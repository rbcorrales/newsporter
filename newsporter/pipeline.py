"""Streaming Extract → Transform → Load orchestration.

Transform workers feed posts into a bounded queue; upload workers drain
it. Per-post artifacts (`posts.jsonl`, `results.jsonl`) flush on every
record so long runs stay observable mid-flight.
"""

from __future__ import annotations

import json
import logging
import queue
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from .cache import TransformCache
from .load.wordpress import WordPressLoader
from .models import Post, Prefab, RawRow
from .sources.base import Source
from .transforms.base import Transformer
from .transforms.llm_synth import random_date_for_range

_SENTINEL = object()


def build_prefabs(rows: list[RawRow], cfg: dict) -> list[Prefab]:
    """Pick the per-row randomized fields up-front in a single thread, so
    worker threads don't need a shared seeded RNG."""
    seed = cfg.get("dataset", {}).get("seed", 42)
    rng = random.Random(seed)

    transform_cfg = cfg.get("transform") or {}
    date_cfg = transform_cfg.get("date") or {}
    category_cfg = transform_cfg.get("category") or {}
    labels: list[str] = list(category_cfg.get("labels") or [])

    # Author pool can live on either `transform.author.pool` (LLM synth use
    # case) or `load.author.pool` (passthrough use case). Whichever's set wins.
    transform_author = transform_cfg.get("author") or {}
    load_author = (cfg.get("load") or {}).get("author") or {}
    pool: list[str] = list(transform_author.get("pool") or load_author.get("pool") or [])

    prefabs: list[Prefab] = []
    for row in rows:
        author = rng.choice(pool) if pool else ""
        date_gmt = random_date_for_range(date_cfg, rng) if date_cfg else ""
        category_fallback = labels[0] if labels else ""
        prefabs.append(
            Prefab(
                row=row,
                author=author,
                date_gmt=date_gmt,
                category_fallback=category_fallback,
            )
        )
    return prefabs


def run_pipeline(
    source: Source,
    transformer: Transformer,
    loader: WordPressLoader,
    cfg: dict,
    log: logging.Logger,
    run_dir: Path,
    cache: TransformCache | None = None,
) -> tuple[list[Post], list[dict]]:
    dataset_cfg = cfg.get("dataset") or {}
    sample_size = int(dataset_cfg.get("sample_size", 0))
    seed = int(dataset_cfg.get("seed", 42))

    log.info("Extracting up to %d rows", sample_size or 0)
    rows = source.fetch(sample_size, seed)
    log.info("Fetched %d rows", len(rows))

    # Resume optimisation: if the loader already knows about posts on
    # the target site (via WP-side `_newsporter_source_id` lookup), skip
    # those rows entirely. Avoids paying transform-cache lookups + REST
    # preflights for every already-uploaded row on a 99k-of-100k resume.
    existing = getattr(loader, "existing_post_ids", {}) or {}
    if existing and not loader.dry_run:
        before = len(rows)
        rows = [r for r in rows if r.source_id not in existing]
        skipped = before - len(rows)
        if skipped:
            log.info(
                "Resume: %d/%d rows already uploaded; processing %d new",
                skipped,
                before,
                len(rows),
            )

    prefabs = build_prefabs(rows, cfg)

    transform_cfg = cfg.get("transform") or {}
    load_cfg = cfg.get("load") or {}
    transform_workers = int(transform_cfg.get("concurrency", 4))
    upload_workers = int(load_cfg.get("concurrency", 4))
    queue_cap = max(upload_workers * 4, 16)

    posts: list[Post] = []
    results: list[dict] = []
    posts_lock = threading.Lock()
    results_lock = threading.Lock()
    post_q: queue.Queue[object] = queue.Queue(maxsize=queue_cap)

    run_dir.mkdir(parents=True, exist_ok=True)
    posts_path = run_dir / "posts.jsonl"
    results_path = run_dir / "results.jsonl"
    posts_fh = posts_path.open("w", encoding="utf-8")
    results_fh = results_path.open("w", encoding="utf-8") if not loader.dry_run else None

    tr_bar = tqdm(total=len(prefabs), desc="Transform", position=0, leave=True)
    up_bar = tqdm(total=len(prefabs), desc="Upload   ", position=1, leave=True)

    def record_post(post: Post) -> None:
        with posts_lock:
            posts.append(post)
            posts_fh.write(json.dumps(asdict(post)) + "\n")
            posts_fh.flush()

    def record_result(result: dict) -> None:
        with results_lock:
            results.append(result)
            if results_fh is not None:
                results_fh.write(json.dumps(result) + "\n")
                results_fh.flush()
            up_bar.update(1)

    def transform_worker(pf: Prefab) -> None:
        try:
            cached = cache.get(pf.row.source_id) if cache is not None else None
            if cached is not None:
                post = cached
            else:
                post = transformer.transform(pf.row, pf)
                if cache is not None:
                    cache.put(post)
            record_post(post)
            post_q.put(post)
        except Exception as e:
            log.warning("Transform failed for %s: %s", pf.row.source_id, e)
            post_q.put(("__err__", pf.row.source_id, str(e)))
        finally:
            tr_bar.update(1)

    def upload_worker() -> None:
        while True:
            item = post_q.get()
            try:
                if item is _SENTINEL:
                    return
                try:
                    if isinstance(item, tuple) and item and item[0] == "__err__":
                        _, source_id, err = item
                        record_result(
                            {"source_id": source_id, "error": f"transform: {err}", "ok": False}
                        )
                        continue
                    if loader.dry_run:
                        assert isinstance(item, Post)
                        record_result({"source_id": item.source_id, "dry_run": True, "ok": True})
                        continue
                    assert isinstance(item, Post)
                    record_result(loader.upload(item))
                except Exception as e:
                    # Per-item failure must not kill the worker. A dying
                    # worker leaves items unprocessed and may not consume
                    # its sentinel, leading to hangs at shutdown.
                    source_id = item.source_id if isinstance(item, Post) else "?"
                    log.exception("Upload worker error on %s", source_id)
                    record_result({"source_id": source_id, "error": f"upload: {e}", "ok": False})
            finally:
                post_q.task_done()

    try:
        with (
            ThreadPoolExecutor(max_workers=upload_workers, thread_name_prefix="upload") as up_ex,
            ThreadPoolExecutor(max_workers=transform_workers, thread_name_prefix="xform") as tr_ex,
        ):
            up_futs = [up_ex.submit(upload_worker) for _ in range(upload_workers)]
            tr_futs = [tr_ex.submit(transform_worker, pf) for pf in prefabs]
            try:
                for f in as_completed(tr_futs):
                    f.result()
            finally:
                # Push sentinels in a finally so a transform exception
                # (KeyboardInterrupt, OOM, etc.) can never deadlock the
                # upload workers in the executor's wait-on-shutdown.
                # Timeout exceeds the worst-case worker stall: each
                # worker may sleep up to attempts x Retry-After-budget
                # on a 429 cascade. 5 min is a generous upper bound.
                sentinel_timeout = 300
                for _ in range(upload_workers):
                    try:
                        post_q.put(_SENTINEL, timeout=sentinel_timeout)
                    except Exception:
                        log.error("Failed to enqueue upload sentinel; workers may hang.")
            for f in up_futs:
                try:
                    f.result()
                except Exception as e:
                    log.error("Upload worker died: %s", e)
    finally:
        tr_bar.close()
        up_bar.close()
        posts_fh.close()
        if results_fh is not None:
            results_fh.close()

    return posts, results
