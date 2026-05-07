"""Newsporter CLI entrypoint.

Usage:
    newsporter --preset <name>
    newsporter --config /path/to/config.yaml --dry-run --sample-size 200
    newsporter --preset <name> --purge

Defaults to dry-run; pass `--live` (or set `load.dry_run: false` in
config.yaml) to actually upload.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .cache import TransformCache
from .config import ConfigError, build_config, repo_root
from .llm import LLM, load_pricing
from .load.wordpress import WordPressClient, WordPressLoader, purge_all_posts
from .models import Post
from .pipeline import run_pipeline
from .sources import SOURCE_REGISTRY, build_source
from .transforms import LLM_REQUIRED, TRANSFORM_REGISTRY, build_transformer
from .upload_log import UploadLog
from .config import validate_config


class _TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write so progress bars don't get
    shredded by interleaved log output during long runs."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:  # noqa: BLE001
            self.handleError(record)


def _setup_logging(level: str, run_dir: Optional[Path] = None) -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Replace any pre-existing stream handlers so re-init in tests is clean.
    for h in list(root.handlers):
        root.removeHandler(h)
    stream = _TqdmLoggingHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_h.setFormatter(fmt)
        root.addHandler(file_h)
    return logging.getLogger("newsporter")


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def _estimate_embedding_cost(posts: list[Post], costs_cfg: dict) -> dict:
    e = (costs_cfg or {}).get("embedding") or {}
    if not e:
        return {}
    total_chars = sum(len(p.content) for p in posts)
    est_tokens = total_chars // 4
    rate = float(e.get("usd_per_1m_tokens", 0.0))
    usd = est_tokens * rate / 1_000_000
    return {
        "posts": len(posts),
        "est_tokens": est_tokens,
        "provider": e.get("provider"),
        "model": e.get("model"),
        "dimensions": e.get("dimensions"),
        "est_usd": round(usd, 4),
    }


def _write_summary(
    run_dir: Path,
    cfg: dict,
    posts: list[Post],
    results: list[dict],
    elapsed_sec: float,
    chat_cost: Optional[dict],
    embedding_estimate: dict,
    llm_calls: int,
    cache_stats: Optional[dict] = None,
) -> dict:
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python_version": sys.version.split()[0],
        "preset": cfg.get("_preset_name"),
        "source": cfg.get("source") or {},
        "transform_type": (cfg.get("transform") or {}).get("type"),
        "wordpress_url": (cfg.get("wordpress") or {}).get("url"),
        "dry_run": bool((cfg.get("load") or {}).get("dry_run")),
        "sample_size_requested": (cfg.get("dataset") or {}).get("sample_size"),
        "rows_fetched": len(posts) + sum(1 for r in results if r.get("ok") is False),
        "posts_prepared": len(posts),
        "posts_uploaded": sum(1 for r in results if r.get("ok") and not r.get("dry_run")),
        "posts_skipped_existing": sum(1 for r in results if r.get("skipped")),
        "posts_failed": sum(1 for r in results if r.get("ok") is False),
        "llm_calls": llm_calls,
        "elapsed_sec": round(elapsed_sec, 2),
        "pace_sec_per_post": (
            round(elapsed_sec / len(posts), 4) if posts else None
        ),
        "transform_concurrency": int((cfg.get("transform") or {}).get("concurrency", 4)),
        "upload_concurrency": int((cfg.get("load") or {}).get("concurrency", 4)),
    }
    if chat_cost is not None:
        summary["chat_cost"] = chat_cost
    if embedding_estimate:
        summary["embedding_estimate"] = embedding_estimate
    if cache_stats is not None:
        summary["cache"] = cache_stats
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _purge_command(cfg: dict, log: logging.Logger, assume_yes: bool) -> int:
    wp_cfg = cfg.get("wordpress") or {}
    for key in ("url", "username", "app_password"):
        if not wp_cfg.get(key):
            log.error("--purge needs wordpress.%s in config.yaml.", key)
            return 2

    url = wp_cfg["url"]
    if not assume_yes:
        # Force the operator to type the URL so a stray --purge can't
        # nuke the wrong site.
        prompt = (
            f"\nAbout to PERMANENTLY DELETE every post on:\n"
            f"  {url}\n"
            f"Type the URL exactly to confirm (or anything else to abort): "
        )
        try:
            entered = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            log.info("Purge aborted.")
            return 1
        if entered != url:
            log.info("URL mismatch (got %r). Purge aborted.", entered)
            return 1

    log.warning("PURGE: deleting ALL posts on %s", url)
    client = WordPressClient(wp_cfg)
    deleted, failed = purge_all_posts(client, log)
    log.info("Purge done. deleted=%d failed=%d", deleted, failed)

    # Local upload log must follow server state or the next run will
    # silently skip rows that no longer exist on WP.
    upload_log_path = Path(
        (cfg.get("load") or {}).get("upload_log_path", "data/uploads.jsonl")
    )
    if upload_log_path.exists():
        ul = UploadLog(upload_log_path, log)
        ul.truncate()
        log.info("Cleared local upload log: %s", upload_log_path)

    return 0


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def main() -> int:
    from . import __version__
    parser = argparse.ArgumentParser(
        prog="newsporter",
        description="Pluggable ETL pipeline that loads a corpus into WordPress.",
    )
    parser.add_argument("--version", action="version", version=f"newsporter {__version__}")

    src = parser.add_argument_group("Config sources")
    src.add_argument("--preset", help="Name of a preset under presets/ (no .yaml)")
    src.add_argument("--config", help="Path to a custom config YAML")

    mode = parser.add_argument_group("Mode").add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run; skip upload (this is also the defaults.yaml default)",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Force live upload (overrides defaults.yaml dry_run: true)",
    )

    runknobs = parser.add_argument_group("Run knobs")
    runknobs.add_argument(
        "--sample-size", type=_positive_int, help="Override dataset.sample_size"
    )
    runknobs.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override logging.level",
    )
    runknobs.add_argument(
        "--no-cache", action="store_true", help="Bypass transform cache for this run"
    )

    resume = parser.add_argument_group("Resume / verification")
    resume.add_argument(
        "--no-resume-check",
        action="store_true",
        help="Skip the resume check entirely. Use when uploading to a "
             "known-empty site, or to force a fresh upload of every row.",
    )
    resume.add_argument(
        "--verify-with-wp",
        action="store_true",
        help="Cross-check the local upload log against WordPress and "
             "REPLACE the log with WP's authoritative state. Use after "
             "wp-admin deletions or cross-machine handoffs.",
    )

    maint = parser.add_argument_group("Maintenance")
    maint.add_argument(
        "--purge",
        action="store_true",
        help="DESTRUCTIVE: delete every post on the target WP site, "
             "clear the local upload log, then exit. Prompts for URL "
             "confirmation unless --yes is also passed.",
    )
    maint.add_argument(
        "--yes",
        action="store_true",
        help="Skip the --purge confirmation prompt (CI / scripted use)",
    )
    maint.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit",
    )

    args = parser.parse_args()

    if args.no_resume_check and args.verify_with_wp:
        parser.error(
            "--no-resume-check and --verify-with-wp are mutually exclusive: "
            "the former skips the check, the latter requires it."
        )

    if args.list_presets:
        presets_dir = repo_root() / "presets"
        if presets_dir.exists():
            for p in sorted(presets_dir.glob("*.yaml")):
                print(p.stem)
        return 0

    try:
        cfg, config_sources = build_config(args)
    except ConfigError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not config_sources:
        print(
            "ERROR: no config layers found. Expected at least defaults.yaml at "
            "the repo root, or pass --preset NAME / --config path.",
            file=sys.stderr,
        )
        return 2

    transform_cfg = cfg.get("transform") or {}
    needs_llm = transform_cfg.get("type") in LLM_REQUIRED

    try:
        validate_config(
            cfg,
            llm_required=needs_llm,
            source_registry=SOURCE_REGISTRY,
            transform_registry=TRANSFORM_REGISTRY,
        )
    except ConfigError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # ── Set up run dir + logging (file handler lives in run_dir) ──────
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path((cfg.get("logging") or {}).get("run_dir", "runs")) / run_id
    log = _setup_logging((cfg.get("logging") or {}).get("level", "INFO"), run_dir)
    for src in config_sources:
        log.info("Config layer: %s", src)

    if args.purge:
        return _purge_command(cfg, log, assume_yes=args.yes)

    # ── Wire components ───────────────────────────────────────────────
    pricing = load_pricing(repo_root() / "pricing.yaml")

    llm: Optional[LLM] = None
    if needs_llm:
        llm = LLM(cfg["llm"], pricing=pricing)
        log.info("LLM: %s @ %s", cfg["llm"]["model"], cfg["llm"]["base_url"])

    source = build_source(cfg.get("source") or {})

    # Source identity: a stable string derived from the source config so
    # the cache signature flips when you switch corpora. Avoids stale
    # cross-corpus replays when source_ids happen to overlap.
    source_cfg = cfg.get("source") or {}
    source_identity = json.dumps(
        {
            "type": source_cfg.get("type"),
            "name": source_cfg.get("name"),
            "config": source_cfg.get("config"),
            "split": source_cfg.get("split"),
            "path": source_cfg.get("path"),
            "field_map": source_cfg.get("field_map"),
        },
        sort_keys=True,
    )

    transformer = build_transformer(transform_cfg, llm, source_identity=source_identity)

    labels: list[str] = list(((transform_cfg.get("category") or {}).get("labels")) or [])
    transform_author = (transform_cfg.get("author") or {})
    load_author = ((cfg.get("load") or {}).get("author") or {})
    author_pool: list[str] = list(
        transform_author.get("pool") or load_author.get("pool") or []
    )

    # Upload log is opt-out via --no-resume-check. Path is configurable
    # via load.upload_log_path, defaults to data/uploads.jsonl.
    upload_log: Optional[UploadLog] = None
    if not args.no_resume_check:
        upload_log_path = Path(
            (cfg.get("load") or {}).get("upload_log_path", "data/uploads.jsonl")
        )
        upload_log = UploadLog(upload_log_path, log)

    loader = WordPressLoader(
        wp_cfg=cfg.get("wordpress") or {},
        load_cfg=cfg.get("load") or {},
        labels=labels,
        author_pool=author_pool,
        upload_log=upload_log,
        skip_resume_check=args.no_resume_check,
        verify_with_wp=args.verify_with_wp,
    )
    if not loader.dry_run and loader.existing_post_ids:
        log.info(
            "Resume: %d posts already uploaded; will skip those",
            len(loader.existing_post_ids),
        )

    # ── Cache ─────────────────────────────────────────────────────────
    cache: Optional[TransformCache] = None
    cache_cfg = (transform_cfg.get("cache") or {})
    if cache_cfg.get("enabled"):
        cache_path = Path(cache_cfg.get("path") or "data/transforms_cache.jsonl")
        cache = TransformCache(cache_path, transformer.signature(), log)

    # ── Run ───────────────────────────────────────────────────────────
    log.info("Run ID: %s  dry_run=%s", run_id, loader.dry_run)
    t0 = time.monotonic()
    try:
        posts, results = run_pipeline(
            source, transformer, loader, cfg, log, run_dir, cache
        )
    finally:
        if cache is not None:
            cache.close()
        if upload_log is not None:
            upload_log.close()
    elapsed = time.monotonic() - t0

    embedding_est = _estimate_embedding_cost(posts, cfg.get("costs") or {})
    if embedding_est:
        log.info(
            "Embedding estimate (future phase): ~%d tokens, ~$%.4f with %s",
            embedding_est["est_tokens"],
            embedding_est["est_usd"],
            embedding_est.get("model"),
        )

    chat_cost = llm.cost_summary() if llm is not None else None
    if chat_cost is not None:
        log.info(
            "Chat usage: %s, in=%d out=%d tokens, est=$%.4f%s",
            chat_cost["model"],
            chat_cost["input_tokens"],
            chat_cost["output_tokens"],
            chat_cost["est_usd"],
            "" if chat_cost["priced"] else " (model not in pricing.yaml)",
        )

    cache_stats = cache.stats() if cache is not None else None
    summary = _write_summary(
        run_dir, cfg, posts, results, elapsed,
        chat_cost, embedding_est, llm.calls if llm else 0, cache_stats,
    )
    log.info(
        "Done in %.1fs. Prepared=%d uploaded=%d skipped=%d failed=%d llm_calls=%d",
        elapsed,
        summary["posts_prepared"],
        summary["posts_uploaded"],
        summary["posts_skipped_existing"],
        summary["posts_failed"],
        summary["llm_calls"],
    )
    log.info("Artifacts: %s", run_dir)
    # Exit non-zero if anything failed so CI / shell scripts notice. A
    # post that uploaded successfully but then a network blip dropped
    # the response (and idempotency recovered it) reports `recovered`,
    # not `failed`, so this only fires on real losses.
    return 1 if summary["posts_failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
