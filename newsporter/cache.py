"""Append-only JSONL transform cache, keyed by source_id and stamped with
a transformer-derived signature.

Cache entries from a previous run with a different prompt, model, or
schema are auto-skipped on load (signature mismatch). Delete the file to
start fresh. Re-running the same corpus against a different WordPress
site is free because the LLM cost is paid only once.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path

from .models import Post


class TransformCache:
    def __init__(self, path: Path, sig: str, log: logging.Logger) -> None:
        self.path = path
        self.sig = sig
        self.log = log
        self._lock = threading.Lock()
        self._by_source: dict[str, Post] = {}
        self._loaded = 0
        self._stale = 0
        self._hits = 0
        self._misses = 0
        self._fh = None  # opened lazily on first write
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("sig") != self.sig:
                    self._stale += 1
                    continue
                post_d = entry.get("post") or {}
                try:
                    self._by_source[post_d["source_id"]] = Post(**post_d)
                    self._loaded += 1
                except Exception:
                    continue
        self.log.info(
            "Transform cache: %d hit-eligible, %d stale (different sig). path=%s",
            self._loaded,
            self._stale,
            self.path,
        )

    def get(self, source_id: str) -> Post | None:
        # Lock the read too. CPython's GIL makes single-key dict.get
        # effectively atomic today, but free-threaded builds (3.13+
        # PEP 703) remove that guarantee. The lock is uncontended on the
        # happy path so the cost is negligible.
        with self._lock:
            self._hits += 1 if source_id in self._by_source else 0
            self._misses += 0 if source_id in self._by_source else 1
            return self._by_source.get(source_id)

    def put(self, post: Post) -> None:
        with self._lock:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            self._by_source[post.source_id] = post
            self._fh.write(json.dumps({"sig": self.sig, "post": asdict(post)}) + "\n")
            self._fh.flush()

    def stats(self) -> dict:
        with self._lock:
            return {
                "loaded_at_start": self._loaded,
                "stale_at_start": self._stale,
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._by_source),
            }

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
