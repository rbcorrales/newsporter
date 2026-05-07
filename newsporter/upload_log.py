"""Append-only local log of successful WordPress uploads.

One JSONL line per upload: `{"source_id": "...", "post_id": 12345}`. On
resume, the loader reads this file to know in O(disk) which rows are
already done — no WP REST roundtrips required.

Authority: the local log records what *this machine* uploaded. If posts
are deleted via wp-admin or you resume from a different machine, the
log can be stale. Pass `--verify-with-wp` to do a bulk WP fetch and
merge that into the log (taking WP as authoritative). Without that
flag, the log is trusted as-is for speed.

This is separate from `TransformCache` on purpose — a row may be
transformed (paid for the LLM) but fail to upload (network blip after
the LLM call). The two states are independent.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional


class UploadLog:
    def __init__(self, path: Path, log: logging.Logger) -> None:
        self.path = path
        self.log = log
        self._lock = threading.Lock()
        self._by_source: dict[str, int] = {}
        self._fh = None  # opened lazily on first write
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        loaded = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                sid = entry.get("source_id")
                pid = entry.get("post_id")
                if sid and isinstance(pid, int):
                    self._by_source[str(sid)] = pid
                    loaded += 1
        self.log.info(
            "Upload log: %d previously-uploaded rows loaded from %s",
            loaded,
            self.path,
        )

    def get(self, source_id: str) -> Optional[int]:
        with self._lock:
            return self._by_source.get(source_id)

    def all(self) -> dict[str, int]:
        with self._lock:
            return dict(self._by_source)

    def put(self, source_id: str, post_id: int) -> None:
        with self._lock:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            self._by_source[source_id] = post_id
            self._fh.write(
                json.dumps({"source_id": source_id, "post_id": post_id}) + "\n"
            )
            self._fh.flush()

    def replace(self, mapping: dict[str, int]) -> int:
        """Atomic replace: rewrite the file from scratch using only the
        provided mapping. Stale entries (locally-tracked posts that no
        longer exist on WP) get dropped. Returns the count written."""
        with self._lock:
            self._by_source = {str(k): int(v) for k, v in mapping.items() if isinstance(v, int) and v > 0}
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as out:
                for sid, pid in self._by_source.items():
                    out.write(json.dumps({"source_id": sid, "post_id": pid}) + "\n")
                out.flush()
            tmp.replace(self.path)
            return len(self._by_source)

    def truncate(self) -> None:
        """Wipe the log entirely. Used by `newsporter --purge` to keep
        the local state consistent with the now-empty WP site."""
        with self._lock:
            self._by_source.clear()
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            if self.path.exists():
                self.path.unlink()

    def merge(self, mapping: dict[str, int]) -> int:
        """Bulk import (e.g. from a WP-side fetch). Returns count of
        new entries written. Existing entries with the same source_id
        are overwritten on disk too — last-write wins."""
        new_count = 0
        with self._lock:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            for sid, pid in mapping.items():
                sid = str(sid)
                if not isinstance(pid, int) or pid <= 0:
                    continue
                if self._by_source.get(sid) != pid:
                    self._by_source[sid] = pid
                    self._fh.write(
                        json.dumps({"source_id": sid, "post_id": pid}) + "\n"
                    )
                    new_count += 1
            self._fh.flush()
        return new_count

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
