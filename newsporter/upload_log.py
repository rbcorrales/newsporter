"""Append-only local log of successful WordPress uploads.

One JSONL line per upload: `{"source_id": "...", "post_id": 12345,
"content_hash": "..."}`. The `content_hash` field is optional (empty or
omitted for pre-dedup entries); when present, it powers cross-run
content-hash dedup on warm-resume paths that don't fetch from WP.

On resume, the loader reads this file to know in O(disk) which rows are
already done (no WP REST roundtrips required) AND which content hashes
have already landed (so an upstream-duplicate ingested in a previous
run can't slip through a fresh fetch).

Authority: the local log records what *this machine* uploaded. If posts
are deleted via wp-admin or you resume from a different machine, the
log can be stale. Pass `--verify-with-wp` to do a bulk WP fetch and
merge that into the log (taking WP as authoritative). Without that
flag, the log is trusted as-is for speed.

This is separate from `TransformCache` on purpose. A row may be
transformed (paid for the LLM) but fail to upload (network blip after
the LLM call). The two states are independent.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path


class UploadLog:
    def __init__(self, path: Path, log: logging.Logger) -> None:
        self.path = path
        self.log = log
        self._lock = threading.Lock()
        self._by_source: dict[str, int] = {}
        self._by_hash: dict[str, int] = {}
        self._fh = None  # opened lazily on first write
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        loaded = 0
        hashes_loaded = 0
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
                if sid is not None and sid != "" and isinstance(pid, int) and pid > 0:
                    self._by_source[str(sid)] = pid
                    loaded += 1
                chash = entry.get("content_hash")
                # setdefault: first-seen post wins as the canonical id for
                # this content. Mirrors the loader's setdefault on
                # `existing_content_hashes` during live uploads. Short-circuit
                # the and-chain so setdefault doesn't fire on invalid rows.
                if (
                    chash
                    and isinstance(pid, int)
                    and pid > 0
                    and self._by_hash.setdefault(str(chash), pid) == pid
                ):
                    hashes_loaded += 1
        self.log.info(
            "Upload log: %d previously-uploaded rows loaded from %s (%d with content hash)",
            loaded,
            self.path,
            hashes_loaded,
        )

    def get(self, source_id: str) -> int | None:
        with self._lock:
            return self._by_source.get(source_id)

    def all(self) -> dict[str, int]:
        with self._lock:
            return dict(self._by_source)

    def all_hashes(self) -> dict[str, int]:
        with self._lock:
            return dict(self._by_hash)

    def put(self, source_id: str, post_id: int, content_hash: str = "") -> None:
        with self._lock:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            self._by_source[source_id] = post_id
            if content_hash:
                self._by_hash.setdefault(content_hash, post_id)
            entry: dict[str, object] = {"source_id": source_id, "post_id": post_id}
            if content_hash:
                entry["content_hash"] = content_hash
            self._fh.write(json.dumps(entry) + "\n")
            self._fh.flush()

    def replace(
        self,
        mapping: dict[str, int],
        hashes: dict[str, int] | None = None,
    ) -> int:
        """Atomic replace: rewrite the file from scratch using only the
        provided mapping. Stale entries (locally-tracked posts that no
        longer exist on WP) get dropped. If `hashes` is provided, lines
        that match a known content hash get the field stamped, so a
        `--verify-with-wp` pass also rebuilds the hash index on disk.
        Returns the count written.
        """
        with self._lock:
            self._by_source = {
                str(k): int(v) for k, v in mapping.items() if isinstance(v, int) and v > 0
            }
            # Invert hashes (content_hash -> post_id) to (post_id -> content_hash)
            # so we can look up by post id while rewriting the source-keyed log.
            pid_to_hash: dict[int, str] = {}
            for chash, pid in (hashes or {}).items():
                if isinstance(pid, int) and pid > 0 and chash:
                    pid_to_hash[pid] = str(chash)
            self._by_hash = {h: p for p, h in pid_to_hash.items()}
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as out:
                for sid, pid in self._by_source.items():
                    entry: dict[str, object] = {"source_id": sid, "post_id": pid}
                    chash = pid_to_hash.get(pid)
                    if chash:
                        entry["content_hash"] = chash
                    out.write(json.dumps(entry) + "\n")
                out.flush()
            tmp.replace(self.path)
            return len(self._by_source)

    def truncate(self) -> None:
        """Wipe the log entirely. Used by `newsporter --purge` to keep
        the local state consistent with the now-empty WP site."""
        with self._lock:
            self._by_source.clear()
            self._by_hash.clear()
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            if self.path.exists():
                self.path.unlink()

    def merge(
        self,
        mapping: dict[str, int],
        hashes: dict[str, int] | None = None,
    ) -> int:
        """Bulk import (e.g. from a WP-side fetch). Returns count of
        new entries written. Existing entries with the same source_id
        are overwritten on disk too (last-write wins). If `hashes` is
        provided, the merge also stamps content_hash on matching posts
        so warm-resume keeps cross-run hash dedup.
        """
        new_count = 0
        pid_to_hash: dict[int, str] = {}
        for chash, pid in (hashes or {}).items():
            if isinstance(pid, int) and pid > 0 and chash:
                pid_to_hash[pid] = str(chash)
        with self._lock:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            for sid, pid in mapping.items():
                sid = str(sid)
                if not isinstance(pid, int) or pid <= 0:
                    continue
                chash = pid_to_hash.get(pid, "")
                changed_source = self._by_source.get(sid) != pid
                changed_hash = bool(chash) and self._by_hash.get(chash) != pid
                if not (changed_source or changed_hash):
                    continue
                self._by_source[sid] = pid
                if chash:
                    self._by_hash.setdefault(chash, pid)
                entry: dict[str, object] = {"source_id": sid, "post_id": pid}
                if chash:
                    entry["content_hash"] = chash
                self._fh.write(json.dumps(entry) + "\n")
                new_count += 1
            self._fh.flush()
        return new_count

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
