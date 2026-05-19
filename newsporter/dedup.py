"""Content-hash dedup primitive.

Source-ID dedup at the loader catches re-uploads of the same row. This
catches a different case: upstream datasets that surface the same
article under multiple row IDs (HuggingFace mirrors, syndicated news
streams). Source IDs differ, so source-ID dedup is blind to it; only
hashing the raw body catches the duplicate.

The hash is computed over the raw source body (post-strip), not over
the WP-side `post.content` (which has block markup wrapped around it).
That keeps the index stable across transformer changes that affect
formatting but not source identity.

MD5 is fine here. This is a dedup key, not a security check.
"""

from __future__ import annotations

import hashlib


def content_hash_for(text: str) -> str:
    """Stable hex digest of a body string. Empty input returns ""."""
    if not text:
        return ""
    normalized = text.strip()
    if not normalized:
        return ""
    return hashlib.md5(normalized.encode("utf-8", errors="replace")).hexdigest()
