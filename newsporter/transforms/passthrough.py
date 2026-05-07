"""Passthrough transformer for sources that already carry clean post data.

Useful for WordPress→WordPress migration or any corpus that has the
title, content, and category already structured. No LLM call, zero cost.

Config schema:
    type: passthrough
    title_field: "title"             # logical name from source.field_map
    body_field: "content"
    category_field: "category"       # optional; falls back to prefab
"""

from __future__ import annotations

import hashlib
import json

from ..models import Post, Prefab, RawRow
from .base import Transformer


class PassthroughTransformer(Transformer):
    SCHEMA_VERSION = "v1-passthrough"

    def __init__(self, cfg: dict, llm=None) -> None:  # llm unused, accepted for symmetry
        self.title_field = cfg.get("title_field", "title")
        self.body_field = cfg.get("body_field", "body")
        self.category_field = cfg.get("category_field")  # may be None

    def signature(self) -> str:
        material = json.dumps(
            {
                "type": "passthrough",
                "version": self.SCHEMA_VERSION,
                "title_field": self.title_field,
                "body_field": self.body_field,
                "category_field": self.category_field,
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def transform(self, row: RawRow, prefab: Prefab) -> Post:
        title = row.fields.get(self.title_field, "").strip() or "Untitled"
        body = row.fields.get(self.body_field, "")
        category = ""
        if self.category_field:
            category = row.fields.get(self.category_field, "").strip()
        if not category:
            category = prefab.category_fallback
        return Post(
            title=title,
            content=body,
            category=category,
            author=prefab.author,
            date_gmt=prefab.date_gmt,
            source_id=row.source_id,
        )
