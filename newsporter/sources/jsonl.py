"""Local JSONL source. One JSON object per line, any schema.

Config schema:
    type: jsonl
    path: "data/my_corpus.jsonl"
    field_map:               # logical name -> source key
      id: "id"
      body: "content"
      summary: "excerpt"     # optional
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..models import RawRow
from .base import Source


class JsonlSource(Source):
    def __init__(self, cfg: dict) -> None:
        self.path = Path(cfg["path"])
        self.field_map: dict[str, str] = cfg.get("field_map") or {}
        self.id_field = self.field_map.get("id", "id")
        self.body_field_map = {k: v for k, v in self.field_map.items() if k != "id"}

    def fetch(self, sample_size: int, seed: int) -> list[RawRow]:
        all_records: list[dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        rng = random.Random(seed)
        total = len(all_records)
        n = min(sample_size, total) if sample_size else total
        indices = rng.sample(range(total), n)
        rows: list[RawRow] = []
        for i in indices:
            r = all_records[i]
            source_id = str(r.get(self.id_field, i))
            fields = {
                logical: str(r.get(src_key, "") or "")
                for logical, src_key in self.body_field_map.items()
            }
            rows.append(RawRow(source_id=source_id, fields=fields))
        return rows
