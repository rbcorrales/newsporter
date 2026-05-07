"""HuggingFace `datasets` source.

Config schema:
    type: huggingface
    name: "<owner>/<dataset>"
    config: "<version>"      # optional, dataset-specific
    split: "train"
    cache_dir: "data/hf_cache"
    field_map:               # logical name -> source column
      id: "id"               # required: column to use as source_id
      body: "<body_column>"  # any other logical names the transformer wants
      summary: "<summary_column>"
"""

from __future__ import annotations

import random

from datasets import load_dataset

from ..models import RawRow
from .base import Source


class HuggingFaceSource(Source):
    def __init__(self, cfg: dict) -> None:
        self.name = cfg["name"]
        self.dataset_config = cfg.get("config")
        self.split = cfg.get("split", "train")
        self.cache_dir = cfg.get("cache_dir")
        self.field_map: dict[str, str] = cfg.get("field_map") or {}
        self.id_field = self.field_map.get("id", "id")
        # Logical -> source mapping for the non-id fields.
        self.body_field_map = {k: v for k, v in self.field_map.items() if k != "id"}

    def fetch(self, sample_size: int, seed: int) -> list[RawRow]:
        kwargs = {"split": self.split}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        ds = (
            load_dataset(self.name, self.dataset_config, **kwargs)
            if self.dataset_config
            else load_dataset(self.name, **kwargs)
        )
        rng = random.Random(seed)
        total = len(ds)
        n = min(sample_size, total) if sample_size else total
        indices = rng.sample(range(total), n)
        rows: list[RawRow] = []
        for i in indices:
            r = ds[i]
            source_id = str(r.get(self.id_field) if self.id_field in r else i)
            fields = {
                logical: str(r.get(src_col, "") or "")
                for logical, src_col in self.body_field_map.items()
            }
            rows.append(RawRow(source_id=source_id, fields=fields))
        return rows
