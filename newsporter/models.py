"""Shared data shapes that flow through Extract → Transform → Load."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawRow:
    """One record from a Source.

    `source_id` is the stable identifier used as the cache key. `fields` is
    a free-form dict of source-named columns, so different sources can
    surface different schemas (e.g. one dataset has `article` + `highlights`,
    a WordPress export has `title` + `content` + `tags`). Transformers pull
    the fields they need by name via the source's `field_map` config.

    `content_hash` is filled by the pipeline after fetch (hash of the raw
    source body via `dedup.content_hash_for`) and is used to drop rows
    whose content was already ingested under a different `source_id`. It
    stays "" if the body field is missing or empty.
    """

    source_id: str
    fields: dict[str, str] = field(default_factory=dict)
    content_hash: str = ""


@dataclass
class Prefab:
    """Per-row randomization picked up-front in the main thread.

    Worker threads must not share a seeded `random.Random` instance, so
    anything that needs deterministic per-row randomness (author choice,
    fallback date, fallback category) is generated here once.
    """

    row: RawRow
    author: str
    date_gmt: str
    category_fallback: str


@dataclass
class Post:
    """The transformed post that the loader will upload."""

    title: str
    content: str
    category: str
    author: str
    date_gmt: str
    source_id: str
    content_hash: str = ""
