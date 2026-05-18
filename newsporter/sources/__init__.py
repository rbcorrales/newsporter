"""Source registry. Add a new source by importing it here and registering
its `type:` discriminator string."""

from __future__ import annotations

from .base import Source
from .huggingface import HuggingFaceSource
from .jsonl import JsonlSource

SOURCE_REGISTRY: dict[str, type[Source]] = {
    "huggingface": HuggingFaceSource,
    "jsonl": JsonlSource,
}


def build_source(cfg: dict) -> Source:
    """Instantiate the right Source subclass given a config block with a
    `type:` key."""
    source_type = cfg.get("type")
    if source_type not in SOURCE_REGISTRY:
        known = ", ".join(sorted(SOURCE_REGISTRY))
        raise ValueError(
            f"Unknown source type {source_type!r}. Known: {known}. "
            f"Set `source.type` in your config."
        )
    return SOURCE_REGISTRY[source_type](cfg)


__all__ = ["SOURCE_REGISTRY", "Source", "build_source"]
