"""Transformer registry. Add a new transformer by importing it here and
registering its `type:` discriminator string."""

from __future__ import annotations

from typing import Optional

from ..llm import LLM
from .base import Transformer
from .llm_synth import LLMSynthTransformer
from .passthrough import PassthroughTransformer

TRANSFORM_REGISTRY: dict[str, type[Transformer]] = {
    "llm_synth": LLMSynthTransformer,
    "passthrough": PassthroughTransformer,
}

# Transformers that need an LLM client. Anything else can be built without.
LLM_REQUIRED: set[str] = {"llm_synth"}


def build_transformer(
    cfg: dict, llm: Optional[LLM], source_identity: str = ""
) -> Transformer:
    transform_type = cfg.get("type")
    if transform_type not in TRANSFORM_REGISTRY:
        known = ", ".join(sorted(TRANSFORM_REGISTRY))
        raise ValueError(
            f"Unknown transform type {transform_type!r}. Known: {known}. "
            f"Set `transform.type` in your config."
        )
    cls = TRANSFORM_REGISTRY[transform_type]
    if transform_type in LLM_REQUIRED:
        if llm is None:
            raise ValueError(
                f"Transform type {transform_type!r} requires an `llm:` block."
            )
        return cls(cfg, llm, source_identity=source_identity)
    return cls(cfg)


__all__ = ["Transformer", "TRANSFORM_REGISTRY", "LLM_REQUIRED", "build_transformer"]
