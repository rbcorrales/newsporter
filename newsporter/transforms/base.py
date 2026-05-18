"""Transformer abstract base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..models import Post, Prefab, RawRow

if TYPE_CHECKING:
    from ..llm import LLM


class Transformer(ABC):
    """Turns a (RawRow, Prefab) pair into a Post."""

    def __init__(
        self,
        cfg: dict,
        llm: LLM | None = None,
        *,
        source_identity: str = "",
    ) -> None:
        """Registry contract: every Transformer is instantiated with the
        `transform:` block, the LLM client (None for transforms that don't
        need one), and an optional `source_identity` string used by cache
        signatures. Subclasses override to do their own validation."""
        self.cfg = cfg
        self.llm = llm
        self.source_identity = source_identity

    @abstractmethod
    def signature(self) -> str:
        """Stable identifier hashed into the cache key. Should change
        whenever the transformer's behaviour would produce different
        outputs (model, prompt, schema, cleaner list)."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, row: RawRow, prefab: Prefab) -> Post:
        raise NotImplementedError
