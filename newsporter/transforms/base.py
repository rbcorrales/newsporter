"""Transformer abstract base."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Post, Prefab, RawRow


class Transformer(ABC):
    """Turns a (RawRow, Prefab) pair into a Post."""

    @abstractmethod
    def signature(self) -> str:
        """Stable identifier hashed into the cache key. Should change
        whenever the transformer's behaviour would produce different
        outputs (model, prompt, schema, cleaner list)."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, row: RawRow, prefab: Prefab) -> Post:
        raise NotImplementedError
