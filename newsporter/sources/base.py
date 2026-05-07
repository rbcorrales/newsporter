"""Source abstract base. Implementations live in sibling modules and
register themselves in `__init__.SOURCE_REGISTRY`."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import RawRow


class Source(ABC):
    """Yields RawRows. Concrete subclasses configure themselves from the
    `source:` block of the active config."""

    @abstractmethod
    def fetch(self, sample_size: int, seed: int) -> list[RawRow]:
        """Return up to `sample_size` rows. `seed` controls deterministic
        sampling when the source supports it."""
        raise NotImplementedError
