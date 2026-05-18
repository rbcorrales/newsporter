"""Declarative text cleaners. Each entry in a transform's `cleaners:`
list is a small dict with a `type:` discriminator. New cleaner types are
added here.

Example config:
    cleaners:
      - { type: regex_strip, pattern: "^\\(SOURCE_TAG\\)\\s*--\\s*" }
      - { type: html_strip }
      - { type: trim }
"""

from __future__ import annotations

import re
from collections.abc import Callable

_CLEANERS: dict[str, Callable[[str, dict], str]] = {}


def cleaner(name: str):
    """Decorator: register a cleaner under its config `type:` value."""

    def deco(fn: Callable[[str, dict], str]) -> Callable[[str, dict], str]:
        _CLEANERS[name] = fn
        return fn

    return deco


@cleaner("regex_strip")
def _regex_strip(text: str, cfg: dict) -> str:
    pattern = cfg["pattern"]
    flags = re.DOTALL | re.MULTILINE
    if cfg.get("ignore_case"):
        flags |= re.IGNORECASE
    count = int(cfg.get("count", 1))
    return re.sub(pattern, "", text, count=count, flags=flags)


@cleaner("regex_replace")
def _regex_replace(text: str, cfg: dict) -> str:
    pattern = cfg["pattern"]
    replacement = cfg.get("replacement", "")
    flags = re.DOTALL | re.MULTILINE
    if cfg.get("ignore_case"):
        flags |= re.IGNORECASE
    count = int(cfg.get("count", 0))
    return re.sub(pattern, replacement, text, count=count, flags=flags)


@cleaner("html_strip")
def _html_strip(text: str, cfg: dict) -> str:
    return re.sub(r"<[^>]+>", "", text)


@cleaner("trim")
def _trim(text: str, cfg: dict) -> str:
    return text.strip()


@cleaner("collapse_whitespace")
def _collapse_whitespace(text: str, cfg: dict) -> str:
    return re.sub(r"[ \t]+", " ", text)


def apply_cleaners(text: str, cleaner_specs: list[dict]) -> str:
    """Run each cleaner in order. Unknown types raise ValueError up-front
    so a typo in config doesn't silently no-op."""
    for spec in cleaner_specs or []:
        ctype = spec.get("type")
        if not isinstance(ctype, str):
            raise ValueError(f"Cleaner spec missing `type` (got {ctype!r}).")
        fn = _CLEANERS.get(ctype)
        if fn is None:
            known = ", ".join(sorted(_CLEANERS))
            raise ValueError(f"Unknown cleaner type {ctype!r}. Known: {known}.")
        text = fn(text, spec)
    return text
