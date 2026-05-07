"""Config loading, layering, env-var substitution, and validation.

The CLI calls `build_config(args)` to get a fully resolved, validated
config plus the list of files that contributed to it. Everything that
touches YAML, deep-merge, or substitution lives here so cli.py stays a
thin orchestrator.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Optional

import yaml

# Match `${VAR}` anywhere in a string. We deliberately do NOT support
# default-value syntax (`${VAR:-default}`) because the surface area is
# small and the failure mode is loud (empty value triggers validation).
_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised by `validate_config` when the merged config is unusable.
    Always carries an actionable message."""


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively overlay onto a deep copy of `base`. Lists are replaced
    wholesale (a preset's `categories.labels` shouldn't be appended to a
    default; same for any list-shaped value). Returns a new dict; neither
    input is mutated."""
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# Allowlist of dotted config paths where `${VAR}` substitution is honored.
# Anything outside this set gets the raw string verbatim — preventing
# poisoned presets from exfiltrating arbitrary env vars (e.g. embedding
# `${AWS_SECRET_ACCESS_KEY}` in a prompt template). Each entry is a
# tuple of segments; a `*` segment matches any single key.
_ENV_SUB_PATHS: set[tuple[str, ...]] = {
    ("llm", "api_key"),
    ("llm", "headers", "*"),
    ("wordpress", "url"),
    ("wordpress", "username"),
    ("wordpress", "app_password"),
}


def _path_matches(path: tuple[str, ...]) -> bool:
    for allowed in _ENV_SUB_PATHS:
        if len(allowed) != len(path):
            continue
        if all(a == "*" or a == p for a, p in zip(allowed, path)):
            return True
    return False


def _expand_at_path(value, path: tuple[str, ...]):
    if isinstance(value, str):
        if _path_matches(path):
            return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
        return value
    if isinstance(value, dict):
        return {k: _expand_at_path(v, path + (k,)) for k, v in value.items()}
    if isinstance(value, list):
        # Lists don't get path segments; their elements aren't credential
        # leaves we care about. (If a leaf inside a list ever needs sub,
        # add `(*, idx)` semantics.)
        return [_expand_at_path(v, path + (None,)) for v in value]  # type: ignore[arg-type]
    return value


def expand_env_vars(cfg: dict) -> dict:
    """Walk the config and substitute `${VAR}` in credential-shaped
    leaves only. Untrusted presets cannot exfiltrate arbitrary env
    vars via prompts, URLs, or other fields."""
    return _expand_at_path(cfg, ())


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_preset(name: str) -> Path:
    available: list[str] = []
    for base in (repo_root(), Path.cwd()):
        presets_dir = base / "presets"
        if presets_dir.exists():
            available.extend(p.stem for p in presets_dir.glob("*.yaml"))
        candidate = base / "presets" / f"{name}.yaml"
        if candidate.exists():
            return candidate
    available_str = (
        ", ".join(sorted(set(available))) if available else "(none found)"
    )
    raise ConfigError(
        f"Preset {name!r} not found in presets/. Available: {available_str}"
    )


def build_config(args: argparse.Namespace) -> tuple[dict, list[str]]:
    """Layer config: defaults → preset → user config → CLI flags. Apply
    env-var substitution. Returns (merged_config, sources_used) where
    sources_used is a list of file paths in load order, for logging."""
    sources: list[str] = []
    cfg: dict = {}

    defaults_path = repo_root() / "defaults.yaml"
    if defaults_path.exists():
        cfg = deep_merge(cfg, load_yaml(defaults_path))
        sources.append(str(defaults_path))

    if args.preset:
        preset_path = resolve_preset(args.preset)
        cfg = deep_merge(cfg, load_yaml(preset_path))
        sources.append(str(preset_path))
        cfg["_preset_name"] = args.preset

    user_config_path: Optional[Path] = None
    if args.config:
        user_config_path = Path(args.config)
    elif Path("config.yaml").exists():
        user_config_path = Path("config.yaml")
    if user_config_path is not None:
        if not user_config_path.exists():
            raise ConfigError(f"Config file not found: {user_config_path}")
        cfg = deep_merge(cfg, load_yaml(user_config_path))
        sources.append(str(user_config_path))

    # CLI flag overrides go last so they always win.
    if getattr(args, "dry_run", False):
        cfg = deep_merge(cfg, {"load": {"dry_run": True}})
    if getattr(args, "live", False):
        cfg = deep_merge(cfg, {"load": {"dry_run": False}})
    if getattr(args, "sample_size", None) is not None:
        cfg = deep_merge(cfg, {"dataset": {"sample_size": args.sample_size}})
    if getattr(args, "log_level", None):
        cfg = deep_merge(cfg, {"logging": {"level": args.log_level}})
    if getattr(args, "no_cache", False):
        cfg = deep_merge(cfg, {"transform": {"cache": {"enabled": False}}})

    cfg = expand_env_vars(cfg)
    return cfg, sources


def validate_config(cfg: dict, *, llm_required: bool, source_registry, transform_registry) -> None:
    """Reject configs that would crash deep into the run. Every error
    message names what to fix and where to fix it."""
    from datetime import datetime as _dt

    errors: list[str] = []

    def _is_pos_int(v) -> bool:
        # `True`/`False` are `int` instances in Python — reject explicitly.
        return isinstance(v, int) and not isinstance(v, bool) and v >= 1

    # ── dataset ──
    dataset = cfg.get("dataset") or {}
    sample_size = dataset.get("sample_size")
    if not _is_pos_int(sample_size):
        errors.append(
            "dataset.sample_size must be a positive integer. Set it in "
            "config.yaml or pass --sample-size N."
        )

    # ── source ──
    source = cfg.get("source") or {}
    source_type = source.get("type")
    if not source_type:
        errors.append(
            "source.type is required. Pass --preset NAME or define a "
            "`source:` block in your config."
        )
    elif source_type not in source_registry:
        known = ", ".join(sorted(source_registry))
        errors.append(f"source.type={source_type!r} unknown. Known: {known}.")
    else:
        # Source-specific shape checks.
        if source_type == "huggingface":
            if not source.get("name"):
                errors.append("source.name is required for type=huggingface.")
            if not source.get("split"):
                errors.append("source.split is required for type=huggingface (e.g. 'train').")
        elif source_type == "jsonl":
            if not source.get("path"):
                errors.append("source.path is required for type=jsonl.")
            else:
                from pathlib import Path as _P
                if not _P(source["path"]).exists():
                    errors.append(f"source.path does not exist: {source['path']!r}")
        fmap = source.get("field_map")
        if fmap is not None and not isinstance(fmap, dict):
            errors.append("source.field_map must be a mapping {logical: source_column}.")

    # ── transform ──
    transform = cfg.get("transform") or {}
    transform_type = transform.get("type")
    if not transform_type:
        errors.append(
            "transform.type is required. Pass --preset NAME or define a "
            "`transform:` block in your config."
        )
    elif transform_type not in transform_registry:
        known = ", ".join(sorted(transform_registry))
        errors.append(
            f"transform.type={transform_type!r} unknown. Known: {known}."
        )
    else:
        # llm_synth has the most config surface — validate eagerly.
        if transform_type == "llm_synth":
            cat = (transform.get("category") or {})
            labels = cat.get("labels")
            if not isinstance(labels, list) or not labels:
                errors.append(
                    "transform.category.labels must be a non-empty list of strings "
                    "for type=llm_synth."
                )
            elif not all(isinstance(lbl, str) and lbl for lbl in labels):
                errors.append(
                    "transform.category.labels must contain non-empty strings only."
                )
            date_cfg = transform.get("date") or {}
            for key in ("range_start", "range_end"):
                if not date_cfg.get(key):
                    errors.append(
                        f"transform.date.{key} is required for type=llm_synth."
                    )
            try:
                if date_cfg.get("range_start") and date_cfg.get("range_end"):
                    s = _dt.fromisoformat(date_cfg["range_start"])
                    e = _dt.fromisoformat(date_cfg["range_end"])
                    if s > e:
                        errors.append(
                            "transform.date.range_start must be <= range_end."
                        )
            except (TypeError, ValueError) as ex:
                errors.append(f"transform.date.* must be ISO-8601 dates: {ex}")
            cleaners = transform.get("cleaners") or []
            if cleaners and not isinstance(cleaners, list):
                errors.append("transform.cleaners must be a list of dicts.")
            else:
                for i, spec in enumerate(cleaners):
                    if not isinstance(spec, dict) or not spec.get("type"):
                        errors.append(
                            f"transform.cleaners[{i}] must be a dict with a 'type' key."
                        )

    # ── llm (only when transform needs it) ──
    if llm_required:
        llm = cfg.get("llm") or {}
        for key in ("base_url", "model"):
            if not llm.get(key):
                errors.append(
                    f"llm.{key} is required because transform.type={transform_type!r} "
                    f"uses an LLM. Set it in defaults.yaml or config.yaml."
                )
        api_key = llm.get("api_key")
        if not api_key:
            errors.append(
                "llm.api_key resolved to empty. If using ${OPENAI_API_KEY}, "
                "export it in your shell."
            )

    # ── wordpress (only when not dry-run) ──
    load = cfg.get("load") or {}
    if not load.get("dry_run", True):
        wp = cfg.get("wordpress") or {}
        for key in ("url", "username", "app_password"):
            if not wp.get(key):
                errors.append(
                    f"wordpress.{key} is required for live uploads. Set it "
                    f"in config.yaml, or pass --dry-run."
                )
        url = wp.get("url") or ""
        if url and not url.startswith(("http://", "https://")):
            errors.append(
                f"wordpress.url must include a scheme (got {url!r}). "
                "Use 'https://...' (or 'http://localhost' for local dev)."
            )

    if errors:
        bullet = "\n  - "
        raise ConfigError("Config validation failed:" + bullet + bullet.join(errors))
