"""LLM-synthesizing transformer.

Given a body of text, asks an OpenAI-compatible chat model for a JSON
object with `title`, `category`, and `date` (YYYY-MM). Uses Structured
Outputs against OpenAI proper to guarantee valid JSON and constrain the
category to an enum; falls back to tolerant JSON parsing on other
endpoints.

Config schema (under `transform:`):
    type: llm_synth
    body_field: "body"               # logical name from source.field_map
    summary_field: "summary"         # optional, used as title fallback
    title:
      max_length: 80
    category:
      labels: [News, Politics, ...]
    date:
      range_start: "2007-04-01"      # ISO
      range_end:   "2015-04-30"
    cleaners:                        # optional, list of declarative steps
      - { type: regex_strip, pattern: "^\\(SOURCE_TAG\\)\\s*--\\s*" }
    prompt_template: |               # optional, has a sensible default
      You are a news editor...
    max_tokens_combined: 120
"""

from __future__ import annotations

import calendar
import hashlib
import json
import random
import re
from datetime import datetime, timedelta

from ..llm import LLM
from ..models import Post, Prefab, RawRow
from .base import Transformer
from .cleaners import apply_cleaners

_TITLE_PREFIXES = ("headline:", "title:")
_DATE_YM_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_FENCE_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


DEFAULT_PROMPT_TEMPLATE = """\
You are a news editor. Read the article and output ONE valid JSON object \
with exactly these keys:
  - "title": a short factual news headline, 60 to 80 characters. Sentence \
case (capitalize the first word and proper nouns; lowercase everything \
else). Do NOT use Title Case.
  - "category": exactly one label from this list: {labels}. Output the \
label string verbatim.
  - "date": plausible publication year and month as YYYY-MM. The article \
was published between {win_start} and {win_end}. Infer from named events, \
people in office, technology mentioned, and any explicit dates in the body. \
If unsure, pick a plausible year and month within the {win_start} to \
{win_end} window.

No markdown fences. No commentary. No extra fields.

Examples:
{{"title": "Powerful earthquake kills at least 15 in central Peru", \
"category": "{example_label}", "date": "{example_date}"}}
"""


def _ym_from_iso(date_str: str) -> tuple[int, int]:
    d = datetime.fromisoformat(date_str)
    return (d.year, d.month)


def _random_date(start_iso: str, end_iso: str, rng: random.Random) -> str:
    s = datetime.fromisoformat(start_iso)
    e = datetime.fromisoformat(end_iso)
    delta = (e - s).total_seconds()
    offset = rng.uniform(0, max(delta, 0))
    return (s + timedelta(seconds=offset)).strftime("%Y-%m-%dT%H:%M:%S")


def _date_iso_from_year_month(
    date_ym: str,
    rng: random.Random,
    min_ym: tuple[int, int],
    max_ym: tuple[int, int],
) -> str | None:
    """Convert YYYY-MM into a full ISO datetime by picking a random valid
    day (calendar.monthrange handles leap years and 30/31-day months) and
    a random time. Returns None when malformed or outside [min_ym, max_ym]."""
    try:
        year_str, month_str = date_ym.split("-", 1)
        year = int(year_str)
        month = int(month_str)
    except (ValueError, AttributeError):
        return None
    if not (1 <= month <= 12):
        return None
    if (year, month) < min_ym or (year, month) > max_ym:
        return None
    max_day = calendar.monthrange(year, month)[1]
    return datetime(
        year,
        month,
        rng.randint(1, max_day),
        rng.randint(0, 23),
        rng.randint(0, 59),
        rng.randint(0, 59),
    ).strftime("%Y-%m-%dT%H:%M:%S")


def _clean_title_output(out: str, max_len: int) -> str:
    out = out.split("\n")[0].strip()
    for prefix in _TITLE_PREFIXES:
        if out.lower().startswith(prefix):
            out = out[len(prefix) :].strip()
            break
    out = out.strip('"').strip("'").strip("*").strip()
    if len(out) > max_len:
        out = out[:max_len].rsplit(" ", 1)[0] + "..."
    return out


def _title_from_summary(summary: str, max_len: int) -> str:
    first = (summary.strip().split(".") or [""])[0].strip()
    if len(first) > max_len:
        first = first[:max_len].rsplit(" ", 1)[0] + "..."
    return first or "Untitled"


def _parse_combined_output(raw: str, labels: list[str], max_title_len: int) -> tuple[str, str, str]:
    """Tolerant of fences, leading commentary, missing fields. Returns
    (title, matched_category, date_ym), each possibly empty."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        m = _FENCE_JSON_RE.search(text)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
    title = ""
    category = ""
    date_ym = ""
    if isinstance(parsed, dict):
        title = str(parsed.get("title") or "").strip()
        category = str(parsed.get("category") or "").strip()
        raw_date = str(parsed.get("date") or "").strip()
        if _DATE_YM_RE.match(raw_date):
            date_ym = raw_date
    title = _clean_title_output(title, max_title_len) if title else ""
    matched = ""
    if category:
        cat_lower = category.lower()
        for lbl in labels:
            if lbl.lower() == cat_lower or lbl.lower() in cat_lower:
                matched = lbl
                break
    return title, matched, date_ym


class LLMSynthTransformer(Transformer):
    def __init__(self, cfg: dict, llm: LLM, source_identity: str = "") -> None:
        self.llm = llm
        self.source_identity = source_identity  # mixed into signature so
        # the cache can never replay a post from a different corpus that
        # happens to share a source_id.
        self.body_field = cfg.get("body_field", "body")
        self.summary_field = cfg.get("summary_field", "summary")

        title_cfg = cfg.get("title") or {}
        self.max_title_len = int(title_cfg.get("max_length", 80))

        category_cfg = cfg.get("category") or {}
        self.labels: list[str] = list(category_cfg.get("labels") or [])
        if not self.labels:
            raise ValueError("transform.category.labels must be non-empty")

        date_cfg = cfg.get("date") or {}
        self.range_start = date_cfg["range_start"]
        self.range_end = date_cfg["range_end"]
        self.min_ym = _ym_from_iso(self.range_start)
        self.max_ym = _ym_from_iso(self.range_end)

        self.cleaners: list[dict] = list(cfg.get("cleaners") or [])
        self.prompt_template = cfg.get("prompt_template") or DEFAULT_PROMPT_TEMPLATE
        self.max_tokens = int(cfg.get("max_tokens_combined", 120))

    def signature(self) -> str:
        """Hash everything that would change the model's output. Cache
        entries built under a different signature get auto-invalidated.
        Source identity is included so the same source_id from a
        different corpus can never replay a stale post."""
        material = json.dumps(
            {
                "source_identity": self.source_identity,
                "model": self.llm.model,
                "prompt_template": self.prompt_template,
                "labels": sorted(self.labels),  # order-independent invalidation
                "cleaners": self.cleaners,
                "date_range": [self.range_start, self.range_end],
                "max_tokens": self.max_tokens,
                "max_title_len": self.max_title_len,
                "body_field": self.body_field,
                "summary_field": self.summary_field,
                "schema_version": "v2-llm-synth-source-scoped",
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def _build_prompt(self) -> str:
        win_start = self.range_start[:7]
        win_end = self.range_end[:7]

        # str.format treats `{` and `}` as field markers. Labels (or any
        # interpolated value) containing literal braces would crash the
        # call. Escape them.
        def _safe(value: str) -> str:
            return str(value).replace("{", "{{").replace("}", "}}")

        return self.prompt_template.format(
            labels=", ".join(_safe(lbl) for lbl in self.labels),
            win_start=_safe(win_start),
            win_end=_safe(win_end),
            example_label=_safe(self.labels[0]),
            example_date=_safe(win_start),
        )

    def _structured_response_format(self) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "title_category_date",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "category": {"type": "string", "enum": self.labels},
                        "date": {
                            "type": "string",
                            "pattern": r"^\d{4}-(0[1-9]|1[0-2])$",
                        },
                    },
                    "required": ["title", "category", "date"],
                    "additionalProperties": False,
                },
            },
        }

    def transform(self, row: RawRow, prefab: Prefab) -> Post:
        body = row.fields.get(self.body_field, "")
        summary = row.fields.get(self.summary_field, "")
        body = apply_cleaners(body, self.cleaners).strip()

        system = self._build_prompt()
        user = f"Article:\n\n{body[:4000]}\n\nJSON:"
        response_format = (
            self._structured_response_format() if self.llm.supports_structured_outputs() else None
        )
        raw = self.llm.chat(system, user, self.max_tokens, response_format=response_format)
        title, category, date_ym = _parse_combined_output(raw, self.labels, self.max_title_len)

        if not title:
            title = _title_from_summary(summary, self.max_title_len) if summary else "Untitled"
        if not category:
            category = prefab.category_fallback or self.labels[0]

        date_gmt = prefab.date_gmt
        if date_ym:
            llm_date = _date_iso_from_year_month(
                date_ym,
                random.Random(row.source_id),
                self.min_ym,
                self.max_ym,
            )
            if llm_date:
                date_gmt = llm_date

        return Post(
            title=title,
            content=body,
            category=category,
            author=prefab.author,
            date_gmt=date_gmt,
            source_id=row.source_id,
        )


def random_date_for_range(date_cfg: dict, rng: random.Random) -> str:
    """Helper exported for the pipeline's prefab-generation step."""
    return _random_date(date_cfg["range_start"], date_cfg["range_end"], rng)
