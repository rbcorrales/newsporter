#!/usr/bin/env python3
"""Pipeline: HuggingFace news dataset -> WordPress posts with synthesized metadata.

Phase 1 only (extract/transform/load). Embedding is a separate pass.
"""

import argparse
import calendar
import html
import json
import logging
import os
import queue
import random
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
import yaml
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


_SENTINEL = object()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_pricing(path: Path) -> dict:
    """Optional OpenAI pricing table. Models absent from the file resolve to
    $0 in cost summaries, which is the right behavior for local servers and
    proxy endpoints where there's no per-token bill."""
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


@dataclass
class RawRow:
    dataset_id: str
    article: str
    highlights: str


@dataclass
class Post:
    title: str
    content: str
    category: str
    author: str
    date_gmt: str
    source_id: str


def extract(cfg: dict) -> list[RawRow]:
    d = cfg["dataset"]
    ds = load_dataset(
        d["name"],
        d["config"],
        split=d["split"],
        cache_dir=d.get("cache_dir"),
    )
    rnd = random.Random(d["seed"])
    total = len(ds)
    indices = rnd.sample(range(total), min(d["sample_size"], total))
    rows: list[RawRow] = []
    for i in indices:
        r = ds[i]
        rows.append(RawRow(
            dataset_id=str(r.get("id") or i),
            article=r["article"],
            highlights=r.get("highlights", ""),
        ))
    return rows


class LLM:
    """OpenAI-compatible chat client (LM Studio, Ollama, llama.cpp, vLLM, or
    Automattic's AI API Proxy at https://public-api.wordpress.com/wpcom/v2/
    ai-api-proxy/v1, which requires an X-WPCOM-AI-Feature header)."""

    def __init__(self, cfg: dict, pricing: Optional[dict] = None):
        api_key = cfg.get("api_key") or "not-needed"
        # Allow ${ENV_VAR} substitution so secrets stay out of YAML.
        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            api_key = os.environ.get(api_key[2:-1], "") or "not-needed"

        # Optional default headers (e.g. a feature flag or tracking header
        # required by some proxy endpoints).
        headers = cfg.get("headers") or {}
        client_kwargs = {"base_url": cfg["base_url"], "api_key": api_key}
        if headers:
            client_kwargs["default_headers"] = headers

        self.client = OpenAI(**client_kwargs)
        self.model = cfg["model"]
        self.temperature = cfg.get("temperature", 0.3)
        self.timeout = cfg.get("timeout", 60)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._calls_lock = threading.Lock()
        self.pricing = pricing or {}

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        response_format: Optional[dict] = None,
    ) -> str:
        with self._calls_lock:
            self.calls += 1
        # GPT-5 and o-series have two API quirks vs older chat models:
        # - require `max_completion_tokens` instead of `max_tokens`
        # - reject any `temperature` other than the default (1)
        is_reasoning = bool(re.match(r"^(gpt-5|o[1-9])", self.model))
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "timeout": self.timeout,
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = self.temperature
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self.client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            with self._calls_lock:
                self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        return (resp.choices[0].message.content or "").strip()

    def cost_summary(self) -> dict:
        rates = self.pricing.get(self.model, {})
        in_rate = float(rates.get("input", 0.0) or 0.0)
        out_rate = float(rates.get("output", 0.0) or 0.0)
        usd = (self.input_tokens * in_rate + self.output_tokens * out_rate) / 1_000_000
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "est_usd": round(usd, 6),
            "priced": bool(rates),
        }


# Newswire dateline like "WASHINGTON (CNN) -- " or "(CNN) -- ".
_PUBLICATION_PREFIX_RE = re.compile(r"^(?:[A-Z][A-Z\s,]*\s+)?\(CNN\)\s*(?:--\s*)?")
# Byline lead like "By . Author Name . UPDATED: . HH:MM ZZZ, DD Month YYYY . ".
_BYLINE_LEAD_RE = re.compile(
    r"^By\s+\.\s+.*?UPDATED:\s+\.\s+\d{1,2}:\d{2}\s+\w+,\s+\d{1,2}\s+\w+\s+\d{4}\s+\.\s+",
    re.DOTALL,
)


def clean_body(text: str, strip_lead_prefixes: bool) -> str:
    if strip_lead_prefixes:
        text = _PUBLICATION_PREFIX_RE.sub("", text, count=1)
        text = _BYLINE_LEAD_RE.sub("", text, count=1)
    return text.strip()


def to_block_content(text: str) -> str:
    """Convert plain text to Gutenberg block markup (one wp:paragraph per line)."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    blocks = [
        f"<!-- wp:paragraph -->\n<p>{html.escape(p)}</p>\n<!-- /wp:paragraph -->"
        for p in paragraphs
    ]
    return "\n\n".join(blocks)


def title_from_highlights(highlights: str, max_len: int) -> str:
    first = (highlights.strip().split(".") or [""])[0].strip()
    if len(first) > max_len:
        first = first[:max_len].rsplit(" ", 1)[0] + "..."
    return first or "Untitled"


TITLE_PREFIXES = ("headline:", "title:")


def _clean_title_output(out: str, max_len: int) -> str:
    out = out.split("\n")[0].strip()
    for prefix in TITLE_PREFIXES:
        if out.lower().startswith(prefix):
            out = out[len(prefix):].strip()
            break
    out = out.strip('"').strip("'").strip("*").strip()
    if len(out) > max_len:
        out = out[:max_len].rsplit(" ", 1)[0] + "..."
    return out


def title_from_llm(
    lm: LLM, article: str, highlights: str, max_len: int, max_tokens: int
) -> str:
    system = (
        "Write one short factual news headline for the article below.\n"
        "Target 60 to 80 characters.\n"
        "Use sentence case. This means: capitalize the first word, AND "
        "always capitalize proper nouns (names of people, places, countries, "
        "cities, organizations, brands, products, teams). Every other word "
        "is lowercase. Do NOT use Title Case where every word starts uppercase.\n"
        "\n"
        "Examples of the correct style:\n"
        "  Powerful earthquake kills at least 15 in central Peru\n"
        "  Del Potro loses opening round match at Thailand Open in Bangkok\n"
        "  Murderer posts jail selfies, asks for credit on Facebook\n"
        "  Apple unveils new MacBook Pro at event in Cupertino\n"
        "\n"
        "Output only the headline text. No quotes, no prefix, no label, no commentary."
    )
    user = f"Article:\n\n{article[:4000]}\n\nHeadline:"
    out = _clean_title_output(lm.chat(system, user, max_tokens), max_len)
    if out:
        return out
    if highlights:
        return title_from_highlights(highlights, max_len)
    return "Untitled"


def category_from_llm(lm: LLM, article: str, labels: list[str], max_tokens: int) -> str:
    labels_str = ", ".join(labels)
    system = (
        f"You are a news category classifier. Given an article, reply with exactly "
        f"one label from this list: {labels_str}. Output only the label."
    )
    user = f"Article:\n\n{article[:3000]}\n\nCategory:"
    out = lm.chat(system, user, max_tokens)
    lower = out.lower()
    for lbl in labels:
        if lbl.lower() in lower:
            return lbl
    return labels[0]


_COMBINED_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


_DATE_YM_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _parse_combined_output(
    raw: str, labels: list[str], max_title_len: int
) -> tuple[str, str, str]:
    """Extract (title, category, date_ym) from a combined LLM response.
    Tolerant of markdown fences, leading commentary, and missing/invalid
    fields. date_ym is a YYYY-MM string when valid, else empty."""
    text = raw.strip()
    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    title = ""
    category = ""
    date_ym = ""
    parsed = None
    # First try a clean json.loads on the whole thing.
    try:
        parsed = json.loads(text)
    except Exception:
        # Fall back to extracting the first {...} block.
        m = _COMBINED_JSON_RE.search(text)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
    if isinstance(parsed, dict):
        title = str(parsed.get("title") or "").strip()
        category = str(parsed.get("category") or "").strip()
        raw_date = str(parsed.get("date") or "").strip()
        if _DATE_YM_RE.match(raw_date):
            date_ym = raw_date
    # Normalize title to existing rules.
    title = _clean_title_output(title, max_title_len) if title else ""
    # Normalize category against the allowed list (case-insensitive).
    matched_category = ""
    if category:
        cat_lower = category.lower()
        for lbl in labels:
            if lbl.lower() == cat_lower or lbl.lower() in cat_lower:
                matched_category = lbl
                break
    return title, matched_category, date_ym


def combined_from_llm(
    lm: LLM,
    article: str,
    highlights: str,
    labels: list[str],
    max_title_len: int,
    max_tokens: int,
    date_window: tuple[str, str],
) -> tuple[str, str, str]:
    """One LLM call -> (title, category, date_ym). date_ym is YYYY-MM or empty.
    Roughly halves per-post LLM time vs separate title/category calls.
    date_window is (min_yyyy_mm, max_yyyy_mm) used to instruct the model on
    the plausible publication range."""
    labels_str = ", ".join(labels)
    win_start, win_end = date_window
    system = (
        "You are a news editor. Read the article and output ONE valid JSON "
        "object with exactly these keys:\n"
        '  - "title": a short factual news headline, 60 to 80 characters. '
        "Sentence case (capitalize the first word and proper nouns; "
        "lowercase everything else). Do NOT use Title Case.\n"
        f'  - "category": exactly one label from this list: {labels_str}. '
        "Output the label string verbatim.\n"
        '  - "date": plausible publication year and month as YYYY-MM. '
        f"The article was published between {win_start} and {win_end}. "
        "Infer from named events, people in office, technology mentioned, "
        "and any explicit dates in the body. If unsure, pick a plausible "
        f"year and month within the {win_start} to {win_end} window.\n"
        "\n"
        "No markdown fences. No commentary. No extra fields.\n"
        "\n"
        "Examples:\n"
        '{"title": "Powerful earthquake kills at least 15 in central Peru", "category": "World", "date": "2007-08"}\n'
        '{"title": "Del Potro loses opening round at Thailand Open in Bangkok", "category": "Sports", "date": "2009-09"}\n'
        '{"title": "Murderer posts jail selfies, asks for credit on Facebook", "category": "Crime", "date": "2014-06"}'
    )
    user = f"Article:\n\n{article[:4000]}\n\nJSON:"
    # Structured Outputs: forces valid JSON and constrains category to the
    # allowed enum. Supported on gpt-4o-mini, gpt-5-*, and most modern OpenAI
    # models. For local/older endpoints that reject it, the kwarg is simply
    # not sent because we only set response_format when the LLM is OpenAI.
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "title_category_date",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "category": {"type": "string", "enum": labels},
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
    use_structured = "openai.com" in (lm.client.base_url.host or "")
    raw = lm.chat(system, user, max_tokens, response_format=response_format if use_structured else None)
    title, category, date_ym = _parse_combined_output(raw, labels, max_title_len)
    if not title:
        title = title_from_highlights(highlights, max_title_len) if highlights else "Untitled"
    if not category:
        category = labels[0]
    return title, category, date_ym


def random_date(start: str, end: str, rng: random.Random) -> str:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    delta = (e - s).total_seconds()
    offset = rng.uniform(0, max(delta, 0))
    d = s + timedelta(seconds=offset)
    return d.strftime("%Y-%m-%dT%H:%M:%S")


def _ym_from_iso(date_str: str) -> tuple[int, int]:
    """Extract (year, month) from an ISO date string like '2007-04-01'."""
    d = datetime.fromisoformat(date_str)
    return (d.year, d.month)


def date_iso_from_year_month(
    date_ym: str,
    rng: random.Random,
    min_ym: tuple[int, int],
    max_ym: tuple[int, int],
) -> Optional[str]:
    """Convert a YYYY-MM string into a full ISO datetime by picking a random
    day within the month (using calendar.monthrange so Feb 30 / Apr 31 can
    never happen) and a random hour/minute/second. Returns None when the
    input is malformed or outside [min_ym, max_ym]."""
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
    day = rng.randint(1, max_day)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return datetime(year, month, day, hour, minute, second).strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class Prefab:
    """Per-row data picked up-front in the main thread (non-thread-safe rng)."""

    row: RawRow
    author: str
    date_gmt: str
    category_fallback: str


class TransformCache:
    """Append-only JSONL cache of transformed Posts keyed by source_id.

    The cache lets you re-upload the same corpus to a different WordPress
    site without paying the LLM cost again. Each entry carries a "sig"
    (model + combined-flag + prompt version) so a config change invalidates
    stale entries automatically. Delete the file to start fresh.
    """

    PROMPT_VERSION = "v3-combined-structured-date"

    def __init__(self, path: Path, sig: str, log: logging.Logger) -> None:
        self.path = path
        self.sig = sig
        self.log = log
        self._lock = threading.Lock()
        self._by_source: dict[str, Post] = {}
        self._loaded = 0
        self._stale = 0
        self._fh = None  # opened lazily on first write
        self._load()

    @staticmethod
    def signature(cfg: dict) -> str:
        t = cfg.get("transform", {})
        llm = cfg.get("llm", {})
        return "|".join([
            f"model={llm.get('model','?')}",
            f"combined={'1' if t.get('combined') else '0'}",
            f"title_method={t.get('title', {}).get('method','?')}",
            f"cat_method={t.get('category', {}).get('method','?')}",
            f"prompts={TransformCache.PROMPT_VERSION}",
        ])

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("sig") != self.sig:
                    self._stale += 1
                    continue
                post_d = entry.get("post") or {}
                try:
                    self._by_source[post_d["source_id"]] = Post(**post_d)
                    self._loaded += 1
                except Exception:
                    continue
        self.log.info(
            "Transform cache: %d hit-eligible entries loaded, %d stale (different sig). path=%s",
            self._loaded,
            self._stale,
            self.path,
        )

    def get(self, source_id: str) -> Optional[Post]:
        return self._by_source.get(source_id)

    def put(self, post: Post) -> None:
        with self._lock:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            self._by_source[post.source_id] = post
            self._fh.write(json.dumps({"sig": self.sig, "post": asdict(post)}) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def prefab_rows(rows: list[RawRow], cfg: dict) -> list[Prefab]:
    """Pre-pick the random-but-deterministic fields so worker threads can run
    without sharing the seeded rng."""
    t = cfg["transform"]
    rng = random.Random(cfg["dataset"]["seed"])
    out: list[Prefab] = []
    for row in rows:
        author = rng.choice(t["author"]["pool"])
        date_gmt = random_date(t["date"]["range_start"], t["date"]["range_end"], rng)
        if t["category"]["method"] == "random":
            category_fallback = rng.choice(t["category"]["labels"])
        else:
            category_fallback = t["category"]["labels"][0]
        out.append(Prefab(row=row, author=author, date_gmt=date_gmt, category_fallback=category_fallback))
    return out


def transform_one(pf: Prefab, cfg: dict, lm: Optional[LLM]) -> Post:
    """Transform a single row. Safe to call from multiple threads."""
    t = cfg["transform"]
    body = clean_body(pf.row.article, t["clean_body"]["strip_lead_prefixes"])

    title_method = t["title"]["method"]
    cat_method = t["category"]["method"]
    combined = bool(t.get("combined"))

    # Combined path: one LLM call returns both title and category as JSON.
    # Roughly cuts per-post LLM time in half. Only kicks in when both fields
    # are LLM-driven; otherwise fall through to the per-field path.
    date_gmt = pf.date_gmt
    if combined and title_method == "llm" and cat_method == "llm":
        assert lm is not None
        max_tokens = cfg["llm"].get(
            "max_tokens_combined",
            cfg["llm"].get("max_tokens_title", 40) + cfg["llm"].get("max_tokens_category", 8) + 16,
        )
        win_start = t["date"]["range_start"][:7]  # YYYY-MM
        win_end = t["date"]["range_end"][:7]
        title, category, date_ym = combined_from_llm(
            lm,
            body,
            pf.row.highlights,
            t["category"]["labels"],
            t["title"]["max_length"],
            max_tokens,
            (win_start, win_end),
        )
        if date_ym:
            # Use the LLM-suggested year/month with a random day+time, clamped
            # to the configured fallback window. Anything outside the window
            # or otherwise malformed falls back to the prefab's random date.
            min_ym = _ym_from_iso(t["date"]["range_start"])
            max_ym = _ym_from_iso(t["date"]["range_end"])
            llm_date = date_iso_from_year_month(
                date_ym, random.Random(pf.row.dataset_id), min_ym, max_ym
            )
            if llm_date:
                date_gmt = llm_date
    else:
        if title_method == "llm":
            assert lm is not None
            title = title_from_llm(
                lm,
                body,
                pf.row.highlights,
                t["title"]["max_length"],
                cfg["llm"]["max_tokens_title"],
            )
        else:
            title = title_from_highlights(pf.row.highlights, t["title"]["max_length"])

        if cat_method == "llm":
            assert lm is not None
            category = category_from_llm(
                lm, body, t["category"]["labels"], cfg["llm"]["max_tokens_category"]
            )
        else:
            category = pf.category_fallback

    return Post(
        title=title,
        content=body,
        category=category,
        author=pf.author,
        date_gmt=date_gmt,
        source_id=pf.row.dataset_id,
    )


class WordPressClient:
    def __init__(self, cfg: dict):
        self.url = cfg["url"].rstrip("/")
        self.session = requests.Session()
        self.session.auth = (cfg["username"], cfg["app_password"])
        self.session.headers.update({"User-Agent": "newsporter/0.1"})

    def _url(self, path: str) -> str:
        return f"{self.url}/wp-json/wp/v2{path}"

    def ensure_categories(self, names: list[str]) -> dict[str, int]:
        mapping: dict[str, int] = {}
        r = self.session.get(self._url("/categories"), params={"per_page": 100})
        r.raise_for_status()
        existing = {c["name"]: c["id"] for c in r.json()}
        for name in names:
            if name in existing:
                mapping[name] = existing[name]
            else:
                r = self.session.post(self._url("/categories"), json={"name": name})
                r.raise_for_status()
                mapping[name] = r.json()["id"]
        return mapping

    def ensure_authors(self, names: list[str], role: str, email_domain: str) -> dict[str, int]:
        mapping: dict[str, int] = {}
        r = self.session.get(
            self._url("/users"), params={"per_page": 100, "context": "edit"}
        )
        r.raise_for_status()
        by_slug = {u["slug"]: u["id"] for u in r.json()}
        for name in names:
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "author"
            if slug in by_slug:
                mapping[name] = by_slug[slug]
                continue
            parts = name.split(" ", 1)
            first = parts[0]
            last = parts[1] if len(parts) > 1 else ""
            payload = {
                "username": slug,
                "email": f"{slug}@{email_domain}",
                "password": secrets.token_urlsafe(24),
                "roles": [role],
                "name": name,
                "first_name": first,
                "last_name": last,
            }
            resp = self.session.post(self._url("/users"), json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"POST /users {resp.status_code} for '{name}': {resp.text[:300]}"
                )
            mapping[name] = resp.json()["id"]
        return mapping

    def create_post(
        self,
        post: Post,
        category_id: int,
        author_id: Optional[int],
        status: str,
        prepend_byline: bool,
    ) -> int:
        content = to_block_content(post.content)
        if prepend_byline:
            byline_block = (
                "<!-- wp:paragraph -->\n"
                f"<p><em>By {html.escape(post.author)}</em></p>\n"
                "<!-- /wp:paragraph -->"
            )
            content = f"{byline_block}\n\n{content}"
        payload: dict = {
            "title": post.title,
            "content": content,
            "status": status,
            "date_gmt": post.date_gmt,
            "categories": [category_id],
            "meta": {
                "_newsporter_source_id": post.source_id,
                "_newsporter_byline": post.author,
            },
        }
        if author_id is not None:
            payload["author"] = author_id
        r = self.session.post(self._url("/posts"), json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {r.status_code}: {r.text[:300]}")
        return r.json()["id"]


def upload_one(
    p: Post,
    cfg: dict,
    cat_mapping: dict[str, int],
    author_mapping: dict[str, int],
    wp: WordPressClient,
) -> dict:
    lcfg = cfg["load"]
    prepend = cfg["transform"]["author"]["prepend_to_content"]
    cid = cat_mapping[p.category]
    aid = author_mapping.get(p.author) if author_mapping else None
    attempts = lcfg["retry"]["attempts"]
    backoff = lcfg["retry"]["backoff_seconds"]
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            pid = wp.create_post(p, cid, aid, lcfg["post_status"], prepend)
            return {"source_id": p.source_id, "post_id": pid, "ok": True}
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    return {"source_id": p.source_id, "error": str(last_err), "ok": False}


def run_pipeline(
    prefabs: list[Prefab],
    cfg: dict,
    lm: Optional[LLM],
    cat_mapping: dict[str, int],
    author_mapping: dict[str, int],
    wp: Optional[WordPressClient],
    log: logging.Logger,
    run_dir: Path,
    cache: Optional[TransformCache] = None,
) -> tuple[list[Post], list[dict]]:
    """Streaming pipeline: transform workers push posts into a bounded queue,
    upload workers consume them. Per-post artifacts are flushed to JSONL as
    they complete so long runs stay observable.
    """
    lcfg = cfg["load"]
    tcfg = cfg["transform"]
    transform_workers = int(tcfg.get("concurrency", 4))
    upload_workers = int(lcfg.get("concurrency", 4))
    queue_cap = max(upload_workers * 4, 16)
    dry_run = bool(lcfg.get("dry_run")) or wp is None

    posts: list[Post] = []
    results: list[dict] = []
    posts_lock = threading.Lock()
    results_lock = threading.Lock()

    post_q: "queue.Queue[object]" = queue.Queue(maxsize=queue_cap)

    posts_path = run_dir / "posts.jsonl"
    results_path = run_dir / "results.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    posts_fh = posts_path.open("w", encoding="utf-8")
    results_fh = results_path.open("w", encoding="utf-8") if not dry_run else None

    tr_bar = tqdm(total=len(prefabs), desc="Transform", position=0, leave=True)
    up_bar = tqdm(total=len(prefabs), desc="Upload   ", position=1, leave=True)

    def record_post(post: Post) -> None:
        with posts_lock:
            posts.append(post)
            posts_fh.write(json.dumps(asdict(post)) + "\n")
            posts_fh.flush()

    def record_result(result: dict) -> None:
        with results_lock:
            results.append(result)
            if results_fh is not None:
                results_fh.write(json.dumps(result) + "\n")
                results_fh.flush()
            up_bar.update(1)

    def transform_worker(pf: Prefab) -> None:
        try:
            cached = cache.get(pf.row.dataset_id) if cache is not None else None
            if cached is not None:
                # Cache hit: skip the LLM entirely. The cached Post carries the
                # original author/date/content/title/category from a prior run.
                post = cached
            else:
                post = transform_one(pf, cfg, lm)
                if cache is not None:
                    cache.put(post)
            record_post(post)
            post_q.put(post)
        except Exception as e:
            log.warning("Transform failed for %s: %s", pf.row.dataset_id, e)
            post_q.put(("__err__", pf.row.dataset_id, str(e)))
        finally:
            tr_bar.update(1)

    def upload_worker() -> None:
        while True:
            item = post_q.get()
            try:
                if item is _SENTINEL:
                    return
                if isinstance(item, tuple) and item and item[0] == "__err__":
                    _, source_id, err = item
                    record_result({"source_id": source_id, "error": f"transform: {err}", "ok": False})
                    continue
                if dry_run:
                    assert isinstance(item, Post)
                    record_result({"source_id": item.source_id, "dry_run": True})
                    continue
                assert isinstance(item, Post) and wp is not None
                result = upload_one(item, cfg, cat_mapping, author_mapping, wp)
                record_result(result)
            finally:
                post_q.task_done()

    try:
        with ThreadPoolExecutor(max_workers=upload_workers, thread_name_prefix="upload") as up_ex, \
             ThreadPoolExecutor(max_workers=transform_workers, thread_name_prefix="xform") as tr_ex:
            up_futs = [up_ex.submit(upload_worker) for _ in range(upload_workers)]
            tr_futs = [tr_ex.submit(transform_worker, pf) for pf in prefabs]
            for f in as_completed(tr_futs):
                # Surface any unexpected exception that escaped the worker.
                f.result()
            # Signal upload workers to drain and stop.
            for _ in range(upload_workers):
                post_q.put(_SENTINEL)
            for f in up_futs:
                f.result()
    finally:
        tr_bar.close()
        up_bar.close()
        posts_fh.close()
        if results_fh is not None:
            results_fh.close()

    return posts, results


def purge_all_posts(wp: "WordPressClient", log: logging.Logger) -> tuple[int, int]:
    """Permanently delete every post (any status) on the WordPress site."""
    deleted = 0
    failed = 0
    while True:
        r = wp.session.get(
            wp._url("/posts"),
            params={"per_page": 100, "status": "any", "context": "edit"},
        )
        if r.status_code >= 400:
            log.error("List posts failed (%d): %s", r.status_code, r.text[:200])
            break
        posts = r.json()
        if not posts:
            break
        progress = 0
        for p in posts:
            resp = wp.session.delete(
                wp._url(f"/posts/{p['id']}"), params={"force": "true"}
            )
            if resp.status_code >= 400:
                log.warning(
                    "Delete %d failed (%d): %s",
                    p["id"],
                    resp.status_code,
                    resp.text[:120],
                )
                failed += 1
            else:
                deleted += 1
                progress += 1
                log.info("Deleted post %d: %s", p["id"], p["title"]["rendered"][:60])
        if progress == 0:
            break
    return deleted, failed


def estimate_embedding_cost(posts: list[Post], costs_cfg: dict) -> dict:
    # Rough heuristic: ~4 chars per token for English prose.
    total_chars = sum(len(p.content) for p in posts)
    est_tokens = total_chars // 4
    e = costs_cfg["embedding"]
    usd = est_tokens * e["usd_per_1m_tokens"] / 1_000_000
    return {
        "posts": len(posts),
        "est_tokens": est_tokens,
        "provider": e["provider"],
        "model": e["model"],
        "dimensions": e["dimensions"],
        "est_usd": round(usd, 4),
    }


def write_run_summary(
    run_dir: Path,
    cfg: dict,
    rows: list[RawRow],
    posts: list[Post],
    results: list[dict],
    lm_calls: int,
    embedding_estimate: dict,
    elapsed_sec: float,
    chat_cost: Optional[dict] = None,
) -> dict:
    """Per-post artifacts are streamed to posts.jsonl / results.jsonl during the
    run. Here we only snapshot the aggregate summary."""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": cfg["dataset"],
        "sample_size_requested": cfg["dataset"]["sample_size"],
        "rows_fetched": len(rows),
        "posts_prepared": len(posts),
        "posts_uploaded": sum(1 for r in results if r.get("ok")),
        "posts_failed": sum(1 for r in results if r.get("ok") is False),
        "llm_calls": lm_calls,
        "elapsed_sec": round(elapsed_sec, 2),
        "transform_concurrency": int(cfg["transform"].get("concurrency", 4)),
        "upload_concurrency": int(cfg["load"].get("concurrency", 4)),
        "embedding_estimate": embedding_estimate,
    }
    if chat_cost is not None:
        summary["chat_cost"] = chat_cost
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("newsporter")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import news corpus into WordPress")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Skip upload")
    parser.add_argument("--sample-size", type=int, help="Override sample size")
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Delete all posts on the WP site (destructive, exits after)",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.dry_run:
        cfg["load"]["dry_run"] = True
    if args.sample_size is not None:
        cfg["dataset"]["sample_size"] = args.sample_size

    log = setup_logging(cfg["logging"]["level"])

    if args.purge:
        log.info("PURGE mode: deleting ALL posts on %s", cfg["wordpress"]["url"])
        wp = WordPressClient(cfg["wordpress"])
        deleted, failed = purge_all_posts(wp, log)
        log.info("Purge done. deleted=%d failed=%d", deleted, failed)
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(cfg["logging"]["run_dir"]) / run_id
    log.info("Run ID: %s", run_id)

    log.info(
        "Extracting %d rows from %s [%s/%s]",
        cfg["dataset"]["sample_size"],
        cfg["dataset"]["name"],
        cfg["dataset"]["config"],
        cfg["dataset"]["split"],
    )
    rows = extract(cfg)
    log.info("Fetched %d rows", len(rows))

    uses_lm = (
        cfg["transform"]["title"]["method"] == "llm"
        or cfg["transform"]["category"]["method"] == "llm"
    )
    pricing = load_pricing(Path(__file__).parent / "pricing.yaml")
    lm: Optional[LLM] = None
    if uses_lm:
        lm = LLM(cfg["llm"], pricing=pricing)
        log.info("LLM: %s @ %s", cfg["llm"]["model"], cfg["llm"]["base_url"])

    author_cfg = cfg["transform"]["author"]
    if cfg["load"]["dry_run"]:
        wp = None
        cat_mapping = {
            lbl: 1000 + i for i, lbl in enumerate(cfg["transform"]["category"]["labels"])
        }
        author_mapping: dict[str, int] = {}
    else:
        log.info("Connecting to WordPress at %s", cfg["wordpress"]["url"])
        wp = WordPressClient(cfg["wordpress"])
        if cfg["transform"]["category"]["ensure_created"]:
            log.info("Ensuring categories: %s", cfg["transform"]["category"]["labels"])
            cat_mapping = wp.ensure_categories(cfg["transform"]["category"]["labels"])
        else:
            cat_mapping = {}
        if author_cfg.get("create_wp_users"):
            log.info("Ensuring %d WP authors", len(author_cfg["pool"]))
            author_mapping = wp.ensure_authors(
                author_cfg["pool"],
                author_cfg.get("role", "author"),
                author_cfg.get("email_domain", "newsporter.local"),
            )
        else:
            author_mapping = {}

    prefabs = prefab_rows(rows, cfg)

    # Optional transform cache: store fully-transformed Posts keyed by
    # source_id so a re-run (e.g. uploading the same corpus to a different
    # site) can skip the LLM entirely.
    cache: Optional[TransformCache] = None
    cache_cfg = cfg["transform"].get("cache") or {}
    if cache_cfg.get("enabled"):
        cache_path = Path(cache_cfg.get("path", "data/transforms_cache.jsonl"))
        cache = TransformCache(cache_path, TransformCache.signature(cfg), log)

    log.info(
        "Pipeline: transform x%d, upload x%d, dry_run=%s, combined=%s, cache=%s",
        int(cfg["transform"].get("concurrency", 4)),
        int(cfg["load"].get("concurrency", 4)),
        cfg["load"]["dry_run"],
        bool(cfg["transform"].get("combined")),
        "on" if cache is not None else "off",
    )
    t0 = time.monotonic()
    try:
        posts, results = run_pipeline(
            prefabs, cfg, lm, cat_mapping, author_mapping, wp, log, run_dir, cache
        )
    finally:
        if cache is not None:
            cache.close()
    elapsed = time.monotonic() - t0

    embedding_est = estimate_embedding_cost(posts, cfg["costs"])
    log.info(
        "Embedding estimate (future phase): ~%d tokens, ~$%.4f with %s",
        embedding_est["est_tokens"],
        embedding_est["est_usd"],
        embedding_est["model"],
    )

    chat_cost = lm.cost_summary() if lm is not None else None
    if chat_cost is not None:
        log.info(
            "Chat usage: %s, in=%d out=%d tokens, est=$%.4f%s",
            chat_cost["model"],
            chat_cost["input_tokens"],
            chat_cost["output_tokens"],
            chat_cost["est_usd"],
            "" if chat_cost["priced"] else " (model not in pricing.yaml)",
        )
    summary = write_run_summary(
        run_dir, cfg, rows, posts, results,
        lm.calls if lm else 0, embedding_est, elapsed,
        chat_cost=chat_cost,
    )
    log.info(
        "Done in %.1fs. Prepared=%d uploaded=%d failed=%d llm_calls=%d",
        elapsed,
        summary["posts_prepared"],
        summary["posts_uploaded"],
        summary["posts_failed"],
        summary["llm_calls"],
    )
    log.info("Artifacts: %s", run_dir)


if __name__ == "__main__":
    main()
