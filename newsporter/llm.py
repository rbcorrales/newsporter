"""OpenAI-compatible chat client with per-call token tracking.

Works against any OpenAI-compatible endpoint: OpenAI direct, LM Studio,
Ollama, llama.cpp server, vLLM, Automattic's AI API Proxy, etc. The
`response_format` kwarg is only sent when the caller asks for it, so older
or non-OpenAI endpoints that reject it are unaffected.
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

try:
    # Newer SDK names. Fall back gracefully so the import doesn't break
    # users on older clients.
    from openai import APITimeoutError, RateLimitError, APIConnectionError
except ImportError:  # pragma: no cover
    RateLimitError = APITimeoutError = APIConnectionError = ()  # type: ignore


_log = logging.getLogger("newsporter")


class LLM:
    def __init__(self, cfg: dict, pricing: Optional[dict] = None) -> None:
        api_key = cfg.get("api_key") or "not-needed"
        # ${ENV_VAR} substitution keeps secrets out of YAML.
        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            api_key = os.environ.get(api_key[2:-1], "") or "not-needed"

        # Optional default headers (e.g. X-WPCOM-AI-Feature for the A8C proxy).
        headers = cfg.get("headers") or {}
        client_kwargs = {"base_url": cfg["base_url"], "api_key": api_key}
        if headers:
            client_kwargs["default_headers"] = headers

        self.client = OpenAI(**client_kwargs)
        self.model = cfg["model"]
        self.temperature = cfg.get("temperature", 0.3)
        self.timeout = cfg.get("timeout", 60)
        self.max_retries = int(cfg.get("max_retries", 5))
        self.retry_base_seconds = float(cfg.get("retry_base_seconds", 2.0))
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.retries_429 = 0
        self.retries_timeout = 0
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
        # GPT-5 / o-series have two API quirks vs older chat models:
        #   - require `max_completion_tokens` instead of `max_tokens`
        #   - reject any `temperature` other than the default (1)
        # Anchored on word boundary so `gpt-4o` doesn't match (not reasoning)
        # but `o4-mini` does. Case-insensitive in case configs paste model
        # IDs from docs as `GPT-5`.
        is_reasoning = bool(re.match(r"^(gpt-[5-9]|o\d+)(-|$)", self.model, re.IGNORECASE))
        kwargs: dict = {
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

        # Retry loop for transient OpenAI failures. Without this, a single
        # 429 mid-run drops the row for this run; resume picks it up but
        # that's a much heavier recovery path than just waiting.
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except RateLimitError as e:  # type: ignore[misc]
                last_err = e
                with self._calls_lock:
                    self.retries_429 += 1
                if attempt >= self.max_retries:
                    raise
                # Honor Retry-After header when present, else exponential
                # backoff with jitter.
                ra = self._extract_retry_after(e)
                sleep_for = ra if ra is not None else (
                    self.retry_base_seconds * (2 ** attempt)
                    + random.uniform(0, 1)
                )
                _log.warning(
                    "OpenAI 429 (attempt %d/%d). Sleeping %.1fs.",
                    attempt + 1, self.max_retries, sleep_for,
                )
                time.sleep(sleep_for)
            except (APITimeoutError, APIConnectionError) as e:  # type: ignore[misc]
                last_err = e
                with self._calls_lock:
                    self.retries_timeout += 1
                if attempt >= self.max_retries:
                    raise
                sleep_for = self.retry_base_seconds * (2 ** attempt) + random.uniform(0, 1)
                _log.warning(
                    "OpenAI %s (attempt %d/%d). Sleeping %.1fs.",
                    type(e).__name__, attempt + 1, self.max_retries, sleep_for,
                )
                time.sleep(sleep_for)
        else:  # pragma: no cover  — only hit if the loop fails to break
            assert last_err is not None
            raise last_err

        # Defensive: some endpoints (Azure content-filter, proxies, custom
        # gateways) return success with empty choices. Don't crash on it.
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""

        usage = getattr(resp, "usage", None)
        if usage is not None:
            with self._calls_lock:
                self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        msg = getattr(choices[0], "message", None)
        return (getattr(msg, "content", None) or "").strip()

    @staticmethod
    def _extract_retry_after(err) -> Optional[float]:
        """Pull a Retry-After hint out of a RateLimitError, if the SDK
        attached the response headers. Returns None when unavailable."""
        resp = getattr(err, "response", None)
        if resp is None:
            return None
        headers = getattr(resp, "headers", None) or {}
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:
            return None
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def supports_structured_outputs(self) -> bool:
        """OpenAI proper supports response_format=json_schema. Most local and
        proxy endpoints don't, and silently break or 400 when it's set."""
        host = self.client.base_url.host or ""
        return "openai.com" in host

    def stats(self) -> dict:
        """Operational counters for summary.json."""
        with self._calls_lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "retries_429": self.retries_429,
                "retries_timeout": self.retries_timeout,
            }

    def cost_summary(self) -> dict:
        # Exact match first (cheap, deterministic). Fall back to longest
        # prefix match (with a `-` separator so `gpt-4` doesn't absorb
        # `gpt-4o-mini`) so dated model IDs like
        # `gpt-4o-mini-2024-07-18` resolve to the `gpt-4o-mini` rates.
        model_lc = self.model.lower()
        rates = self.pricing.get(self.model) or self.pricing.get(model_lc) or {}
        if not rates:
            prefixes = [
                k for k in self.pricing
                if model_lc == k.lower() or model_lc.startswith(k.lower() + "-")
            ]
            if prefixes:
                best = max(prefixes, key=len)
                rates = self.pricing[best]
        in_rate = float((rates or {}).get("input", 0.0) or 0.0)
        out_rate = float((rates or {}).get("output", 0.0) or 0.0)
        usd = (self.input_tokens * in_rate + self.output_tokens * out_rate) / 1_000_000
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "est_usd": round(usd, 6),
            "priced": bool(rates),
        }


def load_pricing(path: Path) -> dict:
    """Optional pricing table. Models absent from the file resolve to $0
    cost in summaries, which is correct for local servers and the A8C
    proxy (no per-token bill). Tokens are still recorded regardless."""
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}
