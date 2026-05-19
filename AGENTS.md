# AGENTS.md (Newsporter)

Project-specific context for agents working in this directory. Global rules (response style, no em dashes in prose, sentence-case headings, etc.) live in `~/.claude/CLAUDE.md` and still apply.

## What this is

Pluggable ETL pipeline that pulls rows from a source (HuggingFace dataset, JSONL file, etc.), transforms each row into a synthesized WordPress `Post` (LLM-synthesized title + category + plausible publication date, plus randomized author + cleaned body), and uploads via the WP REST API. Designed to seed test sites with realistic-looking content corpora.

## Config layering

Four layers, deep-merged in this order (later overrides earlier):

1. `defaults.yaml` (in repo): environment baseline, shared across teammates.
2. `presets/<name>.yaml` (in repo): dataset pack with `source:` + `transform:` only, never credentials or per-target knobs.
3. `config.yaml` (gitignored): wp creds, per-run knobs, environment-specific overrides.
4. CLI flags: final word.

The principle: **a preset describes a dataset; a config describes an environment.** Don't put credentials in a preset; don't put cleaners in a config.

`expand_env_vars` only walks **credential leaves**: `wordpress.{url,username,app_password}`, `llm.api_key`, `llm.headers.*`. Anything else with `${VAR}` is left literal. This prevents a poisoned preset from exfiltrating arbitrary env vars (e.g. `${AWS_SECRET_ACCESS_KEY}` in a prompt template).

`validate_config` checks every required key plus does source-type-specific shape checks (huggingface needs `name`+`split`; jsonl needs `path` to exist; llm_synth needs non-empty labels and an ISO-valid date range). Errors are accumulated and reported all at once, BEFORE the dataset loads or the pipeline starts.

## Idempotency contract (layered)

Four independent layers, each catches different failure modes:

1. **WP server-side meta** (`_newsporter_source_id`). Requires `tools/newsporter-meta.php` in `wp-content/mu-plugins/`. Without it, the meta is silently dropped on every POST and idempotency is local-only. The `auth_callback` is `manage_options` (admin-only), so the user uploading via REST must have admin caps for the meta to be writable.
2. **Local upload log** (`data/uploads.jsonl`). Append-only JSONL of `{source_id, post_id}`. Read at startup → in-memory dict → pipeline filters those rows before transform. A 99k-of-100k resume costs a single local-disk read instead of 99k REST roundtrips.
3. **Pre-create lookup on retry**. If a `create_post` retry fires after a `ConnectionError`/`Timeout`, the loader does a single `find_post_id_by_source_id` to detect a server-side success whose response we lost. Closes the duplicate-on-network-blip window.
4. **Content-hash dedup** (`_newsporter_content_hash`). Catches the case where the upstream dataset ships the same article under multiple source IDs (HuggingFace mirrors, syndicated news, etc.). Source-ID dedup is blind to that because the IDs differ. The hash is MD5 of the raw source body post-strip; the pipeline computes it after fetch, the loader writes it both as post meta on WP and as an optional `content_hash` field on each `data/uploads.jsonl` line. Bulk fetch via `list_metas()` rebuilds the index from WP; warm resume from the local log reads it via `UploadLog.all_hashes()`. Either path keeps cross-run dedup live, so warm resume after an interrupted run still catches duplicates landed in the previous batch. Disable with `dataset.dedup_content: false`.

Authority order on resume:

- Local log is trusted by default.
- `--verify-with-wp` does a bulk WP fetch via `list_source_ids`, REPLACES the local log with WP's authoritative state, and reports drift counts (`only_in_log`, `only_on_wp`, `differing_post_ids`) to `run.log`. Use after wp-admin deletions or cross-machine handoffs.
- `--purge` deletes all posts on the WP site AND truncates the local upload log so the next run is consistent.
- `--no-resume-check` skips both paths (force-fresh upload).

The mu-plugin uniqueness enforcement is NOT implemented server-side. That's the long-term right answer (Codex flagged it) but requires a real REST-route plugin. The pre-create lookup on retry is the practical mitigation.

## Cache signature

`TransformCache` keys are `<sha256(signature)[:16]>`. `LLMSynthTransformer.signature()` hashes:

- **source identity** (type + name + path + config + split + field_map). Switching corpora invalidates.
- model name
- prompt template (the actual string, not a version number; edit the prompt, signature flips)
- sorted labels (order-independent so reordering YAML doesn't invalidate)
- cleaners list (order-dependent; order changes behavior)
- date range
- max_tokens, max_title_len, body_field, summary_field
- `schema_version` constant. Bump in code to force-invalidate every cache without editing config.

Cross-corpus collision (same `source_id` from a different dataset replaying a stale post) is impossible because source identity is in the signature.

`PassthroughTransformer` has its own simpler signature with `SCHEMA_VERSION = "v1-passthrough"`.

## Reasoning model quirks (gpt-5*, o-series)

- Use `max_completion_tokens` instead of `max_tokens`
- Reject any `temperature` other than the default (1)
- Detection: `re.match(r"^(gpt-[5-9]|o\d+)(-|$)", model, re.IGNORECASE)`. `gpt-4o` does NOT match (good); `o4-mini`, `gpt-5-nano`, `O1` all match.

If you add a new reasoning family, extend the regex. If you add a non-OpenAI provider that uses different param names, don't try to hide it in the regex; add a config knob.

## Pricing longest-prefix

`pricing.yaml` lookup: exact match first; on miss, longest prefix match with a required `-` separator. So `gpt-4o-mini-2024-07-18` resolves to `gpt-4o-mini` rates, but a hypothetical `gpt-4` entry would NOT absorb `gpt-4o-anything`. Both sides are lowercased.

Models not in the pricing table resolve to $0 (`priced: false` in summary.json), correct for local servers and for proxy endpoints where the project is internally billed.

## OpenAI / WP retry semantics

**LLM (`LLM.chat`)**: retries on `RateLimitError` (honors `Retry-After` from response headers when present), `APITimeoutError`, `APIConnectionError`. Defaults: `max_retries=5`, `retry_base_seconds=2`, exponential backoff with jitter. Counts in `LLM.stats()`.

**WP (`WordPressLoader.upload`)**: retries on 429 (parses `Retry-After`) and 5xx. Permanent 4xx fails fast. `attempts` and `backoff_seconds` are configured under `load.retry`. `WordPressUploadError` carries `status`, redacted `body_excerpt` (long base64-shaped strings replaced with `[redacted]`), and full response `headers` for the retry path.

## Streaming pipeline

`pipeline.run_pipeline` runs two `ThreadPoolExecutor`s connected by a bounded `queue.Queue` (cap = `max(upload_workers*4, 16)`). Sentinel-finally pattern guarantees no deadlock when transform raises:

```python
try:
    for f in as_completed(tr_futs): f.result()
finally:
    for _ in range(upload_workers):
        post_q.put(_SENTINEL, timeout=300)  # 5 min covers worst-case retry stalls
```

Per-thread `requests.Session` (the Session is documented as not thread-safe), stashed in `threading.local`, lazy-initialized.

Per-item try/except in `upload_worker` so a single failure can't kill the worker (which would otherwise leave its share of the queue undrained, eventually deadlocking shutdown).

## Source / transformer registries

Module-level dicts in `sources/__init__.py:SOURCE_REGISTRY` and `transforms/__init__.py:TRANSFORM_REGISTRY`. `LLM_REQUIRED` set lists transformer types that need an LLM client. Add a new source: write the class, register it, reference `type:` in YAML. No plugin / entry-point machinery (deliberate; overkill for a small internal tool).

`build_transformer` accepts `source_identity` and threads it to the transformer so the cache signature can include it.

## Date inference

The `llm_synth` transformer asks the model for `YYYY-MM` only. The Python side picks a random valid day via `calendar.monthrange` (handles leap years and 30/31-day months) and a random hour/minute/second. Out-of-window months fall back to the prefab's uniform-random date.

Each preset declares its own `transform.date.range_start` / `range_end` so the prompt and the clamp share a single source of truth. Verify the dataset's actual coverage from its source (HuggingFace dataset card, RSS feed range, etc.) before setting the window: a too-wide range invites the model to guess wildly; a too-narrow one masks legitimate dates outside it.

## CLI flag interaction matrix

- `--dry-run` and `--live` are mutually exclusive (argparse group)
- `--no-resume-check` and `--verify-with-wp` are mutex (we error out)
- `--purge` requires either typed-URL confirmation or `--yes`
- `--purge` clears the local upload log on success
- `--no-cache` bypasses the transform cache for this run only
- `--list-presets` prints and exits (one per line, shell-friendly)
- Process exit code is `1` when `posts_failed > 0`

## Things deferred (don't redo unless asked)

- **WP-side atomic upsert via custom REST endpoint**. Codex's preferred fix; would eliminate the network-blip duplicate window for sure. Requires shipping a real plugin (not just an mu-plugin meta-registration). Pre-create lookup on retry is the practical mitigation we landed on.
- **Streaming HF iteration**. `load_dataset` materializes the train split before sampling. Memory cost for 100k is fine (~1-1.5 GB total resident). Defer until needed.
- **Cache compaction**. File growth is bounded for our scale.
- **Cross-process file locking**. Single-writer is the assumed deployment.
- **KeyboardInterrupt graceful cancel via `cancel_futures`**. Current behavior waits ≤60s for in-flight LLM, then exits cleanly. Double-Ctrl-C is the failure mode and we accept ≤1 row of loss.
- **ReDoS protection on user cleaner regex**. Trust model is "configs come from teammates, not the corpus." Documented; not enforced.
- **Test suite**. Separate scope. The validation-up-front and signature-based cache invalidation patterns are designed to be testable.
- **Slug Unicode handling**. Author pool is ASCII; not a concern for any current preset.
- **Streaming JSONL source for huge files**. JSONL source loads the whole file into memory before sampling. Documented; defer.

## Things easy to get wrong

- **Forgetting `pip install -e .`**. `python -m newsporter` will fail with `ModuleNotFoundError`. Editable install is the install pattern.
- **Editing a preset thinking it'll override a default**. Presets only override their `source:` and `transform:` blocks. Per-environment overrides (wordpress, llm, sample_size) belong in `config.yaml`.
- **Running `--purge` on the wrong site**. There's a typed-URL prompt for a reason. Never use `--yes` with `--purge` outside of CI / scripted contexts.
- **Resuming with a different `--sample-size`**. `random.sample(range(total), n)` produces different sets for different n, not a strict superset. The same source_ids you want to resume against may not be in the new sample.
- **Bumping the prompt and not understanding why old runs invalidate**. Prompt template is hashed into the signature. Edit the prompt, the cache is stale. That's the design.
- **Forgetting the mu-plugin**. `--verify-with-wp` returns empty silently. The local-log path keeps working, but cross-machine resume becomes impossible.
</content>
</invoke>