# Newsporter

Pluggable ETL pipeline that seeds WordPress with synthesized post corpora. Designed so you can swap the source dataset, the per-row transform, or the WP target without touching code. Useful whenever you need a realistic-looking content corpus on a WordPress site for testing, benchmarking, or development.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The editable install registers the package and the `newsporter` console script. `python -m newsporter` works the same. If you're hacking on the code, `pip install -e .` keeps your edits live without reinstalling.

### One-time WordPress side (for live runs)

Drop [`tools/newsporter-meta.php`](tools/newsporter-meta.php) into the target site's `wp-content/mu-plugins/`. It registers two meta keys (`_newsporter_source_id`, `_newsporter_byline`) so Newsporter can look up existing posts by source id and avoid duplicating on retry.

> [!IMPORTANT]
> Without the mu-plugin installed on the WordPress side, every retry creates a duplicate post. Cross-machine resume becomes impossible too.

## Quickstart

```bash
# Smoke test. defaults.yaml ships with dry_run: true. Nothing uploads.
export OPENAI_API_KEY=sk-...
python -m newsporter --list-presets             # see what's available
python -m newsporter --preset <name> --sample-size 5

# Real run: copy the template, set creds, and pass --live.
cp config.example.yaml config.yaml
$EDITOR config.yaml   # set wordpress.url, username, app_password, sample_size
python -m newsporter --preset <name> --live
```

> [!NOTE]
> `load.dry_run` defaults to **true**, so a missing or typo'd config can never accidentally upload to a live site. Pass `--live` (or set `load.dry_run: false` in `config.yaml`) to actually upload.

### Common flags

| Flag | Purpose |
|---|---|
| `--preset NAME` | Pick a dataset pack from `presets/` |
| `--config PATH` | Override the auto-loaded `./config.yaml` |
| `--sample-size N` | Override `dataset.sample_size` |
| `--dry-run` / `--live` | Force dry-run / force upload |
| `--log-level LEVEL` | DEBUG / INFO / WARNING / ERROR |
| `--no-cache` | Bypass the transform cache for this run |
| `--purge` | Delete every post on the target site, then exit |
| `--list-presets` | Print available presets |

## Config layering

The CLI loads in this order, deep-merging at each step (later overrides earlier):

```
defaults.yaml         (in repo)         environment baseline: load knobs, llm choice, costs
presets/<name>.yaml   (in repo)         dataset pack: source + transform only
config.yaml           (gitignored)      your environment: wordpress site + creds + per-run knobs
CLI flags                               final overrides (--dry-run, --sample-size)
```

A preset is dataset-only. The same preset works against any WP site, any LLM, any sample size. Your `config.yaml` carries credentials and per-run choices, and stays out of git.

## Where state lives

```
config.yaml             your environment, gitignored, never leaves your machine
defaults.yaml           shared environment baseline (in repo)
presets/                shared dataset packs (in repo)
data/transforms_cache.jsonl   completed (RawRow → Post) results, replays on resume
data/uploads.jsonl            successful (source_id → post_id) entries; the resume map
data/hf_cache/                HuggingFace dataset cache (when using a HuggingFace source)
runs/<timestamp>/             per-run artifacts (see below)
```

Each run writes:

- `summary.json`: counts, elapsed time, chat token usage + estimated USD, embedding cost estimate, retry counters, drift report when `--verify-with-wp` was used
- `posts.jsonl`: synthesized post payloads, one per line, streamed as they're transformed
- `results.jsonl`: per-post upload result with WP post id (or error + status), streamed as they complete
- `run.log`: full log file (everything from stderr also written here)

## Architecture

```
newsporter/
├── newsporter/         # the Python package
│   ├── sources/        # Extract: pluggable source implementations
│   ├── transforms/     # Transform: pluggable transformer implementations + cleaners
│   └── load/           # Load: WordPress REST sink
├── presets/            # dataset packs (source + transform only)
├── defaults.yaml       # environment baseline (load knobs, llm, costs)
└── config.yaml         # gitignored: your wordpress + creds + per-run knobs
```

Three abstractions:

- **`Source`** (`sources/base.py`): yields `RawRow`s. Implementations: `huggingface`, `jsonl`. Add more by writing a new class and registering it in `sources/__init__.py`.
- **`Transformer`** (`transforms/base.py`): turns `(RawRow, Prefab)` into a `Post`. Implementations: `llm_synth` (current behavior), `passthrough` (no LLM, for already-structured corpora). The cache signature is derived from the transformer so config edits auto-invalidate stale entries.
- **`WordPressLoader`** (`load/wordpress.py`): single sink. Resolves categories and authors, retries with backoff, supports dry run.

A run picks one of each via the `type:` discriminator in YAML:

```yaml
source:    { type: huggingface, ... }
transform: { type: llm_synth, ... }
```

## Adding a new source

1. Subclass `Source` in `newsporter/sources/yourname.py`. Implement `fetch(sample_size, seed) -> list[RawRow]`.
2. Register it in `newsporter/sources/__init__.py` under a `type:` key.
3. Reference it in your config: `source: { type: yourname, ... }`.

## Adding a new transformer

Same shape: subclass `Transformer`, implement `signature()` and `transform()`, register in `transforms/__init__.py`. If your transformer needs an LLM client, add its type to `LLM_REQUIRED`.

## Cleaners

The `llm_synth` transformer applies a list of declarative cleaning steps from config:

```yaml
transform:
  cleaners:
    - { type: regex_strip, pattern: '^\(SOURCE_TAG\)\s*--\s*' }
    - { type: html_strip }
    - { type: trim }
```

Cleaner types: `regex_strip`, `regex_replace`, `html_strip`, `trim`, `collapse_whitespace`. Add new ones in `newsporter/transforms/cleaners.py` with the `@cleaner("name")` decorator.

## Pricing and cost tracking

`pricing.yaml` at the repo root maps model IDs to `{input, output}` USD-per-million-tokens. Models not listed resolve to $0 cost (correct for local servers and the A8C proxy). Tokens are recorded regardless of pricing presence.

`summary.json` carries a `chat_cost` block when an LLM was used:

```json
"chat_cost": {
  "model": "gpt-4o-mini",
  "input_tokens": 181613,
  "output_tokens": 4639,
  "est_usd": 0.030025,
  "priced": true
}
```

## Cache

When `transform.cache.enabled: true`, completed `Post` objects are appended to a JSONL file keyed by `source_id`. The cache key is the SHA-256 prefix of the transformer's signature (model + prompt + cleaners + labels + schema version), so changing any of those auto-invalidates stale entries on next load. Re-uploading the same corpus to a different WP site is free because the LLM cost is paid only once.

Delete the cache file to force a clean rebuild.

## Resume

A killed-and-restarted run picks up where it left off automatically. Three pieces make this work:

1. **Deterministic source sampling**: same `seed` + same `sample_size` → same set of `source_id`s comes back.
2. **Local upload log**: every successful upload appends `{"source_id": "...", "post_id": N}` to `data/uploads.jsonl`. On startup, the loader reads it into an in-memory dict and the pipeline filters those rows out before transform. A 99k-of-100k resume costs a single local file read (instant) and processes only the missing 1k.
3. **Transform cache**: any rows whose transform completed before the crash replay from disk without re-hitting the LLM.

**Resume recipe:** re-run with the exact same flags. Already-uploaded rows are skipped automatically.

### Upload log vs WP-side authority

The local log is fast but local. If posts are deleted via wp-admin, or you resume from a different machine, the log is stale. Two ways to reconcile:

- **`--verify-with-wp`**: bulk-fetch every `_newsporter_source_id` already on the site (paginated REST sweep, takes ~5 min per 99k posts on WP Cloud's rate-limited edge), merge into the local log, take WP as authoritative. Use after `--purge` or after a cross-machine handoff.
- **`--no-resume-check`**: skip both the local log read and any WP fetch. Use when uploading to a known-empty site or to force a fresh upload of every row.

Caveats:
- The mu-plugin at `tools/newsporter-meta.php` is required for `--verify-with-wp` (queries the meta key on WP) and for any external tool that wants to find posts by source_id. The local-log path doesn't need it, since the log is the source of truth.

> [!CAUTION]
> Sample size must match across resume runs. `random.sample(range(total), n)` produces a different set for every `n`, not a strict superset. Resuming with `--sample-size 5000` after an initial `--sample-size 1000` does not give you "the original 1000 plus 4000 new"; it reshuffles the lot, which usually isn't what you want.

## Content and licensing

Newsporter is a generic ETL infrastructure. It does not ship any source content in this repo. Whatever dataset a preset points at gets fetched at runtime into your local HuggingFace (or other source) cache.

> [!IMPORTANT]
> **You are responsible for what you upload.** Source content may be owned by third parties. Using Newsporter to populate a private development or test site is generally fine for research and engineering purposes. Using it to publish copyrighted content on a public site is not. If you're not sure whether your use case is in bounds, consult your legal team before flipping `--live` against a publicly-accessible site.

## Production-scale tuning

> [!TIP]
> For runs in the tens-of-thousands range or against rate-limited WP infrastructure, the defaults are conservative but worth checking. In your `config.yaml`:

```yaml
load:
  concurrency: 2          # below most WP edge rate limits
  retry:
    attempts: 8           # extra budget for 429 cascades

llm:
  max_retries: 6          # smooths over OpenAI tier-1 RPM ceilings
```

The OpenAI client retries `RateLimitError` and timeout errors automatically, using the Retry-After header when the SDK surfaces them. The WP loader does the same for 429s and 5xxs. Permanent 4xxs fail fast (no retry).
