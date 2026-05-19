"""WordPress REST loader.

`WordPressClient` is the low-level wrapper around `/wp-json/wp/v2/`.
`WordPressLoader` is the per-post orchestrator the pipeline calls; it
handles category/author resolution, retries with proper 429/Retry-After
support, idempotent upload via `_newsporter_source_id` meta, and dry run.

Each thread gets its own `requests.Session` (the Session object is
documented as not thread-safe). Sessions are created lazily on first use
and stashed in `threading.local`.

Idempotency: each upload sends the source_id in `meta._newsporter_source_id`
and consults that key on retry / re-run to avoid duplicating posts.
This requires the meta key to be REGISTERED on the WordPress side. Drop
the snippet in `tools/newsporter-meta.php` into wp-content/mu-plugins/.

Config schema (relevant subset):
    wordpress:
      url: "https://..."
      username: "..."
      app_password: "..."
    load:
      concurrency: 4
      post_status: "publish"
      retry: { attempts: 3, backoff_seconds: 2 }
      dry_run: true               # default! pass --live or set false to upload
      timeout_connect: 5
      timeout_read: 30
      author:
        prepend_to_content: false
        create_wp_users: true
        role: "author"
        email_domain: "newsporter.local"
"""

from __future__ import annotations

import email.utils
import html
import logging
import re
import secrets
import threading
import time
from typing import Any

import requests
import requests.exceptions

from ..models import Post

log = logging.getLogger("newsporter")

# How long to wait when a server returns 429/503 with no Retry-After.
_DEFAULT_RETRY_AFTER_SECONDS = 5.0


class WordPressUploadError(RuntimeError):
    """Carries the HTTP status, response excerpt, response headers, and
    source_id so the pipeline can write a structured failure record and
    the retry path can honor Retry-After."""

    def __init__(
        self,
        status: int,
        body_excerpt: str,
        source_id: str = "",
        headers: dict | None = None,
    ) -> None:
        # Avoid leaking auth-shaped substrings if a misconfigured WP
        # security plugin echoes the request back.
        sanitized = _redact_credentials(body_excerpt)
        super().__init__(f"HTTP {status}: {sanitized[:300]}")
        self.status = status
        self.body_excerpt = sanitized
        self.source_id = source_id
        self.headers = dict(headers or {})


_REDACT_RE = re.compile(r"[A-Za-z0-9+/_-]{32,}={0,2}")


def _redact_credentials(text: str) -> str:
    if not text:
        return text
    return _REDACT_RE.sub("[redacted]", text)


def _to_block_content(text: str) -> str:
    """Convert plain text to Gutenberg paragraph blocks. Splits on blank
    lines (one or more); within a paragraph, bare \\n becomes <br/>."""
    paragraphs = re.split(r"(?:\r?\n\s*){2,}", text)
    blocks = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # Within a paragraph, preserve hard line breaks as <br/>.
        escaped = "<br/>".join(html.escape(line) for line in p.splitlines() if line.strip())
        if not escaped:
            continue
        blocks.append(f"<!-- wp:paragraph -->\n<p>{escaped}</p>\n<!-- /wp:paragraph -->")
    return "\n\n".join(blocks)


def _parse_retry_after(header_value: str | None) -> float | None:
    if not header_value:
        return None
    try:
        return float(header_value)
    except (TypeError, ValueError):
        pass
    # HTTP-date format
    try:
        dt = email.utils.parsedate_to_datetime(header_value)
        if dt is None:
            return None
        delta = dt.timestamp() - time.time()
        return max(delta, 0.0)
    except (TypeError, ValueError):
        return None


class WordPressClient:
    """Thread-aware low-level REST client. Each thread gets its own
    Session via threading.local."""

    def __init__(
        self, cfg: dict, *, timeout_connect: float = 5.0, timeout_read: float = 30.0
    ) -> None:
        if not cfg.get("url"):
            raise ValueError("wordpress.url is required")
        url = cfg["url"].rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"wordpress.url must include scheme: {url}")
        if url.startswith("http://") and "localhost" not in url and "127.0.0.1" not in url:
            log.warning("wordpress.url is HTTP, not HTTPS. App password will travel in cleartext.")
        self.url = url
        self.username = cfg["username"]
        self.app_password = cfg["app_password"]
        self.timeout = (timeout_connect, timeout_read)
        self._tls = threading.local()

    def _session(self) -> requests.Session:
        s = getattr(self._tls, "session", None)
        if s is None:
            s = requests.Session()
            s.auth = (self.username, self.app_password)
            s.headers.update({"User-Agent": "newsporter/0.2"})
            self._tls.session = s
        return s

    def _url(self, path: str) -> str:
        return f"{self.url}/wp-json/wp/v2{path}"

    # ── Categories / authors ────────────────────────────────────────

    def _paginated_get(self, path: str, params: dict | None = None) -> list[dict]:
        """Fetch all pages of a list endpoint. Stops when the page is
        shorter than per_page or X-WP-TotalPages says we're done."""
        out: list[dict] = []
        params = dict(params or {})
        params.setdefault("per_page", 100)
        page = 1
        while True:
            params["page"] = page
            r = self._session().get(self._url(path), params=params, timeout=self.timeout)
            if r.status_code >= 400:
                # WP returns 400 with code rest_post_invalid_page_number when
                # we've gone past the end. Treat that as "no more pages".
                if r.status_code == 400 and "rest_post_invalid_page_number" in r.text:
                    break
                raise WordPressUploadError(r.status_code, r.text)
            chunk = r.json()
            out.extend(chunk)
            total_pages = int(r.headers.get("X-WP-TotalPages") or 1)
            if page >= total_pages or len(chunk) < params["per_page"]:
                break
            page += 1
        return out

    def ensure_categories(self, names: list[str]) -> dict[str, int]:
        """Map every name to a category id, creating missing ones. Match
        is case-insensitive on display name AND slug, so 'World' won't
        duplicate an existing 'world'."""
        existing = self._paginated_get("/categories")
        by_name_lc = {c["name"].lower(): c["id"] for c in existing}
        by_slug = {c["slug"]: c["id"] for c in existing}
        mapping: dict[str, int] = {}
        for name in names:
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "category"
            if name.lower() in by_name_lc:
                mapping[name] = by_name_lc[name.lower()]
                continue
            if slug in by_slug:
                mapping[name] = by_slug[slug]
                continue
            r = self._session().post(
                self._url("/categories"),
                json={"name": name, "slug": slug},
                timeout=self.timeout,
            )
            if r.status_code >= 400:
                # `term_exists` carries the existing term id we want.
                try:
                    body = r.json()
                except Exception:
                    body = {}
                term_id = (body.get("data") or {}).get("term_id")
                if r.status_code == 400 and body.get("code") == "term_exists" and term_id:
                    mapping[name] = int(term_id)
                    continue
                raise WordPressUploadError(r.status_code, r.text)
            mapping[name] = r.json()["id"]
        return mapping

    def ensure_authors(self, names: list[str], role: str, email_domain: str) -> dict[str, int]:
        """Map every name to a user id, creating missing ones. If WP
        rejects the create with `existing_user_login`/`existing_user_email`,
        look the user up and adopt the id rather than failing."""
        existing = self._paginated_get("/users", {"context": "edit"})
        by_slug = {u["slug"]: u["id"] for u in existing}
        by_name_lc = {u["name"].lower(): u["id"] for u in existing}
        mapping: dict[str, int] = {}
        for name in names:
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "author"
            if slug in by_slug:
                mapping[name] = by_slug[slug]
                continue
            if name.lower() in by_name_lc:
                mapping[name] = by_name_lc[name.lower()]
                continue
            parts = name.split(" ", 1)
            payload = {
                "username": slug,
                "email": f"{slug}@{email_domain}",
                "password": secrets.token_urlsafe(24),
                "roles": [role],
                "name": name,
                "first_name": parts[0],
                "last_name": parts[1] if len(parts) > 1 else "",
            }
            r = self._session().post(self._url("/users"), json=payload, timeout=self.timeout)
            if r.status_code >= 400:
                try:
                    body = r.json()
                except Exception:
                    body = {}
                # Adopt existing user when WP rejects the duplicate.
                if body.get("code") in ("existing_user_login", "existing_user_email"):
                    found = self._session().get(
                        self._url("/users"),
                        params={"slug": slug, "context": "edit"},
                        timeout=self.timeout,
                    )
                    if found.status_code < 400:
                        users = found.json()
                        if users:
                            mapping[name] = users[0]["id"]
                            continue
                raise WordPressUploadError(r.status_code, f"create user '{name}': {r.text}")
            mapping[name] = r.json()["id"]
        return mapping

    # ── Posts ──────────────────────────────────────────────────────

    def find_post_id_by_source_id(self, source_id: str) -> int | None:
        """Single-row lookup by `_newsporter_source_id`. Used by the
        retry-recovery path to detect whether a prior POST that returned
        a network error actually succeeded server-side. The bulk
        startup fetch (`list_source_ids`) is the primary idempotency
        path; this is the in-flight safety net."""
        try:
            r = self._session().get(
                self._url("/posts"),
                params={
                    "meta_key": "_newsporter_source_id",
                    "meta_value": source_id,
                    "status": "any",
                    "context": "view",
                    "per_page": 1,
                },
                timeout=self.timeout,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return None
        if r.status_code >= 400:
            return None
        items = r.json()
        if isinstance(items, list) and items:
            pid = items[0].get("id")
            return int(pid) if isinstance(pid, int) and pid > 0 else None
        return None

    def list_metas(self) -> tuple[dict[str, int], dict[str, int]]:
        """Bulk fetch `_newsporter_source_id` AND `_newsporter_content_hash`
        from every post on the site in a single paginated scan. Returns
        `(source_id_to_post_id, content_hash_to_post_id)`. One traversal
        instead of two; large corpora make REST roundtrips the dominant
        cost here.

        Both meta keys must be registered with `show_in_rest: true` on
        the WP side (see `tools/newsporter-meta.php`). Returns empty
        dicts on any error (caller treats absence as "no idempotency
        available, behave like a fresh upload").
        """
        sids: dict[str, int] = {}
        hashes: dict[str, int] = {}
        page = 1
        per_page = 100
        while True:
            try:
                r = self._session().get(
                    self._url("/posts"),
                    params={
                        "per_page": per_page,
                        "page": page,
                        "status": "any",
                        "context": "edit",
                        "_fields": "id,meta",
                    },
                    timeout=self.timeout,
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ):
                return sids, hashes
            if r.status_code >= 400:
                # 400 with rest_post_invalid_page_number = past the end.
                if r.status_code == 400 and "rest_post_invalid_page_number" in r.text:
                    break
                # Anything else: bail with what we have. The pipeline
                # will fall back to per-create with no idempotency,
                # which still works (just slower / not duplicate-safe).
                return sids, hashes
            items = r.json()
            if not isinstance(items, list) or not items:
                break
            for it in items:
                meta = it.get("meta")
                # WP returns meta as a dict when keys are show_in_rest,
                # but some plugin combos (and older WP) return a list of
                # {key, value} records. Normalize both shapes.
                if isinstance(meta, list):
                    meta = {m.get("key"): m.get("value") for m in meta if isinstance(m, dict)}
                elif not isinstance(meta, dict):
                    meta = {}
                pid = it.get("id")
                if not (isinstance(pid, int) and pid > 0):
                    continue
                sid = meta.get("_newsporter_source_id")
                # Be explicit: "0" and 0 are valid source_ids and must
                # not be silently dropped by a truthy check.
                if sid is not None and sid != "":
                    sids[str(sid)] = pid
                chash = meta.get("_newsporter_content_hash")
                if chash:
                    hashes[str(chash)] = pid
            total_pages = int(r.headers.get("X-WP-TotalPages") or 1)
            if page >= total_pages or len(items) < per_page:
                break
            page += 1
        return sids, hashes

    def list_source_ids(self) -> dict[str, int]:
        """Backward-compatible wrapper around `list_metas`. Returns just
        the source-id index. Prefer `list_metas` when both indices are
        needed in the same startup pass.
        """
        sids, _ = self.list_metas()
        return sids

    def create_post(
        self,
        post: Post,
        category_id: int,
        author_id: int | None,
        status: str,
        prepend_byline: bool,
    ) -> tuple[int, requests.Response]:
        content = _to_block_content(post.content)
        if prepend_byline:
            byline = (
                "<!-- wp:paragraph -->\n"
                f"<p><em>By {html.escape(post.author)}</em></p>\n"
                "<!-- /wp:paragraph -->"
            )
            content = f"{byline}\n\n{content}"
        # Title is rendered by themes; escape to prevent stored XSS via
        # poisoned LLM output or a passthrough source.
        payload: dict[str, Any] = {
            "title": html.escape(post.title),
            "content": content,
            "status": status,
            "date_gmt": post.date_gmt,
            "categories": [category_id],
            "meta": {
                "_newsporter_source_id": post.source_id,
                "_newsporter_byline": post.author,
                # Empty string when the body field is missing/empty; WP
                # treats that the same as "no value" for sanitize_text_field,
                # so the row just won't surface in the content-hash index.
                "_newsporter_content_hash": post.content_hash,
            },
        }
        if author_id is not None:
            payload["author"] = author_id
        r = self._session().post(self._url("/posts"), json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            raise WordPressUploadError(
                r.status_code, r.text, source_id=post.source_id, headers=dict(r.headers)
            )
        try:
            body = r.json()
        except Exception as e:
            raise WordPressUploadError(
                r.status_code,
                f"non-JSON success body: {e}",
                source_id=post.source_id,
                headers=dict(r.headers),
            ) from e
        post_id = body.get("id") if isinstance(body, dict) else None
        if not isinstance(post_id, int) or post_id <= 0:
            raise WordPressUploadError(
                r.status_code,
                f"missing/invalid post id in success body: {body!r}"[:300],
                source_id=post.source_id,
                headers=dict(r.headers),
            )
        return int(post_id), r


class WordPressLoader:
    """Per-post upload orchestrator with idempotency, retry, and dry-run.

    Holds the resolved category/author maps so the pipeline doesn't have
    to know about them. In dry-run mode, never instantiates a client and
    never touches the network — useful for offline transform inspection."""

    def __init__(
        self,
        wp_cfg: dict,
        load_cfg: dict,
        labels: list[str],
        author_pool: list[str],
        *,
        upload_log=None,
        skip_resume_check: bool = False,
        verify_with_wp: bool = False,
    ) -> None:
        self.cfg = load_cfg
        self.dry_run = bool(load_cfg.get("dry_run"))
        self.status = load_cfg.get("post_status", "publish")
        retry = load_cfg.get("retry") or {}
        self.attempts = int(retry.get("attempts", 3))
        self.backoff = float(retry.get("backoff_seconds", 2))
        timeout_connect = float(load_cfg.get("timeout_connect", 5))
        timeout_read = float(load_cfg.get("timeout_read", 30))

        author_cfg = load_cfg.get("author") or {}
        self.prepend_byline = bool(author_cfg.get("prepend_to_content"))
        self.create_wp_users = bool(author_cfg.get("create_wp_users"))
        self.role = author_cfg.get("role", "author")
        self.email_domain = author_cfg.get("email_domain", "newsporter.local")

        self.client: WordPressClient | None = None
        self.cat_mapping: dict[str, int] = {}
        self.author_mapping: dict[str, int] = {}
        self.upload_log = upload_log
        # source_id → post_id for posts already on the site. Populated
        # at startup from the local upload log (fast path) and
        # optionally cross-checked against WP. Mutated under
        # `_existing_lock` as new uploads complete, so concurrent
        # workers in the same run don't race to create duplicates.
        self.existing_post_ids: dict[str, int] = {}
        # content_hash → post_id, for catching upstream-dataset duplicates
        # (same article under multiple source_ids). Populated from the
        # local upload log on warm resume, replaced/augmented by the WP
        # bulk fetch when it runs.
        self.existing_content_hashes: dict[str, int] = {}
        self._existing_lock = threading.Lock()
        # Counters surfaced to summary.json by the pipeline. Owned here
        # rather than passed through return values so the loader stays
        # the single source of truth for upload-side stats.
        self.dedup_stats: dict[str, int] = {
            "cross_run": 0,
            "within_run": 0,
            "empty_hash": 0,
        }
        # Row-funnel counters. Lets summary.json show the true number of
        # rows that came back from source.fetch() (pre-filter) alongside
        # what survived each successive skip. Without these, post-filter
        # counts hide both source-ID resume skips and content-hash skips.
        self.pipeline_stats: dict[str, int] = {
            "rows_fetched": 0,
            "rows_after_source_id_resume": 0,
            "rows_after_content_dedup": 0,
        }

        if self.dry_run:
            self.cat_mapping = {lbl: 1000 + i for i, lbl in enumerate(labels)}
        else:
            self.client = WordPressClient(
                wp_cfg, timeout_connect=timeout_connect, timeout_read=timeout_read
            )
            self.cat_mapping = self.client.ensure_categories(labels)
            if self.create_wp_users and author_pool:
                self.author_mapping = self.client.ensure_authors(
                    author_pool, self.role, self.email_domain
                )
            if not skip_resume_check:
                # Fast path: trust the local log. Falls back to WP bulk
                # fetch when the log is empty (first run on this machine)
                # or when the operator explicitly asks for verification.
                if self.upload_log is not None:
                    self.existing_post_ids = self.upload_log.all()
                    # Seed cross-run content-hash dedup from the local log
                    # on the warm path. The WP bulk fetch below replaces
                    # or augments this when it runs, but if the log is
                    # already populated AND verify_with_wp is False, this
                    # is the only source of cross-run hash protection.
                    self.existing_content_hashes = self.upload_log.all_hashes()
                if verify_with_wp or not self.existing_post_ids:
                    log.info(
                        "Cross-checking already-uploaded source_ids against WP "
                        "(can take a few minutes on large sites)..."
                    )
                    server_set, server_hashes = self.client.list_metas()
                    self.existing_content_hashes = dict(server_hashes)
                    if server_hashes:
                        log.info(
                            "Loaded %d content hashes from WP for cross-run dedup.",
                            len(server_hashes),
                        )

                    # Drift report: what does WP say vs our local log?
                    local_set = set(self.existing_post_ids)
                    server_keys = set(server_set)
                    only_local = local_set - server_keys
                    only_server = server_keys - local_set
                    differing = {
                        sid
                        for sid in (local_set & server_keys)
                        if self.existing_post_ids.get(sid) != server_set.get(sid)
                    }
                    if only_local or only_server or differing:
                        log.warning(
                            "Resume drift: only_in_log=%d only_on_wp=%d differing_post_ids=%d",
                            len(only_local),
                            len(only_server),
                            len(differing),
                        )

                    if verify_with_wp:
                        # Operator explicitly asked us to trust WP.
                        # Replace, don't merge — stale local entries
                        # (deletes, machine swaps) get pruned.
                        self.existing_post_ids = dict(server_set)
                        if self.upload_log is not None:
                            self.upload_log.replace(server_set, server_hashes)
                            log.info(
                                "Upload log replaced with %d entries from WP (verify-with-wp).",
                                len(server_set),
                            )
                    else:
                        # Empty log + no explicit verify: trust WP, no
                        # drift reconciliation needed (nothing local to
                        # disagree with).
                        self.existing_post_ids.update(server_set)
                        if self.upload_log is not None and server_set:
                            self.upload_log.merge(server_set, server_hashes)

    def upload(self, post: Post) -> dict:
        if self.dry_run:
            return {"source_id": post.source_id, "dry_run": True, "ok": True}
        assert self.client is not None
        cid = self.cat_mapping.get(post.category)
        if cid is None:
            return {
                "source_id": post.source_id,
                "error": f"category not in mapping: {post.category!r}",
                "status": None,
                "ok": False,
            }
        aid = self.author_mapping.get(post.author)

        # Idempotency: in-memory lookup against the bulk-fetched set.
        # The pipeline already filters most already-uploaded rows up
        # front; this is a belt-and-suspenders check for races between
        # the bulk fetch and an upload landing.
        with self._existing_lock:
            existing = self.existing_post_ids.get(post.source_id)
        if existing is not None:
            return {
                "source_id": post.source_id,
                "post_id": existing,
                "ok": True,
                "skipped": True,
            }

        last_err: Exception | None = None
        last_status: int | None = None
        for i in range(self.attempts):
            try:
                # On retry after a network blip, the previous POST may
                # have created the post server-side; we just lost the
                # response. Probe by source_id to avoid duplicating.
                if i > 0:
                    try:
                        existing_id = self.client.find_post_id_by_source_id(post.source_id)
                    except Exception:
                        existing_id = None
                    if existing_id is not None:
                        with self._existing_lock:
                            self.existing_post_ids[post.source_id] = existing_id
                            if post.content_hash:
                                self.existing_content_hashes.setdefault(
                                    post.content_hash, existing_id
                                )
                        if self.upload_log is not None:
                            self.upload_log.put(post.source_id, existing_id, post.content_hash)
                        return {
                            "source_id": post.source_id,
                            "post_id": existing_id,
                            "ok": True,
                            "skipped": True,
                            "recovered_after_retry": True,
                        }

                pid, _ = self.client.create_post(post, cid, aid, self.status, self.prepend_byline)
                with self._existing_lock:
                    self.existing_post_ids[post.source_id] = pid
                    if post.content_hash:
                        # setdefault: keep whichever post landed first as
                        # the "canonical" id for this content. Later writes
                        # with the same hash shouldn't have reached this
                        # point (pipeline filters them) but guard anyway.
                        self.existing_content_hashes.setdefault(post.content_hash, pid)
                if self.upload_log is not None:
                    self.upload_log.put(post.source_id, pid, post.content_hash)
                return {"source_id": post.source_id, "post_id": pid, "ok": True}
            except WordPressUploadError as e:
                last_err = e
                last_status = e.status
                if e.status == 429:
                    ra = _parse_retry_after(e.headers.get("Retry-After"))
                    sleep_for = ra if ra is not None else (_DEFAULT_RETRY_AFTER_SECONDS * (2**i))
                    log.warning(
                        "WP 429 (attempt %d/%d). Sleeping %.1fs (Retry-After=%s).",
                        i + 1,
                        self.attempts,
                        sleep_for,
                        ra,
                    )
                    time.sleep(sleep_for)
                    continue
                if 500 <= e.status < 600:
                    time.sleep(self.backoff * (i + 1))
                    continue
                # Permanent 4xx — bail without retry.
                break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                last_err = e
                time.sleep(self.backoff * (i + 1))
        return {
            "source_id": post.source_id,
            "error": str(last_err),
            "status": last_status,
            "ok": False,
        }


def purge_all_posts(client: WordPressClient, log: logging.Logger) -> tuple[int, int]:
    """Permanently delete every post (any status) on the target site.
    Used by `newsporter --purge`."""
    deleted = 0
    failed = 0
    while True:
        r = client._session().get(
            client._url("/posts"),
            params={"per_page": 100, "status": "any", "context": "edit"},
            timeout=client.timeout,
        )
        if r.status_code >= 400:
            log.error("List posts failed (%d): %s", r.status_code, r.text[:200])
            break
        posts = r.json()
        if not posts:
            break
        progress = 0
        for p in posts:
            d = client._session().delete(
                client._url(f"/posts/{p['id']}"),
                params={"force": "true"},
                timeout=client.timeout,
            )
            if d.status_code < 400:
                deleted += 1
                progress += 1
            else:
                failed += 1
                log.warning("Delete %s failed (%d): %s", p["id"], d.status_code, d.text[:120])
        if progress == 0:
            break
    return deleted, failed
