"""Stage ``commons``: resolve each unique file title to URLs, size, MIME, SHA-1 and license data.

Queries Commons ``prop=imageinfo`` 50 titles per request; titles Commons reports as
missing are retried against the source wiki's own API (locally uploaded files).
"""
from __future__ import annotations

import datetime as _dt
import html
import logging
import mimetypes
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from . import ids
from .config import COMMONS_API, ImagesConfig, StageError
from .http import HttpError, make_session, request_with_retry
from .storage import PartWriter, Store, last_wins

logger = logging.getLogger("wikivoyage.images.commons")

BATCH = 50
DONE_STATUSES = {"ok", "missing", "skipped"}
EXTMETADATA_KEYS = (
    "LicenseShortName|LicenseUrl|UsageTerms|AttributionRequired|Artist|Credit|"
    "ImageDescription|DateTimeOriginal"
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def html_to_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", str(value)))).strip()
    return text or None


def strip_tracking(url: Optional[str]) -> Optional[str]:
    """Drop ``utm_*`` query parameters Commons appends to thumbnail URLs."""
    if not url or "?" not in url:
        return url
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.startswith("utm_")]
    return urlunsplit(parts._replace(query=urlencode(query)))


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def query_imageinfo(
    session,
    api_url: str,
    titles: list[str],
    *,
    thumb_width: int,
    sleep=time.sleep,
    max_lag_retries: int = 6,
) -> dict[str, Optional[dict]]:
    """Map each input title to its API page object (None when it cannot be resolved).

    Follows ``continue`` and the ``normalized``/``redirects`` maps.
    """
    params: dict[str, Any] = {
        "action": "query",
        "prop": "imageinfo",
        "iiprop": "url|size|mime|sha1|timestamp|extmetadata",
        "iiurlwidth": str(thumb_width),
        "iiextmetadatafilter": EXTMETADATA_KEYS,
        "iiextmetadatalanguage": "en",
        "format": "json",
        "formatversion": "2",
        "maxlag": "5",
        "redirects": "1",
        "titles": "|".join(titles),
    }
    pages: dict[str, dict] = {}
    alias: dict[str, str] = {}
    cont: dict[str, Any] = {}
    lag_retries = 0
    while True:
        resp = request_with_retry(session, "POST", api_url, data={**params, **cont})
        data = resp.json()
        err = data.get("error")
        if err:
            if err.get("code") == "maxlag" and lag_retries < max_lag_retries:
                lag_retries += 1
                wait = float(resp.headers.get("Retry-After") or 5)
                logger.info("API maxlag (%s); waiting %.0fs", err.get("info"), wait)
                sleep(wait)
                continue
            raise HttpError(f"API error {err.get('code')}: {err.get('info')}", None)
        q = data.get("query", {})
        for m in q.get("normalized", []) + q.get("redirects", []):
            alias[m["from"]] = m["to"]
        for p in q.get("pages", []):
            t = p.get("title")
            prev = pages.get(t)
            if prev is None:
                pages[t] = p
            elif p.get("imageinfo") and not prev.get("imageinfo"):
                prev["imageinfo"] = p["imageinfo"]
        if "continue" in data:
            cont = data["continue"]
            continue
        break

    out: dict[str, Optional[dict]] = {}
    for t in titles:
        c, seen = t, set()
        while c in alias and c not in seen:
            seen.add(c)
            c = alias[c]
        out[t] = pages.get(c)
    return out


def _meta(ext: dict, key: str) -> Optional[str]:
    v = (ext or {}).get(key)
    if isinstance(v, dict):
        v = v.get("value")
    return None if v in (None, "") else str(v)


def build_row(
    file_title: str,
    page: Optional[dict],
    repository: str,
    allowed_mime: Iterable[str],
) -> dict:
    """Turn an API page object into a commons/part-*.jsonl row."""
    row: dict[str, Any] = {
        "file_title": file_title,
        "file_key": ids.file_key(file_title),
        "canonical_title": (page or {}).get("title", file_title),
        "repository": repository,
        "fetched_at": _now(),
    }
    infos = (page or {}).get("imageinfo") or []
    if not infos:
        row.update(status="missing", reason="missing")
        return row
    ii = infos[0]
    ext = ii.get("extmetadata") or {}
    mime = (ii.get("mime") or "").lower()
    thumb_url = strip_tracking(ii.get("thumburl"))
    row.update({
        "thumb_url": thumb_url,
        "thumb_width": ii.get("thumbwidth"),
        "thumb_height": ii.get("thumbheight"),
        "thumb_mime": (mimetypes.guess_type(thumb_url.split("?", 1)[0])[0] if thumb_url else None),
        "original_url": strip_tracking(ii.get("url")),
        "width": ii.get("width"),
        "height": ii.get("height"),
        "mime": mime,
        "size": ii.get("size"),
        "commons_sha1": ii.get("sha1"),
        "upload_timestamp": ii.get("timestamp"),
        "description_url": ii.get("descriptionurl"),
        "license_short_name": html_to_text(_meta(ext, "LicenseShortName")),
        "license_url": _meta(ext, "LicenseUrl"),
        "usage_terms": html_to_text(_meta(ext, "UsageTerms")),
        "attribution_required": _meta(ext, "AttributionRequired"),
        "artist_html": _meta(ext, "Artist"),
        "artist": html_to_text(_meta(ext, "Artist")),
        "credit_html": _meta(ext, "Credit"),
        "credit": html_to_text(_meta(ext, "Credit")),
        "image_description": html_to_text(_meta(ext, "ImageDescription")),
        "date_time_original": html_to_text(_meta(ext, "DateTimeOriginal")),
    })
    if mime not in set(allowed_mime):
        row.update(status="skipped", reason=f"mime_not_allowed:{mime or 'unknown'}")
    elif not thumb_url:
        if mime == "image/svg+xml":
            row.update(status="skipped", reason="svg_no_render")
        else:
            row.update(status="ok", reason=None, thumb_url=strip_tracking(ii.get("url")), thumb_mime=mime)
    else:
        row.update(status="ok", reason=None)
    return row


def load_done(store: Store, cfg: ImagesConfig) -> dict[str, dict]:
    return last_wins(store.iter_jsonl_parts(cfg.commons_dir()), key=lambda r: r["file_title"])


def unique_titles(store: Store, cfg: ImagesConfig) -> list[str]:
    if not store.exists(cfg.refs_rel()):
        raise StageError(f"{store.uri(cfg.refs_rel())} not found; run 'extract' first", 1)
    seen: dict[str, None] = {}
    for r in store.iter_jsonl(cfg.refs_rel()):
        seen.setdefault(r["file_title"], None)
    return list(seen)


def run(
    cfg: ImagesConfig,
    store: Store,
    *,
    run_id: str,
    retry_errors: bool = True,
    limit: int = 0,
    session=None,
) -> dict:
    titles = unique_titles(store, cfg)
    done = load_done(store, cfg)

    def pending(t: str) -> bool:
        r = done.get(t)
        if r is None:
            return True
        return r["status"] == "error" and retry_errors

    todo = [t for t in titles if pending(t)]
    if limit > 0:
        todo = todo[:limit]
    logger.info("commons: %d unique titles, %d already done, %d to query", len(titles), len(titles) - len(todo), len(todo))

    session = session or make_session(cfg.user_agent)
    writer = PartWriter(store, cfg.commons_dir(), run_id, cfg.flush_every)
    counts: dict[str, int] = {}
    lock = threading.Lock()

    def record(row: dict) -> None:
        writer.add(row)
        with lock:
            counts[row["status"]] = counts.get(row["status"], 0) + 1

    def work(batch: list[str]) -> None:
        try:
            res = query_imageinfo(session, COMMONS_API, batch, thumb_width=cfg.thumb_width)
            missing = [t for t in batch if not (res.get(t) or {}).get("imageinfo")]
            local: dict[str, Optional[dict]] = {}
            if missing:
                try:
                    local = query_imageinfo(session, cfg.local_api_url, missing, thumb_width=cfg.thumb_width)
                except HttpError as e:
                    logger.warning("local API fallback failed: %s", e)
            for t in batch:
                lp = local.get(t)
                if t in missing and lp and lp.get("imageinfo") and lp.get("imagerepository") == "local":
                    record(build_row(t, lp, "local", cfg.allowed_mime))
                else:
                    record(build_row(t, res.get(t), "commons", cfg.allowed_mime))
        except Exception as e:
            reason = e.reason if isinstance(e, HttpError) else f"{type(e).__name__}: {e}"[:300]
            logger.error("commons batch of %d failed: %s", len(batch), reason)
            for t in batch:
                record({"file_title": t, "file_key": ids.file_key(t), "status": "error",
                        "reason": reason, "fetched_at": _now()})

    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    with ThreadPoolExecutor(max_workers=cfg.api_concurrency) as pool:
        for i, _ in enumerate(pool.map(work, batches), 1):
            if i % 20 == 0:
                logger.info("commons: %d/%d batches (%s)", i, len(batches), counts)
    writer.close()
    stats = {"unique_titles": len(titles), "queried": len(todo), **{f"status_{k}": v for k, v in counts.items()}}
    logger.info("commons done: %s", stats)
    return stats
