"""Stage ``fetch``: download thumbnails, validate and preprocess them, record a manifest.

Images and the manifest are shared across dumps; an image is fetched again only
when its Commons SHA-1 changes (re-upload) or a previous attempt failed.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import mimetypes
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from . import commons as commons_stage
from .config import ImagesConfig
from .http import HttpError, RateLimiter, make_session, request_with_retry
from .preprocess import ImageRejected, encode_jpeg, load_and_validate, to_rgb
from .storage import PartWriter, Store, last_wins

logger = logging.getLogger("wikivoyage.images.fetch")

MAX_BYTES = 20 * 1024 * 1024
DONE_STATUSES = {"ok", "skipped"}


def manifest_key(row: dict) -> tuple[str, Optional[str]]:
    return row["file_key"], row.get("commons_sha1")


def load_manifest(store: Store, cfg: ImagesConfig) -> dict:
    return last_wins(store.iter_jsonl_parts(cfg.manifest_dir()), key=manifest_key)


def _ext_for(url: str, content_type: Optional[str]) -> str:
    path = url.split("?", 1)[0].rsplit("/", 1)[-1]
    if "." in path:
        ext = path.rsplit(".", 1)[-1].lower()
        if 1 <= len(ext) <= 5 and ext.isalnum():
            return ext
    return (mimetypes.guess_extension(content_type or "") or ".bin").lstrip(".")


def _download(session, url: str, limiter: RateLimiter) -> tuple[bytes, Optional[str]]:
    resp = request_with_retry(session, "GET", url, limiter=limiter, stream=True, max_attempts=4)
    with resp:
        ctype = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        length = int(resp.headers.get("Content-Length") or 0)
        if length > MAX_BYTES:
            raise ImageRejected("too_large_bytes", "skipped")
        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            buf.extend(chunk)
            if len(buf) > MAX_BYTES:
                raise ImageRejected("too_large_bytes", "skipped")
    if ctype and not ctype.startswith("image/"):
        raise ImageRejected(f"content_type:{ctype}", "failed")
    return bytes(buf), ctype or None


def fetch_one(cfg: ImagesConfig, store: Store, session, limiter: RateLimiter, meta: dict, attempts: int) -> dict:
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    row = {
        "file_key": meta["file_key"],
        "file_title": meta["file_title"],
        "commons_sha1": meta.get("commons_sha1"),
        "source_url": meta["thumb_url"],
        "attempts": attempts,
        "fetched_at": now,
    }
    try:
        data, ctype = _download(session, meta["thumb_url"], limiter)
        row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), http_content_type=ctype)
        img = load_and_validate(data, min_side=cfg.min_side, max_side=cfg.max_side)
        raw_rel = cfg.raw_rel(meta["file_key"], _ext_for(meta["thumb_url"], ctype))
        store.write_bytes(raw_rel, data)
        row.update(raw_path=raw_rel, raw_width=img.width, raw_height=img.height)
        rgb, ops = to_rgb(img, pad_aspect=cfg.pad_aspect)
        proc_rel = cfg.processed_rel(meta["file_key"])
        store.write_bytes(proc_rel, encode_jpeg(rgb))
        row.update(
            status="ok", reason=None, processed_path=proc_rel,
            processed_width=rgb.width, processed_height=rgb.height, preprocess_ops=ops,
        )
    except ImageRejected as e:
        row.update(status=e.status, reason=e.reason)
    except HttpError as e:
        row.update(status="failed", reason=e.reason)
    except Exception as e:  # never stop the run for one image
        logger.exception("fetch failed for %s", meta["file_title"])
        row.update(status="failed", reason=f"{type(e).__name__}: {e}"[:300])
    return row


def run(cfg: ImagesConfig, store: Store, *, run_id: str, limit: int = 0, session=None) -> dict:
    metas = [
        r for r in commons_stage.load_done(store, cfg).values()
        if r.get("status") == "ok" and r.get("thumb_url")
    ]
    manifest = load_manifest(store, cfg)
    todo: list[tuple[dict, int]] = []
    already = exhausted = 0
    for m in metas:
        prev = manifest.get((m["file_key"], m.get("commons_sha1")))
        if prev is None:
            todo.append((m, 1))
        elif prev["status"] in DONE_STATUSES:
            already += 1
        elif prev.get("attempts", 1) >= cfg.fetch_max_attempts:
            exhausted += 1
        else:
            todo.append((m, prev.get("attempts", 1) + 1))
    if limit > 0:
        todo = todo[:limit]
    logger.info(
        "fetch: %d fetchable images, %d done, %d gave up after %d attempts, %d to fetch",
        len(metas), already, exhausted, cfg.fetch_max_attempts, len(todo),
    )

    session = session or make_session(cfg.user_agent)
    limiter = RateLimiter(cfg.download_rps)
    writer = PartWriter(store, cfg.manifest_dir(), run_id, cfg.flush_every)
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    lock = threading.Lock()

    def work(item: tuple[dict, int]) -> None:
        row = fetch_one(cfg, store, session, limiter, *item)
        writer.add(row)
        with lock:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if row.get("reason"):
                r = row["reason"].split(":", 1)[0]
                reasons[r] = reasons.get(r, 0) + 1
            n = sum(counts.values())
        if n % 500 == 0:
            logger.info("fetch: %d/%d (%s)", n, len(todo), counts)

    with ThreadPoolExecutor(max_workers=cfg.download_concurrency) as pool:
        list(pool.map(work, todo))
    writer.close()
    stats = {
        "fetchable": len(metas), "already_done": already, "gave_up": exhausted, "attempted": len(todo),
        **{f"status_{k}": v for k, v in counts.items()}, "reasons": reasons,
    }
    logger.info("fetch done: %s", stats)
    return stats
