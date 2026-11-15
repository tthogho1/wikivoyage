"""Stage ``extract``: stream the dump and write every article's image references to refs.jsonl."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Optional

from wikivoyage_stream import WikivoyageDumpStreamReader
from wikivoyage_stream import filters

from . import ids
from .config import ImagesConfig, StageError
from .storage import Store
from .wikitext_images import extract_refs

logger = logging.getLogger("wikivoyage.images.extract")


def _to_int(v) -> Optional[int]:
    return int(v) if v is not None and str(v).isdigit() else None


def _page_type_filter(cfg: ImagesConfig):
    if cfg.page_types.strip() in ("", "*"):
        return None
    allowed = {t.strip() for t in cfg.page_types.split(",") if t.strip()}
    from get_wikivoyage import detect_page_type_and_status

    return lambda text: detect_page_type_and_status(text)[0] in allowed


def resolve_source(cfg: ImagesConfig, store: Store, dump_path: Optional[str]) -> str:
    source = dump_path or os.environ.get("WIKIVOYAGE_DUMP", "").strip()
    if source:
        return source
    rel = cfg.dump_rel()
    if not store.exists(rel):
        raise StageError(f"dump not found at {store.uri(rel)}; run the 'dump' stage or pass --dump-path", 1)
    return store.uri(rel)


def run(cfg: ImagesConfig, store: Store, *, dump_path: Optional[str] = None, limit: int = 0) -> dict:
    date = cfg.require_dump_date()
    source = resolve_source(cfg, store, dump_path)
    type_ok = _page_type_filter(cfg)
    logger.info("extracting image refs from %s (dump_date=%s)", source, date)

    reader = WikivoyageDumpStreamReader(source)
    stats = {
        "pages_seen": 0, "articles": 0, "articles_with_images": 0, "refs": 0,
        "truncated": 0, "ref_errors": 0, "skipped_not_article": 0, "skipped_type": 0,
    }
    fd, tmp_refs = tempfile.mkstemp(prefix="refs-", suffix=".jsonl")
    fd_err, tmp_err = tempfile.mkstemp(prefix="extract-errors-", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out, os.fdopen(fd_err, "w", encoding="utf-8") as err_out:
            for page in reader.iter_pages():
                stats["pages_seen"] += 1
                if not filters.is_article(page):
                    stats["skipped_not_article"] += 1
                    continue
                if type_ok is not None and not type_ok(page.text):
                    stats["skipped_type"] += 1
                    continue
                page_id = _to_int(page.page_id)
                if page_id is None:
                    stats["skipped_not_article"] += 1
                    continue
                stats["articles"] += 1
                errors: list[dict] = []
                try:
                    refs, truncated = extract_refs(page.text, max_per_page=cfg.max_per_page, errors=errors)
                except Exception as e:  # never stop the whole run for one page
                    logger.exception("extract failed for %r", page.title)
                    errors.append({"raw": "", "reason": f"page_error:{type(e).__name__}"})
                    refs, truncated = [], 0
                for e in errors:
                    e.update({"source_page_id": page_id, "source_page_title": page.title})
                    err_out.write(json.dumps(e, ensure_ascii=False) + "\n")
                stats["ref_errors"] += len(errors)
                if truncated:
                    stats["truncated"] += truncated
                    logger.info("%s: %d image refs over the cap of %d dropped", page.title, truncated, cfg.max_per_page)
                if refs:
                    stats["articles_with_images"] += 1
                url = filters.page_url(page.title or "", cfg.site_base)
                for r in refs:
                    out.write(json.dumps({
                        "wiki": cfg.wiki,
                        "dump_date": date,
                        "source_page_id": page_id,
                        "source_page_title": page.title,
                        "page_url": url,
                        "revision_id": _to_int(page.revision_id),
                        "revision_timestamp": page.timestamp,
                        "file_title": r.file_title,
                        "file_key": ids.file_key(r.file_title),
                        "record_id": ids.record_id(cfg.wiki, page_id, r.file_title),
                        "position": r.position,
                        "caption": r.caption,
                        "context": r.context,
                        "source_kind": r.source_kind,
                        "template_name": r.template_name,
                        "param_name": r.param_name,
                    }, ensure_ascii=False) + "\n")
                stats["refs"] += len(refs)
                if limit > 0 and stats["articles"] >= limit:
                    break
        store.put_file(cfg.refs_rel(), tmp_refs)
        store.put_file(cfg.extract_errors_rel(), tmp_err)
    finally:
        for p in (tmp_refs, tmp_err):
            if os.path.exists(p):
                os.unlink(p)
    stats["reader_compressed_bytes"] = reader.compressed_bytes
    logger.info("extract done: %s → %s", stats, store.uri(cfg.refs_rel()))
    return stats
