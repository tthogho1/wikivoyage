"""Stages ``upsert`` and ``replay``: join refs × Commons metadata × embeddings and write to Zilliz.

One row per (article, image) pair, keyed by the deterministic ``record_id``. Upserted
ids are recorded per dump so an interrupted run resumes; failed batches are written
to ``runs/<date>/upsert/failed/`` with their vectors so ``replay`` can resend them.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional
from urllib.parse import quote

from . import commons as commons_stage
from . import embed as embed_stage
from .config import ImagesConfig, StageError
from .storage import PartWriter, Store
from .zilliz import ImageCollection, SchemaMismatch, make_client

logger = logging.getLogger("wikivoyage.images.upsert")

COMMONS_FILE_PAGE = "https://commons.wikimedia.org/wiki/"


def build_record(ref: dict, meta: dict, emb: dict, *, created_at: int) -> dict:
    image_page_url = meta.get("description_url") or (
        COMMONS_FILE_PAGE + quote(ref["file_title"].replace(" ", "_"), safe=":_()',-")
    )
    return {
        "id": ref["record_id"],
        "vector": emb["vector"],
        "wiki": ref["wiki"],
        "page_id": ref["source_page_id"],
        "page_title": ref["source_page_title"],
        "page_url": ref["page_url"],
        "revision_id": ref.get("revision_id"),
        "image_title": ref["file_title"],
        "image_url": meta["thumb_url"],
        "original_url": meta.get("original_url"),
        "image_page_url": image_page_url,
        "caption": ref.get("caption"),
        "position": ref["position"],
        "license": meta.get("license_short_name") or "unknown",
        "license_url": meta.get("license_url"),
        "artist": meta.get("artist"),
        "credit": meta.get("credit"),
        "mime_type": meta.get("mime") or "unknown",
        "width": meta.get("width"),
        "height": meta.get("height"),
        "file_key": ref["file_key"],
        "commons_sha1": meta.get("commons_sha1"),
        "dump_date": ref["dump_date"],
        "embedding_model": emb["model_id"],
        "embedding_revision": emb["model_revision"],
        "created_at": created_at,
    }


def load_done_ids(store: Store, cfg: ImagesConfig) -> set[int]:
    return {r["id"] for r in store.iter_jsonl_parts(cfg.upsert_dir(), stem="done-")}


def _open_collection(cfg: ImagesConfig, dim: int, client=None, *, scalar_indexes: bool = True) -> ImageCollection:
    if client is None:
        if not cfg.zilliz_uri or not cfg.zilliz_token:
            raise StageError("ZILLIZ_URI and ZILLIZ_TOKEN must be set", 1)
        client = make_client(cfg.zilliz_uri, cfg.zilliz_token)
    return ImageCollection(client, cfg.collection, dim, scalar_indexes=scalar_indexes)


class _BatchSender:
    def __init__(self, store: Store, cfg: ImagesConfig, coll: Optional[ImageCollection], run_id: str, dry_run: bool) -> None:
        self.store, self.cfg, self.coll, self.run_id, self.dry_run = store, cfg, coll, run_id, dry_run
        self.done = PartWriter(store, cfg.upsert_dir(), run_id, cfg.flush_every, stem="done-")
        self.batch: list[dict] = []
        self.seq = 0
        self.upserted = 0
        self.failed_rows = 0
        self.failed_batches: list[str] = []

    def add(self, row: dict) -> None:
        self.batch.append(row)
        if len(self.batch) >= self.cfg.zilliz_batch_size:
            self.send()

    def send(self) -> None:
        if not self.batch:
            return
        rows, self.batch = self.batch, []
        if self.dry_run:
            self.upserted += len(rows)
            return
        t0 = time.monotonic()
        try:
            self.coll.upsert(rows)
        except Exception as e:
            rel = f"{self.cfg.upsert_dir()}/failed/batch-{self.run_id}-{self.seq:05d}.jsonl"
            self.seq += 1
            self.store.write_bytes(rel, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
            logger.error("upsert batch of %d failed (%s); saved to %s", len(rows), e, rel)
            self.failed_rows += len(rows)
            self.failed_batches.append(rel)
            return
        for r in rows:
            self.done.add({"id": r["id"]})
        self.upserted += len(rows)
        logger.info("upserted %d rows in %.2fs (total %d)", len(rows), time.monotonic() - t0, self.upserted)

    def close(self) -> None:
        self.send()
        self.done.close()


def run(
    cfg: ImagesConfig,
    store: Store,
    *,
    run_id: str,
    prune: bool = False,
    dry_run: bool = False,
    model_slug: Optional[str] = None,
    client: Any = None,
    scalar_indexes: bool = True,
) -> dict:
    if not store.exists(cfg.refs_rel()):
        raise StageError(f"{store.uri(cfg.refs_rel())} not found; run 'extract' first", 1)
    slug = embed_stage.resolve_slug(store, cfg, model_slug)
    info = store.read_json(f"{cfg.embeddings_dir(slug)}/model_info.json")
    dim = int(info["dim"])

    coll: Optional[ImageCollection] = None
    if not dry_run or client is not None or (cfg.zilliz_uri and cfg.zilliz_token):
        coll = _open_collection(cfg, dim, client, scalar_indexes=scalar_indexes)
        try:
            created = coll.ensure(create=not dry_run)
        except SchemaMismatch as e:
            raise StageError(str(e), 3) from e
        if created:
            logger.info("created collection %r", cfg.collection)
    validator = coll or ImageCollection(None, cfg.collection, dim)

    commons = {
        r["file_key"]: r for r in commons_stage.load_done(store, cfg).values() if r.get("status") == "ok"
    }
    refs_by_key: dict[str, list[dict]] = {}
    n_refs = 0
    for ref in store.iter_jsonl(cfg.refs_rel()):
        n_refs += 1
        if ref["file_key"] in commons:
            refs_by_key.setdefault(ref["file_key"], []).append(ref)
    done = load_done_ids(store, cfg)
    logger.info("upsert: %d refs, %d with usable metadata, %d already upserted for %s (slug=%s, dim=%d)",
                n_refs, sum(map(len, refs_by_key.values())), len(done), cfg.dump_date, slug, dim)

    sender = _BatchSender(store, cfg, coll, run_id, dry_run)
    produced: set[int] = set()
    invalid = 0
    created_at = int(time.time() * 1000)
    for table in embed_stage.iter_parquet_parts(store, slug):
        for emb in table.to_pylist():
            meta = commons.get(emb["file_key"])
            if meta is None or meta.get("commons_sha1") != emb["commons_sha1"]:
                continue  # stale vector for an image that was re-uploaded since
            for ref in refs_by_key.get(emb["file_key"], ()):
                rid = ref["record_id"]
                if rid in done or rid in produced:
                    continue
                produced.add(rid)
                try:
                    row = validator.validate(build_record(ref, meta, emb, created_at=created_at))
                except ValueError as e:
                    invalid += 1
                    logger.warning("invalid record for %s / %s: %s", ref["source_page_title"], ref["file_title"], e)
                    continue
                sender.add(row)
    sender.close()

    eligible = {r["record_id"] for refs in refs_by_key.values() for r in refs}
    pending = len(eligible - done - produced)
    stats: dict[str, Any] = {
        "refs": n_refs, "eligible": len(eligible), "already_done": len(eligible & done),
        "upserted": sender.upserted, "invalid": invalid, "pending_embedding": pending,
        "failed_rows": sender.failed_rows, "failed_batches": len(sender.failed_batches),
        "dry_run": dry_run, "model_slug": slug,
    }
    if coll is not None and not dry_run:
        coll.finish()
        if prune:
            if sender.failed_rows or pending:
                logger.warning("skipping --prune: %d failed rows, %d refs still pending embedding",
                               sender.failed_rows, pending)
                stats["pruned"] = "skipped"
            else:
                coll.prune(cfg.wiki, cfg.require_dump_date())
                stats["pruned"] = True
        stats["collection_rows"] = coll.count()
    logger.info("upsert done: %s", stats)
    return stats


def replay(
    cfg: ImagesConfig, store: Store, *, run_id: str, file: Optional[str] = None,
    client: Any = None, scalar_indexes: bool = True,
) -> dict:
    failed_dir = f"{cfg.upsert_dir()}/failed"
    files = [file] if file else [p for p in store.list(failed_dir) if p.endswith(".jsonl")]
    if not files:
        logger.info("replay: nothing to replay in %s", store.uri(failed_dir))
        return {"files": 0, "upserted": 0, "failed_batches": 0}
    done = PartWriter(store, cfg.upsert_dir(), run_id, cfg.flush_every, stem="done-")
    coll: Optional[ImageCollection] = None
    upserted = 0
    still_failed: list[str] = []
    for rel in files:
        rows = list(store.iter_jsonl(rel))
        if not rows:
            store.delete(rel)
            continue
        if coll is None:
            coll = _open_collection(cfg, len(rows[0]["vector"]), client, scalar_indexes=scalar_indexes)
            try:
                coll.ensure()
            except SchemaMismatch as e:
                raise StageError(str(e), 3) from e
        try:
            coll.upsert([coll.validate(r) for r in rows])
        except Exception as e:
            logger.error("replay of %s failed: %s", rel, e)
            still_failed.append(rel)
            continue
        for r in rows:
            done.add({"id": r["id"]})
        upserted += len(rows)
        store.move(rel, f"{cfg.upsert_dir()}/replayed/{rel.rsplit('/', 1)[-1]}")
        logger.info("replayed %s (%d rows)", rel, len(rows))
    done.close()
    if coll is not None:
        coll.finish()
    return {"files": len(files), "upserted": upserted, "failed_batches": len(still_failed)}
