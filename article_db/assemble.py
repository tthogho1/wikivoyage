"""Stage ``assemble``: join chunks × pages × images by page_id into one entity per article and
write them to ``snapshots/<snapshot_id>/part-*.parquet``. ``snapshot.json`` is written last and
marks the snapshot as complete; ``upsert`` refuses snapshots without it."""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import logging
import math
import os
from collections import Counter
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from image_embedding.storage import Store, open_store

from . import sources
from .config import ArticlesConfig, StageError
from .schema import IMAGE_DIM, TEXT_DIM, clean_row, element_bytes, row_bytes, snapshot_schema

logger = logging.getLogger("wikivoyage.articles.assemble")

HASH_EXCLUDED = {"article_embedding", "chunks", "images", "snapshot_id", "content_hash", "created_at"}


# ---------------------------------------------------------------------------
# One article
# ---------------------------------------------------------------------------
def _chunk_order(chunk_id: str) -> tuple[int, int]:
    _pid, sec, part = chunk_id.rsplit("-", 2)
    return int(sec), int(part)


def _unix(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        return int(_dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def content_hash(row: dict) -> str:
    """SHA-256 of everything that ends up in Zilliz except snapshot_id / created_at, so an
    unchanged article is not re-sent by the next snapshot."""
    h = hashlib.sha256()
    meta = {k: v for k, v in row.items() if k not in HASH_EXCLUDED}
    h.update(json.dumps(meta, sort_keys=True, default=str).encode())
    for kind in ("chunks", "images"):
        h.update(f"|{kind}|".encode())
        for e in row.get(kind) or []:
            h.update(json.dumps({k: v for k, v in e.items() if k != "embedding"}, sort_keys=True).encode())
            h.update(np.asarray(e["embedding"], dtype=np.float32).tobytes())
    return h.hexdigest()


def build_article(page: dict, chunks: sources.PageChunks, images: list[dict], *, ctx: dict,
                  max_chunks: int, max_images: int, max_row_bytes: int,
                  truncated_fields: Optional[list[str]] = None) -> tuple[dict, dict]:
    """Return (row, info). ``page`` is the stage-1 page row (or the first chunk row when the
    page is unknown). info: chunks/images dropped by the capacity and the 2 MiB row limits."""
    order = sorted(range(len(chunks.rows)), key=lambda i: _chunk_order(chunks.rows[i]["chunk_id"]))
    elems = []
    for idx, i in enumerate(order):
        r = chunks.rows[i]
        path = r.get("section") or "Overview"
        parts = path.split(" > ")
        elems.append({
            "chunk_id": r["chunk_id"], "chunk_index": idx, "heading": parts[-1], "top_section": parts[0],
            "section_path": path, "url": r.get("url") or "", "text": r["chunk_text"],
            "embedding": chunks.vectors[i],
        })
    imgs = sorted(images, key=lambda e: e["position"])
    info = {"chunks_cut_capacity": max(0, len(elems) - max_chunks),
            "images_cut_capacity": max(0, len(imgs) - max_images),
            "chunks_cut_size": 0, "images_cut_size": 0}
    elems, imgs = elems[:max_chunks], imgs[:max_images]

    mean = np.mean(np.stack([e["embedding"] for e in elems]), axis=0) if elems else np.zeros(TEXT_DIM, np.float32)
    norm = float(np.linalg.norm(mean))
    article_vec = (mean / norm).astype(np.float32) if norm > 0 else mean.astype(np.float32)

    row: dict[str, Any] = {
        "page_id": chunks.page_id,
        "wiki": ctx["wiki"],
        "title": page.get("title"),
        "url": page.get("url"),
        "page_type": page.get("page_type"),
        "status": page.get("status"),
        "is_part_of": page.get("is_part_of"),
        "revision_id": page.get("revision_id"),
        "last_updated": _unix(page.get("timestamp")),
        "article_embedding": article_vec,
        "text_model": ctx["text_model"],
        "image_model": ctx["image_model"],
        "image_model_revision": ctx["image_model_revision"],
        "chunk_manifest_sha": ctx["chunk_manifest_sha"],
        "image_dump_date": ctx["image_dump_date"],
        "snapshot_id": ctx["snapshot_id"],
        "chunks": elems,
        "images": imgs,
    }
    row["chunk_count"], row["image_count"] = len(elems), len(imgs)
    row["chunks_truncated"] = row["images_truncated"] = False
    row = clean_row(row, image_dim=ctx["image_dim"], truncated=truncated_fields)

    # Zilliz rejects a row above 2 MiB: drop images from the end first, then chunks.
    size = row_bytes(row)
    while size > max_row_bytes and (row["images"] or len(row["chunks"]) > 1):
        if row["images"]:
            size -= element_bytes(row["images"].pop(), "images")
            info["images_cut_size"] += 1
        else:
            size -= element_bytes(row["chunks"].pop(), "chunks")
            info["chunks_cut_size"] += 1
    row["chunk_count"], row["image_count"] = len(row["chunks"]), len(row["images"])
    row["chunks_truncated"] = bool(info["chunks_cut_capacity"] or info["chunks_cut_size"])
    row["images_truncated"] = bool(info["images_cut_capacity"] or info["images_cut_size"])
    row["content_hash"] = content_hash(row)
    info["row_bytes"] = row_bytes(row)
    return row, info


# ---------------------------------------------------------------------------
# Snapshot output
# ---------------------------------------------------------------------------
def snapshot_dir(snapshot_id: str) -> str:
    return f"snapshots/{snapshot_id}"


def write_parquet(store: Store, rel: str, rows: list[dict]) -> None:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=snapshot_schema()), buf, compression="zstd")
    store.write_bytes(rel, buf.getvalue())


class SnapshotWriter:
    def __init__(self, store: Store, prefix: str, per_part: int) -> None:
        self.store, self.prefix, self.per_part = store, prefix, per_part
        self.rows: list[dict] = []
        self.seq = 0
        self.total = 0

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.per_part:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        rel = f"{self.prefix}/part-{self.seq:05d}.parquet"
        write_parquet(self.store, rel, self.rows)
        self.total += len(self.rows)
        logger.info("wrote %s (%d articles, %d total)", rel, len(self.rows), self.total)
        self.seq += 1
        self.rows = []


def _write_jsonl(store: Store, rel: str, rows: list[dict]) -> None:
    if rows:
        store.write_bytes(rel, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())


def _cache_root(cfg: ArticlesConfig) -> str:
    return os.path.join(cfg.workdir if not cfg.workdir.startswith("s3://") else "articles_cache", "cache")


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------
def run(cfg: ArticlesConfig, *, run_id: str, store: Optional[Store] = None, limit: int = 0,
        verify: bool = True, allow_partial: bool = False, strict_revision: bool = False,
        images: Optional[sources.ImageSource] = None) -> dict:
    store = store or open_store(cfg.workdir)
    manifest, manifest_sha = sources.read_manifest(cfg.chunks_source)
    if manifest.get("model") != cfg.text_model:
        raise StageError(f"chunks were embedded with {manifest.get('model')!r}, expected {cfg.text_model!r}", 1)

    cache = _cache_root(cfg)
    chunk_paths = sources.list_parquet(cfg.chunks_source, os.path.join(cache, "chunks"), name_prefix="part-")
    expected_parts = math.ceil(manifest["chunks"] / manifest["rows_per_part"])
    partial = False
    if len(chunk_paths) != expected_parts:
        msg = f"{cfg.chunks_source}: {len(chunk_paths)} of {expected_parts} chunk parts present"
        if not allow_partial:
            raise StageError(msg + " (embed_chunks.py not finished?); --allow-partial to build from what exists", 5)
        logger.warning("%s; building a partial snapshot", msg)
        partial = True

    pages_source = cfg.resolved_pages_source()
    pages = sources.load_pages(sources.list_parquet(pages_source, os.path.join(cache, "pages")), with_sections=verify)
    logger.info("pages: %d from %s; chunk parts: %d from %s", len(pages), pages_source, len(chunk_paths), cfg.chunks_source)

    if images is None and not cfg.no_images:
        images = sources.load_images(cfg)
    image_dim = images.dim if images is not None else IMAGE_DIM
    if image_dim != IMAGE_DIM:
        raise StageError(f"image embeddings have dim {image_dim}, the collection schema expects {IMAGE_DIM}", 3)

    snapshot_id = run_id
    ctx = {
        "wiki": cfg.wiki, "text_model": cfg.text_model, "chunk_manifest_sha": manifest_sha,
        "image_model": images.model_id if images else "", "image_model_revision": images.model_revision if images else "",
        "image_dump_date": images.dump_date if images else "", "image_dim": image_dim, "snapshot_id": snapshot_id,
    }
    prefix = snapshot_dir(snapshot_id)
    writer = SnapshotWriter(store, prefix, cfg.articles_per_part)
    chunk_stats = sources.ChunkStats()
    stats: Counter = Counter()
    mismatches: list[dict] = []
    overflow: list[dict] = []
    skipped: list[dict] = []
    truncated_fields: list[str] = []
    seen: set[int] = set()
    max_bytes = 0
    limited = False

    for pcs in sources.iter_page_chunks(chunk_paths, TEXT_DIM, chunk_stats):
        if limit and stats["articles"] >= limit:
            limited = True
            break
        seen.add(pcs.page_id)
        page = pages.get(pcs.page_id)
        if page is None:
            stats["page_missing"] += 1
            mismatches.append({"page_id": pcs.page_id, "reason": "page not in page Parquet"})
            page = {k: pcs.rows[0].get(k) for k in ("title", "page_type", "status", "is_part_of")}
            page["url"] = pcs.rows[0]["url"].split("#", 1)[0]
        elif verify:
            exp = dict(sources.expected_chunks(page, manifest))
            got = {r["chunk_id"]: r["chunk_text"] for r in pcs.rows}
            if any(exp.get(cid) != text for cid, text in got.items()) or not set(exp) >= set(got):
                mismatches.append({"page_id": pcs.page_id, "reason": "chunk_id/chunk_text differ from embed_chunks rules",
                                   "unexpected": sorted(set(got) - set(exp))[:5]})
        if not pcs.rows:
            stats["skipped_no_chunks"] += 1
            skipped.append({"page_id": pcs.page_id, "reason": "no valid chunks"})
            continue

        elems: list[dict] = []
        if images is not None:
            elems, pending, revs = images.elements(pcs.page_id)
            stats["images_pending"] += pending
            if revs and any(r is not None and r != page.get("revision_id") for r in revs):
                stats["revision_mismatch"] += 1
                if strict_revision:
                    stats["images_dropped_revision"] += len(elems)
                    elems = []

        row, info = build_article(page, pcs, elems, ctx=ctx, max_chunks=cfg.max_chunks, max_images=cfg.max_images,
                                  max_row_bytes=cfg.max_row_bytes, truncated_fields=truncated_fields)
        cut = {k: v for k, v in info.items() if k != "row_bytes" and v}
        if cut:
            overflow.append({"page_id": pcs.page_id, "title": row["title"], **cut})
            for k, v in cut.items():
                stats[k] += v
        max_bytes = max(max_bytes, info["row_bytes"])
        stats["articles"] += 1
        stats["chunks"] += row["chunk_count"]
        stats["images"] += row["image_count"]
        stats["articles_with_images"] += bool(row["image_count"])
        writer.add(row)

    complete = not (partial or limited)
    if complete:
        if chunk_stats.rows != manifest["chunks"]:
            mismatches.append({"reason": f"chunk rows {chunk_stats.rows} != manifest chunks {manifest['chunks']}"})
        if verify:
            for pid, page in pages.items():
                if pid not in seen and sources.expected_chunks(page, manifest):
                    mismatches.append({"page_id": pid, "reason": "page should have chunks but has none"})
    if mismatches and verify:
        _write_jsonl(store, f"{prefix}/mismatches.jsonl", mismatches)
        raise StageError(f"{len(mismatches)} inconsistencies between chunks and pages (first: {mismatches[0]}); "
                         f"see {store.uri(prefix + '/mismatches.jsonl')}. The page Parquet must be the one "
                         "embed_chunks.py read (--pages-source), or pass --no-verify-chunks", 5)

    writer.flush()
    _write_jsonl(store, f"{prefix}/overflow.jsonl", overflow)
    _write_jsonl(store, f"{prefix}/skipped.jsonl", skipped)
    result = {
        **{k: stats[k] for k in sorted(stats)},
        "chunk_rows_read": chunk_stats.rows, "invalid_chunks": chunk_stats.invalid,
        "duplicate_chunks": chunk_stats.duplicate, "invalid_examples": chunk_stats.invalid_examples,
        "truncated_fields": dict(Counter(truncated_fields)), "max_row_bytes": max_bytes,
        "mismatches": len(mismatches), "parts": writer.seq,
    }
    store.write_json(f"{prefix}/snapshot.json", {
        "snapshot_id": snapshot_id, "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "complete": complete, "partial_parts": partial, "limit": limit or None,
        "chunks_source": cfg.chunks_source, "chunk_manifest": manifest, "chunk_manifest_sha": manifest_sha,
        "pages_source": pages_source, "images": None if images is None else {
            "workdir": cfg.images_workdir, "dump_date": images.dump_date, "slug": images.slug,
            "model_id": images.model_id, "model_revision": images.model_revision, "dim": images.dim},
        "text_dim": TEXT_DIM, "image_dim": image_dim, "max_chunks": cfg.max_chunks, "max_images": cfg.max_images,
        "max_row_bytes": cfg.max_row_bytes, "stats": result,
    })
    logger.info("snapshot %s: %s", snapshot_id, result)
    return {"snapshot_id": snapshot_id, **result}
