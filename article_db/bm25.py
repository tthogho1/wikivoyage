"""Keyword (BM25) side of the hybrid text search: Zilliz collection ``wikivoyage_chunks_bm25``.

StructArray sub-fields cannot carry an analyzer or a BM25 Function (design §12 item 7), so the chunk
texts of ``wikivoyage_articles`` are mirrored into a flat collection, one row per chunk:

    id = page_id * 1024 + chunk_offset     (max_capacity of chunks[] is 512)
    page_id, chunk_offset                  → the same (page_id, offset) an element-level hit on
                                             chunks[embedding] returns, so both lists merge per element
    text (english analyzer) ──BM25 Function──▶ sparse (SPARSE_INVERTED_INDEX, metric BM25)
    content_hash                           → the article version the offsets belong to; the searcher
                                             drops keyword hits whose hash differs from the article row

Zilliz computes the sparse vectors server-side; nothing is embedded locally.

``build`` reads the same snapshot as ``upsert`` and, for every article whose content_hash differs from
what was last sent (``bm25_state/done-*.jsonl``, last-wins), deletes its old rows and inserts the new ones.
"""
from __future__ import annotations

import io
import logging
import time
from typing import Any, Iterator, Optional

import pyarrow.parquet as pq

from image_embedding.storage import PartWriter, Store, last_wins
from image_embedding.zilliz import RETRY_DELAYS, SchemaMismatch, _type_name, make_client, truncate_utf8

from .config import ArticlesConfig, StageError

logger = logging.getLogger("wikivoyage.articles.bm25")

DEFAULT_COLLECTION = "wikivoyage_chunks_bm25"
STATE_DIR = "bm25_state"
ID_STRIDE = 1024
ANALYZER = {"type": "english"}  # standard tokenizer + lowercase + English stemmer + stop words

# (name, kind, max_length)
FIELDS: tuple[tuple[str, str, int], ...] = (
    ("id", "INT64", 0),
    ("page_id", "INT64", 0),
    ("chunk_offset", "INT64", 0),
    ("chunk_index", "INT64", 0),
    ("title", "VARCHAR", 512),
    ("page_type", "VARCHAR", 32),
    ("is_part_of", "VARCHAR", 512),
    ("top_section", "VARCHAR", 256),
    ("content_hash", "VARCHAR", 64),
    ("text", "VARCHAR", 8192),
    ("sparse", "SPARSE_FLOAT_VECTOR", 0),
)
SCALAR_INDEXED = ("page_id", "page_type", "is_part_of", "title", "top_section")
ARTICLE_COLUMNS = ["page_id", "content_hash", "title", "page_type", "is_part_of"]
CHUNK_COLUMNS = ["chunk_index", "top_section", "text"]


def chunk_rows(article: dict) -> list[dict]:
    """One BM25 row per element of ``article["chunks"]``, in array order (offset = position)."""
    pid = int(article["page_id"])
    out = []
    for off, c in enumerate(article.get("chunks") or []):
        out.append({
            "id": pid * ID_STRIDE + off,
            "page_id": pid,
            "chunk_offset": off,
            "chunk_index": int(c.get("chunk_index", -1) if c.get("chunk_index") is not None else -1),
            "title": truncate_utf8(article.get("title") or "", 512),
            "page_type": truncate_utf8(article.get("page_type") or "", 32),
            "is_part_of": truncate_utf8(article.get("is_part_of") or "", 512),
            "top_section": truncate_utf8(c.get("top_section") or "", 256),
            "content_hash": article.get("content_hash") or "",
            "text": truncate_utf8(c.get("text") or "", 8192),
        })
    return out


class Bm25Collection:
    def __init__(self, client: Any, name: str = DEFAULT_COLLECTION) -> None:
        self.client, self.name = client, name

    def build_schema(self):
        from pymilvus import DataType, Function, FunctionType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False,
                                           description="Wikivoyage chunk texts for BM25 (mirrors wikivoyage_articles.chunks)")
        for name, kind, max_len in FIELDS:
            kw: dict[str, Any] = {}
            if name == "id":
                kw["is_primary"] = True
            if kind == "VARCHAR":
                kw["max_length"] = max_len
            if name == "text":
                kw.update(enable_analyzer=True, analyzer_params=ANALYZER)
            schema.add_field(name, getattr(DataType, kind), **kw)
        schema.add_function(Function(name="text_bm25", function_type=FunctionType.BM25,
                                     input_field_names=["text"], output_field_names=["sparse"]))
        return schema

    def index_params(self):
        p = self.client.prepare_index_params()
        p.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                    params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75})
        for name in SCALAR_INDEXED:
            p.add_index(field_name=name, index_type="INVERTED")
        return p

    def ensure(self, *, create: bool = True) -> bool:
        if self.client.has_collection(self.name):
            self.verify_schema()
            return False
        if not create:
            return False
        logger.info("creating collection %r", self.name)
        self.client.create_collection(collection_name=self.name, schema=self.build_schema(),
                                      index_params=self.index_params())
        return True

    def verify_schema(self) -> None:
        desc = self.client.describe_collection(self.name)
        actual = {f["name"]: f for f in desc.get("fields", [])}
        errors = []
        for name, kind, max_len in FIELDS:
            got = actual.get(name)
            if got is None:
                errors.append(f"missing field {name}")
            elif _type_name(got.get("type")) != kind:
                errors.append(f"{name}: type {_type_name(got.get('type'))} != {kind}")
            elif kind == "VARCHAR" and int((got.get("params") or {}).get("max_length", 0)) < max_len:
                errors.append(f"{name}: max_length too small")
        if not any(str(fn.get("type", "")).upper().endswith("BM25") or fn.get("type") == 1
                   for fn in desc.get("functions", []) or []):
            errors.append("no BM25 function text → sparse")
        if errors:
            raise SchemaMismatch(f"collection {self.name!r} schema mismatch: " + "; ".join(errors))

    def replace_articles(self, page_ids: list[int], rows: list[dict], *, delete: bool = True,
                         sleep=time.sleep) -> None:
        """Delete every row of these articles, then insert their new rows (chunk counts can shrink)."""
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                if delete and page_ids:
                    self.client.delete(collection_name=self.name, filter=f"page_id in {page_ids}")
                if rows:
                    self.client.insert(collection_name=self.name, data=rows)
                return
            except Exception as e:
                if attempt >= len(RETRY_DELAYS):
                    raise
                logger.warning("BM25 batch of %d rows failed (%s); retrying in %ds", len(rows), e, RETRY_DELAYS[attempt])
                sleep(RETRY_DELAYS[attempt])

    def count(self) -> Optional[int]:
        try:
            return int(self.client.get_collection_stats(self.name).get("row_count"))
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Build stage
# ---------------------------------------------------------------------------
def _iter_articles(store: Store, sid: str) -> Iterator[dict]:
    """Snapshot rows with only the columns BM25 needs. ParquetFile.read takes leaf paths, so the
    1024-d chunk vectors and the images are never decoded."""
    from .upsert import snapshot_parts

    leaves = ARTICLE_COLUMNS + [f"chunks.list.element.{c}" for c in CHUNK_COLUMNS]
    for rel in snapshot_parts(store, sid):
        yield from pq.ParquetFile(io.BytesIO(store.read_bytes(rel))).read(columns=leaves).to_pylist()


def load_done(store: Store) -> dict[int, str]:
    merged = last_wins(store.iter_jsonl_parts(STATE_DIR, stem="done-"), key=lambda r: r["page_id"])
    return {pid: r.get("content_hash", "") for pid, r in merged.items()}


def run(cfg: ArticlesConfig, store: Store, *, run_id: str, snapshot: Optional[str] = None,
        collection: str = DEFAULT_COLLECTION, force: bool = False, prune: bool = False, dry_run: bool = False,
        limit: int = 0, batch_rows: int = 2000, max_batch_bytes: int = 8 * 1024 * 1024, client: Any = None) -> dict:
    from .upsert import resolve_snapshot

    sid, meta = resolve_snapshot(store, snapshot)
    coll: Optional[Bm25Collection] = None
    created = False
    if not dry_run:
        if client is None:
            if not cfg.zilliz_uri or not cfg.zilliz_token:
                raise StageError("ZILLIZ_URI and ZILLIZ_TOKEN must be set", 1)
            client = make_client(cfg.zilliz_uri, cfg.zilliz_token)
        coll = Bm25Collection(client, collection)
        try:
            created = coll.ensure()
        except SchemaMismatch as e:
            raise StageError(str(e), 3) from e
    done = {} if (force or created) else load_done(store)
    logger.info("bm25 snapshot %s → %s (%d articles recorded as sent)%s", sid, collection, len(done),
                " [dry run]" if dry_run else "")

    writer = PartWriter(store, STATE_DIR, run_id, flush_every=10 ** 9, stem="done-")
    pending_ids: list[int] = []
    pending_hash: dict[int, str] = {}
    pending_rows: list[dict] = []
    pending_bytes = 0
    stats = {"snapshot_id": sid, "articles": 0, "unchanged": 0, "articles_sent": 0, "rows_sent": 0,
             "mb_sent": 0.0, "dry_run": dry_run, "created_collection": created}
    keep: set[int] = set()
    started = time.monotonic()

    def flush() -> None:
        nonlocal pending_ids, pending_rows, pending_bytes, pending_hash
        if not pending_ids:
            return
        if not dry_run:
            # A new collection is empty: skip the delete. Otherwise delete the articles we have state for
            # and also any we have none for (a previous run may have died between insert and state flush).
            coll.replace_articles(pending_ids, pending_rows, delete=not created)
            for pid in pending_ids:
                writer.add({"page_id": pid, "content_hash": pending_hash[pid], "snapshot_id": sid})
            writer.flush()
        stats["articles_sent"] += len(pending_ids)
        stats["rows_sent"] += len(pending_rows)
        stats["mb_sent"] = round(stats["mb_sent"] + pending_bytes / 1e6, 1)
        rate = stats["rows_sent"] / max(1e-6, time.monotonic() - started)
        logger.info("bm25: %d articles / %d chunks sent (%.0f chunks/s)", stats["articles_sent"], stats["rows_sent"], rate)
        pending_ids, pending_rows, pending_bytes, pending_hash = [], [], 0, {}

    for art in _iter_articles(store, sid):
        stats["articles"] += 1
        pid = int(art["page_id"])
        keep.add(pid)
        if done.get(pid) == art["content_hash"]:
            stats["unchanged"] += 1
            continue
        if limit and stats["articles_sent"] + len(pending_ids) >= limit:
            continue
        rows = chunk_rows(art)
        nbytes = sum(len(r["text"].encode("utf-8")) + 256 for r in rows)
        if pending_rows and (len(pending_rows) + len(rows) > batch_rows or pending_bytes + nbytes > max_batch_bytes):
            flush()
        pending_ids.append(pid)
        pending_hash[pid] = art["content_hash"]
        pending_rows += rows
        pending_bytes += nbytes
    flush()
    writer.close()

    if coll is not None:
        if prune:
            if not meta.get("complete") or limit:
                logger.warning("skipping --prune (complete snapshot=%s, limit=%s)", meta.get("complete"), limit)
                stats["pruned"] = "skipped"
            else:
                extra = sorted(pid for pid, h in load_done(store).items() if h and pid not in keep)
                for i in range(0, len(extra), 500):
                    coll.client.delete(collection_name=collection, filter=f"page_id in {extra[i:i + 500]}")
                if extra:
                    gone = PartWriter(store, STATE_DIR, run_id + "-prune", flush_every=10 ** 9, stem="done-")
                    for pid in extra:
                        gone.add({"page_id": pid, "content_hash": "", "snapshot_id": sid})
                    gone.close()
                stats["pruned"] = len(extra)
        try:
            coll.client.flush(collection_name=collection)
        except Exception as e:  # noqa: BLE001 — advisory on Zilliz Cloud
            logger.warning("flush failed: %s", e)
        stats["collection_rows"] = coll.count()
    logger.info("bm25 done: %s", stats)
    return stats
