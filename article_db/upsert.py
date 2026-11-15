"""Stages ``upsert`` / ``replay`` and ``status``.

``upsert`` sends the articles of a snapshot whose ``content_hash`` differs from what was last
sent (``state/done-*.jsonl``, last-wins per page_id). Batches are bounded by count and bytes;
a failed batch (after retries) is saved as Parquet under ``snapshots/<id>/upsert/failed/`` and
can be resent with ``replay``.
"""
from __future__ import annotations

import io
import logging
import time
from typing import Any, Iterator, Optional

import pyarrow.parquet as pq

from image_embedding.storage import PartWriter, Store, last_wins

from .assemble import snapshot_dir, write_parquet
from .collection import ArticleCollection, SchemaMismatch, make_client
from .config import ArticlesConfig, StageError
from .schema import IMAGE_DIM, TEXT_DIM, row_bytes

logger = logging.getLogger("wikivoyage.articles.upsert")

STATE_DIR = "state"


# ---------------------------------------------------------------------------
# Snapshots and state
# ---------------------------------------------------------------------------
def list_snapshots(store: Store) -> list[str]:
    """Complete snapshots (those with snapshot.json), oldest first."""
    out = set()
    for rel in store.list("snapshots"):
        parts = rel.split("/")
        if len(parts) == 3 and parts[2] == "snapshot.json":
            out.add(parts[1])
    return sorted(out)


def resolve_snapshot(store: Store, explicit: Optional[str] = None) -> tuple[str, dict]:
    if explicit:
        rel = f"{snapshot_dir(explicit)}/snapshot.json"
        if not store.exists(rel):
            raise StageError(f"snapshot {explicit!r} not found or incomplete ({store.uri(rel)})", 1)
        return explicit, store.read_json(rel)
    snaps = list_snapshots(store)
    if not snaps:
        raise StageError(f"no snapshot in {store.uri('snapshots')}; run 'assemble' first", 1)
    return snaps[-1], store.read_json(f"{snapshot_dir(snaps[-1])}/snapshot.json")


def snapshot_parts(store: Store, sid: str) -> list[str]:
    prefix = snapshot_dir(sid)
    return [r for r in store.list(prefix) if r.count("/") == 2 and r.rsplit("/", 1)[-1].startswith("part-")
            and r.endswith(".parquet")]


def read_parquet(store: Store, rel: str, columns: Optional[list[str]] = None):
    return pq.read_table(io.BytesIO(store.read_bytes(rel)), columns=columns)


def iter_snapshot_rows(store: Store, sid: str) -> Iterator[dict]:
    for rel in snapshot_parts(store, sid):
        yield from read_parquet(store, rel).to_pylist()


def load_done(store: Store) -> dict[int, str]:
    """page_id → content_hash last sent ("" = deleted)."""
    merged = last_wins(store.iter_jsonl_parts(STATE_DIR, stem="done-"), key=lambda r: r["page_id"])
    return {pid: r.get("content_hash", "") for pid, r in merged.items()}


def open_collection(cfg: ArticlesConfig, meta: Optional[dict] = None, client: Any = None) -> ArticleCollection:
    if client is None:
        if not cfg.zilliz_uri or not cfg.zilliz_token:
            raise StageError("ZILLIZ_URI and ZILLIZ_TOKEN must be set", 1)
        client = make_client(cfg.zilliz_uri, cfg.zilliz_token)
    meta = meta or {}
    return ArticleCollection(client, cfg.collection, text_dim=meta.get("text_dim", TEXT_DIM),
                             image_dim=meta.get("image_dim", IMAGE_DIM),
                             max_chunks=meta.get("max_chunks", cfg.max_chunks),
                             max_images=meta.get("max_images", cfg.max_images))


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------
class _Sender:
    def __init__(self, store: Store, coll: Optional[ArticleCollection], cfg: ArticlesConfig, sid: str,
                 run_id: str, dry_run: bool) -> None:
        self.store, self.coll, self.cfg, self.sid, self.run_id, self.dry_run = store, coll, cfg, sid, run_id, dry_run
        self.done = PartWriter(store, STATE_DIR, run_id, flush_every=10 ** 9, stem="done-")
        self.batch: list[dict] = []
        self.bytes = 0
        self.seq = 0
        self.sent = self.sent_bytes = self.failed_rows = 0
        self.failed_files: list[str] = []
        self.started = time.monotonic()

    def add(self, row: dict) -> None:
        n = row_bytes(row)
        if self.batch and (len(self.batch) >= self.cfg.batch_size or self.bytes + n > self.cfg.max_batch_bytes):
            self.send()
        self.batch.append(row)
        self.bytes += n

    def send(self) -> None:
        if not self.batch:
            return
        rows, nbytes = self.batch, self.bytes
        self.batch, self.bytes = [], 0
        if self.dry_run:
            self.sent += len(rows)
            self.sent_bytes += nbytes
            return
        now = int(time.time() * 1000)
        t0 = time.monotonic()
        try:
            self.coll.upsert([{**r, "created_at": now} for r in rows])
        except Exception as e:  # noqa: BLE001 — saved for replay
            rel = f"{snapshot_dir(self.sid)}/upsert/failed/batch-{self.run_id}-{self.seq:05d}.parquet"
            self.seq += 1
            write_parquet(self.store, rel, rows)
            self.failed_rows += len(rows)
            self.failed_files.append(rel)
            logger.error("batch of %d articles failed (%s); saved to %s", len(rows), str(e)[:300], rel)
            return
        for r in rows:
            self.done.add({"page_id": r["page_id"], "content_hash": r["content_hash"], "snapshot_id": self.sid})
        self.done.flush()  # resume granularity = one batch
        self.sent += len(rows)
        self.sent_bytes += nbytes
        rate = self.sent / max(1e-6, time.monotonic() - self.started)
        logger.info("upserted %d articles (%.1f MB) in %.1fs — total %d (%.1f/s)",
                    len(rows), nbytes / 1e6, time.monotonic() - t0, self.sent, rate)

    def close(self) -> None:
        self.send()
        self.done.close()


def run(cfg: ArticlesConfig, store: Store, *, run_id: str, snapshot: Optional[str] = None, prune: bool = False,
        dry_run: bool = False, force: bool = False, limit: int = 0, client: Any = None) -> dict:
    sid, meta = resolve_snapshot(store, snapshot)
    coll: Optional[ArticleCollection] = None
    created = False
    if not dry_run:
        coll = open_collection(cfg, meta, client)
        try:
            created = coll.ensure()
        except SchemaMismatch as e:
            raise StageError(str(e), 3) from e
        if created:
            logger.info("created collection %r", cfg.collection)

    # A freshly created collection is empty whatever the state files say.
    done = {} if (force or created) else load_done(store)
    logger.info("upsert snapshot %s → %s (%d articles recorded as sent)%s", sid, cfg.collection, len(done),
                " [dry run]" if dry_run else "")

    sender = _Sender(store, coll, cfg, sid, run_id, dry_run)
    total = unchanged = 0
    keep: set[int] = set()
    for row in iter_snapshot_rows(store, sid):
        total += 1
        keep.add(row["page_id"])
        if done.get(row["page_id"]) == row["content_hash"]:
            unchanged += 1
            continue
        if limit and sender.sent + len(sender.batch) >= limit:
            continue
        sender.add(row)
    sender.close()

    stats: dict[str, Any] = {
        "snapshot_id": sid, "articles": total, "unchanged": unchanged, "upserted": sender.sent,
        "upserted_mb": round(sender.sent_bytes / 1e6, 1), "failed_rows": sender.failed_rows,
        "failed_batches": len(sender.failed_files), "dry_run": dry_run, "created_collection": created,
    }
    if coll is not None:
        coll.finish()
        if prune:
            if sender.failed_rows or not meta.get("complete") or limit:
                logger.warning("skipping --prune (failed rows=%d, complete snapshot=%s, limit=%s)",
                               sender.failed_rows, meta.get("complete"), limit)
                stats["pruned"] = "skipped"
            else:
                extra = sorted(pid for pid in coll.iter_page_ids() if pid not in keep)
                if extra:
                    coll.delete_ids(extra)
                    gone = PartWriter(store, STATE_DIR, run_id + "-prune", flush_every=10 ** 9, stem="done-")
                    for pid in extra:
                        gone.add({"page_id": pid, "content_hash": "", "snapshot_id": sid})
                    gone.close()
                stats["pruned"] = len(extra)
        stats["collection_rows"] = coll.count()
    logger.info("upsert done: %s", stats)
    return stats


def replay(cfg: ArticlesConfig, store: Store, *, run_id: str, snapshot: Optional[str] = None,
           file: Optional[str] = None, client: Any = None) -> dict:
    if file:
        files = [file]
    else:
        prefixes = [snapshot_dir(snapshot)] if snapshot else ["snapshots"]
        files = [r for p in prefixes for r in store.list(p) if "/upsert/failed/" in r and r.endswith(".parquet")]
    if not files:
        logger.info("replay: no failed batches")
        return {"files": 0, "upserted": 0, "failed_batches": 0}
    coll = open_collection(cfg, None, client)
    try:
        coll.ensure()
    except SchemaMismatch as e:
        raise StageError(str(e), 3) from e
    done = PartWriter(store, STATE_DIR, run_id, flush_every=10 ** 9, stem="done-")
    upserted, still_failed = 0, []
    for rel in files:
        rows = read_parquet(store, rel).to_pylist()
        now = int(time.time() * 1000)
        try:
            coll.upsert([{**r, "created_at": now} for r in rows])
        except Exception as e:  # noqa: BLE001
            logger.error("replay of %s failed: %s", rel, str(e)[:300])
            still_failed.append(rel)
            continue
        for r in rows:
            done.add({"page_id": r["page_id"], "content_hash": r["content_hash"], "snapshot_id": r["snapshot_id"]})
        done.flush()
        upserted += len(rows)
        store.move(rel, rel.replace("/upsert/failed/", "/upsert/replayed/"))
        logger.info("replayed %s (%d articles)", rel, len(rows))
    done.close()
    coll.finish()
    return {"files": len(files), "upserted": upserted, "failed_batches": len(still_failed)}


def status(cfg: ArticlesConfig, store: Store, *, snapshot: Optional[str] = None, client: Any = None) -> dict:
    out: dict[str, Any] = {"workdir": store.uri(""), "collection": cfg.collection, "snapshots": list_snapshots(store)}
    if out["snapshots"]:
        sid, meta = resolve_snapshot(store, snapshot)
        done = load_done(store)
        hashes = {}
        for rel in snapshot_parts(store, sid):
            t = read_parquet(store, rel, columns=["page_id", "content_hash"])
            hashes.update(zip(t.column("page_id").to_pylist(), t.column("content_hash").to_pylist()))
        sent = sum(1 for pid, h in hashes.items() if done.get(pid) == h)
        out["snapshot"] = {"id": sid, "complete": meta.get("complete"), "articles": len(hashes),
                           "sent": sent, "pending": len(hashes) - sent, "stats": meta.get("stats"),
                           "failed_batches": [r for r in store.list(snapshot_dir(sid)) if "/upsert/failed/" in r]}
    if client is not None or (cfg.zilliz_uri and cfg.zilliz_token):
        coll = open_collection(cfg, None, client)
        out["collection_exists"] = coll.client.has_collection(cfg.collection)
        out["collection_rows"] = coll.count() if out["collection_exists"] else None
    return out
