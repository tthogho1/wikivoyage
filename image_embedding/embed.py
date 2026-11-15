"""Stage ``embed``: SigLIP2 vectors for every fetched image not embedded yet.

Output: ``embeddings/<model_slug>/part-<run>-<seq>.parquet`` (vector = list<float32>),
shared across dumps and keyed by (file_key, commons_sha1).

Unless ``keep_images`` (``--keep-images`` / ``IMAGES_KEEP_FILES=true``), the downloaded files
(``images/raw/`` and ``images/processed/``) of every embedded image are deleted at the end of the
run, after the vectors are written. Only the vectors and metadata are needed downstream; fetch and
embed skip finished images by their manifest / embedding records, not by the files. Files of
images that failed are kept so the next run can retry them. Re-embedding with another model means
downloading again.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import logging
import sys
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from . import fetch as fetch_stage
from . import ids
from .config import ImagesConfig, StageError
from .storage import PartWriter, Store

logger = logging.getLogger("wikivoyage.images.embed")

SCHEMA = pa.schema([
    ("file_key", pa.string()),
    ("file_title", pa.string()),
    ("commons_sha1", pa.string()),
    ("processed_sha256", pa.string()),
    ("vector", pa.list_(pa.float32())),
    ("model_id", pa.string()),
    ("model_revision", pa.string()),
    ("embedded_at", pa.timestamp("ms", tz="UTC")),
])


def write_parquet_part(store: Store, rel: str, rows: list[dict]) -> None:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), buf, compression="zstd")
    store.write_bytes(rel, buf.getvalue())


def iter_parquet_parts(store: Store, slug: str, columns: Optional[list[str]] = None):
    """Yield one pyarrow Table per part file, oldest first."""
    for rel in store.list(ImagesConfig.embeddings_dir(slug)):
        name = rel.rsplit("/", 1)[-1]
        if name.startswith("part-") and name.endswith(".parquet"):
            yield pq.read_table(io.BytesIO(store.read_bytes(rel)), columns=columns)


def done_keys(store: Store, slug: str) -> set[tuple[str, Optional[str]]]:
    out: set = set()
    for t in iter_parquet_parts(store, slug, columns=["file_key", "commons_sha1"]):
        out.update(zip(t.column("file_key").to_pylist(), t.column("commons_sha1").to_pylist()))
    return out


def resolve_slug(store: Store, cfg: ImagesConfig, explicit: Optional[str] = None) -> str:
    """Embedding dir to read: explicit slug, pinned revision, or the newest dir for the model."""
    if explicit:
        return explicit
    if cfg.model_revision:
        return ids.model_slug(cfg.model_id, cfg.model_revision)
    prefix = cfg.model_id.replace("/", "__") + "@"
    best: tuple[str, str] | None = None
    for rel in store.list("embeddings"):
        parts = rel.split("/")
        if len(parts) == 3 and parts[2] == "model_info.json" and parts[1].startswith(prefix):
            created = store.read_json(rel).get("created_at", "")
            if best is None or created > best[0]:
                best = (created, parts[1])
    if best is None:
        raise StageError(f"no embeddings found for {cfg.model_id}; run 'embed' first", 1)
    return best[1]


class _Sink:
    def __init__(self, store: Store, slug: str, run_id: str, flush_every: int) -> None:
        self.store, self.slug, self.run_id, self.flush_every = store, slug, run_id, flush_every
        self.rows: list[dict] = []
        self.seq = 0
        self.total = 0

    def add(self, row: dict) -> None:
        self.rows.append(row)
        self.total += 1
        if len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        rel = f"{ImagesConfig.embeddings_dir(self.slug)}/part-{self.run_id}-{self.seq:05d}.parquet"
        write_parquet_part(self.store, rel, self.rows)
        logger.info("embed: wrote %s (%d vectors, %d total)", rel, len(self.rows), self.total)
        self.seq += 1
        self.rows = []


def delete_embedded_files(store: Store, ok_rows: list[dict], done: set) -> int:
    """Delete raw/processed files of images whose current (file_key, commons_sha1) is embedded."""
    keys = {r["file_key"] for r in ok_rows if (r["file_key"], r.get("commons_sha1")) in done}
    n = 0
    for prefix in ("images/raw", "images/processed"):
        for rel in store.list(prefix):
            if rel.rsplit("/", 1)[-1].split(".", 1)[0] in keys:
                store.delete(rel)
                n += 1
    return n


def _embed_with_split(embedder, images: list, metas: list[dict], on_ok, on_err) -> None:
    """Embed a batch; on failure halve it, down to single images, so only bad ones are dropped."""
    try:
        vecs = embedder.embed_images(images)
    except Exception as e:
        if len(images) == 1:
            on_err(metas[0], f"{type(e).__name__}: {e}"[:300])
            return
        logger.warning("batch of %d failed (%s); splitting", len(images), type(e).__name__)
        _clear_cache()
        mid = len(images) // 2
        _embed_with_split(embedder, images[:mid], metas[:mid], on_ok, on_err)
        _embed_with_split(embedder, images[mid:], metas[mid:], on_ok, on_err)
        return
    for m, v in zip(metas, vecs):
        on_ok(m, v)


def _clear_cache() -> None:
    torch = sys.modules.get("torch")  # only a loaded model has a device cache to free
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def run(cfg: ImagesConfig, store: Store, *, run_id: str, limit: int = 0, embedder=None) -> dict:
    from PIL import Image

    ok_rows = [r for r in fetch_stage.load_manifest(store, cfg).values() if r.get("status") == "ok"]
    if embedder is None:
        from .embedder import SiglipImageEmbedder

        try:
            embedder = SiglipImageEmbedder(
                cfg.model_id, cfg.model_revision, device=cfg.device, dtype=cfg.dtype, hf_token=cfg.hf_token
            )
        except Exception as e:
            raise StageError(f"failed to load model {cfg.model_id}: {e}", 1) from e
    slug = ids.model_slug(embedder.model_id, embedder.model_revision)
    done = done_keys(store, slug)
    todo = [r for r in ok_rows if (r["file_key"], r.get("commons_sha1")) not in done]
    already = len(ok_rows) - len(todo)
    if limit > 0:
        todo = todo[:limit]
    logger.info("embed: %d fetched images, %d already embedded, %d to embed (slug=%s)",
                len(ok_rows), already, len(todo), slug)

    sink = _Sink(store, slug, run_id, cfg.flush_every)
    errors = PartWriter(store, f"{cfg.embeddings_dir(slug)}/errors", run_id, cfg.flush_every)
    now = lambda: _dt.datetime.now(_dt.timezone.utc)  # noqa: E731

    embedded: set = set()

    def on_ok(meta: dict, vec) -> None:
        embedded.add((meta["file_key"], meta.get("commons_sha1")))
        sink.add({
            "file_key": meta["file_key"], "file_title": meta["file_title"],
            "commons_sha1": meta.get("commons_sha1"), "processed_sha256": meta.get("_processed_sha256"),
            "vector": vec.tolist(), "model_id": embedder.model_id,
            "model_revision": embedder.model_revision, "embedded_at": now(),
        })

    def on_err(meta: dict, reason: str) -> None:
        logger.warning("embed failed for %s: %s", meta["file_title"], reason)
        errors.add({"file_key": meta["file_key"], "file_title": meta["file_title"],
                    "commons_sha1": meta.get("commons_sha1"), "reason": reason,
                    "at": now().isoformat(timespec="seconds")})

    bs = max(1, cfg.embed_batch_size)
    for i in range(0, len(todo), bs):
        images, metas = [], []
        for meta in todo[i:i + bs]:
            try:
                data = store.read_bytes(meta["processed_path"])
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as e:
                on_err(meta, f"read_failed: {type(e).__name__}: {e}"[:300])
                continue
            images.append(img)
            metas.append({**meta, "_processed_sha256": hashlib.sha256(data).hexdigest()})
        if images:
            _embed_with_split(embedder, images, metas, on_ok, on_err)
        if (i // bs) % 20 == 0:
            logger.info("embed: %d/%d", min(i + bs, len(todo)), len(todo))
    sink.flush()
    errors.close()

    info = dict(embedder.info)
    info["created_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    info_rel = f"{cfg.embeddings_dir(slug)}/model_info.json"
    if store.exists(info_rel):
        info["created_at"] = store.read_json(info_rel).get("created_at", info["created_at"])
    store.write_json(info_rel, info)
    stats = {"fetched_ok": len(ok_rows), "attempted": len(todo), "embedded": sink.total,
             "errors": errors.total, "model_slug": slug, "dim": embedder.dim}
    # After sink.flush(): the vectors are on disk, so the image files are no longer needed.
    # The sweep also removes files left by earlier runs (e.g. one that crashed mid-way).
    stats["deleted_files"] = 0 if cfg.keep_images else delete_embedded_files(store, ok_rows, done | embedded)
    logger.info("embed done: %s", stats)
    return stats
