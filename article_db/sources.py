"""Read the existing outputs: embedded chunks (scripts/embed_chunks.py), stage-1 pages, and the
image pipeline's refs / Commons metadata / SigLIP2 embeddings. Nothing is re-embedded."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import quote

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import ArticlesConfig, StageError

logger = logging.getLogger("wikivoyage.articles.sources")

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMONS_FILE_PAGE = "https://commons.wikimedia.org/wiki/"
NORM_TOL = 1e-3


# ---------------------------------------------------------------------------
# Local / S3 files
# ---------------------------------------------------------------------------
def _split_s3(uri: str) -> tuple[str, str]:
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, (prefix.rstrip("/") + "/") if prefix else ""


def _wanted(name: str, prefix: Optional[str]) -> bool:
    return (name.endswith(".parquet") and not name.startswith(("_", "."))
            and (prefix is None or name.startswith(prefix)))


def list_parquet(source: str, cache_dir: str, *, name_prefix: Optional[str] = None) -> list[str]:
    """Local paths of the Parquet files directly under ``source`` (S3 objects are cached in
    ``cache_dir`` and only re-downloaded when the size differs)."""
    if source.startswith("s3://"):
        import boto3

        bucket, prefix = _split_s3(source)
        s3 = boto3.client("s3")
        objs = [o for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")
                for o in page.get("Contents", []) if _wanted(os.path.basename(o["Key"]), name_prefix)]
        os.makedirs(cache_dir, exist_ok=True)
        out = []
        for o in sorted(objs, key=lambda o: o["Key"]):
            local = os.path.join(cache_dir, os.path.basename(o["Key"]))
            if not (os.path.exists(local) and os.path.getsize(local) == o["Size"]):
                logger.info("downloading s3://%s/%s", bucket, o["Key"])
                s3.download_file(bucket, o["Key"], local + ".tmp")
                os.replace(local + ".tmp", local)
            out.append(local)
        return out
    if not os.path.isdir(source):
        raise StageError(f"{source} not found", 1)
    return sorted(os.path.join(source, f) for f in os.listdir(source) if _wanted(f, name_prefix))


def read_manifest(source: str) -> tuple[dict, str]:
    """manifest.json of the embedded chunks and its SHA-256."""
    if source.startswith("s3://"):
        import boto3

        bucket, prefix = _split_s3(source)
        data = boto3.client("s3").get_object(Bucket=bucket, Key=prefix + "manifest.json")["Body"].read()
    else:
        path = os.path.join(source, "manifest.json")
        if not os.path.exists(path):
            raise StageError(f"{path} not found (written by scripts/embed_chunks.py)", 5)
        with open(path, "rb") as fh:
            data = fh.read()
    # Hash the parsed content so whitespace differences between copies don't matter.
    manifest = json.loads(data)
    sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest, sha


# ---------------------------------------------------------------------------
# scripts/embed_chunks.py (chunking rules are taken from the script itself, not copied)
# ---------------------------------------------------------------------------
_EMBED_CHUNKS = None


def embed_chunks_module():
    global _EMBED_CHUNKS
    if _EMBED_CHUNKS is None:
        path = REPO_ROOT / "scripts" / "embed_chunks.py"
        spec = importlib.util.spec_from_file_location("_wikivoyage_embed_chunks", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EMBED_CHUNKS = mod
    return _EMBED_CHUNKS


def manifest_page_types(manifest: dict) -> Optional[set[str]]:
    v = str(manifest.get("page_types", "*")).strip()
    return None if v == "*" else {t.strip() for t in v.split(",")}


def expected_chunks(page: dict, manifest: dict) -> list[tuple[str, str]]:
    """(chunk_id, chunk_text) that embed_chunks.py produces for this page with the manifest settings."""
    ec = embed_chunks_module()
    return [(c["chunk_id"], c["chunk_text"]) for c in ec.build_chunks(
        [page], manifest["max_chars"], manifest["min_chars"], manifest_page_types(manifest),
        bool(manifest.get("skip_stub")))]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
PAGE_COLUMNS = ["page_id", "title", "page_type", "status", "is_part_of", "url", "revision_id", "timestamp"]


def load_pages(paths: list[str], *, with_sections: bool) -> dict[int, dict]:
    """page_id → page row. Duplicate page_ids keep the first, like embed_chunks.read_pages."""
    cols = PAGE_COLUMNS + (["sections"] if with_sections else [])
    tables = [pq.read_table(p, columns=cols) for p in paths]
    if not tables:
        raise StageError("no page Parquet files found", 5)
    rows = pa.concat_tables(tables, promote_options="default").to_pylist()
    rows.sort(key=lambda r: r["page_id"])
    pages: dict[int, dict] = {}
    for r in rows:
        pages.setdefault(r["page_id"], r)
    return pages


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------
@dataclass
class PageChunks:
    page_id: int
    rows: list[dict]          # chunk_id, title, page_type, status, is_part_of, section, url, chunk_text
    vectors: np.ndarray       # (len(rows), dim) float32


@dataclass
class ChunkStats:
    rows: int = 0
    invalid: int = 0
    duplicate: int = 0
    invalid_examples: list = field(default_factory=list)


def _vectors(col: pa.ChunkedArray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """(matrix (n, dim), valid mask). Rows with a wrong length / null / non-unit norm are invalid."""
    arr = col.combine_chunks()
    n = len(arr)
    lengths = pc.fill_null(pc.list_value_length(arr), 0).to_numpy(zero_copy_only=False)
    valid = lengths == dim
    mat = np.zeros((n, dim), dtype=np.float32)
    if valid.all():
        mat[:] = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, dim)
    else:
        offsets = arr.offsets.to_numpy()
        values = arr.values.to_numpy(zero_copy_only=False)
        for i in np.flatnonzero(valid):
            mat[i] = values[offsets[i]:offsets[i + 1]]
    norms = np.linalg.norm(mat, axis=1)
    valid &= np.abs(norms - 1.0) <= NORM_TOL
    return mat, valid


def iter_page_chunks(paths: list[str], dim: int, stats: ChunkStats) -> Iterator[PageChunks]:
    """Stream chunks grouped by page_id. The parts are in page_id order (embed_chunks sorts pages);
    an out-of-order page_id is an error rather than a silently split article."""
    cur_id: Optional[int] = None
    rows: list[dict] = []
    vecs: list[np.ndarray] = []
    seen_ids: set[str] = set()
    done_pages: set[int] = set()

    def flush() -> Optional[PageChunks]:
        if cur_id is None or not rows:
            return None
        return PageChunks(cur_id, list(rows), np.stack(vecs) if vecs else np.zeros((0, dim), np.float32))

    for path in paths:
        t = pq.read_table(path)
        mat, valid = _vectors(t.column("embedding"), dim)
        scalars = t.drop_columns(["embedding"]).to_pylist()
        for i, r in enumerate(scalars):
            stats.rows += 1
            pid = r["page_id"]
            if pid != cur_id:
                if pid in done_pages or (cur_id is not None and pid < cur_id):
                    raise StageError(f"{path}: page_id {pid} after {cur_id}; chunk parts are not in page_id order", 5)
                pc_ = flush()
                if pc_ is not None:
                    yield pc_
                if cur_id is not None:
                    done_pages.add(cur_id)
                cur_id, rows, vecs, seen_ids = pid, [], [], set()
            if not valid[i]:
                stats.invalid += 1
                if len(stats.invalid_examples) < 10:
                    stats.invalid_examples.append(r["chunk_id"])
                continue
            if r["chunk_id"] in seen_ids:
                stats.duplicate += 1
                continue
            seen_ids.add(r["chunk_id"])
            rows.append(r)
            vecs.append(mat[i])
    pc_ = flush()
    if pc_ is not None:
        yield pc_


# ---------------------------------------------------------------------------
# Images (outputs of the image_embedding pipeline)
# ---------------------------------------------------------------------------
@dataclass
class ImageSource:
    refs_by_page: dict[int, list[dict]]
    commons: dict[str, dict]            # file_key → Commons row (status ok)
    vectors: np.ndarray                 # (n, dim)
    index: dict[str, int]               # file_key → row in vectors (sha1 matches Commons)
    model_id: str
    model_revision: str
    slug: str
    dim: int
    dump_date: str

    def elements(self, page_id: int) -> tuple[list[dict], int, list[Optional[int]]]:
        """Image elements of one article (position order, one per record_id), the number of refs
        still pending (no metadata / no embedding yet) and the refs' revision ids."""
        out, pending, revs, seen = [], 0, [], set()
        for ref in self.refs_by_page.get(page_id, ()):
            revs.append(ref.get("revision_id"))
            meta = self.commons.get(ref["file_key"])
            idx = self.index.get(ref["file_key"])
            if meta is None or idx is None:
                pending += 1
                continue
            if ref["record_id"] in seen:
                continue
            seen.add(ref["record_id"])
            out.append(image_element(ref, meta, self.vectors[idx]))
        return out, pending, revs


def image_element(ref: dict, meta: dict, vec: np.ndarray) -> dict:
    """Same field rules as image_embedding.upsert.build_record."""
    page = meta.get("description_url") or (
        COMMONS_FILE_PAGE + quote(ref["file_title"].replace(" ", "_"), safe=":_()',-"))
    return {
        "record_id": ref["record_id"],
        "position": ref["position"],
        "file_title": ref["file_title"],
        "file_key": ref["file_key"],
        "commons_sha1": meta.get("commons_sha1") or "",
        "image_url": meta.get("thumb_url") or "",
        "image_page_url": page,
        "caption": ref.get("caption") or "",
        "section_path": "",
        "license": meta.get("license_short_name") or "unknown",
        "license_url": meta.get("license_url") or "",
        "artist": meta.get("artist") or "",
        "embedding": vec,
    }


def load_images(cfg: ArticlesConfig) -> ImageSource:
    from image_embedding import commons as commons_stage
    from image_embedding import embed as embed_stage
    from image_embedding.config import ImagesConfig
    from image_embedding.storage import open_store

    icfg = ImagesConfig.from_env()
    icfg.workdir = cfg.images_workdir
    icfg.dump_date = cfg.image_dump_date
    dump_date = icfg.require_dump_date()
    store = open_store(icfg.workdir)
    if not store.exists(icfg.refs_rel()):
        raise StageError(f"{store.uri(icfg.refs_rel())} not found; run the image pipeline "
                         f"(python -m image_embedding all --dump-date {dump_date}) or pass --no-images", 1)
    slug = embed_stage.resolve_slug(store, icfg, cfg.image_model_slug)
    info = store.read_json(f"{icfg.embeddings_dir(slug)}/model_info.json")
    dim = int(info["dim"])

    commons = {r["file_key"]: r for r in commons_stage.load_done(store, icfg).values() if r.get("status") == "ok"}

    # Embedding parts are read oldest first, so a newer vector for the same file_key wins.
    latest: dict[str, np.ndarray] = {}
    for table in embed_stage.iter_parquet_parts(store, slug, columns=["file_key", "commons_sha1", "vector"]):
        mat, valid = _vectors(table.column("vector"), dim)
        for i, (key, sha) in enumerate(zip(table.column("file_key").to_pylist(),
                                           table.column("commons_sha1").to_pylist())):
            meta = commons.get(key)
            # A stale vector (image re-uploaded since) never replaces a matching one.
            if valid[i] and meta is not None and meta.get("commons_sha1") == sha:
                latest[key] = mat[i]
    keys = list(latest)
    vectors = np.stack([latest[k] for k in keys]) if keys else np.zeros((0, dim), np.float32)
    index = {k: i for i, k in enumerate(keys)}

    refs_by_page: dict[int, list[dict]] = {}
    for ref in store.iter_jsonl(icfg.refs_rel()):
        refs_by_page.setdefault(ref["source_page_id"], []).append(ref)
    for refs in refs_by_page.values():
        refs.sort(key=lambda r: r["position"])

    logger.info("images: %d refs on %d pages, %d Commons ok, %d usable vectors (slug=%s, dim=%d)",
                sum(map(len, refs_by_page.values())), len(refs_by_page), len(commons), len(keys), slug, dim)
    return ImageSource(refs_by_page, commons, vectors, index, info.get("model_id", icfg.model_id),
                       info.get("model_revision") or "", slug, dim, dump_date)
