"""Field layout of ``wikivoyage_articles``, shared by the snapshot Parquet and the Zilliz collection.

Sub-fields are never null (StructArray element-level null is not used); missing strings are
``""`` and missing integers ``-1``. Top-level fields are not nullable either, for the same rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pyarrow as pa

from image_embedding.zilliz import truncate_utf8

TEXT_DIM = 1024   # Qwen/Qwen3-Embedding-0.6B
IMAGE_DIM = 1152  # google/siglip2-so400m-patch16-384

# Zilliz rejects a single row above 2 MiB ("too large to fit in one WAL message"); the estimate
# below matches the server's within ~0.1 % in the spike, so keep a margin.
WAL_ROW_LIMIT = 2_097_152
DEFAULT_MAX_ROW_BYTES = 1_900_000


@dataclass(frozen=True)
class F:
    name: str
    kind: str  # INT64 | VARCHAR | BOOL | FLOAT_VECTOR
    max_length: int = 0


TOP: tuple[F, ...] = (
    F("page_id", "INT64"),
    F("wiki", "VARCHAR", 32),
    F("title", "VARCHAR", 512),
    F("url", "VARCHAR", 2048),
    F("page_type", "VARCHAR", 32),
    F("status", "VARCHAR", 32),
    F("is_part_of", "VARCHAR", 512),
    F("revision_id", "INT64"),
    F("last_updated", "INT64"),
    F("chunk_count", "INT64"),
    F("image_count", "INT64"),
    F("chunks_truncated", "BOOL"),
    F("images_truncated", "BOOL"),
    F("article_embedding", "FLOAT_VECTOR"),
    F("text_model", "VARCHAR", 128),
    F("image_model", "VARCHAR", 128),
    F("image_model_revision", "VARCHAR", 64),
    F("chunk_manifest_sha", "VARCHAR", 64),
    F("image_dump_date", "VARCHAR", 16),
    F("snapshot_id", "VARCHAR", 64),
    F("content_hash", "VARCHAR", 64),
    F("created_at", "INT64"),  # set by upsert, not stored in the snapshot
)

CHUNK: tuple[F, ...] = (
    F("chunk_id", "VARCHAR", 64),
    F("chunk_index", "INT64"),
    F("heading", "VARCHAR", 256),
    F("top_section", "VARCHAR", 256),
    F("section_path", "VARCHAR", 1024),
    F("url", "VARCHAR", 2048),
    F("text", "VARCHAR", 8192),
    F("embedding", "FLOAT_VECTOR"),
)

IMAGE: tuple[F, ...] = (
    F("record_id", "INT64"),
    F("position", "INT64"),
    F("file_title", "VARCHAR", 1024),
    F("file_key", "VARCHAR", 40),
    F("commons_sha1", "VARCHAR", 40),
    F("image_url", "VARCHAR", 2048),
    F("image_page_url", "VARCHAR", 2048),
    F("caption", "VARCHAR", 2048),
    F("section_path", "VARCHAR", 1024),
    F("license", "VARCHAR", 256),
    F("license_url", "VARCHAR", 1024),
    F("artist", "VARCHAR", 1024),
    F("embedding", "FLOAT_VECTOR"),
)

STRUCTS = {"chunks": CHUNK, "images": IMAGE}
SNAPSHOT_ONLY_EXCLUDED = {"created_at"}
VECTOR_DIMS = {"article_embedding": TEXT_DIM, "chunks[embedding]": TEXT_DIM, "images[embedding]": IMAGE_DIM}


# ---------------------------------------------------------------------------
# Arrow (snapshot) schema
# ---------------------------------------------------------------------------
def _arrow_type(f: F) -> pa.DataType:
    return {"INT64": pa.int64(), "VARCHAR": pa.string(), "BOOL": pa.bool_(),
            "FLOAT_VECTOR": pa.list_(pa.float32())}[f.kind]


def snapshot_schema() -> pa.Schema:
    cols = [(f.name, _arrow_type(f)) for f in TOP if f.name not in SNAPSHOT_ONLY_EXCLUDED]
    for name, fields in STRUCTS.items():
        cols.append((name, pa.list_(pa.struct([(f.name, _arrow_type(f)) for f in fields]))))
    return pa.schema(cols)


# ---------------------------------------------------------------------------
# Row cleaning and size estimate
# ---------------------------------------------------------------------------
def _clean(values: dict, fields: tuple[F, ...], dims: dict[str, int], where: str,
           truncated: Optional[list[str]] = None) -> dict:
    out: dict[str, Any] = {}
    for f in fields:
        v = values.get(f.name)
        if f.kind == "VARCHAR":
            v = "" if v is None else str(v)
            t = truncate_utf8(v, f.max_length)
            if t != v and truncated is not None:
                truncated.append(f"{where}.{f.name}")
            v = t
        elif f.kind == "INT64":
            v = -1 if v is None else int(v)
        elif f.kind == "BOOL":
            v = bool(v)
        elif f.kind == "FLOAT_VECTOR":
            if v is None or len(v) != dims[f.name]:
                raise ValueError(f"{where}.{f.name}: expected {dims[f.name]} dims, got {None if v is None else len(v)}")
        out[f.name] = v
    return out


def clean_row(row: dict, *, text_dim: int = TEXT_DIM, image_dim: int = IMAGE_DIM,
              truncated: Optional[list[str]] = None) -> dict:
    """Defaults for missing values, UTF-8 byte truncation of VARCHARs, vector dim checks."""
    top = [f for f in TOP if f.name not in SNAPSHOT_ONLY_EXCLUDED or f.name in row]
    out = _clean(row, tuple(top), {"article_embedding": text_dim}, "article", truncated)
    out["chunks"] = [_clean(c, CHUNK, {"embedding": text_dim}, "chunks", truncated) for c in row.get("chunks") or []]
    out["images"] = [_clean(i, IMAGE, {"embedding": image_dim}, "images", truncated) for i in row.get("images") or []]
    return out


def _bytes(values: dict, fields: tuple[F, ...]) -> int:
    n = 0
    for f in fields:
        v = values.get(f.name)
        if f.kind == "VARCHAR":
            n += len((v or "").encode("utf-8"))
        elif f.kind == "INT64":
            n += 8
        elif f.kind == "BOOL":
            n += 1
        elif f.kind == "FLOAT_VECTOR":
            n += 4 * (len(v) if v is not None else 0)
    return n


def element_bytes(elem: dict, kind: str) -> int:
    return _bytes(elem, STRUCTS[kind])


def row_bytes(row: dict) -> int:
    """Approximate serialized size of one entity (what Zilliz checks against the 2 MiB WAL limit)."""
    return (_bytes(row, TOP)
            + sum(element_bytes(c, "chunks") for c in row.get("chunks") or [])
            + sum(element_bytes(i, "images") for i in row.get("images") or []))
