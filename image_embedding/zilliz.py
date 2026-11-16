"""Zilliz collection ``wikivoyage_images``: schema, create/verify, record validation, upsert, prune."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from .upserter import Upserter

logger = logging.getLogger("wikivoyage.images.zilliz")

RETRY_DELAYS = (1, 2, 4)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str  # INT64 | VARCHAR | FLOAT_VECTOR
    max_length: int = 0
    nullable: bool = False
    is_primary: bool = False


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("id", "INT64", is_primary=True),
    FieldSpec("vector", "FLOAT_VECTOR"),
    FieldSpec("wiki", "VARCHAR", 32),
    FieldSpec("page_id", "INT64"),
    FieldSpec("page_title", "VARCHAR", 512),
    FieldSpec("page_url", "VARCHAR", 2048),
    FieldSpec("revision_id", "INT64", nullable=True),
    FieldSpec("image_title", "VARCHAR", 1024),
    FieldSpec("image_url", "VARCHAR", 2048),
    FieldSpec("original_url", "VARCHAR", 2048, nullable=True),
    FieldSpec("image_page_url", "VARCHAR", 2048),
    FieldSpec("caption", "VARCHAR", 2048, nullable=True),
    FieldSpec("position", "INT64"),
    FieldSpec("license", "VARCHAR", 256),
    FieldSpec("license_url", "VARCHAR", 1024, nullable=True),
    FieldSpec("artist", "VARCHAR", 1024, nullable=True),
    FieldSpec("credit", "VARCHAR", 2048, nullable=True),
    FieldSpec("mime_type", "VARCHAR", 64),
    FieldSpec("width", "INT64", nullable=True),
    FieldSpec("height", "INT64", nullable=True),
    FieldSpec("file_key", "VARCHAR", 40),
    FieldSpec("commons_sha1", "VARCHAR", 40, nullable=True),
    FieldSpec("dump_date", "VARCHAR", 16),
    FieldSpec("embedding_model", "VARCHAR", 128),
    FieldSpec("embedding_revision", "VARCHAR", 64),
    FieldSpec("created_at", "INT64"),
)
FIELD_NAMES = [f.name for f in FIELDS]
SCALAR_INDEXED = ("page_title", "page_id", "dump_date")
OUTPUT_FIELDS = [f.name for f in FIELDS if f.name != "vector"]


class SchemaMismatch(Exception):
    pass


def truncate_utf8(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    return b[:max_bytes].decode("utf-8", errors="ignore")


def escape_filter_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _type_name(t: Any) -> str:
    name = getattr(t, "name", None)
    if name:
        return str(name)
    from pymilvus import DataType

    try:
        return DataType(int(t)).name
    except Exception:
        return str(t)


class ImageCollection:
    def __init__(self, client: Any, name: str, dim: int, *, scalar_indexes: bool = True) -> None:
        self.client = client
        self.name = name
        self.dim = dim
        self.scalar_indexes = scalar_indexes
        self.upserter = Upserter(client)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _build_schema(self):
        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        for f in FIELDS:
            kw: dict[str, Any] = {}
            if f.is_primary:
                kw["is_primary"] = True
            if f.kind == "VARCHAR":
                kw["max_length"] = f.max_length
            if f.kind == "FLOAT_VECTOR":
                kw["dim"] = self.dim
            if f.nullable:
                kw["nullable"] = True
            schema.add_field(f.name, getattr(DataType, f.kind), **kw)
        return schema

    def _index_params(self):
        params = self.client.prepare_index_params()
        params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        if self.scalar_indexes:
            for name in SCALAR_INDEXED:
                params.add_index(field_name=name, index_type="INVERTED")
        return params

    def ensure(self, *, create: bool = True) -> bool:
        """Create the collection if absent (returns True), else verify its schema."""
        if self.client.has_collection(self.name):
            self.verify_schema()
            return False
        if not create:
            return False
        logger.info("creating collection %r (dim=%d)", self.name, self.dim)
        self.client.create_collection(
            collection_name=self.name,
            schema=self._build_schema(),
            index_params=self._index_params(),
            description="Wikivoyage article images – SigLIP2 image embeddings",
        )
        return True

    def verify_schema(self) -> list[str]:
        """Raise SchemaMismatch on incompatible differences; return warnings for harmless ones."""
        desc = self.client.describe_collection(self.name)
        actual = {f["name"]: f for f in desc.get("fields", [])}
        errors: list[str] = []
        warnings: list[str] = []
        for spec in FIELDS:
            f = actual.get(spec.name)
            if f is None:
                errors.append(f"missing field {spec.name!r}")
                continue
            tname = _type_name(f.get("type"))
            if tname != spec.kind:
                errors.append(f"field {spec.name!r}: type {tname} != {spec.kind}")
                continue
            params = f.get("params", {}) or {}
            if spec.is_primary and not f.get("is_primary"):
                errors.append(f"field {spec.name!r} is not the primary key")
            if spec.kind == "FLOAT_VECTOR" and int(params.get("dim", 0)) != self.dim:
                errors.append(f"vector dim {params.get('dim')} != {self.dim}")
            if spec.kind == "VARCHAR" and int(params.get("max_length", 0)) < spec.max_length:
                errors.append(f"field {spec.name!r}: max_length {params.get('max_length')} < {spec.max_length}")
            if spec.nullable and not f.get("nullable", False):
                errors.append(f"field {spec.name!r} must be nullable")
        for name, f in actual.items():
            if name not in FIELD_NAMES:
                (warnings if f.get("nullable") else errors).append(f"unexpected field {name!r}")
        if desc.get("auto_id"):
            errors.append("auto_id must be disabled")
        if desc.get("enable_dynamic_field"):
            errors.append("dynamic fields must be disabled")
        try:
            idx = self.client.describe_index(self.name, index_name="vector") or {}
            metric = (idx.get("metric_type") or "").upper()
            if metric and metric != "COSINE":
                errors.append(f"vector metric {metric} != COSINE")
        except Exception as e:
            warnings.append(f"could not read vector index: {e}")
        for w in warnings:
            logger.warning("schema: %s", w)
        if errors:
            raise SchemaMismatch(f"collection {self.name!r} schema mismatch: " + "; ".join(errors))
        return warnings

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------
    def validate(self, row: dict) -> dict:
        """Check required fields, types and dim; truncate VARCHARs by UTF-8 bytes. Returns a clean row."""
        out: dict[str, Any] = {}
        for spec in FIELDS:
            v = row.get(spec.name)
            if v is None:
                if not spec.nullable:
                    raise ValueError(f"{spec.name} is required")
                out[spec.name] = None
                continue
            if spec.kind == "INT64":
                if isinstance(v, bool) or not isinstance(v, int):
                    raise ValueError(f"{spec.name} must be int, got {type(v).__name__}")
            elif spec.kind == "VARCHAR":
                v = str(v)
                t = truncate_utf8(v, spec.max_length)
                if t != v:
                    logger.warning("truncated %s for id=%s (%d → %d bytes)",
                                   spec.name, row.get("id"), len(v.encode()), len(t.encode()))
                v = t
            elif spec.kind == "FLOAT_VECTOR":
                v = [float(x) for x in v]
                if len(v) != self.dim:
                    raise ValueError(f"vector dim {len(v)} != {self.dim}")
            out[spec.name] = v
        return out

    def upsert(self, rows: list[dict], *, sleep=time.sleep) -> None:
        """Upsert with up to 3 retries (1, 2, 4 s); re-raises the last error."""
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                self.upserter.upsert(self.name, rows)
                return
            except Exception as e:
                if attempt >= len(RETRY_DELAYS):
                    raise
                logger.warning("upsert of %d rows failed (%s); retrying in %ds", len(rows), e, RETRY_DELAYS[attempt])
                sleep(RETRY_DELAYS[attempt])

    def finish(self) -> None:
        self.upserter.flush(self.name)
        self.upserter.load_collection(self.name)

    def prune(self, wiki: str, dump_date: str) -> Any:
        flt = f'wiki == "{escape_filter_value(wiki)}" and dump_date != "{escape_filter_value(dump_date)}"'
        logger.info("pruning rows matching %s", flt)
        return self.client.delete(collection_name=self.name, filter=flt)

    def count(self) -> Optional[int]:
        try:
            return int(self.upserter.get_collection_stats(self.name).get("row_count"))
        except Exception:
            return None


def make_client(uri: str, token: str):
    from pymilvus import MilvusClient

    if not uri:
        raise ValueError("ZILLIZ_URI is not set")
    return MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)
