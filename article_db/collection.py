"""Zilliz collection ``wikivoyage_articles``: create, verify, upsert (whole-entity replace), prune."""
from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Optional

from image_embedding.zilliz import RETRY_DELAYS, SchemaMismatch, _type_name, make_client  # noqa: F401

from .schema import CHUNK, IMAGE, IMAGE_DIM, STRUCTS, TEXT_DIM, TOP

logger = logging.getLogger("wikivoyage.articles.collection")

HNSW = {"M": 16, "efConstruction": 200}
SCALAR_INDEXED = ("page_type", "title", "is_part_of", "snapshot_id")
STRUCT_SCALAR_INDEXED = ("chunks[top_section]", "chunks[heading]")


class ArticleCollection:
    def __init__(self, client: Any, name: str, *, text_dim: int = TEXT_DIM, image_dim: int = IMAGE_DIM,
                 max_chunks: int = 512, max_images: int = 64) -> None:
        self.client, self.name = client, name
        self.dims = {"article_embedding": text_dim, "chunks": text_dim, "images": image_dim}
        self.capacity = {"chunks": max_chunks, "images": max_images}

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _add(self, target, f, dim: int, **extra) -> None:
        from pymilvus import DataType

        kw: dict[str, Any] = dict(extra)
        if f.kind == "VARCHAR":
            kw["max_length"] = f.max_length
        if f.kind == "FLOAT_VECTOR":
            kw["dim"] = dim
        target.add_field(f.name, getattr(DataType, f.kind), **kw)

    def build_schema(self):
        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False,
                                           description="Wikivoyage articles: Qwen3 text chunks + SigLIP2 images")
        for f in TOP:
            self._add(schema, f, self.dims["article_embedding"], **({"is_primary": True} if f.name == "page_id" else {}))
        for name, fields in STRUCTS.items():
            st = self.client.create_struct_field_schema()
            for f in fields:
                self._add(st, f, self.dims[name])
            schema.add_field(name, DataType.ARRAY, element_type=DataType.STRUCT, struct_schema=st,
                             max_capacity=self.capacity[name])
        return schema

    def index_params(self):
        p = self.client.prepare_index_params()
        p.add_index(field_name="article_embedding", index_type="AUTOINDEX", metric_type="COSINE")
        # Element-level search (COSINE), not EmbeddingList/MAX_SIM: a hit carries the element offset.
        p.add_index(field_name="chunks[embedding]", index_type="HNSW", metric_type="COSINE", params=HNSW)
        p.add_index(field_name="images[embedding]", index_type="HNSW", metric_type="COSINE", params=HNSW)
        for name in SCALAR_INDEXED + STRUCT_SCALAR_INDEXED:
            p.add_index(field_name=name, index_type="INVERTED")
        return p

    def ensure(self, *, create: bool = True) -> bool:
        """Create the collection if absent (returns True), else verify its schema."""
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
        errors: list[str] = []

        def check(f, got: Optional[dict], where: str, dim: int) -> None:
            if got is None:
                errors.append(f"missing field {where}{f.name}")
                return
            tname = _type_name(got.get("type"))
            if tname != f.kind:
                errors.append(f"{where}{f.name}: type {tname} != {f.kind}")
                return
            params = got.get("params") or {}
            if f.kind == "FLOAT_VECTOR" and int(params.get("dim", 0)) != dim:
                errors.append(f"{where}{f.name}: dim {params.get('dim')} != {dim}")
            if f.kind == "VARCHAR" and int(params.get("max_length", 0)) < f.max_length:
                errors.append(f"{where}{f.name}: max_length {params.get('max_length')} < {f.max_length}")

        for f in TOP:
            check(f, actual.get(f.name), "", self.dims["article_embedding"])
        if not actual.get("page_id", {}).get("is_primary"):
            errors.append("page_id is not the primary key")
        for name, fields in STRUCTS.items():
            got = actual.get(name)
            if got is None or "struct_fields" not in got:
                errors.append(f"missing StructArray field {name}")
                continue
            cap = int((got.get("params") or {}).get("max_capacity", 0))
            if cap < self.capacity[name]:
                errors.append(f"{name}: max_capacity {cap} < {self.capacity[name]}")
            subs = {s["name"]: s for s in got["struct_fields"]}
            for f in fields:
                check(f, subs.get(f.name), f"{name}.", self.dims[name])
        expected = {f.name for f in TOP} | set(STRUCTS)
        errors += [f"unexpected field {n}" for n in actual if n not in expected]
        if desc.get("auto_id"):
            errors.append("auto_id must be disabled")
        for field in ("chunks[embedding]", "images[embedding]", "article_embedding"):
            try:
                metric = (self.client.describe_index(self.name, index_name=field) or {}).get("metric_type", "")
                if metric and metric.upper() != "COSINE":
                    errors.append(f"{field}: metric {metric} != COSINE")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{field}: no index ({e})")
        if errors:
            raise SchemaMismatch(f"collection {self.name!r} schema mismatch: " + "; ".join(errors))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def upsert(self, rows: list[dict], *, sleep=time.sleep) -> None:
        """Whole-entity replace (partial update is not supported with StructArray fields).
        Retries 1, 2, 4 s, then re-raises."""
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                self.client.upsert(collection_name=self.name, data=rows)
                return
            except Exception as e:
                if attempt >= len(RETRY_DELAYS) or "too large" in str(e):
                    raise
                logger.warning("upsert of %d rows failed (%s); retrying in %ds", len(rows), e, RETRY_DELAYS[attempt])
                sleep(RETRY_DELAYS[attempt])

    def iter_page_ids(self, batch_size: int = 5000) -> Iterator[int]:
        it = self.client.query_iterator(collection_name=self.name, batch_size=batch_size, filter="page_id >= 0",
                                        output_fields=["page_id"])
        try:
            while True:
                batch = it.next()
                if not batch:
                    return
                for r in batch:
                    yield r["page_id"]
        finally:
            it.close()

    def delete_ids(self, ids: list[int], chunk: int = 500) -> int:
        n = 0
        for i in range(0, len(ids), chunk):
            part = ids[i:i + chunk]
            self.client.delete(collection_name=self.name, filter=f"page_id in {part}")
            n += len(part)
        return n

    def finish(self) -> None:
        try:
            self.client.flush(collection_name=self.name)
        except Exception as e:  # noqa: BLE001 — flush is advisory on Zilliz Cloud
            logger.warning("flush failed: %s", e)

    def count(self) -> Optional[int]:
        try:
            return int(self.client.get_collection_stats(self.name).get("row_count"))
        except Exception:  # noqa: BLE001
            return None


__all__ = ["ArticleCollection", "SchemaMismatch", "make_client", "CHUNK", "IMAGE"]
