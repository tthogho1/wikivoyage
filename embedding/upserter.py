"""Milvus/Zilliz collection management and upsert helpers.

Provides an `Upserter` class that wraps `MilvusClient` operations:
- ensure_collection(collection_name)
- upsert(collection_name, rows)
- flush(collection_name)
- load_collection(collection_name)
- get_collection_stats(collection_name)
- drop_collection(collection_name)
- query(collection_name, filter, output_fields, limit)

Rows passed to `upsert` should be a list of dicts matching the collection schema.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger("wikivoyage.upserter")


class Upserter:
    def __init__(self, client: Any):
        """Wrap a `pymilvus.MilvusClient` instance."""
        self.client = client

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    def ensure_collection(self, name: str, *, dense_dim: int = 1024, varchar_max: int = 65535, description: str | None = None) -> None:
        """Create collection with dense + sparse fields if it does not exist."""
        from pymilvus import DataType

        if self.client.has_collection(name):
            logger.info("Collection %r already exists.", name)
            return

        logger.info("Creating collection %r …", name)
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("page_id",        DataType.INT64,          is_primary=True)
        schema.add_field("title",          DataType.VARCHAR,        max_length=512)
        schema.add_field("slug",           DataType.VARCHAR,        max_length=512)
        schema.add_field("page_type",      DataType.VARCHAR,        max_length=64)
        schema.add_field("status",         DataType.VARCHAR,        max_length=64,  nullable=True)
        schema.add_field("url",            DataType.VARCHAR,        max_length=1024, nullable=True)
        schema.add_field("source",         DataType.VARCHAR,        max_length=64)
        schema.add_field("retrieval_text", DataType.VARCHAR,        max_length=varchar_max)
        schema.add_field("dense_vector",   DataType.FLOAT_VECTOR,   dim=dense_dim)
        schema.add_field("sparse_vector",  DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )

        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            description=description or "Wikivoyage pages – BGE-M3 dense + sparse",
        )
        logger.info("Created collection %r.", name)

    # ------------------------------------------------------------------
    # Upsert / data operations
    # ------------------------------------------------------------------
    def upsert(self, collection_name: str, rows: Iterable[dict]) -> dict:
        """Upsert rows into the collection. Returns the raw result from the client."""
        rows_list = list(rows)
        if not rows_list:
            logger.info("No rows to upsert for collection %r", collection_name)
            return {}
        logger.info("Upserting %d rows into %r …", len(rows_list), collection_name)
        result = self.client.upsert(collection_name=collection_name, data=rows_list)
        logger.debug("Upsert result: %s", result)
        return result

    def flush(self, collection_name: str) -> None:
        logger.info("Flushing collection %r …", collection_name)
        self.client.flush(collection_name=collection_name)

    def load_collection(self, collection_name: str) -> None:
        logger.info("Loading collection %r into memory …", collection_name)
        self.client.load_collection(collection_name=collection_name)

    def get_collection_stats(self, collection_name: str) -> dict:
        return self.client.get_collection_stats(collection_name=collection_name)

    def drop_collection(self, collection_name: str) -> None:
        logger.info("Dropping collection %r …", collection_name)
        self.client.drop_collection(collection_name=collection_name)

    def query(self, collection_name: str, *, filter: str | None = None, output_fields: list[str] | None = None, limit: int = 20) -> list[dict]:
        qf = filter or "page_id > 0"
        ofs = output_fields or ["page_id", "title", "slug", "page_type", "status"]
        return self.client.query(collection_name=collection_name, filter=qf, output_fields=ofs, limit=limit)
