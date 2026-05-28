"""Embed Wikivoyage JSONL pages with BGE-M3 (dense + sparse) and upsert to Zilliz Cloud.

Usage
-----
Set environment variables, then run:

    python embedding/upsert_zilliz.py

Environment variables (required):
    ZILLIZ_URI        Zilliz Cloud endpoint, e.g. "https://xxxx.zillizcloud.com"
    ZILLIZ_TOKEN      API token  (user:password  or  API-key string)

Environment variables (optional):
    WIKIVOYAGE_PAGES_DIR   Directory of per-page JSONL files  (default: pages)
    ZILLIZ_COLLECTION      Collection name                     (default: wikivoyage_pages)
    ZILLIZ_BATCH_SIZE      Upsert batch size                   (default: 32)
    BGE_M3_DEVICE          "cpu" | "cuda" | "mps"             (default: cpu)
    BGE_M3_USE_FP16        "1" to enable fp16 on GPU          (default: 0)

Collection schema (auto-created if not exists)
-----------------------------------------------
    page_id       INT64   primary key
    title         VARCHAR
    slug          VARCHAR
    page_type     VARCHAR
    status        VARCHAR
    url           VARCHAR
    source        VARCHAR
    dense_vector  FLOAT_VECTOR  (1024 dims, BGE-M3 dense output)
    sparse_vector SPARSE_FLOAT_VECTOR (BGE-M3 lexical weights)
    retrieval_text VARCHAR (stored for inspection / hybrid re-rank)
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Third-party imports (checked at runtime with friendly messages)
# ---------------------------------------------------------------------------
try:
    from pymilvus import (
        MilvusClient,
        DataType,
        Function,
        FunctionType,
    )
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction
except ImportError as e:
    sys.exit(
        f"[ERROR] Missing dependency: {e}\n"
        "Install with:  pip install 'pymilvus[model]>=2.4'"
    )

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs): pass  # type: ignore

logger = logging.getLogger("wikivoyage.embed")

# Ensure project root is on sys.path so `import embedding.*` works when running
# this script directly (not as a package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DENSE_DIM = 1024                   # BGE-M3 dense output dimension
MAX_VARCHAR = 65_535               # Milvus VARCHAR limit
COLLECTION_DESC = "Wikivoyage city pages – BGE-M3 dense + sparse vectors"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
def _create_collection(client: MilvusClient, name: str) -> None:
    """Create collection with dense + sparse index if it does not exist."""
    from pymilvus import CollectionSchema, FieldSchema

    if client.has_collection(name):
        logger.info("Collection %r already exists – skipping creation.", name)
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("page_id",        DataType.INT64,          is_primary=True)
    schema.add_field("title",          DataType.VARCHAR,        max_length=512)
    schema.add_field("slug",           DataType.VARCHAR,        max_length=512)
    schema.add_field("page_type",      DataType.VARCHAR,        max_length=64)
    schema.add_field("status",         DataType.VARCHAR,        max_length=64,  nullable=True)
    schema.add_field("url",            DataType.VARCHAR,        max_length=1024, nullable=True)
    schema.add_field("source",         DataType.VARCHAR,        max_length=64)
    schema.add_field("retrieval_text", DataType.VARCHAR,        max_length=MAX_VARCHAR)
    schema.add_field("dense_vector",   DataType.FLOAT_VECTOR,   dim=DENSE_DIM)
    schema.add_field("sparse_vector",  DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
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

    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
        description=COLLECTION_DESC,
    )
    logger.info("Created collection %r.", name)


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------
def _load_records(pages_dir: str) -> list[dict]:
    pattern = os.path.join(pages_dir, "*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning("No .jsonl files found in %r", pages_dir)
        return []
    records = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping bad JSON in %s: %s", path, e)
    logger.info("Loaded %d records from %s", len(records), pages_dir)
    return records


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def _embed_batch(
    ef: BGEM3EmbeddingFunction,
    texts: list[str],
) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Return (dense_list, sparse_list) for a batch of texts.

    BGEM3EmbeddingFunction may return sparse values as scipy csr_matrix objects.
    We normalise them to plain {int: float} dicts that Milvus accepts.
    """
    output = ef(texts)
    dense: list[list[float]] = [list(map(float, v)) for v in output["dense"]]

    sparse: list[dict[int, float]] = []
    for s in output["sparse"]:
        # scipy sparse matrix (1 x vocab)
        try:
            cx = s.tocoo()
            sparse.append({int(j): float(v) for j, v in zip(cx.col, cx.data)})
        except AttributeError:
            # already a dict-like object
            sparse.append({int(k): float(v) for k, v in s.items()})

    return dense, sparse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load .env file (project root or current directory; does not override existing env vars)
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
    logger.info(".env loaded from %s", _env_path)

    # --- config -----------------------------------------------------------
    zilliz_uri   = os.environ.get("ZILLIZ_URI", "").strip()
    zilliz_token = os.environ.get("ZILLIZ_TOKEN", "").strip()
    if not zilliz_uri or not zilliz_token:
        sys.exit(
            "[ERROR] ZILLIZ_URI and ZILLIZ_TOKEN environment variables must be set.\n"
            "  export ZILLIZ_URI='https://xxxx.zillizcloud.com'\n"
            "  export ZILLIZ_TOKEN='<api-key>'"
        )

    pages_dir   = os.environ.get("WIKIVOYAGE_PAGES_DIR", "pages")
    collection  = os.environ.get("ZILLIZ_COLLECTION", "wikivoyage_pages")
    batch_size  = int(os.environ.get("ZILLIZ_BATCH_SIZE", "32"))
    device      = os.environ.get("BGE_M3_DEVICE", "cpu")
    use_fp16    = os.environ.get("BGE_M3_USE_FP16", "0") == "1"

    # --- load JSONL records -----------------------------------------------
    records = _load_records(pages_dir)
    if not records:
        sys.exit("[ERROR] No records found. Check WIKIVOYAGE_PAGES_DIR.")

    # --- initialise components: Chunker, Embedder, Upserter ----------------
    chunk_method = os.environ.get("CHUNK_METHOD", "recursive")
    chunk_size = int(os.environ.get("CHUNK_SIZE", str(CHUNK_SIZE)))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", str(CHUNK_OVERLAP)))
    embedding_batch = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))

    logger.info("Initializing components: chunker=%s size=%d overlap=%d embed_batch=%d", chunk_method, chunk_size, chunk_overlap, embedding_batch)
    try:
        from embedding.chunker import make_chunker
        from embedding.embedder import make_embedder
        from embedding.upserter import Upserter
    except Exception as e:
        logger.error("Failed to import embedding helpers: %s", e)
        sys.exit(1)

    chunker = make_chunker(kind=chunk_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedder = make_embedder(model_name="BAAI/bge-m3", device=device, use_fp16=use_fp16)

    # connect to Zilliz
    logger.info("Connecting to Zilliz Cloud: %s", zilliz_uri)
    client = MilvusClient(uri=zilliz_uri, token=zilliz_token)
    upserter = Upserter(client)
    upserter.ensure_collection(collection, dense_dim=DENSE_DIM, varchar_max=MAX_VARCHAR, description=COLLECTION_DESC)

    total_upserted = 0
    total_skipped = 0

    # process in batches of pages; chunk each page and embed chunks in batch
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]

        valid = [r for r in batch if r.get("retrieval_text") and r.get("page_id")]
        if not valid:
            continue

        chunked_records = []
        for r in valid:
            text = (r.get("retrieval_text") or "").strip()
            if not text:
                total_skipped += 1
                continue
            chunks = chunker.chunk(text)
            for i, c in enumerate(chunks, start=1):
                chunked_records.append({
                    "parent_page_id": int(r["page_id"]),
                    "chunk_index": i,
                    "title": r.get("title"),
                    "slug": r.get("slug"),
                    "page_type": r.get("page_type"),
                    "status": r.get("status"),
                    "url": r.get("url"),
                    "source": r.get("source", "enwikivoyage"),
                    "retrieval_text": c,
                })

        if not chunked_records:
            continue

        texts = [c["retrieval_text"] for c in chunked_records]
        logger.info("Embedding batch %d–%d (pages=%d chunks=%d) …", batch_start + 1, batch_start + len(valid), len(valid), len(texts))

        try:
            dense_vecs, sparse_vecs = embedder.embed_texts(texts, batch_size=embedding_batch)
        except Exception as e:
            logger.error("Embedding failed for batch %d: %s", batch_start, e)
            total_skipped += len(chunked_records)
            continue

        rows = []
        for c, dv, sv in zip(chunked_records, dense_vecs, sparse_vecs):
            rt = c["retrieval_text"]
            if len(rt) > MAX_VARCHAR:
                rt = rt[:MAX_VARCHAR]
            parent = int(c["parent_page_id"])
            idx = int(c["chunk_index"])
            rows.append({
                "page_id": parent * 1000 + idx,
                "title": (c.get("title") or "")[:512],
                "slug": (c.get("slug") or "")[:512],
                "page_type": (c.get("page_type") or "other")[:64],
                "status": (c.get("status") or "")[:64],
                "url": (c.get("url") or "")[:1024],
                "source": (c.get("source") or "enwikivoyage")[:64],
                "retrieval_text": rt,
                "dense_vector": dv,
                "sparse_vector": sv,
            })

        try:
            result = upserter.upsert(collection, rows)
            upserted = result.get("upsert_count", len(rows)) if isinstance(result, dict) else len(rows)
            total_upserted += upserted
            logger.info("Upserted %d records (total so far: %d).", upserted, total_upserted)
        except Exception as e:
            logger.error("Upsert failed for batch %d: %s", batch_start, e)
            total_skipped += len(rows)

    # flush and load collection for visibility
    try:
        upserter.flush(collection)
        upserter.load_collection(collection)
    except Exception as e:
        logger.warning("Flush/load failed (non-fatal): %s", e)

    try:
        stats = upserter.get_collection_stats(collection)
    except Exception:
        stats = {}

    logger.info("Done. upserted=%d, skipped=%d, collection=%r, stats=%s", total_upserted, total_skipped, collection, stats)


if __name__ == "__main__":
    main()
