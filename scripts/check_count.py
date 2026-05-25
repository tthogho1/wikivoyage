"""Check the number of records in the Zilliz Cloud collection.

Usage
-----
    python scripts/check_count.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*a, **kw): pass  # type: ignore

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

try:
    from pymilvus import MilvusClient
except ImportError:
    sys.exit("[ERROR] pymilvus not installed. Run: pip install 'pymilvus[model]>=2.4'")


def main() -> None:
    uri        = os.environ.get("ZILLIZ_URI", "").strip()
    token      = os.environ.get("ZILLIZ_TOKEN", "").strip()
    collection = os.environ.get("ZILLIZ_COLLECTION", "wikivoyage_pages")

    if not uri or not token:
        sys.exit("[ERROR] ZILLIZ_URI and ZILLIZ_TOKEN must be set in .env")

    client = MilvusClient(uri=uri, token=token)

    if not client.has_collection(collection):
        print(f"[WARN] Collection {collection!r} does not exist.")
        return

    desc = client.describe_collection(collection_name=collection)

    # query count (works without loading the collection into memory)
    result = client.query(
        collection_name=collection,
        filter="page_id > 0",
        output_fields=["count(*)"],
    )
    row_count = result[0].get("count(*)", "unknown") if result else "unknown"

    print(f"Collection : {collection}")
    print(f"Row count  : {row_count:,}" if isinstance(row_count, int) else f"Row count  : {row_count}")
    print(f"Description: {desc.get('description', '')}")
    print(f"Fields     : {[f['name'] for f in desc.get('fields', [])]}")


if __name__ == "__main__":
    main()
