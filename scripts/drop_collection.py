"""Drop and recreate the Zilliz collection (clears all data).

Usage
-----
    python scripts/drop_collection.py
    python scripts/drop_collection.py --yes   # skip confirmation prompt
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Drop Zilliz collection")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    uri        = os.environ.get("ZILLIZ_URI", "").strip()
    token      = os.environ.get("ZILLIZ_TOKEN", "").strip()
    collection = os.environ.get("ZILLIZ_COLLECTION", "wikivoyage_pages")

    if not uri or not token:
        sys.exit("[ERROR] ZILLIZ_URI and ZILLIZ_TOKEN must be set in .env")

    client = MilvusClient(uri=uri, token=token)

    if not client.has_collection(collection):
        print(f"Collection {collection!r} does not exist. Nothing to drop.")
        return

    stats = client.get_collection_stats(collection_name=collection)
    row_count = stats.get("row_count", "?")
    print(f"Collection : {collection}")
    print(f"Row count  : {row_count}")

    if not args.yes:
        ans = input(f"\nDrop collection {collection!r}? This cannot be undone. [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return

    client.drop_collection(collection_name=collection)
    print(f"Dropped collection {collection!r}.")


if __name__ == "__main__":
    main()
