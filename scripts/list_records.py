"""List records in the Zilliz Cloud collection.

Usage
-----
    # Show first 20 records (default)
    python scripts/list_records.py

    # Show first 50 records
    python scripts/list_records.py --limit 50

    # Filter by page_type
    python scripts/list_records.py --page-type city

    # Filter by status
    python scripts/list_records.py --status guide

    # Output as JSON
    python scripts/list_records.py --json

    # Filter by keyword in title
    python scripts/list_records.py --title Tokyo
"""
from __future__ import annotations

import argparse
import json
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

_OUTPUT_FIELDS = ["page_id", "title", "slug", "page_type", "status", "url", "source"]


def _build_filter(page_type: str | None, status: str | None, title: str | None) -> str:
    parts = []
    if page_type:
        parts.append(f'page_type == "{page_type}"')
    if status:
        parts.append(f'status == "{status}"')
    if title:
        parts.append(f'title like "%{title}%"')
    return " and ".join(parts) if parts else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="List Zilliz collection records")
    parser.add_argument("--limit",     type=int, default=20,  help="Max records to show (default: 20)")
    parser.add_argument("--page-type", type=str, default=None, help="Filter by page_type (city/region/country/other)")
    parser.add_argument("--status",    type=str, default=None, help="Filter by status (outline/usable/guide/star)")
    parser.add_argument("--title",     type=str, default=None, help="Filter by title keyword")
    parser.add_argument("--json",      action="store_true",    help="Output as JSON")
    args = parser.parse_args()

    uri        = os.environ.get("ZILLIZ_URI", "").strip()
    token      = os.environ.get("ZILLIZ_TOKEN", "").strip()
    collection = os.environ.get("ZILLIZ_COLLECTION", "wikivoyage_pages")

    if not uri or not token:
        sys.exit("[ERROR] ZILLIZ_URI and ZILLIZ_TOKEN must be set in .env")

    client = MilvusClient(uri=uri, token=token)

    if not client.has_collection(collection):
        sys.exit(f"[ERROR] Collection {collection!r} does not exist.")

    filter_expr = _build_filter(args.page_type, args.status, args.title)

    results = client.query(
        collection_name=collection,
        filter=filter_expr if filter_expr else "page_id > 0",
        output_fields=_OUTPUT_FIELDS,
        limit=args.limit,
    )

    if not results:
        print("No records found.")
        return

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # --- table output ---
    header = f"{'page_id':>10}  {'page_type':<10}  {'status':<10}  {'title'}"
    print(header)
    print("-" * 80)
    for r in results:
        print(
            f"{r.get('page_id', ''):>10}  "
            f"{r.get('page_type', ''):.<10}  "
            f"{r.get('status', '') or '':.<10}  "
            f"{r.get('title', '')}"
        )

    print()
    print(f"Showing {len(results)} record(s)"
          + (f"  [filter: {filter_expr}]" if filter_expr else ""))


if __name__ == "__main__":
    main()
