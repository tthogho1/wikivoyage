"""Search the Databricks Vector Search index of Wikivoyage chunks from the command line.

The question is embedded with the Qwen3 serving endpoint (with Qwen3's query instruction,
since the chunks were embedded without one by scripts/embed_chunks.py), then searched
against the precomputed-vector index.

Needs in .env: DATABRICKS_HOST, DATABRICKS_TOKEN.

  python scripts/query_index.py "cheap hostel near the city centre in Aachen"
  python scripts/query_index.py "best beaches" --page-type city -n 10 --full
  python scripts/query_index.py --status
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

try:  # databricks-vectorsearch was renamed to databricks-ai-search
    from databricks.ai_search.client import VectorSearchClient
except ImportError:
    from databricks.vector_search.client import VectorSearchClient

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

VS_ENDPOINT = os.environ.get("DATABRICKS_VS_ENDPOINT", "wikivoyage-ai-search-endpoint")
VS_INDEX = os.environ.get("DATABRICKS_VS_INDEX", "workspace.default.wikivoyage_page_index_ai_search")
EMBEDDING_ENDPOINT = os.environ.get("DATABRICKS_EMBEDDING_ENDPOINT", "databricks-qwen3-embedding-0-6b")

# Qwen3-Embedding: queries get this instruction; documents were embedded without one.
QUERY_PREFIX = ("Instruct: Given a web search query, retrieve relevant passages "
                "that answer the query\nQuery:")


def _host() -> str:
    host = os.environ["DATABRICKS_HOST"].strip().rstrip("/")
    return host if host.startswith(("https://", "http://")) else "https://" + host


def embed_query(question: str) -> list[float]:
    from databricks.sdk import WorkspaceClient

    ws = WorkspaceClient(host=_host(), token=os.environ["DATABRICKS_TOKEN"])
    resp = ws.serving_endpoints.query(name=EMBEDDING_ENDPOINT, input=[QUERY_PREFIX + question])
    vec = np.asarray(resp.data[0].embedding, dtype="float32")
    return (vec / np.linalg.norm(vec)).tolist()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Search the Wikivoyage Vector Search index.")
    p.add_argument("question", nargs="?", help="what to search for")
    p.add_argument("-n", "--num-results", type=int, default=5)
    p.add_argument("--page-type", help="filter: city / region / country")
    p.add_argument("--full", action="store_true", help="print the whole chunk text")
    p.add_argument("--status", action="store_true", help="only print the index status")
    args = p.parse_args(argv)

    vsc = VectorSearchClient(workspace_url=_host(),
                             personal_access_token=os.environ["DATABRICKS_TOKEN"],
                             disable_notice=True)
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
    status = index.describe().get("status", {})
    print(f"{VS_INDEX}: ready={status.get('ready')} state={status.get('detailed_state')} "
          f"indexed={status.get('indexed_row_count')}")
    if args.status:
        return 0
    if not args.question:
        p.error("give a question, or --status")
    if not status.get("ready"):
        print("The index is not ready yet; try again later (or --status to watch it).")
        return 1

    results = index.similarity_search(
        query_vector=embed_query(args.question),
        columns=["title", "section", "url", "chunk_text"],
        filters={"page_type": args.page_type} if args.page_type else None,
        num_results=args.num_results,
    )
    rows = results.get("result", {}).get("data_array", [])
    if not rows:
        print("no results")
    for i, (title, section, url, text, score) in enumerate(rows, 1):
        body = text.split("\n", 1)[-1]  # drop the "Title (type, region) — Section" header line
        print(f"\n{i}. [{score:.3f}] {title} — {section}\n   {url}")
        print("   " + (body if args.full else body[:300].replace("\n", " ") + ("…" if len(body) > 300 else "")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
