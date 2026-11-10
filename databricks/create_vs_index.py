"""Create (or sync) the Vector Search index over the precomputed Qwen3 embeddings.

Runs locally. The source table already holds chunk_id, chunk_text, ... and a 1024-dim
`embedding` from scripts/embed_chunks.py, so Databricks embeds nothing here.

Needs in .env: DATABRICKS_HOST, DATABRICKS_TOKEN.
Needs once on the table (Databricks SQL editor), or index creation fails:
  ALTER TABLE workspace.default.wikivoyage_pages_index
  SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

  pip install databricks-vectorsearch python-dotenv
  python databricks/create_vs_index.py

Safe to re-run: an existing endpoint is reused and an existing index is synced.
The index builds in the background; re-run to see its progress.
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv

try:  # databricks-vectorsearch was renamed to databricks-ai-search
    from databricks.ai_search.client import VectorSearchClient
except ImportError:
    from databricks.vector_search.client import VectorSearchClient

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ENDPOINT = "wikivoyage-ai-search-endpoint"
INDEX = "workspace.default.wikivoyage_page_index_ai_search"
TABLE = "workspace.default.wikivoyage_pages_index"
PRIMARY_KEY = "chunk_id"
EMBEDDING_COLUMN = "embedding"
EMBEDDING_DIM = 1024  # Qwen3-Embedding-0.6B

host = os.environ["DATABRICKS_HOST"].strip().rstrip("/")
if not host.startswith(("https://", "http://")):
    host = "https://" + host

vsc = VectorSearchClient(
    workspace_url=host,
    personal_access_token=os.environ["DATABRICKS_TOKEN"],
    disable_notice=True,
)

# Endpoint: create only if missing, then wait until it is ONLINE (first time: minutes).
if ENDPOINT not in [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]:
    vsc.create_endpoint(name=ENDPOINT, endpoint_type="STANDARD")
    print(f"creating endpoint {ENDPOINT}")
while (state := vsc.get_endpoint(ENDPOINT).get("endpoint_status", {}).get("state")) != "ONLINE":
    print(f"{time.strftime('%H:%M:%S')} endpoint {ENDPOINT}: {state}")
    time.sleep(30)
print(f"endpoint {ENDPOINT}: ONLINE")

# Index: a new index starts its first sync by itself; an existing one is synced.
try:
    index = vsc.get_index(endpoint_name=ENDPOINT, index_name=INDEX)
except Exception:
    index = None

if index is None:
    try:
        index = vsc.create_delta_sync_index(
            endpoint_name=ENDPOINT,
            index_name=INDEX,
            source_table_name=TABLE,
            pipeline_type="TRIGGERED",
            primary_key=PRIMARY_KEY,
            embedding_vector_column=EMBEDDING_COLUMN,
            embedding_dimension=EMBEDDING_DIM,
        )
    except Exception as e:
        if "change data feed" in str(e).lower() or "changedatafeed" in str(e).lower():
            raise SystemExit(
                f"Change Data Feed is off on {TABLE}. Run this once in the Databricks SQL editor:\n"
                f"  ALTER TABLE {TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
            ) from e
        raise
    print(f"created index {INDEX}")
else:
    status = index.describe().get("status", {})
    if status.get("ready"):
        index.sync()
        print(f"index {INDEX} exists; started a sync")
    else:
        print(f"index {INDEX} exists and is still building")

status = index.describe().get("status", {})
print(f"ready={status.get('ready')} state={status.get('detailed_state')} "
      f"indexed={status.get('indexed_row_count')}")
if status.get("message"):
    print(status["message"])
