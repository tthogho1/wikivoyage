# Databricks notebook source
# MAGIC %md
# MAGIC # Wikivoyage: embedded chunks (S3) → Delta → Vector Search
# MAGIC
# MAGIC The chunks are built and embedded **locally** by `scripts/embed_chunks.py`
# MAGIC (Qwen3-Embedding-0.6B, 1024 dims, normalized) and uploaded to S3 as `part-*.parquet`.
# MAGIC This notebook does no embedding of documents:
# MAGIC
# MAGIC 1. **Check** that the `databricks-qwen3-embedding-0-6b` endpoint returns the same vectors as
# MAGIC    the local model, using `_check.parquet` from `embed_chunks.py --write-check`. If it does,
# MAGIC    the chatbot can embed questions with the endpoint.
# MAGIC 2. **Load** the parts into the Delta table `chunks_embedded` (Change Data Feed on).
# MAGIC 3. **Index** the precomputed `embedding` column with a Delta Sync index.
# MAGIC 4. **Query** with a question embedded by the endpoint.
# MAGIC
# MAGIC Re-run steps 2–3 whenever more parts have been uploaded; the index is synced, not recreated.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("embedded_path", "s3://wikivoyage-bucket/chunks_embedded/", "Embedded chunks (S3)")
dbutils.widgets.text("catalog", "", "Catalog (blank = current)")
dbutils.widgets.text("schema", "wikivoyage", "Schema")
dbutils.widgets.text("embedding_dim", "1024", "Embedding dimension")
dbutils.widgets.text("query_endpoint", "databricks-qwen3-embedding-0-6b", "Query embedding endpoint")
dbutils.widgets.text("vs_endpoint", "wikivoyage-vs", "Vector Search endpoint")

EMBEDDED_PATH = dbutils.widgets.get("embedded_path").rstrip("/") + "/"
CATALOG = dbutils.widgets.get("catalog").strip() or spark.sql("SELECT current_catalog()").first()[0]
SCHEMA = dbutils.widgets.get("schema")
EMBEDDING_DIM = int(dbutils.widgets.get("embedding_dim"))
QUERY_ENDPOINT = dbutils.widgets.get("query_endpoint")
VS_ENDPOINT = dbutils.widgets.get("vs_endpoint")

EMBEDDED_TABLE = f"{CATALOG}.{SCHEMA}.chunks_embedded"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.chunks_vec_index"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"table: {EMBEDDED_TABLE}\nindex: {INDEX_NAME}")

# COMMAND ----------

import numpy as np
from databricks.sdk import WorkspaceClient

ws = WorkspaceClient()


def endpoint_embed(texts):
    """Embed texts with the serving endpoint; returns L2-normalized float32 vectors."""
    resp = ws.serving_endpoints.query(name=QUERY_ENDPOINT, input=list(texts))
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

# COMMAND ----------

# MAGIC %md ## 1. Check the endpoint against the local model
# MAGIC
# MAGIC Cosine similarity between the local vector and the endpoint's vector for the same input.
# MAGIC About **0.99 or higher** means they match (fp16 vs. the endpoint's precision causes tiny
# MAGIC differences). For `query` rows, `input` already contains Qwen3's query instruction; the
# MAGIC `raw text` column shows whether the endpoint adds the instruction by itself.

# COMMAND ----------

check = spark.read.parquet(EMBEDDED_PATH + "_check.parquet").toPandas()
local = np.stack(check["embedding"].map(lambda v: np.asarray(v, dtype="float32")))
same_input = (local * endpoint_embed(check["input"])).sum(axis=1)
raw_text = (local * endpoint_embed(check["text"])).sum(axis=1)

report = check[["kind", "prompt", "text"]].assign(
    same_input=same_input.round(4), raw_text=raw_text.round(4))
display(report)

QUERY_PREFIX = ""
q = check[(check["kind"] == "query") & (check["prompt"] == "query")].iloc[0]
if q["input"].endswith(q["text"]):
    QUERY_PREFIX = q["input"][: len(q["input"]) - len(q["text"])]

matches = bool((same_input >= 0.99).all())
print("endpoint matches the local model:", matches)
print("query prefix:", repr(QUERY_PREFIX))
if not matches:
    print("Do not use the endpoint for queries; embed questions with the local model instead.")

# COMMAND ----------

# MAGIC %md ## 2. Load the embedded chunks into Delta

# COMMAND ----------

from pyspark.sql import functions as F

parts = spark.read.parquet(EMBEDDED_PATH + "part-*.parquet")
(parts.write.format("delta")
      .mode("overwrite")
      .option("overwriteSchema", "true")
      .saveAsTable(EMBEDDED_TABLE))
spark.sql(f"ALTER TABLE {EMBEDDED_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

display(spark.table(EMBEDDED_TABLE).select(
    F.count("*").alias("chunks"),
    F.countDistinct("page_id").alias("pages"),
    F.min(F.size("embedding")).alias("min_dim"),
    F.max(F.size("embedding")).alias("max_dim"),
))

# COMMAND ----------

# MAGIC %md ## 3. Vector Search endpoint and index (precomputed vectors)

# COMMAND ----------

import time

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)


def wait_for(check_fn, what, timeout_s=3600, interval_s=60):
    """Poll until check_fn() is true; returns False on timeout instead of raising."""
    start = time.time()
    while time.time() - start < timeout_s:
        if check_fn():
            return True
        time.sleep(interval_s)
    print(f"{what} not ready after {timeout_s}s; it keeps building in the background")
    return False


try:
    vsc.get_endpoint(VS_ENDPOINT)
except Exception:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")

wait_for(
    lambda: vsc.get_endpoint(VS_ENDPOINT).get("endpoint_status", {}).get("state") == "ONLINE",
    f"endpoint {VS_ENDPOINT}",
)

try:
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
except Exception:
    index = None

if index is not None:
    index.sync()
    print(f"synced existing index {INDEX_NAME}")
else:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=INDEX_NAME,
        source_table_name=EMBEDDED_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_vector_column="embedding",
        embedding_dimension=EMBEDDING_DIM,
    )
    print(f"created index {INDEX_NAME}")

table_rows = spark.table(EMBEDDED_TABLE).count()


def index_ready():
    st = index.describe().get("status", {})
    print(f"{time.strftime('%H:%M:%S')} {st.get('detailed_state')}: "
          f"{st.get('indexed_row_count')}/{table_rows} rows")
    return st.get("ready", False)


wait_for(index_ready, f"index {INDEX_NAME}")

# COMMAND ----------

# MAGIC %md ## 4. Try a query
# MAGIC
# MAGIC The question gets Qwen3's query instruction (`QUERY_PREFIX`); documents were embedded
# MAGIC without one. The chatbot must do the same.

# COMMAND ----------

question = "cheap hotel near the city centre in Aachen"
query_vec = endpoint_embed([QUERY_PREFIX + question])[0].tolist()

results = index.similarity_search(
    query_vector=query_vec,
    columns=["title", "section", "url", "chunk_text"],
    filters={"page_type": "city"},
    num_results=5,
)
cols = [c["name"] for c in results["manifest"]["columns"]]
display(spark.createDataFrame(results["result"]["data_array"], cols))
