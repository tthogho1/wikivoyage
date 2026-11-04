# Databricks notebook source
# MAGIC %md
# MAGIC # Wikivoyage: Parquet → chunks → embeddings → Vector Search
# MAGIC
# MAGIC Builds the RAG chunk table from the Parquet written by `get_wikivoyage.py --format parquet`,
# MAGIC embeds it with `BAAI/bge-base-en-v1.5` (768 dims) on the cluster, and indexes the
# MAGIC precomputed vectors with Databricks Vector Search.
# MAGIC
# MAGIC 1. Read `part-*.parquet` from S3 (one row per page).
# MAGIC 2. Explode `sections` and split each section's text into chunks of at most `max_chars`,
# MAGIC    breaking only between lines so a listing (name, address, price, ...) stays whole.
# MAGIC 3. Build `chunk_text` = header line + body, e.g.
# MAGIC    ```
# MAGIC    Aachen (city, Eifel (North Rhine-Westphalia)) — Sleep > Budget
# MAGIC    Hostel Aachen, Mauerstraße 116. Phone: +49 241 4305 288. Close to the city centre.
# MAGIC    ```
# MAGIC 4. Save `chunks` (skipped when it already exists, unless `rebuild_chunks` = `true`).
# MAGIC 5. Embed `chunk_text` into `chunks_embedded.embedding`. **Resumable:** it works in passes of
# MAGIC    `rows_per_pass` rows and appends each pass, so re-running continues where it stopped.
# MAGIC 6. Create (or sync) a Delta Sync index over the precomputed vectors.
# MAGIC 7. Try a query. The chatbot must embed questions with the **same model**, normalized.
# MAGIC
# MAGIC Embedding on CPU is slow: measured on an M4 Mac CPU, ~320k chunks take ~14 h with
# MAGIC bge-base, ~4 h with `BAAI/bge-small-en-v1.5` (384 dims) and 60+ h with
# MAGIC `Alibaba-NLP/gte-large-en-v1.5` (1024 dims). Set `embedding_model` and `embedding_dim`
# MAGIC together, and use the same model in the chatbot. Run this notebook as a **Job**
# MAGIC (Workflows → Create job) so it does not depend on an open browser; if the run stops,
# MAGIC run it again.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch "sentence-transformers>=4,<5" "transformers>=4.44,<5"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("source_path", "s3://wikivoyage-bucket/pages/", "Parquet input (S3)")
dbutils.widgets.text("catalog", "", "Catalog (blank = current)")
dbutils.widgets.text("schema", "wikivoyage", "Schema")
dbutils.widgets.text("max_chars", "2000", "Max chars per chunk")
dbutils.widgets.text("min_chars", "50", "Skip sections shorter than")
dbutils.widgets.dropdown("rebuild_chunks", "false", ["false", "true"], "Rebuild chunks table")
dbutils.widgets.text("embedding_model", "BAAI/bge-base-en-v1.5", "Embedding model (Hugging Face)")
dbutils.widgets.text("embedding_dim", "768", "Embedding dimension")
dbutils.widgets.text("rows_per_pass", "20000", "Rows embedded per pass")
dbutils.widgets.text("partitions", "16", "Parallel tasks per pass")
dbutils.widgets.text("vs_endpoint", "wikivoyage-vs", "Vector Search endpoint")

SOURCE_PATH = dbutils.widgets.get("source_path")
CATALOG = dbutils.widgets.get("catalog").strip() or spark.sql("SELECT current_catalog()").first()[0]
SCHEMA = dbutils.widgets.get("schema")
MAX_CHARS = int(dbutils.widgets.get("max_chars"))
MIN_CHARS = int(dbutils.widgets.get("min_chars"))
REBUILD_CHUNKS = dbutils.widgets.get("rebuild_chunks") == "true"
EMBEDDING_MODEL = dbutils.widgets.get("embedding_model")
EMBEDDING_DIM = int(dbutils.widgets.get("embedding_dim"))
ROWS_PER_PASS = int(dbutils.widgets.get("rows_per_pass"))
PARTITIONS = int(dbutils.widgets.get("partitions"))
VS_ENDPOINT = dbutils.widgets.get("vs_endpoint")

# transformers is pinned below 5 in the %pip cell so gte-large-en-v1.5 also works: its remote
# code fails on 5.x (IndexError on rope position_ids).
# Chunks are at most ~2000 chars (~500 tokens). bge models accept 512 tokens, gte-large 8192;
# capping the sequence length also bounds CPU cost.
MAX_SEQ_LENGTH = 512 if "bge-" in EMBEDDING_MODEL else 1024

CHUNK_TABLE = f"{CATALOG}.{SCHEMA}.chunks"
EMBEDDED_TABLE = f"{CATALOG}.{SCHEMA}.chunks_embedded"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.chunks_vec_index"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"chunks: {CHUNK_TABLE}\nembedded: {EMBEDDED_TABLE}\nindex: {INDEX_NAME}")

# COMMAND ----------

import time

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)


def get_index_or_none(name):
    try:
        return vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=name)
    except Exception:
        return None

# COMMAND ----------

# MAGIC %md ## 1. Read pages

# COMMAND ----------

from pyspark.sql import functions as F, types as T

pages = spark.read.parquet(SOURCE_PATH)
# Parquet from before is_part_of was added has no such column.
if "is_part_of" not in pages.columns:
    pages = pages.withColumn("is_part_of", F.lit(None).cast("string"))
# Overlapping runs written to the same prefix can repeat a page.
pages = pages.dropDuplicates(["page_id"])

# COMMAND ----------

# MAGIC %md ## 2–3. Split sections into chunks and build `chunk_text`

# COMMAND ----------


def split_lines(text, max_chars):
    """Pack lines into chunks of <= max_chars; only a single over-long line is cut mid-line."""
    chunks, cur = [], ""
    for line in (text or "").split("\n"):
        while len(line) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if not line:
            continue
        if cur and len(cur) + 1 + len(line) > max_chars:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


split_lines_udf = F.udf(lambda t: split_lines(t, MAX_CHARS), T.ArrayType(T.StringType()))

sections = (
    pages
    .select("page_id", "title", "page_type", "status", "is_part_of", "url",
            F.posexplode("sections").alias("sec_idx", "s"))
    .where(F.length("s.text") >= MIN_CHARS)
)

chunks = (
    sections
    .select("*", F.posexplode(split_lines_udf("s.text")).alias("part", "body"))
    .withColumn("section", F.when(F.col("s.path") == "", F.lit("Overview")).otherwise(F.col("s.path")))
    .withColumn(
        "place",
        F.when(F.col("is_part_of").isNotNull(), F.concat_ws(", ", "page_type", "is_part_of"))
         .otherwise(F.col("page_type")),
    )
    .withColumn(
        "chunk_text",
        F.concat(F.col("title"), F.lit(" ("), F.col("place"), F.lit(") — "),
                 F.col("section"), F.lit("\n"), F.col("body")),
    )
    # Link to the section itself when it has an anchor.
    .withColumn(
        "url",
        F.when(F.col("s.anchor") != "", F.concat(F.col("url"), F.lit("#"), F.col("s.anchor")))
         .otherwise(F.col("url")),
    )
    .withColumn("chunk_id", F.concat_ws("-", "page_id", "sec_idx", "part"))
    .select("chunk_id", "page_id", "title", "page_type", "status", "is_part_of",
            "section", "url", "chunk_text")
)

# COMMAND ----------

# MAGIC %md ## 4. Save the chunk table
# MAGIC
# MAGIC Kept as-is when it already exists. With `rebuild_chunks` = `true` it is rebuilt, and the
# MAGIC embeddings and index built from the old chunks are dropped so they are rebuilt too.

# COMMAND ----------

if REBUILD_CHUNKS or not spark.catalog.tableExists(CHUNK_TABLE):
    (chunks.write.format("delta")
           .mode("overwrite")
           .option("overwriteSchema", "true")
           .saveAsTable(CHUNK_TABLE))
    spark.sql(f"DROP TABLE IF EXISTS {EMBEDDED_TABLE}")
    if get_index_or_none(INDEX_NAME) is not None:
        vsc.delete_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
        print(f"deleted index {INDEX_NAME} (built from the old chunks)")
    print(f"built {CHUNK_TABLE}")
else:
    print(f"keeping existing {CHUNK_TABLE} (set rebuild_chunks = true to rebuild)")

display(spark.table(CHUNK_TABLE).select(
    F.count("*").alias("chunks"),
    F.countDistinct("page_id").alias("pages"),
    F.round(F.avg(F.length("chunk_text"))).alias("avg_chars"),
    F.max(F.length("chunk_text")).alias("max_chars"),
))

# COMMAND ----------

# MAGIC %md ## 5. Embed `chunk_text` (resumable)

# COMMAND ----------

from typing import Iterator

import pandas as pd

model_name, max_seq = EMBEDDING_MODEL, MAX_SEQ_LENGTH


@F.pandas_udf(T.ArrayType(T.FloatType()))
def embed_udf(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
    from sentence_transformers import SentenceTransformer

    # Loaded once per task; the Hugging Face download is cached on each machine.
    model = SentenceTransformer(model_name, trust_remote_code=True, device="cpu")
    model.max_seq_length = max_seq
    for texts in batches:
        vecs = model.encode(texts.tolist(), batch_size=16,
                            normalize_embeddings=True, show_progress_bar=False)
        yield pd.Series([v.astype("float32").tolist() for v in vecs])


if not spark.catalog.tableExists(EMBEDDED_TABLE):
    (spark.table(CHUNK_TABLE).limit(0)
          .withColumn("embedding", F.lit(None).cast(T.ArrayType(T.FloatType())))
          .write.format("delta").saveAsTable(EMBEDDED_TABLE))
    spark.sql(f"ALTER TABLE {EMBEDDED_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

total = spark.table(CHUNK_TABLE).count()
while True:
    done = spark.table(EMBEDDED_TABLE).select("chunk_id")
    todo = spark.table(CHUNK_TABLE).join(done, "chunk_id", "left_anti")
    left = todo.count()
    print(f"{time.strftime('%H:%M:%S')} embedded {total - left}/{total}, {left} left")
    if left == 0:
        break
    (todo.limit(ROWS_PER_PASS)
         .repartition(PARTITIONS)
         .withColumn("embedding", embed_udf("chunk_text"))
         .write.format("delta").mode("append").saveAsTable(EMBEDDED_TABLE))

# COMMAND ----------

# MAGIC %md ## 6. Vector Search endpoint and index (precomputed vectors)

# COMMAND ----------


def wait_for(check, what, timeout_s=3600, interval_s=60):
    """Poll until check() is true; returns False on timeout instead of raising."""
    start = time.time()
    while time.time() - start < timeout_s:
        if check():
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

index = get_index_or_none(INDEX_NAME)
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

embedded_rows = spark.table(EMBEDDED_TABLE).count()


def index_ready():
    st = index.describe().get("status", {})
    print(f"{time.strftime('%H:%M:%S')} {st.get('detailed_state')}: "
          f"{st.get('indexed_row_count')}/{embedded_rows} rows")
    return st.get("ready", False)


wait_for(index_ready, f"index {INDEX_NAME}")

# COMMAND ----------

# MAGIC %md ## 7. Try a query
# MAGIC
# MAGIC Questions are embedded with the same model and normalized, exactly like the chunks.

# COMMAND ----------

from sentence_transformers import SentenceTransformer

query_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device="cpu")
query_vec = query_model.encode("cheap hotel near the city centre in Aachen",
                               normalize_embeddings=True).astype("float32").tolist()

results = index.similarity_search(
    query_vector=query_vec,
    columns=["title", "section", "url", "chunk_text"],
    filters={"page_type": "city"},
    num_results=5,
)
cols = [c["name"] for c in results["manifest"]["columns"]]
display(spark.createDataFrame(results["result"]["data_array"], cols))
