# Databricks notebook source
# MAGIC %md
# MAGIC # Wikivoyage: Parquet → chunks → Vector Search
# MAGIC
# MAGIC Builds the RAG chunk table from the Parquet written by `get_wikivoyage.py --format parquet`
# MAGIC and indexes it with Databricks Vector Search (managed embeddings).
# MAGIC
# MAGIC 1. Read `part-*.parquet` from S3 (one row per page).
# MAGIC 2. Explode `sections` and split each section's text into chunks of at most `max_chars`,
# MAGIC    breaking only between lines so a listing (name, address, price, ...) stays whole.
# MAGIC 3. Build `chunk_text` = header line + body, e.g.
# MAGIC    ```
# MAGIC    Aachen (city, Eifel (North Rhine-Westphalia)) — Sleep > Budget
# MAGIC    Hostel Aachen, Mauerstraße 116. Phone: +49 241 4305 288. Close to the city centre.
# MAGIC    ```
# MAGIC 4. Save as a Delta table with Change Data Feed enabled.
# MAGIC 5. Create (or sync) a Delta Sync index that embeds `chunk_text` only; the other columns are
# MAGIC    kept for `filters` and citations.
# MAGIC
# MAGIC Re-run the whole notebook after a new dump run; an existing index is synced, not recreated.
# MAGIC `bge-large-en` accepts 512 tokens, so lower `max_chars` to ~1500 if you switch to it
# MAGIC (`gte-large-en` accepts 8192).

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("source_path", "s3://wikivoyage-bucket/pages/", "Parquet input (S3)")
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "wikivoyage", "Schema")
dbutils.widgets.text("max_chars", "2000", "Max chars per chunk")
dbutils.widgets.text("min_chars", "50", "Skip sections shorter than")
dbutils.widgets.text("vs_endpoint", "wikivoyage-vs", "Vector Search endpoint")
dbutils.widgets.text("embedding_model", "databricks-gte-large-en", "Embedding model endpoint")

SOURCE_PATH = dbutils.widgets.get("source_path")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MAX_CHARS = int(dbutils.widgets.get("max_chars"))
MIN_CHARS = int(dbutils.widgets.get("min_chars"))
VS_ENDPOINT = dbutils.widgets.get("vs_endpoint")
EMBEDDING_MODEL = dbutils.widgets.get("embedding_model")

CHUNK_TABLE = f"{CATALOG}.{SCHEMA}.chunks"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.chunks_index"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

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

print(f"pages: {pages.count()}")
display(pages.groupBy("page_type").count().orderBy(F.desc("count")))

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

# COMMAND ----------

(chunks.write.format("delta")
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(CHUNK_TABLE))
spark.sql(f"ALTER TABLE {CHUNK_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

stats = spark.table(CHUNK_TABLE).select(
    F.count("*").alias("chunks"),
    F.countDistinct("page_id").alias("pages"),
    F.round(F.avg(F.length("chunk_text"))).alias("avg_chars"),
    F.max(F.length("chunk_text")).alias("max_chars"),
)
display(stats)
display(spark.table(CHUNK_TABLE).where("title = 'Aachen'").limit(10))

# COMMAND ----------

# MAGIC %md ## 5. Vector Search endpoint and index

# COMMAND ----------

import time

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)


def wait_for(check, what, timeout_s=3600, interval_s=30):
    start = time.time()
    while not check():
        if time.time() - start > timeout_s:
            raise TimeoutError(f"{what} not ready after {timeout_s}s")
        print(f"waiting for {what} ...")
        time.sleep(interval_s)


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
        source_table_name=CHUNK_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=EMBEDDING_MODEL,
    )
    print(f"created index {INDEX_NAME}")

wait_for(lambda: index.describe().get("status", {}).get("ready", False), f"index {INDEX_NAME}")

# COMMAND ----------

# MAGIC %md ## 6. Try a query

# COMMAND ----------

results = index.similarity_search(
    query_text="cheap hotel near the city centre in Aachen",
    columns=["title", "section", "url", "chunk_text"],
    filters={"page_type": "city"},
    num_results=5,
    query_type="HYBRID",
)
cols = [c["name"] for c in results["manifest"]["columns"]]
display(spark.createDataFrame(results["result"]["data_array"], cols))
