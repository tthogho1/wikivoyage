# Wikivoyage RAG Pipeline

A pipeline that extracts city information from Wikivoyage XML dumps, generates dense + sparse vectors using BGE-M3, and upserts them into Zilliz Cloud for hybrid vector search.

---

## Architecture

```
Wikivoyage XML dump (.bz2)
        │
        ▼
[1] WikivoyageDumpStreamReader     (wikivoyage_stream/)
        │  yields Page objects
        ▼
[2] get_wikivoyage.py              city page filtering + pages/*.jsonl output
        │  retrieval_text / sections / clean_text
        ▼
[3] embedding/upsert_zilliz.py     BGE-M3 embed → Zilliz Cloud upsert
```

---

## Directory Structure

```
wikivoyage/
├── get_wikivoyage.py           Main script (dump → per-page JSONL)
├── dump_to_jsonl.py            Bulk raw JSONL writer (single file)
├── pipeline.py                 Compatibility wrapper
│
├── wikivoyage_stream/          Dump reader package
│   ├── __init__.py
│   ├── models.py               Page dataclass
│   ├── parser.py               NamespaceAwareXmlParser
│   └── reader.py               WikivoyageDumpStreamReader
│
├── wikivoyage_pipeline/        RAG preprocessing package
│   ├── __init__.py
│   ├── writer.py               WikivoyageJsonlWriter
│   └── preprocessor.py         RagPreprocessor
│
├── embedding/
│   └── upsert_zilliz.py        BGE-M3 embed + Zilliz Cloud upsert
│
├── pages/                      Output: per-page JSONL files (auto-generated)
│   └── <PageTitle>.jsonl
│
├── .env                        Environment variables (not committed)
├── .env.example                Environment variable template
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```ini
# Zilliz Cloud (required)
ZILLIZ_URI=https://<your-cluster-endpoint>.zillizcloud.com
ZILLIZ_TOKEN=<your-api-key>

# Pipeline settings (optional)
WIKIVOYAGE_DUMP=             # omit to download automatically
WIKIVOYAGE_LIMIT=500         # 0 = no limit
WIKIVOYAGE_OUTDIR=pages
WIKIVOYAGE_PAGE_TYPES=city   # city, region, country, or * for all

# Zilliz settings (optional)
ZILLIZ_COLLECTION=wikivoyage_pages
ZILLIZ_BATCH_SIZE=32

# BGE-M3 settings (optional)
BGE_M3_DEVICE=cpu            # cpu / cuda / mps
BGE_M3_USE_FP16=0            # set to 1 when using GPU
```

> **How to find your Zilliz Cloud credentials**
> 1. Log in to [https://cloud.zilliz.com](https://cloud.zilliz.com)
> 2. Select your cluster → click **Connect**
> 3. Copy **Public Endpoint** → `ZILLIZ_URI`
> 4. Generate or copy an **API Key** → `ZILLIZ_TOKEN`

---

## Usage

### Step 1 – Extract city pages to JSONL

```bash
# Extract 500 city pages
WIKIVOYAGE_LIMIT=500 python3 get_wikivoyage.py

# All pages, all types (takes a long time)
WIKIVOYAGE_PAGE_TYPES='*' WIKIVOYAGE_LIMIT=0 python3 get_wikivoyage.py

# Use a local dump file
WIKIVOYAGE_DUMP=/path/to/enwikivoyage-latest-pages-articles.xml.bz2 \
WIKIVOYAGE_LIMIT=0 python3 get_wikivoyage.py
```

Output: `pages/Tokyo.jsonl`, `pages/Abadeh.jsonl`, …

### Step 2 – Embed and upsert to Zilliz Cloud

```bash
python3 embedding/upsert_zilliz.py
```

On the first run, the BGE-M3 model (~2.2 GB) is downloaded automatically.

---

## JSONL Schema (`pages/*.jsonl`)

One file = one page = one JSON line.

| Field | Type | Description |
|---|---|---|
| `page_id` | int | Wikivoyage page ID |
| `title` | str | Page title |
| `slug` | str | Filename-safe title (spaces → `_`) |
| `namespace` | int | Namespace (0 = article) |
| `page_type` | str | `city` / `region` / `country` / `other` |
| `status` | str | `outline` / `usable` / `guide` / `star` |
| `is_redirect` | bool | Whether the page is a redirect |
| `redirect_target` | str\|null | Redirect target title |
| `source` | str | `enwikivoyage` |
| `url` | str | Page URL |
| `retrieved_at` | str | Retrieval timestamp (ISO 8601) |
| `revision_id` | int | Revision ID |
| `timestamp` | str | Last edited timestamp |
| `wikitext` | str | Raw wikitext |
| `clean_text` | str | Plain text (templates stripped) |
| `sections` | list | Section-split result |
| `retrieval_text` | str | Labeled text for RAG embedding |

---

## Zilliz Cloud Collection Schema

| Field | Type | Description |
|---|---|---|
| `page_id` | INT64 (PK) | Page ID |
| `title` | VARCHAR(512) | Title |
| `slug` | VARCHAR(512) | Slug |
| `page_type` | VARCHAR(64) | Page type |
| `status` | VARCHAR(64) | Status |
| `url` | VARCHAR(1024) | URL |
| `source` | VARCHAR(64) | Data source |
| `retrieval_text` | VARCHAR(65535) | Text for retrieval |
| `dense_vector` | FLOAT_VECTOR(1024) | BGE-M3 dense (HNSW / COSINE) |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | BGE-M3 sparse lexical weights (IP) |

The collection and indexes are **created automatically** on the first run if they do not exist.

---

## GPU Acceleration

**Apple Silicon (M1/M2/M3)**
```ini
BGE_M3_DEVICE=mps
BGE_M3_USE_FP16=0
```

**NVIDIA GPU**
```ini
BGE_M3_DEVICE=cuda
BGE_M3_USE_FP16=1
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pymilvus[model]>=2.4` | Milvus client + BGE-M3 embedding function |
| `mwparserfromhell>=0.6.6` | Wikitext parsing and template stripping |
| `requests>=2.34` | Downloading the dump |
| `python-dotenv>=1.0` | Loading `.env` files |
| `torch>=2.0` | BGE-M3 inference backend |

```bash
pip install -r requirements.txt
```
