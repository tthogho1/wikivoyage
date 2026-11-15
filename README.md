# Wikivoyage — RAG chatbot

This repository contains a pipeline that converts Wikivoyage pages into JSONL, chunks and embeds them with BGE‑M3 (dense + sparse), upserts to a Milvus/Zilliz collection, and exposes a small FastAPI RAG chatbot with a minimal web UI.

Quick start
1. Activate the virtualenv and install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file at the project root by copying `.env.example` and filling credentials:

```bash
cp .env.example .env
# Edit .env and set ZILLIZ_URI, ZILLIZ_TOKEN, OPENAI_API_KEY
```

3. If you haven't upserted embeddings yet, run the upsert orchestrator (from repo root):

```bash
python -m embedding.upsert_zilliz
```

4. Start the FastAPI chatbot server (from repo root):

```bash
uvicorn chatbot.main:app --reload --host 127.0.0.1 --port 8000
```

5. Open the web UI in your browser:

  http://127.0.0.1:8000/

Health and example curl

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Where should I stay in Paris?","top_k":5}'
```

Notes
- The web UI is served from `chatbot/static/` and the endpoint `/` returns `index.html`.
- Ensure `.env` contains valid Zilliz credentials; otherwise the retriever will fail on startup.
- If you want token-based chunking, install `langchain` and `tiktoken` (requirements currently reference them as optional).

If you'd like, I can also: add README sections for embedding tuning, add a Docker dev container, or add smoke tests for chunking/embed normalization.
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

---

## Image pipeline (`python -m image_embedding`)

Embeds every image used in Wikivoyage articles with
[SigLIP2](https://huggingface.co/google/siglip2-so400m-patch16-384) and stores them in the Zilliz
collection `wikivoyage_images`, one row per (article, image) pair, with page URL/title, image URL and
license/attribution. Design: [`design/image-embedding-design.md`](design/image-embedding-design.md).

```bash
pip install -r image_embedding/requirements-embed.txt   # fetch stages only: image_embedding/requirements.txt

D=20261101   # a dump date listed at https://dumps.wikimedia.org/enwikivoyage/
python -m image_embedding dump    --dump-date $D   # download + SHA-1 check (resumable)
python -m image_embedding extract --dump-date $D   # [[File:]], <gallery>, template image= → refs.jsonl
python -m image_embedding commons --dump-date $D   # Commons imageinfo: URLs, MIME, SHA-1, license
python -m image_embedding fetch   --dump-date $D   # 768px thumbnails → validate → RGB JPEG
python -m image_embedding embed   --dump-date $D   # SigLIP2 vectors (GPU / Apple Silicon recommended)
python -m image_embedding upsert  --dump-date $D   # Zilliz upsert, 100 rows per batch
python -m image_embedding all     --dump-date $D --limit 20   # everything, small trial run

python -m image_embedding status  --dump-date $D
python -m image_embedding replay  --dump-date $D   # re-send failed upsert batches
python -m image_embedding search  --text "beach at sunset" [--page-title Tokyo]
python -m image_embedding search  --image photo.jpg | --image-url https://...
```

Every stage can be re-run: finished work is skipped, so an interrupted run resumes. Images and
embeddings are cached across dumps (keyed by the Commons SHA-1). On a new dump,
`upsert --prune` deletes rows whose image was removed from the article. It runs only if every row
was upserted. Intermediate files live under `IMAGES_WORKDIR` (local dir or `s3://…`).

Exit codes: `1` error, `2` dump checksum mismatch, `3` collection schema mismatch, `4` failed upsert
batches (run `replay`).

Tests: `pip install -r image_embedding/requirements-dev.txt && pytest` (the real-model test runs only
with `pytest -m model`, a ~4.5 GB download).

On AWS Batch, build `Dockerfile.images`
(`DOCKERFILE=Dockerfile.images REPO_NAME=wikivoyage-images ./scripts/publish_ecr.sh`) and submit jobs
with `--parameters stage=<stage>,dump_date=<date>` against `wikivoyage-images-fargate`. The image has
no torch: run `embed` on a GPU or Apple Silicon machine against the same `IMAGES_WORKDIR=s3://…`.
