# Wikivoyage RAG

Travel chat with retrieval-augmented generation (RAG) over English Wikivoyage. You can ask with text, a
photo, or both. Text is embedded with **Qwen3-Embedding-0.6B** and images with **SigLIP2**. Each article
is stored as one row of the Zilliz Cloud collection **`wikivoyage_articles`**, with its text chunks and
images in StructArray fields. A Next.js web app answers with OpenAI `gpt-4o-mini`.

```
Wikivoyage dump (.xml.bz2)
 ├─ 1. get_wikivoyage.py ─────────── pages (Parquet) ── s3://…/pages/
 │     └─ 2. scripts/embed_chunks.py ─ section chunks + Qwen3 vectors ── chunks_embedded/
 └─ 3. python -m image_embedding ─── article images + SigLIP2 vectors ── images_work/
                          │
       4. python -m article_db all  (join by page_id, no re-embedding)
                          ▼
            Zilliz  wikivoyage_articles   1 row = 1 article
              chunks[]  text + Qwen3 (1024)      images[]  Commons URL/licence + SigLIP2 (1152)
                          ▲
       5. article_db.api (FastAPI :8001, Qwen3 + SigLIP2 loaded)  ◀──  webapp/ (Next.js :3000) ──▶ OpenAI
```

## Repository layout

| Path | What it is |
|---|---|
| `get_wikivoyage.py`, `wikivoyage_stream/` | Stage 1: stream the dump, keep articles, write one record per page (Parquet or JSONL, local or S3) |
| `scripts/embed_chunks.py` | Stage 2: split sections into chunks (≤ 2,000 chars) and embed them with Qwen3 locally (MPS) |
| `image_embedding/` | Stage 3: article images → Commons metadata → thumbnails → SigLIP2 vectors (`python -m image_embedding`) |
| `article_db/` | Stage 4: build and upsert `wikivoyage_articles`. Also the search (`search.py`, CLI `search`) and the retrieval API (`api.py`) |
| `webapp/` | Next.js (TypeScript) chat UI and `/api/chat` route. See [webapp/README.md](webapp/README.md) |
| `databricks/` | Alternative search backend: the same Qwen3 chunks in a Databricks Vector Search index (`create_vs_index.py`, `build_index.py`, `query_index.py`) |
| `design/` | Design documents and the StructArray spike results (Japanese) |
| `tests/` | `pytest` tests for `image_embedding` and `article_db` |
| `infra/`, `Dockerfile`, `Dockerfile.images`, `scripts/publish_ecr.sh` | AWS Batch (Fargate) jobs for stage 1 and the image pipeline's download stages |
| `scripts/spike_structarray.py` | One-off check of Zilliz StructArray features (results in `design/structarray-spike-result.md`) |
| `legacy/` | Retired first version (BGE-M3 → `wikivoyage_pages`, FastAPI chatbot on Hugging Face Spaces). Not maintained |

Local data folders (gitignored): `chunks_embedded/` (stage 2), `images_work/` (stage 3),
`articles_work/` (stage 4 snapshots).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_all.txt     # everything; or only the set a stage needs (below)
cp .env.example .env                    # fill in ZILLIZ_URI, ZILLIZ_TOKEN, OPENAI_API_KEY, AWS creds
```

| Requirements file | For |
|---|---|
| `requirements.txt` | Stage 1 only (also what the stage-1 Docker image installs) |
| `image_embedding/requirements.txt` | Image pipeline without torch (`dump`, `extract`, `commons`, `fetch`, `upsert`) |
| `image_embedding/requirements-embed.txt` | Adds torch + transformers for `embed` |
| `article_db/requirements-search.txt` | Search CLI and retrieval API (Qwen3 + SigLIP2, FastAPI) |
| `image_embedding/requirements-dev.txt` | `pytest` |

Every entry point loads the repo-root `.env`; variables already set in the environment win.
`.env.example` lists every variable with its default.

## Pipeline

All commands run from the repo root with the venv active.

```bash
# 1. Dump → pages (Parquet on S3; WIKIVOYAGE_PAGE_TYPES=* keeps every article type)
python get_wikivoyage.py --format parquet --s3-bucket wikivoyage-bucket --outdir pages

# 2. Pages → chunks + Qwen3 vectors (~2 days on an M4; resumable)
caffeinate -i python scripts/embed_chunks.py

# 3. Images → SigLIP2 vectors (each stage resumable; `status` shows progress)
D=20261101   # a dump date from https://dumps.wikimedia.org/enwikivoyage/
for s in dump extract commons fetch embed; do python -m image_embedding $s --dump-date $D || break; done

# 4. Join by page_id and upsert to Zilliz (~1 h, mostly upload)
python -m article_db all --image-dump-date $D
python -m article_db status
```

Notes:
- **Stage 3:** downloaded image files are deleted once embedded (`IMAGES_KEEP_FILES=true` keeps them).
  `image_embedding upsert`/`search` write to and read a separate `wikivoyage_images` collection. They are
  optional and not needed by `article_db`.
- **Stage 4:** `upsert` sends only articles whose content changed, `--prune` deletes articles that are
  gone, and `replay` resends failed batches. Exit codes: 1 error, 3 schema mismatch, 4 failed batches,
  5 inconsistent inputs.

## Search and chat

```bash
# CLI
python -m article_db search --text "cheap hostel near the city centre in Aachen"
python -m article_db search --image photo.jpg --top-k 5
python -m article_db search --text "castle" --page-type city --heading See --json

# Retrieval API (terminal 1) and web app (terminal 2) → http://localhost:3000
uvicorn article_db.api:app --host 127.0.0.1 --port 8001
cd webapp && npm install && npm run dev
```

Which model embeds the query:

| Query | Text chunks (`chunks[embedding]`) | Images (`images[embedding]`) |
|---|---|---|
| text | Qwen3, with its query instruction | SigLIP2 text encoder |
| photo | the first chunks of the matched articles | SigLIP2 image encoder |
| text + photo | Qwen3 on the text | SigLIP2 image encoder |

Results are merged per article with weighted reciprocal rank fusion (RRF): `ARTICLES_W_TEXT`=1.0,
`ARTICLES_W_IMAGE`=0.5, `ARTICLES_RRF_K`=60.

## Databricks (alternative backend)

The stage-2 Parquet can also be loaded into a Delta table and indexed by Databricks Vector Search.
Queries are embedded by the `databricks-qwen3-embedding-0-6b` serving endpoint.

```bash
python databricks/create_vs_index.py                       # create or sync the index (runs locally)
python databricks/query_index.py "best beaches" --page-type city
```

## Tests

```bash
pytest                       # image_embedding + article_db, no network
pytest -m model              # real SigLIP2 (downloads ~4.5 GB)
cd webapp && npm test && npm run typecheck && npm run build
```

## Deployment (AWS Batch on Fargate)

- `Dockerfile` (stage 1) and `Dockerfile.images` (the image pipeline's download stages, no torch).
  Build and push with `scripts/publish_ecr.sh` (`DOCKERFILE=Dockerfile.images REPO_NAME=wikivoyage-images`
  for the second).
- Job definitions and CloudFormation are in `infra/`. Jobs take their settings through
  container-override environment variables, e.g. `WIKIVOYAGE_S3_BUCKET`, `WIKIVOYAGE_LIMIT`,
  `IMAGES_WORKDIR=s3://…`.
- Run `embed` and stage 2 on a GPU or Apple Silicon machine against the same S3 locations.
