# Wikivoyage Chat (web app)

Next.js (TypeScript) chat UI with RAG over the Zilliz collection `wikivoyage_articles`.
Questions can be text, a photo, or both. Retrieval runs in the Python API (`article_db/api.py`),
because the stored vectors come from Qwen3-Embedding-0.6B and SigLIP2 (PyTorch).
The answer is generated here with OpenAI.

| Query | Text chunks (`chunks[embedding]`) | Images (`images[embedding]`) |
|---|---|---|
| text | Qwen3 (query instruction) + BM25 | SigLIP2 text encoder |
| photo | first chunks of the matched articles | SigLIP2 image encoder |
| text + photo | Qwen3 + BM25 on the text | SigLIP2 image encoder |

The lists are merged per article with weighted RRF (`ARTICLES_W_TEXT`=1.0, `ARTICLES_W_BM25`=1.0,
`ARTICLES_W_IMAGE`=0.5). BM25 runs on the `wikivoyage_chunks_bm25` collection; without it the text search
is dense only.
For follow-up turns, `gpt-4o-mini` first rewrites the question into a standalone query.
The photo is also sent to `gpt-4o-mini` with the context.

## Run (two terminals, from the repo root)

```bash
# 1. Retrieval API (Qwen3 + SigLIP2 stay loaded; needs ZILLIZ_URI / ZILLIZ_TOKEN in ../.env)
source .venv/bin/activate
pip install -r article_db/requirements-search.txt
uvicorn article_db.api:app --host 127.0.0.1 --port 8001
curl localhost:8001/health

# 2. Web app (uses OPENAI_API_KEY from ../.env)
cd webapp && npm install && npm run dev   # http://localhost:3000
```

Settings come from the repo-root `.env`. `webapp/.env.local` overrides it (see `.env.local.example`):
`OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4o-mini`), `RETRIEVAL_URL` (default `http://127.0.0.1:8001`).

## API

`POST /api/chat` takes multipart data: `question?`, `image?` (jpeg/png/webp/gif, at most 20 MB),
and `history?` (JSON `[{role, content}]`). It streams NDJSON:
`{"type":"sources", searchQuery, search, citations}`, then `{"type":"delta", text}` events,
then `{"type":"done"}`, or `{"type":"error", message}` on failure.

## Checks

```bash
npm test          # vitest: prompt building, NDJSON parsing
npm run typecheck
npm run build
```
