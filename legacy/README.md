# Legacy: BGE-M3 → Zilliz `wikivoyage_pages` (not maintained)

The first version of the project, replaced by `article_db/` + `webapp/`. Kept for reference only.
Its Zilliz collection `wikivoyage_pages` no longer exists, so none of this works without
re-running the old upsert first.

| Path | What it was |
|---|---|
| `embedding/` | Stage 2: `retrieval_text` (1,500 chars) → char/LangChain chunks → BGE-M3 dense + sparse → Zilliz upsert. `upserter.py` is now a re-export of `image_embedding.upserter` |
| `chatbot/` | FastAPI + static UI, hybrid BGE-M3 search, OpenAI answers. Deployed to the Hugging Face Space `tthogho1/wikivoyage` (`hf` remote), port 7860 |
| `scripts/check_count.py`, `list_records.py`, `drop_collection.py` | Inspect / drop `ZILLIZ_COLLECTION` |
| `scripts/set_hf_secrets.py` | Push `.env` to the Space as secrets |

Known problems when it was retired: the "recursive" chunker silently fell back to character slicing
(LangChain 1.x moved `langchain.text_splitter`), and `retrieval_text` was cut to 1,500 chars, so pages
had only 1–2 chunks.

Running it (from the repo root, which still holds `.env` and `image_embedding/`):

```bash
pip install -r legacy/chatbot/requirements.txt langchain-text-splitters tiktoken
PYTHONPATH=legacy:. python -m embedding.upsert_zilliz
cd legacy && PYTHONPATH=..:. uvicorn chatbot.main:app --port 8000
```
