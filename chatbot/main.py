"""Wikivoyage RAG Chatbot – FastAPI application.

Usage
-----
    uvicorn chatbot.main:app --reload --port 8000

Environment variables (all loaded from .env):
    ZILLIZ_URI            Zilliz Cloud endpoint
    ZILLIZ_TOKEN          Zilliz API token
    OPENAI_API_KEY        OpenAI API key
    ZILLIZ_COLLECTION     Collection name (default: wikivoyage_pages)
    BGE_M3_DEVICE         cpu / cuda / mps (default: cpu)
    BGE_M3_USE_FP16       1 to enable fp16 (default: 0)
    OPENAI_MODEL          Chat model (default: gpt-4o-mini)
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# dotenv
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*a, **kw): pass  # type: ignore

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
try:
    from pymilvus import MilvusClient
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction
except ImportError as e:
    sys.exit(f"[ERROR] {e}\nInstall: pip install 'pymilvus[model]>=2.4'")

try:
    from openai import OpenAI
except ImportError as e:
    sys.exit(f"[ERROR] {e}\nInstall: pip install openai")

from chatbot.chat import generate_answer
from chatbot.models import ChatRequest, ChatResponse, HealthResponse, Source
from chatbot.retriever import HybridRetriever

logger = logging.getLogger("wikivoyage.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# App state (initialised during lifespan)
# ---------------------------------------------------------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup, release on shutdown."""
    # --- config ---
    zilliz_uri   = os.environ.get("ZILLIZ_URI", "").strip()
    zilliz_token = os.environ.get("ZILLIZ_TOKEN", "").strip()
    openai_key   = os.environ.get("OPENAI_API_KEY", "").strip()
    collection   = os.environ.get("ZILLIZ_COLLECTION", "wikivoyage_pages")
    device       = os.environ.get("BGE_M3_DEVICE", "cpu")
    use_fp16     = os.environ.get("BGE_M3_USE_FP16", "0") == "1"

    if not zilliz_uri or not zilliz_token:
        raise RuntimeError("ZILLIZ_URI and ZILLIZ_TOKEN must be set in .env")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY must be set in .env")

    # --- BGE-M3 ---
    logger.info("Loading BGE-M3 on device=%s fp16=%s …", device, use_fp16)
    ef = BGEM3EmbeddingFunction(
        model_name="BAAI/bge-m3",
        device=device,
        use_fp16=use_fp16,
    )

    # --- Zilliz ---
    logger.info("Connecting to Zilliz Cloud: %s", zilliz_uri)
    milvus_client = MilvusClient(uri=zilliz_uri, token=zilliz_token)
    if not milvus_client.has_collection(collection):
        raise RuntimeError(f"Collection {collection!r} not found. Run upsert_zilliz.py first.")
    logger.info("Collection %r ready.", collection)

    # --- OpenAI ---
    openai_client = OpenAI(api_key=openai_key)

    _state["retriever"] = HybridRetriever(milvus_client, ef, collection)
    _state["openai"]    = openai_client
    _state["collection"] = collection
    _state["milvus"]    = milvus_client

    logger.info("Startup complete.")
    yield

    # Cleanup
    _state.clear()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Wikivoyage RAG Chatbot",
    description="Hybrid dense+sparse retrieval (BGE-M3) over Wikivoyage city pages with OpenAI answer generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Check service health and Zilliz collection status."""
    milvus: MilvusClient = _state.get("milvus")
    collection: str = _state.get("collection", "wikivoyage_pages")
    exists = milvus.has_collection(collection) if milvus else False
    return HealthResponse(
        status="ok",
        collection=collection,
        collection_exists=exists,
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest) -> ChatResponse:
    """Answer a travel question using RAG over Wikivoyage pages.

    1. Embed the question with BGE-M3
    2. Hybrid search (dense + sparse) against Zilliz Cloud
    3. Generate an answer with OpenAI using the retrieved passages
    """
    retriever: HybridRetriever = _state.get("retriever")
    openai_client = _state.get("openai")

    if not retriever or not openai_client:
        raise HTTPException(status_code=503, detail="Service not ready.")

    try:
        hits = retriever.search(
            question=req.question,
            top_k=req.top_k,
            dense_weight=req.dense_weight,
            sparse_weight=req.sparse_weight,
        )
    except Exception as e:
        logger.error("Retrieval error: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    try:
        answer = generate_answer(req.question, hits, openai_client)
    except Exception as e:
        logger.error("LLM error: %s", e)
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {e}")

    sources = [
        Source(
            title=h["title"],
            page_type=h["page_type"],
            status=h.get("status") or None,
            url=h.get("url") or None,
            retrieval_text=h["retrieval_text"],
            score=round(h["score"], 4),
        )
        for h in hits
    ]

    return ChatResponse(answer=answer, sources=sources, question=req.question)
