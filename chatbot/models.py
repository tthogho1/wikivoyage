"""Pydantic models for the Wikivoyage RAG chatbot API."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for dense search (0-1)")
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for sparse search (0-1)")


class Source(BaseModel):
    title: str
    page_type: str
    status: Optional[str] = None
    url: Optional[str] = None
    retrieval_text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    question: str


class HealthResponse(BaseModel):
    status: str
    collection: str
    collection_exists: bool
