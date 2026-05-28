"""Chunking utilities for embedding texts.

Provides a `Chunker` class that supports:
- LangChain `RecursiveCharacterTextSplitter` (preferred)
- LangChain `TokenTextSplitter` (token-aware, uses `tiktoken`)
- Direct `tiktoken` token-based chunking fallback
- Simple char-based chunking fallback

Usage:
    from embedding.chunker import Chunker
    c = Chunker(method="recursive", chunk_size=1000, chunk_overlap=200)
    chunks = c.chunk(text)

Also supports `chunk_sections(sections)` where `sections` is a list of
objects with a `text` or `content` field.
"""
from __future__ import annotations

import logging
from typing import List, Iterable, Optional

logger = logging.getLogger("wikivoyage.chunker")


class Chunker:
    def __init__(
        self,
        method: str = "recursive",  # "recursive" | "token" | "char"
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        token_encoding: str = "cl100k_base",
    ) -> None:
        self.method = method
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.token_encoding = token_encoding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chunk(self, text: str) -> List[str]:
        """Return a list of chunks for `text` using the configured method."""
        text = (text or "").strip()
        if not text:
            return []

        if self.method == "recursive":
            return self._chunk_with_langchain_recursive(text)
        if self.method == "token":
            # prefer LangChain token splitter, fallback to tiktoken implementation
            return self._chunk_with_token_splitter(text)
        # default to char-based
        return self._chunk_char(text, self.chunk_size, self.chunk_overlap)

    def chunk_sections(self, sections: Iterable[dict]) -> List[str]:
        """Create chunks from a sequence of section dicts.

        Each section should contain a text field named either `text` or `content`.
        The chunker will keep section headers with the text when possible.
        """
        out: List[str] = []
        for s in sections:
            if not s:
                continue
            title = s.get("title") or s.get("header") or ""
            text = s.get("text") or s.get("content") or ""
            combined = (title + "\n\n" + text).strip() if title else text.strip()
            if not combined:
                continue
            out.extend(self.chunk(combined))
        return out

    # ------------------------------------------------------------------
    # Implementations / fallbacks
    # ------------------------------------------------------------------
    def _chunk_with_langchain_recursive(self, text: str) -> List[str]:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            return splitter.split_text(text)
        except Exception as e:
            logger.debug("LangChain recursive splitter unavailable: %s", e)
            return self._chunk_char(text, self.chunk_size, self.chunk_overlap)

    def _chunk_with_token_splitter(self, text: str) -> List[str]:
        # Try LangChain TokenTextSplitter first
        try:
            from langchain.text_splitter import TokenTextSplitter

            splitter = TokenTextSplitter(
                encoding_name=self.token_encoding,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            return splitter.split_text(text)
        except Exception:
            logger.debug("LangChain TokenTextSplitter unavailable, falling back to tiktoken")

        # Fallback: use tiktoken directly
        try:
            import tiktoken

            enc = tiktoken.get_encoding(self.token_encoding)
            ids = enc.encode(text)
            chunks: List[str] = []
            i = 0
            step = max(1, self.chunk_size - self.chunk_overlap)
            while i < len(ids):
                window = ids[i : i + self.chunk_size]
                chunks.append(enc.decode(window).strip())
                if i + self.chunk_size >= len(ids):
                    break
                i += step
            return chunks
        except Exception as e:
            logger.debug("tiktoken unavailable or failed: %s", e)
            return self._chunk_char(text, self.chunk_size, self.chunk_overlap)

    @staticmethod
    def _chunk_char(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
        if not text:
            return []
        chunks: List[str] = []
        L = len(text)
        s = 0
        while s < L:
            e = min(L, s + max_chars)
            chunk = text[s:e].strip()
            if chunk:
                chunks.append(chunk)
            if e == L:
                break
            s = max(0, e - overlap)
        return chunks


# convenience factory
def make_chunker(kind: str = "recursive", chunk_size: int = 1000, chunk_overlap: int = 200) -> Chunker:
    return Chunker(method=kind, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
