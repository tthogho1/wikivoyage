"""Qwen3-Embedding query encoder for ``chunks[embedding]``.

The chunks were embedded by ``scripts/embed_chunks.py`` with no prompt (documents), so queries get
Qwen3's retrieval instruction, as in ``databricks/query_index.py``. torch is imported lazily.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("wikivoyage.articles.text_embedder")

QUERY_PREFIX = ("Instruct: Given a web search query, retrieve relevant passages "
                "that answer the query\nQuery:")


class QwenQueryEmbedder:
    def __init__(self, model_id: str = "Qwen/Qwen3-Embedding-0.6B", device: Optional[str] = None,
                 max_seq_length: int = 512) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if not device or device == "auto":
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16} if device in ("mps", "cuda") else {}
        logger.info("loading %s on %s", model_id, device)
        self.model = SentenceTransformer(model_id, device=device, model_kwargs=kwargs)
        self.model.max_seq_length = max_seq_length
        self.model_id = model_id
        self.device = device

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """(n, 1024) float32, L2-normalised."""
        vecs = self.model.encode(texts, prompt=QUERY_PREFIX, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
