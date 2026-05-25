"""Hybrid retriever: BGE-M3 (dense + sparse) search against Zilliz Cloud."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("wikivoyage.retriever")

# Output fields to fetch from Zilliz
_OUTPUT_FIELDS = ["page_id", "title", "page_type", "status", "url", "retrieval_text"]


class HybridRetriever:
    """Encapsulates BGE-M3 embedding + hybrid Zilliz search.

    Initialise once at app startup and reuse for every query.
    """

    def __init__(
        self,
        client: Any,                  # MilvusClient
        ef: Any,                      # BGEM3EmbeddingFunction
        collection: str,
    ) -> None:
        self._client = client
        self._ef = ef
        self._collection = collection

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _embed_query(self, text: str) -> tuple[list[float], dict[int, float]]:
        """Embed a single query text. Returns (dense_vec, sparse_vec)."""
        output = self._ef([text])
        dense = output["dense"][0]

        raw_sparse = output["sparse"][0]
        try:
            cx = raw_sparse.tocoo()
            sparse: dict[int, float] = {int(j): float(v) for j, v in zip(cx.col, cx.data)}
        except AttributeError:
            sparse = {int(k): float(v) for k, v in raw_sparse.items()}

        return dense, sparse

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------
    def search(
        self,
        question: str,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[dict]:
        """Run hybrid dense+sparse search and return ranked results.

        Each result dict contains the output fields plus a ``score`` key.
        """
        from pymilvus import AnnSearchRequest, WeightedRanker

        dense_vec, sparse_vec = self._embed_query(question)

        # Dense ANN request
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k,
        )

        # Sparse ANN request
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k,
        )

        ranker = WeightedRanker(dense_weight, sparse_weight)

        results = self._client.hybrid_search(
            collection_name=self._collection,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
        )

        hits = []
        for hit in results[0]:
            entity = hit.get("entity", hit)
            hits.append({
                "page_id":        entity.get("page_id"),
                "title":          entity.get("title", ""),
                "page_type":      entity.get("page_type", ""),
                "status":         entity.get("status", ""),
                "url":            entity.get("url", ""),
                "retrieval_text": entity.get("retrieval_text", ""),
                "score":          hit.get("distance", 0.0),
            })

        logger.info("Retrieved %d hits for question: %.60s…", len(hits), question)
        return hits
