"""Embedder wrapper for BGE-M3 producing dense + sparse vectors.

Provides an `Embedder` class with a simple `embed_texts()` API that returns
(normalized_dense_list, normalized_sparse_list).

Dense vectors are lists[float]. Sparse vectors are dict[int,float].
"""
from __future__ import annotations

from typing import Any, Iterable, List, Tuple, Dict
import logging

logger = logging.getLogger("wikivoyage.embedder")


class Embedder:
    def __init__(
        self,
        ef: Any | None = None,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        use_fp16: bool = False,
    ) -> None:
        """Create an Embedder.

        If `ef` (a BGEM3EmbeddingFunction) is provided it will be used; otherwise
        the class will try to construct one lazily when needed.
        """
        self._ef = ef
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16

    def _ensure_ef(self) -> None:
        if self._ef is not None:
            return
        try:
            from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        except Exception as e:
            raise RuntimeError("BGEM3EmbeddingFunction not available: %s" % e)
        logger.info("Initialising BGEM3EmbeddingFunction model=%s device=%s fp16=%s",
                    self.model_name, self.device, self.use_fp16)
        self._ef = BGEM3EmbeddingFunction(
            model_name=self.model_name,
            device=self.device,
            use_fp16=self.use_fp16,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_texts(self, texts: Iterable[str], batch_size: int = 32) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """Embed an iterable of texts, yielding (dense_list, sparse_list).

        - `dense_list` is a list of lists of floats
        - `sparse_list` is a list of dict[int,float]
        """
        texts_list = list(texts)
        if not texts_list:
            return [], []

        self._ensure_ef()

        dense_out: List[List[float]] = []
        sparse_out: List[Dict[int, float]] = []

        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            try:
                output = self._ef(batch)
            except Exception as e:
                logger.error("Embedding function failed for batch starting at %d: %s", i, e)
                raise

            # Normalize dense vectors to plain lists of floats
            raw_dense = output.get("dense", [])
            for dv in raw_dense:
                # ensure Python floats
                dense_out.append([float(x) for x in dv])

            # Normalize sparse vectors to dict[int,float]
            raw_sparse = output.get("sparse", [])
            for s in raw_sparse:
                sparse_out.append(self._normalize_sparse(s))

        return dense_out, sparse_out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_sparse(sparse_obj: Any) -> Dict[int, float]:
        """Convert a sparse object (scipy matrix or dict-like) into {int: float}."""
        try:
            # scipy sparse matrix -> convert to COO
            cx = sparse_obj.tocoo()
            return {int(c): float(v) for c, v in zip(cx.col, cx.data)}
        except Exception:
            pass

        # dict-like
        try:
            return {int(k): float(v) for k, v in sparse_obj.items()}
        except Exception:
            logger.debug("Failed to normalise sparse object: %r", sparse_obj)
            return {}


# convenience factory
def make_embedder(**kwargs) -> Embedder:
    return Embedder(**kwargs)
