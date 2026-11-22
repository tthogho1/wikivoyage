"""Configuration from environment variables (+ CLI overrides)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from image_embedding.config import StageError  # noqa: F401 — re-exported

from .schema import DEFAULT_MAX_ROW_BYTES

DEFAULT_TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_PAGES_S3 = "s3://wikivoyage-bucket/pages/"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v else default


@dataclass
class ArticlesConfig:
    workdir: str = "articles_work"
    chunks_source: str = "chunks_embedded"
    pages_source: str = ""            # "" → resolved by resolved_pages_source()
    text_model: str = DEFAULT_TEXT_MODEL
    images_workdir: str = "images_work"
    image_dump_date: Optional[str] = None
    image_model_slug: Optional[str] = None
    no_images: bool = False
    collection: str = "wikivoyage_articles"
    bm25_collection: str = "wikivoyage_chunks_bm25"
    wiki: str = "enwikivoyage"
    max_chunks: int = 512
    max_images: int = 64
    max_row_bytes: int = DEFAULT_MAX_ROW_BYTES
    batch_size: int = 50
    max_batch_bytes: int = 8 * 1024 * 1024
    articles_per_part: int = 500
    zilliz_uri: str = ""
    zilliz_token: str = ""

    @classmethod
    def from_env(cls) -> "ArticlesConfig":
        return cls(
            workdir=_env("ARTICLES_WORKDIR", "articles_work"),
            chunks_source=_env("ARTICLES_CHUNKS_SOURCE", "chunks_embedded"),
            pages_source=_env("ARTICLES_PAGES_SOURCE"),
            text_model=_env("ARTICLES_TEXT_MODEL", DEFAULT_TEXT_MODEL),
            images_workdir=_env("IMAGES_WORKDIR", "images_work"),
            image_dump_date=_env("IMAGES_DUMP_DATE") or None,
            collection=_env("ZILLIZ_ARTICLE_COLLECTION", "wikivoyage_articles"),
            bm25_collection=_env("ZILLIZ_BM25_COLLECTION", "wikivoyage_chunks_bm25"),
            wiki=_env("IMAGES_WIKI", "enwikivoyage"),
            max_chunks=_env_int("ARTICLES_MAX_CHUNKS", 512),
            max_images=_env_int("ARTICLES_MAX_IMAGES", 64),
            max_row_bytes=_env_int("ARTICLES_MAX_ROW_BYTES", DEFAULT_MAX_ROW_BYTES),
            batch_size=_env_int("ARTICLES_BATCH_SIZE", 50),
            max_batch_bytes=_env_int("ARTICLES_MAX_BATCH_BYTES", 8 * 1024 * 1024),
            zilliz_uri=_env("ZILLIZ_URI"),
            zilliz_token=_env("ZILLIZ_TOKEN"),
        )

    def resolved_pages_source(self) -> str:
        """Stage-1 page Parquet. Default: the page cache embed_chunks.py left next to its parts
        (same pages the chunks were built from), else the S3 page prefix."""
        if self.pages_source:
            return self.pages_source
        if not self.chunks_source.startswith("s3://"):
            cache = os.path.join(self.chunks_source, "pages_cache")
            if os.path.isdir(cache):
                return cache
        return DEFAULT_PAGES_S3
