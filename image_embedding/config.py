"""Configuration (environment variables + CLI overrides) and work-dir layout."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from wikivoyage_stream.reader import DEFAULT_USER_AGENT

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
DUMPS_BASE = "https://dumps.wikimedia.org"
DEFAULT_MODEL_ID = "google/siglip2-so400m-patch16-384"
DEFAULT_ALLOWED_MIME = "image/jpeg,image/png,image/webp,image/svg+xml"


class StageError(Exception):
    """A stage failure with the CLI exit code to use."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v else default


@dataclass
class ImagesConfig:
    workdir: str = "images_work"
    dump_date: Optional[str] = None
    wiki: str = "enwikivoyage"
    site_base: str = "https://en.wikivoyage.org/wiki/"
    local_api: str = ""
    user_agent: str = DEFAULT_USER_AGENT
    max_per_page: int = 50
    page_types: str = "*"
    thumb_width: int = 768
    allowed_mime: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_ALLOWED_MIME.split(",")))
    api_concurrency: int = 2
    download_concurrency: int = 3
    download_rps: float = 4.0
    min_side: int = 100
    max_side: int = 8192
    pad_aspect: float = 2.0
    flush_every: int = 1000
    fetch_max_attempts: int = 3
    model_id: str = DEFAULT_MODEL_ID
    model_revision: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    embed_batch_size: int = 32
    text_template: str = "{}"
    keep_images: bool = False  # delete raw/processed image files once their vectors are written
    hf_token: Optional[str] = None
    collection: str = "wikivoyage_images"
    zilliz_batch_size: int = 100
    zilliz_uri: str = ""
    zilliz_token: str = ""

    @classmethod
    def from_env(cls) -> "ImagesConfig":
        site_base = _env("IMAGES_SITE_BASE", "https://en.wikivoyage.org/wiki/")
        return cls(
            workdir=_env("IMAGES_WORKDIR", "images_work"),
            dump_date=_env("IMAGES_DUMP_DATE") or None,
            wiki=_env("IMAGES_WIKI", "enwikivoyage"),
            site_base=site_base,
            local_api=_env("IMAGES_LOCAL_API"),
            user_agent=_env("WIKIVOYAGE_USER_AGENT") or DEFAULT_USER_AGENT,
            max_per_page=_env_int("IMAGES_MAX_PER_PAGE", 50),
            page_types=_env("IMAGES_PAGE_TYPES", "*") or "*",
            thumb_width=_env_int("IMAGES_THUMB_WIDTH", 768),
            allowed_mime=tuple(
                m.strip().lower()
                for m in _env("IMAGES_ALLOWED_MIME", DEFAULT_ALLOWED_MIME).split(",")
                if m.strip()
            ),
            api_concurrency=min(4, max(1, _env_int("IMAGES_API_CONCURRENCY", 2))),
            download_concurrency=max(1, _env_int("IMAGES_DOWNLOAD_CONCURRENCY", 3)),
            download_rps=_env_float("IMAGES_DOWNLOAD_RPS", 4.0),
            min_side=_env_int("IMAGES_MIN_SIDE", 100),
            max_side=_env_int("IMAGES_MAX_SIDE", 8192),
            pad_aspect=_env_float("IMAGES_PAD_ASPECT", 2.0),
            flush_every=max(1, _env_int("IMAGES_FLUSH_EVERY", 1000)),
            fetch_max_attempts=_env_int("IMAGES_FETCH_MAX_ATTEMPTS", 3),
            model_id=_env("SIGLIP_MODEL_ID", DEFAULT_MODEL_ID),
            model_revision=_env("SIGLIP_REVISION") or None,
            device=_env("SIGLIP_DEVICE", "auto") or "auto",
            dtype=_env("SIGLIP_DTYPE", "auto") or "auto",
            embed_batch_size=_env_int("SIGLIP_BATCH_SIZE", 32),
            text_template=os.environ.get("SIGLIP_TEXT_TEMPLATE", "{}") or "{}",
            keep_images=_env("IMAGES_KEEP_FILES").lower() in ("1", "true", "yes"),
            hf_token=_env("HF_TOKEN") or None,
            collection=_env("ZILLIZ_IMAGE_COLLECTION", "wikivoyage_images"),
            zilliz_batch_size=_env_int("IMAGES_ZILLIZ_BATCH_SIZE", 100),
            zilliz_uri=_env("ZILLIZ_URI"),
            zilliz_token=_env("ZILLIZ_TOKEN"),
        )

    # ------------------------------------------------------------------
    # Derived endpoints
    # ------------------------------------------------------------------
    @property
    def local_api_url(self) -> str:
        """MediaWiki API of the source wiki (for files uploaded locally, not on Commons)."""
        if self.local_api:
            return self.local_api
        base = self.site_base.rstrip("/")
        if base.endswith("/wiki"):
            base = base[: -len("/wiki")]
        return base + "/w/api.php"

    def require_dump_date(self) -> str:
        d = (self.dump_date or "").strip()
        if not (len(d) == 8 and d.isdigit()):
            raise StageError(
                f"--dump-date YYYYMMDD is required (got {self.dump_date!r}; 'latest' is not allowed)", 1
            )
        return d

    # ------------------------------------------------------------------
    # Work-dir layout (paths relative to IMAGES_WORKDIR)
    # ------------------------------------------------------------------
    def dump_filename(self) -> str:
        return f"{self.wiki}-{self.require_dump_date()}-pages-articles.xml.bz2"

    def dump_url(self) -> str:
        return f"{DUMPS_BASE}/{self.wiki}/{self.require_dump_date()}/{self.dump_filename()}"

    def dump_rel(self) -> str:
        return f"dumps/{self.require_dump_date()}/{self.dump_filename()}"

    def run_dir(self) -> str:
        return f"runs/{self.require_dump_date()}"

    def refs_rel(self) -> str:
        return f"{self.run_dir()}/refs.jsonl"

    def extract_errors_rel(self) -> str:
        return f"{self.run_dir()}/extract_errors.jsonl"

    def commons_dir(self) -> str:
        return f"{self.run_dir()}/commons"

    def upsert_dir(self) -> str:
        return f"{self.run_dir()}/upsert"

    def summary_dir(self) -> str:
        return f"{self.run_dir()}/summary"

    @staticmethod
    def manifest_dir() -> str:
        return "images/manifest"

    @staticmethod
    def raw_rel(file_key: str, ext: str) -> str:
        return f"images/raw/{file_key[:2]}/{file_key}.{ext}"

    @staticmethod
    def processed_rel(file_key: str) -> str:
        return f"images/processed/{file_key[:2]}/{file_key}.jpg"

    @staticmethod
    def embeddings_dir(slug: str) -> str:
        return f"embeddings/{slug}"
