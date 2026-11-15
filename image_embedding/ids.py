"""Deterministic identifiers used across stages."""
from __future__ import annotations

import hashlib
import secrets
import time
from typing import Optional


def file_key(file_title: str) -> str:
    """40-char hex SHA-1 of the normalised ``File:...`` title (cache / file-name key)."""
    return hashlib.sha1(file_title.encode("utf-8")).hexdigest()


def record_id(wiki: str, page_id: int, file_title: str) -> int:
    """Positive INT64 Zilliz primary key for one (article, image) pair.

    Independent of dump date, position and model, so re-runs and later dumps
    overwrite the same row.
    """
    digest = hashlib.sha1(f"{wiki}|{page_id}|{file_title}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def model_slug(model_id: str, revision: Optional[str]) -> str:
    return model_id.replace("/", "__") + "@" + (revision or "unknown")[:12]


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(2)
