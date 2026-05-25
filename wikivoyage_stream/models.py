"""Data models for the wikivoyage stream pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Page:
    page_id: Optional[str]
    title: Optional[str]
    namespace: Optional[str]
    text: str
    revision_id: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "title": self.title,
            "namespace": self.namespace,
            "text": self.text,
            "revision_id": self.revision_id,
            "timestamp": self.timestamp,
        }
