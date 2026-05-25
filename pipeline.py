"""Compatibility wrapper: re-export package classes."""
from __future__ import annotations

from wikivoyage_pipeline.writer import WikivoyageJsonlWriter
from wikivoyage_pipeline.preprocessor import RagPreprocessor

__all__ = ["WikivoyageJsonlWriter", "RagPreprocessor"]
