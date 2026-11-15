"""Small package for streaming Wikivoyage dump parsing.

Exports: Page, NamespaceAwareXmlParser, WikivoyageDumpStreamReader
"""
from .models import Page
from .parser import NamespaceAwareXmlParser
from .reader import WikivoyageDumpStreamReader
from . import filters

__all__ = ["Page", "NamespaceAwareXmlParser", "WikivoyageDumpStreamReader", "filters"]
