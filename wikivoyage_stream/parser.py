"""Namespace-aware incremental XML parser for MediaWiki export XML."""
from __future__ import annotations

import logging
from typing import Iterator, Optional
import xml.etree.ElementTree as ET

from .models import Page

logger = logging.getLogger("wikivoyage.stream")


class NamespaceAwareXmlParser:
    """Wraps XMLPullParser and auto-detects the MediaWiki XML namespace.

    The namespace URI of MediaWiki export XML changes across versions
    so we read it from the root ``<mediawiki>`` element instead of hard-coding it.
    """

    def __init__(self) -> None:
        self._parser = ET.XMLPullParser(events=("start", "end"))
        self._ns_uri: Optional[str] = None
        self._ns: str = ""
        self._last_title: Optional[str] = None

    @property
    def namespace_uri(self) -> Optional[str]:
        return self._ns_uri

    @property
    def last_title(self) -> Optional[str]:
        return self._last_title

    def feed(self, text: str) -> None:
        self._parser.feed(text)

    def _detect_namespace(self, elem: ET.Element) -> None:
        tag = elem.tag
        if tag.startswith("{") and "}" in tag:
            uri = tag.split("}", 1)[0].lstrip("{")
            local = tag.split("}", 1)[1]
        else:
            uri = ""
            local = tag
        if local != "mediawiki":
            logger.warning("Root element is not <mediawiki>: %s", tag)
        self._ns_uri = uri
        self._ns = f"{{{uri}}}" if uri else ""
        logger.info("Detected MediaWiki XML namespace: %r", uri)

    def iter_pages(self) -> Iterator[Page]:
        ns = self._ns
        for event, elem in self._parser.read_events():
            if event == "start":
                if self._ns_uri is None:
                    self._detect_namespace(elem)
                    ns = self._ns
                continue

            if not ns:
                continue

            if elem.tag != f"{ns}page":
                continue

            title = elem.findtext(f"{ns}title")
            ns_id = elem.findtext(f"{ns}ns")
            page_id: Optional[str] = None
            for child in elem.findall(f"{ns}id"):
                page_id = child.text
                break
            revision_node = elem.find(f"{ns}revision")
            revision_id: Optional[str] = None
            timestamp: Optional[str] = None
            page_text = ""
            if revision_node is not None:
                rid_node = revision_node.find(f"{ns}id")
                if rid_node is not None and rid_node.text:
                    revision_id = rid_node.text
                ts_node = revision_node.find(f"{ns}timestamp")
                if ts_node is not None and ts_node.text:
                    timestamp = ts_node.text
                text_node = revision_node.find(f"{ns}text")
                if text_node is not None and text_node.text:
                    page_text = text_node.text

            self._last_title = title

            yield Page(
                page_id=page_id,
                title=title,
                namespace=ns_id,
                text=page_text,
                revision_id=revision_id,
                timestamp=timestamp,
            )

            elem.clear()
