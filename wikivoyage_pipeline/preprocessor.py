"""RAG preprocessor (wikitext -> sectioned plain text JSONL)."""
from __future__ import annotations

import json
import re
from typing import Iterator, Optional

try:
    import mwparserfromhell
except Exception:
    mwparserfromhell = None


class RagPreprocessor:
    """Convert raw JSONL (wikitext) -> RAG-ready JSONL (sections + plain text)."""

    HEADING_RE = re.compile(r"(?m)^(={2,})\s*(.+?)\s*\1\s*$")

    def __init__(self, input_path: str, output_path: str, language: str = "en") -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.language = language

    def _wikitext_to_text(self, wikitext: str) -> str:
        if not wikitext:
            return ""
        if mwparserfromhell:
            try:
                return mwparserfromhell.parse(wikitext).strip_code().strip()
            except Exception:
                pass
        text = re.sub(r"\{\{.*?\}\}", " ", wikitext, flags=re.S)
        text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
        text = re.sub(r"''+", "", text)
        text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _split_sections(self, wikitext: str) -> Iterator[tuple[Optional[str], str]]:
        if not wikitext:
            yield None, ""
            return
        matches = list(self.HEADING_RE.finditer(wikitext))
        if not matches:
            yield None, wikitext
            return
        first = matches[0]
        lead = wikitext[: first.start()]
        yield None, lead
        for i, m in enumerate(matches):
            title = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)
            body = wikitext[start:end]
            yield title, body

    def run(self) -> None:
        with open(self.input_path, "r", encoding="utf-8") as inf, open(
            self.output_path, "w", encoding="utf-8"
        ) as outf:
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("is_redirect"):
                    continue
                page_id = rec.get("id")
                page_title = rec.get("title")
                wikitext = rec.get("wikitext", "") or ""

                for section_title, section_wikitext in self._split_sections(wikitext):
                    text = self._wikitext_to_text(section_wikitext)
                    if not text:
                        continue
                    anchor = (
                        re.sub(r"[^\w\-]", "_", section_title).strip("_")
                        if section_title
                        else ""
                    )
                    out = {
                        "page_id": page_id,
                        "page_title": page_title,
                        "section_title": section_title,
                        "section_anchor": anchor,
                        "text": text,
                        "url": rec.get("url"),
                        "source": rec.get("source", "enwikivoyage"),
                        "language": self.language,
                    }
                    outf.write(json.dumps(out, ensure_ascii=False) + "\n")
