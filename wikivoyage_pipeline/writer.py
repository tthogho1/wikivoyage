"""JSONL writer for Wikivoyage Page objects."""
from __future__ import annotations

import json
import re
from typing import Iterable, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from get_wikivoyage import Page


class WikivoyageJsonlWriter:
    """Write Page objects to a raw JSONL file (one page per line)."""

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def _is_redirect(self, wikitext: str) -> tuple[bool, Optional[str]]:
        if not wikitext:
            return False, None
        first = next((ln for ln in (l.strip() for l in wikitext.splitlines()) if ln), "")
        if not first:
            return False, None
        if first.upper().startswith("#REDIRECT"):
            m = re.search(r"\[\[([^\]\|]+)", first)
            return True, (m.group(1).strip() if m else None)
        return False, None

    def write_pages(self, pages: Iterable[Page]) -> None:
        with open(self.output_path, "w", encoding="utf-8") as fh:
            for p in pages:
                ns = None
                try:
                    ns = int(p.namespace) if p.namespace is not None else None
                except Exception:
                    pass
                if ns is not None and ns != 0:
                    continue

                is_redir, redir_target = self._is_redirect(p.text or "")
                rec = {
                    "id": int(p.page_id) if p.page_id and str(p.page_id).isdigit() else p.page_id,
                    "title": p.title,
                    "namespace": ns if ns is not None else p.namespace,
                    "is_redirect": is_redir,
                    "redirect_target": redir_target,
                    "wikitext": p.text,
                    "timestamp": None,
                    "url": None,
                    "source": "enwikivoyage",
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
