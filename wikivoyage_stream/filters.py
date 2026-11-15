"""Page filters and URL helpers shared by the text and image pipelines."""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import quote

from .models import Page

SITE_BASE = "https://en.wikivoyage.org/wiki/"

_REDIRECT_LINK_RE = re.compile(r"\[\[([^\]|]+)")


def is_redirect(wikitext: str) -> tuple[bool, Optional[str]]:
    """Return (is_redirect, target) judged from the first non-blank line of wikitext."""
    first = next((l.strip() for l in wikitext.splitlines() if l.strip()), "")
    if first.upper().startswith("#REDIRECT"):
        m = _REDIRECT_LINK_RE.search(first)
        return True, (m.group(1).strip() if m else None)
    return False, None


def page_url(title: str, site_base: str = SITE_BASE) -> str:
    """Article URL for a page title, e.g. "New York City" → .../wiki/New_York_City."""
    return site_base + quote((title or "").replace(" ", "_"), safe="_()',-")


def is_article(page: Page) -> bool:
    """True for a main-namespace (ns=0), non-redirect page with a non-blank body."""
    if page.namespace != "0":
        return False
    if page.redirect_title:
        return False
    text = page.text or ""
    if not text.strip():
        return False
    return not is_redirect(text)[0]
