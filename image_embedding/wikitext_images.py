"""Extract every image referenced by an article's wikitext.

Sources handled:
- links       ``[[File:X.jpg|thumb|Caption]]`` / ``[[Image:...]]`` (not ``[[:File:...]]``, not ``[[Media:...]]``)
- galleries   ``<gallery>`` lines, ``File:`` prefix optional
- templates   any template argument whose value looks like an image file name
              (``{{pagebanner|Banner.jpg}}``, listings' ``image=``, infoboxes, ...)
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import unquote

logger = logging.getLogger("wikivoyage.images.extract")

try:
    import mwparserfromhell
except Exception:  # pragma: no cover - dependency missing
    mwparserfromhell = None

MAX_CAPTION = 1000
CONTEXT_CHARS = 200
MAX_NAME_BYTES = 240

FILE_NAMESPACES = {"file", "image"}
LINK_OPTIONS = {
    "thumb", "thumbnail", "frame", "framed", "frameless", "border", "left", "right", "center",
    "centre", "none", "upright", "baseline", "middle", "sub", "super", "top", "text-top",
    "bottom", "text-bottom",
}
LINK_OPTION_KEYS = ("upright=", "alt=", "link=", "page=", "class=", "lang=", "thumb=", "thumbnail=")
_SIZE_RE = re.compile(r"^\d*\s*x?\s*\d+\s*px$", re.IGNORECASE)
_IMAGE_VALUE_RE = re.compile(r"^[^\[\]{}|#<>]+\.(?:jpe?g|png|gif|svg|webp|tiff?)$", re.IGNORECASE)
_PREFIXED_VALUE_RE = re.compile(r"^\s*(?:file|image)\s*:[^\[\]{}|#<>]+$", re.IGNORECASE)
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")
_INVALID_TITLE_CHARS = re.compile(r"[\[\]{}|<>]")
_LISTING_TEMPLATES = {"see", "do", "buy", "eat", "drink", "sleep", "go", "listing", "marker", "vcard"}
_FALLBACK_LINK_RE = re.compile(r"\[\[\s*(?:File|Image)\s*:([^|\]]+)", re.IGNORECASE)
_FALLBACK_TEMPLATE_VALUE_RE = re.compile(
    r"[|=]\s*([^\[\]{}|#<>=\n]+\.(?:jpe?g|png|gif|svg|webp|tiff?))\s*(?=[|}])", re.IGNORECASE
)


@dataclass
class ImageRef:
    file_title: str
    position: int
    caption: Optional[str]
    context: str
    source_kind: str  # link | gallery | template
    template_name: Optional[str] = None
    param_name: Optional[str] = None
    offset: int = field(default=0, repr=False)


def normalize_file_title(raw: str) -> Optional[str]:
    """Normalise a reference to ``File:<Name>``; None when it is not a usable file title."""
    if raw is None:
        return None
    s = html.unescape(str(raw))
    if "{{" in s or "}}" in s:
        return None  # needs template expansion
    if "%" in s:
        s = unquote(s)
    s = s.split("#", 1)[0]
    s = _CTRL_RE.sub(" ", s).replace("_", " ")
    s = _WS_RE.sub(" ", s).strip()
    if ":" in s:
        prefix, rest = s.split(":", 1)
        if prefix.strip().lower() in FILE_NAMESPACES:
            s = rest.strip()
    if not s or _INVALID_TITLE_CHARS.search(s):
        return None
    if len(s.encode("utf-8")) > MAX_NAME_BYTES:
        return None
    return "File:" + s[0].upper() + s[1:]


def _split_top_level(s: str, sep: str = "|") -> list[str]:
    """Split on ``sep`` outside of [[...]] and {{...}}."""
    parts, buf, depth_link, depth_tpl, i = [], [], 0, 0, 0
    while i < len(s):
        two = s[i:i + 2]
        if two == "[[":
            depth_link += 1; buf.append(two); i += 2; continue
        if two == "]]" and depth_link:
            depth_link -= 1; buf.append(two); i += 2; continue
        if two == "{{":
            depth_tpl += 1; buf.append(two); i += 2; continue
        if two == "}}" and depth_tpl:
            depth_tpl -= 1; buf.append(two); i += 2; continue
        if s[i] == sep and not depth_link and not depth_tpl:
            parts.append("".join(buf)); buf = []; i += 1; continue
        buf.append(s[i]); i += 1
    parts.append("".join(buf))
    return parts


def _plain(text: str) -> Optional[str]:
    text = text or ""
    if mwparserfromhell is not None:
        try:
            text = mwparserfromhell.parse(text).strip_code()
        except Exception:
            pass
    text = re.sub(r"<[^>]+>", "", text)
    text = _WS_RE.sub(" ", html.unescape(text)).strip()
    return text[:MAX_CAPTION] or None


def _is_option(arg: str) -> bool:
    a = arg.strip().lower()
    return a in LINK_OPTIONS or bool(_SIZE_RE.match(a)) or a.startswith(LINK_OPTION_KEYS)


def caption_from_args(args: list[str]) -> Optional[str]:
    """Last non-option argument as plain text; falls back to ``alt=``."""
    alt: Optional[str] = None
    for a in args:
        if a.strip().lower().startswith("alt="):
            alt = a.split("=", 1)[1]
    for a in reversed(args):
        if a.strip() and not _is_option(a):
            return _plain(a)
    return _plain(alt) if alt else None


@dataclass
class _Candidate:
    offset: int
    raw: str
    caption: Optional[str]
    kind: str
    template_name: Optional[str] = None
    param_name: Optional[str] = None


def _locate(text: str, needle: str, start: int) -> int:
    pos = text.find(needle, start)
    return pos if pos >= 0 else len(text)


def _candidates_mwparser(text: str) -> list[_Candidate]:
    code = mwparserfromhell.parse(text)
    out: list[_Candidate] = []

    cursor = 0
    for link in code.filter_wikilinks(recursive=True):
        title = str(link.title).strip()
        off = _locate(text, str(link), cursor)
        if off < len(text):
            cursor = off + 1
        if title.startswith(":") or ":" not in title:
            continue
        ns = title.split(":", 1)[0].strip().lower()
        if ns not in FILE_NAMESPACES:
            continue
        args = _split_top_level(str(link.text)) if link.text is not None else []
        out.append(_Candidate(off, title, caption_from_args(args), "link"))

    cursor = 0
    for tag in code.filter_tags(recursive=True, matches=lambda n: str(n.tag).strip().lower() == "gallery"):
        off = _locate(text, str(tag), cursor)
        if off < len(text):
            cursor = off + 1
        contents = str(tag.contents) if tag.contents is not None else ""
        line_cursor = off
        for line in contents.splitlines():
            line = line.strip()
            if not line:
                continue
            line_off = _locate(text, line, line_cursor)
            if line_off < len(text):
                line_cursor = line_off + 1
            parts = _split_top_level(line)
            out.append(_Candidate(line_off, parts[0], caption_from_args(parts[1:]), "gallery"))

    cursor = 0
    for tpl in code.filter_templates(recursive=True):
        tpl_str = str(tpl)
        off = _locate(text, tpl_str, cursor)
        if off < len(text):
            cursor = off + 1
        name = str(tpl.name).strip().lower().replace("_", " ")
        params = {str(p.name).strip().lower(): p for p in tpl.params}
        for p in tpl.params:
            value = str(p.value).strip()
            if not value or not (_IMAGE_VALUE_RE.match(value) or _PREFIXED_VALUE_RE.match(value)):
                continue
            if name in _LISTING_TEMPLATES and "name" in params:
                caption = _plain(str(params["name"].value))
            elif name == "pagebanner":
                caption = None
            elif "caption" in params:
                caption = _plain(str(params["caption"].value))
            else:
                caption = None
            v_off = _locate(text, value, off) if off < len(text) else len(text)
            out.append(_Candidate(v_off, value, caption, "template", name, str(p.name).strip()))
    return out


def _candidates_regex(text: str) -> list[_Candidate]:
    out = [_Candidate(m.start(), "File:" + m.group(1), None, "link") for m in _FALLBACK_LINK_RE.finditer(text)]
    out += [
        _Candidate(m.start(1), m.group(1).strip(), None, "template")
        for m in _FALLBACK_TEMPLATE_VALUE_RE.finditer(text)
    ]
    return out


def extract_refs(
    wikitext: str,
    *,
    max_per_page: int = 50,
    errors: Optional[list[dict]] = None,
) -> tuple[list[ImageRef], int]:
    """Return (refs in document order, de-duplicated and capped; number truncated by the cap).

    Unusable references are appended to ``errors`` as ``{"raw", "reason"}`` and skipped.
    """
    text = wikitext or ""
    if not text.strip():
        return [], 0
    try:
        if mwparserfromhell is None:
            raise RuntimeError("mwparserfromhell not installed")
        cands = _candidates_mwparser(text)
    except Exception as e:
        logger.warning("wikitext parse failed (%s); using regex fallback", e)
        cands = _candidates_regex(text)

    cands.sort(key=lambda c: c.offset)
    by_title: dict[str, ImageRef] = {}
    for c in cands:
        title = normalize_file_title(c.raw)
        if title is None:
            if errors is not None:
                errors.append({"raw": c.raw[:500], "reason": "unparsable_title", "kind": c.kind})
            continue
        existing = by_title.get(title)
        if existing is not None:
            if existing.caption is None and c.caption:
                existing.caption = c.caption
            continue
        start = max(0, c.offset - CONTEXT_CHARS)
        by_title[title] = ImageRef(
            file_title=title,
            position=0,
            caption=c.caption,
            context=text[start:c.offset + CONTEXT_CHARS],
            source_kind=c.kind,
            template_name=c.template_name,
            param_name=c.param_name,
            offset=c.offset,
        )

    refs = list(by_title.values())
    truncated = max(0, len(refs) - max_per_page) if max_per_page > 0 else 0
    if truncated:
        refs = refs[:max_per_page]
    for i, r in enumerate(refs, 1):
        r.position = i
    return refs, truncated
