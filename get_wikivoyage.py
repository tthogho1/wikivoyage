"""CLI: stream Wikivoyage dump and write *city pages* as JSONL or Parquet.

  jsonl    1 file per page: <outdir>/<slug>.jsonl (one JSON line each)
  parquet  batched part files: <outdir>/part-00000.parquet, part-00001.parquet, ...

City-focused schema (one JSON line / Parquet row per page):

  {
    "page_id": int,
    "title": str,
    "slug": str,                    # filename stem (spaces -> "_")
    "namespace": int,
    "page_type": "city" | "region" | "country" | "district" | "park" | "other",
    "status": "stub|outline|usable|guide|star" | null,
    "is_redirect": bool,
    "redirect_target": str | null,
    "source": "enwikivoyage",
    "url": str,
    "retrieved_at": str,            # ISO 8601 (UTC)
    "revision_id": int | null,
    "timestamp": str | null,        # last-revision timestamp from dump
    "wikitext": str,                # original wikitext
    "clean_text": str,              # plain text; listings rendered as one line each
    "sections": [ { "title": str, "path": str,  # path: "Sleep > Budget"
                    "anchor": str, "text": str }, ... ],
    "retrieval_text": str,
    "is_part_of": str | null        # parent region from {{IsPartOf|...}}
  }

Command-line options (each overrides the matching environment variable):
  --format {jsonl,parquet}   WIKIVOYAGE_FORMAT     (default: jsonl)
  --s3-bucket BUCKET         WIKIVOYAGE_S3_BUCKET  "bucket" or "s3://bucket[/prefix]";
                                                   when set, output goes to S3.
  --outdir DIR               WIKIVOYAGE_OUTDIR     local dir, or S3 key prefix
                                                   (default: pages)

Environment variables:
  WIKIVOYAGE_DUMP        URL or local path to the .bz2 dump.
  WIKIVOYAGE_LIMIT       Max *city* pages to write (0 = unlimited, default 200).
  WIKIVOYAGE_PAGE_TYPES  Comma-separated allowed page types
                         (default: "city"; use "city,region,country,district"
                         to broaden or "*" to disable the filter).
  WIKIVOYAGE_PARQUET_ROWS_PER_FILE  Rows per Parquet part file (default 10000).
  WIKIVOYAGE_SITE        Base wiki URL (default: https://en.wikivoyage.org/wiki/).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import tempfile
from typing import Optional
from urllib.parse import quote

from wikivoyage_stream.reader import WikivoyageDumpStreamReader
from wikivoyage_stream.models import Page

try:
    import mwparserfromhell  # optional, nicer wikitext -> plain text
except Exception:
    mwparserfromhell = None

try:
    import boto3
except Exception:
    boto3 = None

logger = logging.getLogger("wikivoyage.stream")

DUMP_URL = (
    "https://dumps.wikimedia.org/enwikivoyage/latest/"
    "enwikivoyage-latest-pages-articles.xml.bz2"
)
SITE_BASE = "https://en.wikivoyage.org/wiki/"

_UNSAFE_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_HEADING_RE = re.compile(r"(?m)^(={2,})\s*(.+?)\s*\1\s*$")

# Wikivoyage status templates: {{outlinecity}}, {{usablecity}}, ...
# Breadcrumb parent: {{IsPartOf|North Rhine-Westphalia}} (older pages use {{IsIn|...}})
_IS_PART_OF_RE = re.compile(r"\{\{\s*(?:IsPartOf|IsIn)\s*\|\s*([^|}]+)", re.IGNORECASE)

_STATUS_RE = re.compile(
    r"\{\{\s*(stub|outline|usable|guide|star)"
    r"(city|region|country|district|park|airport|continent|huge ?city|small ?city)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_filename(title: str) -> str:
    name = title.strip().replace(" ", "_")
    name = _UNSAFE_RE.sub("_", name)
    return name[:200] or "_unnamed_"


def _is_redirect(wikitext: str) -> tuple[bool, Optional[str]]:
    first = next((l.strip() for l in wikitext.splitlines() if l.strip()), "")
    if first.upper().startswith("#REDIRECT"):
        m = re.search(r"\[\[([^\]|]+)", first)
        return True, (m.group(1).strip() if m else None)
    return False, None


def extract_is_part_of(wikitext: str) -> Optional[str]:
    """Return the parent region named by {{IsPartOf|...}}, or None."""
    m = _IS_PART_OF_RE.search(wikitext or "")
    if not m:
        return None
    parent = m.group(1).replace("_", " ").strip()
    return parent or None


def detect_page_type_and_status(wikitext: str) -> tuple[str, Optional[str]]:
    """Classify a Wikivoyage page using its status template, e.g. {{usablecity}}.

    Returns (page_type, status). page_type is "other" when nothing matches.
    """
    m = _STATUS_RE.search(wikitext or "")
    if not m:
        return "other", None
    status = m.group(1).lower()
    kind = re.sub(r"\s+", "", m.group(2).lower())
    if kind in {"city", "hugecity", "smallcity"}:
        ptype = "city"
    elif kind == "region":
        ptype = "region"
    elif kind == "country":
        ptype = "country"
    elif kind == "district":
        ptype = "district"
    elif kind in {"park", "airport", "continent"}:
        ptype = kind
    else:
        ptype = "other"
    return ptype, status


# Wikivoyage listing templates: {{eat | name=... | address=... | content=...}}
_LISTING_TYPES = {"see", "do", "buy", "eat", "drink", "sleep", "listing", "go"}
_LISTING_LABELLED = (
    ("phone", "Phone"),
    ("hours", "Hours"),
    ("checkin", "Check-in"),
    ("checkout", "Check-out"),
    ("price", "Price"),
)


def _tpl_param(tpl, key: str) -> str:
    if not tpl.has(key):
        return ""
    return re.sub(r"\s+", " ", _render_wikicode(tpl.get(key).value)).strip()


def _render_listing(tpl) -> str:
    """Render a listing template as one plain-text line.

    e.g. "Sushi Dai (alt), Tsukiji (directions). Hours: 5-14. Price: ¥4000. content"
    """
    head = _tpl_param(tpl, "name")
    alt = _tpl_param(tpl, "alt")
    if alt:
        head = f"{head} ({alt})" if head else alt
    address = _tpl_param(tpl, "address")
    directions = _tpl_param(tpl, "directions")
    if directions:
        address = f"{address} ({directions})" if address else directions
    if address:
        head = f"{head}, {address}" if head else address
    parts = [head]
    parts += [f"{label}: {v}" for key, label in _LISTING_LABELLED if (v := _tpl_param(tpl, key))]
    parts.append(_tpl_param(tpl, "content"))
    parts = [p.rstrip(" .") for p in parts if p and p.strip(" .")]
    return ". ".join(parts) + "." if parts else ""


def _render_wikicode(code) -> str:
    """strip_code(), but keep listings as text and {{marker}} as its name.

    Plain strip_code() drops every template, which removes all of a page's
    see/eat/sleep/... listings. Other templates are still dropped, as are
    [[File:...]] / [[Image:...]] / [[Category:...]] links (strip_code would
    otherwise leave "thumb|caption" text behind).
    """
    for link in code.filter_wikilinks(recursive=False):
        target = str(link.title).strip().lower()
        if target.startswith(("file:", "image:", "category:")):
            code.remove(link)
    for tpl in code.filter_templates(recursive=False):
        name = str(tpl.name).strip().lower()
        if name in _LISTING_TYPES:
            rendered = _render_listing(tpl)
        elif name == "marker":
            rendered = _tpl_param(tpl, "name")
        else:
            rendered = ""
        if rendered:
            code.replace(tpl, rendered)
        else:
            code.remove(tpl)
    return code.strip_code()


def wikitext_to_plain(wikitext: str) -> str:
    if not wikitext:
        return ""
    if mwparserfromhell is not None:
        try:
            text = _render_wikicode(mwparserfromhell.parse(wikitext))
            # Keep line breaks (one listing per line); collapse gaps left by removed templates.
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r" *\n *", "\n", text)
            return re.sub(r"\n{3,}", "\n\n", text).strip()
        except Exception:
            pass
    # Fallback without mwparserfromhell: listings are dropped with other templates.
    text = re.sub(r"\{\{.*?\}\}", " ", wikitext, flags=re.S)
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"''+", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sections(wikitext: str) -> list[dict]:
    """Split wikitext into sections by ==headings== and clean each body.

    Every heading level gets its own section; `path` joins the enclosing
    headings ("Sleep > Budget") since titles like "Budget" repeat under
    several parents.
    """
    if not wikitext:
        return []
    matches = list(_HEADING_RE.finditer(wikitext))
    sections: list[dict] = []

    def _push(title: Optional[str], path: str, body: str) -> None:
        text = wikitext_to_plain(body)
        if not text:
            return
        anchor = re.sub(r"[^\w\-]", "_", title).strip("_") if title else ""
        sections.append({"title": title or "", "path": path, "anchor": anchor, "text": text})

    if not matches:
        _push(None, "", wikitext)
        return sections

    _push(None, "", wikitext[: matches[0].start()])  # lead
    stack: list[tuple[int, str]] = []  # (heading level, title) of enclosing headings
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)
        _push(title, " > ".join(t for _, t in stack), wikitext[start:end])
    return sections


# Mapping: canonical section label -> set of lower-cased wikitext headings that match.
_SECTION_LABEL_MAP: list[tuple[str, set[str]]] = [
    ("Transportation", {"get in", "get around", "by bus", "by car", "by bus or car",
                        "by train", "by plane", "by boat", "by foot", "by taxi",
                        "by bike", "by bicycle"}),
    ("Attractions",    {"see", "do", "attractions", "sights", "sightseeing"}),
    ("Shopping",       {"buy", "shop", "shopping", "markets", "souvenirs"}),
    ("Food and Drink", {"eat", "drink", "food", "restaurants", "dining",
                        "food and drink", "eat and drink"}),
    ("Accommodation",  {"sleep", "hotels", "lodging", "accommodation",
                        "where to stay"}),
    ("Health and Safety", {"stay healthy", "stay safe", "health", "safety", "stay",
                           "stay healthy and safe", "medical"}),
    ("Communication",  {"connect", "communication", "internet", "phone"}),
    ("Nearby",         {"go next", "nearby", "around", "surrounding", "see also",
                        "next destinations"}),
]


def _label_for(heading: str) -> Optional[str]:
    h = heading.strip().lower()
    for label, keywords in _SECTION_LABEL_MAP:
        if h in keywords or any(h.startswith(k) for k in keywords):
            return label
    return None


_JUNK_RE = re.compile(
    r"\{\{[^}]*\}\}"          # {{template}}
    r"|\[\[(?:[^|\]]*\|)?([^\]]+)\]\]"  # [[link]] or [[link|text]] → keep text via sub below
    r"|File:[^\n]*"
    r"|thumb\|[^\n]*"
    r"|<[^>]+>"
    r"|<ref[^>]*>.*?</ref>",
    re.S | re.I,
)


def _clean_for_retrieval(text: str) -> str:
    """Aggressive clean: remove all wikitext markup, collapse whitespace."""
    if mwparserfromhell is not None:
        try:
            wcode = mwparserfromhell.parse(text)
            # Remove templates wholesale (marker, eat, sleep, see, …) before strip_code
            for tpl in wcode.filter_templates():
                wcode.remove(tpl)
            text = wcode.strip_code()
        except Exception:
            pass
    else:
        # fallback regex path: multi-pass nested {{...}} removal
        for _ in range(4):
            text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
        text = _JUNK_RE.sub(" ", text)
        text = re.sub(r"''+", "", text)

    text = re.sub(r"^[*#:;]+", "", text, flags=re.M)               # list bullets
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)   # any leftover links
    text = re.sub(r"File:[^\n]*", "", text, flags=re.I)
    text = re.sub(r"thumb\|[^\n]*", "", text, flags=re.I)
    # artifact cleanup: stray commas/colons left when templates were stripped
    text = re.sub(r"\bat\s*\.", ".", text)                   # "stopping at."
    text = re.sub(r"\bthe\s*,\s*", "the ", text)            # "the , located"
    text = re.sub(r",\s*located\b", " located", text)
    text = re.sub(r":\s*,", ":", text)
    text = re.sub(r"\s*,\s*\.", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    # drop short sentences that carry no real information (< 25 chars)
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 25]
    text = ". ".join(sentences)
    if text and not text.endswith("."):
        text += "."
    return text


def build_retrieval_text(
    title: str,
    page_type: str,
    wikitext: str,
    max_chars: int = 1500,
) -> str:
    """Build a labelled, embedding-ready text directly from raw wikitext."""
    parts: list[str] = [
        f"Title: {title}",
        f"Type: {page_type}",
    ]

    # Split wikitext into (heading, body) pairs the same way as split_sections.
    matches = list(_HEADING_RE.finditer(wikitext))
    raw_sections: list[tuple[Optional[str], str]] = []
    if not matches:
        raw_sections.append((None, wikitext))
    else:
        raw_sections.append((None, wikitext[: matches[0].start()]))
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)
            raw_sections.append((m.group(2).strip(), wikitext[start:end]))

    grouped: dict[str, list[str]] = {}
    for heading, body in raw_sections:
        cleaned = _clean_for_retrieval(body)
        if not cleaned:
            continue
        if not heading:
            label = "Overview"
        else:
            label = _label_for(heading) or heading.title()
        grouped.setdefault(label, []).append(cleaned)

    for label, texts in grouped.items():
        combined = " ".join(texts).strip()
        if combined:
            parts.append(f"{label}: {combined}")

    result = "\n".join(parts)
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0] + " ..."
    return result


def _unique_path(outdir: str, stem: str, used: dict[str, int], s3_mode: bool = False) -> str:
    """Return <outdir>/<stem>.jsonl, adding _2, _3 ... if needed (in-run dedup).

    When `s3_mode` is True we only perform in-run deduplication and do not
    check the local filesystem for existing files (S3 checks would be costly).
    """
    base = stem
    n = used.get(base, 0) + 1
    used[base] = n
    candidate_stem = base if n == 1 else f"{base}_{n}"
    path = os.path.join(outdir, candidate_stem + ".jsonl")
    if not s3_mode:
        # Also avoid colliding with existing files on disk
        while os.path.exists(path):
            n += 1
            used[base] = n
            candidate_stem = f"{base}_{n}"
            path = os.path.join(outdir, candidate_stem + ".jsonl")
    return path


def build_record(page: Page, page_type: str, status: Optional[str], slug: str) -> dict:
    is_redir, redir_target = _is_redirect(page.text or "")
    pid = (
        int(page.page_id)
        if page.page_id and str(page.page_id).isdigit()
        else page.page_id
    )
    nsv = (
        int(page.namespace)
        if page.namespace and str(page.namespace).isdigit()
        else page.namespace
    )
    rev = (
        int(page.revision_id)
        if page.revision_id and str(page.revision_id).isdigit()
        else page.revision_id
    )
    url = SITE_BASE + quote((page.title or "").replace(" ", "_"), safe="_()',-")
    retrieved_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    sections = split_sections(page.text or "")

    return {
        "page_id": pid,
        "title": page.title,
        "slug": slug,
        "namespace": nsv,
        "page_type": page_type,
        "status": status,
        "is_redirect": is_redir,
        "redirect_target": redir_target,
        "source": "enwikivoyage",
        "url": url,
        "retrieved_at": retrieved_at,
        "revision_id": rev,
        "timestamp": page.timestamp,
        "wikitext": page.text,
        "clean_text": wikitext_to_plain(page.text or ""),
        "sections": sections,
        "retrieval_text": build_retrieval_text(page.title or "", page_type, page.text or ""),
        "is_part_of": extract_is_part_of(page.text or ""),
    }


# ---------------------------------------------------------------------------
# Parquet output
# ---------------------------------------------------------------------------
def _parquet_schema(pa):
    """Explicit schema so all-null columns in a batch don't get type `null`."""
    section = pa.struct([
        ("title", pa.string()),
        ("path", pa.string()),
        ("anchor", pa.string()),
        ("text", pa.string()),
    ])
    return pa.schema([
        ("page_id", pa.int64()),
        ("title", pa.string()),
        ("slug", pa.string()),
        ("namespace", pa.int64()),
        ("page_type", pa.string()),
        ("status", pa.string()),
        ("is_redirect", pa.bool_()),
        ("redirect_target", pa.string()),
        ("source", pa.string()),
        ("url", pa.string()),
        ("retrieved_at", pa.string()),
        ("revision_id", pa.int64()),
        ("timestamp", pa.string()),
        ("wikitext", pa.string()),
        ("clean_text", pa.string()),
        ("sections", pa.list_(section)),
        ("retrieval_text", pa.string()),
        ("is_part_of", pa.string()),
    ])


class ParquetSink:
    """Write records to rolling part-NNNNN.parquet files, locally or to S3.

    Records are buffered into row groups of `row_group_size`; a new part file is
    started every `rows_per_file` rows. In S3 mode each part is written to a
    temp file and uploaded when it is closed.
    """

    def __init__(
        self,
        outdir: str,
        s3_client=None,
        s3_bucket: str = "",
        s3_prefix: str = "",
        rows_per_file: int = 10000,
        row_group_size: int = 500,
    ) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as e:  # pragma: no cover - dependency/runtime
            raise RuntimeError("pyarrow is required for --format parquet") from e
        self._pa, self._pq = pa, pq
        self._schema = _parquet_schema(pa)
        self.outdir = outdir
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.rows_per_file = max(1, rows_per_file)
        self.row_group_size = max(1, row_group_size)
        self._buf: list[dict] = []
        self._writer = None
        self._path: Optional[str] = None
        self._rows_in_file = 0
        self._part = 0
        self.files_written: list[str] = []

    def _open(self) -> None:
        name = f"part-{self._part:05d}.parquet"
        self._part += 1
        if self.s3_client is not None:
            fd, self._path = tempfile.mkstemp(suffix=".parquet")
            os.close(fd)
            self._dest = f"s3://{self.s3_bucket}/{self.s3_prefix}{name}"
        else:
            self._path = os.path.join(self.outdir, name)
            self._dest = self._path
        self._writer = self._pq.ParquetWriter(self._path, self._schema, compression="zstd")
        self._rows_in_file = 0

    def _flush(self) -> None:
        if not self._buf:
            return
        if self._writer is None:
            self._open()
        table = self._pa.Table.from_pylist(self._buf, schema=self._schema)
        self._writer.write_table(table)
        self._rows_in_file += len(self._buf)
        self._buf.clear()

    def _close_file(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        if self.s3_client is not None:
            key = self._dest.split("/", 3)[3]
            try:
                self.s3_client.upload_file(self._path, self.s3_bucket, key)
            finally:
                os.remove(self._path)
        logger.info("wrote %s (%d rows)", self._dest, self._rows_in_file)
        self.files_written.append(self._dest)

    def add(self, record: dict) -> None:
        self._buf.append(record)
        room = self.rows_per_file - self._rows_in_file
        if len(self._buf) >= min(self.row_group_size, room):
            self._flush()
            if self._rows_in_file >= self.rows_per_file:
                self._close_file()

    def close(self) -> None:
        self._flush()
        self._close_file()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_s3_bucket(value: str) -> tuple[str, Optional[str]]:
    """Split "bucket" or "s3://bucket[/prefix]" into (bucket, prefix | None)."""
    value = value.strip()
    if value.startswith("s3://"):
        value = value[5:]
    if "/" in value:
        bucket, prefix = value.split("/", 1)
        return bucket, prefix.strip("/")
    return value, None


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream the Wikivoyage dump and write city pages as JSONL or Parquet.",
    )
    p.add_argument(
        "--format",
        choices=("jsonl", "parquet"),
        default=os.environ.get("WIKIVOYAGE_FORMAT", "jsonl").strip().lower() or "jsonl",
        help="output format (env WIKIVOYAGE_FORMAT, default: jsonl)",
    )
    p.add_argument(
        "--s3-bucket",
        default=os.environ.get("WIKIVOYAGE_S3_BUCKET", "").strip(),
        help='output bucket, "bucket" or "s3://bucket[/prefix]"; '
             "writes to S3 when set (env WIKIVOYAGE_S3_BUCKET)",
    )
    p.add_argument(
        "--outdir",
        default=os.environ.get("WIKIVOYAGE_OUTDIR", "pages").strip(),
        help="local output dir, or S3 key prefix in S3 mode "
             "(env WIKIVOYAGE_OUTDIR, default: pages)",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    if args.format not in ("jsonl", "parquet"):
        raise SystemExit(f"invalid format {args.format!r} (expected jsonl or parquet)")

    source = os.environ.get("WIKIVOYAGE_DUMP", DUMP_URL)
    limit = int(os.environ.get("WIKIVOYAGE_LIMIT", "200"))
    rows_per_file = int(os.environ.get("WIKIVOYAGE_PARQUET_ROWS_PER_FILE", "10000"))
    outdir_env = args.outdir
    types_env = os.environ.get("WIKIVOYAGE_PAGE_TYPES", "city").strip()
    allowed_types: Optional[set[str]] = (
        None if types_env == "*" else {t.strip() for t in types_env.split(",") if t.strip()}
    )

    # S3 configuration: if --s3-bucket / WIKIVOYAGE_S3_BUCKET is set, write to S3.
    # A prefix given in "s3://bucket/prefix" takes precedence over --outdir.
    s3_bucket, s3_url_prefix = _parse_s3_bucket(args.s3_bucket)
    s3_prefix = (s3_url_prefix if s3_url_prefix is not None else outdir_env).strip("/")
    if s3_prefix:
        s3_prefix = s3_prefix + "/"

    s3_mode = False
    s3_client = None
    if s3_bucket:
        if boto3 is None:
            raise RuntimeError("boto3 is required to write to S3 (set WIKIVOYAGE_S3_BUCKET)")
        s3_mode = True
        s3_client = boto3.client("s3")

    # Local output dir only needed when not using S3
    if not s3_mode:
        outdir = outdir_env or "pages"
        os.makedirs(outdir, exist_ok=True)
    else:
        # when using S3, we'll use the prefix (may be empty)
        outdir = s3_prefix

    sink: Optional[ParquetSink] = None
    if args.format == "parquet":
        sink = ParquetSink(
            outdir,
            s3_client=s3_client,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            rows_per_file=rows_per_file,
        )

    reader = WikivoyageDumpStreamReader(source)
    used_stems: dict[str, int] = {}
    written = 0
    skipped_ns = 0
    skipped_redir = 0
    skipped_type = 0

    for page in reader.iter_pages():
        # only articles (ns=0)
        if page.namespace != "0":
            skipped_ns += 1
            continue

        is_redir, _ = _is_redirect(page.text or "")
        if is_redir:
            skipped_redir += 1
            continue

        page_type, status = detect_page_type_and_status(page.text or "")
        if allowed_types is not None and page_type not in allowed_types:
            skipped_type += 1
            continue

        title = page.title or "_unnamed_"
        slug = _safe_filename(title)
        # Parquet doesn't write per-page files, so only dedup slugs within the run.
        outpath = _unique_path(outdir, slug, used_stems, s3_mode=s3_mode or sink is not None)
        # Update slug if path was disambiguated
        final_stem = os.path.splitext(os.path.basename(outpath))[0]
        record = build_record(page, page_type, status, final_stem)

        if sink is not None:
            sink.add(record)
        elif s3_mode:
            key = (s3_prefix or "") + os.path.basename(outpath)
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=key,
                Body=(json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"),
            )
        else:
            with open(outpath, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        written += 1
        if sink is None:
            logger.info("wrote %s (type=%s)", os.path.basename(outpath), page_type)
        if limit > 0 and written >= limit:
            break

    if sink is not None:
        sink.close()

    dest = f"s3://{s3_bucket}/{s3_prefix}" if s3_mode else f"{outdir}/"
    what = (
        f"{written} pages in {len(sink.files_written)} Parquet file(s)"
        if sink is not None
        else f"{written} city-page JSONL files"
    )
    print(
        f"Wrote {what} → {dest} "
        f"(seen={reader.page_count}, skipped_ns={skipped_ns}, "
        f"skipped_redirect={skipped_redir}, skipped_type={skipped_type}, "
        f"compressed={reader.compressed_bytes} B, "
        f"decompressed={reader.decompressed_bytes} B)."
    )


if __name__ == "__main__":
    main()