"""Build RAG chunks from the stage-1 Parquet and embed them locally.

Reads the page Parquet written by `get_wikivoyage.py --format parquet`, splits each
page's sections into chunks exactly like the old databricks/build_chunks.py did (same
chunk_ids and chunk_text), embeds chunk_text with Qwen3-Embedding-0.6B (MPS when
available), and writes part-NNNNN.parquet files for databricks/build_index.py:

  chunk_id, page_id, title, page_type, status, is_part_of, section, url,
  chunk_text, embedding (list<float32>, normalized)

Resumable: parts are written to --workdir first (atomically) and uploaded to --dest;
a rerun skips parts that are already done. Documents are embedded without a prompt;
queries must use the model's "query" prompt (see build_index.py).

  pip install sentence-transformers torch pyarrow boto3
  caffeinate -i python scripts/embed_chunks.py            # full run (~2 days on an M4)
  python scripts/embed_chunks.py --write-check            # endpoint comparison file only
  python scripts/embed_chunks.py --max-parts 1 --rows-per-part 64 --dest ./out   # quick test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("embed_chunks")

OUT_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("page_id", pa.int64()),
    ("title", pa.string()),
    ("page_type", pa.string()),
    ("status", pa.string()),
    ("is_part_of", pa.string()),
    ("section", pa.string()),
    ("url", pa.string()),
    ("chunk_text", pa.string()),
    ("embedding", pa.list_(pa.float32())),
])

PAGE_COLUMNS = ["page_id", "title", "page_type", "status", "is_part_of", "url", "sections"]


# ---------------------------------------------------------------------------
# S3 / local paths
# ---------------------------------------------------------------------------
def _split_s3(uri: str) -> tuple[str, str]:
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, (prefix.rstrip("/") + "/") if prefix else ""


def _s3():
    import boto3

    return boto3.client("s3")


def read_pages(source: str, cache_dir: str) -> list[dict]:
    """Read page rows from s3://... or a local dir of part-*.parquet (S3 parts cached locally)."""
    tables = []
    if source.startswith("s3://"):
        bucket, prefix = _split_s3(source)
        s3 = _s3()
        keys = sorted(
            o["Key"]
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for o in page.get("Contents", [])
            if o["Key"].endswith(".parquet") and not os.path.basename(o["Key"]).startswith(("_", "."))
        )
        if not keys:
            raise SystemExit(f"no .parquet files under {source}")
        os.makedirs(cache_dir, exist_ok=True)
        for key in keys:
            local = os.path.join(cache_dir, os.path.basename(key))
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if not (os.path.exists(local) and os.path.getsize(local) == size):
                logger.info("downloading s3://%s/%s", bucket, key)
                s3.download_file(bucket, key, local)
            tables.append(pq.read_table(local, columns=PAGE_COLUMNS))
    else:
        files = sorted(f for f in os.listdir(source) if f.endswith(".parquet") and not f.startswith(("_", ".")))
        if not files:
            raise SystemExit(f"no .parquet files in {source}")
        tables = [pq.read_table(os.path.join(source, f), columns=PAGE_COLUMNS) for f in files]

    rows = pa.concat_tables(tables, promote_options="default").to_pylist()
    # Overlapping runs written to the same prefix can repeat a page; keep the first.
    seen, pages = set(), []
    for r in sorted(rows, key=lambda r: r["page_id"]):
        if r["page_id"] not in seen:
            seen.add(r["page_id"])
            pages.append(r)
    return pages


# ---------------------------------------------------------------------------
# Chunking (must match the chunk_id / chunk_text rules used on Databricks)
# ---------------------------------------------------------------------------
def split_lines(text: Optional[str], max_chars: int) -> list[str]:
    """Pack lines into chunks of <= max_chars; only a single over-long line is cut mid-line."""
    chunks, cur = [], ""
    for line in (text or "").split("\n"):
        while len(line) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if not line:
            continue
        if cur and len(cur) + 1 + len(line) > max_chars:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def build_chunks(
    pages: list[dict],
    max_chars: int,
    min_chars: int,
    page_types: Optional[set[str]] = None,
    skip_stub: bool = False,
) -> Iterator[dict]:
    for p in pages:
        if page_types is not None and p["page_type"] not in page_types:
            continue
        if skip_stub and p["status"] == "stub":
            continue
        place = f"{p['page_type']}, {p['is_part_of']}" if p.get("is_part_of") else p["page_type"]
        # sec_idx counts every section, including the short ones skipped below.
        for sec_idx, s in enumerate(p["sections"] or []):
            if len(s["text"] or "") < min_chars:
                continue
            section = s["path"] or "Overview"
            url = f"{p['url']}#{s['anchor']}" if s["anchor"] else p["url"]
            for part, body in enumerate(split_lines(s["text"], max_chars)):
                yield {
                    "chunk_id": f"{p['page_id']}-{sec_idx}-{part}",
                    "page_id": p["page_id"],
                    "title": p["title"],
                    "page_type": p["page_type"],
                    "status": p["status"],
                    "is_part_of": p.get("is_part_of"),
                    "section": section,
                    "url": url,
                    "chunk_text": f"{p['title']} ({place}) — {section}\n{body}",
                }


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def load_model(name: str, device: Optional[str], max_seq_length: int):
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"torch_dtype": torch.float16} if device in ("mps", "cuda") else {}
    logger.info("loading %s on %s", name, device)
    model = SentenceTransformer(name, device=device, model_kwargs=kwargs)
    model.max_seq_length = max_seq_length
    return model


def embed(model, texts: list[str], batch_size: int, query: bool = False) -> list[list[float]]:
    # Qwen3-Embedding: documents get no prompt, queries get the "query" instruction.
    kw = {"prompt_name": "query"} if query and "query" in model.prompts else {"prompt": ""}
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=False, **kw)
    return [v.astype("float32").tolist() for v in vecs]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
class Dest:
    """Upload target: s3://bucket/prefix or a local directory."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        if uri.startswith("s3://"):
            self.bucket, self.prefix = _split_s3(uri)
            self.s3 = _s3()
        else:
            os.makedirs(uri, exist_ok=True)
            self.s3 = None

    def exists(self, name: str) -> bool:
        if self.s3 is None:
            return os.path.exists(os.path.join(self.uri, name))
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self.prefix + name)
            return True
        except Exception:
            return False

    def put(self, local: str, name: str) -> None:
        if self.s3 is None:
            if os.path.abspath(os.path.dirname(local)) != os.path.abspath(self.uri):
                import shutil

                shutil.copyfile(local, os.path.join(self.uri, name))
        else:
            self.s3.upload_file(local, self.bucket, self.prefix + name)


def write_parquet(rows: list[dict], path: str, schema: pa.Schema = OUT_SCHEMA) -> None:
    tmp = path + ".tmp"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
    os.replace(tmp, path)  # a part on disk is always complete


def write_check(model, chunks: list[dict], dest: Dest, workdir: str, batch_size: int) -> None:
    """Write _check.parquet: a few documents and queries embedded locally, for comparing
    against the Databricks endpoint in build_index.py."""
    docs = [c["chunk_text"] for c in chunks[:5]]
    queries = ["cheap hostel near the city centre in Aachen",
               "best time to visit Tokyo", "how to get from the airport to the city"]
    # `input` is the exact string the model saw (query prompt + text), so the endpoint can be
    # called with the same input.
    query_prefix = model.prompts.get("query", "")
    rows = []
    for text, vec in zip(docs, embed(model, docs, batch_size)):
        rows.append({"kind": "document", "prompt": "none", "text": text, "input": text, "embedding": vec})
    for text, vec in zip(queries, embed(model, queries, batch_size, query=True)):
        rows.append({"kind": "query", "prompt": "query", "text": text,
                     "input": query_prefix + text, "embedding": vec})
    for text, vec in zip(queries, embed(model, queries, batch_size)):
        rows.append({"kind": "query", "prompt": "none", "text": text, "input": text, "embedding": vec})
    schema = pa.schema([("kind", pa.string()), ("prompt", pa.string()), ("text", pa.string()),
                        ("input", pa.string()), ("embedding", pa.list_(pa.float32()))])
    path = os.path.join(workdir, "_check.parquet")
    write_parquet(rows, path, schema)
    dest.put(path, "_check.parquet")
    logger.info("wrote %s/_check.parquet (%d rows)", dest.uri.rstrip("/"), len(rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", default="s3://wikivoyage-bucket/pages/", help="stage-1 page Parquet (s3:// or dir)")
    p.add_argument("--dest", default="s3://wikivoyage-bucket/chunks_embedded/", help="output (s3:// or dir)")
    p.add_argument("--workdir", default="chunks_embedded", help="local work dir (cache, parts, manifest)")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--device", default=None, help="mps / cuda / cpu (default: best available)")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--min-chars", type=int, default=50)
    p.add_argument("--rows-per-part", type=int, default=5000)
    p.add_argument("--page-types", default="*", help='comma list, e.g. "city"; "*" = all')
    p.add_argument("--skip-stub", action="store_true", help="skip pages with status=stub")
    p.add_argument("--max-parts", type=int, default=0, help="stop after N new parts (0 = all)")
    p.add_argument("--write-check", action="store_true", help="only write _check.parquet and exit")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    os.makedirs(args.workdir, exist_ok=True)
    page_types = None if args.page_types.strip() == "*" else {t.strip() for t in args.page_types.split(",")}

    pages = read_pages(args.source, os.path.join(args.workdir, "pages_cache"))
    chunks = list(build_chunks(pages, args.max_chars, args.min_chars, page_types, args.skip_stub))
    n_parts = (len(chunks) + args.rows_per_part - 1) // args.rows_per_part
    logger.info("%d pages -> %d chunks -> %d parts of %d", len(pages), len(chunks), n_parts, args.rows_per_part)

    # Parts are only resumable when the chunking and model are unchanged.
    manifest = {
        "model": args.model, "max_chars": args.max_chars, "min_chars": args.min_chars,
        "page_types": args.page_types, "skip_stub": args.skip_stub,
        "rows_per_part": args.rows_per_part, "chunks": len(chunks),
    }
    manifest_path = os.path.join(args.workdir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            old = json.load(fh)
        if old != manifest:
            raise SystemExit(f"{manifest_path} was written with different settings:\n  {old}\nvs\n  {manifest}\n"
                             "Use a new --workdir (and --dest) or delete the old one.")
    else:
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    dest = Dest(args.dest)
    model = load_model(args.model, args.device, args.max_seq_length)

    if args.write_check:
        write_check(model, chunks, dest, args.workdir, args.batch_size)
        return

    done_rows, new_parts, started = 0, 0, time.time()
    for i in range(n_parts):
        name = f"part-{i:05d}.parquet"
        local = os.path.join(args.workdir, name)
        rows = chunks[i * args.rows_per_part:(i + 1) * args.rows_per_part]
        if os.path.exists(local):
            if not dest.exists(name):
                dest.put(local, name)
                logger.info("uploaded existing %s", name)
            continue
        if args.max_parts and new_parts >= args.max_parts:
            logger.info("stopping after %d new part(s) (--max-parts)", new_parts)
            break

        t = time.time()
        vecs = embed(model, [r["chunk_text"] for r in rows], args.batch_size)
        for r, v in zip(rows, vecs):
            r["embedding"] = v
        write_parquet(rows, local)
        dest.put(local, name)
        new_parts += 1
        done_rows += len(rows)

        rate = done_rows / (time.time() - started)
        left = sum(1 for j in range(i + 1, n_parts)
                   if not os.path.exists(os.path.join(args.workdir, f"part-{j:05d}.parquet")))
        logger.info("%s: %d chunks in %.0fs (%.1f/s) | %d/%d parts left, ETA %.1f h",
                    name, len(rows), time.time() - t, rate, left, n_parts,
                    left * args.rows_per_part / rate / 3600 if rate else 0)

    complete = all(os.path.exists(os.path.join(args.workdir, f"part-{j:05d}.parquet")) for j in range(n_parts))
    print(f"{'Done' if complete else 'Stopped'}: {n_parts} parts total → {args.dest}")


if __name__ == "__main__":
    sys.exit(main())
