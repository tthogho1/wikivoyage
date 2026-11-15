"""CLI: ``python -m article_db <assemble|upsert|all|replay|status|search>`` (run from the repo root).

  assemble  chunks_embedded (Qwen3) + stage-1 pages + image pipeline outputs (SigLIP2)
            → articles_work/snapshots/<id>/part-*.parquet (one row per article)
  upsert    snapshot → Zilliz `wikivoyage_articles` (only articles whose content changed)
  all       assemble, then upsert the new snapshot
  replay    resend failed upsert batches
  status    snapshot / sent / pending counts and collection rows
  search    --text and/or --image/--image-url → top articles with matching chunks and images

Exit codes: 0 ok · 1 error · 3 collection schema mismatch · 4 some upsert batches failed ·
5 inconsistent inputs (missing chunk parts, chunks vs pages mismatch).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
from typing import Optional

from .config import ArticlesConfig, StageError

logger = logging.getLogger("wikivoyage.articles")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", help="local dir or s3://bucket/prefix (env ARTICLES_WORKDIR, default articles_work)")
    common.add_argument("--collection", help="Zilliz collection (env ZILLIZ_ARTICLE_COLLECTION, default wikivoyage_articles)")
    common.add_argument("--log-level", default="INFO")

    asm = argparse.ArgumentParser(add_help=False)
    asm.add_argument("--chunks-source", help="embed_chunks.py output (env ARTICLES_CHUNKS_SOURCE, default chunks_embedded)")
    asm.add_argument("--pages-source", help="stage-1 page Parquet the chunks were built from "
                                            "(default <chunks-source>/pages_cache, else s3://wikivoyage-bucket/pages/)")
    asm.add_argument("--images-workdir", help="image pipeline work dir (env IMAGES_WORKDIR)")
    asm.add_argument("--image-dump-date", help="image pipeline dump date YYYYMMDD (env IMAGES_DUMP_DATE)")
    asm.add_argument("--image-model-slug", help="embeddings/<slug> to read (default: newest for SIGLIP_MODEL_ID)")
    asm.add_argument("--no-images", action="store_true", help="text chunks only (images = [])")
    asm.add_argument("--no-verify-chunks", dest="verify", action="store_false",
                     help="skip re-deriving chunk_ids from the pages with embed_chunks.py's rules")
    asm.add_argument("--allow-partial", action="store_true", help="build from the chunk parts that exist")
    asm.add_argument("--strict-revision", action="store_true",
                     help="drop an article's images when their revision differs from the page's")
    asm.add_argument("--articles", type=int, default=0, help="assemble at most N articles (testing)")

    up = argparse.ArgumentParser(add_help=False)
    up.add_argument("--batch-size", type=int, help="articles per request (env ARTICLES_BATCH_SIZE, default 50)")
    up.add_argument("--max-batch-bytes", type=int, help="bytes per request (env ARTICLES_MAX_BATCH_BYTES, default 8 MiB)")
    up.add_argument("--prune", action="store_true", help="delete articles that are not in the snapshot")
    up.add_argument("--dry-run", action="store_true", help="do not write to Zilliz")
    up.add_argument("--force", action="store_true", help="send every article, ignoring state/done-*")
    up.add_argument("--limit", type=int, default=0, help="send at most N articles (testing)")

    p = argparse.ArgumentParser(prog="python -m article_db", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("assemble", parents=[common, asm], help="build a snapshot")
    s = sub.add_parser("upsert", parents=[common, up], help="send a snapshot to Zilliz")
    s.add_argument("--snapshot", help="snapshot id (default: newest complete)")
    sub.add_parser("all", parents=[common, asm, up], help="assemble + upsert")
    s = sub.add_parser("replay", parents=[common], help="resend failed batches")
    s.add_argument("--snapshot")
    s.add_argument("--file", help="one failed batch (path relative to the work dir)")
    s = sub.add_parser("status", parents=[common], help="show progress")
    s.add_argument("--snapshot")
    s = sub.add_parser("search", parents=[common], help="search by text and/or image")
    s.add_argument("--text", help="English question / description")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--image", help="local image file")
    g.add_argument("--image-url", help="http(s) image URL")
    s.add_argument("--mode", choices=["both", "text", "image"], default="both",
                   help="text = chunks only, image = images only (default both)")
    s.add_argument("--top-k", type=int, default=5, help="articles to return")
    s.add_argument("--page-type")
    s.add_argument("--is-part-of")
    s.add_argument("--title")
    s.add_argument("--heading", help="top-level section of the chunk, e.g. Sleep (text search only)")
    s.add_argument("--json", action="store_true")
    return p


def _search(cfg: ArticlesConfig, a: argparse.Namespace) -> int:
    from .search import SearchFilters, build_searcher

    if not a.text and not a.image and not a.image_url:
        raise StageError("give --text, --image or --image-url", 1)
    img = None
    if a.image:
        from PIL import Image

        img = Image.open(a.image)
    elif a.image_url:
        from image_embedding.config import ImagesConfig
        from image_embedding.search import load_image_from_url

        img = load_image_from_url(a.image_url, ImagesConfig.from_env().user_agent)
    need_text = bool(a.text) and a.mode in ("both", "text")
    need_image = a.mode in ("both", "image") and (img is not None or bool(a.text))
    try:
        searcher = build_searcher(cfg, text=need_text, image=need_image)
    except ValueError as e:
        raise StageError(str(e), 1) from e
    res = searcher.search(a.text, img, top_k=a.top_k, mode=a.mode,
                          filters=SearchFilters(a.page_type, a.is_part_of, a.title, a.heading))
    if a.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"mode={res.mode} encoders={','.join(res.encoders)}")
    for i, art in enumerate(res.articles, 1):
        print(f"\n{i}. [{art.score:.4f}] {art.title} ({art.page_type}, {art.is_part_of})  {art.url}")
        for c in art.chunks:
            body = c.text.split("\n", 1)[-1].replace("\n", " ")
            print(f"   - {c.section_path or '(intro)'} [{c.score:.3f}] {body[:200]}{'…' if len(body) > 200 else ''}")
        for m in art.images:
            print(f"   * [{m.score:.3f}] {m.caption or m.file_title}  {m.image_url}  ({m.license}, {m.artist[:40]})")
    return 0


def _apply(cfg: ArticlesConfig, a: argparse.Namespace) -> None:
    for arg, attr in {"workdir": "workdir", "collection": "collection", "chunks_source": "chunks_source",
                      "pages_source": "pages_source", "images_workdir": "images_workdir",
                      "image_dump_date": "image_dump_date", "image_model_slug": "image_model_slug",
                      "batch_size": "batch_size", "max_batch_bytes": "max_batch_bytes"}.items():
        v = getattr(a, arg, None)
        if v is not None:
            setattr(cfg, attr, v)
    if getattr(a, "no_images", False):
        cfg.no_images = True


def _summary(store, stage: str, run_id: str, started: float, stats: dict) -> None:
    try:
        store.write_json(f"summary/{stage}-{run_id}.json", {
            "stage": stage, "run_id": run_id,
            "started_at": _dt.datetime.fromtimestamp(started, _dt.timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.time() - started, 1), "stats": stats})
    except Exception as e:  # noqa: BLE001 — a summary must never fail the stage
        logger.warning("could not write summary: %s", e)


def main(argv: Optional[list[str]] = None) -> int:
    _load_dotenv()
    a = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(a.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = ArticlesConfig.from_env()
    _apply(cfg, a)
    if a.cmd == "search":
        try:
            return _search(cfg, a)
        except StageError as e:
            logger.error("%s", e)
            return e.code

    from image_embedding.ids import new_run_id
    from image_embedding.storage import open_store

    from . import assemble, upsert

    store = open_store(cfg.workdir)
    run_id = new_run_id()
    try:
        if a.cmd == "status":
            print(json.dumps(upsert.status(cfg, store, snapshot=a.snapshot), ensure_ascii=False, indent=2, default=str))
            return 0
        if a.cmd == "replay":
            started = time.time()
            stats = upsert.replay(cfg, store, run_id=run_id, snapshot=a.snapshot, file=a.file)
            _summary(store, "replay", run_id, started, stats)
            return 4 if stats["failed_batches"] else 0

        snapshot = getattr(a, "snapshot", None)
        if a.cmd in ("assemble", "all"):
            started = time.time()
            stats = assemble.run(cfg, run_id=run_id, store=store, limit=a.articles, verify=a.verify,
                                 allow_partial=a.allow_partial, strict_revision=a.strict_revision)
            _summary(store, "assemble", run_id, started, stats)
            snapshot = stats["snapshot_id"]
        if a.cmd in ("upsert", "all"):
            started = time.time()
            stats = upsert.run(cfg, store, run_id=run_id, snapshot=snapshot, prune=a.prune,
                               dry_run=a.dry_run, force=a.force, limit=a.limit)
            _summary(store, "upsert", run_id, started, stats)
            if stats["failed_batches"]:
                logger.error("%d batches failed; run 'python -m article_db replay'", stats["failed_batches"])
                return 4
        return 0
    except StageError as e:
        logger.error("%s", e)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
