"""CLI: ``python -m image_embedding <subcommand> [options]`` (run from the repo root).

Exit codes: 0 ok · 1 error / bad input · 2 dump checksum mismatch ·
3 Zilliz schema mismatch · 4 some upsert batches failed (see runs/<date>/upsert/failed/).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
from typing import Callable, Optional

from .config import ImagesConfig, StageError

logger = logging.getLogger("wikivoyage.images")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ("dump", "extract", "commons", "fetch", "embed", "upsert")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dump-date", help="dump date YYYYMMDD (env IMAGES_DUMP_DATE)")
    common.add_argument("--workdir", help="local dir or s3://bucket/prefix (env IMAGES_WORKDIR, default images_work)")
    common.add_argument("--log-level", default="INFO")

    p = argparse.ArgumentParser(prog="python -m image_embedding",
                                description="Wikivoyage image pipeline: dump → refs → Commons → thumbnails → SigLIP2 → Zilliz")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[common], help=help_)

    def limit(sp):
        sp.add_argument("--limit", type=int, default=0, help="process at most N items (debugging)")

    def stage_opts(sp, names: tuple[str, ...]):
        if "dump" in names:
            sp.add_argument("--force", action="store_true", help="download the dump again")
        if "extract" in names:
            sp.add_argument("--dump-path", help="dump to read instead of the downloaded one (path, s3:// or URL)")
            sp.add_argument("--max-per-page", type=int, help="image cap per article (env IMAGES_MAX_PER_PAGE=50)")
        if "commons" in names:
            sp.add_argument("--no-retry-errors", dest="retry_errors", action="store_false",
                            help="do not re-query titles that previously errored")
        if {"commons", "fetch"} & set(names):
            sp.add_argument("--concurrency", type=int, help="parallel requests")
        if "fetch" in names:
            sp.add_argument("--rps", type=float, help="image downloads per second (env IMAGES_DOWNLOAD_RPS=4)")
        if "embed" in names:
            sp.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))
            sp.add_argument("--batch-size", type=int, help="images per forward pass (env SIGLIP_BATCH_SIZE=32)")
            sp.add_argument("--keep-images", action="store_true", default=None,
                            help="keep downloaded image files after embedding (default: delete; env IMAGES_KEEP_FILES)")
        if "upsert" in names:
            sp.add_argument("--collection", help="collection name (env ZILLIZ_IMAGE_COLLECTION)")
            sp.add_argument("--zilliz-batch-size", type=int, help="rows per upsert (default 100)")
            sp.add_argument("--prune", action="store_true",
                            help="after a fully successful run, delete rows from other dump dates")
            sp.add_argument("--dry-run", action="store_true", help="build and validate rows without writing")
            sp.add_argument("--model-slug", help="embeddings/<slug> to use (default: newest for the model)")

    for name in STAGES:
        sp = add(name, f"run the {name} stage")
        stage_opts(sp, (name,))
        if name != "dump" and name != "upsert":
            limit(sp)
    sp = add("all", "run dump → extract → commons → fetch → embed → upsert")
    stage_opts(sp, STAGES)
    limit(sp)

    sp = add("replay", "re-send failed upsert batches")
    sp.add_argument("--file", help="one failed batch (path relative to the work dir); default: all")
    sp.add_argument("--collection")

    sp = add("search", "search by image, image URL or text")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="local image file")
    g.add_argument("--image-url", help="http(s) image URL")
    g.add_argument("--text", help="text query (SigLIP2 text encoder)")
    sp.add_argument("--page-title", help="restrict to one article")
    sp.add_argument("--top-k", type=int, default=10)
    sp.add_argument("--collection")
    sp.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))
    sp.add_argument("--json", action="store_true", help="print JSON")

    add("status", "show per-stage counts for a dump date")
    return p


def _apply_args(cfg: ImagesConfig, a: argparse.Namespace) -> None:
    mapping = {
        "dump_date": "dump_date", "workdir": "workdir", "max_per_page": "max_per_page",
        "rps": "download_rps", "device": "device", "batch_size": "embed_batch_size",
        "collection": "collection", "zilliz_batch_size": "zilliz_batch_size", "keep_images": "keep_images",
    }
    for arg, field in mapping.items():
        v = getattr(a, arg, None)
        if v is not None:
            setattr(cfg, field, v)
    if getattr(a, "concurrency", None):
        cfg.api_concurrency = min(4, a.concurrency)
        cfg.download_concurrency = a.concurrency


def _write_summary(cfg: ImagesConfig, store, stage: str, run_id: str, started: float, stats: dict) -> None:
    if not cfg.dump_date:
        return
    summary = {
        "stage": stage, "run_id": run_id, "dump_date": cfg.dump_date,
        "started_at": _dt.datetime.fromtimestamp(started, _dt.timezone.utc).isoformat(timespec="seconds"),
        "finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 1),
        "stats": stats,
    }
    try:
        store.write_json(f"{cfg.summary_dir()}/{stage}-{run_id}.json", summary)
    except Exception as e:  # a summary must never fail the stage
        logger.warning("could not write summary: %s", e)


def _run_stage(name: str, cfg: ImagesConfig, store, a: argparse.Namespace, run_id: str) -> int:
    from . import commons, dump, embed, extract, fetch, upsert

    lim = getattr(a, "limit", 0) or 0
    runners: dict[str, Callable[[], dict]] = {
        "dump": lambda: dump.run(cfg, store, force=getattr(a, "force", False)),
        "extract": lambda: extract.run(cfg, store, dump_path=getattr(a, "dump_path", None), limit=lim),
        "commons": lambda: commons.run(cfg, store, run_id=run_id,
                                       retry_errors=getattr(a, "retry_errors", True), limit=lim),
        "fetch": lambda: fetch.run(cfg, store, run_id=run_id, limit=lim),
        "embed": lambda: embed.run(cfg, store, run_id=run_id, limit=lim),
        "upsert": lambda: upsert.run(cfg, store, run_id=run_id, prune=getattr(a, "prune", False),
                                     dry_run=getattr(a, "dry_run", False),
                                     model_slug=getattr(a, "model_slug", None)),
    }
    started = time.time()
    logger.info("=== %s (dump_date=%s, run_id=%s) ===", name, cfg.dump_date, run_id)
    stats = runners[name]()
    _write_summary(cfg, store, name, run_id, started, stats)
    if name == "upsert" and stats.get("failed_batches"):
        logger.error("%d upsert batches failed; run 'replay' to resend them", stats["failed_batches"])
        return 4
    return 0


def _search(cfg: ImagesConfig, a: argparse.Namespace) -> int:
    from PIL import Image

    from .embedder import SiglipImageEmbedder
    from .search import ImageSearcher
    from .zilliz import make_client

    if not cfg.zilliz_uri or not cfg.zilliz_token:
        raise StageError("ZILLIZ_URI and ZILLIZ_TOKEN must be set", 1)
    embedder = SiglipImageEmbedder(cfg.model_id, cfg.model_revision, device=cfg.device,
                                   dtype=cfg.dtype, hf_token=cfg.hf_token)
    searcher = ImageSearcher(make_client(cfg.zilliz_uri, cfg.zilliz_token), embedder, cfg.collection,
                             text_template=cfg.text_template, user_agent=cfg.user_agent)
    kw = {"top_k": a.top_k, "page_title": a.page_title}
    try:
        if a.image:
            hits = searcher.search_by_image(Image.open(a.image), **kw)
        elif a.image_url:
            hits = searcher.search_by_image_url(a.image_url, **kw)
        else:
            hits = searcher.search_by_text(a.text, **kw)
    except (ValueError, OSError) as e:
        raise StageError(str(e), 1) from e
    if a.json:
        print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
    else:
        for i, h in enumerate(hits, 1):
            print(f"{i:2d}. {h.score:.4f}  {h.page_title}  {h.image_title}")
            print(f"     article: {h.page_url}")
            print(f"     image:   {h.image_url}")
            print(f"     license: {h.license}" + (f" ({h.license_url})" if h.license_url else "")
                  + (f"  by {h.artist}" if h.artist else ""))
            if h.caption:
                print(f"     caption: {h.caption}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    _load_dotenv()
    a = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(a.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = ImagesConfig.from_env()
    _apply_args(cfg, a)

    from . import ids
    from .storage import open_store

    try:
        if a.cmd == "search":
            return _search(cfg, a)
        store = open_store(cfg.workdir)
        run_id = ids.new_run_id()
        if a.cmd == "status":
            from .status import collect

            cfg.require_dump_date()
            print(json.dumps(collect(cfg, store), ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "replay":
            from .upsert import replay

            cfg.require_dump_date()
            stats = replay(cfg, store, run_id=run_id, file=a.file)
            _write_summary(cfg, store, "replay", run_id, time.time(), stats)
            return 4 if stats["failed_batches"] else 0
        cfg.require_dump_date()
        names = STAGES if a.cmd == "all" else (a.cmd,)
        for name in names:
            code = _run_stage(name, cfg, store, a, run_id)
            if code:
                return code
        return 0
    except StageError as e:
        logger.error("%s", e)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
