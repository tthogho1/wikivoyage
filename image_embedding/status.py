"""Stage ``status``: counts per stage for one dump date."""
from __future__ import annotations

from collections import Counter

from . import commons as commons_stage
from . import embed as embed_stage
from . import fetch as fetch_stage
from .config import ImagesConfig, StageError
from .storage import Store
from .upsert import load_done_ids


def collect(cfg: ImagesConfig, store: Store) -> dict:
    out: dict = {"dump_date": cfg.dump_date, "workdir": store.uri("")}
    rel = cfg.dump_rel()
    out["dump"] = "verified" if store.exists(rel + ".verified") else ("present" if store.exists(rel) else "missing")

    if store.exists(cfg.refs_rel()):
        refs = list(store.iter_jsonl(cfg.refs_rel()))
        titles = {r["file_title"] for r in refs}
        out["extract"] = {"refs": len(refs), "articles": len({r["source_page_id"] for r in refs}),
                          "unique_images": len(titles)}
    else:
        titles = set()
        out["extract"] = "not run"

    commons = commons_stage.load_done(store, cfg)
    c_status = Counter(r["status"] for r in commons.values())
    out["commons"] = {**c_status, "not_queried": len(titles - set(commons))}

    ok = {r["file_key"]: r for r in commons.values() if r["status"] == "ok"}
    manifest = fetch_stage.load_manifest(store, cfg)
    m_status = Counter()
    fetched_ok = set()
    for fk, meta in ok.items():
        m = manifest.get((fk, meta.get("commons_sha1")))
        m_status[m["status"] if m else "not_fetched"] += 1
        if m and m["status"] == "ok":
            fetched_ok.add((fk, meta.get("commons_sha1")))
    out["fetch"] = dict(m_status)

    try:
        slug = embed_stage.resolve_slug(store, cfg)
        done = embed_stage.done_keys(store, slug)
        out["embed"] = {"model_slug": slug, "embedded": len(fetched_ok & done),
                        "not_embedded": len(fetched_ok - done)}
    except StageError:
        out["embed"] = "not run"

    done_ids = load_done_ids(store, cfg)
    failed = [p for p in store.list(f"{cfg.upsert_dir()}/failed") if p.endswith(".jsonl")]
    out["upsert"] = {"upserted_ids": len(done_ids), "failed_batches": len(failed)}
    return out
