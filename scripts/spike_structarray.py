"""Spike: check Zilliz Cloud StructArray behaviour for design/structarray-db-design.md (§10-1).

Creates throwaway collections `wikivoyage_articles_spike*`, runs checks (a)–(k) and drops
them (unless --keep). Vectors are random but have the real dims (Qwen3 1024, SigLIP2 1152).
Results go to stdout and --out (JSON).

  python scripts/spike_structarray.py [--keep] [--out articles_work/spike.json]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient, AnnSearchRequest, RRFRanker

ROOT = Path(__file__).resolve().parents[1]
BASE = "wikivoyage_articles_spike"
TEXT_DIM, IMAGE_DIM = 1024, 1152
CHUNK_CAP, IMAGE_CAP = 512, 64
SECTIONS = ["Understand", "Get in", "See", "Eat", "Sleep"]

rng = np.random.default_rng(42)
results: dict[str, dict] = {}


def vec(dim: int) -> list[float]:
    v = rng.standard_normal(dim).astype("float32")
    return (v / np.linalg.norm(v)).tolist()


def check(key: str, desc: str):
    def deco(fn):
        def run(*a, **kw):
            t = time.time()
            try:
                detail = fn(*a, **kw)
                results[key] = {"desc": desc, "ok": True, "detail": detail}
            except Exception as e:  # noqa: BLE001 — record every failure
                results[key] = {"desc": desc, "ok": False, "error": f"{type(e).__name__}: {e}"[:1500]}
                traceback.print_exc()
            results[key]["sec"] = round(time.time() - t, 2)
            r = results[key]
            print(f"[{key}] {'OK ' if r['ok'] else 'NG '} {desc}: {r.get('detail', r.get('error'))}", flush=True)
            return r["ok"]
        return run
    return deco


def make_schema(client: MilvusClient, *, top_vector: bool = True, chunk_cap: int = CHUNK_CAP):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("page_id", DataType.INT64, is_primary=True)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("page_type", DataType.VARCHAR, max_length=32)
    if top_vector:
        schema.add_field("article_embedding", DataType.FLOAT_VECTOR, dim=TEXT_DIM)

    chunk = client.create_struct_field_schema()
    chunk.add_field("chunk_id", DataType.VARCHAR, max_length=64)
    chunk.add_field("chunk_index", DataType.INT64)
    chunk.add_field("top_section", DataType.VARCHAR, max_length=128)
    chunk.add_field("text", DataType.VARCHAR, max_length=8192)
    chunk.add_field("embedding", DataType.FLOAT_VECTOR, dim=TEXT_DIM)
    schema.add_field("chunks", DataType.ARRAY, element_type=DataType.STRUCT,
                     struct_schema=chunk, max_capacity=chunk_cap)

    image = client.create_struct_field_schema()
    image.add_field("file_title", DataType.VARCHAR, max_length=1024)
    image.add_field("caption", DataType.VARCHAR, max_length=2048)
    image.add_field("position", DataType.INT64)
    image.add_field("embedding", DataType.FLOAT_VECTOR, dim=IMAGE_DIM)
    schema.add_field("images", DataType.ARRAY, element_type=DataType.STRUCT,
                     struct_schema=image, max_capacity=IMAGE_CAP)
    return schema


def make_index(client: MilvusClient, *, top_vector: bool = True, vec_index: str = "HNSW"):
    idx = client.prepare_index_params()
    params = {"M": 16, "efConstruction": 200} if vec_index == "HNSW" else {}
    if top_vector:
        idx.add_index(field_name="article_embedding", index_type="AUTOINDEX", metric_type="COSINE")
    idx.add_index(field_name="chunks[embedding]", index_type=vec_index, metric_type="COSINE", params=params)
    idx.add_index(field_name="images[embedding]", index_type=vec_index, metric_type="COSINE", params=params)
    idx.add_index(field_name="chunks[top_section]", index_type="INVERTED")
    return idx


def article(pid: int, n_chunks: int, n_images: int, text_len: int = 300, page_type: str = "city") -> dict:
    chunks = [{
        "chunk_id": f"{pid}-{i}-0",
        "chunk_index": i,
        "top_section": SECTIONS[i % len(SECTIONS)],
        "text": f"[{pid}#{i}] " + "x" * text_len,
        "embedding": vec(TEXT_DIM),
    } for i in range(n_chunks)]
    images = [{
        "file_title": f"File:Spike_{pid}_{j}.jpg",
        "caption": f"caption {pid}/{j}",
        "position": j + 1,
        "embedding": vec(IMAGE_DIM),
    } for j in range(n_images)]
    mean = np.mean([c["embedding"] for c in chunks], axis=0) if chunks else np.ones(TEXT_DIM)
    return {
        "page_id": pid, "title": f"Spike {pid}", "page_type": page_type,
        "article_embedding": (mean / np.linalg.norm(mean)).astype("float32").tolist(),
        "chunks": chunks, "images": images,
    }


def drop(client: MilvusClient, name: str) -> None:
    if client.has_collection(name):
        client.drop_collection(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default="articles_work/spike.json")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    client = MilvusClient(uri=os.environ["ZILLIZ_URI"], token=os.environ["ZILLIZ_TOKEN"])
    import pymilvus
    results["k"] = {"desc": "pymilvus version", "ok": True, "detail": pymilvus.__version__}
    try:
        results["server"] = {"desc": "server version", "ok": True, "detail": client.get_server_version()}
    except Exception as e:  # noqa: BLE001
        results["server"] = {"desc": "server version", "ok": False, "error": str(e)}
    print("server:", results["server"])

    main_coll = BASE
    store: dict = {}

    @check("a", "StructArray collection create (2 struct fields, cap 512/64, HNSW COSINE on subfields)")
    def a():
        drop(client, main_coll)
        client.create_collection(main_coll, schema=make_schema(client), index_params=make_index(client))
        d = client.describe_collection(main_coll)
        return {"struct_fields": [s.get("name") for s in d.get("struct_array_fields", d.get("struct_fields", []))]
                or list(d.keys())}

    @check("h", "subfield index AUTOINDEX (vector) + INVERTED (scalar)")
    def h():
        name = BASE + "_autoidx"
        drop(client, name)
        try:
            client.create_collection(name, schema=make_schema(client), index_params=make_index(client, vec_index="AUTOINDEX"))
            return "AUTOINDEX accepted on chunks[embedding]/images[embedding]"
        finally:
            drop(client, name)

    @check("b", "collection without top-level vector field")
    def b():
        name = BASE + "_notop"
        drop(client, name)
        try:
            client.create_collection(name, schema=make_schema(client, top_vector=False),
                                     index_params=make_index(client, top_vector=False))
            client.insert(name, [{k: v for k, v in article(1, 3, 1).items() if k != "article_embedding"}])
            return "accepted (top-level vector not required)"
        finally:
            drop(client, name)

    @check("c", "insert entity with 512 chunks (max_capacity) and 64 images")
    def c():
        r = client.insert(main_coll, [article(100, CHUNK_CAP, IMAGE_CAP, text_len=100)])
        return {"insert_count": r.get("insert_count")}

    @check("c2", "insert 513 chunks (over capacity) is rejected")
    def c2():
        try:
            client.insert(main_coll, [article(101, CHUNK_CAP + 1, 1, text_len=10)])
        except Exception as e:  # noqa: BLE001
            return f"rejected as expected: {str(e)[:200]}"
        raise RuntimeError("over-capacity insert was ACCEPTED")

    @check("d", "insert with images=[] (empty StructArray)")
    def d():
        r = client.insert(main_coll, [article(200, 5, 0)])
        return {"insert_count": r.get("insert_count")}

    @check("d0", "insert with chunks=[] (empty) as well")
    def d0():
        r = client.insert(main_coll, [article(201, 0, 2)])
        return {"insert_count": r.get("insert_count")}

    @check("base", "insert normal articles (ids 1..20)")
    def base():
        rows = [article(i, int(rng.integers(3, 40)), int(rng.integers(0, 6)),
                        page_type="city" if i % 2 else "region") for i in range(1, 21)]
        store["rows"] = {r["page_id"]: r for r in rows}
        r = client.insert(main_coll, rows)
        client.flush(main_coll)
        client.load_collection(main_coll)
        return {"insert_count": r.get("insert_count"), "stats": client.get_collection_stats(main_coll)}

    @check("d2", "query entity with empty images back")
    def d2():
        res = client.query(main_coll, filter="page_id in [200, 201]", output_fields=["page_id", "images", "chunks"])
        return [{"page_id": r["page_id"], "n_images": len(r.get("images") or []),
                 "images_value": repr(r.get("images"))[:60], "n_chunks": len(r.get("chunks") or [])} for r in res]

    @check("e", "upsert on StructArray collection (replace chunks of page 3)")
    def e():
        new = article(3, 2, 1)
        new["chunks"][0]["text"] = "UPSERTED"
        r = client.upsert(main_coll, [new])
        res = client.query(main_coll, filter="page_id == 3", output_fields=["chunks"],
                           consistency_level="Strong")
        ch = res[0]["chunks"]
        return {"upsert_count": r.get("upsert_count"), "n_chunks_after": len(ch), "first_text": ch[0]["text"]}

    @check("e2", "partial update (upsert with only page_id + title)")
    def e2():
        r = client.upsert(main_coll, [{"page_id": 4, "title": "Partial 4"}], partial_update=True)
        res = client.query(main_coll, filter="page_id == 4", output_fields=["title", "chunks"],
                           consistency_level="Strong")
        return {"upsert_count": r.get("upsert_count"), "title": res[0]["title"],
                "chunks_kept": len(res[0]["chunks"])}

    def target(pid: int = 7, idx: int = 2):
        row = store["rows"][pid]
        idx = min(idx, len(row["chunks"]) - 1)
        return row, idx

    @check("f", "element-level search on chunks[embedding]: offset + output_fields chunks[text]")
    def f():
        row, idx = target()
        res = client.search(main_coll, data=[row["chunks"][idx]["embedding"]], anns_field="chunks[embedding]",
                            limit=5, output_fields=["page_id", "title", "chunks[text]", "chunks[top_section]"],
                            search_params={"metric_type": "COSINE"}, consistency_level="Strong")
        hits = []
        for h in res[0]:
            ent = h.get("entity", {})
            hits.append({"id": h.get("id"), "distance": round(h.get("distance", 0), 4), "offset": h.get("offset"),
                         "keys": sorted(h.keys()), "entity_keys": sorted(ent.keys()),
                         "entity_preview": repr(ent)[:300]})
        return {"expected": {"page_id": row["page_id"], "offset": idx, "text": row["chunks"][idx]["text"][:20]},
                "hits": hits}

    @check("f2", "element-level search: are multiple elements of the same entity returned?")
    def f2():
        row, idx = target()
        res = client.search(main_coll, data=[row["chunks"][idx]["embedding"]], anns_field="chunks[embedding]",
                            limit=10, output_fields=["page_id"], consistency_level="Strong")
        return [(h.get("id"), h.get("offset")) for h in res[0]]

    @check("g", "filter: entity predicate + element_filter(chunks, $[top_section] == ...)")
    def g():
        row, idx = target()
        sec = row["chunks"][idx]["top_section"]
        flt = f'page_type == "city" && element_filter(chunks, $[top_section] == "{sec}")'
        res = client.search(main_coll, data=[row["chunks"][idx]["embedding"]], anns_field="chunks[embedding]",
                            limit=5, filter=flt, output_fields=["page_id", "page_type", "chunks[top_section]"],
                            consistency_level="Strong")
        return {"filter": flt, "hits": [{"id": h.get("id"), "offset": h.get("offset"),
                                         "entity": repr(h.get("entity"))[:200]} for h in res[0]]}

    @check("g2", "entity-level filters: MATCH_ANY / array_contains / array_length")
    def g2():
        out = {}
        for flt in ['MATCH_ANY(chunks, $[top_section] == "Sleep")',
                    'array_contains(chunks[top_section], "Sleep")',
                    'array_length(images) == 0']:
            try:
                out[flt] = sorted(r["page_id"] for r in client.query(main_coll, filter=flt, output_fields=["page_id"],
                                                                     consistency_level="Strong"))
            except Exception as e:  # noqa: BLE001
                out[flt] = f"ERROR {str(e)[:200]}"
        return out

    @check("f3", "image search on images[embedding] (incl. entity with images=[])")
    def f3():
        row = next(r for r in store["rows"].values() if r["images"])
        res = client.search(main_coll, data=[row["images"][0]["embedding"]], anns_field="images[embedding]",
                            limit=5, output_fields=["page_id", "images[file_title]"], consistency_level="Strong")
        return {"expected": row["page_id"],
                "hits": [{"id": h.get("id"), "offset": h.get("offset"), "entity": repr(h.get("entity"))[:200]}
                         for h in res[0]]}

    @check("j", "hybrid_search over chunks[embedding] + images[embedding]")
    def j():
        row = next(r for r in store["rows"].values() if r["images"])
        reqs = [AnnSearchRequest([row["chunks"][0]["embedding"]], "chunks[embedding]", {"metric_type": "COSINE"}, 5),
                AnnSearchRequest([row["images"][0]["embedding"]], "images[embedding]", {"metric_type": "COSINE"}, 5)]
        res = client.hybrid_search(main_coll, reqs, RRFRanker(60), limit=5, output_fields=["page_id"])
        return {"expected": row["page_id"], "hits": [(h.get("id"), h.get("offset")) for h in res[0]]}

    @check("i", "request size: batches of ~2 / 8 / 32 MB")
    def i():
        out = {}
        pid = 10_000
        # one article with 200 chunks of 2000 chars + 20 images ≈ 200*(4 KB vec as JSON-ish + 2 KB) — measure raw
        for mb in (2, 8, 32):
            rows, size = [], 0
            while size < mb * 1024 * 1024:
                r = article(pid, 200, 20, text_len=2000)
                pid += 1
                rows.append(r)
                size += 200 * (TEXT_DIM * 4 + 2000) + 20 * IMAGE_DIM * 4 + TEXT_DIM * 4
            t = time.time()
            try:
                res = client.insert(main_coll, rows)
                out[f"{mb}MB"] = {"rows": len(rows), "ok": True, "count": res.get("insert_count"),
                                  "sec": round(time.time() - t, 1)}
            except Exception as e:  # noqa: BLE001
                out[f"{mb}MB"] = {"rows": len(rows), "ok": False, "error": str(e)[:300]}
        return out

    @check("x", "delete by page_id filter (replace mode fallback)")
    def x():
        r = client.delete(main_coll, filter="page_id in [5, 6]")
        return {"delete_count": r.get("delete_count") if isinstance(r, dict) else r}

    try:
        if a():
            for fn in (h, b, c, c2, d, d0, base, d2, e, e2, f, f2, g, g2, f3, j, i, x):
                fn()
    finally:
        if not args.keep:
            for n in (main_coll, BASE + "_autoidx", BASE + "_notop"):
                try:
                    drop(client, n)
                except Exception:  # noqa: BLE001
                    pass
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
