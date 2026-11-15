"""article_db: assemble (chunks × pages × images) and upsert, with synthetic inputs and a fake client."""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pymilvus import MilvusClient

from article_db import assemble, sources, upsert
from article_db.config import ArticlesConfig, StageError
from article_db.schema import IMAGE_DIM, TEXT_DIM, clean_row, row_bytes, snapshot_schema
from image_embedding import ids
from image_embedding.storage import LocalStore

SLUG = "google__siglip2-so400m-patch16-384@abc123def456"
DUMP = "20261101"


def unit(rng, dim):
    v = rng.standard_normal(dim).astype("float32")
    return v / np.linalg.norm(v)


def page(pid, title, sections, **kw):
    return {"page_id": pid, "title": title, "page_type": kw.get("page_type", "city"), "status": "usable",
            "is_part_of": kw.get("is_part_of", "Region"), "url": f"https://en.wikivoyage.org/wiki/{title}",
            "revision_id": pid * 100, "timestamp": "2026-11-01T00:00:00Z",
            "sections": [{"title": t, "path": p, "anchor": p.replace(" > ", "_").replace(" ", "_") if p else "", "text": x}
                         for t, p, x in sections]}


PAGES = [
    page(1, "Tokyo", [("", "", "Tokyo is big. " * 10), ("See", "See", "Temples and towers. " * 10),
                      ("Budget", "Sleep > Budget", "Hostels. " * 10), ("Tiny", "Tiny", "short")]),
    page(2, "Kyoto", [("", "", "Kyoto is old. " * 10), ("Eat", "Eat", ("Tofu. " * 300 + "\n") * 2)]),
    page(3, "Nara", [("", "", "Deer. " * 20)]),
]


@pytest.fixture
def inputs(tmp_path):
    """chunks_embedded/ (+ pages_cache) and an image work dir, like the real pipelines write them."""
    rng = np.random.default_rng(0)
    ec = sources.embed_chunks_module()
    chunks = list(ec.build_chunks(PAGES, 2000, 50, None, False))
    for c in chunks:
        c["embedding"] = unit(rng, TEXT_DIM).tolist()
    cdir = tmp_path / "chunks_embedded"
    (cdir / "pages_cache").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(chunks[:4], schema=ec.OUT_SCHEMA), cdir / "part-00000.parquet")
    pq.write_table(pa.Table.from_pylist(chunks[4:], schema=ec.OUT_SCHEMA), cdir / "part-00001.parquet")
    (cdir / "manifest.json").write_text(json.dumps({
        "model": "Qwen/Qwen3-Embedding-0.6B", "max_chars": 2000, "min_chars": 50, "page_types": "*",
        "skip_stub": False, "rows_per_part": 4, "chunks": len(chunks)}))
    pq.write_table(pa.Table.from_pylist(PAGES), cdir / "pages_cache" / "part-00000.parquet")

    # Image work dir: Tokyo has 2 usable images + 1 without metadata, Kyoto 1 with a stale vector.
    istore = LocalStore(str(tmp_path / "images_work"))
    refs, commons, embs = [], [], []
    for pid, title, pos, ok, stale in [(1, "File:Shibuya.jpg", 2, True, False), (1, "File:Tower.jpg", 1, True, False),
                                       (1, "File:Missing.jpg", 3, False, False), (2, "File:Kinkaku.jpg", 1, True, True)]:
        key = ids.file_key(title)
        refs.append({"wiki": "enwikivoyage", "dump_date": DUMP, "source_page_id": pid, "source_page_title": "x",
                     "page_url": "u", "revision_id": pid * 100, "file_title": title, "file_key": key,
                     "record_id": ids.record_id("enwikivoyage", pid, title), "position": pos, "caption": f"cap {title}"})
        if ok:
            commons.append({"file_title": title, "file_key": key, "status": "ok", "thumb_url": f"https://up/{key}.jpg",
                            "description_url": f"https://commons/{title}", "license_short_name": "CC BY-SA 4.0",
                            "license_url": "https://cc", "artist": "Someone", "commons_sha1": "sha-new"})
            embs.append({"file_key": key, "file_title": title, "commons_sha1": "sha-old" if stale else "sha-new",
                         "processed_sha256": "p", "vector": unit(rng, IMAGE_DIM).tolist(), "model_id": "m",
                         "model_revision": "abc123def456", "embedded_at": None})
    istore.write_bytes(f"runs/{DUMP}/refs.jsonl", "".join(json.dumps(r) + "\n" for r in refs).encode())
    istore.write_bytes(f"runs/{DUMP}/commons/part-1-00000.jsonl", "".join(json.dumps(r) + "\n" for r in commons).encode())
    from image_embedding.embed import write_parquet_part
    write_parquet_part(istore, f"embeddings/{SLUG}/part-1-00000.parquet", embs)
    istore.write_json(f"embeddings/{SLUG}/model_info.json", {"model_id": "google/siglip2-so400m-patch16-384",
                                                            "model_revision": "abc123def456", "dim": IMAGE_DIM,
                                                            "created_at": "2026-11-02"})
    cfg = ArticlesConfig(workdir=str(tmp_path / "aw"), chunks_source=str(cdir), images_workdir=str(tmp_path / "images_work"),
                         image_dump_date=DUMP)
    return cfg, chunks


def snapshot_rows(cfg, sid):
    return {r["page_id"]: r for r in upsert.iter_snapshot_rows(LocalStore(cfg.workdir), sid)}


def test_assemble_joins_chunks_pages_and_images(inputs):
    cfg, chunks = inputs
    stats = assemble.run(cfg, run_id="r1")
    assert stats["articles"] == 3 and stats["chunks"] == len(chunks) and stats["mismatches"] == 0
    assert stats["images"] == 2 and stats["images_pending"] == 2  # missing metadata + stale vector
    rows = snapshot_rows(cfg, "r1")

    tokyo = rows[1]
    assert [c["chunk_id"] for c in tokyo["chunks"]] == ["1-0-0", "1-1-0", "1-2-0"]  # "Tiny" is < min_chars
    assert [c["chunk_index"] for c in tokyo["chunks"]] == [0, 1, 2]
    budget = tokyo["chunks"][2]
    assert (budget["section_path"], budget["heading"], budget["top_section"]) == ("Sleep > Budget", "Budget", "Sleep")
    assert budget["text"] == next(c["chunk_text"] for c in chunks if c["chunk_id"] == "1-2-0")
    assert [i["file_title"] for i in tokyo["images"]] == ["File:Tower.jpg", "File:Shibuya.jpg"]  # position order
    img = tokyo["images"][0]
    assert img["license"] == "CC BY-SA 4.0" and img["record_id"] == ids.record_id("enwikivoyage", 1, "File:Tower.jpg")
    assert len(img["embedding"]) == IMAGE_DIM
    assert abs(np.linalg.norm(tokyo["article_embedding"]) - 1) < 1e-5
    assert tokyo["last_updated"] == 1793491200 and tokyo["image_dump_date"] == DUMP

    assert rows[2]["images"] == [] and rows[2]["image_count"] == 0
    assert [c["chunk_id"] for c in rows[2]["chunks"]] == ["2-0-0", "2-1-0", "2-1-1"]  # long section split


def test_content_hash_is_stable_and_tracks_changes(inputs):
    cfg, _ = inputs
    assemble.run(cfg, run_id="r1")
    assemble.run(cfg, run_id="r2")
    a, b = snapshot_rows(cfg, "r1"), snapshot_rows(cfg, "r2")
    assert all(a[p]["content_hash"] == b[p]["content_hash"] for p in a)  # snapshot_id is not hashed
    cfg.no_images = True
    assemble.run(cfg, run_id="r3")
    c = snapshot_rows(cfg, "r3")
    assert c[1]["content_hash"] != a[1]["content_hash"] and c[3]["content_hash"] != a[3]["content_hash"]


def test_row_size_limit_drops_images_then_chunks(inputs):
    cfg, _ = inputs
    base = snapshot_rows(cfg, assemble.run(cfg, run_id="r1")["snapshot_id"])[1]
    cfg.max_row_bytes = row_bytes(base) - 100  # one image too many
    stats = assemble.run(cfg, run_id="r2")
    tokyo = snapshot_rows(cfg, "r2")[1]
    assert tokyo["image_count"] == 1 and tokyo["images_truncated"] and not tokyo["chunks_truncated"]
    assert stats["images_cut_size"] == 1
    cfg.max_row_bytes = row_bytes({**base, "images": [], "chunks": base["chunks"][:2]})
    tokyo = snapshot_rows(cfg, assemble.run(cfg, run_id="r3")["snapshot_id"])[1]
    assert tokyo["image_count"] == 0 and tokyo["chunk_count"] == 2 and tokyo["chunks_truncated"]


def test_capacity_limit(inputs):
    cfg, _ = inputs
    cfg.max_chunks, cfg.max_images = 2, 1
    stats = assemble.run(cfg, run_id="r1")
    tokyo = snapshot_rows(cfg, "r1")[1]
    assert tokyo["chunk_count"] == 2 and tokyo["image_count"] == 1 and tokyo["images"][0]["position"] == 1
    assert stats["chunks_cut_capacity"] == 2  # Tokyo 3→2, Kyoto 3→2
    overflow = LocalStore(cfg.workdir).iter_jsonl("snapshots/r1/overflow.jsonl")
    assert {r["page_id"] for r in overflow} == {1, 2}


def test_verify_detects_chunk_text_drift(inputs, tmp_path):
    cfg, chunks = inputs
    ec = sources.embed_chunks_module()
    bad = [dict(c, chunk_text=c["chunk_text"] + " (edited)") if c["chunk_id"] == "3-0-0" else c for c in chunks[4:]]
    pq.write_table(pa.Table.from_pylist(bad, schema=ec.OUT_SCHEMA), tmp_path / "chunks_embedded" / "part-00001.parquet")
    with pytest.raises(StageError) as e:
        assemble.run(cfg, run_id="r1")
    assert e.value.code == 5
    assert not LocalStore(cfg.workdir).exists("snapshots/r1/snapshot.json")
    assert assemble.run(cfg, run_id="r2", verify=False)["articles"] == 3


def test_missing_part_needs_allow_partial(inputs, tmp_path):
    cfg, _ = inputs
    os.remove(tmp_path / "chunks_embedded" / "part-00001.parquet")
    with pytest.raises(StageError) as e:
        assemble.run(cfg, run_id="r1")
    assert e.value.code == 5
    stats = assemble.run(cfg, run_id="r2", allow_partial=True)
    assert stats["articles"] == 2
    assert LocalStore(cfg.workdir).read_json("snapshots/r2/snapshot.json")["complete"] is False


def test_out_of_order_chunks_rejected(inputs, tmp_path):
    cfg, chunks = inputs
    ec = sources.embed_chunks_module()
    d = tmp_path / "chunks_embedded"
    pq.write_table(pa.Table.from_pylist(chunks[4:], schema=ec.OUT_SCHEMA), d / "part-00000.parquet")
    pq.write_table(pa.Table.from_pylist(chunks[:4], schema=ec.OUT_SCHEMA), d / "part-00001.parquet")
    with pytest.raises(StageError) as e:
        assemble.run(cfg, run_id="r1")
    assert e.value.code == 5


def test_clean_row_defaults_and_truncation():
    v = np.zeros(TEXT_DIM, np.float32)
    trunc: list[str] = []
    row = clean_row({"page_id": 1, "title": None, "revision_id": None, "article_embedding": v,
                     "chunks": [{"chunk_id": "1-0-0", "text": "é" * 5000, "embedding": v}], "images": []},
                    truncated=trunc)
    assert row["title"] == "" and row["revision_id"] == -1 and row["chunks"][0]["heading"] == ""
    assert len(row["chunks"][0]["text"].encode()) <= 8192 and trunc == ["chunks.text"]
    with pytest.raises(ValueError):
        clean_row({"page_id": 1, "article_embedding": v[:10], "chunks": [], "images": []})
    assert {f.name for f in snapshot_schema()} >= {"page_id", "chunks", "images", "content_hash"}


# ---------------------------------------------------------------------------
# upsert with a fake client
# ---------------------------------------------------------------------------
class FakeClient:
    create_schema = MilvusClient.create_schema
    create_struct_field_schema = MilvusClient.create_struct_field_schema
    prepare_index_params = MilvusClient.prepare_index_params

    def __init__(self, fail_on: set[int] = frozenset()):
        self.rows: dict[int, dict] = {}
        self.exists = False
        self.calls = 0
        self.fail_on = set(fail_on)

    def has_collection(self, name):
        return self.exists

    def create_collection(self, collection_name, schema, index_params):
        self.exists = True
        self.schema = schema

    def describe_collection(self, name):
        from pymilvus import DataType

        def field(f):
            d = {"name": f.name, "type": f.dtype, "params": dict(f.params), "is_primary": f.is_primary}
            return d
        fields = [field(f) for f in self.schema.fields]
        for s in self.schema.struct_fields:
            fields.append({"name": s.name, "type": DataType.ARRAY, "params": {"max_capacity": s.max_capacity},
                           "struct_fields": [field(f) for f in s.fields]})
        return {"fields": fields, "auto_id": False}

    def describe_index(self, name, index_name):
        return {"metric_type": "COSINE"}

    def upsert(self, collection_name, data):
        self.calls += 1
        if any(r["page_id"] in self.fail_on for r in data):
            raise RuntimeError("boom")
        for r in data:
            self.rows[r["page_id"]] = r

    def flush(self, collection_name):
        pass

    def get_collection_stats(self, name):
        return {"row_count": len(self.rows)}

    def query_iterator(self, collection_name, batch_size, filter, output_fields):
        ids = [{"page_id": p} for p in self.rows]

        class It:
            def __init__(self):
                self.done = False

            def next(self):
                if self.done:
                    return []
                self.done = True
                return ids

            def close(self):
                pass
        return It()

    def delete(self, collection_name, filter):
        for pid in json.loads(filter.split(" in ", 1)[1]):
            self.rows.pop(pid, None)


def test_upsert_resume_diff_failure_replay_and_prune(inputs, monkeypatch):
    cfg, _ = inputs
    monkeypatch.setattr("article_db.collection.RETRY_DELAYS", ())
    store = LocalStore(cfg.workdir)
    assemble.run(cfg, run_id="s1")
    cfg.batch_size = 1

    client = FakeClient(fail_on={2})
    st = upsert.run(cfg, store, run_id="u1", client=client)
    assert st["created_collection"] and st["upserted"] == 2 and st["failed_batches"] == 1
    assert set(client.rows) == {1, 3}
    row = client.rows[1]
    assert isinstance(row["created_at"], int) and len(row["chunks"][0]["embedding"]) == TEXT_DIM

    client.fail_on = set()
    st = upsert.replay(cfg, store, run_id="u2", client=client)
    assert st == {"files": 1, "upserted": 1, "failed_batches": 0} and set(client.rows) == {1, 2, 3}

    client.calls = 0
    st = upsert.run(cfg, store, run_id="u3", client=client)  # schema verified, nothing changed
    assert st["upserted"] == 0 and st["unchanged"] == 3 and client.calls == 0

    cfg.max_images = 1  # only Tokyo changes (loses its 2nd image) → only Tokyo is re-sent
    assemble.run(cfg, run_id="s2")
    client.rows[99] = {"page_id": 99}
    st = upsert.run(cfg, store, run_id="u4", client=client, prune=True)
    assert st["upserted"] == 1 and len(client.rows[1]["images"]) == 1 and st["pruned"] == 1 and 99 not in client.rows
    status = upsert.status(cfg, store, client=client)
    assert status["snapshot"]["pending"] == 0 and status["collection_rows"] == 3


def test_upsert_batches_by_bytes(inputs):
    cfg, _ = inputs
    store = LocalStore(cfg.workdir)
    assemble.run(cfg, run_id="s1")
    rows = list(upsert.iter_snapshot_rows(store, "s1"))
    cfg.max_batch_bytes = max(row_bytes(r) for r in rows) + 1  # every article alone
    client = FakeClient()
    upsert.run(cfg, store, run_id="u1", client=client)
    assert client.calls == 3


def test_schema_mismatch_exit_3(inputs):
    cfg, _ = inputs
    store = LocalStore(cfg.workdir)
    assemble.run(cfg, run_id="s1")
    client = FakeClient()
    upsert.run(cfg, store, run_id="u1", client=client)
    cfg.max_chunks = 1024  # new snapshot wants capacity 1024, the collection has 512
    assemble.run(cfg, run_id="s2")
    with pytest.raises(StageError) as e:
        upsert.run(cfg, store, run_id="u2", client=client)
    assert e.value.code == 3


@pytest.mark.milvus
def test_real_zilliz_roundtrip(inputs):
    """Opt-in (-m milvus): synthetic articles with images → real Zilliz test collection → search → drop."""
    from dotenv import load_dotenv

    from article_db.upsert import open_collection

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    cfg, _ = inputs
    cfg.zilliz_uri, cfg.zilliz_token = os.environ.get("ZILLIZ_URI", ""), os.environ.get("ZILLIZ_TOKEN", "")
    if not cfg.zilliz_uri:
        pytest.skip("ZILLIZ_URI not set")
    cfg.collection = "wikivoyage_articles_pytest"
    store = LocalStore(cfg.workdir)
    assemble.run(cfg, run_id="s1")
    coll = open_collection(cfg)
    if coll.client.has_collection(cfg.collection):
        coll.client.drop_collection(cfg.collection)
    try:
        st = upsert.run(cfg, store, run_id="u1")
        assert st["upserted"] == 3 and st["failed_batches"] == 0
        assert upsert.run(cfg, store, run_id="u2")["upserted"] == 0  # verify_schema passes on the real server
        tokyo = snapshot_rows(cfg, "s1")[1]
        res = coll.client.search(cfg.collection, data=[tokyo["images"][1]["embedding"]], anns_field="images[embedding]",
                                 limit=1, output_fields=["title"], consistency_level="Strong")
        assert res[0][0]["page_id"] == 1 and res[0][0]["offset"] == 1
        got = coll.client.query(cfg.collection, filter="page_id in [1, 2]", output_fields=["page_id", "images"],
                                consistency_level="Strong")
        by_id = {r["page_id"]: r for r in got}
        assert by_id[2]["images"] == [] and by_id[1]["images"][0]["license"] == "CC BY-SA 4.0"
    finally:
        coll.client.drop_collection(cfg.collection)
