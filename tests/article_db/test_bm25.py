"""Hybrid text search (Qwen3 dense + BM25) and the BM25 mirror build (no torch, no network)."""
from __future__ import annotations

import copy


from article_db import bm25
from article_db.config import ArticlesConfig
from article_db.search import ArticleSearcher, SearchFilters

from .test_search import ROWS, FakeSiglip, FakeText

HASH = {1: "h1", 2: "h2", 3: "h3"}


class HybridClient:
    """Dense hits by anns_field; BM25 hits as plain-collection hits with an entity."""

    def __init__(self, dense, sparse, rows=None):
        self.dense, self.sparse = dense, sparse
        self.rows = rows or {pid: {**r, "content_hash": HASH[pid]} for pid, r in ROWS.items()}
        self.searches: list[dict] = []
        self.queries: list[dict] = []

    def search(self, **kw):
        self.searches.append(kw)
        if kw["anns_field"] == "sparse":
            return [[{"id": p * 1024 + o, "distance": d,
                      "entity": {"page_id": p, "chunk_offset": o, "content_hash": h}}
                     for p, o, d, h in self.sparse]]
        return [[{"page_id": p, "offset": o, "distance": d, "entity": {"page_id": p}}
                 for p, o, d in self.dense.get(kw["anns_field"], [])]]

    def query(self, **kw):
        self.queries.append(kw)
        ids = eval(kw["filter"].split(" in ", 1)[1])  # noqa: S307 — test-only, our own string
        return [self.rows[i] for i in ids if i in self.rows]


DENSE = {"chunks[embedding]": [(2, 0, 0.70), (1, 1, 0.65)], "images[embedding]": []}
# "Aachen Cathedral": only BM25 finds Aachen chunk 4, and ranks it first.
SPARSE = [(1, 4, 12.5, "h1"), (1, 1, 7.0, "h1"), (2, 0, 3.0, "h2")]


def make(dense=DENSE, sparse=SPARSE, rows=None, **kw):
    client = HybridClient(dense, sparse, rows)
    s = ArticleSearcher(client, "coll", text_embedder=FakeText(), image_embedder=FakeSiglip(),
                        bm25_collection="bm25coll", **kw)
    return s, client


def test_text_search_is_hybrid():
    s, client = make()
    res = s.search("Aachen Cathedral", mode="text")
    assert res.encoders == ["qwen3", "bm25"]
    sparse = [c for c in client.searches if c["anns_field"] == "sparse"][0]
    assert sparse["collection_name"] == "bm25coll" and sparse["data"] == ["Aachen Cathedral"]
    assert sparse["search_params"] == {"metric_type": "BM25"}
    # Aachen: dense rank 2 + BM25 rank 1; Bonn: dense rank 1 + BM25 rank 2 → equal RRF, page_id breaks the tie.
    assert [a.page_id for a in res.articles] == [1, 2]
    assert res.articles[0].score == res.articles[1].score


def test_bm25_only_chunk_is_returned_with_both_scores():
    s, _ = make()
    aachen = next(a for a in s.search("Aachen Cathedral", mode="text").articles if a.page_id == 1)
    by_off = {c.offset: c for c in aachen.chunks}
    assert list(by_off) == [1, 4]           # chunk 1 is in both lists → fused first
    assert by_off[4].score == 0.0 and by_off[4].bm25 == 12.5
    assert by_off[1].score == 0.65 and by_off[1].bm25 == 7.0
    assert by_off[4].section_path == "S4"


def test_keyword_off_and_no_collection_are_dense_only():
    s, client = make()
    assert s.search("x", mode="text", keyword=False).encoders == ["qwen3"]
    assert not any(c["anns_field"] == "sparse" for c in client.searches)
    plain = ArticleSearcher(HybridClient(DENSE, SPARSE), "coll", text_embedder=FakeText(), image_embedder=FakeSiglip())
    assert plain.search("x", mode="text").encoders == ["qwen3"]


def test_image_query_does_not_use_bm25():
    from PIL import Image

    s, client = make()
    res = s.search(None, Image.new("RGB", (8, 8)))
    assert res.encoders == ["siglip2-image"]
    assert not any(c["anns_field"] == "sparse" for c in client.searches)


def test_stale_bm25_hits_are_ignored():
    rows = {pid: {**r, "content_hash": HASH[pid]} for pid, r in ROWS.items()}
    rows[1]["content_hash"] = "newer"
    s, _ = make(rows=rows)
    aachen = next(a for a in s.search("x", mode="text").articles if a.page_id == 1)
    assert [c.offset for c in aachen.chunks] == [1] and aachen.chunks[0].bm25 == 0.0


def test_bm25_failure_falls_back_to_dense():
    s, client = make()

    def boom(**kw):
        if kw["anns_field"] == "sparse":
            raise RuntimeError("collection not loaded")
        return HybridClient.search(client, **kw)

    client.search = boom
    res = s.search("x", mode="text")
    assert [c.offset for c in res.articles[0].chunks]  # still answers from the dense list


def test_bm25_filter_uses_flat_top_section():
    f = SearchFilters(page_type="city", heading="Sleep")
    assert f.bm25_expr() == 'page_type == "city" && top_section == "Sleep"'
    s, client = make()
    s.search("x", filters=f, mode="text")
    sparse = [c for c in client.searches if c["anns_field"] == "sparse"][0]
    assert sparse["filter"] == 'page_type == "city" && top_section == "Sleep"'


def test_bm25_weight_zero_keeps_dense_order():
    s, _ = make(w_bm25=0.0)
    assert [a.page_id for a in s.search("x", mode="text").articles] == [2, 1]


# ---------------------------------------------------------------- build
def test_chunk_rows_follow_array_order():
    art = {"page_id": 7, "content_hash": "c", "title": "T", "page_type": "city", "is_part_of": "P",
           "chunks": [{"chunk_index": 5, "top_section": "See", "text": "a"},
                      {"chunk_index": 2, "top_section": "Eat", "text": "b" * 9000}]}
    rows = bm25.chunk_rows(art)
    assert [(r["id"], r["chunk_offset"], r["chunk_index"]) for r in rows] == [(7168, 0, 5), (7169, 1, 2)]
    assert len(rows[1]["text"]) == 8192 and rows[0]["content_hash"] == "c"


class BuildClient:
    def __init__(self, exists=False):
        self.exists, self.inserted, self.deleted = exists, [], []

    def has_collection(self, name):
        return self.exists

    def create_collection(self, **kw):
        self.exists = True

    def create_schema(self, **kw):
        from pymilvus import MilvusClient

        return MilvusClient.create_schema(**kw)

    def prepare_index_params(self):
        from pymilvus import MilvusClient

        return MilvusClient.prepare_index_params()

    def describe_collection(self, name):
        return {"fields": [{"name": n, "type": k, "params": {"max_length": m}} for n, k, m in bm25.FIELDS],
                "functions": [{"type": "BM25"}]}

    def delete(self, **kw):
        self.deleted.append(kw["filter"])

    def insert(self, **kw):
        self.inserted += kw["data"]

    def flush(self, **kw):
        pass

    def get_collection_stats(self, name):
        return {"row_count": len(self.inserted)}


def _snapshot(tmp_path, articles):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from image_embedding.storage import open_store

    store = open_store(str(tmp_path))
    store.write_json("snapshots/S1/snapshot.json", {"complete": True})
    full = [{**a, "chunks": [{**c, "embedding": [0.0] * 4} for c in a["chunks"]]} for a in articles]
    buf = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(full), buf)
    store.write_bytes("snapshots/S1/part-00000.parquet", buf.getvalue().to_pybytes())
    return store


ARTS = [{"page_id": pid, "content_hash": HASH[pid], "title": r["title"], "page_type": "city", "is_part_of": "NRW",
         "chunks": [{"chunk_index": c["chunk_index"], "top_section": "S", "text": c["text"]} for c in r["chunks"]]}
        for pid, r in ROWS.items()]


def test_build_sends_changed_articles_only(tmp_path):
    store = _snapshot(tmp_path, ARTS)
    cfg = ArticlesConfig()
    client = BuildClient()
    st = bm25.run(cfg, store, run_id="r1", client=client)
    assert st["created_collection"] and st["rows_sent"] == 6 and not client.deleted  # 5 + 1 + 0 chunks
    assert {r["page_id"] for r in client.inserted} == {1, 2}

    # Second run: nothing changed.
    client2 = BuildClient(exists=True)
    assert bm25.run(cfg, store, run_id="r2", client=client2)["articles_sent"] == 0

    # Aachen changes: its rows are deleted and re-inserted, Bonn is untouched.
    arts = copy.deepcopy(ARTS)
    arts[0]["content_hash"] = "h1b"
    arts[0]["chunks"] = arts[0]["chunks"][:2]
    store3 = _snapshot(tmp_path, arts)
    client3 = BuildClient(exists=True)
    st = bm25.run(cfg, store3, run_id="r3", client=client3)
    assert st["articles_sent"] == 1 and client3.deleted == ["page_id in [1]"]
    assert [r["chunk_offset"] for r in client3.inserted] == [0, 1]


def test_schema_has_bm25_function():
    schema = bm25.Bm25Collection(BuildClient()).build_schema()
    fields = {f.name: f for f in schema.fields}
    assert list(fields)[0] == "id" and fields["id"].is_primary and "sparse" in fields
    assert [fn.name for fn in schema.functions] == ["text_bm25"]
