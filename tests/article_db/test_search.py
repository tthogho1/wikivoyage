"""article_db.search: encoder routing, RRF, grouping, offset slicing and filters (no torch, no network)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from article_db.search import (ArticleSearcher, SearchFilters, _Elem, element_limit, rrf_merge)

ROWS = {
    1: {"page_id": 1, "title": "Aachen", "url": "u1", "page_type": "city", "status": "usable", "is_part_of": "NRW",
        "chunks": [{"chunk_index": i, "section_path": f"S{i}", "url": f"u1#{i}", "text": f"Aachen ({i})\nbody {i}"}
                   for i in range(5)],
        "images": [{"image_url": f"img1-{i}", "image_page_url": "", "caption": f"cap {i}", "section_path": "",
                    "license": "CC BY-SA 4.0", "license_url": "", "artist": "A", "file_title": f"File:{i}.jpg"}
                   for i in range(3)]},
    2: {"page_id": 2, "title": "Bonn", "url": "u2", "page_type": "city", "status": "", "is_part_of": "NRW",
        "chunks": [{"chunk_index": 0, "section_path": "", "url": "u2", "text": "Bonn\nintro"}], "images": []},
    3: {"page_id": 3, "title": "Cologne", "url": "u3", "page_type": "city", "status": "", "is_part_of": "NRW",
        "chunks": [], "images": [{"image_url": "img3-0", "caption": "", "license": "", "artist": "",
                                  "image_page_url": "", "section_path": "", "license_url": "", "file_title": "F"}]},
}


class FakeClient:
    def __init__(self, hits: dict[str, list[tuple[int, int, float]]]):
        self.hits = hits
        self.searches: list[dict] = []
        self.queries: list[dict] = []

    def search(self, **kw):
        self.searches.append(kw)
        return [[{"page_id": p, "offset": o, "distance": d, "entity": {"page_id": p}}
                 for p, o, d in self.hits.get(kw["anns_field"], [])]]

    def query(self, **kw):
        self.queries.append(kw)
        ids = eval(kw["filter"].split(" in ", 1)[1])  # noqa: S307 — test-only, our own string
        return [ROWS[i] for i in ids if i in ROWS]


class FakeText:
    def __init__(self):
        self.calls = []

    def embed_queries(self, texts):
        self.calls.append(texts)
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeSiglip:
    def __init__(self):
        self.text_calls, self.image_calls = [], []

    def embed_texts(self, texts):
        self.text_calls.append(texts)
        return np.ones((len(texts), 3), dtype=np.float32)

    def embed_images(self, images):
        self.image_calls.append(images)
        return np.ones((len(images), 3), dtype=np.float32)


HITS = {
    "chunks[embedding]": [(1, 3, 0.80), (2, 0, 0.70), (1, 1, 0.65), (1, 3, 0.60)],
    "images[embedding]": [(3, 0, 0.20), (1, 2, 0.15)],
}


def make(hits=HITS):
    client, text, sig = FakeClient(hits), FakeText(), FakeSiglip()
    return ArticleSearcher(client, "coll", text_embedder=text, image_embedder=sig), client, text, sig


def img():
    return Image.new("RGB", (32, 32), (0, 0, 255))


# ---------------------------------------------------------------- routing
def test_text_only_uses_qwen_and_siglip_text():
    s, client, text, sig = make()
    res = s.search("cheap hostel")
    assert res.mode == "text" and res.encoders == ["qwen3", "siglip2-text"]
    assert text.calls == [["cheap hostel"]] and sig.text_calls == [["cheap hostel"]] and not sig.image_calls
    assert [c["anns_field"] for c in client.searches] == ["chunks[embedding]", "images[embedding]"]


def test_image_only_uses_siglip_image_and_intro_chunks():
    s, client, text, sig = make()
    res = s.search(None, img())
    assert res.mode == "image" and res.encoders == ["siglip2-image"]
    assert not text.calls and not sig.text_calls and len(sig.image_calls) == 1
    assert [c["anns_field"] for c in client.searches] == ["images[embedding]"]
    top = res.articles[0]
    assert top.title == "Cologne" and top.chunks == [] and top.images[0].image_url == "img3-0"
    aachen = next(a for a in res.articles if a.page_id == 1)
    assert [c.chunk_index for c in aachen.chunks] == [0, 1, 2]  # intro chunks, no Qwen3 vector


def test_text_and_image_uses_qwen_and_siglip_image():
    s, client, text, sig = make()
    res = s.search("where to stay", img())
    assert res.mode == "text+image" and res.encoders == ["siglip2-image", "qwen3"] and res.ranking == "photo"
    assert not sig.text_calls and len(sig.image_calls) == 1 and text.calls == [["where to stay"]]


def test_text_and_photo_ranks_by_photo_and_searches_text_inside():
    s, client, *_ = make()
    res = s.search("where to stay", img(), top_k=2)
    # Photo list: Cologne (3) then Aachen (1); text alone would have put Aachen and Bonn first.
    assert [a.page_id for a in res.articles] == [3, 1]
    image_search, text_search = client.searches
    assert image_search["anns_field"] == "images[embedding]" and image_search["filter"] == ""
    assert text_search["anns_field"] == "chunks[embedding]" and text_search["filter"] == "page_id in [3, 1]"
    aachen = res.articles[1]
    assert [c.offset for c in aachen.chunks] == [3, 1]       # passages picked by the text
    assert [c.chunk_index for c in res.articles[0].chunks] == []  # Cologne has no chunks at all


def test_photo_first_can_be_turned_off():
    client = FakeClient(HITS)
    s = ArticleSearcher(client, "coll", text_embedder=FakeText(), image_embedder=FakeSiglip(), photo_first=False)
    res = s.search("where to stay", img())
    assert res.ranking == "fused" and res.encoders == ["qwen3", "siglip2-image"]
    assert res.articles[0].page_id == 1


def test_mode_text_skips_images():
    s, client, text, sig = make()
    res = s.search("x", mode="text")
    assert res.encoders == ["qwen3"] and not sig.text_calls
    assert all(a.images == [] for a in res.articles)


def test_needs_some_input():
    s, *_ = make()
    with pytest.raises(ValueError):
        s.search("  ", None)


def test_missing_encoder_is_an_error():
    s = ArticleSearcher(FakeClient(HITS), "coll", text_embedder=FakeText(), image_embedder=None)
    with pytest.raises(ValueError, match="image embedder"):
        s.search(None, img())


# ---------------------------------------------------------------- merge + hydrate
def test_rrf_uses_best_element_rank_and_weights():
    lists = {"text": [_Elem(1, 0, .9), _Elem(1, 1, .8), _Elem(2, 0, .7)], "image": [_Elem(2, 0, .2)]}
    ranked = dict(rrf_merge(lists, {"text": 1.0, "image": 0.5}, k=60))
    assert ranked[1] == pytest.approx(1 / 61)
    assert ranked[2] == pytest.approx(1 / 62 + 0.5 / 61)
    assert list(dict(rrf_merge(lists, {"text": 1.0, "image": 0.0}, k=60))) == [1, 2]


def test_matched_chunks_are_sliced_by_offset_and_deduplicated():
    s, client, *_ = make()
    res = s.search("x", mode="text")
    aachen = res.articles[0]
    assert aachen.title == "Aachen"
    assert [(c.offset, c.section_path, c.score) for c in aachen.chunks] == [(3, "S3", 0.8), (1, "S1", 0.65)]
    assert len(client.queries) == 1 and "chunks[text]" in client.queries[0]["output_fields"]
    assert not any(f.startswith("images[") for f in client.queries[0]["output_fields"])


def test_top_k_and_out_of_range_offsets():
    hits = {"chunks[embedding]": [(1, 99, .9), (2, 0, .8), (3, 0, .7)], "images[embedding]": []}
    s, client, *_ = make(hits)
    res = s.search("x", top_k=2, mode="text")
    assert [a.page_id for a in res.articles] == [1, 2]
    # Offset beyond the stored array is dropped (no IndexError); with no match left, the intro is used.
    assert [c.chunk_index for c in res.articles[0].chunks] == [0, 1, 2]
    assert "[1, 2]" in client.queries[0]["filter"]


def test_to_dict_shape():
    s, *_ = make()
    d = s.search("x").to_dict()
    assert set(d) == {"query", "articles", "images"} and d["query"]["encoders"] == ["qwen3", "siglip2-text"]
    assert d["query"]["ranking"] == "fused" and {"page_id", "title", "url", "image_url"} <= set(d["images"][0])
    assert {"page_id", "title", "url", "score", "chunks", "images"} <= set(d["articles"][0])


# ---------------------------------------------------------------- filters
def test_filters_go_to_the_right_searches():
    s, client, *_ = make()
    s.search("x", filters=SearchFilters(page_type="city", heading='Sl"eep'))
    chunk_f, image_f = (c["filter"] for c in client.searches)
    assert chunk_f == 'page_type == "city" && element_filter(chunks, $[top_section] == "Sl\\"eep")'
    assert image_f == 'page_type == "city"'  # no heading on images


def test_filter_expr_empty_and_combined():
    assert SearchFilters().chunk_expr() == ""
    assert SearchFilters(heading="See").chunk_expr() == 'element_filter(chunks, $[top_section] == "See")'
    assert SearchFilters(is_part_of="NRW", title="Bonn").article_expr() == 'is_part_of == "NRW" && title == "Bonn"'


def test_element_limit_bounds():
    assert element_limit(1, 1) == 50 and element_limit(5, 3) == 50 and element_limit(20, 3) == 180
    assert element_limit(1000, 3) == 1000


# ---------------------------------------------------------------- gallery
def test_gallery_shows_image_hits_outside_the_top_articles():
    hits = {"chunks[embedding]": [(1, 0, .9), (2, 0, .8)],
            "images[embedding]": [(3, 0, .30), (1, 2, .20), (1, 1, .19), (1, 0, .18)]}
    s, client, *_ = make(hits)
    res = s.search("x", top_k=1)
    assert [a.page_id for a in res.articles] == [1]
    # Cologne is not in the top 1 but its image is the best match; at most 2 images per article.
    assert [(m.page_id, m.offset, m.title) for m in res.images] == [(3, 0, "Cologne"), (1, 2, "Aachen"), (1, 1, "Aachen")]
    assert "page_id in [3]" in client.queries[1]["filter"]  # only the missing article is fetched again


def test_gallery_size_and_empty_without_image_search():
    s, *_ = make()
    assert s.search("x", mode="text").images == []
    s.gallery_size = 1
    assert len(s.search("x").images) == 1
