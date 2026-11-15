"""End-to-end run of every stage on a mini dump, with HTTP mocked and Milvus Lite as Zilliz."""
from __future__ import annotations

import re
from urllib.parse import parse_qs

import pytest

from image_embedding import commons, embed, extract, fetch, ids, upsert
from image_embedding.__main__ import main as cli_main
from image_embedding.config import COMMONS_API, StageError
from image_embedding.search import ImageSearcher
from image_embedding.status import collect
from image_embedding.zilliz import ImageCollection

from .conftest import FakeEmbedder, jpeg_bytes, png_bytes

COLORS = {
    "Tokyo banner.jpg": (250, 10, 10), "Shibuya crossing.jpg": (10, 250, 10), "Sensoji.jpg": (10, 10, 250),
    "Gallery one.jpg": (200, 200, 10), "Kinkakuji.jpg": (10, 200, 200),
}


def _thumb(name: str) -> str:
    return "https://upload.test/thumb/" + name.replace(" ", "_")


def _page(title: str) -> dict:
    name = title.split(":", 1)[1]
    if name == "Does not exist.jpg":
        return {"title": title, "missing": True}
    mime = "image/gif" if name.endswith(".gif") else "image/jpeg"
    return {"title": title, "imagerepository": "shared", "imageinfo": [{
        "url": "https://upload.test/orig/" + name, "thumburl": _thumb(name), "thumbwidth": 300,
        "thumbheight": 200, "width": 3000, "height": 2000, "mime": mime, "size": 1000,
        "sha1": ids.file_key(title), "descriptionurl": "https://commons.wikimedia.org/wiki/" + title,
        "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"}, "Artist": {"value": "<b>Ann</b>"}},
    }]}


@pytest.fixture
def http(requests_mock):
    def api(request, context):
        titles = parse_qs(request.text)["titles"][0].split("|")
        return {"query": {"pages": [_page(t) for t in titles]}}

    def local_api(request, context):
        titles = parse_qs(request.text)["titles"][0].split("|")
        return {"query": {"pages": [{"title": t, "missing": True} for t in titles]}}

    def image(request, context):
        name = request.url.rsplit("/", 1)[-1].replace("_", " ")
        context.headers["Content-Type"] = "image/jpeg"
        if name == "Kinkakuji.jpg":
            return png_bytes((50, 50))  # too small → skipped
        return jpeg_bytes((300, 200), COLORS[name])

    requests_mock.post(COMMONS_API, json=api)
    requests_mock.post("https://en.wikivoyage.org/w/api.php", json=local_api)
    requests_mock.get(re.compile(r"https://upload\.test/thumb/.*"), content=image)
    return requests_mock


@pytest.fixture
def milvus(tmp_path):
    from pymilvus import MilvusClient

    client = MilvusClient(str(tmp_path / "milvus.db"))
    yield client
    client.close()


def _count(client, name="wikivoyage_images") -> int:
    return len(client.query(name, filter="id > 0", output_fields=["id"], limit=1000))


def test_full_pipeline(cfg, store, mini_dump, http, milvus):
    # extract ------------------------------------------------------------
    s = extract.run(cfg, store, dump_path=mini_dump)
    assert s["articles"] == 4  # Tokyo, Kyoto, No images, Missing image (redirect/talk/empty skipped)
    refs = list(store.iter_jsonl(cfg.refs_rel()))
    assert [(r["source_page_title"], r["file_title"]) for r in refs] == [
        ("Tokyo", "File:Tokyo banner.jpg"), ("Tokyo", "File:Shibuya crossing.jpg"),
        ("Tokyo", "File:Sensoji.jpg"), ("Tokyo", "File:Gallery one.jpg"), ("Tokyo", "File:Animated.gif"),
        ("Kyoto", "File:Kinkakuji.jpg"), ("Kyoto", "File:Shibuya crossing.jpg"),
        ("Missing image", "File:Does not exist.jpg"),
    ]
    assert refs[0]["page_url"] == "https://en.wikivoyage.org/wiki/Tokyo" and refs[0]["revision_id"] == 100

    # commons ------------------------------------------------------------
    s = commons.run(cfg, store, run_id="20261101T000000Z-0001")
    assert (s["status_ok"], s["status_skipped"], s["status_missing"]) == (5, 1, 1)
    n_api = len([r for r in http.request_history if r.method == "POST"])
    assert commons.run(cfg, store, run_id="20261101T000001Z-0002")["queried"] == 0
    assert len([r for r in http.request_history if r.method == "POST"]) == n_api

    # fetch --------------------------------------------------------------
    s = fetch.run(cfg, store, run_id="20261101T000002Z-0003")
    assert (s["status_ok"], s["status_skipped"]) == (4, 1)
    n_get = len([r for r in http.request_history if r.method == "GET"])
    assert fetch.run(cfg, store, run_id="20261101T000003Z-0004")["attempted"] == 0
    assert len([r for r in http.request_history if r.method == "GET"]) == n_get
    assert store.exists(cfg.processed_rel(ids.file_key("File:Sensoji.jpg")))

    # embed --------------------------------------------------------------
    emb = FakeEmbedder()
    s = embed.run(cfg, store, run_id="20261101T000004Z-0005", embedder=emb)
    assert (s["embedded"], s["errors"]) == (4, 0)
    # image files are deleted by default once their vectors are written (raw + processed per image)
    assert s["deleted_files"] == 8 and store.list("images/raw") == [] and store.list("images/processed") == []
    assert embed.run(cfg, store, run_id="20261101T000005Z-0006", embedder=emb)["attempted"] == 0
    assert fetch.run(cfg, store, run_id="20261101T000005Z-0007")["attempted"] == 0  # no re-download

    # upsert -------------------------------------------------------------
    s = upsert.run(cfg, store, run_id="20261101T000006Z-0007", client=milvus)
    assert s["upserted"] == 5 and s["failed_batches"] == 0  # Shibuya counted for both articles
    assert _count(milvus) == 5
    row = milvus.query("wikivoyage_images", filter='page_title == "Kyoto"',
                       output_fields=["image_title", "license", "artist", "caption", "page_url"])[0]
    assert row["image_title"] == "File:Shibuya crossing.jpg" and row["license"] == "CC BY-SA 4.0"
    assert row["artist"] == "Ann" and row["caption"] == "Same image, other article"

    # re-run is a no-op and does not duplicate --------------------------
    assert upsert.run(cfg, store, run_id="20261101T000007Z-0008", client=milvus)["upserted"] == 0
    assert _count(milvus) == 5

    # search -------------------------------------------------------------
    searcher = ImageSearcher(milvus, emb)
    from PIL import Image
    hits = searcher.search_by_image(Image.new("RGB", (300, 200), COLORS["Sensoji.jpg"]), top_k=3)
    assert hits[0].image_title == "File:Sensoji.jpg" and hits[0].score > 0.99
    hits = searcher.search_by_image(Image.new("RGB", (300, 200), COLORS["Sensoji.jpg"]), page_title="Kyoto")
    assert {h.page_title for h in hits} == {"Kyoto"}
    assert len(searcher.search_by_text("temple", top_k=2)) == 2

    st = collect(cfg, store)
    assert st["upsert"]["upserted_ids"] == 5 and st["embed"]["not_embedded"] == 0


def test_failed_batch_replay_and_prune(cfg, store, mini_dump, http, milvus, monkeypatch):
    extract.run(cfg, store, dump_path=mini_dump)
    commons.run(cfg, store, run_id="20261101T000000Z-0001")
    fetch.run(cfg, store, run_id="20261101T000001Z-0002")
    embed.run(cfg, store, run_id="20261101T000002Z-0003", embedder=FakeEmbedder())
    cfg.zilliz_batch_size = 2

    real = ImageCollection.upsert
    calls = {"n": 0}

    def flaky(self, rows, sleep=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("zilliz down")
        return real(self, rows, sleep=lambda s: None)

    monkeypatch.setattr(ImageCollection, "upsert", flaky)
    s = upsert.run(cfg, store, run_id="20261101T000003Z-0004", client=milvus, prune=True)
    assert s["failed_batches"] == 1 and s["failed_rows"] == 2 and s["pruned"] == "skipped"
    assert _count(milvus) == 3
    monkeypatch.setattr(ImageCollection, "upsert", real)

    r = upsert.replay(cfg, store, run_id="20261101T000004Z-0005", client=milvus)
    assert (r["upserted"], r["failed_batches"]) == (2, 0) and _count(milvus) == 5
    assert store.list(f"{cfg.upsert_dir()}/failed") == []

    # A newer dump in which Kyoto lost its images: prune removes the stale Kyoto rows.
    cfg.dump_date = "20261201"
    from .conftest import MINI_PAGES, make_dump
    pages = [p if p[2] != "Kyoto" else (2, 0, "Kyoto", "Kyoto, no pictures now.", None) for p in MINI_PAGES]
    dump2 = store.uri("new.xml.bz2")
    with open(dump2, "wb") as fh:
        fh.write(make_dump(pages))
    extract.run(cfg, store, dump_path=dump2)
    commons.run(cfg, store, run_id="20261201T000000Z-0001")
    assert fetch.run(cfg, store, run_id="20261201T000001Z-0002")["attempted"] == 0  # shared image cache
    s = upsert.run(cfg, store, run_id="20261201T000002Z-0003", client=milvus, prune=True)
    assert s["upserted"] == 4 and s["pruned"] is True
    rows = milvus.query("wikivoyage_images", filter="id > 0", output_fields=["page_title", "dump_date"])
    assert {r["dump_date"] for r in rows} == {"20261201"} and "Kyoto" not in {r["page_title"] for r in rows}


def test_embed_isolates_bad_image(cfg, store, mini_dump, http):
    extract.run(cfg, store, dump_path=mini_dump)
    commons.run(cfg, store, run_id="20261101T000000Z-0001")
    fetch.run(cfg, store, run_id="20261101T000001Z-0002")
    s = embed.run(cfg, store, run_id="20261101T000002Z-0003", embedder=FakeEmbedder(fail_on=300))
    assert s["embedded"] == 0 and s["errors"] == 4  # every image is 300px wide here → each fails alone
    assert s["deleted_files"] == 0 and len(store.list("images/processed")) == 4  # kept for the retry


def test_schema_mismatch_is_exit_3(cfg, store, mini_dump, http, milvus):
    extract.run(cfg, store, dump_path=mini_dump)
    commons.run(cfg, store, run_id="20261101T000000Z-0001")
    fetch.run(cfg, store, run_id="20261101T000001Z-0002")
    embed.run(cfg, store, run_id="20261101T000002Z-0003", embedder=FakeEmbedder(dim=8))
    ImageCollection(milvus, "wikivoyage_images", dim=16).ensure()
    with pytest.raises(StageError) as e:
        upsert.run(cfg, store, run_id="20261101T000003Z-0004", client=milvus)
    assert e.value.code == 3 and "dim" in str(e.value)


def test_cli_status_and_missing_date(cfg, store, mini_dump, http, capsys, monkeypatch):
    monkeypatch.delenv("IMAGES_DUMP_DATE", raising=False)
    assert cli_main(["extract", "--workdir", cfg.workdir]) == 1
    assert cli_main(["extract", "--workdir", cfg.workdir, "--dump-date", "20261101", "--dump-path", mini_dump]) == 0
    assert cli_main(["status", "--workdir", cfg.workdir, "--dump-date", "20261101"]) == 0
    assert '"refs": 8' in capsys.readouterr().out
    assert store.list(cfg.summary_dir())


def test_embed_keep_images(cfg, store, mini_dump, http):
    extract.run(cfg, store, dump_path=mini_dump)
    commons.run(cfg, store, run_id="20261101T000000Z-0001")
    fetch.run(cfg, store, run_id="20261101T000001Z-0002")
    cfg.keep_images = True
    s = embed.run(cfg, store, run_id="20261101T000002Z-0003", embedder=FakeEmbedder())
    assert s["embedded"] == 4 and s["deleted_files"] == 0 and len(store.list("images/processed")) == 4
    cfg.keep_images = False  # a later run sweeps files of images embedded earlier
    s = embed.run(cfg, store, run_id="20261101T000003Z-0004", embedder=FakeEmbedder())
    assert s["attempted"] == 0 and s["deleted_files"] == 8 and store.list("images/raw") == []
