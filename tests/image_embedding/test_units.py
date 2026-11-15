from __future__ import annotations

import io

import pytest
import requests
from PIL import Image

from image_embedding import ids
from image_embedding.commons import build_row, html_to_text, query_imageinfo
from image_embedding.http import HttpError, request_with_retry
from image_embedding.preprocess import ImageRejected, load_and_validate, to_rgb
from image_embedding.storage import LocalStore, PartWriter, S3Store, last_wins
from image_embedding.wikitext_images import extract_refs, normalize_file_title
from image_embedding.zilliz import ImageCollection, truncate_utf8
from wikivoyage_stream import Page, filters

from .conftest import jpeg_bytes, png_bytes


# ---------------------------------------------------------------- titles / refs
@pytest.mark.parametrize("raw,expected", [
    ("image:foo_bar.JPG", "File:Foo bar.JPG"),
    ("File:Foo%20Bar.jpg", "File:Foo Bar.jpg"),
    ("  FILE :  x  y.png ", "File:X y.png"),
    ("Caf&eacute;.jpg", "File:Café.jpg"),
    ("File:A.jpg#section", "File:A.jpg"),
    ("{{{image}}}.jpg", None),
    ("File:", None),
    ("File:bad|name.jpg", None),
    ("x" * 300 + ".jpg", None),
])
def test_normalize_file_title(raw, expected):
    assert normalize_file_title(raw) == expected


def test_extract_refs_all_sources_order_and_dedup():
    text = (
        "{{pagebanner|Banner.jpg}}\n"
        "[[File:A_b.jpg|thumb|left|200px|Caption with [[Link|label]]]]\n"
        "[[:File:NotEmbedded.jpg]] [[Media:Sound.ogg]] [[Category:X]]\n"
        "[[Image:alt.png|thumb|alt=Alt text]]\n"
        "{{infobox|photo=Info.webp|caption=Infobox caption}}\n"
        "{{see|name=Temple|image=File:Temple.jpg}}\n"
        "<gallery>\nFile:G1.jpg|One\nG2.jpg\n</gallery>\n"
        "[[File:A b.jpg|later caption]]\n"
    )
    refs, truncated = extract_refs(text)
    assert truncated == 0
    assert [r.file_title for r in refs] == [
        "File:Banner.jpg", "File:A b.jpg", "File:Alt.png", "File:Info.webp",
        "File:Temple.jpg", "File:G1.jpg", "File:G2.jpg",
    ]
    by = {r.file_title: r for r in refs}
    assert by["File:A b.jpg"].caption == "Caption with label"
    assert by["File:Alt.png"].caption == "Alt text"
    assert by["File:Info.webp"].caption == "Infobox caption"
    assert by["File:Temple.jpg"].caption == "Temple"
    assert (by["File:Temple.jpg"].template_name, by["File:Temple.jpg"].param_name) == ("see", "image")
    assert by["File:Banner.jpg"].caption is None
    assert by["File:G1.jpg"].caption == "One" and by["File:G2.jpg"].caption is None
    assert [r.position for r in refs] == list(range(1, 8))
    assert "A_b.jpg" in by["File:A b.jpg"].context


def test_extract_refs_dedup_takes_later_caption_and_caps():
    refs, _ = extract_refs("[[File:X.jpg|thumb]] [[File:X.jpg|thumb|Real caption]]")
    assert len(refs) == 1 and refs[0].caption == "Real caption"
    many = "".join(f"[[File:I{i}.jpg]]" for i in range(60))
    refs, truncated = extract_refs(many, max_per_page=50)
    assert len(refs) == 50 and truncated == 10 and refs[-1].file_title == "File:I49.jpg"


def test_extract_refs_records_unusable_titles():
    errors: list = []
    refs, _ = extract_refs("[[File:{{PAGENAME}}.jpg]] [[File:Ok.jpg]]", errors=errors)
    assert [r.file_title for r in refs] == ["File:Ok.jpg"]
    assert errors and errors[0]["reason"] == "unparsable_title"


# ---------------------------------------------------------------- filters / ids
def test_filters():
    page = lambda **kw: Page(page_id="1", title="T", namespace=kw.get("ns", "0"),  # noqa: E731
                             text=kw.get("text", "body"), redirect_title=kw.get("redir"))
    assert filters.is_article(page())
    assert not filters.is_article(page(ns="1"))
    assert not filters.is_article(page(text="  \n"))
    assert not filters.is_article(page(text="#REDIRECT [[X]]"))
    assert not filters.is_article(page(redir="X"))
    assert filters.is_redirect("\n#redirect [[Tokyo|x]]") == (True, "Tokyo")
    assert filters.page_url("New York City") == "https://en.wikivoyage.org/wiki/New_York_City"


def test_page_url_matches_get_wikivoyage():
    import get_wikivoyage

    p = Page(page_id="9", title="São Paulo (state)", namespace="0", text="x")
    assert get_wikivoyage.build_record(p, "city", None, "s")["url"] == filters.page_url(p.title)


def test_ids_deterministic():
    a = ids.record_id("enwikivoyage", 1, "File:A.jpg")
    assert a == ids.record_id("enwikivoyage", 1, "File:A.jpg")
    assert a != ids.record_id("enwikivoyage", 2, "File:A.jpg")
    assert 0 <= a < 2**63
    assert len(ids.file_key("File:A.jpg")) == 40
    assert ids.model_slug("google/siglip2-x", "abcdef0123456789") == "google__siglip2-x@abcdef012345"


# ---------------------------------------------------------------- Commons
def _ii(**kw):
    base = {"url": "https://upload/orig.jpg", "thumburl": "https://upload/thumb/768px-x.jpg",
            "thumbwidth": 768, "thumbheight": 512, "width": 2000, "height": 1333, "mime": "image/jpeg",
            "size": 12345, "sha1": "a" * 40, "timestamp": "2020-01-01T00:00:00Z",
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:X.jpg",
            "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"},
                            "Artist": {"value": '<a href="//x">Jane &amp; Co</a>'}}}
    base.update(kw)
    return base


def test_build_row_ok_and_mime_rules():
    row = build_row("File:X.jpg", {"title": "File:X.jpg", "imageinfo": [_ii()]}, "commons",
                    ["image/jpeg", "image/svg+xml"])
    assert row["status"] == "ok" and row["license_short_name"] == "CC BY-SA 4.0"
    assert row["artist"] == "Jane & Co" and row["artist_html"].startswith("<a")
    assert row["thumb_mime"] == "image/jpeg"
    gif = build_row("File:G.gif", {"title": "File:G.gif", "imageinfo": [_ii(mime="image/gif")]}, "commons", ["image/jpeg"])
    assert (gif["status"], gif["reason"]) == ("skipped", "mime_not_allowed:image/gif")
    svg = build_row("File:S.svg", {"imageinfo": [_ii(mime="image/svg+xml", thumburl="https://u/768px-S.svg.png")]},
                    "commons", ["image/svg+xml"])
    assert svg["status"] == "ok" and svg["thumb_mime"] == "image/png"
    svg2 = build_row("File:S.svg", {"imageinfo": [_ii(mime="image/svg+xml", thumburl=None)]}, "commons", ["image/svg+xml"])
    assert svg2["reason"] == "svg_no_render"
    assert build_row("File:M.jpg", {"title": "File:M.jpg", "missing": True}, "commons", [])["status"] == "missing"
    assert html_to_text("<b>a</b>\n  b") == "a b"
    tracked = build_row("File:X.jpg", {"imageinfo": [_ii(thumburl="https://t/960px-x.jpg?utm_source=c&utm_campaign=i")]},
                        "commons", ["image/jpeg"])
    assert tracked["thumb_url"] == "https://t/960px-x.jpg"


def test_query_imageinfo_normalized_redirect_continue(requests_mock):
    api = "https://api.test/w/api.php"
    responses = [
        {"json": {"continue": {"iicontinue": "x", "continue": "||"},
                  "query": {"normalized": [{"from": "File:a.jpg", "to": "File:A.jpg"}],
                            "redirects": [{"from": "File:A.jpg", "to": "File:B.jpg"}],
                            "pages": [{"title": "File:B.jpg"}, {"title": "File:C.jpg", "imageinfo": [_ii()]}]}}},
        {"json": {"query": {"pages": [{"title": "File:B.jpg", "imageinfo": [_ii(sha1="b" * 40)]}]}}},
    ]
    requests_mock.post(api, responses)
    out = query_imageinfo(requests.Session(), api, ["File:a.jpg", "File:C.jpg", "File:Z.jpg"], thumb_width=768)
    assert out["File:a.jpg"]["imageinfo"][0]["sha1"] == "b" * 40
    assert out["File:C.jpg"]["imageinfo"]
    assert out["File:Z.jpg"] is None
    assert "iicontinue=x" in requests_mock.request_history[1].text


def test_query_imageinfo_maxlag(requests_mock):
    api = "https://api.test/w/api.php"
    requests_mock.post(api, [
        {"json": {"error": {"code": "maxlag", "info": "lagged"}}, "headers": {"Retry-After": "1"}},
        {"json": {"query": {"pages": [{"title": "File:C.jpg", "imageinfo": [_ii()]}]}}},
    ])
    slept: list = []
    out = query_imageinfo(requests.Session(), api, ["File:C.jpg"], thumb_width=768, sleep=slept.append)
    assert out["File:C.jpg"] and slept == [1.0]


# ---------------------------------------------------------------- HTTP retry
def test_retry_honours_retry_after_then_succeeds(requests_mock):
    requests_mock.get("https://x.test/a", [
        {"status_code": 429, "headers": {"Retry-After": "3"}},
        {"status_code": 503},
        {"status_code": 200, "text": "ok"},
    ])
    slept: list = []
    r = request_with_retry(requests.Session(), "GET", "https://x.test/a", sleep=slept.append)
    assert r.text == "ok" and slept[0] == 3.0 and 0.8 <= slept[1] <= 2.4


def test_retry_404_is_immediate_and_exhaustion_raises(requests_mock):
    requests_mock.get("https://x.test/404", status_code=404)
    with pytest.raises(HttpError) as e:
        request_with_retry(requests.Session(), "GET", "https://x.test/404", sleep=lambda s: None)
    assert e.value.status == 404 and requests_mock.call_count == 1
    requests_mock.get("https://x.test/500", status_code=500)
    with pytest.raises(HttpError) as e:
        request_with_retry(requests.Session(), "GET", "https://x.test/500", max_attempts=3, sleep=lambda s: None)
    assert e.value.status == 500


# ---------------------------------------------------------------- preprocess
def test_preprocess_rules():
    with pytest.raises(ImageRejected) as e:
        load_and_validate(b"")
    assert e.value.reason == "zero_bytes"
    with pytest.raises(ImageRejected) as e:
        load_and_validate(b"<html>error</html>")
    assert (e.value.reason, e.value.status) == ("corrupt", "failed")
    with pytest.raises(ImageRejected) as e:
        load_and_validate(png_bytes((99, 300)))
    assert e.value.reason == "too_small:99x300"
    with pytest.raises(ImageRejected) as e:
        load_and_validate(png_bytes((200, 200)), max_side=150)
    assert e.value.reason.startswith("too_large")

    buf = io.BytesIO()
    frames = [Image.new("RGB", (120, 120), c) for c in ((255, 0, 0), (0, 255, 0))]
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:])
    with pytest.raises(ImageRejected) as e:
        load_and_validate(buf.getvalue())
    assert e.value.reason == "animated"

    rgba = load_and_validate(png_bytes((200, 200), (0, 0, 0, 0), mode="RGBA"))
    out, ops = to_rgb(rgba)
    assert out.mode == "RGB" and out.getpixel((5, 5)) == (255, 255, 255) and "rgba_on_white" in ops

    cmyk = Image.new("CMYK", (150, 150), (0, 0, 0, 0))
    out, ops = to_rgb(cmyk)
    assert out.mode == "RGB" and ops == ["convert_rgb"]

    wide = load_and_validate(jpeg_bytes((300, 100)))
    out, ops = to_rgb(wide, pad_aspect=2.0)
    assert out.size == (300, 300) and "pad_square" in ops
    out, ops = to_rgb(load_and_validate(jpeg_bytes((199, 100))), pad_aspect=2.0)
    assert out.size == (199, 100)


def test_preprocess_exif_rotation():
    img = Image.new("RGB", (200, 100), (1, 2, 3))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90° CW
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    out, ops = to_rgb(load_and_validate(buf.getvalue()), pad_aspect=None)
    assert out.size == (100, 200) and "exif_transpose" in ops


# ---------------------------------------------------------------- storage
def _exercise_store(store):
    w = PartWriter(store, "runs/x/commons", "20260101T000000Z-aaaa", flush_every=2)
    for i in range(3):
        w.add({"k": "a" if i != 1 else "b", "v": i})
    w.close()
    w2 = PartWriter(store, "runs/x/commons", "20260102T000000Z-bbbb", flush_every=10)
    w2.add({"k": "a", "v": 99})
    w2.close()
    assert len(store.list("runs/x/commons")) == 3
    merged = last_wins(store.iter_jsonl_parts("runs/x/commons"), key=lambda r: r["k"])
    assert merged["a"]["v"] == 99 and merged["b"]["v"] == 1
    store.write_bytes("a/b.bin", b"123")
    assert store.exists("a/b.bin") and store.read_bytes("a/b.bin") == b"123"
    store.move("a/b.bin", "a/c.bin")
    assert not store.exists("a/b.bin") and store.exists("a/c.bin")


def test_local_store(tmp_path):
    _exercise_store(LocalStore(str(tmp_path)))
    assert not [p for p in (tmp_path / "a").iterdir() if p.name.startswith(".tmp-")]


def test_s3_store():
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="bkt")
        _exercise_store(S3Store("bkt", "prefix/work", client=client))
        assert all(k["Key"].startswith("prefix/work/") for k in client.list_objects_v2(Bucket="bkt")["Contents"])


# ---------------------------------------------------------------- record validation
def test_validate_and_truncate():
    coll = ImageCollection(None, "c", dim=3)
    row = {f: None for f in ("revision_id", "original_url", "caption", "license_url", "artist",
                             "credit", "width", "height", "commons_sha1")}
    row.update(id=1, vector=[1, 0, 0], wiki="w", page_id=1, page_title="あ" * 300, page_url="u",
               image_title="File:x", image_url="u", image_page_url="u", position=1, license="l",
               mime_type="m", file_key="k", dump_date="d", embedding_model="m",
               embedding_revision="r", created_at=1)
    out = coll.validate(row)
    assert len(out["page_title"].encode()) <= 512 and out["page_title"] == "あ" * 170
    with pytest.raises(ValueError):
        coll.validate({**row, "vector": [1, 0]})
    with pytest.raises(ValueError):
        coll.validate({**row, "license": None})
    with pytest.raises(ValueError):
        coll.validate({**row, "page_id": "1"})
    assert truncate_utf8("aé", 2) == "a"
