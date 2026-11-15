from __future__ import annotations

import bz2
import io
from xml.sax.saxutils import escape

import numpy as np
import pytest
from PIL import Image

from image_embedding.config import ImagesConfig
from image_embedding.storage import LocalStore

NS = "http://www.mediawiki.org/xml/export-0.11/"

# (page_id, ns, title, text, redirect_title)
MINI_PAGES = [
    (1, 0, "Tokyo", "{{pagebanner|Tokyo banner.jpg}}\n'''Tokyo''' is a city.\n"
     "[[File:Shibuya crossing.jpg|thumb|300px|The [[Shibuya]] crossing]]\n"
     "* {{see|name=Senso-ji|image=Sensoji.jpg|content=Temple}}\n"
     "<gallery>\nFile:Gallery one.jpg|First\nAnimated.gif|Moving\n</gallery>\n", None),
    (2, 0, "Kyoto", "Kyoto.\n[[Image:Kinkakuji.jpg|left|Golden Pavilion]]\n"
     "[[File:Shibuya crossing.jpg|thumb|Same image, other article]]\n", None),
    (3, 0, "Old Tokyo", "#REDIRECT [[Tokyo]]", "Tokyo"),
    (4, 1, "Talk:Tokyo", "[[File:Talk.jpg]]", None),
    (5, 0, "Empty", "   ", None),
    (6, 0, "No images", "Just text with a [[link]].", None),
    (7, 0, "Missing image", "[[File:Does not exist.jpg|thumb|Gone]]", None),
]


def make_dump(pages=MINI_PAGES) -> bytes:
    parts = [f'<mediawiki xmlns="{NS}" version="0.11" xml:lang="en">\n']
    for pid, ns, title, text, redir in pages:
        parts.append("<page>\n")
        parts.append(f"<title>{escape(title)}</title>\n<ns>{ns}</ns>\n<id>{pid}</id>\n")
        if redir:
            parts.append(f'<redirect title="{escape(redir)}" />\n')
        parts.append(f"<revision><id>{pid * 100}</id><timestamp>2026-11-01T00:00:00Z</timestamp>"
                     f'<text bytes="{len(text)}" xml:space="preserve">{escape(text)}</text></revision>\n')
        parts.append("</page>\n")
    parts.append("</mediawiki>\n")
    return bz2.compress("".join(parts).encode("utf-8"))


@pytest.fixture
def mini_dump(tmp_path):
    p = tmp_path / "mini-dump.xml.bz2"
    p.write_bytes(make_dump())
    return str(p)


@pytest.fixture
def cfg(tmp_path) -> ImagesConfig:
    c = ImagesConfig(workdir=str(tmp_path / "work"), dump_date="20261101")
    c.flush_every = 3  # exercise multi-part writes
    c.download_rps = 0
    c.api_concurrency = 1
    c.download_concurrency = 2
    c.model_id = FakeEmbedder.model_id
    return c


@pytest.fixture
def store(cfg) -> LocalStore:
    return LocalStore(cfg.workdir)


def png_bytes(size=(200, 150), color=(200, 30, 30), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes(size=(300, 200), color=(10, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


class FakeEmbedder:
    """Deterministic stand-in for SiglipImageEmbedder (vector from mean colour)."""

    model_id = "test/fake-siglip"
    model_revision = "0123456789abcdef"

    def __init__(self, dim: int = 8, fail_on: int | None = None) -> None:
        self.dim = dim
        self.fail_on = fail_on
        self.info = {"model_id": self.model_id, "model_revision": self.model_revision, "dim": dim}
        self.calls = 0

    def _vec(self, img) -> np.ndarray:
        r, g, b = [float(c) for c in np.asarray(img.convert("RGB")).reshape(-1, 3).mean(axis=0)]
        v = np.array([r, g, b, 1.0] + [0.5] * (self.dim - 4), dtype=np.float32)
        return v / np.linalg.norm(v)

    def embed_images(self, images):
        self.calls += 1
        if self.fail_on is not None and len(images) > 1:
            raise RuntimeError("simulated OOM")
        if self.fail_on is not None and images[0].size[0] == self.fail_on:
            raise RuntimeError("bad image")
        return np.stack([self._vec(i) for i in images])

    def embed_texts(self, texts):
        return np.stack([np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim) for _ in texts])
