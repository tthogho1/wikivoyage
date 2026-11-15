"""Real SigLIP2 checks. Downloads ~4.5 GB on first run: ``pytest -m model``."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder():
    from image_embedding.config import ImagesConfig
    from image_embedding.embedder import SiglipImageEmbedder

    cfg = ImagesConfig.from_env()
    return SiglipImageEmbedder(cfg.model_id, cfg.model_revision, device=cfg.device, dtype=cfg.dtype)


def _sun() -> Image.Image:
    img = Image.new("RGB", (384, 384), (90, 160, 255))
    ImageDraw.Draw(img).ellipse((120, 120, 264, 264), fill=(255, 220, 0))
    return img


def test_image_vectors(embedder):
    vecs = embedder.embed_images([_sun(), _sun(), Image.new("RGB", (500, 200), (20, 120, 20))])
    assert vecs.shape == (3, 1152) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)
    assert float(vecs[0] @ vecs[1]) > 0.99
    assert embedder.info["model_revision"] not in ("", "unknown")


def test_text_matches_image(embedder):
    img = embedder.embed_images([_sun()])[0]
    texts = embedder.embed_texts(["a yellow sun in a blue sky", "a plate of sushi"])
    assert float(texts[0] @ img) > float(texts[1] @ img)
