"""Search ``wikivoyage_images`` by image, image URL or text (SigLIP2 shares one space for both)."""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Any, Optional

from .http import make_session
from .zilliz import escape_filter_value

MAX_IMAGE_BYTES = 20 * 1024 * 1024
RESULT_FIELDS = [
    "page_title", "page_url", "image_url", "image_page_url", "caption", "license",
    "license_url", "artist", "image_title", "page_id",
]


@dataclass
class ImageHit:
    score: float
    page_title: str
    page_url: str
    image_url: str
    image_page_url: str
    caption: Optional[str]
    license: str
    license_url: Optional[str]
    artist: Optional[str]
    image_title: str
    page_id: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_image_from_url(url: str, user_agent: str):
    from PIL import Image

    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("image URL must be http(s)")
    session = make_session(user_agent)
    with session.get(url, stream=True, timeout=(10, 60)) as resp:
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype and not ctype.startswith("image/"):
            raise ValueError(f"URL did not return an image (Content-Type {ctype})")
        buf = bytearray()
        for chunk in resp.iter_content(64 * 1024):
            buf.extend(chunk)
            if len(buf) > MAX_IMAGE_BYTES:
                raise ValueError("image larger than 20 MB")
    try:
        return Image.open(io.BytesIO(bytes(buf))).convert("RGB")
    except Exception as e:
        raise ValueError(f"could not decode image: {e}") from e


class ImageSearcher:
    def __init__(
        self,
        client: Any,
        embedder: Any,
        collection: str = "wikivoyage_images",
        *,
        text_template: str = "{}",
        user_agent: str = "",
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.text_template = text_template or "{}"
        self.user_agent = user_agent

    def _search(self, vector, *, top_k: int = 10, page_title: Optional[str] = None) -> list[ImageHit]:
        flt = f'page_title == "{escape_filter_value(page_title)}"' if page_title else ""
        res = self.client.search(
            collection_name=self.collection,
            data=[list(map(float, vector))],
            anns_field="vector",
            limit=top_k,
            filter=flt,
            output_fields=RESULT_FIELDS,
            search_params={"metric_type": "COSINE"},
        )
        hits: list[ImageHit] = []
        for h in (res[0] if res else []):
            ent = h.get("entity", h)
            hits.append(ImageHit(score=float(h.get("distance", 0.0)), **{k: ent.get(k) for k in RESULT_FIELDS}))
        return hits

    def search_by_image(self, image, *, top_k: int = 10, page_title: Optional[str] = None) -> list[ImageHit]:
        from .preprocess import to_rgb

        rgb, _ = to_rgb(image)
        return self._search(self.embedder.embed_images([rgb])[0], top_k=top_k, page_title=page_title)

    def search_by_image_url(self, url: str, **kw) -> list[ImageHit]:
        return self.search_by_image(load_image_from_url(url, self.user_agent), **kw)

    def search_by_text(self, text: str, *, top_k: int = 10, page_title: Optional[str] = None) -> list[ImageHit]:
        query = self.text_template.format(text)
        return self._search(self.embedder.embed_texts([query])[0], top_k=top_k, page_title=page_title)
