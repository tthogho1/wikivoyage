"""Search ``wikivoyage_articles`` with text and/or an image (design §5.3, §7.3).

Which encoder embeds the query depends on the input:

    text only     Qwen3 → chunks[embedding]   +  SigLIP2 text  → images[embedding]
    image only                                    SigLIP2 image → images[embedding]
    text + image  Qwen3 → chunks[embedding]   +  SigLIP2 image → images[embedding]

The two cosine scales differ, so the lists are merged per article with weighted RRF.
Element-level search returns (page_id, offset); matched chunk/image contents are fetched in
a second ``query`` for the top articles only (spike option A: ``chunks[text]`` in a search
returns the whole array of every hit).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from image_embedding.zilliz import escape_filter_value

logger = logging.getLogger("wikivoyage.articles.search")

ARTICLE_FIELDS = ["page_id", "title", "url", "page_type", "status", "is_part_of"]
CHUNK_FIELDS = ["chunk_index", "section_path", "url", "text"]
IMAGE_FIELDS = ["image_url", "image_page_url", "caption", "section_path", "license", "license_url", "artist",
                "file_title"]

TEXT_FIELD = "chunks[embedding]"
IMAGE_FIELD = "images[embedding]"


@dataclass
class SearchFilters:
    page_type: Optional[str] = None
    is_part_of: Optional[str] = None
    title: Optional[str] = None
    heading: Optional[str] = None  # top_section of the chunk; text search only

    def article_expr(self) -> str:
        parts = []
        for name in ("page_type", "is_part_of", "title"):
            v = getattr(self, name)
            if v:
                parts.append(f'{name} == "{escape_filter_value(v)}"')
        return " && ".join(parts)

    def chunk_expr(self) -> str:
        expr = self.article_expr()
        if self.heading:
            elem = f'element_filter(chunks, $[top_section] == "{escape_filter_value(self.heading)}")'
            expr = f"{expr} && {elem}" if expr else elem
        return expr


@dataclass
class ChunkMatch:
    offset: int
    score: float
    chunk_index: int = -1
    section_path: str = ""
    url: str = ""
    text: str = ""


@dataclass
class ImageMatch:
    offset: int
    score: float
    image_url: str = ""
    image_page_url: str = ""
    caption: str = ""
    section_path: str = ""
    license: str = ""
    license_url: str = ""
    artist: str = ""
    file_title: str = ""


@dataclass
class ArticleHit:
    page_id: int
    score: float
    title: str = ""
    url: str = ""
    page_type: str = ""
    status: str = ""
    is_part_of: str = ""
    chunks: list[ChunkMatch] = field(default_factory=list)
    images: list[ImageMatch] = field(default_factory=list)


@dataclass
class SearchResult:
    mode: str
    encoders: list[str]
    articles: list[ArticleHit]

    def to_dict(self) -> dict:
        return {"query": {"mode": self.mode, "encoders": self.encoders},
                "articles": [asdict(a) for a in self.articles]}


@dataclass
class _Elem:
    page_id: int
    offset: int
    score: float


def rrf_merge(lists: dict[str, list[_Elem]], weights: dict[str, float], k: int = 60) -> list[tuple[int, float]]:
    """Weighted RRF over articles. An article's rank in a list is the rank of its best element."""
    scores: dict[int, float] = {}
    for name, elems in lists.items():
        w = weights.get(name, 1.0)
        rank = 0
        seen: set[int] = set()
        for e in elems:
            if e.page_id in seen:
                continue
            seen.add(e.page_id)
            rank += 1
            scores[e.page_id] = scores.get(e.page_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def element_limit(top_k: int, per_article: int) -> int:
    return max(50, min(1000, top_k * per_article * 3))


class ArticleSearcher:
    def __init__(self, client: Any, collection: str = "wikivoyage_articles", *,
                 text_embedder: Any = None, image_embedder: Any = None,
                 rrf_k: int = 60, w_text: float = 1.0, w_image: float = 0.5,
                 per_article: int = 3, intro_chunks: int = 3, ef: int = 128) -> None:
        self.client = client
        self.collection = collection
        self.text_embedder = text_embedder    # QwenQueryEmbedder: embed_queries(list[str])
        self.image_embedder = image_embedder  # SiglipImageEmbedder: embed_texts / embed_images
        self.rrf_k = rrf_k
        self.weights = {"text": w_text, "image": w_image}
        self.per_article = per_article
        self.intro_chunks = intro_chunks
        self.ef = ef

    # ------------------------------------------------------------------ public
    def search(self, text: Optional[str] = None, image: Any = None, *, top_k: int = 5,
               filters: Optional[SearchFilters] = None, mode: str = "both") -> SearchResult:
        """``image`` is a PIL image. ``mode``: both / text (chunks only) / image (images only)."""
        text = (text or "").strip() or None
        if text is None and image is None:
            raise ValueError("give text, an image, or both")
        filters = filters or SearchFilters()
        lim = element_limit(top_k, self.per_article)
        lists: dict[str, list[_Elem]] = {}
        encoders: list[str] = []

        if text is not None and mode in ("both", "text"):
            qvec = self._need(self.text_embedder, "text").embed_queries([text])[0]
            lists["text"] = self._elem_search(qvec, TEXT_FIELD, filters.chunk_expr(), lim)
            encoders.append("qwen3")
        if mode in ("both", "image"):
            if image is not None:
                from image_embedding.preprocess import to_rgb

                rgb, _ = to_rgb(image)
                ivec = self._need(self.image_embedder, "image").embed_images([rgb])[0]
                lists["image"] = self._elem_search(ivec, IMAGE_FIELD, filters.article_expr(), lim)
                encoders.append("siglip2-image")
            elif text is not None:
                tvec = self._need(self.image_embedder, "image").embed_texts([text])[0]
                lists["image"] = self._elem_search(tvec, IMAGE_FIELD, filters.article_expr(), lim)
                encoders.append("siglip2-text")
        if not lists:
            raise ValueError(f"mode={mode!r} needs {'text' if mode == 'text' else 'an image or text'}")

        ranked = rrf_merge(lists, self.weights, self.rrf_k)[:top_k]
        articles = self._hydrate(ranked, lists, intro=("text" not in lists))
        query_mode = "+".join(n for n, v in (("text", text), ("image", image)) if v is not None)
        return SearchResult(mode=query_mode, encoders=encoders, articles=articles)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _need(embedder: Any, what: str) -> Any:
        if embedder is None:
            raise ValueError(f"no {what} embedder loaded")
        return embedder

    def _elem_search(self, vector, anns_field: str, expr: str, limit: int) -> list[_Elem]:
        res = self.client.search(
            collection_name=self.collection,
            data=[[float(x) for x in vector]],
            anns_field=anns_field,
            limit=limit,
            filter=expr,
            output_fields=["page_id"],
            search_params={"metric_type": "COSINE", "params": {"ef": max(self.ef, limit)}},
        )
        out: list[_Elem] = []
        for h in (res[0] if res else []):
            ent = h.get("entity") or {}
            pid = h.get("page_id", ent.get("page_id", h.get("id")))
            off = h.get("offset")
            if pid is None or off is None:
                continue
            out.append(_Elem(int(pid), int(off), float(h.get("distance", 0.0))))
        return out

    def _hydrate(self, ranked: list[tuple[int, float]], lists: dict[str, list[_Elem]],
                 *, intro: bool) -> list[ArticleHit]:
        if not ranked:
            return []
        ids = [pid for pid, _ in ranked]
        want_chunks = "text" in lists or intro
        want_images = "image" in lists
        fields = list(ARTICLE_FIELDS)
        if want_chunks:
            fields += [f"chunks[{f}]" for f in CHUNK_FIELDS]
        if want_images:
            fields += [f"images[{f}]" for f in IMAGE_FIELDS]
        rows = self.client.query(collection_name=self.collection, filter=f"page_id in {ids}",
                                 output_fields=fields)
        by_id = {int(r["page_id"]): r for r in rows}

        def best(name: str, pid: int) -> list[_Elem]:
            seen, out = set(), []
            for e in lists.get(name, []):
                if e.page_id == pid and e.offset not in seen:
                    seen.add(e.offset)
                    out.append(e)
                    if len(out) >= self.per_article:
                        break
            return out

        hits: list[ArticleHit] = []
        for pid, score in ranked:
            row = by_id.get(pid)
            if row is None:  # deleted between search and query
                continue
            hit = ArticleHit(page_id=pid, score=round(score, 6),
                             **{f: row.get(f) or "" for f in ARTICLE_FIELDS if f != "page_id"})
            chunks = row.get("chunks") or []
            images = row.get("images") or []
            if "text" in lists:
                hit.chunks = [self._chunk(chunks, e.offset, e.score) for e in best("text", pid)
                              if e.offset < len(chunks)]
            elif intro:
                hit.chunks = [self._chunk(chunks, i, 0.0) for i in range(min(self.intro_chunks, len(chunks)))]
            hit.images = [self._image(images, e.offset, e.score) for e in best("image", pid)
                          if e.offset < len(images)]
            hits.append(hit)
        return hits

    @staticmethod
    def _chunk(chunks: list[dict], offset: int, score: float) -> ChunkMatch:
        c = chunks[offset]
        return ChunkMatch(offset=offset, score=round(score, 4), **{f: c.get(f, ChunkMatch.__dataclass_fields__[f].default)
                                                                   for f in CHUNK_FIELDS})

    @staticmethod
    def _image(images: list[dict], offset: int, score: float) -> ImageMatch:
        m = images[offset]
        return ImageMatch(offset=offset, score=round(score, 4), **{f: m.get(f) or "" for f in IMAGE_FIELDS})


def build_searcher(cfg: Any, *, text: bool = True, image: bool = True, client: Any = None) -> ArticleSearcher:
    """Load the encoders that are asked for (each takes seconds and GBs) and connect to Zilliz."""
    import os

    from image_embedding.zilliz import make_client

    if client is None:
        if not cfg.zilliz_uri or not cfg.zilliz_token:
            raise ValueError("ZILLIZ_URI and ZILLIZ_TOKEN must be set")
        client = make_client(cfg.zilliz_uri, cfg.zilliz_token)
    if not client.has_collection(cfg.collection):
        raise ValueError(f"collection {cfg.collection!r} not found; run 'python -m article_db upsert' first")
    text_emb = image_emb = None
    if text:
        from .text_embedder import QwenQueryEmbedder

        text_emb = QwenQueryEmbedder(cfg.text_model, device=os.environ.get("QWEN_DEVICE", "auto"))
    if image:
        from image_embedding.config import ImagesConfig
        from image_embedding.embedder import SiglipImageEmbedder

        icfg = ImagesConfig.from_env()
        image_emb = SiglipImageEmbedder(icfg.model_id, icfg.model_revision, device=icfg.device, dtype=icfg.dtype)
    env = os.environ.get
    return ArticleSearcher(client, cfg.collection, text_embedder=text_emb, image_embedder=image_emb,
                           rrf_k=int(env("ARTICLES_RRF_K", "60")), w_text=float(env("ARTICLES_W_TEXT", "1.0")),
                           w_image=float(env("ARTICLES_W_IMAGE", "0.5")))
