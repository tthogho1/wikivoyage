"""Search ``wikivoyage_articles`` with text and/or an image (design §5.3, §7.3).

Which encoder embeds the query depends on the input:

    text only     Qwen3 → chunks[embedding]  +  BM25 → wikivoyage_chunks_bm25  +  SigLIP2 text  → images[embedding]
    image only                                                                    SigLIP2 image → images[embedding]
    text + image  Qwen3 → chunks[embedding]  +  BM25 → wikivoyage_chunks_bm25  +  SigLIP2 image → images[embedding]

Text search is hybrid: dense (Qwen3) for meaning, BM25 for exact words (place, station and shop
names). BM25 hits carry the same (page_id, offset) as dense chunk hits (see ``bm25.py``). The scores
are not comparable, so the lists are merged per article with weighted RRF, and an article's matched
chunks are picked by element-level RRF of the dense and BM25 lists.

Text + photo is ranked differently ("photo-first"): the photo says *where*, the text says *what*.
Merging both by RRF lets a vague question ("where to stay?") outrank the photo, so the photo alone
picks the articles and the text search then runs only inside them to pick the passages.

Independently of the article ranking, ``SearchResult.images`` is a gallery of the best image hits
(at most ``gallery_per_article`` per article), so good photo matches are shown even when their
article does not make the top_k.
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
BM25_FIELD = "sparse"
CHUNK_LISTS = ("text", "bm25")
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

    def bm25_expr(self) -> str:
        """Same conditions on the flat BM25 collection (top_section is a plain field there)."""
        expr = self.article_expr()
        if self.heading:
            elem = f'top_section == "{escape_filter_value(self.heading)}"'
            expr = f"{expr} && {elem}" if expr else elem
        return expr


@dataclass
class ChunkMatch:
    offset: int
    score: float           # Qwen3 cosine (0 when only BM25 matched)
    chunk_index: int = -1
    section_path: str = ""
    url: str = ""
    text: str = ""
    bm25: float = 0.0      # BM25 score (0 when only the dense search matched)


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
class GalleryImage(ImageMatch):
    """An image hit with the article it belongs to (gallery entries need not be in ``articles``)."""
    page_id: int = -1
    title: str = ""
    url: str = ""


@dataclass
class SearchResult:
    mode: str
    encoders: list[str]
    articles: list[ArticleHit]
    images: list[GalleryImage] = field(default_factory=list)
    ranking: str = "fused"  # fused (RRF over every list) | photo (text + photo: the photo picks the articles)

    def to_dict(self) -> dict:
        return {"query": {"mode": self.mode, "encoders": self.encoders, "ranking": self.ranking},
                "articles": [asdict(a) for a in self.articles],
                "images": [asdict(m) for m in self.images]}


@dataclass
class _Elem:
    page_id: int
    offset: int
    score: float
    content_hash: Optional[str] = None  # BM25 hits: article version the offset belongs to


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
                 bm25_collection: Optional[str] = None,
                 rrf_k: int = 60, w_text: float = 1.0, w_image: float = 0.5, w_bm25: float = 1.0,
                 per_article: int = 3, intro_chunks: int = 3, ef: int = 128,
                 gallery_size: int = 8, gallery_per_article: int = 2, photo_first: bool = True) -> None:
        self.client = client
        self.collection = collection
        self.text_embedder = text_embedder    # QwenQueryEmbedder: embed_queries(list[str])
        self.image_embedder = image_embedder  # SiglipImageEmbedder: embed_texts / embed_images
        self.bm25_collection = bm25_collection  # None → dense-only text search
        self.rrf_k = rrf_k
        self.weights = {"text": w_text, "image": w_image, "bm25": w_bm25}
        self.per_article = per_article
        self.intro_chunks = intro_chunks
        self.ef = ef
        self.gallery_size = gallery_size
        self.gallery_per_article = gallery_per_article
        self.photo_first = photo_first

    # ------------------------------------------------------------------ public
    def search(self, text: Optional[str] = None, image: Any = None, *, top_k: int = 5,
               filters: Optional[SearchFilters] = None, mode: str = "both",
               keyword: bool = True) -> SearchResult:
        """``image`` is a PIL image. ``mode``: both / text (chunks only) / image (images only).
        ``keyword=False`` turns off the BM25 half of the text search (dense only)."""
        text = (text or "").strip() or None
        if text is None and image is None:
            raise ValueError("give text, an image, or both")
        filters = filters or SearchFilters()
        lim = element_limit(top_k, self.per_article)
        lists: dict[str, list[_Elem]] = {}
        encoders: list[str] = []
        want_text = text is not None and mode in ("both", "text")
        want_image = mode in ("both", "image")
        photo_first = self.photo_first and want_text and want_image and image is not None

        if photo_first:
            # 1. The photo picks the articles ...
            lists["image"] = self._photo_search(image, filters, lim)
            encoders.append("siglip2-image")
            ranked = rrf_merge({"image": lists["image"]}, {"image": 1.0}, self.rrf_k)[:top_k]
            # 2. ... and the text picks the passages inside them.
            if ranked:
                inside = f"page_id in {[pid for pid, _ in ranked]}"
                encoders += self._text_lists(lists, text, filters, lim, keyword, restrict=inside)
        else:
            if want_text:
                encoders += self._text_lists(lists, text, filters, lim, keyword)
            if want_image:
                if image is not None:
                    lists["image"] = self._photo_search(image, filters, lim)
                    encoders.append("siglip2-image")
                elif text is not None:
                    tvec = self._need(self.image_embedder, "image").embed_texts([text])[0]
                    lists["image"] = self._elem_search(tvec, IMAGE_FIELD, filters.article_expr(), lim)
                    encoders.append("siglip2-text")
            if not lists:
                raise ValueError(f"mode={mode!r} needs {'text' if mode == 'text' else 'an image or text'}")
            ranked = rrf_merge(lists, self.weights, self.rrf_k)[:top_k]

        rows: dict[int, dict] = {}
        articles = self._hydrate(ranked, lists, rows)
        gallery = self._gallery(lists.get("image", []), rows)
        query_mode = "+".join(n for n, v in (("text", text), ("image", image)) if v is not None)
        return SearchResult(mode=query_mode, encoders=encoders, articles=articles, images=gallery,
                            ranking="photo" if photo_first else "fused")

    def _text_lists(self, lists: dict[str, list[_Elem]], text: str, filters: SearchFilters, lim: int,
                    keyword: bool, restrict: str = "") -> list[str]:
        """Dense (+ BM25) chunk search into ``lists``; returns the encoder names used."""
        qvec = self._need(self.text_embedder, "text").embed_queries([text])[0]
        lists["text"] = self._elem_search(qvec, TEXT_FIELD, _and(filters.chunk_expr(), restrict), lim)
        used = ["qwen3"]
        if keyword and self.bm25_collection:
            lists["bm25"] = self._bm25_search(text, _and(filters.bm25_expr(), restrict), lim)
            used.append("bm25")
        return used

    def _photo_search(self, image: Any, filters: SearchFilters, lim: int) -> list[_Elem]:
        from image_embedding.preprocess import to_rgb

        rgb, _ = to_rgb(image)
        ivec = self._need(self.image_embedder, "image").embed_images([rgb])[0]
        return self._elem_search(ivec, IMAGE_FIELD, filters.article_expr(), lim)

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

    def _bm25_search(self, text: str, expr: str, limit: int) -> list[_Elem]:
        try:
            res = self.client.search(
                collection_name=self.bm25_collection,
                data=[text],
                anns_field=BM25_FIELD,
                limit=limit,
                filter=expr,
                output_fields=["page_id", "chunk_offset", "content_hash"],
                search_params={"metric_type": "BM25"},
            )
        except Exception as e:  # noqa: BLE001 — keyword search is an add-on; never fail the query for it
            logger.warning("BM25 search failed, using dense only: %s", e)
            return []
        out: list[_Elem] = []
        for h in (res[0] if res else []):
            ent = h.get("entity") or {}
            pid, off = ent.get("page_id"), ent.get("chunk_offset")
            if pid is None or off is None:
                continue
            out.append(_Elem(int(pid), int(off), float(h.get("distance", 0.0)), ent.get("content_hash")))
        return out

    def _best_chunks(self, lists: dict[str, list[_Elem]], pid: int) -> list[tuple[int, float, float]]:
        """(offset, dense cosine, bm25) of the article's best chunks: element-level weighted RRF of the
        dense and BM25 lists (rank = rank among this article's elements)."""
        fused: dict[int, float] = {}
        raw: dict[int, dict[str, float]] = {}
        for name in CHUNK_LISTS:
            rank = 0
            for e in lists.get(name, []):
                if e.page_id != pid or name in raw.get(e.offset, {}):
                    continue
                rank += 1
                fused[e.offset] = fused.get(e.offset, 0.0) + self.weights.get(name, 1.0) / (self.rrf_k + rank)
                raw.setdefault(e.offset, {})[name] = e.score
        best = sorted(fused, key=lambda o: (-fused[o], o))[: self.per_article]
        return [(o, raw[o].get("text", 0.0), raw[o].get("bm25", 0.0)) for o in best]

    def _hydrate(self, ranked: list[tuple[int, float]], lists: dict[str, list[_Elem]],
                 rows_out: dict[int, dict]) -> list[ArticleHit]:
        """Fetch the ranked articles; rows are also put in ``rows_out`` for the gallery."""
        if not ranked:
            return []
        ids = [pid for pid, _ in ranked]
        want_images = "image" in lists
        fields = list(ARTICLE_FIELDS)
        if "bm25" in lists:
            fields.append("content_hash")
        fields += [f"chunks[{f}]" for f in CHUNK_FIELDS]  # matched chunks, or the intro as a fallback
        if want_images:
            fields += [f"images[{f}]" for f in IMAGE_FIELDS]
        rows = self.client.query(collection_name=self.collection, filter=f"page_id in {ids}",
                                 output_fields=fields)
        by_id = {int(r["page_id"]): r for r in rows}
        rows_out.update(by_id)
        if "bm25" in lists:
            # The BM25 mirror may lag behind wikivoyage_articles: offsets of another version are meaningless.
            stale = {e.page_id for e in lists["bm25"]
                     if e.page_id in by_id and e.content_hash and e.content_hash != by_id[e.page_id].get("content_hash")}
            if stale:
                logger.warning("ignoring BM25 hits of %d articles whose content changed; run 'python -m article_db bm25'",
                               len(stale))
                lists = {**lists, "bm25": [e for e in lists["bm25"] if e.page_id not in stale]}

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
            if any(n in lists for n in CHUNK_LISTS):
                hit.chunks = [self._chunk(chunks, off, dense, bm25) for off, dense, bm25 in self._best_chunks(lists, pid)
                              if off < len(chunks)]
            if not hit.chunks:
                # Found by its images only (photo query, or no passage matched): give the intro as context.
                hit.chunks = [self._chunk(chunks, i, 0.0) for i in range(min(self.intro_chunks, len(chunks)))]
            hit.images = [self._image(images, e.offset, e.score) for e in best("image", pid)
                          if e.offset < len(images)]
            hits.append(hit)
        return hits

    def _gallery(self, elems: list[_Elem], rows: dict[int, dict]) -> list[GalleryImage]:
        """Best image hits in score order, at most ``gallery_per_article`` per article."""
        picked: list[_Elem] = []
        per: dict[int, int] = {}
        seen: set[tuple[int, int]] = set()
        for e in elems:
            if len(picked) >= self.gallery_size:
                break
            if (e.page_id, e.offset) in seen or per.get(e.page_id, 0) >= self.gallery_per_article:
                continue
            seen.add((e.page_id, e.offset))
            per[e.page_id] = per.get(e.page_id, 0) + 1
            picked.append(e)
        missing = sorted({e.page_id for e in picked if "images" not in rows.get(e.page_id, {})})
        if missing:
            fetched = self.client.query(collection_name=self.collection, filter=f"page_id in {missing}",
                                        output_fields=["page_id", "title", "url"] + [f"images[{f}]" for f in IMAGE_FIELDS])
            rows = {**rows, **{int(r["page_id"]): r for r in fetched}}
        out: list[GalleryImage] = []
        for e in picked:
            row = rows.get(e.page_id)
            images = (row or {}).get("images") or []
            if e.offset >= len(images):
                continue
            m = self._image(images, e.offset, e.score)
            out.append(GalleryImage(**asdict(m), page_id=e.page_id, title=row.get("title") or "", url=row.get("url") or ""))
        return out

    @staticmethod
    def _chunk(chunks: list[dict], offset: int, score: float, bm25: float = 0.0) -> ChunkMatch:
        c = chunks[offset]
        return ChunkMatch(offset=offset, score=round(score, 4), bm25=round(bm25, 4), **{f: c.get(f, ChunkMatch.__dataclass_fields__[f].default)
                                                                   for f in CHUNK_FIELDS})

    @staticmethod
    def _image(images: list[dict], offset: int, score: float) -> ImageMatch:
        m = images[offset]
        return ImageMatch(offset=offset, score=round(score, 4), **{f: m.get(f) or "" for f in IMAGE_FIELDS})


def _and(*exprs: str) -> str:
    return " && ".join(e for e in exprs if e)


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
    env = os.environ.get
    bm25 = None
    if text and env("ARTICLES_BM25", "true").strip().lower() not in ("0", "false", "no", "off"):
        bm25 = cfg.bm25_collection
        if not client.has_collection(bm25):
            logger.warning("BM25 collection %r not found; text search is dense only "
                           "(run 'python -m article_db bm25')", bm25)
            bm25 = None
    text_emb = image_emb = None
    if text:
        from .text_embedder import QwenQueryEmbedder

        text_emb = QwenQueryEmbedder(cfg.text_model, device=os.environ.get("QWEN_DEVICE", "auto"))
    if image:
        from image_embedding.config import ImagesConfig
        from image_embedding.embedder import SiglipImageEmbedder

        icfg = ImagesConfig.from_env()
        image_emb = SiglipImageEmbedder(icfg.model_id, icfg.model_revision, device=icfg.device, dtype=icfg.dtype)
    return ArticleSearcher(client, cfg.collection, text_embedder=text_emb, image_embedder=image_emb,
                           bm25_collection=bm25,
                           rrf_k=int(env("ARTICLES_RRF_K", "60")), w_text=float(env("ARTICLES_W_TEXT", "1.0")),
                           w_image=float(env("ARTICLES_W_IMAGE", "0.5")), w_bm25=float(env("ARTICLES_W_BM25", "1.0")),
                           gallery_size=int(env("ARTICLES_GALLERY_SIZE", "8")),
                           photo_first=env("ARTICLES_PHOTO_FIRST", "true").strip().lower() not in ("0", "false", "no", "off"))
