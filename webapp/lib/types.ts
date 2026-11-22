// Mirrors article_db.search.SearchResult.to_dict() (Python retrieval API, POST /search).

export interface ChunkMatch {
  offset: number;
  score: number; // Qwen3 cosine (0 when only BM25 matched)
  bm25?: number; // BM25 score (0 when only the dense search matched)
  chunk_index: number;
  section_path: string;
  url: string;
  text: string;
}

export interface ImageMatch {
  offset: number;
  score: number;
  image_url: string;
  image_page_url: string;
  caption: string;
  section_path: string;
  license: string;
  license_url: string;
  artist: string;
  file_title: string;
}

/** Gallery entry: an image hit with its article (the article need not be in `articles`). */
export interface GalleryImage extends ImageMatch {
  page_id: number;
  title: string;
  url: string;
}

export interface ArticleHit {
  page_id: number;
  score: number;
  title: string;
  url: string;
  page_type: string;
  status: string;
  is_part_of: string;
  chunks: ChunkMatch[];
  images: ImageMatch[];
}

export interface SearchResult {
  query: { mode: string; encoders: string[]; ranking?: "fused" | "photo" };
  articles: ArticleHit[];
  /** Best image matches straight from the image search, independent of the article ranking. */
  images?: GalleryImage[];
}

/** One numbered passage given to the LLM; the answer cites it as [n]. */
export interface Citation {
  n: number;
  title: string;
  section: string;
  url: string;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

/** NDJSON events streamed by POST /api/chat, one JSON object per line. */
export type ChatEvent =
  | { type: "sources"; searchQuery: string | null; search: SearchResult; citations: Citation[] }
  | { type: "delta"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };
