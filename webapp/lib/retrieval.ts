import { env } from "./env";
import type { SearchResult } from "./types";

export class RetrievalError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

/** POST /search on the Python retrieval API (article_db.api). */
export async function searchArticles(opts: {
  text: string | null;
  image: Blob | null;
  imageName?: string;
  topK?: number;
  signal?: AbortSignal;
}): Promise<SearchResult> {
  const form = new FormData();
  if (opts.text) form.set("text", opts.text);
  if (opts.image) form.set("image", opts.image, opts.imageName ?? "query.jpg");
  form.set("top_k", String(opts.topK ?? 5));
  const RETRIEVAL_URL = env.retrievalUrl;
  let resp: Response;
  try {
    resp = await fetch(`${RETRIEVAL_URL}/search`, { method: "POST", body: form, signal: opts.signal });
  } catch (e) {
    throw new RetrievalError(
      `Retrieval API not reachable at ${RETRIEVAL_URL} (start: uvicorn article_db.api:app --port 8001): ${String(e)}`,
      502,
    );
  }
  if (!resp.ok) {
    let detail = await resp.text();
    try {
      detail = (JSON.parse(detail) as { detail?: string }).detail ?? detail;
    } catch {}
    throw new RetrievalError(`Retrieval API ${resp.status}: ${detail}`, resp.status);
  }
  return (await resp.json()) as SearchResult;
}
