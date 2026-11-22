import type { ArticleHit, Citation, GalleryImage, SearchResult } from "@/lib/types";

/** The API's gallery (best image hits, any article); older APIs: the images of the top articles. */
function galleryOf(search: SearchResult): GalleryImage[] {
  if (search.images) return search.images;
  return search.articles.flatMap((a) => a.images.map((m) => ({ ...m, page_id: a.page_id, title: a.title, url: a.url })));
}

export function Sources({ search, citations }: { search: SearchResult; citations: Citation[] }) {
  const images = galleryOf(search);
  return (
    <div className="sources">
      {images.length > 0 && (
        <div className="gallery">
          {images.slice(0, 8).map((m) => (
            <figure key={`${m.page_id}-${m.offset}`}>
              <a href={m.image_page_url || m.image_url} target="_blank" rel="noreferrer">
                <img src={m.image_url} alt={m.caption || m.file_title} loading="lazy" />
              </a>
              <figcaption>
                <span className="fig-title">{m.caption || m.title}</span>
                <span className="attrib">
                  <a href={m.url} target="_blank" rel="noreferrer">
                    {m.title}
                  </a>{" "}
                  · {m.artist ? `${stripHtml(m.artist)}, ` : ""}
                  {m.license_url ? (
                    <a href={m.license_url} target="_blank" rel="noreferrer">
                      {m.license}
                    </a>
                  ) : (
                    m.license
                  )}
                </span>
              </figcaption>
            </figure>
          ))}
        </div>
      )}
      <details>
        <summary>
          {search.articles.length} articles · {citations.length} passages · encoders: {search.query.encoders.join(" + ")}
          {search.query.ranking === "photo" ? " · ranked by photo" : ""}
        </summary>
        <ol className="citations">
          {citations.map((c) => (
            <li key={c.n} value={c.n}>
              <a href={c.url} target="_blank" rel="noreferrer">
                {c.title} — {c.section}
              </a>
            </li>
          ))}
        </ol>
        <ul className="articles">
          {search.articles.map((a: ArticleHit) => (
            <li key={a.page_id}>
              <a href={a.url} target="_blank" rel="noreferrer">
                {a.title}
              </a>{" "}
              <span className="muted">
                {[a.page_type, a.is_part_of].filter(Boolean).join(", ")} · score {a.score.toFixed(4)}
              </span>
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

function stripHtml(s: string) {
  return s.replace(/<[^>]*>/g, "").trim().slice(0, 80);
}
