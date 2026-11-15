import type { ChatTurn, Citation, SearchResult } from "./types";

export const SYSTEM_PROMPT = `You are a travel assistant that answers from Wikivoyage.
Use ONLY the numbered context passages. Cite the passages you use as [n], right after the sentence.
If the user attached a photo, say what it shows and which of the listed destinations it most likely is,
but do not claim a place that is not in the context.
If the context does not answer the question, say so briefly and suggest what to ask instead.
Be concise and specific; name the destination. Answer in the user's language.`;

const MAX_PASSAGE_CHARS = 1800;

/** Drop the "Title (type, region) — Section" header line embed_chunks.py puts on every chunk. */
export function chunkBody(text: string): string {
  const i = text.indexOf("\n");
  return (i >= 0 ? text.slice(i + 1) : text).trim();
}

/** Number every chunk and every image caption so the answer can cite them as [n]. */
export function buildContext(search: SearchResult): { context: string; citations: Citation[] } {
  const parts: string[] = [];
  const citations: Citation[] = [];
  for (const a of search.articles) {
    const where = [a.page_type, a.is_part_of].filter(Boolean).join(", ");
    for (const c of a.chunks) {
      const n = citations.length + 1;
      const section = c.section_path || "Introduction";
      citations.push({ n, title: a.title, section, url: c.url || a.url });
      let body = chunkBody(c.text);
      if (body.length > MAX_PASSAGE_CHARS) body = body.slice(0, MAX_PASSAGE_CHARS) + " …";
      parts.push(`[${n}] ${a.title}${where ? ` (${where})` : ""} — ${section}\n${body}`);
    }
    const captions = a.images.map((m) => m.caption).filter(Boolean);
    if (captions.length) {
      const n = citations.length + 1;
      citations.push({ n, title: a.title, section: "Photos", url: a.url });
      parts.push(`[${n}] ${a.title} — photos in the article: ${captions.map((c) => `"${c}"`).join("; ")}`);
    }
  }
  return { context: parts.join("\n\n---\n\n"), citations };
}

/** Prompt that turns a follow-up ("hotels there?") into a standalone search query. */
export function condenseMessages(history: ChatTurn[], question: string): { role: "system" | "user"; content: string }[] {
  const convo = history
    .slice(-6)
    .map((t) => `${t.role === "user" ? "User" : "Assistant"}: ${t.content.slice(0, 600)}`)
    .join("\n");
  return [
    {
      role: "system",
      content:
        "Rewrite the user's last message as one standalone English search query for a travel guide. " +
        "Resolve pronouns and 'there' using the conversation. Keep place names. Output only the query.",
    },
    { role: "user", content: `Conversation:\n${convo}\n\nLast message: ${question}` },
  ];
}

export function userPrompt(context: string, question: string | null, hasImage: boolean): string {
  const q =
    question ??
    "What place is in this photo? Describe it and tell me about the matching destination.";
  const img = hasImage ? "\n(The user attached the photo shown with this message.)" : "";
  return `Context passages:\n\n${context || "(no passages found)"}\n\nQuestion: ${q}${img}`;
}
