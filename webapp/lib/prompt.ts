import type { ChatTurn, Citation, SearchResult } from "./types";

export const SYSTEM_PROMPT = `You are a travel assistant that answers from Wikivoyage.
Use ONLY the numbered context passages. Cite the passages you use as [n], right after the sentence.

Write a helpful, well-developed answer, not a one-liner:
1. Start with a 1–2 sentence direct answer that names the destination(s).
2. Then give the useful details from the passages as a list, one item per line starting with "- ":
   each item names a specific place, dish, venue, sight or option in **bold**, says what makes it worth it,
   and adds practical details the passages give (area, address, price, opening hours, how to get there).
   Every item must end with the [n] of the passage it comes from; leave out anything you cannot cite.
   Give 3–6 items when the passages support them, and cover several destinations if more than one fits.
3. End with a short practical tip (best time, booking, getting around) if the passages mention one.
Use every relevant passage; never invent names, prices or facts that are not in them.

If the user attached a photo, first say in one sentence what it shows and which of the listed destinations
it most likely is (do not claim a place that is not in the context), then answer the question for that place.
If the context only partly answers the question, give what it does say and state what is missing.
If it does not answer at all, say so briefly and suggest what to ask instead.
Answer in the user's language.`;

const MAX_PASSAGE_CHARS = 1800;

/** How many past turns to keep, both for the follow-up rewrite prompt and for the answer's chat memory. */
export const RECENT_TURNS = 6;

// Matches either a leading connective ("And...", "What about...") or a pronoun/deictic word that can
// only be resolved from the previous turn ("there", "it", "same place", ...). Used to decide whether a
// question needs to be rewritten into a standalone search query before embedding (see looksLikeFollowUp).
const FOLLOW_UP_RE =
  /(^\s*(and|also|what about|how about)\b)|\b(there|that|those|it|they|them|this|these|same|another|again|nearby|around here|in that area)\b/i;

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
    .slice(-RECENT_TURNS)
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

/** Heuristic: rewrite only likely follow-up turns; keep standalone questions untouched. */
export function looksLikeFollowUp(question: string): boolean {
  const q = question.trim();
  if (!q) return false;
  return FOLLOW_UP_RE.test(q);
}

export function userPrompt(context: string, question: string | null, hasImage: boolean): string {
  const q =
    question ??
    "What place is in this photo? Describe it and tell me about the matching destination.";
  const img = hasImage ? "\n(The user attached the photo shown with this message.)" : "";
  return `Context passages:\n\n${context || "(no passages found)"}\n\nQuestion: ${q}${img}`;
}
