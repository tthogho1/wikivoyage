import OpenAI from "openai";
import type { ChatCompletionContentPart, ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { z } from "zod";

import { env } from "@/lib/env";
import { encodeEvent } from "@/lib/ndjson";
import { SYSTEM_PROMPT, RECENT_TURNS, buildContext, condenseMessages, looksLikeFollowUp, userPrompt } from "@/lib/prompt";
import { RetrievalError, searchArticles } from "@/lib/retrieval";
import type { ChatEvent } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"];

const History = z
  .array(z.object({ role: z.enum(["user", "assistant"]), content: z.string().max(8000) }))
  .max(20);

const Mode = z.enum(["both", "text", "image"]).catch("both");

function badRequest(message: string, status = 400) {
  return Response.json({ error: message }, { status });
}

/**
 * POST multipart: question?, image? (file), history? (JSON [{role, content}]).
 * Streams NDJSON: {type:"sources"} → {type:"delta"}* → {type:"done"} (or {type:"error"}).
 */
export async function POST(req: Request) {
  if (!env.openaiKey) return badRequest("OPENAI_API_KEY is not set", 500);

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return badRequest("expected multipart/form-data");
  }
  const question = (form.get("question")?.toString() ?? "").trim() || null;
  const rawImage = form.get("image");
  const image = rawImage instanceof File && rawImage.size > 0 ? rawImage : null;
  const mode = Mode.parse(form.get("mode")?.toString());
  if (!question && !image) return badRequest("send a question, an image, or both");
  if (question && question.length > 2000) return badRequest("question longer than 2000 characters");
  if (image && image.size > MAX_IMAGE_BYTES) return badRequest("image larger than 20 MB", 413);
  if (image && !IMAGE_TYPES.includes(image.type)) return badRequest(`unsupported image type ${image.type}`);

  let rawHistory: unknown;
  try {
    rawHistory = JSON.parse(form.get("history")?.toString() || "[]");
  } catch {
    return badRequest("history is not JSON");
  }
  const parsed = History.safeParse(rawHistory);
  if (!parsed.success) {
    console.error("[chat] invalid history", parsed.error.issues);
    return badRequest(`invalid history: ${parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ")}`);
  }
  const history = parsed.data;

  const openai = new OpenAI({ apiKey: env.openaiKey });
  const MODEL = env.openaiModel;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (ev: ChatEvent) => controller.enqueue(encodeEvent(ev));
      try {
        // 1. Follow-ups ("what about food there?") become a standalone query before embedding.
        //    Chat memory for generation is NOT gated on this: a missed follow-up heuristic should only
        //    degrade the search query, never wipe the assistant's conversational memory.
        let searchQuery = question;
        if (question && history.length > 0 && looksLikeFollowUp(question)) {
          const r = await openai.chat.completions.create({
            model: MODEL,
            messages: condenseMessages(history, question),
            temperature: 0,
            max_completion_tokens: 60,
          });
          searchQuery = r.choices[0]?.message?.content?.trim() || question;
        }

        // 2. Retrieval: Qwen3 / SigLIP2 run in the Python API.
        const search = await searchArticles({ text: searchQuery, image, imageName: image?.name, mode, signal: req.signal });
        const { context, citations } = buildContext(search);
        send({ type: "sources", searchQuery, search, citations });

        // 3. Answer, with the photo itself when there is one (gpt-4o-mini reads images).
        const content: ChatCompletionContentPart[] = [{ type: "text", text: userPrompt(context, question, !!image) }];
        if (image) {
          const b64 = Buffer.from(await image.arrayBuffer()).toString("base64");
          content.push({ type: "image_url", image_url: { url: `data:${image.type};base64,${b64}`, detail: "low" } });
        }
        const messages: ChatCompletionMessageParam[] = [
          { role: "system", content: SYSTEM_PROMPT },
          ...history.slice(-RECENT_TURNS).map((t) => ({ role: t.role, content: t.content }) as ChatCompletionMessageParam),
          { role: "user", content },
        ];
        const completion = await openai.chat.completions.create(
          { model: MODEL, messages, temperature: 0.3, max_completion_tokens: 1500, stream: true },
          { signal: req.signal },
        );
        for await (const chunk of completion) {
          const text = chunk.choices[0]?.delta?.content;
          if (text) send({ type: "delta", text });
        }
        send({ type: "done" });
      } catch (e) {
        if (!req.signal.aborted) {
          const message = e instanceof RetrievalError || e instanceof Error ? e.message : String(e);
          console.error("[chat]", message);
          send({ type: "error", message });
        }
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-store" },
  });
}
