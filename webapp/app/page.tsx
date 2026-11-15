"use client";

import { useEffect, useRef, useState } from "react";

import { Answer } from "@/components/Answer";
import { Sources } from "@/components/Sources";
import { readNdjson } from "@/lib/ndjson";
import type { ChatEvent, ChatTurn, Citation, SearchResult } from "@/lib/types";

interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  imageUrl?: string; // object URL of the attached photo (user turns)
  search?: SearchResult;
  citations?: Citation[];
  searchQuery?: string | null;
  error?: string;
  pending?: boolean;
}

const EXAMPLES = [
  "Cheap places to stay near the old town in Aachen",
  "Best beaches in Okinawa",
  "What can I eat in Kyoto?",
];

let nextId = 1;

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function attach(file: File | undefined | null) {
    if (!file || !file.type.startsWith("image/")) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
  }

  function clearImage() {
    setImage(null);
    setPreview(null);
    if (fileRef.current) fileRef.current.value = "";
  }

  function update(id: number, patch: Partial<Message> | ((m: Message) => Partial<Message>)) {
    setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m)));
  }

  async function send(text = input) {
    const question = text.trim();
    if ((!question && !image) || busy) return;

    const history: ChatTurn[] = messages
      .filter((m) => !m.error && !m.pending && m.text)
      .map((m) => ({ role: m.role, content: m.text }));
    const userMsg: Message = { id: nextId++, role: "user", text: question, imageUrl: preview ?? undefined };
    const botMsg: Message = { id: nextId++, role: "assistant", text: "", pending: true };
    setMessages((ms) => [...ms, userMsg, botMsg]);
    setInput("");

    const form = new FormData();
    if (question) form.set("question", question);
    if (image) form.set("image", image, image.name);
    form.set("history", JSON.stringify(history));
    setImage(null);
    setPreview(null);
    if (fileRef.current) fileRef.current.value = "";

    setBusy(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const resp = await fetch("/api/chat", { method: "POST", body: form, signal: ctrl.signal });
      if (!resp.ok || !resp.body) {
        const body = await resp.json().catch(() => ({}));
        throw new Error((body as { error?: string }).error ?? `HTTP ${resp.status}`);
      }
      for await (const ev of readNdjson<ChatEvent>(resp.body)) {
        if (ev.type === "sources") update(botMsg.id, { search: ev.search, citations: ev.citations, searchQuery: ev.searchQuery });
        else if (ev.type === "delta") update(botMsg.id, (m) => ({ text: m.text + ev.text }));
        else if (ev.type === "error") update(botMsg.id, { error: ev.message });
      }
    } catch (e) {
      if (!ctrl.signal.aborted) update(botMsg.id, { error: e instanceof Error ? e.message : String(e) });
    } finally {
      update(botMsg.id, { pending: false });
      setBusy(false);
      abortRef.current = null;
    }
  }

  return (
    <main
      className={dragging ? "dragging" : ""}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        attach(e.dataTransfer.files[0]);
      }}
    >
      <header>
        <h1>Wikivoyage Chat</h1>
        <p className="muted">Ask about a destination, or attach a photo to find where it is.</p>
      </header>

      <section className="thread">
        {messages.length === 0 && (
          <div className="empty">
            {EXAMPLES.map((q) => (
              <button key={q} className="chip" onClick={() => send(q)}>
                {q}
              </button>
            ))}
          </div>
        )}
        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="msg user">
              {m.imageUrl && <img className="thumb" src={m.imageUrl} alt="attached photo" />}
              {m.text && <p>{m.text}</p>}
            </div>
          ) : (
            <div key={m.id} className="msg bot">
              {m.searchQuery && messages.length > 2 && <p className="muted small">searched: “{m.searchQuery}”</p>}
              {m.text ? <Answer text={m.text} citations={m.citations ?? []} /> : m.pending && <p className="muted">Searching…</p>}
              {m.error && <p className="error">{m.error}</p>}
              {m.search && <Sources search={m.search} citations={m.citations ?? []} />}
            </div>
          ),
        )}
        <div ref={endRef} />
      </section>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        {preview && (
          <div className="attached">
            <img src={preview} alt="photo to search with" />
            <button type="button" onClick={clearImage} aria-label="Remove photo">
              ×
            </button>
          </div>
        )}
        <div className="row">
          <button type="button" className="icon" onClick={() => fileRef.current?.click()} aria-label="Attach photo" title="Attach photo">
            📷
          </button>
          <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp,image/gif" hidden onChange={(e) => attach(e.target.files?.[0])} />
          <textarea
            value={input}
            placeholder={image ? "Ask about this photo (optional)…" : "Ask about a destination…"}
            rows={1}
            onChange={(e) => setInput(e.target.value)}
            onPaste={(e) => attach(Array.from(e.clipboardData.files).find((f) => f.type.startsWith("image/")))}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send();
              }
            }}
          />
          {busy ? (
            <button type="button" onClick={() => abortRef.current?.abort()}>
              Stop
            </button>
          ) : (
            <button type="submit" disabled={!input.trim() && !image}>
              Send
            </button>
          )}
        </div>
      </form>
    </main>
  );
}
