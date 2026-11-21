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

const MODE_OPTIONS: { value: "both" | "text" | "image"; icon: string; label: string }[] = [
  { value: "both", icon: "🔎", label: "Text + Images" },
  { value: "text", icon: "📝", label: "Text only" },
  { value: "image", icon: "🖼️", label: "Images only" },
];

export default function Page() {
  const nextIdRef = useRef(1);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [mode, setMode] = useState<"both" | "text" | "image">("both");
  const [modeMenuOpen, setModeMenuOpen] = useState(false);
  const modeMenuRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (modeMenuRef.current && !modeMenuRef.current.contains(e.target as Node)) setModeMenuOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setModeMenuOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, []);

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
    const userMsg: Message = { id: nextIdRef.current++, role: "user", text: question, imageUrl: preview ?? undefined };
    const botMsg: Message = { id: nextIdRef.current++, role: "assistant", text: "", pending: true };
    setMessages((ms) => [...ms, userMsg, botMsg]);
    setInput("");

    const form = new FormData();
    if (question) form.set("question", question);
    if (image) form.set("image", image, image.name);
    form.set("history", JSON.stringify(history));
    form.set("mode", mode);
    setImage(null);
    setPreview(null);
    setMode("both"); // don't let a one-off mode choice silently apply to unrelated future questions
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
          <div className="mode-menu" ref={modeMenuRef}>
            <button
              type="button"
              className="icon"
              onClick={() => setModeMenuOpen((o) => !o)}
              aria-haspopup="true"
              aria-expanded={modeMenuOpen}
              aria-label="Search mode"
              title={MODE_OPTIONS.find((o) => o.value === mode)?.label}
            >
              {MODE_OPTIONS.find((o) => o.value === mode)?.icon}
            </button>
            {modeMenuOpen && (
              <div className="mode-dropdown" role="menu" aria-label="Search mode">
                {MODE_OPTIONS.map((o) => (
                  <button
                    key={o.value}
                    type="button"
                    role="menuitemradio"
                    aria-checked={mode === o.value}
                    className={mode === o.value ? "active" : ""}
                    onClick={() => {
                      setMode(o.value);
                      setModeMenuOpen(false);
                    }}
                  >
                    <span className="mode-icon">{o.icon}</span>
                    <span>{o.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
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
