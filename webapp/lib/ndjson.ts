/** Incremental NDJSON parser: feed text as it arrives, get back the complete JSON lines. */
export class NdjsonParser<T = unknown> {
  private buf = "";

  push(chunk: string): T[] {
    this.buf += chunk;
    const lines = this.buf.split("\n");
    this.buf = lines.pop() ?? "";
    return lines.filter((l) => l.trim() !== "").map((l) => JSON.parse(l) as T);
  }

  /** Whatever is left after the stream ends (a last line without "\n"). */
  flush(): T[] {
    const rest = this.buf.trim();
    this.buf = "";
    return rest ? [JSON.parse(rest) as T] : [];
  }
}

export function encodeEvent(event: unknown): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(event) + "\n");
}

export async function* readNdjson<T>(body: ReadableStream<Uint8Array>): AsyncGenerator<T> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new NdjsonParser<T>();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const ev of parser.push(decoder.decode(value, { stream: true }))) yield ev;
  }
  for (const ev of parser.push(decoder.decode())) yield ev;
  for (const ev of parser.flush()) yield ev;
}
