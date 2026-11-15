import { describe, expect, it } from "vitest";

import { NdjsonParser, encodeEvent, readNdjson } from "./ndjson";

describe("NdjsonParser", () => {
  it("handles lines split across chunks", () => {
    const p = new NdjsonParser<{ a: number }>();
    expect(p.push('{"a":1}\n{"a"')).toEqual([{ a: 1 }]);
    expect(p.push(":2}\n\n")).toEqual([{ a: 2 }]);
    expect(p.push('{"a":3}')).toEqual([]);
    expect(p.flush()).toEqual([{ a: 3 }]);
  });
});

describe("readNdjson", () => {
  it("reads a byte stream, including multibyte characters split between chunks", async () => {
    const bytes = new Uint8Array([...encodeEvent({ t: "日本" }), ...encodeEvent({ t: "x" })]);
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(bytes.slice(0, 8)); // splits inside "日"
        c.enqueue(bytes.slice(8));
        c.close();
      },
    });
    const out = [];
    for await (const ev of readNdjson(stream)) out.push(ev);
    expect(out).toEqual([{ t: "日本" }, { t: "x" }]);
  });
});
