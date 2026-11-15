import { describe, expect, it } from "vitest";

import { buildContext, chunkBody, condenseMessages, userPrompt } from "./prompt";
import type { SearchResult } from "./types";

const search: SearchResult = {
  query: { mode: "text", encoders: ["qwen3", "siglip2-text"] },
  articles: [
    {
      page_id: 1, score: 0.03, title: "Aachen", url: "https://w/Aachen", page_type: "city", status: "", is_part_of: "NRW",
      chunks: [
        { offset: 4, score: 0.74, chunk_index: 4, section_path: "Sleep > Budget", url: "https://w/Aachen#Budget",
          text: "Aachen (city, NRW) — Sleep > Budget\nHostel A near the station." },
        { offset: 0, score: 0, chunk_index: 0, section_path: "", url: "", text: "Aachen (city, NRW)\nAachen is a spa city." },
      ],
      images: [
        { offset: 0, score: 0.2, image_url: "i", image_page_url: "", caption: "Cathedral", section_path: "", license: "CC0",
          license_url: "", artist: "", file_title: "File:C.jpg" },
        { offset: 1, score: 0.1, image_url: "j", image_page_url: "", caption: "", section_path: "", license: "CC0",
          license_url: "", artist: "", file_title: "File:D.jpg" },
      ],
    },
    { page_id: 2, score: 0.01, title: "Bonn", url: "https://w/Bonn", page_type: "city", status: "", is_part_of: "",
      chunks: [], images: [] },
  ],
};

describe("buildContext", () => {
  it("numbers chunks, then the article's photo captions", () => {
    const { context, citations } = buildContext(search);
    expect(citations).toEqual([
      { n: 1, title: "Aachen", section: "Sleep > Budget", url: "https://w/Aachen#Budget" },
      { n: 2, title: "Aachen", section: "Introduction", url: "https://w/Aachen" },
      { n: 3, title: "Aachen", section: "Photos", url: "https://w/Aachen" },
    ]);
    expect(context).toContain("[1] Aachen (city, NRW) — Sleep > Budget\nHostel A near the station.");
    expect(context).not.toContain("— Sleep > Budget\nAachen (city");
    expect(context).toContain('[3] Aachen — photos in the article: "Cathedral"');
  });

  it("is empty for no articles", () => {
    expect(buildContext({ ...search, articles: [] })).toEqual({ context: "", citations: [] });
  });

  it("truncates very long passages", () => {
    const long = structuredClone(search);
    long.articles[0].chunks[0].text = "h\n" + "x".repeat(5000);
    expect(buildContext(long).context.length).toBeLessThan(2500);
  });
});

describe("prompts", () => {
  it("chunkBody drops the header line", () => {
    expect(chunkBody("Title — S\nbody\nmore")).toBe("body\nmore");
    expect(chunkBody("no header")).toBe("no header");
  });

  it("condense keeps the last 6 turns and the new question", () => {
    const history = Array.from({ length: 10 }, (_, i) => ({ role: i % 2 ? "assistant" : "user", content: `t${i}` }) as const);
    const msgs = condenseMessages([...history], "food there?");
    expect(msgs[1].content).not.toContain("t3");
    expect(msgs[1].content).toContain("t4");
    expect(msgs[1].content).toContain("Last message: food there?");
  });

  it("userPrompt asks about the photo when there is no question", () => {
    expect(userPrompt("ctx", null, true)).toContain("What place is in this photo?");
    expect(userPrompt("", "q", false)).toContain("(no passages found)");
  });
});
