import type { Citation } from "@/lib/types";

/** Plain text with paragraphs and **bold**; [n] becomes a link to the cited section. */
export function Answer({ text, citations }: { text: string; citations: Citation[] }) {
  const byN = new Map(citations.map((c) => [c.n, c]));
  return (
    <div className="answer">
      {text.split(/\n{2,}/).map((para, i) => (
        <p key={i}>{renderInline(para, byN)}</p>
      ))}
    </div>
  );
}

function renderInline(text: string, byN: Map<number, Citation>) {
  const out: React.ReactNode[] = [];
  const re = /\[(\d+)\]|\*\*([^*]+)\*\*|\n/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const key = `${m.index}`;
    if (m[1]) {
      const c = byN.get(Number(m[1]));
      out.push(
        c ? (
          <a key={key} className="cite" href={c.url} target="_blank" rel="noreferrer" title={`${c.title} — ${c.section}`}>
            {m[1]}
          </a>
        ) : (
          `[${m[1]}]`
        ),
      );
    } else if (m[2]) {
      out.push(<strong key={key}>{m[2]}</strong>);
    } else {
      out.push(<br key={key} />);
    }
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
