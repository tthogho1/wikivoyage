"""RAG answer generation using OpenAI chat completions."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("wikivoyage.chat")

_SYSTEM_PROMPT = """\
You are a knowledgeable travel assistant specialising in Wikivoyage destination information.
Answer the user's question using ONLY the provided context passages.
If the answer is not contained in the context, say "I don't have enough information about that destination."
Be concise, helpful, and specific. Mention the city/destination name when relevant.
"""

_CONTEXT_TEMPLATE = """\
[{i}] {title} ({page_type}{status_str})
{text}
"""


def _build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        status_str = f", {h['status']}" if h.get("status") else ""
        parts.append(
            _CONTEXT_TEMPLATE.format(
                i=i,
                title=h["title"],
                page_type=h["page_type"],
                status_str=status_str,
                text=h["retrieval_text"],
            )
        )
    return "\n---\n".join(parts)


def generate_answer(question: str, hits: list[dict], openai_client: Any) -> str:
    """Call OpenAI chat completions with retrieved context and return the answer."""
    if not hits:
        return "I could not find any relevant information for your question."

    context = _build_context(hits)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context passages:\n\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    logger.info("Calling OpenAI model=%s with %d context passages …", model, len(hits))
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()
