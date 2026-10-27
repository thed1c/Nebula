"""nebula/sizing.py: how big a request is, and which ticked models can hold it (Switchyard plumbing step C).

The size is counted before routing: with the engine's tokenizer (`POST /tokenize`, llama-server's) when the engine
has one, else four characters a token with a margin, plus a few tokens a message and the tools' JSON. What a
request needs is that plus the output it asked for (`max_tokens`, else a reserve). A model holds it when its
context per slot is at least that; a model whose context is unknown (a provider with none configured) is taken
to hold it, since nothing says otherwise.
"""
from __future__ import annotations

import json

from nebula.events import canonical_json

PER_MESSAGE = 6
MARGIN = 1.1
OUTPUT_RESERVE = 1024


def text_of_request(messages: list[dict], tools) -> str:
    parts = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") in ("text", "input_text") and p.get("text"):
                    parts.append(p["text"])
        if m.get("tool_calls"):
            parts.append(canonical_json(m["tool_calls"]))
        if m.get("reasoning_content"):
            parts.append(str(m["reasoning_content"]))
    if tools:
        parts.append(canonical_json(tools))
    return "\n".join(parts)


def estimate_tokens(engine, messages: list[dict], tools, margin: float = MARGIN) -> tuple[int, str]:
    """(tokens, how): the engine's tokenizer when it answers, else the character estimate."""
    text = text_of_request(messages, tools)
    n = engine.tokenize(text) if engine is not None else None
    if n is not None:
        return n + PER_MESSAGE * len(messages or []), "tokenizer"
    return int(len(text) / 4 * margin) + PER_MESSAGE * len(messages or []), "estimate"


def output_for(max_tokens, reserve: int = OUTPUT_RESERVE, context: int | None = None) -> int:
    """The output a request needs room for: what it asked for, else the reserve, never more than an eighth of the
    slot when the request named nothing (a small slot is not refused for a large default)."""
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0:
        return max_tokens
    if context:
        return max(16, min(reserve, int(context) // 8))
    return reserve


def needed(estimate: int, max_tokens, reserve: int = OUTPUT_RESERVE, context: int | None = None) -> int:
    return estimate + output_for(max_tokens, reserve, context)


def holds(context: int | None, estimate: int, max_tokens=None, reserve: int = OUTPUT_RESERVE) -> bool:
    return context is None or int(context) >= needed(estimate, max_tokens, reserve, context)
