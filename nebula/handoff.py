"""nebula/handoff.py: the packet a crossing carries, the prompt a crossed tier reads, structured compaction.

The packet is built from the ledger, not from the request: the original task, the constraints the client stated
(its system messages, quoted), why the turn is moving, the latest briefing if there is one, every transcript event
after it in canonical form, the tool state, the coverage and the budget. It carries no ids and no times, so the
same conversation gives the same packet bytes, and its hash with the request's is the handoff's idempotency key.

A crossed tier reads a prompt rendered from the packet: a preamble (deterministic text from the packet's fields),
then the events since the briefing as messages. Later turns on that tier are rendered the same way, so the
destination's prompt cache holds the conversation from the crossing on.

Compaction asks the tier's own engine for a fixed shape (objective, constraints, tried and observed, open, pending
tool effects) as JSON under a schema the engine enforces, and the ledger stores it as a briefing event. It runs on
the user's slot: after a reply whose prompt passed the configured share of the context, or before a crossing whose
packet would not fit the destination.
"""
from __future__ import annotations

import json
import logging

from nebula.events import canonical_json, canonical_message, sha256_of, template_to_openai
from nebula.ledger import text_of

LOG = logging.getLogger("cockpit.handoff")
PACKET_VERSION = 1
ORIGINAL_TASK_CHARS = 2000                 # the first message, verbatim up to here; a briefing's objective carries the rest
BRIEFING_SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {"type": "string"},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "tried": {"type": "array", "items": {"type": "object", "properties": {
            "what": {"type": "string"}, "observed": {"type": "string"}}, "required": ["what", "observed"],
            "additionalProperties": False}},
        "open": {"type": "array", "items": {"type": "string"}},
        "pending_tool_effects": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["objective", "constraints", "tried", "open", "pending_tool_effects"],
    "additionalProperties": False,
}
COMPACTION_INSTRUCTION = (
    "Summarize the conversation so far for another model that will continue it. Answer with JSON only, in this "
    "shape: objective (the user's goal, one sentence), constraints (the user's stated requirements, quoted), tried "
    "(what was attempted and what each attempt showed), open (what is still unresolved), pending_tool_effects "
    "(tool calls whose outcome is not yet known). Do not add facts the conversation does not contain.")


# ------------------------------------------------------------------------------------------------ rendering
def _text_content(content) -> str | list:
    """Canonical content for an engine: text as it is; image references as a note, since the bytes are not kept."""
    if isinstance(content, list):
        parts = []
        for p in content:
            if p.get("type") == "image_ref":
                parts.append({"type": "text", "text": f"[image omitted: sha256 {p['sha256'][:16]}, {p['length']} bytes]"})
            else:
                parts.append({"type": "text", "text": p.get("text", "")})
        return "".join(x["text"] for x in parts)
    return content if isinstance(content, str) else ""


def canonical_to_template(c: dict) -> dict:
    """A canonical message -> the template form template_to_openai takes (ids are given there, by position)."""
    out = {"role": c.get("role"), "content": _text_content(c.get("content"))}
    if c.get("tool_calls"):
        out["tool_calls"] = [{"function": {"name": t.get("name"), "arguments": t.get("arguments")
                              if isinstance(t.get("arguments"), dict) else {"arguments": t.get("arguments")}}}
                             for t in c["tool_calls"]]
    return out


def rendered_message(payload: dict) -> dict:
    """A stored payload as a message an engine can read: images as notes, everything else as stored."""
    out = dict(payload)
    out["content"] = _text_content(payload.get("content")) if isinstance(payload.get("content"), list) else payload.get("content")
    return out


def preamble(packet: dict) -> str:
    b = packet.get("briefing")
    lines = ["You are continuing a conversation that another model handed over."]
    if not b:                                # with a briefing, its objective stands for the task
        lines.append(f"Original task: {packet['original_task']}")
    if packet["constraints"]:
        lines.append("Constraints the user stated:")
        lines += [f"- {c}" for c in packet["constraints"]]
    reason = packet["escalation_reason"]
    lines.append(f"Why it was handed over: {reason['source']}: {reason['text']}")
    if b:
        lines.append("Briefing from the previous model:")
        lines.append(f"Objective: {b.get('objective', '')}")
        if b.get("constraints"):
            lines.append("Constraints: " + "; ".join(b["constraints"]))
        for t in b.get("tried") or []:
            lines.append(f"Tried: {t.get('what', '')} -> observed: {t.get('observed', '')}")
        if b.get("open"):
            lines.append("Open: " + "; ".join(b["open"]))
        if b.get("pending_tool_effects"):
            lines.append("Pending tool effects: " + "; ".join(b["pending_tool_effects"]))
    if packet["tool_state"]["pending"]:
        lines.append("Tool calls without a result yet: " + ", ".join(packet["tool_state"]["pending"]))
    lines.append("Continue from the messages that follow and answer the last one.")
    return "\n".join(lines)


def render_messages(packet: dict) -> list[dict]:
    """The prompt a crossed tier reads: the preamble, then the events since the briefing."""
    events = [canonical_to_template(c) for c in packet["events_since_briefing"]]
    return [{"role": "system", "content": preamble(packet)}] + template_to_openai(events)


def estimate_tokens(messages: list[dict]) -> int:
    """About four characters a token, plus a few per message: an estimate for budgets, not a count."""
    return len(canonical_json(messages)) // 4 + 8 * len(messages)


# ------------------------------------------------------------------------------------------------ the packet
def build_packet(ledger, session_id: str, *, reason: dict, destination_context: int | None) -> dict:
    conv = ledger.transcript(session_id)
    latest = ledger.latest_briefing(session_id)
    # the boundary is what the briefing covers, not where its event sits: a briefing made for a crossing is written
    # after the request's own new messages, which must follow it in the packet
    briefing_seq = latest[1].get("covers_through_seq", latest[0]) if latest else 0
    briefing = {k: v for k, v in latest[1].items() if k in BRIEFING_SCHEMA["properties"]} if latest else None
    constraints = [text_of(p.get("content")) for _s, p in conv if p.get("role") == "system" and text_of(p.get("content"))]
    first = next((p for _s, p in conv if p.get("role") == "user"), None)
    events = [canonical_message(p) for seq, p in conv if seq > briefing_seq and p.get("role") != "system"]
    pending = []
    if conv and conv[-1][1].get("role") == "assistant":
        pending = [t.get("function", t).get("name") for t in conv[-1][1].get("tool_calls") or []]
    task = text_of(first.get("content")) if first else ""
    if len(task) > ORIGINAL_TASK_CHARS:
        task = task[:ORIGINAL_TASK_CHARS] + f" [... {len(task) - ORIGINAL_TASK_CHARS} more characters]"
    packet = {"version": PACKET_VERSION,
              "original_task": task,
              "constraints": constraints,
              "escalation_reason": {"source": reason.get("source", ""), "text": reason.get("text", "")},
              "briefing": briefing,
              "events_since_briefing": events,
              "tool_state": {"pending": pending, "uncertain": []},
              "coverage": {"through_seq": conv[-1][0] if conv else 0, "briefing_seq": briefing_seq or None, "gaps": []},
              "budget": {"estimated_tokens": 0, "destination_context": destination_context, "fits": True}}
    est = estimate_tokens(render_messages(packet))
    packet["budget"] = {"estimated_tokens": est, "destination_context": destination_context,
                        "fits": destination_context is None or est <= int(destination_context * 0.9)}
    return packet


def packet_sha256(packet: dict) -> str:
    return sha256_of(packet)


def idempotency_key(packet_sha: str, request_sha: str) -> str:
    return sha256_of({"packet_sha256": packet_sha, "request_sha256": request_sha})


# ------------------------------------------------------------------------------------------------ compaction
def full_read_due(briefings: int, every: int) -> bool:
    """A briefing that reads the whole conversation again instead of building on the last one: every `every`-th of a
    session (0: never). Updating a briefing from itself drifts a little over many updates; the Halo model's 2026-10-21
    measurement (24 held-out conversations, ~7 updates each, graded against the conversation): never 1.79 faithful and 5/24
    with an invented detail, every 3rd 1.83 and 4/24, every 5th 1.92 and 2/24 (the default)."""
    return bool(every) and briefings > 0 and (briefings + 1) % int(every) == 0


def transcript_messages(ledger, session_id: str, until_seq: int | None = None, full: bool = False) -> list[dict]:
    """The conversation as an engine would read it now: the latest briefing as a system note, then the events
    since it; or all of it when there is no briefing (or `full`). `until_seq` stops before a request's own new messages."""
    latest = None if full else ledger.latest_briefing(session_id)
    out = []
    since = 0
    if latest:
        b = latest[1]
        since = b.get("covers_through_seq", latest[0])
        out.append({"role": "system", "content": "Summary of the conversation so far:\n" + preamble(
            {"original_task": b.get("objective", ""), "constraints": [], "escalation_reason": {"source": "", "text": ""},
             "briefing": b, "tool_state": {"pending": []}}).split("\n", 1)[1]})
    out += [rendered_message(p) for _s, p in ledger.transcript(session_id, since, until_seq)]
    return out


def compact(engine, messages: list[dict]) -> dict | None:
    """One engine call for the briefing; None when the engine's answer is not the shape asked for."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[-1].get("role") == "user" and isinstance(msgs[-1].get("content"), str):
        msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + COMPACTION_INSTRUCTION
    else:
        msgs.append({"role": "user", "content": COMPACTION_INSTRUCTION})
    body = {"messages": msgs, "stream": False, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "briefing", "strict": True,
                                                                        "schema": BRIEFING_SCHEMA}}}
    obj = engine.chat(body)
    text = ((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        b = json.loads(text)
    except ValueError:
        LOG.warning("compaction: the engine's answer is not JSON (%d chars)", len(text))
        return None
    if not isinstance(b, dict) or any(k not in b for k in BRIEFING_SCHEMA["required"]):
        LOG.warning("compaction: the engine's answer is missing fields: %s", sorted(b) if isinstance(b, dict) else type(b))
        return None
    return {k: b[k] for k in BRIEFING_SCHEMA["properties"]}


def compaction_due(prompt_tokens: int, n_ctx: int | None, fraction: float, briefing_seq: int | None,
                   transcript_seq: int) -> bool:
    if not n_ctx or prompt_tokens < fraction * n_ctx:
        return False
    return briefing_seq is None or transcript_seq > briefing_seq
