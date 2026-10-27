"""nebula/dialects/messages.py: Anthropic Messages, translated.

The request goes through serve/frontend.py's anthropic_to_messages (lifted), then to the engine as a streamed chat
completion; the engine's events come back as Anthropic's: message_start, content_block_start / _delta / _stop for
thinking, text and tool_use blocks, message_delta with the stop reason and the counts, message_stop. Modelled on
serve/server.py's anthropic_events and anthropic_collect, without Strata's piecewise tool-call streaming: the
engine sends a call's arguments as they come and the cockpit delivers the call whole.

Counts, Anthropic's way (serve/server.py, #1013): input_tokens is what the engine read this turn
(timings.prompt_n) and cache_read_input_tokens what its prompt cache already held (timings.cache_n); input_tokens
is never 0, since Claude Code reads 0 as "not given" and then counts the prompt twice. message_start cannot know
the prompt's size before the engine has read it, so it says 0 there; the real counts come in message_delta.
"""
from __future__ import annotations

import json
import uuid

from nebula._vendor.strata_serve.frontend import anthropic_to_messages, tool_choice_of

from nebula.dialects import JsonReply, SseReply, framed, requested_model
from nebula.events import engine_request, template_to_openai, template_tools_to_openai
from nebula.turn import Turn

STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def messages(engine, ctx, req: dict):
    cfg = ctx.cfg
    model = requested_model(req, cfg)
    msgs, tools, kw = anthropic_to_messages(req, True)
    kind, name = tool_choice_of(req.get("tool_choice"))
    if kind == "none":                           # Anthropic's {"type": "none"}: no tools offered
        tools = None
    choice = None
    if tools and kind == "required":
        choice = "required"
    elif tools and kind == "named":
        choice = {"type": "function", "function": {"name": name}}
    openai_messages, openai_tools = template_to_openai(msgs), template_tools_to_openai(tools)
    body = engine_request(openai_messages, openai_tools, kw,
                          max_tokens=req.get("max_tokens"), temperature=req.get("temperature"),
                          top_p=req.get("top_p"), top_k=req.get("top_k"),
                          stop=req.get("stop_sequences") or None, tool_choice=choice)
    thinking = False if kw.get("enable_thinking") is False else (True if kw.get("reasoning_effort") else None)
    turn = Turn(ctx, "messages", req)
    engine, body = turn.prepare(body, messages=openai_messages, tools=openai_tools, requested_model=model,
                                thinking_requested=thinking)
    events = anthropic_events(turn.run(engine, body), model, turn)
    if not req.get("stream"):
        return JsonReply(200, anthropic_collect(events))
    return SseReply(framed(events, lambda e: e[0], lambda e: e[1]))


def anthropic_events(events, model: str, turn: Turn):
    """(event name, event) pairs from the turn's events. The engine's first event is read before message_start
    goes out, so an engine that refuses the request raises here, before any answer was started."""
    it = iter(events)
    try:
        first = next(it)
    except StopIteration:
        first = None

    def rest():
        if first is not None:
            yield first
        yield from it

    mid = "msg_" + uuid.uuid4().hex[:24]
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}}
    index, open_kind, used_tool = -1, None, False
    for ev in rest():
        want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use"}.get(ev.kind)
        if want is None or (ev.kind != "tool_call" and not ev.text):
            continue
        if open_kind != want or want == "tool_use":
            if open_kind is not None:
                yield "content_block_stop", {"type": "content_block_stop", "index": index}
            index += 1
            open_kind = want
            block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                     "text": {"type": "text", "text": ""},
                     "tool_use": {"type": "tool_use", "id": ev.call.id if ev.call else "",
                                  "name": ev.call.name if ev.call else "", "input": {}}}[want]
            yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}
        if want == "thinking":
            delta = {"type": "thinking_delta", "thinking": ev.text}
        elif want == "text":
            delta = {"type": "text_delta", "text": ev.text}
        else:
            used_tool = True
            delta = {"type": "input_json_delta", "partial_json": json.dumps(ev.call.arguments, ensure_ascii=False)}
        yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}
    if open_kind is not None:
        yield "content_block_stop", {"type": "content_block_stop", "index": index}
    d = turn.done
    stop = "tool_use" if used_tool and d.finish in ("stop", "tool_calls") else STOP.get(d.finish, "end_turn")
    read, cached = d.read_tokens, d.cache_tokens
    if read < 1 and cached > 0:                  # the last prompt token is always read (#1013)
        read, cached = 1, cached - 1
    yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                            "usage": {"input_tokens": read, "cache_read_input_tokens": cached,
                                      "output_tokens": d.completion_tokens}}
    yield "message_stop", {"type": "message_stop"}


def anthropic_collect(events) -> dict:
    """The non-streaming answer: the message, its blocks built from the same events."""
    msg, blocks = None, []
    for name, e in events:
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            else:
                b["_json"] = b.get("_json", "") + d["partial_json"]
        elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
            b = blocks[-1]
            try:
                b["input"] = json.loads(b.pop("_json") or "{}")
            except ValueError:                   # a call the output ended inside: it has no input to give
                blocks.pop()
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["stop_sequence"] = e["delta"].get("stop_sequence")
            msg["usage"].update(e["usage"])
    msg["content"] = blocks
    return msg
