"""nebula/events.py: one engine stream read as events, and the lifted translators' messages made into the engine's.

The engine streams OpenAI chat chunks: `reasoning_content`, `content` and `tool_calls` deltas, a chunk carrying
`finish_reason`, and (with stream_options.include_usage) one with `usage` and llama-server's `timings`. `Stream`
turns them into serve.frontend's Event objects (reasoning, content, tool_call) so that serve.responses' Assembler
and the Anthropic event generator in nebula/dialects/messages.py read one shape, and keeps the counts in `done`
once the stream has ended.

The lifted translators (serve/frontend.py's openai_to_messages and anthropic_to_messages, serve/responses.py's
input_messages) write the chat template's message form, because Strata's own server renders the template itself.
The cockpit's engine renders its own, so that form is converted here to OpenAI chat messages. Tool-call ids are not
part of the template form: they are generated here, and a tool message takes the id of the call it answers by
position (serve/responses.py's _order_tool_results has already put the results in the calls' order).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field

from nebula._vendor.strata_serve.frontend import Event, ToolCall, tool_arguments


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


@dataclass
class Done:
    finish: str = "stop"                     # the engine's finish_reason: stop | length | tool_calls
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0                   # usage.prompt_tokens_details.cached_tokens
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    model: str = ""
    id: str = ""

    @property
    def read_tokens(self) -> int:
        """Prompt tokens the engine read this time: timings.prompt_n where given, else prompt minus cached."""
        n = self.timings.get("prompt_n")
        return n if isinstance(n, int) else max(self.prompt_tokens - self.cached_tokens, 0)

    @property
    def cache_tokens(self) -> int:
        n = self.timings.get("cache_n")
        return n if isinstance(n, int) else self.cached_tokens


class Stream:
    """Iterate it for Events; read `done` and `used_tool` after the iteration ended. Or, without chunks, feed it
    one chunk at a time (`feed`, then `end`) where the chunks are also wanted as they are."""

    def __init__(self, chunks=None):
        self.chunks = chunks
        self.done = Done()
        self.calls: dict[int, dict] = {}     # tool-call index -> {"id", "name", "arguments"} while it is written
        self.used_tool = False

    def __iter__(self):
        for chunk in self.chunks or ():
            yield from self.feed(chunk)
        yield from self.end()

    def feed(self, chunk: dict):
        """The events of one chunk."""
        self.done.model = chunk.get("model") or self.done.model
        self.done.id = chunk.get("id") or self.done.id
        if isinstance(chunk.get("usage"), dict):
            self._usage(chunk["usage"])
        if isinstance(chunk.get("timings"), dict):
            self.done.timings = chunk["timings"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                yield Event("reasoning", delta["reasoning_content"])
            if delta.get("content"):
                yield Event("content", delta["content"])
            for c in delta.get("tool_calls") or []:
                if isinstance(c, dict):
                    self._call_delta(c)
            if choice.get("finish_reason"):
                self.done.finish = choice["finish_reason"]
                yield from self._flush()

    def end(self):
        """The events of a call the stream ended inside, once there are no more chunks."""
        yield from self._flush()

    def _usage(self, u: dict) -> None:
        self.done.usage = u
        self.done.prompt_tokens = int(u.get("prompt_tokens") or 0)
        self.done.completion_tokens = int(u.get("completion_tokens") or 0)
        details = u.get("prompt_tokens_details") or {}
        self.done.cached_tokens = int(details.get("cached_tokens") or 0)

    def _call_delta(self, c: dict) -> None:
        idx = c.get("index") if isinstance(c.get("index"), int) else 0
        call = self.calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
        if c.get("id"):
            call["id"] = c["id"]
        fn = c.get("function") or {}
        if fn.get("name"):
            call["name"] += fn["name"]
        if fn.get("arguments"):
            call["arguments"] += fn["arguments"]

    def _flush(self):
        for idx in sorted(self.calls):
            c = self.calls[idx]
            self.used_tool = True
            yield Event("tool_call", call=ToolCall(c["name"], tool_arguments(c["arguments"]), c["id"] or new_call_id()))
        self.calls.clear()


def completion_chunks(obj: dict):
    """A whole chat completion as the chunks a stream would have sent, for code that reads one shape."""
    choices = []
    for ch in obj.get("choices") or []:
        msg = ch.get("message") or {}
        delta = {k: msg[k] for k in ("reasoning_content", "content", "tool_calls") if msg.get(k)}
        choices.append({"index": ch.get("index", 0), "delta": delta, "finish_reason": ch.get("finish_reason")})
    yield {"id": obj.get("id"), "model": obj.get("model"), "choices": choices,
           "usage": obj.get("usage"), "timings": obj.get("timings")}


def assemble_reply(events: list[Event]) -> dict:
    """The assistant message a stream's events add up to, in the engine's own (OpenAI) shape: what a client would
    send back as history, plus `reasoning_content` where the engine exposed it."""
    content, reasoning, calls = [], [], []
    for ev in events:
        if ev.kind == "content":
            content.append(ev.text)
        elif ev.kind == "reasoning":
            reasoning.append(ev.text)
        elif ev.kind == "tool_call" and ev.call is not None:
            calls.append({"id": ev.call.id, "type": "function", "function": {
                "name": ev.call.name, "arguments": json.dumps(ev.call.arguments, ensure_ascii=False)}})
    msg: dict = {"role": "assistant", "content": "".join(content) if content or not calls else None}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    return msg


def completion_from(events: list[Event], done: Done, model: str) -> dict:
    """A whole chat completion from a reply's events, for the chat dialect when the cockpit ran the turn itself."""
    import time
    msg = assemble_reply(events)
    finish = "tool_calls" if msg.get("tool_calls") else ("length" if done.finish == "length" else "stop")
    usage = done.usage or {"prompt_tokens": done.prompt_tokens, "completion_tokens": done.completion_tokens,
                           "total_tokens": done.prompt_tokens + done.completion_tokens}
    return {"id": done.id or "chatcmpl-cockpit", "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}], "usage": usage, "timings": done.timings}


def chunks_from(events, model: str, done_after, want_usage: bool):
    """OpenAI chat chunks from a reply's events: for the chat dialect's stream when the cockpit ran the turn itself.
    `done_after()` is read once the events ended, for the finish and the counts."""
    import time
    base = {"id": "chatcmpl-cockpit", "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
    calls = 0
    for ev in events:
        if ev.kind == "reasoning":
            delta = {"reasoning_content": ev.text}
        elif ev.kind == "content":
            delta = {"content": ev.text}
        elif ev.kind == "tool_call" and ev.call is not None:
            delta = {"tool_calls": [{"index": calls, "id": ev.call.id, "type": "function", "function": {
                "name": ev.call.name, "arguments": json.dumps(ev.call.arguments, ensure_ascii=False)}}]}
            calls += 1
        else:
            continue
        yield {**base, "choices": [{"index": 0, "finish_reason": None, "delta": delta}]}
    done = done_after()
    finish = "tool_calls" if calls else ("length" if done.finish == "length" else "stop")
    yield {**base, "choices": [{"index": 0, "finish_reason": finish, "delta": {}}], "timings": done.timings}
    if want_usage:
        usage = done.usage or {"prompt_tokens": done.prompt_tokens, "completion_tokens": done.completion_tokens,
                               "total_tokens": done.prompt_tokens + done.completion_tokens}
        yield {**base, "choices": [], "usage": usage, "timings": done.timings}


# ------------------------------------------------------------------------------------------------ canonical forms
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def image_ref(url: str) -> dict:
    """An image's source as a reference: its hash and size, not its bytes. The ledger keeps references; a data:
    URL of a few megabytes would otherwise be copied into every event that carries it."""
    data = url.encode("utf-8")
    return {"type": "image_ref", "sha256": hashlib.sha256(data).hexdigest(), "length": len(data)}


def stored_message(m: dict) -> dict:
    """A message as the ledger stores it: as sent (any dialect's OpenAI-shaped form), images as references."""
    out = dict(m)
    content = m.get("content")
    if isinstance(content, list):
        parts = []
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") in ("image_url", "input_image", "image"):
                url = p.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                if p.get("type") == "image":
                    src = p.get("source")
                    url = src if isinstance(src, str) else (src or {}).get("url") or (src or {}).get("data") or ""
                parts.append(image_ref(str(url or "")))
            elif p.get("type") == "image_ref":
                parts.append(p)
            else:
                parts.append({"type": "text", "text": p.get("text") or ""})
        out["content"] = parts
    return out


def canonical_message(m: dict) -> dict:
    """The form two copies of one message are compared in, and the form the transcript is hashed in. It leaves out
    what clients legitimately change when they send history back: tool-call ids (regenerated by every translator),
    reasoning (dropped by Anthropic clients), and a tool message's call id. Images are their references."""
    role = "system" if m.get("role") == "developer" else m.get("role")
    stored = stored_message(m)
    content = stored.get("content")
    if isinstance(content, list):
        content = [p if p.get("type") == "image_ref" else {"type": "text", "text": p.get("text", "")} for p in content]
        if all(p["type"] == "text" for p in content):
            content = "".join(p["text"] for p in content)
    out = {"role": role, "content": "" if content is None else content}
    calls = []
    for c in m.get("tool_calls") or []:
        fn = c.get("function", c) if isinstance(c, dict) else {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        calls.append({"name": fn.get("name"), "arguments": args})
    if calls:
        out["tool_calls"] = calls
    return out


def transcript_sha256(canonical_messages: list[dict]) -> str:
    return sha256_of(canonical_messages)


# ------------------------------------------------------------------------------------------------ to the engine
def template_to_openai(messages: list[dict]) -> list[dict]:
    """The template message form -> OpenAI chat messages for the engine."""
    out: list[dict] = []
    pending: list[str] = []                  # ids of the last assistant turn's calls, taken by the tool messages after it
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            content = [{"type": "image_url", "image_url": {"url": p.get("source", "")}} if p.get("type") == "image"
                       else {"type": "text", "text": p.get("text", "")}
                       for p in content if isinstance(p, dict)]
        msg: dict = {"role": role, "content": "" if content is None else content}
        if role == "tool":
            msg["tool_call_id"] = pending.pop(0) if pending else new_call_id()
            if isinstance(content, list) and not any(p["type"] == "image_url" for p in content):
                msg["content"] = "".join(p["text"] for p in content)
        else:
            pending = []
        if m.get("reasoning_content"):
            msg["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c.get("function", c) if isinstance(c, dict) else {}
                args = fn.get("arguments")
                calls.append({"id": new_call_id(), "type": "function", "function": {
                    "name": fn.get("name"),
                    "arguments": args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False)}})
            msg["tool_calls"] = calls
            pending = [c["id"] for c in calls]
        out.append(msg)
    return out


def template_tools_to_openai(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [{"type": "function", "function": {
        "name": t["name"], "description": t.get("description") or "",
        "parameters": t.get("parameters") or {"type": "object", "properties": {}}}} for t in tools]


def engine_request(messages: list[dict], tools: list[dict] | None, kwargs: dict, **fields) -> dict:
    """The body of a streamed engine request. `kwargs` are the template kwargs the translators return
    (enable_thinking False, or a reasoning_effort level): the engine takes them as chat_template_kwargs
    (lite/docs/API.md). Fields given as None are left out."""
    body: dict = {"messages": messages, "stream": True, "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    template: dict = {}
    if kwargs.get("enable_thinking") is False:
        template["enable_thinking"] = False
    elif kwargs.get("reasoning_effort"):
        template["reasoning_effort"] = kwargs["reasoning_effort"]
    if template:
        body["chat_template_kwargs"] = template
    for k, v in fields.items():
        if v is not None:
            body[k] = v
    return body
