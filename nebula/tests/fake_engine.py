"""A fake strata-lite-server for the gateway's tests: the endpoints and chunk shapes of the real one (read from a
live v0.2.1 engine on 2026-10-15), answered deterministically from the request.

- thinking on (the default) adds a reasoning chunk; chat_template_kwargs.enable_thinking false leaves it out;
- a request with tools whose last user text mentions "weather" answers with one get_weather call, its arguments
  split over two chunks, and finish_reason tool_calls;
- any other request answers "Hello" + " world" with finish_reason stop;
- a body over OVERFLOW bytes is llama-server's 400 for a prompt that does not fit;
- the usage chunk carries usage and timings as the engine's does;
- a last user text containing "slowly" is streamed with a pause between chunks, for disconnection tests;
- `on_request(body)`, when set, runs before the answer, for a look at the ledger while the request is in flight;
- a last user text containing "stuck" is answered "I can't do that.";
- a last user text containing "look up <words>", with a `*__search` tool, calls it with {"query": "<words>"};
- a last user text containing "call the tool", with tools, calls the first `__echo` tool (else the first tool);
- a last message of role tool is answered "Tool said: " plus its content;
- a json_schema response_format named "briefing" (the cockpit's compaction) is answered with a fixed briefing;
- usage.prompt_tokens grows with the messages' length (usage_for), and /props reports the `n_ctx` given at start.

`requests` keeps every chat body received, for assertions.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OVERFLOW = 200_000
MODEL_PATH = "/models/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
USAGE = {"prompt_tokens": 26, "completion_tokens": 3, "total_tokens": 29, "prompt_tokens_details": {"cached_tokens": 2}}
TIMINGS = {"cache_n": 2, "prompt_n": 24, "prompt_ms": 300.0, "prompt_per_second": 80.0,
           "predicted_n": 3, "predicted_ms": 75.0, "predicted_per_second": 40.0, "draft_n": 2, "draft_n_accepted": 1}


def last_user_text(body: dict) -> str:
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            return "".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
    return ""


def usage_for(body: dict) -> dict:
    """26 tokens plus a quarter of every message's characters: enough to grow past a small context in a test."""
    n = 26 + sum((len(c) if isinstance(c, str) else len(json.dumps(c))) // 4
                 for c in ((m.get("content") or "") for m in body.get("messages") or []))
    return {"prompt_tokens": n, "completion_tokens": 3, "total_tokens": n + 3, "prompt_tokens_details": {"cached_tokens": 2}}


BRIEFING = {"objective": "answer the user", "constraints": [], "tried": [{"what": "answered", "observed": "ok"}],
            "open": [], "pending_tool_effects": []}


def first_user_text(body: dict) -> str:
    for m in body.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else "".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
    return ""


def chunks_for(body: dict, alias: str = "fake") -> list[dict]:
    base = {"id": "chatcmpl-fake", "model": alias, "object": "chat.completion.chunk", "created": 1}
    thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True) is not False
    out = []
    fmt = body.get("response_format") or {}
    if fmt.get("type") == "json_schema" and (fmt.get("json_schema") or {}).get("name") == "briefing":
        text = json.dumps({**BRIEFING, "objective": first_user_text(body).split("\n")[0][:60] or BRIEFING["objective"]})
        out.append({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": text}}]})
        out.append({**base, "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}]})
        out.append({**base, "choices": [], "usage": usage_for(body), "timings": TIMINGS})
        return out
    if thinking:
        out.append({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"reasoning_content": "Let me think."}}]})
    msgs = body.get("messages") or []
    text_low = last_user_text(body).lower()
    tool_names = [t.get("function", t).get("name") for t in body.get("tools") or [] if isinstance(t, dict)]
    if msgs and msgs[-1].get("role") == "tool":                 # a tool answered: say what it said
        said = msgs[-1].get("content") if isinstance(msgs[-1].get("content"), str) else json.dumps(msgs[-1].get("content"))
        out.append({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Tool said: " + said}}]})
        out.append({**base, "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}]})
    elif "stuck" in text_low:
        out.append({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "I can't do that."}}]})
        out.append({**base, "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}]})
    elif tool_names and "weather" in text_low:
        out.append({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": 0, "id": "tc_fake_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":'}}]}}]})
        out.append({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ' "Paris"}'}}]}}]})
        out.append({**base, "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}], "timings": TIMINGS})
    elif tool_names and "look up " in text_low and any(n.endswith("__search") for n in tool_names):   # Lookup's search
        name = next(n for n in tool_names if n.endswith("__search"))
        q = last_user_text(body)[text_low.index("look up ") + len("look up "):].strip()
        out.append({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": 0, "id": "tc_fake_3", "type": "function", "function": {"name": name, "arguments": json.dumps({"query": q})}}]}}]})
        out.append({**base, "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}], "timings": TIMINGS})
    elif tool_names and "call the tool" in text_low:              # an agent's own tool: the first echo, else the first tool
        name = next((n for n in tool_names if n.endswith("__echo")), tool_names[0])
        args = '{"text": "hi from the engine"}' if name.endswith("__echo") else "{}"
        out.append({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Calling it. "}}]})
        out.append({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": 0, "id": "tc_fake_2", "type": "function", "function": {"name": name, "arguments": args}}]}}]})
        out.append({**base, "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}], "timings": TIMINGS})
    else:
        out.append({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Hello"}}]})
        out.append({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": " world"}}]})
        out.append({**base, "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}]})
    out.append({**base, "choices": [], "usage": usage_for(body), "timings": TIMINGS})
    return out


def scripted(said, alias: str, body: dict) -> list[dict]:
    """A test's own answer: tool calls ([{"name", "arguments"}]) or a text."""
    base = {"id": "chatcmpl-script", "model": alias, "object": "chat.completion.chunk", "created": 1}
    if isinstance(said, str):
        out = [{**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": said}}]},
               {**base, "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}]}]
    else:
        out = [{**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": i, "id": f"tc_s{i}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}}
            for i, c in enumerate(said)]}}]},
               {**base, "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}], "timings": TIMINGS}]
    return out + [{**base, "choices": [], "usage": usage_for(body), "timings": TIMINGS}]


def completion_for(body: dict, alias: str = "fake", chunks: list | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": ""}
    finish = "stop"
    calls: dict[int, dict] = {}
    for ch in (chunks if chunks is not None else chunks_for(body, alias)):
        for c in ch.get("choices") or []:
            d = c.get("delta") or {}
            if d.get("reasoning_content"):
                msg["reasoning_content"] = msg.get("reasoning_content", "") + d["reasoning_content"]
            if d.get("content"):
                msg["content"] += d["content"]
            for tc in d.get("tool_calls") or []:
                call = calls.setdefault(tc["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                call["id"] = tc.get("id") or call["id"]
                call["function"]["name"] += tc.get("function", {}).get("name", "")
                call["function"]["arguments"] += tc.get("function", {}).get("arguments", "")
            if c.get("finish_reason"):
                finish = c["finish_reason"]
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
        msg["content"] = None
    return {"id": "chatcmpl-fake", "model": alias, "object": "chat.completion", "created": 1,
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}], "usage": usage_for(body), "timings": TIMINGS}


class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self._key_refused():
            return
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/props":
            self._json(200, {"model_path": getattr(self.server, "model_path", None) or MODEL_PATH, "model_alias": self.server.alias,
                             "modalities": {"vision": True}, "total_slots": getattr(self.server, "slots", 1),
                             "default_generation_settings": {"n_ctx": self.server.n_ctx}})
        elif self.path == "/slots":
            self._json(200, [{"id": 0, "n_ctx": 262144, "is_processing": False}])
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.server.alias, "object": "model", "owned_by": "llamacpp"}]})
        elif self.path == "/metrics":
            data = b"# HELP fake 1\nfake 1\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json(404, {"error": {"code": 404, "message": "not found", "type": "not_found_error"}})

    def _key_refused(self) -> bool:
        key = getattr(self.server, "api_key", None)
        if key and self.path != "/health" and self.headers.get("Authorization") != f"Bearer {key}":
            self._json(401, {"error": {"code": 401, "message": "Invalid API Key", "type": "authentication_error"}})
            return True
        return False

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        if self._key_refused():
            return
        cache = getattr(self.server, "prefix_cache", None)
        if self.path.startswith("/slots/") and self.path.endswith("?action=erase") and cache is not None:
            slot = int(self.path[len("/slots/"):].split("?")[0])
            cache.erase(slot)
            self.server.erased.append(slot)
            return self._json(200, {"id_slot": slot, "n_erased": 1})
        if self.path == "/tokenize":
            if not self.server.tokenizer:
                return self._json(404, {"error": {"code": 404, "message": "not found", "type": "not_found_error"}})
            text = (json.loads(raw) or {}).get("content") or ""
            return self._json(200, {"tokens": list(range(len(text) // 2))})
        if self.path == "/v1/embeddings":            # a vector per input, from the text's length (a test's stand-in)
            body = json.loads(raw)
            items = body.get("input") if isinstance(body.get("input"), list) else [body.get("input")]
            self.server.requests.append(body)
            return self._json(200, {"object": "list", "model": self.server.alias,
                                    "data": [{"object": "embedding", "index": i, "embedding": [float(len(str(x))), 1.0, 0.5]} for i, x in enumerate(items)],
                                    "usage": {"prompt_tokens": len(items), "total_tokens": len(items)}})
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": {"code": 404, "message": "not found", "type": "not_found_error"}})
        body = json.loads(raw)
        self.server.requests.append(body)
        self.server.headers_seen.append({k: v for k, v in self.headers.items()})
        if self.server.on_request is not None:
            self.server.on_request(body)         # a test's look at the world before the engine answers
        if len(raw) > OVERFLOW:
            return self._json(400, {"error": {"code": 400, "message": "the request exceeds the available context size",
                                              "type": "exceed_context_size_error"}})
        cached = None
        if cache is not None:
            prompt = json.dumps(body.get("messages") or [], sort_keys=True)
            cached = cache.serve(prompt, body.get("id_slot"), body.get("cache_prompt", True))
        if not body.get("stream"):
            said = self.server.script(body) if self.server.script is not None else None
            obj = completion_for(body, self.server.alias, scripted(said, self.server.alias, body) if said is not None else None)
            if cached is not None:
                obj["usage"]["prompt_tokens_details"] = {"cached_tokens": cached}
            return self._json(200, obj)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        slow = "slowly" in last_user_text(body)
        said = self.server.script(body) if self.server.script is not None else None
        for ch in (scripted(said, self.server.alias, body) if said is not None else chunks_for(body, self.server.alias)):
            if cached is not None and isinstance(ch.get("usage"), dict):
                ch["usage"] = dict(ch["usage"], prompt_tokens_details={"cached_tokens": cached})
            if not ch.get("choices") and not want_usage:
                continue
            self.wfile.write(b"data: " + json.dumps(ch).encode() + b"\n\n")
            self.wfile.flush()
            if slow:
                time.sleep(0.3)                  # a client that hangs up mid-stream is seen as a broken pipe
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class PrefixCache:
    """A llama-server-like prompt cache per slot, for the tenancy tests: a request reuses the longest prefix its slot
    holds (its `id_slot`, else the slot whose prompt matches best, else the least recently used); `cache_prompt: false`
    reuses nothing; a slot can be erased. The reused part is reported as cached tokens (a quarter of its characters)."""

    def __init__(self, slots: int = 2):
        self.held = [""] * slots
        self.used = [0.0] * slots
        self.lock = threading.Lock()

    def serve(self, prompt: str, id_slot=None, cache_prompt=True) -> int:
        with self.lock:
            lcp = lambda a: len(os.path.commonprefix([a, prompt]))
            if isinstance(id_slot, int) and 0 <= id_slot < len(self.held):
                s = id_slot
            else:
                best = max(range(len(self.held)), key=lambda i: lcp(self.held[i]))
                s = best if lcp(self.held[best]) > 0.1 * len(prompt) else min(range(len(self.held)), key=lambda i: self.used[i])
            reused = lcp(self.held[s]) if cache_prompt is not False else 0
            self.held[s], self.used[s] = prompt, time.time()
            return reused // 4

    def erase(self, slot: int) -> None:
        with self.lock:
            self.held[slot] = ""


class FakeEngine(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionError, OSError)):   # a cockpit that hung up: no trace
            super().handle_error(request, client_address)

    def __init__(self, alias: str = "fake", n_ctx: int = 262144, port: int = 0, background: bool = True):
        super().__init__(("127.0.0.1", port), FakeHandler)
        self.requests: list[dict] = []
        self.erased: list[int] = []                  # slots erased (POST /slots/N?action=erase), with a prefix_cache
        self.prefix_cache = None                     # a PrefixCache to report cached prompt tokens like llama-server
        self.api_key = None                          # when set, every path but /health needs it
        self.headers_seen: list[dict] = []
        self.on_request = None
        self.script = None                           # body -> a list of tool calls ({name, arguments}) or a text, or None
        self.alias, self.n_ctx = alias, n_ctx
        self.tokenizer = False                       # POST /tokenize answers one token per two characters when on
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        if background:
            self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def stop(self):
        self.shutdown()
        self.server_close()
