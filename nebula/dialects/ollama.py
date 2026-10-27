"""nebula/dialects/ollama.py: Ollama's API, for apps that speak it (an API entry point with the Ollama style only;
its paths, /api/chat and the like, are the page's on Nebula's own port).

    POST /api/chat       {model, messages (images: base64), stream (default true), options, think, format}
    POST /api/generate   {model, prompt, system, images, stream, options}
    GET  /api/tags       the models this entry point offers
    GET  /api/version

A request becomes an OpenAI chat request and goes through the same Switchyard (nebula/dialects/chat.py: its routing,
gates, tenancy and ledger); the answer comes back as Ollama's: one JSON object, or newline-delimited objects while
streaming, ending with `done: true` and the counts.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from nebula.dialects import JsonReply
from nebula.dialects.chat import chat

VERSION = "0.6.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _content(m: dict):
    imgs = m.get("images") or []
    text = m.get("content") or ""
    if not imgs:
        return text
    return [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + i}} for i in imgs]


def _openai(req: dict, prompt_mode: bool) -> dict:
    o = req.get("options") if isinstance(req.get("options"), dict) else {}
    if prompt_mode:
        msgs = ([{"role": "system", "content": req["system"]}] if req.get("system") else []) + \
            [{"role": "user", "content": _content({"content": req.get("prompt") or "", "images": req.get("images")})}]
    else:
        msgs = [{"role": m.get("role") or "user", "content": _content(m)} for m in req.get("messages") or [] if isinstance(m, dict)]
    out = {"model": req.get("model"), "messages": msgs, "stream": req.get("stream", True) is not False}
    for k_in, k_out in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("seed", "seed"),
                        ("num_predict", "max_tokens"), ("stop", "stop"), ("repeat_penalty", "repeat_penalty")):
        if o.get(k_in) is not None:
            out[k_out] = o[k_in]
    if req.get("think") is not None:
        out["chat_template_kwargs"] = {"enable_thinking": bool(req["think"])}
    if req.get("format") == "json":
        out["response_format"] = {"type": "json_object"}
    elif isinstance(req.get("format"), dict):
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": req["format"]}}
    if out["stream"]:
        out["stream_options"] = {"include_usage": True}
    return out


def _done(model: str, usage: dict | None, reason: str | None, t0: float) -> dict:
    u = usage or {}
    return {"model": model, "created_at": _now(), "done": True, "done_reason": reason or "stop",
            "total_duration": int((time.time() - t0) * 1e9), "prompt_eval_count": u.get("prompt_tokens") or 0,
            "eval_count": u.get("completion_tokens") or 0}


def handle(engine, ctx, path: str, req: dict):
    """(status, a JSON body) or (200, an iterator of NDJSON lines)."""
    prompt_mode = path == "/api/generate"
    model = req.get("model")
    if not isinstance(model, str) or not model:
        return 400, {"error": "model is required"}
    t0 = time.time()
    oreq = _openai(req, prompt_mode)
    reply = chat(engine, ctx, oreq)
    key = "response" if prompt_mode else "message"
    if isinstance(reply, JsonReply):
        if reply.status != 200:
            return reply.status, {"error": str((reply.body.get("error") or {}).get("message") or reply.body)}
        c = (reply.body.get("choices") or [{}])[0]
        msg = c.get("message") or {}
        body = {"model": model, "created_at": _now(),
                key: (msg.get("content") or "") if prompt_mode else {"role": "assistant", "content": msg.get("content") or ""},
                **_done(model, reply.body.get("usage"), c.get("finish_reason"), t0)}
        if not prompt_mode and msg.get("reasoning_content"):
            body["message"]["thinking"] = msg["reasoning_content"]
        return 200, body

    def lines():
        usage, reason = None, None
        for frame in reply.frames:
            for ln in frame.decode("utf-8", "replace").splitlines():
                if not ln.startswith("data: ") or ln[6:].strip() == "[DONE]":
                    continue
                try:
                    ch = json.loads(ln[6:])
                except ValueError:
                    continue
                usage = ch.get("usage") or usage
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    reason = c.get("finish_reason") or reason
                    if d.get("content") or d.get("reasoning_content"):
                        out = {"model": model, "created_at": _now(), "done": False}
                        if prompt_mode:
                            out["response"] = d.get("content") or ""
                        else:
                            out["message"] = {"role": "assistant", "content": d.get("content") or ""}
                            if d.get("reasoning_content"):
                                out["message"]["thinking"] = d["reasoning_content"]
                        yield (json.dumps(out) + "\n").encode()
        last = _done(model, usage, reason, t0)
        last[key] = "" if prompt_mode else {"role": "assistant", "content": ""}
        yield (json.dumps(last) + "\n").encode()
    return 200, lines()


def tags(cfg, allowed: list[str] | None) -> dict:
    names = [t.name for t in cfg.tiers if not allowed or t.name in allowed]
    return {"models": [{"name": n, "model": n, "modified_at": _now(), "size": 0, "digest": "",
                        "details": {"format": "gguf", "family": "", "parameter_size": "", "quantization_level": ""}} for n in names]}
