#!/usr/bin/env python3
"""One Engine P2: impulse-fx's request / answer layer (src/fx/fx_front.*) against serve/server.py's own code, on the CPU.

    python fx_front_parity.py --fx BUILD/impulse-fx --gguf DENSE_OR_META.gguf --tokenizer PACK/tokenizer [--out summary.json]

Needs: the pack's tokenizer folder (vocab.json, merges.txt, token_type.json, chat_template.jinja: what server.py reads)
and the dense GGUF or tools/gguf_meta_only.py's metadata copy of it (what impulse-fx reads); a Python with jinja2 and
regex (Pillow for the picture-format cases).  Nothing runs on a GPU.

1. PROMPTS: for every request of the corpus, server.py's path (_openai up to Service.prepare: with_shared, logit_bias,
   openai_to_messages, tool_choice, forced_call, prepare_format, max tokens, the 400 checks, the literal marks, the
   template, the tokenizer, the images' pads, the context check) and impulse-fx's (--fx-front-test "prepare") must give
   the SAME token ids, the same GEN head (max_new and the sampling keys) - or both refuse the request.  The corpus: the
   pack's chat_golden.json, the request shapes of the reference engine's callers (Nebula, an agent router's role agents, the pipeline's
   picture checks, an app harness, a watchdog's probe; docs/FX-PARITY.md), long real texts, and edge cases (literal
   tags, control-token text, Unicode whitespace, floats in schemas, tool histories, efforts, formats, sampling types).
2. PARSER: OutputParser's events for model outputs fed token by token (the pack's tokenizer's pieces) and in random
   pieces: reasoning, answers, tool calls of every parameter type, calls inside the reasoning, quoted tags, code
   fences, unclosed calls, undeclared names (with and without tool_name_repair).
3. RUN: Service._run + openai_chunks/openai_collect on a scripted engine against impulse-fx's ("tokens"): the answer
   (content, reasoning_content, tool calls, finish_reason, usage) and every GEN head sent - EOS literals, the thinking
   budget's wrap-up, forced calls, stop strings, the repeat guard.
"""
import argparse
import copy
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
IMPULSE = HERE.parent
FLASHNEXT = IMPULSE / "flashnext"
sys.path.insert(0, str(FLASHNEXT / "tools"))
sys.path.insert(0, str(FLASHNEXT))

from serve import server as SV                       # noqa: E402
from serve import frontend as FE                     # noqa: E402
from serve import logit_bias as BIAS                 # noqa: E402
from serve.structured import prepare_format          # noqa: E402
import strata_tokenizer as ST                        # noqa: E402

CTX = 262144


# ------------------------------------------------------------------------------------------------ server.py's side
def strata_tokenizer(tdir):
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tdir / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


class FakeVision:
    """Vision's interface for prepare(): (an embeddings file named by the source's hash, its tokens) from a table."""
    elastic = False

    def __init__(self, table, d):
        self.table, self.dir = table, pathlib.Path(d)

    def alive(self):
        return True

    def encode(self, src, keep=()):
        if src not in self.table:
            raise ValueError("the image could not be read: " + src[:40])
        p = self.dir / (hashlib.sha256(src.encode()).hexdigest()[:32] + ".sve")
        p.write_bytes(b"x" * 16)
        return p, self.table[src]

    def encode_all(self, sources, cancel=None):
        return [self.encode(s) for s in sources]


def py_prepare(svc, tok, req, images, ctx):
    """_openai's steps up to Service.prepare -> (ids, head, max_new, thinking, force) or ("error", message)."""
    req = copy.deepcopy(req)
    svc.vision = FakeVision(images, svc._vdir) if images is not None else None
    svc.engine.max_context = ctx
    try:
        req = svc.with_shared(req, "openai")
        BIAS.normalize(req.get("logit_bias"), len(tok.tokens))
        messages, tools, kw = FE.openai_to_messages(req)
        if FE.tool_choice_of(req.get("tool_choice"))[0] == "none":
            tools = None
        force = FE.forced_call(req.get("tool_choice"), tools)
        messages, validator = prepare_format(req.get("response_format"), messages)
        if validator is not None and (tools or req.get("strata_mcp")):
            raise ValueError("structured response_format with tools/MCP is not supported")
        max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)
        svc.reasoning_budget(req)
        SV.stop_strings(req)
        SV.check_request_sampling(req)
        ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, force=force, req=req)
        svc.drop_embeddings()
        head = f"GEN {max_new}{SV.StrataEngine.sampling_keys(req)}"
        return {"ids": ids, "head": head, "max_new": max_new, "thinking": thinking, "force": force or ""}
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, UnicodeError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------------------------------------ impulse-fx's side
class Fx:
    def __init__(self, binary, gguf):
        self.p = subprocess.Popen([binary, "--fx-front-test", gguf, "--context", str(CTX)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self.errlines = []
        threading.Thread(target=lambda: [self.errlines.append(x) for x in self.p.stderr], daemon=True).start()

    def ask(self, q):
        self.p.stdin.write(json.dumps(q, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit("impulse-fx --fx-front-test ended:\n" + "".join(self.errlines[-20:]))
        return json.loads(line)

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=30)


# ------------------------------------------------------------------------------------------------ the corpus
def real_texts():
    out = []
    for p in [IMPULSE.parent.parent / "x"] + sorted((IMPULSE.parent.parent / "docs").glob("*.md"))[:6] + \
            [FLASHNEXT / "README.md", FLASHNEXT / "IMPULSE.md", IMPULSE / "README.md"]:
        try:
            out.append(p.read_text(encoding="utf-8"))
        except OSError:
            pass
    return out


def png_bytes(w=2, h=2):
    import struct
    import zlib
    raw = b"".join(b"\x00" + b"\x10\x20\x30" * w for _ in range(h))
    def chunk(k, d):
        return struct.pack(">I", len(d)) + k + d + struct.pack(">I", zlib.crc32(k + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def data_url(b, mime="image/png"):
    import base64
    return f"data:{mime};base64," + base64.b64encode(b).decode()


ROLE_TOOLS = [
    {"type": "function", "function": {
        "name": "memory",
        "description": "Save durable facts that apply to every session. target 'user' = who the user is; 'memory' = your notes.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["add", "replace", "remove"]},
            "target": {"type": "string", "enum": ["memory", "user"]},
            "content": {"type": "string"}, "old_text": {"type": "string"}}, "required": ["action", "target"]}}},
    {"type": "function", "function": {
        "name": "tool_result_page",
        "description": "Read the next page of a tool result that was longer than 60000 characters.",
        "parameters": {"type": "object", "properties": {"cursor": {"type": "string", "maxLength": 80}}, "required": ["cursor"]}}},
    {"type": "function", "function": {
        "name": "scene_lookup",
        "description": "Look up a scene of the story by its code: its card, cast, place and the approved keyframe.",
        "parameters": {"type": "object", "properties": {
            "scene_code": {"type": "string", "pattern": "^[A-Z0-9_]+$"},
            "include": {"type": "array", "items": {"type": "string", "enum": ["card", "cast", "keyframe", "notes"]}},
            "max_chars": {"type": "integer", "minimum": 1000, "maximum": 60000, "default": 20000},
            "strictness": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
            "dry_run": {"type": "boolean"}}, "required": ["scene_code"]}}},
    {"type": "function", "function": {
        "name": "session_search",
        "description": "Search earlier sessions' notes. Returns at most `limit` hits, newest first.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer", "default": 5},
            "since": {"anyOf": [{"type": "string", "format": "date"}, {"type": "null"}]},
            "weights": {"type": "object", "properties": {"title": {"type": "number", "default": 2.5},
                                                        "body": {"type": "number", "default": 1.0}}}},
            "required": ["query"]}}},
]


def corpus(texts, rng):
    C = []

    def add(name, req, images=None, ctx=None):
        C.append({"name": name, "req": req, "images": images, "ctx": ctx})

    # the pack's golden cases, as chat requests
    for case in json.loads((FLASHNEXT / "serve" / "chat_golden.json").read_text()):
        req = {"model": "flash-next", "messages": case["messages"]}
        if case.get("tools"):
            req["tools"] = [{"type": "function", "function": t} for t in case["tools"]]
        if case.get("enable_thinking") is False:
            req["chat_template_kwargs"] = {"enable_thinking": False}
        add("golden: " + case["name"], req)

    long = texts[0] if texts else "x " * 5000
    sys_director = ("You are the visual director of a story project. Answer as JSON when asked. " * 4) + "\n\n" + long[:6000]
    hist = [{"role": "system", "content": sys_director},
            {"role": "user", "content": "Plan the keyframe for scene C1_S04. Use the tools."},
            {"role": "assistant", "content": "", "reasoning_content": "I should look the scene up first.",
             "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "scene_lookup",
                                                                              "arguments": json.dumps({"scene_code": "C1_S04", "include": ["card", "cast"], "max_chars": 20000, "strictness": 0.75})}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"card": long[:1500], "cast": ["Ann", "Ben"]})},
            {"role": "assistant", "content": "The scene is in the library.",
             "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "memory", "arguments": {"action": "add", "target": "memory", "content": "C1_S04: library, evening"}}}]},
            {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
            {"role": "user", "content": "Now write the shot list."}]
    # an agent router's role agents: {model, messages, tools, tool_choice: auto}, non-streaming, + the generation policy
    policy = {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_format": "deepseek", "reasoning_budget_tokens": 8192, "temperature": 0.2}
    add("role agent + policy", {"model": "flash-next", "messages": hist, "tools": ROLE_TOOLS, "tool_choice": "auto", **policy})
    add("role agent, stream", {"model": "flash-next", "messages": hist[:4], "tools": ROLE_TOOLS, "tool_choice": "auto", "stream": True, **policy})
    add("long-form writer", {"model": "flash-next", "max_tokens": 20000, "temperature": 0.4,
                              "messages": [{"role": "system", "content": "You are a long-form writer."},
                                           {"role": "user", "content": long[:9000]}]})
    add("long-form writer + policy", {"model": "flash-next", "max_tokens": 20000, "temperature": 0.4, "messages": hist[:2] + [{"role": "user", "content": long[:3000]}], **policy})
    # Nebula's door: stream + include_usage, enable_thinking false, thinking_budget, response_format json_schema
    schema = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["ok", "fix"]}, "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                               "notes": {"type": "array", "items": {"type": "string"}}}, "required": ["verdict"], "additionalProperties": False}
    add("nebula typed tier", {"model": "FlashNext", "messages": [{"role": "user", "content": "Check this line: " + long[:500]}], "stream": True,
                              "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
                              "response_format": {"type": "json_schema", "json_schema": {"name": "check", "schema": schema, "strict": True}}, "max_tokens": 4000})
    add("nebula clamp", {"model": "FlashNext", "messages": [{"role": "user", "content": "Summarize: " + long[:2000]}], "max_tokens": 1200,
                         "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 800}, "reasoning_budget_tokens": 800})
    add("nebula probe", {"model": "FlashNext", "messages": [{"role": "user", "content": "Say: ready"}], "max_tokens": 200})
    add("watchdog probe", {"model": "flash-next", "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                           "messages": [{"role": "user", "content": "ping"}]})
    add("speed probe", {"model": "flash-next", "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": 500,
                    "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": "Write a short story about the sky."}]})
    add("app harness", {"model": "auto", "messages": [{"role": "system", "content": "You help write a visual novel."},
                                                      {"role": "user", "content": "Suggest three beats for the next scene."}], "stream": False,
                        "chat_template_kwargs": {"enable_thinking": True}})
    # the pipeline's picture checks: text + image_url data URIs (labelled png / jpeg)
    p1, p2 = data_url(png_bytes(2, 2)), data_url(png_bytes(3, 1), "image/jpeg")
    add("pipeline picture check", {"model": "flash-next-alias", "max_tokens": 6000, "stream": False,
                                   "messages": [{"role": "system", "content": "You review scene renders."},
                                                {"role": "user", "content": [{"type": "text", "text": json.dumps({"design": long[:800], "standing_checks": ["eyes", "hands"]})},
                                                                             {"type": "image_url", "image_url": {"url": p1}}]}]}, {p1: 1024})
    add("pipeline crops", {"model": "flash-next-alias", "max_tokens": 3000,
                           "messages": [{"role": "user", "content": [{"type": "text", "text": "These are close-up crops, numbered in order."}] +
                                         [{"type": "image_url", "image_url": {"url": u}} for u in (p1, p2, p1)]}]}, {p1: 256, p2: 81})
    add("picture without an encoder", {"model": "x", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": p1}}]}]}, None)
    add("tool picture unreadable", {"model": "x", "messages": [{"role": "user", "content": "look"},
                                                               {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Read", "arguments": "{\"path\": \"a.png\"}"}}]},
                                                               {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": "the file"}]},
                                                               {"role": "user", "content": "and?"}]}, {p1: 4})
    add("picture + vision tags in text", {"model": "x", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "the template writes <|vision_start|><|image_pad|><|vision_end|> for a picture: "},
        {"type": "image_url", "image_url": p1}]}]}, {p1: 7})
    add("anthropic-style image part", {"model": "x", "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": p1.split(",", 1)[1]}}, {"type": "text", "text": "what is it"}]}]},
        {"data:image/png;base64," + p1.split(",", 1)[1]: 9})

    # long real texts (17K-token class)
    big = "\n\n".join(texts)
    add("long read", {"model": "flash-next", "messages": [{"role": "user", "content": big[:70000] + "\n\nSummarize the above."}], "max_tokens": 2048})

    # edge cases
    lit = ["<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>", "<tool_response>",
           "</tool_response>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>"]
    for i, t in enumerate(lit):
        add(f"literal {t} in user", {"messages": [{"role": "user", "content": f"Quote the tag {t} exactly, then `{t}` and \"{t}\"."}]})
        add(f"literal {t} in system+tool", {"messages": [{"role": "system", "content": "Docs: " + t}, {"role": "user", "content": "hi"},
                                                         {"role": "assistant", "content": f"see {t}", "tool_calls": [{"function": {"name": "f", "arguments": {"x": t}}}]},
                                                         {"role": "tool", "content": "result " + t}, {"role": "user", "content": "ok"}],
                                            "tools": [{"type": "function", "function": {"name": "f", "description": "d " + t, "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}}]})
    add("assistant inline think", {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "<think>\nmy </think> plan <tool_call>\n</think>\n\nanswer <|im_end|>"}, {"role": "user", "content": "b"}]})
    add("assistant inline think, leading space", {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "  <think>x</think>y"}, {"role": "user", "content": "b"}]})
    add("user wrapped tool response", {"messages": [{"role": "user", "content": "q"}, {"role": "user", "content": "<tool_response>\nthe <think> result\n</tool_response>"}]})
    add("user wrapped tool response, parts", {"messages": [{"role": "user", "content": "q"}, {"role": "user", "content": [{"type": "text", "text": " <tool_response>a</think>"}, {"type": "text", "text": "b</tool_response> "}]}]})
    add("only tool responses (template refuses)", {"messages": [{"role": "user", "content": "<tool_response>\nx\n</tool_response>"}]})
    for ws in [" ", "　", " ", "\x1c", "\x85", " \t\n", "​", "﻿"]:
        add(f"whitespace {ws!r}", {"messages": [{"role": "system", "content": ws + "sys" + ws}, {"role": "user", "content": ws + "hello" + ws},
                                                {"role": "assistant", "content": ws + "hi" + ws, "reasoning_content": ws + "r" + ws}, {"role": "user", "content": "x" + ws}]})
    add("empty assistant turns dropped", {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "b"},
                                                       {"role": "assistant", "content": "  "}, {"role": "user", "content": "c"}, {"role": "assistant", "content": ""}]})
    add("developer and late system", {"messages": [{"role": "developer", "content": "dev"}, {"role": "user", "content": "a"}, {"role": "system", "content": "late"},
                                                   {"role": "developer", "content": "later"}, {"role": "user", "content": "b"}]})
    add("parts: refusal, unknown, input_text, untyped", {"messages": [{"role": "user", "content": [{"type": "input_text", "text": "a"}, {"text": "b"}, {"type": "file", "file": {}},
                                                                                                  {"type": "refusal", "refusal": "no"}, {"type": "text", "text": "c"}]},
                                                                      {"role": "assistant", "content": [{"type": "refusal", "refusal": "I won't"}]}, {"role": "user", "content": "d"}]})
    add("messages as a JSON string", {"messages": json.dumps([{"role": "user", "content": "x"}])})
    add("messages not a list", {"messages": "not json"})
    add("message without role", {"messages": [{"content": "x"}]})
    add("no messages", {"model": "x"})
    add("content a number", {"messages": [{"role": "user", "content": 5}]})
    floats = {"type": "object", "properties": {"t": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7, "multipleOf": 1e-7, "big": 1e20,
                                                     "pi": 3.14159265358979, "n": -0.0, "e": 2.5e-05, "x": 1e16, "y": 1e15, "z": 123456789.125}}}
    add("floats in schemas", {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function", "function": {"name": "f", "parameters": floats}}]})
    add("floats in args", {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 0.5, "b": 1.0, "c": 1e-07, "d": [1.5, 2.0], "e": {"f": 3.0}, "g": True, "h": None, "i": "s"}}}]},
                                           {"role": "tool", "content": "r"}, {"role": "user", "content": "y"}], "tools": [{"type": "function", "function": {"name": "f", "parameters": floats}}]})
    add("non-ascii tool schema", {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function", "function": {"name": "f", "description": "Löwe 日本 \U0001f600   \x7f", "parameters": {"type": "object"}}}]})
    for args in ['{"a": 1}', "", "   ", "{bad json", "[1, 2]", "5", None, {"a": [1, {"b": 2}]}, [1, 2], 7, '"str"']:
        add(f"tool args {args!r}", {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "c", "tool_calls": [{"function": {"name": "f", "arguments": args}}]},
                                                 {"role": "tool", "content": "r"}, {"role": "tool", "content": "r2"}, {"role": "user", "content": "y"}]})
    add("tool_calls as a JSON string", {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": json.dumps([{"function": {"name": "f", "arguments": "{}"}}])}, {"role": "user", "content": "y"}]})
    add("tool_calls bad", {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [{"function": "f"}]}]})
    add("tool call without function wrapper", {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [{"name": "g", "arguments": {"k": "v"}}]}, {"role": "user", "content": "y"}]})
    add("bare tools", {"messages": [{"role": "user", "content": "x"}], "tools": [{"name": "a", "parameters": {"type": "object"}}, {"type": "function", "function": {"name": "b"}}]})
    add("tools bad", {"messages": [{"role": "user", "content": "x"}], "tools": "auto"})
    add("tool without name", {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function", "function": {"description": "d"}}]})
    add("tool parameters a string", {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function", "function": {"name": "a", "parameters": "{}"}}]})
    # the stable tool order: the same set in other orders, one request after another (both sides keep their own LRU)
    names = ["alpha", "beta", "gamma", "delta"]
    for k in range(4):
        order = names[:]
        rng.shuffle(order)
        add(f"tool order {k}", {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}} for n in order]})
    for tc in ["auto", "none", "required", {"type": "function", "function": {"name": "memory"}}, {"type": "function", "name": "session_search"},
               {"type": "tool", "name": "nope"}, {"type": "any"}, "weird", 5]:
        for think in (True, False):
            add(f"tool_choice {tc!r} thinking {think}", {"messages": [{"role": "user", "content": "x"}], "tools": ROLE_TOOLS, "tool_choice": tc,
                                                         "chat_template_kwargs": {"enable_thinking": think}})
    add("tool_choice required, one tool", {"messages": [{"role": "user", "content": "x"}], "tools": ROLE_TOOLS[:1], "tool_choice": "required", "chat_template_kwargs": {"enable_thinking": False}})
    for eff in ["low", "medium", "high", "xhigh", "max", "none", "off", "minimal", " LOW ", False, "", None, "bogus", True, 3]:
        add(f"reasoning_effort {eff!r}", {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": eff})
    add("reasoning.effort", {"messages": [{"role": "user", "content": "x"}], "reasoning": {"effort": "low"}})
    add("effort + kwargs off", {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}})
    add("kwargs effort", {"messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": {"reasoning_effort": "medium"}})
    add("kwargs effort after off", {"messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": {"enable_thinking": 0, "reasoning_effort": "low"}})
    add("kwargs effort then off", {"messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": {"reasoning_effort": "low", "enable_thinking": ""}})
    add("kwargs thinking on + effort", {"messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}})
    add("kwargs not a dict", {"messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": ["a"]})
    add("low effort + tools", {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "x"}], "tools": ROLE_TOOLS[:2], "reasoning_effort": "low"})
    for rf in [{"type": "json_object"}, {"type": "text"}, {"type": "json_schema", "json_schema": {"name": "n", "schema": schema}},
               {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "array"}}}, {"type": "json_schema", "json_schema": {"schema": schema}},
               {"type": "json_schema", "json_schema": {"name": "n", "schema": {"$ref": "#/$defs/a", "$defs": {"a": {"type": "object"}}}}},
               {"type": "json_schema", "json_schema": {"name": "n", "schema": {"anyOf": [{"type": "object"}, {"type": "object", "properties": {"x": {"$ref": "http://x"}}}]}}},
               {"type": "yaml"}, "json"]:
        add(f"response_format {json.dumps(rf)[:60]}", {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "x"}], "response_format": rf})
    add("response_format, no system, list system", {"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}})
    add("response_format + tools", {"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}, "tools": ROLE_TOOLS[:1]})
    for st in ["END", ["a", "b"], ["", "x"], [1], ["a", "b", "c", "d", "e"], None]:
        add(f"stop {st!r}", {"messages": [{"role": "user", "content": "x"}], "stop": st})
    for mt in [100, "100", 100.7, -1, 0, "abc", True, CTX * 2, None]:
        add(f"max_tokens {mt!r}", {"messages": [{"role": "user", "content": "x"}], "max_tokens": mt})
    add("max_completion_tokens wins", {"messages": [{"role": "user", "content": "x"}], "max_completion_tokens": 50, "max_tokens": 70})
    add("context overflow", {"messages": [{"role": "user", "content": long[:4000]}]}, ctx=500)
    add("context overflow with max_tokens", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 600}, ctx=500)
    for samp in [{"temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "seed": 7}, {"temperature": "0.7", "top_k": "40"}, {"temperature": 1, "top_k": 100},
                 {"temperature": 0.123456789012, "top_p": 1.0, "top_k": 0}, {"repetition_penalty": 1.1}, {"repeat_penalty": 1.05, "repeat_last_n": 128},
                 {"presence_penalty": 1.5, "frequency_penalty": 0.25, "penalty_last_n": 256}, {"top_k": 40.0}, {"seed": True}, {"temperature": "hot"},
                 {"strata_tune": {"pcie_frac": 0.04, "spec_min_p": 0.5, "aux_cpus": True}}, {"strata_checkpoint": False},
                 {"experimental_speed_projection": False}, {"logit_bias": {"12": -100}}, {"logit_bias": [[5, 1.5], [6, False]]}, {"logit_bias": {"x": 1}},
                 {"reasoning_budget_tokens": 8192}, {"reasoning_budget_tokens": "lots"}, {"reasoning_budget_tokens": 100.0}]:
        add(f"sampling {json.dumps(samp)[:70]}", {"messages": [{"role": "user", "content": "x"}], **samp})
    # random mixes of real text pieces with tags and whitespace
    pieces = lit + ["\n", "\n\n", " ", "\t", " ", "```", "`", "\"", "日本語", "\U0001f600", "<function=f>", "<parameter=x>"]
    for k in range(400):
        def rtext():
            t = texts[rng.randrange(len(texts))] if texts else "abc"
            a = rng.randrange(max(1, len(t) - 400))
            s = t[a:a + rng.randrange(1, 400)]
            for _ in range(rng.randrange(6)):
                at = rng.randrange(len(s) + 1)
                s = s[:at] + rng.choice(pieces) + s[at:]
            return s
        msgs = [{"role": "system", "content": rtext()}] if rng.random() < 0.5 else []
        for j in range(rng.randrange(1, 5)):
            msgs.append({"role": "user", "content": rtext()})
            if rng.random() < 0.6:
                m = {"role": "assistant", "content": rtext()}
                if rng.random() < 0.4:
                    m["reasoning_content"] = rtext()
                msgs.append(m)
        if msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": rtext()})
        req = {"messages": msgs}
        if rng.random() < 0.3:
            req["tools"] = ROLE_TOOLS
        if rng.random() < 0.3:
            req["reasoning_effort"] = rng.choice(["low", "medium", "high", "none"])
        add(f"random mix {k}", req)
    return C


# ------------------------------------------------------------------------------------------------ parser and run cases
def parser_outputs():
    T = [
        "Let me think about this.</think>\n\nThe answer is 42.",
        "plan</think>\n\n\n\nAnswer with newlines\n\nand more",
        "I'll call it.\n</think>\n\n<tool_call>\n<function=scene_lookup>\n<parameter=scene_code>\nC1_S04\n</parameter>\n<parameter=include>\n[\"card\", \"cast\"]\n</parameter>\n<parameter=max_chars>\n20000\n</parameter>\n<parameter=strictness>\n0.75\n</parameter>\n<parameter=dry_run>\nTrue\n</parameter>\n</function>\n</tool_call>",
        "x</think>\n\nSome text first.\n\n<tool_call>\n<function=memory>\n<parameter=action>\nadd\n</parameter>\n<parameter=target>\nmemory\n</parameter>\n<parameter=content>\nA value with </parameter> inside and \"quotes\" and 日本 \U0001f600\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=tool_result_page>\n<parameter=cursor>\nabc\n</parameter>\n</function>\n</tool_call>",
        "t</think>\n\n<tool_call>\n<function=session_search>\n<parameter=query>\nfalse\n</parameter>\n<parameter=limit>\nFalse\n</parameter>\n<parameter=since>\nnull\n</parameter>\n<parameter=weights>\n{\"title\": 2.5, \"body\": 1.0}\n</parameter>\n</function>\n</tool_call>",
        "t</think>\n\n<tool_call>\n<function=memory>\n<parameter=action>\nadd",                       # unclosed: the output ends inside it
        "t</think>\n\n<tool_call>\n<function=memory>\n<parameter=action>\nadd\n</parameter>\n</function>",   # no </tool_call>
        "t</think>\n\n<tool_call>\n<function=memory>\n<parameter=action>\nadd\n</parameter>\n</function>\nThen text after.",
        "t</think>\n\n<tool_call>\n<function=undeclared_tool>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>",
        "t</think>\n\n<tool_call>\n<function=Memory>\n<parameter=action>\nadd\n</parameter>\n</function>\n</tool_call>",
        "t</think>\n\n<tool_call>\n<function=tools>\n<function=memory>\n<parameter=action>\nadd\n</parameter>\n</function>\n</function>\n</tool_call>",
        "t</think>\n\nI'd use a <tool_call> block here, like `<tool_call>` in code.",
        "t</think>\n\n```\n<tool_call>\n<function=memory>\n</function>\n</tool_call>\n```\nafter",
        "thinking about `</think>` and \"</think>\" as quotes</think>\n\nanswer",
        "in reasoning:\n<tool_call>\n<function=memory>\n<parameter=action>\nadd\n</parameter>\n</function>\n</tool_call>\n</think>\n\n",
        "in reasoning:\n<tool_call>\n<function=memory>\n<parameter=action>\nadd\n</parameter>\n</function>\n</tool_call>\nthen more thinking</think>\n\nok",
        "a quoted <tool_call> mid-sentence</think>\n\nok",
        "no end of thinking at all",
        "</think>",
        "t</think>\n\n<tool_call>\n<function=memory>\n<parameter=action>\n\n\n</parameter>\n<parameter=content>\nx\n</parameter>\n</function>\n</tool_call>",
        "t</think>\n\n<tool_call>\n<function= memory >\n<parameter=action>\nadd\n</parameter>\n</function>\n</tool_call>",
        "t</think>\n\n<tool_call>\n  <function=scene_lookup>\n<parameter=include>\nnot json\n</parameter>\n<parameter=max_chars>\n1e3\n</parameter>\n</function>\n</tool_call>",
        "t</think>\n\nanswer with a ``` fence\n```python\nprint('<tool_call>')\n```\n",
    ]
    return T


def run_cases():
    """(name, req, prompt text, passes [texts], config): the scripted engine's passes for Service._run."""
    return [
        ("answer", {}, True, ["thinking</think>\n\nThe answer.<|im_end|>"], {}),
        ("length", {}, True, ["thinking for a long time and never"], {}),
        ("no thinking", {"chat_template_kwargs": {"enable_thinking": False}}, False, ["Just the answer.<|im_end|>"], {}),
        ("tool call", {"tools": ROLE_TOOLS}, True, ["plan</think>\n\n<tool_call>\n<function=memory>\n<parameter=action>\nadd\n</parameter>\n<parameter=target>\nuser\n</parameter>\n</function>\n</tool_call><|im_end|>"], {}),
        ("two calls", {"tools": ROLE_TOOLS}, True, ["p</think>\n\n<tool_call>\n<function=memory>\n<parameter=action>\nadd\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=scene_lookup>\n<parameter=scene_code>\nC1\n</parameter>\n<parameter=max_chars>\n500\n</parameter>\n</function>\n</tool_call><|im_end|>"], {}),
        ("unclosed call", {"tools": ROLE_TOOLS}, True, ["p</think>\n\n<tool_call>\n<function=memory>\n<parameter=action>\nadd<|im_end|>"], {}),
        ("call in reasoning", {"tools": ROLE_TOOLS}, True, ["r\n<tool_call>\n<function=memory>\n<parameter=action>\nadd\n</parameter>\n</function>\n</tool_call>\n<|im_end|>"], {}),
        ("stop string", {"stop": ["STOP"]}, True, ["t</think>\n\nbefore STOP after<|im_end|>"], {}),
        ("stop string split", {"stop": ["END OF"]}, True, ["t</think>\n\nsome END", " OF text<|im_end|>"], {}),
        ("eos literal after space", {}, True, ["t</think>\n\nWrite the token: <|im_end|>", "> done.<|im_end|>"], {}),
        ("eos literal in backticks", {}, True, ["t</think>\n\nIt is `<|im_end|>", ">` here.<|im_end|>"], {}),
        ("eos literal off", {}, True, ["t</think>\n\nWrite the token: <|im_end|>"], {"stop_token_text": False}),
        ("eos literal in a call", {"tools": ROLE_TOOLS}, True, ["t</think>\n\n<tool_call>\n<function=memory>\n<parameter=content>\nthe end is <|im_end|>", ">\n</parameter>\n</function>\n</tool_call><|im_end|>"], {}),
        ("budget wrap-up", {"reasoning_budget_tokens": 5, "max_tokens": 4096}, True, ["one two three four five six seven", "The answer after the wrap-up.<|im_end|>"], {}),
        ("budget capped by max_tokens", {"reasoning_budget_tokens": 8192, "max_tokens": 600}, True, ["a b c d e f g h i j k l m n o p q r s t u v w x y z " * 30, "ok<|im_end|>"], {}),
        ("forced call, thinking", {"tools": ROLE_TOOLS, "tool_choice": {"type": "function", "function": {"name": "memory"}}}, True,
         ["I will call it.</think>\n\n", "<parameter=action>\nadd\n</parameter>\n</function>\n</tool_call><|im_end|>"], {}),
        ("forced call, no thinking", {"tools": ROLE_TOOLS, "tool_choice": "required", "chat_template_kwargs": {"enable_thinking": False}}, False,
         ["memory>\n<parameter=action>\nadd\n</parameter>\n</function>\n</tool_call><|im_end|>"], {}),
        ("repeat guard", {}, True, ["!" * 300], {"repeat_stop_tokens": 50}),
        ("unicode split", {}, True, ["t</think>\n\n日本語 \U0001f600 Löwe<|im_end|>"], {}),
        ("undeclared, repair", {"tools": ROLE_TOOLS}, True, ["t</think>\n\n<tool_call>\n<function=Memory>\n<parameter=action>\nadd\n</parameter>\n</function>\n</tool_call><|im_end|>"], {"tool_name_repair": True}),
        ("undeclared, no repair", {"tools": ROLE_TOOLS}, True, ["t</think>\n\n<tool_call>\n<function=nope>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call><|im_end|>"], {}),
        ("sampling keys", {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "seed": 3, "strata_checkpoint": False}, True, ["t</think>\n\nx<|im_end|>"], {}),
        ("uncensored", {"uncensored": False}, True, ["t</think>\n\nx<|im_end|>"], {}),
        ("uncensored, explicit key wins", {"uncensored": False, "experimental_speed_projection": True}, True, ["t</think>\n\nx<|im_end|>"], {}),
        ("tune keys", {"strata_tune": {"pcie_frac": 0.04, "aux_cpus": 1}, "logit_bias": {"5": -100}}, True, ["t</think>\n\nx<|im_end|>"], {}),
        ("cancelled by stop in reasoning? no: stop only cuts content", {"stop": ["think"]}, True, ["I think so</think>\n\nthink twice<|im_end|>"], {}),
        ("eos literal in code block example", {}, True, ["t</think>\n\n```\n<|im_start|>user\nhi<|im_end|>", "\n```\nok<|im_end|>"], {}),
    ]


class ScriptEngine:
    """generate() as StrataEngine's for Service._run: each pass's ids from a list, the GEN heads recorded."""
    batch = 0

    def __init__(self, passes):
        self.passes, self.heads, self.last, self.info, self.max_context = list(passes), [], {}, {}, CTX

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.heads.append(f"GEN {int(max_new)}{SV.StrataEngine.sampling_keys(sampling or {})} #{len(ids)}")
        out = self.passes.pop(0) if self.passes else []
        eos = bool(out) and out[-1] in SV.EOS_IDS
        try:
            for t in out:
                yield t
        finally:
            self.last = {"generated": len(out), "prompt_tokens": 10, "prompt_ms": 1.0, "decode_ms": 2.0,
                         "finish": "stop" if eos else "length", "drafts_accepted": 0, "drafts_offered": 0, "reused": 0,
                         "hits": 0, "lookups": 0, "ram_blobs": 0, "file_blobs": 0, "file_mb": 0.0, "prompt_read": 10, "offloaded": 0}


def strip_ids(r):
    if isinstance(r, dict):
        r = {k: strip_ids(v) for k, v in r.items() if k not in ("id", "created")}
    elif isinstance(r, list):
        r = [strip_ids(x) for x in r]
    return r


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    tdir = pathlib.Path(a.tokenizer)
    tok = strata_tokenizer(tdir)
    template = FE.ChatTemplate(tdir / "chat_template.jinja")
    engine = SV.MockEngine(tok, "x", max_context=CTX)
    svc = SV.Service(engine, tok, template, model_name="flash-next")
    svc._vdir = tempfile.mkdtemp(prefix="fxparity-")
    svc.set_aliases(["flash-next-alias", "FlashNext"])
    fx = Fx(a.fx, a.gguf)
    rng = random.Random(a.seed)
    texts = real_texts()
    t0 = time.time()
    summary = {"prompts": {"cases": 0, "accepted": 0, "refused": 0, "tokens": 0, "plain_spans": 0, "refused_cases": [], "message_differs": [], "mismatches": []},
               "parser": {"cases": 0, "mismatches": []}, "run": {"cases": 0, "mismatches": []}}

    # 1. prompts
    P = summary["prompts"]
    for case in corpus(texts, rng):
        ctx = case["ctx"] or CTX
        py = py_prepare(svc, tok, case["req"], case["images"], ctx)
        q = {"op": "prepare", "req": case["req"], "context": ctx}
        if case["images"] is not None:
            q["images"] = case["images"]
        cx = fx.ask(q)
        P["cases"] += 1
        bad = None
        if "error" in py or "error" in cx:
            if ("error" in py) != ("error" in cx):
                bad = {"python": py.get("error", "accepted"), "fx": cx.get("error", "accepted")}
            else:
                P["refused"] += 1
                P["refused_cases"].append(case["name"])
                pm, cm = py["error"].split(": ", 1)[-1], cx["error"].split(": ", 1)[-1]
                if pm != cm:   # the 400's text (informational: the status is what callers branch on)
                    P["message_differs"].append({"case": case["name"], "python": pm[:160], "fx": cm[:160]})
        else:
            P["accepted"] += 1
            P["tokens"] += len(py["ids"])
            P["plain_spans"] += sum(1 for t in ("<think>", "</think>", "<|im_end|>", "<tool_call>", "<|vision_start|>") if t in json.dumps(case["req"], ensure_ascii=False))
            for k in ("ids", "head", "max_new", "thinking", "force"):
                if py[k] != cx[k]:
                    first = next((i for i, (x, y) in enumerate(zip(py["ids"], cx["ids"])) if x != y), min(len(py["ids"]), len(cx["ids"]))) if k == "ids" else None
                    bad = {"field": k, "python": py[k] if k != "ids" else py["ids"][max(0, (first or 0) - 5):(first or 0) + 8],
                           "fx": cx[k] if k != "ids" else cx["ids"][max(0, (first or 0) - 5):(first or 0) + 8], "at": first,
                           "lens": [len(py["ids"]), len(cx["ids"])]}
                    if k == "ids":
                        bad["python_text"] = tok.decode(py["ids"][max(0, first - 20):first + 20])
                        bad["fx_text"] = tok.decode(cx["ids"][max(0, first - 20):first + 20])
                    break
        if bad:
            bad["case"] = case["name"]
            P["mismatches"].append(bad)
            if a.verbose:
                print("MISMATCH", json.dumps(bad, ensure_ascii=False)[:600])
    print(f"prompts: {P['cases']} requests ({P['accepted']} accepted, {P['refused']} refused by both), {P['tokens']:,} tokens, "
          f"{len(P['mismatches'])} mismatches", flush=True)

    # 2. the parser: token by token (the tokenizer's pieces) and in random pieces
    R = summary["parser"]
    tools = [t["function"] for t in ROLE_TOOLS]
    for text in parser_outputs():
        ids = tok.encode(text, parse_special=True)
        dec = SV.Detokenizer(tok)
        by_token = [dec.push(t) for t in ids]
        splits = [by_token, [text]]
        for _ in range(3):
            cut = sorted(rng.sample(range(1, len(text)), min(len(text) - 1, rng.randrange(1, 12)))) if len(text) > 2 else []
            splits.append([text[i:j] for i, j in zip([0] + cut, cut + [len(text)])])
        for thinking in (True, False):
            for with_tools in (True, False):
                for repair in (False, True):
                    for deltas in splits:
                        for finish in ("stop", "length"):
                            p = FE.OutputParser(thinking=thinking, tools=tools if with_tools else None, stream_tools=True, repair_names=repair)
                            pev, perr = [], None
                            try:
                                for d in deltas:
                                    pev += p.feed(d)
                                pev += p.finish(finish)
                            except ValueError as e:
                                perr = str(e)
                            pj = [{"kind": e.kind, "text": e.text or "", **({"name": e.call.name, "arguments": e.call.arguments} if e.call else {})} for e in pev]
                            cx = fx.ask({"op": "parse", "thinking": thinking, "tools": tools if with_tools else None, "deltas": deltas,
                                         "finish": finish, "repair_names": repair})
                            R["cases"] += 1
                            cj = cx.get("events", [])
                            if pj != cj or (perr is None) != ("error" not in cx):
                                R["mismatches"].append({"text": text[:80], "thinking": thinking, "tools": with_tools, "repair": repair, "finish": finish,
                                                        "deltas": len(deltas), "python": pj[:12], "fx": cj[:12], "python_error": perr, "fx_error": cx.get("error")})
    # the recorded reasoning outputs of #804/#1058 (serve/fixtures/rcall_specimens.json), with their own tools and finish
    spec = json.loads((FLASHNEXT / "serve" / "fixtures" / "rcall_specimens.json").read_text())["specimens"]
    for sp in spec:
        stools = [{"name": n, "parameters": {"type": "object"}} for n in sp.get("tools") or []]
        ids = tok.encode(sp["input"], parse_special=True)
        dec = SV.Detokenizer(tok)
        deltas = [dec.push(t) for t in ids]
        for repair in (False, True):
            p = FE.OutputParser(thinking=True, tools=stools or None, stream_tools=True, repair_names=repair)
            pev, perr = [], None
            try:
                for d in deltas:
                    pev += p.feed(d)
                pev += p.finish(sp.get("finish", "stop"))
            except ValueError as e:
                perr = str(e)
            pj = [{"kind": e.kind, "text": e.text or "", **({"name": e.call.name, "arguments": e.call.arguments} if e.call else {})} for e in pev]
            cx = fx.ask({"op": "parse", "thinking": True, "tools": stools or None, "deltas": deltas, "finish": sp.get("finish", "stop"), "repair_names": repair})
            R["cases"] += 1
            if pj != cx.get("events", []) or (perr is None) != ("error" not in cx):
                R["mismatches"].append({"specimen": sp["name"], "repair": repair, "python": pj[:12], "fx": cx.get("events", [])[:12]})
    print(f"parser: {R['cases']} feeds, {len(R['mismatches'])} mismatches", flush=True)

    # 3. Service._run on a scripted engine
    U = summary["run"]
    base = [{"role": "user", "content": "Go."}]
    for name, extra, thinking, passes, config in run_cases():
        req = {"model": "flash-next", "messages": base, **extra}
        prep = py_prepare(svc, tok, req, None, CTX)
        if "error" in prep:
            U["mismatches"].append({"case": name, "python": prep["error"]})
            continue
        pass_ids = [tok.encode(t, parse_special=True) for t in passes]
        eng = ScriptEngine(pass_ids)
        svc.engine = eng
        svc.stop_token_text = config.get("stop_token_text", True)
        svc.repeat_stop_tokens = config.get("repeat_stop_tokens", SV.REPEAT_STOP_TOKENS)
        svc.tool_name_repair = config.get("tool_name_repair", False)
        _, tl, kw = FE.openai_to_messages(copy.deepcopy(req))
        r2 = copy.deepcopy(req)
        SV.check_request_sampling(r2)
        cancel = threading.Event()
        try:
            res = SV.openai_collect(SV.openai_chunks(svc, r2, prep["ids"], prep["thinking"], tl, prep["max_new"], cancel, force=prep["force"] or None))
            py = strip_ids(res)
        except ValueError as e:
            py = {"error": f"ValueError: {e}"}
        cx = fx.ask({"op": "tokens", "req": r2, "prompt": prep["ids"], "thinking": prep["thinking"], "tools": tl, "max_new": prep["max_new"],
                     "force": prep["force"], "passes": pass_ids, "config": config,
                     "model_name": "flash-next", "aliases": ["flash-next-alias", "FlashNext"]})
        heads = cx.pop("gens", [])
        cxr = strip_ids(cx)
        U["cases"] += 1
        bad = []
        if ("error" in py) != ("error" in cxr):
            bad.append("error")
        if "error" not in py and "error" not in cxr:
            for c in (py, cxr):   # tool call ids are random
                for tc in c["choices"][0]["message"].get("tool_calls") or []:
                    tc.pop("id", None)
            if py != cxr:
                bad.append("answer")
        if eng.heads != heads:
            bad.append("heads")
        if bad:
            U["mismatches"].append({"case": name, "what": bad, "python": py, "fx": cxr, "python_heads": eng.heads, "fx_heads": heads})
        svc.engine = engine
    print(f"run: {U['cases']} scripted requests, {len(U['mismatches'])} mismatches", flush=True)
    fx.close()
    summary["seconds"] = round(time.time() - t0, 1)
    summary["gate"] = "One Engine P2, the dev machine CPU; impulse/tests/fx_front_parity.py; serve/server.py's own functions vs impulse-fx --fx-front-test"
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1, ensure_ascii=False)
    ok = not (P["mismatches"] or R["mismatches"] or U["mismatches"])
    if not ok and not a.out:
        for k in ("prompts", "parser", "run"):
            for m in summary[k]["mismatches"][:5]:
                print(k, json.dumps(m, ensure_ascii=False)[:800])
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
