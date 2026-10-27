#!/usr/bin/env python3
"""One Engine P1: impulse-server's fx engine path end to end without a GPU or a model.  A made-up Flash-Next pack folder
(pack/index.txt + a tiny *dense*.gguf: a byte-level BPE tokenizer with Qwen's special tokens and a ChatML template that
knows enable_thinking) and tests/fx_mock_engine.cpp (libimpulse_fx_mock: the engine's serve protocol, scripted) in place
of libimpulse_fx.  Checks: the pack folder selects the fx engine; /health is 503 until READY; the GEN line (max_new, the
sampling keys server.py would send, the prompt's ids = the chat template's text); non-streamed and streamed chat answers
(text, finish_reason stop / length, usage); max_tokens; stop strings (STOP sent, text cut); enable_thinking; /completion;
the engine's arguments (--fx-args / --fx-config / --fx-config-json); the run config's "log" (server.py's engine log).  P2 (server parity): /health, /v1/models, /props and /slots as
server.py answers them; the reasoning split; tool calls (streamed pieces and whole); an EOS literal's second pass;
response_format; a client that leaves (STOP); pictures through a stand-in strata-vision (GENI, the image pads, a WebP
converted by Pillow as server.py's Vision.normalize does); a silent engine (#481: the request ends, the server exits).
Standard library only (Pillow for the WebP case, when the Python running this has it).

    python fx_server_test.py BUILD/bin        (impulse-server or impulse-fx, and libimpulse_fx_mock.so there)
"""
import http.client
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import gguf_meta_only as G   # noqa: E402  (its stdlib GGUF writer)

TEMPLATE = (
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{% if m.content is string %}{{ m.content }}{% else %}{% for it in m.content %}"
    "{% if it.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ it.text }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}"
    "{% endif %}")


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def tiny_gguf(path: pathlib.Path):
    """A gpt2-BPE tokenizer: the 256 byte tokens, a few merges, then Qwen's special tokens (CONTROL 3, USER_DEFINED 4)."""
    b2u = bytes_to_unicode()
    tokens = [b2u[b] for b in range(256)]
    merges = [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("H", "e"), ("l", "l"), ("He", "ll"), ("Hell", "o")]
    for a, b in merges:
        tokens.append(a + b)
    types = [1] * len(tokens)
    specials = [("<|endoftext|>", 3), ("<|im_start|>", 3), ("<|im_end|>", 3), ("<think>", 4), ("</think>", 4),
                ("<|vision_start|>", 3), ("<|image_pad|>", 3), ("<|vision_end|>", 3)]
    ids = {}
    for t, ty in specials:
        ids[t] = len(tokens)
        tokens.append(t)
        types.append(ty)
    S = lambda s: s.encode()   # noqa: E731
    kv = [
        (S("general.architecture"), G.STR, S("gpt2")),
        (S("general.name"), G.STR, S("fx-test-tiny")),
        (S("gpt2.context_length"), G.U32, 4096),
        (S("gpt2.embedding_length"), G.U32, 64),
        (S("gpt2.feed_forward_length"), G.U32, 256),
        (S("gpt2.block_count"), G.U32, 1),
        (S("gpt2.attention.head_count"), G.U32, 4),
        (S("gpt2.attention.layer_norm_epsilon"), G.F32, 1e-5),
        (S("tokenizer.ggml.model"), G.STR, S("gpt2")),
        (S("tokenizer.ggml.pre"), G.STR, S("qwen35")),
        (S("tokenizer.ggml.tokens"), G.ARR, (G.STR, [S(t) for t in tokens])),
        (S("tokenizer.ggml.token_type"), G.ARR, (G.I32, types)),
        (S("tokenizer.ggml.merges"), G.ARR, (G.STR, [S(a + " " + b) for a, b in merges])),
        (S("tokenizer.ggml.eos_token_id"), G.U32, ids["<|im_end|>"]),
        (S("tokenizer.ggml.bos_token_id"), G.U32, ids["<|endoftext|>"]),
        (S("tokenizer.ggml.padding_token_id"), G.U32, ids["<|endoftext|>"]),
        (S("tokenizer.ggml.add_bos_token"), G.BOOL, False),
        (S("tokenizer.chat_template"), G.STR, S(TEMPLATE)),
    ]
    G.write_meta(str(path), 3, kv)
    return ids


class Server:
    def __init__(self, bindir, model, mockdir, extra, env_extra=None):
        self.port = free_port()
        env = dict(os.environ, IMPULSE_FX_MOCK_DIR=str(mockdir), **(env_extra or {}))
        env.pop("CUDA_MODULE_LOADING", None)
        self.log = open(mockdir / f"server-{self.port}.log", "w")
        exe = bindir / "impulse-server" if (bindir / "impulse-server").exists() else bindir / "impulse-fx"
        self.p = subprocess.Popen([str(exe), "-m", str(model), "--port", str(self.port), "--alias", "bb",
                                   "--fx-lib", str(bindir / "libimpulse_fx_mock.so"), "--fx-tokenize-endpoints", *extra],
                                  stdout=self.log, stderr=subprocess.STDOUT, env=env)

    def req(self, method, path, body=None, timeout=30):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        c.request(method, path, body=json.dumps(body) if body is not None else None,
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read().decode("utf-8")
        c.close()
        return r.status, data

    def wait_up(self, deadline=30.0):
        t0 = time.time()
        while time.time() - t0 < deadline:
            if self.p.poll() is not None:
                raise SystemExit(f"impulse-server exited ({self.p.returncode}); see {self.log.name}")
            try:
                return self.req("GET", "/health", timeout=2)
            except OSError:
                time.sleep(0.1)
        raise SystemExit("impulse-server did not listen")

    def close(self):
        self.p.terminate()
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


FAILS = []


def check(what, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + what + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILS.append(what)


def sse(data: str):
    events = []
    for block in data.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                events.append(line[6:])
    return events


def tok(S, text):
    return json.loads(S.req("POST", "/tokenize", {"content": text})[1])["tokens"]


def chat(S, body):
    st, d = S.req("POST", "/v1/chat/completions", body)
    try:
        return st, json.loads(d)
    except ValueError:
        return st, d


def p2_checks(S, mock, sp):
    end = sp["<|im_end|>"]
    st, d = S.req("GET", "/health")
    h = json.loads(d)
    check("P2 /health: server.py's fields", st == 200 and h.get("status") == "ok" and h.get("loaded") is True and h.get("service") == "strata"
          and h.get("max_context") == 4096 and h.get("model") == "bb" and h.get("images") is False, h)
    st, d = S.req("GET", "/v1/models")
    m = json.loads(d)
    check("P2 /v1/models: the model, n_ctx, modalities", m["data"][0]["id"] == "bb" and m["data"][0]["meta"]["n_ctx"] == 4096
          and m["data"][0]["architecture"]["input_modalities"] == ["text"], m)
    st, d = S.req("GET", "/props")
    pr = json.loads(d)
    check("P2 /props: n_ctx, total_slots, modalities, model_alias, model_path, chat_template",
          pr["default_generation_settings"]["n_ctx"] == 4096 and pr["total_slots"] == 1 and pr["modalities"] == {"vision": False}
          and pr["model_alias"] == "bb" and pr["model_path"].endswith("tiny-dense.gguf") and "<|im_start|>" in pr["chat_template"], pr)
    st, d = S.req("GET", "/props?model=other")
    check("P2 /props?model=other: 404", st == 404, st)
    st, d = S.req("GET", "/slots")
    sl = json.loads(d)
    check("P2 /slots: one slot, idle, n_ctx", sl[0]["id"] == 0 and sl[0]["is_processing"] is False and sl[0]["n_ctx"] == 4096, sl)
    st, d = S.req("POST", "/v1/nope", {})
    check("P2 unknown path: 404", st == 404, st)

    # the reasoning split (thinking on: the prompt opens <think>, the reply closes it)
    (mock / "reply.txt").write_text(",".join(map(str, tok(S, "Let me think.</think>\n\nThe answer.") + [end])))
    st, j = chat(S, {"messages": [{"role": "user", "content": "q"}]})
    msg = j["choices"][0]["message"]
    check("P2 reasoning split: reasoning_content and content", msg.get("reasoning_content") == "Let me think." and msg.get("content") == "The answer.", msg)
    st, d = S.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "q"}], "stream": True})
    chunks = [json.loads(e) for e in sse(d) if e != "[DONE]"]
    rc = "".join(c["choices"][0]["delta"].get("reasoning_content") or "" for c in chunks)
    cc = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    check("P2 reasoning split, streamed", rc == "Let me think." and cc == "The answer." and chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}, (rc, cc))
    check("P2 stream: the last chunk carries usage and timings", "usage" in chunks[-1] and "timings" in chunks[-1] and chunks[-1]["choices"][0]["finish_reason"] == "stop", chunks[-1])

    # a tool call: whole (non-streamed) and in pieces (streamed), finish_reason tool_calls
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {
        "q": {"type": "string"}, "n": {"type": "integer"}, "deep": {"type": "boolean"}}}}}]
    call = "ok</think>\n\n<tool_call>\n<function=lookup>\n<parameter=q>\nsky \"blue\"\n</parameter>\n<parameter=n>\n3\n</parameter>\n<parameter=deep>\nTrue\n</parameter>\n</function>\n</tool_call>"
    (mock / "reply.txt").write_text(",".join(map(str, tok(S, call) + [end])))
    st, j = chat(S, {"messages": [{"role": "user", "content": "q"}], "tools": tools})
    tcs = j["choices"][0]["message"].get("tool_calls") or []
    check("P2 tool call: name, arguments (Python bools where the schema says boolean)", len(tcs) == 1 and tcs[0]["function"]["name"] == "lookup"
          and json.loads(tcs[0]["function"]["arguments"]) == {"q": "sky \"blue\"", "n": 3, "deep": True} and tcs[0]["id"].startswith("call_"), tcs)
    check("P2 tool call: finish_reason tool_calls, no content", j["choices"][0]["finish_reason"] == "tool_calls" and j["choices"][0]["message"]["content"] is None, j["choices"][0])
    st, d = S.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "q"}], "tools": tools, "stream": True})
    chunks = [json.loads(e) for e in sse(d) if e != "[DONE]"]
    pieces = [tc for c in chunks for tc in (c["choices"][0]["delta"].get("tool_calls") or [])]
    args = "".join(p["function"].get("arguments") or "" for p in pieces)
    check("P2 tool call, streamed: the name first, then the arguments in pieces", len(pieces) > 2 and pieces[0]["function"]["name"] == "lookup"
          and json.loads(args) == {"q": "sky \"blue\"", "n": 3, "deep": True}, (len(pieces), args))
    # an EOS literal: <|im_end|> written after a space is its text, and the reply goes on in a second GEN
    n_gen = len([x for x in (mock / "requests.log").read_text().split("\n") if x.startswith("GEN ")])
    (mock / "reply.txt").write_text(",".join(map(str, tok(S, "Write it: ") + [end])) + "\n" + ",".join(map(str, tok(S, "> done.") + [end])) + "\n")
    st, j = chat(S, {"messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": {"enable_thinking": False}})
    gens = [x for x in (mock / "requests.log").read_text().split("\n") if x.startswith("GEN ")]
    check("P2 EOS literal: the token's text kept, a second pass", j["choices"][0]["message"]["content"] == "Write it: <|im_end|> done." and len(gens) == n_gen + 2,
          (j["choices"][0]["message"], len(gens) - n_gen))
    # response_format: one JSON object, canonical; a reply that is not one: 502
    (mock / "reply.txt").write_text(",".join(map(str, tok(S, "```json\n{\"a\": 1, \"b\": [1.0, 2]}\n```") + [end])))
    st, j = chat(S, {"messages": [{"role": "user", "content": "q"}], "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False}})
    check("P2 response_format json_object: the canonical JSON", st == 200 and j["choices"][0]["message"]["content"] == '{"a":1,"b":[1.0,2]}', (st, j))
    (mock / "reply.txt").write_text(",".join(map(str, tok(S, "not json") + [end])))
    st, j = chat(S, {"messages": [{"role": "user", "content": "q"}], "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False}})
    check("P2 response_format, not JSON: 502 structured_output_failed", st == 502 and j["error"]["type"] == "structured_output_failed", (st, j))
    # 400s as server.py words them
    st, j = chat(S, {"messages": [{"role": "user", "content": "q"}], "max_tokens": 5000})
    check("P2 context: 400 (prompt + max tokens over the context, never truncated)", st == 400 and "never truncated" in j["error"]["message"], (st, j))
    st, j = chat(S, {"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "bogus"})
    check("P2 unknown effort: 400", st == 400 and j["error"]["type"] == "invalid_request_error", (st, j))
    # a client that leaves mid-stream: STOP reaches the engine
    long_reply = tok(S, "word " * 400)
    (mock / "reply.txt").write_text(",".join(map(str, long_reply)))
    n_stop = (mock / "requests.log").read_text().count("\nSTOP\n")
    c = http.client.HTTPConnection("127.0.0.1", S.port, timeout=30)
    c.request("POST", "/v1/chat/completions", body=json.dumps({"messages": [{"role": "user", "content": "q"}], "stream": True,
                                                                "chat_template_kwargs": {"enable_thinking": False}}), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    r.read(200)
    c.close()
    for _ in range(100):
        if (mock / "requests.log").read_text().count("\nSTOP\n") > n_stop:
            break
        time.sleep(0.1)
    check("P2 a streaming client that leaves: STOP", (mock / "requests.log").read_text().count("\nSTOP\n") > n_stop)
    # ... and a non-streamed one
    n_stop = (mock / "requests.log").read_text().count("\nSTOP\n")
    c = http.client.HTTPConnection("127.0.0.1", S.port, timeout=30)
    c.request("POST", "/v1/chat/completions", body=json.dumps({"messages": [{"role": "user", "content": "q"}],
                                                                "chat_template_kwargs": {"enable_thinking": False}}), headers={"Content-Type": "application/json"})
    time.sleep(0.5)
    c.sock.close()
    for _ in range(100):
        if (mock / "requests.log").read_text().count("\nSTOP\n") > n_stop:
            break
        time.sleep(0.1)
    check("P2 a non-streamed client that leaves: STOP", (mock / "requests.log").read_text().count("\nSTOP\n") > n_stop)
    time.sleep(0.5)
    st, d = S.req("GET", "/slots")
    check("P2 /slots: idle again", json.loads(d)[0]["is_processing"] is False, d)


FAKE_VISION = r'''#!%s
import hashlib, os, sys
log = open(os.environ["FX_FAKE_VISION_LOG"], "a")
print("READY", flush=True)
for line in sys.stdin:
    f = line.split()
    if not f or f[0] == "QUIT":
        break
    data = open(f[1], "rb").read()
    log.write("ENC %%s %%d\n" %% (hashlib.sha256(data).hexdigest(), len(data))); log.flush()
    n = 3 + len(data) %% 5
    open(f[2], "wb").write(b"E" * (16 * n))
    print("OK %%d" %% n, flush=True)
'''


def vision_checks(bindir, root, mock, td):
    import base64
    import hashlib
    fv = td / "strata-vision-fake"
    fv.write_text(FAKE_VISION % sys.executable)
    fv.chmod(0o755)
    vlog = td / "vision.log"
    vlog.write_text("")
    cfg = td / "run-config-vision.json"
    cfg.write_text(json.dumps({"args": ["--pack", str(root / "pack"), "--native", str(root / "tiny-dense.gguf")], "model_name": "bb",
                               "aliases": ["FlashNext"], "vision": {"exe": str(fv), "mmproj": "m.gguf", "model": "d.gguf", "gpu": False, "threads": 2}}))
    (mock / "requests.log").write_text("")
    extra = ["--fx-config", str(cfg)]
    try:
        import PIL  # noqa: F401
        extra += ["--fx-image-python", sys.executable]
        have_pil = True
    except ImportError:
        have_pil = False
    S = Server(bindir, root, mock, extra, {"FX_FAKE_VISION_LOG": str(vlog)})
    try:
        S.wait_up()
        for _ in range(100):
            if S.req("GET", "/health")[0] == 200:
                break
            time.sleep(0.1)
        h = json.loads(S.req("GET", "/health")[1])
        check("vision: /health images true, /props modalities.vision", h.get("images") is True and json.loads(S.req("GET", "/props")[1])["modalities"]["vision"] is True, h)
        m = json.loads(S.req("GET", "/v1/models")[1])
        check("vision: /v1/models lists the alias with alias_of", [x["id"] for x in m["data"]] == ["bb", "FlashNext"] and m["data"][1]["alias_of"] == "bb"
              and m["data"][0]["architecture"]["input_modalities"] == ["text", "image"], m)
        end = json.loads(S.req("POST", "/tokenize", {"content": "<|im_end|>"})[1])["tokens"]
        (mock / "reply.txt").write_text(",".join(map(str, tok(S, "A red square.") + end)))
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDAxMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg==")
        st, j = chat(S, {"model": "FlashNext", "messages": [{"role": "user", "content": [{"type": "text", "text": "what"},
                                                                                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}],
                         "chat_template_kwargs": {"enable_thinking": False}})
        check("vision: the answer, the alias as the model name", st == 200 and j["choices"][0]["message"]["content"] == "A red square." and j["model"] == "FlashNext", (st, j))
        log = (mock / "requests.log").read_text().split("\n")
        geni = [x for x in log if x.startswith("GENI ")]
        n_pad = 3 + len(png) % 5
        check("vision: a GENI line with the embeddings file and img_keys", geni and " img_keys=" in geni[-1] and "GENI-FILE %d" % (16 * n_pad) in log,
              (geni[-1][:200] if geni else None, [x for x in log if x.startswith("GENI-FILE")]))
        enc = vlog.read_text().split("\n")
        check("vision: a PNG goes to the encoder as it is", ("ENC %s %d" % (hashlib.sha256(png).hexdigest(), len(png))) in enc, enc)
        pads = [int(x) for x in geni[-1].split(" ")[-1].split(",")]
        pad = json.loads(S.req("POST", "/tokenize", {"content": "<|image_pad|>"})[1])["tokens"]
        check("vision: one image pad per image token", len(pad) == 1 and pads.count(pad[0]) == n_pad, (pad, pads.count(pad[0]) if pad else None))
        if have_pil:
            import io
            from PIL import Image
            im = Image.new("RGBA", (3, 2), (10, 200, 30, 128))
            b = io.BytesIO()
            im.save(b, format="WEBP")
            webp = b.getvalue()
            want = io.BytesIO()
            ref = Image.open(io.BytesIO(webp)); ref.load()
            from PIL import ImageOps
            ref = ImageOps.exif_transpose(ref).convert("RGBA")
            bg = Image.new("RGB", ref.size, (255, 255, 255)); bg.paste(ref, mask=ref.split()[-1]); bg.save(want, format="PNG")
            st, j = chat(S, {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(webp).decode()}}]}],
                             "chat_template_kwargs": {"enable_thinking": False}})
            enc = vlog.read_text().split("\n")
            check("vision: a WebP converted to the PNG server.py's Pillow makes", st == 200 and any(x.startswith("ENC " + hashlib.sha256(want.getvalue()).hexdigest()) for x in enc), (st, enc[-2:]))
        st, j = chat(S, {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,bm90IGFuIGltYWdl"}}]}]})
        check("vision: bytes that are no picture: 400", st == 400, (st, j))
        st, j = chat(S, {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]})
        check("vision: an http(s) URL: 400 (impulse-fx does not download)", st == 400, (st, j))
    finally:
        S.close()


def silent_checks(bindir, root, mock):
    cfg = mock / "run-config-silent.json"
    cfg.write_text(json.dumps({"args": ["--pack", str(root / "pack"), "--native", str(root / "tiny-dense.gguf")], "engine_silence_s": 2}))
    S = Server(bindir, root, mock, ["--fx-config", str(cfg)], {"IMPULSE_FX_MOCK_HANG": "1"})
    try:
        S.wait_up()
        for _ in range(100):
            if S.req("GET", "/health")[0] == 200:
                break
            time.sleep(0.1)
        t0 = time.time()
        st, d = S.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "q"}], "max_tokens": 5}, timeout=120)
        check("a silent engine: the request ends with 503 after engine_silence_s", st == 503 and "said nothing" in d and time.time() - t0 < 60, (st, d[:200]))
        try:
            S.p.wait(timeout=10)
            exited = True
        except subprocess.TimeoutExpired:
            exited = False
        check("a silent engine: the server exits (its supervisor starts it again)", exited and S.p.returncode != 0, S.p.returncode)
    finally:
        S.close()


def log_checks(bindir, root, mock, td):
    """The run config's "log", as server.py's StrataEngine keeps it: appended (never truncated), its "[strata] <time>
    engine started: ..." line, the engine's stderr there and not on the console; the front's own lines on the console
    and in the log.  The config passed inline (--fx-config-json, what Nebula's flashnext.py writes)."""
    import re
    elog = td / "engine.log"
    elog.write_text("an earlier run\n")
    cfg = {"args": ["--pack", str(root / "pack"), "--native", str(root / "tiny-dense.gguf"), "--spec", "4"],
           "gpu": [0, 1], "layer_split": "16", "log": str(elog)}
    consoles = []
    for _ in range(2):   # twice: the second start appends to the same log
        if (mock / "requests.log").exists():
            (mock / "requests.log").unlink()
        S = Server(bindir, root, mock, ["--fx-config-json", json.dumps(cfg)])
        try:
            S.wait_up()
            for _ in range(100):
                if S.req("GET", "/health")[0] == 200:
                    break
                time.sleep(0.1)
            args = (mock / "requests.log").read_text().split("\n")[0]
        finally:
            S.close()
        consoles.append(pathlib.Path(S.log.name).read_text())
    check("--fx-config-json: the run config inline (args + --layer-split 16)",
          args == f"ARGS strata --serve --pack {root}/pack --native {root}/tiny-dense.gguf --spec 4 --layer-split 16", args)
    text = elog.read_text()
    check("log: appended, never truncated (server.py's \"a\")", text.startswith("an earlier run\n"), text[:80])
    started = re.findall(r"^\[strata\] \d{4}-\d\d-\d\d \d\d:\d\d:\d\d engine started: \S+libimpulse_fx_mock\.so --serve --pack "
                         + re.escape(str(root)) + r"/pack .*--layer-split 16$", text, re.M)
    check("log: server.py's \"[strata] <time> engine started: <engine> --serve <args>\" line, one per start", len(started) == 2, text[-600:])
    check("log: the engine's stderr", text.count("fx mock engine: loaded (its stderr)\n") == 2, text[-600:])
    check("log: the front's own lines too", "impulse fx: engine " in text and "impulse fx: the engine is ready" in text, text[-600:])
    check("console: the front's lines, not the engine's stderr (that is the log's)",
          all("impulse fx: engine " in c and "impulse-server: listening" in c and "fx mock engine: loaded" not in c
              and "engine started:" not in c for c in consoles), consoles[-1][-600:])
    r = subprocess.run([str(bindir / "impulse-fx" if (bindir / "impulse-fx").exists() else bindir / "impulse-server"), "-m", str(root),
                        "--fx-config-json", json.dumps(cfg), "--fx-args", "--spec", "4"], capture_output=True, text=True, timeout=30)
    check("--fx-config-json with --fx-args: refused", r.returncode == 2 and "one of --fx-config" in r.stderr, (r.returncode, r.stderr[-200:]))


def main() -> int:
    bindir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "build/bin").resolve()
    td = pathlib.Path(tempfile.mkdtemp(prefix="fx-server-test-"))
    try:
        root = td / "tiny-pack"
        (root / "pack").mkdir(parents=True)
        (root / "pack" / "index.txt").write_text("test\n")
        sp = tiny_gguf(root / "tiny-dense.gguf")
        mock = td / "mock"
        mock.mkdir()
        S = Server(bindir, root, mock, ["--fx-args", "--spec", "4", "--kv", "int8"], {"IMPULSE_FX_MOCK_DELAY_MS": "1500"})
        try:
            st, _ = S.wait_up()
            check("/health is 503 while the engine loads", st == 503, st)
            for _ in range(100):
                st, _ = S.req("GET", "/health")
                if st == 200:
                    break
                time.sleep(0.1)
            check("/health is 200 after READY", st == 200, st)
            env_seen = (mock / "env.txt").read_text().strip()
            check("the engine sees CUDA_MODULE_LOADING=EAGER (set before anything starts CUDA)", env_seen == "CUDA_MODULE_LOADING=EAGER", env_seen)
            log = (mock / "requests.log").read_text()
            args = log.split("\n")[0]
            check("no \"log\": the engine's stderr stays on stderr", "fx mock engine: loaded (its stderr)" in pathlib.Path(S.log.name).read_text())
            check("the engine's arguments: --serve, --pack, the --fx-args, --native",
                  args == f"ARGS strata --serve --pack {root}/pack --spec 4 --kv int8 --native {root}/tiny-dense.gguf", args)

            st, d = S.req("POST", "/tokenize", {"content": "Hello there, the world! Ünï ✓ 你好"})
            reply = json.loads(d)["tokens"]
            text = "Hello there, the world! Ünï ✓ 你好"
            (mock / "reply.txt").write_text(",".join(map(str, reply + [sp["<|im_end|>"]])))

            def gen_lines():
                return [x for x in (mock / "requests.log").read_text().split("\n") if x.startswith("GEN ")]

            # non-streamed chat, greedy (no sampling keys), max_tokens
            st, d = S.req("POST", "/v1/chat/completions", {"model": "bb", "messages": [{"role": "user", "content": "Hello"}],
                                                          "max_tokens": 100, "chat_template_kwargs": {"enable_thinking": False}})
            j = json.loads(d)
            check("chat: 200", st == 200, d[:300])
            msg = j["choices"][0]["message"]
            check("chat: the reply's text", msg.get("content") == text, msg)
            check("chat: finish_reason stop at <|im_end|>", j["choices"][0]["finish_reason"] == "stop", j["choices"][0])
            check("chat: usage", j["usage"]["completion_tokens"] == len(reply) + 1 and j["usage"]["prompt_tokens"] > 0, j.get("usage"))
            g = gen_lines()[-1].split(" ")
            check("GEN: max_new 100, no sampling keys (greedy)", g[1] == "100" and len(g) == 3, " ".join(g[:4]))
            st, d2 = S.req("POST", "/detokenize", {"tokens": [int(x) for x in g[2].split(",")]})
            prompt = json.loads(d2)["content"]
            want = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            check("GEN: the prompt is the template's text (enable_thinking false)", prompt == want, repr(prompt))

            # thinking on: the template opens <think> (the tiny template is not detected as a thinking template, so on is asked)
            st, d = S.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5,
                                                          "chat_template_kwargs": {"enable_thinking": True}})
            g = gen_lines()[-1].split(" ")
            prompt = json.loads(S.req("POST", "/detokenize", {"tokens": [int(x) for x in g[-1].split(",")]})[1])["content"]
            check("GEN: enable_thinking true (<think> opened)", prompt.endswith("<|im_start|>assistant\n<think>\n"), repr(prompt))
            j = json.loads(d)
            check("max_tokens 5: finish_reason length", j["choices"][0]["finish_reason"] == "length", j["choices"][0])
            check("max_tokens 5: 5 tokens", j["usage"]["completion_tokens"] == 5, j.get("usage"))

            # sampling keys as server.py sends them
            S.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 3, "temperature": 0.7,
                                                  "top_p": 0.9, "top_k": 40, "min_p": 0.05, "presence_penalty": 1.5, "seed": 7})
            g = gen_lines()[-1]
            want_keys = " temperature=0.7 top_p=0.9 top_k=40 min_p=0.05 penalty_present=1.5 penalty_last_n=64 seed=7 "
            check("GEN: the sampling keys", want_keys in g, g[:160])
            S.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 3, "temperature": 0,
                                                  "top_k": 0})
            g = gen_lines()[-1]
            check("GEN: temperature 0 = greedy (no key), top_k 0 = 64", " temperature=" not in g and " top_k=64 " in g, g[:80])

            # streamed
            st, d = S.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100,
                                                          "stream": True, "stream_options": {"include_usage": True},
                                                          "chat_template_kwargs": {"enable_thinking": False}})
            ev = sse(d)
            check("stream: ends with [DONE]", ev and ev[-1] == "[DONE]", ev[-2:])
            chunks = [json.loads(e) for e in ev if e != "[DONE]"]
            content = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c.get("choices"))
            check("stream: the deltas make the text", content == text, repr(content))
            check("stream: more than one content chunk", sum(1 for c in chunks if c.get("choices") and c["choices"][0]["delta"].get("content")) > 3)
            fin = [c["choices"][0].get("finish_reason") for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
            check("stream: finish_reason stop", fin == ["stop"], fin)
            check("stream: a usage chunk", any(c.get("usage") for c in chunks), chunks[-1])

            # a stop string: the text is cut before it and the engine is told to STOP
            n_stop = (mock / "requests.log").read_text().count("\nSTOP\n")
            st, d = S.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "stop": ["world"],
                                                          "chat_template_kwargs": {"enable_thinking": False}})
            j = json.loads(d)
            check("stop string: the text before it", j["choices"][0]["message"]["content"] == "Hello there, the ", j["choices"][0])
            check("stop string: finish_reason stop", j["choices"][0]["finish_reason"] == "stop")
            check("stop string: STOP sent to the engine", (mock / "requests.log").read_text().count("\nSTOP\n") == n_stop + 1)
            st, d = S.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "stop": ["world"],
                                                          "stream": True, "chat_template_kwargs": {"enable_thinking": False}})
            content = "".join((json.loads(e)["choices"][0]["delta"].get("content") or "") for e in sse(d)
                              if e != "[DONE]" and json.loads(e).get("choices"))
            check("stop string, streamed: the text before it", content == "Hello there, the ", repr(content))

            # /completion: text prompt
            st, d = S.req("POST", "/completion", {"prompt": "Hello", "n_predict": 4})
            j = json.loads(d)
            check("/completion: 200 and 4 tokens", st == 200 and j.get("tokens_predicted", 4) == 4 and j.get("content") == "Hello there", d[:300])

            # an image without an encoder: 400, server.py's words
            st, d = S.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}, {"type": "text", "text": "x"}]}]})
            check("an image without the encoder: 400", st == 400 and "without the vision encoder" in d, (st, d[:200]))

            p2_checks(S, mock, sp)
        finally:
            S.close()

        # pictures: a stand-in strata-vision (READY, ENC -> OK n, its .sve file) through the run config's "vision"
        vision_checks(bindir, root, mock, td)
        # #481: an engine that says nothing during a request
        silent_checks(bindir, root, mock)
        # the run config's "log" (server.py's engine log), the config inline
        log_checks(bindir, root, mock, td)

        # --fx-config: Strata's run config (args, gpu + layer_split, env)
        cfg = td / "run-config.json"
        cfg.write_text(json.dumps({"args": ["--pack", str(root / "pack"), "--native", str(root / "tiny-dense.gguf"), "--spec", "4"],
                                   "gpu": [0, 1], "layer_split": "16", "env": {"FX_TEST_ENV": "1"}}))
        (mock / "requests.log").unlink()
        S = Server(bindir, root / "tiny-dense.gguf", mock, ["--fx-config", str(cfg)])
        try:
            S.wait_up()
            for _ in range(100):
                if (mock / "requests.log").exists():
                    break
                time.sleep(0.1)
            args = (mock / "requests.log").read_text().split("\n")[0]
            check("--fx-config (and -m the dense GGUF): args + --layer-split 16",
                  args == f"ARGS strata --serve --pack {root}/pack --native {root}/tiny-dense.gguf --spec 4 --layer-split 16", args)
        finally:
            S.close()
        # --fx-cli: the engine's own command line in this process (no --serve, no server), stdin / stdout its own
        (mock / "requests.log").unlink()
        exe = bindir / "impulse-server" if (bindir / "impulse-server").exists() else bindir / "impulse-fx"
        r = subprocess.run([str(exe), "-m", str(root), "--fx-cli", "--fx-lib", str(bindir / "libimpulse_fx_mock.so"),
                            "--fx-args", "--max-new", "3"], input="QUIT\n", capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, IMPULSE_FX_MOCK_DIR=str(mock)))
        args = (mock / "requests.log").read_text().split("\n")[0]
        check("--fx-cli: the engine's output on stdout, its exit code", "READY 4096 stop" in r.stdout and r.returncode == 0, (r.returncode, r.stdout[-200:]))
        check("--fx-cli: no --serve", args == f"ARGS strata --pack {root}/pack --max-new 3 --native {root}/tiny-dense.gguf", args)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("PASSED" if not FAILS else f"FAILED: {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
