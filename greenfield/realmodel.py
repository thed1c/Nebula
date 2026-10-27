"""Driven by realmodel.sh inside the container: Nebula's own API, a real llama.cpp build and a tiny real GGUF.
    python realmodel.py pull      # (network on) resolve + pull the GGUF through Nebula's HF code
    python realmodel.py deploy    # (network off) deploy it with llama.cpp, verify, chat, drain
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8090"
URL = "https://huggingface.co/ggml-org/models/blob/main/tinyllamas/stories260K.gguf"


def call(path, body=None, timeout=120):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def step(name, ok, detail=""):
    print(("ok    " if ok else "FAIL  ") + name + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        sys.exit(1)


def pull():
    st, r = call("/api/yard/hf/resolve", {"url": URL})
    step("resolve the link", st == 200 and r.get("chosen") == "tinyllamas/stories260K.gguf", f"{st} {str(r)[:200]}")
    f = next(x for x in r["files"] if x["path"] == r["chosen"])
    st, p = call("/api/yard/hf/pull", {"repo": r["repo"], "revision": r["revision"], "files": [f["path"]], "meta": [f]})
    step("start the pull", st == 200, str(p)[:200])
    for _ in range(300):
        v = call("/api/yard/hf/pulls")[1]["pulls"][0]
        if v["state"] in ("done", "failed"):
            break
        time.sleep(1)
    step("downloaded and checksum-checked", v["state"] == "done", json.dumps(v)[:300])
    open("/tmp/model-path", "w").write(v["files"][0]["target"])


def deploy():
    model = open("/tmp/model-path").read().strip()
    binary = open("/tmp/llama-server-path").read().strip()
    engine = {"name": "stories", "kind": "llama.cpp", "binary": binary, "model": model, "context": 256, "slots": 1,
              "options": {"temp": 0.0, "threads": 2, "seed": 1}}
    st, pv = call("/api/yard/preview", {"engine": engine})
    step("the command line", st == 200, " ".join(pv.get("command") or [str(pv)]))
    st, out = call("/api/yard/deploy", {"engine": engine, "wait": True}, timeout=300)
    step("deploy (add, make a model, start)", st == 200 and out.get("engine", {}).get("state") == "ready", json.dumps(out)[:600])
    st, v = call("/api/yard/engines/stories/verify", {})
    step("verify: it answers and runs the pulled file", v.get("ok") is True, json.dumps(v))
    st, c = call("/v1/chat/completions", {"model": "stories", "max_tokens": 16,
                                          "messages": [{"role": "user", "content": "Once upon a time"}]}, timeout=300)
    text = ((c.get("choices") or [{}])[0].get("message") or {}).get("content")
    step("a real answer through Nebula", st == 200 and bool(text), repr(text)[:120])
    st, d = call("/api/yard/engines/stories/drain", {"timeout_s": 30})
    for _ in range(60):
        if call("/api/yard/engines/stories")[1]["state"] == "stopped":
            break
        time.sleep(0.5)
    step("drain → stopped", call("/api/yard/engines/stories")[1]["state"] == "stopped")


if __name__ == "__main__":
    {"pull": pull, "deploy": deploy}[sys.argv[1]]()
