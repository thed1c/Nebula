#!/usr/bin/env python3
"""Flint quality and speed benchmark: one GGUF through Impulse, on one card, repeatable.

    bench.py run --gguf FILE --label NAME --card GPU-UUID [--ref DIR] [--humaneval 40|164] [--contexts 8192,32768,65536]
    bench.py ref --bf16 FILE --oracle ROUTE_ORACLE --card GPU-UUID --out DIR

`run` measures, in order:
1. **Code quality, HumanEval** (openai/human-eval, the file pinned below by commit and sha256, fetched at run time):
   `--humaneval 40` takes 40 problems evenly spaced over the 164 (the same 40 every time), 164 takes them all; greedy,
   through impulse-server's OpenAI API, max 6144 tokens; each answer's last python block run with the task's tests in its
   own process (15 s). pass@1.
2. **Agreement with the BF16 weights** (`--ref`, made once by `ref`): six HumanEval prompts (fixed: tasks 0, 32, 64,
   96, 128, 160) answered greedily by the BF16 file in llama.cpp (route-oracle, b6b003d + the Flint patch, every expert in
   RAM, CUDA fusion off), 256 tokens each; this file teacher-forced on the same tokens by impulse-qwen35: top-1 agreement
   and mean KL(BF16 || this), over the 6 x 257 rows.
3. **Speed and memory** per context: impulse-server started with `-c N` (Flint's defaults: every expert in RAM, the expert
   cache on the card), a prompt of about N - 1.5K tokens read, 256 tokens written, a follow-up of 400 written; the
   process's peak card memory (nvidia-smi) and its RAM: the expert arena (RssShmem) + its own (RssAnon); the model file's
   page cache (RssFile) is reclaimable and reported apart.

Each run appends one JSON line to `--json` and rewrites the Markdown table `--md` from every line there. No prompt or
generated text is written, numbers only (CONTRIBUTING.md). Needs: impulse-server and impulse-qwen35 (`--bin DIR`),
nvidia-smi, Python 3.10+.
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, os, re, subprocess, sys, tempfile, threading, time, urllib.request

HE_URL = "https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz"
HE_SHA = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
KL_TASKS = (0, 32, 64, 96, 128, 160)
ASK = "Complete this Python function. Reply with the whole function (and any imports it needs) in one ```python code block.\n\n"


def humaneval(cache: str) -> list[dict]:
    p = os.path.join(cache, "HumanEval.jsonl.gz")
    if not os.path.isfile(p):
        os.makedirs(cache, exist_ok=True)
        urllib.request.urlretrieve(HE_URL, p + ".part")
        os.replace(p + ".part", p)
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != HE_SHA:
        sys.exit(f"{p}: not the pinned HumanEval file")
    return [json.loads(l) for l in gzip.open(p, "rt")]


class Server:
    """impulse-server on one card; peak card memory polled; RSS read at the end."""
    def __init__(self, binp, gguf, card, ctx, port, threads, extra=()):
        self.port, self.peak = port, 0
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=card)
        self.p = subprocess.Popen([os.path.join(binp, "impulse-server"), "-m", gguf, "--host", "127.0.0.1", "--port", str(port),
                                   "-c", str(ctx), "-t", str(threads), *extra], stdout=self.log, stderr=self.log, env=env)
        threading.Thread(target=self._poll, daemon=True).start()
        for _ in range(400):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); return
            except Exception:
                if self.p.poll() is not None:
                    raise RuntimeError("impulse-server exited: see " + self.log.name)
                time.sleep(2)
        raise RuntimeError("impulse-server did not come up")

    def _poll(self):
        while self.p.poll() is None:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout
            self.peak = max(self.peak, sum(int(l.split(",")[1]) for l in out.splitlines() if l.split(",")[0].strip() == str(self.p.pid)))
            time.sleep(0.5)

    def post(self, path, body, t=7200):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(r, timeout=t))

    def mem(self) -> dict:
        st = {l.split(":")[0]: int(l.split()[1]) for l in open(f"/proc/{self.p.pid}/status") if ":" in l and l.split()[1:2] and l.split()[1].isdigit()}
        slots = re.search(r"(\d+) expert slots per layer", open(self.log.name).read())
        return {"vram_gb": round(self.peak / 1024, 1), "ram_gb": round((st.get("RssShmem", 0) + st.get("RssAnon", 0)) / 2**20, 1),
                "page_cache_gb": round(st.get("RssFile", 0) / 2**20, 1), "slots": int(slots.group(1)) if slots else None}

    def stop(self):
        self.p.terminate(); self.p.wait(60); os.unlink(self.log.name)


def run_humaneval(srv: Server, probs: list[dict], n: int) -> dict:
    pick = probs if n >= len(probs) else [probs[round(i * (len(probs) - 1) / (n - 1))] for i in range(n)]
    passed, toks, t0, cut = 0, 0, time.time(), 0
    for p in pick:
        d = srv.post("/v1/chat/completions", {"messages": [{"role": "user", "content": ASK + p["prompt"]}], "max_tokens": 6144, "temperature": 0})
        text = d["choices"][0]["message"].get("content") or ""
        cut += d["choices"][0]["finish_reason"] == "length"
        toks += d.get("usage", {}).get("completion_tokens", 0)
        blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
        code = blocks[-1] if blocks else text
        if f"def {p['entry_point']}" not in code:
            code = p["prompt"] + code
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "t.py")
            open(f, "w").write(code + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n")
            try:
                passed += subprocess.run([sys.executable, "-I", f], cwd=td, capture_output=True, timeout=15).returncode == 0
            except subprocess.TimeoutExpired:
                pass
    return {"he_n": len(pick), "he_pass": passed, "he_cut": cut, "he_tokens": toks, "he_min": round((time.time() - t0) / 60)}


def run_speed(a, ctx: int) -> dict:
    srv = Server(a.bin, a.gguf, a.card, ctx, a.port, a.threads)
    try:
        unit = "Ledger %d: the harbour office logged the ship Marigold, forty barrels of salt, wind from the north-west, fog by noon. "
        per = len(srv.post("/tokenize", {"content": "".join(unit % i for i in range(200))})["tokens"]) / 200
        msgs = [{"role": "user", "content": "".join(unit % i for i in range(int((ctx - 1500) / per))) + "\nWhich ship did the office log in ledger 7? One word."}]
        d = srv.post("/v1/chat/completions", {"messages": msgs, "max_tokens": 256, "temperature": 0})
        t = d["timings"]
        msgs += [{"role": "assistant", "content": d["choices"][0]["message"].get("content") or ""}, {"role": "user", "content": "Now write a short poem about the harbour."}]
        w2 = srv.post("/v1/chat/completions", {"messages": msgs, "max_tokens": 400, "temperature": 0})["timings"]["predicted_per_second"]
        return {"ctx": ctx, "read_tok": t["prompt_n"], "read_tps": round(t["prompt_per_second"]), "write_tps": round(t["predicted_per_second"], 1),
                "write2_tps": round(w2, 1), "right": "Marigold" in (d["choices"][0]["message"].get("content") or ""), **srv.mem()}
    finally:
        srv.stop()


def run_kl(a) -> dict:
    agree = rows = 0; kls = []
    for k in KL_TASKS:
        base = os.path.join(a.ref, f"he{k}")
        out = tempfile.mktemp()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.card)
        subprocess.run([os.path.join(a.bin, "impulse-qwen35"), "-m", a.gguf, "--tokens", base + ".tokens", "--oracle", base + ".logits",
                        "--mode", "force", "--n-cpu-moe", "49", "-c", "8192", "-ctk", "f16", "-ctv", "f16", "-t", str(a.threads), "--out", out],
                       env=env, capture_output=True, timeout=3600)
        d = json.load(open(out + ".json")); os.unlink(out + ".json")
        agree += d["argmax_same"]; rows += d["steps_compared"]; kls.append(d["kl_mean"])
    return {"kl_rows": rows, "kl_top1": round(100 * agree / rows, 1), "kl_mean": round(sum(kls) / len(kls), 4)}


def make_ref(a):
    probs = humaneval(a.cache)
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.card, GGML_CUDA_DISABLE_FUSION="1")
    for k in KL_TASKS:
        req = os.path.join(a.out, f"he{k}.req.json")
        json.dump({"messages": [{"role": "user", "content": ASK + probs[k]["prompt"]}]}, open(req, "w"))
        subprocess.run([a.oracle, "-m", a.bf16, "-ngl", "99", "--n-cpu-moe", "49", "-fa", "on", "-c", "8192", "-ctk", "f16", "-ctv", "f16",
                        "-b", "2048", "-ub", "512", "-t", str(a.threads), "--temp", "0"],
                       env={**env, "ROUTE_REQ": req, "ROUTE_OUT": os.path.join(a.out, f"he{k}"), "ROUTE_NGEN": "256", "ROUTE_LOGITS": "1"},
                       check=True, capture_output=True)
        os.unlink(req)
    print("reference written to", a.out)


def table(rows: list[dict]) -> str:
    ctxs = sorted({s["ctx"] for r in rows for s in r.get("speed", [])})
    h = ["Option", "File GB", "HumanEval", "Top-1 vs BF16", "KL vs BF16"] + [f"{c // 1024}K read / write tok/s, card GB, RAM GB" for c in ctxs]
    out = ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
    for r in rows:
        sp = {s["ctx"]: s for s in r.get("speed", [])}
        cells = [r["label"], f"{r['size_gb']:.1f}",
                 f"{r['he_pass']}/{r['he_n']}" + (f" ({r['he_cut']} cut)" if r.get("he_cut") else "") if "he_pass" in r else "-",
                 f"{r['kl_top1']} %" if "kl_top1" in r else "-", f"{r['kl_mean']}" if "kl_mean" in r else "-"]
        for c in ctxs:
            s = sp.get(c)
            cells.append(f"{s['read_tps']} / {s['write_tps']}-{s['write2_tps']}, {s['vram_gb']}, {s['ram_gb']}" + ("" if s["right"] else " (wrong answer)") if s else "-")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gguf", required=True); r.add_argument("--label", required=True); r.add_argument("--card", required=True)
    r.add_argument("--bin", default="build/bin"); r.add_argument("--ref"); r.add_argument("--humaneval", type=int, default=40)
    r.add_argument("--contexts", default="8192,32768,65536"); r.add_argument("--threads", type=int, default=8)
    r.add_argument("--port", type=int, default=18960); r.add_argument("--cache", default=os.path.expanduser("~/.cache/flint-bench"))
    r.add_argument("--json", default="flint-bench.jsonl"); r.add_argument("--md", default="flint-bench.md")
    r.add_argument("--skip", default="", help="comma list of humaneval,kl,speed to leave out")
    r.add_argument("--known", default="{}", help="JSON merged into the row: numbers measured before with the same settings "
                                                 "(he_n, he_pass, he_cut, kl_rows, kl_top1, kl_mean), for a part left out")
    f = sub.add_parser("ref")
    f.add_argument("--bf16", required=True); f.add_argument("--oracle", required=True); f.add_argument("--card", required=True)
    f.add_argument("--out", required=True); f.add_argument("--threads", type=int, default=12)
    f.add_argument("--cache", default=os.path.expanduser("~/.cache/flint-bench"))
    a = ap.parse_args()
    if a.cmd == "ref":
        return make_ref(a)
    skip = set(filter(None, a.skip.split(",")))
    row = {"label": a.label, "file": os.path.basename(a.gguf), "size_gb": os.path.getsize(a.gguf) / 1e9, "card": a.card, "when": time.strftime("%Y-%m-%d %H:%M")}
    row.update(json.loads(a.known))
    if "humaneval" not in skip:
        srv = Server(a.bin, a.gguf, a.card, 16384, a.port, a.threads)
        try:
            row.update(run_humaneval(srv, humaneval(a.cache), a.humaneval))
        finally:
            srv.stop()
    if a.ref and "kl" not in skip:
        row.update(run_kl(a))
    if "speed" not in skip:
        row["speed"] = [run_speed(a, int(c)) for c in a.contexts.split(",")]
    with open(a.json, "a") as f:
        f.write(json.dumps(row) + "\n")
    rows = [json.loads(l) for l in open(a.json)]
    open(a.md, "w").write(table(rows))
    print(table([row]))


if __name__ == "__main__":
    main()
