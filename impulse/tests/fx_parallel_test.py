#!/usr/bin/env python3
"""impulse-fx's "parallel": N (the engine's batch slots, src/fx/fx_slots.hpp) end to end without a GPU or a model: the
made-up pack and the mock engine of fx_server_test.py, the mock with IMPULSE_FX_MOCK_BATCH (its BGEN / BT / BDONE).
Checks: "parallel": 2 adds --batch 2 to the engine's arguments (none for 1, none when the args name --batch); /props
total_slots and /slots report the engine's slots; the startup line on what a slot costs the expert cache; requests at
once all answer in full (streamed and not), the second one admitted into a slot (BGEN) while the first was promoted
from the solo path (GEN, STOP, BGEN); a client that stops reading frees its slot (BSTOP) and the next requests still
answer.  Standard library only.

    python fx_parallel_test.py BUILD/bin      (impulse-fx or impulse-server, and libimpulse_fx_mock.so there)
"""
import http.client
import json
import pathlib
import shutil
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fx_server_test as T   # noqa: E402  (its pack, tokenizer, mock server and checks)


def chat(port, text, stream=False, max_tokens=64, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = {"model": "bb", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}
    c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    if not stream:
        d = json.loads(r.read().decode())
        c.close()
        return r.status, d["choices"][0]["message"]["content"] if r.status == 200 else d
    text_out = ""
    for raw in r:
        line = raw.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            j = json.loads(line[6:])
            if "choices" in j:
                text_out += j["choices"][0]["delta"].get("content") or ""
    c.close()
    return r.status, text_out


def up(S):
    S.wait_up()
    for _ in range(200):
        st, _ = S.req("GET", "/health")
        if st == 200:
            return
        time.sleep(0.05)
    raise SystemExit("not ready")


def main() -> int:
    bindir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "build/bin").resolve()
    td = pathlib.Path(tempfile.mkdtemp(prefix="fx-parallel-test-"))
    try:
        root = td / "tiny-pack"
        (root / "pack").mkdir(parents=True)
        (root / "pack" / "index.txt").write_text("test\n")
        sp = T.tiny_gguf(root / "tiny-dense.gguf")
        end = sp["<|im_end|>"]
        reply_text = "Hello there, the world is wide and the sea is deep and the night is long."

        # "parallel": 1 (and absent): no --batch, one slot, as before
        mock = td / "m1"
        mock.mkdir()
        cfg = {"args": ["--spec", "4"], "parallel": 1}
        S = T.Server(bindir, root, mock, ["--fx-config-json", json.dumps(cfg)])
        try:
            up(S)
            args = (mock / "requests.log").read_text().split("\n")[0]
            T.check("parallel 1: no --batch in the engine's arguments", "--batch" not in args, args)
            st, d = S.req("GET", "/props")
            T.check("parallel 1: total_slots 1", json.loads(d)["total_slots"] == 1, d[:200])
        finally:
            S.close()

        # "parallel": 2 -> --batch 2; the mock runs 2 slots
        mock = td / "m2"
        mock.mkdir()
        cfg = {"args": ["--spec", "4"], "parallel": 2}
        S = T.Server(bindir, root, mock, ["--fx-config-json", json.dumps(cfg)],
                     {"IMPULSE_FX_MOCK_BATCH": "2", "IMPULSE_FX_MOCK_EOS": str(end)})
        try:
            up(S)
            args = (mock / "requests.log").read_text().split("\n")[0]
            T.check("parallel 2: --batch 2 in the engine's arguments", " --batch 2" in args, args)
            st, d = S.req("GET", "/props")
            T.check("parallel 2: /props total_slots 2", json.loads(d)["total_slots"] == 2, d[:200])
            st, d = S.req("GET", "/slots")
            sl = json.loads(d)
            T.check("parallel 2: /slots lists 2 slots, idle", len(sl) == 2 and all(not s["is_processing"] for s in sl), d)
            log = pathlib.Path(S.log.name).read_text()
            T.check("startup: what a slot costs the expert cache", "one slot's sessions take 512 MiB = about 73 expert-cache slots" in log,
                    [x for x in log.split("\n") if "batch slots" in x])
            st, d = S.req("POST", "/tokenize", {"content": reply_text})
            reply = json.loads(d)["tokens"] + [end]
            (mock / "reply.txt").write_text(",".join(map(str, reply)))
            # four requests at once, two streamed: every answer in full
            res = [None] * 4

            def go(i):
                res[i] = chat(S.port, f"question {i}", stream=i % 2 == 1, max_tokens=200)
            th = [threading.Thread(target=go, args=(i,)) for i in range(4)]
            for t in th:
                t.start()
                time.sleep(0.02)
            for t in th:
                t.join()
            T.check("4 requests at once on 2 slots: all 200 with the whole reply",
                    all(r and r[0] == 200 and r[1] == reply_text for r in res), res)
            reqs = (mock / "requests.log").read_text()
            T.check("... the slots were used (BGEN) and a solo request promoted (STOP)", reqs.count("BGEN ") >= 3 and "STOP" in reqs,
                    (reqs.count("BGEN "), reqs.count("STOP")))
            st, d = S.req("GET", "/slots")
            T.check("... every slot idle again", all(not s["is_processing"] for s in json.loads(d)), d)
            # a streamed client that stops reading after the first chunk while another runs: BSTOP, then more answers
            c = http.client.HTTPConnection("127.0.0.1", S.port, timeout=30)
            other = [None]
            t2 = threading.Thread(target=lambda: other.__setitem__(0, chat(S.port, "the other", max_tokens=200)))
            t2.start()
            time.sleep(0.05)
            body = {"model": "bb", "messages": [{"role": "user", "content": "leave early"}], "max_tokens": 200, "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
            r = c.getresponse()
            r.readline()
            c.close()
            t2.join()
            T.check("a client that leaves: the other request still answers in full", other[0] and other[0][1] == reply_text, other[0])
            ok = True
            for _ in range(100):
                st, d = S.req("GET", "/slots")
                if all(not s["is_processing"] for s in json.loads(d)):
                    break
                time.sleep(0.1)
            else:
                ok = False
            T.check("... its slot freed (BSTOP or the stop at its end)", ok, d)
            st, txt = chat(S.port, "after", max_tokens=200)
            T.check("... and the next request answers", st == 200 and txt == reply_text, (st, txt))
        finally:
            S.close()

        # the args name --batch themselves: no second one
        mock = td / "m3"
        mock.mkdir()
        cfg = {"args": ["--batch", "3"], "parallel": 2}
        S = T.Server(bindir, root, mock, ["--fx-config-json", json.dumps(cfg)], {"IMPULSE_FX_MOCK_BATCH": "3"})
        try:
            up(S)
            args = (mock / "requests.log").read_text().split("\n")[0]
            T.check("the args' own --batch wins (no second one)", args.count("--batch") == 1 and "--batch 3" in args, args)
            st, d = S.req("GET", "/props")
            T.check("... total_slots = the engine's 3", json.loads(d)["total_slots"] == 3, d[:120])
        finally:
            S.close()
    finally:
        shutil.rmtree(td, ignore_errors=True)
    if T.FAILS:
        print(f"FAILED: {len(T.FAILS)}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
