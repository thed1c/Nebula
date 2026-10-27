#!/usr/bin/env python3
"""impulse-server's --api-key against a running binary (any small model, e.g. Halo's LFM2.5-1.2B QAD Q4_0):

    api_key_live.py IMPULSE_SERVER MODEL [--port P] [extra server args...]

Two servers in turn: one with --api-key (and a second key from --api-key-file), one without. With a key: /health open; /v1/*,
/completion and /lora-adapters (GET and POST) refuse a missing or wrong key with llama-server's 401 body, accept the right one
(Bearer or X-Api-Key, either key). Without a key: everything open, as before. Stdlib only; prints one JSON line, exit 1 on a failure."""
import json, os, subprocess, sys, tempfile, time, urllib.error, urllib.request


def req(base, path, body=None, key=None, header="Authorization", method=None):
    h = {"Content-Type": "application/json"}
    if key is not None:
        h[header] = ("Bearer " + key) if header == "Authorization" else key
    r = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=300) as f:
            return f.status, f.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def serve(binary, model, port, extra):
    p = subprocess.Popen([binary, "-m", model, "--port", str(port), "-c", "4096", *extra], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            if req(base, "/health")[0] == 200:
                return p, base
        except Exception:
            pass
        time.sleep(0.5)
    p.kill()
    raise SystemExit("server did not come up")


def main():
    binary, model = sys.argv[1], sys.argv[2]
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 18620
    extra = [a for i, a in enumerate(sys.argv[3:], 3) if a != "--port" and sys.argv[i - 1] != "--port"]
    chat = {"messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 4, "temperature": 0}
    comp = {"prompt": "Hello", "n_predict": 2}
    out, bad = {}, []
    kf = tempfile.NamedTemporaryFile("w", suffix=".keys", delete=False)
    kf.write("# comment\nfile-key-2\n"); kf.close()
    p, base = serve(binary, model, port, ["--api-key", "right-key-1", "--api-key-file", kf.name, *extra])
    try:
        refused = json.dumps({"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}, separators=(",", ":"))
        out["health_open"] = req(base, "/health")[0]
        for name, path, body, method in (("chat", "/v1/chat/completions", chat, None), ("models", "/v1/models", None, None),
                                         ("completion", "/completion", comp, None), ("lora_get", "/lora-adapters", None, "GET"),
                                         ("lora_post", "/lora-adapters", [], "POST")):
            nokey, wrong = req(base, path, body, None, method=method), req(base, path, body, "wrong-key", method=method)
            right, fkey = req(base, path, body, "right-key-1", method=method), req(base, path, body, "file-key-2", "X-Api-Key", method=method)
            out[name] = [nokey[0], wrong[0], right[0], fkey[0]]
            if nokey[0] != 401 or wrong[0] != 401 or nokey[1] != refused: bad.append(name + " not refused")
            if right[0] == 401 or fkey[0] == 401: bad.append(name + " refused with the right key")
        if out["health_open"] != 200: bad.append("/health not open")
    finally:
        p.terminate(); p.wait(); os.unlink(kf.name)
    p, base = serve(binary, model, port, extra)
    try:
        out["no_key_server"] = [req(base, "/v1/chat/completions", chat)[0], req(base, "/lora-adapters")[0]]
        if 401 in out["no_key_server"]: bad.append("no-key server refused")
    finally:
        p.terminate(); p.wait()
    print(json.dumps({"ok": not bad, "failures": bad, **out}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
