"""A pretend Impulse (strata-lite-server) binary: llama-server's flags in, fake_engine.py's answers out. It prints load
lines the way the real one does, reports the model file it was given in /props, and runs until it is signalled.
Nebula's managed-engine tests and the greenfield container start it as an engine's binary.

    python3 -m nebula.tests.fake_impulse -m MODEL.gguf --host 127.0.0.1 --port P [-c CTX] [-np SLOTS] [other flags]

Flags it does not know are accepted and ignored, as a newer engine's would be by an older wrapper. With
FAKE_IMPULSE_FAIL=1 it fails to load (exit code 3).
"""
import argparse
import json
import os
import sys
import time

from nebula.tests.fake_engine import FakeEngine


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("-c", "--ctx-size", type=int, default=4096)
    p.add_argument("-np", "--parallel", type=int, default=1)
    p.add_argument("--api-key-file")
    p.add_argument("--api-key")
    a, rest = p.parse_known_args()
    for i in range(2):
        print(f"impulse: loading {os.path.basename(a.model)}, step {i + 1} of 2", file=sys.stderr, flush=True)
        time.sleep(0.1)
    if os.environ.get("FAKE_IMPULSE_FAIL") == "1":
        print("impulse: error: cannot load the model (fake)", file=sys.stderr, flush=True)
        return 3
    srv = FakeEngine(os.path.splitext(os.path.basename(a.model))[0], n_ctx=a.ctx_size // max(1, a.parallel), port=a.port,
                     background=False)
    srv.model_path, srv.slots = os.path.abspath(a.model), a.parallel
    if a.api_key_file:                           # llama-server's: every path but /health needs the key
        srv.api_key = open(a.api_key_file, encoding="utf-8").read().split()[0]
    elif a.api_key:
        srv.api_key = a.api_key
    if os.environ.get("FAKE_IMPULSE_ARGS"):      # the tests read the flags it was started with
        with open(os.environ["FAKE_IMPULSE_ARGS"], "a", encoding="utf-8") as f:
            f.write(json.dumps(sys.argv[1:]) + "\n")
    print(f"impulse: listening on http://{a.host}:{a.port} ({a.parallel} slots of {a.ctx_size // max(1, a.parallel)})",
          file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
