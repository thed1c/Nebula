"""A pretend local Gravity router (internally: the Personal AI Router's OpenAI-compatible client proxy), for the tests
and the greenfield container. Its "machines" are pretend engines, each with its own models. It schedules as that
router does: only a machine with the exact model; the least pending work first, counting its own reservations so a
burst spreads; one machine per request; a failure before the first byte is retried on another machine; none left: 503
with Retry-After. Its identity probe answers 409 "proxy-facade". Loopback only.

    python3 -m nebula.tests.fake_pair --port 18280 --node alpha=m1,m2 --node beta=m2
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nebula.tests.fake_engine import FakeEngine

PROBE = "X-NVPAIR-Engine-Identity-Probe"


class Node:
    def __init__(self, name: str, models: list[str], engine=None):
        self.name, self.models = name, list(models)
        self.engine = engine or FakeEngine(name)
        self.pending = 0
        self.served = 0
        self.down = False


class FakeRouter(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, nodes: list[Node], port: int = 0):
        self.nodes, self.lock = nodes, threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def _json(self, code, obj, extra=None):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.headers.get(PROBE) == "1":
                    return self._json(409, {"error": {"code": "proxy-facade", "message": "the compatibility facade is not the engine"}})
                models = sorted({m for n in outer.nodes if not n.down for m in n.models})
                if self.path == "/v1/models":
                    return self._json(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in models]})
                if self.path == "/api/tags":
                    return self._json(200, {"models": [{"name": m} for m in models]})
                self._json(404, {"error": "no"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path != "/v1/chat/completions":
                    return self._json(404, {"error": "no"})
                model = body.get("model")
                tried = set()
                while True:
                    with outer.lock:
                        owners = [n for n in outer.nodes if model in n.models and n.name not in tried and not n.down]
                        if not owners:
                            return self._json(503, {"error": {"message": f"no machine serves {model!r} now"}}, {"Retry-After": "5"})
                        n = min(owners, key=lambda x: (x.pending, x.name))
                        n.pending += 1                       # the reservation, before the machine reports load
                    tried.add(n.name)
                    try:
                        req = urllib.request.Request(n.engine.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                     headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(req, timeout=30) as r:
                            data, ctype = r.read(), r.headers.get("Content-Type") or "application/json"
                    except (urllib.error.URLError, OSError):
                        with outer.lock:
                            n.pending -= 1
                        continue                             # nothing was sent back yet: another machine
                    time.sleep(0.05)                         # a call takes a moment, so a burst overlaps
                    with outer.lock:
                        n.pending -= 1
                        n.served += 1
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

        super().__init__(("127.0.0.1", port), H)
        threading.Thread(target=self.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server_port}"

    def close(self):
        self.shutdown()
        self.server_close()
        for n in self.nodes:
            n.engine.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18280)
    p.add_argument("--node", action="append", default=[], help="name=model1,model2")
    a = p.parse_args()
    nodes = [Node(spec.split("=")[0], spec.split("=")[1].split(",")) for spec in a.node or ["alpha=tiny"]]
    r = FakeRouter(nodes, a.port)
    print(f"pretend Gravity router: {r.url} ({', '.join(n.name + ': ' + ','.join(n.models) for n in nodes)})", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
