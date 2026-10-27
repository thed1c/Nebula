"""nebula/pairface.py (internal): this Nebula as its machine's engine for the Personal AI Router (model servers on
your network, nebula/pairnet.py).

PAIR runs llama.cpp in its router mode on 127.0.0.1:8081 and adopts a server already there when it answers like one:
`/props` says `"role": "router"`, `/health` answers, and `/models` lists its models. This face answers so, with this
Nebula's models, and passes every inference request to this Nebula's own API on loopback. So a request that reaches
this machine over PAIR's mutual TLS lands in Nebula (its routing, its locked engines, its tenancy), never in a bare
engine.

Who a request is for comes with it from the Nebula that sent it (the household's main Nebula, where guests enter):
`X-Nebula-Tenant`, signed with the secret the two Nebulas share (set when this machine was installed). A request
without a valid signature is the owner's own (the household's machines are the owner's).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PASS = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses", "/completion", "/v1/models", "/tokenize")


def sign(secret: str, tenant: str, body: bytes) -> str:
    return hmac.new(secret.encode(), tenant.encode() + b"\n" + hashlib.sha256(body).digest(), hashlib.sha256).hexdigest()


class Face:
    def __init__(self, server, port: int = 8081, models=None, secret: str | None = None):
        self.server, self.models, self.secret = server, models, secret
        face = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/health":
                    return self._json(200, {"status": "ok"})
                if path == "/props":
                    return self._json(200, {"role": "router", "model_alias": "nebula", "total_slots": 8})
                if path in ("/models", "/v1/models"):
                    names = face.model_names()
                    if path == "/models":
                        return self._json(200, {"object": "list", "data": [{"id": n, "status": {"value": "loaded"}} for n in names]})
                    return self._json(200, {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "nebula"} for n in names]})
                return self._json(404, {"error": {"message": "not here", "type": "not_found_error"}})

            def do_POST(self):
                path = self.path.split("?")[0]
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                if path not in PASS:
                    return self._json(404, {"error": {"message": "not here", "type": "not_found_error"}})
                headers = {"Content-Type": "application/json", "X-Nebula-Internal": getattr(face.server, "internal_token", "") or ""}
                tenant, sig = self.headers.get("X-Nebula-Tenant"), self.headers.get("X-Nebula-Tenant-Sig")
                if tenant and face.secret and sig and hmac.compare_digest(sign(face.secret, tenant, raw), sig):
                    if tenant != "owner":                 # a guest of the household: a guest tenant here too
                        headers["X-Nebula-From"] = "via:" + tenant
                url = f"http://127.0.0.1:{face.server.server_port}{path}"
                try:
                    resp = urllib.request.urlopen(urllib.request.Request(url, data=raw, headers=headers, method="POST"), timeout=900)
                except urllib.error.HTTPError as e:
                    body = e.read()
                    self.send_response(e.code)
                    self.send_header("Content-Type", e.headers.get("Content-Type") or "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    return self.wfile.write(body)
                except OSError as e:
                    return self._json(502, {"error": {"message": str(e)}})
                with resp:
                    self.send_response(resp.status)
                    self.send_header("Content-Type", resp.headers.get("Content-Type") or "application/json")
                    self.end_headers()
                    while True:
                        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()

        self.srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, name="pair-face", daemon=True).start()
        self.port = self.srv.server_port

    def model_names(self) -> list[str]:
        if self.models is not None:
            return list(self.models)
        return [t.name for t in self.server.cfg.tiers if not getattr(t, "mesh", None) and not getattr(t, "provider", None)
                and not getattr(t, "is_command", False)]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
