"""A pretend router on a loopback port (another OpenAI-compatible server with several models behind one address):
POST <prefix>/v1/chat/completions (streamed or not) and GET <prefix>/v1/models; it records each request's path,
headers and body."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class HttpRouter:
    def __init__(self, prefix: str = "/model", models=("default-route", "big-model", "small-model")):
        self.prefix, self.models, self.seen = prefix, list(models), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                outer.seen.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
                if self.path == outer.prefix + "/v1/models":
                    return self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in outer.models]})
                self._send(404, {"error": "no"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.seen.append({"method": "POST", "path": self.path, "headers": dict(self.headers), "body": body})
                if self.path != outer.prefix + "/v1/chat/completions":
                    return self._send(404, {"error": "no"})
                model = body.get("model") or "default-route"
                text = f"answered by {model}"
                if not body.get("stream"):
                    return self._send(200, {"id": "c1", "object": "chat.completion", "model": model,
                                            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                         "finish_reason": "stop"}],
                                            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in ({"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}], "model": model},
                              {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "model": model,
                               "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}):
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    @property
    def address(self) -> str:
        return f"http://127.0.0.1:{self.srv.server_port}" + self.prefix

    def posts(self):
        return [s for s in self.seen if s["method"] == "POST"]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
