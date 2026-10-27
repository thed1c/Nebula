"""OpenTelemetry (nebula/otel.py): off by default and then sends nothing; on, one span per turn to the collector with
the turn's metadata (model, tokens, tenant, class, how it ended) and never its prompt, reply or a key; the collector's
address is https or this machine's."""
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call


class Collector:
    def __init__(self):
        self.posts = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                outer.posts.append((self.path, self.rfile.read(int(self.headers.get("Content-Length") or 0))))
                self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1/traces"


class OtelTest(unittest.TestCase):
    def setUp(self):
        self.engine, self.col = FakeEngine("small"), Collector()
        tmp = tempfile.mkdtemp()
        self.path = os.path.join(tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({"port": 0, "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False,
                                               "tiers": [{"name": "small", "engine": self.engine.url}],
                                               "assistant": {"halo": {"on": False}},
                                               "lan": {"decided": True, "discoverable": False, "uuid": "u-1", "name": "n"}}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        self.engine.stop(); self.col.srv.shutdown()

    def chat(self, text):
        return call(self.base, "/v1/chat/completions", {"model": "small", "messages": [{"role": "user", "content": text}]})[0]

    def test_off_by_default_then_metadata_only(self):
        self.assertFalse(call(self.base, "/api/otel")[1]["enabled"])
        self.chat("SECRET-PROMPT-1")
        self.srv.otel.flush()
        self.assertEqual(self.col.posts, [])                                          # off: nothing sent
        st, v = call(self.base, "/api/otel", {"enabled": True, "endpoint": self.col.url})
        self.assertEqual((st, v["enabled"]), (200, True), v)
        self.assertEqual(self.chat("SECRET-PROMPT-2 with a password hunter2"), 200)
        self.srv.otel.flush()
        self.assertEqual(len(self.col.posts), 1)
        path, raw = self.col.posts[0]
        self.assertEqual(path, "/v1/traces")
        for secret in (b"SECRET-PROMPT", b"hunter2", b"Hello world", b"Let me think"):
            self.assertNotIn(secret, raw)                                                # never a prompt or a reply
        span = json.loads(raw)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attrs = {a["key"]: list(a["value"].values())[0] for a in span["attributes"]}
        self.assertEqual((span["name"], attrs["nebula.tier"], attrs["gen_ai.request.model"], attrs["nebula.tenant"]),
                         ("nebula.turn", "small", "small", "owner"))
        self.assertGreater(int(attrs["gen_ai.usage.output_tokens"]), 0)
        self.assertEqual(span["status"]["code"], 1)
        self.assertIn("otel.settings", [e["kind"] for e in self.srv.seclog.read()])

    def test_the_collector_is_https_or_this_machine(self):
        self.assertEqual(call(self.base, "/api/otel", {"endpoint": "http://10.0.0.5:4318/v1/traces"})[0], 400)
        self.assertEqual(call(self.base, "/api/otel", {"endpoint": "https://otel.example.com/v1/traces"})[0], 200)
        self.assertEqual(call(self.base, "/api/otel", {"enabled": True, "endpoint": "http://127.0.0.1:4318/v1/traces"})[0], 200)


if __name__ == "__main__":
    unittest.main()
