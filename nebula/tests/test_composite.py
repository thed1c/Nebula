"""Composite routing (nebula/turn.py): one explainable choice (the stages in every decision), a retry on the next
eligible model when an engine fails before it answered anything (all three paths), passthrough callers that are
never moved, and a route's own deadline."""
import json
import os
import tempfile
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call


class Refusing(ThreadingHTTPServer):
    """Healthy to the registry, but every chat is refused (an engine overloaded or failing before its first byte)."""
    daemon_threads = True

    def __init__(self, status=503):
        self.calls = 0
        outer = self

        class H(BaseHTTPRequestHandler):
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
                if self.path == "/props":
                    return self._json(200, {"total_slots": 1, "default_generation_settings": {"n_ctx": 32768}})
                self._json(200, {"status": "ok"} if self.path == "/health" else {"data": []})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path == "/tokenize":                 # the size estimate: no tokenizer here
                    return self._json(404, {"error": "no"})
                outer.calls += 1
                self._json(status, {"error": {"message": "overloaded (pretend)"}})
        super().__init__(("127.0.0.1", 0), H)
        threading.Thread(target=self.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server_port}"


class CompositeTest(unittest.TestCase):
    def setUp(self, rules=None):
        self.bad, self.good = Refusing(), FakeEngine("good")
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False, "rules": rules or {},
            "tiers": [{"name": "first", "engine": self.bad.url, "timeout_s": 42}, {"name": "second", "engine": self.good.url}],
            "callers": [{"id": "strict", "tags": ["strict/*"], "models": ["first"], "mode": "passthrough"},
                        {"id": "app", "tags": ["app/*"], "models": ["first", "second"]}],
            "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        self.bad.shutdown(); self.bad.server_close(); self.good.shutdown()

    def post(self, path, body, label="app/x"):
        if body.get("stream"):
            import urllib.request
            req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "X-Nebula-Caller": label,
                                                  "X-Nebula-Session": uuid.uuid4().hex})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode()
        return call(self.base, path, body, {"X-Nebula-Caller": label, "X-Nebula-Session": uuid.uuid4().hex})

    def decisions(self):
        rows = self.srv.ledger.db.execute("SELECT tier, reason, inputs FROM decisions ORDER BY created_at, rowid").fetchall()
        return [(t, r, json.loads(i)) for t, r, i in rows]

    def attempt_tier(self):
        return self.srv.ledger.db.execute("SELECT tier FROM attempts ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()[0]

    def test_retry_before_the_first_byte_plain(self):
        st, body = self.post("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200, body)
        self.assertEqual(self.bad.calls, 1)
        (t1, r1, i1), (t2, r2, i2) = self.decisions()[-2:]
        self.assertEqual((t1, t2, r2), ("first", "second", "retry_before_first_byte:first"))
        self.assertEqual((i2["from"], i2["status"], i2["caller"]), ("first", 503, "app"))
        self.assertEqual(self.attempt_tier(), "second")

    def test_retry_before_the_first_byte_streamed_and_through_an_agent_turn(self):
        st, text = self.post("/v1/chat/completions", {"model": "x", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200)
        self.assertTrue(text.rstrip().endswith("data: [DONE]"), text[-200:])
        self.assertEqual(self.decisions()[-1][:2], ("second", "retry_before_first_byte:first"))
        st, body = self.post("/v1/messages", {"model": "x", "max_tokens": 32, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200, body)                                     # the Anthropic dialect runs the turn itself
        self.assertEqual(self.decisions()[-1][:2], ("second", "retry_before_first_byte:first"))

    def test_a_passthrough_caller_is_never_moved(self):
        st, body = self.post("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]}, label="strict/1")
        self.assertEqual(st, 503, body)
        self.assertIn("overloaded", json.dumps(body))
        d = self.decisions()[-1]
        self.assertEqual((d[0], d[2]["stages"][0]["mode"]), ("first", "passthrough"))

    def test_a_model_the_request_names_is_not_moved(self):
        st, body = self.post("/v1/chat/completions", {"model": "first", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 503, body)
        self.assertFalse(any(r.startswith("retry") for _t, r, _i in self.decisions()))

    def test_the_stages_explain_the_choice(self):
        self.post("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        first = [i for _t, r, i in self.decisions() if not r.startswith("retry")][-1]
        names = [s["stage"] for s in first["stages"]]
        self.assertEqual(names, ["ticks", "allowed", "answers", "fits", "choice"])
        self.assertEqual(first["stages"][0]["models"], ["first", "second"])
        self.assertEqual(first["stages"][4]["model"], "first")

    def test_a_routes_own_deadline(self):
        self.assertEqual(self.srv.engines["first"].timeout_s, 42.0)
        self.assertEqual(self.srv.engines["second"].timeout_s, self.srv.cfg.engine_timeout_s)


class NoRetryTest(CompositeTest):
    def setUp(self):
        super().setUp(rules={"retries_before_first_byte": 0})

    def test_retry_before_the_first_byte_plain(self):
        st, body = self.post("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 503, body)

    def test_retry_before_the_first_byte_streamed_and_through_an_agent_turn(self):
        pass


if __name__ == "__main__":
    unittest.main()
