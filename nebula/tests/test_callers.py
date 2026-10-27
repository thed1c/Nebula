"""Callers (nebula/callers.py): a key names the client, a label the origin inside it; a caller's models are its job's
ticks and its class its place at the gates; both are in the Plan, and Apply writes them."""
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from nebula import callers as C
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine


def call(base, path, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(base + path, data=data, headers={**h, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class CallersTest(unittest.TestCase):
    def setUp(self):
        self.big, self.small = FakeEngine("big"), FakeEngine("small")
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
            "tiers": [{"name": "big", "engine": self.big.url}, {"name": "small", "engine": self.small.url}],
            "callers": [{"id": "writer-app", "name": "The writing app", "class": "background", "tags": ["writer/*"], "models": ["small"]},
                        {"id": "coder", "tags": ["code/*"], "models": ["big"]}],
            "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        self.big.shutdown(); self.small.shutdown()

    def chat(self, headers=None):
        import uuid                                                   # each its own conversation: no floor carried over
        return call(self.base, "/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]},
                    {"X-Nebula-Session": uuid.uuid4().hex, **(headers or {})})

    def last(self):
        tier, inputs = self.srv.ledger.db.execute("SELECT tier, inputs FROM decisions ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        return tier, json.loads(inputs)

    def test_a_label_picks_the_caller_and_its_ticks(self):
        self.assertEqual(self.chat()[0], 200)
        self.assertEqual(self.last()[0], "big")                                   # no caller: the default order
        self.assertEqual(self.chat({"X-Nebula-Caller": "writer/draft"})[0], 200)
        tier, inputs = self.last()
        self.assertEqual((tier, inputs["caller"], inputs["label"], inputs["priority"]), ("small", "writer-app", "writer/draft", "background"))
        self.assertEqual(self.chat({"X-Nebula-Caller": "someone/else"})[0], 200)
        tier, inputs = self.last()
        self.assertEqual((tier, inputs["caller"], inputs["label"]), ("big", None, "someone/else"))   # a label alone is still recorded

    def test_the_key_names_the_client_and_wins_over_the_label(self):
        st, r = call(self.base, "/api/yard/callers/coder/keys", {"label": "ci"})
        self.assertEqual(st, 200, r)
        key = r["key"]
        self.assertNotIn(key, Path(self.path).read_text())                        # only its hash is kept
        self.assertNotIn("sha256", json.dumps(call(self.base, "/api/yard/callers")[1]))
        self.assertEqual(self.chat({"Authorization": f"Bearer {key}", "X-Nebula-Caller": "writer/draft"})[0], 200)
        tier, inputs = self.last()
        self.assertEqual((tier, inputs["caller"], inputs["label"]), ("big", "coder", "writer/draft"))
        # with the API Server requiring a key, a caller's key opens it; revoked, it does not
        self.srv.cfg.api = {"require_key": True}
        self.assertEqual(self.chat()[0], 401)
        self.assertEqual(self.chat({"x-api-key": key})[0], 200)
        call(self.base, f"/api/yard/callers/coder/keys/{r['row']['id']}/revoke", {})
        self.assertEqual(self.chat({"x-api-key": key})[0], 401)

    def test_plan_jobs_and_apply(self):
        st, v = call(self.base, "/api/plan")
        jobs = {j["id"]: j for j in v["live"]["jobs"]}
        self.assertEqual((jobs["caller:writer-app"]["order"], jobs["caller:writer-app"]["class"], jobs["caller:writer-app"]["tags"]),
                         (["small"], "background", ["writer/*"]))
        plan = json.loads(json.dumps(v["live"]))
        for j in plan["jobs"]:
            if j["id"] == "caller:writer-app":
                j["order"], j["class"] = ["big", "small"], "interactive"
        st, out = call(self.base, "/api/plan/apply", {"plan": plan})
        self.assertEqual(st, 200, out)
        self.assertIn("caller:writer-app", out["applied"]["changed"])
        saved = {c["id"]: c for c in json.loads(Path(self.path).read_text())["callers"]}
        self.assertEqual((saved["writer-app"]["models"], saved["writer-app"]["class"]), (["big", "small"], "interactive"))
        self.chat({"X-Nebula-Caller": "writer/x"})
        self.assertEqual(self.last()[0], "big")

    def test_checks_and_the_api(self):
        with self.assertRaises(C.CallerError):
            C.check({"id": "Bad Id"})
        with self.assertRaises(C.CallerError):
            C.check({"id": "a", "class": "urgent"})
        with self.assertRaises(C.CallerError):
            C.check({"id": "a", "models": ["nope"]}, ["big"])
        with self.assertRaises(C.CallerError):
            C.check({"id": "a", "keys2": []})
        st, r = call(self.base, "/api/yard/callers", {"id": "phone", "tags": ["phone/*"], "models": ["big"]})
        self.assertEqual(st, 200, r)
        self.assertEqual(call(self.base, "/api/yard/callers", {"id": "phone", "models": ["nope"]})[0], 400)
        self.assertEqual(call(self.base, "/api/yard/callers/phone/remove", {})[1]["removed"], True)
        self.assertEqual(call(self.base, "/api/yard/callers/phone/keys", {})[0], 404)
        self.assertEqual([c["id"] for c in json.loads(Path(self.path).read_text())["callers"]], ["writer-app", "coder"])


if __name__ == "__main__":
    unittest.main()
