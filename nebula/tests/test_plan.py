"""The decision plan (nebula/plan.py, GET/POST /api/plan): this Nebula's own plan read from and applied to its
config; with the optional watch on, the watched Switchyard's plan beside it, read only."""
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from nebula import plan as P
from nebula.config import Config
from nebula.examples.fake_router import FakeRouter
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine

ROOT = Path(__file__).resolve().parents[2]


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def serve(d: dict):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "nebula.json")
    Path(path).write_text(json.dumps({"port": 0, "monitor": False, "ledger": os.path.join(tmp, "l.sqlite3"),
                                      "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}, **d}))
    srv = make_server(Config.from_args(["--config", path]))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}", path


class OwnPlanTest(unittest.TestCase):
    def setUp(self):
        self.a, self.b = FakeEngine("a"), FakeEngine("b")
        self.srv, self.base, self.path = serve({
            "tiers": [{"name": "fast", "engine": self.a.url}, {"name": "big", "engine": self.b.url}],
            "agents": [{"name": "writer", "system": "w", "models": ["big", "fast"]}, {"name": "helper", "system": "h"}]})

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close(); self.a.stop(); self.b.stop()

    def test_read_edit_apply(self):
        status, v = call(self.base, "/api/plan")
        self.assertEqual((status, v["source"]["kind"], v["source"]["editable"]), (200, "own", True))
        jobs = {j["id"]: j for j in v["live"]["jobs"]}
        self.assertEqual(jobs["writer"]["order"], ["big", "fast"])
        self.assertEqual((jobs["helper"]["ticked"], jobs[P.DEFAULT_JOB]["order"]), (False, ["fast", "big"]))
        draft = json.loads(json.dumps(v["live"]))
        for j in draft["jobs"]:
            if j["id"] == "helper":
                j["order"], j["ticked"], j["class"] = ["big"], True, "background"
        draft["classes"][0]["tolerance_s"] = 30
        status, v = call(self.base, "/api/plan/apply", {"plan": draft})
        self.assertEqual(status, 200, v)
        self.assertTrue(v["applied"])
        saved = json.loads(Path(self.path).read_text())
        helper = next(a for a in saved["agents"] if a["name"] == "helper")
        self.assertEqual((helper["models"], helper["priority"]), (["big"], "background"))
        self.assertEqual(saved["rules"]["tolerance_seconds"]["interactive"], 30)
        self.assertEqual(call(self.base, "/api/plan/apply", {})[0], 400)          # nothing to apply

    def test_a_bad_plan_is_refused(self):
        _s, v = call(self.base, "/api/plan")
        bad = json.loads(json.dumps(v["live"])); bad["jobs"][1]["order"] = ["nope"]
        self.assertEqual(call(self.base, "/api/plan/apply", {"plan": bad})[0], 400)
        bad["jobs"][1]["order"] = ["big", "big"]
        self.assertEqual(call(self.base, "/api/plan/apply", {"plan": bad})[0], 400)


class WatchedPlanTest(unittest.TestCase):
    def setUp(self):
        self.router = FakeRouter(every=0).start()
        self.srv, self.base, self.path = serve({"ledger": None, "watch": {"url": self.router.base, "poll_seconds": 0.5}})

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close(); self.router.close()

    def test_beside_this_nebulas_own_read_only(self):
        status, v = call(self.base, "/api/plan")
        self.assertEqual((status, v["source"]["kind"], v["source"]["editable"]), (200, "own", True))
        self.assertEqual(v["watched"]["plan"]["schema"], P.SCHEMA)
        self.assertEqual([j["id"] for j in v["watched"]["plan"]["jobs"]][:1], ["writer-app"])
        self.assertEqual(self.router.other, [])                        # nothing is ever sent to it
        for gone in ("/api/plan/draft", "/api/plan/discard"):
            self.assertEqual(call(self.base, gone, {"plan": v["watched"]["plan"]})[0], 404)

    def test_no_watch_no_plan(self):
        self.assertIsNone(P.watched_plan(None))
        self.assertIn("error", P.watched_plan("http://127.0.0.1:9/switchyard/plan", ttl=0))


if __name__ == "__main__":
    unittest.main()
