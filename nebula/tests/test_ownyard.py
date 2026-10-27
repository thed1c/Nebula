"""This Nebula's own Switchyard (nebula/ownyard.py): its status in the router-status-v1 shape (served at
/switchyard/status), its lanes with the calls holding them, its decisions with their stages, its stats; and a second
Nebula watching it the way it watches a router."""
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
import uuid
from pathlib import Path

from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call
from nebula.tests.test_router_status import SCHEMA, problems


def serve(d):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "nebula.json")
    Path(path).write_text(json.dumps({"port": 0, "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False,
                                      "lan": {"decided": True, "discoverable": False, "uuid": uuid.uuid4().hex, "name": "n"}, **d}))
    srv = make_server(Config.from_args(["--config", path]))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


class OwnYardTest(unittest.TestCase):
    def setUp(self):
        self.big, self.small = FakeEngine("big"), FakeEngine("small")
        self.srv, self.base = serve({
            "tiers": [{"name": "big", "engine": self.big.url}, {"name": "small", "engine": self.small.url}],
            "callers": [{"id": "writer", "name": "The writer", "tags": ["writer/*"], "models": ["small", "big"]}]})

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        self.big.shutdown(); self.small.shutdown()

    def ask(self, text="hi", label="writer/draft", stream=False):
        body = {"model": "x", "stream": stream, "messages": [{"role": "user", "content": text}]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-Nebula-Caller": label,
                                              "X-Nebula-Session": uuid.uuid4().hex})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()

    def test_the_status_is_router_status_v1(self):
        self.ask()
        st, s = call(self.base, "/switchyard/status")
        self.assertEqual(st, 200)
        self.assertEqual(problems(SCHEMA, s), [])
        self.assertEqual(set(s["capacity"]["destinations"]), {"big", "small"})
        self.assertEqual(s["callers"], [{"id": "writer", "name": "The writer", "tags": ["writer/*"]}])
        self.assertEqual(s["switchyard"]["last"]["tier"], "small")
        self.assertEqual(call(self.base, "/switchyard/plan")[1]["jobs"][-1]["id"], "caller:writer")
        self.assertEqual(call(self.base, "/switchyard/registry")[1]["callers"][0]["id"], "writer")
        self.srv.cfg.api = {"require_key": True}
        self.assertEqual(call(self.base, "/switchyard/status")[0], 401)       # behind the API's key, like /v1

    def test_a_call_holds_its_lane_while_it_runs(self):
        t = threading.Thread(target=self.ask, kwargs={"text": "tell it slowly", "stream": True})
        t.start()
        seen = None
        for _ in range(60):
            lanes = {l["name"]: l for l in call(self.base, "/api/yard/lanes")[1]["lanes"]}
            if lanes["small"]["busy"]:
                seen = lanes["small"]
                break
            time.sleep(0.05)
        t.join()
        self.assertIsNotNone(seen, "the lane never showed the call")
        self.assertEqual((seen["slots"], seen["calls"][0]["caller"], seen["calls"][0]["source"]), (1, "writer/draft", "The writer"))
        self.assertEqual(call(self.base, "/api/yard/lanes")[1]["lanes"][1]["busy"], 0)   # given back

    def test_decisions_and_stats(self):
        self.ask()
        self.ask(label="other/x")
        d = call(self.base, "/api/yard/decisions")[1]
        self.assertEqual([r["tier"] for r in d["decisions"]], ["big", "small"])
        self.assertEqual(d["decisions"][1]["caller"], "writer/draft")
        self.assertEqual([s["stage"] for s in d["decisions"][1]["stages"]], ["ticks", "allowed", "answers", "fits", "choice"])
        self.assertEqual(d["counters"]["route"], {"big": 1, "small": 1})
        s = call(self.base, "/api/yard/stats")[1]
        self.assertEqual((s["models"]["small"]["24h"]["calls"], s["callers"]["writer"]["24h"]["calls"]), (1, 1))
        self.assertEqual(s["models"]["small"]["24h"]["bands"]["<2k"], 1)
        self.assertIsNotNone(s["models"]["small"]["10m"]["latency"]["avg"])

    def test_another_nebula_watches_this_one(self):
        self.ask()
        watcher, wbase = serve({"watch": {"url": self.base, "poll_seconds": 0.5}})
        try:
            time.sleep(0.8)
            self.assertEqual([t["name"] for t in call(wbase, "/api/config")[1]["tiers"]][:2], ["big", "small"])
            plan = call(wbase, "/api/plan")[1]
            self.assertEqual(plan["watched"]["plan"]["jobs"][-1]["id"], "caller:writer")
            self.assertEqual(call(wbase, "/api/config")[1]["watch"]["name"], "n")       # the name it reports
            lanes = call(wbase, "/api/watch/lanes")[1]["lanes"]
            self.assertEqual([l["name"] for l in lanes], ["big", "small"])
        finally:
            watcher.shutdown(); watcher.server_close(); watcher.tools.close()


if __name__ == "__main__":
    unittest.main()
