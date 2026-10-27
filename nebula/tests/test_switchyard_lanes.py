"""Watching a router's lanes and decisions (nebula/switchyard.py, GET /api/watch/lanes and /api/watch/decisions):
each lane's slots, holders, waits, rules and routes; the decisions feed kept by this Nebula; all read only."""
import json
import threading
import unittest
import urllib.request

from nebula import switchyard
from nebula.config import Config
from nebula.examples.fake_router import FakeRouter, status as fake_status
from nebula.server import make_server
from nebula.switchyard import SwitchyardWatch
from nebula.tests.test_router_status import SCHEMA, problems
from nebula.tests.test_switchyard import status as bare_status

CALL = {"caller": "writer-app/draft", "route": "big", "actor": "writer", "run": "r1", "since": 1000.0}


class Feed:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class LanesTest(unittest.TestCase):
    def test_a_lane_with_its_holder_rules_and_routes(self):
        st = fake_status([CALL], n=1)
        st["waiting"] = [{"id": "w1", "actor": "planner", "caller": "job.summaries", "group": "big", "class": "background",
                          "since": 1001.0, "reason": "waiting for Big model", "behind": ["writer"]}]
        st["routes"]["small"].update(ready=False, reason="health check failed", model_file="a.gguf",
                                     expected_model_file="b.gguf")
        self.assertEqual(problems(SCHEMA, st), [])
        w = SwitchyardWatch({"url": "http://x:1"}, fetch=Feed(st))
        w.poll_once()
        lanes = {lane["name"]: lane for lane in w.lanes()}
        big, small = lanes["big"], lanes["small"]
        self.assertEqual((big["label"], big["slots"], big["busy"]), ("Big model", 1, 1))
        self.assertEqual([(h["actor"], h["route"]) for h in big["holders"]], [("writer", "big")])
        self.assertEqual(big["rules"], {"slots": 1, "timeout_seconds": 600, "max_queued": {"background": 4}})
        self.assertEqual([(x["actor"], x["class"], x["behind"]) for x in big["waiting"]], [("planner", "background", ["writer"])])
        self.assertEqual([c["actor"] for c in big["calls"]], ["writer"])
        r = big["routes"][0]
        self.assertEqual((r["id"], r["ready"], r["priority"], r["policy_state"], r["eligible"]["tools"]),
                         ("big", True, 1, "qualified", True))
        self.assertFalse(small["up"])
        self.assertEqual((small["routes"][0]["reason"], small["routes"][0]["expected_model_file"]),
                         ("health check failed", "b.gguf"))

    def test_without_the_optional_sections(self):
        w = SwitchyardWatch({"url": "http://x:1"}, fetch=Feed(bare_status(calls=[dict(CALL, route="owner-big")])))
        w.poll_once()
        lanes = {lane["name"]: lane for lane in w.lanes()}
        self.assertEqual(set(lanes), {"big", "dense"})
        self.assertEqual((lanes["big"]["slots"], lanes["big"]["holders"], lanes["big"]["rules"]), (1, [], {}))
        self.assertEqual([c["actor"] for c in lanes["big"]["calls"]], ["writer"])
        self.assertEqual(w.decisions_feed()["decisions"], [])


class DecisionsTest(unittest.TestCase):
    def test_kept_once_newest_first_and_bounded(self):
        feed = Feed(fake_status([CALL], n=1))
        w = SwitchyardWatch({"url": "http://x:1"}, fetch=feed)
        w.poll_once()
        w.poll_once()                                  # the same last decisions: not kept twice
        d = w.decisions_feed()
        self.assertEqual(sorted(x["kind"] for x in d["decisions"]), ["admission", "capacity", "route"])
        self.assertTrue(all(x["seen_at_start"] for x in d["decisions"]))
        cap = next(x for x in d["decisions"] if x["kind"] == "capacity")
        self.assertEqual((cap["selected"], cap["candidates"], cap["agree"], cap["source"]), ("big", ["big", "small"], True, "writer-app/draft"))
        self.assertEqual(d["counters"]["capacity"]["agree"], 1)
        feed.value = fake_status([dict(CALL, run="r2", since=2000.0, route="small", actor="planner")], n=2)
        w.poll_once()
        d = w.decisions_feed()
        self.assertEqual(len(d["decisions"]), 6)
        self.assertEqual((d["decisions"][0]["actor"], d["decisions"][0]["seen_at_start"]), ("planner", False))
        old = switchyard.DECISIONS_KEPT
        for i in range(old):                           # more than it keeps: the oldest go
            feed.value = fake_status([dict(CALL, run=f"x{i}", since=3000.0 + i)], n=3 + i)
            w.poll_once()
        self.assertEqual(len(w.decisions), old)


class WatchedLanesServerTest(unittest.TestCase):
    """End to end on the fake router: the two feeds answer, and nothing but reads reach the router."""
    def setUp(self):
        self.router = FakeRouter(every=0).start()
        self.router.calls = [dict(CALL, since=1000.0)]
        cfg = Config.from_dict({"port": 0, "ledger": None, "monitor": False,
                                "watch": {"url": self.router.url, "poll_seconds": 0.5}})
        cfg.lan = {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}
        self.srv = make_server(cfg)
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.router.close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as r:
            return json.loads(r.read())

    def test_the_feeds(self):
        lanes = self.get("/api/watch/lanes")
        self.assertEqual([lane["name"] for lane in lanes["lanes"]], ["big", "small"])
        self.assertEqual(lanes["lanes"][0]["holders"][0]["actor"], "writer")
        self.assertTrue(lanes["read_at"])
        d = self.get("/api/watch/decisions")
        self.assertEqual(sorted(x["kind"] for x in d["decisions"]), ["admission", "capacity", "route"])
        self.assertEqual(d["decisions"][0]["source"], "The writing app")     # named by the Switchyard's callers
        self.assertEqual(self.router.other, [])

    def test_the_page_has_the_widgets(self):
        with urllib.request.urlopen(self.base + "/", timeout=5) as r:
            page = r.read().decode()
        for needle in ('data-w="lanes"', 'data-w="decisions"', "/api/watch/lanes", "/api/watch/decisions", "function openLane"):
            self.assertIn(needle, page)


if __name__ == "__main__":
    unittest.main()
