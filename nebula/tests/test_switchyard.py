"""Watching another Switchyard by URL (nebula/switchyard.py, optional): its status read into the page's feeds, read only."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nebula.config import Config
from nebula.server import make_server
from nebula.switchyard import SwitchyardWatch


def status(calls=(), waiting=(), parking=()):
    return {
        "switchyard": {"actors": {"writer": "live", "planner": "off"}},
        "capacity": {"destinations": {
            "big": {"label": "The Big One", "routes": ["big", "owner-big"], "slots": 1, "busy": len(calls), "free": 1 - len(calls),
                    "queued": len(waiting), "slot_context": 262144, "avg_call_seconds": 90.0, "unavailable": None},
            "dense": {"label": "Dense lanes", "routes": ["dense"], "slots": 2, "busy": 0, "free": 2, "queued": 0,
                      "slot_context": 98304, "unavailable": None}},
            "parking": list(parking)},
        "routes": {"big": {"base_url": "http://127.0.0.1:9001/v1", "ready": True},
                   "owner-big": {"base_url": "http://127.0.0.1:9001/v1", "ready": True},
                   "dense": {"base_url": "http://127.0.0.1:9002/v1", "ready": False}},
        "in_flight_calls": list(calls),
        "waiting": list(waiting),
        "external_destinations": {"destinations": [{"id": "cli-a", "kind": "consultation", "enabled": True, "running": 0},
                                                   {"id": "cli-off", "kind": "decision", "enabled": False}]},
        "admission": {"groups": {"big": {"size": 1}}},
        "callers": [{"id": "writer-app", "name": "The writing app", "class": "interactive", "tags": ["writer-app/*"]}],
    }


CALL = {"caller": "writer-app/draft", "run": "r1", "route": "owner-big", "actor": "writer", "since": 1000.0}


class Feed:
    def __init__(self):
        self.value = status()
        self.fail = False

    def __call__(self):
        if self.fail:
            raise OSError("connection refused")
        return self.value


class WatchTest(unittest.TestCase):
    def setUp(self):
        self.feed = Feed()
        self.w = SwitchyardWatch({"url": "http://x:1"}, fetch=self.feed)

    def test_a_call_starts_and_lands(self):
        self.w.poll_once()
        self.assertEqual(self.w.live()["running"], [])
        self.feed.value = status(calls=[CALL])
        self.w.poll_once()
        t = self.w.tick()
        self.assertEqual(len(t["started"]), 1)
        a = t["started"][0]
        self.assertEqual((a["dialect"], a["title"], a["tier"], a["agent"]), ("writer-app", "The writing app", "big", "writer"))
        self.assertEqual(self.w.inputs(), [{"input": "writer-app", "attempts": 1, "sessions": 0, "last_at": a["created_at"]}])
        self.feed.value = status()
        self.w.poll_once()
        t = self.w.tick()
        self.assertEqual(t["running"], [])
        self.assertEqual([x["state"] for x in t["landed"]], ["completed"])
        self.assertEqual(self.w.recent()["attempts"][0]["title"], "The writing app")
        self.assertEqual(self.w.summary()["calls_seen"], 1)

    def test_the_models_agents_and_sources(self):
        self.w.poll_once()
        self.assertEqual([t["name"] for t in self.w.tiers()], ["big", "dense", "cli-a"])
        self.assertEqual(self.w.tiers()[2]["provider"], "consultation")
        e = {x["tier"]: x for x in self.w.engines()}
        self.assertTrue(e["big"]["up"])
        self.assertFalse(e["dense"]["up"])                      # its one route is not ready
        self.assertEqual((e["big"]["slots"], e["big"]["n_ctx"], e["big"]["model"]), (1, 262144, "The Big One"))
        self.assertEqual([a["name"] for a in self.w.agents()], ["writer", "planner"])
        self.assertEqual(self.w.sources(), [["writer-app", "The writing app", "interactive"]])
        self.assertEqual(self.w.source_of("agent.other/x"), ("agent.other/x", "agent.other/x", ""))

    def test_a_watch_is_its_url_only(self):
        w = SwitchyardWatch({"url": "http://x:1/switchyard/status"}, fetch=self.feed)
        self.assertEqual((w.base, w.url, w.registry_url, w.plan_url),
                         ("http://x:1", "http://x:1/switchyard/status", "http://x:1/switchyard/registry", "http://x:1/switchyard/plan"))
        self.assertEqual(w.name, "x:1")                          # nothing named here: its address until it reports a name
        self.feed.value = dict(status(), name="The shop's Nebula")
        w.poll_once()
        self.assertEqual(w.name, "The shop's Nebula")
        for old in ({"status_url": "http://x/s"}, {"url": "http://x", "plan_" + "command": "x"}, {"url": "http://x", "registry_file": "r.json"},
                    {"url": "http://x", "agents": ["a"]}, {"url": "http://x", "name": "n"}):
            with self.assertRaises(ValueError):
                SwitchyardWatch(old)

    def test_waits_and_parked(self):
        self.feed.value = status(waiting=[{"id": "w1", "actor": "writer", "group": "big", "since": 5.0, "reason": "waiting for big",
                                           "class": "agent"}], parking=[{"caller": "x", "reason": "held: too big"}])
        self.w.poll_once()
        lv = self.w.live()
        self.assertEqual(lv["waiting"][0]["tiers"], ["big"])
        self.assertEqual(lv["parked"][0]["reason"], "held: too big")

    def test_unreachable_keeps_the_last_view(self):
        self.feed.value = status(calls=[CALL])
        self.w.poll_once()
        self.feed.fail = True
        self.w.poll_once()
        self.assertFalse(self.w.summary()["reachable"])
        self.assertIn("connection refused", self.w.summary()["error"])
        self.assertEqual(len(self.w.live()["running"]), 1)
        self.assertEqual(self.w.tick()["started"], [])           # nothing is launched twice


class StatusServer:
    """The watched router: one GET path, and a count of everything else it was asked (it must stay zero)."""
    def __init__(self, value):
        outer = self
        self.value, self.other = value, []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/switchyard/status":
                    body = json.dumps(outer.value).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    outer.other.append(("GET", self.path))
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                outer.other.append(("POST", self.path))
                self.send_response(404)
                self.end_headers()

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/switchyard/status"


class WatchServerTest(unittest.TestCase):
    def setUp(self):
        self.router = StatusServer(status(calls=[CALL]))
        cfg = Config.from_dict({"port": 0, "ledger": None, "monitor": False,
                                "watch": {"url": self.router.url, "poll_seconds": 0.5}})
        cfg.lan = {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}
        self.srv = make_server(cfg)
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.router.srv.shutdown()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as r:
            return json.loads(r.read())

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def test_the_page_feeds(self):
        c = self.get("/api/config")
        self.assertTrue(c["watch"]["name"].startswith("127.0.0.1:"))
        self.assertEqual([t["name"] for t in c["tiers"]], ["big", "dense", "cli-a"])
        m = self.get("/api/monitor")
        self.assertEqual(m["engines"][0]["tier"], "big")
        self.assertEqual(m["sessions"]["attempts"], 1)
        self.assertEqual(self.get("/api/live")["running"][0]["tier"], "big")
        self.assertEqual(self.get("/api/sessions"), [])

    def test_nothing_is_sent(self):
        self.assertEqual(self.post("/api/chat", {"text": "hi"}), 409)
        self.assertEqual(self.post("/v1/chat/completions", {"model": "big", "messages": [{"role": "user", "content": "hi"}]}), 409)
        self.assertEqual(self.post("/api/models", {"name": "x", "kind": "engine", "url": "http://127.0.0.1:1"}), 409)
        self.assertEqual(self.router.other, [])

    def test_the_look_is_saved_in_memory_without_a_config_file(self):
        req = urllib.request.Request(self.base + "/api/look", data=json.dumps({"router": [0.3, 0.6], "junk": 1}).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertFalse(json.loads(r.read())["saved"])
        self.assertEqual(self.get("/api/look"), {"router": [0.3, 0.6]})

    def test_the_stream_starts_with_what_runs(self):
        with urllib.request.urlopen(self.base + "/api/stream", timeout=5) as r:
            head = r.readline().decode() + r.readline().decode()
        self.assertTrue(head.startswith("event: tick"))
        self.assertIn('"tier": "big"', head)


class LookFileTest(unittest.TestCase):
    """The page's layout beside the config: NAME.look.json, replaced whole, only its known sections, while watching."""
    def test_round_trip(self):
        router = StatusServer(status())
        d = Path(tempfile.mkdtemp())
        (d / "home.json").write_text(json.dumps({"port": 0, "ledger": None, "monitor": False,
                                                  "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"},
                                                  "watch": {"url": router.url}}))
        srv = make_server(Config.from_args(["--config", str(d / "home.json")]))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_port}"
        try:
            look = {"pins": {"m:big": [0.8, 0.2]}, "nodes": {"m:big": {"name": "The Big One", "size": 1.5, "colour": "#ffbe7f"}},
                    "router": [0.4, 0.5], "widgets": {"room": {"x": 0, "y": 0, "w": 300, "h": 200}}, "other": "dropped"}
            req = urllib.request.Request(base + "/api/look", data=json.dumps(look).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                self.assertTrue(json.loads(r.read())["saved"])
            look.pop("other")
            self.assertEqual(json.loads((d / "home.look.json").read_text()), look)
            with urllib.request.urlopen(base + "/api/look", timeout=5) as r:
                self.assertEqual(json.loads(r.read()), look)
            self.assertEqual(router.other, [])
        finally:
            srv.shutdown(); srv.server_close(); router.srv.shutdown()


if __name__ == "__main__":
    unittest.main()
