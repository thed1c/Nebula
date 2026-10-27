"""The documented router status (docs/router-status-v1.md): the schema holds for the examples, the callers come
from the status, a URL or a file, and a Nebula watching the fake router draws it and only reads."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from nebula.config import Config
from nebula.examples.fake_router import FakeRouter, status as fake_status
from nebula.server import make_server
from nebula.switchyard import SwitchyardWatch
from nebula.tests.test_switchyard import CALL, status as test_status

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "docs" / "router-status-v1.schema.json").read_text())
TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
         "integer": int, "number": (int, float)}


def problems(schema: dict, value, where: str = "$") -> list[str]:
    """The schema's subset the document uses (type, required, properties, additionalProperties, items)."""
    out = []
    types = schema.get("type")
    if types is not None:
        names = types if isinstance(types, list) else [types]
        ok = any(isinstance(value, TYPES[n]) and not (n in ("integer", "number") and isinstance(value, bool))
                 for n in names)
        if not ok:
            return [f"{where}: {type(value).__name__} is not {'/'.join(names)}"]
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                out.append(f"{where}: {k} is missing")
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                out += problems(props[k], v, f"{where}.{k}")
            elif isinstance(schema.get("additionalProperties"), dict):
                out += problems(schema["additionalProperties"], v, f"{where}.{k}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            out += problems(schema["items"], v, f"{where}[{i}]")
    return out


class SchemaTest(unittest.TestCase):
    def test_the_examples_hold(self):
        call = dict(CALL)
        for st in (fake_status([]), fake_status([dict(call, route="big")], "http://x/registry.json"),
                   test_status(calls=[call], waiting=[{"id": "w1", "actor": "writer", "group": "big", "since": 5.0}],
                               parking=[{"caller": "x", "reason": "held"}])):
            self.assertEqual(problems(SCHEMA, st), [])

    def test_the_checker_catches(self):
        self.assertEqual(problems(SCHEMA, {"capacity": {"destinations": {}}}), ["$: in_flight_calls is missing"])
        self.assertTrue(problems(SCHEMA, {"capacity": {"destinations": {"a": {"slots": "two"}}}, "in_flight_calls": []}))


class CallersTest(unittest.TestCase):
    def test_from_the_status_itself(self):
        st = dict(test_status(calls=[CALL]), callers=[{"id": "writer-app", "name": "The writing app", "tags": ["writer-app/*"]}])
        w = SwitchyardWatch({"url": "http://x:1"}, fetch=lambda: st)
        w.poll_once()
        self.assertEqual(w.live()["running"][0]["title"], "The writing app")

    def test_from_its_registry(self):
        r = FakeRouter(every=0).start()
        try:
            r.calls = [{"caller": "job.summaries", "run": "r1", "route": "small", "actor": "planner", "since": 1.0}]
            w = SwitchyardWatch({"url": r.base})
            w.poll_once()
            self.assertEqual(w.live()["running"][0]["title"], "Nightly jobs")
            self.assertEqual([s[0] for s in w.sources()], ["writer-app", "nightly"])
            self.assertEqual(r.other, [])
        finally:
            r.close()

    def test_a_path_the_status_names_is_never_read(self):
        d = tempfile.mkdtemp()
        reg = Path(d, "registry.json")
        reg.write_text(json.dumps({"callers": [{"id": "writer-app", "name": "The writing app", "tags": ["writer-app/*"]}]}))
        st = test_status(calls=[CALL])
        st.pop("callers")
        st["capacity"]["registry"] = str(reg)
        w = SwitchyardWatch({"url": "http://127.0.0.1:9"}, fetch=lambda: st)
        w.poll_once()
        self.assertEqual(w.live()["running"][0]["title"], CALL["caller"])     # a file on this machine: not read


class WatchTheFakeRouterTest(unittest.TestCase):
    """End to end: nebula/examples/watch-router.json's shape against the fake Switchyard; nothing but reads reach it."""
    def setUp(self):
        self.router = FakeRouter(every=0).start()
        self.router.calls = [{"caller": "writer-app/draft", "run": "r1", "route": "big", "actor": "writer", "since": 1.0}]
        example = json.loads((ROOT / "nebula" / "examples" / "watch-router.json").read_text())
        example.update(port=0, monitor=False)
        example["watch"].update(url=self.router.base, poll_seconds=0.5)
        self.srv = make_server(Config.from_dict(example))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.router.close()

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

    def test_drawn_and_only_read(self):
        self.assertEqual([t["name"] for t in self.get("/api/config")["tiers"]], ["big", "small", "cli"])
        live = self.get("/api/live")["running"]
        self.assertEqual((live[0]["tier"], live[0]["title"], live[0]["agent"]), ("big", "The writing app", "writer"))
        self.assertEqual([a["name"] for a in self.get("/api/config")["agents"]], ["writer", "planner", "reviewer"])
        self.assertEqual(self.post("/v1/chat/completions", {"model": "big", "messages": [{"role": "user", "content": "hi"}]}), 409)
        self.assertEqual(self.router.other, [])

    def test_the_plan_over_http(self):
        v = self.get("/api/plan")
        self.assertNotIn("error", v["watched"]["plan"])
        self.assertEqual([j["id"] for j in v["watched"]["plan"]["jobs"]][:1], ["writer-app"])
        self.assertEqual(self.router.other, [])


class NoPairingTest(unittest.TestCase):
    """LAN discovery and pairing come in a later release: this one says so and has no pairing routes."""
    def test_no_pairing(self):
        from nebula.tests.test_escalation import Rig
        rig = Rig()
        try:
            with urllib.request.urlopen(rig.base + "/api/lan", timeout=5) as r:
                lan = json.loads(r.read())
            self.assertEqual((lan["pairing"], lan["found"], lan["peers"], lan["window"]["open"]), (False, [], [], False))
            for path in ("/pair", "/api/lan/open", "/api/lan/pair", "/api/lan/settings"):
                req = urllib.request.Request(rig.base + path, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError, msg=path) as cm:
                    urllib.request.urlopen(req, timeout=5)
                self.assertIn(cm.exception.code, (400, 404), path)
        finally:
            rig.stop()


if __name__ == "__main__":
    unittest.main()
