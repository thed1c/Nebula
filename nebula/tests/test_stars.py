"""The core stars (nebula/stars.py): the API Server (pass-keys, a dedicated model), the Assistant (the page's chat as
the built-in agent), Lookup (the MCP servers, two built in); a router (several models behind one address) with caller headers;
watching a router and serving at once."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from nebula import stars
from nebula.config import Config
from nebula.examples.fake_router import FakeRouter
from nebula.lookup.docs import Docs
from nebula.lookup.web import page_text, parse_duckduckgo
from nebula.server import make_server
from nebula.tests.fake_router_http import HttpRouter

ROOT = Path(__file__).resolve().parents[2]


def call(base, path, body=None, headers=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(base + path, data=data, headers={**h, **(headers or {})}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class Rig:
    """A Nebula whose one tier is a router on a loopback port, with a config file."""
    def __init__(self, extra=None):
        self.router = HttpRouter()
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "nebula.json")
        d = {"port": 0, "ledger": os.path.join(self.dir, "ledger.sqlite3"), "monitor": False,
             "tiers": [{"name": "router", "engine": self.router.address, "context": 32768, "model": "default-route",
                        "headers": {"X-Caller": "nebula.{input}"}, "health_path": "/v1/models"}],
             "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}
        d.update(extra or {})
        Path(self.path).write_text(json.dumps(d))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.srv.tools.close()
        self.router.close()

    def chat(self, **kw):
        return call(self.base, "/v1/chat/completions", {"model": "anything", "messages": [{"role": "user", "content": "hi"}], **kw})

    def saved(self):
        return json.loads(Path(self.path).read_text())


class RouterTierTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()

    def tearDown(self):
        self.rig.stop()

    def test_api_calls_go_to_the_router_tagged_as_the_api(self):
        status, raw = self.rig.chat()
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "answered by default-route")
        p = self.rig.router.posts()[-1]
        self.assertEqual(p["path"], "/model/v1/chat/completions")
        self.assertEqual(p["headers"]["X-Caller"], "nebula.api")
        self.assertEqual(p["body"]["model"], "default-route")   # Nebula routes: the tier's own model name, the router's default

    def test_the_router_tier_is_up_by_its_catalog(self):
        nodes = json.loads(call(self.rig.base, "/api/registry")[1])
        self.assertTrue(next(n for n in nodes if n["name"] == "router")["up"])
        self.assertNotIn("/model/health", [s["path"] for s in self.rig.router.seen])

    def test_the_page_chat_is_the_assistant(self):
        status, raw = call(self.rig.base, "/api/chat", {"text": "hello"})
        self.assertEqual(status, 200, raw)
        p = self.rig.router.posts()[-1]
        self.assertEqual(p["headers"]["X-Caller"], "nebula.assistant")
        self.assertIn("You are Halo", p["body"]["messages"][0]["content"])   # the built-in agent's prompt

    def test_a_dedicated_model_by_name_and_the_choices(self):
        v = json.loads(call(self.rig.base, "/api/stars")[1])
        self.assertIn({"tier": "router", "model": "big-model", "label": "router › big-model"}, v["api"]["choices"])
        status, raw = call(self.rig.base, "/api/stars/api", {"route": {"tier": "router", "model": "big-model"}})
        self.assertEqual(status, 200, raw)
        self.rig.chat()
        self.assertEqual(self.rig.router.posts()[-1]["body"]["model"], "big-model")
        self.assertEqual(self.rig.saved()["api"]["route"], {"tier": "router", "model": "big-model"})
        call(self.rig.base, "/api/stars/assistant", {"route": {"tier": "router", "model": "small-model"}})
        call(self.rig.base, "/api/chat", {"text": "hello"})
        self.assertEqual(self.rig.router.posts()[-1]["body"]["model"], "small-model")
        call(self.rig.base, "/api/stars/api", {"route": {}})              # back to Nebula routes
        self.rig.chat()
        self.assertEqual(self.rig.router.posts()[-1]["body"]["model"], "default-route")
        self.assertEqual(call(self.rig.base, "/api/stars/api", {"route": {"tier": "nope"}})[0], 400)


class PassKeyTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()

    def tearDown(self):
        self.rig.stop()

    def test_require_make_use_and_revoke(self):
        b = self.rig.base
        self.assertEqual(call(b, "/api/stars/api", {"require_key": True})[0], 400)   # no key yet: refused
        status, raw = call(b, "/api/stars/api/keys", {"label": "laptop"})
        got = json.loads(raw)
        key, row = got["key"], got["row"]
        self.assertTrue(key.startswith("nbk_"))
        self.assertNotIn(key, Path(self.rig.path).read_text())                       # only its hash is kept
        self.assertEqual(call(b, "/api/stars/api", {"require_key": True})[0], 200)
        self.assertEqual(self.rig.chat()[0], 401)
        self.assertEqual(call(b, "/v1/models")[0], 401)
        self.assertEqual(call(b, "/v1/models", headers={"Authorization": "Bearer " + key})[0], 200)
        status, _ = call(b, "/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]},
                         headers={"x-api-key": key})
        self.assertEqual(status, 200)
        self.assertEqual(call(b, "/v1/models", headers={"Authorization": "Bearer nbk_wrong"})[0], 401)
        self.assertEqual(call(b, "/api/stars")[0], 200)                               # the page itself is not the API
        v = json.loads(call(b, "/api/stars")[1])
        self.assertEqual((v["api"]["require_key"], v["api"]["keys"][0]["label"]), (True, "laptop"))
        self.assertNotIn("sha256", v["api"]["keys"][0])
        out = json.loads(call(b, f"/api/stars/api/keys/{row['id']}/revoke", {})[1])
        self.assertIn("last key", out["note"])
        self.assertEqual(self.rig.chat()[0], 401)                                     # no keys left: still closed (never opens itself)


class LookupTest(unittest.TestCase):
    def setUp(self):
        self.docs = tempfile.mkdtemp()
        Path(self.docs, "orbits.md").write_text("The black hole keeps three gold stars on its orbit.\nNothing else.\n")
        self.rig = Rig()

    def tearDown(self):
        self.rig.stop()

    def wait_tools(self, server, n, seconds=20):
        for _ in range(int(seconds * 10)):
            row = next((s for s in json.loads(call(self.rig.base, "/api/stars")[1])["lookup"]["servers"] if s["name"] == server), None)
            if row and len(row["tools"]) >= n:
                return row
            time.sleep(0.1)
        self.fail(f"{server} did not list {n} tools: {row}")

    def test_builtins_listed_off_then_on_and_used(self):
        v = json.loads(call(self.rig.base, "/api/stars")[1])["lookup"]["servers"]
        self.assertEqual({s["name"]: s["enabled"] for s in v if s["builtin"]}, {"library": True, "security": True, "docs": False, "web": False})
        status, raw = call(self.rig.base, "/api/stars/lookup", {"name": "docs", "folders": [self.docs], "disabled": False})
        self.assertEqual(status, 200, raw)
        row = self.wait_tools("docs", 3)
        self.assertEqual(sorted(t["name"] for t in row["tools"]), ["list_folders", "read_document", "search_documents"])
        out = self.rig.srv.tools.call("docs__search_documents", {"query": "gold stars"})
        self.assertIn("orbits.md", out)
        self.assertEqual(self.rig.srv.cfg.agent("assistant").tools, ["library", "security", "docs"])   # the Assistant may use it
        self.assertEqual(self.rig.saved()["mcp_servers"]["docs"]["folders"], [self.docs])
        call(self.rig.base, "/api/stars/lookup/docs/remove", {})                         # a built-in is turned off
        self.assertTrue(self.rig.saved()["mcp_servers"]["docs"]["disabled"])

    def test_add_and_remove_a_server(self):
        fake = str(ROOT / "nebula" / "_vendor" / "strata_serve" / "mcp_fake_server.py")
        status, raw = call(self.rig.base, "/api/stars/lookup", {"name": "fake", "command": sys.executable, "args": [fake]})
        self.assertEqual(status, 200, raw)
        self.wait_tools("fake", 2)
        self.assertEqual(self.rig.srv.tools.call("fake__echo", {"text": "hi"}), "hi")
        self.assertEqual(call(self.rig.base, "/api/stars/lookup", {"name": "bad"})[0], 400)
        call(self.rig.base, "/api/stars/lookup/fake/remove", {})
        self.assertNotIn("fake", self.rig.saved()["mcp_servers"])


class BuiltinServersTest(unittest.TestCase):
    def test_docs_search_read_and_fence(self):
        d = tempfile.mkdtemp()
        Path(d, "a.txt").write_text("alpha beta\ngamma\n")
        Path(d, "b.md").write_text("beta beta beta\n")
        docs = Docs([d])
        out = docs.search("beta")
        self.assertLess(out.index("b.md"), out.index("a.txt"))                         # the better match first
        self.assertIn("gamma", docs.read(str(Path(d, "a.txt"))))
        with self.assertRaises(PermissionError):
            docs.read("/etc/hostname")
        self.assertIn("no document folders", Docs([]).search("xyz"))

    def test_web_parsing_without_the_network(self):
        raw = ('<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fx&amp;rut=1">'
               'Example <b>Org</b></a><a class="result__snippet" href="#">A <b>snippet</b> here</a>')
        self.assertEqual(parse_duckduckgo(raw, 5), [{"title": "Example Org", "url": "https://example.org/x", "snippet": "A snippet here"}])
        self.assertEqual(page_text("<p>One<script>x()</script></p><p>Two &amp; three</p>"), "One\nTwo & three")


class WatchAndServeTest(unittest.TestCase):
    """Watching a router's status while serving through a tier: the map reads the status, the chat and API answer."""
    def setUp(self):
        self.watched = FakeRouter(every=0).start()
        self.rig = Rig({"watch": {"url": self.watched.url, "poll_seconds": 0.5}})

    def tearDown(self):
        self.rig.stop()
        self.watched.close()

    def test_both(self):
        c = json.loads(call(self.rig.base, "/api/config")[1])
        self.assertTrue(c["serving"])
        self.assertEqual([t["name"] for t in c["tiers"]], ["big", "small", "cli"])        # the map: the watched router
        self.assertEqual([t["name"] for t in c["own_tiers"]], ["router"])
        self.assertEqual(self.rig.chat()[0], 200)                                        # not 409: it serves
        self.assertEqual(call(self.rig.base, "/api/chat", {"text": "hi"})[0], 200)
        self.assertEqual(len(json.loads(call(self.rig.base, "/api/sessions")[1])), 2)      # its own ledger
        self.assertEqual(self.watched.other, [])                                          # the watched router: read only


class ConfigTest(unittest.TestCase):
    def test_round_trip_keeps_the_stars_and_not_the_builtin_agent(self):
        cfg = Config.from_dict({"tiers": [{"name": "t", "engine": "http://127.0.0.1:1", "headers": {"X": "y"}}],
                                "api": {"require_key": False}, "mcp_servers": {"web": {"builtin": "web", "disabled": True}}})
        stars.assistant_agent(cfg)
        d = cfg.to_dict()
        self.assertEqual(d["tiers"][0]["headers"], {"X": "y"})
        self.assertNotIn("assistant", [a["name"] for a in d["agents"]])
        self.assertEqual(d["mcp_servers"]["web"]["builtin"], "web")
        self.assertEqual(stars.hub_servers(cfg), {})                                     # off: nothing started


if __name__ == "__main__":
    unittest.main()
