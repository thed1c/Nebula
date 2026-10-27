"""The API Server's entry points (nebula/entries.py): each its own listener; its scope enforced (this machine, or
private addresses only, never outside), its model list enforced, a bad key bounced (and counted for its Quasar), its
API styles (OpenAI, Anthropic, Ollama), its tenant per key; added and undone by its wizard; kept across restarts."""
import json
import os
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from nebula import entries as EP
from nebula import lan
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call


from nebula.tests import free_port  # noqa: E402,F401  (outside the system's outgoing ports)


def raw(url, body=None, key=None, method=None):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


class EntryTest(unittest.TestCase):
    def setUp(self):
        self.small, self.big = FakeEngine("small"), FakeEngine("big")
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
            "tiers": [{"name": "small", "engine": self.small.url, "shared": True}, {"name": "big", "engine": self.big.url}],
            "assistant": {"halo": {"on": False}},
            "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        self.start()

    def start(self):
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()

    def tearDown(self):
        self.stop()
        self.small.stop(); self.big.stop()

    def add(self, **kw):
        st, e = call(self.base, "/api/api-servers", {"name": "Kids", "scope": "loopback", "port": free_port(), "formats": ["openai", "ollama"],
                                                    "models": ["small"], **kw})
        self.assertEqual(st, 200, e)
        st, k = call(self.base, f"/api/api-servers/{e['id']}/keys", {"label": "tablet"})
        self.assertEqual(st, 200, k)
        return e, k["key"], f"http://127.0.0.1:{e['port']}"

    def chat(self, url, key, model="small"):
        return raw(url + "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}]}, key)

    def test_model_list_and_keys_enforced_and_bounces_counted(self):
        e, key, url = self.add()
        self.assertEqual(self.chat(url, key)[0], 200)
        self.assertNotEqual(self.chat(url, key, "big")[0], 200)                          # not on its list
        models = json.loads(raw(url + "/v1/models", key=key)[2])["data"]
        self.assertEqual([m["id"] for m in models if not m["id"].startswith("nebula")], ["small"])
        self.assertEqual(self.chat(url, None)[0], 401)                                     # no key: bounced
        self.assertEqual(self.chat(url, "nbc_wrong")[0], 401)
        other, okey, _ourl = self.add(name="House", models=[])
        self.assertEqual(self.chat(url, okey)[0], 401)                                    # another entry's key: bounced
        st, made = call(self.base, "/api/stars/api/keys", {"label": "main"})
        self.assertEqual(self.chat(url, made["key"])[0], 401)                             # the main API's key: not this one's
        v = {x["id"]: x for x in call(self.base, "/api/api-servers")[1]["entries"]}
        self.assertEqual(v[e["id"]]["stats"]["bounces"], 4)
        self.assertEqual(v[e["id"]]["stats"]["requests"], 1)                             # the refused model is not a call
        refused = [r for r in self.srv.seclog.read(kinds=["auth.refused"])]
        self.assertEqual(sum(r["count"] for r in refused), 4)
        row = self.srv.ledger.db.execute("SELECT inputs FROM decisions WHERE advisory = 0 ORDER BY rowid LIMIT 1").fetchone()
        self.assertEqual(json.loads(row[0])["tenant"], f"caller:entry-{e['id']}")        # its own tenant
        self.assertIn("main", v)

    def test_a_clients_own_model_name(self):
        """An older app asks for "old-name"; the door gives it the model Nebula calls "small", and lists it as "old-name"."""
        e, key, url = self.add(rename={"old-name": "small"})
        self.assertEqual(self.chat(url, key, "old-name")[0], 200)
        self.assertEqual(len(self.small.requests), 1)
        models = json.loads(raw(url + "/v1/models", key=key)[2])["data"]
        self.assertIn("old-name", [m["id"] for m in models])
        self.assertNotIn("small", [m["id"] for m in models])
        st, bad = call(self.base, "/api/api-servers", {"name": "X", "scope": "loopback", "port": free_port(), "formats": ["openai"],
                                                       "rename": {"old-name": "nope"}})
        self.assertEqual(st, 400, bad)

    def test_formats(self):
        e, key, url = self.add(formats=["openai"])
        self.assertEqual(raw(url + "/v1/messages", {"model": "small", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}, key)[0], 404)
        self.assertEqual(raw(url + "/api/chat", {"model": "small", "messages": []}, key)[0], 404)
        self.assertEqual(raw(url + "/api/stars", key=key)[0], 404)                       # never the page's API
        e2, key2, url2 = self.add(name="Claude apps", formats=["anthropic"])
        st, _ct, body = raw(url2 + "/v1/messages", {"model": "small", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}, key2)
        self.assertEqual(st, 200, body)
        self.assertEqual(json.loads(body)["type"], "message")

    def test_ollama(self):
        e, key, url = self.add()
        tags = json.loads(raw(url + "/api/tags", key=key)[2])
        self.assertEqual([m["name"] for m in tags["models"]], ["small"])
        st, ct, body = raw(url + "/api/chat", {"model": "small", "stream": False, "messages": [{"role": "user", "content": "hi"}]}, key)
        self.assertEqual(st, 200, body)
        d = json.loads(body)
        self.assertEqual((d["message"]["content"], d["done"]), ("Hello world", True))
        st, ct, body = raw(url + "/api/chat", {"model": "small", "messages": [{"role": "user", "content": "hi"}]}, key)
        self.assertEqual((st, ct), (200, "application/x-ndjson"))
        lines = [json.loads(x) for x in body.decode().splitlines() if x.strip()]
        self.assertEqual("".join(x["message"]["content"] for x in lines), "Hello world")
        self.assertTrue(lines[-1]["done"])
        self.assertTrue(lines[-1]["eval_count"] > 0)
        st, ct, body = raw(url + "/api/generate", {"model": "small", "prompt": "hi", "stream": False}, key)
        self.assertEqual(json.loads(body)["response"], "Hello world")
        self.assertEqual(raw(url + "/api/chat", {"model": "small", "messages": []}, "nbc_wrong")[0], 401)
        self.assertEqual(json.loads(raw(url + "/api/chat", {"model": "small", "messages": []}, "nbc_wrong")[2])["error"],
                         "this entry point needs one of its own keys")                     # Ollama's error shape

    def test_scope_enforced(self):
        e, key, url = self.add()
        real = lan.is_loopback
        lan.is_loopback = lambda a: False                                                 # as if from another machine
        try:
            self.assertEqual(self.chat(url, key)[0], 403)
        finally:
            lan.is_loopback = real
        lan_e, lkey, lurl = self.add(name="House LAN", scope="lan", models=[])
        self.assertTrue(self.srv.entries.spec(lan_e["id"])["require_key"])               # on the network: a key by default
        open_e = call(self.base, "/api/api-servers", {"name": "Open LAN", "scope": "lan", "port": free_port(), "require_key": False})[1]
        self.assertFalse(open_e["require_key"])                                           # design decision: encouraged, not required
        self.assertIn("anyone on it", open_e["warning"])                                  # and warned about
        self.assertIsNone(lan_e.get("warning"))
        self.assertEqual(self.chat(lurl, lkey)[0], 200)
        real = EP.on_lan
        EP.on_lan = lambda a: False                                                       # a public address
        try:
            self.assertEqual(self.chat(lurl, lkey)[0], 403)
        finally:
            EP.on_lan = real
        for addr, ok in (("192.168.1.20", True), ("10.0.0.5", True), ("172.20.1.1", True), ("fe80::1", True), ("127.0.0.1", True),
                         ("8.8.8.8", False), ("100.64.3.4", False), ("2001:4860::8888", False), ("::ffff:192.168.1.2", True)):
            self.assertEqual(EP.on_lan(addr), ok, addr)
        self.assertEqual(call(self.base, "/api/api-servers", {"name": "Out", "scope": "internet", "port": free_port()})[0], 400)

    def test_guest_entry_and_restart(self):
        e, key, url = self.add(name="Friend", guest=True, models=[])
        self.assertEqual(self.chat(url, key, "small")[0], 200)                          # shared
        self.assertNotEqual(self.chat(url, key, "big")[0], 200)                          # not shared with guests
        self.stop()
        self.start()
        self.assertEqual(self.chat(url, key, "small")[0], 200)                          # listening again after a restart
        st, v = call(self.base, f"/api/api-servers/{e['id']}/remove", {})
        self.assertEqual(st, 200, v)
        with self.assertRaises(OSError):
            urllib.request.urlopen(url + "/health", timeout=3)
        self.assertIsNone(self.srv.cfg.raw.get("callers") and next((c for c in self.srv.cfg.callers if c["id"] == "entry-" + e["id"]), None))

    def test_the_wizard(self):
        before = Path(self.path).read_bytes()
        port = free_port()
        answers = {"name": "Home Assistant", "scope": "lan", "formats": ["ollama"], "models": ["small"], "port": port, "make_key": True}
        st, p = call(self.base, "/api/wizards/entry/plan", {"answers": answers})
        self.assertEqual(st, 200, p)
        self.assertTrue(p["warnings"])
        self.assertEqual(Path(self.path).read_bytes(), before)                           # Cancel: nothing
        st, job = call(self.base, "/api/wizards/entry/apply", {"answers": answers, "wait": True})
        self.assertEqual((st, job["state"]), (200, "done"), job)
        key = job["result"]["key"]
        st, _ct, body = raw(f"http://127.0.0.1:{port}/api/chat", {"model": "small", "stream": False, "messages": [{"role": "user", "content": "hi"}]}, key)
        self.assertEqual(st, 200, body)
        call(self.base, f"/api/wizards/jobs/{job['id']}/undo", {})
        self.assertEqual(self.srv.entries.specs(), [])
        self.assertEqual(call(self.base, "/api/wizards/entry/plan", {"answers": {**answers, "port": self.srv.server_port}})[0], 409)


if __name__ == "__main__":
    unittest.main()
