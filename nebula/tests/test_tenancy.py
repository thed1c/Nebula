"""Sharing with others (nebula/tenancy.py, docs/sharing.md): no tenant ever gets another's cached prefix (a slot each,
erased when it changes hands; or no cache for guests), a tenant's own prefix is still reused; guests reach only shared
models; sessions are per tenant; a paired peer reaches the model API and its own status only; engines Nebula runs
are locked to Nebula (a key only it holds, their own pages off)."""
import json
import os
import tempfile
import threading
import unittest
import urllib.request
import uuid
from pathlib import Path

from nebula.config import Config
from nebula.tests import WINDOWS, linux_only  # noqa: F401
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine, PrefixCache
from nebula.tests.test_callers import call
from nebula.tests.test_engines import FAKE, ROOT, free_port, make_gguf, wait_for

PREFIX = "You are reviewing this contract. " + "Clause %d: the parties agree to the terms set out. " * 200


def prefix(n):
    return "".join(PREFIX % i if "%d" in PREFIX else PREFIX for i in range(1))


class Rig:
    def __init__(self, isolation="slots", slots=2):
        self.shared = FakeEngine("shared")
        self.shared.prefix_cache = PrefixCache(slots)
        self.private = FakeEngine("private")
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
            "tiers": [{"name": "shared", "engine": self.shared.url, "shared": True, "cache_isolation": isolation, "slots": slots},
                      {"name": "private", "engine": self.private.url}],
            "callers": [{"id": "alice", "guest": True, "models": ["shared"]}, {"id": "bob", "guest": True, "models": ["shared"]},
                        {"id": "carol", "guest": True, "models": ["shared"]}, {"id": "mine", "models": ["private", "shared"]}],
            "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.keys = {c: call(self.base, f"/api/yard/callers/{c}/keys", {"label": "t"})[1]["key"] for c in ("alice", "bob", "carol", "mine")}

    def ask(self, who, text, model="x"):
        h = {"X-Nebula-Session": uuid.uuid4().hex}
        if who != "owner":
            h["Authorization"] = f"Bearer {self.keys[who]}"
        return call(self.base, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}]}, h)

    def cached(self, who, text, model="x"):
        st, body = self.ask(who, text, model)
        assert st == 200, body
        return body["usage"]["prompt_tokens_details"]["cached_tokens"]

    def stop(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        self.shared.shutdown(); self.private.shutdown()


class SlotIsolationTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig("slots", slots=2)

    def tearDown(self):
        self.rig.stop()

    def test_the_same_long_prefix_never_hits_across_tenants(self):
        r = self.rig
        self.assertEqual(r.cached("alice", PREFIX + "question one"), 0)
        self.assertGreater(r.cached("alice", PREFIX + "question two"), 500)        # her own prefix is reused
        self.assertEqual(r.cached("bob", PREFIX + "question one"), 0)              # the same prefix: no hit for bob
        self.assertGreater(r.cached("bob", PREFIX + "question three"), 500)
        self.assertEqual([b.get("id_slot") for b in r.shared.requests], [0, 0, 1, 1])
        self.assertEqual(r.cached("carol", PREFIX + "x"), 0)                       # a third tenant, two slots: one erased
        self.assertEqual(r.shared.erased, [0])
        self.assertEqual(r.cached("alice", PREFIX + "question four"), 0)           # alice's slot went to carol: no hit
        row = r.srv.ledger.db.execute("SELECT inputs FROM decisions WHERE advisory = 1 ORDER BY rowid DESC LIMIT 1").fetchone()
        self.assertIn("erased", json.loads(row[0])["isolation"])

    def test_guests_reach_only_shared_models_and_sessions_are_per_tenant(self):
        r = self.rig
        st, body = r.ask("alice", "hi", model="private")
        self.assertEqual(st, 503, body)                                             # named, but not shared with guests
        self.assertIn("not shared with guests", body["error"]["message"])
        self.assertEqual(r.ask("mine", "hi", model="private")[0], 200)              # the owner's own app may
        same = {"model": "x", "messages": [{"role": "user", "content": "the same opening"}]}
        for who in ("alice", "bob"):
            call(r.base, "/v1/chat/completions", same, {"Authorization": f"Bearer {r.keys[who]}"})
        rows = r.srv.ledger.db.execute("SELECT DISTINCT session_id FROM attempts a JOIN events e USING (session_id) "
                                       "WHERE e.payload LIKE '%the same opening%'").fetchall()
        self.assertEqual(len(rows), 2)                                              # never one session for two tenants


class NoCacheIsolationTest(unittest.TestCase):
    """An engine that cannot pin slots (Impulse today): guests never use the cache; the owner still does."""

    def setUp(self):
        self.rig = Rig("no-cache", slots=2)

    def tearDown(self):
        self.rig.stop()

    def test_guests_never_hit(self):
        r = self.rig
        self.assertEqual(r.cached("owner", PREFIX + "a"), 0)
        self.assertEqual(r.cached("alice", PREFIX + "b"), 0)
        self.assertEqual(r.cached("alice", PREFIX + "c"), 0)
        self.assertEqual([b.get("cache_prompt") for b in r.shared.requests], [None, False, False])


class WhatAGuestSeesTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig("slots")

    def tearDown(self):
        self.rig.stop()

    def test_a_peer_reaches_the_model_api_and_its_own_status_only(self):
        r = self.rig
        r.ask("owner", "the owner's own secret plan")
        token = "peer-token-" + uuid.uuid4().hex
        from nebula import lan
        if r.srv.pairing is None:
            r.srv.pairing = lan.Pairing(r.srv.cfg) if hasattr(lan, "Pairing") else None
        self.assertIsNotNone(r.srv.pairing, "this Nebula has no pairing store")
        r.srv.pairing.remember({"uuid": "friend-1", "name": "a friend", "address": None, "port": None}, token)
        r.srv.cfg.lan_auth_loopback = True                     # treat this machine's calls as from beyond it
        try:
            def get(path):
                req = urllib.request.Request(r.base + path, headers={"Authorization": f"Bearer {token}"})
                try:
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        return resp.status, json.loads(resp.read())
                except urllib.error.HTTPError as e:
                    return e.code, None
            for path in ("/api/sessions", "/api/recent", "/api/yard/decisions", "/api/yard/stats", "/api/config", "/api/plan"):
                self.assertEqual(get(path)[0], 403, path)
            self.assertEqual(get("/v1/models")[0], 200)
            st, s = get("/switchyard/status")
            self.assertEqual((st, s["callers"], s["in_flight_calls"]), (200, [], []))
            self.assertNotIn("last", s["switchyard"])
        finally:
            r.srv.cfg.lan_auth_loopback = False

    def test_a_callers_view_is_its_own(self):
        r = self.rig
        r.ask("bob", "bob's question")
        st, s = call(r.base, "/switchyard/status", None, {"Authorization": f"Bearer {r.keys['alice']}"})
        self.assertEqual([c["id"] for c in s["callers"]], ["alice"])
        st, p = call(r.base, "/switchyard/plan", None, {"Authorization": f"Bearer {r.keys['alice']}"})
        self.assertEqual([j["id"] for j in p["jobs"]], ["caller:alice"])
        st, s = call(r.base, "/switchyard/status")                                  # the owner sees everyone
        self.assertEqual(len(s["callers"]), 4)


class LockedEnginesTest(unittest.TestCase):
    """Engines Nebula runs: their own pages off, a key only Nebula holds, no cross-slot cache when shared."""

    def test_flags_and_the_key(self):
        tmp = tempfile.mkdtemp()
        make_gguf(os.path.join(tmp, "m.gguf"))
        argsfile = os.path.join(tmp, "args.log")
        os.environ["FAKE_IMPULSE_ARGS"] = argsfile
        port = free_port()
        path = os.path.join(tmp, "nebula.json")
        Path(path).write_text(json.dumps({"port": 0, "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False,
                                          "engines": [{"name": "e", "kind": "llama.cpp", "binary": FAKE, "model": "m.gguf",
                                                       "port": port, "context": 2048, "slots": 2, "cwd": ROOT, "guests": True}],
                                          "lan": {"decided": True, "discoverable": False}}))
        srv = make_server(Config.from_args(["--config", path]))
        base = f"http://127.0.0.1:{srv.server_port}"
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            self.assertEqual(call(base, "/api/yard/engines/e/start", {"wait": True})[1]["state"], "ready")
            argv = json.loads(open(argsfile).read().splitlines()[-1])
            for w in ("--no-webui", "--cache-ram", "--no-cache-idle-slots", "-sps", "--api-key-file"):
                self.assertIn(w, argv)
            self.assertNotIn("--no-slots", argv)                                  # shared: Nebula erases slots through it
            keyfile = argv[argv.index("--api-key-file") + 1]
            if not WINDOWS:
                self.assertEqual(oct(os.stat(keyfile).st_mode & 0o777), "0o600")
            self.assertNotIn(open(keyfile).read().strip(), json.dumps(call(base, "/api/yard/engines/e")[1]))   # never shown
            st, _ = call(f"http://127.0.0.1:{port}", "/v1/models")
            self.assertEqual(st, 401)                                             # not even from this machine without it
            call(base, "/api/yard/engines/e/route", {})
            tier = srv.cfg.tier("e")
            self.assertEqual((tier.shared, tier.cache_isolation, tier.api_key_file), (True, "slots", keyfile))
            self.assertEqual(call(base, "/v1/chat/completions", {"model": "e", "messages": [{"role": "user", "content": "hi"}]})[0], 200)
            self.assertTrue(call(base, "/api/yard/engines/e/verify", {})[1]["ok"])
        finally:
            srv.shutdown(); srv.server_close(); srv.tools.close()
            del os.environ["FAKE_IMPULSE_ARGS"]


if __name__ == "__main__":
    unittest.main()
