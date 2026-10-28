"""The Gravity node (nebula/gravity.py): Nebulas joined by invite only, nothing shared until ticked, a peer's granted
model used here like any other (as that peer's tenant there), revoking, owner machines administered over the admin
ALPN, friends not. Real iroh endpoints, direct on loopback, no relay, no network."""
import json
import os
import tempfile
import threading
import unittest
import urllib.request
import uuid
from pathlib import Path

from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call

try:
    import iroh  # noqa: F401
    HAVE_IROH = True
except ImportError:
    HAVE_IROH = False


def nebula(name, tiers=()):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "nebula.json")
    Path(path).write_text(json.dumps({"port": 0, "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False, "tiers": list(tiers),
                                      "lan": {"decided": True, "discoverable": False, "uuid": uuid.uuid4().hex, "name": name}}))
    srv = make_server(Config.from_args(["--config", path]))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    st, v = call(base, "/api/yard/node/on", {"name": name})
    assert st == 200 and v["on"], v
    return srv, base


@unittest.skipUnless(HAVE_IROH, "iroh is not installed (pip install nebula-eh[gravity])")
class GravityTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine("big")
        self.a, self.abase = nebula("alice", [{"name": "big", "engine": self.engine.url, "shared": True},
                                             {"name": "mine", "engine": self.engine.url},
                                             {"name": "cloud", "provider": "openai-compatible", "base_url": "https://api.example.com/v1",
                                              "model": "m", "shared": True}])
        self.b, self.bbase = nebula("bob", [{"name": "local", "engine": self.engine.url}])

    def tearDown(self):
        for s in (self.a, self.b):
            s.shutdown(); s.server_close(); s.tools.close()
        self.engine.shutdown()

    def friends(self):
        inv = call(self.abase, "/api/yard/node/invite", {"kind": "friend"})[1]["invite"]
        st, j = call(self.bbase, "/api/yard/node/join", {"invite": inv})
        self.assertEqual(st, 200, j)
        return inv, j["peer"]["id"], call(self.abase, "/api/yard/node")[1]["peers"][0]["id"]

    def test_invite_join_nothing_until_ticked_then_used_then_revoked(self):
        inv, a_id, b_id = self.friends()
        self.assertEqual(call(self.bbase, "/api/yard/node/join", {"invite": inv})[0], 403)   # an invite works once
        self.assertEqual(call(self.bbase, f"/api/yard/node/peers/{a_id}/adopt", {})[1]["made"], [])   # nothing granted
        self.assertEqual(call(self.abase, f"/api/yard/node/peers/{b_id}/grants", {"models": ["mine"]})[0], 400)  # not shared
        st, r = call(self.abase, f"/api/yard/node/peers/{b_id}/grants", {"models": ["big"]})
        self.assertEqual((st, r["needs_confirm"], r["granted"]), (200, ["big"], False))      # Nebula does not run its engine
        self.assertIn("may not be safe", r["warning"])
        self.assertEqual(call(self.abase, f"/api/yard/node/peers/{b_id}/grants", {"models": ["big"], "confirm": ["big"]})[1]["grants"], ["big"])
        out = call(self.bbase, f"/api/yard/node/peers/{a_id}/adopt", {})[1]
        self.assertEqual(out["made"], ["alice-big"])
        self.assertIn("not guaranteed safe", out["warnings"][0])                             # and the guest is told
        self.assertTrue(self.b.cfg.tier("alice-big").not_guaranteed_safe)
        again = Config.from_args(["--config", self.b.cfg.path])                              # the mark survives a restart
        self.assertTrue(again.tier("alice-big").not_guaranteed_safe)
        st, body = call(self.bbase, "/v1/chat/completions", {"model": "alice-big", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200, body)
        row = self.a.ledger.db.execute("SELECT inputs FROM decisions WHERE advisory = 0 ORDER BY rowid DESC LIMIT 1").fetchone()
        self.assertEqual(json.loads(row[0])["tenant"], f"peer:{b_id}")                       # bob's own tenant at alice's
        bridge = self.b.gravity.bridge.url                                                   # a model not granted: refused
        req = urllib.request.Request(f"{bridge}/peer/{a_id}/v1/chat/completions", data=json.dumps(
            {"model": "mine", "messages": [{"role": "user", "content": "hi"}]}).encode(), headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=30)
        self.assertEqual(cm.exception.code, 403)
        self.assertEqual(call(self.abase, f"/api/yard/node/peers/{b_id}/revoke", {})[1]["revoked"], True)
        st, body = call(self.bbase, "/v1/chat/completions", {"model": "alice-big", "messages": [{"role": "user", "content": "again"}]})
        self.assertNotEqual(st, 200, body)                                                   # no longer a peer: nothing
        kinds = [(e["kind"], e["who"]) for e in self.a.seclog.read(limit=100)]                # alice's security log
        for k in (("gravity.join", f"peer:{b_id}"), ("gravity.grants", "owner"), ("guest.refused", f"peer:{b_id}"),
                  ("gravity.revoked", "owner"), ("gravity.invite", "owner")):
            self.assertIn(k, kinds)
        self.assertIn("gravity.refused", [k for k, _ in kinds])                              # the used invite, again
        self.assertTrue(self.a.seclog.verify()["ok"])

    def test_a_stale_address_is_replaced_by_the_one_the_peer_says_now(self):
        """BUGS #3: addresses were frozen at invite time. Each look says where each side is now, and both keep that."""
        _inv, a_id, b_id = self.friends()
        self.a.gravity.peer(b_id)["addr"] = {"id": b_id, "relay": None, "direct": ["127.0.0.1:9"]}   # bob "moved": alice's copy is stale
        self.b.gravity.peer(a_id)["addr"] = {**self.a.gravity.addr(), "relay": "https://old-relay.example"}  # and bob's of alice
        self.b.gravity.conns.clear()
        self.assertTrue(self.b.gravity.look(a_id)["reachable"])
        self.assertEqual(self.a.gravity.peer(b_id)["addr"], self.b.gravity.addr())           # alice now has bob's current address
        self.assertEqual(self.b.gravity.peer(a_id)["addr"], self.a.gravity.addr())
        again = Config.from_args(["--config", self.a.cfg.path])                              # and it is saved
        self.assertEqual(next(p for p in again.raw["gravity"]["peers"] if p["id"] == b_id)["addr"], self.b.gravity.addr())
        self.assertFalse(self.a.gravity.fresh_addr(b_id, {"id": "someone-else", "direct": []}))   # only its own id's
        self.assertFalse(self.a.gravity.fresh_addr(b_id, {"id": b_id, "direct": "1.2.3.4:5"}))

    def test_the_invite_says_how_a_friend_gets_here(self):
        r = call(self.abase, "/api/yard/node/invite", {"kind": "friend"})[1]
        self.assertEqual(r["reach"], "lan")
        self.assertIn("works only on your own network", r["note"])
        call(self.abase, "/api/yard/node/public-relay", {"on": True})
        r = call(self.abase, "/api/yard/node/invite", {"kind": "friend"})[1]
        self.assertEqual(r["reach"], "public")
        self.assertIn('must also turn on "Use the public relay"', r["note"])

    def test_a_member_sees_every_model_with_no_route_to_the_ones_not_shared(self):
        """They would see them, just with "no route". Cloud models stay Anti-Gravity: never listed."""
        _inv, a_id, b_id = self.friends()
        seen = self.b.gravity.look(a_id)                                                     # bob looks at alice
        self.assertEqual((seen["offers"], sorted(seen["unrouted"]), seen["reachable"]), ([], ["big", "mine"], True))
        call(self.abase, f"/api/yard/node/peers/{b_id}/grants", {"models": ["big"], "confirm": ["big"]})
        seen = self.b.gravity.look(a_id)
        self.assertEqual((seen["offers"], seen["unrouted"]), (["big"], ["mine"]))
        self.assertEqual(self.a.gravity.look(b_id)["unrouted"], ["local"])                   # and alice sees bob's
        peer = call(self.bbase, "/api/yard/node")[1]["peers"][0]                             # the page reads it here
        self.assertEqual((peer["offers"], peer["unrouted"], peer["reachable"]), (["big"], ["mine"], True))

    def test_cloud_models_are_anti_gravity(self):
        _inv, _a_id, b_id = self.friends()
        st, err = call(self.abase, f"/api/yard/node/peers/{b_id}/grants", {"models": ["cloud"], "confirm": ["cloud"]})
        self.assertEqual(st, 400, err)                                                       # never, even confirmed, even by API
        self.assertIn("Anti-Gravity", err["error"]["message"])
        m = {x["name"]: x for x in call(self.abase, "/api/yard/node/models")[1]["models"]}
        self.assertIn("Anti-Gravity", m["cloud"]["anti_gravity"])
        self.assertIsNone(m["big"]["anti_gravity"])
        self.assertIsNotNone(m["big"]["needs_care"])
        call(self.abase, "/api/yard/node/models/big/risky", {"risky": True})
        self.assertTrue(Config.from_args(["--config", self.a.cfg.path]).tier("big").risky)

    def test_owner_machines_are_administered_friends_are_not(self):
        _inv, a_id, _b = self.friends()
        st, err = call(self.bbase, f"/api/yard/node/peers/{a_id}/admin/engines")
        self.assertEqual(st, 403, err)                                                       # a friend administers nothing
        c, cbase = nebula("carol-laptop")
        try:
            inv = call(self.abase, "/api/yard/node/invite", {"kind": "owner"})[1]["invite"]   # alice's own machine joins
            st, j = call(cbase, "/api/yard/node/join", {"invite": inv})
            self.assertEqual((st, j["peer"]["role"]), (200, "admin"), j)
            st, eng = call(cbase, f"/api/yard/node/peers/{a_id}/admin/engines")
            self.assertEqual(st, 200, eng)
            self.assertIn("cards", eng)
            st, out = call(cbase, f"/api/yard/node/peers/{a_id}/admin/hosts", {"name": "far-box"})
            self.assertEqual(st, 200, out)                                                   # a write, over the admin ALPN
            self.assertEqual(self.a.cfg.hosts[-1]["name"], "far-box")
        finally:
            c.shutdown(); c.server_close(); c.tools.close()

    def test_nothing_is_announced(self):
        v = call(self.abase, "/api/yard/node")[1]
        self.assertIsNone(v["addr"]["relay"])                                                # no relay unless a gateway is set
        self.assertEqual(v["relays"], [])
        self.assertTrue(v["id"])


if __name__ == "__main__":
    unittest.main()
