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
        tier = f"friend-{a_id[:4]}-big".lower()                                              # never its machine name
        self.assertEqual(out["made"], [tier])
        self.assertNotIn("alice", out["warnings"][0])
        self.assertIn("not guaranteed safe", out["warnings"][0])                             # and the guest is told
        self.assertTrue(self.b.cfg.tier(tier).not_guaranteed_safe)
        again = Config.from_args(["--config", self.b.cfg.path])                              # the mark survives a restart
        self.assertTrue(again.tier(tier).not_guaranteed_safe)
        st, body = call(self.bbase, "/v1/chat/completions", {"model": tier, "messages": [{"role": "user", "content": "hi"}]})
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
        st, body = call(self.bbase, "/v1/chat/completions", {"model": tier, "messages": [{"role": "user", "content": "again"}]})
        self.assertNotEqual(st, 200, body)                                                   # no longer a peer: nothing
        kinds = [(e["kind"], e["who"]) for e in self.a.seclog.read(limit=100)]                # alice's security log
        for k in (("gravity.join", f"peer:{b_id}"), ("gravity.grants", "owner"), ("guest.refused", f"peer:{b_id}"),
                  ("gravity.revoked", "owner"), ("gravity.invite", "owner")):
            self.assertIn(k, kinds)
        self.assertIn("gravity.refused", [k for k, _ in kinds])                              # the used invite, again
        self.assertTrue(self.a.seclog.verify()["ok"])

    def test_a_blocked_peer_gets_nothing_cannot_rejoin_and_comes_back_on_unblock(self):
        _inv, a_id, b_id = self.friends()
        call(self.abase, f"/api/yard/node/peers/{b_id}/grants", {"models": ["big"], "confirm": ["big"]})
        tier = call(self.bbase, f"/api/yard/node/peers/{a_id}/adopt", {})[1]["made"][0]
        ask = lambda: call(self.bbase, "/v1/chat/completions", {"model": tier, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(ask()[0], 200)
        st, r = call(self.abase, f"/api/yard/node/peers/{b_id}/block", {"on": True})
        self.assertEqual(st, 200, r)
        self.assertTrue(r["peer"]["blocked"])
        self.assertEqual(r["peer"]["grants"], ["big"])                                       # kept for an unblock
        self.assertNotEqual(ask()[0], 200)                                                   # blocked: nothing
        peers = call(self.abase, "/api/yard/node")[1]["peers"]
        self.assertTrue(peers[0]["blocked"])                                                 # still listed
        inv2 = call(self.abase, "/api/yard/node/invite", {"kind": "friend"})[1]["invite"]
        self.assertEqual(call(self.bbase, "/api/yard/node/join", {"invite": inv2})[0], 403)  # no way back by an invite
        self.assertTrue(call(self.abase, "/api/yard/node")[1]["peers"][0]["blocked"])
        st, r = call(self.abase, f"/api/yard/node/peers/{b_id}/block", {"on": False})
        self.assertEqual(st, 200, r)
        self.assertNotIn("blocked", r["peer"])
        self.assertEqual(ask()[0], 200)                                                      # back, with its shares
        kinds = [(e["kind"], e["who"]) for e in self.a.seclog.read(limit=100)]
        self.assertIn(("gravity.blocked", "owner"), kinds)
        self.assertTrue(self.a.seclog.verify()["ok"])

    def test_limits_on_a_shared_model_hold_its_calls_and_its_replies(self):
        _inv, a_id, b_id = self.friends()
        self.assertEqual(call(self.abase, f"/api/yard/node/peers/{b_id}/limits", {"model": "big", "calls_per_hour": 2})[0], 400)  # not shared
        call(self.abase, f"/api/yard/node/peers/{b_id}/grants", {"models": ["big"], "confirm": ["big"]})
        for bad in ({"calls_per_hour": 0.5}, {"calls_per_hour": "x"}, {"max_output_tokens": 3}, {"max_concurrent": 1000}):
            st, _ = call(self.abase, f"/api/yard/node/peers/{b_id}/limits", {"model": "big", **bad})
            self.assertEqual(st, 400, bad)
        st, p = call(self.abase, f"/api/yard/node/peers/{b_id}/limits", {"model": "big", "calls_per_hour": 2, "max_output_tokens": 64})
        self.assertEqual((st, p["limits"]), (200, {"big": {"calls_per_hour": 2, "max_output_tokens": 64}}))
        tier = call(self.bbase, f"/api/yard/node/peers/{a_id}/adopt", {})[1]["made"][0]
        offered = self.b.gravity.look(a_id)
        self.assertEqual(offered["offer_limits"], {"big": {"calls_per_hour": 2, "max_output_tokens": 64}})   # bob sees them
        for i in range(2):
            st, body = call(self.bbase, "/v1/chat/completions", {"model": tier, "max_tokens": 4000, "messages": [{"role": "user", "content": f"hi {i}"}]})
            self.assertEqual(st, 200, body)
        self.assertLessEqual(self.engine.requests[-1].get("max_tokens"), 64)                  # the reply asked for no longer
        st, body = call(self.bbase, "/v1/chat/completions", {"model": tier, "messages": [{"role": "user", "content": "a third"}]})
        self.assertNotEqual(st, 200, body)                                                   # the hour's two are spent
        self.assertIn("2 calls an hour", json.dumps(body))
        self.assertIn(("guest.limited", f"peer:{b_id}"), [(e["kind"], e["who"]) for e in self.a.seclog.read(limit=50)])
        call(self.abase, f"/api/yard/node/peers/{b_id}/grants", {"models": []})                  # no longer shared: no limits kept
        self.assertNotIn("big", self.a.gravity.peer(b_id).get("limits") or {})

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

    def test_a_peer_that_does_not_answer_is_away_with_a_reason_and_no_machine_name(self):
        """It said "<name> cannot be reached: " and nothing more (a timeout's message is empty), and kept a dead
        connection for the next try. Now: a reason, the role instead of the name, and the next try dials again."""
        _inv, a_id, b_id = self.friends()
        self.assertTrue(self.a.gravity.look(b_id)["reachable"])
        self.assertTrue(any(k[0] == b_id for k in self.a.gravity.conns))
        self.b.gravity.close()                                                               # bob's Nebula closes
        self.a.gravity.peer(b_id)["addr"] = {"id": b_id, "relay": None, "direct": ["127.0.0.1:9"]}
        with self.assertLogs("nebula.gravity", "WARNING") as logs:
            seen = self.a.gravity.look(b_id)
        self.assertFalse(seen["reachable"])
        self.assertRegex(seen["error"], r"cannot be reached: \S")                           # a reason after the colon
        self.assertNotIn("bob", seen["error"])
        self.assertTrue(seen["error"].startswith("a friend's Nebula ("))
        self.assertIn("is away", logs.output[0])
        self.assertNotIn("bob", logs.output[0])
        self.assertFalse(any(k[0] == b_id for k in self.a.gravity.conns))                    # the next try dials again
        self.b.gravity = None

    def test_a_relay_change_keeps_what_peers_offered(self):
        _inv, a_id, b_id = self.friends()
        self.b.gravity.look(a_id)
        st, v = call(self.bbase, "/api/yard/node/public-relay", {"on": False})
        self.assertEqual(st, 200, v)
        self.assertEqual(sorted(v["peers"][0]["unrouted"]), ["big", "mine"])                 # the map does not empty meanwhile

    def test_an_adopted_model_follows_the_bridge_and_is_not_adopted_twice(self):
        """The bridge listens on a new port each start: an adopted model kept the old one and was dead."""
        _inv, a_id, b_id = self.friends()
        call(self.abase, f"/api/yard/node/peers/{b_id}/grants", {"models": ["big"], "confirm": ["big"]})
        tier = call(self.bbase, f"/api/yard/node/peers/{a_id}/adopt", {})[1]["made"][0]
        old = self.b.cfg.tier(tier).engine
        self.assertEqual(call(self.bbase, "/api/yard/node/public-relay", {"on": False})[0], 200)   # a new endpoint, a new bridge
        new = self.b.cfg.tier(tier).engine
        self.assertNotEqual(old, new)
        self.assertEqual(new, f"{self.b.gravity.bridge.url}/peer/{a_id}")
        self.assertEqual(self.b.engines[tier].address if hasattr(self.b.engines[tier], "address") else new, new)
        st, body = call(self.bbase, "/v1/chat/completions", {"model": tier, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200, body)                                                      # and it still answers
        self.assertEqual(call(self.bbase, f"/api/yard/node/peers/{a_id}/adopt", {})[1]["made"], [])   # not a second copy
        self.assertEqual(call(self.bbase, "/api/config")[1]["tiers"][-1]["gravity_peer"], a_id)

    def test_a_peer_has_a_random_name_new_each_time_it_comes_online(self):
        _inv, a_id, b_id = self.friends()
        first = self.a.gravity.look(b_id)["alias"]
        self.assertRegex(first, r"^[A-Z][a-z]+ [A-Z][a-z]+$")
        self.assertEqual(self.a.gravity.look(b_id)["alias"], first)                         # the same while it stays
        self.a.gravity.seen[b_id]["reachable"] = False                                      # it went away
        names = {self.a.gravity.look(b_id)["alias"]}
        for _ in range(5):
            self.a.gravity.seen[b_id]["reachable"] = False
            names.add(self.a.gravity.look(b_id)["alias"])
        self.assertGreater(len(names | {first}), 1)                                          # a new one each time it is back
        self.assertNotIn("bob", call(self.abase, "/api/yard/node")[1]["peers"][0]["alias"])

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

    def test_a_friend_sees_each_model_by_its_own_name_and_its_calls_are_marked_for_the_map(self):
        """A model's alias here ("big") is not what it is: a friend sees its own name (its Hugging Face name, else its
        file's), and a friend's call shows on this map as coming in over Gravity."""
        self.a.cfg.tier("big").hf_repo = "acme/Tiny-Model-1B-GGUF"
        self.a.cfg.tier("mine").hf_repo = "acme/Other-Model-2B"
        _inv, a_id, b_id = self.friends()
        self.assertEqual(self.b.gravity.look(a_id)["real"], {"big": "Tiny-Model-1B-GGUF", "mine": "Other-Model-2B"})
        call(self.abase, f"/api/yard/node/peers/{b_id}/grants", {"models": ["big"], "confirm": ["big"]})
        self.assertEqual(self.b.gravity.look(a_id)["real"]["big"], "Tiny-Model-1B-GGUF")
        tier = call(self.bbase, f"/api/yard/node/peers/{a_id}/adopt", {})[1]["made"][0]
        self.assertEqual(tier, f"friend-{a_id[:4]}-tiny-model-1b-gguf".lower())
        st, body = call(self.bbase, "/v1/chat/completions", {"model": tier, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200, body)
        from nebula.activity import Watch
        att = self.a.ledger.db.execute("SELECT attempt_id FROM attempts ORDER BY rowid DESC LIMIT 1").fetchone()[0]
        self.assertEqual(Watch._callers(self.a.ledger.db, [att]), {att: f"gravity:{b_id}"})

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


class GrantLimitsTest(unittest.TestCase):
    def test_calls_at_once_and_an_hour_and_the_reply_cap(self):
        from nebula import gravity
        lim = {"calls_per_hour": 3, "max_concurrent": 1}
        self.assertIsNone(gravity.admit_call("p1", "m", lim, now=1000.0))
        self.assertIn("1 calls at once", gravity.admit_call("p1", "m", lim, now=1001.0))   # one running
        gravity.call_done("p1", "m")
        self.assertIsNone(gravity.admit_call("p1", "m", lim, now=1002.0)); gravity.call_done("p1", "m")
        self.assertIsNone(gravity.admit_call("p1", "m", lim, now=1003.0)); gravity.call_done("p1", "m")
        self.assertIn("3 calls an hour", gravity.admit_call("p1", "m", lim, now=1004.0))
        self.assertIsNone(gravity.admit_call("p1", "m", lim, now=1000.0 + 3601)); gravity.call_done("p1", "m")   # an hour on
        self.assertIsNone(gravity.admit_call("p2", "m", lim, now=1004.0)); gravity.call_done("p2", "m")         # each peer its own
        for path, body, want in (("/v1/chat/completions", {}, {"max_tokens": 100}), ("/v1/chat/completions", {"max_tokens": 5000}, {"max_tokens": 100}),
                                 ("/v1/chat/completions", {"max_tokens": 20}, {"max_tokens": 20}), ("/v1/responses", {}, {"max_output_tokens": 100}),
                                 ("/v1/chat/completions", {"max_completion_tokens": 900}, {"max_completion_tokens": 100})):
            gravity.cap_reply(path, body, 100)
            self.assertEqual(body, want)


class InferPathTest(unittest.TestCase):
    """A peer's INFER call goes to loopback with the internal token: only the model API may be its path."""

    def test_only_the_model_api(self):
        from nebula.gravity import infer_path_ok
        for ok in ("/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/completions", "/v1/embeddings"):
            self.assertTrue(infer_path_ok(ok), ok)
        for bad in ("/api/config", "/api/yard/node", "/api/signin/open", "/v1/../api/config", "/v1//api", "/v1/x?y=1",
                    "/v1/", "v1/chat/completions", "/V1/chat", "", None, 3, "/v1/chat%2f..%2fapi"):
            self.assertFalse(infer_path_ok(bad), bad)


if __name__ == "__main__":
    unittest.main()
