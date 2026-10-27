"""Halo (nebula/halo.py) and the guard (nebula/guard.py): a guest's holds (paused, limited, a model held back) checked
before the Switchyard; Halo waking on the security log, reading it, acting through its small tools on the Assistant's
model with the interactive class, every action logged with an undo that works; its tools refused to anyone else;
the owner never held; Halo off does nothing."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nebula import halo as H
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call


def serve(d: dict):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "nebula.json")
    Path(path).write_text(json.dumps({"port": 0, "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False,
                                      "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}, **d}))
    srv = make_server(Config.from_args(["--config", path]))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}", path


class Rig(unittest.TestCase):
    halo_on = True

    def setUp(self):
        self.small, self.big = FakeEngine("small"), FakeEngine("big")
        self.srv, self.base, self.path = serve({
            "tiers": [{"name": "small", "engine": self.small.url, "shared": True}, {"name": "big", "engine": self.big.url}],
            "callers": [{"id": "friend-app", "name": "A friend's app", "guest": True}],
            "assistant": {"route": {"tier": "big"}, "halo": {"on": self.halo_on, "settle_s": 0.2}}})
        st, made = call(self.base, "/api/yard/callers/friend-app/keys", {"label": "theirs"})
        self.key, self.key_id = made["key"], made["row"]["id"]

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        self.small.stop(); self.big.stop()

    def guest(self, model="small", key=None):
        return call(self.base, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}]},
                    {"Authorization": f"Bearer {key or self.key}"})[0]

    def wait(self, pred, s=20):
        end = time.time() + s
        while time.time() < end:
            if pred():
                return True
            time.sleep(0.1)
        return False


class GuardTest(Rig):
    halo_on = False

    def test_pause_limit_deny_and_their_undo(self):
        g = self.srv.guard
        self.assertEqual(self.guest(), 200)
        u = g.pause("caller:friend-app", 600)
        self.assertEqual(self.guest(), 403)
        self.assertEqual(self.srv.seclog.read(kinds=["guest.paused"])[0]["who"], "caller:friend-app")
        g.undo(u)
        self.assertEqual(self.guest(), 200)
        u = g.limit("caller:friend-app", 2, 600)
        self.assertEqual([self.guest() for _ in range(3)], [200, 200, 429])
        g.undo(u)
        u = g.deny("caller:friend-app", "small")
        self.assertNotEqual(self.guest(), 200)                                    # held back
        g.undo(u)
        self.assertEqual(self.guest(), 200)
        again = Config.from_args(["--config", self.path])                       # kept in the config
        self.assertIn("guard", again.raw)

    def test_the_owner_is_never_held(self):
        with self.assertRaises(ValueError):
            self.srv.guard.pause("owner", 600)
        self.assertIsNone(self.srv.guard.check("owner"))

    def test_halo_off_does_nothing(self):
        for _ in range(3):
            self.guest("big")
        time.sleep(1.0)
        self.assertEqual(self.big.requests, [])                                  # no model was asked
        self.assertEqual(call(self.base, "/api/halo")[1]["state"], "off")


class HaloTest(Rig):
    def script(self, body):
        names = [t["function"]["name"] for t in body.get("tools") or []]
        if "halo__pause_guest" not in names:
            return None
        msgs = body.get("messages") or []
        last = msgs[-1]
        if last.get("role") == "user":
            return [{"name": "security__read_log", "arguments": {"min_severity": "notice", "limit": 20}}]
        prev_calls = [c["function"]["name"] for m in msgs if m.get("role") == "assistant" for c in m.get("tool_calls") or []]
        if prev_calls[-1] == "security__read_log":
            assert "guest.refused" in last["content"]                           # it read the log
            return [{"name": "halo__pause_guest", "arguments": {"tenant": "caller:friend-app", "minutes": 30,
                                                                 "reason": "probing a model not shared with it"}}]
        return "Paused the friend's app for 30 minutes."

    def test_halo_sees_acts_and_undo_works(self):
        self.big.script = self.script
        for _ in range(3):
            self.assertNotEqual(self.guest("big"), 200)                          # not shared with guests: refused
        self.assertTrue(self.wait(lambda: call(self.base, "/api/halo")[1]["actions"]), call(self.base, "/api/halo")[1])
        v = call(self.base, "/api/halo")[1]
        self.assertEqual(v["state"], "orange")                                    # it took a step: orange (red: it asks you)
        a = v["actions"][0]
        self.assertEqual((a["tool"], a["args"]["tenant"], a["reason"]), ("pause_guest", "caller:friend-app", "probing a model not shared with it"))
        self.assertEqual(self.guest("small"), 403)                                # paused
        posted = self.big.requests[0]
        self.assertEqual(posted["messages"][0]["content"][:9], "You are H")     # Halo's prompt, on the Assistant's model
        rows = [json.loads(r[0]) for r in self.srv.ledger.db.execute("SELECT inputs FROM decisions WHERE advisory = 0")]
        d = next(x for x in rows if x.get("label") == "halo")
        self.assertEqual((d.get("label"), d.get("priority")), ("halo", "interactive"))   # the interactive class at the gates
        kinds = [e["kind"] for e in self.srv.seclog.read()]
        self.assertIn("halo.action", kinds)
        st, v = call(self.base, f"/api/halo/actions/{a['id']}/undo", {})
        self.assertEqual(st, 200, v)
        self.assertTrue(v["actions"][0]["undone"])
        self.assertEqual(self.guest("small"), 200)
        self.assertIn("halo.undo", [e["kind"] for e in self.srv.seclog.read()])
        self.assertEqual(call(self.base, f"/api/halo/actions/{a['id']}/undo", {})[0], 400)    # once
        self.assertEqual(call(self.base, "/api/halo/seen", {})[1]["state"], "glow")

    def test_its_tools_are_its_own(self):
        self.assertTrue(self.srv.tools.call("halo__pause_guest", {"tenant": "caller:friend-app", "minutes": 5, "reason": "x"}).startswith("error:"))
        self.assertIsNone(self.srv.guard.check("caller:friend-app"))
        self.assertNotIn("halo", self.srv.cfg.agent("assistant").tools)          # the Assistant does not get them
        tok = H.HALO_TURN.set(True)
        try:
            out = self.srv.tools.call("halo__pause_guest", {"tenant": "owner", "minutes": 5, "reason": "x"})
            self.assertTrue(out.startswith("error:"), out)                       # never the owner
            out = self.srv.tools.call("halo__revoke_key", {"key": f"caller:friend-app/{self.key_id}", "reason": "leaked"})
            self.assertTrue(out.startswith("revoked"), out)
        finally:
            H.HALO_TURN.reset(tok)
        self.assertEqual(self.guest(), 401 if self.srv.cfg.api.get("require_key") else 200)   # a revoked key names no caller
        a = self.srv.halo.actions[-1]
        self.srv.halo.undo(a["id"])
        from nebula import callers
        self.assertEqual(callers.by_key(self.srv.cfg, self.key)["id"], "friend-app")   # the key works again

    def test_nothing_new_stays_calm(self):
        self.assertEqual(self.srv.halo.check(), {"looked": 0})
        self.assertEqual(call(self.base, "/api/halo")[1]["state"], "glow")
        self.assertEqual(self.big.requests, [])


if __name__ == "__main__":
    unittest.main()
