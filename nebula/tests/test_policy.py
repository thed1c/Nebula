"""This Nebula's own routing policy (nebula/policy.py): what the Configure panels set, enforced by its router."""
import json
import time
import unittest
import urllib.error
import urllib.request

from nebula.config import Tier
from nebula.policy import Policy, PolicyError, check
from nebula.tests.test_escalation import Rig


def at(day: int, hhmm: str) -> float:
    """A local time on a weekday of a fixed week (day 0 = Monday, 2026-10-12)."""
    h, m = map(int, hhmm.split(":"))
    return time.mktime((2026, 10, 12 + day, h, m, 0, 0, 0, -1))


class PolicyTest(unittest.TestCase):
    def test_checked(self):
        self.assertEqual(check({"models": {"a": {"enabled": False, "priority": 2, "junk_none": None}}}),
                         {"models": {"a": {"enabled": False, "priority": 2}}, "advertise": {}})
        for bad in ({"models": {"a": {"priority": 0}}}, {"models": {"a": {"enabled": "no"}}}, {"models": {"a": {"color": 1}}},
                    {"models": {"a": {"hours": {"from": "8am"}}}}, {"models": {"a": {"limits": {"max_tokens": 5}}}},
                    {"advertise": {"u": "a"}}, []):
            with self.assertRaises(PolicyError):
                check(bad)

    def test_off_hours_and_hourly_count(self):
        p = Policy()
        p.put({"models": {"a": {"enabled": False}, "b": {"hours": {"days": [0, 1, 2, 3, 4], "from": "08:00", "to": "18:00"}},
                          "c": {"hours": {"days": [4], "from": "22:00", "to": "02:00"}}, "d": {"limits": {"calls_per_hour": 2}}}})
        self.assertEqual(p.allowed("a"), (False, "disabled"))
        self.assertTrue(p.allowed("b", at(0, "09:30"))[0])
        self.assertEqual(p.allowed("b", at(0, "19:00")), (False, "outside its hours"))
        self.assertFalse(p.allowed("b", at(5, "09:30"))[0])                 # Saturday
        self.assertTrue(p.allowed("c", at(4, "23:00"))[0])                  # Friday night
        self.assertTrue(p.allowed("c", at(5, "01:00"))[0])                  # past midnight, Friday's window
        self.assertFalse(p.allowed("c", at(6, "01:00"))[0])                 # Sunday 01:00 belongs to Saturday: no
        now = at(2, "12:00")
        p.note_call("d", now - 100); p.note_call("d", now - 50)
        self.assertEqual(p.allowed("d", now), (False, "its hourly limit (2)"))
        self.assertTrue(p.allowed("d", now + 3600)[0])

    def test_order_capacity_vision(self):
        p = Policy()
        p.put({"models": {"b": {"priority": 1}, "c": {"prefer": True, "priority": 9}, "a": {"slots": 2, "limits": {"max_concurrent": 1}},
                          "x": {"limits": {"max_concurrent": 3}}, "v": {"vision": False}}, "advertise": {"u1": ["a"], "u2": []}})
        tiers = [Tier(n) for n in ("a", "b", "c", "d")]
        self.assertEqual([t.name for t in p.order(tiers)], ["c", "b", "a", "d"])
        self.assertEqual((p.slots("a", 4), p.slots("x", 4), p.slots("x", 2), p.slots("y", 4)), (1, 3, 2, 4))
        self.assertEqual((p.vision("v", True), p.vision("w", True)), (False, True))


class EnforcedTest(unittest.TestCase):
    """Through the gateway, on fake engines: fast and think share one, big is the other."""
    def setUp(self):
        self.rig = Rig()

    def tearDown(self):
        self.rig.stop()

    def put(self, policy, base=None):
        req = urllib.request.Request((base or self.rig.base) + "/api/routing", data=json.dumps(policy).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def error(self, **kw):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.rig.chat([{"role": "user", "content": "hi"}], **kw)
        return cm.exception.code, json.loads(cm.exception.read())["error"]["message"]

    def test_disabled_and_named(self):
        self.put({"models": {"fast": {"enabled": False}}})
        self.rig.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(self.rig.decisions()[-1]["tier"], "think")          # the ladder starts past it
        code, msg = self.error(model="fast")
        self.assertEqual(code, 503); self.assertIn("fast is not to be used now: disabled", msg)
        self.assertIn("policy:disabled", [d["reason"] for d in self.rig.decisions()])

    def test_none_allowed(self):
        self.put({"models": {n: {"enabled": False} for n in ("fast", "think", "big")}})
        code, msg = self.error()
        self.assertEqual(code, 503); self.assertIn("no model may be used now", msg)

    def test_prefer_and_caps(self):
        view = self.put({"models": {"big": {"prefer": True, "thinking": False, "limits": {"max_output_tokens": 64}, "slots": 1}}})
        self.assertEqual({m["name"]: m["slots"] for m in view["models"]}["big"], 1)
        self.rig.chat([{"role": "user", "content": "hi"}], max_tokens=5000)
        self.assertEqual(self.rig.decisions()[-1]["tier"], "big")
        sent = self.rig.b.requests[-1]
        self.assertEqual(sent["max_tokens"], 64)
        self.assertEqual(sent["chat_template_kwargs"]["enable_thinking"], False)

    def test_a_bad_policy_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.put({"models": {"fast": {"priority": 42}}})
        self.assertEqual(cm.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
