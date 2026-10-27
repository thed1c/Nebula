"""Step A: the registry and the ticks. The ladder is what an agent, a hook or the default list ticks; nothing
lands outside it; a ticked model that is down is skipped, and none up is a wait the request does not get yet."""
import json
import unittest
import urllib.request

from nebula.config import Agent, Config, Tier
from nebula.replay import verify
from nebula.tests.test_escalation import Rig
from nebula.tests.test_gateway import call
from nebula.tests.test_hooks import post

AGENTS = [Agent("ticked", system="Ticked.", models=["think", "big"]),
          Agent("bigonly", system="Big only.", models=["big"]),
          Agent("short", system="Short form.", tier="think", ceiling="think")]


class TicksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig(agents=AGENTS, hooks=[{"name": "h", "agent": "short", "models": ["big"]}])

    @classmethod
    def tearDownClass(cls):
        cls.rig.stop()

    def setUp(self):
        self.rig.a.requests.clear()
        self.rig.b.requests.clear()

    def test_the_ticks_are_the_ladder(self):
        rig = self.rig
        a1 = rig.chat([{"role": "user", "content": "hi"}], model="cockpit/ticked")
        self.assertEqual((len(rig.a.requests), len(rig.b.requests)), (1, 0))
        self.assertEqual(rig.a.requests[-1]["chat_template_kwargs"], {"enable_thinking": True})   # ticks[0] is think, not fast
        self.assertEqual(rig.attempts()[-1]["tier"], "think")
        d = json.loads(rig.decisions()[-1]["inputs"])
        self.assertEqual((d["ticks"], d["usable"], d["floor"]), (["think", "big"], ["think", "big"], "think"))
        hist = [{"role": "user", "content": "hi"}, a1, {"role": "user", "content": "more"}]
        rig.chat(hist, model="cockpit/ticked", headers={"X-Cockpit-Escalate": "1"})                 # one up inside the ticks
        self.assertEqual(rig.attempts()[-1]["tier"], "big")
        self.assertEqual(len(rig.b.requests), 1)
        ok, lines = verify(rig.path)
        self.assertTrue(ok, "\n".join(lines))

    def test_a_model_named_outside_the_ticks_is_not_used(self):
        rig = self.rig
        rig.chat([{"role": "user", "content": "outside"}], model="fast", headers={"X-Cockpit-Agent": "ticked"})
        att, dec = rig.attempts()[-1], rig.decisions()[-1]
        self.assertEqual(att["tier"], "think")
        self.assertIn("outside_ticks:fast", dec["reason"])
        rig.chat([{"role": "user", "content": "outside"}], model="cockpit/ticked", headers={"X-Cockpit-Escalate": "fast"})
        dec = rig.decisions()[-1]
        self.assertIn("outside_ticks:fast", dec["reason"])
        self.assertIn("escalate:1", dec["reason"])

    def test_a_hooks_ticks_win_over_the_agents(self):
        rig = self.rig
        status, js = post(rig.base, "/hooks/h", b'{"x": 1}')
        self.assertEqual(status, 200, js)
        self.assertEqual((len(rig.b.requests), rig.attempts()[-1]["tier"], rig.attempts()[-1]["agent"]), (1, "big", "short"))
        self.assertEqual(json.loads(rig.decisions()[-1]["inputs"])["ticks"], ["big"])

    def test_short_form_still_works(self):
        rig = self.rig
        rig.chat([{"role": "user", "content": "short"}], model="cockpit/short", headers={"X-Cockpit-Escalate": "big"})
        self.assertEqual((rig.attempts()[-1]["tier"], rig.decisions()[-1]["reason"]), ("think", "escalate:big,ceiling:think"))

    def test_registry_nodes(self):
        nodes = json.loads(urllib.request.urlopen(self.rig.base + "/api/registry", timeout=5).read())
        self.assertEqual([(n["name"], n["up"], n["slots"], n["context"], n["vision"], n["model"], n["power_class"]) for n in nodes],
                         [("fast", True, 1, 262144, True, "fastfake", "local"), ("think", True, 1, 262144, True, "fastfake", "local"),
                          ("big", True, 1, 262144, True, "bigfake", "local")])


class DownTest(unittest.TestCase):
    def test_a_down_ticked_model_is_skipped_and_none_up_waits(self):
        rig = Rig(agents=AGENTS)
        try:
            rig.b.stop()
            rig.srv.registry.probe_every = 0.0
            nodes = json.loads(urllib.request.urlopen(rig.base + "/api/registry", timeout=5).read())
            self.assertEqual([n["up"] for n in nodes], [True, True, False])
            rig.chat([{"role": "user", "content": "hi"}], model="cockpit/ticked")             # big is down: think answers
            self.assertEqual(rig.attempts()[-1]["tier"], "think")
            self.assertEqual(json.loads(rig.decisions()[-1]["inputs"])["usable"], ["think"])
            status, _, body = call(rig.base, "/v1/chat/completions", {"model": "cockpit/bigonly", "messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(status, 503)
            self.assertIn("waiting for a ticked model", json.loads(body)["error"]["message"])
            att, dec = rig.attempts()[-1], rig.decisions()[-1]
            self.assertEqual((att["state"], dec["reason"], dec["tier"]), ("failed", "waiting:no_usable_ticked_model", "big"))
            self.assertIn("waiting for a ticked model", att["error"])
            ok, lines = verify(rig.path)
            self.assertTrue(ok, "\n".join(lines))
        finally:
            rig.stop()


class ConfigTest(unittest.TestCase):
    def test_ticks_must_be_tiers(self):
        base = {"tiers": [{"name": "fast", "engine": "http://x"}]}
        with self.assertRaises(ValueError):
            Config.from_dict({**base, "agents": [{"name": "a", "models": ["nope"]}]})
        with self.assertRaises(ValueError):
            Config.from_dict({**base, "default_models": ["fast", "fast"]})
        with self.assertRaises(ValueError):
            Config.from_dict({**base, "hooks": [{"name": "h", "models": ["nope"]}]})
        cfg = Config.from_dict({**base, "default_models": ["fast"], "agents": [{"name": "a", "models": ["fast"]}]})
        self.assertEqual((cfg.default_models, cfg.agents[0].models), (["fast"], ["fast"]))


if __name__ == "__main__":
    unittest.main()
