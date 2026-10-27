"""The page's models API: endpoints to paste elsewhere, a provider's model list, adding and removing a model
(saved to the config file, the key in a 0600 file), and the policy ticks."""
import json
import os
import stat
import tempfile
import unittest
import urllib.request
from pathlib import Path

from nebula.config import Agent, Config
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_escalation import Rig
from nebula.tests.test_gateway import call
from nebula.tests.test_ui import req as ui_req


class ModelsApiTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig(agents=[Agent("writer", system="You write.", tier="fast")])
        self.cfgdir = tempfile.TemporaryDirectory()
        cfg = self.rig.srv.cfg
        cfg.path = str(Path(self.cfgdir.name) / "nebula.json")
        cfg.base_dir = self.cfgdir.name
        cfg.raw = {"port": 0, "_note": "kept as it was"}
        self.provider = FakeEngine("cloudfake")

    def tearDown(self):
        self.provider.stop()
        self.rig.stop()
        self.cfgdir.cleanup()

    def test_endpoints_use_the_host_the_request_came_by(self):
        r = urllib.request.Request(self.rig.base + "/api/endpoints", headers={"Host": "192.168.1.20:8090"})
        ep = json.loads(urllib.request.urlopen(r, timeout=5).read())
        self.assertEqual((ep["server_url"], ep["openai_base"], ep["anthropic_base"]), ("http://192.168.1.20:8090", "http://192.168.1.20:8090/v1", "http://192.168.1.20:8090"))
        self.assertEqual([m["model"] for m in ep["models"]], ["nebula/fast", "nebula/think", "nebula/big"])
        self.assertEqual(ep["agents"][0]["model"], "nebula/writer")
        self.assertIn("nebula/fast", ep["models"][0]["curl"])

    def test_list_add_use_and_remove_a_provider(self):
        status, _h, raw = ui_req(self.rig.base, "/api/providers/models", {"base_url": self.provider.url + "/v1", "api_key": "k-1"})
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["models"], ["cloudfake"])
        status, _h, raw = ui_req(self.rig.base, "/api/models", {"name": "cloud", "kind": "provider", "url": self.provider.url + "/v1",
                                                                "model": "cloudfake", "api_key": "k-1", "power_class": "cloud", "cost_per_mtoken": 2.5, "context": 200000})
        js = json.loads(raw)
        self.assertEqual(status, 200, raw)
        self.assertEqual((js["added"], js["node"]["up"], js["node"]["model"], js["node"]["power_class"], js["node"]["context"]), ("cloud", True, "cloudfake", "cloud", 200000))
        saved = json.load(open(js["saved"]))
        self.assertEqual(saved["_note"], "kept as it was")
        tier = [t for t in saved["tiers"] if t["name"] == "cloud"][0]
        self.assertEqual((tier["provider"], tier["base_url"], tier["model"], tier["cost_per_mtoken"]), ("openai-compatible", self.provider.url + "/v1", "cloudfake", 2.5))
        self.assertNotIn("api_key", tier)
        self.assertTrue(tier["api_key_file"].endswith(os.path.join("nebula-keys", "cloud.key")))
        if os.name != "nt":                                   # Windows: the user's profile is the wall, not a mode
            self.assertEqual(stat.S_IMODE(os.stat(tier["api_key_file"]).st_mode), 0o600)
        self.assertEqual(open(tier["api_key_file"]).read().strip(), "k-1")
        nodes = json.loads(ui_req(self.rig.base, "/api/registry")[2])
        self.assertEqual([n["name"] for n in nodes], ["fast", "think", "big", "cloud"])
        # a turn to the new tier goes to the provider with the key and its model id
        status, _, body = call(self.rig.base, "/v1/chat/completions", {"model": "cockpit/cloud", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, body)
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(self.provider.requests[0]["model"], "cloudfake")
        self.assertEqual(self.provider.headers_seen[-1].get("Authorization"), "Bearer k-1")
        self.assertEqual(self.rig.attempts()[-1]["tier"], "cloud")
        # ticked by an agent: cannot be removed until unticked
        self.assertEqual(ui_req(self.rig.base, "/api/policy", {"agent": "writer", "models": ["cloud", "fast"]})[0], 200)
        self.assertEqual(json.load(open(js["saved"]))["agents"][0]["models"], ["cloud", "fast"])
        call(self.rig.base, "/v1/chat/completions", {"model": "cockpit/writer", "messages": [{"role": "user", "content": "to the cloud first"}]})
        self.assertEqual(self.rig.attempts()[-1]["tier"], "cloud")
        self.assertEqual(ui_req(self.rig.base, "/api/models/cloud/remove", {})[0], 409)
        ui_req(self.rig.base, "/api/policy", {"agent": "writer", "models": []})
        status, _h, raw = ui_req(self.rig.base, "/api/models/cloud/remove", {})
        self.assertEqual(status, 200, raw)
        self.assertEqual([t["name"] for t in json.load(open(js["saved"]))["tiers"]], ["fast", "think", "big"])
        self.assertEqual(len(json.loads(ui_req(self.rig.base, "/api/registry")[2])), 3)

    def test_refusals(self):
        self.assertEqual(ui_req(self.rig.base, "/api/models", {"name": "fast", "kind": "engine", "url": self.provider.url})[0], 409)
        self.assertEqual(ui_req(self.rig.base, "/api/models", {"name": "bad name", "kind": "engine", "url": self.provider.url})[0], 400)
        self.assertEqual(ui_req(self.rig.base, "/api/models", {"name": "dead", "kind": "engine", "url": "http://127.0.0.1:9"})[0], 502)
        self.assertEqual(ui_req(self.rig.base, "/api/models", {"name": "nomodel", "kind": "provider", "url": self.provider.url + "/v1"})[0], 400)
        self.assertEqual(ui_req(self.rig.base, "/api/models", {"name": "wrong", "kind": "provider", "url": self.provider.url + "/v1", "model": "gpt-x"})[0], 400)
        self.assertEqual(ui_req(self.rig.base, "/api/policy", {"models": ["nope"]})[0], 400)
        self.assertEqual(ui_req(self.rig.base, "/api/policy", {"agent": "nobody", "models": ["fast"]})[0], 404)
        self.assertEqual(ui_req(self.rig.base, "/api/models/fast/remove", {})[0], 409)        # the writer's floor names it
        # the default list, saved
        self.assertEqual(ui_req(self.rig.base, "/api/policy", {"models": ["think"]})[0], 200)
        self.assertEqual(json.load(open(self.rig.srv.cfg.path))["default_models"], ["think"])
        call(self.rig.base, "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(self.rig.attempts()[-1]["tier"], "think")

    def test_an_engine_added_by_url(self):
        status, _h, raw = ui_req(self.rig.base, "/api/models", {"name": "second", "kind": "engine", "url": self.provider.url, "power_class": "second-box"})
        self.assertEqual(status, 200, raw)
        js = json.loads(raw)
        self.assertEqual((js["node"]["slots"], js["node"]["context"], js["node"]["vision"], js["node"]["model"]), (1, 262144, True, "cloudfake"))
        tier = [t for t in json.load(open(js["saved"]))["tiers"] if t["name"] == "second"][0]
        self.assertEqual((tier["engine"], tier["power_class"]), (self.provider.url, "second-box"))
        self.assertNotIn("api_key_file", tier)


class ConfigRoundTripTest(unittest.TestCase):
    def test_save_keeps_unknown_sections_and_drops_defaults(self):
        d = {"port": 1, "_note": "x", "tiers": [{"name": "a", "engine": "http://a"}], "agents": [{"name": "w", "system": "s", "tools": ["t"]}],
             "mcp_servers": {"t": {"command": "x"}}, "default_models": ["a"]}
        cfg = Config.from_dict(d)
        out = cfg.to_dict()
        self.assertEqual(out["_note"], "x")
        self.assertEqual(out["tiers"], [{"name": "a", "engine": "http://a"}])
        self.assertEqual(out["agents"], [{"name": "w", "system": "s", "tools": ["t"]}])
        self.assertEqual(out["default_models"], ["a"])
        cfg.default_models = []
        self.assertNotIn("default_models", cfg.to_dict())
        with self.assertRaises(ValueError):
            cfg.save()


if __name__ == "__main__":
    unittest.main()
