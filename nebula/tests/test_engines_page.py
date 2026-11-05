"""The Engines page's pieces (2026-11-05): Build an engine takes the owner's name for it (checked, not taken) and offers
Impulse (L) there; each machine lists its sandboxes, read per OpenShell gateway in the background."""
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nebula import openshell, wizards as W


class NameTest(unittest.TestCase):
    def server(self):
        return SimpleNamespace(cfg=SimpleNamespace(tiers=[SimpleNamespace(name="writer")], engines=[{"name": "big"}]))

    def test_the_name_given_or_a_free_one(self):
        srv = self.server()
        self.assertEqual(W._wanted_name(srv, {"name": "My Coder"}, "x"), "my-coder")
        self.assertEqual(W._wanted_name(srv, {}, "writer"), "writer-2")
        for taken in ("writer", "big", "Big"):
            with self.assertRaises(W.WizardError):
                W._wanted_name(srv, {"name": taken}, "x")
        with self.assertRaises(W.WizardError):
            W._wanted_name(srv, {"name": "!!!"}, "x")

    def test_build_an_engine_has_a_name_and_offers_impulse_l(self):
        fields = [f["key"] for st in W.DeployOwn.steps for f in st["fields"]]
        self.assertIn("name", fields)
        self.assertIn("legacy", fields)
        srv = SimpleNamespace(cfg=SimpleNamespace(base_dir="/tmp"))
        with mock.patch("nebula.engines.legacy_status", return_value=None), \
                mock.patch("nebula.engines.legacy_optional", return_value={"here": False, "optional": True, "offer": "downloads Impulse (L) (470 MB): why", "why": None}):
            o = W.DeployOwn().options(srv, "legacy", {})
        self.assertEqual((o[0]["post"], o[0]["button"]), ("/api/yard/pieces/impulse-legacy", "Download Impulse (L)"))
        with mock.patch("nebula.engines.legacy_status", return_value=None), \
                mock.patch("nebula.engines.legacy_optional", return_value={"here": True, "optional": True, "offer": None}):
            self.assertEqual(W.DeployOwn().options(srv, "legacy", {}), [])     # here already: nothing to offer


class SandboxesTest(unittest.TestCase):
    def test_each_gateway_read_in_the_background(self):
        calls = []

        def fake_json(args, exe=None, gateway=None, timeout=120):
            calls.append((tuple(args), gateway))
            if args[:2] == ["gateway", "list"]:
                return [{"name": "openshell"}, {"name": "nebula-rig2"}]
            if gateway == "nebula-rig2":
                raise openshell.OpenShellError("ssh: no route to host")
            return [{"name": "nebula-box-1", "phase": "Error", "message": "pull failed", "secret": "x"}]
        openshell._BY_GATEWAY.update(at=0.0, gateways={}, busy=False)
        with mock.patch.object(openshell, "_json", side_effect=fake_json), mock.patch.object(openshell, "binary", return_value="/usr/bin/openshell"):
            first = openshell.by_gateway()
            self.assertEqual(first["gateways"], {})                 # no wait: the first look starts the reading
            for _ in range(100):
                if openshell._BY_GATEWAY["at"]:
                    break
                time.sleep(0.02)
            v = openshell.by_gateway()
        self.assertEqual(v["gateways"]["openshell"]["sandboxes"], [{"name": "nebula-box-1", "phase": "Error", "status": None, "message": "pull failed", "reason": None}])
        self.assertIn("no route", v["gateways"]["nebula-rig2"]["error"])
        self.assertIn((("sandbox", "list"), "nebula-rig2"), calls)


if __name__ == "__main__":
    unittest.main()
