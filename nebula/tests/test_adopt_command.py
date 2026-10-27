"""A command engine on a model machine (the reference engine: an image and a command, no "model") is taken back after Nebula restarts
while its sandbox kept running - before, adopt_sandboxes read s["model"] and skipped it silently, leaving it "stopped"
while it ran on the test server (2026-10-25, the YaRN gate found it)."""
import json
import unittest
from unittest import mock

from nebula import engines as E
from nebula import machines, openshell


class Cfg:
    base_dir = None

    def __init__(self, engines):
        self.engines = engines


REF_ENGINE = {"name": "FlashNext", "kind": "command", "machine": "Server", "image": "nebula-flashnext:impulse-36f62a1-r768",
            "folder": "/var/lib/models", "port": 18123, "card": "GPU-a,GPU-b",
            "command": ["/opt/impulse/bin/impulse-fx", "-m", "/models/x", "--port", "18123"], "host": "127.0.0.1", "where": "openshell"}


class TestAdoptCommand(unittest.TestCase):
    def test_command_engine_taken_back(self):
        en = E.Engines(Cfg([dict(REF_ENGINE)]))
        en.network = object()
        services = {"services": [{"sandbox": openshell.sandbox_name("FlashNext"), "url": "http://default--nebeng-reference.openshell.localhost:17680/"}]}
        row = {"name": "Server", "gateway": "nebula-server", "gateway_port": 17680, "address": "192.0.2.10", "host": "server"}
        with mock.patch.object(E, "check_spec", side_effect=lambda s: s), \
             mock.patch.object(openshell, "_json", side_effect=Exception("no local gateway")), \
             mock.patch.object(machines, "row", return_value=row), \
             mock.patch.object(machines, "tunnel", return_value=17680), \
             mock.patch.object(machines, "target", return_value=None), \
             mock.patch.object(machines, "router_url", return_value="http://127.0.0.1:9/router"), \
             mock.patch.object(openshell, "run", return_value=json.dumps(services)):
            took = en.adopt_sandboxes()
        self.assertEqual(took, ["FlashNext"])
        p = en.procs["FlashNext"]
        self.assertEqual(p.state, "ready")
        self.assertEqual(p.models_dir, REF_ENGINE["folder"])
        self.assertEqual(p.image, REF_ENGINE["image"])


if __name__ == "__main__":
    unittest.main()
