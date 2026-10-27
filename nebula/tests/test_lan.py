"""This machine on the network: loopback, and adopting engines already running here (step G1). LAN discovery and
PIN pairing (step G3) come in a later release."""
import json
import unittest
import urllib.request

from nebula.lan import is_loopback, probe_local_engines
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_escalation import Rig


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return json.loads(r.read())


class LoopbackTest(unittest.TestCase):
    def test_is_loopback(self):
        self.assertTrue(is_loopback("127.0.0.1") and is_loopback("::1") and not is_loopback("192.0.2.9"))


class AdoptTest(unittest.TestCase):
    def test_probe_finds_a_running_engine(self):
        fake = FakeEngine("found")
        try:
            got = probe_local_engines(ports=[(fake.server_port, "llama"), (1, "none")])
            self.assertEqual([(e["kind"], e["url"], e["model"]) for e in got], [("llama-server", fake.url, "found")])
            rig = Rig()
            try:
                import nebula.lan as lanmod
                old = lanmod.LOCAL_ENGINE_PORTS
                lanmod.LOCAL_ENGINE_PORTS = [(fake.server_port, "llama")]
                try:
                    found = get(rig.base, "/api/lan/engines")
                finally:
                    lanmod.LOCAL_ENGINE_PORTS = old
                self.assertEqual((found[0]["url"], found[0]["added"]), (fake.url, False))
            finally:
                rig.stop()
        finally:
            fake.stop()


if __name__ == "__main__":
    unittest.main()
