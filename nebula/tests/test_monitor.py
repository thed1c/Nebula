"""The page's /api/monitor asks every engine at once: one that does not answer (a friend's away Nebula through the
Gravity bridge, 20 s) no longer holds the whole poll, and is not asked again while it is still being asked."""
import threading
import time
import unittest
from types import SimpleNamespace

from nebula import ui


class FakeEngine:
    def __init__(self, delay=0.0, status=200):
        self.delay, self.status, self.calls = delay, status, 0
        self.health_path = "/health"
        self.gate = threading.Event()

    def get(self, path):
        self.calls += 1
        if self.delay:
            self.gate.wait(self.delay)
        return self.status, "application/json", b"{}"

    def props(self):
        return {"total_slots": 2, "model_alias": "m"}

    def busy(self):
        return 0

    def n_ctx(self):
        return 4096


class MonitorTest(unittest.TestCase):
    def server(self, engines):
        tiers = [SimpleNamespace(name=n, address=f"http://{n}", model=n, managed=None) for n in engines]
        return SimpleNamespace(cfg=SimpleNamespace(tiers=tiers), engines=engines, telemetry=None, ledger=None, errors=None)

    def test_a_model_that_does_not_answer_does_not_hold_the_poll(self):
        slow, fast = FakeEngine(delay=10), FakeEngine()
        srv = self.server({"away": slow, "local": fast})
        t = time.monotonic()
        out = ui.monitor(srv)
        self.assertLess(time.monotonic() - t, ui.MONITOR_WAIT_S + 1.0)
        up = {e["tier"]: e["up"] for e in out["engines"]}
        self.assertEqual(up, {"away": False, "local": True})
        ui.monitor(srv)                                          # still being asked: not asked again
        self.assertEqual(slow.calls, 1)
        self.assertEqual(fast.calls, 2)
        slow.gate.set()
        time.sleep(0.2)
        self.assertTrue({e["tier"]: e["up"] for e in ui.monitor(srv)["engines"]}["away"])   # it answered: up again

    def test_a_friends_model_takes_gravitys_hello_instead_of_a_call(self):
        friend = FakeEngine(delay=10)
        srv = self.server({"friend-m": friend})
        srv.cfg.tiers[0].mesh, srv.cfg.tiers[0].engine, srv.cfg.tiers[0].model = "gravity-peer", "http://127.0.0.1:9/peer/abc", "m"
        srv.gravity = SimpleNamespace(seen={"abc": {"reachable": False, "offers": ["m"]}})
        t = time.monotonic()
        self.assertFalse(ui.monitor(srv)["engines"][0]["up"])                 # away: down at once, nothing dialled
        self.assertLess(time.monotonic() - t, 0.5)
        srv.gravity.seen["abc"]["reachable"] = True
        self.assertTrue(ui.monitor(srv)["engines"][0]["up"])                  # its hello answered: up
        self.assertEqual(friend.calls, 0)


if __name__ == "__main__":
    unittest.main()
