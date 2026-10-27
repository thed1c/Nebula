"""A sandboxed engine that was ready and whose sandbox later died (its program killed: OpenShell's phase Error) is
shown "failed" with the reason, instead of "ready" for ever (seen 2026-10-26: two engines' sandboxes went to Error
at night and the page kept saying ready). Checked at most every SANDBOX_RECHECK_S, in the background."""
import time
import unittest
from unittest import mock

from nebula import engines as E


class Cfg:
    base_dir, path, raw = None, None, {}

    def __init__(self, engines):
        self.engines = engines
        self.tiers = []


class Boxed:
    """openshell.SandboxedEngine's face, with exited() answering what the test says."""

    def __init__(self, why=None):
        self.state, self.error, self.proc, self.why, self.asked, self.lines = "ready", None, None, why, 0, []

    def exited(self):
        self.asked += 1
        return self.why

    def _say(self, line):
        self.lines.append(line)

    def status(self):
        return {"state": self.state, "lines": self.lines, "pid": None, "error": self.error, "seconds": None}


SPEC = {"name": "critic", "kind": "llama.cpp", "model": "/m/x.gguf", "port": 18300}


def settle(p, want, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end and p.state != want:
        time.sleep(0.02)


class RecheckTest(unittest.TestCase):
    def engines(self, p):
        en = E.Engines(Cfg([dict(SPEC)]))
        en.procs["critic"] = p
        en.SANDBOX_RECHECK_S = 0.0
        return en

    def test_a_dead_sandbox_is_failed_with_the_reason(self):
        p = Boxed("its sandbox is Error")
        en = self.engines(p)
        changed = []
        en.on_change = lambda: changed.append(1)
        en.state("critic")                                   # starts the check
        settle(p, "failed")
        end = time.time() + 3
        while not changed and time.time() < end:            # the page is told right after the state flips
            time.sleep(0.02)
        self.assertEqual(en.state("critic"), "failed")
        self.assertIn("its sandbox is Error", p.error)
        self.assertTrue(changed)

    def test_a_running_sandbox_stays_ready(self):
        p = Boxed(None)
        en = self.engines(p)
        en.state("critic")
        time.sleep(0.2)
        self.assertEqual(en.state("critic"), "ready")
        self.assertGreaterEqual(p.asked, 1)

    def test_asked_at_most_every_interval(self):
        p = Boxed(None)
        en = self.engines(p)
        en.SANDBOX_RECHECK_S = 60.0
        for _ in range(5):
            en.state("critic")
        time.sleep(0.2)
        self.assertEqual(p.asked, 1)

    def test_a_failing_check_says_nothing(self):
        p = Boxed()
        p.exited = mock.Mock(side_effect=RuntimeError("the gateway is down"))
        en = self.engines(p)
        en.state("critic")
        time.sleep(0.2)
        self.assertEqual(en.state("critic"), "ready")


if __name__ == "__main__":
    unittest.main()
