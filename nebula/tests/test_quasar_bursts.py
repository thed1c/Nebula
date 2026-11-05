"""Calls on the Home map (nebula/ui/index.html): callers are not objects on the map. An API call is a flash through its
Quasar: its Quasar's strands pulse in, its model's strands pulse out, and the answer comes back the same way; a failed
call bounces at the Quasar. A model that talks (streams) pulses its star. Nebula's own work (Halo, the wizard, Lookup)
has no Quasar. A registered caller is still remembered for the Apps page. Drawing only."""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "ui" / "index.html"


def source() -> str:
    m = re.search(r"// quasarCalls:begin\n(.*?)\n\s*// quasarCalls:end", PAGE.read_text(), re.S)
    if not m:
        raise AssertionError("quasarCalls is not marked in index.html")
    return m.group(1)


class QuasarCallsTest(unittest.TestCase):
    def test_the_page_wires_it(self):
        html = PAGE.read_text()
        self.assertIn("if (isRealCaller(a.caller)) live.callers.set(a.caller", html)     # remembered for the Apps page
        body = html[html.index("function syncSources()"):]
        body = body[:body.index("\n  }\n")]
        self.assertIn('if (b.kind === "src" && !b.leaving) swallow(b);', body)          # no caller stars on the map
        self.assertNotIn('ensureBody(k, "src")', html)
        self.assertIn("function callQuasar(a)", html)
        self.assertIn('if (q) { quasarFx(q, "infall"); onward(streamOf(q.key, "in")); }', html)
        self.assertIn('quasarFx(q, a.state && a.state !== "completed" ? "bounce" : "jetflash")', html)   # a failed call
        self.assertIn("live.flights.add({ pulse: true, key: own.key, dir: own.dir", html)  # the strands pulse
        self.assertIn("talk(modelEl(a.tier));", html)                                      # a talking model's star
        self.assertIn("@media (prefers-reduced-motion: reduce) { .body.talk .core", html)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_which_callers_are_apps(self):
        js = source() + """
console.log(JSON.stringify(["writer", "my-agent", "entry-lan-1", "chat", "messages", "responses", "halo", "wizard",
  "lookup", "", null, undefined, 7, "gravity:3f9a"].map(isRealCaller)));"""
        r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), [True, True] + [False] * 12)     # a friend's call is not an app

    def test_gravity_calls_run_the_wormhole(self):
        html = PAGE.read_text()
        self.assertIn("function wormPath(id, dir)", html)
        self.assertIn("if (gravityPeerOf(a.caller)) return null;", html)              # a friend's call has no Quasar
        self.assertIn('else if (w) { flashHole(peerRouterEl(from), GRAVITY_PINK); onward(w, GRAVITY_PINK); }', html)
        self.assertIn('if (to) light(wormPath(to, "out")', html)                     # out to a friend's model
        self.assertIn("const w = from && wormPath(from, \"out\");", html)           # the answer back to the friend
        self.assertNotIn("chained · not shared with you", html)
        self.assertNotIn("shared with you · you share", html)


if __name__ == "__main__":
    unittest.main()
