"""The Home stage's caller stars and the reply's bursts (nebula/ui/index.html, quasarCalls): a call's caller is a star of
its own only when it is a registered caller (not the API's doors, not an entry point's own calls, not Nebula's own work:
Halo has its own star); the reply goes back down its stream as 3 to 5 bursts, more for a longer reply; the last one
blows the star apart and it re-forms. A failed call bounces. Drawing only."""
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


class QuasarBurstsTest(unittest.TestCase):
    def test_the_page_wires_it(self):
        html = PAGE.read_text()
        self.assertIn("if (isRealCaller(a.caller)) live.callers.set(a.caller", html)     # remembered from the feed
        self.assertIn('const nb = ensureBody(k, "src"); nb.input = seen.dialect;', html)   # its star, into its Quasar
        self.assertIn("laserThrough(q, src)", html)                                      # the last burst: a laser through it
        self.assertIn(".laser { position:absolute", html)
        self.assertIn('quasarFx(q, "bounce"); burst(srcEl, colour); return;', html)    # a failed call: no destroy
        self.assertIn("@media (prefers-reduced-motion: reduce) { .sspark { display:none; }", html)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_which_callers_are_stars_and_how_many_bursts(self):
        js = source() + """
console.log(JSON.stringify({
  stars: ["writer", "my-agent", "entry-lan-1", "chat", "messages", "responses", "halo", "wizard", "lookup", "", null, undefined, 7]
    .map(isRealCaller),
  bursts: [0, 5, 40, 400, 4000, 400000, undefined, -3].map(replyBursts) }));"""
        r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(out["stars"], [True, True] + [False] * 11)
        self.assertEqual(out["bursts"], [3, 3, 4, 5, 5, 5, 3, 3])


if __name__ == "__main__":
    unittest.main()
