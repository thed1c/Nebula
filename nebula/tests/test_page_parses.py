"""The page's own scripts parse as a whole (node --check). The Python tests read pieces of nebula/ui/index.html, so a
script that no longer parses (two `const routeOf` in one scope after a merge, 2026-11-04) passed every one of them while
the window would have opened blank."""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "ui" / "index.html"


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class PageParsesTest(unittest.TestCase):
    def test_every_inline_script_parses(self):
        src = PAGE.read_text(encoding="utf-8")
        scripts = [m.group(1) for m in re.finditer(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", src, re.S)]
        self.assertTrue(scripts, "no inline script in the page")
        with tempfile.TemporaryDirectory() as d:
            for n, body in enumerate(scripts):
                f = Path(d) / f"script-{n}.js"
                f.write_text(body, encoding="utf-8")
                r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 0, f"script {n} of the page does not parse:\n{r.stderr[-2000:]}")


if __name__ == "__main__":
    unittest.main()
