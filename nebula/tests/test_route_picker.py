"""A model picker's option value (nebula/ui/index.html routeVal / routeOfVal) survives being written as HTML: it joined
tier and model with a NUL character, which the HTML parser turns into U+FFFD, so picking a model answered "no tier named
'x�'" (in every release since 0.1.3). routeVal and routeOfVal are run in node, the value passed through the page's
own esc() and an HTML attribute decode as the parser would."""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "ui" / "index.html"


def grab(src: str, name: str) -> str:
    m = re.search(r"^  const %s = .*$" % name, src, re.M)
    if not m:
        raise AssertionError(f"{name} is not in the page")
    return m.group(0)


class RoutePickerTest(unittest.TestCase):
    def test_no_nul_in_the_page(self):
        self.assertNotIn("\\u0000", PAGE.read_text(encoding="utf-8"))

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_value_survives_html(self):
        src = PAGE.read_text(encoding="utf-8")
        js = "\n".join(grab(src, n) for n in ("esc", "routeVal", "routeOfVal")) + r"""
const unhtml = (s) => s.replace(/\0/g, "�").replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
const out = [];
for (const c of [{ tier: "everyday", model: "lfm25-24b" }, { tier: "larger", model: "" }, { tier: "x", model: 'a"b<c>&d' }])
  out.push(routeOfVal(unhtml(esc(routeVal(c)))));
out.push(routeOfVal(""), routeVal(null));
console.log(JSON.stringify(out));
"""
        r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), [
            {"tier": "everyday", "model": "lfm25-24b"},
            {"tier": "larger", "model": None},
            {"tier": "x", "model": 'a"b<c>&d'},
            {}, ""])


if __name__ == "__main__":
    unittest.main()
