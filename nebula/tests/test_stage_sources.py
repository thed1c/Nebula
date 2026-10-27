"""The Home stage's blue stars (nebula/ui/index.html isStageSource): the API's doors (OpenAI chat, Anthropic messages,
Responses) and Nebula's own work (Halo, the wizard, Lookup) are not stars; the page's room, webhooks and schedules
are. Only the drawing changes: routing and the ledger still see every input."""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "ui" / "index.html"


def source() -> str:
    m = re.search(r"// isStageSource:begin\n(.*?)\n\s*// isStageSource:end", PAGE.read_text(), re.S)
    if not m:
        raise AssertionError("isStageSource is not marked in index.html")
    return m.group(1)


class StageSourcesTest(unittest.TestCase):
    def test_sync_sources_asks_it(self):
        html = PAGE.read_text()
        body = html[html.index("function syncSources()"):]
        body = body[:body.index("\n  }\n")]
        self.assertIn("if (!isStageSource(key)) continue;", body)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_which_inputs_are_stars(self):
        keys = ["chat", "messages", "responses", "halo", "wizard", "lookup", "room", "hook:nightly", "mcp"]
        js = source() + f"\nconsole.log(JSON.stringify({json.dumps(keys)}.map(isStageSource)));"
        r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(dict(zip(keys, json.loads(r.stdout))),
                         {"chat": False, "messages": False, "responses": False, "halo": False, "wizard": False,
                          "lookup": False, "room": True, "hook:nightly": True, "mcp": True})


if __name__ == "__main__":
    unittest.main()
