"""The boundary: Nebula reaches a router it watches or routes through only by its interfaces (HTTP, a caller address,
a config), never by its files, its code or its machine's paths. This fails when Nebula's code or examples name a
machine path, a router's private runtime or config directory, or run another system's script.

The patterns here are generic, so the public repository names no one's system. A private list can be added on the
owner's machine: NEBULA_BOUNDARY_DENY=<file> (one extended regex per line, # for comments) checks those names too.
"""
from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {"node_modules", "__pycache__", "_vendor", ".git"}
CODE = (".py", ".js", ".html", ".json")

GENERIC = [
    # /mnt/<one letter> is WSL's mount of a Windows drive (nebula/wslbox.py), generic; any other /mnt/... is a machine's
    (r"""["'](?:/mnt/(?![a-z](?:/|["'])|\{)|/(?:home|srv|media|data)/)""", "an absolute machine path"),
    (r"""["']~/(?!\.config/nebula)""", "a home-relative path outside Nebula's own config"),
    (r"collaboration/", "another system's tree"),
    (r"runtime/broker|broker\.json|broker-switchyard", "a router's private runtime or config"),
    (r"http\+unix://%2F", "a fixed socket address (a socket path belongs in the user's config)"),
    (r"""plan_command["']\s*:\s*\[""", "a watch running another system's script"),
]


def files():
    for p in (ROOT / "nebula").rglob("*"):
        if p.is_file() and p.suffix in CODE and not SKIP_DIRS.intersection(p.relative_to(ROOT).parts):
            if p.name not in ("package-lock.json", Path(__file__).name):
                yield p


def private():
    f = os.environ.get("NEBULA_BOUNDARY_DENY")
    if not f:
        return []
    lines = Path(f).read_text(encoding="utf-8").splitlines()
    return [(ln.strip(), "a name on the private list") for ln in lines if ln.strip() and not ln.startswith("#")]


class BoundaryTest(unittest.TestCase):
    def hits(self, rules, paths):
        out = []
        for p in paths:
            text = p.read_text(encoding="utf-8", errors="replace")
            for n, line in enumerate(text.splitlines(), 1):
                for rx, why in rules:
                    if re.search(rx, line, re.I):
                        out.append(f"{p.relative_to(ROOT)}:{n}: {why}: {line.strip()[:120]}")
        return out

    def test_code_and_examples_name_no_other_system(self):
        hits = self.hits(GENERIC, list(files()))
        self.assertEqual(hits, [], "\n" + "\n".join(hits))

    def test_the_private_list(self):
        rules = private()
        if not rules:
            self.skipTest("NEBULA_BOUNDARY_DENY not set")
        paths = list(files()) + [p for p in (ROOT / "docs").rglob("*.md")] + [ROOT / "README.md"]
        hits = self.hits(rules, paths)
        self.assertEqual(hits, [], "\n" + "\n".join(hits))

    def test_the_rules_catch(self):
        sample = ['"engine": "http+unix://%2Fsrv%2Fx.sock/m"', 'C = Path("/mnt/x/collaboration/tools")',
                  '"plan_command": ["python3", "x.py"]', 'cfg = "~/somewhere/broker.json"']
        for s in sample:
            self.assertTrue(any(re.search(rx, s, re.I) for rx, _ in GENERIC), s)


if __name__ == "__main__":
    unittest.main()
