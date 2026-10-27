"""A Gravity router on this machine (nebula/pairlocal.py): found on loopback by its identity probe (never a plain
engine), adopted as one model per model it offers, its scheduling (exact model, least pending, retry before the first
byte); and no user-facing text says "PAIR" (design decision: users only see Gravity)."""
import ast
import json
import os
import re
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nebula import pairlocal
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.fake_pair import FakeRouter, Node
from nebula.tests.test_callers import call

ROOT = Path(__file__).resolve().parents[2]


class GravityLocalTest(unittest.TestCase):
    def setUp(self):
        self.nodes = [Node("alpha", ["tiny", "chat-7b"]), Node("beta", ["chat-7b"])]
        self.router = FakeRouter(self.nodes)
        self.plain = FakeEngine("plain")
        port = lambda u: int(u.rsplit(":", 1)[1])
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
            "tiers": [{"name": "local", "engine": self.plain.url}],
            "gravity_local_ports": [port(self.router.url), port(self.plain.url)],
            "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        self.router.close(); self.plain.shutdown()

    def ask(self, model):
        return call(self.base, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}]},
                    {"X-Nebula-Session": uuid.uuid4().hex})

    def test_found_on_this_machine_only_by_its_probe(self):
        here = call(self.base, "/api/yard/gravity")[1]["here"]
        self.assertEqual([h["address"] for h in here], [self.router.url])          # the plain engine is not one
        self.assertEqual(here[0]["models"], ["chat-7b", "tiny"])

    def test_adopted_and_routed(self):
        st, r = call(self.base, "/api/yard/gravity/adopt", {"address": self.router.url})
        self.assertEqual((st, r["made"]), (200, ["gravity-chat-7b", "gravity-tiny"]), r)
        self.assertEqual(call(self.base, "/api/yard/gravity/adopt", {"address": self.plain.url})[0], 400)
        self.assertEqual(call(self.base, "/api/yard/gravity/adopt", {"address": "http://10.0.0.5:8080"})[0], 400)
        node = [n for n in call(self.base, "/api/registry")[1] if n["name"] == "gravity-tiny"][0]
        self.assertEqual((node["kind"], node["up"]), ("gravity", True))
        self.assertEqual(self.ask("gravity-tiny")[0], 200)
        self.assertEqual((self.nodes[0].served, self.nodes[1].served), (1, 0))      # only alpha has "tiny"
        with ThreadPoolExecutor(4) as ex:                                            # a burst spreads over both
            list(ex.map(lambda _i: self.ask("gravity-chat-7b"), range(4)))
        self.assertTrue(self.nodes[1].served >= 1 and self.nodes[0].served >= 2, [n.served for n in self.nodes])
        self.nodes[0].engine.shutdown()                                               # alpha's engine gone
        self.nodes[0].engine.server_close()
        st, body = self.ask("gravity-chat-7b")
        self.assertEqual(st, 200, body)                                               # retried on beta, unseen
        st, body = self.ask("gravity-tiny")
        self.assertEqual(st, 503, body)                                               # nobody else has it

    def test_no_user_facing_text_says_pair(self):
        """Users only see Gravity. Internal names, comments and the license credits may say it."""
        hits = []
        word = re.compile(r"\bPAIR\b|Personal AI Router")
        html = (ROOT / "nebula" / "ui" / "index.html").read_text(encoding="utf-8")
        for n, line in enumerate(html.splitlines(), 1):
            if word.search(line) and not line.strip().startswith("//"):
                hits.append(f"index.html:{n}")
        for p in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if word.search(line):
                    hits.append(f"{p.relative_to(ROOT)}:{n}")
        for p in sorted((ROOT / "nebula").glob("*.py")):            # strings a user can be shown (not docstrings)
            tree = ast.parse(p.read_text(encoding="utf-8"))
            docs = {id(n.body[0].value) for n in ast.walk(tree)
                    if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body
                    and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs \
                        and word.search(n.value) and not n.value.startswith("X-NVPAIR"):
                    hits.append(f"{p.relative_to(ROOT)}:{n.lineno}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
