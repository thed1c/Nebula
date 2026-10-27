"""Custom routing's rules: checked when applied, tried in order, the first
that holds sends the request to a model, to the larger or everyday ones, to one model only, or refuses it; a request
none matches goes on as before; a rule's name is on the decision."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nebula import route_rules as RR
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call

PIC = [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]


class CheckTest(unittest.TestCase):
    def test_bad_rules_are_refused(self):
        for bad in ([{"if": [], "then": {"to": "x"}}], [{"if": [{"what": "moon"}], "then": {"to": "x"}}],
                    [{"if": [{"what": "long", "value": "lots"}], "then": {"to": "x"}}],
                    [{"if": [{"what": "hours", "value": "25:00-1"}], "then": {"to": "x"}}],
                    [{"if": [{"what": "guest"}], "then": {"to": "x", "refuse": "y"}}], "not a list"):
            with self.assertRaises(RR.RuleError):
                RR.check(bad)
        ok = RR.check([{"if": [{"what": "pictures"}, {"what": "guest", "is": False}], "then": {"to": "big"}}])
        self.assertEqual(ok[0]["if"][1], {"what": "guest", "is": False})
        self.assertEqual(ok[0]["name"], "rule 1")

    def test_hours_cross_midnight(self):
        t = lambda h, m: time.struct_time((2026, 10, 20, h, m, 0, 0, 0, -1))
        self.assertTrue(RR._in_span("22:00-07:00", t(23, 30)) and RR._in_span("22:00-07:00", t(6, 59)))
        self.assertFalse(RR._in_span("22:00-07:00", t(12, 0)))
        self.assertTrue(RR._in_span("09:00-17:00", t(9, 0)) and not RR._in_span("09:00-17:00", t(17, 0)))


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.small, self.big = FakeEngine("small"), FakeEngine("big")
        self.tmp = tempfile.mkdtemp()
        p = os.path.join(self.tmp, "nebula.json")
        Path(p).write_text(json.dumps({"port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
                                       "tiers": [{"name": "small", "engine": self.small.url}, {"name": "big", "engine": self.big.url}],
                                       "assistant": {"halo": {"on": False}}, "lan": {"decided": True, "discoverable": False}}))
        self.srv = make_server(Config.from_args(["--config", p]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close(); self.small.stop(); self.big.stop()

    def rules(self, rr, policy=None):
        body = {"rules": {"route_rules": rr}, **({"policy": policy} if policy else {})}
        return call(self.base, "/api/plan/apply", body)

    def ask(self, content, headers=None):
        n0, b0 = len(self.small.requests), len(self.big.requests)
        st, out = call(self.base, "/v1/chat/completions", {"messages": [{"role": "user", "content": content}]}, headers)
        went = "big" if len(self.big.requests) > b0 else "small" if len(self.small.requests) > n0 else None
        return st, went, out

    def test_if_and_if_not_then(self):
        self.assertEqual(self.ask("hello")[1], "small")                                   # no rules: as before
        st, v = self.rules([{"name": "pictures to big", "if": [{"what": "pictures"}], "then": {"to": "big"}},
                            {"name": "no secrets", "if": [{"what": "words", "value": "password, secret"}], "then": {"refuse": "not that here"}},
                            {"name": "not the writer: small only", "if": [{"what": "app", "is": False, "value": "writer-*"},
                                                                          {"what": "long", "value": 5000}], "then": {"only": "small"}}])
        self.assertEqual(st, 200, v)
        self.assertEqual(self.ask(PIC)[1], "big")
        st, went, out = self.ask("tell me the secret")
        self.assertEqual((st, went), (403, None))
        self.assertIn("not that here", json.dumps(out))
        self.assertEqual(self.ask("good morning")[1], "small")                            # none matches: as before
        rows = \
            [r[0] for r in self.srv.ledger.db.execute("SELECT reason FROM decisions")]
        self.assertTrue(any("rule:pictures to big" in r for r in rows), rows)
        self.assertTrue(any("rule:no secrets" in r for r in rows), rows)

    def test_larger_and_a_rule_switched_off(self):
        st, v = self.rules([{"name": "code: larger", "if": [{"what": "code"}], "then": {"to": "larger"}}],
                           policy={"big": {"stage": "capable"}, "small": {"stage": "everyday"}})
        self.assertEqual(st, 200, v)
        self.assertEqual(self.ask("fix this\\n```py\\nx=1\\n```")[1], "big")
        self.rules([{"name": "code: larger", "off": True, "if": [{"what": "code"}], "then": {"to": "larger"}}])
        self.assertEqual(self.ask("and this one\\n```py\\ny=2\\n```")[1], "small")

    def test_bad_rules_are_not_applied(self):
        st, v = self.rules([{"if": [{"what": "moon"}], "then": {"to": "big"}}])
        self.assertEqual(st, 400, v)
        self.assertEqual(call(self.base, "/api/plan")[1]["rules"]["route_rules"], [])


if __name__ == "__main__":
    unittest.main()
