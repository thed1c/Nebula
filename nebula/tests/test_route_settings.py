"""Each route's own instructions: checked when applied, saved in the config, put first in the system prompt of a request
on that route (before the app's own, which is kept), chosen by a rule's route, else by the answering model's route;
none for a passthrough caller; the route is on the decision."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from nebula import route_settings as RS
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call


class CheckTest(unittest.TestCase):
    def test_bad_routes_are_refused(self):
        for bad in ("text", {"moon": {"instructions": "x"}}, {"writing": "x"}, {"writing": {"instructions": 5}},
                    {"writing": {"instructions": "x" * (RS.MAX_INSTRUCTIONS + 1)}}):
            with self.assertRaises(RS.RouteError):
                RS.check(bad)

    def test_empty_is_dropped_and_other_keys_pass(self):
        out = RS.check({"writing": {"instructions": "  Write scenes.  "}, "everyday": {"instructions": " "},
                        "critique": {"instructions": "", "adapter": "review-v1"}})
        self.assertEqual(out, {"writing": {"instructions": "Write scenes."}, "critique": {"adapter": "review-v1"}})

    def test_apply_puts_them_first(self):
        b = {"messages": [{"role": "user", "content": "hi"}]}
        RS.apply(b, "R")
        self.assertEqual(b["messages"][0], {"role": "system", "content": "R"})
        b = {"messages": [{"role": "system", "content": "App"}, {"role": "user", "content": "hi"}]}
        RS.apply(b, "R")
        self.assertEqual(b["messages"][0]["content"], "R\n\nApp")
        b = {"messages": [{"role": "system", "content": [{"type": "text", "text": "App"}]}]}
        RS.apply(b, "R")
        self.assertEqual(b["messages"][0]["content"][0]["text"], "R\n\n")
        b = {"messages": [{"role": "user", "content": "hi"}]}
        RS.apply(b, "")
        self.assertEqual(len(b["messages"]), 1)


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.small, self.big = FakeEngine("small"), FakeEngine("big")
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
            "tiers": [{"name": "small", "engine": self.small.url}, {"name": "big", "engine": self.big.url}],
            "assistant": {"halo": {"on": False}}, "lan": {"decided": True, "discoverable": False},
            "callers": [{"id": "raw-app", "tags": ["raw/*"], "mode": "passthrough"}]}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close(); self.small.stop(); self.big.stop()

    def apply(self, rules, policy=None):
        return call(self.base, "/api/plan/apply", {"rules": rules, **({"policy": policy} if policy else {})})

    def ask(self, messages, headers=None):
        n0, b0 = len(self.small.requests), len(self.big.requests)
        st, out = call(self.base, "/v1/chat/completions", {"messages": messages}, headers)
        eng = self.big if len(self.big.requests) > b0 else self.small if len(self.small.requests) > n0 else None
        return st, ("big" if eng is self.big else "small" if eng is self.small else None), (eng.requests[-1] if eng else None)

    def test_each_route_gets_its_own(self):
        st, v = self.apply({"routes": {"everyday": {"instructions": "Be brief."}, "writing": {"instructions": "Write scenes."}}},
                           policy={"small": {"stage": "everyday"}, "big": {"stage": "writing"}})
        self.assertEqual(st, 200, v)
        self.assertEqual(v["rules"]["routes"]["writing"], {"instructions": "Write scenes."})
        saved = json.loads(Path(self.path).read_text())
        self.assertEqual(saved["rules"]["routes"]["everyday"], {"instructions": "Be brief."})
        # an everyday answer: its route's text first, the app's own kept after it
        st, went, sent = self.ask([{"role": "system", "content": "You are the app."}, {"role": "user", "content": "hello"}])
        self.assertEqual((st, went), (200, "small"))
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "Be brief.\n\nYou are the app."})
        self.assertEqual(sum(1 for m in sent["messages"] if m["role"] == "system"), 1)
        # a rule that sends it to the writing route: the writing text
        st, v = self.apply({"route_rules": [{"name": "stories", "if": [{"what": "words", "value": "story"}], "then": {"to": "writing"}}]})
        self.assertEqual(st, 200, v)
        st, went, sent = self.ask([{"role": "user", "content": "write me a story"}])
        self.assertEqual((st, went), (200, "big"))
        self.assertEqual(sent["messages"][0]["role"], "system")         # a move to another engine brings its handoff note
        self.assertTrue(sent["messages"][0]["content"].startswith("Write scenes."), sent["messages"][0])
        self.assertNotIn("Be brief.", sent["messages"][0]["content"])
        rows = [r[0] for r in self.srv.ledger.db.execute("SELECT reason FROM decisions")]
        self.assertTrue(any("instructions:writing" in r for r in rows), rows)
        self.assertTrue(any("instructions:everyday" in r for r in rows), rows)
        # the history the ledger keeps is the app's own, without the route's text
        self.assertFalse(any("Write scenes." in (r[0] or "") for r in self.srv.ledger.db.execute("SELECT payload FROM events")))

    def test_none_set_changes_nothing_and_passthrough_gets_none(self):
        st, went, sent = self.ask([{"role": "user", "content": "hello"}])
        self.assertEqual(sent["messages"], [{"role": "user", "content": "hello"}])
        self.apply({"routes": {"everyday": {"instructions": "Be brief."}}})
        st, went, sent = self.ask([{"role": "user", "content": "hello"}], {"X-Nebula-Caller": "raw/one"})
        self.assertEqual(st, 200)
        self.assertEqual(sent["messages"], [{"role": "user", "content": "hello"}])
        st, went, sent = self.ask([{"role": "user", "content": "hello again"}])
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "Be brief."})

    def test_bad_routes_are_not_applied(self):
        st, v = self.apply({"routes": {"moon": {"instructions": "x"}}})
        self.assertEqual(st, 400, v)
        self.assertEqual(call(self.base, "/api/plan")[1]["rules"]["routes"], {})


if __name__ == "__main__":
    unittest.main()
