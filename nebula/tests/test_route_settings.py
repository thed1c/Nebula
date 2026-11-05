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


class OwnRoutesTest(unittest.TestCase):
    """Routes of the owner's own (Star Calling's "Add a new route") and each route's kind: checked, said to Halo with
    their words, a model put on one, a rule sending to one, and a thinking route's answers thinking."""
    ROUTES = {"pictures": {"name": " Pictures ", "about": "photos and screenshots", "kind": ["vision", "thinking"]},
              "larger": {"kind": ["no_thinking"], "instructions": "Show your steps."}}

    def test_check_names_kinds_and_order(self):
        out = RS.check(self.ROUTES)
        self.assertEqual(out["pictures"], {"name": "Pictures", "about": "photos and screenshots", "kind": ["thinking", "vision"]})
        rules = {"routes": out}
        self.assertEqual(RS.names(rules), ["everyday", "larger", "writing", "critique", "pictures"])
        self.assertEqual(RS.label(rules, "pictures"), "Pictures")
        self.assertEqual(RS.words(rules, "pictures"), "photos and screenshots (for requests with pictures)")
        self.assertEqual(RS.words(rules, "writing", {"writing": "stories"}), "stories")
        self.assertIs(RS.thinking(rules, "pictures"), True)
        self.assertIs(RS.thinking(rules, "larger"), False)
        self.assertIsNone(RS.thinking(rules, "everyday"))
        for bad in ({"Moon Base": {"name": "x"}}, {"moon": {"about": "no name"}}, {"moon": {"name": "M", "kind": ["flying"]}},
                    {"moon": {"name": "M", "kind": ["thinking", "no_thinking"]}}, {"moon": {"name": "M" * 41}},
                    {f"r{i:02d}": {"name": "x"} for i in range(RS.MAX_ROUTES + 1)}):
            with self.assertRaises(RS.RouteError):
                RS.check(bad)

    def test_a_model_on_an_own_route_and_halo_hears_its_words(self):
        from types import SimpleNamespace
        from unittest import mock
        from nebula import policy as P, stage
        from nebula.router import Rules
        pol = P.Policy()
        pol.put({"models": {"eye": {"stage": "pictures"}, "small": {}}})
        self.assertEqual(pol.stage("eye"), "pictures")
        with self.assertRaises(P.PolicyError):
            pol.put({"models": {"eye": {"stage": "Not A Route"}}})
        rules = Rules(); rules.routes = RS.check(self.ROUTES); rules.stage_router = {"mode": "live", "judge": "halo"}
        tiers = [SimpleNamespace(name="small"), SimpleNamespace(name="eye")]
        turn = SimpleNamespace(rules=rules, policy=pol, pinned=False, passthrough=False, ladder=list(tiers),
                               req={"messages": [{"role": "user", "content": "what is in this photo?"}]})
        seen = {}
        def halo(req, routes, timeout=12.0, words=None):
            seen.update(routes=routes, words=words)
            return {"route": "pictures", "why": "a photo", "model": "halo", "seconds": 0.1}
        with mock.patch.object(stage, "ask_halo", halo):
            got = stage.pick(turn, 100)
        self.assertEqual(seen["routes"], ["everyday", "pictures"])
        self.assertEqual(seen["words"]["pictures"], "photos and screenshots (for requests with pictures)")
        self.assertEqual(got["picked"], "pictures")
        self.assertEqual([t.name for t in turn.ladder], ["eye", "small"])
        turn.stage, turn.routed = got, None
        self.assertEqual(RS.route_of(turn, "eye"), "pictures")
        self.assertEqual(RS.route_of(SimpleNamespace(rules=Rules(), policy=pol), "eye"), "everyday")    # removed: everyday

    def test_a_rule_sends_to_an_own_route(self):
        from types import SimpleNamespace
        from nebula import policy as P, route_rules
        from nebula.router import Rules
        pol = P.Policy(); pol.put({"models": {"eye": {"stage": "pictures"}}})
        rules = Rules(); rules.routes = RS.check(self.ROUTES)
        rules.route_rules = [{"name": "photos", "if": [{"what": "pictures", "is": False}], "then": {"to": "pictures"}}]
        tiers = [SimpleNamespace(name="small"), SimpleNamespace(name="eye")]
        turn = SimpleNamespace(rules=rules, policy=pol, pinned=False, ladder=list(tiers), req={"messages": [{"role": "user", "content": "hi"}]},
                               caller=None, agent=None, tools=None)
        out = route_rules.apply(turn, 10, None, None)
        self.assertTrue(out and out["moved"], out)
        self.assertEqual([t.name for t in turn.ladder], ["eye", "small"])


class AnyTaskOrderTest(unittest.TestCase):
    """Star Calling's rows say "left to right": a chosen list (Any task, a helper's own) keeps its order (after "ask
    first"), and a route put first by smart routing keeps its own order (its models' priority)."""
    def test_a_chosen_list_keeps_its_order_and_a_route_its_own(self):
        from types import SimpleNamespace
        from unittest import mock
        from nebula import policy as P, stage
        from nebula.router import Rules
        pol = P.Policy()
        pol.put({"models": {"a": {"priority": 1}, "b": {"priority": 2, "stage": "larger"}, "c": {"priority": 1, "stage": "larger"},
                            "d": {"prefer": True, "priority": 9}}})
        T = [SimpleNamespace(name=n) for n in ("b", "a", "c", "d")]
        self.assertEqual([t.name for t in pol.kept(T)], ["d", "b", "a", "c"])          # ask first, then the list's order
        self.assertEqual([t.name for t in pol.order(T)], ["d", "a", "c", "b"])          # no list chosen: priority
        rules = Rules(); rules.stage_router = {"mode": "live", "judge": "halo"}
        turn = SimpleNamespace(rules=rules, policy=pol, pinned=False, passthrough=False, ladder=pol.kept(T),
                               req={"messages": [{"role": "user", "content": "debug this"}]})
        with mock.patch.object(stage, "ask_halo", lambda req, routes, timeout=12.0, words=None: {"route": "larger", "why": "code", "model": "halo", "seconds": 0}):
            stage.pick(turn, 10)
        self.assertEqual([t.name for t in turn.ladder], ["c", "b", "d", "a"])            # the route by its order, then the list


class PutAwayTest(unittest.TestCase):
    """Star Calling shows the built-in routes but everyday as suggestions: one put away is out of smart routing until it
    is brought back; everyday stays."""
    def test_a_suggested_route_put_away(self):
        out = RS.check({"writing": {"hidden": True}, "critique": {"hidden": False, "instructions": "Quote it."}})
        self.assertEqual(out, {"writing": {"hidden": True}, "critique": {"instructions": "Quote it."}})
        rules = {"routes": {**out, "pictures": {"name": "Pictures"}}}
        self.assertEqual(RS.names(rules), ["everyday", "larger", "critique", "pictures"])
        self.assertFalse(RS.is_route(rules, "writing"))
        for bad in ({"everyday": {"hidden": True}}, {"pictures": {"name": "P", "hidden": True}}, {"writing": {"hidden": "yes"}}):
            with self.assertRaises(RS.RouteError):
                RS.check(bad)
