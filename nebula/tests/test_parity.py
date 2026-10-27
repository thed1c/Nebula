"""The Switchyard's function set in Nebula's router (nebula/stage.py, turn.py, admission.py, policy.py, registry.py,
dialects/embeddings.py, schema.py): the stage router and its judge, escalation that ignores outages and lets go after
its hold, route eligibility, explicit-only and typed destinations, a model's own queue cap, timeout and wait, the
parking cap, the capacity modes, run stickiness and its limit, the output budget, pinned embeddings, readiness by
the loaded file."""
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine


def post(base, path, body, headers=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), dict(e.headers)


class Rig(unittest.TestCase):
    rules: dict = {}
    policy: dict = {}
    extra_tiers: list = []

    def setUp(self):
        self.small, self.big = FakeEngine("small"), FakeEngine("big")
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False, "rules": self.rules,
            "tiers": [{"name": "small", "engine": self.small.url, "context": 4096}, {"name": "big", "engine": self.big.url, "context": 8192}]
            + [dict(t, engine=getattr(self, t["engine"]).url) if t.get("engine") in ("small", "big") else t for t in self.extra_tiers],
            "assistant": {"halo": {"on": False}},
            "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        Path(self.path).with_suffix(".routing.json").write_text(json.dumps({"models": self.policy}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        self.small.stop(); self.big.stop()

    def chat(self, messages, headers=None, **kw):
        return post(self.base, "/v1/chat/completions", {"messages": messages, **kw}, headers)

    def last(self):
        row = self.srv.ledger.db.execute("SELECT tier, inputs, reason FROM decisions WHERE advisory = 0 ORDER BY rowid DESC LIMIT 1").fetchone()
        return row[0], json.loads(row[1]), row[2]

    def hold(self, tier_name):
        """Take the model's only slot (as a long call would), until the returned release() is called."""
        g = self.srv.admission.gate(self.srv.cfg.tier(tier_name))
        g.acquire(0, {"attempt_id": "holder"}, timeout=5)
        return lambda: g.release({"attempt_id": "holder"})


HARD = [{"role": "user", "content": "debug this step by step:\n```\nx = 1\n```"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "error: the test failed"}]


class StageRouterTest(Rig):
    rules = {"stage_router": {"mode": "live", "threshold": 0.5}}
    policy = {"small": {"stage": "everyday"}, "big": {"stage": "capable"}}

    def test_live_everyday_and_capable(self):
        self.assertEqual(self.chat([{"role": "user", "content": "hi"}])[0], 200)
        tier, inputs, _ = self.last()
        self.assertEqual(tier, "small")
        self.assertEqual(next(s for s in inputs["stages"] if s["stage"] == "tier")["picked"], "everyday")
        self.assertEqual(self.chat(HARD)[0], 200)
        tier, inputs, _ = self.last()
        st = next(s for s in inputs["stages"] if s["stage"] == "tier")
        self.assertEqual((tier, st["picked"], st["how"]), ("big", "larger", "score"))
        self.assertIn("tool error(s)", " ".join(st["signals"]))
        self.assertEqual(self.chat(HARD, model="small")[0], 200)                    # a model it named itself is never moved
        self.assertEqual(self.last()[0], "small")


class StageShadowTest(Rig):
    rules = {"stage_router": {"mode": "shadow"}}
    policy = {"small": {"stage": "everyday"}, "big": {"stage": "capable"}}

    def test_shadow_only_records(self):
        self.chat(HARD)
        tier, inputs, _ = self.last()
        self.assertEqual(tier, "small")                                               # nothing changed
        self.assertEqual(next(s for s in inputs["stages"] if s["stage"] == "tier")["picked"], "larger")


class JudgeTest(Rig):
    rules = {"stage_router": {"mode": "live", "threshold": 0.5, "unsure": 0.5, "judge": "judge"}}
    policy = {"small": {"stage": "everyday"}, "big": {"stage": "capable"}, "judge": {"explicit": True}}
    extra_tiers = [{"name": "judge", "engine": "small"}]

    def test_an_unsure_request_goes_to_the_judge(self):
        def script(body):
            sys_msg = (body.get("messages") or [{}])[0].get("content") or ""
            return '{"capable": true, "confidence": 0.9}' if "You route requests" in sys_msg else None
        self.small.script = script
        self.assertEqual(self.chat([{"role": "user", "content": "hi"}])[0], 200)
        tier, inputs, _ = self.last()
        st = next(s for s in inputs["stages"] if s["stage"] == "tier")
        self.assertEqual((tier, st["how"], st["judge"]["capable"], st["judge"]["model"]), ("big", "judge", True, "judge"))


class HaloRouterTest(Rig):
    """Halo as the smart router (design decision 2026-10-21): it reads every turn and picks the route; the reply says why."""
    rules = {"stage_router": {"mode": "live", "judge": "halo"}}
    policy = {"small": {"stage": "everyday"}, "big": {"stage": "writing"}}

    def setUp(self):
        super().setUp()
        from types import SimpleNamespace
        from nebula import stage
        self.halo = FakeEngine("halo")
        self.halo.script = lambda body: ('{"route": "writing", "why": "asks for a story"}'
                                         if "story" in body["messages"][-1]["content"] else '{"route": "everyday", "why": "a greeting"}')
        own = SimpleNamespace(ready=lambda: True, tier=SimpleNamespace(address=self.halo.url, model="halo", context=4096))
        self.old, stage.HALO = stage.HALO, (lambda: own)

    def tearDown(self):
        from nebula import stage
        stage.HALO = self.old
        self.halo.stop()
        super().tearDown()

    def test_halo_picks_each_turn_and_says_why(self):
        self.assertEqual(self.chat([{"role": "user", "content": "write me a story about a fox"}])[0], 200)
        tier, inputs, reason = self.last()
        self.assertEqual(tier, "big")
        self.assertIn("route:writing (Halo: asks for a story)", reason)
        self.assertEqual(self.chat([{"role": "user", "content": "hi"}])[0], 200)
        self.assertEqual(self.last()[0], "small")
        self.halo.script = lambda body: "not json"                             # Halo out of sorts: the score decides
        self.assertEqual(self.chat([{"role": "user", "content": "thanks, that is all"}])[0], 200)
        self.assertEqual(self.last()[0], "small")


class CueRouteTest(unittest.TestCase):
    def test_words_point_to_a_route(self):
        from nebula.stage import cue_route
        r = ["everyday", "larger", "writing", "critique"]
        self.assertEqual(cue_route("Write a poem about autumn leaves.", r), "writing")
        self.assertEqual(cue_route("Give me feedback on this: 'The rain fell.'", r), "critique")
        self.assertEqual(cue_route("Why does my python function raise an error?", r), "larger")
        self.assertEqual(cue_route("Recommend a movie for tonight.", r), "everyday")
        self.assertEqual(cue_route("Write a poem", ["everyday", "larger"]), "everyday")       # only routes it has


class EscalationTest(Rig):
    rules = {"tool_errors_to_rise": 2, "escalation_hold_seconds": 1}

    def turn_msgs(self, errors):
        return [{"role": "user", "content": "do it"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "run", "arguments": "{}"}}
                                                                    for i in range(len(errors))]}] + \
            [{"role": "tool", "tool_call_id": f"c{i}", "content": e} for i, e in enumerate(errors)]

    def turn(self, errors, sess="s1"):
        msgs = [{"role": "user", "content": "do it"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "run", "arguments": "{}"}}
                                                                    for i in range(len(errors))]}]
        msgs += [{"role": "tool", "tool_call_id": f"c{i}", "content": e} for i, e in enumerate(errors)]
        return self.chat(msgs, {"X-Nebula-Session": sess})

    def test_outages_are_no_strike_and_the_hold_ends(self):
        self.turn(["error: 503 Service Unavailable", "error: connection refused"])
        self.assertEqual(self.last()[0], "small")                                     # its tools' service was down
        msgs, (st, out, _h) = self.turn_msgs(["error: bad argument", "error: still wrong"]), self.turn(["error: bad argument", "error: still wrong"], "s2")
        self.assertEqual(self.last()[0], "big")                                       # the model's own failures: up
        time.sleep(1.5)
        reply = out["choices"][0]["message"]["content"]
        self.chat(msgs + [{"role": "assistant", "content": reply}, {"role": "user", "content": "next"}], {"X-Nebula-Session": "s2"})
        tier, inputs, reason = self.last()
        self.assertEqual(tier, "small")                                               # the hold ran out: back to the floor


class EligibilityTest(Rig):
    policy = {"small": {"eligible": {"tools": False}}, "big": {"prefer": True}}

    def test_a_need_skips_a_model_not_eligible_for_it(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        self.policy_small_first()
        self.assertEqual(self.chat([{"role": "user", "content": "hi"}], tools=tools)[0], 200)
        self.assertEqual(self.last()[0], "big")
        st, err, _ = self.chat([{"role": "user", "content": "hi"}], tools=tools, model="small")
        self.assertEqual(st, 503, err)
        self.assertIn("not eligible for tools", err["error"]["message"])

    def policy_small_first(self):
        self.srv.policy.put({"models": {"small": {"eligible": {"tools": False}, "priority": 1}, "big": {"priority": 2}}})


class TypedDestinationTest(Rig):
    policy = {"decide": {"explicit": True, "schema": {"type": "object", "required": ["verdict"],
                                                      "properties": {"verdict": {"enum": ["yes", "no"]}}}}}
    extra_tiers = [{"name": "decide", "engine": "big"}]

    def test_named_only_and_its_answer_checked(self):
        self.chat([{"role": "user", "content": "hi"}])
        self.assertNotEqual(self.last()[0], "decide")                                  # never a fallback
        self.big.script = lambda body: '{"verdict": "yes"}'
        st, out, _ = self.chat([{"role": "user", "content": "ok?"}], model="decide")
        self.assertEqual(st, 200, out)
        self.big.script = lambda body: '{"verdict": "maybe"}'
        st, err, _ = self.chat([{"role": "user", "content": "ok?"}], model="decide")
        self.assertEqual((st, err["error"]["type"]), (502, "invalid_response"), err)
        st, err, _ = self.chat([{"role": "user", "content": "ok?"}], model="decide", stream=True)
        self.assertEqual(st, 400, err)


class QueueTest(Rig):
    policy = {"small": {"queue": {"max_queued": {"interactive": 0}}}}

    def test_queue_full_at_once(self):
        release = self.hold("small")
        try:
            t0 = time.time()
            st, err, _ = self.chat([{"role": "user", "content": "hi"}], model="small")
            self.assertEqual(st, 503, err)
            self.assertIn("queue_full", err["error"]["message"])
            self.assertLess(time.time() - t0, 3)
        finally:
            release()


class QueueTimeoutTest(Rig):
    policy = {"small": {"queue": {"timeout_s": 5}}}

    def test_its_own_timeout(self):
        release = self.hold("small")
        try:
            t0 = time.time()
            st, err, _ = self.chat([{"role": "user", "content": "hi"}], model="small")
            self.assertEqual(st, 503, err)
            self.assertLess(time.time() - t0, 9)                                       # its 5 s, not the rules' 600
        finally:
            release()


class WaitAndModesTest(Rig):
    policy = {"small": {"wait_s": 0, "priority": 1}, "big": {"priority": 2}}

    def test_its_own_wait_overflows_at_once(self):
        release = self.hold("small")
        try:
            self.assertEqual(self.chat([{"role": "user", "content": "hi"}])[0], 200)
            tier, _inputs, _ = self.last()
            self.assertEqual(tier, "big")                                              # interactive would have waited 60 s
        finally:
            release()

    def test_shadow_and_off(self):
        self.srv.cfg.rules = {"capacity_mode": "shadow"}
        release = self.hold("small")
        threading.Timer(1.0, release).start()
        self.assertEqual(self.chat([{"role": "user", "content": "hi"}])[0], 200)
        tier, inputs, _ = self.last()
        self.assertEqual((tier, inputs.get("capacity_would_pick")), ("small", "big"))
        self.srv.cfg.rules = {"capacity_mode": "off"}
        release = self.hold("small")
        threading.Timer(1.0, release).start()
        self.chat([{"role": "user", "content": "another question"}])
        tier, inputs, _ = self.last()
        self.assertEqual((tier, inputs.get("capacity_would_pick")), ("small", None))


class ParkingAndBudgetTest(Rig):
    rules = {"parking_max_calls": 0}

    def test_parking_full(self):
        st, err, _ = self.chat([{"role": "user", "content": "x " * 40000}])
        self.assertEqual(st, 503, err)
        self.assertIn("parking_full", err["error"]["message"])

    def test_the_budget_is_cut_to_the_slot_and_said(self):
        st, out, headers = self.chat([{"role": "user", "content": "hello " * 1000}], model="small", max_tokens=20000)
        self.assertEqual(st, 200, out)
        self.assertIn("clamped", headers.get("X-Nebula-Budget", ""))
        sent = self.small.requests[-1]["max_tokens"]
        self.assertLess(sent, 4096)
        self.assertEqual(self.last()[1]["budget"]["clamped"], True)


class StickyRunTest(Rig):
    rules = {"sticky_seconds": 1}

    def test_a_run_sticks_then_may_move(self):
        h = {"X-Nebula-Run": "job-7"}
        self.chat([{"role": "user", "content": "one"}], h, model="big")
        self.chat([{"role": "user", "content": "one"}, {"role": "assistant", "content": "Hello world"}, {"role": "user", "content": "two"}], h)
        self.assertEqual(self.last()[0], "big")                                        # the run stays where it started
        sessions = self.srv.ledger.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        self.assertEqual(sessions, 1)
        time.sleep(1.5)
        self.chat([{"role": "user", "content": "one"}, {"role": "assistant", "content": "Hello world"}, {"role": "user", "content": "two"},
                   {"role": "assistant", "content": "Hello world"}, {"role": "user", "content": "three"}], h)
        tier, _inputs, reason = self.last()
        self.assertEqual(tier, "small")                                                # idle past its limit: free to move


class EmbeddingsAndReadinessTest(Rig):
    extra_tiers = [{"name": "emb", "engine": "small"}]
    policy = {"big": {"expect_model": "right-model"}}

    def test_embeddings_are_pinned(self):
        st, out, headers = post(self.base, "/v1/embeddings", {"model": "emb", "input": ["a", "bb"]})
        self.assertEqual(st, 200, out)
        self.assertEqual((len(out["data"]), out["model"]), (2, "emb"))
        self.assertIn("pinned", headers.get("X-Nebula-Route", ""))
        self.assertEqual(post(self.base, "/v1/embeddings", {"input": "a"})[0], 400)
        self.assertEqual(post(self.base, "/v1/embeddings", {"model": "nope", "input": "a"})[0], 404)
        self.small.stop()
        st, err, _ = post(self.base, "/v1/embeddings", {"model": "emb", "input": "a"})
        self.assertEqual(st, 502, err)                                                 # down: never another model's vectors

    def test_ready_only_with_the_expected_file(self):
        self.big.model_path = "/models/wrong-model.gguf"
        self.assertFalse(self.srv.registry.up("big", force=True))
        node = self.srv.registry.node("big")
        self.assertIn("expected right-model", node["why_down"])
        self.big.model_path = "/models/right-model-Q4.gguf"
        self.assertTrue(self.srv.registry.up("big", force=True))


class PlanApiTest(Rig):
    """Every setting above, set from the Plan tab's API (GET /api/plan; POST /api/plan/apply {policy, rules})."""

    def test_set_from_the_plan_and_used(self):
        st, v, _h = post(self.base, "/api/plan/apply", {
            "policy": {"small": {"stage": "everyday", "queue": {"max_queued": {"background": 2}, "timeout_s": 30}, "wait_s": 5,
                                 "eligible": {"structured": False}},
                       "big": {"stage": "capable", "expect_model": "fake", "explicit": False}},
            "rules": {"stage_router": {"mode": "live", "threshold": 0.5, "judge": None}, "capacity_mode": "shadow",
                      "sticky_seconds": 120, "parking_max_calls": 3, "clamp_output": True, "escalation_hold_seconds": 600,
                      "escalation_ignore_outages": True, "long_context_tokens": 16384}})
        self.assertEqual(st, 200, v)
        got = json.loads(urllib.request.urlopen(self.base + "/api/plan").read())
        self.assertEqual((got["policy"]["big"]["stage"], got["rules"]["capacity_mode"], got["rules"]["stage_router"]["mode"],
                          got["rules"]["parking_max_calls"]), ("larger", "shadow", "live", 3))
        self.assertEqual(self.chat(HARD)[0], 200)
        self.assertEqual(self.last()[0], "big")                                         # the stage router, live from the Plan
        self.assertEqual(json.loads(Path(self.path).read_text())["rules"]["sticky_seconds"], 120)    # saved to the config
        self.assertEqual(post(self.base, "/api/plan/apply", {"rules": {"capacity_mode": "sometimes"}})[0], 400)
        self.assertEqual(post(self.base, "/api/plan/apply", {"policy": {"small": {"stage": "huge"}}})[0], 400)


if __name__ == "__main__":
    unittest.main()
