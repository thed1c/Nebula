"""The scripted escalations, through the gateway, with two fake engines: fast and think on one, big on the other."""
import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from nebula.config import Config, Tier
from nebula.replay import verify
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine


def rows(path, sql, *args):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(sql, args)]
    finally:
        db.close()


class Rig:
    """Two fake engines, a cockpit with fast / think / big tiers, and a ledger in a temporary directory."""

    def __init__(self, big_ctx=262144, fast_ctx=None, ledger=True, rules=None, agents=None, mcp_servers=None, secretary=None,
                 hooks=None, schedules=None, extra_tiers=None):
        self.a, self.b = FakeEngine("fastfake"), FakeEngine("bigfake")
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "ledger.sqlite3")
        tiers = [Tier("fast", self.a.url, thinking=False, context=fast_ctx), Tier("think", self.a.url, thinking=True, context=fast_ctx),
                 Tier("big", self.b.url, context=big_ctx)] + list(extra_tiers or [])
        cfg = Config(port=0, tiers=tiers, ledger=self.path if ledger else None, rules=rules or {}, monitor=False,
                     agents=list(agents or []), mcp_servers=dict(mcp_servers or {}), secretary=secretary,
                     mcp_settings={"timeout_s": 10, "start_timeout_s": 30}, hooks=list(hooks or []), schedules=list(schedules or []))
        cfg.base_dir = self.tmp.name                  # keys, policies and work directories land in the rig's temp dir
        self.srv = make_server(cfg)
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.a.stop()
        self.b.stop()
        self.tmp.cleanup()

    def chat(self, messages, model="m", headers=None, **extra):
        body = {"model": model, "messages": messages, **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())["choices"][0]["message"]

    def attempts(self):
        return rows(self.path, "SELECT * FROM attempts ORDER BY created_at")

    def decisions(self):
        return rows(self.path, "SELECT * FROM decisions ORDER BY created_at")

    def handoffs(self):
        out = rows(self.path, "SELECT * FROM handoffs ORDER BY created_at")
        for h in out:
            h["packet"] = json.loads(h["packet"])
        return out

    def kinds(self):
        return [r["kind"] for r in rows(self.path, "SELECT kind FROM events ORDER BY seq")]


class EscalationTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()

    def tearDown(self):
        self.rig.stop()

    def assert_replays(self):
        ok, lines = verify(self.rig.path)
        self.assertTrue(ok, "\n".join(lines))

    def test_named_tier_crosses_with_a_packet_the_destination_answers_from(self):
        rig = self.rig
        a1 = rig.chat([{"role": "system", "content": "Answer in French."}, {"role": "user", "content": "Name a colour."}])
        self.assertEqual(len(rig.a.requests), 1)
        a2 = rig.chat([{"role": "system", "content": "Answer in French."}, {"role": "user", "content": "Name a colour."},
                       a1, {"role": "user", "content": "Another?"}], model="cockpit/big")
        self.assertEqual(a2["content"], "Hello world")
        self.assertEqual((len(rig.a.requests), len(rig.b.requests)), (1, 1))
        sent = rig.b.requests[0]["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("handed over", sent[0]["content"])
        self.assertIn("Original task: Name a colour.", sent[0]["content"])
        self.assertIn("- Answer in French.", sent[0]["content"])
        self.assertIn("named the big tier", sent[0]["content"])
        self.assertEqual([m["role"] for m in sent[1:]], ["user", "assistant", "user"])
        self.assertEqual(sent[-1]["content"], "Another?")
        self.assertNotIn("reasoning_content", sent[2])      # the packet's events are canonical: no reasoning, no ids
        attempts = rig.attempts()
        self.assertEqual([(a["tier"], a["state"]) for a in attempts], [("fast", "completed"), ("fast", "escalated"), ("big", "completed")])
        self.assertEqual(len({a["task_id"] for a in attempts}), 1)
        self.assertEqual(attempts[2]["request_sha256"], attempts[1]["request_sha256"])
        decisions = rig.decisions()
        self.assertEqual([d["reason"] for d in decisions], ["floor", "named:big"])
        inputs = json.loads(decisions[1]["inputs"])
        self.assertIn("move_cost_tokens", inputs)
        hs = rig.handoffs()
        self.assertEqual(len(hs), 1)
        h = hs[0]
        self.assertEqual((h["from_attempt"], h["to_attempt"], h["to_tier"]), (attempts[1]["attempt_id"], attempts[2]["attempt_id"], "big"))
        self.assertIsNotNone(h["sent_at"])
        self.assertIsNotNone(h["acknowledged_at"])
        self.assertEqual(h["packet"]["escalation_reason"]["source"], "named:big")
        self.assertEqual(h["packet"]["coverage"]["through_seq"], 5)
        self.assertEqual([e["role"] for e in h["packet"]["events_since_briefing"]], ["user", "assistant", "user"])
        self.assertTrue(h["packet"]["budget"]["fits"])
        self.assertEqual(rig.kinds(), ["message", "message", "request", "reply", "message", "request", "handoff", "request", "reply"])
        self.assertEqual(rows(rig.path, "SELECT value FROM outcomes ORDER BY recorded_at")[1]["value"], "escalated")
        self.assert_replays()

    def test_escalate_header_rises_on_the_same_engine_then_crosses(self):
        rig = self.rig
        a1 = rig.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(rig.a.requests[-1]["chat_template_kwargs"], {"enable_thinking": False})
        hist = [{"role": "user", "content": "hi"}, a1, {"role": "user", "content": "more"}]
        a2 = rig.chat(hist, headers={"X-Cockpit-Escalate": "1"})
        self.assertEqual(rig.a.requests[-1]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(a2["reasoning_content"], "Let me think.")
        self.assertEqual([(a["tier"], a["state"], a["thinking"]) for a in rig.attempts()], [("fast", "completed", 0), ("think", "completed", 1)])
        self.assertEqual(rig.decisions()[-1]["reason"], "escalate:1")
        self.assertEqual(rig.handoffs(), [])
        hist += [a2, {"role": "user", "content": "again"}]
        a3 = rig.chat(hist, model="fast")                  # named, one step down on the same engine
        self.assertEqual([(a["tier"], a["thinking"]) for a in rig.attempts()][-1], ("fast", 0))
        self.assertEqual(rig.a.requests[-1]["chat_template_kwargs"], {"enable_thinking": False})
        hist += [a3, {"role": "user", "content": "once more"}]
        rig.chat(hist, headers={"X-Cockpit-Escalate": "big"})
        self.assertEqual(len(rig.b.requests), 1)
        self.assertEqual([a["tier"] for a in rig.attempts()], ["fast", "think", "fast", "fast", "big"])
        self.assertEqual(len(rig.handoffs()), 1)
        self.assert_replays()

    def test_regenerations_rise_then_cross(self):
        rig = self.rig
        msgs = [{"role": "user", "content": "hi"}]
        rig.chat(msgs)
        rig.chat(msgs)
        self.assertEqual(rig.decisions()[-1]["reason"], "regenerations:1")
        self.assertEqual(rig.attempts()[-1]["tier"], "think")
        rig.chat(msgs)
        self.assertEqual(rig.decisions()[-1]["reason"], "regenerations:2")
        self.assertEqual([a["tier"] for a in rig.attempts()], ["fast", "think", "think", "big"])
        self.assertEqual(len(rig.b.requests), 1)
        self.assertIn("asked for this turn 3 times", rig.handoffs()[0]["packet"]["escalation_reason"]["text"])
        self.assert_replays()

    def test_tool_errors_rise_then_cross(self):
        rig = self.rig
        calls = [{"id": f"c{i}", "type": "function", "function": {"name": "run", "arguments": "{}"}} for i in range(4)]
        results = [{"role": "tool", "tool_call_id": f"c{i}", "content": f"Error: step {i} failed"} for i in range(4)]
        base = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": None, "tool_calls": calls[:2]}]
        rig.chat(base + results[:2])
        self.assertEqual((rig.attempts()[-1]["tier"], rig.decisions()[-1]["reason"]), ("think", "tool_errors:2"))
        hist = [{"role": "user", "content": "do it again"}, {"role": "assistant", "content": None, "tool_calls": calls}] + results
        rig.chat(hist)
        self.assertEqual((rig.attempts()[-1]["tier"], rig.decisions()[-1]["reason"]), ("big", "tool_errors:4"))
        self.assertEqual(rig.handoffs()[-1]["packet"]["tool_state"]["pending"], [])
        self.assert_replays()

    def test_stuck_claim_rises(self):
        rig = self.rig
        a1 = rig.chat([{"role": "user", "content": "I am stuck on this"}])
        self.assertEqual(a1["content"], "I can't do that.")
        rig.chat([{"role": "user", "content": "I am stuck on this"}, a1, {"role": "user", "content": "please try"}])
        self.assertEqual((rig.attempts()[-1]["tier"], rig.decisions()[-1]["reason"]), ("think", "model_claim:stuck"))
        self.assert_replays()

    def test_pending_tool_blocks_the_crossing(self):
        rig = self.rig
        hist = [{"role": "user", "content": "do it"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]}]
        rig.chat(hist, model="cockpit/big")
        self.assertEqual((len(rig.a.requests), len(rig.b.requests)), (1, 0))
        d = rig.decisions()[-1]
        self.assertEqual((d["tier"], d["reason"]), ("fast", "named:big,blocked:pending_tool"))
        self.assertTrue(json.loads(d["inputs"])["pending_tool"])
        self.assertEqual(rig.handoffs(), [])
        self.assert_replays()

    def test_crossed_tier_is_sticky_and_its_prefix_stable(self):
        rig = self.rig
        a1 = rig.chat([{"role": "user", "content": "hi"}])
        hist = [{"role": "user", "content": "hi"}, a1, {"role": "user", "content": "more"}]
        a2 = rig.chat(hist, model="cockpit/big")
        hist += [a2, {"role": "user", "content": "and more"}]
        rig.chat(hist)
        self.assertEqual(len(rig.b.requests), 2)
        first, second = rig.b.requests
        self.assertEqual(first["messages"][0], second["messages"][0])
        self.assertEqual([m["role"] for m in second["messages"]], ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual([a["tier"] for a in rig.attempts()], ["fast", "fast", "big", "big"])
        self.assertEqual(rig.decisions()[-1]["reason"], "floor")
        self.assertEqual(len(rig.handoffs()), 1)
        hist += [rig.chat(hist), {"role": "user", "content": "back to fast"}]
        rig.chat(hist, model="fast")
        self.assertEqual([(a["tier"], a["state"]) for a in rig.attempts()][-3:],
                         [("big", "completed"), ("big", "escalated"), ("fast", "completed")])
        self.assertEqual(len(rig.handoffs()), 2)
        self.assertEqual(rig.a.requests[-1]["messages"][0]["role"], "user")   # back on the origin: the raw history
        self.assertEqual(len(rig.a.requests[-1]["messages"]), len(hist))
        self.assert_replays()

    def test_the_same_conversation_gives_the_same_packet_bytes(self):
        def run(rig):
            a1 = rig.chat([{"role": "user", "content": "hi"}])
            rig.chat([{"role": "user", "content": "hi"}, a1, {"role": "user", "content": "more"}], model="cockpit/big")
            return rig.handoffs()[0]

        other = Rig()
        try:
            h1, h2 = run(self.rig), run(other)
        finally:
            other.stop()
        self.assertEqual(h1["packet"], h2["packet"])
        self.assertEqual(h1["packet_sha256"], h2["packet_sha256"])
        self.assertEqual(h1["idempotency_key"], h2["idempotency_key"])
        self.assertNotEqual(h1["handoff_id"], h2["handoff_id"])

    def test_no_ledger_routes_on_the_request_alone(self):
        rig = Rig(ledger=False)
        try:
            rig.chat([{"role": "user", "content": "hi"}], model="cockpit/big")
            self.assertEqual((len(rig.a.requests), len(rig.b.requests)), (0, 1))
            self.assertEqual(rig.b.requests[0]["messages"][0]["role"], "user")
            rig.chat([{"role": "user", "content": "hi"}], headers={"X-Cockpit-Escalate": "1"})
            self.assertEqual(rig.a.requests[-1]["chat_template_kwargs"], {"enable_thinking": True})
        finally:
            rig.stop()


class CompactionTest(unittest.TestCase):
    def test_compaction_after_the_context_threshold(self):
        rig = Rig(fast_ctx=200)                  # three quarters of 200 is 150; "hi" costs 26
        try:
            long = "w" * 600                     # 600 characters: 150 more tokens in the fake's count
            a1 = rig.chat([{"role": "user", "content": long}])
            self.assertEqual(rig.kinds(), ["message", "request", "reply", "briefing"])
            self.assertEqual(rig.a.requests[-1]["response_format"]["json_schema"]["name"], "briefing")
            b = json.loads(rows(rig.path, "SELECT payload FROM events WHERE kind='briefing'")[0]["payload"])
            self.assertEqual((b["covers_through_seq"], b["tier"], b["objective"][:10]), (3, "fast", "w" * 10))
            rig.chat([{"role": "user", "content": long}, a1, {"role": "user", "content": "next"}], model="cockpit/big")
            h = rig.handoffs()[0]
            self.assertEqual(h["packet"]["briefing"]["tried"], [{"what": "answered", "observed": "ok"}])
            self.assertEqual(h["packet"]["coverage"]["briefing_seq"], 3)   # what it covers: through the reply
            self.assertEqual([e["content"] for e in h["packet"]["events_since_briefing"]], ["next"])
            sent = rig.b.requests[0]["messages"]
            self.assertIn("Briefing from the previous model", sent[0]["content"])
            self.assertEqual([m["role"] for m in sent], ["system", "user"])
            ok, lines = verify(rig.path)
            self.assertTrue(ok, "\n".join(lines))
        finally:
            rig.stop()

    def test_packet_too_big_compacts_first_then_crosses(self):
        rig = Rig(big_ctx=400)                   # the packet must fit in 360 tokens
        try:
            long = "w" * 1500                    # a paste of about 375 tokens, mid-conversation
            a1 = rig.chat([{"role": "user", "content": "hi"}])
            hist = [{"role": "user", "content": "hi"}, a1, {"role": "user", "content": long}]
            a2 = rig.chat(hist)
            hist += [a2, {"role": "user", "content": "short"}]
            rig.chat(hist, model="cockpit/big")
            self.assertEqual(rig.a.requests[-1]["response_format"]["json_schema"]["name"], "briefing")
            compacted = rig.a.requests[-1]["messages"]
            self.assertEqual([m["role"] for m in compacted], ["user", "assistant", "user", "assistant", "user"])
            self.assertEqual(compacted[-1]["content"].split("\n")[0], "Summarize the conversation so far for another model that will continue it. Answer with JSON only, in this shape: objective (the user's goal, one sentence), constraints (the user's stated requirements, quoted), tried (what was attempted and what each attempt showed), open (what is still unresolved), pending_tool_effects (tool calls whose outcome is not yet known). Do not add facts the conversation does not contain.")
            self.assertEqual(len(rig.b.requests), 1)
            h = rig.handoffs()[0]
            self.assertIsNotNone(h["packet"]["briefing"])
            self.assertTrue(h["packet"]["budget"]["fits"])
            self.assertEqual([e["content"] for e in h["packet"]["events_since_briefing"]], ["short"])
            b = json.loads(rows(rig.path, "SELECT payload FROM events WHERE kind='briefing'")[0]["payload"])
            self.assertEqual(b["covers_through_seq"], 6)
            self.assertEqual(rig.kinds(), ["message", "request", "reply", "message", "request", "reply", "message", "request", "briefing", "handoff", "request", "reply"])
            sent = rig.b.requests[0]["messages"]
            self.assertNotIn("Original task", sent[0]["content"])
            self.assertIn("Objective: hi", sent[0]["content"])
            ok, lines = verify(rig.path)
            self.assertTrue(ok, "\n".join(lines))
        finally:
            rig.stop()

    def test_packet_that_cannot_fit_stays_on_the_floor(self):
        rig = Rig(big_ctx=40)
        try:
            a1 = rig.chat([{"role": "user", "content": "hi"}])
            rig.chat([{"role": "user", "content": "hi"}, a1, {"role": "user", "content": "more"}], model="cockpit/big")
            self.assertEqual(len(rig.b.requests), 0)
            d = rig.decisions()[-1]
            self.assertEqual((d["tier"], d["reason"]), ("fast", "named:big,blocked:does_not_fit"))
            self.assertIn("packet_estimated_tokens", json.loads(d["inputs"]))
            self.assertEqual(rig.handoffs(), [])
            self.assertEqual([a["state"] for a in rig.attempts()], ["completed", "completed"])
            ok, lines = verify(rig.path)
            self.assertTrue(ok, "\n".join(lines))
        finally:
            rig.stop()


if __name__ == "__main__":
    unittest.main()
