"""Starweave phase 1 (nebula/team.py): the answering model hands work up (this answer) and down (the next turns),
Halo rules on each move, Practice records without moving, and every call is a ledger row."""
import json
import unittest

from nebula import team
from nebula.replay import verify
from nebula.router import Rules
from nebula.tests.test_escalation import Rig, rows


def tool_names(body):
    return [t.get("function", t).get("name") for t in body.get("tools") or []]


def last_user(body):
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            return m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
    return ""


def wants(to, word, call):
    """A fake model that makes `call` to `to` when the last user text holds `word` and the tool is offered."""
    def script(body):
        msgs = body.get("messages") or []
        if msgs and msgs[-1].get("role") == "tool":
            return None
        if call in tool_names(body) and word in last_user(body):
            return [{"name": call, "arguments": {"to": to, "why": "it needs more depth"}}]
        return None
    return script


class StarweaveTest(unittest.TestCase):
    def rig(self, mode="on", **rules):
        r = Rig(rules={"starweave": {"mode": mode, **rules}})
        self.addCleanup(r.stop)
        return r

    def assert_replays(self, rig):
        ok, lines = verify(rig.path)
        self.assertTrue(ok, "\n".join(lines))

    def actions(self, rig):
        return [json.loads(r["payload"]) for r in rows(rig.path, "SELECT payload FROM events WHERE kind='team_action' ORDER BY seq")]

    def test_off_offers_no_team_tools(self):
        rig = self.rig("off")
        rig.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(tool_names(rig.a.requests[-1]), [])

    def test_on_offers_hand_up_to_the_models_above(self):
        rig = self.rig()
        rig.chat([{"role": "user", "content": "hi"}])
        body = rig.a.requests[-1]
        self.assertEqual(tool_names(body), [team.HAND_UP, team.DELEGATE, team.COLLECT, team.ASK, team.ASK_OWNER, team.NOTE])   # nothing below fast
        spec = body["tools"][0]["function"]
        self.assertEqual(spec["parameters"]["properties"]["to"]["enum"], ["think", "big"])

    def test_hand_up_to_another_engine_finishes_the_same_answer(self):
        rig = self.rig()
        rig.a.script = wants("big", "hard", team.HAND_UP)
        reply = rig.chat([{"role": "user", "content": "this is hard"}])
        self.assertEqual(reply["content"], "Hello world")             # the larger model answered, in the same turn
        self.assertEqual(len(rig.b.requests), 1)
        sent = rig.b.requests[0]["messages"]
        self.assertIn("fast handed this turn to you: it needs more depth", sent[0]["content"])
        self.assertNotIn(team.HAND_UP, tool_names(rig.b.requests[0]))  # big is the top: nothing to hand up to
        self.assertIn(team.HAND_DOWN, tool_names(rig.b.requests[0]))
        attempts = rig.attempts()
        self.assertEqual([(a["tier"], a["state"]) for a in attempts], [("fast", "escalated"), ("big", "completed")])
        self.assertIn("team:hand_up:fast->big", [d["reason"] for d in rig.decisions()])
        acts = self.actions(rig)
        self.assertEqual([(a["action"], a["from"], a["to"], a["ruling"], a["mode"]) for a in acts],
                         [("hand_up", "fast", "big", "allowed", "on")])
        h = rig.handoffs()
        self.assertEqual(len(h), 1)
        self.assertEqual(h[0]["packet"]["escalation_reason"]["source"], "team:hand_up:fast")
        self.assert_replays(rig)

    def test_the_session_stays_up_after_a_hand_up(self):
        rig = self.rig()
        rig.a.script = wants("big", "hard", team.HAND_UP)
        first = [{"role": "user", "content": "this is hard"}]
        r1 = rig.chat(first)
        rig.chat(first + [r1, {"role": "user", "content": "and then?"}])
        self.assertEqual(len(rig.b.requests), 2)                       # the floor is now big
        self.assert_replays(rig)

    def test_hand_up_on_the_same_engine_re_tiers_the_attempt(self):
        rig = self.rig()
        rig.a.script = wants("think", "hard", team.HAND_UP)
        reply = rig.chat([{"role": "user", "content": "this is hard"}])
        self.assertEqual(reply["content"], "Tool said: Handed to think.")
        self.assertEqual(len(rig.b.requests), 0)
        self.assertEqual([(a["tier"], a["state"]) for a in rig.attempts()], [("think", "completed")])
        self.assertEqual(rig.a.requests[-1]["chat_template_kwargs"]["enable_thinking"], True)
        spec = {t["function"]["name"]: t["function"] for t in rig.a.requests[-1]["tools"]}
        self.assertEqual(spec[team.HAND_UP]["parameters"]["properties"]["to"]["enum"], ["big"])     # offered from think now
        self.assertEqual(spec[team.HAND_DOWN]["parameters"]["properties"]["to"]["enum"], ["fast"])
        self.assert_replays(rig)

    def test_practice_records_the_move_and_makes_none(self):
        rig = self.rig("practice")
        rig.a.script = wants("big", "hard", team.HAND_UP)
        reply = rig.chat([{"role": "user", "content": "this is hard"}])
        self.assertIn("Practice mode: the move to big was recorded but not made", reply["content"])
        self.assertEqual(len(rig.b.requests), 0)
        self.assertEqual([(a["ruling"], a["mode"]) for a in self.actions(rig)], [("allowed", "practice")])
        self.assert_replays(rig)

    def test_halo_refuses_a_move_that_is_not_up_the_ladder(self):
        rig = self.rig()
        rig.a.script = wants("nowhere", "hard", team.HAND_UP)
        reply = rig.chat([{"role": "user", "content": "this is hard"}])
        self.assertIn("Not moved: nowhere is not above you on this request's ladder; you may hand to think, big", reply["content"])
        self.assertEqual(len(rig.b.requests), 0)
        self.assertEqual([a["ruling"] for a in self.actions(rig)], ["refused"])
        self.assert_replays(rig)

    def test_the_hand_move_budget_holds(self):
        rig = self.rig(max_hand_moves=0)
        rig.a.script = wants("big", "hard", team.HAND_UP)
        reply = rig.chat([{"role": "user", "content": "this is hard"}])
        self.assertIn("already used its 0 hand moves", reply["content"])
        self.assertEqual(len(rig.b.requests), 0)

    def test_a_named_model_is_never_moved_but_may_lead(self):
        rig = self.rig()
        rig.chat([{"role": "user", "content": "hi"}], model="nebula/big")
        names = tool_names(rig.b.requests[-1])
        self.assertNotIn(team.HAND_UP, names)
        self.assertNotIn(team.HAND_DOWN, names)
        self.assertEqual(names, [team.DELEGATE, team.COLLECT, team.ASK, team.ASK_OWNER, team.NOTE])

    def test_hand_down_moves_the_next_turn_to_the_smaller_model_from_a_briefing(self):
        rig = self.rig()
        rig.b.script = wants("fast", "done", team.HAND_DOWN)
        first = [{"role": "user", "content": "the hard part is done"}]
        r1 = rig.chat(first, headers={"X-Nebula-Escalate": "big"})
        self.assertEqual(r1["content"], "Tool said: Noted: after this answer, fast takes the next turns. Finish your answer now.")
        self.assertEqual(len(rig.b.requests), 3)                       # the call, the answer after it, and the briefing
        briefs = [b for b in rig.b.requests if ((b.get("response_format") or {}).get("json_schema") or {}).get("name") == "briefing"]
        self.assertEqual(len(briefs), 1)                               # no secretary set: the answering model writes it
        self.assertIn("briefing", rig.kinds())                          # written before the next turn moves down
        before = len(rig.b.requests)
        rig.chat(first + [r1, {"role": "user", "content": "thanks, one more small thing"}])
        self.assertEqual(len(rig.b.requests), before)                  # big did not answer the next turn
        last = rig.decisions()[-1]
        self.assertIn("team:hand_down:fast", last["reason"])
        self.assertEqual(rig.attempts()[-1]["tier"], "fast")
        # and only that next turn: the hand-down is used once
        r2 = rig.chat(first + [r1, {"role": "user", "content": "thanks, one more small thing"}, {"role": "assistant", "content": "Hello world"},
                               {"role": "user", "content": "ok"}])
        self.assertEqual(r2["content"], "Hello world")
        self.assertNotIn("team:hand_down", rig.decisions()[-1]["reason"])
        self.assert_replays(rig)


class SettingsTest(unittest.TestCase):
    def test_settings_default_off(self):
        self.assertEqual({k: v for k, v in team.settings(Rules.from_dict({})).items() if k in ("mode", "max_hand_moves")},
                         {"mode": "off", "max_hand_moves": 4})
        self.assertEqual(team.settings(Rules.from_dict({"starweave": {"mode": "on", "max_hand_moves": 2}}))["max_hand_moves"], 2)

    def test_check_settings(self):
        self.assertEqual(team.check_settings({"mode": "practice"}), {"mode": "practice"})
        with self.assertRaises(team.TeamError):
            team.check_settings({"mode": "sometimes"})
        with self.assertRaises(team.TeamError):
            team.check_settings({"mode": "on", "max_hand_moves": 99})


if __name__ == "__main__":
    unittest.main()
