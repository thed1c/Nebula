"""Starweave phase 3: the reviewer's verdict (approve, or reject with blocking issues), the rework loop, the team's
questions to the owner as cards whose answers reach the lead's next turn, and the probe's measured evidence."""
import json
import re
import time
import unittest
import urllib.request

from nebula import ledger as L
from nebula import profiles, team
from nebula.tests.test_escalation import Rig, rows
from nebula.tests.test_starweave_work import lead, tool_names


def mode_of(body):
    msgs = body.get("messages") or []
    first = str(msgs[0].get("content")) if msgs and msgs[0].get("role") == "system" else ""
    for m, text in team.MODE_TEXT.items():
        if text in first:
            return m
    return None


def verdict(v, issues=(), summary="ok"):
    return json.dumps({"verdict": v, "summary": summary, "blocking_issues": list(issues), "followups": []})


class ReviewTest(unittest.TestCase):
    def rig(self, mode="on", **rules):
        r = Rig(rules={"starweave": {"mode": mode, "reviewer": "big", **rules}})
        self.addCleanup(r.stop)
        return r

    def items(self, rig):
        return rows(rig.path, "SELECT * FROM work_items ORDER BY created_at, rowid")

    def test_the_lead_offers_review_and_ask_owner(self):
        rig = self.rig()
        rig.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(tool_names(rig.a.requests[-1]),
                         [team.HAND_UP, team.DELEGATE, team.COLLECT, team.ASK, team.REVIEW, team.ASK_OWNER, team.NOTE])

    def test_a_review_of_the_leads_own_draft(self):
        rig = self.rig()
        rig.a.script = lead("draft", (team.REVIEW, {"work": "My draft.", "criteria": "Clear and short."}))
        seen = []

        def big(body):
            if mode_of(body) == "review":
                seen.append(body)
                return verdict("approve", summary="clear")
            return None
        rig.b.script = big
        reply = rig.chat([{"role": "user", "content": "check my draft"}])
        self.assertEqual(reply["content"], "Tool said: verdict: approve\nsummary: clear")
        (body,) = seen
        self.assertEqual(body["response_format"]["json_schema"]["name"], "verdict")
        self.assertIn("Criteria:\nClear and short.", body["messages"][-1]["content"])
        self.assertIn("The work:\nMy draft.", body["messages"][-1]["content"])
        self.assertEqual(tool_names(body), [])                       # a reviewer only answers with its verdict
        (item,) = self.items(rig)
        self.assertEqual((item["mode"], item["label"], item["state"]), ("review", "review-1", "done"))

    def test_a_rejected_item_is_reworked_then_approved(self):
        rig = self.rig()
        verdicts = iter([verdict("reject", ["it is too short"]), verdict("approve", summary="now complete")])

        def then(body):
            said = body["messages"][-1]["content"]
            if said.startswith("[d] think: done"):
                return [{"name": team.REVIEW, "arguments": {"item": "d", "criteria": "complete"}}]
            return None
        lead_script = lead("write", (team.DELEGATE, {"items": [{"id": "d", "to": "think", "task": "Write the intro."}]}), then=then)
        reworks = []

        def a(body):
            if mode_of(body) == "rework":
                reworks.append(body)
                return "A longer intro."
            return lead_script(body)
        rig.a.script = a
        rig.b.script = lambda body: next(verdicts) if mode_of(body) == "review" else None
        reply = rig.chat([{"role": "user", "content": "write it"}])
        self.assertIn("[d] approved by big after 1 rework.", reply["content"])
        self.assertIn("The approved result:\nA longer intro.", reply["content"])
        (rw,) = reworks
        self.assertIn("Blocking issues to fix:\n- it is too short", rw["messages"][-1]["content"])
        self.assertIn("Your result:\nHello world", rw["messages"][-1]["content"])
        d = next(i for i in self.items(rig) if i["label"] == "d")
        self.assertEqual((d["state"], d["rework_count"], d["result"], json.loads(d["verdict"])["verdict"]),
                         ("done", 1, "A longer intro.", "approve"))
        sid = rig.srv.ledger.summaries()[0]["session_id"]
        reply_item = [i for i in rig.srv.ledger.view(sid)["items"] if i["kind"] == "reply"][-1]
        self.assertEqual(reply_item["team"], "Lead: fast · delegated to think · reviewed by big (approved, 1 rework)")

    def test_rejected_past_the_rework_rounds_goes_back_to_the_lead(self):
        rig = self.rig(max_rework=1)

        def then(body):
            if body["messages"][-1]["content"].startswith("[d] think: done"):
                return [{"name": team.REVIEW, "arguments": {"item": "d", "criteria": "complete"}}]
            return None
        lead_script = lead("write", (team.DELEGATE, {"items": [{"id": "d", "to": "think", "task": "Write."}]}), then=then)
        rig.a.script = lambda body: "Still short." if mode_of(body) == "rework" else lead_script(body)
        rig.b.script = lambda body: verdict("reject", ["too short"]) if mode_of(body) == "review" else None
        reply = rig.chat([{"role": "user", "content": "write it"}])
        self.assertIn("[d] not approved by big after 1 rework. Decide:", reply["content"])
        self.assertEqual(len(re.findall(r"Verdict \d:", reply["content"])), 2)

    def test_a_verdict_that_does_not_hold_is_asked_again_once(self):
        rig = self.rig()
        rig.a.script = lead("draft", (team.REVIEW, {"work": "W.", "criteria": "C."}))
        said = iter([verdict("reject", []), verdict("approve")])
        seen = []

        def big(body):
            if mode_of(body) == "review":
                seen.append(body)
                return next(said)
            return None
        rig.b.script = big
        reply = rig.chat([{"role": "user", "content": "check my draft"}])
        self.assertIn("verdict: approve", reply["content"])
        self.assertEqual(len(seen), 2)
        self.assertIn("a reject needs at least one blocking issue", seen[1]["messages"][-1]["content"])

    def test_without_a_reviewer_review_is_not_offered_and_is_refused(self):
        rig = Rig(rules={"starweave": {"mode": "on"}})
        self.addCleanup(rig.stop)
        rig.chat([{"role": "user", "content": "hi"}])
        self.assertNotIn(team.REVIEW, tool_names(rig.a.requests[-1]))

    def test_the_owner_is_asked_and_the_answer_reaches_the_next_turn(self):
        rig = self.rig()
        rig.a.script = lead("release", (team.ASK_OWNER, {"question": "Publish 0.1.7 now?", "options": ["Publish", "Hold"]}))
        first = [{"role": "user", "content": "release it"}]
        r1 = rig.chat(first)
        self.assertIn("Asked the owner", r1["content"])
        (item,) = self.items(rig)
        self.assertEqual((item["mode"], item["state"], json.loads(item["verdict"])["options"]), ("ask_owner", "awaiting_owner", ["Publish", "Hold"]))
        req = urllib.request.Request(rig.base + "/api/starweave", headers={})
        with urllib.request.urlopen(req, timeout=10) as r:
            v = json.loads(r.read())
        self.assertEqual([(q["question"], q["options"]) for q in v["owner_asks"]], [("Publish 0.1.7 now?", ["Publish", "Hold"])])
        sid = rig.srv.ledger.summaries()[0]["session_id"]
        reply_item = [i for i in rig.srv.ledger.view(sid)["items"] if i["kind"] == "reply"][-1]
        self.assertIn("asked you 1 (1 waiting)", reply_item["team"])
        self.assertEqual(reply_item["work_items"][0]["options"], ["Publish", "Hold"])
        body = json.dumps({"item_id": item["item_id"], "choice": "Hold"}).encode()
        req = urllib.request.Request(rig.base + "/api/starweave/answer", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            self.assertEqual(json.loads(r.read())["owner_asks"], [])
        rig.a.script = None
        rig.chat(first + [r1, {"role": "user", "content": "and now?"}])
        system = rig.a.requests[-1]["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("The owner answered your team's question “Publish 0.1.7 now?”: Hold", system["content"])
        self.assertIn("team_note", rig.kinds())
        rig.chat(first + [r1, {"role": "user", "content": "and now?"}, {"role": "assistant", "content": "Hello world"},
                          {"role": "user", "content": "ok"}])
        self.assertNotIn("The owner answered", json.dumps(rig.a.requests[-1]["messages"]))   # read once
        with self.assertRaises(ValueError):
            rig.srv.ledger.answer_owner(item["item_id"], "Publish")

    def test_practice_asks_nobody(self):
        rig = self.rig("practice")
        rig.a.script = lead("release", (team.ASK_OWNER, {"question": "Publish?", "options": ["Yes", "No"]}))
        reply = rig.chat([{"role": "user", "content": "release it"}])
        self.assertIn("Practice mode: the question was recorded but the owner was not asked", reply["content"])
        self.assertEqual(self.items(rig), [])


class ProbeTest(unittest.TestCase):
    def test_the_probe_measures_speed_json_and_tool_calls(self):
        rig = Rig(rules={"starweave": {"mode": "on"}})
        self.addCleanup(rig.stop)

        def big(body):
            fmt = (body.get("response_format") or {}).get("json_schema") or {}
            if fmt.get("name") == "probe":
                return json.dumps({"answer": "Canberra", "confidence": 0.9})
            names = tool_names(body)
            if "get_time" in names:
                city = re.search(r"in (\w+)\?", body["messages"][-1]["content"]).group(1)
                return [{"name": "get_time", "arguments": {"city": city}}]
            return None
        rig.b.script = big
        with urllib.request.urlopen(rig.base + "/api/starweave", timeout=10) as r:
            self.assertGreater(json.loads(r.read())["probe_estimate"]["big"], 0)
        req = urllib.request.Request(rig.base + "/api/starweave/probe", data=json.dumps({"model": "big"}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=10).read()
        end = time.time() + 30
        while time.time() < end and (profiles.load(rig.srv.cfg)["profiles"].get("big", {}).get("probe") or {}).get("state") == "running":
            time.sleep(0.2)
        prof = profiles.load(rig.srv.cfg)["profiles"]["big"]
        self.assertEqual(prof["probe"]["state"], "done", prof["probe"])
        m = prof["measured"]
        self.assertEqual((m["json_ok"], m["tools_ok"], m["calls"], m["tok_s"]), (1.0, 1.0, 11, 40.0))
        self.assertEqual(len(rig.b.requests), 11)
        sc, why = profiles.score("big", "larger", prof, None, None)
        self.assertAlmostEqual(sc, profiles.WEIGHT_MEASURED * 1.0)
        self.assertIn("40 tokens/s, JSON 5/5, tools 5/5", why[0])
        attempts = rows(rig.path, "SELECT dialect FROM attempts")
        self.assertTrue(all(a["dialect"] == "probe" for a in attempts))


class TeamLineReviewTest(unittest.TestCase):
    def test_line(self):
        work = [{"mode": "review", "state": "done", "model": "check", "result": verdict("reject", ["x"])}]
        self.assertEqual(L.team_line("big", [{"action": "review", "ruling": "allowed", "mode": "on"}], work),
                         "Lead: big · reviewed by check (rejected)")



class LeadReviewsTest(unittest.TestCase):
    """The lead's own model may review an item a teammate did (never its own draft, never the item's own model)."""

    def test_the_lead_reviews_its_teammates_item(self):
        rig = Rig(rules={"starweave": {"mode": "on", "reviewer": "fast"}})
        self.addCleanup(rig.stop)

        def then(body):
            if body["messages"][-1]["content"].startswith("[d] think: done"):
                return [{"name": team.REVIEW, "arguments": {"item": "d", "criteria": "complete"}}]
            return None
        lead_script = lead("write", (team.DELEGATE, {"items": [{"id": "d", "to": "think", "task": "Write it."}]}), then=then)
        reviews = []

        def a(body):
            if mode_of(body) == "review":
                reviews.append(body)
                return verdict("approve", summary="fine")
            return lead_script(body)
        rig.a.script = a
        reply = rig.chat([{"role": "user", "content": "write it"}])
        self.assertIn(team.REVIEW, tool_names(rig.a.requests[0]))
        self.assertIn("[d] approved by fast.", reply["content"])
        self.assertEqual(len(reviews), 1)
        rv = next(i for i in rows(rig.path, "SELECT * FROM work_items") if i["mode"] == "review")
        self.assertEqual((rv["model"], rv["state"]), ("fast", "done"))

    def test_the_lead_does_not_review_its_own_draft(self):
        rig = Rig(rules={"starweave": {"mode": "on", "reviewer": "fast"}})
        self.addCleanup(rig.stop)
        seen = []

        def then(body):
            seen.append(body["messages"][-1]["content"])
            return None
        rig.a.script = lead("draft", (team.REVIEW, {"work": "My draft.", "criteria": "Short."}), then=then)
        rig.chat([{"role": "user", "content": "check my draft"}])
        self.assertTrue(any("you are the reviewer here" in x for x in seen), seen)


class TokenCapTest(unittest.TestCase):
    def test_free_command_tiers_stay_out_of_the_cap(self):
        from types import SimpleNamespace
        from nebula import teamwork
        from nebula.config import Tier
        tiers = {"codex": Tier("codex", cli="codex"), "paid": Tier("paid", cli="claude", cost_per_mtoken=3.0), "big": Tier("big", engine="http://x")}
        turn = SimpleNamespace(cfg=SimpleNamespace(tier=tiers.get))
        items = [{"model": "codex", "tokens": 4_000_000}, {"model": "paid", "tokens": 10}, {"model": "big", "tokens": 5}]
        self.assertEqual(teamwork.capped_tokens(turn, items), 15)

    def test_cached_input_is_not_counted(self):
        from nebula import teamwork
        self.assertEqual(teamwork.uncached_tokens({"prompt_tokens": 1000, "cached_tokens": 900, "completion_tokens": 50}), 150)

if __name__ == "__main__":
    unittest.main()
