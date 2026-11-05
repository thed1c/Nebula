"""Team contracts (nebula/contracts.py): each Star Weaving team has a name, the routes it serves, and one page that says
what it is for, who does what, what it may spend and touch, what done means and when it asks. Halo holds a team task
to the contract its task took when it started: roles, independent review, budgets, tools per role and done checks are
enforced in code; the rest is the lead's instructions."""
import json
import os
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock

from nebula import contracts as C
from nebula import team, teamtasks, toolbox
from nebula.tests.test_escalation import Rig, rows
from nebula.tests.test_starweave_review import mode_of, verdict
from nebula.tests.test_starweave_work import is_teammate, last_user, tool_names
from nebula.tests.test_toolbox import FakeBox


def wait(pred, s=20):
    end = time.time() + s
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def code_team(**over):
    c = {"name": "Code team", "purpose": "Builds and fixes code with tests.", "examples": ["fix a failing test"],
         "routes": ["everyday"], "lead": "fast", "helpers": ["think"], "reviewer": "big", "others": False}
    c.update(over)
    return c


class ShapeTest(unittest.TestCase):
    def test_a_blank_part_falls_back_and_the_old_per_route_shape_reads_as_a_team(self):
        teams = C.teams({"starweave": {"teams": {"larger": {"lead": "big", "helpers": ["codex"], "reviewer": "big"}}}})
        c = teams["larger"]
        self.assertEqual((c["name"], c["routes"], c["lead"], c["helpers"], c["reviewer"]), ("Larger", ["larger"], "big", ["codex"], "big"))
        self.assertTrue(c["others"])                                    # today: anyone in the pool may help
        self.assertEqual(c["budgets"], {})                              # Star Weaving's own numbers
        self.assertEqual(c["done"], {"tests": "", "review": False, "report": False, "council": False})   # done when the lead says so
        self.assertEqual(C.limits({"max_task_tokens": 5}, c), {"max_task_tokens": 5})

    def test_named_teams_serve_routes_and_a_route_can_have_two(self):
        rules = {"starweave": {"teams": {"code": code_team(routes=["larger"]), "research": code_team(name="Research", routes=["larger", "writing"])}}}
        self.assertEqual([k for k, _c in C.for_route(rules, "larger")], ["code", "research"])
        self.assertEqual([k for k, _c in C.for_route(rules, "writing")], ["research"])
        self.assertEqual(C.for_route(rules, "critique"), [])

    def test_save_refuses_with_the_reason(self):
        names = ["fast", "think", "big"]
        bad = [({"helpers": ["think"]}, "every team has a lead"),
               (code_team(lead="nope"), "no model 'nope'"),
               (code_team(helpers=[], reviewer="fast"), "would review only its own work"),
               (code_team(budgets={"max_task_tokens": 5}), "max_task_tokens is a number from 10000"),
               (code_team(routes=["nowhere"]), "is not a route"),
               (code_team(tools={"lead": "everything"}), "write, read or none"),
               (code_team(reviewer=None, helpers=[], done={"review": True}), "needs a reviewer"),
               (code_team(helpers=["fast"]), "it is not also a helper"),
               (code_team(tools={"folder": "relative/dir"}), "full path"),
               (code_team(**{"class": "band"}), "the class is one of coding, council, research"),
               (code_team(helpers=[], done={"council": True}), "a council needs a crew")]
        for raw, why in bad:
            with self.assertRaises(C.ContractError) as e:
                C.check("code", raw, names)
            self.assertIn(why, str(e.exception))
        with self.assertRaises(C.ContractError):
            C.check("bad id!", code_team(), names)
        ok = C.check("code", code_team(budgets={"max_task_tokens": 3_000_000}), names)
        self.assertEqual(ok["budgets"], {"max_task_tokens": 3_000_000})
        self.assertNotIn("stand_ins", ok)                                # blank parts are left out of the config

    def test_a_reviewer_never_writes(self):
        c = C.normalise("code", code_team(tools={"reviewer": "write"}))
        self.assertEqual(c["tools"]["reviewer"], "read")

    def test_classes_that_change_nothing_hold_every_role_to_read(self):
        for k in C.READ_ONLY_CLASSES:
            c = C.normalise("t", code_team(**{"class": k, "tools": {"lead": "write", "helpers": None, "reviewer": "none"}}))
            self.assertEqual((c["tools"]["lead"], c["tools"]["helpers"], c["tools"]["reviewer"]), ("read", "read", "none"))
        c = C.normalise("t", code_team(**{"class": "coding", "tools": {"lead": "write"}}))
        self.assertEqual(c["tools"]["lead"], "write")

    def test_the_version_follows_the_contract(self):
        a = C.normalise("code", code_team())
        self.assertEqual(C.version(a), C.version(C.compact(a)))
        self.assertEqual(C.version(a), C.version(C.snapshot("code", a)))
        self.assertNotEqual(C.version(a), C.version(C.normalise("code", code_team(house_rules="Never commit."))))

    def test_templates_fill_in_the_members(self):
        d = C.template("coding", lead="big", helpers=["codex"], reviewer="codex", routes=["larger"])
        self.assertEqual((d["lead"], d["helpers"], d["reviewer"], d["routes"]), ("big", ["codex"], "codex", ["larger"]))
        self.assertTrue(d["done"]["review"])
        C.check("code", d, ["big", "codex"])                             # a template's draft saves as it is
        self.assertEqual(sorted(C.TEMPLATES), sorted(C.CLASSES))     # one per class
        for name in C.TEMPLATES:
            C.check(name, C.template(name, lead="big", helpers=["codex"], reviewer="codex"), ["big", "codex"])
        here = ["big", "codex", "writer"]                         # only a lead picked: the page's New team buttons
        for name in C.TEMPLATES:
            d = C.template(name, lead="big", routes=["everyday"], models=here)
            C.check(name, d, here)
            self.assertNotEqual(d.get("reviewer"), "big")
        self.assertEqual(C.template("council", lead="big", models=here)["helpers"], ["codex", "writer"])
        self.assertEqual(C.template("coding", lead="big", models=here)["reviewer"], "codex")
        with self.assertRaises(C.ContractError):
            C.template("nope")

    def test_the_card_reads_in_plain_sentences(self):
        lines = C.plain(C.normalise("code", code_team(budgets={"max_task_tokens": 3_000_000, "max_background_seconds": 2700},
                                                      done={"tests": "auto", "review": True})))
        self.assertEqual(lines[0], "The Code team builds and fixes code with tests.")
        self.assertIn("fast leads, think helps, big reviews. Only they work on it.", lines)
        self.assertIn("Up to 3M tokens, 45 min.", lines)
        self.assertIn("Done when the tests pass and big approves.", lines)

    def test_the_page_editing_a_route_card_keeps_the_rest_of_the_contract(self):
        had = {"code": code_team(routes=["larger"], house_rules="Never commit.")}
        out = C.merge_route_teams(had, {"larger": {"lead": "big", "helpers": ["fast"], "reviewer": "think"},
                                        "writing": {"lead": "fast", "helpers": [], "reviewer": None}})
        self.assertEqual((out["code"]["lead"], out["code"]["helpers"], out["code"]["reviewer"]), ("big", ["fast"], "think"))
        self.assertEqual(out["code"]["house_rules"], "Never commit.")
        self.assertEqual(out["writing"]["routes"], ["writing"])         # a route with no team gets one


class FirstContractTest(unittest.TestCase):
    def test_larger_becomes_its_first_contract_and_codex_reviews(self):
        sw = {"mode": "on", "max_task_tokens": 5_000_000, "max_task_seconds": 1200, "cloud_cap_usd": 1.0,
              "good_at": {"codex": "fast, careful edits"},
              "teams": {"larger": {"lead": "big", "helpers": ["codex"], "reviewer": "big"}}}
        new = C.first_contracts(sw)["larger"]
        self.assertEqual((new["lead"], new["helpers"], new["reviewer"]), ("big", ["codex"], "codex"))   # the one change
        self.assertEqual(new["routes"], ["larger"])
        self.assertEqual(new["budgets"]["max_task_tokens"], 5_000_000)  # the same numbers, now the team's own
        self.assertEqual(new["budgets"]["max_task_seconds"], 1200)
        self.assertEqual(new["done"], {"tests": "auto", "review": True})
        self.assertEqual(new["tools"], {"lead": "write", "helpers": "write", "reviewer": "read"})
        self.assertEqual(new["good_at"], {"codex": "fast, careful edits"})
        C.check("larger", new, ["big", "codex"])

    def test_convert_writes_the_file_once_with_a_copy_beside_it(self):
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        p = os.path.join(d, "nebula.json")
        before = {"rules": {"starweave": {"mode": "on", "teams": {"larger": {"lead": "big", "helpers": ["codex"], "reviewer": "big"}}}}}
        with open(p, "w") as f:
            json.dump(before, f)
        said = C.convert_file(p)
        self.assertIn("converted 1 team", said)
        self.assertIn("big leads, codex helps, codex reviews.", said)
        with open(p + ".before-contracts.bak") as f:
            self.assertEqual(json.load(f), before)
        with open(p) as f:
            self.assertEqual(json.load(f)["rules"]["starweave"]["teams"]["larger"]["reviewer"], "codex")
        self.assertIn("nothing to convert", C.convert_file(p))


class TaskRig:
    """A rig with Star Weaving on and team contracts; the fast engine is the lead."""

    def __init__(self, case, teams, **rules):
        self.rig = Rig(rules={"starweave": {"mode": "on", "teams": teams}, **rules})
        case.addCleanup(self.rig.stop)

    def __getattr__(self, k):
        return getattr(self.rig, k)

    def run(self, goal="Team task: build it.", **kw):
        t = teamtasks.start(self.rig.srv, goal, **kw)
        assert wait(lambda: teamtasks.tasks(self.rig.srv, t["id"])[0]["state"] != "running", 30)
        return teamtasks.tasks(self.rig.srv, t["id"])[0]

    def actions(self):
        return [json.loads(r["payload"]) for r in rows(self.rig.path, "SELECT payload FROM events WHERE kind='team_action' ORDER BY seq")]


def lead_says(first, *later):
    """The lead: its first answer (a list of team calls, or a text); each send-back answered by the next of `later`."""
    replies = list(later)

    def script(body):
        if is_teammate(body) or mode_of(body):
            return None
        msgs = body.get("messages") or []
        if msgs and msgs[-1].get("role") == "tool":
            return None                                  # "Tool said: …"
        text = last_user(body)
        if text.startswith("Halo checked your answer"):
            return replies.pop(0) if replies else "Still the same."
        if "Team task" in text:
            return first
        return None
    return script


class ContractTaskTest(unittest.TestCase):
    def test_a_task_takes_its_contract_and_an_edit_changes_only_the_next_task(self):
        rig = TaskRig(self, {"code": code_team()})
        rig.a.script = lead_says("Done: files a.py; tests pass.")
        t = teamtasks.start(rig.srv, "Team task: one.", run=False)
        self.assertEqual((t["team"], t["contract"]["name"]), ("code", "Code team"))
        v1 = t["contract_version"]
        self.assertEqual(v1, C.version(C.teams(rig.srv.cfg.rules)["code"]))
        from nebula import ui
        ui.contract_post(rig.srv, "/api/starweave/contracts/save", {"id": "code", "contract": code_team(house_rules="Never commit.")})
        turn = SimpleNamespace(team_ctx={"task_id": f"task:{t['id']}"}, cfg=rig.srv.cfg, headers={})
        self.assertEqual(C.of_turn(turn)["version"], v1)             # the running task keeps what it started with
        C._SNAP.clear()
        self.assertEqual(C.of_turn(turn)["version"], v1)             # read back from the task's record too
        t2 = teamtasks.start(rig.srv, "Team task: two.", run=False)
        self.assertNotEqual(t2["contract_version"], v1)
        self.assertEqual(t2["contract"]["house_rules"], "Never commit.")

    def test_the_lead_reads_its_contract(self):
        rig = TaskRig(self, {"code": code_team(house_rules="Never commit.", escalation={"ask_before": ["deleting files"]},
                                               reporting={"sections": ["files", "tests"]})})
        rig.a.script = lead_says("Files: a.py. Tests: pass.")
        done = rig.run()
        self.assertEqual(done["state"], "done", done)
        prompt = rig.a.requests[0]["messages"][-1]["content"]
        self.assertIn("Your team's contract (Code team, v", prompt)
        self.assertIn("- Purpose: Builds and fixes code with tests.", prompt)
        self.assertIn("Only these models may be given work.", prompt)
        self.assertIn("- Ask the owner (team__ask_owner) before: deleting files.", prompt)
        self.assertIn("- House rules: Never commit.", prompt)
        self.assertIn("Answer with a part for each of: files, tests", prompt)

    def test_halo_refuses_work_for_a_model_off_the_team(self):
        rig = TaskRig(self, {"code": code_team(reviewer=None)})
        rig.a.script = lead_says([{"name": team.DELEGATE, "arguments": {"items": [{"id": "p", "to": "big", "task": "Do it."}]}}])
        done = rig.run()
        self.assertIn("big is not on your team now; your team: think", done["result"])
        spec = {t["function"]["name"]: t["function"] for t in rig.a.requests[0]["tools"]}
        self.assertEqual(spec[team.DELEGATE]["parameters"]["properties"]["items"]["items"]["properties"]["to"]["enum"], ["think"])
        refused = [a for a in rig.actions() if a["ruling"] == "refused"]
        self.assertEqual(refused[0]["contract"]["team"], "code")      # ruled under the contract, by version
        self.assertTrue(refused[0]["contract"]["version"].startswith("v"))
        self.assertEqual([r["action"] for r in done["refusals"]], ["delegate"])     # the task card shows it

    def test_others_may_help_when_the_contract_says_so(self):
        rig = TaskRig(self, {"code": code_team(reviewer=None, others=True)})
        rig.a.script = lead_says([{"name": team.DELEGATE, "arguments": {"items": [{"id": "p", "to": "big", "task": "Do it."}]}}])
        done = rig.run()
        self.assertEqual([(w["label"], w["model"], w["state"]) for w in done["items"]], [("p", "big", "done")])

    def test_a_stand_in_works_while_its_member_is_down(self):
        rig = TaskRig(self, {"code": code_team(helpers=["think"], reviewer=None, stand_ins={"think": "big"})})
        rig.srv.registry.mark_down("think")
        rig.a.script = lead_says([{"name": team.DELEGATE, "arguments": {"items": [{"id": "p", "to": "big", "task": "Do it."}]}}])
        done = rig.run()
        self.assertEqual([(w["model"], w["state"]) for w in done["items"]], [("big", "done")])

    def test_teammates_read_the_house_rules(self):
        rig = TaskRig(self, {"code": code_team(reviewer=None, house_rules="Never commit.")})
        rig.a.script = lead_says([{"name": team.DELEGATE, "arguments": {"items": [{"id": "p", "to": "think", "task": "Do it."}]}}])
        rig.run()
        mate = [b for b in rig.a.requests if is_teammate(b)][0]
        self.assertIn("Your team's house rules (Code team): Never commit.", mate["messages"][0]["content"])

    def test_the_lead_cannot_approve_its_own_draft_when_the_reviewer_must(self):
        rig = TaskRig(self, {"code": code_team(reviewer=None, review={"lead_reviews": False})})
        rig.a.script = lead_says("Done.")
        rig.run()
        self.assertNotIn(team.REVIEW, tool_names(rig.a.requests[0]))  # nobody but the lead could review: no review tool
        rig2 = TaskRig(self, {"code": code_team(reviewer="big", review={"lead_reviews": False})})
        rig2.a.script = lead_says("Done.")
        rig2.run()
        spec = {t["function"]["name"]: t["function"] for t in rig2.a.requests[0]["tools"]}
        self.assertIn("Reviewer: big.", spec[team.REVIEW]["description"])


class DoneCheckTest(unittest.TestCase):
    def test_done_needs_an_approved_review_and_is_sent_back_once(self):
        rig = TaskRig(self, {"code": code_team(done={"review": True})})
        rig.a.script = lead_says("All done.", "All done, really.")
        done = rig.run()
        self.assertEqual(done["state"], "failed", done)
        self.assertEqual(done["sent_back"], 1)
        self.assertIn("review: no review approved the work", done["error"])
        self.assertEqual(done["result"], "All done, really.")          # the lead's last answer is kept
        back = [b for b in rig.a.requests if last_user(b).startswith("Halo checked your answer")]
        self.assertEqual(len(back), 1)
        self.assertIn("send-back 1 of 1", last_user(back[0]))
        self.assertEqual([(a["action"], a["ruling"]) for a in rig.actions() if a["action"] == "done_check"],
                         [("done_check", "failed"), ("done_check", "failed")])
        note = rig.srv.halo.view()["notes"][0]
        self.assertEqual(note["kind"], "failed")
        self.assertIn("Checks: review failed", note["text"])

    def test_an_approved_review_passes(self):
        rig = TaskRig(self, {"code": code_team(done={"review": True})})
        rig.a.script = lead_says([{"name": team.REVIEW, "arguments": {"work": "the change", "criteria": "it works"}}])
        rig.b.script = lambda body: verdict("approve") if mode_of(body) == "review" else None
        done = rig.run()
        self.assertEqual(done["state"], "done", done)
        self.assertEqual([(k["check"], k["ok"]) for k in done["checks"]], [("review", True)])

    def test_a_missing_report_part_is_sent_back_and_fixed(self):
        rig = TaskRig(self, {"code": code_team(done={"report": True}, reporting={"sections": ["files", "tests"]}, review={"max_rework": 2})})
        rig.a.script = lead_says("Files: a.py", "Files: a.py\nTests: 3 passed")
        done = rig.run()
        self.assertEqual((done["state"], done["sent_back"]), ("done", 1), done)
        back = [b for b in rig.a.requests if last_user(b).startswith("Halo checked your answer")]
        self.assertIn("the report has no part for: tests", last_user(back[0]))

    def test_halo_runs_the_tests_itself_in_the_sandbox(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, ignore_errors=True))
        p = mock.patch.object(toolbox, "WORKBOX", FakeBox)
        p.start()
        self.addCleanup(p.stop)
        toolbox._BOXES.clear()
        FakeBox.ran = []
        rig = TaskRig(self, {"code": code_team(done={"tests": "python3 -m unittest fail"}, tools={"folder": folder})},
                      toolbox={"mode": "on", "folder": folder})
        rig.a.script = lead_says("Done.", "Fixed.")
        done = rig.run()
        self.assertEqual(done["folder"], os.path.realpath(folder))     # the team's own folder
        self.assertEqual(done["state"], "failed")
        self.assertEqual([c for _f, c, _t in FakeBox.ran], ["python3 -m unittest fail"] * 2)
        self.assertIn("failed (exit code 1)", done["checks"][0]["detail"])
        ui_save = {"id": "code", "contract": code_team(done={"tests": "python3 -m unittest"}, tools={"folder": folder})}
        from nebula import ui
        ui.contract_post(rig.srv, "/api/starweave/contracts/save", ui_save)
        done = rig.run()
        self.assertEqual((done["state"], done["checks"][0]["ok"]), ("done", True), done)

    def test_auto_tests_are_skipped_when_the_folder_has_none(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, ignore_errors=True))
        rig = TaskRig(self, {"code": code_team(done={"tests": "auto"})})
        rig.a.script = lead_says("Done.")
        done = rig.run(folder=folder)
        self.assertEqual(done["state"], "done", done)
        self.assertEqual((done["checks"][0]["check"], done["checks"][0]["ok"]), ("tests", None))
        self.assertIsNone(C.test_command(C.normalise("c", code_team(done={"tests": "auto"})), folder))
        open(os.path.join(folder, "test_x.py"), "w").close()
        self.assertEqual(C.test_command(C.normalise("c", code_team(done={"tests": "auto"})), folder), "python3 -m unittest discover")

    def test_no_contract_no_checks(self):
        rig = TaskRig(self, {"everyday": {"lead": "fast", "helpers": ["big"]}})       # today's per-route team
        rig.a.script = lambda body: "All done."
        done = rig.run()
        self.assertEqual((done["state"], done["checks"], done["sent_back"]), ("done", [], 0))


class CouncilTest(unittest.TestCase):
    def council(self, **over):
        return code_team(**{"class": "council", "helpers": ["think", "big"], "reviewer": None,
                            "done": {"council": True}, **over})

    def test_a_council_hears_every_member_before_its_answer_reaches_you(self):
        rig = TaskRig(self, {"council": self.council()})
        ask = lambda *who: [{"name": team.DELEGATE, "arguments": {"items": [{"id": f"view-{m}", "to": m, "task": "Which design?"} for m in who]}}]
        rig.a.script = lead_says(ask("think"), "Weighed: think says A.")
        done = rig.run()
        self.assertEqual(done["state"], "failed", done)
        self.assertIn("council: not heard from: big", done["error"])
        prompt = rig.a.requests[0]["messages"][-1]["content"]
        self.assertIn("- Class: Council. This is a council: give the question to every member", prompt)

    def test_a_council_that_heard_everyone_is_done(self):
        rig = TaskRig(self, {"council": self.council()})
        rig.a.script = lead_says([{"name": team.DELEGATE, "arguments": {"items": [
            {"id": "a", "to": "think", "task": "Which design?"}, {"id": "b", "to": "big", "task": "Which design?"}]}}])
        done = rig.run()
        self.assertEqual(done["state"], "done", done)
        self.assertEqual([(k["check"], k["ok"]) for k in done["checks"]], [("council", True)])

    def test_the_card_says_it_is_a_council(self):
        lines = C.plain(C.normalise("council", self.council()))
        self.assertIn("A council: every crew member gives its own view, and fast weighs them.", lines)
        self.assertIn("Done when every crew member was heard.", lines)


class ToolsAndBudgetsTest(unittest.TestCase):
    def test_a_read_only_lead_gets_no_write_tools(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, ignore_errors=True))
        p = mock.patch.object(toolbox, "WORKBOX", FakeBox)
        p.start()
        self.addCleanup(p.stop)
        rig = TaskRig(self, {"code": code_team(tools={"lead": "read"})}, toolbox={"mode": "on", "folder": folder})
        rig.a.script = lead_says("Read it.")
        rig.run()
        names = tool_names(rig.a.requests[0])
        self.assertIn(toolbox.READ, names)
        self.assertNotIn(toolbox.WRITE, names)
        self.assertNotIn(toolbox.RUN, names)

    def test_a_role_with_no_tools(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, ignore_errors=True))
        p = mock.patch.object(toolbox, "WORKBOX", FakeBox)
        p.start()
        self.addCleanup(p.stop)
        rig = TaskRig(self, {"code": code_team(tools={"lead": "none"})}, toolbox={"mode": "on", "folder": folder})
        rig.a.script = lead_says("No tools.")
        rig.run()
        self.assertFalse([n for n in tool_names(rig.a.requests[0]) if toolbox.is_box(n)])

    def test_the_teams_budget_holds_and_halo_warns_first(self):
        big_task = "x" * 36_000                         # about 9,000 tokens: past 80% of a 10,000-token budget
        rig = TaskRig(self, {"code": code_team(reviewer=None, budgets={"max_task_tokens": 10_000})})
        calls = [[{"name": team.DELEGATE, "arguments": {"items": [{"id": "a", "to": "think", "task": big_task}]}}],
                 [{"name": team.DELEGATE, "arguments": {"items": [{"id": "b", "to": "think", "task": big_task + "y"}]}}],
                 [{"name": team.DELEGATE, "arguments": {"items": [{"id": "c", "to": "think", "task": big_task + "z"}]}}]]

        def script(body):
            if is_teammate(body):
                return None
            msgs = body.get("messages") or []
            if msgs and msgs[-1].get("role") == "tool":
                return calls.pop(0) if calls else None
            if "Team task" in last_user(body):
                return calls.pop(0)
            return None
        rig.a.script = script
        done = rig.run()
        tools = [m["content"] for b in rig.a.requests for m in b["messages"] if m.get("role") == "tool"]
        self.assertTrue(any("Halo: this task has used" in t and "of its tokens" in t for t in tools), tools)
        self.assertTrue(any("this task used its 10000 tokens" in t for t in tools), tools)
        self.assertEqual([w["label"] for w in done["items"]], ["a", "b"])     # c is refused: the budget is spent
        self.assertEqual(sum("Halo: this task has used" in t for t in tools if "[b]" in t), 0)   # warned once
        self.assertIn("budget_warning", [a["action"] for a in rig.actions()])

    def test_limits_put_the_contract_over_star_weaving(self):
        base = team.settings(SimpleNamespace(starweave={"max_task_tokens": 400_000, "max_rework": 2}))
        lim = C.limits(base, C.normalise("c", code_team(budgets={"max_task_tokens": 50_000}, review={"max_rework": 0})))
        self.assertEqual((lim["max_task_tokens"], lim["max_rework"], lim["max_in_flight"]), (50_000, 0, 4))


class EscalationTest(unittest.TestCase):
    def test_halo_asks_the_owner_when_the_same_part_is_rejected_twice(self):
        rig = TaskRig(self, {"code": code_team(review={"max_rework": 1}, escalation={"ask_after_rejects": 2})})
        rig.a.script = lead_says([{"name": team.DELEGATE, "arguments": {"items": [{"id": "p", "to": "think", "task": "Do it."}]}}])

        def lead2(body):
            msgs = body.get("messages") or []
            if not is_teammate(body) and not mode_of(body) and msgs and msgs[-1].get("role") == "tool" \
                    and "[p] think: done" in str(msgs[-1].get("content")):
                return [{"name": team.REVIEW, "arguments": {"item": "p", "criteria": "it works"}}]
            return lead_says([{"name": team.DELEGATE, "arguments": {"items": [{"id": "p", "to": "think", "task": "Do it."}]}}])(body)
        rig.a.script = lead2
        rig.b.script = lambda body: verdict("reject", ["it does not work"]) if mode_of(body) == "review" else None
        done = rig.run()
        owner = [w for w in done["items"] if w["mode"] == "ask_owner"]
        self.assertEqual(len(owner), 1, done["items"])
        row = rig.srv.ledger.work_item(owner[0]["item_id"])
        self.assertIn("big rejected part p (think) 2 times", row["task_text"])
        self.assertEqual(row["state"], "awaiting_owner")
        tools = [m["content"] for b in rig.a.requests for m in b["messages"] if m.get("role") == "tool"]
        self.assertTrue(any("Halo asked the owner what to do" in t for t in tools))


class PageTest(unittest.TestCase):
    def post(self, rig, path, body):
        r = urllib.request.Request(rig.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def get(self, rig, path):
        with urllib.request.urlopen(rig.base + path, timeout=10) as resp:
            return json.loads(resp.read())

    def test_save_template_delete_and_the_view(self):
        rig = TaskRig(self, {"larger": {"lead": "big", "helpers": ["fast"], "reviewer": "big"}})
        st, v = self.post(rig, "/api/starweave/contracts/template", {"template": "research", "lead": "big", "routes": ["writing"]})
        self.assertEqual(st, 200, v)
        self.assertEqual((v["draft"]["name"], v["draft"]["lead"]), ("Research team", "big"))
        st, v = self.post(rig, "/api/starweave/contracts/save", {"id": "research", "contract": v["draft"]})
        self.assertEqual(st, 200, v)
        self.assertEqual(list(v["contracts"]), ["larger", "research"])  # the first save turned the old team into a contract
        self.assertEqual(v["contracts"]["larger"]["contract"]["reviewer"], "fast")
        self.assertIn("big leads, fast helps, fast reviews.", v["contracts"]["larger"]["plain"])
        self.assertTrue(v["contracts"]["research"]["version"].startswith("v"))
        self.assertEqual(v["teams"]["writing"]["lead"], "big")       # the per-route cards still read
        st, v = self.post(rig, "/api/starweave/contracts/save", {"id": "research", "contract": code_team(lead="nope")})
        self.assertEqual((st, v["error"]["message"] if isinstance(v.get("error"), dict) else v.get("error")),
                         (400, "team research: there is no model 'nope' here"))
        st, v = self.post(rig, "/api/starweave/contracts/save", {"id": "deep", "was": "research", "contract": code_team(name="Deep")})
        self.assertEqual(list(v["contracts"]), ["larger", "deep"])      # renamed in place
        st, v = self.post(rig, "/api/starweave/contracts/delete", {"id": "deep"})
        self.assertEqual(list(v["contracts"]), ["larger"])
        self.assertEqual(self.get(rig, "/api/starweave")["templates"], sorted(C.CLASSES))

    def test_halo_picks_the_team_by_its_purpose(self):
        rig = TaskRig(self, {"code": code_team(), "docs": code_team(name="Docs team", purpose="Writes the docs.", lead="big",
                                                                    helpers=[], reviewer="fast")})
        spec = {t["name"]: t for t in rig.srv.tools.local["starweave"].tools}
        team_param = spec["start_task"]["inputSchema"]["properties"]["team"]
        self.assertEqual(team_param["enum"], ["code", "docs"])
        self.assertIn("docs: Writes the docs.", team_param["description"])
        rig.a.script = lambda body: "ok"
        rig.b.script = lambda body: "ok"
        out = rig.srv.tools.call("starweave__start_task", {"goal": "Write the README.", "why": "docs", "team": "docs"})
        self.assertIn("with the Docs team (big leads, fast reviews)", out)
        self.assertTrue(rig.srv.tools.call("starweave__start_task", {"goal": "x", "why": "y", "team": "nope"}).startswith("error:"))

    def test_the_route_cards_apply_keeps_the_contracts(self):
        rig = TaskRig(self, {"code": code_team(routes=["larger"], house_rules="Never commit.")})
        st, out = self.post(rig, "/api/plan/apply", {"rules": {"starweave": {"mode": "on", "teams": {
            "larger": {"lead": "big", "helpers": ["fast"], "reviewer": "think"}}}}})
        self.assertEqual(st, 200, out)
        c = C.teams(rig.srv.cfg.rules)["code"]
        self.assertEqual((c["lead"], c["helpers"], c["reviewer"], c["house_rules"]), ("big", ["fast"], "think", "Never commit."))


class PageWordsTest(unittest.TestCase):
    def test_the_teams_panel_reads_contracts_and_saves_them_on_their_own(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parents[1] / "ui" / "index.html").read_text()
        for want in ('data-wvk="${esc(id)}"', 'data-wvtvedit="${esc(id)}"', 'data-wvedbox', 'id="wv-teams"', '"/api/starweave/tags"', 'if (id) { wv.sel = id; wv.editing = null;', '"/api/starweave/contracts/save"', '"/api/starweave/contracts/delete"',
                     '"/api/starweave/contracts/template"', "teams are saved by their own Save", "refused ${m.t.refusals.length} move",
                     "under its contract ${esc(m.t.contract_version"):
            self.assertIn(want, html)
        self.assertNotIn("one per route: drag a model in", html)
        from nebula import contracts
        for k in contracts.CLASSES:                        # every class Halo knows can be picked on the page
            self.assertIn(f'["{k}", "{contracts.CLASS_NAMES[k]}:', html)


class OwnRoutesAndTagsTest(unittest.TestCase):
    """A team serves a route of the owner's own (Star Calling's custom routes) once that route exists, and carries a few
    tags, shown on its row."""
    def test_a_team_serves_an_own_route_and_has_tags(self):
        names = ["fast", "think", "big"]
        with self.assertRaises(C.ContractError) as e:
            C.check("code", code_team(routes=["pictures"]), names)               # not a route the caller knows of
        self.assertIn("is not a route", str(e.exception))
        ok = C.check("code", code_team(routes=["pictures", "larger"], tags=["Fast ", "code", "fast", ""]), names,
                     ["everyday", "larger", "pictures"])
        self.assertEqual(ok["routes"], ["pictures", "larger"])
        self.assertEqual(ok["tags"], ["fast", "code"])
        with self.assertRaises(C.ContractError):
            C.check("code", code_team(routes=["writing"]), names, ["everyday", "larger", "pictures"])   # put away
        self.assertEqual(C.normalise("code", code_team(routes=["pictures"]))["routes"], ["pictures"])
        self.assertEqual(C.normalise("code", code_team(tags="a; b"))["tags"], ["a", "b"])
        self.assertNotEqual(C.version(code_team()), C.version(code_team(tags=["code"])))


if __name__ == "__main__":
    unittest.main()
