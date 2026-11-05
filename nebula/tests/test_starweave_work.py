"""Starweave phase 2 (nebula/team.py, nebula/teamwork.py): the lead delegates work items to teammates, collects
their results, asks a teammate, and leaves notes; Halo rules on each call; Practice runs nothing; teammates work in
child sessions of their own and see only what their item needs."""
import json
import unittest

from nebula import team
from nebula.activity import Watch
from nebula.replay import verify
from nebula.tests.test_escalation import Rig, rows


def tool_names(body):
    return [t.get("function", t).get("name") for t in body.get("tools") or []]


def last_user(body):
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            return m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
    return ""


def is_teammate(body):
    msgs = body.get("messages") or []
    return bool(msgs) and msgs[0].get("role") == "system" and "teammate on a Nebula team" in str(msgs[0].get("content"))


def lead(word, *calls, then=None):
    """A fake lead: on a request whose last user text holds `word` (and not a teammate's), makes `calls`; after a
    tool result, `then(body)` may make more calls (else the engine says what the tool said)."""
    def script(body):
        if is_teammate(body):
            return None
        msgs = body.get("messages") or []
        if msgs and msgs[-1].get("role") == "tool":
            return then(body) if then else None
        if word in last_user(body) and team.DELEGATE in tool_names(body):
            return [{"name": n, "arguments": a} for n, a in calls]
        return None
    return script


def teammate_requests(engine):
    return [b for b in engine.requests if is_teammate(b)]


class WorkTest(unittest.TestCase):
    def rig(self, mode="on", **rules):
        r = Rig(rules={"starweave": {"mode": mode, **rules}})
        self.addCleanup(r.stop)
        return r

    def items(self, rig):
        out = rows(rig.path, "SELECT * FROM work_items ORDER BY created_at, rowid")
        for r in out:
            r["depends_on"], r["chain"] = json.loads(r["depends_on"]), json.loads(r["chain"])
        return out

    def actions(self, rig):
        return [json.loads(r["payload"]) for r in rows(rig.path, "SELECT payload FROM events WHERE kind='team_action' ORDER BY seq")]

    def assert_replays(self, rig):
        ok, lines = verify(rig.path)
        self.assertTrue(ok, "\n".join(lines))

    def test_the_lead_gets_the_team_sheet(self):
        rig = self.rig(good_at={"big": "long code and analysis"})
        rig.chat([{"role": "user", "content": "hi"}])
        body = rig.a.requests[-1]
        spec = {t["function"]["name"]: t["function"] for t in body["tools"]}
        self.assertEqual(spec[team.DELEGATE]["parameters"]["properties"]["items"]["items"]["properties"]["to"]["enum"], ["think", "big"])
        self.assertIn("- big: everyday; long code and analysis", spec[team.DELEGATE]["description"])
        self.assertIn("local", spec[team.DELEGATE]["description"])
        self.assertNotIn("- fast:", spec[team.DELEGATE]["description"])         # never itself: it is up the chain

    def test_delegate_runs_the_item_on_the_teammate_and_returns_its_result(self):
        rig = self.rig()
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "sum", "to": "big", "task": "Summarise the plan."}]}))
        reply = rig.chat([{"role": "user", "content": "split this up please"}])
        self.assertEqual(reply["content"], "Tool said: [sum] big: done\nHello world")
        sent = teammate_requests(rig.b)
        self.assertEqual(len(sent), 1)
        prompt = sent[0]["messages"][-1]["content"]
        self.assertIn("The user's request (for context):\nsplit this up please", prompt)
        self.assertIn("Your task:\nSummarise the plan.", prompt)
        self.assertEqual(len(sent[0]["messages"]), 2)                         # never the conversation
        (item,) = self.items(rig)
        self.assertEqual((item["label"], item["from_model"], item["model"], item["mode"], item["state"], item["result"]),
                         ("sum", "fast", "big", "execute", "done", "Hello world"))
        self.assertEqual(item["chain"], ["fast"])
        self.assertGreater(item["tokens"], 0)
        child = rows(rig.path, "SELECT * FROM sessions WHERE session_id=?", item["child_session_id"])[0]
        self.assertEqual((child["session_key"], child["key_source"]), (f"team:{item['item_id']}", "team"))
        self.assertEqual([(a["action"], a["to"], a["ruling"]) for a in self.actions(rig)], [("delegate", "big", "allowed")])
        lead_session = rows(rig.path, "SELECT session_id FROM sessions WHERE key_source<>'team'")
        self.assertEqual(len(lead_session), 1)                                 # the lists leave the child out
        self.assert_replays(rig)

    def test_the_request_event_names_the_model_the_item_runs_on(self):
        rig = self.rig()
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "sum", "to": "big", "task": "Summarise the plan."}]}))
        rig.chat([{"role": "user", "content": "split this up please"}])
        (item,) = self.items(rig)
        (req,) = rows(rig.path, "SELECT payload FROM events WHERE session_id=? AND kind='request'", item["child_session_id"])
        self.assertEqual(json.loads(req["payload"])["tier"], "big")            # not the first tier in the config ("fast")
        (att,) = rows(rig.path, "SELECT tier FROM attempts WHERE session_id=?", item["child_session_id"])
        self.assertEqual(att["tier"], "big")
        self.assert_replays(rig)

    def test_a_first_request_naming_a_model_opens_on_it(self):
        rig = self.rig(mode="off")
        rig.chat([{"role": "user", "content": "hello"}], model="big")
        (req,) = rows(rig.path, "SELECT payload FROM events WHERE kind='request'")
        self.assertEqual(json.loads(req["payload"])["tier"], "big")
        self.assertEqual(rows(rig.path, "SELECT * FROM handoffs"), [])                # nothing to hand off on a first turn
        self.assert_replays(rig)

    def test_a_model_used_only_when_named_is_never_recorded_for_another_turn(self):
        rig = self.rig(mode="off")
        rig.srv.policy.put({"models": {"fast": {"explicit": True}}, "advertise": {}})   # the first tier in the config
        rig.chat([{"role": "user", "content": "hello"}])
        (req,) = rows(rig.path, "SELECT payload FROM events WHERE kind='request'")
        (att,) = rows(rig.path, "SELECT tier FROM attempts")
        self.assertEqual((json.loads(req["payload"])["tier"], att["tier"]), ("think", "think"))

    def test_a_dependent_item_waits_and_builds_on_the_result_even_on_the_leads_own_engine(self):
        rig = self.rig()
        rig.a.script = lead("split", (team.DELEGATE, {"items": [
            {"id": "facts", "to": "big", "task": "List the facts."},
            {"id": "draft", "to": "think", "task": "Draft from the facts.", "depends_on": ["facts"]}]}))
        reply = rig.chat([{"role": "user", "content": "split it"}])
        self.assertIn("[facts] big: done\nHello world", reply["content"])
        self.assertIn("[draft] think: done\nHello world", reply["content"])
        (draft,) = teammate_requests(rig.a)                                    # think runs on the lead's engine: its slot was given back
        self.assertIn("Results you build on:\n[facts] big:\nHello world", draft["messages"][-1]["content"])
        facts, d = self.items(rig)
        self.assertEqual(d["depends_on"], [facts["item_id"]])
        self.assertEqual((facts["state"], d["state"]), ("done", "done"))

    def test_wait_none_then_collect_whole(self):
        rig = self.rig()

        def then(body):
            said = body["messages"][-1]["content"]
            if said.startswith("Started:"):
                return [{"name": team.COLLECT, "arguments": {"ids": ["bg"]}}]
            return None
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "bg", "to": "big", "task": "Do it."}], "wait": "none"}), then=then)
        reply = rig.chat([{"role": "user", "content": "split it"}])
        self.assertEqual(reply["content"], "Tool said: [bg] big: done\nHello world")
        self.assertEqual([a["action"] for a in self.actions(rig)], ["delegate", "collect"])

    def test_a_restart_marks_running_work_stopped_instead_of_losing_it(self):
        # an item still running when Nebula closes: it gets a few seconds, then it is marked stopped while the ledger is
        # open (its answer came after the close, was lost, and the item stayed "running" for ever); the next Nebula
        # marks anything the last one left unfinished
        import threading, time
        from nebula import teamwork
        from nebula.ledger import Ledger
        from nebula.server import make_server
        rig = Rig(rules={"starweave": {"mode": "on"}})
        release = threading.Event()

        def held(body):
            if is_teammate(body):
                release.wait(30)
            return None
        rig.b.script = held
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "bg", "to": "big", "task": "Do it."}], "wait": "none"}))
        rig.chat([{"role": "user", "content": "split it"}])
        for _ in range(100):
            if [i for i in self.items(rig) if i["state"] == "running"]:
                break
            time.sleep(0.05)
        t0 = time.monotonic()
        rig.srv.shutdown()
        rig.srv.server_close()
        self.assertLess(time.monotonic() - t0, 10)
        (item,) = self.items(rig)
        self.assertEqual((item["state"], item["error"]), ("failed", teamwork.STOPPED))
        release.set()
        time.sleep(0.3)                                                    # the late answer finds the ledger closed
        (item,) = self.items(rig)
        self.assertEqual(item["state"], "failed")
        # a ledger an older Nebula left with an item mid-way: the next start marks it
        led = Ledger(rig.path)
        led.update_work_item(item["item_id"], state="running", error=None)
        led.close()
        cfg = rig.srv.cfg
        srv = make_server(cfg)
        try:
            (item,) = self.items(rig)
            self.assertEqual((item["state"], item["error"]), ("failed", teamwork.STOPPED))
        finally:
            srv.server_close()
            rig.a.stop()
            rig.b.stop()
            rig.tmp.cleanup()

    def test_the_cloud_cap_gives_each_priced_piece_a_budget_it_cannot_pass(self):
        # cap $1 at $10 a million tokens = 100,000 tokens for the task's priced pieces. Each piece is set aside a share
        # of what is left (shared among the pieces that may run at once) and its answer is held to it; what a piece
        # has used counts whatever state it is in (under review too); none starts once less than 1,000 tokens are left
        import threading
        from types import SimpleNamespace
        from nebula import teamwork
        from nebula.router import Rules
        items = []
        cloud = SimpleNamespace(cost_per_mtoken=10.0, is_command=False)
        turn = SimpleNamespace(rules=Rules.from_dict({"starweave": {"mode": "on", "cloud_cap_usd": 1, "max_in_flight": 4}}),
                               cfg=SimpleNamespace(tier=lambda name: cloud if name == "cloud" else SimpleNamespace(cost_per_mtoken=None)),
                               ledger=SimpleNamespace(work_items=lambda task_id=None, session_id=None: list(items)),
                               team_ctx=None, headers={}, team_task="task:t", priority="interactive", contract=None)
        T = "task:budget"
        why, a = teamwork.admit_priced(turn, T, "a", "cloud")
        self.assertEqual((why, a), (None, 25_000))                       # $1 shared four ways
        why, b = teamwork.admit_priced(turn, T, "b", "cloud")
        self.assertEqual((why, b), (None, 25_000))                       # $0.75 left, three ways
        self.assertEqual(teamwork.admit_priced(turn, T, "x", "free"), (None, None))   # an unpriced model: no budget
        items.append({"item_id": "a", "model": "cloud", "state": "in_review", "tokens": 25_000})   # used, under review
        teamwork._priced_done(T, "a")
        why, c = teamwork.admit_priced(turn, T, "c", "cloud")            # $1 - $0.25 used - $0.25 set aside for b
        self.assertEqual((why, c), (None, 16_666))
        items.append({"item_id": "b", "model": "cloud", "state": "done", "tokens": 25_000})
        items.append({"item_id": "c", "model": "cloud", "state": "done", "tokens": 49_500})
        teamwork._priced_done(T, "b"); teamwork._priced_done(T, "c")
        why, d = teamwork.admit_priced(turn, T, "d", "cloud")            # 500 tokens left: none starts
        self.assertIn("cloud cap reached", why)
        self.assertLessEqual(sum(i["tokens"] for i in items) * 10 / 1e6, 1.0)
        # a piece's own request is held to its budget
        self.assertGreater(teamwork.prompt_tokens([{"role": "user", "content": "x" * 3000}]), 1000)

    def test_a_priced_teammates_request_carries_its_budget(self):
        from nebula.config import Tier
        rig = Rig(rules={"starweave": {"mode": "on", "cloud_cap_usd": 1, "max_in_flight": 4}})
        self.addCleanup(rig.stop)
        rig.srv.cfg.tiers.append(Tier("cloudy", rig.b.url, cost_per_mtoken=10.0))
        from nebula.server import reload_models
        reload_models(rig.srv)
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "c", "to": "cloudy", "task": "Do it."}]}))
        rig.chat([{"role": "user", "content": "split it"}])
        sent = teammate_requests(rig.b)
        self.assertEqual(len(sent), 1)
        self.assertLessEqual(sent[0]["max_tokens"], 25_000)                # $1 shared four ways, less its prompt
        self.assertGreater(sent[0]["max_tokens"], 20_000)

    def test_ask_gets_a_short_answer(self):
        rig = self.rig()
        rig.a.script = lead("split", (team.ASK, {"to": "big", "question": "Is 7 prime?"}))
        reply = rig.chat([{"role": "user", "content": "split hairs"}])
        self.assertEqual(reply["content"], "Tool said: Hello world")
        (sent,) = teammate_requests(rig.b)
        self.assertEqual(sent["max_tokens"], team.ASK_TOKENS)
        self.assertIn("The lead's question:\nIs 7 prime?", sent["messages"][-1]["content"])
        self.assertEqual([(i["mode"], i["label"]) for i in self.items(rig)], [("ask", "ask-1")])

    def test_a_note_reaches_every_later_teammate(self):
        rig = self.rig()
        rig.a.script = lead("split", (team.NOTE, {"text": "Mara is left-handed."}),
                            (team.DELEGATE, {"items": [{"id": "ch", "to": "big", "task": "Write the scene."}]}))
        rig.chat([{"role": "user", "content": "split the chapter"}])
        (sent,) = teammate_requests(rig.b)
        self.assertIn("Team notes:\n- Mara is left-handed. (fast)", sent["messages"][-1]["content"])
        self.assertIn("team_note", rig.kinds())

    def test_a_teammate_may_ask_and_note_but_not_delegate(self):
        rig = self.rig()
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "x", "to": "big", "task": "Do it."}]}))
        rig.chat([{"role": "user", "content": "split"}])
        (sent,) = teammate_requests(rig.b)
        self.assertEqual(tool_names(sent), [team.ASK, team.NOTE])
        spec = {t["function"]["name"]: t["function"] for t in sent["tools"]}
        self.assertEqual(spec[team.ASK]["parameters"]["properties"]["to"]["enum"], ["think"])   # fast and big are up its chain

    def test_at_the_last_depth_only_note(self):
        rig = self.rig(max_depth=1)
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "x", "to": "big", "task": "Do it."}]}))
        rig.chat([{"role": "user", "content": "split"}])
        (sent,) = teammate_requests(rig.b)
        self.assertEqual(tool_names(sent), [team.NOTE])

    def test_halo_refuses_what_breaks_the_rules(self):
        cases = [
            ({"items": [{"id": "a", "to": "fast", "task": "t"}]}, "item a: fast is already up this task's chain"),
            ({"items": [{"id": "a", "to": "nowhere", "task": "t"}]}, "item a: nowhere is not on your team now; your team: think, big"),
            ({"items": [{"id": "a", "to": "big", "task": "t"}, {"id": "a", "to": "think", "task": "u"}]}, "item id a is used twice"),
            ({"items": [{"id": "a", "to": "big", "task": "t", "depends_on": ["zz"]}]}, "item a: depends_on zz, which is not an item before it"),
            ({"items": [{"id": "a", "to": "big", "task": " "}]}, "item a: the task is empty"),
            ({"items": []}, "no items"),
        ]
        for args, why in cases:
            with self.subTest(why=why):
                rig = self.rig()
                rig.a.script = lead("split", (team.DELEGATE, args))
                reply = rig.chat([{"role": "user", "content": "split"}])
                self.assertIn("Not done: " + why, reply["content"])
                self.assertEqual(teammate_requests(rig.b), [])
                self.assertEqual(self.items(rig), [])
                self.assertEqual([a["ruling"] for a in self.actions(rig)], ["refused"])

    def test_the_same_task_twice_to_the_same_model_is_refused(self):
        rig = self.rig()
        call = (team.DELEGATE, {"items": [{"id": "a", "to": "big", "task": "Same."}]})
        again = [{"name": team.DELEGATE, "arguments": {"items": [{"id": "b", "to": "big", "task": "Same."}]}}]
        rig.a.script = lead("split", call, then=lambda body: again if "[a] big: done" in body["messages"][-1]["content"] else None)
        reply = rig.chat([{"role": "user", "content": "split"}])
        self.assertIn("Not done: item b: big already had this same task in this task", reply["content"])
        self.assertEqual(len(teammate_requests(rig.b)), 1)

    def test_the_task_budget_and_x_nebula_task(self):
        rig = self.rig(max_task_tokens=10_000)
        from nebula.ledger import Ledger
        led = Ledger(rig.path)
        wi = led.add_work_item(task_id="task:t1", session_id="ses_x", model="big", mode="execute", task_text="earlier", label="old")
        led.update_work_item(wi, state="done", tokens=10_000)
        led.close()
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "a", "to": "big", "task": "More."}]}))
        reply = rig.chat([{"role": "user", "content": "split"}], headers={"X-Nebula-Task": "t1"})
        self.assertIn("Not done: this task used its 10000 tokens; put the results together and answer now", reply["content"])
        self.assertEqual(teammate_requests(rig.b), [])

    def test_practice_records_and_runs_nothing(self):
        rig = self.rig("practice")
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "a", "to": "big", "task": "Do it."}]}))
        reply = rig.chat([{"role": "user", "content": "split"}])
        self.assertIn("skipped (practice)", reply["content"])
        self.assertEqual(teammate_requests(rig.b), [])
        self.assertEqual(self.items(rig), [])
        self.assertEqual([(a["action"], a["mode"], a["ruling"], a["items"]) for a in self.actions(rig)],
                         [("delegate", "practice", "allowed", [{"id": "a", "to": "big"}])])

    def test_the_live_view_shows_the_beams(self):
        rig = self.rig()
        watch = Watch(rig.srv.ledger)
        watch.tick()
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "a", "to": "big", "task": "Do it."}]}))
        rig.chat([{"role": "user", "content": "split"}])
        t = watch.tick()["team"]
        self.assertEqual([(b["beam"], b["from_model"], b["model"], b["label"]) for b in t["beams"]],
                         [("started", "fast", "big", "a"), ("landed", "fast", "big", "a")])
        self.assertEqual(t["items"], [])
        self.assertEqual(watch.tick()["team"]["beams"], [])

    def test_work_calls_are_not_hand_moves(self):
        rig = self.rig(max_hand_moves=1)
        rig.b.script = lead("done", (team.HAND_DOWN, {"to": "fast", "why": "the rest is small"}),
                            (team.NOTE, {"text": "The hard part is solved."}))
        first = [{"role": "user", "content": "the hard part is done"}]
        r1 = rig.chat(first, headers={"X-Nebula-Escalate": "big"})
        rig.chat(first + [r1, {"role": "user", "content": "one more small thing"}])
        self.assertIn("team:hand_down:fast", rig.decisions()[-1]["reason"])   # the note after it did not undo the move
        self.assertEqual(rig.attempts()[-1]["tier"], "fast")

    def test_the_lead_session_view_lists_its_items(self):
        rig = self.rig()
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "a", "to": "big", "task": "Do it."}]}))
        rig.chat([{"role": "user", "content": "split"}])
        sid = rows(rig.path, "SELECT session_id FROM sessions WHERE key_source<>'team'")[0]["session_id"]
        view = rig.srv.ledger.view(sid)
        self.assertEqual([(i["label"], i["state"]) for i in view["work_items"]], [("a", "done")])
        self.assertEqual([s["session_id"] for s in rig.srv.ledger.summaries()], [sid])


class WorkLedgerTest(unittest.TestCase):
    def test_a_ledger_from_before_starweave_work_gains_its_table(self):
        import os
        import sqlite3
        import tempfile
        from nebula import ledger as L
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.sqlite3")
            c = sqlite3.connect(path)
            for stmt in L.SCHEMA.split(";"):
                if stmt.strip():
                    c.execute(stmt)
            c.execute("PRAGMA user_version = 2")
            c.commit()
            c.close()
            led = L.Ledger(path)
            try:
                self.assertEqual((led.upgraded["from"], led.upgraded["to"]), (2, len(L.STEPS)))
                self.assertTrue(os.path.exists(led.upgraded["backup"]))
                wi = led.add_work_item(task_id="t", session_id="s", model="big", mode="execute", task_text="x", label="a")
                led.update_work_item(wi, state="running", tokens=5)
                led.update_work_item(wi, tokens=7)
                self.assertEqual((led.work_item(wi)["state"], led.work_item(wi)["tokens"]), ("running", 12))
                with self.assertRaises(ValueError):
                    led.update_work_item(wi, state="sideways")
            finally:
                led.close()


class WorkSettingsTest(unittest.TestCase):
    def test_limits(self):
        self.assertEqual(team.check_settings({"mode": "on", "max_depth": 3, "cloud_cap_usd": 0.5, "good_at": {"big": " code "}}),
                         {"mode": "on", "max_depth": 3, "cloud_cap_usd": 0.5, "good_at": {"big": "code"}})
        for bad in ({"max_in_flight": 0}, {"max_task_tokens": 5}, {"cloud_cap_usd": -1}, {"good_at": ["x"]}):
            with self.assertRaises(team.TeamError):
                team.check_settings({"mode": "on", **bad})


if __name__ == "__main__":
    unittest.main()
