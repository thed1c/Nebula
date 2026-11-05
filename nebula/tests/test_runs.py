"""Detached runs (nebula/runs.py): a command tier's Starweave work runs on its own with no deadline, is collected
later (even after a restart), can be cancelled, and continues its native session for rework and follow-ups."""
import json
import os
import stat
import sys
import tempfile
import textwrap
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nebula import ledger as L
from nebula import runs, team, teamwork
from nebula.config import Tier
from nebula.tests.test_escalation import Rig, rows
from nebula.tests.test_starweave_review import mode_of, verdict
from nebula.tests.test_starweave_work import lead

FAKE_CODEX = textwrap.dedent('''\
    #!{python}
    import json, os, sys, time, uuid
    args = sys.argv[1:]
    prompt = sys.stdin.read()
    out = args[args.index("-o") + 1]
    resumed = "resume" in args
    sid = args[args.index("resume") + 1] if resumed else "th-" + uuid.uuid4().hex[:8]
    with open("calls.jsonl", "a") as f:
        f.write(json.dumps({{"args": args, "prompt": prompt}}) + "\\n")
    print(json.dumps({{"type": "thread.started", "thread_id": sid}}), flush=True)
    if os.path.exists("sleep.txt"):
        time.sleep(float(open("sleep.txt").read()))
    if "with code" in prompt:
        open("made.py", "w").write("x = 1\\n")
    with open(out, "w") as f:
        if "Answer with one JSON object" in prompt:
            f.write('Here it is: {{"verdict": "approve", "summary": "codex checked", "blocking_issues": [], "followups": []}}')
        else:
            f.write(("resumed: " if resumed else "") + "Hello from codex")
    print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "ignored"}}}}))
    print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 100, "output_tokens": 20}}}}))
''')


def fake_codex(test) -> str:
    """A `codex` on the PATH that behaves like `codex exec --json`, and its working folder."""
    bindir, work = tempfile.mkdtemp(), tempfile.mkdtemp()
    path = os.path.join(bindir, "codex")
    with open(path, "w") as f:
        f.write(FAKE_CODEX.format(python=sys.executable))
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    p = mock.patch.dict(os.environ, {"PATH": bindir + os.pathsep + os.environ.get("PATH", "")})
    p.start()
    test.addCleanup(p.stop)
    import shutil
    test.addCleanup(shutil.rmtree, bindir, True)
    test.addCleanup(shutil.rmtree, work, True)
    return work


def calls(work) -> list[dict]:
    try:
        with open(os.path.join(work, "calls.jsonl")) as f:
            return [json.loads(x) for x in f]
    except OSError:
        return []


class FormsTest(unittest.TestCase):
    def test_codex_new_and_resumed(self):
        argv, how = runs.run_form("codex", None, "gpt-5", None, True, None, runs.ANSWER)
        self.assertEqual((argv[:3], argv[-1], how), (["codex", "exec", "--json"], "-", "codex"))
        self.assertEqual(argv[argv.index("-s") + 1], "workspace-write")
        self.assertNotIn("--ephemeral", argv)                     # the session is kept, so it can be continued
        argv, _h = runs.run_form("codex", None, None, None, False, "th-1", runs.ANSWER)
        self.assertEqual(argv[-3:], ["resume", "th-1", "-"])
        self.assertEqual(argv[argv.index("-s") + 1], "read-only")

    def test_claude_new_and_resumed(self):
        argv, how = runs.run_form("claude", None, None, "Be brief.", True, None, runs.ANSWER)
        self.assertEqual((how, argv[argv.index("--system-prompt") + 1], argv[argv.index("--permission-mode") + 1]),
                         ("claude", "Be brief.", "acceptEdits"))
        argv, _h = runs.run_form("claude", None, None, "Be brief.", False, "s-9", runs.ANSWER)
        self.assertEqual(argv[argv.index("--resume") + 1], "s-9")
        self.assertNotIn("--system-prompt", argv)                 # a continued session keeps its own

    def test_parse(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("Final answer")
        self.addCleanup(os.unlink, f.name)
        out = "\n".join(json.dumps(e) for e in [{"type": "thread.started", "thread_id": "th-7"},
                                                {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}])
        self.assertEqual(runs.parse_codex(out + "\nnot json", f.name),
                         ("Final answer", "th-7", {"prompt_tokens": 5, "completion_tokens": 2, "cached_tokens": 0}))
        text, sid, usage = runs.parse_claude(json.dumps({"type": "result", "result": "ok", "session_id": "s-1",
                                                         "usage": {"input_tokens": 3, "output_tokens": 4}}))
        self.assertEqual((text, sid, usage["completion_tokens"]), ("ok", "s-1", 4))
        _t, _s, usage = runs.parse_claude(json.dumps({"result": "ok", "usage": {"input_tokens": 3, "output_tokens": 4,
                                                                                "cache_read_input_tokens": 90}}))
        self.assertEqual((usage["prompt_tokens"], usage["cached_tokens"]), (93, 90))     # cached inside, as Codex counts


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.work = tempfile.mkdtemp()
        import shutil
        self.addCleanup(shutil.rmtree, self.base, True)
        self.addCleanup(shutil.rmtree, self.work, True)

    def start(self, code: str, prompt: str = "hi") -> str:
        return runs.start(self.base, tier="cmd", cli=None, argv=[sys.executable, "-c", code], how="stdout", cwd=self.work, prompt=prompt)

    def test_a_run_outlives_no_deadline_and_returns(self):
        r = self.start("import sys, time; time.sleep(1.5); print('got ' + sys.stdin.read())", "the prompt")
        self.assertEqual(runs.inspect(self.base, r)["state"], "running")
        (info,) = runs.wait(self.base, [r])
        self.assertEqual((info["state"], info["outcome"], info["result"]["content"]), ("returned", "completed", "got the prompt"))
        self.assertGreater(info["seconds"], 1.0)

    def test_the_child_does_not_think_it_is_inside_another_agent(self):
        with mock.patch.dict(os.environ, {"CODEX_THREAD_ID": "parent", "CLAUDECODE": "1"}):
            r = self.start("import os; print(os.environ.get('CODEX_THREAD_ID'), os.environ.get('CLAUDECODE'))")
        self.assertEqual(runs.wait(self.base, [r])[0]["result"]["content"], "None None")

    def test_cancel_stops_the_child(self):
        r = self.start("import time; time.sleep(60)")
        runs.cancel(self.base, r)
        (info,) = runs.wait(self.base, [r], until=time.monotonic() + 20)
        self.assertEqual((info["state"], info["outcome"]), ("returned", "cancelled"))

    def test_a_failed_command_says_why(self):
        r = self.start("import sys; sys.stderr.write('boom'); sys.exit(3)")
        (info,) = runs.wait(self.base, [r])
        self.assertEqual(info["outcome"], "failed")
        self.assertIn("exited 3: boom", info["error"])

    def test_a_runner_that_died_is_interrupted(self):
        r = self.start("import time; time.sleep(60)")
        h = runs.read_json(os.path.join(runs.folder_of(self.base, r), "handle.json"))
        import signal
        os.kill(h["pid"], signal.SIGKILL)
        os.kill(h["child_pid"], signal.SIGKILL) if h.get("child_pid") else None
        end = time.monotonic() + 10
        while runs.inspect(self.base, r)["state"] == "running" and time.monotonic() < end:
            time.sleep(0.1)
        self.assertEqual(runs.inspect(self.base, r)["state"], "interrupted")

    def test_one_run_per_session(self):
        runs.start(self.base, tier="cmd", cli=None, argv=[sys.executable, "-c", "import time; time.sleep(30)"], how="stdout",
                   cwd=self.work, prompt="", session="s-1")
        with self.assertRaises(runs.RunError):
            runs.start(self.base, tier="cmd", cli=None, argv=[sys.executable, "-c", "pass"], how="stdout", cwd=self.work,
                       prompt="", session="s-1")
        for r in runs.runs(self.base):
            runs.cancel(self.base, r["run_id"])

    def test_old_runs_are_removed(self):
        ids = [self.start("print(1)") for _ in range(4)]
        runs.wait(self.base, ids)
        self.assertEqual(runs.prune(self.base, keep=1), 3)
        self.assertEqual(len(runs.runs(self.base)), 1)


class TeamRunTest(unittest.TestCase):
    def rig(self, **sw):
        self.work = fake_codex(self)
        codex = Tier("codex", cli="codex", cwd=self.work, writes=True, slots=1, timeout_s=1)
        r = Rig(rules={"starweave": {"mode": "on", "teams": {"everyday": {"lead": "fast", "helpers": ["codex"], "reviewer": "big"}}, **sw}},
                extra_tiers=[codex])
        self.addCleanup(r.stop)
        return r

    def items(self, rig):
        return rows(rig.path, "SELECT * FROM work_items ORDER BY created_at, rowid")

    def test_codex_runs_detached_past_its_engine_timeout(self):
        rig = self.rig()
        with open(os.path.join(self.work, "sleep.txt"), "w") as f:
            f.write("2")                                       # longer than the tier's 1 s timeout
        rig.a.script = lead("build", (team.DELEGATE, {"items": [{"id": "s", "to": "codex", "task": "Build it with code."}]}))
        reply = rig.chat([{"role": "user", "content": "build it"}])
        self.assertIn("Hello from codex", reply["content"])
        (item,) = self.items(rig)
        self.assertEqual((item["state"], item["tokens"]), ("done", 120))
        self.assertTrue(item["run_id"].startswith("run_") and item["native_session"].startswith("th-"))
        self.assertTrue(os.path.exists(os.path.join(self.work, "made.py")))          # it wrote in its own folder
        (c,) = calls(self.work)
        self.assertEqual(c["args"][c["args"].index("-s") + 1], "workspace-write")
        self.assertIn("System instructions:", c["prompt"])
        self.assertIn("Your task:\nBuild it with code.", c["prompt"])

    def test_rework_continues_the_same_session_with_only_the_issues(self):
        rig = self.rig()
        verdicts = iter([verdict("reject", ["add a test"]), verdict("approve", summary="good")])

        def then(body):
            if body["messages"][-1]["content"].startswith("[d] codex: done"):
                return [{"name": team.REVIEW, "arguments": {"item": "d", "criteria": "has tests"}}]
            return None
        rig.a.script = lead("build", (team.DELEGATE, {"items": [{"id": "d", "to": "codex", "task": "Write the parser."}]}), then=then)
        rig.b.script = lambda body: next(verdicts) if mode_of(body) == "review" else None
        reply = rig.chat([{"role": "user", "content": "build it"}])
        self.assertIn("[d] approved by big after 1 rework.", reply["content"])
        self.assertIn("resumed: Hello from codex", reply["content"])
        first, second = calls(self.work)
        sid = json.loads(json.dumps(first))["args"]
        session = self.items(rig)[0]["native_session"]
        self.assertEqual(second["args"][-3:], ["resume", session, "-"])
        self.assertTrue(second["prompt"].startswith("The reviewer did not approve your result."))
        self.assertIn("- add a test", second["prompt"])
        self.assertNotIn("Write the parser.", second["prompt"])                     # the session remembers it
        self.assertTrue(sid)

    def test_a_follow_up_continues_an_items_session(self):
        rig = self.rig()
        step = {"n": 0}

        def then(body):
            step["n"] += 1
            if step["n"] == 1:
                return [{"name": team.DELEGATE, "arguments": {"items": [{"id": "s2", "to": "codex", "task": "Now add docs.", "continue": "s1"}]}}]
            return None
        rig.a.script = lead("build", (team.DELEGATE, {"items": [{"id": "s1", "to": "codex", "task": "Write it."}]}), then=then)
        rig.chat([{"role": "user", "content": "build it"}])
        s1, s2 = self.items(rig)
        self.assertEqual((s1["state"], s2["state"]), ("done", "done"))
        self.assertEqual(s2["native_session"], s1["native_session"])
        second = calls(self.work)[1]
        self.assertEqual(second["args"][-3:], ["resume", s1["native_session"], "-"])
        self.assertEqual(second["prompt"], "A follow-up from the lead:\nNow add docs.")

    def test_a_later_task_continues_an_earlier_tasks_item(self):
        rig = self.rig()
        h = {"X-Nebula-Session": "follow-up-chat"}
        seen = []

        def script(body):
            if body.get("messages") and body["messages"][-1].get("role") == "tool":
                seen.append(body["messages"][-1]["content"])
                return None
            text = body["messages"][-1].get("content") if body.get("messages") else ""
            if team.DELEGATE not in [t["function"]["name"] for t in body.get("tools") or [] if "function" in t]:
                return None
            if text == "build it":
                return [{"name": team.DELEGATE, "arguments": {"items": [{"id": "s1", "to": "codex", "task": "Write it."}]}}]
            if text == "now the docs":
                return [{"name": team.DELEGATE, "arguments": {"items": [{"id": "s1", "to": "codex", "task": "Now add docs.", "continue": "s1"}]}}]
            return None
        rig.a.script = script
        first = rig.chat([{"role": "user", "content": "build it"}], headers=h)
        rig.chat([{"role": "user", "content": "build it"}, first, {"role": "user", "content": "now the docs"}], headers=h)
        s1, s2 = self.items(rig)
        self.assertNotEqual(s1["task_id"], s2["task_id"])                            # two tasks, one chat
        self.assertEqual((s1["state"], s2["state"]), ("done", "done"), seen)
        self.assertEqual(s2["native_session"], s1["native_session"])
        second = calls(self.work)[1]
        self.assertEqual(second["args"][-3:], ["resume", s1["native_session"], "-"])
        self.assertEqual(second["prompt"], "A follow-up from the lead:\nNow add docs.")

    def test_continue_by_an_earlier_items_full_id(self):
        rig = self.rig()
        led = rig.srv.ledger
        old = led.add_work_item(task_id="task:old", session_id="elsewhere", model="codex", mode="execute", task_text="t", label="x")
        led.update_work_item(old, state="done", native_session="th-old")
        turn = SimpleNamespace(ledger=led, team_ctx={"task_id": "task:new", "session_id": "here"})
        self.assertEqual(team.earlier_item(turn, old)["item_id"], old)
        self.assertIsNone(team.earlier_item(turn, "x"))                             # its label belongs to another chat
        self.assertEqual(teamwork.continued_session(turn, {"to": "codex", "continue": old}), "th-old")
        self.assertIsNone(teamwork.continued_session(turn, {"to": "big", "continue": old}))

    def test_halo_refuses_continuing_another_models_item(self):
        rig = self.rig()
        step = {"n": 0}
        seen = []

        def then(body):
            step["n"] += 1
            seen.append(body["messages"][-1]["content"])
            if step["n"] == 1:
                return [{"name": team.DELEGATE, "arguments": {"items": [{"id": "s2", "to": "codex", "task": "More.", "continue": "t1"}]}}]
            return None
        rig.a.script = lead("build", (team.DELEGATE, {"items": [{"id": "t1", "to": "big", "task": "Think."}]}), then=then)
        rig.chat([{"role": "user", "content": "build it"}])
        self.assertTrue(any("t1 was big's; only that teammate can go on with it" in s for s in seen), seen)

    def test_a_run_left_behind_is_collected_after_a_restart(self):
        rig = self.rig()
        led = rig.srv.ledger
        item = led.add_work_item(task_id="task:x", session_id="s", model="codex", mode="execute", task_text="t", label="x")
        r = runs.start(rig.srv.cfg.base_dir, tier="codex", cli=None, argv=[sys.executable, "-c", "import time; time.sleep(1); print('late')"],
                       how="stdout", cwd=self.work, prompt="", item_id=item)
        led.update_work_item(item, state="running", run_id=r)
        self.assertEqual([x["item_id"] for x in led.running_runs()], [item])
        self.assertEqual(teamwork.recover(rig.srv), 1)
        end = time.monotonic() + 20
        while led.work_item(item)["state"] == "running" and time.monotonic() < end:
            time.sleep(0.1)
        self.assertEqual((led.work_item(item)["state"], led.work_item(item)["result"]), ("done", "late"))

    def test_the_runs_api(self):
        rig = self.rig()
        r = runs.start(rig.srv.cfg.base_dir, tier="codex", cli=None, argv=[sys.executable, "-c", "import time; print('x', flush=True); time.sleep(30)"],
                       how="stdout", cwd=self.work, prompt="", label="slow")
        import urllib.request

        def post(path, body):
            req = urllib.request.Request(rig.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        with urllib.request.urlopen(rig.base + "/api/runs", timeout=10) as resp:
            v = json.loads(resp.read())
        self.assertEqual((v["running"], v["runs"][0]["label"]), (1, "slow"))
        time.sleep(0.5)
        self.assertIn("x", post("/api/runs/log", {"run_id": r})["stdout"])
        post("/api/runs/cancel", {"run_id": r})
        (info,) = runs.wait(rig.srv.cfg.base_dir, [r], until=time.monotonic() + 20)
        self.assertEqual(info["outcome"], "cancelled")


class LedgerTest(unittest.TestCase):
    def test_step_four_adds_the_run_columns(self):
        self.assertIn(L._run_columns, L.STEPS)
        self.assertEqual(L.STEPS[3], L._run_columns)
        with tempfile.TemporaryDirectory() as tmp:
            led = L.Ledger(os.path.join(tmp, "l.sqlite3"))
            self.addCleanup(led.close)
            cols = [r["name"] for r in led.db.execute("PRAGMA table_info(work_items)")]
            self.assertEqual((L.STEPS[-1], "skills" in cols), (L._work_skills, True))


if __name__ == "__main__":
    unittest.main()
