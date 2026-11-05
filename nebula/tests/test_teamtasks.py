"""Team tasks (nebula/teamtasks.py): "Halo, I want to do this task". Halo's chat gets team tools while Star Weaving is
on; a task runs in the background with the route's team, in its own conversation, its parts the task's; when it ends
Halo says so in the Mission Log; only the owner starts one."""
import json
import os
import tempfile
import time
import unittest

from nebula import stars, team, teamtasks
from nebula.tests.test_escalation import Rig
from nebula.tests.test_starweave_work import lead


def wait(pred, s=20):
    end = time.time() + s
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


class TeamTaskTest(unittest.TestCase):
    def rig(self, mode="on"):
        rig = Rig(rules={"starweave": {"mode": mode, "teams": {"everyday": {"lead": "fast", "helpers": ["big"]}}}})
        self.addCleanup(rig.stop)
        return rig

    def test_a_task_runs_in_the_background_with_the_team_and_halo_says_when_it_is_done(self):
        rig = self.rig()
        rig.a.script = lead("Team task", (team.DELEGATE, {"items": [{"id": "part", "to": "big", "task": "Do the part."}]}))
        t = teamtasks.start(rig.srv, "Write the thing, with its tests.")
        self.assertEqual((t["state"], t["lead"], t["helpers"], t["route"]), ("running", "fast", ["big"], "everyday"))
        self.assertTrue(wait(lambda: teamtasks.tasks(rig.srv, t["id"])[0]["state"] != "running"), teamtasks.tasks(rig.srv))
        done = teamtasks.tasks(rig.srv, t["id"])[0]
        self.assertEqual(done["state"], "done", done)
        self.assertTrue(done["result"])
        self.assertEqual([(w["label"], w["model"], w["state"]) for w in done["items"]], [("part", "big", "done")])  # its parts are the task's
        s = rig.srv.ledger.session(done["session_id"])
        self.assertEqual(s["session_key"], f"starweave-task:{t['id']}")             # its own conversation
        sent = rig.a.requests[0]
        self.assertIn("Write the thing, with its tests.", sent["messages"][-1]["content"])
        note = rig.srv.halo.view()["notes"][0]
        self.assertEqual((note["task_id"], note["kind"]), (t["id"], "done"))       # the Mission Log hears of it
        self.assertTrue(note["title"].startswith("Team task done"))

    def test_a_task_running_when_nebula_closes_stays_stopped_when_its_answer_comes_late(self):
        import threading
        rig = Rig(rules={"starweave": {"mode": "on", "teams": {"everyday": {"lead": "fast", "helpers": ["big"]}}}})
        release = threading.Event()
        rig.a.script = lambda body: (release.wait(30), None)[1]
        t = teamtasks.start(rig.srv, "Write the thing, with its tests.")
        time.sleep(0.5)
        rig.srv.shutdown()
        rig.srv.server_close()
        stopped = [x for x in teamtasks._load(rig.srv) if x["id"] == t["id"]][0]
        self.assertEqual((stopped["state"], stopped["error"]), ("failed", teamtasks.STOPPED))
        release.set()
        time.sleep(1.0)                                   # the late answer finds Nebula closed: it writes nothing
        again = [x for x in teamtasks._load(rig.srv) if x["id"] == t["id"]][0]
        self.assertEqual((again["state"], again["error"]), ("failed", teamtasks.STOPPED))
        rig.a.stop(); rig.b.stop(); rig.tmp.cleanup()

    def test_the_api_starts_and_lists_and_refuses_what_cannot_run(self):
        rig = self.rig()
        rig.a.script = lambda body: "All done."
        st, out = self.post(rig, "/api/starweave/tasks", {"goal": "Tidy the folder."})
        self.assertEqual(st, 200, out)
        tid = out["task"]["id"]
        self.assertTrue(wait(lambda: self.get(rig, "/api/starweave/tasks")["tasks"][0]["state"] == "done"))
        self.assertEqual(self.get(rig, "/api/starweave/tasks")["tasks"][0]["id"], tid)
        self.assertEqual(self.post(rig, "/api/starweave/tasks", {"goal": ""})[0], 400)
        self.assertEqual(self.post(rig, "/api/starweave/tasks", {"goal": "x", "folder": "/no/such/folder"})[0], 400)

    def test_star_weaving_off_starts_nothing_and_halo_has_no_team_tools(self):
        rig = self.rig(mode="off")
        with self.assertRaises(teamtasks.TaskError):
            teamtasks.start(rig.srv, "Write the thing.")
        a = stars.assistant_agent(rig.srv.cfg)
        self.assertNotIn("starweave", a.tools)

    def test_halo_gets_the_team_tools_while_star_weaving_is_on_and_only_the_owner_runs_them(self):
        rig = self.rig()
        a = stars.assistant_agent(rig.srv.cfg)
        self.assertIn("starweave", a.tools)
        self.assertIn("starweave__start_task", a.system)
        names = [t["function"]["name"] for t in rig.srv.tools.openai_tools(["starweave"])]
        self.assertEqual(sorted(names), ["starweave__start_task", "starweave__tasks"])
        rig.a.script = lambda body: "Done."
        out = rig.srv.tools.call("starweave__start_task", {"goal": "Build it.", "why": "it has parts"})
        self.assertTrue(out.startswith("Started team task"), out)
        self.assertIn("fast leads", out)
        from nebula.engine import CALL_TENANT
        tok = CALL_TENANT.set("caller:friend")
        try:
            self.assertTrue(rig.srv.tools.call("starweave__start_task", {"goal": "x", "why": "y"}).startswith("error:"))
        finally:
            CALL_TENANT.reset(tok)
        self.assertTrue(wait(lambda: "[done]" in rig.srv.tools.call("starweave__tasks", {})))

    def test_no_team_is_said(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(rules={"starweave": {"mode": "on", "teams": {}}}, tier=lambda n: None)
        with self.assertRaises(teamtasks.TaskError):
            teamtasks.team_for(cfg, "writing")

    # --- http
    def post(self, rig, path, body):
        import urllib.error
        import urllib.request
        r = urllib.request.Request(rig.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def get(self, rig, path):
        import urllib.request
        with urllib.request.urlopen(rig.base + path, timeout=10) as resp:
            return json.loads(resp.read())


if __name__ == "__main__":
    unittest.main()
