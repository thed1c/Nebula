"""Nebula's own tools (the team's, the toolbox, skills) go only on agent work: Halo's chats, team tasks and their work
items, and callers that ask (X-Nebula-Tools: on, or rules.toolbox.callers). A plain API call gets none: a Harness check
sent answer-only requests and got 60 tool rounds and 2.2M tokens back."""
import tempfile
import unittest
from types import SimpleNamespace

from nebula import team, toolbox
from nebula.router import Rules
from nebula.tests.test_escalation import Rig


def tool_names(body):
    return [t.get("function", t).get("name") for t in body.get("tools") or []]


class ThroughTheApp(unittest.TestCase):
    def rig(self, agent_work=False, **toolbox_rules):
        box = tempfile.TemporaryDirectory()
        self.addCleanup(box.cleanup)
        r = Rig(rules={"starweave": {"mode": "on"}, "toolbox": {"mode": "on", "folder": box.name, **toolbox_rules}},
                agent_work=agent_work)
        self.addCleanup(r.stop)
        return r

    def test_a_plain_api_call_gets_no_tools(self):
        rig = self.rig()
        rig.chat([{"role": "user", "content": "Answer only JSON: any conflicts?"}], headers={"X-Nebula-Caller": "avnh"})
        self.assertEqual(tool_names(rig.a.requests[-1]), [])

    def test_a_caller_that_asks_gets_them(self):
        rig = self.rig()
        rig.chat([{"role": "user", "content": "fix the tests"}], headers={"X-Nebula-Caller": "avnh", "X-Nebula-Tools": "on"})
        names = tool_names(rig.a.requests[-1])
        self.assertIn(team.HAND_UP, names)
        self.assertTrue(any(toolbox.is_box(n) for n in names))

    def test_a_caller_named_in_the_rules_gets_them(self):
        rig = self.rig(callers=["builder"])
        rig.chat([{"role": "user", "content": "fix the tests"}], headers={"X-Nebula-Caller": "builder"})
        self.assertTrue(any(toolbox.is_box(n) for n in tool_names(rig.a.requests[-1])))
        rig.chat([{"role": "user", "content": "fix the tests"}], headers={"X-Nebula-Caller": "avnh"})
        self.assertEqual(tool_names(rig.a.requests[-1]), [])


class Gate(unittest.TestCase):
    def turn(self, **kw):
        base = dict(headers={}, team_ctx=None, caller=None, dialect="chat", agent=None, rules=Rules.from_dict({}))
        return SimpleNamespace(**{**base, **kw})

    def test_who_gets_them(self):
        self.assertFalse(team.agent_work(self.turn(caller={"label": "avnh"})))
        self.assertTrue(team.agent_work(self.turn(dialect="room")))                       # the page's chat (Halo)
        self.assertTrue(team.agent_work(self.turn(dialect="halo")))                       # Halo's own turns
        self.assertTrue(team.agent_work(self.turn(agent=SimpleNamespace(name="assistant"))))
        self.assertTrue(team.agent_work(self.turn(team_ctx={"mode": "work"})))            # a team's work item
        self.assertTrue(team.agent_work(self.turn(caller={"label": "starweave"}, headers={"X-Nebula-Task": "t1"})))
        self.assertFalse(team.agent_work(self.turn(caller={"label": "starweave"})))       # the label alone is not a task
        self.assertFalse(team.agent_work(self.turn(dialect="room", headers={"X-Nebula-Tools": "off"})))


if __name__ == "__main__":
    unittest.main()
