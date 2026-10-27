"""Agents through the gateway: selection, the system prompt, floors and ceilings, the output shape, the tool loop
through a real (fake) MCP server, and the secretary's compactions."""
import json
import sys
import unittest
import urllib.request

from nebula.config import Agent
from nebula.replay import verify
from nebula.tests.test_escalation import Rig, rows
from nebula.tests.test_gateway import call, sse
from nebula.tests.test_ui import req as ui_req, frames as ui_frames

FAKE_MCP = {"fake": {"command": sys.executable, "args": ["nebula/_vendor/strata_serve/mcp_fake_server.py"]}}
AGENTS = [Agent("helper", system="You are Helper.", tier="think", ceiling="think", tools=["fake"]),
          Agent("writer", system="You write.", tier="fast", output_schema={"type": "object", "properties": {"text": {"type": "string"}}}),
          Agent("secretary", system="You keep notes.", tier="fast", event=True),
          Agent("plain", system="Plain agent.", default=True)]


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig(agents=AGENTS, mcp_servers=FAKE_MCP, fast_ctx=230)

    @classmethod
    def tearDownClass(cls):
        cls.rig.stop()

    def setUp(self):
        self.rig.a.requests.clear()
        self.rig.b.requests.clear()

    def test_selection_system_floor_and_ceiling(self):
        rig = self.rig
        a1 = rig.chat([{"role": "user", "content": "hi helper"}], model="cockpit/helper")
        sent = rig.a.requests[-1]
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "You are Helper."})
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": True})      # the agent's floor is think
        self.assertTrue(any(t["function"]["name"] == "fake__echo" for t in sent["tools"]))
        att = rig.attempts()[-1]
        self.assertEqual((att["agent"], att["tier"]), ("helper", "think"))
        self.assertEqual(json.loads(rig.decisions()[-1]["inputs"])["agent"], "helper")
        # the ceiling holds the ladder: an escalate to big stays on think
        rig.chat([{"role": "user", "content": "hi helper"}, a1, {"role": "user", "content": "more"}], model="cockpit/helper",
                 headers={"X-Cockpit-Escalate": "big"})
        self.assertEqual(len(rig.b.requests), 0)
        d = rig.decisions()[-1]
        self.assertEqual((d["tier"], d["reason"]), ("think", "escalate:big,ceiling:think"))
        ok, lines = verify(rig.path)
        self.assertTrue(ok, "\n".join(lines))

    def test_header_default_and_an_existing_system_prompt(self):
        rig = self.rig
        rig.chat([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}], model="m",
                 headers={"X-Cockpit-Agent": "writer"})
        sent = rig.a.requests[-1]
        self.assertEqual(sent["messages"][0]["content"], "You write.\n\nBe brief.")
        self.assertEqual(sent["response_format"]["json_schema"]["name"], "writer")
        self.assertEqual(rig.attempts()[-1]["agent"], "writer")
        rig.chat([{"role": "user", "content": "no agent named"}], model="m")
        self.assertEqual(rig.a.requests[-1]["messages"][0]["content"], "Plain agent.")   # the default agent
        self.assertEqual(rig.attempts()[-1]["agent"], "plain")
        status, _, body = call(rig.base, "/v1/chat/completions", {"model": "cockpit/nobody", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(status, 200)                                                     # not an agent, not a tier: plain
        self.assertEqual(rig.attempts()[-1]["agent"], "plain")
        r = urllib.request.Request(rig.base + "/v1/chat/completions", data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "x"}]}).encode(),
                                   headers={"Content-Type": "application/json", "X-Cockpit-Agent": "nobody"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(r, timeout=5)
        self.assertEqual(cm.exception.code, 400)

    def test_floors_are_per_agent(self):
        rig = self.rig
        hist = [{"role": "user", "content": "shared session"}]
        a1 = rig.chat(hist, model="cockpit/helper", headers={"X-Cockpit-Session": "shared"})
        hist += [a1, {"role": "user", "content": "now the writer"}]
        rig.chat(hist, model="cockpit/writer", headers={"X-Cockpit-Session": "shared"})
        self.assertEqual([(a["agent"], a["tier"]) for a in rig.attempts()][-2:], [("helper", "think"), ("writer", "fast")])

    def test_tool_loop_runs_the_agents_tools_and_hides_them(self):
        rig = self.rig
        status, _, body = call(rig.base, "/v1/chat/completions",
                               {"model": "cockpit/helper", "messages": [{"role": "user", "content": "please call the tool"}]})
        js = json.loads(body)
        self.assertEqual(status, 200, body)
        msg = js["choices"][0]["message"]
        self.assertEqual(msg["content"], "Calling it. Tool said: hi from the engine")
        self.assertNotIn("tool_calls", msg)
        self.assertEqual(js["choices"][0]["finish_reason"], "stop")
        from nebula.tests.fake_engine import usage_for
        self.assertEqual(js["usage"]["prompt_tokens"], sum(usage_for(r)["prompt_tokens"] for r in rig.a.requests))   # two rounds, summed
        self.assertGreater(js["usage"]["prompt_tokens"], 2 * 26)
        self.assertEqual(len(rig.a.requests), 2)
        second = rig.a.requests[1]["messages"]
        self.assertEqual([m["role"] for m in second[-2:]], ["assistant", "tool"])
        self.assertEqual(second[-1]["content"], "hi from the engine")
        self.assertEqual(second[-2]["tool_calls"][0]["function"]["name"], "fake__echo")
        kinds = [r["kind"] for r in rows(rig.path, "SELECT kind FROM events WHERE session_id=(SELECT session_id FROM attempts ORDER BY created_at DESC, rowid DESC LIMIT 1) ORDER BY seq")]
        self.assertEqual(kinds, ["message", "request", "tool_round", "reply"])
        cost = rows(rig.path, "SELECT * FROM costs ORDER BY recorded_at DESC, rowid DESC LIMIT 1")[0]
        self.assertEqual((cost["prompt_n"], cost["predicted_n"]), (48, 6))
        ok, lines = verify(rig.path)
        self.assertTrue(ok, "\n".join(lines))

    def test_the_live_feed_shows_a_tool_round_with_its_agent_and_server(self):
        from nebula.activity import Watch
        watch = Watch(self.rig.srv.ledger, ["fast", "think", "big"])
        watch.tick()
        status, _, body = call(self.rig.base, "/v1/chat/completions",
                               {"model": "cockpit/helper", "messages": [{"role": "user", "content": "now call the tool"}]})
        self.assertEqual(status, 200, body)
        rounds = watch.tick()["tool_rounds"]
        self.assertEqual([(r["agent"], r["tier"], r["tools"], r["servers"]) for r in rounds],
                         [("helper", "think", ["fake__echo"], ["fake"])])

    def test_tool_loop_streams_and_through_the_other_dialects(self):
        rig = self.rig
        status, _, raw = call(rig.base, "/v1/chat/completions",
                              {"model": "cockpit/helper", "stream": True, "messages": [{"role": "user", "content": "please call the tool"}]})
        fr = [d for _n, d in sse(raw) if isinstance(d, dict) and d.get("choices")]
        text = "".join(f["choices"][0]["delta"].get("content", "") for f in fr)
        self.assertEqual(text, "Calling it. Tool said: hi from the engine")
        self.assertFalse(any("tool_calls" in f["choices"][0]["delta"] for f in fr))
        js = json.loads(call(rig.base, "/v1/messages", {"model": "cockpit/helper", "max_tokens": 50,
                                                        "messages": [{"role": "user", "content": "please call the tool"}]})[2])
        self.assertEqual(js["stop_reason"], "end_turn")                                 # the tool round is not a tool_use stop
        self.assertEqual([b["type"] for b in js["content"]], ["thinking", "text", "thinking", "text"])   # thinking on, two rounds
        self.assertEqual("".join(b["text"] for b in js["content"] if b["type"] == "text"), "Calling it. Tool said: hi from the engine")
        js = json.loads(call(rig.base, "/v1/responses", {"model": "cockpit/helper", "input": "please call the tool"})[2])
        self.assertEqual([o["type"] for o in js["output"]], ["reasoning", "message", "reasoning", "message"])
        self.assertEqual("".join(o["content"][0]["text"] for o in js["output"] if o["type"] == "message"),
                         "Calling it. Tool said: hi from the engine")

    def test_a_clients_own_tools_still_go_back_to_it(self):
        rig = self.rig
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
        js = json.loads(call(rig.base, "/v1/chat/completions", {"model": "cockpit/helper", "tools": tools,
                                                                "messages": [{"role": "user", "content": "weather in Paris"}]})[2])
        self.assertEqual(js["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(js["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(len(rig.a.requests), 1)

    def test_secretary_compacts(self):
        rig = self.rig
        rig.chat([{"role": "user", "content": "w" * 600}], model="cockpit/writer")     # past three quarters of 230
        briefing = rows(rig.path, "SELECT payload FROM events WHERE kind='briefing' ORDER BY seq DESC LIMIT 1")
        self.assertTrue(briefing)
        b = json.loads(briefing[0]["payload"])
        self.assertEqual((b["agent"], b["tier"]), ("secretary", "fast"))
        comp = rig.a.requests[-1]
        self.assertEqual(comp["response_format"]["json_schema"]["name"], "briefing")
        self.assertEqual(comp["messages"][0], {"role": "system", "content": "You keep notes."})

    def test_page_picks_an_agent(self):
        rig = self.rig
        cfg = json.loads(ui_req(rig.base, "/api/config")[2])
        self.assertEqual([a["name"] for a in cfg["agents"]], ["helper", "writer", "secretary", "plain", "assistant"])
        status, headers, raw = ui_req(rig.base, "/api/chat", {"text": "please call the tool", "agent": "helper"})
        self.assertEqual(status, 200)
        sid = headers.get("X-Cockpit-Session-Id")
        v = json.loads(ui_req(rig.base, "/api/sessions/" + sid)[2])
        self.assertEqual([i["kind"] for i in v["items"]], ["message", "tool_round", "reply"])
        self.assertEqual(v["items"][1]["calls"][0]["name"], "fake__echo")
        self.assertEqual(v["items"][1]["results"], ["hi from the engine"])
        self.assertEqual(v["items"][2]["attempt"]["agent"], "helper")
        self.assertEqual(ui_req(rig.base, "/api/chat", {"text": "x", "agent": "nobody"})[0], 400)


if __name__ == "__main__":
    unittest.main()
