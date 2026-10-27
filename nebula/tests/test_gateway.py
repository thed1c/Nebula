"""The gateway against the fake engine: the three dialects, the passed-through GETs, the error shapes."""
import json
import threading
import os
import unittest
import urllib.error
import urllib.request

from nebula.config import Config, Tier
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine, OVERFLOW

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
ANTHROPIC_TOOLS = [{"name": "get_weather", "description": "weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]


def call(base, path, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def sse(raw: bytes):
    """(event name or None, parsed data or the raw payload) per frame."""
    out = []
    for frame in raw.decode().split("\n\n"):
        if not frame.strip():
            continue
        name, data = None, None
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                payload = line[5:].strip()
                try:
                    data = json.loads(payload)
                except ValueError:
                    data = payload
        out.append((name, data))
    return out


class GatewayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeEngine()
        cfg = Config(host="127.0.0.1", port=0, tiers=[Tier(name="local", engine=cls.fake.url)], aliases=["reference"], monitor=False)
        cls.srv = make_server(cfg)
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.fake.stop()

    def setUp(self):
        self.fake.requests.clear()

    def last_engine_request(self):
        return self.fake.requests[-1]

    # --- GET
    def test_passthrough_and_models(self):
        for path in ("/health", "/props", "/slots"):
            status, ctype, body = call(self.base, path)
            self.assertEqual(status, 200, path)
            self.assertIn("json", ctype)
        self.assertEqual(call(self.base, "/metrics")[0], 200)
        props = json.loads(call(self.base, "/props")[2])
        self.assertTrue(props["modalities"]["vision"])
        self.assertIn("is_processing", json.loads(call(self.base, "/slots")[2])[0])
        models = json.loads(call(self.base, "/v1/models")[2])
        self.assertEqual([m["id"] for m in models["data"]], ["fake", "local", "reference"])
        self.assertEqual(call(self.base, "/nope")[0], 404)

    # --- chat completions
    def test_chat_echoes_the_requested_model(self):
        status, _, body = call(self.base, "/v1/chat/completions", {"model": "reference", "messages": [{"role": "user", "content": "hi"}]})
        js = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(js["model"], "reference")
        self.assertEqual(js["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(js["usage"]["prompt_tokens"], 26)
        self.assertIn("cache_n", js["timings"])
        self.assertNotIn("model", self.last_engine_request())

    def test_chat_stream_usage_only_when_asked(self):
        req = {"model": "x", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        frames = sse(call(self.base, "/v1/chat/completions", req)[2])
        self.assertEqual(frames[-1][1], "[DONE]")
        self.assertTrue(all(f[1]["model"] == "x" for f in frames[:-1]))
        self.assertFalse(any("usage" in f[1] for f in frames[:-1]))
        self.assertTrue(self.last_engine_request()["stream_options"]["include_usage"])
        req["stream_options"] = {"include_usage": True}
        frames = sse(call(self.base, "/v1/chat/completions", req)[2])
        self.assertEqual(frames[-2][1]["usage"]["completion_tokens"], 3)

    def test_chat_overflow_is_the_engines_400(self):
        req = {"model": "x", "messages": [{"role": "user", "content": "w" * (OVERFLOW + 10)}]}
        status, _, body = call(self.base, "/v1/chat/completions", req)
        self.assertEqual(status, 400)
        self.assertIn("context", json.loads(body)["error"]["message"])
        status, ctype, body = call(self.base, "/v1/chat/completions", dict(req, stream=True))
        self.assertEqual(status, 400)
        self.assertIn("json", ctype)

    def test_bad_json_is_a_400(self):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=b"[1, 2", headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 400)

    # --- Anthropic messages
    def test_messages_thinking_text_and_counts(self):
        req = {"model": "claude-x", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
        status, _, body = call(self.base, "/v1/messages", req)
        js = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(js["model"], "claude-x")
        self.assertEqual([b["type"] for b in js["content"]], ["thinking", "text"])
        self.assertEqual(js["content"][0]["thinking"], "Let me think.")
        self.assertEqual(js["content"][1]["text"], "Hello world")
        self.assertEqual(js["stop_reason"], "end_turn")
        self.assertEqual(js["usage"], {"input_tokens": 24, "cache_read_input_tokens": 2, "output_tokens": 3})
        sent = self.last_engine_request()
        self.assertEqual(sent["max_tokens"], 64)
        self.assertNotIn("chat_template_kwargs", sent)

    def test_messages_thinking_disabled(self):
        req = {"model": "m", "max_tokens": 8, "thinking": {"type": "disabled"}, "system": "be brief",
               "messages": [{"role": "user", "content": "hi"}]}
        js = json.loads(call(self.base, "/v1/messages", req)[2])
        self.assertEqual([b["type"] for b in js["content"]], ["text"])
        sent = self.last_engine_request()
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "be brief"})

    def test_messages_tool_use_and_tool_result_round_trip(self):
        req = {"model": "m", "max_tokens": 8, "tools": ANTHROPIC_TOOLS, "thinking": {"type": "disabled"},
               "messages": [{"role": "user", "content": "weather in Paris?"}]}
        js = json.loads(call(self.base, "/v1/messages", req)[2])
        self.assertEqual(js["stop_reason"], "tool_use")
        block = js["content"][0]
        self.assertEqual((block["type"], block["name"], block["input"], block["id"]), ("tool_use", "get_weather", {"city": "Paris"}, "tc_fake_1"))
        sent = self.last_engine_request()
        self.assertEqual(sent["tools"][0]["function"]["name"], "get_weather")
        # the client sends the call and its result back
        req["messages"] += [{"role": "assistant", "content": [block]},
                            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block["id"], "content": "sunny"}]},
                            {"role": "user", "content": "thanks"}]
        js = json.loads(call(self.base, "/v1/messages", req)[2])
        self.assertEqual(js["content"][-1]["text"], "Hello world")
        sent = self.last_engine_request()
        roles = [m["role"] for m in sent["messages"]]
        self.assertEqual(roles, ["user", "assistant", "tool", "user"])
        self.assertEqual(sent["messages"][2]["tool_call_id"], sent["messages"][1]["tool_calls"][0]["id"])
        self.assertEqual(sent["messages"][2]["content"], "sunny")

    def test_messages_tool_choice(self):
        req = {"model": "m", "max_tokens": 8, "tools": ANTHROPIC_TOOLS, "tool_choice": {"type": "any"},
               "messages": [{"role": "user", "content": "weather?"}]}
        call(self.base, "/v1/messages", req)
        self.assertEqual(self.last_engine_request()["tool_choice"], "required")
        req["tool_choice"] = {"type": "tool", "name": "get_weather"}
        call(self.base, "/v1/messages", req)
        self.assertEqual(self.last_engine_request()["tool_choice"], {"type": "function", "function": {"name": "get_weather"}})
        req["tool_choice"] = {"type": "none"}
        call(self.base, "/v1/messages", req)
        self.assertNotIn("tools", self.last_engine_request())

    def test_messages_stream_event_order(self):
        req = {"model": "m", "max_tokens": 8, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        status, ctype, body = call(self.base, "/v1/messages", req)
        self.assertEqual(status, 200)
        self.assertIn("event-stream", ctype)
        names = [n for n, _ in sse(body)]
        self.assertEqual(names, ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                                 "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
                                 "message_delta", "message_stop"])
        frames = sse(body)
        self.assertEqual(frames[-2][1]["usage"]["input_tokens"], 24)
        self.assertEqual(frames[0][1]["message"]["model"], "m")

    def test_messages_errors_in_anthropic_shape(self):
        req = {"model": "m", "max_tokens": 8, "stream": True, "messages": [{"role": "user", "content": "w" * (OVERFLOW + 10)}]}
        status, _, body = call(self.base, "/v1/messages", req)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["type"], "error")
        status, _, body = call(self.base, "/v1/messages", {"model": "m", "messages": "not a list"})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["type"], "invalid_request_error")

    # --- Responses
    def test_responses_items_and_usage(self):
        req = {"model": "r", "input": "hi", "instructions": "be brief"}
        status, _, body = call(self.base, "/v1/responses", req)
        js = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual((js["object"], js["status"], js["model"]), ("response", "completed", "r"))
        self.assertEqual([o["type"] for o in js["output"]], ["reasoning", "message"])
        self.assertEqual(js["output"][1]["content"][0]["text"], "Hello world")
        self.assertEqual(js["usage"]["input_tokens"], 28)       # 26 plus two for "be brief" in the fake's count
        self.assertEqual(js["usage"]["input_tokens_details"]["cached_tokens"], 2)
        self.assertEqual(self.last_engine_request()["messages"][0], {"role": "system", "content": "be brief"})

    def test_responses_function_call_and_its_output(self):
        tools = [{"type": "function", "name": "get_weather", "parameters": TOOLS[0]["function"]["parameters"]}]
        req = {"model": "r", "input": [{"role": "user", "content": "weather in Paris"}], "tools": tools,
               "reasoning": {"effort": "none"}}
        js = json.loads(call(self.base, "/v1/responses", req)[2])
        item = js["output"][0]
        self.assertEqual((item["type"], item["name"], json.loads(item["arguments"]), item["call_id"]),
                         ("function_call", "get_weather", {"city": "Paris"}, "tc_fake_1"))
        self.assertEqual(self.last_engine_request()["chat_template_kwargs"], {"enable_thinking": False})
        req["input"] += [item, {"type": "function_call_output", "call_id": item["call_id"], "output": "sunny"},
                         {"role": "user", "content": "thanks"}]
        js = json.loads(call(self.base, "/v1/responses", req)[2])
        self.assertEqual(js["output"][-1]["content"][0]["text"], "Hello world")
        sent = self.last_engine_request()
        self.assertEqual([m["role"] for m in sent["messages"]], ["user", "assistant", "tool", "user"])
        self.assertEqual(sent["messages"][2]["tool_call_id"], sent["messages"][1]["tool_calls"][0]["id"])

    def test_responses_stream_and_encrypted_reasoning(self):
        req = {"model": "r", "input": "hi", "stream": True, "include": ["reasoning.encrypted_content"]}
        status, _, body = call(self.base, "/v1/responses", req)
        frames = sse(body)
        self.assertEqual(status, 200)
        self.assertEqual(frames[0][0], "response.created")
        self.assertEqual(frames[-1][0], "response.completed")
        reasoning = [i for i in frames[-1][1]["response"]["output"] if i["type"] == "reasoning"][0]
        self.assertTrue(reasoning["encrypted_content"].startswith("strata.r1:"))

    def test_responses_errors(self):
        status, _, body = call(self.base, "/v1/responses", {"model": "r", "input": "hi", "previous_response_id": "x"})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["param"], "previous_response_id")
        req = {"model": "r", "input": "w" * (OVERFLOW + 10), "stream": True}
        status, ctype, body = call(self.base, "/v1/responses", req)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["type"], "invalid_request_error")

    def test_json_schema_format_goes_to_the_engine(self):
        fmt = {"type": "json_schema", "name": "r", "strict": True, "schema": {"type": "object"}}
        js = json.loads(call(self.base, "/v1/responses", {"model": "r", "input": "hi", "text": {"format": fmt},
                                                          "reasoning": {"effort": "none"}})[2])
        self.assertEqual(js["output"][-1]["content"][0]["text"], "Hello world")
        self.assertEqual(self.last_engine_request()["response_format"]["type"], "json_schema")


class EngineDownTest(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", 'Windows takes about 2 s per refused loopback connection: known, see docs')
    def test_unreachable_engine_is_a_502(self):
        dead = FakeEngine()
        url = dead.url
        dead.stop()
        srv = make_server(Config(port=0, tiers=[Tier(name="local", engine=url)], monitor=False))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{srv.server_port}"
            status, _, body = call(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(status, 502)
            self.assertIn("could not be reached", json.loads(body)["error"]["message"])
            self.assertEqual(call(base, "/health")[0], 502)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
