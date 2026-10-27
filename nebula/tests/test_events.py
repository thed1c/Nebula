"""cockpit.events: the engine's chunks as events, and the template message form made into the engine's."""
import unittest

from nebula.events import Stream, completion_chunks, engine_request, template_to_openai, template_tools_to_openai


def chunks():
    yield {"model": "m", "id": "x", "choices": [{"delta": {"reasoning_content": "think "}}]}
    yield {"choices": [{"delta": {"reasoning_content": "more"}}]}
    yield {"choices": [{"delta": {"content": "Hel"}}]}
    yield {"choices": [{"delta": {"content": "lo"}}]}
    yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "tc1", "function": {"name": "f", "arguments": '{"a"'}}]}}]}
    yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ': 1}'}}]}}]}
    yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "timings": {"prompt_n": 10, "cache_n": 5}}
    yield {"choices": [], "usage": {"prompt_tokens": 15, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 5}}}


class StreamTest(unittest.TestCase):
    def test_events_and_done(self):
        s = Stream(chunks())
        ev = list(s)
        self.assertEqual([(e.kind, e.text) for e in ev[:4]],
                         [("reasoning", "think "), ("reasoning", "more"), ("content", "Hel"), ("content", "lo")])
        self.assertEqual(ev[4].kind, "tool_call")
        self.assertEqual((ev[4].call.id, ev[4].call.name, ev[4].call.arguments), ("tc1", "f", {"a": 1}))
        self.assertTrue(s.used_tool)
        self.assertEqual((s.done.finish, s.done.prompt_tokens, s.done.completion_tokens), ("tool_calls", 15, 7))
        self.assertEqual((s.done.read_tokens, s.done.cache_tokens), (10, 5))
        self.assertEqual(s.done.model, "m")

    def test_counts_without_timings(self):
        s = Stream(iter([{"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]},
                         {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 1,
                                                   "prompt_tokens_details": {"cached_tokens": 4}}}]))
        list(s)
        self.assertEqual((s.done.read_tokens, s.done.cache_tokens), (5, 4))

    def test_call_without_id_gets_one(self):
        s = Stream(iter([{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "g", "arguments": "{}"}}]},
                                       "finish_reason": "tool_calls"}]}]))
        ev = list(s)
        self.assertTrue(ev[0].call.id.startswith("call_"))

    def test_whole_completion_as_chunks(self):
        obj = {"id": "c", "model": "m", "choices": [{"index": 0, "finish_reason": "stop",
                                                     "message": {"role": "assistant", "content": "hi", "reasoning_content": "r"}}],
               "usage": {"prompt_tokens": 3, "completion_tokens": 1}, "timings": {"prompt_n": 3, "cache_n": 0}}
        s = Stream(completion_chunks(obj))
        self.assertEqual([(e.kind, e.text) for e in s], [("reasoning", "r"), ("content", "hi")])
        self.assertEqual(s.done.finish, "stop")


class ConvertTest(unittest.TestCase):
    def test_tool_calls_and_results_share_ids(self):
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "reasoning_content": "r",
                 "tool_calls": [{"function": {"name": "a", "arguments": {"x": 1}}}, {"function": {"name": "b", "arguments": {}}}]},
                {"role": "tool", "content": "ra"}, {"role": "tool", "content": "rb"},
                {"role": "user", "content": [{"type": "text", "text": "see"}, {"type": "image", "source": "data:image/png;base64,AA"}]}]
        out = template_to_openai(msgs)
        calls = out[1]["tool_calls"]
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])
        self.assertEqual(calls[0]["function"]["arguments"], '{"x": 1}')
        self.assertEqual(out[1]["reasoning_content"], "r")
        self.assertEqual([m["tool_call_id"] for m in out[2:4]], [c["id"] for c in calls])
        self.assertEqual(out[4]["content"][1], {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}})

    def test_tool_result_without_a_call_gets_an_id(self):
        out = template_to_openai([{"role": "tool", "content": "orphan"}])
        self.assertTrue(out[0]["tool_call_id"].startswith("call_"))

    def test_tools_and_request(self):
        tools = template_tools_to_openai([{"name": "f", "description": "d", "parameters": {"type": "object"}}])
        self.assertEqual(tools[0], {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}})
        body = engine_request([{"role": "user", "content": "x"}], tools, {"enable_thinking": False}, max_tokens=5, temperature=None)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["max_tokens"], 5)
        self.assertNotIn("temperature", body)
        self.assertTrue(body["stream"] and body["stream_options"]["include_usage"])
        self.assertEqual(engine_request([], None, {"reasoning_effort": "low"})["chat_template_kwargs"], {"reasoning_effort": "low"})
        self.assertNotIn("chat_template_kwargs", engine_request([], None, {}))


if __name__ == "__main__":
    unittest.main()
