"""The ledger through the gateway: sessions found again, rewinds, rows before the engine is called, costs and
outcomes, interrupted streams, and the replay that rebuilds every transcript."""
import json
import os
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nebula.config import Config, Tier
from nebula.replay import verify
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine, OVERFLOW
from nebula.tests.test_gateway import ANTHROPIC_TOOLS, call, sse


def rows(path, sql, *args):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(sql, args)]
    finally:
        db.close()


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeEngine()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "ledger.sqlite3")
        cfg = Config(port=0, tiers=[Tier(name="local", engine=self.fake.url)], ledger=self.path, monitor=False)
        self.srv = make_server(cfg)
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.fake.stop()
        self.tmp.cleanup()

    def kinds(self, session_id=None):
        where = "WHERE session_id=?" if session_id else ""
        return [(r["kind"], r["role"]) for r in rows(self.path, f"SELECT kind, role FROM events {where} ORDER BY seq",
                                                      *([session_id] if session_id else []))]

    def chat(self, messages, **extra):
        status, _, body = call(self.base, "/v1/chat/completions", {"model": "m", "messages": messages, **extra})
        self.assertEqual(status, 200, body)
        return json.loads(body)["choices"][0]["message"]

    def assert_replays(self):
        ok, lines = verify(self.path)
        self.assertTrue(ok, "\n".join(lines))
        return lines

    # --- sessions and transcripts
    def test_turns_extend_one_session(self):
        a1 = self.chat([{"role": "user", "content": "hi"}])
        a2 = self.chat([{"role": "user", "content": "hi"}, a1, {"role": "user", "content": "more"}])
        self.assertEqual(a2["content"], "Hello world")
        sessions = rows(self.path, "SELECT * FROM sessions")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["key_source"], "opening")
        self.assertEqual(self.kinds(), [("message", "user"), ("request", None), ("reply", "assistant"),
                                        ("message", "user"), ("request", None), ("reply", "assistant")])
        attempts = rows(self.path, "SELECT * FROM attempts ORDER BY created_at")
        self.assertEqual([a["state"] for a in attempts], ["completed", "completed"])
        self.assertEqual(attempts[0]["task_id"], attempts[1]["task_id"])
        self.assertEqual(attempts[0]["tier"], "local")
        reply = rows(self.path, "SELECT * FROM events WHERE kind='reply' ORDER BY seq")[0]
        payload = json.loads(reply["payload"])
        self.assertEqual((payload["content"], payload["reasoning_content"]), ("Hello world", "Let me think."))
        self.assertEqual(reply["reasoning_visibility"], "text")
        self.assertEqual(reply["content_sha256"], attempts[0]["reply_sha256"])
        self.assert_replays()

    def test_history_sent_back_without_reasoning_still_matches(self):
        a1 = self.chat([{"role": "user", "content": "hi"}])
        a1.pop("reasoning_content")
        self.chat([{"role": "user", "content": "hi"}, a1, {"role": "user", "content": "more"}])
        self.assertNotIn("rewind", [k for k, _ in self.kinds()])
        self.assert_replays()

    def test_regeneration_records_a_rewind(self):
        self.chat([{"role": "user", "content": "hi"}])
        self.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(self.kinds(), [("message", "user"), ("request", None), ("reply", "assistant"),
                                        ("rewind", None), ("request", None), ("reply", "assistant")])
        rewind = rows(self.path, "SELECT payload FROM events WHERE kind='rewind'")[0]
        self.assertEqual(json.loads(rewind["payload"]), {"keep": 1, "dropped": 1})
        self.assert_replays()

    def test_edited_turn_rewinds_and_appends(self):
        a1 = self.chat([{"role": "user", "content": "hi"}])
        self.chat([{"role": "user", "content": "hi"}, a1, {"role": "user", "content": "A"}])
        self.chat([{"role": "user", "content": "hi"}, a1, {"role": "user", "content": "B"}])
        kinds = self.kinds()
        self.assertEqual(kinds[6:], [("rewind", None), ("message", "user"), ("request", None), ("reply", "assistant")])
        self.assertEqual(json.loads(rows(self.path, "SELECT payload FROM events WHERE kind='rewind'")[0]["payload"]),
                         {"keep": 2, "dropped": 2})
        self.assertEqual(len(rows(self.path, "SELECT * FROM sessions")), 1)
        self.assert_replays()

    def test_different_openings_are_different_sessions(self):
        self.chat([{"role": "user", "content": "hi"}])
        self.chat([{"role": "user", "content": "hello"}])
        self.chat([{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}])
        self.assertEqual(len(rows(self.path, "SELECT * FROM sessions")), 3)
        self.assert_replays()

    def test_session_key_sources(self):
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        import urllib.request
        r = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json", "X-Cockpit-Session": "abc"})
        urllib.request.urlopen(r, timeout=5).read()
        call(self.base, "/v1/messages", {"model": "m", "max_tokens": 8, "metadata": {"user_id": "user_1_session_2"},
                                         "messages": [{"role": "user", "content": "hi"}]})
        call(self.base, "/v1/chat/completions", dict(req, user="u9"))
        call(self.base, "/v1/chat/completions", req)
        got = {(s["session_key"], s["key_source"]) for s in rows(self.path, "SELECT * FROM sessions")}
        self.assertEqual(got, {("abc", "header"), ("user_1_session_2", "metadata.user_id"), ("u9", "user"), ("", "opening")})
        self.assert_replays()

    def test_anthropic_tool_round_trip_matches_by_content(self):
        req = {"model": "m", "max_tokens": 8, "tools": ANTHROPIC_TOOLS, "thinking": {"type": "disabled"},
               "messages": [{"role": "user", "content": "weather in Paris?"}]}
        js = json.loads(call(self.base, "/v1/messages", req)[2])
        block = js["content"][0]
        req["messages"] += [{"role": "assistant", "content": [block]},
                            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block["id"], "content": "sunny"}]},
                            {"role": "user", "content": "thanks"}]
        call(self.base, "/v1/messages", req)
        self.assertEqual(self.kinds(), [("message", "user"), ("request", None), ("reply", "assistant"),
                                        ("message", "tool"), ("message", "user"), ("request", None), ("reply", "assistant")])
        reply = json.loads(rows(self.path, "SELECT payload FROM events WHERE kind='reply' ORDER BY seq")[0]["payload"])
        self.assertEqual(reply["tool_calls"][0]["function"], {"name": "get_weather", "arguments": '{"city": "Paris"}'})
        self.assertIsNone(reply["content"])
        self.assert_replays()

    def test_responses_turns_share_the_session(self):
        req = {"model": "r", "input": [{"role": "user", "content": "hi"}], "instructions": "be brief"}
        js = json.loads(call(self.base, "/v1/responses", req)[2])
        req["input"] += [o for o in js["output"]] + [{"role": "user", "content": "more"}]
        call(self.base, "/v1/responses", req)
        self.assertEqual(len(rows(self.path, "SELECT * FROM sessions")), 1)
        self.assertNotIn("rewind", [k for k, _ in self.kinds()])
        self.assert_replays()

    def test_images_are_stored_as_references(self):
        url = "data:image/png;base64," + "A" * 5000
        self.chat([{"role": "user", "content": [{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {"url": url}}]}],
                  chat_template_kwargs={"enable_thinking": False})
        payload = json.loads(rows(self.path, "SELECT payload FROM events WHERE kind='message'")[0]["payload"])
        self.assertEqual(payload["content"][1]["type"], "image_ref")
        self.assertEqual(payload["content"][1]["length"], len(url))
        self.assertNotIn("AAAA", rows(self.path, "SELECT payload FROM events WHERE kind='message'")[0]["payload"])
        self.assert_replays()

    # --- the ordering invariant
    def test_rows_exist_before_the_engine_is_called(self):
        seen = {}

        def look(body):
            seen["attempts"] = rows(self.path, "SELECT * FROM attempts")
            seen["events"] = rows(self.path, "SELECT kind FROM events ORDER BY seq")

        self.fake.on_request = look
        self.chat([{"role": "user", "content": "hi"}])
        self.assertEqual([a["state"] for a in seen["attempts"]], ["running"])
        self.assertEqual([e["kind"] for e in seen["events"]], ["message", "request"])
        self.assertEqual([a["state"] for a in rows(self.path, "SELECT * FROM attempts")], ["completed"])

    # --- what the engine answered, or did not
    def test_costs_and_outcome(self):
        self.chat([{"role": "user", "content": "hi"}])
        cost = rows(self.path, "SELECT * FROM costs")[0]
        self.assertEqual((cost["prompt_tokens"], cost["completion_tokens"], cost["prompt_n"], cost["cache_n"], cost["draft_n_accepted"]),
                         (26, 3, 24, 2, 1))
        out = rows(self.path, "SELECT * FROM outcomes")[0]
        self.assertEqual((out["kind"], out["value"]), ("transport", "stop"))
        dec = rows(self.path, "SELECT * FROM decisions")[0]
        self.assertEqual((dec["tier"], dec["reason"], dec["advisory"], dec["thinking"]), ("local", "floor", 0, None))
        self.assertEqual(rows(self.path, "SELECT * FROM attempts")[0]["model_id"], "fake")

    def test_refused_request_is_a_failed_attempt(self):
        status, _, _ = call(self.base, "/v1/chat/completions",
                            {"model": "m", "messages": [{"role": "user", "content": "w" * (OVERFLOW + 10)}]})
        self.assertEqual(status, 400)
        att = rows(self.path, "SELECT * FROM attempts")[0]
        self.assertEqual(att["state"], "failed")
        self.assertIn("context", att["error"])
        self.assertEqual(rows(self.path, "SELECT value FROM outcomes")[0]["value"], "error")
        self.assertEqual([k for k, _ in self.kinds()], ["message", "request"])
        self.assert_replays()
        for path, req in (("/v1/messages", {"model": "m", "max_tokens": 8, "stream": True,
                                             "messages": [{"role": "user", "content": "w" * (OVERFLOW + 10)}]}),
                          ("/v1/responses", {"model": "m", "input": "w" * (OVERFLOW + 10), "stream": True})):
            self.assertEqual(call(self.base, path, req)[0], 400)
        self.assertEqual([a["state"] for a in rows(self.path, "SELECT * FROM attempts")], ["failed"] * 3)
        self.assert_replays()

    def test_client_hanging_up_leaves_an_interrupted_reply(self):
        req = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "answer slowly"}]}
        data = json.dumps(req).encode()
        s = socket.create_connection(("127.0.0.1", self.srv.server_port), timeout=5)
        s.sendall(b"POST /v1/chat/completions HTTP/1.0\r\nContent-Type: application/json\r\n"
                  + f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
        got = b""
        while b"data:" not in got:
            got += s.recv(4096)
        s.close()
        for _ in range(50):
            att = rows(self.path, "SELECT * FROM attempts")[0]
            if att["state"] != "running":
                break
            time.sleep(0.1)
        self.assertEqual(att["state"], "interrupted")
        reply = rows(self.path, "SELECT * FROM events WHERE kind='reply'")
        self.assertEqual(len(reply), 1)
        self.assertEqual(reply[0]["completeness"], "interrupted")
        self.assertEqual(rows(self.path, "SELECT value FROM outcomes")[0]["value"], "interrupted")
        self.assert_replays()

    # --- the replay
    def test_replay_detects_tampering(self):
        a1 = self.chat([{"role": "user", "content": "hi"}])
        self.chat([{"role": "user", "content": "hi"}, a1, {"role": "user", "content": "more"}])
        lines = self.assert_replays()
        self.assertIn("6 events, 2 attempts, 4 messages", lines[0])
        db = sqlite3.connect(self.path)
        db.execute("UPDATE events SET payload=? WHERE seq=1", (json.dumps({"role": "user", "content": "hi!"}),))
        db.commit()
        db.close()
        ok, lines = verify(self.path)
        self.assertFalse(ok)
        self.assertTrue(any("content_sha256" in l for l in lines))
        self.assertTrue(any("not the request the attempt recorded" in l for l in lines))

    def test_replay_detects_a_lost_event(self):
        self.chat([{"role": "user", "content": "hi"}])
        db = sqlite3.connect(self.path)
        db.execute("DELETE FROM events WHERE seq=2")
        db.commit()
        db.close()
        ok, lines = verify(self.path)
        self.assertFalse(ok)
        self.assertTrue(any("seq gap" in l for l in lines))


class NoLedgerTest(unittest.TestCase):
    def test_no_ledger_still_serves(self):
        fake = FakeEngine()
        srv = make_server(Config(port=0, tiers=[Tier(name="local", engine=fake.url)], ledger=None, monitor=False))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            status, _, body = call(f"http://127.0.0.1:{srv.server_port}", "/v1/chat/completions",
                                   {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(status, 200)
            self.assertIsNone(srv.ledger)
        finally:
            srv.shutdown()
            srv.server_close()
            fake.stop()


class OneWriterTest(unittest.TestCase):
    def test_a_second_nebula_on_the_same_ledger_is_refused_with_the_reason(self):
        import tempfile
        from nebula.config import Config
        from nebula.server import LedgerInUse, make_server
        d = tempfile.mkdtemp()
        cfg = lambda: Config.from_dict({"port": 0, "monitor": False, "ledger": os.path.join(d, "shared.sqlite3"),
                                        "tiers": [{"name": "fast", "engine": "http://127.0.0.1:9"}],
                                        "lan": {"decided": True, "discoverable": False}})
        first = make_server(cfg())
        try:
            with self.assertRaises(LedgerInUse) as cm:
                make_server(cfg())
            self.assertIn("is using the ledger", str(cm.exception))
            self.assertIn(f"pid {os.getpid()}", str(cm.exception))
        finally:
            first.server_close(); first.tools.close()
        again = make_server(cfg())                          # free once the first one closed
        again.server_close(); again.tools.close()


if __name__ == "__main__":
    unittest.main()
