"""Webhook and schedule inputs through the gateway."""
import hashlib
import hmac
import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from nebula.config import Agent
from nebula.replay import verify
from nebula.tests.test_escalation import Rig, rows

AGENTS = [Agent("assistant", system="You assist.", tier="fast", default=True),
          Agent("reviewer", system="You review.", tier="think")]


class Receiver(HTTPServer):
    """A callback target: keeps every body and header it receives."""

    def __init__(self):
        self.got = []
        self.event = threading.Event()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.got.append((dict(self.headers), json.loads(self.rfile.read(n))))
                outer.event.set()
                self.send_response(204)
                self.end_headers()
        super().__init__(("127.0.0.1", 0), H)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}/cb"


def post(base, path, data: bytes, headers=None):
    r = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class HookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["COCKPIT_TEST_HOOK_SECRET"] = "s3cret"
        cls.receiver = Receiver()
        hooks = [{"name": "open", "agent": "assistant"},
                 {"name": "locked", "secret": "env:COCKPIT_TEST_HOOK_SECRET", "agent": "reviewer", "session": "field:issue.number",
                  "message": "Issue {issue.number}: {issue.title}"},
                 {"name": "back", "secret": "cb-secret", "agent": "assistant", "callback": cls.receiver.url, "session": "per-hook"},
                 {"name": "picky", "agent": "assistant", "tier": "field:tier", "message": "{text}"}]
        cls.rig = Rig(agents=AGENTS, hooks=hooks)

    @classmethod
    def tearDownClass(cls):
        cls.rig.stop()
        cls.receiver.shutdown()
        cls.receiver.server_close()

    def test_open_hook_answers_with_the_reply(self):
        status, js = post(self.rig.base, "/hooks/open", json.dumps({"event": "ping", "n": 1}).encode())
        self.assertEqual(status, 200, js)
        self.assertEqual(js["reply"], "Hello world")
        self.assertTrue(js["session_id"].startswith("ses_"))
        self.assertTrue(js["session_key"].startswith("hook-open-"))
        sent = self.rig.a.requests[-1]["messages"]
        self.assertEqual(sent[0]["content"], "You assist.")
        self.assertIn('"event": "ping"', sent[1]["content"])
        att = self.rig.attempts()[-1]
        self.assertEqual((att["dialect"], att["agent"], att["tier"]), ("hook:open", "assistant", "fast"))
        self.assertEqual(rows(self.rig.path, "SELECT key_source FROM sessions WHERE session_id=?", js["session_id"])[0]["key_source"], "header")

    def test_a_hook_takes_its_tier_from_the_payload(self):
        for tier, want in (("big", "big"), (None, "fast"), ("nope", "fast")):
            body = {"text": f"which tier, {tier}?"} | ({"tier": tier} if tier else {})
            status, js = post(self.rig.base, "/hooks/picky", json.dumps(body).encode())
            self.assertEqual(status, 200, js)
            done = rows(self.rig.path, "SELECT tier FROM attempts WHERE session_id=? AND state='completed'", js["session_id"])
            self.assertEqual([r["tier"] for r in done], [want], tier)

    def test_secret_bearer_and_signature(self):
        body = json.dumps({"issue": {"number": 7, "title": "Lights out"}}).encode()
        self.assertEqual(post(self.rig.base, "/hooks/locked", body)[0], 401)
        self.assertEqual(post(self.rig.base, "/hooks/locked", body, {"X-Cockpit-Secret": "wrong"})[0], 401)
        status, js = post(self.rig.base, "/hooks/locked", body, {"Authorization": "Bearer s3cret"})
        self.assertEqual(status, 200, js)
        self.assertEqual(self.rig.a.requests[-1]["messages"][1]["content"], "Issue 7: Lights out")
        self.assertEqual(js["session_key"], "hook-locked-7")
        self.assertEqual(self.rig.attempts()[-1]["tier"], "think")                     # the reviewer's floor
        sig = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        status, js2 = post(self.rig.base, "/hooks/locked", body, {"X-Hub-Signature-256": sig})
        self.assertEqual(status, 200)
        self.assertEqual(js2["session_id"], js["session_id"])                            # the same issue, the same session
        self.assertEqual(self.rig.attempts()[-1]["dialect"], "hook:locked")
        ok, lines = verify(self.rig.path)
        self.assertTrue(ok, "\n".join(lines))

    def test_callback_hook_is_accepted_and_posts_back(self):
        self.receiver.event.clear()
        status, js = post(self.rig.base, "/hooks/back", b'{"text": "late"}', {"X-Cockpit-Secret": "cb-secret"})
        self.assertEqual((status, js["accepted"], js["session_key"]), (202, True, "hook-back"))
        self.assertTrue(self.receiver.event.wait(10), "no callback arrived")
        headers, body = self.receiver.got[-1]
        self.assertEqual(headers.get("Authorization"), "Bearer cb-secret")
        self.assertEqual((body["hook"], body["reply"], body["session_key"]), ("back", "Hello world", "hook-back"))

    def test_unknown_hook_and_empty_payload(self):
        self.assertEqual(post(self.rig.base, "/hooks/nope", b"{}")[0], 404)
        self.assertEqual(post(self.rig.base, "/hooks/open", b"   ")[0], 400)
        cfg = json.loads(urllib.request.urlopen(self.rig.base + "/api/config", timeout=5).read())
        self.assertEqual([(h["name"], h["secret"], h["callback"]) for h in cfg["hooks"]],
                         [("open", False, False), ("locked", True, False), ("back", True, True), ("picky", False, False)])


class ScheduleTest(unittest.TestCase):
    def test_a_schedule_runs_its_agent(self):
        rig = Rig(agents=AGENTS, schedules=[{"name": "tick", "every": "0.3s", "agent": "assistant", "message": "Scheduled check.", "session": "per-hook"}])
        try:
            for _ in range(60):
                if rig.srv.scheduler.runs.get("tick", 0) >= 2:
                    break
                time.sleep(0.1)
            self.assertGreaterEqual(rig.srv.scheduler.runs.get("tick", 0), 2)
            attempts = rig.attempts()
            self.assertTrue(all(a["dialect"] == "hook:tick" for a in attempts))
            self.assertEqual(len(rows(rig.path, "SELECT * FROM sessions")), 1)         # per-hook: one continuing session
            self.assertEqual(rig.a.requests[-1]["messages"][-1]["content"], "Scheduled check.")
            ok, lines = verify(rig.path)
            self.assertTrue(ok, "\n".join(lines))
        finally:
            rig.stop()


if __name__ == "__main__":
    unittest.main()
