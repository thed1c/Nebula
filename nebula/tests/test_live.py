"""The page's live view: what is running, what started and landed, the handoffs, the deltas, and the SSE stream."""
import json
import threading
import time
import unittest
import urllib.request

from nebula.activity import Activity, Watch
from nebula.tests.test_escalation import Rig
from nebula.tests.test_ui import req


class ActivityTest(unittest.TestCase):
    def test_wait_wakes_on_a_bump_and_times_out_otherwise(self):
        a = Activity()
        t0 = time.monotonic()
        self.assertEqual(a.wait(a.version, 0.05), 0)
        self.assertGreaterEqual(time.monotonic() - t0, 0.04)
        threading.Timer(0.05, a.bump).start()
        self.assertEqual(a.wait(0, 5), 1)

    def test_tokens_count_per_attempt_and_go_when_it_finishes(self):
        a = Activity()
        a.token("att_1")
        a.token("att_1", 3)
        a.token(None)
        self.assertEqual(a.snapshot()[0], {"att_1": 4})
        a.finished("att_1")
        self.assertEqual(a.snapshot()[0], {})

    def test_close_releases_a_waiter(self):
        a = Activity()
        threading.Timer(0.05, a.close).start()
        t0 = time.monotonic()
        a.wait(a.version, 5)
        self.assertLess(time.monotonic() - t0, 2)
        self.assertTrue(a.closing)


class LiveTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()
        self.ledger = self.rig.srv.ledger

    def tearDown(self):
        self.rig.stop()

    def test_a_turn_starts_runs_and_lands(self):
        watch = Watch(self.ledger)
        self.assertEqual(watch.tick()["running"], [])
        seen = {}

        def look(_body):                         # in flight: the attempt is running, the stream has not begun
            seen["live"] = json.loads(req(self.rig.base, "/api/live")[2])
        self.rig.a.on_request = look
        self.rig.chat([{"role": "user", "content": "hi there"}])
        self.rig.a.on_request = None
        running = seen["live"]["running"]
        self.assertEqual([(a["tier"], a["dialect"], a["state"], a["title"], a["tokens"]) for a in running],
                         [("fast", "chat", "running", "hi there", 0)])
        t = watch.tick()
        self.assertEqual(t["running"], [])
        self.assertEqual([(a["tier"], a["dialect"], a["handoff"]) for a in t["started"]], [("fast", "chat", None)])
        self.assertEqual([(a["attempt_id"], a["state"], a["model_id"], a["predicted_n"], a["title"]) for a in t["landed"]],
                         [(running[0]["attempt_id"], "completed", "fastfake", 3, "hi there")])
        again = watch.tick()                     # nothing new since
        self.assertEqual([again[k] for k in ("started", "landed", "handoffs")], [[], [], []])

    def test_each_call_says_who_called_it(self):
        """The map sends a caller's light through the Quasar it called (an entry point's calls are its caller's)."""
        watch = Watch(self.ledger)
        watch.tick()
        self.rig.chat([{"role": "user", "content": "hi"}])
        att = self.rig.attempts()[-1]["attempt_id"]
        t = watch.tick()
        self.assertEqual([a["caller"] for a in t["started"] + t["landed"]], [None, None])
        self.rig.chat([{"role": "user", "content": "again"}])
        att = self.rig.attempts()[-1]["attempt_id"]
        self.ledger.decide(att, tier="fast", thinking=None, reason="test", inputs={"caller": "entry-kids"})
        t = watch.tick()
        self.assertEqual([a["caller"] for a in t["started"] + t["landed"]], ["entry-kids", "entry-kids"])

    def test_a_crossing_is_a_handoff_from_one_tier_to_another(self):
        self.rig.chat([{"role": "user", "content": "first"}])
        watch = Watch(self.ledger)
        watch.tick()                             # the first look reports only what runs
        self.rig.chat([{"role": "user", "content": "first"}, {"role": "assistant", "content": "Hello world"},
                       {"role": "user", "content": "now on big"}], model="cockpit/big")
        t = watch.tick()
        self.assertEqual([(h["from_tier"], h["to_tier"], h["dialect"], h["title"]) for h in t["handoffs"]],
                         [("fast", "big", "chat", "first")])
        states = sorted((a["tier"], a["state"]) for a in t["landed"])
        self.assertEqual(states, [("big", "completed"), ("fast", "escalated")])
        dest = [a for a in t["started"] if a["tier"] == "big"]
        self.assertEqual(dest[0]["handoff"], t["handoffs"][0]["handoff_id"])

    def test_a_move_up_is_an_escalation_with_a_note_and_a_move_down_is_not(self):
        first = [{"role": "user", "content": "first"}]
        self.rig.chat(first)
        watch = Watch(self.ledger, ["fast", "think", "big"])
        watch.tick()
        self.rig.chat(first + [{"role": "assistant", "content": "Hello world"}, {"role": "user", "content": "go big"}],
                      model="cockpit/big")
        t = watch.tick()
        self.assertEqual([(e["from_tier"], e["to_tier"], e["reason"], e["title"]) for e in t["escalations"]],
                         [("fast", "big", "named:big", "first")])
        note = json.loads(req(self.rig.base, "/api/escalations/" + t["escalations"][0]["decision_id"])[2])
        self.assertEqual((note["from_tier"], note["to_tier"], note["title"]), ("fast", "big", "first"))
        h = note["handoff"]
        self.assertEqual(h["handoff_id"], t["handoffs"][0]["handoff_id"])
        self.assertIn("You are continuing a conversation that another model handed over.", h["note"])
        self.assertEqual(h["packet"]["original_task"], "first")
        by_handoff = json.loads(req(self.rig.base, "/api/escalations/" + h["handoff_id"])[2])
        self.assertEqual(by_handoff["decision_id"], note["decision_id"])
        sid = note["session_id"]
        marks = [i for i in json.loads(req(self.rig.base, "/api/sessions/" + sid)[2])["items"] if i["kind"] == "handoff"]
        self.assertEqual([m["handoff_id"] for m in marks], [h["handoff_id"]])
        self.rig.chat([{"role": "user", "content": "other"}], model="cockpit/think")   # a new session named think: up
        self.rig.chat([{"role": "user", "content": "other"}, {"role": "assistant", "content": "Hello world"},
                       {"role": "user", "content": "back down"}], model="cockpit/fast")
        moves = [(e["from_tier"], e["to_tier"]) for e in watch.tick()["escalations"]]
        self.assertNotIn(("think", "fast"), moves)
        self.assertEqual(req(self.rig.base, "/api/escalations/dec_nothing")[0], 404)

    def test_streamed_deltas_count_while_the_attempt_runs(self):
        seen = {}
        orig = self.ledger.activity.token

        def token(attempt_id, n=1):
            orig(attempt_id, n)
            seen.setdefault("counts", []).append(self.ledger.activity.snapshot()[0].get(attempt_id))
        self.ledger.activity.token = token
        status, _h, _raw = req(self.rig.base, "/v1/chat/completions",
                               {"model": "m", "stream": True, "messages": [{"role": "user", "content": "count"}]})
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(seen["counts"]), 2)
        self.assertEqual(seen["counts"], sorted(seen["counts"]))
        self.assertEqual(self.ledger.activity.snapshot()[0], {})   # gone once the attempt closed

    def test_the_stream_sends_a_tick_for_a_turn(self):
        r = urllib.request.urlopen(self.rig.base + "/api/stream", timeout=10)
        self.assertIn("text/event-stream", r.headers["Content-Type"])

        def read_tick():
            event, data = None, None
            while True:
                line = r.readline().decode().rstrip("\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:])
                elif line == "" and event:
                    return event, data
        self.assertEqual(read_tick()[0], "tick")
        threading.Thread(target=self.rig.chat, args=([{"role": "user", "content": "streamed"}],), daemon=True).start()
        landed = []
        deadline = time.monotonic() + 10
        while not landed and time.monotonic() < deadline:
            event, data = read_tick()
            self.assertEqual(event, "tick")
            landed = data["landed"]
        self.assertEqual([(a["tier"], a["state"], a["title"]) for a in landed], [("fast", "completed", "streamed")])
        r.close()


if __name__ == "__main__":
    unittest.main()
