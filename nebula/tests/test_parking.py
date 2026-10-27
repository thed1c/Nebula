"""Step C: the size of a request before routing, and the parking lot for what no ticked model holds."""
import json
import threading
import time
import unittest
import urllib.request

from nebula.config import Agent
from nebula.engine import Engine
from nebula.parking import Lot, ParkedTooBig
from nebula.replay import verify
from nebula.sizing import estimate_tokens, holds, needed
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_escalation import Rig, rows
from nebula.tests.test_gateway import call
from nebula.tests.test_ui import req as ui_req

RULES = {"parking_wait_seconds": 1.5, "parking_recheck_seconds": 0.2}


class SizingTest(unittest.TestCase):
    def test_estimate_and_tokenizer(self):
        fake = FakeEngine()
        try:
            eng = Engine(fake.url)
            msgs = [{"role": "user", "content": "x" * 400}]
            n, how = estimate_tokens(eng, msgs, None)
            self.assertEqual((n, how), (int(400 / 4 * 1.1) + 6, "estimate"))        # no tokenizer: remembered after one 404
            self.assertTrue(eng._no_tokenize)
            fake.tokenizer = True
            eng2 = Engine(fake.url)
            self.assertEqual(estimate_tokens(eng2, msgs, None), (200 + 6, "tokenizer"))
            self.assertEqual(estimate_tokens(None, msgs, [{"type": "function", "function": {"name": "f"}}])[1], "estimate")
            self.assertEqual(needed(100, 50), 150)
            self.assertEqual(needed(100, None), 1124)
            self.assertEqual(needed(100, None, context=800), 200)                    # the reserve capped at an eighth
            self.assertTrue(holds(None, 10 ** 9))
            self.assertTrue(holds(150, 100, 50))
            self.assertFalse(holds(149, 100, 50))
            self.assertTrue(holds(160, 140))                                          # 140 + min(1024, 20) = 160
        finally:
            fake.stop()


class LotTest(unittest.TestCase):
    def test_release_and_refusal(self):
        lot = Lot(recheck_seconds=0.05, wait_seconds=0.3)
        with self.assertRaises(ParkedTooBig):
            lot.park({"reason": "too big"}, lambda: None)
        self.assertEqual(lot.parked(), [])
        self.assertEqual(lot.history[-1]["left"], "refused")
        box = {"tier": None}
        out = {}
        t = threading.Thread(target=lambda: out.update(dict(zip(("item", "tier"), lot.park({"reason": "r"}, lambda: box["tier"], wait_seconds=5)))))
        t.start()
        time.sleep(0.1)
        self.assertEqual(len(lot.parked()), 1)
        box["tier"] = "big"
        lot.poke()
        t.join(5)
        self.assertEqual((out["tier"], out["item"]["left"], out["item"]["to"]), ("big", "released", "big"))
        self.assertEqual(lot.status()["recent"][-1]["left"], "released")


AGENTS = [Agent("small", system="Small.", models=["fast"]), Agent("wide", system="Wide.", models=["fast", "big"])]


class GatewayParkingTest(unittest.TestCase):
    def test_parked_visible_then_refused(self):
        rig = Rig(agents=AGENTS, fast_ctx=100, big_ctx=100, rules=RULES)
        try:
            out = {}
            t = threading.Thread(target=lambda: out.update(zip(("status", "ctype", "body"), call(rig.base, "/v1/chat/completions",
                {"model": "cockpit/small", "max_tokens": 10, "messages": [{"role": "user", "content": "w" * 1000}]}))))
            t.start()
            time.sleep(0.4)
            live = json.loads(urllib.request.urlopen(rig.base + "/api/live", timeout=5).read())
            self.assertEqual(len(live["parked"]), 1)
            p = live["parked"][0]
            self.assertEqual((p["agent"], p["input"], p["ticks"]), ("small", "chat", ["fast"]))
            self.assertIn("the ticked models hold at most 100", p["reason"])
            self.assertGreater(p["need"], 100)
            t.join(10)
            self.assertEqual(out["status"], 503)
            self.assertIn("parked_too_big", json.loads(out["body"])["error"]["message"])
            att = rig.attempts()[-1]
            self.assertEqual(att["state"], "failed")
            self.assertIn("parked_too_big", att["error"])
            dec = rig.decisions()[-1]
            self.assertEqual(dec["reason"], "parked:too_big")
            inputs = json.loads(dec["inputs"])
            self.assertEqual((inputs["contexts"], inputs["size"]["how"]), ({"fast": 100}, "estimate"))
            self.assertEqual(len(rig.a.requests), 0)                                    # nothing was sent to fail
            parking = json.loads(urllib.request.urlopen(rig.base + "/api/parking", timeout=5).read())
            self.assertEqual(parking["recent"][-1]["left"], "refused")
            ok, lines = verify(rig.path)
            self.assertTrue(ok, "\n".join(lines))
        finally:
            rig.stop()

    def test_released_when_a_bigger_model_is_ticked(self):
        rig = Rig(agents=AGENTS, fast_ctx=100, big_ctx=100, rules={"parking_wait_seconds": 10, "parking_recheck_seconds": 0.2})
        huge = FakeEngine("hugefake")
        try:
            out = {}
            t = threading.Thread(target=lambda: out.update(zip(("status", "ctype", "body"), call(rig.base, "/v1/chat/completions",
                {"model": "cockpit/small", "max_tokens": 10, "messages": [{"role": "user", "content": "w" * 1000}]}, timeout=30))))
            t.start()
            time.sleep(0.4)
            self.assertEqual(len(json.loads(urllib.request.urlopen(rig.base + "/api/live", timeout=5).read())["parked"]), 1)
            # a model that holds it is added and ticked for the agent: the parked request takes it
            status, _h, raw = ui_req(rig.base, "/api/models", {"name": "huge", "kind": "engine", "url": huge.url, "context": 100000})
            self.assertEqual(status, 200, raw)
            self.assertEqual(ui_req(rig.base, "/api/policy", {"agent": "small", "models": ["fast", "huge"]})[0], 200)
            t.join(15)
            self.assertEqual(out["status"], 200, out.get("body"))
            self.assertEqual(json.loads(out["body"])["choices"][0]["message"]["content"], "Hello world")
            self.assertEqual(len(huge.requests), 1)
            dec = json.loads(rig.decisions()[-1]["inputs"])
            self.assertEqual(dec["parked"]["released_to"], "huge")
            self.assertGreater(dec["parked"]["waited_s"], 0.3)
            self.assertEqual([d["reason"] for d in rig.decisions()][-2:][0], "parked:too_big")
            self.assertEqual(rig.attempts()[-1]["tier"], "huge")
            self.assertEqual(len(json.loads(urllib.request.urlopen(rig.base + "/api/live", timeout=5).read())["parked"]), 0)
            ok, lines = verify(rig.path)
            self.assertTrue(ok, "\n".join(lines))
        finally:
            huge.stop()
            rig.stop()

    def test_tokenizer_counts_and_a_pinned_small_model_parks(self):
        rig = Rig(agents=AGENTS, fast_ctx=400, big_ctx=100000, rules=RULES)
        try:
            msgs = [{"role": "user", "content": "w" * 1000}]
            status, _, _ = call(rig.base, "/v1/chat/completions", {"model": "cockpit/small", "max_tokens": 10, "messages": msgs})
            self.assertEqual(status, 200)                                               # the estimate: 281 + 10 fits 400
            self.assertEqual(json.loads(rig.decisions()[-1]["inputs"])["size"]["how"], "estimate")
            rig.a.tokenizer = True
            rig.srv.engines["fast"]._no_tokenize = False
            status, _, body = call(rig.base, "/v1/chat/completions", {"model": "cockpit/small", "max_tokens": 10, "messages": msgs})
            self.assertEqual(status, 503)                                               # the tokenizer counts 506 + 10: too big
            inputs = json.loads(rig.decisions()[-1]["inputs"])
            self.assertEqual(inputs["size"]["how"], "tokenizer")
            self.assertGreater(inputs["size"]["need"], 500)                           # 500 from the tokenizer, plus the messages and the 10
            # the wide agent's ticks hold it on big; naming fast pins the small one, which is parked, not trimmed
            status, _, _ = call(rig.base, "/v1/chat/completions", {"model": "cockpit/wide", "max_tokens": 10, "messages": msgs})
            self.assertEqual((status, rig.attempts()[-1]["tier"]), (200, "big"))
            status, _, body = call(rig.base, "/v1/chat/completions", {"model": "fast", "max_tokens": 10, "messages": msgs},
                                   )
            self.assertEqual(status, 503)
            self.assertIn("the pinned model fast holds 400", json.loads(body)["error"]["message"])
        finally:
            rig.stop()


if __name__ == "__main__":
    unittest.main()
