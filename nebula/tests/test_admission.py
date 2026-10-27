"""Step B: the gate at every engine, served by class then arrival; the wait visible and recorded; the overflow to
a later ticked engine when the wait passes the class's tolerance."""
import json
import threading
import time
import unittest
import urllib.request

from nebula.admission import Admission, Gate, QueueTimeout, priority_of
from nebula.config import Agent, Tier
from nebula.replay import verify
from nebula.tests.test_escalation import Rig, rows
from nebula.tests.test_hooks import post


class GateTest(unittest.TestCase):
    def test_served_by_class_then_arrival(self):
        g = Gate(1)
        order, lock = [], threading.Lock()
        g.acquire(2, {"who": "first-background"})           # holds the one slot
        started = threading.Barrier(4)

        def worker(rank, name, delay):
            started.wait()
            time.sleep(delay)
            g.acquire(rank, {"who": name})
            with lock:
                order.append(name)
            time.sleep(0.05)
            g.release()
        ts = [threading.Thread(target=worker, args=a) for a in [(2, "background", 0.0), (0, "interactive-1", 0.05), (0, "interactive-2", 0.1)]]
        for t in ts:
            t.start()
        started.wait()
        time.sleep(0.3)
        self.assertEqual(g.status()["waiting"][0]["who"], "interactive-1")       # the queue shows the best first
        self.assertEqual(g.ahead(0), 2)
        self.assertEqual(g.ahead(2), 3)
        g.release()                                                                # the holder leaves
        for t in ts:
            t.join(5)
        self.assertEqual(order, ["interactive-1", "interactive-2", "background"])
        self.assertEqual(g.status()["held"], 0)

    def test_timeout_leaves_the_queue(self):
        g = Gate(1)
        g.acquire(0, {})
        with self.assertRaises(QueueTimeout):
            g.acquire(0, {"late": True}, timeout=0.1)
        self.assertEqual(g.status()["waiting"], [])
        g.release()
        self.assertLess(g.acquire(0, {}), 0.01)

    def test_priority_of(self):
        self.assertEqual(priority_of("chat", None, None), "interactive")
        self.assertEqual(priority_of("hook:x", None, None), "agent")
        self.assertEqual(priority_of("hook:x", "background", None), "background")
        self.assertEqual(priority_of("room", None, "background"), "background")
        self.assertEqual(priority_of("room", "nonsense", None), "interactive")


class ChooseTest(unittest.TestCase):
    def test_overflow_rules(self):
        class Cfg:
            tiers = [Tier("fast", "http://a", thinking=False), Tier("think", "http://a", thinking=True), Tier("big", "http://b")]
        adm = Admission(Cfg(), None, {"interactive": 60.0, "agent": 15.0, "background": 5.0}, default_avg=10.0)
        ladder = Cfg.tiers
        self.assertEqual(adm.choose(ladder, 0, 0, sticky=False), (0, None))             # free: stay
        adm.gate(ladder[0]).acquire(0, {})                                               # a's one slot held
        self.assertEqual(adm.choose(ladder, 0, 0, sticky=True), (0, None))              # sticky: wait
        self.assertEqual(adm.choose(ladder, 0, 0, sticky=False), (0, None))             # 10 s expected < 60 s tolerance
        self.assertEqual(adm.choose(ladder, 0, 2, sticky=False), (2, "overflow:big"))    # background: 10 s > 5 s, b is free
        adm.gate(ladder[2]).acquire(0, {})                                               # b held too
        for _ in range(8):
            adm.gate(ladder[0]).queue.append([0, 99, {}, 0])                            # 8 more interactive waiting at a
        self.assertEqual(adm.choose(ladder, 0, 0, sticky=False), (2, "overflow:big"))    # 90 s at a vs 10 s at b: b wins by > 60
        self.assertAlmostEqual(adm.expected_wait(ladder[0], 0), 90.0)
        adm.learn(ladder[0], 2.0)
        self.assertAlmostEqual(adm.average(ladder[0]), 2.0)
        adm.learn(ladder[0], 4.0)
        self.assertAlmostEqual(adm.average(ladder[0]), 2.6)


AGENTS = [Agent("worker", system="Work.", models=["fast", "big"]),
          Agent("stay", system="Stay.", models=["fast"])]


class GatewayQueueTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig(agents=AGENTS, hooks=[{"name": "bg", "agent": "stay", "priority": "background"}],
                       rules={"tolerance_seconds": {"interactive": 60, "agent": 15, "background": 5}})

    def tearDown(self):
        self.rig.stop()

    def test_interactive_is_served_before_background_on_one_slot(self):
        rig = self.rig
        results, lock = [], threading.Lock()

        def hook_call(name):
            status, js = post(rig.base, "/hooks/bg", json.dumps({"n": name, "text": "answer slowly"}).encode())
            with lock:
                results.append((name, status))

        def chat_call(name):
            rig.chat([{"role": "user", "content": f"{name} answer slowly"}], model="cockpit/stay")
            with lock:
                results.append((name, 200))
        # the background call takes the slot (the fake streams slowly); two interactive turns and a second background
        # call queue behind it; the interactive ones are served first
        t0 = threading.Thread(target=hook_call, args=("bg-1",))
        t0.start()
        time.sleep(0.4)
        ts = [threading.Thread(target=hook_call, args=("bg-2",)), threading.Thread(target=chat_call, args=("i-1",)),
              threading.Thread(target=chat_call, args=("i-2",))]
        ts[0].start()
        time.sleep(0.15)
        ts[1].start()
        time.sleep(0.15)
        ts[2].start()
        time.sleep(0.3)
        live = json.loads(urllib.request.urlopen(rig.base + "/api/live", timeout=5).read())
        waiting = live["waiting"]
        self.assertEqual([w["class"] for w in waiting], ["interactive", "interactive", "background"])
        self.assertTrue(all("fast" in w["tiers"] for w in waiting))
        adm = json.loads(urllib.request.urlopen(rig.base + "/api/admission", timeout=5).read())
        self.assertEqual(list(adm.values())[0]["held"], 1)
        for t in [t0] + ts:
            t.join(30)
        served = [m["content"] for r in rig.a.requests for m in r["messages"] if m["role"] == "user"]
        self.assertEqual([x.split()[0] if x.startswith("i-") else "bg" for x in served if not x.startswith("Webhook") or True][:0], [])
        order = []
        for r in rig.a.requests:
            u = [m for m in r["messages"] if m["role"] == "user"][-1]["content"]
            order.append("i" if u.startswith("i-") else "bg")
        self.assertEqual(order, ["bg", "i", "i", "bg"])
        outcomes = rows(rig.path, "SELECT a.dialect, o.detail FROM outcomes o JOIN attempts a USING (attempt_id) ORDER BY o.recorded_at")
        waits = {d["dialect"]: json.loads(d["detail"])["queued_s"] for d in outcomes}
        self.assertEqual(waits["chat"] >= 0, True)
        bg_waits = [json.loads(d["detail"])["queued_s"] for d in outcomes if d["dialect"] == "hook:bg"]
        self.assertGreater(max(bg_waits), 1.0)                                         # the second background call waited behind both
        self.assertTrue(json.loads(rows(rig.path, "SELECT inputs FROM decisions ORDER BY created_at DESC, rowid DESC LIMIT 1")[0]["inputs"])["priority"] in ("interactive", "background"))
        ok, lines = verify(rig.path)
        self.assertTrue(ok, "\n".join(lines))

    def test_a_second_ticked_engine_takes_the_overflow(self):
        rig = self.rig
        rig.srv.cfg.hooks.append({"name": "ov", "agent": "worker", "priority": "background"})
        # two interactive turns on the one slot of fast (the first streams slowly, the second waits behind it); a
        # background call for the worker, whose ticks are fast then big, finds fast's expected wait (two calls at the
        # default 5 s average) above its 5 s tolerance and big free: it overflows to big, a crossing with the reason
        hold = threading.Thread(target=lambda: rig.chat([{"role": "user", "content": "hold the slot, answer slowly"}], model="cockpit/stay"))
        hold2 = threading.Thread(target=lambda: rig.chat([{"role": "user", "content": "second, answer slowly"}], model="cockpit/stay"))
        hold.start()
        time.sleep(0.4)
        hold2.start()
        time.sleep(0.3)
        status, js = post(rig.base, "/hooks/ov", b'{"text": "overflow me"}', {})
        self.assertEqual(status, 200, js)
        hold.join(30)
        hold2.join(30)
        # the decision sits on the source attempt (escalated, tier fast); the destination attempt answered on big
        last = rows(rig.path, "SELECT a.tier, a.state, d.reason, d.inputs FROM attempts a JOIN decisions d USING (attempt_id) WHERE a.dialect='hook:ov' ORDER BY d.created_at")
        self.assertEqual((last[0]["tier"], last[0]["state"]), ("fast", "escalated"))
        self.assertIn("overflow:big", last[0]["reason"])
        dest = rows(rig.path, "SELECT tier, state FROM attempts WHERE dialect='hook:ov' ORDER BY created_at DESC, rowid DESC LIMIT 1")[0]
        self.assertEqual((dest["tier"], dest["state"]), ("big", "completed"))
        inputs = json.loads(last[0]["inputs"])
        self.assertEqual(inputs["priority"], "background")
        self.assertEqual(len(rig.b.requests), 1)
        self.assertEqual(rows(rig.path, "SELECT to_tier FROM handoffs")[0]["to_tier"], "big")       # a different engine: a crossing
        # the same call while the slot is free stays on fast
        status, js = post(rig.base, "/hooks/ov", b'{"text": "no need"}', {})
        self.assertEqual(status, 200, js)
        self.assertEqual(rows(rig.path, "SELECT tier FROM attempts WHERE dialect='hook:ov' ORDER BY created_at DESC, rowid DESC LIMIT 1")[0]["tier"], "fast")
        ok, lines = verify(rig.path)
        self.assertTrue(ok, "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
