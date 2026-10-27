"""cockpit.router: the rules, on their own."""
import unittest

from nebula.config import Tier
from nebula.ledger import SessionState
from nebula.router import Rules, Signals, decide, tier_named

TIERS = [Tier("fast", "http://a", thinking=False), Tier("think", "http://a", thinking=True), Tier("big", "http://b")]


def state(**kw):
    return SessionState("s", **kw)


class DecideTest(unittest.TestCase):
    def test_floor_is_the_first_tier_then_sticky(self):
        d = decide(TIERS, Rules(), state(), Signals())
        self.assertEqual((d.tier.name, d.reason, d.thinking, d.crossing), ("fast", "floor", False, False))
        d = decide(TIERS, Rules(), state(current_tier="big"), Signals())
        self.assertEqual((d.tier.name, d.reason, d.crossing), ("big", "floor", False))

    def test_regenerations_rise_then_cross(self):
        self.assertEqual(decide(TIERS, Rules(), state(regenerations=1), Signals()).tier.name, "think")
        d = decide(TIERS, Rules(), state(regenerations=2), Signals())
        self.assertEqual((d.tier.name, d.crossing, d.reasons), ("big", True, ["regenerations:2"]))
        self.assertEqual(decide(TIERS, Rules(), state(current_tier="think", regenerations=1), Signals()).tier.name, "big")

    def test_tool_errors(self):
        self.assertEqual(decide(TIERS, Rules(), state(tool_error_streak=2), Signals()).tier.name, "think")
        self.assertEqual(decide(TIERS, Rules(), state(tool_error_streak=4), Signals()).tier.name, "big")
        self.assertEqual(decide(TIERS, Rules(), state(tool_error_streak=1), Signals()).tier.name, "fast")

    def test_stuck_claim(self):
        d = decide(TIERS, Rules(), state(last_reply_text="Sorry, I can't do that."), Signals())
        self.assertEqual((d.tier.name, d.reasons, d.inputs["stuck_phrase"]), ("think", ["model_claim:stuck"], "i can't"))

    def test_escalate_header(self):
        self.assertEqual(decide(TIERS, Rules(), state(), Signals(escalate="1")).tier.name, "think")
        self.assertEqual(decide(TIERS, Rules(), state(), Signals(escalate="big")).tier.name, "big")
        self.assertEqual(decide(TIERS, Rules(), state(current_tier="big"), Signals(escalate="1")).tier.name, "big")

    def test_named_tier_sets_it_either_way(self):
        self.assertEqual(tier_named("cockpit/big", TIERS), "big")
        self.assertEqual(tier_named("big", TIERS), "big")
        self.assertIsNone(tier_named("gpt-5", TIERS))
        d = decide(TIERS, Rules(), state(current_tier="big"), Signals(named_tier="fast"))
        self.assertEqual((d.tier.name, d.crossing, d.reasons), ("fast", True, ["named:fast"]))
        d = decide(TIERS, Rules(), state(regenerations=5), Signals(named_tier="fast"))
        self.assertEqual(d.tier.name, "fast")

    def test_pending_tool_blocks_a_crossing(self):
        d = decide(TIERS, Rules(), state(pending_tool=True), Signals(named_tier="big"))
        self.assertEqual((d.tier.name, d.blocked, d.crossing), ("fast", "pending_tool", False))
        self.assertIn("blocked:pending_tool", d.reasons)
        d = decide(TIERS, Rules(), state(pending_tool=True), Signals(escalate="1"))
        self.assertEqual((d.tier.name, d.blocked), ("think", None))

    def test_thinking_comes_from_the_tier_else_the_request(self):
        self.assertFalse(decide(TIERS, Rules(), state(), Signals(thinking_requested=True)).thinking)
        self.assertTrue(decide(TIERS, Rules(), state(current_tier="think"), Signals(thinking_requested=False)).thinking)
        self.assertEqual(decide(TIERS, Rules(), state(current_tier="big"), Signals(thinking_requested=False)).thinking, False)
        self.assertIsNone(decide(TIERS, Rules(), state(current_tier="big"), Signals()).thinking)

    def test_rules_from_dict(self):
        r = Rules.from_dict({"regenerations_to_cross": 3, "stuck_phrases": ["NO IDEA"], "unknown": 1})
        self.assertEqual((r.regenerations_to_cross, r.stuck_phrases, r.regenerations_to_rise), (3, ("no idea",), 1))


if __name__ == "__main__":
    unittest.main()
