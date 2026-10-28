"""The Gravity tab's invite (nebula/ui/index.html): it says the friend needs "Use the public relay" ticked too when that
is how the two would meet away from home (BUGS #2), and docs/gravity.md no longer says public relays are never used."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class GravityPageTest(unittest.TestCase):
    def test_the_invite_names_the_relay_tick(self):
        html = (ROOT / "nebula" / "ui" / "index.html").read_text()
        self.assertIn("turn Gravity on and tick “Use the public relay”", html)              # the friend wizard's message
        self.assertIn('r.reach === "lan" ?', html)

    def test_the_docs_say_the_relay_is_a_tick(self):
        doc = (ROOT / "docs" / "gravity.md").read_text()
        self.assertNotIn("Public relays are never used", doc)
        self.assertIn("Use the public relay", doc)


if __name__ == "__main__":
    unittest.main()
