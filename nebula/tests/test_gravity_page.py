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

    def test_a_picked_model_survives_the_html(self):
        """A NUL between tier and model in an <option> value came back as U+FFFD: "no tier named '<tier>\ufffd'"."""
        html = (ROOT / "nebula" / "ui" / "index.html").read_text()
        self.assertNotIn("\\u0000", html)
        self.assertIn("const routeOfVal", html)

    def test_gravity_is_only_its_wormholes_on_the_map(self):
        html = (ROOT / "nebula" / "ui" / "index.html").read_text()
        self.assertNotIn('ensureBody(gk, "worm")', html)                                    # no Gravity star of its own
        self.assertIn("syncModels(rows);", html)                                             # a friend's model you use stands with yours
        self.assertIn("if (adopted[m]) continue;", html)                                     # ... and not twice, at its own Nebula

    def test_an_away_peer_leaves_the_map(self):
        html = (ROOT / "nebula" / "ui" / "index.html").read_text()
        # by default an away peer leaves the map; Widgets › "Away Nebulas too" keeps it, dimmed; "Only my Nebula" shows none
        self.assertIn("state.gravityOn && !opt.mineOnly ? (state.peers || []).filter(p => p.reachable === true || opt.always) : []", html)
        self.assertIn(".hole.remote.away", html)
        self.assertIn("if (p.alias) return p.alias;", html)

    def test_the_docs_say_the_relay_is_a_tick(self):
        doc = (ROOT / "docs" / "gravity.md").read_text()
        self.assertNotIn("Public relays are never used", doc)
        self.assertIn("Use the public relay", doc)


if __name__ == "__main__":
    unittest.main()
