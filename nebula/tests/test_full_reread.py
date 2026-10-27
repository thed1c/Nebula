"""Compaction's periodic full read (nebula/handoff.py full_read_due, transcript_messages(full=True)): every Nth briefing of a
session reads the whole conversation instead of the last briefing + what came after, so rolling briefings do not drift."""
import unittest

from nebula.config import Config
from nebula.handoff import full_read_due, transcript_messages


class FakeLedger:
    def __init__(self):
        self.msgs = [(1, {"role": "user", "content": "build it with Python 3.9"}), (2, {"role": "assistant", "content": "ok"}),
                     (3, {"role": "user", "content": "now add tests"})]
        self.brief = (2, {"objective": "build it", "constraints": ["Python 3.9"], "tried": [], "open": [], "pending_tool_effects": [],
                          "covers_through_seq": 2})

    def latest_briefing(self, sid):
        return self.brief

    def transcript(self, sid, since_seq=0, until_seq=None):
        return [(s, p) for s, p in self.msgs if s > since_seq and (until_seq is None or s < until_seq)]


class FullReread(unittest.TestCase):
    def test_due(self):
        self.assertEqual([n for n in range(12) if full_read_due(n, 5)], [4, 9])     # the 5th, 10th, ... briefing
        self.assertFalse(any(full_read_due(n, 0) for n in range(12)))              # 0: never
        self.assertFalse(full_read_due(0, 1))                                      # the first briefing reads it all anyway
        self.assertEqual(Config().compaction_full_every, 5)
        self.assertEqual(Config.from_dict({"compaction_full_every": 0}).compaction_full_every, 0)

    def test_transcript(self):
        L = FakeLedger()
        rolling = transcript_messages(L, "s")
        self.assertTrue(rolling[0]["role"] == "system" and rolling[0]["content"].startswith("Summary of the conversation so far:"))
        self.assertEqual([m["content"] for m in rolling[1:]], ["now add tests"])
        full = transcript_messages(L, "s", full=True)
        self.assertEqual([m["content"] for m in full], ["build it with Python 3.9", "ok", "now add tests"])


if __name__ == "__main__":
    unittest.main()
