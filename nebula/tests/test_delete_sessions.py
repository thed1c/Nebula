"""Deleting conversations for good (nebula/ledger.py delete_sessions, POST /api/sessions/ID/delete and
/api/sessions/delete): every row of the conversation goes from every ledger table, the others stay whole and the ledger
still verifies; "closed": true deletes the closed ones, "all": true every one; one with a reply on the way is kept; the page asks first."""
import json
import sqlite3
import unittest
from pathlib import Path

from nebula.ledger import Ledger
from nebula.replay import verify
from nebula.tests.test_escalation import Rig
from nebula.tests.test_ui import req

PAGE = Path(__file__).resolve().parents[1] / "ui" / "index.html"


class DeleteSessionsTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()
        self.base = self.rig.base

    def tearDown(self):
        self.rig.stop()

    def chat(self, text, sid=None):
        status, headers, raw = req(self.base, "/api/chat", {"text": text, **({"session_id": sid} if sid else {})})
        self.assertEqual(status, 200, raw)
        return headers.get("X-Cockpit-Session-Id")

    def rows(self, sid):
        db = sqlite3.connect(self.rig.path)
        try:
            return {t: db.execute(f"SELECT count(*) FROM {t} WHERE session_id=?", (sid,)).fetchone()[0]
                    for t in Ledger.SESSION_TABLES}
        finally:
            db.close()

    def listed(self):
        return [s["session_id"] for s in json.loads(req(self.base, "/api/sessions")[2])]

    def test_one_conversation_is_deleted_for_good(self):
        a, b = self.chat("first"), self.chat("second")
        self.chat("more", a)
        self.assertTrue(all(n for t, n in self.rows(a).items() if t in ("sessions", "events", "attempts")))
        status, _h, raw = req(self.base, f"/api/sessions/{a}/delete", {})
        self.assertEqual((status, json.loads(raw)["deleted"]), (200, [a]))
        self.assertEqual(self.listed(), [b])
        self.assertEqual(req(self.base, f"/api/sessions/{a}")[0], 404)
        self.assertEqual(set(self.rows(a).values()), {0})                  # nothing of it in any table
        self.assertEqual(self.rows(b)["events"] > 0, True)                  # the other one whole
        ok, lines = verify(self.rig.path)
        self.assertTrue(ok, "\n".join(lines))
        self.assertEqual(req(self.base, f"/api/sessions/{a}/delete", {})[0], 404)

    def test_the_closed_ones(self):
        a, b, c = self.chat("one"), self.chat("two"), self.chat("three")
        for sid in (a, c):
            req(self.base, f"/api/sessions/{sid}/close", {})
        status, _h, raw = req(self.base, "/api/sessions/delete", {"closed": True})
        self.assertEqual((status, sorted(json.loads(raw)["deleted"])), (200, sorted([a, c])))
        self.assertEqual(self.listed(), [b])

    def test_all_of_them(self):
        a, b, c = self.chat("one"), self.chat("two"), self.chat("three")
        req(self.base, f"/api/sessions/{b}/close", {})
        status, _h, raw = req(self.base, "/api/sessions/delete", {"all": True})
        self.assertEqual((status, sorted(json.loads(raw)["deleted"])), (200, sorted([a, b, c])))
        self.assertEqual(self.listed(), [])
        ok, lines = verify(self.rig.path)
        self.assertTrue(ok, "\n".join(lines))

    def test_a_list_of_ids(self):
        a, b = self.chat("one"), self.chat("two")
        out = json.loads(req(self.base, "/api/sessions/delete", {"ids": [a, "ses_nope", a]})[2])
        self.assertEqual((out["deleted"], out["missing"], out["busy"]), ([a], ["ses_nope"], []))
        self.assertEqual(self.listed(), [b])
        self.assertEqual(req(self.base, "/api/sessions/delete", {"ids": "all"})[0], 400)
        self.assertEqual(req(self.base, "/api/sessions/delete", {"all": "yes"})[0], 400)
        self.assertEqual(req(self.base, "/api/sessions/delete", {})[0], 400)

    def test_one_with_a_reply_on_the_way_is_kept(self):
        a = self.chat("one")
        db = sqlite3.connect(self.rig.path)
        db.execute("UPDATE attempts SET state='running' WHERE session_id=?", (a,))
        db.commit()
        db.close()
        status, _h, raw = req(self.base, f"/api/sessions/{a}/delete", {})
        self.assertEqual(status, 409)
        self.assertIn("reply on the way", json.loads(raw)["error"]["message"])
        out = json.loads(req(self.base, "/api/sessions/delete", {"ids": [a]})[2])
        self.assertEqual((out["deleted"], out["busy"]), ([], [a]))
        self.assertEqual(self.listed(), [a])

    def test_a_call_left_running_long_ago_does_not_block(self):
        """A crash or a restart can leave an attempt "running" for ever: older than STUCK_AFTER_S it is not in flight."""
        a = self.chat("one")
        db = sqlite3.connect(self.rig.path)
        db.execute("UPDATE attempts SET state='running', created_at='2026-01-01T00:00:00+00:00' WHERE session_id=?", (a,))
        db.commit()
        db.close()
        status, _h, raw = req(self.base, f"/api/sessions/{a}/delete", {})
        self.assertEqual((status, json.loads(raw)["deleted"]), (200, [a]))
        self.assertEqual(set(self.rows(a).values()), {0})

    def test_the_page_asks_first(self):
        html = PAGE.read_text()
        for want in ('class="sdel" data-del=', 'data-delclosed="1"', 'data-delall="1"', 'id="del"', 'if (!confirm(question)) return;',
                     '"/api/sessions/delete"', "state.tabs = state.tabs.filter(t => !gone.has(t.sid)); saveTabs();"):
            self.assertIn(want, html)


if __name__ == "__main__":
    unittest.main()
