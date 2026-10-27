"""The security log (nebula/seclog.py): append-only, a hash chain that shows tampering, repeats collapsed; refusals,
key use (by id, never the key), the page's writes that change who may do what, guests refused a model."""
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nebula import seclog
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call


class LogTest(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "l.sqlite3.security.sqlite3")
        self.log = seclog.SecLog(self.path)

    def tearDown(self):
        self.log.close()

    def test_append_only(self):
        self.log.add("auth.refused", "10.0.0.9", "v1")
        self.log.add("auth.refused", "10.0.0.9", "v1")                 # a repeat row too
        for sql in ("UPDATE events SET kind = 'x'", "DELETE FROM events", "DELETE FROM repeats"):
            with self.assertRaises(sqlite3.DatabaseError):
                self.log.db.execute(sql)
        self.assertEqual(self.log.read()[0]["kind"], "auth.refused")

    def test_the_chain_shows_a_change_made_by_hand(self):
        for i in range(3):
            self.log.add("key.made", "owner", f"api:{i}")
        self.assertEqual(self.log.verify(), {"ok": True, "rows": 3})
        db = sqlite3.connect(self.path)                   # someone with the file: drops the guard, edits a row
        db.execute("DROP TRIGGER events_no_update")
        db.execute("UPDATE events SET what = 'api:x' WHERE seq = 2")
        db.commit(); db.close()
        self.assertEqual(self.log.verify(), {"ok": False, "broken_at": 2, "rows": 1})

    def test_repeats_collapse_into_a_count(self):
        for _ in range(50):
            self.log.add("auth.refused", "10.0.0.9", "v1")
        self.log.add("auth.refused", "10.0.0.8", "v1")
        rows = self.log.read()
        self.assertEqual([(r["who"], r["count"]) for r in rows], [("10.0.0.8", 1), ("10.0.0.9", 50)])
        self.assertEqual(self.log.counts()["auth.refused"], 51)
        self.assertEqual(self.log.verify()["rows"], 2)

    def test_filters(self):
        self.log.add("gravity.join", "peer:a", "friend")
        self.log.add("key.used", "api", "api/1")
        self.log.add("gravity.refused", "node:b", "enroll")
        self.assertEqual([r["kind"] for r in self.log.read(kinds=["gravity.*"])], ["gravity.refused", "gravity.join"])
        self.assertEqual([r["kind"] for r in self.log.read(min_severity="warn")], ["gravity.refused", "gravity.join"])
        self.assertEqual([r["kind"] for r in self.log.read(since_seq=2)], ["gravity.refused"])

    def test_listeners_hear_each_event(self):
        heard = []
        self.log.listeners.append(heard.append)
        self.log.add("auth.refused", "x", "v1")
        self.log.add("auth.refused", "x", "v1")
        self.assertEqual([(h["kind"], h["fresh"]) for h in heard], [("auth.refused", True), ("auth.refused", False)])


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine("tiny")
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
            "tiers": [{"name": "tiny", "engine": self.engine.url, "shared": True}, {"name": "private", "engine": self.engine.url}],
            "callers": [{"id": "friend-app", "name": "A friend's app", "guest": True}],
            "api": {"require_key": True},
            "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close(); self.engine.stop()

    def events(self, **q):
        qs = "&".join(f"{k}={v}" for k, v in q.items())
        return call(self.base, "/api/security" + ("?" + qs if qs else ""))[1]["events"]

    def chat(self, key=None, model="tiny"):
        return call(self.base, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}]},
                    {"Authorization": f"Bearer {key}"} if key else None)[0]

    def test_refusals_key_use_and_writes_never_a_key(self):
        self.assertEqual(self.chat("nbk_wrong"), 401)
        self.assertEqual(self.chat(), 401)
        st, made = call(self.base, "/api/stars/api/keys", {"label": "phone"})
        key = made["key"]
        self.assertEqual(self.chat(key), 200)
        self.assertEqual(self.chat(key), 200)
        st, ck = call(self.base, "/api/yard/callers/friend-app/keys", {"label": "theirs"})
        self.assertEqual(st, 200, ck)
        self.assertNotEqual(self.chat(ck["key"], model="private"), 200)               # a guest: not a shared model
        call(self.base, f"/api/yard/callers/friend-app/keys/{ck['row']['id']}/revoke", {})
        ev = self.events(limit=50)
        kinds = [e["kind"] for e in ev]
        refused = next(e for e in ev if e["kind"] == "auth.refused")
        self.assertEqual((refused["who"], refused["count"]), ("127.0.0.1", 2))       # two refusals, one row
        self.assertIn("key.made", kinds)
        self.assertIn("key.revoked", kinds)
        used = [e for e in ev if e["kind"] == "key.used"]
        self.assertEqual(len([u for u in used if u["who"] == "api"]), 1)              # once per key per hour
        self.assertIn("guest.refused", kinds)
        g = next(e for e in ev if e["kind"] == "guest.refused")
        self.assertEqual((g["who"], g["what"]), ("caller:friend-app", "model:private"))
        raw = Path(seclog.path_for(self.srv.cfg.ledger)).read_bytes()
        for secret in (key, ck["key"], "nbk_wrong"):
            self.assertNotIn(secret.encode(), raw)                                   # never a key, right or wrong
        self.assertTrue(call(self.base, "/api/security/verify")[1]["ok"])

    def test_beside_the_ledger_and_kept_across_restarts(self):
        self.chat("nbk_wrong")
        p = seclog.path_for(self.srv.cfg.ledger)
        self.assertTrue(os.path.exists(p))
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        time.sleep(0.1)
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.assertEqual(self.events()[0]["kind"], "auth.refused")


if __name__ == "__main__":
    unittest.main()
