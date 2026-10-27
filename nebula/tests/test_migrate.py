"""What Nebula keeps on disk carries its version (nebula/migrate.py): a new file starts at the newest; a file from
before versions (a ledger without the agent column) is copied aside and brought up; a step that fails changes
nothing; a file from a newer Nebula is refused and left as it was; the config is versioned the same way."""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nebula import ledger as L
from nebula import migrate
from nebula.config import Config
from nebula.rag.catalog import Catalog
from nebula.seclog import SecLog


def version(path):
    c = sqlite3.connect(path)
    try:
        return c.execute("PRAGMA user_version").fetchone()[0]
    finally:
        c.close()


class SqliteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def p(self, name):
        return os.path.join(self.tmp, name)

    def test_new_files_start_at_the_newest_without_a_backup(self):
        led = L.Ledger(self.p("l.sqlite3")); led.close()
        s = SecLog(self.p("s.sqlite3")); s.add("auth.refused", "x", "y"); s.close()
        c = Catalog(self.p("c.sqlite3")); c.close()
        self.assertEqual(version(self.p("l.sqlite3")), len(L.STEPS))
        self.assertEqual(version(self.p("s.sqlite3")), 1)
        self.assertEqual(version(self.p("c.sqlite3")), 1)
        self.assertEqual([f for f in os.listdir(self.tmp) if ".bak" in f], [])
        # opened again: nothing to do
        led = L.Ledger(self.p("l.sqlite3"))
        self.assertEqual(led.upgraded, {"from": len(L.STEPS), "to": len(L.STEPS), "backup": None})
        led.close()

    def test_an_old_ledger_is_copied_aside_and_brought_up(self):
        path = self.p("old.sqlite3")
        old = L.SCHEMA.replace(", agent TEXT);", ");")
        self.assertNotEqual(old, L.SCHEMA)
        c = sqlite3.connect(path)
        c.executescript(old)
        c.execute("INSERT INTO tasks VALUES ('t1','s1','local',NULL,'2026-01-01')")
        c.commit(); c.close()
        led = L.Ledger(path)
        self.assertEqual(led.upgraded["from"], 0)
        self.assertTrue(os.path.exists(led.upgraded["backup"]))
        self.assertIn("agent", {r["name"] for r in led.db.execute("PRAGMA table_info(attempts)")})
        self.assertEqual(led.db.execute("SELECT count(*) FROM tasks").fetchone()[0], 1)
        led.close()
        b = sqlite3.connect(self.p("old.sqlite3.v0.bak"))
        self.assertNotIn("agent", {r[1] for r in b.execute("PRAGMA table_info(attempts)")})
        self.assertEqual(b.execute("SELECT count(*) FROM tasks").fetchone()[0], 1)
        b.close()

    def test_a_failing_step_changes_nothing(self):
        path = self.p("f.sqlite3")
        c = sqlite3.connect(path, isolation_level=None)
        migrate.sqlite_upgrade(c, "t", ["CREATE TABLE a (x INTEGER);"], path)

        def bad(db):
            db.execute("ALTER TABLE a ADD COLUMN y TEXT")
            raise RuntimeError("half way")
        with self.assertRaises(RuntimeError):
            migrate.sqlite_upgrade(c, "t", ["CREATE TABLE a (x INTEGER);", bad], path)
        self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], 1)
        self.assertEqual([r[1] for r in c.execute("PRAGMA table_info(a)")], ["x"])
        c.close()

    def test_a_newer_file_is_refused_and_left_alone(self):
        path = self.p("new.sqlite3")
        led = L.Ledger(path); led.close()
        c = sqlite3.connect(path); c.execute(f"PRAGMA user_version = {len(L.STEPS) + 1}"); c.commit(); c.close()
        with self.assertRaises(migrate.SchemaError) as e:
            L.Ledger(path)
        self.assertIn("newer Nebula", str(e.exception))
        self.assertEqual(version(path), len(L.STEPS) + 1)
        self.assertEqual([f for f in os.listdir(self.tmp) if ".bak" in f], [])

    def test_append_only_triggers_survive_the_step_runner(self):
        s = SecLog(self.p("s.sqlite3"))
        s.add("auth.refused", "x", "y")
        with self.assertRaises(sqlite3.DatabaseError):
            s.db.execute("DELETE FROM events")
        s.close()


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")

    def test_saved_with_its_version_and_read_back(self):
        Path(self.path).write_text(json.dumps({"port": 0, "tiers": [], "_note": "kept"}))
        cfg = Config.from_args(["--config", self.path])
        cfg.save()
        d = json.loads(Path(self.path).read_text())
        self.assertEqual((d["config_version"], d["_note"]), (migrate.CONFIG_VERSION, "kept"))

    def test_a_newer_config_is_refused_and_left_alone(self):
        text = json.dumps({"config_version": migrate.CONFIG_VERSION + 1, "tiers": []})
        Path(self.path).write_text(text)
        with self.assertRaises(migrate.SchemaError) as e:
            Config.from_args(["--config", self.path])
        self.assertIn("newer Nebula", str(e.exception))
        self.assertEqual(Path(self.path).read_text(), text)

    def test_an_older_config_is_copied_aside_and_brought_up(self):
        Path(self.path).write_text(json.dumps({"tiers": [], "old_name": 5}))

        def step(d):
            d["new_name"] = d.pop("old_name")
            return d
        with mock.patch.object(migrate, "CONFIG_VERSION", 2), mock.patch.dict(migrate.CONFIG_STEPS, {1: step}):
            cfg = Config.from_args(["--config", self.path])
        self.assertEqual(cfg.raw["new_name"], 5)
        d = json.loads(Path(self.path).read_text())
        self.assertEqual((d["config_version"], d["new_name"]), (2, 5))
        self.assertEqual(json.loads(Path(self.path + ".v1.bak").read_text()), {"tiers": [], "old_name": 5})


if __name__ == "__main__":
    unittest.main()
