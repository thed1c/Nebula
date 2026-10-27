"""Nebula's public names (nebula/names.py): "nebula/<name>" and X-Nebula-* are written, and the older "cockpit/<name>",
X-Cockpit-* and cockpit.* files are still read, so an install from before the rename keeps working."""
import os
import tempfile
import unittest

from nebula import names
from nebula.tests.test_escalation import Rig, rows


class NamesTest(unittest.TestCase):
    def test_model_names(self):
        self.assertEqual((names.model("big"), names.model(None)), ("nebula/big", "nebula"))
        self.assertEqual([names.bare_model(m) for m in ("nebula/big", "cockpit/big", "big", "other/big")],
                         ["big", "big", "big", "other/big"])

    def test_headers_new_first_then_old(self):
        self.assertEqual(names.header({"X-Cockpit-Agent": "old"}, "Agent"), "old")
        self.assertEqual(names.header({"X-Nebula-Agent": "new", "X-Cockpit-Agent": "old"}, "Agent"), "new")
        self.assertIsNone(names.header(None, "Agent"))
        h = {"X-Cockpit-Session": "s"}
        names.set_header(h, "Session", "t")
        self.assertEqual(h, {"X-Nebula-Session": "t"})
        names.set_header(h, "Session", None)
        self.assertEqual(h, {})

    def test_an_older_file_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(names.local_path(d, "nebula-keys", "cockpit-keys"), os.path.join(d, "nebula-keys"))
            os.mkdir(os.path.join(d, "cockpit-keys"))
            self.assertEqual(names.local_path(d, "nebula-keys", "cockpit-keys"), os.path.join(d, "cockpit-keys"))
            os.mkdir(os.path.join(d, "nebula-keys"))
            self.assertEqual(names.local_path(d, "nebula-keys", "cockpit-keys"), os.path.join(d, "nebula-keys"))


class OldClientsTest(unittest.TestCase):
    """A client still sending the older names reaches the same tier and agent."""
    def setUp(self):
        self.rig = Rig()

    def tearDown(self):
        self.rig.stop()

    def test_old_and_new_model_names_and_headers(self):
        self.rig.chat([{"role": "user", "content": "new"}], model="nebula/big")
        self.assertEqual(self.rig.decisions()[-1]["tier"], "big")
        self.rig.chat([{"role": "user", "content": "old"}], model="cockpit/big")
        self.assertEqual(self.rig.decisions()[-1]["tier"], "big")
        self.rig.chat([{"role": "user", "content": "x"}], headers={"X-Nebula-Session": "k-new"})
        self.rig.chat([{"role": "user", "content": "y"}], headers={"X-Cockpit-Session": "k-old"})
        got = {r["session_key"]: r["key_source"] for r in rows(self.rig.path, "SELECT * FROM sessions")}
        self.assertEqual((got.get("k-new"), got.get("k-old")), ("header", "header"), got)


if __name__ == "__main__":
    unittest.main()
