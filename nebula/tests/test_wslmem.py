"""WSL's share of the PC's RAM (nebula/wslmem.py): the plan, the change (a backup first, only the memory line), undo,
and the restart's second yes; a fake Windows profile folder, never a real .wslconfig."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from nebula import engines as E
from nebula import wslmem as W


class WslMemTest(unittest.TestCase):
    def setUp(self):
        self.prof = tempfile.mkdtemp()
        os.environ.update({"NEBULA_FAKE_WINPROFILE": self.prof, "NEBULA_FAKE_WIN_RAM_MB": str(32 * 1024)})

    def tearDown(self):
        for k in ("NEBULA_FAKE_WINPROFILE", "NEBULA_FAKE_WIN_RAM_MB"):
            os.environ.pop(k, None)

    def test_plan_leaves_windows_8_gb_and_at_least_half(self):
        p = W.plan()
        self.assertEqual((p["memory_new"], p["memory_now"], p["exists"], p["windows_total_gb"]), ("24GB", None, False, 32))
        os.environ["NEBULA_FAKE_WIN_RAM_MB"] = str(12 * 1024)                       # a small PC: half, not 4 GB
        self.assertEqual(W.plan()["memory_new"], "6GB")

    def test_only_the_memory_line_changes_and_undo_puts_it_back(self):
        cfg = Path(self.prof, ".wslconfig")
        before = "# mine\n[wsl2]\nprocessors=8\nmemory=16GB\nswap=0\n\n[experimental]\nautoMemoryReclaim=gradual\n"
        cfg.write_text(before)
        a = W.apply()
        after = cfg.read_text()
        self.assertEqual(after, before.replace("memory=16GB", "memory=24GB"))       # the rest exactly as it was
        self.assertEqual(Path(a["backup"]).read_text(), before)
        self.assertEqual(W.plan()["memory_now"], "24GB")
        u = W.undo(a["backup"], a["had_file"])
        self.assertEqual((cfg.read_text(), u["memory_now"]), (before, "16GB"))
        self.assertFalse(Path(a["backup"]).exists())

    def test_a_missing_key_or_file_is_added_and_undo_removes_the_new_file(self):
        cfg = Path(self.prof, ".wslconfig")
        cfg.write_text("[wsl2]\nprocessors=4\n")
        W.apply()
        self.assertEqual(cfg.read_text(), "[wsl2]\nmemory=24GB\nprocessors=4\n")
        cfg.write_text("[boot]\nsystemd=true\n")
        W.apply()
        self.assertEqual(cfg.read_text(), "[boot]\nsystemd=true\n\n[wsl2]\nmemory=24GB\n")
        cfg.unlink()
        a = W.apply()                                                              # no file before
        self.assertEqual(cfg.read_text(), "[wsl2]\nmemory=24GB\n")
        W.undo(a["backup"], a["had_file"])
        self.assertFalse(cfg.exists())

    def test_restart_needs_its_own_yes(self):
        with self.assertRaises(W.WslMemError):
            W.restart(False)
        self.assertTrue(W.restart(True)["fake"])                                   # (fake: no wsl --shutdown in tests)

    def test_undo_takes_only_its_own_backups(self):
        Path(self.prof, "other.txt").write_text("x")
        with self.assertRaises(W.WslMemError):
            W.undo(os.path.join(self.prof, "other.txt"), True)

    def test_the_fit_refusal_under_wsl_offers_the_fix(self):
        from nebula.tests.test_engines import make_flint_gguf
        m = os.path.join(self.prof, "f.gguf")
        make_flint_gguf(m)
        os.environ.update({"NEBULA_FAKE_RAM_FREE_MB": "1000", "NEBULA_FAKE_WSL": "1"})
        try:
            f = E.fit({"name": "f", "kind": "impulse", "model": m, "context": 8192}, None, [{"uuid": "a", "memory_mb": 8192, "used_mb": 0}])
        finally:
            for k in ("NEBULA_FAKE_RAM_FREE_MB", "NEBULA_FAKE_WSL"):
                os.environ.pop(k, None)
        self.assertEqual((f["fits"], f.get("wsl_fix")), (False, True))

    def test_the_routes(self):
        from nebula.tests.test_wizards import serve, stop
        from nebula.tests.test_callers import call
        srv, base, _ = serve({})
        try:
            st, p = call(base, "/api/wsl/memory")
            self.assertEqual((st, p["memory_new"]), (200, "24GB"))
            self.assertFalse(Path(self.prof, ".wslconfig").exists())                # a look changes nothing
            st, a = call(base, "/api/wsl/memory/apply", {})
            self.assertEqual(st, 200, a)
            self.assertEqual(call(base, "/api/wsl/memory/restart", {})[0], 409)       # no confirm, no restart
            st, u = call(base, "/api/wsl/memory/undo", {"backup": a["backup"], "had_file": a["had_file"]})
            self.assertEqual((st, Path(self.prof, ".wslconfig").exists()), (200, False))
        finally:
            stop(srv)


if __name__ == "__main__":
    unittest.main()
