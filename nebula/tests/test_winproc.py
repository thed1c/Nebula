"""nebula/winproc.py: on Windows every program Nebula starts is hidden (CREATE_NO_WINDOW), unless the caller chose."""
import subprocess
import sys
import unittest
from unittest import mock

from nebula import winproc


class WinProcTest(unittest.TestCase):
    def test_flags(self):
        self.assertEqual(winproc.flags(None), winproc.CREATE_NO_WINDOW)
        self.assertEqual(winproc.flags(0x200), 0x200 | winproc.CREATE_NO_WINDOW)       # a new process group: hidden too
        self.assertEqual(winproc.flags(0x10), 0x10)                                     # CREATE_NEW_CONSOLE: the caller's
        self.assertEqual(winproc.flags(0x08), 0x08)                                     # DETACHED_PROCESS: the caller's

    def test_nothing_changes_off_windows(self):
        if sys.platform == "win32":
            self.skipTest("Windows")
        before = subprocess.Popen.__init__
        self.assertFalse(winproc.hide_consoles())
        self.assertIs(subprocess.Popen.__init__, before)

    def test_on_windows_every_popen_gets_the_flag(self):
        seen = []

        class FakePopen:
            def __init__(self, *args, **kw):
                seen.append((args, kw.get("creationflags")))

        with mock.patch.object(winproc.os, "name", "nt"), mock.patch.object(winproc.subprocess, "Popen", FakePopen):
            self.assertTrue(winproc.hide_consoles())
            self.assertTrue(winproc.hide_consoles())                                    # once only
            FakePopen(["nvidia-smi"], capture_output=True)
            FakePopen(["wsl.exe"], creationflags=0x10)
            FakePopen(*([None] * 14))                                                   # by position: left alone
        self.assertEqual([f for _a, f in seen], [winproc.CREATE_NO_WINDOW, 0x10, None])


if __name__ == "__main__":
    unittest.main()
