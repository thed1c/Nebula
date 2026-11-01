"""private_dir: mode 0700 on Linux; on Windows a plain folder (Python 3.12 turns 0o700 into an owner-only access list,
and a Nebula once run as administrator then owned, and locked the user out of, its own keys folder)."""
import os
import stat
import tempfile
import unittest
from unittest import mock

from nebula import privdir


class PrivateDirTest(unittest.TestCase):
    def test_windows_gets_no_mode(self):
        with mock.patch.object(privdir.os, "name", "nt"), mock.patch.object(privdir.os, "makedirs") as mk:
            privdir.private_dir("C:/x/keys")
        mk.assert_called_once_with("C:/x/keys", exist_ok=True)

    @unittest.skipIf(os.name == "nt", "POSIX modes")
    def test_posix_is_0700(self):
        d = os.path.join(tempfile.mkdtemp(), "keys")
        privdir.private_dir(d)
        self.assertEqual(stat.S_IMODE(os.stat(d).st_mode) & 0o077, 0)
        privdir.private_dir(d)                                              # again: no error

    def test_nothing_else_asks_windows_for_0o700(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        for dp, _dn, fn in os.walk(root):
            if "tests" in dp or "_vendor" in dp:
                continue
            for f in fn:
                if f.endswith(".py") and f != "privdir.py":
                    with open(os.path.join(dp, f), encoding="utf-8") as fh:
                        self.assertNotIn("mode=0o700", fh.read(), f)


if __name__ == "__main__":
    unittest.main()
