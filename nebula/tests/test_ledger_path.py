"""Where the ledger goes: a relative one beside the config (as the keys are), a command line's from where it ran.
An installer's icon starts Nebula from its program folder; its data must not land there (an uninstall removes it)."""
import json
import os
import tempfile
import unittest

from nebula.config import Config
from nebula.server import make_server
from nebula.tests import free_port


class LedgerPathTest(unittest.TestCase):
    def serve(self, args):
        srv = make_server(Config.from_args(args))
        try:
            return srv.cfg.ledger
        finally:
            srv.server_close(); srv.tools.close()

    def test_relative_ledger_is_beside_the_config(self):
        with tempfile.TemporaryDirectory() as conf, tempfile.TemporaryDirectory() as elsewhere:
            p = os.path.join(conf, "nebula.json")
            for ledger in ("nebula.sqlite3", "mine.sqlite3"):
                with open(p, "w") as f:
                    json.dump({"port": free_port(), "ledger": ledger, "tiers": []}, f)
                old = os.getcwd(); os.chdir(elsewhere)
                try:
                    got = self.serve(["--config", p])
                finally:
                    os.chdir(old)
                self.assertEqual(os.path.realpath(got), os.path.realpath(os.path.join(conf, ledger)))

    def test_a_command_lines_ledger_is_from_where_it_ran(self):
        with tempfile.TemporaryDirectory() as conf, tempfile.TemporaryDirectory() as here:
            p = os.path.join(conf, "nebula.json")
            with open(p, "w") as f:
                json.dump({"port": free_port(), "tiers": []}, f)
            old = os.getcwd(); os.chdir(here)
            try:
                got = self.serve(["--config", p, "--ledger", "run.sqlite3"])
            finally:
                os.chdir(old)
            self.assertEqual(os.path.realpath(got), os.path.realpath(os.path.join(here, "run.sqlite3")))


if __name__ == "__main__":
    unittest.main()
