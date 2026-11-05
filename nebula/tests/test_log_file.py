"""Warnings and errors outlive the window: NAME.log beside the config. And the System tab's error log hears Nebula's own
loggers (nebula.*: Halo, Starweave, the toolbox), not only the cockpit's, with a job that keeps failing as one line."""
import logging
import os
import tempfile
import unittest
from types import SimpleNamespace

from nebula.__main__ import add_log_file
from nebula.errors import ErrorLog


class LogFileTest(unittest.TestCase):
    def test_warnings_go_to_the_log_beside_the_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = SimpleNamespace(path=os.path.join(d, "nebula.json"))
            root = logging.getLogger()
            before = list(root.handlers)
            try:
                path = add_log_file(cfg)
                self.assertEqual(path, os.path.join(d, "nebula.log"))
                self.assertEqual(add_log_file(cfg), path)                     # once, however often it is asked
                self.assertEqual(len(root.handlers), len(before) + 1)
                logging.getLogger("nebula.teamwork").warning("a work item failed: boom")
                logging.getLogger("nebula.teamwork").info("not this")
                for h in root.handlers:
                    h.flush()
                text = open(path, encoding="utf-8").read()
                self.assertIn("WARNING nebula.teamwork: a work item failed: boom", text)
                self.assertNotIn("not this", text)
            finally:
                for h in list(root.handlers):
                    if h not in before:
                        root.removeHandler(h)
                        h.close()

    def test_the_error_log_hears_nebulas_own_loggers_and_folds_repeats(self):
        log = ErrorLog().attach()
        try:
            for name in ("nebula.halo", "nebula.teamwork", "nebula.teamtasks", "nebula.toolbox", "cockpit"):
                logging.getLogger(name).warning("from %s", name)
            for _ in range(5):
                logging.getLogger("nebula.halo").warning("Halo could not look: no model")
            items = log.list()["errors"]
            self.assertEqual([i["source"] for i in items][:5],
                             ["nebula.halo", "nebula.teamwork", "nebula.teamtasks", "nebula.toolbox", "cockpit"])
            folded = [i for i in items if i["message"] == "Halo could not look: no model"]
            self.assertEqual(len(folded), 1)
            self.assertEqual(folded[0]["repeats"], 5)
            # an error with the same words is its own entry, with its traceback; 40 other lines between do not split
            # a repeat inside the window
            try:
                raise RuntimeError("boom")
            except RuntimeError:
                logging.getLogger("nebula.halo").error("Halo could not look: no model", exc_info=True)
            for n in range(40):
                logging.getLogger("nebula.toolbox").warning("other %d", n)
            logging.getLogger("nebula.halo").warning("Halo could not look: no model")
            same = [i for i in log.list()["errors"] if i["message"] == "Halo could not look: no model"]
            self.assertEqual([(i["level"], i.get("repeats", 1)) for i in same], [("warning", 6), ("error", 1)])
            self.assertIn("RuntimeError: boom", same[1]["detail"])
        finally:
            log.detach()


if __name__ == "__main__":
    unittest.main()
