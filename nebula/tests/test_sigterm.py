"""The window closing sends the server SIGTERM: it closes in order (team work marked stopped while the ledger is open,
the ledger closed), as Ctrl+C does, and exits 0 instead of dying on the signal; unlike Ctrl+C it leaves the engines it
started running (a restart must not stop a model on another machine). Warnings go to NAME.log."""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipIf(os.name == "nt", "SIGTERM is a POSIX signal")
class SigtermTest(unittest.TestCase):
    def test_sigterm_closes_in_order_and_exits_0(self):
        with tempfile.TemporaryDirectory() as d:
            port = free_port()
            cfg = Path(d, "nebula.json")
            cfg.write_text(json.dumps({"port": port, "ledger": str(Path(d, "l.sqlite3")), "monitor": False, "tiers": []}))
            p = subprocess.Popen([sys.executable, "-m", "nebula", "--config", str(cfg)], cwd=str(Path(__file__).resolve().parents[2]),
                                 env=dict(os.environ, HOME=d), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                for _ in range(100):
                    try:
                        socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                        break
                    except OSError:
                        time.sleep(0.1)
                p.send_signal(signal.SIGTERM)
                self.assertEqual(p.wait(30), 0)
                self.assertTrue(Path(d, "nebula.log").exists())
            finally:
                if p.poll() is None:
                    p.kill()
                p.stdout.close()


class KeepEnginesTest(unittest.TestCase):
    def test_a_close_for_a_restart_leaves_the_engines_running(self):
        from nebula.tests.test_escalation import Rig
        for keep, stopped in ((True, []), (False, ["yard"])):
            rig = Rig()
            calls = []
            rig.srv.yard.close = lambda: calls.append("yard")
            rig.srv.keep_engines = keep
            rig.stop()
            self.assertEqual(calls, stopped)


if __name__ == "__main__":
    unittest.main()
