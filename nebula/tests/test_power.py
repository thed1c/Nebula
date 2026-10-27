"""Machines that sleep (nebula/power.py): an interactive call wakes one and waits for it; an agent or background call
never does; the wake itself (Wake-on-LAN packet, or a command); the hosts API."""
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path

from nebula import power as PW
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.test_callers import call
from nebula.tests.test_engines import ROOT, free_port


class PowerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.port = free_port()
        self.model = os.path.join(self.tmp, "m.gguf")
        Path(self.model).write_bytes(b"GGUF")
        # waking "box" = starting the pretend engine there (a real machine would be a WoL packet)
        wake = [sys.executable, "-m", "nebula.tests.fake_impulse", "-m", self.model, "--port", str(self.port)]
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
            "tiers": [{"name": "far", "engine": f"http://127.0.0.1:{self.port}", "host": "box"}],
            "hosts": [{"name": "box", "wake": {"kind": "command", "command": wake}, "wake_seconds": 20}],
            "callers": [{"id": "nightly", "tags": ["nightly/*"], "class": "background", "models": ["far"]}],
            "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        self.old_path = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = ROOT
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        for p in self.srv.power.procs:
            p.terminate()
            p.wait(timeout=5)
        if self.old_path is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = self.old_path

    def ask(self, label=None):
        h = {"X-Nebula-Session": uuid.uuid4().hex, **({"X-Nebula-Caller": label} if label else {})}
        return call(self.base, "/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]}, h)

    def test_a_background_call_never_wakes_a_machine(self):
        st, body = self.ask("nightly/run")
        self.assertEqual(st, 503, body)
        self.assertIn("only an interactive call wakes a machine", body["error"]["message"])
        self.assertEqual(self.srv.power.procs, [])
        self.assertFalse(call(self.base, "/api/yard/hosts")[1]["hosts"][0]["awake"])

    def test_an_interactive_call_wakes_it_and_waits(self):
        st, body = self.ask()
        self.assertEqual(st, 200, body)
        self.assertEqual(len(self.srv.power.procs), 1)
        reason = self.srv.ledger.db.execute("SELECT reason, inputs FROM decisions ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        self.assertIn("woke:box", json.loads(reason[1])["stages"][4]["why"])
        h = call(self.base, "/api/yard/hosts")[1]["hosts"][0]
        self.assertEqual((h["awake"], h["models"]), (True, ["far"]))
        self.assertEqual(self.ask("nightly/run")[0], 200)              # awake now: background calls use it too

    def test_wake_on_lan_and_checks(self):
        pkt = PW.magic_packet("aa:bb:cc:dd:ee:ff")
        self.assertEqual((len(pkt), pkt[:6], pkt[6:12]), (102, b"\xff" * 6, bytes.fromhex("aabbccddeeff")))
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(5)
        st, r = call(self.base, "/api/yard/hosts", {"name": "lan-box", "wake": {"kind": "wol", "mac": "AA-BB-CC-DD-EE-FF",
                                                                                "broadcast": "127.0.0.1", "port": rx.getsockname()[1]}})
        self.assertEqual(st, 200, r)
        self.assertEqual(call(self.base, "/api/yard/hosts/lan-box/wake", {})[1]["sent"], True)
        self.assertEqual(rx.recv(200), pkt)
        rx.close()
        with self.assertRaises(PW.HostError):
            PW.check({"name": "x", "wake": {"kind": "wol", "mac": "nope"}})
        with self.assertRaises(PW.HostError):
            PW.check({"name": "x", "wake": {"kind": "teleport"}})
        self.assertEqual(json.loads(Path(self.path).read_text())["hosts"][-1]["wake"]["mac"], "aa:bb:cc:dd:ee:ff")


if __name__ == "__main__":
    unittest.main()
