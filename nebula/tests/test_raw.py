"""The Raw tab (nebula/packages.py parse_command, /api/yard/raw): settings → command line → settings is the same,
flags without a setting are kept verbatim, Nebula's hardening is locked and put back, a change is a diff and then a
new deployment that Undo takes back."""
import json
import os
import shlex
import tempfile
import threading
import unittest
from pathlib import Path

from nebula import engines as E
from nebula import packages as P
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.test_callers import call


class ParseTest(unittest.TestCase):
    def test_round_trip(self):
        spec = E.check_spec({"name": "a", "kind": "llama.cpp", "binary": "/opt/llama/llama-server", "model": "/m/x.gguf",
                             "port": 18300, "context": 8192, "slots": 2,
                             "options": {"gpu_layers": 99, "flash_attn": "on", "temp": 0.7, "thinking": False, "jinja": True,
                                         "cache_type_k": "q8_0", "cache_type_v": "q8_0", "top_k": 40},
                             "flags": ["--some-new-flag", "3", "--verbose"]})
        text = " ".join(shlex.quote(w) for w in E.command_of(spec))
        r = P.parse_command("llama.cpp", text)
        self.assertEqual(r["invalid"], [])
        back = r["spec"]
        for k in ("binary", "model", "port", "context", "slots", "flags"):
            self.assertEqual(back.get(k), spec.get(k), k)
        self.assertEqual(back["options"], spec["options"])
        self.assertIn("--no-webui: Nebula turns the engine's own chat page off", r["locked"])     # put back, not kept
        self.assertEqual([w for w in r["warnings"] if "--some-new-flag" in w][0].split(":")[0], "--some-new-flag is not one of llama.cpp's known settings")

    def test_locked_and_invalid(self):
        r = P.parse_command("llama.cpp", "llama-server -m x.gguf --host 0.0.0.0 --port 9 --api-key secret --cache-ram 8192 --webui")
        self.assertNotIn("host", r["spec"])
        self.assertEqual(len(r["locked"]), 4)
        self.assertTrue(any("this machine only" in x for x in r["locked"]))
        self.assertNotIn("secret", json.dumps(r["spec"]))                                       # never kept
        self.assertTrue(P.parse_command("llama.cpp", "llama-server --top-p 7")["invalid"])        # out of range
        self.assertTrue(P.parse_command("llama.cpp", "llama-server -m 'unclosed")["invalid"])


class RawApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({"port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
                                               "engines": [{"name": "e", "kind": "llama.cpp", "model": "m.gguf", "port": 18301,
                                                            "context": 4096, "slots": 1, "guests": True, "card": "0"}],
                                               "lan": {"decided": True, "discoverable": False}}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()

    def test_diff_save_undo_and_hardening_put_back(self):
        text = "llama-server -m m.gguf --port 18301 -c 8192 -np 1 --temp 0.5 --no-webui --my-flag x --cache-ram 4096"
        st, r = call(self.base, "/api/yard/raw", {"engine": "e", "text": text})
        self.assertEqual((st, r["invalid"]), (200, []), r)
        changed = {d["setting"]: d["to"] for d in r["diff"]}
        self.assertEqual((changed["context"], changed["temp"], changed["flags"]), (8192, 0.5, ["--my-flag", "x"]))
        self.assertEqual(r["spec"]["card"], "0")                                                 # what the line does not say is kept
        self.assertTrue(r["spec"]["guests"])
        cmd = r["command"]
        for w in ("--no-webui", "--cache-ram", "--no-cache-idle-slots", "-sps", "--api-key-file", "--my-flag"):
            self.assertIn(w, cmd)                                                                # Nebula's own, back
        self.assertEqual(cmd[cmd.index("--cache-ram") + 1], "0")                                 # not the 4096 asked
        st, out = call(self.base, "/api/yard/engines", r["spec"])
        self.assertEqual((st, out["version"]), (200, 1), out)
        self.assertEqual(self.srv.yard.spec("e")["options"]["temp"], 0.5)
        st, r2 = call(self.base, "/api/yard/raw", {"engine": "e", "text": "llama-server -m m.gguf --port 18301 -c 4096"})
        call(self.base, "/api/yard/engines", r2["spec"])
        call(self.base, "/api/yard/engines/e/undo", {})
        self.assertEqual(self.srv.yard.spec("e")["flags"], ["--my-flag", "x"])                     # Undo: the raw edit back

    def test_an_engine_listens_on_this_machine_only(self):
        st, err = call(self.base, "/api/yard/engines", dict(self.srv.yard.spec("e"), host="0.0.0.0"))
        self.assertEqual(st, 400)
        self.assertIn("this machine only", err["error"]["message"])


if __name__ == "__main__":
    unittest.main()
