"""The engine's own warnings and errors into the error log: which lines count, following a file or a command."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from nebula.config import Config
from nebula.tests import WINDOWS, linux_only  # noqa: F401
from nebula.enginelog import Follower, classify
from nebula.errors import ErrorLog


def wait_for(cond, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


class ClassifyTest(unittest.TestCase):
    def test_problem_lines_and_echoes(self):
        cases = {   # the first four are from a live strata-lite-server's journal (2026-10-15)
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 13632.89 MiB on device 0: cudaMalloc failed: out of memory": "error",
            "/build/ggml/src/ggml-backend.cpp:162: GGML_ASSERT(buffer) failed": "error",
            "Grammar still awaiting trigger after token 3914 (` cannot`)": None,
            "ggml_backend_cuda_graph_compute: CUDA graph warmup reset": None,
            "add_text: please fix this error: it failed to load": None,
            "<|turn>model": None,
            "strata-lite-server: listening on http://127.0.0.1:8080 (alias a, ctx 262144, vision 1, mtp 1)": None,
            "strata-lite-server: cannot listen on 127.0.0.1:8080": "error",
            "engine: cannot load the model": "error",
            "CUDA error: an illegal memory access was encountered": "error",
            "terminate called after throwing an instance of 'std::runtime_error'": "error",
            "llama_model_loader: warning: unknown key": "warning",
            "W the cache is cold": "warning",
            "the error was in my code": None,
            "": None,
        }
        for line, want in cases.items():
            with self.subTest(line=line):
                self.assertEqual(classify(line), want)


class FollowTest(unittest.TestCase):
    def setUp(self):
        self.log = ErrorLog()

    def messages(self):
        return [(e["source"], e["level"], e["message"]) for e in self.log.list()["errors"]]

    @linux_only

    def test_a_file_from_its_end_across_truncation_and_rotation(self):
        d = Path(tempfile.mkdtemp())
        path = d / "engine.log"
        path.write_text("engine: cannot load the model (old, before the cockpit started)\n")
        f = Follower({"tier": "fast", "file": str(path)}, self.log, poll_s=0.05).start()
        self.addCleanup(f.close)
        time.sleep(0.3)
        with open(path, "a") as out:
            out.write("strata-lite: KV on the card 32768 -> 49152 cells\nggml: cudaMalloc failed: out of")
            out.flush()
            time.sleep(0.2)
            out.write(" memory\n")
        self.assertTrue(wait_for(lambda: len(self.messages()) == 1), self.messages())
        self.assertEqual(self.messages(), [("engine:fast", "error", "ggml: cudaMalloc failed: out of memory")])
        path.write_text("GGML_ASSERT(buffer) failed\n")                       # truncated and written again
        self.assertTrue(wait_for(lambda: len(self.messages()) == 2), self.messages())
        os.rename(path, d / "engine.log.1")                                   # rotated
        path.write_text("CUDA error: out of memory\n")
        self.assertTrue(wait_for(lambda: len(self.messages()) == 3), self.messages())
        self.assertEqual(self.messages()[-1], ("engine:fast", "error", "CUDA error: out of memory"))

    def test_a_command_and_repeats_collapsed(self):
        code = ("import time\nfor m in ['loading', 'engine: cannot load the model', 'engine: cannot load the model', "
                "'llama: warning: slow']: print(m, flush=True)\ntime.sleep(30)")
        f = Follower({"tier": "big", "command": [sys.executable, "-c", code]}, self.log).start()
        self.assertTrue(wait_for(lambda: len(self.messages()) == 2), self.messages())
        f.close()
        got = self.log.list()["errors"]
        self.assertEqual([(e["level"], e["message"], e.get("repeats", 1)) for e in got],
                         [("error", "engine: cannot load the model", 2), ("warning", "llama: warning: slow", 1)])
        self.assertIsNotNone(f.proc.poll())                                   # the command ended with the follower

    def test_a_log_that_cannot_be_followed_says_so(self):
        Follower({"tier": "big", "command": ["no-such-log-command-here", "-f"]}, self.log).start()
        Follower({"tier": "small", "command": [sys.executable, "-c", "raise SystemExit(1)"]}, self.log).start()
        self.assertTrue(wait_for(lambda: len(self.log.list()["errors"]) == 2), self.log.list())
        got = sorted((e["level"], e["message"]) for e in self.log.list()["errors"])
        self.assertEqual(got[0][0], "warning")
        self.assertIn("the engine log command for small ended (exit code 1)", got[0][1])
        self.assertIn("the engine log for big cannot be followed", got[1][1])

    def test_a_bad_entry_is_refused(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"tiers": [{"name": "fast", "engine": "http://x"}], "engine_logs": [{"tier": "fast"}]})
        cfg = Config.from_dict({"tiers": [{"name": "fast", "engine": "http://x"}], "engine_logs": [{"tier": "fast", "file": "/tmp/x.log"}]})
        self.assertEqual(cfg.engine_logs, [{"tier": "fast", "file": "/tmp/x.log"}])


if __name__ == "__main__":
    unittest.main()
