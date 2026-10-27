"""Engines on Windows in a container through WSL2 and Podman (nebula/wslbox.py, Setup's Safe sandbox): the state Setup
left decides, a Windows path is a /mnt/<drive> path inside, the container is started with read-only mounts, no
capabilities and its port on 127.0.0.1 only, and nebula/engines.py picks it on Windows (else a plain process, warned).
wsl.exe is a fake that records what it was asked."""
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from nebula import engines as E
from nebula import wslbox as W


class FakeWsl:
    def __init__(self, answers=None):
        self.calls, self.answers = [], dict(answers or {})

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        inner = argv[argv.index("--exec") + 1:]
        assert not inner or inner[0] != "podman", "Podman is reached through Setup's nebula-podman wrapper"
        inner = [("podman" if w == "nebula-podman" else w) for w in inner[:1]] + inner[1:]
        for key, (code, out) in self.answers.items():
            if " ".join(inner).startswith(key):
                return subprocess.CompletedProcess(argv, code, out, "")
        return subprocess.CompletedProcess(argv, 0, "", "")


def write_state(d, **st):
    with open(os.path.join(d, "sandbox.json"), "w", encoding="utf-8") as f:
        json.dump(st, f)


class PathTest(unittest.TestCase):
    def test_windows_paths(self):
        self.assertEqual(W.to_wsl(r"C:\Users\me\AppData\Roaming\nebula\models"), "/mnt/c/Users/me/AppData/Roaming/nebula/models")
        self.assertEqual(W.to_wsl("D:/Models/x.gguf"), "/mnt/d/Models/x.gguf")
        self.assertEqual(W.to_wsl("E:\\"), "/mnt/e")
        self.assertEqual(W.to_wsl("/mnt/c/already"), "/mnt/c/already")
        with self.assertRaises(W.WslError):
            W.to_wsl(r"\\nas\share\model.gguf")
        with self.assertRaises(W.WslError):
            W.to_wsl("relative\\x")

    def test_container_name(self):
        self.assertEqual(W.container_name("My Model 7B"), "nebeng-my-model-7b")


class StateTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        W._READY.clear()
        self.env = mock.patch.dict(os.environ, {"NEBULA_WSL": "wsl.exe"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_each_state_says_why(self):
        self.assertIn("not set up", W.why_not(self.d))
        write_state(self.d, state="declined")
        self.assertIn("you chose", W.why_not(self.d))
        write_state(self.d, state="reboot")
        self.assertIn("restart", W.why_not(self.d))
        write_state(self.d, state="failed", why="Ubuntu's root filesystem did not check out (sha256)")
        self.assertIn("did not check out", W.why_not(self.d))

    def test_ready_and_podman_answers(self):
        write_state(self.d, state="ready", images={"impulse": "nebula-engine-impulse:0.1.3"})
        fake = FakeWsl({"podman info": (0, "4.9.3\n")})
        self.assertIsNone(W.why_not(self.d, runner=fake))
        self.assertEqual(fake.calls[0][:4], ["wsl.exe", "-d", "NebulaEH", "--exec"])
        self.assertEqual(W.image_for(self.d, "impulse"), "nebula-engine-impulse:0.1.3")

    def test_a_state_file_with_a_bom(self):
        """Windows PowerShell 5.1's Set-Content -Encoding UTF8 writes a BOM (seen on a test PC, 2026-10-26: "ready" read as unset)."""
        with open(os.path.join(self.d, "sandbox.json"), "w", encoding="utf-8-sig") as f:
            json.dump({"state": "ready", "images": {"impulse": "nebula-engine-impulse:0.1.3"}}, f)
        self.assertIsNone(W.why_not(self.d, runner=FakeWsl({"podman info": (0, "4.9.3\n")})))
        self.assertEqual(W.image_for(self.d, "impulse"), "nebula-engine-impulse:0.1.3")

    def test_ready_but_podman_does_not_answer(self):
        write_state(self.d, state="ready")
        why = W.why_not(self.d, runner=FakeWsl({"podman info": (125, "")}))
        self.assertIn("does not answer", why)


class EngineTest(unittest.TestCase):
    def spec(self):
        return {"name": "Small", "kind": "impulse", "port": 18301}

    def test_run_argv(self):
        argv = W.run_argv("Small", "nebula-engine-impulse:0.1.3", ["/opt/engine/impulse-server", "-m", "/models/a.gguf"], 18301,
                          [(r"C:\Users\me\models", "/models"), (r"C:\Users\me\AppData\Roaming\nebula\nebula-keys\sandbox-Small", "/keys")],
                          {"CUDA_VISIBLE_DEVICES": "GPU-1"}, True)
        s = " ".join(argv)
        for want in ("podman run -d --replace --name nebeng-small", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                     "--network pasta -p 127.0.0.1:18301:18301", "--device nvidia.com/gpu=all", "-v /mnt/c/Users/me/models:/models:ro",
                     "/keys:ro", "-e CUDA_VISIBLE_DEVICES=GPU-1", "nebula-engine-impulse:0.1.3 /opt/engine/impulse-server -m /models/a.gguf"):
            self.assertIn(want, s)
        self.assertNotIn("--device", " ".join(W.run_argv("x", "i", [], 1, [], {}, False)))

    def test_start_exited_stop(self):
        fake = FakeWsl({"podman inspect": (0, "exited 139\n"), "podman logs": (0, "segfault in load\n")})
        e = W.WslEngine(self.spec(), ["/opt/engine/impulse-server"], [("/mnt/c/m", "/models")], "img", runner=fake)
        with mock.patch.dict(os.environ, {"NEBULA_WSL": "wsl.exe"}):
            e.start()
            self.assertEqual((e.state, e.address), ("loading", "http://127.0.0.1:18301"))
            self.assertIn("exited (139): segfault in load", e.exited())
            e.stop()
        self.assertEqual(e.state, "stopped")
        self.assertEqual(fake.calls[-1][4:8], ["nebula-podman", "rm", "-f", "-t"])
        self.assertEqual(e.status()["where"], "wsl2-podman")

    def test_a_container_that_does_not_start(self):
        fake = FakeWsl({"podman run": (125, "")})
        e = W.WslEngine(self.spec(), [], [], "img", runner=fake)
        with mock.patch.dict(os.environ, {"NEBULA_WSL": "wsl.exe"}):
            e.start()
        self.assertEqual(e.state, "failed")
        self.assertIn("did not start", e.error)

    def test_no_wsl(self):
        e = W.WslEngine(self.spec(), [], [], "img")
        with mock.patch.object(W, "wsl_exe", return_value=None):
            e.start()
        self.assertEqual((e.state, e.error), ("failed", "WSL is not installed on this machine"))


class PlanTest(unittest.TestCase):
    """nebula/engines.py on Windows: the sandbox is WSL2 + Podman when Setup made it, else a plain process, warned."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        W._READY.clear()

    def plan(self, spec):
        real = W.why_not
        with mock.patch.object(E.os, "name", "nt"), mock.patch.object(W, "why_not", lambda c, **k: real(c, runner=FakeWsl({"podman info": (0, "4.9.3")}))):
            with mock.patch.dict(os.environ, {"NEBULA_WSL": "wsl.exe"}):
                return E.sandbox_plan(spec, conf_dir=self.d)

    def test_declined_runs_plainly(self):
        write_state(self.d, state="declined")
        self.assertEqual(self.plan({"name": "a", "kind": "impulse"})[0], "local")

    def test_ready_with_an_image(self):
        write_state(self.d, state="ready", images={"impulse": "nebula-engine-impulse:0.1.3"})
        self.assertEqual(self.plan({"name": "a", "kind": "impulse"}), ("openshell", None))

    def test_ready_without_impulse_image(self):
        write_state(self.d, state="ready", images={})
        where, why = self.plan({"name": "a", "kind": "impulse"})
        self.assertEqual(where, "local")
        self.assertIn("no image for", why)

    def test_llama_cpp_pulls_its_own_image(self):
        write_state(self.d, state="ready", images={})
        self.assertEqual(self.plan({"name": "a", "kind": "llama.cpp"}), ("openshell", None))


class Cfg:
    def __init__(self, d, engines):
        self.engines, self.base_dir, self.path, self.raw = engines, d, os.path.join(d, "nebula.json"), {"models_dir": os.path.join(d, "models")}


class SandboxedTest(unittest.TestCase):
    """Engines._sandboxed on Windows builds a WslEngine: the model under /models, the engine on all of the container's
    addresses, the image Setup built, Impulse's Linux program in it, the card picked by CUDA_VISIBLE_DEVICES."""

    def test_impulse_on_windows(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "models"))
        model = os.path.join(d, "models", "small.gguf")
        open(model, "wb").close()
        write_state(d, state="ready", images={"impulse": "nebula-engine-impulse:0.1.3"})
        spec = {"name": "small", "kind": "impulse", "model": model, "port": 18301}
        en = E.Engines(Cfg(d, [spec]))
        with mock.patch.object(E.os, "name", "nt"), mock.patch.object(E, "cards", return_value=[]):
            e = en._sandboxed(spec, [{"uuid": "GPU-aaaa", "index": 0}])
        self.assertIsInstance(e, W.WslEngine)
        self.assertEqual(e.image, "nebula-engine-impulse:0.1.3")
        self.assertEqual(e.mounts[0], (os.path.realpath(os.path.join(d, "models")), "/models"))
        cmd = " ".join(e.inner_cmd)
        self.assertTrue(cmd.startswith("/opt/engine/impulse-server"), cmd)
        self.assertIn("-m /models/small.gguf", cmd)
        self.assertIn("--host 0.0.0.0", cmd)
        self.assertEqual(e.env["CUDA_VISIBLE_DEVICES"], "GPU-aaaa")
        self.assertTrue(e.gpu)


if __name__ == "__main__":
    unittest.main()
