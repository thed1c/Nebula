"""The desktop application's lifetimes, without a display: the engine it starts (or finds), the server behind the
window, the window's choice of backend, the icon, and the entry point under a signal."""
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
from unittest import mock

from nebula.app import shortcut, window
from nebula.tests import WINDOWS, linux_only  # noqa: F401
from nebula.app.engine import EngineProcess, health
from nebula.app.runner import App
from nebula.config import Config, Tier
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_ui import req

ROOT = Path(__file__).resolve().parents[2]


from nebula.tests import free_port  # noqa: E402,F401  (outside the system's outgoing ports)


def fake_cmd(port, *extra):
    return [sys.executable, "-m", "nebula.tests.fake_engine_proc", "--port", str(port), *extra]


def alive(pid: int) -> bool:
    if os.name == "nt":                          # Windows: os.kill(pid, 0) would send it a Ctrl+C
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)    # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259                 # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:                                         # a zombie (not yet reaped) counts as gone
        return Path(f"/proc/{pid}/stat").read_text().split(")")[1].split()[0] != "Z"
    except OSError:
        return True


def gone_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.1)
    return False


class EngineTest(unittest.TestCase):
    def engine(self, *extra, **kw):
        self.port = free_port()
        e = EngineProcess(fake_cmd(self.port, *extra), f"http://127.0.0.1:{self.port}", cwd=str(ROOT), **kw)
        self.addCleanup(e.stop)
        return e

    def test_starts_shows_its_lines_is_ready_and_stops(self):
        e = self.engine("--steps", "3")
        e.start()
        self.assertEqual(e.state, "loading")
        self.assertTrue(e.wait_ready(poll_s=0.1))
        st = e.status()
        self.assertEqual((st["state"], st["managed"]), ("ready", True))
        text = "\n".join(st["lines"])
        self.assertIn("loading the model, step 3 of 3", text)
        self.assertIn("listening on", text)
        self.assertEqual(health(e.address), 200)
        pid = e.proc.pid
        e.stop()
        self.assertEqual(e.state, "stopped")
        self.assertTrue(gone_within(pid, 5))
        self.assertIsNone(health(e.address))

    def test_an_engine_already_answering_is_used_not_started(self):
        fake = FakeEngine("already")
        self.addCleanup(fake.stop)
        e = EngineProcess(["false"], fake.url)
        e.start()
        self.assertEqual(e.state, "attached")
        self.assertIsNone(e.proc)
        self.assertTrue(e.wait_ready())
        e.stop()                                 # not ours: left running
        self.assertEqual(health(fake.url), 200)

    def test_an_engine_that_ends_before_it_is_ready_fails_with_its_lines(self):
        e = self.engine("--fail", "--steps", "1")
        e.start()
        self.assertFalse(e.wait_ready(poll_s=0.1))
        st = e.status()
        self.assertEqual(st["state"], "failed")
        self.assertIn("exit code 3", st["error"])
        self.assertIn("cannot load the model (fake)", "\n".join(st["lines"]))

    def test_nothing_answers_and_nothing_to_start(self):
        e = EngineProcess([], f"http://127.0.0.1:{free_port()}")
        e.start()
        self.assertEqual(e.state, "failed")
        self.assertIn("no command", e.error)

    def test_a_command_that_does_not_exist(self):
        e = EngineProcess(["/nonexistent/strata-lite-server"], f"http://127.0.0.1:{free_port()}")
        e.start()
        self.assertEqual(e.state, "failed")
        self.assertIn("cannot start the engine", e.error)

    def test_a_string_command_is_split(self):
        e = EngineProcess("strata-lite-server -m 'a model.gguf' --port 8080", "http://127.0.0.1:1")
        self.assertEqual(e.command, ["strata-lite-server", "-m", "a model.gguf", "--port", "8080"])

    @unittest.skipIf(os.name == "nt", 'Windows: terminate is already immediate (TerminateProcess)')

    def test_one_that_ignores_sigterm_is_killed(self):
        e = self.engine("--ignore-term", "--steps", "0", stop_timeout_s=1.0)
        e.start()
        self.assertTrue(e.wait_ready(poll_s=0.1))
        pid, t0 = e.proc.pid, time.monotonic()
        e.stop()
        self.assertTrue(gone_within(pid, 5))
        self.assertLess(time.monotonic() - t0, 10)
        self.assertIn("killing it", "\n".join(e.status()["lines"]))

    @unittest.skipIf(os.name == "nt", "process groups are POSIX; Windows has the job object")
    def test_what_the_engine_started_ends_with_it(self):
        pidfile = Path(tempfile.mkdtemp()) / "child.pid"
        e = self.engine("--steps", "0", "--child", str(pidfile))
        e.start()
        self.assertTrue(e.wait_ready(poll_s=0.1))
        child = int(pidfile.read_text())
        self.assertTrue(alive(child))
        e.stop()
        self.assertTrue(gone_within(child, 5))

    @unittest.skipUnless(sys.platform.startswith("linux"), "PR_SET_PDEATHSIG is Linux's")
    def test_the_engine_dies_with_an_app_that_is_killed(self):
        port = free_port()
        script = ("import sys, time\nfrom nebula.app.engine import EngineProcess\n"
                  f"e = EngineProcess({fake_cmd(port, '--steps', '0')!r}, 'http://127.0.0.1:{port}')\n"
                  "e.start(); e.wait_ready(poll_s=0.1); print(e.proc.pid, flush=True); time.sleep(600)\n")
        app = subprocess.Popen([sys.executable, "-c", script], cwd=str(ROOT), stdout=subprocess.PIPE, text=True)
        self.addCleanup(app.stdout.close)
        self.addCleanup(app.kill)
        pid = int(app.stdout.readline())
        self.assertTrue(alive(pid))
        app.kill()                               # SIGKILL: the app runs no cleanup at all
        app.wait()
        self.assertTrue(gone_within(pid, 5))


class AppTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.eport = free_port()
        self.cfg = Config(port=0, tiers=[Tier("fast", f"http://127.0.0.1:{self.eport}", thinking=False)],
                          ledger=str(Path(self.tmp.name) / "ledger.sqlite3"), monitor=False, base_dir=str(ROOT))

    def app(self, *extra, **app_cfg):
        a = App(self.cfg, {"engine": {"command": fake_cmd(self.eport, *extra), "stop_timeout_s": 5}, **app_cfg})
        self.addCleanup(a.close)
        a.start()
        self.base = a.url.rstrip("/")
        return a

    def status(self):
        return json.loads(req(self.base, "/api/app")[2])

    def wait_state(self, *states, seconds=20):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            st = self.status()
            if st["engine"]["state"] in states:
                return st
            time.sleep(0.1)
        self.fail(f"the engine never reached {states}: {self.status()}")

    def test_the_page_is_served_at_once_and_the_engine_comes_up_behind_it(self):
        a = self.app("--steps", "4", "--step-s", "0.3")
        self.assertEqual(req(self.base, "/")[0], 200)              # the window has its page while the engine loads
        st = self.status()
        self.assertTrue(st["app"])
        self.assertIn(st["engine"]["state"], ("idle", "starting", "loading"))
        st = self.wait_state("ready")
        self.assertEqual(st["engine"]["tier"], "fast")
        self.assertIn("listening on", "\n".join(st["engine"]["lines"]))
        status, _h, raw = req(self.base, "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, raw)
        pid = a.engine.proc.pid
        a.close()
        self.assertTrue(gone_within(pid, 10))
        self.assertEqual(a.engine.state, "stopped")
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", int(self.base.rsplit(":", 1)[1])), timeout=1).close()

    def test_a_failed_start_can_be_asked_again_from_the_page(self):
        a = self.app("--fail", "--steps", "1")
        st = self.wait_state("failed")
        self.assertIn("exit code 3", st["engine"]["error"])
        status, _h, raw = req(self.base, "/api/app/engine/start", {})
        self.assertEqual(status, 200, raw)
        self.wait_state("failed")                                    # the same command fails the same way
        self.assertEqual(sum("starting the engine" in x for x in a.engine.status()["lines"]), 2)

    def test_the_engine_s_own_errors_reach_the_error_log(self):
        a = self.app("--fail", "--steps", "1")
        self.wait_state("failed")
        got = [(e["source"], e["level"], e["message"]) for e in json.loads(req(self.base, "/api/errors")[2])["errors"]]
        self.assertIn(("engine:fast", "error", "engine: cannot load the model (fake)"), got)
        self.assertTrue(any(src == "cockpit.app.engine" and "exit code 3" in msg for src, _lvl, msg in got), got)
        self.assertFalse(any("loading the model, step" in msg for _s, _l, msg in got))   # the ordinary lines stay out

    def test_asking_again_while_it_runs_is_refused(self):
        self.app("--steps", "0")
        self.wait_state("ready")
        self.assertEqual(req(self.base, "/api/app/engine/start", {})[0], 409)

    def test_without_an_engine_section_the_app_only_attaches(self):
        a = App(self.cfg, {})
        self.addCleanup(a.close)
        a.start()
        st = json.loads(req(a.url.rstrip("/"), "/api/app")[2])
        self.assertEqual((st["app"], st["engine"]), (True, None))

    def test_close_twice_is_harmless(self):
        a = self.app("--steps", "0")
        self.wait_state("ready")
        a.close()
        a.close()

    def test_keep_awake_is_held_while_an_attempt_runs(self):
        calls = []

        class Fake:
            def acquire(self): calls.append("acquire")
            def release(self): calls.append("release")
            def close(self): calls.append("close")
        with mock.patch("nebula._vendor.strata_serve.keepawake.KeepAwake.create", return_value=Fake()):
            a = self.app("--steps", "0", prevent_sleep=True)
        self.wait_state("ready")
        ledger, seen = a.server.ledger, {}
        orig = ledger.activity.token

        def token(attempt_id, n=1):              # in flight: the watcher has had a moment to see the running attempt
            orig(attempt_id, n)
            if not seen:
                time.sleep(0.3)
                seen["calls"] = list(calls)
        ledger.activity.token = token
        status, _h, raw = req(self.base, "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200, raw)
        deadline = time.monotonic() + 5
        while calls[-1:] != ["release"] and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(seen["calls"], ["acquire"])
        self.assertEqual(calls, ["acquire", "release"])
        a.close()
        self.assertEqual(calls[-1], "close")

    def test_a_browser_session_is_not_the_app(self):
        from nebula.tests.test_escalation import Rig
        rig = Rig()
        self.addCleanup(rig.stop)
        self.assertEqual(json.loads(req(rig.base, "/api/app")[2]), {"app": False})
        self.assertEqual(req(rig.base, "/api/app/engine/start", {})[0], 404)


class WindowTest(unittest.TestCase):
    def test_backends_best_first(self):
        with mock.patch.object(window, "_has_pywebview", return_value=True), \
             mock.patch.object(window, "_has_gtk", return_value="4.1"), mock.patch.object(window.sys, "platform", "linux"):
            self.assertEqual(window.backends(), ["pywebview", "gtk", "browser"])
        with mock.patch.object(window, "_has_pywebview", return_value=False), \
             mock.patch.object(window, "_has_gtk", return_value=None), mock.patch.object(window.sys, "platform", "linux"):
            self.assertEqual(window.backends(), ["browser"])
        with mock.patch.object(window, "_has_pywebview", return_value=True), \
             mock.patch.object(window, "_has_gtk", return_value="4.1"), mock.patch.object(window.sys, "platform", "win32"):
            self.assertEqual(window.backends(), ["pywebview", "browser"])

    def test_a_backend_that_is_not_here_is_refused(self):
        with mock.patch.object(window, "backends", return_value=["browser"]):
            with self.assertRaises(RuntimeError):
                window.show("http://127.0.0.1:1/", backend="pywebview")

    def test_error_page_escapes(self):
        page = window.error_page("<t>", "a & b", "<script>")
        self.assertIn("&lt;t&gt;", page)
        self.assertIn("a &amp; b", page)
        self.assertNotIn("<script>", page)


class ShortcutTest(unittest.TestCase):
    @linux_only
    def test_desktop_entry(self):
        text = shortcut.desktop_entry("/opt/venv/bin/python3", "my config.json", root=Path("/repo"))
        lines = dict(x.split("=", 1) for x in text.splitlines() if "=" in x)
        self.assertEqual(lines["Exec"], f"/opt/venv/bin/python3 -m nebula.app --config '{Path('my config.json').resolve()}'")
        self.assertEqual((lines["Path"], lines["Terminal"], lines["StartupWMClass"]), ("/repo", "false", window.APP_ID))
        self.assertTrue(lines["Icon"].endswith("icon.svg") and Path(lines["Icon"]).exists())

    def test_install_linux_writes_an_executable_entry(self):
        d = Path(tempfile.mkdtemp())
        path = shortcut.install_linux(None, apps_dir=d, python="/usr/bin/python3")
        self.assertEqual(path, d / "nebula-eh.desktop")
        self.assertTrue(os.access(path, os.X_OK))
        self.assertIn("Exec=/usr/bin/python3 -m nebula.app\n", path.read_text())

    def test_windows_script(self):
        s = shortcut.windows_script(r"C:\venv\Scripts\python.exe", None, root=Path("C:/strata"))
        self.assertIn("pythonw.exe", s)
        self.assertIn("$s.Arguments = '-m nebula.app'", s)
        self.assertIn("Nebula EH.lnk", s)


@unittest.skipIf(os.name == "nt", "SIGTERM to a child is POSIX")
class MainTest(unittest.TestCase):
    def test_no_window_runs_until_sigterm_then_stops_the_engine(self):
        tmp = Path(tempfile.mkdtemp())
        eport, port = free_port(), free_port()
        cfg = {"port": port, "ledger": "ledger.sqlite3", "monitor": False,
               "tiers": [{"name": "fast", "engine": f"http://127.0.0.1:{eport}"}],
               "app": {"engine": {"command": fake_cmd(eport, "--steps", "2"), "cwd": str(ROOT)}}}
        (tmp / "cockpit.json").write_text(json.dumps(cfg))
        p = subprocess.Popen([sys.executable, "-m", "nebula.app", "--config", str(tmp / "cockpit.json"), "--no-window"],
                             cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(p.stdout.close)
        self.addCleanup(p.kill)
        line = p.stdout.readline()
        self.assertIn(f"http://127.0.0.1:{port}/", line)
        base = f"http://127.0.0.1:{port}"
        deadline, st = time.monotonic() + 20, {}
        while time.monotonic() < deadline:
            st = json.loads(req(base, "/api/app")[2])
            if st["engine"]["state"] == "ready":
                break
            time.sleep(0.1)
        self.assertEqual(st["engine"]["state"], "ready")
        self.assertTrue((tmp / "ledger.sqlite3").exists())          # the ledger beside the config, not in the cwd
        pid = st["engine"]["pid"]
        p.send_signal(signal.SIGTERM)
        out = p.communicate(timeout=30)[0]
        self.assertEqual(p.returncode, 0, out)
        self.assertIn("[cockpit] stopped", out)
        self.assertTrue(gone_within(pid, 5))

    def test_a_taken_port_is_said_and_exits(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            tmp = Path(tempfile.mkdtemp())
            (tmp / "c.json").write_text(json.dumps({"port": s.getsockname()[1], "ledger": None, "monitor": False,
                                                    "tiers": [{"name": "fast", "engine": "http://127.0.0.1:1"}]}))
            r = subprocess.run([sys.executable, "-m", "nebula.app", "--config", str(tmp / "c.json"), "--no-window"],
                               cwd=str(ROOT), capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 1)
        self.assertIn("cannot listen", r.stderr)


if __name__ == "__main__":
    unittest.main()
