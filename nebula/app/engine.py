"""nebula/app/engine.py: the engine the application starts, when it is told to.

    eng = EngineProcess(command=[...], address="http://127.0.0.1:8080")
    eng.start()          # from the thread that lives as long as the app (see below); returns at once
    eng.wait_ready()     # True once GET /health answers 200; False if the process ended or the time ran out
    eng.status()         # {"state", "lines", "pid", "seconds", ...} for the window
    eng.stop()           # SIGTERM (Windows: terminate), then kill after stop_timeout_s

An engine that already answers at the address is used as it is and never started or stopped by the app: the same
model may run as a service the owner started some other way, and a second copy would not fit on the card.

strata-lite-server loads the model before it listens, so its own startup progress is its stderr: every line is kept
(the last 400) for the window, and the state reads "loading" until /health answers.

Lifetimes, as Strata's own server does it (serve/winjob.py): on Windows the engine goes into a job object that kills it
when the app's last handle closes, however the app ends; on Linux it is started with PR_SET_PDEATHSIG, so it is sent
SIGTERM when the thread that started it ends. That is a thread, not the process, so `start()` must be called from a
thread that lives as long as the app: the runner's supervisor (nebula/app/runner.py), never a request's thread. It
also gets its own process group, so a Ctrl+C in the app's terminal reaches the app, which then stops the engine in
order.
"""
from __future__ import annotations

import collections
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import logging
import urllib.error
import urllib.request

LOG = logging.getLogger("cockpit.app.engine")

from nebula.cmdline import split as split_command  # noqa: E402  (POSIX on Linux; Windows paths kept on Windows)


PR_SET_PDEATHSIG = 1


def _die_with_parent():                       # runs in the child between fork and exec (Linux only)
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:  # noqa: BLE001 - a libc without prctl: the app's own stop() still ends it
        pass


def health(address: str, timeout: float = 2.0) -> int | None:
    """The engine's GET /health status, or None when nothing answers."""
    try:
        with urllib.request.urlopen(address.rstrip("/") + "/health", timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (OSError, ValueError):
        return None


class EngineProcess:
    def __init__(self, command, address: str, cwd: str | None = None, env: dict | None = None,
                 ready_timeout_s: float = 900.0, stop_timeout_s: float = 30.0, log_path: str | None = None):
        self.command = split_command(command) if isinstance(command, str) else [str(c) for c in command or []]
        self.address = address
        self.cwd, self.env = cwd, env
        self.ready_timeout_s, self.stop_timeout_s = ready_timeout_s, stop_timeout_s
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.lines: collections.deque[str] = collections.deque(maxlen=400)
        self.state = "idle"        # idle | attached | starting | loading | ready | failed | stopping | stopped
        self.error: str | None = None
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self.lock = threading.Lock()
        self.reader: threading.Thread | None = None
        self.on_line = None                     # called with each line of the engine's output (the error log's filter)

    @property
    def managed(self) -> bool:
        return self.proc is not None

    def _say(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)
        if self.on_line is not None and not line.startswith("[cockpit]"):
            try:
                self.on_line(line)
            except Exception:  # noqa: BLE001
                pass
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    def start(self) -> None:
        """Start the engine unless one already answers at the address. Returns at once; `wait_ready` waits."""
        if health(self.address) == 200:
            self.state = "attached"
            self._say(f"[cockpit] an engine already answers at {self.address}: using it, not starting another")
            return
        if not self.command:
            self.state, self.error = "failed", "no engine answers and no command is configured to start one"
            self._say(f"[cockpit] {self.error}")
            return
        kw: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                    "cwd": self.cwd, "env": {**os.environ, **{k: str(v) for k, v in (self.env or {}).items()}},
                    "text": True, "encoding": "utf-8", "errors": "replace", "bufsize": 1}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kw["start_new_session"] = True
            if sys.platform.startswith("linux"):
                kw["preexec_fn"] = _die_with_parent
        self.state, self.started_at = "starting", time.time()
        self._say("[cockpit] starting the engine: " + " ".join(shlex.quote(c) for c in self.command))
        try:
            self.proc = subprocess.Popen(self.command, **kw)
        except OSError as e:
            self.state, self.error = "failed", f"cannot start the engine: {e}"
            self._say(f"[cockpit] {self.error}")
            LOG.error("engine at %s: %s", self.address, self.error)
            return
        if os.name == "nt":
            try:
                from nebula._vendor.strata_serve.winjob import contain
                contain(self.proc)            # killed with the app's last handle, however the app ends
            except Exception:  # noqa: BLE001
                pass
        self.state = "loading"
        self.reader = threading.Thread(target=self._read, name="engine-output", daemon=True)
        self.reader.start()

    def _read(self) -> None:
        """Drains the engine's output for its whole life (a full pipe would stop it), keeping the lines."""
        assert self.proc is not None and self.proc.stdout is not None
        out = self.proc.stdout
        try:
            for line in out:
                self._say(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        finally:
            out.close()

    def wait_ready(self, poll_s: float = 0.5) -> bool:
        if self.state == "attached":
            return True
        if self.proc is None:
            return False
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            if self.state in ("stopping", "stopped"):
                return False
            if self.proc.poll() is not None:
                if self.reader is not None:
                    self.reader.join(timeout=2)
                self.state, self.error = "failed", f"the engine ended with exit code {self.proc.returncode} before it was ready"
                self._say(f"[cockpit] {self.error}")
                LOG.error("engine at %s: %s", self.address, self.error)
                return False
            if health(self.address, timeout=1.0) == 200:
                self.state, self.ready_at = "ready", time.time()
                self._say(f"[cockpit] the engine is ready after {self.ready_at - self.started_at:.1f} s")
                return True
            time.sleep(poll_s)
        self.state, self.error = "failed", f"the engine did not answer within {self.ready_timeout_s:.0f} s"
        self._say(f"[cockpit] {self.error}")
        LOG.error("engine at %s: %s", self.address, self.error)
        return False

    def stop(self) -> None:
        """Ends an engine the app started: SIGTERM (Windows: terminate), then kill. An attached one is left alone."""
        p = self.proc
        if p is None:
            return
        if p.poll() is None:
            self.state = "stopping"
            self._say("[cockpit] stopping the engine")
            try:
                if os.name == "nt":
                    p.terminate()
                else:
                    os.killpg(p.pid, signal.SIGTERM)      # its group: whatever it started ends with it
                p.wait(timeout=self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                self._say(f"[cockpit] the engine did not end within {self.stop_timeout_s:.0f} s: killing it")
                try:
                    if os.name == "nt":
                        p.kill()
                    else:
                        os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
                p.wait(timeout=10)
            except OSError:
                pass
        self.state = "stopped"
        if self.reader is not None:
            self.reader.join(timeout=2)

    def status(self) -> dict:
        with self.lock:
            lines = list(self.lines)
        now = time.time()
        return {"state": self.state, "managed": self.managed, "address": self.address, "error": self.error,
                "pid": self.proc.pid if self.proc else None,
                "seconds": round((self.ready_at or now) - self.started_at, 1) if self.started_at else None,
                "lines": lines[-60:]}
