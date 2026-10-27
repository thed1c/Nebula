"""nebula/wslbox.py: engines on Windows in a container, through WSL2 and Podman (Setup's "Safe sandbox" step).

Setup (greenfield/windows/sandbox.ps1) makes a WSL2 distro of Nebula's own (`NebulaEH`, Ubuntu 24.04 from a pinned,
sha256-checked root filesystem), installs a pinned Podman and NVIDIA's container toolkit in it, describes the NVIDIA
card to Podman (CDI, `nvidia-ctk cdi generate --mode=wsl`), builds the engine's image from the Linux Impulse, and
writes what it did to `sandbox.json` beside the config:

    {"state": "ready" | "declined" | "failed" | "reboot", "why": "...", "distro": "NebulaEH", "podman": "4.9.3",
     "gpu": true, "images": {"impulse": "nebula-engine-impulse:0.1.3"}}

Here, an engine Nebula would put in an OpenShell sandbox on Linux runs instead as a Podman container in that distro:
its models folder read-only at /models (C:\\... is /mnt/c/... inside WSL), its card through CDI (WSL has one GPU
device for all cards; CUDA_VISIBLE_DEVICES picks the card), no capabilities, no new privileges, its port published
on 127.0.0.1 only (WSL forwards it to Windows' own 127.0.0.1). OpenShell itself is not on Windows yet: the
container is the sandbox, with no network policy of OpenShell's (it reaches what the distro reaches).

Without the distro (declined, failed, not installed), why_not() says so and the engine runs as a plain process with
the usual warning.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time

DISTRO = "NebulaEH"
STATE_FILE = "sandbox.json"
_READY: dict = {}                        # the config folder -> {"at", "why"}


def wsl_exe() -> str | None:
    return os.environ.get("NEBULA_WSL") or shutil.which("wsl.exe") or shutil.which("wsl")


def state(conf_dir: str | None) -> dict:
    """What Setup's sandbox step left (sandbox.json beside the config), or {} when it never ran."""
    try:
        with open(os.path.join(conf_dir or ".", STATE_FILE), encoding="utf-8-sig") as f:   # PowerShell 5.1 writes a BOM
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def run(args: list[str], timeout: float = 120, input_text: str | None = None, runner=None) -> subprocess.CompletedProcess:
    """`podman <args>` (or any program) inside the distro, without a shell: wsl.exe -d NebulaEH --exec <args>."""
    exe = wsl_exe()
    if exe is None:
        raise WslError("WSL is not installed on this machine")
    if args and args[0] == "podman":                     # Setup's wrapper: Podman with the user's own runtime folder (WSLg's
        args = ["nebula-podman"] + args[1:]               # /mnt/wslg/runtime-dir is uid 1000's, which Podman refuses)
    argv = [exe, "-d", DISTRO, "--exec"] + args
    r = (runner or subprocess.run)(argv, capture_output=True, text=True, timeout=timeout, input=input_text,
                                    encoding="utf-8", errors="replace")
    return r


class WslError(RuntimeError):
    pass


def why_not(conf_dir: str | None, max_age_s: float = 60.0, runner=None) -> str | None:
    """None when engines can go into a WSL2 container here; else why not, in a few words (checked once a minute)."""
    st = state(conf_dir)
    s = st.get("state")
    if s == "declined":
        return "you chose to run engines without a sandbox (Setup's Safe sandbox step)"
    if s == "reboot":
        return "the Safe sandbox needs Windows to restart once to finish"
    if s == "failed":
        return "the Safe sandbox could not be set up: " + str(st.get("why") or "unknown")[:160]
    if s != "ready":
        return "the Safe sandbox (WSL2 + Podman) is not set up: run Setup again to add it"
    key = os.path.realpath(conf_dir or ".")
    seen = _READY.get(key)
    if seen and time.time() - seen["at"] < max_age_s:
        return seen["why"]
    try:
        r = run(["podman", "info", "--format", "{{.Version.Version}}"], timeout=60, runner=runner)
        why = None if r.returncode == 0 and r.stdout.strip() else \
            "Podman in the NebulaEH distro does not answer: " + (r.stderr or r.stdout or "").strip()[:160]
    except (OSError, subprocess.TimeoutExpired, WslError) as e:
        why = f"the NebulaEH distro does not start: {e}"[:200]
    _READY[key] = {"at": time.time(), "why": why}
    return why


def image_for(conf_dir: str | None, kind: str) -> str | None:
    return (state(conf_dir).get("images") or {}).get(kind)


def to_wsl(path: str) -> str:
    """A Windows path as WSL sees it: C:\\Users\\me\\models -> /mnt/c/Users/me/models. A network (UNC) path is refused."""
    p = path.replace("/", "\\")
    m = re.match(r"^([A-Za-z]):(\\.*)?$", p)
    if not m:
        if p.startswith("\\\\"):
            raise WslError(f"{path}: a network folder cannot be mounted into the sandbox; copy the model to a local drive")
        if path.startswith("/"):
            return path                                        # already a Linux path (tests, a path inside WSL)
        raise WslError(f"{path}: not an absolute Windows path")
    rest = (m.group(2) or "").replace("\\", "/").rstrip("/")
    return f"/mnt/{m.group(1).lower()}{rest}"


def container_name(engine: str) -> str:
    return "nebeng-" + (re.sub(r"[^a-z0-9_.-]+", "-", engine.lower()).strip("-.") or "engine")[:40]


def run_argv(name: str, image: str, inner_cmd: list[str], port: int, mounts: list[tuple[str, str]], env: dict,
             gpu: bool) -> list[str]:
    """`podman run` for one engine: detached, replaced if a container of that name is left, read-only mounts, no
    capabilities, no new privileges, its port on 127.0.0.1 only."""
    argv = ["podman", "run", "-d", "--replace", "--name", container_name(name), "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--pids-limit=4096",
            "--network", "pasta",                         # rootless Podman 4.9 wants slirp4netns otherwise; Setup installs passt
            "-p", f"127.0.0.1:{int(port)}:{int(port)}",
            "--label", "ai.nebula.engine=" + name]
    if gpu:
        argv += ["--device", "nvidia.com/gpu=all"]
    for host, inside in mounts:
        argv += ["-v", f"{to_wsl(host)}:{inside}:ro"]
    for k, v in sorted((env or {}).items()):
        argv += ["-e", f"{k}={v}"]
    return argv + [image] + list(inner_cmd)


class WslEngine:
    """An engine as a Podman container in the NebulaEH distro, with the same face as EngineProcess and
    openshell.SandboxedEngine (start, stop, wait_ready, state, status), so nebula/engines.py runs it like those."""

    def __init__(self, spec: dict, inner_cmd: list[str], mounts: list[tuple[str, str]], image: str, env: dict | None = None,
                 gpu: bool = True, ready_timeout_s: float = 900.0, runner=None):
        self.spec, self.inner_cmd, self.mounts, self.image = spec, list(inner_cmd), list(mounts), image
        self.env, self.gpu, self.ready_timeout_s, self.runner = dict(env or {}), gpu, ready_timeout_s, runner
        self.name = container_name(spec["name"])
        self.address: str | None = None
        self.state, self.error, self.proc = "idle", None, None
        self.lines: list[str] = []
        self.started_at = self.ready_at = None

    def _say(self, line: str) -> None:
        self.lines = (self.lines + [line])[-400:]

    def start(self) -> None:
        self.state, self.started_at = "starting", time.time()
        argv = run_argv(self.spec["name"], self.image, self.inner_cmd, int(self.spec["port"]), self.mounts, self.env, self.gpu)
        self._say("[nebula] wsl -d " + DISTRO + " " + " ".join(argv))
        try:
            r = run(argv, timeout=600, runner=self.runner)
        except (OSError, subprocess.TimeoutExpired, WslError) as e:
            self.state, self.error = "failed", str(e)[:400]
            self._say(f"[nebula] {self.error}")
            return
        if r.returncode != 0:
            self.state, self.error = "failed", "its container did not start: " + (r.stderr or r.stdout or "").strip()[-300:]
            self._say(f"[nebula] {self.error}")
            return
        self.address = f"http://127.0.0.1:{int(self.spec['port'])}"
        self._say(f"[nebula] {self.name} started; it answers at {self.address} when loaded")
        self.state = "loading"

    def exited(self) -> str | None:
        """Why the container stopped, or None while it runs."""
        try:
            r = run(["podman", "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", self.name], timeout=30, runner=self.runner)
        except (OSError, subprocess.TimeoutExpired, WslError):
            return None
        out = (r.stdout or "").split()
        if r.returncode != 0 or not out:
            return "its container is gone"
        if out[0] in ("exited", "stopped", "dead"):
            tail = ""
            try:
                lg = run(["podman", "logs", "--tail", "5", self.name], timeout=30, runner=self.runner)
                tail = ((lg.stdout or "") + (lg.stderr or "")).strip().splitlines()[-1:] or [""]
                tail = ": " + tail[0][:200] if tail[0] else ""
            except (OSError, subprocess.TimeoutExpired, WslError):
                pass
            return f"it exited ({out[1] if len(out) > 1 else '?'}){tail}"
        return None

    def wait_ready(self, poll_s: float = 0.5) -> bool:
        from nebula.app.engine import health
        if self.state != "loading" or not self.address:
            return False
        end, look = time.monotonic() + self.ready_timeout_s, time.monotonic() + 10
        while time.monotonic() < end:
            if self.state in ("stopping", "stopped"):
                return False
            if time.monotonic() >= look:
                look = time.monotonic() + 10
                why = self.exited()
                if why:
                    self.state, self.error = "failed", f"{self.name}: {why} while loading"
                    self._say(f"[nebula] {self.error}")
                    return False
            if health(self.address, timeout=1.0) == 200:
                self.state, self.ready_at = "ready", time.time()
                self._say("[nebula] the engine in its container answers")
                return True
            time.sleep(poll_s)
        self.state, self.error = "failed", f"the engine in {self.name} did not answer within {self.ready_timeout_s:.0f} s"
        return False

    def stop(self) -> None:
        self.state = "stopping"
        try:
            run(["podman", "rm", "-f", "-t", "10", self.name], timeout=120, runner=self.runner)
        except (OSError, subprocess.TimeoutExpired, WslError) as e:
            self._say(f"[nebula] {e}")
        self.state, self.address = "stopped", None

    def status(self) -> dict:
        return {"state": self.state, "lines": list(self.lines[-60:]), "pid": None, "error": self.error,
                "seconds": None, "sandbox": self.name, "address": self.address, "where": "wsl2-podman"}
