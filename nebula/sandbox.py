"""nebula/sandbox.py: a command tier's sandbox, run through NVIDIA's OpenShell (docs/DESIGN.md, step O1).

OpenShell puts a process in a container (Docker, Podman, Kubernetes or a MicroVM) whose only way out is a
supervisor that checks every connection against a policy and holds the credentials. Nebula uses it through its CLI,
nothing else: one sandbox per command tier, created once and kept (`openshell sandbox create --name nebula-<tier>
--from <image> --detach ... -- sleep infinity`), and every call an `openshell sandbox exec -n nebula-<tier> --
<the CLI's argv>` with the prompt on its stdin (OpenShell caps piped stdin at 4 MiB), the answer from its stdout
or from a file read back with another exec. The CLI's own account key, when it is to have one, is an OpenShell
provider attached to the sandbox, substituted by the supervisor at the vendor's host only; the sandbox never holds
it. With `model_via_nebula`, the sandbox's `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL` point back at this Nebula on
its LAN address with a token minted for the sandbox (step O3), so the CLI's own model calls are this Nebula's
attempts.

`openshell` missing, or the gateway off, makes the tier down with the reason; nothing falls back to running the
CLI bare on the host.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time

from nebula.engine import EngineError

STDIN_CAP = 4 * 1024 * 1024
KEEPALIVE = ["sleep", "infinity"]
IMAGES = {                                    # a default image per CLI; the owner may name another
    "claude": "ghcr.io/nvidia/openshell/claude-code:latest",
    "codex": "ghcr.io/nvidia/openshell/codex:latest",
    "grok": "ubuntu:24.04",
}


VENDOR_HOSTS = {                              # where each CLI's own account lives: the only hosts a tier gets by default
    "claude": ["api.anthropic.com"],
    "codex": ["api.openai.com", "chatgpt.com", "auth.openai.com"],
    "grok": ["api.x.ai"],
}
BINARIES = {                                  # the programs inside the image that make the connections
    "claude": ["/usr/local/bin/node", "/usr/bin/node", "/usr/local/bin/claude", "/usr/bin/claude"],
    "codex": ["/usr/local/bin/codex", "/usr/bin/codex", "/usr/local/bin/node", "/usr/bin/node"],
    "grok": ["/usr/local/bin/grok", "/usr/bin/grok", "/sandbox/.local/bin/grok"],
}
COMMON_BINARIES = ["/usr/bin/bash", "/bin/bash", "/usr/bin/curl", "/usr/bin/python3", "/usr/local/bin/python3"]
FILESYSTEM_DEFAULT = (                        # OpenShell 0.1.4's default, as `openshell policy get <sandbox> --base` prints it
    "filesystem_policy:\n  include_workdir: true\n  read_only:\n    - /bin\n    - /usr\n    - /lib\n    - /proc\n"
    "    - /dev/urandom\n    - /etc\n    - /var/log\n  read_write:\n    - /tmp\n    - /dev/null\nlandlock:\n  compatibility: best_effort\n")


def sandbox_name(tier_name: str) -> str:
    from nebula.openshell import short_name                 # within OpenShell's 19 characters
    return short_name("nebula-", tier_name, "tier")


def parse_host(item) -> dict | None:
    """An owner's host entry: "host", "host:port", "host:port:read-only", or {"host", "port", "access", "path"}."""
    if isinstance(item, dict):
        if not item.get("host"):
            return None
        return {"host": str(item["host"]), "port": int(item.get("port") or 443),
                "access": str(item.get("access") or "read-only"), "path": item.get("path")}
    parts = str(item).strip().split(":")
    if not parts or not parts[0]:
        return None
    port = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 443
    access = parts[2] if len(parts) > 2 else "read-only"
    return {"host": parts[0], "port": port, "access": access, "path": None}


def policy_yaml(cli: str | None, hosts: list, binaries: list[str], nebula: tuple[str, int] | None) -> str:
    """The sandbox's network policy (OpenShell's schema, version 1): the CLI's vendor hosts, this Nebula when the
    sandbox takes its model from it, and what the owner ticked; nothing else. Filesystem and process stay the
    defaults."""
    def endpoint(host, port, access, path=None, protocol="rest"):
        lines = [f"      - host: {host}", f"        port: {int(port)}", f"        protocol: {protocol}",
                 f"        access: {access}", "        enforcement: enforce"]
        if path:
            lines.append(f"        path: {path}")
        return "\n".join(lines)

    def rule(name, endpoints):
        bins = "\n".join(f"      - path: {b}" for b in binaries)
        return f"  {name}:\n    endpoints:\n" + "\n".join(endpoints) + f"\n    binaries:\n{bins}"

    rules = []
    if cli in VENDOR_HOSTS:
        rules.append(rule("vendor", [endpoint(h, 443, "read-write") for h in VENDOR_HOSTS[cli]]))
    if nebula:
        rules.append(rule("nebula", [endpoint(nebula[0], nebula[1], "read-write")]))
    owner = [h for h in (parse_host(x) for x in hosts or []) if h]
    if owner:
        rules.append(rule("owner", [endpoint(h["host"], h["port"], h["access"], h["path"]) for h in owner]))
    # OpenShell 0.1.4 drops its default filesystem rules when a policy file has no network rule (the sandbox then
    # cannot even start its entrypoint), so the defaults are written here as `openshell policy get --base` shows them
    head = ("# written by Nebula for this command tier; edit the tier's hosts on the page, not this file\n"
            "version: 1\n" + FILESYSTEM_DEFAULT)
    if not rules:
        return head
    return head + "network_policies:\n" + "\n".join(rules) + "\n"


class Sandbox:
    """One command tier's sandbox: `spec` is the tier's `sandbox` object ({"image", "env", "providers", "policy",
    "openshell", "model_via_nebula"})."""

    def __init__(self, tier_name: str, spec: dict, base_dir: str | None = None, timeout_s: float = 600.0):
        self.name = sandbox_name(tier_name)
        self.spec = dict(spec or {})
        self.image = self.spec.get("image")
        self.base_dir = base_dir
        self.timeout_s = timeout_s
        self.openshell = self.spec.get("openshell") or "openshell"
        self.lock = threading.Lock()
        self.ready = False
        self.last: dict = {}
        self.nebula_env: dict[str, str] = {}      # set by the server for model_via_nebula (step O3)
        self.nebula_endpoint: tuple[str, int] | None = None
        self.cli = self.spec.get("cli")
        self.policy_written: str | None = None
        self.asks_seen: dict[str, dict] = {}      # host:port:binary -> the denial, accumulated from the sandbox's log

    # --- the CLI
    def binary(self) -> str | None:
        b = self.openshell
        return shutil.which(b) or (b if os.path.isabs(b) and os.access(b, os.X_OK) else None)

    def _cli(self, args: list[str], stdin: bytes | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
        b = self.binary()
        if b is None:
            raise EngineError(503, {"error": {"message": "openshell is not installed on this machine", "type": "server_error"}})
        argv = [b] + args
        try:
            return subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout or self.timeout_s, check=False,
                                  env=dict(os.environ, NO_COLOR="1", TERM="dumb"))
        except subprocess.TimeoutExpired:
            raise EngineError(504, {"error": {"message": f"openshell {' '.join(args[:3])} did not finish in {int(timeout or self.timeout_s)} s",
                                              "type": "server_error"}}) from None
        except OSError as e:
            raise EngineError(502, {"error": {"message": f"openshell could not run: {e}", "type": "server_error"}}) from None

    # --- the policy (step O2)
    def policy_path(self) -> str | None:
        """The policy file the sandbox runs under: the owner's own (`policy`), else one written from the tier's CLI,
        this Nebula and the owner's `hosts`, under cockpit-sandbox beside the config."""
        if self.spec.get("policy"):
            path = self.spec["policy"]
            return path if os.path.isabs(path) or not self.base_dir else os.path.join(self.base_dir, path)
        text = policy_yaml(self.cli, self.spec.get("hosts") or [], self.binaries(), self.nebula_endpoint)
        d = os.path.join(self.base_dir or os.getcwd(), "cockpit-sandbox")
        os.makedirs(d, mode=0o700, exist_ok=True)
        path = os.path.join(d, f"{self.name}.policy.yaml")
        try:
            with open(path, encoding="utf-8") as f:
                same = f.read() == text
        except OSError:
            same = False
        if not same:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        self.policy_written = path
        return path

    def binaries(self) -> list[str]:
        extra = [str(b) for b in self.spec.get("binaries") or []]
        return list(dict.fromkeys(BINARIES.get(self.cli or "", []) + COMMON_BINARIES + extra))

    def apply_policy(self) -> dict:
        """Write the policy again and set it on the running sandbox (OpenShell reloads network rules live)."""
        path = self.policy_path()
        st = self.status()
        if not st["exists"]:
            return {"policy": path, "applied": False, "detail": "no sandbox yet: applied at its creation"}
        p = self._cli(["policy", "set", self.name, "--policy", path, "--wait"], timeout=90)
        return {"policy": path, "applied": p.returncode == 0, "detail": (p.stderr or p.stdout).decode("utf-8", "replace").strip()[-300:]}

    def hosts(self) -> list[dict]:
        return [h for h in (parse_host(x) for x in self.spec.get("hosts") or []) if h]

    # --- the asks (step O2): what the sandbox tried to reach and was refused
    DENIED = re.compile(r"\[(\d+(?:\.\d+)?)\].*?DENIED\s+(\S+?)\((\d+)\)\s+->\s+([^:\s]+):(\d+)\s+\[reason:([^\]]+)\]")

    def read_denials(self) -> list[dict]:
        """The denials in `openshell logs <sandbox>` (OpenShell keeps only its last lines, so they are accumulated
        here across reads), newest last."""
        if self.binary() is None or not self.status()["exists"]:
            return []
        p = self._cli(["logs", self.name], timeout=30)
        for m in self.DENIED.finditer(p.stdout.decode("utf-8", "replace")):
            at, binary, _pid, host, port, reason = m.groups()
            key = f"{host}:{port}:{binary}"
            seen = self.asks_seen.get(key)
            when = float(at)
            if seen is None:
                self.asks_seen[key] = {"key": key, "host": host, "port": int(port), "binary": binary, "reason": reason,
                                       "first": when, "last": when, "count": 1}
            elif when > seen["last"]:
                seen["last"], seen["count"] = when, seen["count"] + 1
        return sorted(self.asks_seen.values(), key=lambda a: a["last"])

    def asks(self) -> list[dict]:
        """The denials not yet answered: not an allowed host of the policy, not rejected by the owner."""
        rejected = {str(r) for r in self.spec.get("rejected") or []}
        allowed = {(h["host"], h["port"]) for h in self.hosts()}
        allowed |= {(h, 443) for h in VENDOR_HOSTS.get(self.cli or "", [])}
        if self.nebula_endpoint:
            allowed.add((self.nebula_endpoint[0], int(self.nebula_endpoint[1])))
        out = []
        for a in self.read_denials():
            if (a["host"], a["port"]) in allowed or f"{a['host']}:{a['port']}" in rejected:
                continue
            out.append(a)
        return out

    def stop(self) -> bool:
        p = self._cli(["sandbox", "stop", self.name], timeout=120)
        self.ready = False
        return p.returncode == 0

    # --- the sandbox's life
    def status(self) -> dict:
        """{"installed", "exists", "running", "detail"}: what `openshell sandbox get` says, without creating anything."""
        if self.binary() is None:
            return {"installed": False, "exists": False, "running": False, "detail": "openshell is not installed"}
        p = self._cli(["sandbox", "get", self.name, "--output", "json"], timeout=30)
        if p.returncode != 0:
            return {"installed": True, "exists": False, "running": False, "detail": (p.stderr or p.stdout).decode("utf-8", "replace").strip()[-300:]}
        text = p.stdout.decode("utf-8", "replace")
        state = None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):                    # OpenShell 0.1.4 prints "phase": "Ready" for a running sandbox
                state = obj.get("phase") or obj.get("status") or obj.get("state") or (obj.get("sandbox") or {}).get("phase")
                if isinstance(state, dict):
                    state = state.get("phase") or state.get("state")
        except ValueError:
            pass
        running = str(state).lower() in ("ready", "running") if state is not None else ("ready" in text.lower() or "running" in text.lower())
        return {"installed": True, "exists": True, "running": bool(running), "detail": str(state or "")[:80]}

    def ensure(self) -> None:
        """Create the sandbox when it is not there (kept alive by a sleep as its main process); start it when stopped."""
        with self.lock:
            if self.ready:
                return
            st = self.status()
            if not st["installed"]:
                raise EngineError(503, {"error": {"message": "openshell is not installed on this machine", "type": "server_error"}})
            if not st["exists"]:
                if not self.image:
                    raise EngineError(503, {"error": {"message": f"the sandbox {self.name} needs an image", "type": "server_error"}})
                args = ["sandbox", "create", "--name", self.name, "--from", self.image, "--detach"]
                for k, v in {**(self.spec.get("env") or {}), **self.nebula_env}.items():
                    args += ["--env", f"{k}={v}"]
                for prov in self.spec.get("providers") or []:
                    args += ["--provider", str(prov)]
                path = self.policy_path()
                if path:
                    args += ["--policy", path]
                args += ["--"] + KEEPALIVE
                p = self._cli(args, timeout=max(120.0, self.timeout_s))
                self.last = {"create": p.returncode, "stderr": p.stderr.decode("utf-8", "replace")[-400:]}
                if p.returncode != 0:
                    raise EngineError(502, {"error": {"message": f"openshell could not create {self.name}: {(p.stderr or p.stdout).decode('utf-8', 'replace').strip()[-400:]}",
                                                      "type": "server_error"}})
            elif not st["running"]:
                p = self._cli(["sandbox", "start", self.name], timeout=120)
                if p.returncode != 0:
                    raise EngineError(502, {"error": {"message": f"openshell could not start {self.name}: {(p.stderr or p.stdout).decode('utf-8', 'replace').strip()[-400:]}",
                                                      "type": "server_error"}})
            self.ready = True

    def exec(self, argv: list[str], stdin: bytes | None = None, env: dict | None = None, timeout: float | None = None) -> tuple[int, bytes, bytes]:
        """One command inside the sandbox: (exit code, stdout, stderr)."""
        if stdin is not None and len(stdin) > STDIN_CAP:
            raise EngineError(413, {"error": {"message": f"the prompt is over OpenShell's 4 MiB stdin cap ({len(stdin)} bytes)", "type": "invalid_request_error"}})
        args = ["sandbox", "exec", "-n", self.name]
        for k, v in (env or {}).items():
            args += ["--env", f"{k}={v}"]
        p = self._cli(args + ["--"] + list(argv), stdin=stdin, timeout=timeout)
        return p.returncode, p.stdout, p.stderr

    def run(self, argv: list[str], prompt: str, on_stdin: bool, how: str, prompt_file: str, output_file: str,
            timeout: float | None = None) -> tuple[int, str, str]:
        """The CLI's call as the engine shapes it: the prompt put where the form wants it, the answer read back."""
        self.ensure()
        work = os.path.dirname(prompt_file) or "/tmp"
        data = prompt.encode("utf-8")
        started = time.monotonic()
        try:
            if not on_stdin:
                rc, _o, err = self.exec(["sh", "-c", f"mkdir -p {work} && cat > {prompt_file}"], stdin=data, timeout=60)
                if rc != 0:
                    raise EngineError(502, {"error": {"message": f"the sandbox could not take the prompt: {err.decode('utf-8', 'replace')[-300:]}", "type": "server_error"}})
            elif how == "file":
                self.exec(["sh", "-c", f"mkdir -p {work}"], timeout=60)
            rc, out, err = self.exec(argv, stdin=data if on_stdin else None, timeout=timeout)
            text = out.decode("utf-8", "replace")
            if rc == 0 and how == "file":
                rc2, out2, _e2 = self.exec(["cat", output_file], timeout=60)
                text = out2.decode("utf-8", "replace") if rc2 == 0 else text
            return rc, text, err.decode("utf-8", "replace")
        finally:
            self.last = dict(self.last, seconds=round(time.monotonic() - started, 2))
            try:
                self.exec(["rm", "-rf", work], timeout=30)
            except EngineError:
                pass

    def delete(self) -> bool:
        p = self._cli(["sandbox", "delete", self.name], timeout=120)
        self.ready = False
        return p.returncode == 0

    def describe(self) -> dict:
        return {"name": self.name, "image": self.image, "providers": list(self.spec.get("providers") or []),
                "policy": self.spec.get("policy") or self.policy_written, "own_policy": bool(self.spec.get("policy")),
                "vendor_hosts": VENDOR_HOSTS.get(self.cli or "", []), "hosts": self.hosts(),
                "model_via_nebula": bool(self.spec.get("model_via_nebula")), "rejected": list(self.spec.get("rejected") or []),
                "nebula": list(self.nebula_endpoint) if self.nebula_endpoint else None, **self.status()}
