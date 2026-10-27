"""nebula/machines.py: engines on a model machine (one added with Add a server, nebula/sshinstall.py), run by this
Nebula like its own.

A machine's cards appear under Engines beside this computer's (`machine` names it). An engine deployed onto one
(`"machine": "<its name>"`) always runs in an OpenShell sandbox there, on rootless Podman, started through that
machine's gateway (registered here, reached over SSH): its image is built here from the engine's program and loaded
there once, the model is a file on that machine (its folder mounted read-only), and its address there goes to the
machine's node keeper, which PAIR's requests for that model reach. This Nebula routes the model through its own
router (the household network, mutual TLS): `http://127.0.0.1:<router port>` with the engine's name as the model.

Nothing on the machine decides anything: which engine runs, on which card, who may use it, is this Nebula's.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.parse

from nebula import openshell

_CARDS: dict[str, tuple[float, list]] = {}
_CARDS_LOCK = threading.Lock()


def row(cfg, name: str) -> dict:
    r = next((s for s in (cfg.raw or {}).get("servers") or [] if s.get("name") == name), None)
    if r is None:
        raise KeyError(name)
    return r


def target(cfg, r: dict):
    from nebula import names, sshinstall
    kh = os.path.join(names.local_path(cfg.base_dir, "nebula-keys", "cockpit-keys"), "known_hosts")
    return sshinstall.Target(r["host"], r["user"], int(r.get("port") or 22), r.get("key_file") or None, kh)


def cards(cfg, max_age_s: float = 10.0) -> list[dict]:
    """Every model machine's cards, as engines.cards() gives this computer's, with `machine` naming where."""
    from nebula import sshinstall
    out = []
    for r in (cfg.raw or {}).get("servers") or []:
        if not r.get("gateway"):                       # one of the older headless servers: not a model machine
            continue
        with _CARDS_LOCK:
            at, have = _CARDS.get(r["name"], (0.0, []))
        if time.time() - at > max_age_s:
            try:
                have = [{"uuid": c["uuid"], "index": c["index"], "name": c["name"], "memory_mb": c["memory_total_mib"],
                         "used_mb": c["memory_used_mib"], "compute_cap": c.get("compute_cap"), "machine": r["name"]}
                        for c in sshinstall.gpus(target(cfg, r))]
            except sshinstall.SshError:
                have = [dict(c, unreachable=True) for c in have]
            with _CARDS_LOCK:
                _CARDS[r["name"]] = (time.time(), have)
        out += have
    return out


_TUNNELS: dict[str, subprocess.Popen] = {}
_TUNNEL_LOCK = threading.Lock()


def free_port(taken: list[int], first: int = 17680) -> int:
    import socket
    for port in range(first, first + 200):
        if port in taken:
            continue
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            s.close()
    raise OSError("no free local port for a gateway tunnel")


def _open(port: int) -> bool:
    import socket
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        return True
    except OSError:
        return False


def tunnel(cfg, r: dict) -> int:
    """This machine's SSH tunnel to the machine's OpenShell gateway (it listens on that machine's loopback only):
    127.0.0.1:<gateway_port> here, kept while Nebula runs and opened again when it has gone."""
    from nebula import sshinstall
    port = int(r.get("gateway_port") or 0)
    if not port:
        raise openshell.OpenShellError(f"{r['name']}'s gateway has no tunnel port: add the machine again")
    with _TUNNEL_LOCK:
        p = _TUNNELS.get(r["name"])
        if p is not None and p.poll() is None and _open(port):
            return port
        if p is not None and p.poll() is None:
            p.terminate()
        argv = target(cfg, r).argv("")[:-1]           # ssh and its options, the destination; no command
        argv[1:1] = ["-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                     "-L", f"127.0.0.1:{port}:127.0.0.1:{sshinstall.GATEWAY_PORT}"]
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _TUNNELS[r["name"]] = p
        for _ in range(60):
            if _open(port):
                return port
            if p.poll() is not None:
                raise openshell.OpenShellError(f"the tunnel to {r['name']}'s gateway did not open: "
                                               f"{p.stderr.read().decode(errors='replace').strip()[-300:]}")
            time.sleep(0.25)
        raise openshell.OpenShellError(f"the tunnel to {r['name']}'s gateway did not open in 15 s")


def close_tunnel(name: str) -> None:
    with _TUNNEL_LOCK:
        p = _TUNNELS.pop(name, None)
    if p is not None and p.poll() is None:
        p.terminate()


def cuda_archs(binary: str) -> set[int]:
    """The card generations a CUDA program carries kernels for (sm_86 -> 86, sm_120a -> 120), read from the file."""
    import re
    found: set[int] = set()
    tail = b""
    with open(openshell.engine_program(binary), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            buf = tail + chunk
            found.update(int(m) for m in re.findall(rb"\bsm_(\d{2,3})a?\b", buf))
            tail = buf[-16:]
    return found


def arch_gap(binary: str, cards: list[dict]) -> str | None:
    """Why this build cannot run on these cards (none of its kernels is for their generation), or None."""
    archs = cuda_archs(binary)
    if not archs:
        return None                                       # not a CUDA build we can read: let it try
    for c in cards:
        cc = str(c.get("compute_cap") or "")
        if not re.match(r"^\d+\.\d+$", cc):
            continue
        need = int(cc.replace(".", ""))
        if need not in archs:
            return (f"its Impulse build has no kernels for {c.get('name')} (sm_{need}); it carries "
                    f"{', '.join('sm_%d' % a for a in sorted(archs))}: use a build that includes sm_{need}")
    return None


def router_url(cfg) -> str:
    return f"http://127.0.0.1:{int(((cfg.raw or {}).get('network') or {}).get('router_port') or 8080)}"


class MachineEngine(openshell.SandboxedEngine):
    """An engine in a sandbox on a model machine, with SandboxedEngine's face (start, wait_ready, stop, status)."""

    def __init__(self, spec: dict, inner_cmd: list[str], cfg, network, policy_dir: str, models_root: str, card_index,
                 build_from: str | None, image: str | None, ready_timeout_s: float = 900.0, env: dict | None = None):
        super().__init__(spec, inner_cmd, models_root, policy_dir, card_index,
                         ready_timeout_s=ready_timeout_s, env=env, build_from=build_from)
        self.cfg, self.network = cfg, network
        self.r = row(cfg, spec["machine"])
        self.t = target(cfg, self.r)
        self.gateway = self.r["gateway"]
        self.image = image
        self.there: str | None = None                    # its address on that machine (the gateway's)

    def _ssh(self, cmd: str, timeout: float = 120, stdin: bytes | None = None) -> str:
        return self.t.run(cmd, stdin=stdin, timeout=timeout)

    def _volume(self) -> str:
        """The model's folder there as a read-only Podman volume OpenShell may attach (its approval labels)."""
        folder = self.models_dir
        vol = openshell.volume_name(folder)
        argv = ["podman"] + openshell.volume_argv(folder, vol)[1:]
        self._ssh(f"podman volume exists {shlex.quote(vol)} || {openshell_quote(argv)}", timeout=60)
        return vol

    def _image_there(self) -> str:
        """The engine's image on that machine: built here (Docker) from its program, loaded there once."""
        import shutil
        docker = os.environ.get("NEBULA_DOCKER") or shutil.which("docker")
        if self.image:
            tag = self.image
        else:
            if not docker:
                raise openshell.OpenShellError("its image is built on this computer from its program, which needs Docker here")
            tag = openshell.engine_image(self.build_from, docker, say=self._say)
        try:
            self._ssh(f"podman image exists {shlex.quote(tag)}", timeout=60)
            return self._sandbox_image_there(tag)
        except openshell.OpenShellError:
            raise
        except Exception:  # noqa: BLE001 (not there yet)
            pass
        if "/" in tag.split(":", 1)[0]:                    # a registry's image (llama.cpp's own): pulled there
            self._say(f"[nebula] pulling {tag} on {self.r['name']} (once)")
            try:
                self._ssh(f"podman pull -q {shlex.quote(tag)}", timeout=3600)
                return self._sandbox_image_there(tag)
            except Exception as e:  # noqa: BLE001 - no registry there: sent from here below
                self._say(f"[nebula] the pull there did not work ({str(e)[:120]}); sending it from here")
        if not docker:
            raise openshell.OpenShellError(f"{tag} is not on {self.r['name']} and there is no Docker here to send it")
        self._say(f"[nebula] sending its image {tag} to {self.r['name']} (once for this build)")
        save = subprocess.Popen([docker, "save", tag], stdout=subprocess.PIPE)
        p = subprocess.run(self.t.argv("podman load -q"), stdin=save.stdout, capture_output=True, timeout=3600)
        save.wait(timeout=60)
        if p.returncode != 0:
            raise openshell.OpenShellError(f"its image could not be loaded on {self.r['name']}: "
                                           f"{p.stderr.decode(errors='replace').strip()[-300:]}")
        return tag

    def _device_nodes(self) -> list[str]:
        """The NVIDIA device nodes of its cards there, from the CDI description Podman reads there."""
        if not self.card_index:
            return []
        try:
            text = self._ssh("cat /etc/cdi/nvidia-podman.json 2>/dev/null || nvidia-ctk cdi generate --format=json 2>/dev/null", timeout=60)
        except Exception:  # noqa: BLE001
            return []
        return openshell.device_nodes(text, self.card_index)

    def _sandbox_image_there(self, tag: str) -> str:
        """As openshell.sandbox_image, there: an image whose working folder the sandbox's user cannot write (llama.cpp's
        /app) gets a thin copy working in /sandbox, with its program's folder on the library path."""
        wd = self._ssh(f"podman image inspect {shlex.quote(tag)} --format '{{{{.Config.WorkingDir}}}}'", timeout=60).strip()
        if wd in ("", "/", "/sandbox"):
            return tag
        thin = tag.replace(":", "-").replace("/", "-") + ":nebula-sandbox-2"
        prog = next((w for w in self.inner_cmd if w.startswith("/")), None)
        lib = os.path.dirname(prog) if prog else wd
        lines = f"FROM {tag}\nWORKDIR /sandbox\nENV LD_LIBRARY_PATH={lib}\n"
        self._ssh(f"podman image exists {shlex.quote(thin)} || printf {shlex.quote(lines)} | podman build -q -t {shlex.quote(thin)} -f - /tmp",
                  timeout=600)
        return thin

    def start(self) -> None:
        self.state, self.started_at = "starting", time.time()
        try:
            tunnel(self.cfg, self.r)
            vol = self._volume() if self.models_dir else None   # a container engine may need no folder
            os.makedirs(self.policy_dir, exist_ok=True)
            pol = os.path.join(self.policy_dir, f"{self.name}.yaml")
            with open(pol, "w", encoding="utf-8") as f:
                f.write(openshell.policy_text(None, self._device_nodes()))
            argv = openshell.deploy_argv(self.spec, self.inner_cmd, self.card_index, vol, pol, [], self.env,
                                         memory=f"{openshell.ram_gib(self._ssh('cat /proc/meminfo', timeout=30))}Gi",
                                         cpus=int(self._ssh("nproc", timeout=30).strip() or 2),
                                         nodes=openshell.node_cpus(self._ssh(openshell.NODE_CPUS_CMD, timeout=30)))
            if vol is None:                                 # no folder: no mount at all
                i = argv.index("--driver-config-json") + 1
                dc = json.loads(argv[i])
                dc["docker"]["mounts"] = []
                argv[i] = json.dumps(dc, separators=(",", ":"))
            i = argv.index("--driver-config-json") + 1
            argv[i] = json.dumps({"podman": json.loads(argv[i])["docker"]}, separators=(",", ":"))
            argv[argv.index("--from") + 1] = img = self._image_there()
            # Podman re-owns a named volume's folder to the first container that mounts it (a sandbox runs as
            # "nobody", which can lock the owner out of their own models folder): a first mount as root settles it
            if vol:
                self._ssh(f"podman run --rm --network none -v {shlex.quote(vol)}:/m:ro --entrypoint /bin/true {shlex.quote(img)}", timeout=300)
            argv.insert(argv.index("--"), "--detach")
            self._say(f"[nebula] on {self.r['name']}: openshell " + " ".join(argv))
            try:
                openshell.run(argv, self.exe, timeout=900, gateway=self.gateway)
            except openshell.OpenShellError as e:          # a sandbox left from an earlier try (failed, or Nebula gone)
                if "already exists" not in str(e):
                    raise
                self._say(f"[nebula] an earlier sandbox {self.name} is still there: deleted, then made again")
                openshell.delete(self.name, self.exe, gateway=self.gateway)
                time.sleep(3)
                openshell.run(argv, self.exe, timeout=900, gateway=self.gateway)
            there = openshell.expose(self.name, int(self.spec["port"]), self.exe, gateway=self.gateway)
            from nebula import sshinstall                # its name says this side's tunnel port; there it is the gateway's own
            u = urllib.parse.urlsplit(there)
            self.there = urllib.parse.urlunsplit((u.scheme, f"{u.hostname}:{sshinstall.GATEWAY_PORT}", u.path, "", "")).rstrip("/")
            self.address = router_url(self.cfg)
            self._say(f"[nebula] on {self.r['name']} at {self.there}; reached here through your network as {self.spec['name']}")
            self.state = "loading"
        except Exception as e:  # noqa: BLE001
            self.state, self.error = "failed", str(e)[:400]
            self._say(f"[nebula] {self.error}")

    def wait_ready(self, poll_s: float = 2.0) -> bool:
        if self.state != "loading" or not self.there:
            return False
        end = time.monotonic() + self.ready_timeout_s
        try:
            node = self.network.node(self.t)
        except Exception as e:  # noqa: BLE001
            self.state, self.error = "failed", f"{self.r['name']}'s node keeper: {e}"
            return False
        look = time.monotonic() + 10
        try:
            while time.monotonic() < end:
                if self.state in ("stopping", "stopped"):
                    return False
                if time.monotonic() >= look:               # a program that died while loading: said within seconds
                    look = time.monotonic() + 10
                    why = openshell.exited(self.name, self.exe, gateway=self.gateway)
                    if why:
                        self.state, self.error = "failed", f"{self.name} on {self.r['name']}: {why} while loading"
                        self._say(f"[nebula] {self.error}")
                        return False
                if node.call("keeper:health", {"url": self.there}, timeout=30).get("status") == 200:
                    node.call("keeper:set-engine", {"model": self.spec["name"], "url": self.there}, timeout=30)
                    self._say("[nebula] waiting for your network to list it (about a minute)")
                    listed = time.monotonic() + 180
                    while time.monotonic() < listed and not self._listed():
                        time.sleep(3)
                    self.state, self.ready_at = "ready", time.time()
                    self._say(f"[nebula] the engine on {self.r['name']} answers; its model is on your network")
                    return True
                time.sleep(poll_s)
        except Exception as e:  # noqa: BLE001
            self.state, self.error = "failed", f"{self.r['name']}: {e}"[:400]
            return False
        finally:
            node.stop()
        self.state, self.error = "failed", f"the engine on {self.r['name']} did not answer within {self.ready_timeout_s:.0f} s"
        return False

    def _listed(self) -> bool:
        """This machine's router sends a request for the model there (PAIR learns a machine's models on its own poll;
        its /v1/models is one machine's answer, not its routing view): a one-token call."""
        import urllib.error
        import urllib.request
        body = json.dumps({"model": self.spec["name"], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
        req = urllib.request.Request(router_url(self.cfg) + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status == 200
        except urllib.error.HTTPError as e:
            return "advertises" not in e.read().decode(errors="replace")
        except (OSError, ValueError):
            return False

    def stop(self) -> None:
        self.state = "stopping"
        try:
            self.network.node_call(self.t, "keeper:set-engine", {"model": self.spec["name"], "url": None})
        except Exception as e:  # noqa: BLE001
            self._say(f"[nebula] {e}")
        try:
            tunnel(self.cfg, self.r)
            openshell.delete(self.name, self.exe, gateway=self.gateway)
        except (openshell.OpenShellError, OSError) as e:
            self._say(f"[nebula] {e}")
        self.state, self.address, self.there = "stopped", None, None

    def status(self) -> dict:
        return dict(super().status(), machine=self.r["name"], there=self.there)


def openshell_quote(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)
