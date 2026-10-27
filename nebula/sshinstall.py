"""nebula/sshinstall.py: Add a server (SSH): make another machine on your network a model machine of this Nebula.

    scan(host, port)           -> its host keys' fingerprints, to show and confirm before anything else
    trust(host, port, fp)      -> the confirmed key kept in Nebula's own known_hosts (nebula-keys/known_hosts)
    preflight(target)          -> OS, architecture, GPUs, free disk, Python, Podman, its firewall; and what is missing
    install(target, …)         -> the node pieces copied and started as the user's own services (below); no Nebula
    uninstall(target)          -> stop and remove them all
    admin_needs(report, …)     -> what needs an administrator there or here (Podman, the firewall, …), as fixed tasks
    run_admin(target, …)       -> those tasks with sudo, the owner's password given once on stdin and never kept

No Nebula is installed on a model machine (design decision 2026-10-21: "this isn't supposed to deploy nebula. Only the
models... it is supposed to be part of YOUR nebula"). It gets only:

- the Personal AI Router's node programs (internal: PAIR), which carry this Nebula's requests to it over mutual TLS;
- nebula/nodekeeper.py, one file: it holds PAIR's broker open, relays this Nebula's calls to it over SSH, and is the
  machine's one engine address for PAIR, passing each request to the engine serving its model;
- NVIDIA OpenShell for this user, its gateway on rootless Podman (Podman itself is the administrator's step), where
  this Nebula starts its engines on that machine's cards (`openshell` here, the gateway there, over SSH).

Your SSH settings are used as `ssh` uses them: a name from ~/.ssh/config (its address, port, key) works, and an address
that one of those names points at gets that name's key, so nothing needs setting up first. SSH is the trusted
bootstrap and the management link: the host key must match the confirmed one (a mismatch is refused), auth is the
user's own SSH agent or key file (never a stored password).
"""
from __future__ import annotations

import io
import os
import re
import shlex
import subprocess
import tarfile

NODE_DIR = ".local/share/nebula-node"            # on the model machine, under the user's home
NODE_STATE = ".local/state/nebula-node"
GATEWAY_PORT = 17670
FACE_PORT = 8081                                  # PAIR's llama.cpp engine address on a machine (its router face)

INSTALL_SH = r"""#!/bin/sh
# A model machine for a Nebula on your network: the network's node programs, the node keeper, OpenShell on rootless Podman,
# as this user's services. Nothing here needs root (Podman was the administrator's step). Idempotent.
set -e
D="$HOME/__NODE_DIR__"; S="$HOME/__NODE_STATE__"; U="$HOME/.config/systemd/user"; OC="$HOME/.config/openshell"
mkdir -p "$D" "$S" "$U" "$OC"; chmod 700 "$S"
rm -rf "$D/new"; mkdir -p "$D/new"
tar -xf "$HOME/nebula-node.tar" -C "$D/new" && rm -f "$HOME/nebula-node.tar"
for x in pair openshell; do if [ -d "$D/new/$x" ]; then rm -rf "$D/$x"; mv "$D/new/$x" "$D/$x"; fi; done
mv -f "$D/new/nodekeeper.py" "$D/nodekeeper.py"; rm -rf "$D/new"
chmod +x "$D/pair/"* "$D/openshell/bin/"* 2>/dev/null || true
GW="$D/openshell/bin/openshell-gateway"
[ -e /usr/bin/openshell-gateway ] && GW=/usr/bin/openshell-gateway
if [ ! -f "$OC/gateway.toml" ] || grep -qs "written by Nebula" "$OC/gateway.toml"; then
  cat > "$OC/gateway.toml" <<'TOML'
# OpenShell gateway (written by Nebula for its engines on this machine): the Podman driver, rootless; per-sandbox
# driver config so Nebula can mount the models folder read-only and pick which cards an engine gets.
[openshell]
version = 2

[openshell.drivers.podman]
allow_driver_config = true
enable_bind_mounts = true
TOML
  echo "OPENSHELL_COMPUTE_DRIVER=podman" > "$OC/gateway.env"
fi
cat > "$U/openshell-gateway.service" <<UNIT
[Unit]
Description=OpenShell Gateway (for a Nebula on your network)
After=podman.socket
Wants=podman.socket

[Service]
Type=simple
StateDirectory=openshell/gateway
Environment=OPENSHELL_LOCAL_TLS_DIR=%h/.local/state/openshell/tls
EnvironmentFile=-%E/openshell/gateway.env
ExecStartPre=$GW config preflight
ExecStartPre=$GW generate-certs --output-dir \${OPENSHELL_LOCAL_TLS_DIR} --server-san host.openshell.internal
ExecStart=$GW
Restart=on-failure
RestartSec=5s
UMask=0077

[Install]
WantedBy=default.target
UNIT
cat > "$U/nebula-node.service" <<UNIT
[Unit]
Description=Model machine for a Nebula on your network (the network's node programs and the node keeper)
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 $D/nodekeeper.py serve --pair-bin $D/pair --state $S --facade-port __FACE_PORT__
Restart=always
RestartSec=3s

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now podman.socket
systemctl --user enable openshell-gateway.service nebula-node.service
systemctl --user restart openshell-gateway.service nebula-node.service
for i in $(seq 1 90); do [ -S "$S/keeper.sock" ] && [ -f "$HOME/.local/state/openshell/tls/client/tls.crt" ] && break; sleep 1; done
[ -S "$S/keeper.sock" ] || { echo "the node keeper did not start:"; journalctl --user -u nebula-node -n 20 --no-pager; exit 1; }
[ -f "$HOME/.local/state/openshell/tls/client/tls.crt" ] || { echo "OpenShell's gateway did not start:"; journalctl --user -u openshell-gateway -n 20 --no-pager; exit 1; }
echo "NODE ready"
"""

UNINSTALL_SH = r"""#!/bin/sh
D="$HOME/__NODE_DIR__"; S="$HOME/__NODE_STATE__"; U="$HOME/.config/systemd/user"; OC="$HOME/.config/openshell"
command -v openshell >/dev/null 2>&1 || PATH="$D/openshell/bin:$PATH"
for sb in $(openshell sandbox list 2>/dev/null | awk '$1 ~ /^nebeng-/ {print $1}'); do openshell sandbox delete "$sb" >/dev/null 2>&1 || true; done
systemctl --user disable --now nebula-node.service 2>/dev/null || true
rm -f "$U/nebula-node.service"
if [ -f "$U/openshell-gateway.service" ] && grep -q "for a Nebula on your network" "$U/openshell-gateway.service"; then
  systemctl --user disable --now openshell-gateway.service 2>/dev/null || true; rm -f "$U/openshell-gateway.service"
  if grep -qs "written by Nebula" "$OC/gateway.toml"; then rm -rf "$OC" "$HOME/.local/state/openshell"; fi
fi
systemctl --user daemon-reload 2>/dev/null || true
pkill -f "$D/" 2>/dev/null || true
rm -rf "$D" "$S" "$HOME/nebula-node.tar"
echo "removed"
"""

PREFLIGHT_SH = r"""uname -sm; (nvidia-smi -L 2>/dev/null || echo "no NVIDIA GPU (nvidia-smi)"); df -Pk "$HOME" | tail -1;
python3 --version 2>&1 || echo "no python3"
[ -S "$HOME/.local/state/nebula-node/keeper.sock" ] && echo "node: installed" || echo "node: not installed"
command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1 && echo "user systemd: yes" || echo "user systemd: no"
echo "ssh: ${SSH_CONNECTION:-}"; echo "user: $(id -un)"
command -v podman >/dev/null 2>&1 && echo "podman: $(podman --version 2>/dev/null | awk '{print $3}')" || echo "podman: missing"
command -v newuidmap >/dev/null 2>&1 && echo "uidmap: yes" || echo "uidmap: no"
grep -qs "^$(id -un):" /etc/subuid && grep -qs "^$(id -un):" /etc/subgid && echo "subuid: yes" || echo "subuid: no"
command -v nvidia-ctk >/dev/null 2>&1 && echo "container toolkit: yes" || echo "container toolkit: no"
{ [ -f /etc/cdi/nvidia.yaml ] || [ -f /var/run/cdi/nvidia.yaml ]; } && echo "cdi: yes" || echo "cdi: no"
if command -v podman >/dev/null 2>&1 && podman --version 2>/dev/null | awk '{print $3}' | grep -q '^4\.' && \
   ! { [ -f /etc/cdi/nvidia-podman.json ] && [ -f /etc/systemd/system/nebula-cdi-podman.path ]; } && \
   grep -qs additionalGids /etc/cdi/*.yaml /etc/cdi/*.json /var/run/cdi/*; then echo "cdi for podman: no"; else echo "cdi for podman: yes"; fi
systemctl is-active ufw >/dev/null 2>&1 && echo "firewall: ufw" || echo "firewall: none"
[ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null)" = yes ] && echo "linger: yes" || echo "linger: no"
command -v sudo >/dev/null 2>&1 && echo "sudo: yes" || echo "sudo: no"; command -v apt-get >/dev/null 2>&1 && echo "apt: yes" || echo "apt: no"
"""

# PAIR between the two machines: its node services (TCP 14318-14323), the engine face (TCP 8080), discovery (UDP 5353)
PAIR_PORTS = "14318:14323"
# Podman 4 (Ubuntu 24.04's) cannot read NVIDIA's CDI description from toolkit 1.18+ (`additionalGids`) and then knows no
# card: a copy without that field in /etc/cdi, written again whenever NVIDIA's own changes (a systemd path unit).
CDI_PODMAN = r"""mkdir -p /etc/cdi
cat > /usr/local/sbin/nebula-cdi-podman <<'SCRIPT'
#!/bin/sh
# Written by Nebula: NVIDIA's CDI description of this machine's cards for Podman 4 (without additionalGids, which it
# cannot read), written again by nebula-cdi-podman.path whenever NVIDIA's own description changes.
set -e
nvidia-ctk cdi generate --format=json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); d["cdiVersion"]="0.6.0"; [e.get("containerEdits", {{}}).pop("additionalGids", None) for e in d.get("devices", []) + [d]]; json.dump(d, open("/etc/cdi/nvidia-podman.json.tmp", "w"), indent=1)'
mv /etc/cdi/nvidia-podman.json.tmp /etc/cdi/nvidia-podman.json
SCRIPT
chmod 755 /usr/local/sbin/nebula-cdi-podman
printf '[Unit]\nDescription=NVIDIA cards described for Podman 4 (written by Nebula)\n\n[Service]\nType=oneshot\nExecStart=/usr/local/sbin/nebula-cdi-podman\n' > /etc/systemd/system/nebula-cdi-podman.service
printf '[Unit]\nDescription=Describe the NVIDIA cards to Podman 4 again when NVIDIA does (written by Nebula)\n\n[Path]\nPathChanged=/var/run/cdi/nvidia.yaml\nPathChanged=/etc/cdi/nvidia.yaml\n\n[Install]\nWantedBy=multi-user.target\n' > /etc/systemd/system/nebula-cdi-podman.path
systemctl daemon-reload
systemctl enable --now nebula-cdi-podman.path
/usr/local/sbin/nebula-cdi-podman"""
# What an administrator may be asked to do, fixed here (the page sends only these names): words, then the command.
ADMIN_TASKS = {
    "podman-install": ("install Podman, rootless (OpenShell's sandboxes for the engines run on it)",
                       "apt-get -o DPkg::Lock::Timeout=600 update -q && DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -q podman uidmap passt slirp4netns"),
    "subuid": ("give {user} its own range of user ids for rootless containers",
               "usermod --add-subuids 100000-165535 --add-subgids 100000-165535 {user}"),
    "cdi": ("describe its NVIDIA cards to containers (CDI)", "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"),
    "toolkit": ("install NVIDIA's container toolkit and describe its cards to containers (CDI)",
                "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor "
                "-o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -fsSL "
                "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "
                "'s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' "
                "> /etc/apt/sources.list.d/nvidia-container-toolkit.list && apt-get -o DPkg::Lock::Timeout=600 update -q && "
                "DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -q nvidia-container-toolkit && mkdir -p /etc/cdi && "
                "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"),
    "cdi-podman": ("describe its NVIDIA cards to Podman 4 too, kept in step with NVIDIA's own description (which has "
                   "a field Podman 4 cannot read)", CDI_PODMAN),
    "linger": ("keep {user}'s services running while nobody is logged in", "loginctl enable-linger {user}"),
    "firewall": ("let {peer} reach this machine through its firewall (your models over mutual TLS; only that machine)",
                 "ufw allow from {peer} comment 'Nebula on your network'"),
}


class SshError(RuntimeError):
    pass


USER_SSH_CONFIG = os.path.join(os.path.expanduser("~"), ".ssh", "config")     # the owner's own SSH settings, read as ssh reads them
_DEFAULT_KEYS = {"id_rsa", "id_ecdsa", "id_ecdsa_sk", "id_ed25519", "id_ed25519_sk", "id_xmss", "id_dsa"}


def ssh_config(host: str, port: int | None = None) -> dict:
    """What `ssh` would use for this host (ssh -G): its address, port, user, keys and host key alias."""
    try:
        f = os.environ.get("NEBULA_SSH_CONFIG")                  # tests: another settings file than ~/.ssh/config
        p = subprocess.run(["ssh", "-G"] + (["-F", f] if f else []) + (["-p", str(int(port))] if port else []) + [host],
                           capture_output=True, text=True,
                           timeout=15, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out: dict = {"identityfile": []}
    for ln in p.stdout.splitlines():
        k, _, v = ln.partition(" ")
        if k == "identityfile":
            out["identityfile"].append(os.path.expanduser(v))
        elif k in ("hostname", "port", "user", "hostkeyalias"):
            out[k] = v
    return out


def config_hosts(path: str | None = None) -> list[str]:
    """The names in ~/.ssh/config (no patterns), for the page to offer."""
    names = []
    try:
        for ln in open(path or os.environ.get("NEBULA_SSH_CONFIG") or USER_SSH_CONFIG, encoding="utf-8"):
            w = ln.split()
            if len(w) > 1 and w[0].lower() == "host":
                names += [n for n in w[1:] if not any(ch in n for ch in "*?!")]
    except OSError:
        pass
    return list(dict.fromkeys(names))


def own_key(host: str, port: int = 22) -> str | None:
    """The key ssh would offer that the owner chose (not one of ssh's defaults): the host's own from ~/.ssh/config, or,
    for an address, the key of a name in ~/.ssh/config that points at it."""
    def chosen(g):
        return next((k for k in g.get("identityfile") or [] if os.path.basename(k) not in _DEFAULT_KEYS and os.path.isfile(k)), None)
    g = ssh_config(host, port)
    if chosen(g):
        return chosen(g)
    for name in config_hosts():
        o = ssh_config(name)
        if o.get("hostname") == g.get("hostname") and str(o.get("port") or 22) == str(port) and chosen(o):
            return chosen(o)
    return None


def kh_name(host: str, port: int = 22) -> str:
    """The name Nebula's known_hosts keeps a server's key under: the host as the owner wrote it."""
    return host if int(port) == 22 else f"[{host}]:{int(port)}"


class Target:
    def __init__(self, host: str, user: str, port: int = 22, key_file: str | None = None, known_hosts: str | None = None):
        if not re.match(r"^[A-Za-z0-9.\-:\[\]]+$", host or "") or not re.match(r"^[A-Za-z0-9._-]+$", user or ""):
            raise SshError("a host (name or address) and a user name")
        self.host, self.user, self.port, self.key_file, self.known_hosts = host, user, int(port), key_file, known_hosts

    def argv(self, remote: str) -> list[str]:
        a = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={self.known_hosts}",
             "-o", f"HostKeyAlias={kh_name(self.host, self.port)}", "-o", "ConnectTimeout=15", "-p", str(self.port)]
        key = os.path.expanduser(self.key_file) if self.key_file else own_key(self.host, self.port)
        if key:
            a += ["-i", key, "-o", "IdentitiesOnly=yes"]
        return a + [f"{self.user}@{self.host}", remote]

    def run(self, remote: str, stdin: bytes | None = None, timeout: float = 600) -> str:
        try:
            p = subprocess.run(self.argv(remote), input=stdin, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise SshError(f"{self.host}: no answer in {int(timeout)} s") from None
        out, err = p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")
        if p.returncode != 0:
            if "REMOTE HOST IDENTIFICATION HAS CHANGED" in err or "Host key verification failed" in err:
                raise SshError(f"{self.host}'s host key does not match the one you confirmed: refused (it may be a different "
                               "machine; confirm its new key only if you know why it changed)")
            raise SshError(f"{self.host}: {err.strip()[-400:] or out.strip()[-400:] or 'ssh failed'}")
        return out


def scan(host: str, port: int = 22) -> list[dict]:
    """The host's keys and their SHA256 fingerprints (ssh-keyscan + ssh-keygen), to show before trusting any. A name
    from ~/.ssh/config is scanned at its address."""
    addr = ssh_config(host, port).get("hostname") or host
    p = subprocess.run(["ssh-keyscan", "-p", str(int(port)), "-T", "10", addr], capture_output=True, text=True, timeout=30)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise SshError(f"{host}:{port} answered with no SSH host key")
    out = []
    for ln in lines:
        f = subprocess.run(["ssh-keygen", "-lf", "-"], input=ln, capture_output=True, text=True)
        fp = (f.stdout.split() or ["", ""])[1]
        out.append({"line": ln, "fingerprint": fp, "type": ln.split()[1]})
    return out


def trust(known_hosts: str, host: str, port: int, fingerprint: str) -> dict:
    """Keep the key whose fingerprint the owner confirmed (scanned again now: a key that changed meanwhile is refused)."""
    keys = [k for k in scan(host, port) if k["fingerprint"] == fingerprint]
    if not keys:
        raise SshError("the host no longer shows that key: refused")
    os.makedirs(os.path.dirname(known_hosts), mode=0o700, exist_ok=True)
    with open(known_hosts, "a", encoding="utf-8") as f:
        f.write(kh_name(host, port) + " " + keys[0]["line"].split(" ", 1)[1] + "\n")     # under the name Target asks for
    os.chmod(known_hosts, 0o600)
    return {"trusted": fingerprint}


def facts(report: list[str]) -> dict:
    """The preflight's "key: value" lines."""
    out = {}
    for ln in report:
        k, sep, v = ln.partition(": ")
        if sep and (" " not in k.strip() or k in ("container toolkit", "user systemd", "cdi for podman")):
            out[k.strip()] = v.strip()
    return out


def admin_needs(report: list[str], local_firewall: bool | None = None) -> list[dict]:
    """[{id, where ("server" | "here"), words, command}]: what an administrator should do for the machine to run
    sandboxed engines, and for the two machines to reach each other; plus notes (no command) for what Nebula cannot do
    for you."""
    f = facts(report)
    user = f.get("user") or ""
    ssh = (f.get("ssh") or "").split()
    me, it = (ssh[0], ssh[2]) if len(ssh) == 4 else (None, None)          # this machine and the server, as each sees the other
    want = []
    if f.get("podman") == "missing" or f.get("uidmap") == "no":
        want.append("podman-install" if f.get("apt") == "yes" else None)
    if f.get("subuid") == "no":
        want.append("subuid")
    if f.get("container toolkit") == "no" and "GPU" in " ".join(report) and f.get("apt") == "yes":
        want.append("toolkit")
    if f.get("container toolkit") == "yes" and f.get("cdi") == "no":
        want.append("cdi")
    if f.get("container toolkit") == "yes" and (f.get("cdi for podman") == "no" or
                                                (f.get("podman") == "missing" and f.get("cdi") == "yes")):
        want.append("cdi-podman")
    if f.get("linger") == "no":
        want.append("linger")
    if f.get("firewall") == "ufw" and _ip(me):
        want.append("firewall")
    out = []
    for w in [x for x in want if x]:
        words, cmd = ADMIN_TASKS[w]
        v = {"user": user, "peer": me, "ports": PAIR_PORTS}
        out.append({"id": w, "where": "server", "words": words.format(**v), "command": cmd.format(**v)})
    if (local_firewall if local_firewall is not None else _ufw_here()) and _ip(it):
        words, cmd = ADMIN_TASKS["firewall"]
        v = {"peer": it, "ports": PAIR_PORTS}
        out.append({"id": "firewall-here", "where": "here", "words": "on this computer: " + words.format(**v), "command": cmd.format(**v)})
    if f.get("podman") == "missing" and f.get("apt") != "yes":
        out.append({"id": None, "where": "server", "words": "Podman is not installed and this machine has no apt: install "
                    "podman, uidmap and passt with its own package manager, then check again", "command": None})
    if f.get("container toolkit") == "no" and "GPU" in " ".join(report) and f.get("apt") != "yes":
        out.append({"id": None, "where": "server", "words": "NVIDIA's container toolkit is not installed: its cards cannot "
                    "reach a sandbox until it is (docs.nvidia.com/datacenter/cloud-native/container-toolkit)", "command": None})
    if f.get("user systemd") == "no":
        out.append({"id": None, "where": "server", "words": "systemd's user services are not available there: the node "
                    "pieces run as your services, so this machine cannot be added yet", "command": None})
    if f.get("python3", "").startswith("no") or "no python3" in report:
        out.append({"id": None, "where": "server", "words": "python3 is not installed there (the node keeper needs it)", "command": None})
    if f.get("sudo") == "no" and any(x["where"] == "server" and x["id"] for x in out):
        out.append({"id": None, "where": "server", "words": "sudo is not installed there: ask its administrator", "command": None})
    return out


def _ip(s) -> bool:
    import ipaddress
    try:
        ipaddress.ip_address(str(s))
        return True
    except ValueError:
        return False


def _ufw_here() -> bool:
    try:
        return subprocess.run(["systemctl", "is-active", "ufw"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _sudo_script(cmds: list[str]) -> str:
    return "set -e\n" + "\n".join(f"echo {shlex.quote('[sudo] ' + c)}\n{c}" for c in cmds) + "\n"


def run_admin(t: Target, chosen: list[str], password: str, local_password: str = "") -> dict:
    """The chosen tasks, worked out again from a fresh preflight (the page names tasks, never commands), with sudo:
    the server's over SSH, this computer's here. The password goes to sudo on stdin once and is not kept or logged."""
    pf = preflight(t)
    needs = {n["id"]: n for n in admin_needs(pf["report"]) if n["id"]}
    pick = [needs[i] for i in chosen if i in needs]
    done = []
    server = [n["command"] for n in pick if n["where"] == "server"]
    if server:
        if not password:
            raise SshError(f"your password on {t.host} is needed for: " + "; ".join(n["words"] for n in pick if n["where"] == "server"))
        script = _sudo_script(server)
        out = _sudo_run(lambda stdin: t.run(f"sudo -S -p '' sh -c {shlex.quote(script)}", stdin=stdin, timeout=1800), password)
        done += [n["words"] for n in pick if n["where"] == "server"]
        _ = out
    here = [n["command"] for n in pick if n["where"] == "here"]
    if here:
        if not local_password:
            raise SshError("this computer's password is needed for: " + "; ".join(n["words"] for n in pick if n["where"] == "here"))
        p = subprocess.run(["sudo", "-S", "-p", "", "sh", "-c", _sudo_script(here)], input=(local_password + "\n").encode(),
                           capture_output=True, timeout=600)
        if p.returncode != 0:
            raise SshError("on this computer: " + _sudo_why(p.stderr.decode(errors="replace")))
        done += [n["words"] for n in pick if n["where"] == "here"]
    return {"done": done}


def _sudo_why(err: str) -> str:
    if "incorrect password" in err or "Sorry, try again" in err:
        return "sudo refused the password"
    if "not in the sudoers" in err:
        return "this user may not use sudo"
    return err.strip()[-300:] or "sudo failed"


def _sudo_run(run, password: str) -> str:
    try:
        return run((password + "\n").encode())
    except SshError as e:
        raise SshError(_sudo_why(str(e))) from None


def confirmed_known_hosts(host: str, port: int, fingerprint: str, folder: str) -> str:
    """A known_hosts file in `folder` with only the key the owner just confirmed (before the wizard keeps it), for
    looking at the machine before anything is installed. A key that changed meanwhile is refused."""
    keys = [k for k in scan(host, port) if k["fingerprint"] == fingerprint]
    if not keys:
        raise SshError("the host no longer shows that key: refused")
    path = os.path.join(folder, "known_hosts")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(kh_name(host, port) + " " + keys[0]["line"].split(" ", 1)[1] + "\n")
    return path


def preflight(t: Target) -> dict:
    out = t.run(PREFLIGHT_SH, timeout=60)
    report = out.strip().splitlines()
    return {"report": report, "needs": admin_needs(report),
            "commands": [f"cat > ~/nebula-node.tar  (the network's node programs, the node keeper, OpenShell; over this SSH link)",
                         "sh -s  (sets them up as your user services: nebula-node, openshell-gateway, podman.socket)"]}


def node_tar(pair_bin: str, openshell_bin: str | None) -> bytes:
    """What a model machine gets: pair/ (PAIR's programs), openshell/bin/ (its CLI and gateway, when this machine has
    them to give), nodekeeper.py."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for f in sorted(os.listdir(pair_bin)):
            p = os.path.join(pair_bin, f)
            if os.path.isfile(p):
                tf.add(p, arcname=f"pair/{f}")
        if openshell_bin:
            for f in ("openshell", "openshell-gateway"):
                p = os.path.join(openshell_bin, f)
                if os.path.isfile(p):
                    tf.add(os.path.realpath(p), arcname=f"openshell/bin/{f}")
        tf.add(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nodekeeper.py"), arcname="nodekeeper.py")
    return buf.getvalue()


def install(t: Target, pair_bin: str, openshell_bin: str | None) -> dict:
    """Copy the node pieces over SSH and start them as the user's services. No Nebula is installed there."""
    t.run("cat > ~/nebula-node.tar", stdin=node_tar(pair_bin, openshell_bin), timeout=1800)
    script = INSTALL_SH.replace("__NODE_DIR__", NODE_DIR).replace("__NODE_STATE__", NODE_STATE) \
        .replace("__FACE_PORT__", str(FACE_PORT))
    out = t.run("sh -s", stdin=script.encode(), timeout=900)
    if "NODE ready" not in out:
        raise SshError("the node pieces did not start: " + out.strip()[-600:])
    return {"log": out.strip().splitlines()[-10:]}


def uninstall(t: Target) -> dict:
    script = UNINSTALL_SH.replace("__NODE_DIR__", NODE_DIR).replace("__NODE_STATE__", NODE_STATE)
    return {"log": t.run("sh -s", stdin=script.encode(), timeout=300).strip().splitlines()}


def keeper_argv(t: Target) -> list[str]:
    """The node keeper's JSON-RPC over SSH (stdio), as PAIR's own broker speaks it (nebula/pairnet.py)."""
    return t.argv(f"exec python3 ~/{NODE_DIR}/nodekeeper.py connect --state ~/{NODE_STATE}")


def gateway_certs(t: Target) -> dict:
    """The OpenShell gateway's client certificate there, for this machine's `openshell` to manage it over SSH."""
    out = {}
    for name, rel in (("ca.crt", "ca.crt"), ("tls.crt", "client/tls.crt"), ("tls.key", "client/tls.key")):
        out[name] = t.run(f"cat ~/.local/state/openshell/tls/{rel}", timeout=30)
    return out


def gpus(t: Target) -> list[dict]:
    """Its cards, as nvidia-smi sees them (the main Nebula's Engines page shows them like its own)."""
    q = "index,uuid,name,memory.total,memory.used,utilization.gpu,compute_cap"
    out = t.run(f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits", timeout=30)
    cards = []
    for ln in out.strip().splitlines():
        w = [x.strip() for x in ln.split(",")]
        if len(w) >= 6:
            cards.append({"index": int(w[0]), "uuid": w[1], "name": w[2], "memory_total_mib": int(float(w[3])),
                          "memory_used_mib": int(float(w[4])), "util": int(float(w[5])) if w[5].replace(".", "").isdigit() else None,
                          "compute_cap": w[6] if len(w) > 6 else None})
    return cards


def quoted(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)
