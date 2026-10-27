"""nebula/gateway.py: the gateway, your own iroh relay (Gravity: Constellation), deployed from Nebula.

A gateway lets your Nebulas (and your friends', once invited) reach each other from anywhere: two that cannot connect
directly meet through it. It only forwards encrypted QUIC between keys and cannot read it; it is optional (on your own
network they connect directly). It is iroh's own relay server in a container (nebula/gateway/Containerfile), with a
config Nebula writes:

- **who may use it:** an allowlist of member keys (this node and its peers), rewritten when a peer joins or is removed;
- **TLS:** with a hostname (and an email for Let's Encrypt) it serves HTTPS on 443 with a certificate of its own;
  without one, plain HTTP on 3340, for a private network or a test.

Deploy it here (`docker run`, needs Docker) or on any host with a public address: Nebula prints the same two commands
to run there (build the image, run it with the config).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

NAME = "nebula-gateway"
IMAGE = "nebula-gateway:1"
HERE = Path(__file__).with_name("gateway")


class GatewayError(RuntimeError):
    pass


def config(members: list[str], hostname: str | None = None, email: str | None = None, port: int = 3340) -> str:
    lines = ["enable_relay = true", "enable_quic_addr_discovery = false"]
    if hostname:
        lines += ['http_bind_addr = "0.0.0.0:80"', "", "[tls]", 'cert_mode = "LetsEncrypt"', f'hostname = ["{hostname}"]',
                  "prod_tls = true", f'contact = "{email or ""}"', 'cert_dir = "/etc/nebula-gateway/certs"',
                  'https_bind_addr = "0.0.0.0:443"']
    else:
        lines += [f'http_bind_addr = "0.0.0.0:{int(port)}"']
    ids = ", ".join(f'"{m}"' for m in dict.fromkeys(members))
    return "\n".join(lines[:3] + ([f"access.allowlist = [{ids}]"] if members else []) + lines[3:]) + "\n"


def url_of(hostname: str | None, host: str = "127.0.0.1", port: int = 3340) -> str:
    return f"https://{hostname}" if hostname else f"http://{host}:{port}"


def docker() -> str:
    d = os.environ.get("NEBULA_DOCKER") or shutil.which("docker")
    if not d:
        raise GatewayError("Docker is not on this machine: deploy the gateway on a host that has it (the commands are shown)")
    return d


def commands(conf_dir: str, hostname: str | None, port: int = 3340) -> list[str]:
    ports = "-p 80:80 -p 443:443 -p 7842:7842/udp" if hostname else f"-p {port}:{port}"
    return [f"docker build -t {IMAGE} -f {HERE}/Containerfile {HERE}",
            f"docker run -d --name {NAME} --restart unless-stopped {ports} -v {conf_dir}:/etc/nebula-gateway {IMAGE}"]


def deploy(conf_dir: str, members: list[str], hostname: str | None = None, email: str | None = None,
           port: int = 3340) -> dict:
    """Here: build the image (once), write the config, run the container. Returns its URL."""
    d = docker()
    os.makedirs(conf_dir, exist_ok=True)
    Path(conf_dir, "relay.toml").write_text(config(members, hostname, email, port), encoding="utf-8")
    if subprocess.run([d, "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        b = subprocess.run([d, "build", "-t", IMAGE, "-f", str(HERE / "Containerfile"), str(HERE)], capture_output=True, text=True)
        if b.returncode != 0:
            raise GatewayError("the gateway image did not build: " + (b.stderr or b.stdout)[-400:])
    subprocess.run([d, "rm", "-f", NAME], capture_output=True)
    ports = ["-p", "80:80", "-p", "443:443", "-p", "7842:7842/udp"] if hostname else ["-p", f"{port}:{port}"]
    r = subprocess.run([d, "run", "-d", "--name", NAME, "--restart", "unless-stopped"] + ports +
                       ["-v", f"{conf_dir}:/etc/nebula-gateway", IMAGE], capture_output=True, text=True)
    if r.returncode != 0:
        raise GatewayError("the gateway did not start: " + (r.stderr or r.stdout)[-400:])
    return {"url": url_of(hostname, port=port), "container": NAME}


def update_members(conf_dir: str, members: list[str], hostname: str | None = None, email: str | None = None,
                   port: int = 3340) -> None:
    """A peer joined or was removed: the allowlist rewritten and the relay restarted (a moment's gap)."""
    Path(conf_dir, "relay.toml").write_text(config(members, hostname, email, port), encoding="utf-8")
    try:
        subprocess.run([docker(), "restart", NAME], capture_output=True, timeout=60)
    except (GatewayError, subprocess.TimeoutExpired):
        pass


def remove() -> None:
    subprocess.run([docker(), "rm", "-f", NAME], capture_output=True)


def ping(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/ping", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def wait_up(url: str, seconds: float = 30) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if ping(url):
            return True
        time.sleep(0.5)
    return False
