"""nebula/lan.py: this machine on the network: its name and addresses, the tokens that may call it from beyond it,
engines already running at their usual ports, and the CLIs installed (adopting local engines, step G1).

LAN discovery and PIN pairing between Nebulas (step G3) come in a later release; the text below describes them.

The beacon is a small JSON datagram on a multicast group every few seconds, sent while this node is discoverable
and heard always, so a fresh install can see who is already running before it decides anything. Nodes are keyed by
a UUID the node keeps, not by hostname. A node that was not heard for a while is dropped.

Pairing: the machine that will be joined opens a window (two minutes) and shows a six-digit PIN; the joining machine
posts the PIN to it with its own identity; the joined machine answers with a token and its registry, and both sides
remember each other: the joined keeps the token's hash beside the peer's UUID, the joining keeps the token in a key
file and adds the joined machine's tiers as its own, named `<peer>/<tier>` and asked for by their direct names. A
request from beyond this machine must carry a paired token; the pairing route itself is open only while the window
is, and the PIN is compared in constant time with a few tries allowed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import socket
import struct
import threading
import time
import uuid

LOG = logging.getLogger("cockpit.lan")
LOCAL_ENGINE_PORTS = [(11434, "ollama"), (1234, "lmstudio"), (8080, "llama"), (18152, "strata-lite")]


def host_name() -> str:
    return socket.gethostname().split(".")[0] or "nebula"


def own_addresses() -> list[str]:
    """This machine's addresses other machines could reach, best first."""
    out = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))    # no packet is sent: the route picks the interface
            out.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            a = info[4][0]
            if a not in out and not a.startswith("127."):
                out.append(a)
    except OSError:
        pass
    return out


def is_loopback(address: str) -> bool:
    return address in ("127.0.0.1", "::1", "localhost") or address.startswith("127.")


# ------------------------------------------------------------------------------------------------ tokens
def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class Tokens:
    """The tokens that may call this Nebula from beyond this machine (a sandboxed CLI's, step O3), by their hashes.
    LAN discovery and PIN pairing come in a later release."""

    def __init__(self, node: dict, peers: dict | None = None):
        self.node = node
        self.peers: dict[str, dict] = dict(peers or {})     # uuid -> {"name", "address", "port", "token_sha256", "paired_at"}
        self.lock = threading.RLock()

    def window(self) -> dict:
        return {"open": False, "pin": None, "seconds_left": 0}

    def remember(self, peer: dict, token: str) -> dict:
        """A peer this node joined gave it this token to use back here (pairing goes both ways): keep its hash."""
        with self.lock:
            self.peers[peer["uuid"]] = {"name": peer.get("name") or peer["uuid"], "address": peer.get("address"),
                                        "port": peer.get("port"), "token_sha256": token_hash(token), "paired_at": time.time()}
            return self.peers[peer["uuid"]]

    def allowed(self, token: str | None) -> dict | None:
        """The peer a bearer token belongs to, or None."""
        if not token:
            return None
        h = token_hash(token)
        with self.lock:
            for uuid_, p in self.peers.items():
                if hmac.compare_digest(p.get("token_sha256", ""), h):
                    return {"uuid": uuid_, **p}
        return None

    def forget(self, uuid_: str) -> bool:
        with self.lock:
            return self.peers.pop(uuid_, None) is not None


def new_node(name: str | None, port: int, version: str) -> dict:
    return {"uuid": str(uuid.uuid4()), "name": name or host_name(), "port": port, "version": version}


# ------------------------------------------------------------------------------------------------ local engines
def probe_local_engines(timeout: float = 0.6, ports=None, host: str = "127.0.0.1") -> list[dict]:
    """Engines already running on this machine at their usual ports, with the models each lists."""
    import urllib.request
    out = []
    for port, kind in (LOCAL_ENGINE_PORTS if ports is None else ports):
        base = f"http://{host}:{port}"
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=timeout) as r:
                obj = json.loads(r.read())
        except Exception:  # noqa: BLE001 - not there, or not an OpenAI-compatible server
            continue
        models = [m.get("id") for m in (obj.get("data") if isinstance(obj, dict) else obj) or [] if isinstance(m, dict) and m.get("id")]
        kind_seen = kind
        try:
            with urllib.request.urlopen(base + "/props", timeout=timeout) as r:
                props = json.loads(r.read())
            if isinstance(props, dict) and props.get("model_alias"):
                kind_seen = "llama-server" if kind != "strata-lite" else kind
        except Exception:  # noqa: BLE001
            props = {}
        out.append({"kind": kind_seen, "url": base, "models": models,
                    "model": (props.get("model_alias") if isinstance(props, dict) else None) or (models[0] if models else None)})
    return out


# ------------------------------------------------------------------------------------------------ CLIs on this machine
CLIS = [("claude", "Claude Code", "anthropic"), ("codex", "Codex CLI", "responses"), ("grok", "Grok", "openai"),
        ("ollama", "Ollama", "engine"), ("lms", "LM Studio", "engine"), ("llama-server", "llama.cpp server", "engine"),
        ("impulse-server", "Impulse", "engine"), ("strata-lite-server", "Impulse (before 0.1.1)", "engine"), ("openshell", "NVIDIA OpenShell", "sandbox")]


def probe_clis() -> list[dict]:
    """Tools installed on this machine that Nebula can be connected to (as inputs) or that run engines."""
    import shutil
    out = []
    for binary, label, kind in CLIS:
        path = shutil.which(binary)
        if path:
            out.append({"binary": binary, "name": label, "kind": kind, "path": path})
    return out


def connect_lines(server_url: str) -> dict:
    """How each CLI is pointed at this Nebula (README.md's lines)."""
    return {"anthropic": f"ANTHROPIC_BASE_URL={server_url} ANTHROPIC_AUTH_TOKEN=nebula claude --model local",
            "responses": (f"codex -c 'model_providers.nebula.name=\"nebula\"' -c 'model_providers.nebula.base_url=\"{server_url}/v1\"' "
                          f"-c 'model_providers.nebula.wire_api=\"responses\"' -c 'model_provider=\"nebula\"' -m local"),
            "openai": f"OPENAI_BASE_URL={server_url}/v1 OPENAI_API_KEY=nebula"}
