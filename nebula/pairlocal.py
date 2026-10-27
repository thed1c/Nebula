"""nebula/pairlocal.py (internal): a router of the Gravity kind installed on this same machine.

Users see "Gravity" only: a Gravity upstream, Gravity peers. Internally this finds a local NVIDIA Personal AI Router
install (Apache-2.0, https://github.com/NVIDIA/Personal-AI-Router), which routes a request to one of the machines in
its own mesh, and adopts it as one upstream. Nothing is read from the network: only loopback ports on this machine
are probed (discovery only on the same machine).

Its client proxies listen on loopback in the engines' own dialects: OpenAI-compatible on 8080 (llama.cpp) and 1234
(LM Studio), Ollama on 11434. A proxy is told from a plain engine on the same port by its identity probe: a request
carrying `X-NVPAIR-Engine-Identity-Probe: 1` is answered 409 "proxy-facade" by the proxy and normally by an engine.

Adopted, it is a tier with `mesh: "gravity"` and the OpenAI-compatible address; the models it lists (the union of its
machines' models) are the tier's model choices, and a request names one exactly, as that router requires.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

PROBE = "X-NVPAIR-Engine-Identity-Probe"
PORTS = [(8080, "openai"), (1234, "openai"), (11434, "ollama")]


def _get(url: str, headers: dict | None = None, timeout: float = 1.5):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (OSError, ValueError):
        return None, b""


def is_router(base: str, dialect: str = "openai") -> bool:
    path = "/api/tags" if dialect == "ollama" else "/v1/models"
    status, body = _get(base + path, {PROBE: "1"})
    return status == 409 and b"proxy-facade" in body


def models(base: str, dialect: str = "openai") -> list[str]:
    if dialect == "ollama":
        status, body = _get(base + "/api/tags", timeout=5)
        try:
            return [m.get("name") for m in json.loads(body).get("models") or [] if m.get("name")] if status == 200 else []
        except ValueError:
            return []
    status, body = _get(base + "/v1/models", timeout=5)
    try:
        return [m.get("id") for m in json.loads(body).get("data") or [] if m.get("id")] if status == 200 else []
    except ValueError:
        return []


def find(ports=None) -> list[dict]:
    """The Gravity routers on this machine: {"address", "dialect", "models"} (loopback only)."""
    out = []
    for port, dialect in ports or PORTS:
        base = f"http://127.0.0.1:{port}"
        if is_router(base, dialect):
            out.append({"address": base, "dialect": dialect, "models": models(base, dialect)})
    return out
