"""nebula/stars.py: Nebula's core stars, the three gold ones that circle the router.

- **API Server**: the gateway (/v1/chat/completions, /v1/messages, /v1/responses). Its settings, in the config's
  `"api"`: `require_key` (when on, every /v1 request needs one of its keys, from this machine too) and `keys` (their
  sha256 only; a key is shown once, when it is made), and `route` (`{}` = Nebula routes each request; `{"tier",
  "model"}` = a dedicated model: that tier, asked for that model by name).
- **Assistant**: the page's chat. `"assistant"`: `route` as above, `system` (its prompt), `tools` (true = the tools
  of every Lookup server that is on). It runs as the built-in agent `assistant` unless the page picks another.
- **Lookup**: the MCP servers (`"mcp_servers"`, Claude Desktop's shape; `"disabled": true` turns one off), including
  two built-in ones, `{"builtin": "docs", "folders": [...]}` and `{"builtin": "web", "provider", "key_env"}`.

A dedicated model's choices are the tiers, and, for a tier in front of a router (another OpenAI-compatible server
with several models), the models its /v1/models lists, read through it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sys
import time

API_INPUTS = ("chat", "messages", "responses")
BUILTIN_AGENT = "assistant"
DEFAULT_SYSTEM = "You are Halo, this Nebula's own AI. Answer plainly. Use the lookup tools when a question needs documents or the web."
BUILTINS = {
    "library": {"label": "Library", "about": "your document collections, indexed here: search with citations (docs/lookup.md)"},
    "security": {"label": "Security log", "about": "the Event Horizon's security log, read only, for you and Halo"},
    "docs": {"label": "Documents", "about": "searches the folders you add"},
    "web": {"label": "Web search", "about": "Brave Search with a key from the environment, else DuckDuckGo"},
}


def star_of(input_name: str | None) -> str | None:
    """Which star a request came in by: the API Server for the API dialects, the Assistant for the page's chat."""
    if input_name in API_INPUTS:
        return "api"
    if input_name == "room":
        return "assistant"
    return None


def call_input(input_name: str | None) -> str:
    """The input a call is for, as a tier's headers name it ({input}): api, assistant, hook, schedule, other."""
    s = star_of(input_name)
    if s:
        return s
    if input_name and input_name.startswith("hook:"):
        return "hook"
    if input_name and input_name.startswith("schedule:"):
        return "schedule"
    return "other"


def section(cfg, star: str) -> dict:
    d = cfg.api if star == "api" else cfg.assistant
    return d if isinstance(d, dict) else {}


def route_of(cfg, star: str | None) -> tuple[str | None, str | None]:
    """(tier, model) of a star's dedicated model, or (None, None) when Nebula routes."""
    if not star:
        return None, None
    r = section(cfg, star).get("route") or {}
    tier = r.get("tier") if isinstance(r, dict) else None
    if not tier or cfg.tier(tier) is None:
        return None, None
    return tier, (r.get("model") or None)


def set_route(cfg, star: str, tier: str | None, model: str | None) -> dict:
    if tier and cfg.tier(tier) is None:
        raise ValueError(f"no tier named {tier!r}")
    d = section(cfg, star)
    d["route"] = {"tier": tier, "model": model} if tier else {}
    setattr(cfg, "api" if star == "api" else "assistant", d)
    return d["route"]


# --- pass-keys
def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def make_key(cfg, label: str = "") -> tuple[dict, str]:
    """A new key: its row (kept) and the key itself (shown once, never kept)."""
    key = "nbk_" + secrets.token_urlsafe(32)
    row = {"id": secrets.token_hex(4), "label": (label or "key")[:60], "sha256": _hash(key),
           "hint": key[:8] + "…" + key[-4:], "created_at": round(time.time())}
    cfg.api = dict(section(cfg, "api"), keys=list(section(cfg, "api").get("keys") or []) + [row])
    return row, key


def revoke_key(cfg, key_id: str) -> bool:
    keys = list(section(cfg, "api").get("keys") or [])
    left = [k for k in keys if k.get("id") != key_id]
    cfg.api = dict(section(cfg, "api"), keys=left)
    return len(left) != len(keys)


def key_ok(cfg, presented: str | None) -> bool:
    if not presented:
        return False
    h = _hash(presented)
    return any(hmac.compare_digest(k.get("sha256", ""), h) for k in section(cfg, "api").get("keys") or [])


def key_row(cfg, presented: str | None) -> tuple[str, dict] | None:
    """(whose, the key's row) of a presented key: ("api", row) for an API Server key, ("caller:<id>", row) for a
    caller's; None when it is neither. For the security log: a key is named by its id and hint, never itself."""
    if not presented:
        return None
    h = _hash(presented)
    for k in section(cfg, "api").get("keys") or []:
        if hmac.compare_digest(k.get("sha256", ""), h):
            return "api", k
    from nebula import callers
    for c in callers.all_of(cfg):
        for k in c.get("keys") or []:
            if hmac.compare_digest(k.get("sha256", ""), h):
                return f"caller:{c['id']}", k
    return None


def presented_key(headers) -> str | None:
    """The key a client sent: `Authorization: Bearer …` (OpenAI, Codex) or `x-api-key` (Anthropic clients)."""
    auth = headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("x-api-key") or headers.get("X-Api-Key") or None


REQUIRE_KEY_DEFAULT = False         # design decision (2026-10-20): apps on this PC need no key unless the owner turns one on


def requires_key(cfg) -> bool:
    """The API Server needs a key, from this machine too: off unless the owner turned it on ("require_key": true)."""
    v = section(cfg, "api").get("require_key")
    return REQUIRE_KEY_DEFAULT if v is None else v is not False


def gateway_allows(cfg, headers) -> bool:
    if not requires_key(cfg):
        return True
    k = presented_key(headers)
    from nebula import callers
    return key_ok(cfg, k) or callers.by_key(cfg, k) is not None     # the API Server's own keys, or a caller's


# --- Lookup: the MCP servers
def hub_servers(cfg) -> dict:
    """The servers to start: every one not disabled, the built-ins made into the commands that run them."""
    out = {}
    for name, c in (cfg.mcp_servers or {}).items():
        if not isinstance(c, dict) or c.get("disabled") is True:
            continue
        b = c.get("builtin")
        if b in IN_PROCESS:
            continue                                     # served in Nebula's own process (nebula/rag/tools.py)
        if b == "docs":
            args = ["-m", "nebula.lookup.docs"]
            for f in c.get("folders") or []:
                args += ["--folder", str(f)]
            out[name] = {"command": sys.executable, "args": args}
        elif b == "web":
            out[name] = {"command": sys.executable, "args": ["-m", "nebula.lookup.web", "--provider", str(c.get("provider") or "auto"),
                                                              "--key-env", str(c.get("key_env") or "BRAVE_API_KEY")]}
        else:
            out[name] = {k: v for k, v in c.items() if k not in ("disabled", "builtin", "label")}
    return out


IN_PROCESS = ("library", "security")


def ensure_builtins(cfg) -> None:
    """Lookup's own: the library and the security log (on; the library is empty until a collection is made), and the
    two simple built-ins, documents and web search (off until the owner turns them on)."""
    servers = dict(cfg.mcp_servers or {})
    have = {c.get("builtin") for c in servers.values() if isinstance(c, dict)}
    for b in IN_PROCESS:
        if b not in have and b not in servers:
            servers[b] = {"builtin": b, "disabled": False}
    if "docs" not in have and "docs" not in servers:
        servers["docs"] = {"builtin": "docs", "folders": [], "disabled": True}
    if "web" not in have and "web" not in servers:
        servers["web"] = {"builtin": "web", "provider": "auto", "key_env": "BRAVE_API_KEY", "disabled": True}
    cfg.mcp_servers = servers


def check_server(name: str, c: dict) -> dict:
    if not name or not isinstance(name, str) or "__" in name or "/" in name:
        raise ValueError("a server's name is a word (no __ or /)")
    if not isinstance(c, dict):
        raise ValueError("a server is an object")
    if c.get("builtin") in BUILTINS:
        out = {"builtin": c["builtin"], "disabled": bool(c.get("disabled"))}
        if c["builtin"] in IN_PROCESS:
            return out
        if c["builtin"] == "docs":
            out["folders"] = [str(f) for f in c.get("folders") or [] if str(f).strip()]
        else:
            out["provider"] = str(c.get("provider") or "auto")
            out["key_env"] = str(c.get("key_env") or "BRAVE_API_KEY")
        return out
    if c.get("url"):
        out = {"url": str(c["url"])}
        if c.get("headers"):
            if not isinstance(c["headers"], dict):
                raise ValueError('"headers" is an object')
            out["headers"] = {str(k): str(v) for k, v in c["headers"].items()}
    elif isinstance(c.get("command"), str) and c["command"].strip():
        out = {"command": c["command"].strip()}
        if c.get("args"):
            if not isinstance(c["args"], list):
                raise ValueError('"args" is a list')
            out["args"] = [str(a) for a in c["args"]]
        if c.get("env"):
            if not isinstance(c["env"], dict):
                raise ValueError('"env" is an object')
            out["env"] = {str(k): str(v) for k, v in c["env"].items()}
    else:
        raise ValueError('a server needs a "command" (a program to start) or a "url"')
    out["disabled"] = bool(c.get("disabled"))
    return out


def lookup_view(cfg, tools) -> list[dict]:
    st = tools.status() if tools is not None else {}
    running = {r.get("name"): r for r in ((st.get("servers") if isinstance(st, dict) else None) or []) if isinstance(r, dict)}
    out = []
    for name, c in (cfg.mcp_servers or {}).items():
        if not isinstance(c, dict):
            continue
        s = running.get(name) if isinstance(running, dict) else None
        kind = c.get("builtin") or ("http" if c.get("url") else "stdio")
        row = {"name": name, "kind": kind, "builtin": bool(c.get("builtin")), "enabled": c.get("disabled") is not True,
               "label": (BUILTINS.get(c.get("builtin")) or {}).get("label") or name,
               "about": (BUILTINS.get(c.get("builtin")) or {}).get("about") or (c.get("url") or " ".join([c.get("command") or ""] + list(c.get("args") or []))),
               "state": (s or {}).get("status") if isinstance(s, dict) else ("off" if c.get("disabled") is True else None),
               "tools": [{"name": t.get("tool"), "description": t.get("description")} for t in ((s or {}).get("tools") or [])]
               if isinstance(s, dict) else [],
               "error": (s or {}).get("error") if isinstance(s, dict) else None}
        if c.get("builtin") == "docs":
            row["folders"] = list(c.get("folders") or [])
        if c.get("builtin") == "web":
            row.update(provider=c.get("provider") or "auto", key_env=c.get("key_env") or "BRAVE_API_KEY")
        out.append(row)
    return out


def enabled_server_names(cfg) -> list[str]:
    return [n for n, c in (cfg.mcp_servers or {}).items() if isinstance(c, dict) and c.get("disabled") is not True]


# --- the Assistant: a built-in agent the page's chat runs as
def assistant_agent(cfg):
    """The built-in `assistant` agent, kept current with the Assistant's settings; None when the config has its own
    agent of that name (the config's wins)."""
    from nebula.config import Agent
    a = cfg.agent(BUILTIN_AGENT)
    if a is not None and not getattr(a, "builtin", False):
        return None
    d = section(cfg, "assistant")
    tools = enabled_server_names(cfg) if d.get("tools", True) else []
    if a is None:
        a = Agent(name=BUILTIN_AGENT)
        a.builtin = True
        cfg.agents.append(a)
    a.system = d.get("system") or DEFAULT_SYSTEM
    a.tools = tools
    return a


# --- a dedicated model's choices: the tiers, and what a router-tier offers
_UPSTREAM: dict = {}                                  # tier -> (read at, [model names])


def upstream_models(engine, tier: str, ttl: float = 30.0) -> list[str]:
    at, got = _UPSTREAM.get(tier, (0.0, None))
    if got is not None and time.time() - at < ttl:
        return got
    names = []
    try:
        status, _ctype, data = engine.get("/v1/models")
        if status == 200:
            d = json.loads(data)
            names = [str(m.get("id")) for m in (d.get("data") or []) if isinstance(m, dict) and m.get("id")]
    except Exception:  # noqa: BLE001 - a router that is down offers nothing now
        names = []
    _UPSTREAM[tier] = (time.time(), names)
    return names


def choices(server) -> list[dict]:
    out = []
    for t in server.cfg.tiers:
        out.append({"tier": t.name, "model": None, "label": t.name})
        eng = (getattr(server, "engines", None) or {}).get(t.name)
        if eng is not None and not t.is_command and getattr(t, "headers", None):
            for m in upstream_models(eng, t.name):
                out.append({"tier": t.name, "model": m, "label": f"{t.name} › {m}"})
    return out
