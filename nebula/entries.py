"""nebula/entries.py: the API Server's entry points (docs/entry-points.md). Each is a Quasar on the map.

Nebula's own port is the first entry point ("This PC"). More can be added from the page, each its own listener:

    "api": {"entries": [{"id": "kids", "name": "Kids", "scope": "lan", "port": 18310,
                         "formats": ["openai", "ollama"], "caller": "entry-kids", "require_key": true}]}

- **scope**: `loopback` (this machine only) or `lan` (your network: private addresses only). A key is required by
  default; one opened without a key is shown with a warning (anyone on the network can use what it offers).
  Never outside: the only door from outside is Gravity.
- **keys, models and tenancy**: every entry point is a caller of its own (`entry-<id>`, nebula/callers.py), with its
  keys and its model list (its ticks). A request is that caller's tenant, as any caller's is; marked `guest`, it gets
  only shared models and its own cache.
- **formats**: `openai` (/v1/chat/completions, /v1/responses, /v1/models), `anthropic` (/v1/messages), `ollama`
  (/api/chat, /api/generate, /api/tags).
- All of them feed the same Switchyard. A call with a bad key bounces off (401, in the security log, and shown on
  the map).
"""
from __future__ import annotations

import ipaddress
import re
import threading
import time
from collections import deque
from http.server import ThreadingHTTPServer

from nebula import callers, stars

FORMATS = {"openai": ("/v1/chat/completions", "/v1/responses", "/v1/models", "/v1/embeddings"),
           "anthropic": ("/v1/messages", "/v1/models"),
           "ollama": ("/api/chat", "/api/generate", "/api/tags", "/api/version")}
SCOPES = ("loopback", "lan")
ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")


class EntryError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def on_lan(addr: str) -> bool:
    """A private address of this network (or this machine); never a public one, nor a carrier-grade NAT/VPN range."""
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if ip.is_loopback or ip.is_link_local:
        return True
    return ip.is_private and not ip.is_global and not (ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"))


class Stats:
    def __init__(self):
        self.requests = 0
        self.inflight = 0
        self.bounces = 0
        self.recent: deque = deque(maxlen=200)          # (when, "call" | "answer" | "bounce")
        self.lock = threading.Lock()

    def mark(self, what: str):
        with self.lock:
            if what == "call":
                self.requests += 1
                self.inflight += 1
            elif what == "answer":
                self.inflight = max(0, self.inflight - 1)
            elif what == "bounce":
                self.bounces += 1
            self.recent.append((time.time(), what))

    def view(self) -> dict:
        with self.lock:
            now = time.time()
            last = [(round(t, 2), w) for t, w in self.recent if now - t < 30]
            return {"requests": self.requests, "inflight": self.inflight, "bounces": self.bounces, "recent": last,
                    "per_min": sum(1 for t, w in self.recent if w == "call" and now - t < 60)}


class EntryServer(ThreadingHTTPServer):
    """One entry point's listener; everything else (the config, the engines, the gates, the ledger) is the main
    server's."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, main, entry: dict, stats: Stats):
        from nebula.server import EntryHandler
        self.main, self.entry, self.stats = main, entry, stats
        host = "127.0.0.1" if entry["scope"] == "loopback" else "0.0.0.0"
        super().__init__((host, int(entry["port"])), EntryHandler)

    def __getattr__(self, name):
        if name in ("main", "entry", "stats"):
            raise AttributeError(name)
        return getattr(self.main, name)

    def handle_error(self, request, client_address):
        import sys
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class Entries:
    def __init__(self, server):
        self.server = server
        self.listeners: dict[str, EntryServer] = {}
        self.stats: dict[str, Stats] = {"main": Stats()}
        self.errors: dict[str, str] = {}

    @property
    def cfg(self):
        return self.server.cfg

    def specs(self) -> list[dict]:
        return list((self.cfg.api or {}).get("entries") or [])

    def spec(self, eid: str) -> dict | None:
        return next((e for e in self.specs() if e.get("id") == eid), None)

    def _write(self, rows: list[dict]):
        self.cfg.api = dict(self.cfg.api or {}, entries=rows)
        if self.cfg.path:
            self.cfg.save()

    # --- life
    def start(self) -> "Entries":
        for e in self.specs():
            self._listen(e)
        return self

    def _listen(self, e: dict):
        st = self.stats.setdefault(e["id"], Stats())
        try:
            srv = EntryServer(self.server, e, st)
        except OSError as x:
            self.errors[e["id"]] = f"port {e['port']}: {x.strerror or x}"
            return
        self.errors.pop(e["id"], None)
        self.listeners[e["id"]] = srv
        threading.Thread(target=srv.serve_forever, name=f"entry-{e['id']}", daemon=True).start()

    def _close(self, eid: str):
        srv = self.listeners.pop(eid, None)
        if srv is not None:
            srv.shutdown()
            srv.server_close()

    def close(self):
        for eid in list(self.listeners):
            self._close(eid)

    # --- changes (the owner's)
    def _check(self, req: dict, old: dict | None = None) -> dict:
        name = str(req.get("name") or (old or {}).get("name") or "").strip()[:40]
        if not name:
            raise EntryError(400, "an entry point needs a name")
        scope = req.get("scope") or (old or {}).get("scope") or "loopback"
        if scope not in SCOPES:
            raise EntryError(400, "scope is this machine (loopback) or your network (lan); never outside: that is Gravity")
        formats = req.get("formats") or (old or {}).get("formats") or ["openai"]
        if not isinstance(formats, list) or not formats or any(f not in FORMATS for f in formats):
            raise EntryError(400, f"formats are some of {', '.join(FORMATS)}")
        port = int(req.get("port") or (old or {}).get("port") or 0)
        if not 1024 <= port <= 65535:
            raise EntryError(400, "a port from 1024 to 65535")
        if port == self.server.server_port or any(e.get("port") == port and e.get("id") != (old or {}).get("id") for e in self.specs()):
            raise EntryError(409, f"port {port} is taken here")
        models = req.get("models") if "models" in req else None
        if models is not None:
            bad = [m for m in models if self.cfg.tier(m) is None]
            if bad:
                raise EntryError(404, f"no model {bad[0]!r}")
        require = bool(req.get("require_key", (old or {}).get("require_key", True)))
        rename = req.get("rename") if "rename" in req else (old or {}).get("rename") or {}
        if not isinstance(rename, dict) or any(not isinstance(k, str) or self.cfg.tier(str(v)) is None for k, v in rename.items()):
            raise EntryError(400, "rename maps a client's own model names to models here")
        return {"name": name, "scope": scope, "formats": list(dict.fromkeys(formats)), "port": port, "require_key": require,
                "models": models, "guest": bool(req.get("guest", (old or {}).get("guest", False))), "rename": dict(rename)}

    def add(self, req: dict) -> dict:
        c = self._check(req)
        eid = str(req.get("id") or re.sub(r"[^a-z0-9]+", "-", c["name"].lower()).strip("-")[:20] or "entry")
        base, n = eid, 2
        while self.spec(eid) is not None or callers.get(self.cfg, f"entry-{eid}") is not None:
            eid, n = f"{base}-{n}", n + 1
        if not ID.match(eid):
            raise EntryError(400, "an entry point's id is lowercase letters, digits and '-'")
        caller = f"entry-{eid}"
        callers.put(self.cfg, {"id": caller, "name": f"{c['name']} (entry point)", "models": c["models"] or [],
                               **({"guest": True} if c["guest"] else {})})
        row = {"id": eid, "name": c["name"], "scope": c["scope"], "port": c["port"], "formats": c["formats"],
               "caller": caller, "require_key": c["require_key"], **({"rename": c["rename"]} if c["rename"] else {})}
        self._write(self.specs() + [row])
        self._listen(row)
        return self.view_one(row)

    def change(self, eid: str, req: dict) -> dict:
        old = self.spec(eid)
        if old is None:
            raise EntryError(404, f"no entry point {eid!r}")
        cur = callers.get(self.cfg, old["caller"]) or {}
        c = self._check({**req, "models": req.get("models", cur.get("models") or []), "guest": req.get("guest", cur.get("guest", False))}, old)
        callers.put(self.cfg, {"id": old["caller"], "name": f"{c['name']} (entry point)", "models": c["models"] or [],
                               **({"guest": True} if c["guest"] else {})})
        row = {**old, "name": c["name"], "scope": c["scope"], "port": c["port"], "formats": c["formats"], "require_key": c["require_key"]}
        row.pop("rename", None)
        if c["rename"]:
            row["rename"] = c["rename"]
        self._write([row if e["id"] == eid else e for e in self.specs()])
        if (old["scope"], old["port"]) != (row["scope"], row["port"]):
            self._close(eid)
            self._listen(row)
        else:
            srv = self.listeners.get(eid)
            if srv is not None:
                srv.entry = row
        return self.view_one(row)

    def remove(self, eid: str) -> dict:
        old = self.spec(eid)
        if old is None:
            raise EntryError(404, f"no entry point {eid!r}")
        self._close(eid)
        callers.remove(self.cfg, old["caller"])
        self._write([e for e in self.specs() if e["id"] != eid])
        self.stats.pop(eid, None)
        return {"removed": eid}

    def make_key(self, eid: str, label: str = "") -> dict:
        e = self.spec(eid)
        if e is None:
            raise EntryError(404, f"no entry point {eid!r}")
        row, key = callers.make_key(self.cfg, e["caller"], label or e["name"])
        if self.cfg.path:
            self.cfg.save()
        return {"key": key, "row": {k: v for k, v in row.items() if k != "sha256"}, "note": "shown once: give it to the app"}

    def revoke_key(self, eid: str, kid: str) -> dict:
        e = self.spec(eid)
        if e is None:
            raise EntryError(404, f"no entry point {eid!r}")
        ok = callers.revoke_key(self.cfg, e["caller"], kid)
        if self.cfg.path:
            self.cfg.save()
        return {"revoked": ok}

    # --- views
    def view_one(self, e: dict) -> dict:
        c = callers.get(self.cfg, e["caller"]) or {}
        return {"id": e["id"], "name": e["name"], "scope": e["scope"], "port": e["port"], "formats": e["formats"], "caller": e["caller"],
                "require_key": e.get("require_key", True), "models": c.get("models") or [], "guest": bool(c.get("guest")),
                "rename": dict(e.get("rename") or {}),
                "keys": [{k: v for k, v in r.items() if k != "sha256"} for r in c.get("keys") or []],
                "listening": e["id"] in self.listeners, "error": self.errors.get(e["id"]),
                "warning": (("open on your network: anyone on it can use what it offers without a key" if e["scope"] == "lan"
                             else "open on this machine: any app here can use it without a key") if not e.get("require_key", True) else None),
                "stats": self.stats.setdefault(e["id"], Stats()).view()}

    def view(self) -> dict:
        api = self.cfg.api or {}
        main = {"id": "main", "name": "This PC", "scope": "loopback" if self.cfg.host in ("127.0.0.1", "localhost", "::1") else "lan",
                "port": self.server.server_port, "formats": ["openai", "anthropic"], "require_key": stars.requires_key(self.cfg),
                "models": [], "main": True, "listening": True, "stats": self.stats["main"].view()}
        return {"entries": [main] + [self.view_one(e) for e in self.specs()]}

    def allowed(self, entry: dict, path: str) -> bool:
        return path == "/health" or any(path == p for f in entry["formats"] for p in FORMATS[f])

    def entry_key_ok(self, entry: dict, presented: str | None) -> bool:
        if not presented:
            return False
        who = callers.by_key(self.cfg, presented)
        return who is not None and who.get("id") == entry["caller"]


def get(server, path: str):
    if path == "/api/api-servers":
        return server.entries.view()
    return None


def post(server, path: str, req: dict):
    ents = server.entries
    if path == "/api/api-servers":
        return ents.add(req)
    m = re.match(r"^/api/api-servers/([^/]+)(?:/(keys|remove)(?:/([^/]+)/revoke)?)?$", path)
    if not m:
        return None
    eid, action, kid = m.groups()
    if eid == "main":
        raise EntryError(400, "This PC is Nebula's own port: its key and model are on the API Server's panel")
    if action is None:
        return ents.change(eid, req)
    if action == "remove":
        return ents.remove(eid)
    if action == "keys" and kid:
        return ents.revoke_key(eid, kid)
    if action == "keys":
        return ents.make_key(eid, str(req.get("label") or ""))
    return None

