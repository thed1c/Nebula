"""nebula/gravity.py: the Gravity node, Nebula to Nebula over iroh (docs/gravity.md).

Every Nebula has two doors. The **API Node** is the HTTP API (apps and agents, a key per caller). The **Gravity node**
is this one: other Nebulas (a friend's, your own laptop, a headless server you own), machine to machine, over iroh
QUIC. A node is its key: an Ed25519 key kept in `nebula-keys/gravity.key` (0600), whose public half is its id.

- **Nothing is found on the network.** No mDNS, no n0 DNS, no DHT (iroh's minimal preset). A Nebula learns of another
  only from an **invite** its owner made (a ticket with the inviter's addresses, the gateway's relay if there is one,
  and a one-time secret), shown as text to copy or a QR code.
- **On the LAN the two dial each other directly; elsewhere through the owner's gateway** (a self-hosted iroh relay:
  `relays`), or, when the owner turns it on (`public_relay`), iroh's free public relays (n0's): they forward the
  same end-to-end encrypted traffic and cannot read it. A gateway of your own wins over the public relays.
- **A peer gets nothing until ticked.** Each model the owner ticks for a peer is one grant; the peer may only use
  granted models that are shared with guests (docs/sharing.md: its own cache, its own history).
- **One ALPN per capability**, and an ALPN with no handler is refused at the socket:
  `gravity/enroll/1` (joining by invite), `gravity/models/1` (the models granted to the asking key, and the names of
  the rest: a member sees every model in the constellation, with "no route" to those not shared with it),
  `gravity/load/1` (pending work, for the peer's scheduler), `gravity/infer/1` (a request and its stream),
  `gravity/admin/1` (owner only: a peer that joined by an owner invite, e.g. a headless server you installed, may
  drive this Nebula's /api/yard over it).
- **Revoking** a peer removes its key; its open connections are closed and every request is checked anyway.

A peer's models are used through a loopback bridge (`http://127.0.0.1:<port>/peer/<id>`), so they are ordinary models
here (mesh "gravity-peer"), routed, queued and recorded like any other.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import queue
import secrets
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = logging.getLogger("nebula.gravity")
ENROLL, MODELS, LOAD, INFER, ADMIN = (b"gravity/enroll/1", b"gravity/models/1", b"gravity/load/1",
                                      b"gravity/infer/1", b"gravity/admin/1")
INVITE_PREFIX = "nebula-invite:"
MAX_REQ = 32 << 20


_LOOP: asyncio.AbstractEventLoop | None = None       # one loop for every node in this process: iroh's callbacks
_LOOP_LOCK = threading.Lock()                        # find their loop through one process-wide setting


def shared_loop() -> asyncio.AbstractEventLoop:
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None:
            _LOOP = asyncio.new_event_loop()
            threading.Thread(target=_LOOP.run_forever, name="gravity", daemon=True).start()
            from iroh import iroh_ffi
            iroh_ffi.uniffi_set_event_loop(_LOOP)
        return _LOOP


def anti_gravity(tier) -> str | None:
    """Why a model may never be shared over Gravity (cloud: your API key, your bill, the provider's terms), else None."""
    if getattr(tier, "provider", None) or (getattr(tier, "base_url", None) and not getattr(tier, "engine", None)):
        return "Anti-Gravity: a cloud model (your API key, your bill, the provider's terms)"
    if getattr(tier, "is_command", False):
        return "Anti-Gravity: a CLI that runs on your own account"
    return None


def needs_care(tier, engines=None) -> str | None:
    """Why sharing a local model warns both ends, else None: the owner marked it, or Nebula does not run its engine
    (so it cannot set that engine's isolation), or it is another router of machines, or its engine runs without a
    sandbox (design decision, 2026-10-20: shared, the other side sees that warning too)."""
    managed = getattr(tier, "managed", None)
    if managed and engines is not None and managed in getattr(engines, "unsandboxed", {}):
        return "its engine runs without a sandbox: " + engines.unsandboxed[managed]
    if getattr(tier, "risky", False):
        return "you marked it as needing care"
    if getattr(tier, "mesh", None):
        return "it is a router of other machines: Nebula cannot keep tenants apart there"
    if not getattr(tier, "managed", None):
        return "Nebula does not run its engine, so it cannot lock it down or keep tenants apart in it"
    return None


class GravityError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def available() -> tuple[bool, str | None]:
    try:
        import iroh  # noqa: F401
        return True, None
    except ImportError:
        return False, 'the Gravity node needs iroh: pip install "nebula-eh[gravity]"'


def _sha(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()


async def _read_all(recv, limit=MAX_REQ) -> bytes:
    return await recv.read_to_end(limit)


async def _reply(bi, obj: dict) -> None:
    await bi.send().write_all(json.dumps(obj).encode())
    await bi.send().finish()


class GravityNode:
    def __init__(self, server):
        self.server, self.cfg = server, server.cfg
        self.ok, self.why = available()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ep = None
        self.bridge = None
        self.conns: dict[tuple[str, bytes], object] = {}
        self.lock = threading.Lock()
        self.seen: dict[str, dict] = {}                  # what each member offers this one: {offers, unrouted, reachable, at}
        self._look = threading.Event()
        self._stop = threading.Event()

    # --- the config section ("gravity" in the config file)
    def section(self) -> dict:
        g = self.cfg.raw.setdefault("gravity", {})
        g.setdefault("peers", [])
        g.setdefault("invites", [])
        g.setdefault("relays", [])
        return g

    def save(self) -> None:
        if self.cfg.path:
            self.cfg.save()

    def members_changed(self) -> None:
        """A peer joined or was removed: a gateway this Nebula runs lets in exactly its members' keys."""
        self.look_soon()
        gw = self.section().get("gateway") or {}
        if not gw.get("managed"):
            return
        from nebula import gateway
        members = [self.node_id()] + [p["id"] for p in self.section()["peers"]]
        try:
            gateway.update_members(gw["dir"], members, gw.get("hostname"), gw.get("email"), int(gw.get("port") or 3340))
        except Exception as e:  # noqa: BLE001 - the page shows the gateway's state
            LOG.warning("the gateway's members were not updated: %s", e)

    def peers(self) -> list[dict]:
        return list(self.section()["peers"])

    def peer(self, pid: str) -> dict | None:
        return next((p for p in self.section()["peers"] if p["id"] == pid), None)

    # --- the key and the endpoint
    def key_path(self) -> str | None:
        from nebula import names
        if not self.cfg.path:
            return None
        return os.path.join(names.local_path(self.cfg.base_dir, "nebula-keys", "cockpit-keys"), "gravity.key")

    def secret(self) -> bytes:
        import iroh
        p = self.key_path()
        if p and os.path.exists(p):
            return open(p, "rb").read()
        raw = iroh.SecretKey.generate().to_bytes()
        if p:
            os.makedirs(os.path.dirname(p), mode=0o700, exist_ok=True)
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
        else:
            self._mem_secret = raw
        return raw if p else self._mem_secret

    def start(self) -> "GravityNode":
        if not self.ok:
            return self
        self.loop = shared_loop()
        try:
            self.ep = self._run(self._bind(), 30)
        except Exception as e:  # noqa: BLE001 - shown on the page
            self.ok, self.why = False, f"the Gravity node could not start: {e}"
            LOG.warning(self.why)
            return self
        self.bridge = _Bridge(self)
        threading.Thread(target=self._looking, name="gravity-look", daemon=True).start()
        return self

    # --- the constellation: every member's models as they offer them to this one (the map draws them)
    LOOK_S = 20.0

    def look_soon(self) -> None:
        self._look.set()

    def _looking(self) -> None:
        while not self._stop.is_set():
            for p in self.peers():
                if self._stop.is_set():
                    return
                self.look(p["id"])
            self._look.wait(self.LOOK_S)
            self._look.clear()

    def look(self, pid: str) -> dict:
        """Ask one member what it offers this Nebula: its models granted here (a route) and the names of the rest (no route)."""
        was = self.seen.get(pid) or {"offers": [], "unrouted": []}
        try:
            r = self.ask(pid, MODELS, {}, timeout=10)
            now = {"offers": [m["name"] for m in r.get("models") or [] if m.get("name")],
                   "unrouted": [str(n) for n in r.get("unrouted") or []], "reachable": True, "at": time.time()}
        except Exception as e:  # noqa: BLE001 - away: keep what it offered last, marked away
            now = {**was, "reachable": False, "error": str(e)[:200], "at": time.time()}
        self.seen[pid] = now
        return now

    def _run(self, coro, timeout: float = 60):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _bind(self):
        import iroh
        relays = [r for r in self.section()["relays"] if isinstance(r, str) and r.startswith(("http://", "https://"))]
        node = self

        class Creator(iroh.ProtocolCreator):
            def __init__(self, alpn):
                self.alpn = alpn

            def create(self, endpoint):
                return _Handler(node, self.alpn)
        port = int(self.section().get("port") or 0)
        opts = iroh.EndpointOptions(
            preset=iroh.preset_minimal(),                               # no n0 DNS, no DHT, no mDNS
            relay_mode=(iroh.RelayMode.custom_from_urls(relays) if relays else
                        iroh.RelayMode.default_mode() if self.section().get("public_relay") else iroh.RelayMode.disabled()),
            bind_addr=f"0.0.0.0:{port}", secret_key=self.secret(),
            protocols={a: Creator(a) for a in (ENROLL, MODELS, LOAD, INFER, ADMIN)})
        ep = await iroh.Endpoint.bind(opts)
        if relays or self.section().get("public_relay"):
            try:
                await asyncio.wait_for(ep.online(), 10)
            except Exception:  # noqa: BLE001 - the relay may be down; direct paths still work
                pass
        return ep

    def node_id(self) -> str | None:
        return str(self.ep.id()) if self.ep is not None else None

    def addr(self) -> dict:
        a = self.ep.addr()
        return {"id": str(a.id()), "relay": a.relay_url(), "direct": list(a.direct_addresses())}

    def close(self) -> None:
        self._stop.set(); self._look.set()
        if self.bridge is not None:
            self.bridge.close()
        if self.ep is not None and self.loop is not None:
            try:
                self._run(self.ep.close(), 10)
            except Exception:  # noqa: BLE001
                pass

    # --- invites
    def invite(self, kind: str = "friend", ttl_s: float = 3600, name: str | None = None, grants: list | None = None,
               confirm: list | None = None) -> dict:
        """A one-time invite. A friend's may carry the models it is given on joining (`grants`, checked now as a tick
        is: never a cloud model, a model that needs care only when `confirm` names it)."""
        if not self.ok or self.ep is None:
            raise GravityError(503, self.why or "the Gravity node is off")
        if kind not in ("friend", "owner"):
            raise GravityError(400, "an invite is for a friend or for one of your own machines (owner)")
        grants = list(dict.fromkeys(grants or []))
        if grants:
            self.check_grants(grants, confirm)
        secret = secrets.token_urlsafe(32)
        g = self.section()
        iid = secrets.token_hex(4)
        g["invites"] = [i for i in g["invites"] if i.get("expires", 0) > time.time() and not i.get("used")] + \
            [{"id": iid, "sha256": _sha(secret), "kind": kind, "expires": time.time() + ttl_s, "for": name,
              **({"grants": grants} if grants else {})}]
        self.save()
        blob = {"v": 1, "addr": self.addr(), "secret": secret, "kind": kind, "name": self.my_name()}
        text = INVITE_PREFIX + base64.urlsafe_b64encode(json.dumps(blob, separators=(",", ":")).encode()).decode().rstrip("=")
        return {"invite": text, "id": iid, "kind": kind, "expires_in_s": ttl_s, "grants": grants,
                "note": "it works once; anyone holding it can join until then, so send it privately"}

    def cancel_invite(self, iid: str) -> bool:
        g = self.section()
        before = len(g["invites"])
        g["invites"] = [i for i in g["invites"] if i.get("id") != iid]
        self.save()
        return len(g["invites"]) != before

    def check_grants(self, models: list, confirm: list | None = None) -> None:
        for m in models:
            t = self.cfg.tier(m)
            if t is None:
                raise GravityError(404, f"no model {m}")
            why = anti_gravity(t)
            if why:
                raise GravityError(400, f"{m}: {why}. It cannot be shared.")
            if not getattr(t, "shared", False):
                raise GravityError(400, f"{m} is not shared with guests: mark it shared first (its own cache per guest)")
        care = {m: needs_care(self.cfg.tier(m)) for m in models}
        unconfirmed = [m for m, why in care.items() if why and m not in (confirm or [])]
        if unconfirmed:
            e = GravityError(409, "; ".join(f"{m}: sharing it may not be safe ({care[m]}). Confirm to share it anyway"
                                            for m in unconfirmed))
            e.needs_confirm = unconfirmed
            raise e

    def my_name(self) -> str:
        return str(self.section().get("name") or (self.cfg.lan or {}).get("name") or "Nebula")

    @staticmethod
    def parse_invite(text: str) -> dict:
        t = text.strip()
        if not t.startswith(INVITE_PREFIX):
            raise GravityError(400, "not a Nebula invite")
        b = t[len(INVITE_PREFIX):]
        try:
            return json.loads(base64.urlsafe_b64decode(b + "=" * (-len(b) % 4)))
        except ValueError:
            raise GravityError(400, "the invite is damaged") from None

    def join(self, text: str) -> dict:
        """Join the Nebula that made the invite: it pins this node's key and this node pins its."""
        if not self.ok or self.ep is None:
            raise GravityError(503, self.why or "the Gravity node is off")
        inv = self.parse_invite(text)
        a = inv["addr"]
        if a["id"] == self.node_id():
            raise GravityError(400, "that invite is this Nebula's own")
        reply = self._run(self._ask(a, ENROLL, {"secret": inv["secret"], "name": self.my_name(), "addr": self.addr()}), 60)
        if not reply.get("ok"):
            raise GravityError(403, reply.get("why") or "the invite was refused")
        row = {"id": a["id"], "name": reply.get("name") or inv.get("name") or a["id"][:8], "addr": a, "grants": [],
               "role": "admin" if inv.get("kind") == "owner" else "friend", "joined_at": time.time()}
        # an owner invite: the inviter is one of my machines and I (who joined) may administer it; it does not
        # administer me. A friend invite: both sides are friends with no grants.
        g = self.section()
        g["peers"] = [p for p in g["peers"] if p["id"] != row["id"]] + [row]
        self.save()
        self.members_changed()
        return row

    # --- grants and revoking
    def set_grants(self, pid: str, models: list[str], confirm: list[str] | None = None) -> dict:
        """Tick models for a peer. A cloud model never (Anti-Gravity); a local model that needs care only when the
        owner confirmed its warning (`confirm` names it); a normal local model by the tick alone."""
        p = self.peer(pid)
        if p is None:
            raise GravityError(404, f"no peer {pid}")
        for m in models:
            t = self.cfg.tier(m)
            if t is None:
                raise GravityError(404, f"no model {m}")
            why = anti_gravity(t)
            if why:
                raise GravityError(400, f"{m}: {why}. It cannot be shared.")
            if not getattr(t, "shared", False):
                raise GravityError(400, f"{m} is not shared with guests: mark it shared first (its own cache per guest)")
        new = [m for m in models if m not in (p.get("grants") or [])]
        care = {m: needs_care(self.cfg.tier(m)) for m in new}
        unconfirmed = [m for m, why in care.items() if why and m not in (confirm or [])]
        if unconfirmed:
            e = GravityError(409, "; ".join(f"{m}: sharing it may not be safe ({care[m]}). Confirm to share it anyway"
                                            for m in unconfirmed))
            e.needs_confirm = unconfirmed
            raise e
        p["grants"] = list(dict.fromkeys(models))
        self.look_soon()
        self.save()
        return p

    def revoke(self, pid: str) -> bool:
        g = self.section()
        before = len(g["peers"])
        g["peers"] = [p for p in g["peers"] if p["id"] != pid]
        self.save()
        self.members_changed()
        with self.lock:
            for k in [k for k in self.conns if k[0] == pid]:
                c = self.conns.pop(k)
                try:
                    c.close(0, b"revoked")
                except Exception:  # noqa: BLE001
                    pass
        from nebula.server import reload_models
        self.cfg.tiers = [t for t in self.cfg.tiers if not (getattr(t, "mesh", None) == "gravity-peer" and (t.engine or "").endswith("/peer/" + pid))]
        reload_models(self.server)
        return len(g["peers"]) != before

    # --- asking a peer
    async def _conn(self, addr: dict, alpn: bytes):
        import iroh
        key = (addr["id"], alpn)
        with self.lock:
            c = self.conns.get(key)
        if c is not None and c.close_reason() is None:
            return c
        ea = iroh.EndpointAddr(iroh.EndpointId.from_string(addr["id"]), addr.get("relay"), list(addr.get("direct") or []))
        c = await asyncio.wait_for(self.ep.connect(ea, alpn), 20)
        with self.lock:
            self.conns[key] = c
        return c

    async def _ask(self, addr: dict, alpn: bytes, payload: dict) -> dict:
        c = await self._conn(addr, alpn)
        bi = await c.open_bi()
        await bi.send().write_all(json.dumps(payload).encode())
        await bi.send().finish()
        return json.loads(await _read_all(bi.recv()))

    def ask(self, pid: str, alpn: bytes, payload: dict, timeout: float = 30) -> dict:
        p = self.peer(pid)
        if p is None:
            raise GravityError(404, f"no peer {pid}")
        try:
            return self._run(self._ask(p["addr"], alpn, payload), timeout)
        except GravityError:
            raise
        except Exception as e:  # noqa: BLE001
            raise GravityError(502, f"{p['name']} cannot be reached: {e}") from None

    def stream(self, pid: str, alpn: bytes, payload: dict, out: "queue.Queue"):
        """A request whose answer streams: (status, content type) first, then the body's chunks, then None."""
        p = self.peer(pid)
        if p is None:
            out.put(GravityError(404, f"no peer {pid}"))
            return

        async def go():
            try:
                c = await self._conn(p["addr"], alpn)
                bi = await c.open_bi()
                await bi.send().write_all(json.dumps(payload).encode())
                await bi.send().finish()
                recv, buf = bi.recv(), b""
                while b"\n" not in buf:
                    chunk = await recv.read(65536)
                    if not chunk:
                        raise GravityError(502, "the peer closed before answering")
                    buf += chunk
                head, _, rest = buf.partition(b"\n")
                out.put(json.loads(head))
                if rest:
                    out.put(rest)
                while True:
                    chunk = await recv.read(65536)
                    if not chunk:
                        break
                    out.put(chunk)
                out.put(None)
            except Exception as e:  # noqa: BLE001
                out.put(e if isinstance(e, GravityError) else GravityError(502, f"{p['name']}: {e}"))
        asyncio.run_coroutine_threadsafe(go(), self.loop)

    def adopt(self, pid: str) -> dict:
        """The models this peer granted, as models here (through the loopback bridge)."""
        from nebula.config import Tier
        from nebula.server import reload_models
        import re
        p = self.peer(pid)
        if p is None:
            raise GravityError(404, f"no peer {pid}")
        offered = self.ask(pid, MODELS, {}).get("models") or []
        made = []
        for m in offered:
            name = re.sub(r"[^a-z0-9._-]+", "-", f"{p['name']}-{m['name']}".lower()).strip("-")[:48]
            if self.cfg.tier(name) is None:
                self.cfg.tiers.append(Tier(name=name, engine=f"{self.bridge.url}/peer/{pid}", model=m["name"],
                                           mesh="gravity-peer", health_path="/v1/models", context=m.get("context"),
                                           slots=int(m.get("slots") or 1), not_guaranteed_safe=bool(m.get("not_guaranteed_safe"))))
                made.append(name)
        reload_models(self.server)
        if self.cfg.path:
            self.cfg.save()
        warn = [f"{p['name']}'s {m['name']} is not guaranteed safe: {m.get('why')}" for m in offered if m.get("not_guaranteed_safe")]
        return {"made": made, "offered": [m["name"] for m in offered], "warnings": warn}

    def view(self) -> dict:
        g = self.section()
        gw = g.get("gateway") or {}
        return {"available": self.ok, "why": self.why, "id": self.node_id(), "name": self.my_name(),
                "gateway": {k: gw.get(k) for k in ("url", "hostname", "managed")} if gw else None,
                "addr": self.addr() if self.ep is not None else None, "relays": g["relays"], "public_relay": bool(g.get("public_relay")),
                "peers": [{**{k: v for k, v in p.items() if k != "addr"}, **self.seen.get(p["id"], {})} for p in g["peers"]],
                "open_invites": len([i for i in g["invites"] if i.get("expires", 0) > time.time() and not i.get("used")]),
                "bridge": self.bridge.url if self.bridge else None}


class _Handler:
    """The answer to one connection on one ALPN (iroh calls accept per connection)."""

    def __init__(self, node: GravityNode, alpn: bytes):
        self.node, self.alpn = node, alpn

    async def accept(self, conn):
        remote = str(conn.remote_id())
        try:
            while True:
                try:
                    bi = await conn.accept_bi()
                except Exception:  # noqa: BLE001 - the peer closed the connection
                    return
                asyncio.ensure_future(self._one(remote, bi, conn))
        except Exception as e:  # noqa: BLE001
            LOG.warning("gravity %s from %s: %s", self.alpn.decode(), remote[:8], e)

    async def shutdown(self):
        pass

    async def _one(self, remote: str, bi, conn):
        node = self.node
        try:
            req = json.loads(await _read_all(bi.recv()) or b"{}")
        except ValueError:
            return await _reply(bi, {"ok": False, "why": "not JSON"})
        if self.alpn == ENROLL:
            return await _reply(bi, self._enroll(remote, req))
        p = node.peer(remote)
        if p is None:                                         # not (or no longer) a peer: nothing at all
            await _reply(bi, {"ok": False, "why": "not a peer of this Nebula"})
            conn.close(1, b"not a peer")
            return
        if self.alpn == MODELS:
            return await _reply(bi, {"ok": True, "models": self._granted(p), "unrouted": self._unrouted(p)})
        if self.alpn == LOAD:
            return await _reply(bi, {"ok": True, "pending": self._pending(p)})
        if self.alpn == INFER:
            model = (req.get("body") or {}).get("model")
            if model not in [m["name"] for m in self._granted(p)]:
                if getattr(node.server, "seclog", None) is not None:
                    node.server.seclog.add("guest.refused", f"peer:{remote}", f"model:{model}", via="gravity")
                return await self._head(bi, 403, {"error": {"message": f"{model!r} is not granted to you", "type": "gravity"}})
            return await self._forward(bi, req, {"X-Nebula-From": remote, "X-Nebula-Internal": node.server.internal_token})
        if self.alpn == ADMIN:
            if p.get("role") != "owner":
                if getattr(node.server, "seclog", None) is not None:
                    node.server.seclog.add("admin.refused", f"peer:{remote}", str(req.get("path") or "")[:60])
                return await self._head(bi, 403, {"error": {"message": "only the owner's own machines administer this Nebula"}})
            if not str(req.get("path") or "").startswith("/api/yard/"):
                return await self._head(bi, 403, {"error": {"message": "admin reaches /api/yard only"}})
            return await self._forward(bi, req, {"X-Nebula-Internal": node.server.internal_token})

    def _enroll(self, remote: str, req: dict) -> dict:
        node = self.node
        g = node.section()
        h = _sha(str(req.get("secret") or ""))
        inv = next((i for i in g["invites"] if i["sha256"] == h and not i.get("used") and i["expires"] > time.time()), None)
        log = getattr(node.server, "seclog", None)
        if inv is None:
            if log is not None:
                log.add("gravity.refused", f"node:{remote[:16]}", "enroll", why="an invite that is not valid")
            return {"ok": False, "why": "this invite is not valid (used, expired or never made here)"}
        inv["used"] = time.time()
        addr = req.get("addr") if isinstance(req.get("addr"), dict) and req["addr"].get("id") == remote else {"id": remote}
        granted = [m for m in inv.get("grants") or [] if node.cfg.tier(m) is not None and getattr(node.cfg.tier(m), "shared", False)
                   and not anti_gravity(node.cfg.tier(m))]                      # the invite's models, still shareable
        row = {"id": remote, "name": str(req.get("name") or remote[:8])[:60], "addr": addr, "grants": granted,
               # an owner invite: the joiner is the owner's own machine and administers this one
               "role": "owner" if inv["kind"] == "owner" else "friend", "joined_at": time.time()}
        g["peers"] = [p for p in g["peers"] if p["id"] != remote] + [row]
        node.save()
        node.members_changed()
        if log is not None:
            log.add("gravity.join", f"peer:{remote}", row["role"], name=row["name"], invite_kind=inv["kind"])
        return {"ok": True, "name": node.my_name(), "id": node.node_id()}

    def _granted(self, p: dict) -> list[dict]:
        out = []
        for m in p.get("grants") or []:
            t = self.node.cfg.tier(m)
            if t is not None and getattr(t, "shared", False) and not anti_gravity(t):
                care = needs_care(t, getattr(self.node.server, "yard", None))
                out.append({"name": m, "context": t.context, "slots": t.slots,
                            **({"not_guaranteed_safe": True, "why": care} if care else {})})
        return out

    def _unrouted(self, p: dict) -> list[str]:
        """This Nebula's own models that this member may see but has no route to: everything not granted to it (or
        granted but not shared). Cloud models and commands are Anti-Gravity and never listed; nor are a member's models
        reached here through Gravity (they are theirs)."""
        routed = {m["name"] for m in self._granted(p)}
        return [t.name for t in self.node.cfg.tiers
                if t.name not in routed and not anti_gravity(t) and getattr(t, "mesh", None) != "gravity-peer" and t.engine]

    def _pending(self, p: dict) -> dict:
        adm = self.node.server.admission
        out = {}
        for m in p.get("grants") or []:
            t = self.node.cfg.tier(m)
            if t is not None and adm is not None:
                st = adm.gate(t).status()
                out[m] = st["held"] + len(st["waiting"])
        return out

    async def _head(self, bi, status: int, body: dict):
        await bi.send().write_all(json.dumps({"status": status, "content_type": "application/json"}).encode() + b"\n" + json.dumps(body).encode())
        await bi.send().finish()

    async def _forward(self, bi, req: dict, headers: dict):
        """The request to this Nebula's own API on loopback (as the peer's tenant), its answer streamed back."""
        loop = asyncio.get_running_loop()
        node = self.node
        url = f"http://127.0.0.1:{node.server.server_port}{req.get('path') or '/v1/chat/completions'}"
        body = req.get("body")
        data = None if body is None else json.dumps(body).encode()
        q: "queue.Queue" = queue.Queue()

        def call():
            r = urllib.request.Request(url, data=data, method=req.get("method") or ("POST" if data else "GET"),
                                       headers={"Content-Type": "application/json", **headers})
            try:
                resp = urllib.request.urlopen(r, timeout=900)
            except urllib.error.HTTPError as e:
                q.put((e.code, e.headers.get("Content-Type") or "application/json"))
                q.put(e.read())
                q.put(None)
                return
            except OSError as e:
                q.put((502, "application/json"))
                q.put(json.dumps({"error": {"message": str(e)}}).encode())
                q.put(None)
                return
            q.put((resp.status, resp.headers.get("Content-Type") or "application/json"))
            with resp:
                while True:
                    chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                    if not chunk:
                        break
                    q.put(chunk)
            q.put(None)
        loop.run_in_executor(None, call)
        status, ctype = await loop.run_in_executor(None, q.get)
        send = bi.send()
        await send.write_all(json.dumps({"status": status, "content_type": ctype}).encode() + b"\n")
        while True:
            chunk = await loop.run_in_executor(None, q.get)
            if chunk is None:
                break
            await send.write_all(chunk)
        await send.finish()


class _Bridge:
    """Loopback HTTP in front of the peers: /peer/<id>/... is that peer's granted models over iroh; /admin/<id>/... is
    its /api/yard (an owner machine's only). Bound to 127.0.0.1: only this machine's Nebula uses it."""

    def __init__(self, node: GravityNode):
        bridge = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def _err(self, e: GravityError):
                b = json.dumps({"error": {"message": str(e), "type": "gravity"}}).encode()
                self.send_response(e.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def _route(self, method: str):
                parts = self.path.split("?")[0].split("/", 3)          # "", "peer"|"admin", id, rest
                if len(parts) < 4 or parts[1] not in ("peer", "admin"):
                    return self._err(GravityError(404, "no such path"))
                kind, pid, rest = parts[1], parts[2], "/" + parts[3]
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else None
                try:
                    if kind == "peer" and rest in ("/health", "/v1/models"):
                        models = node.ask(pid, MODELS, {}, timeout=20).get("models") or []
                        obj = {"status": "ok"} if rest == "/health" else \
                            {"object": "list", "data": [{"id": m["name"], "object": "model", "owned_by": "gravity"} for m in models]}
                        b = json.dumps(obj).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(b)))
                        self.end_headers()
                        return self.wfile.write(b)
                    if kind == "peer" and rest == "/props":
                        return self._err(GravityError(404, "no props through Gravity"))
                    q: "queue.Queue" = queue.Queue()
                    payload = {"path": rest, "method": method, "body": body}
                    node.stream(pid, INFER if kind == "peer" else ADMIN, payload, q)
                    first = q.get(timeout=900)
                    if isinstance(first, Exception):
                        raise first
                    self.send_response(int(first.get("status") or 502))
                    self.send_header("Content-Type", first.get("content_type") or "application/json")
                    self.end_headers()
                    while True:
                        chunk = q.get(timeout=900)
                        if chunk is None:
                            break
                        if isinstance(chunk, Exception):
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except GravityError as e:
                    return self._err(e)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                self._route("GET")

            def do_POST(self):
                self._route("POST")

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, name="gravity-bridge", daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
