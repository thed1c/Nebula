"""nebula/server.py: the HTTP server. Routes, request parsing, each dialect's error shape, SSE writing.

Every POST is recorded in the ledger (nebula/ledger.py) when one is configured: its rows before the engine is
called, the reply after. GET / is the cockpit's own page and /api/... its JSON API (nebula/ui.py); POST /hooks/<name> is a webhook input
(nebula/hooks.py). GET /health, /props, /slots and /metrics
are the engine's, passed through. GET /v1/models is the engine's list
with the tiers' names and the configured aliases added. POST /v1/chat/completions, /v1/messages and /v1/responses
are the dialects. A streamed reply's first frame is read before the 200 goes out, so an engine that refuses the
request (a prompt that does not fit is its 400) is answered with that status and the dialect's error shape, not
with a 200 and an error inside the stream.
"""
from __future__ import annotations

import json
from pathlib import Path
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nebula import names, stars
from nebula._vendor.strata_serve.responses import ResponsesError, error_body

from nebula import __version__
from nebula.dialects import Context, JsonReply, sse_frame
from nebula.dialects.chat import chat
from nebula.dialects.messages import messages
from nebula.dialects.responses import responses
from nebula.dialects.embeddings import embeddings
from nebula import hooks, origin, seclog, signin, ui
from nebula.agents import Tools
from nebula.engine import Engine, EngineError
from nebula.enginelog import Follower
from nebula.errors import ErrorLog
from nebula.ledger import Ledger
from nebula import lan
from nebula.admission import Admission
from nebula.parking import Lot
from nebula.registry import Registry
from nebula.router import Rules

LOG = logging.getLogger("cockpit")
PASSTHROUGH = ("/health", "/props", "/slots", "/metrics")
POST_ROUTES = {"/v1/chat/completions": ("chat", chat), "/v1/messages": ("messages", messages),
               "/v1/responses": ("responses", responses), "/v1/embeddings": ("chat", embeddings)}
MAX_BODY = 512 * 2**20                        # a 262k-token prompt with images is tens of MB


def error_shape(dialect: str, status: int, message: str, param=None, code=None) -> dict:
    if dialect == "messages":
        return {"type": "error", "error": {"type": "invalid_request_error" if status < 500 else "api_error",
                                           "message": message}}
    kind = "invalid_request_error" if status < 500 else "server_error"
    if dialect == "responses":
        return error_body(message, kind, param, code)
    return {"error": {"message": message, "type": kind, "param": param, "code": code}}


class Handler(BaseHTTPRequestHandler):
    server_version = f"nebula/{__version__}"

    def log_message(self, fmt, *args):
        LOG.info("%s " + fmt, self.address_string(), *args)

    @property
    def cfg(self):
        return self.server.cfg

    @property
    def engine(self) -> Engine:
        return self.server.engine

    # --- writing
    def _bytes(self, status: int, ctype: str, data: bytes, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, obj, headers: dict | None = None) -> None:
        self._bytes(status, "application/json", json.dumps(obj, ensure_ascii=False).encode("utf-8"), headers)

    # --- who is asking: this machine, or a paired one
    def _lan_ok(self, path: str) -> bool:
        """A request from beyond this machine needs a paired token; the pairing route is open while its window is."""
        addr = self.client_address[0] if self.client_address else "127.0.0.1"
        if lan.is_loopback(addr) and not getattr(self.server.cfg, "lan_auth_loopback", False):
            return True
        pairing = getattr(self.server, "pairing", None)
        auth = self.headers.get("Authorization") or ""
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else (names.header(self.headers, "Token") or "")
        p = pairing.allowed(token) if pairing is not None else None
        if p is None:
            return False
        if str(p.get("uuid", "")).startswith("sandbox:"):        # a sandboxed CLI's token opens the model API only
            return path == "/health" or path.startswith("/v1/")
        # a paired Nebula: the model API and its own view of the switchyard, never the page's API (sessions, transcripts,
        # decisions, stats, settings: the owner's own)
        return path == "/health" or path.startswith("/v1/") or path == "/switchyard/status"

    def _peer(self) -> str | None:
        """The paired Nebula asking: by its token, or (from this machine only) by the X-Nebula-From it sends."""
        pairing = getattr(self.server, "pairing", None)
        auth = self.headers.get("Authorization") or ""
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else (names.header(self.headers, "Token") or "")
        p = pairing.allowed(token) if pairing is not None and token else None
        if p is not None:
            return p["uuid"]
        addr = self.client_address[0] if self.client_address else "127.0.0.1"
        return names.header(self.headers, "From") if lan.is_loopback(addr) else None

    def _tenant(self) -> str | None:
        """None for the owner (this machine without a caller key); else the asking guest's tenant (nebula/tenancy.py)."""
        peer = self._peer()
        if peer and not str(peer).startswith("sandbox:"):
            return f"peer:{peer}"
        from nebula import callers
        who = callers.by_key(self.cfg, stars.presented_key(self.headers))
        return f"caller:{who['id']}" if who else None

    def _key_ok(self) -> bool:
        """The API Server's pass-key, when it requires one: one of its keys, or a sandbox's own token."""
        if self._internal() or stars.gateway_allows(self.cfg, self.headers):
            return True
        pairing = getattr(self.server, "pairing", None)
        token = stars.presented_key(self.headers)
        return bool(pairing is not None and token and pairing.allowed(token))

    def _context(self) -> Context:
        """A request's context: who asks (a paired peer, a caller by its key or label), its ticks and class, and the
        server's parts."""
        peer = self._peer()
        from nebula import callers
        who, label = callers.identify(self.cfg, self.headers, stars.presented_key(self.headers)) if not peer else (None, None)
        return Context(self.cfg, self.server.ledger, self.headers, self.server.engines, self.server.tools,
                       registry=self.server.registry, admission=self.server.admission, lot=self.server.lot,
                       policy=self.server.policy, peer=peer, yard=self.server.yard,
                       ticks=(who or {}).get("models") or None, priority=(who or {}).get("class"),
                       caller={"id": who["id"] if who else None, "label": label, "guest": bool((who or {}).get("guest"))}
                       if (who or label) else None,
                       mode=(who or {}).get("mode"), power=self.server.power, slots=self.server.slots,
                       seclog=getattr(self.server, "seclog", None), guard=getattr(self.server, "guard", None),
                       otel=getattr(self.server, "otel", None))

    # --- the security log (nebula/seclog.py): refusals and key use, never a key itself
    def _addr(self) -> str:
        return self.client_address[0] if self.client_address else "?"

    def _stats(self):
        """This entry point's counters (the map's Quasar): Nebula's own port is the entry point "main"."""
        ents = getattr(self.server, "entries", None)
        return ents.stats.get("main") if ents is not None else None

    def _refused(self, kind: str, path: str) -> None:
        st = self._stats()
        if st is not None and kind == "auth.refused":
            st.mark("bounce")
        log = getattr(self.server, "seclog", None)
        if log is not None:
            log.add(kind, self._addr(), path.split("/")[1] if path.count("/") > 1 else path,
                    presented_key=bool(stars.presented_key(self.headers)), path=path[:80])

    def _held(self):
        """A guest Halo (or the owner) holds: paused or over its limit (nebula/guard.py); logged."""
        g = getattr(self.server, "guard", None)
        tenant = self._tenant()
        hit = g.check(tenant) if g is not None and tenant else None
        if hit is not None and getattr(self.server, "seclog", None) is not None:
            self.server.seclog.add(hit[2], tenant, "request")
        return hit

    def _key_use(self) -> None:
        log = getattr(self.server, "seclog", None)
        if log is None:
            return
        hit = stars.key_row(self.cfg, stars.presented_key(self.headers))
        if hit is not None:
            log.key_used(f"{hit[0]}/{hit[1].get('id')}", hit[0], self._addr())

    def _internal(self) -> bool:
        """Nebula's own callers on this machine (Gravity's forwards, the household link, its commands)."""
        import hmac
        t = names.header(self.headers, "Internal") or ""
        own = getattr(self.server, "internal_token", None)
        return bool(t and own and hmac.compare_digest(t, own))

    def _signin(self, method: str, path: str) -> bool:
        """The sign-in routes, and the page API's gate: once a passkey or a PIN is set, a signed-in browser (or Nebula
        itself). True when the request was answered here."""
        si = getattr(self.server, "signin", None)
        if si is None:
            return False
        if path == "/api/signin" or path.startswith("/api/signin/"):
            req = {}
            if method == "POST":
                try:
                    n = min(int(self.headers.get("Content-Length") or 0), 65536)
                    req = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    req = {}
            status, body, cookie = signin.handle(self.server, method, path, req if isinstance(req, dict) else {}, self.headers, self._addr())
            self._json(status, body, {"Set-Cookie": cookie, "Cache-Control": "no-store"} if cookie else {"Cache-Control": "no-store"})
            return True
        if path.startswith("/api/") and si.configured() and not self._internal() and not si.signed_in(self.headers):
            self._refused("auth.signin", path)
            self._json(401, {"error": {"message": "sign in to this Nebula first", "code": "signin"}})
            return True
        return False

    def _origin_ok(self, method: str, path: str) -> bool:
        """Only this page drives this Nebula (nebula/origin.py): a foreign Host, or another site's page, is refused."""
        hit = origin.check(self, method, path)
        if hit is None:
            return True
        log = getattr(self.server, "seclog", None)
        if log is not None:
            log.add(hit[0], self._addr(), path.split("/")[1] if path.count("/") > 1 else path, path=path[:80],
                    host=str(self.headers.get("Host") or "")[:80], origin=str(self.headers.get("Origin") or "")[:80])
        self._json(403, error_shape("chat", 403, hit[1]))
        return False

    # --- GET
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if not self._origin_ok("GET", path):
            return None
        if self._signin("GET", path):
            return None
        if not self._lan_ok(path):
            self._refused("auth.unpaired", path)
            return self._json(403, error_shape("chat", 403, "this machine is not paired with this Nebula"))
        if (path.startswith("/v1/") or path.startswith("/switchyard/")) and not self._key_ok():
            self._refused("auth.refused", path)
            return self._json(401, error_shape("chat", 401, "this Nebula's API needs a key (the API Server star)"))
        if path in ("/switchyard/status", "/switchyard/plan", "/switchyard/registry"):
            return self._json(200, ui.switchyard_get(self.server, path, self._tenant()))   # its own, as a router is watched
        if path in PASSTHROUGH and self.engine is None:
            self._json(404, error_shape("chat", 404, "this cockpit only watches another router: it has no engine"))
        elif path in PASSTHROUGH:
            status, ctype, data = self.engine.get(path)
            self._bytes(status, ctype, data)
        elif path == "/v1/models":
            self._json(200, self._models())
        elif path.startswith("/fonts/"):             # the page's own web fonts (nebula/ui/fonts): no outside request
            name = path[len("/fonts/"):]
            f = Path(__file__).with_name("ui") / "fonts" / name
            if "/" in name or not name.endswith((".woff2", ".css")) or not f.is_file():
                return self._json(404, error_shape("chat", 404, "no such font"))
            self._bytes(200, "font/woff2" if name.endswith(".woff2") else "text/css; charset=utf-8", f.read_bytes())
        elif path == "/favicon.svg":                 # the app's icon (nebula/app/icon.svg), the page's tab icon
            self._bytes(200, "image/svg+xml", (Path(__file__).parent / "app" / "icon.svg").read_bytes())
        elif path == "/favicon.ico":                 # the page names its own icon; a browser asks for this one anyway
            self.send_response(204)
            self.end_headers()
        elif path == "/api/stream":
            self._stream()
        elif path == "/" or path.startswith("/api/"):
            status, ctype, data = ui.get(self.server, path, self.path.partition("?")[2], self.headers)
            self._bytes(status, ctype, data)
        else:
            self._json(404, error_shape("chat", 404, f"no such path: {path}"))

    def _stream(self) -> None:
        if self.server.watch is not None and not self.cfg.tiers:
            frames = ui.watch_stream(self.server)
        elif self.server.ledger is None:
            return self._json(503, {"error": "no ledger: start the cockpit with one for the live view"})
        else:
            frames = ui.stream(self.server)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for frame in frames:
                self.wfile.write(frame)
                self.wfile.flush()
        except Exception:  # noqa: BLE001 - the page went away, or the server is closing under it
            pass
        finally:
            frames.close()

    def _models(self) -> dict:
        status, _ctype, data = self.engine.get("/v1/models") if self.engine is not None else (404, "", b"")
        try:
            obj = json.loads(data) if status == 200 else {}
        except ValueError:
            obj = {}
        listed = obj.get("data") if isinstance(obj, dict) and isinstance(obj.get("data"), list) else []
        have = {d.get("id") for d in listed if isinstance(d, dict)}
        for name in self.cfg.model_names:
            if name not in have:
                listed.append({"id": name, "object": "model", "created": int(time.time()), "owned_by": "nebula"})
        return {"object": "list", "data": listed}

    # --- POST
    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if not self._origin_ok("POST", path):
            return None
        if self._signin("POST", path):
            return None
        if not self._lan_ok(path):
            self._refused("auth.unpaired", path)
            return self._json(403, error_shape("chat", 403, "this machine is not paired with this Nebula"))
        if path.startswith("/v1/") and not self._key_ok():
            self._refused("auth.refused", path)
            d = POST_ROUTES.get(path, ("chat",))[0]
            return self._json(401, error_shape(d, 401, "this Nebula's API needs a key (the API Server star)"))
        if path.startswith("/v1/") or path == "/mcp":
            held = self._held()
            if held is not None:
                return self._json(held[0], error_shape(POST_ROUTES.get(path, ("chat",))[0], held[0], held[1]))
        if path == "/mcp":                              # Lookup's library over MCP (nebula/rag/mcp.py), the API's key rules
            if not self._key_ok():
                self._refused("auth.refused", path)
                return self._json(401, {"jsonrpc": "2.0", "id": None, "error": {"code": -32001, "message": "this Nebula's API needs a key"}})
            self._key_use()
            from nebula.rag import mcp
            n = int(self.headers.get("Content-Length") or 0)
            status, body = mcp.handle(self.server, self._tenant(), self.rfile.read(min(n, 2**20)))
            if body is None:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            return self._json(status, body)
        if path.startswith("/v1/"):
            self._key_use()
            st = self._stats()
            if st is not None:
                st.mark("call")
                try:
                    return self._post(path)
                finally:
                    st.mark("answer")
        return self._post(path)

    def _post(self, path: str):
        if self.server.watch is not None and not self.cfg.tiers and path not in ("/api/errors", "/api/errors/clear", "/api/look", "/api/routing") \
                and not path.startswith("/api/stars") and not path.startswith("/api/plan") and not path.startswith("/api/lookup/") \
                and not path.startswith("/api/wizards/") and not path.startswith("/api/updates"):
            return self._json(409, error_shape("chat", 409, "this Nebula has no models of its own yet: it only "
                                                            "watches another router, and changes nothing there"))
        route = POST_ROUTES.get(path)
        if route is None and path.startswith("/api/"):
            route = ("chat", lambda engine, ctx, req: ui.post(self.server, path, engine, ctx, req))
        if route is None and not path.startswith("/hooks/"):
            return self._json(404, error_shape("chat", 404, f"no such path: {path}"))
        dialect = route[0] if route else "chat"
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n > MAX_BODY:
            return self._json(413, error_shape(dialect, 413, "the request body is too large"))
        raw = self.rfile.read(n)
        if route is None:                        # a webhook: its own auth, its own body shape
            try:
                reply = hooks.post(self.server, path[len("/hooks/"):].strip("/"), raw, self.headers)
            except hooks.HookError as e:
                return self._json(e.status, error_shape("chat", e.status, str(e)))
            except EngineError as e:
                LOG.warning("POST %s: the engine answered %s: %s", path, e.status, e.message)
                return self._json(e.status, error_shape("chat", e.status, e.message))
            except (ValueError, KeyError, TypeError) as e:
                return self._json(400, error_shape("chat", 400, str(e) or e.__class__.__name__))
            return self._json(reply.status, reply.body)
        dialect, handler = route
        try:
            req = json.loads(raw)
            if not isinstance(req, dict):
                raise ValueError
        except ValueError:
            return self._json(400, error_shape(dialect, 400, "the body must be a JSON object"))
        ctx = self._context()
        page_write = path.startswith("/api/")
        try:
            reply = handler(self.engine, ctx, req)
            if isinstance(reply, JsonReply):
                if page_write:
                    seclog.page_write(ctx.seclog, path, req, reply.status)
                return self._json(reply.status, reply.body, ctx.reply_headers)
            frames = reply.frames
            first = next(frames, None)
        except ResponsesError as e:
            return self._json(e.status, e.body())
        except ui.UiError as e:
            if page_write:
                seclog.page_write(ctx.seclog, path, req, e.status)
            return self._json(e.status, error_shape(dialect, e.status, str(e)))
        except EngineError as e:
            LOG.warning("POST %s: the engine answered %s: %s", path, e.status, e.message)
            return self._json(e.status, error_shape(dialect, e.status, e.message))
        except (ValueError, KeyError, TypeError) as e:   # a translator refused the request's shape
            if not isinstance(e, ValueError) or not str(e):
                LOG.warning("POST %s: %s while preparing the request", path, e.__class__.__name__, exc_info=True)
            return self._json(400, error_shape(dialect, 400, str(e) or e.__class__.__name__))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        for k, v in {**(ctx.reply_headers or {}), **(reply.headers or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        try:
            if first is not None:
                self.wfile.write(first)
                self.wfile.flush()
            for frame in frames:
                self.wfile.write(frame)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            frames.close()                       # the client went away: so does the engine request
        except EngineError as e:                 # the engine died mid-stream: the dialect's error, as an event
            LOG.warning("POST %s: the engine failed mid-stream (%s): %s", path, e.status, e.message)
            self.wfile.write(sse_frame("error", error_shape(dialect, e.status, e.message)))
        except OSError:
            frames.close()


class EntryHandler(Handler):
    """An API entry point's listener (nebula/entries.py): its scope (this machine, or private addresses of your
    network), its paths (its formats), its own keys (its caller's), its model list; then the same Switchyard."""

    @property
    def entry(self) -> dict:
        return self.server.entry

    def _stats(self):
        return self.server.stats

    def _scope_ok(self) -> bool:
        from nebula.entries import on_lan
        addr = self.client_address[0] if self.client_address else "?"
        return lan.is_loopback(addr) if self.entry["scope"] == "loopback" else on_lan(addr)

    def _lan_ok(self, path: str) -> bool:
        return self._scope_ok()

    def _peer(self):
        return None

    def _key_ok(self) -> bool:
        k = stars.presented_key(self.headers)
        if k:
            return self.server.entries.entry_key_ok(self.entry, k)
        return not self.entry.get("require_key", True)

    def _tenant(self):
        return f"caller:{self.entry['caller']}"

    def _context(self) -> Context:
        from nebula import callers
        ctx = super()._context()
        who = callers.get(self.cfg, self.entry["caller"]) or {"id": self.entry["caller"]}
        ctx.caller = {"id": who["id"], "label": names.header(self.headers, "Caller") or self.entry["name"], "guest": bool(who.get("guest"))}
        ctx.ticks, ctx.priority, ctx.mode = who.get("models") or None, who.get("class"), who.get("mode")
        return ctx

    def _err(self, status: int, message: str, path: str):
        if path.startswith("/api/"):                         # Ollama's shape
            return self._json(status, {"error": message})
        return self._json(status, error_shape(POST_ROUTES.get(path, ("chat",))[0], status, message))

    def _gate(self, path: str) -> bool:
        if not self._origin_ok(self.command, path):
            return False
        if not self._scope_ok():
            self._refused("auth.refused", path)
            self._err(403, "this entry point is for " + ("this machine only" if self.entry["scope"] == "loopback" else "your network only"), path)
            return False
        if not self.server.entries.allowed(self.entry, path):
            self._err(404, f"{path} is not offered here (this entry point speaks {', '.join(self.entry['formats'])})", path)
            return False
        if path != "/health" and not self._key_ok():
            self._refused("auth.refused", path)
            self._err(401, "this entry point needs one of its own keys", path)
            return False
        return True

    def _allowed_models(self) -> list[str]:
        from nebula import callers
        return list((callers.get(self.cfg, self.entry["caller"]) or {}).get("models") or [])

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if not self._gate(path):
            return None
        if path in ("/api/tags", "/api/version"):
            from nebula.dialects import ollama
            return self._json(200, ollama.tags(self.cfg, self._allowed_models()) if path == "/api/tags" else {"version": ollama.VERSION})
        if path == "/v1/models":
            allowed = self._allowed_models()
            m = self._models()
            m["data"] = [d for d in m["data"] if not allowed or d.get("id") in allowed]
            outside: dict = {}                               # a model may have several client names: each listed
            for k, v in (self.entry.get("rename") or {}).items():
                outside.setdefault(v, []).append(k)
            m["data"] = [dict(d, id=o) for d in m["data"] for o in (outside.get(d.get("id")) or [d.get("id")])]
            return self._json(200, m)
        return super().do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if not self._gate(path):
            return None
        import io
        try:
            n = min(int(self.headers.get("Content-Length") or 0), MAX_BODY)
        except ValueError:
            n = 0
        raw = self.rfile.read(n)
        rename = self.entry.get("rename") or {}
        if rename:                                           # a client's own name for a model (an older app's name for it)
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = None
            if isinstance(body, dict) and isinstance(body.get("model"), str) and body["model"] in rename:
                body["model"] = rename[body["model"]]
                raw = json.dumps(body).encode()
                del self.headers["Content-Length"]
                self.headers["Content-Length"] = str(len(raw))
        self.rfile = io.BytesIO(raw)                         # read again by the dialect
        allowed = self._allowed_models()
        if allowed:                                          # a model it does not offer is refused, not swapped
            try:
                named = (json.loads(raw or b"{}") or {}).get("model")
            except ValueError:
                named = None
            bare = names.bare_model(named) if isinstance(named, str) else None
            if bare and bare != "nebula" and bare not in allowed:
                return self._err(404, f"{bare!r} is not offered here (this entry point offers {', '.join(allowed)})", path)
        if path in ("/api/chat", "/api/generate"):
            return self._ollama(path)
        return super().do_POST()

    def _ollama(self, path: str):
        from nebula.dialects import ollama
        held = self._held()
        if held is not None:
            return self._json(held[0], {"error": held[1]})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self._json(400, {"error": "the body must be JSON"})
        st = self._stats()
        st.mark("call")
        try:
            try:
                status, out = ollama.handle(self.engine, self._context(), path, req if isinstance(req, dict) else {})
            except EngineError as e:
                return self._json(e.status, {"error": e.message})
            except (ValueError, KeyError, TypeError) as e:
                return self._json(400, {"error": str(e) or e.__class__.__name__})
            if isinstance(out, dict):
                return self._json(status, out)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                for line in out:
                    self.wfile.write(line)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return None
        finally:
            st.mark("answer")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    admission = None
    lot = None
    registry = None
    ledger = None

    telemetry = None
    tools = None
    app = None                                   # cockpit.app.runner.App when the desktop app runs this server
    scheduler = None
    watch = None                                 # nebula.switchyard.SwitchyardWatch: the optional watch of another Switchyard
    yard = None                                  # nebula.engines.Engines: the engines this Nebula runs
    power = None                                 # nebula.power.Power: the machines that sleep
    slots = None                                 # nebula.tenancy.Slots: each tenant's slot on a shared engine
    network = None                               # nebula.network.Network: model servers on your network, when on
    seclog = None                                # nebula.seclog.SecLog: the security log
    library = None                               # nebula.rag.library.Library: Lookup's collections, when its parts are here
    guard = None                                 # nebula.guard.Guard: holds on guests
    halo = None                                  # nebula.halo.Halo: the Assistant's guardian
    wizards = None                               # nebula.wizards.Wizards: the guided flows and their jobs
    entries = None                               # nebula.entries.Entries: the API Server's entry points
    otel = None                                  # nebula.otel.Exporter: OpenTelemetry, off unless turned on
    updates = None                               # nebula.updates.Updates: a newer Nebula, looked for only when the owner chose so
    library_why = None                           # why there is none
    gravity = None                               # nebula.gravity.GravityNode: Nebula to Nebula over iroh, when on

    errors = None
    followers: list = []
    beacon = None
    pairing = None
    node = None

    def handle_error(self, request, client_address):
        import sys
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):   # the client hung up
            return
        # anything else is logged, a refused or timed-out call made while answering too (it used to vanish: the client
        # saw only a closed connection, as Halo's undo test did once on CI 2026-10-20)
        LOG.error("a request from %s failed", client_address[0], exc_info=True)

    def server_close(self):
        if self.halo is not None:                    # first: it reads the security log the close below shuts
            self.halo.close()
        if getattr(self, "halo_engine", None) is not None:
            self.halo_engine.close()
        if self.network is not None:
            self.network.close()
        if self.gravity is not None:
            self.gravity.close()
        if self.yard is not None:
            self.yard.close()
        if self.watch is not None:
            self.watch.close()
        for f in self.followers:
            f.close()
        if self.errors is not None:
            self.errors.detach()
        if self.ledger is not None:
            self.ledger.activity.close()         # the live streams end before the ledger does
        super().server_close()
        if self.beacon is not None:
            self.beacon.close()
        if self.scheduler is not None:
            self.scheduler.close()
        if self.ledger is not None:
            self.ledger.close()
        if getattr(self, "ledger_hold", None) is not None:
            os.close(self.ledger_hold)
            self.ledger_hold = None
        if self.telemetry is not None:
            self.telemetry.close()
        if self.seclog is not None:
            self.seclog.close()
        if self.tools is not None:
            self.tools.close()
        if self.library is not None:
            self.library.close()
        if self.entries is not None:
            self.entries.close()
        if self.otel is not None:
            self.otel.close()
        if self.updates is not None:
            self.updates.close()


def tier_key(t) -> str | None:
    """A tier's API key: from its file, else its environment variable."""
    if t.api_key_file:
        try:
            with open(t.api_key_file, encoding="utf-8") as f:
                return f.read().strip() or None
        except OSError as e:
            LOG.warning("tier %s: key file %s cannot be read: %s", t.name, t.api_key_file, e)
            return None
    return os.environ.get(t.api_key_env) if t.api_key_env else None


def reload_models(srv) -> None:
    """The engines and the registry rebuilt from the config's tiers, live: after the page added or removed a
    model. The ledger, the tools and the schedules are untouched."""
    cfg = srv.cfg
    from nebula.commands import CommandEngine
    srv.engines = {t.name: (CommandEngine(t, cfg.engine_timeout_s, cfg.base_dir) if t.is_command
                            else Engine(t.address, float(t.timeout_s or cfg.engine_timeout_s), model=t.model, context=t.context, api_key=tier_key(t),
                                        health_path=t.health_path,
                                        sign_secret=(cfg.raw.get("network") or {}).get("secret") if getattr(t, "mesh", None) == "network" else None,
                                        headers={**(t.headers or {}), **({"X-Nebula-From": cfg.lan["uuid"]} if t.peer and cfg.lan.get("uuid") else {})}))
                   for t in cfg.tiers if t.address}
    srv.engine = srv.engines.get(cfg.tiers[0].name) if cfg.tiers else None   # the passed-through GETs are the first tier's
    if getattr(srv, "pairing", None) is not None:
        sandbox_tokens(srv)
    if getattr(srv, "policy", None) is None:
        from nebula.policy import Policy
        srv.policy = Policy(str(Path(cfg.path).with_suffix(".routing.json")) if cfg.path else None)
    if getattr(srv, "registry", None) is None:
        srv.registry = Registry(cfg, srv.engines)
        srv.registry.policy = srv.policy
    else:                                            # the same registry, so a parked request sees the new models
        srv.registry.cfg, srv.registry.engines = cfg, srv.engines
    rules = Rules.from_dict(cfg.rules)
    if getattr(srv, "admission", None) is None:
        srv.admission = Admission(cfg, srv.registry, rules.tolerance_seconds)
    else:
        srv.admission.refresh(cfg, srv.registry)
    if getattr(srv, "lot", None) is None:
        srv.lot = Lot(rules.parking_recheck_seconds, rules.parking_wait_seconds)
    else:
        srv.lot.poke()                                   # a model was added or removed: the parked ask again


class LedgerInUse(OSError):                      # an OSError, as a taken port is: the app shows both the same way
    pass


def hold_ledger(path: str):
    """The ledger has one writer: a second Nebula started on the same file (two configs left at the default
    ./nebula.sqlite3, say) is refused with the reason, not left to interleave its rows with the first one's."""
    lock = os.path.abspath(path) + ".lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":                          # Windows: a lock on the file's first byte, held while it runs
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            os.lseek(fd, 1 if os.name == "nt" else 0, os.SEEK_SET)
            pid = os.read(fd, 32).decode(errors="replace").strip()
        except OSError:
            pid = ""
        os.close(fd)
        raise LedgerInUse(f"another Nebula{' (pid ' + pid + ')' if pid else ''} is using the ledger {os.path.abspath(path)}: "
                          "give this one its own (--ledger PATH, or \"ledger\" in its config)") from None
    if os.name == "nt":                              # the locked first byte stays; the pid goes after it
        os.lseek(fd, 1, os.SEEK_SET)
        os.write(fd, str(os.getpid()).encode().ljust(16))
    else:
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
    return fd


def yard_changed(srv) -> None:
    """An engine started, stopped or moved (a sandbox's gateway address): its models follow, then the models reload."""
    from nebula import yard
    names = [t.managed for t in srv.cfg.tiers if getattr(t, "managed", None)]
    for n in dict.fromkeys(names):
        yard.sync_tiers(srv, n)
    reload_models(srv)


def make_server(cfg) -> Server:
    srv = Server((cfg.host, cfg.port), Handler)
    srv.cfg = cfg
    srv.errors = ErrorLog().attach()                 # the System tab's error log
    srv.followers = [Follower(spec, srv.errors).start() for spec in cfg.engine_logs]   # the engines' own problem lines
    reload_models(srv)
    if cfg.ledger == "nebula.sqlite3":               # the default: an install's older cockpit.sqlite3 is kept
        cfg.ledger = names.local_path(cfg.base_dir, "nebula.sqlite3", "cockpit.sqlite3")
    elif cfg.ledger and cfg.base_dir and not os.path.isabs(cfg.ledger):
        cfg.ledger = os.path.join(cfg.base_dir, cfg.ledger)      # beside the config, as its keys are (not wherever it started)
    try:
        srv.ledger_hold = hold_ledger(cfg.ledger) if cfg.ledger else None
    except LedgerInUse:
        srv.server_close()
        raise
    srv.ledger = Ledger(cfg.ledger, cfg.tenant) if cfg.ledger else None
    srv.seclog = seclog.SecLog(seclog.path_for(cfg.ledger))       # beside the ledger (in memory without one)
    import secrets as _secrets
    srv.internal_token = _secrets.token_urlsafe(32)   # Nebula's own callers (Gravity, the household link, its commands)
    srv.signin = signin.SignIn(srv)                 # the page's sign-in: a passkey or a PIN (nebula/signin.py)
    try:
        signin.write_admin_token(cfg, srv.internal_token)
    except OSError as e:
        LOG.warning("the admin token file was not written: %s", e)
    from nebula.otel import Exporter
    srv.otel = Exporter(cfg, (cfg.lan or {}).get("uuid"))          # sends nothing unless "otel" is turned on
    srv.telemetry = ui.start_telemetry() if cfg.monitor else None
    stars.ensure_builtins(cfg)                      # the Lookup star's two built-in servers, listed (off)
    srv.library = lookup_library(srv)              # Lookup's library (nebula/rag): collections, search with citations
    from nebula.guard import Guard
    from nebula.halo import Halo, halo_agent
    srv.guard = Guard(cfg)
    srv.halo = Halo(srv)                            # the Assistant's guardian (on unless turned off)
    halo_agent(cfg)
    from nebula.halo_engine import HaloEngine
    srv.halo_engine = HaloEngine(srv).start()       # Halo's own model, when it is on this machine (a thread: never waits)
    from nebula import stage as _stage
    _stage.HALO = lambda: getattr(srv, "halo_engine", None)   # Halo as the smart router's judge (nebula/stage.py)
    stars.assistant_agent(cfg)                      # the Assistant star's agent, unless the config has its own
    from nebula.rag.tools import local_servers
    srv.tools = Tools(stars.hub_servers(cfg), cfg.mcp_settings, local_servers(srv))
    srv.halo.start()
    from nebula.wizards import Wizards
    srv.wizards = Wizards(srv)
    from nebula.entries import Entries
    srv.entries = Entries(srv)                      # the API Server's entry points beyond this port (each a Quasar)
    srv.scheduler = hooks.Scheduler(srv) if cfg.schedules else None
    if cfg.watch:
        from nebula.switchyard import SwitchyardWatch
        try:
            srv.watch = SwitchyardWatch(cfg.watch).start()
        except ValueError as e:                      # an old-style watch: said once, and Nebula starts without it
            LOG.error("the watch is off: %s", e)
    from nebula.tenancy import Slots
    srv.slots = Slots()
    from nebula.power import Power
    srv.power = Power(cfg, srv.registry)
    from nebula.engines import Engines
    srv.yard = Engines(cfg, srv.admission)
    srv.yard.on_change = lambda: yard_changed(srv)
    srv.yard.autostart()
    if (cfg.raw.get("network") or {}).get("enabled"):            # model servers on your network (nebula/network.py)
        from nebula.network import Network
        srv.network = Network(srv).start()
        srv.yard.network = srv.network
    import threading as _t                                       # engines whose sandboxes outlived the last run
    _t.Thread(target=srv.yard.adopt_sandboxes, name="adopt-sandboxes", daemon=True).start()
    if (cfg.raw.get("gravity") or {}).get("enabled"):            # the Gravity node is on only when its owner turned it on
        from nebula.gravity import GravityNode
        srv.gravity = GravityNode(srv).start()
    lan_start(srv)
    srv.entries.start()
    from nebula.updates import Updates
    srv.updates = Updates(srv).start()          # one look a minute after the start only in "launch"; else on "Look now"
    return srv


LIBRARY_HOOK = None


def lookup_library(srv):
    """Lookup's library, started on its own thread (its Qdrant with it), or None and why (its parts missing, or
    `"lookup": {"library": false}`)."""
    cfg = srv.cfg
    settings = dict(cfg.raw.get("lookup") or {})
    if settings.get("library") is False:
        srv.library_why = "turned off in the config (lookup.library)"
        return None
    from nebula.rag import library as L
    ok, why = L.available()
    if not ok:
        srv.library_why = why
        return None
    base = cfg.base_dir or os.getcwd()
    data = settings.get("data_dir") or os.path.join(base, "nebula-lookup")
    keys = names.local_path(cfg.base_dir, "nebula-keys", "cockpit-keys")
    lib = L.Library(settings, data, (cfg.ledger + ".lookup.sqlite3") if cfg.ledger else None, os.path.join(keys, "qdrant.key"),
                    engine_of=lambda tier: srv.engines.get(tier))
    hook = LIBRARY_HOOK                              # tests: an embedder and vectors of their own
    if hook is not None:
        hook(lib)
    return lib.start()


SANDBOX_HOST = "host.openshell.internal"         # OpenShell's name, inside a sandbox, for the machine its gateway runs on


def sandbox_tokens(srv) -> None:
    """Step O3: a command tier whose sandbox takes its model from this Nebula gets a token of its own (kept in a 0600
    key file beside the config, its hash remembered like a peer's under `sandbox:<tier>`) and the environment that
    points the CLI's own API calls back here, on this machine's LAN address. A machine with none (offline, a container
    with no network) gives OpenShell's own name for the gateway's host: inside a sandbox, loopback is the sandbox."""
    cfg = srv.cfg
    addresses = [a for a in lan.own_addresses() if not lan.is_loopback(a)]
    host = addresses[0] if addresses else SANDBOX_HOST
    for t in cfg.tiers:
        eng = srv.engines.get(t.name)
        sb = getattr(eng, "sandbox", None)
        if sb is None or not (t.sandbox or {}).get("model_via_nebula"):
            continue
        keys = names.local_path(cfg.base_dir, "nebula-keys", "cockpit-keys")
        os.makedirs(keys, mode=0o700, exist_ok=True)
        path = os.path.join(keys, f"sandbox-{t.name}.key")
        try:
            with open(path, encoding="utf-8") as f:
                token = f.read().strip()
        except OSError:
            token = ""
        if not token:
            import secrets
            token = secrets.token_urlsafe(32)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(token + "\n")
        srv.pairing.remember({"uuid": f"sandbox:{t.name}", "name": f"sandbox {t.name}", "address": None, "port": None}, token)
        base = f"http://{host}:{srv.server_port}"
        sb.nebula_endpoint = (host, srv.server_port)
        sb.nebula_env = {"OPENAI_BASE_URL": base + "/v1", "OPENAI_API_KEY": token,
                         "ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": token, "NEBULA_URL": base}


def lan_start(srv) -> None:
    """This node's identity (kept in the config once decided) and the tokens that may call it from beyond this
    machine. LAN discovery and pairing come in a later release."""
    cfg = srv.cfg
    settings = cfg.lan
    if not settings.get("uuid"):
        settings.update(lan.new_node(settings.get("name"), srv.server_port, __version__))
        settings.setdefault("peers", {})
    settings["port"] = srv.server_port
    srv.node = {"uuid": settings["uuid"], "name": settings.get("name") or lan.host_name(), "port": srv.server_port,
                "version": __version__}
    srv.pairing = lan.Tokens(srv.node, settings.get("peers") or {})
    sandbox_tokens(srv)
