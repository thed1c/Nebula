"""nebula/engine.py: the HTTP client for one strata-lite-server (or any llama-server-compatible server).

Everything the cockpit says to an engine goes through here: a GET passed through as it is, a chat completion as
one JSON answer, or a chat completion as a stream of parsed chunks. Standard library only.
"""
from __future__ import annotations

import contextvars
import http.client
import json
import socket
import urllib.error
import urllib.parse
import urllib.request

# The input a call is for ("api", "assistant", "hook", ...), set by the turn while it calls the engine, so a tier's
# headers can name it (`{input}` in a header value), e.g. a caller tag for a router that counts its callers.
CALL_INPUT: contextvars.ContextVar = contextvars.ContextVar("nebula_call_input", default="nebula")
# Whose call it is (nebula/tenancy.py), signed onto calls to the household's model servers (nebula/network.py).
CALL_TENANT: contextvars.ContextVar = contextvars.ContextVar("nebula_call_tenant", default="owner")


def plain_remote(base: str) -> bool:
    """A plain-HTTP address on another machine: never sent to (machines talk over mTLS or iroh only)."""
    u = urllib.parse.urlparse(base)
    if u.scheme != "http":
        return False
    host = (u.hostname or "").lower()
    # *.localhost is this machine by definition (RFC 6761): OpenShell's gateway exposes a sandbox's engine there
    return not (host in ("localhost", "::1") or host.startswith("127.") or host.endswith(".localhost"))


class EngineError(Exception):
    """The engine answered an HTTP error (a 400 for a prompt that does not fit, a 503 while it loads) or could not
    be reached. `status` is the HTTP status (502 when it was unreachable); `body` the parsed JSON, or the text."""

    def __init__(self, status: int, body):
        super().__init__(f"engine answered {status}: {body}")
        self.status, self.body = status, body

    @property
    def message(self) -> str:
        """The engine's own message when its body has llama-server's {"error": {"message": ...}} shape."""
        b = self.body
        if isinstance(b, dict):
            err = b.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
            if isinstance(err, str):
                return err
            if b.get("message"):
                return str(b["message"])
        return str(b)


class Engine:
    """One engine, or one OpenAI-compatible provider: `api_key` goes in the Authorization header and `model` into
    every chat body (a strata-lite-server ignores the model name; a provider needs one)."""

    def __init__(self, base: str, timeout_s: float = 600.0, api_key: str | None = None, model: str | None = None,
                 context: int | None = None, headers: dict | None = None, health_path: str | None = None,
                 sign_secret: str | None = None):
        self.base = base.rstrip("/")
        self.timeout_s = timeout_s
        self.headers = dict(headers or {})          # sent on every request (a paired Nebula's: which Nebula asks)
        self.api_key, self.model, self.context = api_key, model, context
        self._props: dict | None = None
        # a router takes its callers' calls only (/v1/models, /v1/chat/completions): its health is its catalog, which
        # the tier's `health_path` says
        self.health_path = health_path or "/health"
        self.sign_secret = sign_secret                # a household model server: the tenant goes with the call, signed

    def _open(self, path: str, body: dict | None = None):
        if body is not None and self.model and not body.get("model"):
            body = dict(body, model=self.model)      # its own model, unless the turn named one (a star's dedicated model)
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        who = CALL_INPUT.get()
        headers.update({k: str(v).replace("{input}", who) for k, v in self.headers.items()})
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.sign_secret:
            from nebula.pairface import sign
            tenant = CALL_TENANT.get()
            headers["X-Nebula-Tenant"] = tenant
            headers["X-Nebula-Tenant-Sig"] = sign(self.sign_secret, tenant, data or b"")
        if plain_remote(self.base):
            raise EngineError(502, f"{self.base} is another machine over plain HTTP: Nebula never sends model traffic "
                                   "unencrypted between machines (add it as a model server on your network, or a Gravity peer)")
        req = urllib.request.Request(self.base + path, data=data, headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = raw.decode("utf-8", "replace")
            raise EngineError(e.code, parsed) from None
        except (urllib.error.URLError, OSError) as e:
            raise EngineError(502, f"the engine at {self.base} could not be reached: {e}") from None

    def get(self, path: str) -> tuple[int, str, bytes]:
        """A GET passed through: (status, content type, body bytes). An error is passed through too."""
        try:
            with self._open(path) as r:
                return r.status, r.headers.get("Content-Type") or "application/json", r.read()
        except EngineError as e:
            if isinstance(e.body, str):
                return e.status, "text/plain; charset=utf-8", e.body.encode("utf-8")
            return e.status, "application/json", json.dumps(e.body).encode("utf-8")

    def props(self) -> dict:
        """The engine's /props, read once; {} for a provider or an engine that has none."""
        if self._props is None:
            status, _ctype, data = self.get("/props")
            try:
                self._props = json.loads(data) if status == 200 else {}
            except ValueError:
                self._props = {}
            if not isinstance(self._props, dict):
                self._props = {}
        return self._props

    def n_ctx(self) -> int | None:
        """The context size: the configured one, else the engine's /props (default_generation_settings.n_ctx)."""
        if self.context:
            return self.context
        p = self.props()
        n = (p.get("default_generation_settings") or {}).get("n_ctx") or p.get("n_ctx")
        return n if isinstance(n, int) and n > 0 else None

    def busy(self) -> int:
        """Requests the engine is answering now, from /slots; 0 when it cannot say."""
        status, _ctype, data = self.get("/slots")
        try:
            slots = json.loads(data) if status == 200 else []
        except ValueError:
            slots = []
        return sum(1 for s in slots if isinstance(s, dict) and s.get("is_processing")) if isinstance(slots, list) else 0

    def tokenize(self, text: str) -> int | None:
        """How many tokens the engine's tokenizer makes of the text (llama-server's POST /tokenize); None when the
        engine has no tokenizer endpoint (remembered) or did not answer."""
        if getattr(self, "_no_tokenize", False):
            return None
        try:
            with self._open("/tokenize", {"content": text}) as r:
                obj = json.loads(r.read())
        except EngineError as e:
            if e.status in (404, 405, 501):
                self._no_tokenize = True
            return None
        except ValueError:
            return None
        toks = obj.get("tokens") if isinstance(obj, dict) else None
        return len(toks) if isinstance(toks, list) else None

    def chat(self, body: dict) -> dict:
        """One chat completion, whole."""
        with self._open("/v1/chat/completions", dict(body, stream=False)) as r:
            return json.loads(r.read())

    def chat_stream(self, body: dict):
        """The chunks of a streamed chat completion, parsed, up to [DONE]. The request is sent on the first
        `next()`, so an engine error is raised there, before anything was answered. Closing the generator closes
        the connection, which is how the engine learns the client went away."""
        r = self._open("/v1/chat/completions", dict(body, stream=True))
        try:
            while True:
                line = r.readline()
                if not line:
                    return
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue                                 # blank lines and SSE comments
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    return
                yield json.loads(payload)
        finally:
            r.close()
