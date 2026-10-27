"""nebula/hooks.py: inputs that are not a chat client. Webhooks and schedules start an agent's turn.

A hook is a config row: a name, a secret, the agent that handles it, how its calls map to sessions, and where the
reply goes. `POST /hooks/<name>` with the secret (an `X-Nebula-Secret` header, a bearer token, or a GitHub-style
`X-Hub-Signature-256` HMAC of the body) runs the agent with the payload as the user's message and answers with the
reply as JSON. Its "tier" names the tier its turns ask for; "field:<path>" takes it from the payload, per call. With a `callback`, the call is accepted at once (202) and the reply is posted there when it is ready.
Every call is a turn like any client's, through the chat dialect, recorded under the input's name (the attempt's
dialect is `hook:<name>`), so the router, the ledger and the page treat it as they treat the room's.

A schedule is a hook the cockpit calls itself every so often ("every": "6h", "30m", "90s") with a fixed message.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
import urllib.request
import uuid

from nebula import names
from nebula.dialects import Context, JsonReply
from nebula.dialects.chat import chat

LOG = logging.getLogger("cockpit.hooks")


class HookError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def secret_of(spec: str | None) -> str | None:
    """A literal secret, or "env:NAME" for one read from the environment."""
    if not spec:
        return None
    if spec.startswith("env:"):
        return os.environ.get(spec[4:]) or None
    return spec


def authorized(hook: dict, headers, raw: bytes) -> bool:
    secret = secret_of(hook.get("secret"))
    if not secret:
        return True                              # an open hook: the config said so
    given = names.header(headers, "Secret") or ""
    auth = headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        given = given or auth[7:].strip()
    if given and hmac.compare_digest(given, secret):
        return True
    sig = headers.get("X-Hub-Signature-256") or ""
    if sig.startswith("sha256="):
        want = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig[7:], want)
    return False


def _field(payload, path: str):
    cur = payload
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)] if int(part) < len(cur) else None
        else:
            return None
    return cur


def session_key(hook: dict, payload) -> str:
    """"per-call" (the default): a new session each time; "per-hook": one continuing session; "field:<path>":
    one session per value of that field of the payload (an issue number, a thread id)."""
    mode = hook.get("session") or "per-call"
    name = hook["name"]
    if mode == "per-hook":
        return f"hook-{name}"
    if mode.startswith("field:"):
        value = _field(payload, mode[6:])
        if value not in (None, ""):
            return f"hook-{name}-{value}"
    return f"hook-{name}-{uuid.uuid4().hex[:12]}"


def tier_of(hook: dict, payload) -> str | None:
    """The tier the hook's turns name: the hook's "tier", or with "field:<path>" the payload's value there (a caller
    that wants a bigger model says so per call). None lets the router choose."""
    spec = hook.get("tier")
    if isinstance(spec, str) and spec.startswith("field:"):
        value = _field(payload, spec[6:]) if isinstance(payload, (dict, list)) else None
        return value if isinstance(value, str) and value else None
    return spec or None


def message_of(hook: dict, payload, raw: bytes) -> str:
    """The user's message: the hook's template with {field.path} substitutions, else the payload itself."""
    template = hook.get("message")
    if template:
        def sub(m):
            v = _field(payload, m.group(1)) if isinstance(payload, (dict, list)) else None
            return "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
        import re
        return re.sub(r"\{([a-zA-Z0-9_.]+)\}", sub, template)
    if isinstance(payload, (dict, list)):
        return f"Webhook {hook['name']} delivered this payload:\n\n" + json.dumps(payload, indent=1, ensure_ascii=False)
    return raw.decode("utf-8", "replace")


def run_turn(server, hook: dict, key: str, text: str, tier: str | None = None) -> dict:
    """One turn through the chat dialect, as the hook's agent; the reply as text with its ids."""
    cfg = server.cfg
    headers = {"X-Nebula-Session": key}
    if hook.get("agent"):
        if cfg.agent(hook["agent"]) is None:
            raise HookError(500, f"hook {hook['name']}: no agent named {hook['agent']!r}")
        headers["X-Nebula-Agent"] = hook["agent"]
    tier = tier if tier is not None else (hook.get("tier") if not str(hook.get("tier") or "").startswith("field:") else None)
    model = names.model(tier if tier and cfg.tier(tier) else None)
    req = {"model": model, "messages": [{"role": "user", "content": text}], "stream": False}
    if hook.get("max_tokens"):
        req["max_tokens"] = int(hook["max_tokens"])
    ctx = Context(cfg, server.ledger, headers, server.engines, server.tools, input=f"hook:{hook['name']}",
                  registry=getattr(server, "registry", None), ticks=hook.get("models") or None,
                  priority=hook.get("priority"), admission=getattr(server, "admission", None), lot=getattr(server, "lot", None),
                  yard=getattr(server, "yard", None))
    reply = chat(server.engine, ctx, req)
    obj = reply.body
    msg = (obj.get("choices") or [{}])[0].get("message") or {}
    row = server.ledger.session_by_key(key) if server.ledger is not None else None
    return {"hook": hook["name"], "session_key": key, "session_id": row["session_id"] if row else None,
            "reply": msg.get("content"), "tool_calls": msg.get("tool_calls") or [], "usage": obj.get("usage")}


def deliver(hook: dict, result: dict) -> None:
    """The reply to the hook's callback URL, with the hook's secret as a bearer token."""
    url = hook.get("callback")
    if not url:
        return
    data = json.dumps(result, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    secret = secret_of(hook.get("secret"))
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=30) as r:
            r.read()
    except Exception as e:  # noqa: BLE001 - the callback's failure is logged, the ledger has the reply
        LOG.warning("hook %s: callback %s failed: %s", hook["name"], url, e)


def post(server, name: str, raw: bytes, headers):
    hook = next((h for h in server.cfg.hooks if h.get("name") == name), None)
    if hook is None:
        raise HookError(404, f"no hook named {name!r}")
    if not authorized(hook, headers, raw):
        raise HookError(401, "the hook's secret does not match")
    if not raw.strip():
        raise HookError(400, "the payload gave the agent nothing to read")
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = None                           # not JSON: the text itself is the message
    key = session_key(hook, payload)
    tier = tier_of(hook, payload)
    text = message_of(hook, payload, raw)
    if not text.strip():
        raise HookError(400, "the payload gave the agent nothing to read")
    if hook.get("callback"):
        def work():
            try:
                deliver(hook, run_turn(server, hook, key, text, tier))
            except Exception as e:  # noqa: BLE001
                LOG.warning("hook %s: the turn failed: %s", name, e)
                deliver(hook, {"hook": name, "session_key": key, "error": str(e)})
        threading.Thread(target=work, daemon=True, name=f"hook-{name}").start()
        return JsonReply(202, {"accepted": True, "hook": name, "session_key": key})
    return JsonReply(200, run_turn(server, hook, key, text, tier))


# ------------------------------------------------------------------------------------------------ schedules
def parse_every(spec) -> float:
    """"90s", "30m", "6h", "1d" -> seconds."""
    if isinstance(spec, (int, float)):
        return float(spec)
    s = str(spec).strip().lower()
    unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(s[-1:], None)
    if unit is None:
        return float(s)
    return float(s[:-1]) * unit


class Scheduler:
    """Each schedule runs its hook-like turn every `every`, on its own thread, until the server closes."""

    def __init__(self, server):
        self.server = server
        self.stop = threading.Event()
        self.threads = []
        self.runs: dict[str, int] = {}
        for s in server.cfg.schedules:
            t = threading.Thread(target=self._loop, args=(s,), daemon=True, name=f"schedule-{s.get('name')}")
            self.threads.append(t)
            t.start()

    def _loop(self, s: dict) -> None:
        every = parse_every(s.get("every") or "1h")
        name = s.get("name") or "schedule"
        hook = {"name": name, "agent": s.get("agent"), "tier": s.get("tier"), "session": s.get("session") or "per-call",
                "callback": s.get("callback"), "secret": s.get("secret"), "max_tokens": s.get("max_tokens"),
                "models": s.get("models"), "priority": s.get("priority") or "background"}
        while not self.stop.wait(every):
            try:
                key = session_key(hook, {})
                result = run_turn(self.server, hook, key, s.get("message") or f"Scheduled run of {name}.")
                self.runs[name] = self.runs.get(name, 0) + 1
                deliver(hook, result)
            except Exception as e:  # noqa: BLE001
                LOG.warning("schedule %s failed: %s", name, e)

    def close(self) -> None:
        self.stop.set()
