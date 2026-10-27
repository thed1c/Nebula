"""nebula/halo.py: Halo, the Assistant's guardian (docs/halo.md). On unless the owner turns it off.

Halo watches the security log (nebula/seclog.py). When something a guest did is worth a look (refusals, a join, a
guest asking for what is not theirs), it asks the Assistant's model (or its own, `assistant.halo.route`) with the
interactive class at the gates, through the Switchyard like any turn, to read the log (Lookup's read-only `security`
tool) and decide. It acts only through small tools that undo:

    pause_guest(tenant, minutes)        no requests from that guest for a while (at most a day)
    rate_limit(tenant, per_minute, minutes)
    untick_share(tenant, model)         hold one model back from that guest
    revoke_key(key)                     an API Server or caller key (kept, so undo brings it back)
    ask_owner(message)                  anything bigger: the owner decides

It never touches the owner's own traffic, never shares anything, never loosens a rule: the Switchyard's rules stay
the real guard. Every action is in the security log (`halo.action`) with its undo; `halo.undo` when undone.

**The ring:** glow (calm), pulse (looking at something), red (it acted, or asks you; until you have seen it).
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import secrets
import threading
import time

LOG = logging.getLogger("nebula.halo")
HALO_TURN: contextvars.ContextVar = contextvars.ContextVar("nebula_halo_turn", default=False)
AGENT = "halo"
OWNER = "owner"

SYSTEM = """You are Halo, the guardian of this Nebula, a personal AI switchyard. You read its security log and protect
it from misuse by guests (other people's apps holding a caller key, and other Nebulas joined over Gravity).

Rules:
- Read the log with security__read_log before deciding. Judge patterns, not single events: one wrong key is a typo,
  hundreds a minute is an attack; one refused model is a mistake, a stream of them is probing.
- Use the smallest action that stops the harm: rate_limit before pause_guest, a short pause before a long one,
  untick_share for one model, revoke_key only for a key clearly being abused.
- You never act on the owner ("owner") or the owner's own machines; you never share or open anything.
- Anything bigger than these tools, or anything you are unsure of: ask_owner, plainly, with what you saw.
- If nothing needs doing, say so in one line and do nothing.
Give a reason for every action; the owner reads it beside the undo button."""


class HaloError(ValueError):
    pass


def settings(cfg) -> dict:
    a = cfg.assistant if isinstance(cfg.assistant, dict) else {}
    h = a.get("halo") if isinstance(a.get("halo"), dict) else {}
    return {"on": h.get("on", True) is not False, "route": h.get("route") or {}, "settle_s": float(h.get("settle_s", 3)),
            "every_s": float(h.get("every_s", 300))}


def halo_agent(cfg):
    """The built-in `halo` agent: Halo's prompt, the security log and its own tools, the interactive class."""
    from nebula.config import Agent
    a = cfg.agent(AGENT)
    if a is not None and not getattr(a, "builtin", False):
        return None
    if a is None:
        a = Agent(name=AGENT)
        a.builtin = True
        cfg.agents.append(a)
    a.system, a.tools, a.priority, a.max_rounds = SYSTEM, ["security", "halo"], "interactive", 6
    a.hidden = True                                       # Halo's own: not offered to pick, not a job on the plan
    return a


class Halo:
    def __init__(self, server):
        self.server = server
        self.state = "glow"
        self.why = None
        self.actions: list[dict] = []
        self.asks: list[dict] = []
        self.last_run = None
        self.last_note = None
        self.seen_seq = server.seclog.last_seq() if server.seclog is not None else 0
        self.wake = threading.Event()
        self.closing = False
        self.lock = threading.Lock()
        self.path = (server.cfg.ledger + ".halo.json") if server.cfg.ledger else None
        self._load()
        self.thread = threading.Thread(target=self._loop, name="halo", daemon=True)

    # --- its ring's colour when there is trouble (design decision 10-20): red while it asks you something you have not seen,
    # orange while a step it took is unseen (a pause, a limit, a key revoked); else None
    def _trouble(self) -> str | None:
        if any(not q.get("seen") for q in self.asks):
            return "red"
        if any(not a.get("seen") for a in self.actions):
            return "orange"
        return None

    # --- records
    def _load(self):
        if not self.path:
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.actions, self.asks, self.seen_seq = d.get("actions") or [], d.get("asks") or [], int(d.get("seen_seq") or self.seen_seq)
            if self._trouble():
                self.state = self._trouble()
        except (OSError, ValueError):
            pass

    def _save(self):
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"actions": self.actions[-500:], "asks": self.asks[-200:], "seen_seq": self.seen_seq}, f)
        os.replace(tmp, self.path)

    # --- life
    def on(self) -> bool:
        return settings(self.server.cfg)["on"]

    def start(self) -> "Halo":
        if self.server.seclog is not None:
            self.server.seclog.listeners.append(self._heard)
        self.thread.start()
        return self

    def close(self):
        self.closing = True
        self.wake.set()

    def _heard(self, ev: dict) -> None:
        if self.notable(ev):
            self.wake.set()

    @staticmethod
    def notable(ev: dict) -> bool:
        if ev.get("who") == OWNER or str(ev.get("kind", "")).startswith("halo."):
            return False
        return ev.get("severity") in ("notice", "warn", "alert")

    def _loop(self):
        st = settings(self.server.cfg)
        while not self.closing:
            self.wake.wait(st["every_s"])
            if self.closing:
                return
            self.wake.clear()
            time.sleep(st["settle_s"])                     # let a burst land, then look once
            st = settings(self.server.cfg)
            if not st["on"]:
                continue
            try:
                self.check()
            except Exception as e:  # noqa: BLE001 - said on its panel; the next event tries again
                if self.closing:
                    return
                self.why = f"{type(e).__name__}: {e}"[:300]
                LOG.warning("Halo could not look: %s", self.why)
                if self.server.seclog is not None:
                    self.server.seclog.add("halo.error", "halo", None, error=self.why)

    # --- looking
    def pending(self) -> list[dict]:
        log = self.server.seclog
        if log is None:
            return []
        return [r for r in log.read(since_seq=self.seen_seq, min_severity="notice", limit=200)
                if r["who"] != OWNER and not r["kind"].startswith("halo.")]

    def check(self) -> dict:
        """One look: nothing new is calm; something new goes to the model with Halo's tools."""
        new = self.pending()
        log = self.server.seclog
        top = log.last_seq() if log is not None else 0
        if not new:
            self.seen_seq = top
            if self.state == "pulse":
                self.state = "glow"
            self._save()
            return {"looked": 0}
        if not self.server.cfg.tiers:
            self.why = "Halo has no model to think with: add one"
            self.state = "pulse"
            return {"looked": len(new), "why": self.why}
        before = len(self.actions) + len(self.asks)
        self.state = "pulse"
        digest = "\n".join(f"#{r['seq']} {time.strftime('%H:%M:%S', time.localtime(r['last_at']))} {r['kind']} who={r['who']} "
                           f"what={r['what']} x{r['count']} {json.dumps(r['detail'], ensure_ascii=False)[:200]}" for r in reversed(new))
        text = self._ask_model(f"New security events since you last looked (oldest first):\n{digest}\n\n"
                               f"Guards now in place: {json.dumps(self.server.guard.view())}\nDecide.")
        self.last_run, self.last_note, self.why = time.time(), (text or "")[:500], None
        self.seen_seq = top
        acted = len(self.actions) + len(self.asks) > before
        self.state = self._trouble() or "glow"
        self._save()
        return {"looked": len(new), "acted": acted, "note": self.last_note}

    def _ask_model(self, prompt: str) -> str:
        from nebula import stars
        from nebula.dialects import Context, JsonReply
        from nebula.dialects.chat import chat
        srv, cfg = self.server, self.server.cfg
        halo_agent(cfg)
        r = settings(cfg)["route"]
        own = getattr(srv, "halo_engine", None)
        engines = srv.engines
        if isinstance(r, dict) and r.get("tier") and cfg.tier(r["tier"]):
            tier = r["tier"]                             # the owner's pick wins
        elif own is not None and own.ready():
            # its own model (nebula/halo_engine.py): a view of the config that holds it alone, so it is never one of
            # the models others are routed to, shown or shared, yet Halo's call takes the same path (tools, ledger)
            import copy
            from nebula.engine import Engine
            cfg = copy.copy(cfg)
            cfg.tiers = [own.tier]
            tier = own.tier.name
            engines = {tier: Engine(own.tier.address, float(srv.cfg.engine_timeout_s), model=own.tier.model, context=own.tier.context,
                                    api_key=getattr(own, "api_key", None))}
        else:
            tier = stars.route_of(cfg, "assistant")[0]
        req = {"messages": [{"role": "user", "content": prompt}], "stream": False, **({"model": tier} if tier else {})}
        ctx = Context(cfg, srv.ledger, {"X-Nebula-Agent": AGENT, "X-Nebula-Caller": "halo"}, engines, srv.tools, input="halo",
                      caller={"id": None, "label": "halo", "guest": False}, priority="interactive",
                      registry=srv.registry if engines is srv.engines else None, admission=srv.admission, lot=srv.lot,
                      policy=srv.policy, yard=srv.yard,
                      power=srv.power, slots=srv.slots, seclog=srv.seclog, guard=srv.guard, otel=srv.otel)
        tok = HALO_TURN.set(True)
        try:
            reply = chat(engines.get(tier) or srv.engine, ctx, req)
        finally:
            HALO_TURN.reset(tok)
        if not isinstance(reply, JsonReply):
            return ""
        return ((reply.body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""

    # --- acting (its tools call these)
    def record(self, tool: str, args: dict, reason: str, undo: dict | None, summary: str) -> dict:
        a = {"id": secrets.token_hex(4), "at": time.time(), "tool": tool, "args": args, "reason": (reason or "")[:500],
             "summary": summary, "undo": undo, "undone": False, "seen": False}
        with self.lock:
            self.actions.append(a)
            self.state = self._trouble() or "orange"
            self._save()
        if self.server.seclog is not None:
            self.server.seclog.add("halo.action", "halo", f"{tool}:{args.get('tenant') or args.get('key') or ''}", id=a["id"],
                                   args=args, reason=a["reason"], undo=bool(undo))
        return a

    def ask(self, message: str) -> dict:
        q = {"id": secrets.token_hex(4), "at": time.time(), "message": str(message)[:1000], "seen": False}
        with self.lock:
            self.asks.append(q)
            self.state = "red"                     # it needs you
            self._save()
        if self.server.seclog is not None:
            self.server.seclog.add("halo.ask", "halo", None, id=q["id"], message=q["message"][:300])
        return q

    def undo(self, action_id: str) -> dict:
        a = next((x for x in self.actions if x["id"] == action_id), None)
        if a is None:
            raise HaloError(f"no action {action_id}")
        if a["undone"]:
            raise HaloError("already undone")
        u = a.get("undo") or {}
        if u.get("op") in ("pause", "limit", "deny"):
            self.server.guard.undo(u)
        elif u.get("op") == "revoke":
            restore_key(self.server.cfg, u)
            if self.server.cfg.path:
                self.server.cfg.save()
        else:
            raise HaloError("this action has no undo")
        with self.lock:
            a["undone"], a["undone_at"], a["seen"] = True, time.time(), True
            if not any(not x.get("seen") for x in self.actions + self.asks):
                self.state = "glow"
            self._save()
        if self.server.seclog is not None:
            self.server.seclog.add("halo.undo", OWNER, f"{a['tool']}:{a['args'].get('tenant') or a['args'].get('key') or ''}", id=a["id"])
        return a

    def seen(self) -> dict:
        with self.lock:
            for x in self.actions + self.asks:
                x["seen"] = True
            self.state = "glow"
            self._save()
        return self.view()

    def view(self) -> dict:
        st = settings(self.server.cfg)
        return {"on": st["on"], "state": self.state if st["on"] else "off", "why": self.why, "route": st["route"],
                "last_run": self.last_run, "last_note": self.last_note,
                "actions": list(reversed(self.actions[-100:])), "asks": list(reversed(self.asks[-50:])),
                "guard": self.server.guard.view(),
                "own_model": self.server.halo_engine.view() if getattr(self.server, "halo_engine", None) is not None else None}


# --- keys: revoke and bring back (the row is kept by Halo, so undo restores it exactly; the key itself was never kept)
def revoke_key(cfg, key: str) -> dict:
    """`api/<id>` (an API Server key) or `caller:<cid>/<id>` (a caller's)."""
    owner, _, kid = str(key).rpartition("/")
    if owner == "api":
        rows = list((cfg.api or {}).get("keys") or [])
        row = next((k for k in rows if k.get("id") == kid), None)
        if row is None:
            raise HaloError(f"no key {key}")
        cfg.api = dict(cfg.api or {}, keys=[k for k in rows if k.get("id") != kid])
        return {"op": "revoke", "owner": "api", "row": row}
    if owner.startswith("caller:"):
        from nebula import callers
        c = callers.get(cfg, owner[len("caller:"):])
        row = next((k for k in (c or {}).get("keys") or [] if k.get("id") == kid), None)
        if row is None:
            raise HaloError(f"no key {key}")
        callers.revoke_key(cfg, c["id"], kid)
        return {"op": "revoke", "owner": owner, "row": row}
    raise HaloError("a key is api/<id> or caller:<id>/<key id> (as the security log names it)")


def restore_key(cfg, u: dict) -> None:
    if u["owner"] == "api":
        cfg.api = dict(cfg.api or {}, keys=list((cfg.api or {}).get("keys") or []) + [u["row"]])
    else:
        from nebula import callers
        c = callers.get(cfg, u["owner"][len("caller:"):])
        if c is None:
            raise HaloError("that caller is gone")
        c["keys"] = list(c.get("keys") or []) + [u["row"]]


class HaloTools:
    """Halo's own tools: only Halo's turn may run them."""
    status, error = "ready", None
    tools = [
        {"name": "pause_guest", "description": "Refuse every request from one guest for some minutes (at most 1440).",
         "inputSchema": {"type": "object", "properties": {"tenant": {"type": "string", "description": "caller:<id> or peer:<id>"},
                                                          "minutes": {"type": "number"}, "reason": {"type": "string"}},
                         "required": ["tenant", "minutes", "reason"]}},
        {"name": "rate_limit", "description": "Let one guest make at most N requests a minute, for some minutes.",
         "inputSchema": {"type": "object", "properties": {"tenant": {"type": "string"}, "per_minute": {"type": "integer"},
                                                          "minutes": {"type": "number"}, "reason": {"type": "string"}},
                         "required": ["tenant", "per_minute", "minutes", "reason"]}},
        {"name": "untick_share", "description": "Hold one model back from one guest (its other models stay).",
         "inputSchema": {"type": "object", "properties": {"tenant": {"type": "string"}, "model": {"type": "string"},
                                                          "reason": {"type": "string"}}, "required": ["tenant", "model", "reason"]}},
        {"name": "revoke_key", "description": "Revoke a key that is clearly abused: api/<id> or caller:<id>/<key id>, as the log names it.",
         "inputSchema": {"type": "object", "properties": {"key": {"type": "string"}, "reason": {"type": "string"}},
                         "required": ["key", "reason"]}},
        {"name": "ask_owner", "description": "Tell the owner what you saw and ask them to decide (for anything bigger).",
         "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
    ]

    def __init__(self, server):
        self.server = server

    def _owner_machine(self, tenant: str) -> bool:
        g = getattr(self.server, "gravity", None)
        if g is None or not tenant.startswith("peer:"):
            return False
        p = g.peer(tenant[len("peer:"):]) if hasattr(g, "peer") else None
        return bool(p and p.get("role") in ("owner", "admin"))

    def call(self, tool: str, a: dict) -> str:
        if not HALO_TURN.get():
            raise PermissionError("only Halo uses these tools")
        halo, guard = self.server.halo, self.server.guard
        reason = str(a.get("reason") or "").strip()
        if tool == "ask_owner":
            halo.ask(str(a.get("message") or ""))
            return "asked the owner"
        tenant = str(a.get("tenant") or "")
        if tool in ("pause_guest", "rate_limit", "untick_share"):
            if self._owner_machine(tenant):
                raise HaloError("that is one of the owner's own machines: ask_owner instead")
            if not reason:
                raise HaloError("give a reason")
        if tool == "pause_guest":
            m = max(1.0, min(float(a.get("minutes") or 10), 1440.0))
            u = guard.pause(tenant, m * 60)
            halo.record(tool, {"tenant": tenant, "minutes": m}, reason, u, f"paused {tenant} for {m:g} min")
            return f"paused {tenant} for {m:g} minutes"
        if tool == "rate_limit":
            n, m = int(a.get("per_minute") or 1), max(1.0, min(float(a.get("minutes") or 30), 1440.0))
            u = guard.limit(tenant, n, m * 60)
            halo.record(tool, {"tenant": tenant, "per_minute": n, "minutes": m}, reason, u, f"limited {tenant} to {n}/min for {m:g} min")
            return f"{tenant}: at most {n} a minute for {m:g} minutes"
        if tool == "untick_share":
            model = str(a.get("model") or "")
            if self.server.cfg.tier(model) is None:
                raise HaloError(f"no model {model}")
            u = guard.deny(tenant, model)
            halo.record(tool, {"tenant": tenant, "model": model}, reason, u, f"held {model} back from {tenant}")
            return f"{model} is held back from {tenant}"
        if tool == "revoke_key":
            if not reason:
                raise HaloError("give a reason")
            u = revoke_key(self.server.cfg, str(a.get("key") or ""))
            if self.server.cfg.path:
                self.server.cfg.save()
            halo.record(tool, {"key": a.get("key")}, reason, u, f"revoked {a.get('key')}")
            return f"revoked {a.get('key')}"
        raise ValueError(f"no tool {tool}")
