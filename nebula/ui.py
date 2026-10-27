"""nebula/ui.py: the cockpit's own page (nebula/ui/index.html) and the JSON API it reads.

    GET  /                          the page
    GET  /api/config                the tiers, for the page's pickers
    GET  /api/sessions              one line per session: title, turns, tier, lifecycle
    GET  /api/sessions/ID           the transcript with each reply's tier, model, thinking, decision and cost,
                                    and the handoffs, briefings and lifecycle changes between the turns
    POST /api/sessions/ID/close     explicit close; /reopen puts it back
    POST /api/chat                  {"session_id"?, "text"?, "tier"?, "agent"?, "escalate"?, "regenerate_from_seq"?}
                                    streams the answer as the chat dialect's SSE; the X-Nebula-Session-Id header
                                    names the session (a new one on the first turn)
    GET  /api/monitor               the hardware sampler's latest readings and each tier's engine: busy, context
    GET  /api/recent?limit=N        the latest attempts, decisions and handoffs across sessions, for the feeds
    GET  /api/inputs                attempts and sessions per input (dialect or hook), for the home page's left column
    GET  /api/registry              every tier as a node: up, slots, context, vision, model, host, power class, cost
    GET  /api/admission             every engine's gate: slots, held, the calls waiting (class, since), the average call
    GET  /api/parking               the parking lot: the requests no ticked model holds, their reasons, the recent ones
    GET  /api/lan                   this node; "pairing": false (LAN discovery and pairing come in a later release)
    GET  /api/lan/engines           engines already running on this machine at their usual ports, to adopt as tiers
    GET  /api/lan/scan              every launch's look around: engines running here, CLIs installed, with how to connect
    GET  /api/endpoints             the server's URL as reached, the base URL per dialect, a direct model name per tier and agent
    POST /api/providers/models      {"base_url", "api_key"?} -> the models an OpenAI-compatible provider lists
    POST /api/models                add a tier: an engine by URL, or a provider by base URL, model id and key (key to a 0600 file)
    POST /api/models/NAME/remove    remove a tier nobody ticks
    POST /api/policy                {"agent"?, "models": [...]} the ticks of an agent, or the default list; saved to the config
    GET  /api/live                  what is running now: the attempts in flight with the deltas each has streamed, and
                                    the work without an attempt row (the secretary's compaction)
    GET  /api/stream                the same as server-sent events: a `tick` whenever the ledger committed a write or a
                                    reply streamed, with what started, landed, crossed (handoffs) and was compacted
                                    since the previous tick (nebula/activity.py); a comment line every 15 s of quiet
    GET  /api/escalations/ID        one decision above the session's floor (ID: its decision_id, or a handoff_id): who, from and to
                                    which tier, the reason and its inputs, and for a crossing the handoff note: the
                                    packet the destination was given and the preamble it read
    GET  /api/app                   {"app": false} in a browser; in the desktop app (nebula/app), its state and the
                                    engine it starts: state, the engine's own output lines, seconds since the start
    POST /api/app/engine/start      start the app's engine again after it failed
    GET  /api/look                  the page's layout (nebula/ui.py look_*): pins, router, node names, sizes, colours, widgets
    POST /api/look                  the same, replaced; kept in NAME.look.json beside the config (also while watching)
    GET  /api/security              the security log (nebula/seclog.py), newest first: ?since=SEQ&kinds=a,b.*&who=&min=warn
                                    &since_s=&limit=; GET /api/security/verify walks its hash chain
    GET  /api/plan                  the decision plan (nebula/plan.py): this Nebula's own, and a watched Switchyard's (read
                                    only) when the optional watch is on; POST /api/plan/apply
    GET  /api/watch/lanes           while watching: each model's lane (slots, holders, waits, queue rules, routes)
    GET  /api/watch/decisions       while watching: the router's decisions as a feed, kept by this Nebula, and its counters
    GET  /api/routing               this Nebula's routing policy (nebula/policy.py), its own models and the ones it reaches
    POST /api/routing               the policy, replaced (the page's Configure panels); read on every request
    GET  /api/errors                the error log (nebula/errors.py): the cockpit's warnings and errors, newest last
    POST /api/errors                {"message", "detail"?}: an error the page itself hit
    POST /api/errors/clear          empty the error log (the ledger keeps its failed attempts)
    GET  /api/updates               is there a newer Nebula (nebula/updates.py): current, latest, notes, can_install,
                                    release_page, last_checked, mode; POST /api/updates/check looks now,
                                    POST /api/updates/install updates (or names the release page to open),
                                    POST /api/updates {"mode": "demand"|"launch"|"off"}

The page's turns go through the chat dialect like any client's, under an X-Nebula-Session header the page never
sees: the ledger and the router treat them as they treat Claude Code's. An escalate button is a regeneration of
that reply with the X-Nebula-Escalate header, which is the strongest label the ledger gets. A closed session
refuses the page's turns until it is reopened; an API client that keeps writing to one reopens it.
"""
from __future__ import annotations

import json
import os
import logging
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs

from nebula import gravity, names, stars
from nebula import __version__
from nebula.activity import Watch
from nebula.dialects import Context, JsonReply, SseReply, sse_frame
from nebula.dialects.chat import chat
from nebula.handoff import preamble, rendered_message
from nebula.ledger import text_of

LOG = logging.getLogger("cockpit.ui")
PAGE = Path(__file__).parent / "ui" / "index.html"


class UiError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def start_telemetry():
    """serve/telemetry.py's sampler, if its readers load here; None keeps the monitor to the engines."""
    try:
        from nebula._vendor.strata_serve.telemetry import Telemetry
        from nebula import engines as E
        idx = [c["index"] for c in E.cards()]                      # every card: the hottest is the black hole's
        return Telemetry(gpu_indices=idx) if len(idx) > 1 else Telemetry()
    except Exception as e:  # noqa: BLE001 - the page works without hardware readings
        LOG.info("no hardware telemetry: %s", e)
        return None


def _json(status: int, obj) -> tuple[int, str, bytes]:
    return status, "application/json", json.dumps(obj, ensure_ascii=False).encode("utf-8")


# ------------------------------------------------------------------------------------------------ GET
def get(server, path: str, query: str, headers=None) -> tuple[int, str, bytes]:
    cfg, ledger = server.cfg, server.ledger
    w = getattr(server, "watch", None)
    if w is not None:                              # watching: the map's feeds come from the watched router
        hit = watch_get(server, w, path)
        if hit is not None:
            return hit
    if path.startswith("/api/yard/"):                 # this Nebula's own Switchyard (nebula/yard.py)
        from nebula import yard
        try:
            hit = yard.get(server, path, query)
        except yard.YardError as e:
            return _json(e.status, {"error": str(e)})
        if hit is not None:
            return _json(200, hit)
    if path == "/api/endpoints":
        return _json(200, endpoints(server, headers))
    if path == "/api/plan":
        return _json(200, plan_view(server))
    if path == "/api/stars":
        return _json(200, stars_view(server, headers))
    if path.startswith("/api/lookup"):                # Lookup's library (nebula/rag/api.py)
        from nebula.rag import api as rag_api
        from nebula.rag.library import LibraryError
        try:
            hit = rag_api.get(server, path)
        except LibraryError as e:
            return _json(e.status, {"error": str(e)})
        if hit is not None:
            return _json(200, hit)
    if path == "/api/browse":                      # the folder picker (nebula/browse.py): folders only
        from nebula import browse
        q = parse_qs(query or "")
        try:
            return _json(200, browse.listing((q.get("path") or [None])[0], (q.get("hidden") or ["0"])[0] == "1"))
        except browse.BrowseError as e:
            return _json(e.status, {"error": str(e)})
    if path.startswith("/api/wizards"):            # the guided flows (nebula/wizards.py)
        from nebula import wizards as W
        try:
            return _json(200, W.get(server, path))
        except W.WizardError as e:
            return _json(e.status, {"error": str(e), **e.extra})
    if path == "/api/updates":                     # a newer Nebula (nebula/updates.py): read from the last look, no network
        u = getattr(server, "updates", None)
        return _json(200, u.view() if u is not None else {"mode": "off", "current": __version__, "newer": False})
    if path == "/api/otel":                        # OpenTelemetry (nebula/otel.py): off by default, metadata only
        return _json(200, server.otel.view() if getattr(server, "otel", None) is not None else {"enabled": False})
    if path == "/api/api-servers":                 # the API Server's entry points (nebula/entries.py), each a Quasar
        from nebula import entries as EP
        return _json(200, EP.get(server, path))
    if path == "/api/halo":                         # Halo (nebula/halo.py): its ring, its actions with their undo, its asks
        h = getattr(server, "halo", None)
        return _json(200, h.view() if h is not None else {"on": False, "state": "off"})
    if path in ("/api/security", "/api/security/verify"):   # the security log (nebula/seclog.py), the owner's own
        log = getattr(server, "seclog", None)
        if log is None:
            return _json(503, {"error": "no security log"})
        if path == "/api/security/verify":
            return _json(200, log.verify())
        q = {k: v[-1] for k, v in parse_qs(query or "").items()}
        try:
            rows = log.read(since_seq=int(q.get("since") or 0), kinds=[k for k in (q.get("kinds") or "").split(",") if k] or None,
                            who=q.get("who") or None, min_severity=q.get("min") or None,
                            since_s=float(q["since_s"]) if q.get("since_s") else None, limit=int(q.get("limit") or 100))
        except ValueError as e:
            return _json(400, {"error": str(e)})
        return _json(200, {"events": rows, "last_seq": log.last_seq(), "counts_1h": log.counts(3600)})
    if path == "/":
        try:
            return 200, "text/html; charset=utf-8", PAGE.read_bytes()
        except OSError:
            return 200, "text/plain; charset=utf-8", f"cockpit {__version__}: the page is missing\n".encode("utf-8")
    if path == "/api/config":
        return _json(200, {"version": __version__, "ledger": ledger is not None,
                           "tiers": [{"name": t.name, "address": t.address, "thinking": t.thinking,
                                      "provider": t.provider, "peer": t.peer, "shared": bool(getattr(t, "shared", False)),
                                      "cloud": bool(gravity.anti_gravity(t)),
                                      "mesh": getattr(t, "mesh", None), "managed": getattr(t, "managed", None)} for t in cfg.tiers],
                           "agents": [{"name": a.name, "tier": a.tier, "ceiling": a.ceiling, "tools": list(a.tools),
                                       "models": list(a.models), "default": a.default, "event": a.event} for a in cfg.agents
                                      if not getattr(a, "hidden", False)],
                           "default_models": list(cfg.default_models),
                           "hooks": [{"name": h["name"], "agent": h.get("agent"), "session": h.get("session") or "per-call",
                                      "callback": bool(h.get("callback")), "secret": bool(h.get("secret"))} for h in cfg.hooks],
                           "schedules": [{"name": x["name"], "agent": x.get("agent"), "every": x.get("every")} for x in cfg.schedules],
                           "mcp_servers": sorted(cfg.mcp_servers)})
    if path == "/api/monitor":
        return _json(200, monitor(server))
    if path == "/api/errors":
        return _json(200, server.errors.list() if server.errors is not None else {"errors": [], "count": 0, "last_id": 0})
    if path == "/api/app":
        app = getattr(server, "app", None)
        return _json(200, app.status() if app is not None else {"app": False})
    if path == "/api/look":
        return _json(200, look_read(server))
    if path == "/api/routing":
        return _json(200, routing_view(server))
    if ledger is None:
        return _json(503, {"error": "no ledger: start the cockpit with one for sessions"})
    if path == "/api/live":
        out = Watch(ledger, [t.name for t in cfg.tiers]).tick()
        out["waiting"] = server.admission.waiting() if getattr(server, "admission", None) else []
        out["parked"] = server.lot.parked() if getattr(server, "lot", None) else []
        return _json(200, out)
    if path == "/api/parking":
        return _json(200, server.lot.status() if getattr(server, "lot", None) else {})
    if path == "/api/admission":
        return _json(200, server.admission.status() if getattr(server, "admission", None) else {})
    if path == "/api/sessions":
        return _json(200, ledger.summaries())
    if path == "/api/registry":
        return _json(200, server.registry.nodes() if getattr(server, "registry", None) else [])
    if path == "/api/lan":
        return _json(200, lan_status(server))
    if path == "/api/lan/engines":
        return _json(200, adopt_engines(server))
    if path == "/api/lan/scan":
        return _json(200, lan_scan(server, headers))
    if path == "/api/sandboxes":
        return _json(200, sandboxes_view(server))
    if path == "/api/inputs":
        return _json(200, ledger.inputs())
    if path == "/api/recent":
        q = parse_qs(query)
        try:
            limit = max(1, min(200, int(q.get("limit", ["40"])[0])))
        except ValueError:
            limit = 40
        return _json(200, ledger.recent(limit))
    if path.startswith("/api/escalations/"):
        note = escalation(ledger, path[len("/api/escalations/"):].strip("/"))
        return _json(200, note) if note else _json(404, {"error": "no such decision"})
    if path.startswith("/api/sessions/"):
        sid = path[len("/api/sessions/"):].strip("/")
        view = ledger.view(sid)
        if view is None:
            return _json(404, {"error": f"no session {sid}"})
        tier = cfg.tier(view["context"].get("tier") or "") or cfg.tiers[0]
        eng = server.engines.get(tier.name)
        view["context"]["n_ctx"] = eng.n_ctx() if eng else None
        return _json(200, view)
    return _json(404, {"error": f"no such path: {path}"})


def escalation(ledger, decision_id: str) -> dict | None:
    """The note behind an escalation notice, read from the ledger when it is opened (the handoff is written after the
    decision, so it is looked up then, not when the notice appeared)."""
    if decision_id.startswith("hand_"):         # a handoff's id: its source attempt's decision
        h = ledger.db.execute("SELECT from_attempt FROM handoffs WHERE handoff_id=?", (decision_id,)).fetchone()
        d = ledger.db.execute("SELECT * FROM decisions WHERE attempt_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                              (h["from_attempt"],)).fetchone() if h else None
    else:
        d = ledger.db.execute("SELECT * FROM decisions WHERE decision_id=?", (decision_id,)).fetchone()
    if d is None:
        return None
    inputs = json.loads(d["inputs"])
    a = ledger.db.execute("SELECT agent, dialect, model_id, state FROM attempts WHERE attempt_id=?", (d["attempt_id"],)).fetchone()
    first = next((p for _q, _h, p in ledger._transcript(ledger.db, d["session_id"]) if p.get("role") == "user"), None)
    out = {"decision_id": d["decision_id"], "session_id": d["session_id"], "attempt_id": d["attempt_id"],
           "title": text_of(first.get("content"))[:80] if first else "", "agent": a["agent"] if a else None,
           "input": a["dialect"] if a else None, "from_tier": inputs.get("floor"), "to_tier": d["tier"],
           "reason": d["reason"], "inputs": inputs, "at": d["created_at"], "handoff": None}
    h = ledger.db.execute("SELECT * FROM handoffs WHERE from_attempt=?", (d["attempt_id"],)).fetchone()
    if h is not None:
        packet = json.loads(h["packet"])
        out["handoff"] = {"handoff_id": h["handoff_id"], "to_tier": h["to_tier"], "created_at": h["created_at"],
                          "acknowledged_at": h["acknowledged_at"], "packet_sha256": h["packet_sha256"],
                          "packet": packet, "note": preamble(packet)}
    return out


# ------------------------------------------------------------------------------------------------ models and policy
def endpoints(server, headers) -> dict:
    """What a user pastes into another app: the server's URL as the request reached it, the base URLs per dialect,
    and a direct model name (and a curl line) per tier and per agent."""
    cfg = server.cfg
    host = (headers.get("Host") if headers is not None else None) or f"{cfg.host}:{server.server_port}"
    base = f"http://{host}"

    def entry(kind, name):
        model = names.model(name)
        return {"kind": kind, "name": name, "model": model, "openai_base": base + "/v1", "anthropic_base": base,
                "curl": (f"curl {base}/v1/chat/completions -H 'Content-Type: application/json' "
                         f"-d '{{\"model\": \"{model}\", \"messages\": [{{\"role\": \"user\", \"content\": \"hello\"}}]}}'")}
    return {"server_url": base, "openai_base": base + "/v1", "anthropic_base": base, "responses_url": base + "/v1/responses",
            "models": [entry("tier", t.name) for t in cfg.tiers],
            "agents": [entry("agent", a.name) for a in cfg.agents if not a.event and not getattr(a, "hidden", False)],
            "hooks": [{"name": h["name"], "url": base + "/hooks/" + h["name"]} for h in cfg.hooks]}


def provider_models(base_url: str, api_key: str | None) -> list[str]:
    """The model ids an OpenAI-compatible provider (or an engine) lists at /v1/models."""
    from nebula.engine import Engine
    eng = Engine(base_url.rstrip("/").removesuffix("/v1"), timeout_s=20.0, api_key=api_key)
    status, _ctype, data = eng.get("/v1/models")
    if status != 200:
        raise UiError(502, f"{base_url}: /v1/models answered {status}")
    try:
        obj = json.loads(data)
    except ValueError:
        raise UiError(502, f"{base_url}: /v1/models did not answer JSON") from None
    items = obj.get("data") if isinstance(obj, dict) else obj
    return [m.get("id") for m in items or [] if isinstance(m, dict) and m.get("id")]


def _keys_dir(cfg) -> str:
    import os
    d = names.local_path(cfg.base_dir, "nebula-keys", "cockpit-keys")
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


def add_model(server, req: dict) -> dict:
    """A tier added from the page: a local engine by URL, or an OpenAI-compatible provider by base URL, model id
    and key. Probed before it is saved; the key goes into a 0600 file beside the config, never into the config."""
    import os
    from nebula.config import Tier
    from nebula.server import reload_models
    cfg = server.cfg
    name = str(req.get("name") or "").strip()
    if not name or any(c in name for c in " /\\"):
        raise UiError(400, "a model needs a short name without spaces or slashes")
    if cfg.tier(name) is not None or cfg.agent(name) is not None:
        raise UiError(409, f"the name {name!r} is taken")
    kind = req.get("kind") or ("command" if req.get("cli") or req.get("command") else "provider" if req.get("base_url") else "engine")
    tier = Tier(name=name, thinking=req.get("thinking") if isinstance(req.get("thinking"), bool) else None,
                context=int(req["context"]) if req.get("context") else None,
                power_class=req.get("power_class") or None,
                cost_per_mtoken=float(req["cost_per_mtoken"]) if req.get("cost_per_mtoken") not in (None, "") else None,
                host=req.get("host") or None)
    key = req.get("api_key") if isinstance(req.get("api_key"), str) and req.get("api_key").strip() else None
    if kind == "command":
        from nebula.commands import CLI_FORMS
        cli = str(req.get("cli") or "").strip() or None
        if cli and cli not in CLI_FORMS:
            raise UiError(400, f"no form for the CLI {cli!r}: one of {sorted(CLI_FORMS)}, or give a command")
        tier.cli = cli
        tier.command = None if cli else (str(req.get("command") or "").strip() or None)
        if not tier.cli and not tier.command:
            raise UiError(400, "a command tier needs a CLI (claude, codex, grok) or a command to run")
        tier.model = str(req.get("model") or "").strip() or None
        tier.cwd = str(req.get("cwd") or "").strip() or None
        tier.slots = int(req["slots"]) if req.get("slots") else None
        sb = req.get("sandbox")
        if sb:                                   # run it in an OpenShell sandbox (step O1): true, an image, or the object
            from nebula.sandbox import IMAGES, parse_host
            sb = {} if sb is True else ({"image": sb} if isinstance(sb, str) else dict(sb))
            sb.setdefault("image", IMAGES.get(cli or "", "ubuntu:24.04"))
            hosts = req.get("hosts") if "hosts" in req else sb.get("hosts")
            if isinstance(hosts, str):
                hosts = [h.strip() for h in hosts.split(",") if h.strip()]
            if hosts:
                bad = [h for h in hosts if parse_host(h) is None]
                if bad:
                    raise UiError(400, f"hosts as host, host:port or host:port:read-write; not {bad}")
                sb["hosts"] = list(hosts)
            tier.sandbox = sb
    elif kind == "engine":
        url = str(req.get("engine") or req.get("url") or "").strip()
        if not url.startswith("http"):
            raise UiError(400, "an engine needs its URL, such as http://127.0.0.1:8080")
        tier.engine = url.rstrip("/")
    else:
        url = str(req.get("base_url") or req.get("url") or "").strip()
        if not url.startswith("http"):
            raise UiError(400, "a provider needs its base URL, such as https://api.example.com/v1")
        tier.provider = "openai-compatible"
        tier.base_url = url.rstrip("/")
        tier.model = str(req.get("model") or "").strip() or None
        if not tier.model:
            raise UiError(400, "a provider needs the model id to ask for (list them first)")
        if req.get("api_key_env"):
            tier.api_key_env = str(req["api_key_env"])
    if key:
        path = os.path.join(_keys_dir(cfg), f"{name}.key")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(key.strip() + "\n")
        tier.api_key_file = path
    # the probe: a command is installed, an engine answers /health, a provider lists its models
    from nebula.engine import Engine
    if kind == "command":
        from nebula.commands import CommandEngine
        probe = CommandEngine(tier, base_dir=cfg.base_dir)
        why = probe.unavailable()
        if why is not None:
            raise UiError(400, f"{why}: not added")
    elif kind == "engine":
        from nebula.engine import plain_remote
        if plain_remote(tier.address or ""):
            raise UiError(400, f"{tier.address} is another machine over plain HTTP: add it as a model server on your network "
                               "(encrypted, Models › Add a server) or a Gravity peer, never an open port")
        probe = Engine(tier.address, timeout_s=20.0, api_key=key or (os.environ.get(tier.api_key_env) if tier.api_key_env else None),
                       headers=tier.headers, health_path=tier.health_path)
        status, _c, _d = probe.get(probe.health_path)
        if status != 200:
            raise UiError(502, f"{tier.address}{probe.health_path} answered {status}: not added")
    else:
        ids = provider_models(tier.base_url, key or (os.environ.get(tier.api_key_env) if tier.api_key_env else None))
        if tier.model not in ids and ids:
            raise UiError(400, f"{tier.model!r} is not among the models {tier.base_url} lists; it lists {ids[:20]}")
    cfg.tiers.append(tier)
    saved = cfg.save() if cfg.path else None
    reload_models(server)
    node = server.registry.node(name)
    return {"added": name, "saved": saved, "node": node}


def set_shared(server, name: str, req: dict) -> dict:
    """A model offered to guests or not ({shared: bool}): each guest then gets its own cache (nebula/tenancy.py). A
    cloud model never (Anti-Gravity)."""
    from nebula.server import reload_models
    from nebula import gravity
    t = server.cfg.tier(name)
    if t is None:
        raise UiError(404, f"no model {name!r}")
    on = bool(req.get("shared"))
    if on and gravity.anti_gravity(t):
        raise UiError(400, f"{name}: {gravity.anti_gravity(t)}. It cannot be shared.")
    t.shared = on
    rows = server.cfg.raw.get("tiers") or []
    for r in rows:
        if isinstance(r, dict) and r.get("name") == name:
            r["shared"] = on
    saved = server.cfg.save() if server.cfg.path else None
    reload_models(server)
    return {"model": name, "shared": on, "saved": saved}


def remove_model(server, name: str) -> dict:
    from nebula.server import reload_models
    cfg = server.cfg
    t = cfg.tier(name)
    if t is None:
        raise UiError(404, f"no model named {name!r}")
    used = [a.name for a in cfg.agents if name in a.models or a.tier == name or a.ceiling == name] \
        + (["default"] if name in cfg.default_models else []) \
        + [h["name"] for h in cfg.hooks + cfg.schedules if name in (h.get("models") or []) or h.get("tier") == name]
    if used:
        raise UiError(409, f"{name!r} is used by {used}: change them first")
    cfg.tiers = [x for x in cfg.tiers if x.name != name]
    saved = cfg.save() if cfg.path else None
    reload_models(server)
    return {"removed": name, "saved": saved}


def set_policy(server, req: dict) -> dict:
    """The ticks of one agent (or the default list when no agent is named): the models it may use, in order."""
    cfg = server.cfg
    models = req.get("models")
    if not isinstance(models, list) or any(not isinstance(m, str) for m in models):
        raise UiError(400, "models must be a list of model names, in the order they may be used")
    unknown = [m for m in models if cfg.tier(m) is None]
    if unknown:
        raise UiError(400, f"not models: {unknown}")
    if len(set(models)) != len(models):
        raise UiError(400, "a model is ticked twice")
    agent = req.get("agent")
    if agent:
        a = cfg.agent(agent)
        if a is None:
            raise UiError(404, f"no agent named {agent!r}")
        a.models = list(models)
    else:
        cfg.default_models = list(models)
    saved = cfg.save() if cfg.path else None
    if getattr(server, "lot", None) is not None:
        server.lot.poke()
    return {"agent": agent, "models": list(models), "saved": saved}


# ------------------------------------------------------------------------------------------------ the network
def lan_status(server) -> dict:
    """This node. LAN discovery and pairing come in a later release: "pairing" is false, nothing is found or paired."""
    return {"node": getattr(server, "node", None), "pairing": False, "decided": True, "discoverable": False,
            "beacon": False, "found": [], "peers": [], "joined": [], "window": {"open": False}}


def _sandboxed(server, tier_name: str):
    t = server.cfg.tier(tier_name)
    eng = server.engines.get(tier_name)
    sb = getattr(eng, "sandbox", None)
    if t is None or sb is None:
        raise UiError(404, f"no sandboxed command tier named {tier_name!r}")
    return t, sb


def sandboxes_view(server) -> list[dict]:
    """Step O2 and O4: every sandboxed command tier with its sandbox, policy hosts and the asks (denied connections)."""
    out = []
    for t in server.cfg.tiers:
        sb = getattr(server.engines.get(t.name), "sandbox", None)
        if sb is None:
            continue
        try:
            d = sb.describe()
            d["asks"] = sb.asks() if d.get("installed") else []
        except Exception as e:  # noqa: BLE001 - one broken gateway must not hide the others
            d = {"name": sb.name, "image": sb.image, "installed": False, "exists": False, "running": False, "detail": str(e), "asks": []}
        out.append({"tier": t.name, "cli": t.cli, **d})
    return out


def sandbox_action(server, tier_name: str, action: str, req: dict) -> dict:
    """hosts: add or remove an allowed host and set the policy live; asks: approve (adds the host) or reject (hides it);
    policy: write and set it again; start, stop, delete: the sandbox's life."""
    from nebula.sandbox import parse_host
    from nebula.server import reload_models
    cfg = server.cfg
    t, sb = _sandboxed(server, tier_name)
    spec = t.sandbox
    if action in ("hosts", "asks"):
        if action == "asks":
            host, port = str(req.get("host") or "").strip(), int(req.get("port") or 443)
            if not host:
                raise UiError(400, "which ask: host and port")
            what = req.get("action") or "approve"
            if what == "reject":
                spec.setdefault("rejected", [])
                if f"{host}:{port}" not in spec["rejected"]:
                    spec["rejected"].append(f"{host}:{port}")
                saved = cfg.save() if cfg.path else None
                reload_models(server)
                return {"rejected": f"{host}:{port}", "saved": saved}
            entry = f"{host}:{port}:{req.get('access') or 'read-only'}"
        elif req.get("remove"):
            h = parse_host(req["remove"])
            spec["hosts"] = [x for x in spec.get("hosts") or [] if parse_host(x) and (parse_host(x)["host"], parse_host(x)["port"]) != (h["host"], h["port"])]
            saved = cfg.save() if cfg.path else None
            reload_models(server)
            _t, sb = _sandboxed(server, tier_name)
            return {"removed": req["remove"], "saved": saved, **sb.apply_policy()}
        else:
            entry = str(req.get("add") or "").strip()
            if parse_host(entry) is None:
                raise UiError(400, "a host as host, host:port or host:port:read-write")
        h = parse_host(entry)
        spec.setdefault("hosts", [])
        if not any(parse_host(x) and (parse_host(x)["host"], parse_host(x)["port"]) == (h["host"], h["port"]) for x in spec["hosts"]):
            spec["hosts"].append(entry)
        if f"{h['host']}:{h['port']}" in (spec.get("rejected") or []):
            spec["rejected"].remove(f"{h['host']}:{h['port']}")
        saved = cfg.save() if cfg.path else None
        reload_models(server)
        _t, sb = _sandboxed(server, tier_name)
        return {"allowed": entry, "saved": saved, **sb.apply_policy()}
    if action == "policy":
        return sb.apply_policy()
    if action == "start":
        sb.ready = False
        sb.ensure()
        return sb.describe()
    if action == "stop":
        return {"stopped": sb.stop(), **sb.describe()}
    if action == "delete":
        return {"deleted": sb.delete(), **sb.describe()}
    raise UiError(404, f"no such sandbox action: {action}")


def routing_view(server) -> dict:
    """The Configure panels' view: the policy, this Nebula's own models with what their engines report, the models it
    reaches on paired Nebulas, and the paired Nebulas a wormhole can advertise to."""
    cfg, reg = server.cfg, getattr(server, "registry", None)
    nodes = {n["name"]: n for n in (reg.nodes() if reg is not None else [])}
    own = [{"name": t.name, "thinking": t.thinking, "provider": t.provider, "command": t.is_command,   # a command is never advertised
            **{k: nodes.get(t.name, {}).get(k) for k in ("up", "slots", "reported_slots", "context", "vision", "allowed", "why_not", "model")}}
           for t in cfg.tiers if not t.peer]
    remote = [{"name": t.name, "peer": t.peer, **{k: nodes.get(t.name, {}).get(k) for k in ("up", "allowed", "why_not")}}
              for t in cfg.tiers if t.peer]
    peers = [{"uuid": u, "name": p.get("name")} for u, p in ((cfg.lan or {}).get("peers") or {}).items()]
    peers += [{"uuid": u, "name": p.get("name")} for u, p in ((cfg.lan or {}).get("joined") or {}).items() if u not in {x["uuid"] for x in peers}]
    return {"uuid": (getattr(server, "node", None) or {}).get("uuid") or (cfg.lan or {}).get("uuid"),
            "watching": bool(cfg.watch), "policy": server.policy.get() if getattr(server, "policy", None) else {},
            "models": own, "remote_models": remote, "peers": peers}


def lan_scan(server, headers=None) -> dict:
    """What is around this machine right now: engines at their usual ports, CLIs installed."""
    from nebula.lan import connect_lines, probe_clis
    st = lan_status(server)
    ep = endpoints(server, headers)
    lines = connect_lines(ep["server_url"])
    have = {t.cli for t in server.cfg.tiers if t.cli}
    clis = [dict(c, connect=lines.get(c["kind"]), offer=c["binary"] in ("claude", "codex", "grok"), added=c["binary"] in have)
            for c in probe_clis()]
    return {"lan": st,
            "engines": adopt_engines(server), "clis": clis, "server_url": ep["server_url"]}


def adopt_engines(server) -> list[dict]:
    """Engines already running on this machine at their usual ports, and whether each is a tier here already."""
    cfg = server.cfg
    have = {t.address for t in cfg.tiers if t.address}
    from nebula.lan import probe_local_engines
    return [dict(e, added=e["url"] in have) for e in probe_local_engines()]


def monitor(server) -> dict:
    out = {"telemetry": None, "engines": [], "sessions": None,
           "errors": server.errors.count() if getattr(server, "errors", None) is not None else 0}
    if server.telemetry is not None:
        try:
            snap = server.telemetry.snapshot()
            out["telemetry"] = {"now": snap.get("now"), "static": snap.get("static")}
        except Exception as e:  # noqa: BLE001
            out["telemetry"] = {"error": str(e)}
    for t in server.cfg.tiers:
        eng = server.engines.get(t.name)
        if eng is None:
            continue
        status, _ctype, _data = eng.get(getattr(eng, "health_path", "/health"))
        props = eng.props() if status == 200 else {}
        out["engines"].append({"tier": t.name, "address": t.address, "up": status == 200,
                               "busy": eng.busy() if status == 200 else None, "n_ctx": eng.n_ctx(),
                               "slots": props.get("total_slots") if isinstance(props.get("total_slots"), int) else None,
                               "model": props.get("model_alias") or t.model,
                               "unsandboxed": (getattr(getattr(server, "yard", None), "unsandboxed", None) or {}).get(getattr(t, "managed", None) or "")})
    if server.ledger is not None:
        rows = server.ledger.db.execute("SELECT (SELECT count(*) FROM sessions) AS s, (SELECT count(*) FROM attempts) "
                                        "AS a, (SELECT count(*) FROM handoffs) AS h").fetchone()
        out["sessions"] = {"sessions": rows["s"], "attempts": rows["a"], "handoffs": rows["h"]}
    return out


def _moved(tick: dict, before) -> tuple[bool, tuple]:
    """(whether the tick has news for the page, its running state for the next comparison)."""
    now = (tuple((a["attempt_id"], a["tier"], a["tokens"]) for a in tick["running"]),
           tuple(sorted(w["key"] for w in tick["working"])))
    news = any(tick.get(k) for k in ("started", "landed", "handoffs", "briefings", "escalations", "tool_rounds"))
    return news or now != before, now


def stream(server, keepalive: float = 15.0, min_gap: float = 0.2):
    """The page's live feed: SSE frames. The first is a `tick` with what is running; then one whenever something
    moved, at most every `min_gap` seconds (a burst of streamed deltas comes as one tick), and a comment line after
    `keepalive` seconds of quiet. Ends when the ledger closes."""
    act = server.ledger.activity
    watch = Watch(server.ledger, [t.name for t in server.cfg.tiers])
    seen = act.version
    tick = watch.tick()
    _, before = _moved(tick, None)
    yield sse_frame("tick", tick)
    while True:
        v = act.wait(seen, keepalive)
        if act.closing:
            return
        if v == seen:
            yield b": quiet\n\n"
            continue
        seen = v
        tick = watch.tick()
        news, before = _moved(tick, before)
        if news:
            yield sse_frame("tick", tick)
            time.sleep(min_gap)


# ------------------------------------------------------------------------------------------------ the look
LOOK_KEYS = ("pins", "router", "nodes", "widgets", "map", "systems", "worms", "dyson", "showPlan")
LOOK_MAX = 256 * 1024


def look_path(server):
    """The page's layout beside the config: `NAME.look.json` for `NAME.json`. None without a config file."""
    path = getattr(server.cfg, "path", None)
    return Path(path).with_suffix(".look.json") if path else None


def look_read(server) -> dict:
    p = look_path(server)
    if p is None:
        return dict(getattr(server, "look_mem", None) or {})
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def look_write(server, req: dict) -> dict:
    """The page's whole layout, replaced: where the stars and the router stand, each node's name, size and colour,
    and the widgets around the map. Only the known sections are kept."""
    look = {k: req[k] for k in LOOK_KEYS if k in req}
    data = json.dumps(look, ensure_ascii=False, indent=1)
    if len(data) > LOOK_MAX:
        raise UiError(413, "the layout is too large")
    p = look_path(server)
    if p is None:
        server.look_mem = look
        return {"saved": False, "look": look}
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, p)
    return {"saved": True, "file": str(p)}


# ------------------------------------------------------------------------------------------------ watching
def watch_get(server, w, path: str):
    """The page's feeds when this cockpit only watches another router (nebula/switchyard.py): the same shapes as the
    cockpit's own, filled from the watched router's status. None for a path the cockpit answers as usual."""
    cfg = server.cfg
    serving = bool(cfg.tiers)                      # it also has models of its own: its chat and API answer
    if path == "/api/config":
        out = {"version": __version__, "ledger": server.ledger is not None, "watch": w.summary(), "tiers": w.tiers(),
               "agents": w.agents(), "default_models": [], "hooks": [], "schedules": [], "mcp_servers": [],
               "sources": w.sources(), "serving": serving}
        if serving:
            out["own_tiers"] = [{"name": t.name, "address": t.address, "thinking": t.thinking, "managed": getattr(t, "managed", None)} for t in cfg.tiers]
            out["own_agents"] = [{"name": a.name, "default": a.default, "event": a.event} for a in cfg.agents]
        return _json(200, out)
    if path == "/api/watch/lanes":
        return _json(200, {"lanes": w.lanes(), "read_at": w.summary()["read_at"]})
    if path == "/api/watch/decisions":
        return _json(200, w.decisions_feed())
    if path == "/api/watch":
        return _json(200, w.summary())
    if path == "/api/monitor":
        out = monitor(server)
        sm = w.summary()
        out["engines"] = w.engines()
        out["sessions"] = {"sessions": sm["runs_seen"], "attempts": sm["calls_seen"], "handoffs": 0}
        out["watch"] = sm
        return _json(200, out)
    if path == "/api/live":
        return _json(200, w.live())
    if path == "/api/inputs":
        return _json(200, w.inputs())
    if path == "/api/recent":
        return _json(200, w.recent())
    if path == "/api/sessions" and not serving:
        return _json(200, [])
    if path == "/api/registry" and not serving:     # serving: the Models screen lists this Nebula's own models
        return _json(200, [{"name": e["tier"], "up": e["up"], "slots": e["slots"], "context": e["n_ctx"],
                            "model": e["model"], "host": e["address"]} for e in w.engines()])
    if path == "/api/admission":
        return _json(200, w.snap.get("admission") or {})
    if path == "/api/parking":
        return _json(200, {"parked": w.parked()})
    if path in ("/api/lan/scan", "/api/lan/engines"):     # no probes of local engine ports while only watching
        return _json(200, [] if path == "/api/lan/engines" else
                     {"lan": lan_status(server), "engines": [], "clis": [], "watching": True,
                      "server_url": endpoints(server, None)["server_url"]})
    return None


def watch_stream(server, keepalive: float = 15.0):
    """The live feed while watching: a `tick` after every poll that saw a call start or land, else a comment line
    after `keepalive` seconds of quiet. Ends when the watch closes."""
    w = server.watch
    seen = w.version
    yield sse_frame("tick", dict(w.tick(), started=[], landed=[]))
    while True:
        with w.cond:
            if w.version == seen and not w.closing:
                w.cond.wait(keepalive)
            if w.closing:
                return
            v = w.version
        if v == seen:
            yield b": quiet\n\n"
            continue
        seen = v
        yield sse_frame("tick", w.tick())


# ------------------------------------------------------------------------------------------------ POST
def stars_view(server, headers=None) -> dict:
    """The three core stars, as their panels show them."""
    cfg = server.cfg
    ep = endpoints(server, headers)
    from nebula.lan import connect_lines
    counts = {}
    if server.ledger is not None:
        try:
            counts = {r["input"]: r["attempts"] for r in server.ledger.inputs()}
        except Exception:  # noqa: BLE001 - counts are a nicety
            counts = {}
    api = cfg.api or {}
    ch = stars.choices(server)
    return {
        "serving": bool(cfg.tiers),
        "api": {"server_url": ep["server_url"], "openai_base": ep["openai_base"], "anthropic_base": ep["anthropic_base"],
                "responses_url": ep["responses_url"], "connect": connect_lines(ep["server_url"]),
                "requests": {k: counts.get(k, 0) for k in stars.API_INPUTS},
                "require_key": stars.requires_key(cfg),
                "keys": [{k: v for k, v in r.items() if k != "sha256"} for r in api.get("keys") or []],
                "route": api.get("route") or {}, "choices": ch},
        "assistant": {"route": (cfg.assistant or {}).get("route") or {}, "choices": ch,
                      "agent": stars.BUILTIN_AGENT if cfg.agent(stars.BUILTIN_AGENT) else None,
                      "tools": stars.enabled_server_names(cfg) if (cfg.assistant or {}).get("tools", True) else [],
                      "requests": counts.get("room", 0)},
        "lookup": {"servers": stars.lookup_view(cfg, getattr(server, "tools", None)), "builtins": stars.BUILTINS,
                   "library": server.library.view() if getattr(server, "library", None) is not None
                   else {"state": "unavailable", "why": getattr(server, "library_why", None), "collections": []}},
    }


def _reload_tools(server) -> None:
    from nebula.agents import Tools
    old = getattr(server, "tools", None)
    from nebula.rag.tools import local_servers
    server.tools = Tools(stars.hub_servers(server.cfg), server.cfg.mcp_settings, local_servers(server))
    stars.assistant_agent(server.cfg)
    if old is not None:
        old.close()


def halo_post(server, path: str, req: dict) -> dict:
    """POST /api/halo {on, route}; /api/halo/actions/<id>/undo; /api/halo/seen; /api/halo/check (look now)."""
    from nebula.halo import HaloError
    h = getattr(server, "halo", None)
    if h is None:
        raise UiError(503, "Halo is not running")
    cfg = server.cfg
    try:
        if path == "/api/halo":
            a = dict(cfg.assistant or {})
            hs = dict(a.get("halo") or {})
            if "on" in req:
                hs["on"] = bool(req["on"])
            if "route" in req:
                r = req["route"] or {}
                if r and cfg.tier(r.get("tier") or "") is None:
                    raise UiError(400, f"no model {r.get('tier')!r}")
                hs["route"] = {"tier": r["tier"], "model": r.get("model")} if r else {}
            if "where" in req:                         # GPU (the default) or CPU by the owner's choice: a restart
                if req["where"] not in ("gpu", "cpu"):
                    raise UiError(400, "where is gpu or cpu")
                hs["where"] = req["where"]
            if "variant" in req:                       # Standard or Uncensored: the adapter's scale, no reload
                if req["variant"] not in ("standard", "uncensored"):
                    raise UiError(400, "variant is standard or uncensored")
                hs["variant"] = req["variant"]
            a["halo"] = hs
            cfg.assistant = a
            if cfg.path:
                cfg.save()
            if "variant" in req and getattr(server, "halo_engine", None) is not None:
                server.halo_engine.set_variant(req["variant"])
            if "where" in req and getattr(server, "halo_engine", None) is not None and server.halo_engine.wrong_place():
                server.halo_engine.restart()            # placed again: the CPU, or a GPU that has room
            return h.view()
        if path.startswith("/api/halo/actions/") and path.endswith("/undo"):
            h.undo(path[len("/api/halo/actions/"):-len("/undo")])
            return h.view()
        if path == "/api/halo/seen":
            return h.seen()
        if path == "/api/halo/check":
            return {**h.check(), **h.view()}
    except HaloError as e:
        raise UiError(400, str(e)) from None
    raise UiError(404, f"no such path: {path}")


def stars_post(server, path: str, req: dict) -> dict:
    """The stars' settings, saved to the config (when it has a file)."""
    cfg = server.cfg
    out: dict = {}
    if path in ("/api/stars/api", "/api/stars/assistant"):
        star = path.rsplit("/", 1)[1]
        if star == "api" and "require_key" in req:
            on = bool(req["require_key"])
            if on and not (cfg.api or {}).get("keys"):
                raise UiError(400, "make a key first: with none, nothing could call the API")
            cfg.api = dict(cfg.api or {}, require_key=on)
        if "route" in req:
            r = req.get("route") or {}
            stars.set_route(cfg, star, r.get("tier") or None, r.get("model") or None)
        if star == "assistant":
            d = dict(cfg.assistant or {})
            if "system" in req:
                d["system"] = str(req["system"] or "")[:8000] or None
            if "tools" in req:
                d["tools"] = bool(req["tools"])
            cfg.assistant = {k: v for k, v in d.items() if v is not None}
            stars.assistant_agent(cfg)
    elif path == "/api/stars/api/keys":
        row, key = stars.make_key(cfg, str(req.get("label") or ""))
        out = {"key": key, "row": {k: v for k, v in row.items() if k != "sha256"},
               "note": "shown once: copy it now; Nebula keeps only its hash"}
    elif path.startswith("/api/stars/api/keys/") and path.endswith("/revoke"):
        key_id = path[len("/api/stars/api/keys/"):-len("/revoke")]
        if not stars.revoke_key(cfg, key_id):
            raise UiError(404, f"no key {key_id}")
        if stars.requires_key(cfg) and not (cfg.api or {}).get("keys"):
            out["note"] = "that was the last key: apps cannot call the API until you make another (or turn the key off)"
    elif path == "/api/stars/lookup":
        name = str(req.get("name") or "").strip()
        cur = dict((cfg.mcp_servers or {}).get(name) or {})
        merged = {**cur, **{k: v for k, v in req.items() if k != "name"}}
        cfg.mcp_servers = dict(cfg.mcp_servers or {}, **{name: stars.check_server(name, merged)})
        _reload_tools(server)
    elif path.startswith("/api/stars/lookup/") and path.endswith("/remove"):
        name = path[len("/api/stars/lookup/"):-len("/remove")]
        servers = dict(cfg.mcp_servers or {})
        if name not in servers:
            raise UiError(404, f"no server {name}")
        if servers[name].get("builtin"):
            servers[name] = dict(servers[name], disabled=True)     # a built-in is turned off, not removed
        else:
            servers.pop(name)
        cfg.mcp_servers = servers
        _reload_tools(server)
    else:
        raise UiError(404, f"no such path: {path}")
    out["saved"] = cfg.save() if cfg.path else None
    out["stars"] = stars_view(server)
    return out


def switchyard_get(server, path: str, tenant: str | None = None) -> dict:
    """What another Nebula reads to watch this one (docs/router-status-v1.md): its status, its plan, its callers. For a
    guest (`tenant`: a caller's key or a paired peer), only what is its own."""
    from nebula import ownyard, callers
    own = tenant[len("caller:"):] if tenant and tenant.startswith("caller:") else None
    if path == "/switchyard/status":
        return ownyard.status(server, tenant)
    if path == "/switchyard/plan":
        from nebula import plan as P
        p = P.own_plan(server.cfg)
        if tenant is not None:
            p["jobs"] = [j for j in p["jobs"] if own and j["id"] == P.CALLER + own]
        return p
    rows = [{k: c[k] for k in ("id", "name", "class", "tags") if k in c} for c in callers.all_of(server.cfg)]
    return {"callers": [r for r in rows if tenant is None or r["id"] == own]}


SWITCHYARD_RULES = {   # the Switchyard-wide settings the Plan tab edits: name -> (kind, low, high)
    "capacity_mode": ("choice", ("live", "shadow", "off"), None), "sticky_seconds": ("number", 0, 86400),
    "parking_wait_seconds": ("number", 5, 3600), "parking_max_calls": ("int", 0, 1000), "queue_timeout_seconds": ("number", 5, 3600),
    "long_context_tokens": ("int", 1024, 2_000_000), "clamp_output": ("bool", None, None), "retries_before_first_byte": ("int", 0, 5),
    "regenerations_to_rise": ("int", 1, 10), "tool_errors_to_rise": ("int", 1, 20), "escalation_hold_seconds": ("number", 0, 86400),
    "escalation_ignore_outages": ("bool", None, None), "output_reserve_tokens": ("int", 64, 1_000_000), "tolerance_seconds": ("tolerance", None, None),
    "stage_router": ("stage", None, None), "route_rules": ("rules", None, None),
}


def switchyard_rules(cfg) -> dict:
    from nebula.router import Rules
    r = Rules.from_dict(cfg.rules)
    out = {k: getattr(r, k) for k in SWITCHYARD_RULES if k != "stage_router"}
    from nebula import stage
    out["stage_router"] = stage.settings(r)
    return out


def check_rules(req: dict) -> dict:
    out = {}
    for k, v in req.items():
        spec = SWITCHYARD_RULES.get(k)
        if spec is None:
            raise UiError(400, f"no routing setting {k!r}")
        kind, lo, hi = spec
        if kind == "choice":
            if v not in lo:
                raise UiError(400, f"{k} is one of {', '.join(lo)}")
        elif kind == "bool":
            v = bool(v)
        elif kind in ("int", "number"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise UiError(400, f"{k} is a number from {lo} to {hi}")
            v = int(v) if kind == "int" else float(v)
        elif kind == "tolerance":
            if not isinstance(v, dict) or any(c not in ("interactive", "agent", "background") or isinstance(x, bool)
                                               or not isinstance(x, (int, float)) or not 0 <= x <= 3600 for c, x in v.items()):
                raise UiError(400, "tolerance_seconds is {interactive|agent|background: seconds}")
        elif kind == "rules":
            from nebula import route_rules
            try:
                v = route_rules.check(v)
            except route_rules.RuleError as e:
                raise UiError(400, str(e)) from None
        elif kind == "stage":
            if not isinstance(v, dict):
                raise UiError(400, "stage_router is an object")
            if v.get("mode", "off") not in ("off", "shadow", "live") or v.get("picker", "efficient_first") not in ("efficient_first", "capable_first"):
                raise UiError(400, "stage_router: mode off|shadow|live, picker efficient_first|capable_first")
            th = v.get("threshold", 0.5)
            if isinstance(th, bool) or not isinstance(th, (int, float)) or not 0 <= th <= 1:
                raise UiError(400, "stage_router.threshold is 0 to 1")
            v = {kk: v[kk] for kk in ("mode", "picker", "threshold", "judge", "unsure") if kk in v}
        out[k] = v
    return out


def plan_view(server) -> dict:
    """The decision plan: this Nebula's own (editable; Apply saves it) with its models' policy and the Switchyard's
    settings, and, when the optional watch is on, the watched Switchyard's own plan beside it, read only."""
    from nebula import plan as P
    cfg, w = server.cfg, getattr(server, "watch", None)
    pol = server.policy.get() if getattr(server, "policy", None) is not None else {"models": {}}
    out = {"source": {"kind": "own", "name": "This Nebula", "editable": True, "hint": None}, "live": P.own_plan(cfg),
           "policy": pol.get("models") or {}, "rules": switchyard_rules(cfg),
           "models": [{"name": t.name, "cloud": bool(gravity.anti_gravity(t)), "context": t.context} for t in cfg.tiers]}
    if w is not None:
        out["watched"] = {"name": w.name, "plan": P.watched_plan(w.plan_url)}
    return out


def plan_post(server, path: str, req: dict) -> dict:
    """POST /api/plan/apply {plan?, policy?: {model: its policy}, rules?: {setting: value}}: whichever is given."""
    from nebula import plan as P
    from nebula.policy import PolicyError
    cfg = server.cfg
    if path == "/api/plan/apply":
        plan, pol, rules = req.get("plan"), req.get("policy"), req.get("rules")
        if not plan and pol is None and rules is None:
            raise UiError(400, "nothing to apply")
        changed = []
        if rules is not None:
            if not isinstance(rules, dict):
                raise UiError(400, "rules is an object")
            new = check_rules(rules)
            merged = dict(cfg.rules or {})
            merged.update(new)
            cfg.rules = merged
            cfg.raw["rules"] = {**(cfg.raw.get("rules") or {}), **new}
            changed += sorted(new)
        if pol is not None:
            if not isinstance(pol, dict):
                raise UiError(400, "policy is {model: its policy}")
            cur = server.policy.get()
            try:
                server.policy.put({"models": pol, "advertise": cur.get("advertise") or {}})
            except PolicyError as e:
                raise UiError(400, str(e)) from None
            changed.append("models")
        if plan:
            try:
                P.check(plan)
            except ValueError as e:
                raise UiError(400, str(e)) from None
            changed += P.apply_own(cfg, plan).get("changed") or []
        saved = cfg.save() if cfg.path else None
        from nebula.server import reload_models
        reload_models(server)
        from nebula.router import Rules
        if getattr(server, "admission", None) is not None:
            server.admission.tolerance = Rules.from_dict(cfg.rules).tolerance_seconds
        if getattr(server, "lot", None) is not None:
            server.lot.wait_seconds = Rules.from_dict(cfg.rules).parking_wait_seconds
            server.lot.poke()
        return {"applied": {"changed": changed}, "saved": saved, **plan_view(server)}
    raise UiError(404, f"no such path: {path}")


def updates_post(server, path: str, req: dict) -> dict:
    from nebula import updates as U
    u = getattr(server, "updates", None)
    if u is None:
        raise UiError(503, "the update check is not running")
    if path == "/api/updates":
        try:
            U.set_mode(server.cfg, str(req.get("mode") or ""))
        except ValueError as e:
            raise UiError(400, str(e)) from None
        return u.view()
    if path == "/api/updates/check":
        return u.check()
    if path == "/api/updates/install":
        return u.install()
    raise UiError(404, f"no such path: {path}")


def post(server, path: str, engine, ctx: Context, req: dict):
    if path.startswith("/api/halo"):
        return JsonReply(200, halo_post(server, path, req))
    if path.startswith("/api/updates"):
        return JsonReply(200, updates_post(server, path, req))
    if path == "/api/otel":
        from nebula import otel
        o = dict(server.cfg.raw.get("otel") or {})
        if "endpoint" in req:
            try:
                o["endpoint"] = otel.check_endpoint(str(req["endpoint"]))
            except otel.OtelError as e:
                raise UiError(400, str(e)) from None
        if "enabled" in req:
            o["enabled"] = bool(req["enabled"])
            if o["enabled"]:
                try:
                    otel.check_endpoint(o.get("endpoint") or otel.settings(server.cfg)["endpoint"])
                except otel.OtelError as e:
                    raise UiError(400, str(e)) from None
        if isinstance(req.get("headers"), dict):
            o["headers"] = {str(k): str(v) for k, v in req["headers"].items()}
        server.cfg.raw["otel"] = o
        if server.cfg.path:
            server.cfg.save()
        return JsonReply(200, server.otel.view())
    if path.startswith("/api/api-servers"):
        from nebula import entries as EP
        try:
            hit = EP.post(server, path, req)
        except EP.EntryError as e:
            raise UiError(e.status, str(e)) from None
        except Exception as e:  # noqa: BLE001 - a caller check (nebula/callers.py) says what is wrong
            from nebula import callers
            if isinstance(e, callers.CallerError):
                raise UiError(400, str(e)) from None
            raise
        if hit is not None:
            return JsonReply(200, hit)
    if path.startswith("/api/wizards/"):
        from nebula import wizards as W
        try:
            return JsonReply(200, W.post(server, path, req))
        except W.WizardError as e:
            return JsonReply(e.status, {"error": {"message": str(e), "type": "wizard"}, **e.extra})
    if path.startswith("/api/lookup/"):              # Lookup's library (nebula/rag/api.py)
        from nebula.rag import api as rag_api
        from nebula.rag.library import LibraryError
        from nebula.rag.vectors import VectorsError
        try:
            hit = rag_api.post(server, path, req)
        except LibraryError as e:
            raise UiError(e.status, str(e)) from None
        except (VectorsError, OSError) as e:
            raise UiError(502, str(e)) from None
        if hit is not None:
            return JsonReply(200, hit)
    if path.startswith("/api/yard/"):
        from nebula import yard
        try:
            hit = yard.post(server, path, req)
        except yard.YardError as e:
            raise UiError(e.status, str(e)) from None
        if hit is not None:
            return JsonReply(200, hit)
    if path.startswith("/api/plan"):
        return JsonReply(200, plan_post(server, path, req))
    if path.startswith("/api/stars"):
        return JsonReply(200, stars_post(server, path, req))
    if path == "/api/look":
        return JsonReply(200, look_write(server, req))
    if path == "/api/routing":
        from nebula.policy import PolicyError
        try:
            server.policy.put(req)
        except PolicyError as e:
            raise UiError(400, str(e)) from None
        if getattr(server, "admission", None) is not None:
            server.admission.refresh(server.cfg, server.registry)   # the gates take the new capacities
        if getattr(server, "lot", None) is not None:
            server.lot.poke()
        return JsonReply(200, routing_view(server))
    if path in ("/api/errors", "/api/errors/clear"):
        log = getattr(server, "errors", None)
        if log is None:
            raise UiError(503, "no error log")
        if path == "/api/errors/clear":
            return JsonReply(200, {"cleared": log.clear()})
        message = req.get("message") if isinstance(req.get("message"), str) else ""
        if not message.strip():
            raise UiError(400, "an error needs a message")
        detail = req.get("detail") if isinstance(req.get("detail"), str) else None
        return JsonReply(200, log.add(message, source="page", level="error", detail=detail))
    if path == "/api/app/engine/start":
        app = getattr(server, "app", None)
        if app is None:
            raise UiError(404, "not the desktop app: its engine is started elsewhere")
        return JsonReply(200 if app.restart_engine() else 409, app.status())
    ledger = server.ledger
    if ledger is None:
        raise UiError(503, "no ledger: start the cockpit with one for sessions")
    if path == "/api/chat":
        return chat_post(server, ctx, req)
    if path == "/api/providers/models":
        base = str(req.get("base_url") or req.get("url") or "")
        if not base.startswith("http"):
            raise UiError(400, "base_url is needed, such as https://api.example.com/v1")
        import os
        key = req.get("api_key") or (os.environ.get(req["api_key_env"]) if req.get("api_key_env") else None)
        return JsonReply(200, {"base_url": base, "models": provider_models(base, key)})
    if path == "/api/models":
        return JsonReply(200, add_model(server, req))
    if path.startswith("/api/models/") and path.endswith("/remove"):
        return JsonReply(200, remove_model(server, path[len("/api/models/"):-len("/remove")]))
    if path.startswith("/api/models/") and path.endswith("/shared"):
        return JsonReply(200, set_shared(server, path[len("/api/models/"):-len("/shared")], req))
    if path == "/api/policy":
        return JsonReply(200, set_policy(server, req))
    if path.startswith("/api/sandboxes/"):
        rest = path[len("/api/sandboxes/"):].strip("/").split("/")
        if len(rest) == 2:
            return JsonReply(200, sandbox_action(server, rest[0], rest[1], req))
    if path.startswith("/api/sessions/"):
        rest = path[len("/api/sessions/"):].strip("/").split("/")
        if len(rest) == 2 and rest[1] in ("close", "reopen"):
            sid = rest[0]
            if ledger.session(sid) is None:
                raise UiError(404, f"no session {sid}")
            ledger.set_lifecycle(sid, "closed" if rest[1] == "close" else "active", "user")
            return JsonReply(200, {"session_id": sid, "lifecycle": ledger.session(sid)["lifecycle"]})
    raise UiError(404, f"no such path: {path}")


IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
MAX_IMAGE, MAX_FILE_B64, MAX_TEXT = 10 * 2**20, 28 * 2**20, 400_000


def chat_content(text: str, atts: list) -> str | list:
    """The chat's message with what was attached: photos as pictures for the model,
    text files as text, documents read into text here (Lookup's reader: PDF, Word, … with Docling). A message with a
    picture is a list of parts; else one text."""
    import base64
    import tempfile
    from pathlib import Path as _P
    files, images, total = [], [], 0
    for a in atts[:20]:
        if not isinstance(a, dict):
            continue
        name = str(a.get("name") or "a file")[:120]
        kind = a.get("kind")
        if kind == "image":
            url = str(a.get("data_url") or "")
            head = url.split(",", 1)[0]
            if not any(head == f"data:{t};base64" for t in IMAGE_TYPES):
                raise UiError(400, f"{name}: a photo is PNG, JPEG, WebP or GIF")
            if len(url) * 3 // 4 > MAX_IMAGE:
                raise UiError(413, f"{name}: a photo up to {MAX_IMAGE // 2**20} MB")
            images.append({"type": "image_url", "image_url": {"url": url}})
            continue
        if kind == "text":
            body = str(a.get("text") or "")
        elif kind == "file":
            raw = str(a.get("data") or "")
            if len(raw) > MAX_FILE_B64:
                raise UiError(413, f"{name}: a document up to {MAX_FILE_B64 * 3 // 4 // 2**20} MB")
            from nebula.rag import ingest
            suffix = _P(name).suffix.lower() or ".bin"
            with tempfile.TemporaryDirectory() as d:
                f = _P(d) / ("upload" + suffix)
                f.write_bytes(base64.b64decode(raw))
                try:
                    sections, _info = ingest.read(f)
                except ingest.Skip as e:
                    raise UiError(415, f"{name}: Nebula cannot read it here ({e})") from None
                except Exception as e:  # noqa: BLE001 - a broken file is the sender's to fix
                    raise UiError(415, f"{name}: it could not be read ({type(e).__name__})") from None
            body = "\n\n".join(x.get("text") or "" for x in sections)
        else:
            continue
        room = MAX_TEXT - total
        if room <= 0:
            files.append(f"[{name}: left out, the attachments are long enough already]")
            continue
        cut = body[:room]
        total += len(cut)
        files.append(f"[{name}]\n{cut}" + ("\n[… the rest was cut]" if len(body) > room else ""))
    full = "\n\n".join([x for x in [text.strip()] if x] + files)
    if not images:
        return full
    return ([{"type": "text", "text": full}] if full else []) + images


def chat_post(server, ctx: Context, req: dict):
    cfg, ledger = server.cfg, server.ledger
    sid = req.get("session_id")
    text = req.get("text") if isinstance(req.get("text"), str) else ""
    regen = req.get("regenerate_from_seq")
    if sid:
        s = ledger.session(sid)
        if s is None:
            raise UiError(404, f"no session {sid}")
        if s["lifecycle"] == "closed":
            raise UiError(409, "this session is closed: reopen it to continue")
        key = s["session_key"]
        until = int(regen) if isinstance(regen, int) else None
        messages = [rendered_message(p) for _seq, p in ledger.transcript(sid, until_seq=until)]
        if until is not None and messages and messages[-1].get("role") == "assistant":
            messages.pop()                   # regenerate: the reply itself goes, its question stays
    else:
        key = "ui-" + uuid.uuid4().hex[:16]
        messages = []
    atts = req.get("attachments") if isinstance(req.get("attachments"), list) else []
    if text.strip() or atts:
        messages.append({"role": "user", "content": chat_content(text, atts)})
    if not messages:
        raise UiError(400, "nothing to send")
    tier = req.get("tier") if isinstance(req.get("tier"), str) and cfg.tier(req["tier"]) else None
    headers = dict(ctx.headers or {})
    names.set_header(headers, "Session", key)
    escalate = req.get("escalate")
    names.set_header(headers, "Escalate", escalate if isinstance(escalate, str) and escalate else None)
    agent = req.get("agent")
    if isinstance(agent, str) and agent:
        if cfg.agent(agent) is None:
            raise UiError(400, f"no agent named {agent!r}")
        names.set_header(headers, "Agent", agent)
    elif cfg.default_agent() is None and cfg.agent(stars.BUILTIN_AGENT) is not None:
        names.set_header(headers, "Agent", stars.BUILTIN_AGENT)     # the Assistant star's own agent
    else:
        names.set_header(headers, "Agent", None)
    body = {"model": names.model(tier), "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}}
    reply = chat(server.engine, Context(cfg, ledger, headers, ctx.engines, ctx.tools, input="room",
                                        registry=getattr(server, "registry", None), admission=getattr(server, "admission", None),
                                        lot=getattr(server, "lot", None), yard=getattr(server, "yard", None),
                                        policy=getattr(server, "policy", None), power=getattr(server, "power", None),
                                        slots=getattr(server, "slots", None), seclog=getattr(server, "seclog", None),
                                        guard=getattr(server, "guard", None)), body)
    frames = reply.frames
    first = next(frames, None)               # the session exists once the first frame was pulled
    row = ledger.session_by_key(key)

    def rest():
        try:
            if first is not None:
                yield first
            yield from frames
        finally:
            frames.close()

    sid = row["session_id"] if row else ""
    return SseReply(rest(), {"X-Nebula-Session-Id": sid, "X-Cockpit-Session-Id": sid})
