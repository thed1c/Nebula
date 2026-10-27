"""nebula/ownyard.py: this Nebula's own Switchyard, as others see a router: its status, its lanes, its decisions and
its stats.

- `status(server)` is the router-status-v1 answer (docs/router-status-v1.md) for this Nebula's own models, gates,
  calls, waits, parking and callers. It is served at `GET /switchyard/status`, so another Nebula (or any client) can
  watch this one the way this one watches a router; and the page's own Lanes come from it through the same reader
  the watch uses (nebula/switchyard.py), so a lane looks the same whoever's it is.
- `decisions(server)` is the decisions feed from the ledger itself: every decision, not the last of each kind, with
  its stages (nebula/turn.py) and its caller.
- `stats(server)` is per model and per caller over the last 10 minutes, 24 hours and 7 days: calls, latency and queue
  wait (average / max), prefill and decode seconds, tokens per second, prompt tokens (average / max / total) and the
  prompt-size bands.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

BANDS = [("<2k", 0, 2048), ("2k-8k", 2048, 8192), ("8k-16k", 8192, 16384), ("16k-32k", 16384, 32768),
         ("32k-64k", 32768, 65536), ("64k-128k", 65536, 131072), (">128k", 131072, None)]
WINDOWS = {"10m": 600, "24h": 86400, "7d": 7 * 86400}


def _epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


def status(server, tenant: str | None = None) -> dict:
    """The status; for a guest (`tenant`), with only its own calls, waits and callers in it."""
    out = _status(server)
    if tenant is None:
        return out
    from nebula import callers
    own_caller = tenant[len("caller:"):] if tenant.startswith("caller:") else None
    labels = {c["id"] for c in callers.all_of(server.cfg)}
    mine = lambda row: (row.get("tenant") == tenant)
    out["in_flight_calls"] = [c for c in out["in_flight_calls"] if mine(c)]
    out["waiting"] = [w for w in out["waiting"] if mine(w)]
    for g in out["admission"]["groups"].values():
        g["holders"] = [h for h in g["holders"] if mine(h)]
    out["callers"] = [c for c in out["callers"] if c["id"] == own_caller]
    out["capacity"]["parking"] = []
    out["switchyard"].pop("last", None)
    out["switchyard"]["actors"] = {}
    del labels
    return out


def _status(server) -> dict:
    cfg, adm, reg = server.cfg, server.admission, server.registry
    yard, pol = getattr(server, "yard", None), getattr(server, "policy", None)
    managed = {}
    if yard is not None:
        for s in yard.specs():
            managed[s["name"]] = s
    dests, routes, groups, flights, waiting = {}, {}, {}, [], []
    for t in cfg.tiers:
        if not t.address:
            continue
        g = adm.gate(t) if adm is not None else None
        gs = g.status() if g is not None else {"slots": 1, "held": 0, "waiting": [], "holding": []}
        up = reg.up(t.name) if reg is not None else None
        why = None
        if pol is not None:
            ok, w = pol.allowed(t.name)
            why = None if ok else w
        if yard is not None and yard.is_draining(t.address):
            why = "draining"
        eng = server.engines.get(t.name)
        n_ctx = None
        try:
            n_ctx = eng.n_ctx() if (eng is not None and up) else None
        except Exception:  # noqa: BLE001 - an engine that cannot say
            n_ctx = None
        dests[t.name] = {"label": t.name, "routes": [t.name], "slots": gs["slots"], "busy": gs["held"],
                         "free": max(0, gs["slots"] - gs["held"]), "queued": len(gs["waiting"]), "slot_context": n_ctx,
                         "avg_call_seconds": round(adm.average(t), 2) if adm is not None else None, "unavailable": why}
        spec = managed.get(t.managed) if getattr(t, "managed", None) else None
        r = {"base_url": t.address, "ready": bool(up) if up is not None else None, "managed": t.managed}
        if spec and spec.get("model"):
            r["expected_model_file"] = os.path.basename(spec["model"])
            props = eng.props() if (eng is not None and up) else {}
            if props.get("model_path"):
                r["model_file"] = os.path.basename(props["model_path"])
        routes[t.name] = r
        groups[t.name] = {"label": t.name, "size": gs["slots"], "busy": gs["held"], "free": max(0, gs["slots"] - gs["held"]),
                          "queued": len(gs["waiting"]), "on_hold": why,
                          "holders": [{"actor": h.get("agent"), "route": t.name, "since": h.get("since"), "tenant": h.get("tenant")}
                                      for h in gs["holding"]]}
        for h in gs["holding"]:
            flights.append({"actor": h.get("agent") or h.get("input"), "caller": h.get("label") or h.get("caller") or h.get("input"),
                            "run": h.get("session_id"), "route": t.name, "since": h.get("since"), "tenant": h.get("tenant")})
        for w in gs["waiting"]:
            waiting.append({"id": w.get("attempt_id"), "caller": w.get("label") or w.get("caller"), "actor": w.get("agent"),
                            "class": w.get("class"), "group": t.name, "reason": "waiting for a slot",
                            "since": time.time() - float(w.get("waited_s") or 0), "label": t.name, "tenant": w.get("tenant")})
    lot = getattr(server, "lot", None)
    parking = []
    if lot is not None:
        try:
            parking = lot.parked()
        except Exception:  # noqa: BLE001
            parking = []
    from nebula import callers
    out = {"switchyard": {"actors": {a.name: ("event" if a.event else "live") for a in cfg.agents}, "mode": "composite"},
           "capacity": {"destinations": dests, "parking": parking, "mode": "composite"},
           "routes": routes, "admission": {"groups": groups}, "in_flight_calls": flights, "waiting": waiting,
           "callers": [{k: c[k] for k in ("id", "name", "class", "tags") if k in c} for c in callers.all_of(cfg)],
           "generated_at": time.time(), "router": "nebula",
           "name": str((cfg.lan or {}).get("name") or "Nebula")}
    last = decisions(server, limit=1)["decisions"]
    if last:
        d = last[0]
        out["switchyard"]["last"] = {"at": _epoch(d.get("created_at")), "actor": d.get("actor"), "caller": d.get("caller"),
                                     "mode": "composite", "tier": d.get("tier"), "selected_route": d.get("tier"),
                                     "switchyard_order": d.get("order"), "routed": d.get("tier")}
    return out


def decisions(server, limit: int = 100, tenant: str | None = None) -> dict:
    """The newest decisions from the ledger, in the watch feed's row shape (kind "route"), with their stages; for a
    guest (`tenant`), only its own."""
    ledger = server.ledger
    if ledger is None:
        return {"decisions": [], "counters": {}}
    rows = ledger.db.execute(
        "SELECT d.decision_id, d.tier, d.reason, d.inputs, d.created_at, a.agent, a.dialect, "
        "(SELECT i.inputs FROM decisions i WHERE i.attempt_id = d.attempt_id AND i.advisory = 1 ORDER BY i.rowid DESC LIMIT 1) "
        "FROM decisions d LEFT JOIN attempts a ON a.attempt_id = d.attempt_id WHERE d.advisory = 0 "
        "ORDER BY d.created_at DESC, d.rowid DESC LIMIT ?", (int(limit) * (20 if tenant else 1),)).fetchall()
    out = []
    for did, tier, reason, inputs, at, agent, dialect, adv in rows:
        try:
            i = json.loads(inputs)
            iso = json.loads(adv) if adv else {}
        except ValueError:
            i, iso = {}, {}
        if tenant is not None and (i.get("tenant") or iso.get("tenant") or "owner") != tenant:
            continue
        if len(out) >= limit:
            break
        stages = i.get("stages") or []
        order = next((s.get("models") for s in stages if s.get("stage") == "ticks"), None)
        out.append({"kind": "retry" if str(reason).startswith("retry") else "route", "id": did, "at": at, "created_at": at,
                    "tier": tier, "route": tier, "applied": tier, "reason": reason, "caller": i.get("label") or i.get("caller"), "source": i.get("caller"),
                    "actor": agent, "input": dialect, "class": i.get("priority"), "order": order if isinstance(order, list) else None,
                    "stages": stages, "wait_s": i.get("expected_wait_s"), "from": i.get("from"),
                    "isolation": iso.get("isolation")})
    if tenant is not None:
        return {"decisions": out, "counters": {}}
    counts = dict(ledger.db.execute("SELECT tier, COUNT(*) FROM decisions WHERE advisory = 0 GROUP BY tier").fetchall())
    retries = ledger.db.execute("SELECT COUNT(*) FROM decisions WHERE reason LIKE 'retry%'").fetchone()[0]
    return {"decisions": out, "counters": {"route": counts, "retries": retries, "route_mode": "composite"}}


def stats(server) -> dict:
    """Per model and per caller, over each window."""
    ledger = server.ledger
    if ledger is None:
        return {"windows": list(WINDOWS), "models": {}, "callers": {}}
    since = (datetime.now(timezone.utc) - timedelta(seconds=max(WINDOWS.values()))).isoformat()
    rows = ledger.db.execute(
        "SELECT a.attempt_id, a.tier, a.created_at, a.finished_at, a.state, c.prompt_tokens, c.completion_tokens, "
        "c.prompt_ms, c.predicted_ms, c.predicted_n, "
        "(SELECT inputs FROM decisions d WHERE d.attempt_id = a.attempt_id AND d.advisory = 0 ORDER BY d.created_at DESC, d.rowid DESC LIMIT 1), "
        "(SELECT value FROM outcomes o WHERE o.attempt_id = a.attempt_id ORDER BY o.recorded_at DESC LIMIT 1) "
        "FROM attempts a LEFT JOIN costs c ON c.attempt_id = a.attempt_id WHERE a.created_at >= ?", (since,)).fetchall()
    now = time.time()
    models: dict = {}
    by_caller: dict = {}
    for aid, tier, created, finished, state, pt, ct, pms, dms, dn, dec, outc in rows:
        t0 = _epoch(created)
        if t0 is None:
            continue
        try:
            caller = (json.loads(dec) if dec else {}).get("caller") or "(none)"
        except ValueError:
            caller = "(none)"
        try:
            queued = (json.loads(outc) if outc else {}).get("queued_s")
        except ValueError:
            queued = None
        t1 = _epoch(finished)
        rec = {"latency": (t1 - t0) if t1 else None, "queued": queued, "prefill": pms / 1000 if pms else None,
               "decode": dms / 1000 if dms else None, "tps": (dn / dms * 1000) if (dn and dms) else None, "prompt": pt}
        for name, window in WINDOWS.items():
            if now - t0 > window:
                continue
            for bucket, key in ((models, tier), (by_caller, caller)):
                b = bucket.setdefault(key, {}).setdefault(name, {"calls": 0, "vals": {k: [] for k in rec}, "bands": {n: 0 for n, _a, _b in BANDS}})
                b["calls"] += 1
                for k, v in rec.items():
                    if isinstance(v, (int, float)) and v >= 0:
                        b["vals"][k].append(v)
                if isinstance(pt, int):
                    for n, lo, hi in BANDS:
                        if pt >= lo and (hi is None or pt < hi):
                            b["bands"][n] += 1

    def summary(b):
        out = {"calls": b["calls"], "bands": b["bands"]}
        for k, vs in b["vals"].items():
            out[k] = {"avg": round(sum(vs) / len(vs), 3), "max": round(max(vs), 3), "n": len(vs),
                      **({"total": int(sum(vs))} if k == "prompt" else {})} if vs else None
        return out
    return {"windows": list(WINDOWS), "bands": [n for n, _a, _b in BANDS],
            "models": {m: {w: summary(b) for w, b in ws.items()} for m, ws in models.items()},
            "callers": {c: {w: summary(b) for w, b in ws.items()} for c, ws in by_caller.items()}}
