"""nebula/plan.py: the decision plan, what chooses to do what, when, as rows you can read and edit.

A plan (plan-v1) is plain data:

    {"schema": "nebula-plan-v1", "name": "...",
     "classes": [{"name": "interactive", "rank": 0, "tolerance_s": 60}, ...],       # who goes first at a full model
     "models":  [{"id": "big", "name": "Big model", "slots": 1}, ...],
     "jobs":    [{"id": "studio", "name": "The Studio", "class": "interactive",
                  "order": ["big", "small"],          # the models it may use, in order: the first with a free slot wins
                  "ticked": true,                     # false = the order is its default (no choice saved for it)
                  "tags": ["writer/*"]}, ...],        # the caller labels it covers (globs)
     "settings": {"park_wait_s": 600, "sticky_s": 7200}}

Rules every row shares: a call takes the first model in its job's order with a free slot; when none is free it waits
up to its class's tolerance for its first choice, then takes the next one that can hold it; too big for all of them,
it is parked (up to park_wait_s); a run stays on its first model for sticky_s.

Two sources:
- **This Nebula's own plan**, from its config: a job per agent (its ticks = its models), and one for requests that
  name no agent (the default ticks). Saving it writes the config (agents' models, default_models, the tolerances).
- **A watched Switchyard's plan** (the optional watch, nebula/switchyard.py): its `/switchyard/plan`, shown read only.
  Nebula never changes a Switchyard it watches.
"""
from __future__ import annotations

import json
import time
import urllib.request

SCHEMA = "nebula-plan-v1"
CALLER = "caller:"                                # a job that is a registered caller, by its id
DEFAULT_JOB = "(no agent)"


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --- this Nebula's own plan
def own_plan(cfg) -> dict:
    from nebula.router import Rules
    rules = Rules.from_dict(cfg.rules)
    classes = [{"name": c, "rank": i, "tolerance_s": rules.tolerance_seconds.get(c)} for i, c in
               enumerate(("interactive", "agent", "background"))]
    models = [{"id": t.name, "name": t.name, "slots": t.slots} for t in cfg.tiers]
    every = [t.name for t in cfg.tiers]
    jobs = [{"id": DEFAULT_JOB, "name": "Requests that name no agent", "class": "interactive",
             "order": list(cfg.default_models) or every, "ticked": bool(cfg.default_models), "tags": []}]
    for a in cfg.agents:
        if a.event or getattr(a, "hidden", False):
            continue
        jobs.append({"id": a.name, "name": "Halo" if a.name == "assistant" else a.name, "class": a.priority or "agent",
                     "order": list(a.models) or list(cfg.default_models) or every, "ticked": bool(a.models), "tags": []})
    for c in getattr(cfg, "callers", None) or []:                # a registered caller is a job too (nebula/callers.py)
        jobs.append({"id": CALLER + c["id"], "name": c.get("name") or c["id"], "class": c.get("class") or "interactive",
                     "order": list(c.get("models") or []) or list(cfg.default_models) or every,
                     "ticked": bool(c.get("models")), "tags": list(c.get("tags") or [])})
    return {"schema": SCHEMA, "name": "This Nebula", "classes": classes, "models": models, "jobs": jobs,
            "settings": {"park_wait_s": rules.parking_wait_seconds, "sticky_s": None}}


def apply_own(cfg, plan: dict) -> dict:
    """Write an edited own plan into the config: each agent's ticks, the default ticks, the class tolerances."""
    check(plan)
    names = {t.name for t in cfg.tiers}
    changed = []
    for j in plan["jobs"]:
        order = [m for m in j.get("order") or [] if m in names]
        if j["id"] == DEFAULT_JOB:
            if j.get("ticked", True) and order != list(cfg.default_models):
                cfg.default_models = order
                changed.append(DEFAULT_JOB)
            continue
        if str(j["id"]).startswith(CALLER):
            c = next((x for x in getattr(cfg, "callers", None) or [] if x["id"] == j["id"][len(CALLER):]), None)
            if c is None:
                continue
            new = order if j.get("ticked", True) else []
            if new != list(c.get("models") or []):
                c["models"] = new
                if not new:
                    c.pop("models", None)
                changed.append(j["id"])
            if j.get("class") in ("interactive", "agent", "background") and j["class"] != (c.get("class") or "interactive"):
                c["class"] = j["class"]
                changed.append(j["id"] + " (class)")
            continue
        a = cfg.agent(j["id"])
        if a is None or a.event:
            continue
        new = order if j.get("ticked", True) else []
        if new != list(a.models):
            a.models = new
            changed.append(a.name)
        if j.get("class") in ("interactive", "agent", "background") and j["class"] != (a.priority or "agent"):
            a.priority = j["class"]
            changed.append(a.name + " (class)")
    tol = {c["name"]: float(c["tolerance_s"]) for c in plan.get("classes") or [] if _num(c.get("tolerance_s")) is not None}
    rules = dict(cfg.rules or {})
    if tol and tol != (rules.get("tolerance_seconds") or {}):
        rules["tolerance_seconds"] = {**(rules.get("tolerance_seconds") or {}), **tol}
        changed.append("tolerances")
    pw = _num((plan.get("settings") or {}).get("park_wait_s"))
    if pw is not None and pw != rules.get("parking_wait_seconds"):
        rules["parking_wait_seconds"] = pw
        changed.append("park wait")
    cfg.rules = rules
    raw_rules = dict(cfg.raw.get("rules") or {})
    raw_rules.update({k: rules[k] for k in ("tolerance_seconds", "parking_wait_seconds") if k in rules})
    cfg.raw["rules"] = raw_rules
    return {"changed": changed}


# --- a watched Switchyard's plan
_WATCHED: dict = {}


def watched_plan(url: str | None, ttl: float = 30.0, timeout: float = 10.0) -> dict | None:
    """The watched Switchyard's plan (its /switchyard/plan), read at most every `ttl` seconds; None without a watch."""
    if not url:
        return None
    at, got = _WATCHED.get(url, (0.0, None))
    if got is not None and time.time() - at < ttl:
        return got
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            plan = json.loads(r.read())
        check(plan)
    except Exception as e:  # noqa: BLE001 - shown on the page, not fatal
        plan = {"error": f"{type(e).__name__}: {e}"[:300]}
    _WATCHED[url] = (time.time(), plan)
    return plan


# --- checks
def check(plan: dict) -> None:
    if not isinstance(plan, dict) or not isinstance(plan.get("jobs"), list) or not isinstance(plan.get("models"), list):
        raise ValueError("a plan has jobs and models")
    ids = {m.get("id") for m in plan["models"] if isinstance(m, dict)}
    for j in plan["jobs"]:
        if not isinstance(j, dict) or not j.get("id") or not isinstance(j.get("order", []), list):
            raise ValueError("each job has an id and an order (a list of model ids)")
        bad = [m for m in j.get("order") or [] if m not in ids]
        if bad:
            raise ValueError(f"job {j['id']}: no model named {bad[0]}")
        if len(set(j.get("order") or [])) != len(j.get("order") or []):
            raise ValueError(f"job {j['id']}: a model is listed twice")
