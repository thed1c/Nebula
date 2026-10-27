"""Watching another Switchyard by URL, read only (optional; off unless the config has a `watch`).

A Nebula whose config has a `watch` reads another Switchyard's status every `poll_seconds` and the page draws that
Switchyard's world in Nebula's look: its destinations are the models, its actors the ring's stars, its callers the
sources, and its calls in flight the lights between them, with its waits and parked requests. Nothing is ever written
to the watched Switchyard, and nothing about it is configured here beyond where it is:

    "watch": {"url": "http://127.0.0.1:18300", "poll_seconds": 2}

`url` is the Switchyard's base address (another Nebula's own port, say); its `/switchyard/status`,
`/switchyard/registry` (the callers) and `/switchyard/plan` are read from there (docs/router-status-v1.md). A `url`
that already ends in `/switchyard/status` is taken as its status. Its name and its actors are the ones its status
reports. Calls are seen only while they are in flight, so a call shorter than one poll can be missed; the counts here
start when this Nebula starts.
"""

from __future__ import annotations

import fnmatch
import json
from collections import deque
import logging
import threading
import time
import urllib.request
from datetime import datetime, timezone

LOG = logging.getLogger("cockpit.switchyard")
DECISIONS_KEPT = 200                         # the decisions feed: the newest this many, kept by the watcher


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


class SwitchyardWatch:
    def __init__(self, spec: dict, fetch=None):
        url = spec.get("url") if isinstance(spec, dict) else None
        if not isinstance(url, str) or not url:
            raise ValueError('"watch" needs a "url" (the other Switchyard\'s address)')
        gone = [k for k in ("status_url", "registry_file", "registry_url", "plan_command", "plan_file", "plan_url", "agents", "name")
                if k in spec]
        if gone:
            raise ValueError(f'"watch": {", ".join(gone)} {"is" if len(gone) == 1 else "are"} no longer read: a watch is its "url" only')
        self.base = url.rstrip("/")
        if self.base.endswith("/switchyard/status"):
            self.base = self.base[:-len("/switchyard/status")]
        self.url = self.base + "/switchyard/status"
        self.registry_url = self.base + "/switchyard/registry" if self.base.startswith(("http://", "https://")) else None
        self.plan_url = self.base + "/switchyard/plan" if self.registry_url else None
        self.registry_every = 30.0
        self._reg_read_at = 0.0
        self.poll = max(0.5, float(spec.get("poll_seconds", 2.0)))
        self.timeout = float(spec.get("timeout_seconds", 5.0))
        self._fetch = fetch or self._http
        self._injected = fetch is not None             # a given fetch (the tests' fake router): nothing else goes out
        self.snap: dict = {}
        self.error: str | None = None
        self.at = 0.0
        self.version = 0
        self.cond = threading.Condition()
        self.closing = False
        self.callers: list = []
        self._reg_mtime = None
        self.running: dict = {}                # call key -> call row, as the page's attempts
        self.started: list = []                # call rows since the last version
        self.landed: list = []
        self.finished: list = []               # the latest finished calls, newest first
        self.counts: dict = {}                 # source key -> {"attempts", "last_at"}
        self.runs: set = set()
        self.decisions: deque = deque(maxlen=DECISIONS_KEPT)   # newest last; the status itself keeps only the last one
        self._decided: dict = {}               # kind -> the key of the last decision of that kind already kept
        self.thread = None

    @property
    def name(self) -> str:
        """The name the watched Switchyard reports, else its address."""
        n = self.snap.get("name") or (self.snap.get("switchyard") or {}).get("name") if isinstance(self.snap, dict) else None
        if isinstance(n, str) and n.strip():
            return n.strip()[:60]
        from urllib.parse import urlparse
        u = urlparse(self.base)
        return u.netloc or self.base

    # --- polling
    def _http(self) -> dict:
        with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
            return json.loads(r.read())

    def start(self) -> "SwitchyardWatch":
        self.poll_once()
        self.thread = threading.Thread(target=self._loop, name="switchyard-watch", daemon=True)
        self.thread.start()
        return self

    def close(self) -> None:
        with self.cond:
            self.closing = True
            self.cond.notify_all()

    def _loop(self) -> None:
        while True:
            with self.cond:
                if self.closing:
                    return
                self.cond.wait(self.poll)
                if self.closing:
                    return
            self.poll_once()

    def poll_once(self) -> None:
        try:
            snap = self._fetch()
            if not isinstance(snap, dict):
                raise ValueError("the status is not a JSON object")
            error = None
            self._load_registry(snap)
        except Exception as e:  # noqa: BLE001 - the watched router is down or restarting: keep the last view
            snap, error = None, f"{type(e).__name__}: {e}"[:300]
        with self.cond:
            changed = error != self.error
            self.error = error
            if snap is not None:
                self.snap, self.at = snap, time.time()
                changed = self._diff(snap) or changed
                changed = self._note_decisions(snap) or changed
            else:
                self.started, self.landed = [], []
            if changed:
                self.version += 1
                self.cond.notify_all()

    def _load_registry(self, snap: dict) -> None:
        inline = snap.get("callers")
        if isinstance(inline, list):
            self.callers = [c for c in inline if isinstance(c, dict) and c.get("id")]
            return
        path = self.registry_url
        if not path or self._injected:                 # with a given fetch, its own answer's callers only (no DNS, no
            return                                     # network: a test's made-up host could stall a runner for ever)
        if time.time() - self._reg_read_at < self.registry_every and self._reg_mtime == path:
            return
        try:
            with urllib.request.urlopen(path, timeout=self.timeout) as r:
                d = json.loads(r.read())
            self.callers = [c for c in d.get("callers") or [] if isinstance(c, dict) and c.get("id")]
            self._reg_mtime = path
        except (OSError, ValueError, AttributeError) as e:
            LOG.warning("the watched Switchyard's callers (%s) cannot be read: %s", path, e)
        self._reg_read_at = time.time()

    # --- what the status means
    def source_of(self, tag: str | None) -> tuple[str, str, str]:
        """(key, label, class) of a caller tag: the first registry caller whose glob matches, else the tag."""
        tag = tag or "untagged"
        for c in self.callers:
            if any(fnmatch.fnmatchcase(tag, g) for g in c.get("tags") or []):
                return str(c["id"]), str(c.get("name") or c["id"]), str(c.get("class") or "")
        return tag, tag, ""

    def destinations(self) -> dict:
        return ((self.snap.get("capacity") or {}).get("destinations")) or {}

    def route_dest(self, route: str | None) -> str | None:
        for name, d in self.destinations().items():
            if route in (d.get("routes") or []):
                return name
        return route

    def _diff(self, snap: dict) -> bool:
        now = {}
        for c in snap.get("in_flight_calls") or []:
            if not isinstance(c, dict):
                continue
            key = "|".join(str(c.get(k)) for k in ("actor", "caller", "run", "since"))
            src, label, cls = self.source_of(c.get("caller"))
            since = float(c.get("since") or time.time())
            now[key] = {"attempt_id": key, "dialect": src, "title": label, "class": cls, "caller": c.get("caller"),
                        "tier": self.route_dest(c.get("route")), "route": c.get("route"), "agent": c.get("actor"),
                        "session_id": c.get("run") or key, "tokens": 0, "since": since, "created_at": _iso(since)}
        started = [a for k, a in now.items() if k not in self.running]
        landed = [dict(a, state="completed", seconds=round(time.time() - a["since"], 1))
                  for k, a in self.running.items() if k not in now]
        for a in started:
            n = self.counts.setdefault(a["dialect"], {"attempts": 0, "last_at": None})
            n["attempts"] += 1
            n["last_at"] = a["created_at"]
            self.runs.add(a["session_id"])
        for a in landed:
            self.counts.setdefault(a["dialect"], {"attempts": 0, "last_at": None})["last_at"] = _iso(time.time())
        self.finished = (landed[::-1] + self.finished)[:40]
        self.running = now
        self.started, self.landed = started, landed
        return bool(started or landed)

    # --- the decisions feed: each poll's last decisions, kept when they are new
    def _note_decisions(self, snap: dict) -> bool:
        """Keep the router's latest decisions (the status shows only the last of each kind) as a feed, newest last."""
        new = False
        cap = snap.get("capacity") or {}
        found = [("capacity", cap.get("last")), ("route", (snap.get("switchyard") or {}).get("last")),
                 ("admission", ((snap.get("admission") or {}).get("stats") or {}).get("last"))]
        for kind, d in found:
            if not isinstance(d, dict) or not d:
                continue
            key = json.dumps([d.get(k) for k in ("id", "at", "since", "ended_at", "run_key", "caller", "actor", "outcome")],
                             default=str)
            if self._decided.get(kind) == key:
                continue
            first = kind not in self._decided
            self._decided[kind] = key
            row = self._decision_row(kind, d)
            row["seen_at_start"] = first           # it was already the last one when the watch began
            self.decisions.append(row)
            new = True
        return new

    def _decision_row(self, kind: str, d: dict) -> dict:
        at = d.get("at") or d.get("ended_at") or d.get("since")
        src, label, _cls = self.source_of(d.get("caller"))
        row = {"kind": kind, "at": _iso(float(at)) if isinstance(at, (int, float)) else None, "caller": d.get("caller"),
               "source": label, "actor": d.get("actor"), "class": d.get("class")}
        if kind == "capacity":                     # the capacity router: what it weighed and what it picked
            row.update(mode=d.get("mode"), decision=d.get("kind"), candidates=d.get("candidates"), rank=d.get("rank"),
                       pick=d.get("pick"), selected=d.get("selected"), actual=d.get("actual"), agree=d.get("agree"),
                       reason=d.get("reason"), wait_s=d.get("wait_s"), est_tokens=d.get("est_tokens"))
        elif kind == "route":                      # the switchyard's own routing: the order it tried, what it chose
            row.update(mode=d.get("mode"), tier=d.get("tier"), order=d.get("switchyard_order"),
                       route=d.get("selected_route"), routed=d.get("routed"), applied=d.get("applied"),
                       differs=d.get("differs"), upstream_failure=d.get("upstream_failure"), ms=d.get("ms"))
        else:                                      # admission: waited for a lane, and how it ended
            row.update(route=d.get("route"), lane=d.get("group"), lane_label=d.get("label"), outcome=d.get("outcome"),
                       reason=d.get("reason"), seconds=d.get("seconds"), behind=d.get("behind"))
        return row

    def decisions_feed(self, limit: int = 100) -> dict:
        cap = self.snap.get("capacity") or {}
        sw = self.snap.get("switchyard") or {}
        adm = ((self.snap.get("admission") or {}).get("stats")) or {}
        with self.cond:
            rows = sorted(self.decisions, key=lambda r: r.get("at") or "", reverse=True)[:limit]
        return {"decisions": rows, "kept": DECISIONS_KEPT,
                "counters": {"capacity": cap.get("counters"), "capacity_mode": cap.get("mode"),
                             "route": sw.get("decisions"), "route_mode": sw.get("mode"), "route_errors": sw.get("errors"),
                             "admission": {k: adm.get(k) for k in ("waits", "timeouts", "longest_seconds") if k in adm},
                             "policy": (self.snap.get("route_policy") or {}).get("counters")}}

    # --- the lanes: each model's slots, who holds them, who waits, its queue rules and its routes
    def lanes(self) -> list[dict]:
        snap = self.snap
        dests = self.destinations()
        groups = ((snap.get("admission") or {}).get("groups")) or {}
        pol = snap.get("route_policy") or {}
        queues, prules = pol.get("queues") or {}, pol.get("routes") or {}
        routes = snap.get("routes") or {}
        waits = [w for w in snap.get("waiting") or [] if isinstance(w, dict)]
        now = time.time()
        out = []
        for name in list(dests) + [g for g in groups if g not in dests]:
            d, g, q = dests.get(name) or {}, groups.get(name) or {}, queues.get(name) or {}
            ids = list(dict.fromkeys(list(d.get("routes") or []) + list(q.get("routes") or [])))
            rows = []
            for rid in ids:
                r, pr = routes.get(rid) or {}, prules.get(rid) or {}
                rows.append({"id": rid, "ready": r.get("ready"), "state": r.get("state"), "reason": r.get("reason"),
                             "since": r.get("since"), "base_url": r.get("base_url"), "model_file": r.get("model_file"),
                             "expected_model_file": r.get("expected_model_file"), "attest": r.get("attest"),
                             "enabled": pr.get("enabled"), "priority": pr.get("priority"), "policy_state": pr.get("state"),
                             "model": pr.get("model"), "configured_context": pr.get("configured_context"),
                             "eligible": pr.get("eligible")})
            holders = [{"actor": h.get("actor"), "route": h.get("route"),
                        "seconds": round(now - float(h["since"]), 1) if isinstance(h.get("since"), (int, float)) else None}
                       for h in g.get("holders") or [] if isinstance(h, dict)]
            calls = [{"actor": c.get("actor"), "caller": c.get("caller"), "source": self.source_of(c.get("caller"))[1],
                      "route": c.get("route"), "seconds": round(now - float(c["since"]), 1)
                      if isinstance(c.get("since"), (int, float)) else None}
                     for c in snap.get("in_flight_calls") or [] if isinstance(c, dict) and c.get("route") in ids]
            waiting = [{"actor": w.get("actor"), "caller": w.get("caller"), "source": self.source_of(w.get("caller"))[1],
                        "class": w.get("class"), "reason": w.get("reason"), "behind": w.get("behind"),
                        "seconds": round(now - float(w["since"]), 1) if isinstance(w.get("since"), (int, float)) else None}
                       for w in waits if w.get("group") == name or w.get("route") in ids]
            slots = d.get("slots") if d.get("slots") is not None else g.get("size")
            busy = d.get("busy") if d.get("busy") is not None else g.get("busy")
            out.append({"name": name, "label": d.get("label") or g.get("label") or name, "slots": slots, "busy": busy,
                        "free": d.get("free") if d.get("free") is not None else g.get("free"),
                        "queued": d.get("queued") if d.get("queued") is not None else g.get("queued"),
                        "queued_by_class": d.get("queued_by_class") or {}, "on_hold": g.get("on_hold"),
                        "slot_context": d.get("slot_context"), "avg_call_seconds": d.get("avg_call_seconds"),
                        "measured": d.get("measured"), "unavailable": d.get("unavailable"),
                        "rules": {k: q.get(k) for k in ("slots", "timeout_seconds", "max_queued") if k in q},
                        "holders": holders, "calls": calls, "waiting": waiting, "routes": rows,
                        "up": (any(r["ready"] for r in rows if r["ready"] is not None) if rows else True)
                        and not d.get("unavailable")})
        return out

    # --- the page's feeds, in the shapes the cockpit's own feeds have
    def tiers(self) -> list[dict]:
        out = [{"name": n, "address": self._address(d), "thinking": None, "provider": None, "label": d.get("label") or n}
               for n, d in self.destinations().items()]
        for x in self._externals():
            out.append({"name": x["id"], "address": None, "thinking": None, "provider": x.get("kind") or "external",
                        "label": x["id"]})
        return out

    def _externals(self) -> list[dict]:
        ext = (self.snap.get("external_destinations") or {}).get("destinations") or []
        return [x for x in ext if isinstance(x, dict) and x.get("id") and x.get("enabled")]

    def _address(self, d: dict) -> str | None:
        routes = self.snap.get("routes") or {}
        for r in d.get("routes") or []:
            if isinstance(routes.get(r), dict) and routes[r].get("base_url"):
                return routes[r]["base_url"]
        return None

    def agents(self) -> list[dict]:
        actors = ((self.snap.get("switchyard") or {}).get("actors")) or {}
        names = list(actors)
        return [{"name": a, "tier": None, "ceiling": None, "tools": [], "models": [], "default": False, "event": False,
                 "mode": actors[a]} for a in names]

    def sources(self) -> list[list]:
        rows, seen = [], set()
        for c in self.callers:
            rows.append([str(c["id"]), str(c.get("name") or c["id"]), str(c.get("class") or "")])
            seen.add(str(c["id"]))
        for key in self.counts:
            if key not in seen:
                rows.append([key, key, ""])
        return rows

    def engines(self) -> list[dict]:
        routes = self.snap.get("routes") or {}
        out = []
        for n, d in self.destinations().items():
            rs = [routes.get(r) for r in d.get("routes") or [] if isinstance(routes.get(r), dict)]
            up = (any(r.get("ready") for r in rs) if rs else True) and not d.get("unavailable")
            out.append({"tier": n, "address": self._address(d), "up": up, "busy": d.get("busy"), "slots": d.get("slots"),
                        "n_ctx": d.get("slot_context"), "model": d.get("label") or n, "queued": d.get("queued"),
                        "unavailable": d.get("unavailable"), "avg_call_seconds": d.get("avg_call_seconds")})
        for x in self._externals():
            out.append({"tier": x["id"], "address": None, "up": x.get("status") not in ("credential_missing", "missing"),
                        "busy": x.get("running") or 0, "slots": None, "n_ctx": None, "model": x.get("kind") or "",
                        "queued": x.get("queued")})
        return out

    def waiting(self) -> list[dict]:
        out = []
        for w in self.snap.get("waiting") or []:
            if not isinstance(w, dict):
                continue
            group = w.get("group")
            out.append({"key": w.get("id"), "who": w.get("caller") or w.get("actor"), "agent": w.get("actor"),
                        "class": w.get("class"), "tiers": [group] if group else [], "reason": w.get("reason"),
                        "since": w.get("since"), "label": w.get("label")})
        return out

    def parked(self) -> list[dict]:
        cap = self.snap.get("capacity") or {}
        return [p for p in cap.get("parking") or [] if isinstance(p, dict)]

    def live(self) -> dict:
        with self.cond:
            return {"running": list(self.running.values()), "working": [], "waiting": self.waiting(),
                    "parked": self.parked()}

    def tick(self) -> dict:
        """The stream's tick: what runs now, and what started and landed since the last poll."""
        with self.cond:
            out = {"running": list(self.running.values()), "working": [], "waiting": self.waiting(),
                   "parked": self.parked(), "started": list(self.started), "landed": list(self.landed),
                   "handoffs": [], "briefings": [], "escalations": [], "tool_rounds": []}
        return out

    def inputs(self) -> list[dict]:
        with self.cond:
            return [{"input": k, "attempts": v["attempts"], "sessions": 0, "last_at": v["last_at"]}
                    for k, v in self.counts.items()]

    def recent(self) -> dict:
        with self.cond:
            rows = list(self.running.values()) + self.finished
        attempts = [{"created_at": a["created_at"], "dialect": a["dialect"], "title": a["title"], "agent": a["agent"],
                     "tier": a["tier"], "model_id": a.get("route"), "state": a.get("state", "running"),
                     "finish_reason": f"{a['seconds']} s" if a.get("seconds") is not None else ""} for a in rows]
        return {"attempts": attempts, "decisions": [], "handoffs": []}

    def summary(self) -> dict:
        with self.cond:
            n = sum(v["attempts"] for v in self.counts.values())
            return {"name": self.name, "url": self.base, "reachable": self.error is None, "error": self.error,
                    "read_at": _iso(self.at) if self.at else None, "poll_seconds": self.poll,
                    "calls_seen": n, "runs_seen": len(self.runs), "in_flight": len(self.running)}
