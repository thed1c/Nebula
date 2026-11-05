"""nebula/activity.py: what is moving now, for the page's live view.

The ledger is the record and this is its pulse. The ledger calls `bump()` after every write it commits; a turn
calls `token()` for every delta it streams and `begin()`/`end()` around work that has no attempt row (the
secretary's compaction). `wait(seen, timeout)` blocks until the version moved past `seen`, so a page's stream wakes
on a write instead of polling the database.

Nothing here is kept: the token counts are of the attempts still running and are dropped when the ledger closes the
attempt (its costs row has the real counts then); the working set is what runs at this moment.

    Watch(ledger).tick()   one page's view since its last look: running, started, landed, handoffs, briefings,
                           escalations (decisions that moved a turn up the ladder from its floor, for the page's notices),
                           and team: Starweave's work items in flight per task and the beams (an item given from one
                           model to another: "started", its result back: "landed") since the last look
"""
from __future__ import annotations

import json
import threading
import time


class Activity:
    def __init__(self):
        self.cond = threading.Condition()
        self.version = 0
        self.tokens: dict[str, int] = {}            # attempt_id -> deltas streamed so far
        self.working: dict[str, dict] = {}          # key -> {"agent", "tier", "session_id", "since", "what"}
        self.closing = False

    def bump(self) -> None:
        with self.cond:
            self.version += 1
            self.cond.notify_all()

    def token(self, attempt_id: str | None, n: int = 1) -> None:
        if not attempt_id:
            return
        with self.cond:
            self.tokens[attempt_id] = self.tokens.get(attempt_id, 0) + n
            self.version += 1
            self.cond.notify_all()

    def finished(self, attempt_id: str) -> None:
        with self.cond:
            self.tokens.pop(attempt_id, None)
            self.version += 1
            self.cond.notify_all()

    def begin(self, key: str, **what) -> None:
        with self.cond:
            self.working[key] = {**what, "since": time.time()}
            self.version += 1
            self.cond.notify_all()

    def end(self, key: str) -> None:
        with self.cond:
            self.working.pop(key, None)
            self.version += 1
            self.cond.notify_all()

    def close(self) -> None:
        """Wakes every waiting stream so it can end (the server is closing)."""
        with self.cond:
            self.closing = True
            self.version += 1
            self.cond.notify_all()

    def wait(self, seen: int, timeout: float) -> int:
        """The version, once it differs from `seen` or after `timeout` seconds."""
        with self.cond:
            self.cond.wait_for(lambda: self.version != seen or self.closing, timeout)
            return self.version

    def snapshot(self) -> tuple[dict, list]:
        with self.cond:
            return dict(self.tokens), [{"key": k, **v} for k, v in self.working.items()]


ROW = ("attempt_id", "session_id", "tier", "agent", "dialect", "state", "model_id", "finish_reason", "error",
       "created_at", "finished_at")


class Watch:
    """One page's view of the ledger's movement. The first tick reports what is running (not as started); later ticks
    add what started, what landed (finished, failed, escalated or interrupted) and the handoffs and briefings made
    since the previous tick. Cursors are rowids, which only grow, so nothing committed between two ticks is missed."""

    def __init__(self, ledger, order=None):
        self.ledger = ledger
        self.order = list(order or [])          # the tiers' names, lowest first: an escalation is a move up it
        db = ledger.db
        self.a_cursor = db.execute("SELECT coalesce(max(rowid), 0) FROM attempts").fetchone()[0]
        self.h_cursor = db.execute("SELECT coalesce(max(rowid), 0) FROM handoffs").fetchone()[0]
        self.e_cursor = db.execute("SELECT coalesce(max(rowid), 0) FROM events").fetchone()[0]
        self.d_cursor = db.execute("SELECT coalesce(max(rowid), 0) FROM decisions").fetchone()[0]
        self.running: set[str] = set()
        self.first = True
        self.w_seen: dict[str, str] = {}         # Starweave work item -> the state the last tick saw

    def _titles(self, session_ids) -> dict:
        from nebula.ledger import text_of       # here: the ledger imports this module
        out = {}
        for sid in set(session_ids):
            for _q, _h, p in self.ledger._transcript(self.ledger.db, sid):
                if p.get("role") == "user":
                    out[sid] = text_of(p.get("content"))[:80]
                    break
        return out

    @staticmethod
    def _callers(db, ids) -> dict:
        """Each attempt's caller (nebula/callers.py), as its routing decision recorded it: an entry point's calls are
        its caller's (entry-<id>), which is how the map knows which Quasar a call came in through."""
        out, ids = {}, list(set(ids))
        for i in range(0, len(ids), 400):
            part = ids[i:i + 400]
            for r in db.execute(f"SELECT attempt_id, inputs FROM decisions WHERE attempt_id IN ({','.join('?' * len(part))}) "
                                "ORDER BY rowid", part):
                if r["attempt_id"] in out:
                    continue
                try:
                    i = json.loads(r["inputs"])
                    c = i.get("caller") or (f"gravity:{i['peer']}" if i.get("peer") else None)
                except (ValueError, AttributeError, TypeError):
                    c = None
                if c:
                    out[r["attempt_id"]] = c
        return out

    def _up(self, a: str, b: str) -> bool:
        if a in self.order and b in self.order:
            return self.order.index(b) > self.order.index(a)
        return a != b

    def tick(self) -> dict:
        with self.ledger.lock:                   # a snapshot between the ledger's transactions, never inside one
            return self._tick()

    def _tick(self) -> dict:
        db, activity = self.ledger.db, self.ledger.activity
        tokens, working = activity.snapshot() if activity is not None else ({}, [])
        cols = ", ".join(ROW)
        running = [dict(r) for r in db.execute(f"SELECT {cols} FROM attempts WHERE state='running' ORDER BY created_at")]
        new = [dict(r) for r in db.execute(f"SELECT rowid AS rid, {cols} FROM attempts WHERE rowid > ? ORDER BY rowid",
                                           (self.a_cursor,))]
        handoffs = []
        for r in db.execute("SELECT h.rowid AS rid, h.handoff_id, h.session_id, h.from_attempt, h.to_attempt, h.to_tier, "
                            "h.packet, h.created_at, a.tier AS from_tier, a.agent, a.dialect FROM handoffs h "
                            "LEFT JOIN attempts a ON a.attempt_id = h.from_attempt WHERE h.rowid > ? ORDER BY h.rowid",
                            (self.h_cursor,)):
            h = dict(r)
            try:
                h["reason"] = json.loads(h.pop("packet"))["escalation_reason"]
            except (ValueError, KeyError, TypeError):
                h["reason"] = None
            handoffs.append(h)
        briefings, tool_rounds = [], []
        for r in db.execute("SELECT e.rowid AS rid, e.session_id, e.attempt_id, e.kind, e.payload, e.recorded_at, a.agent, a.tier "
                            "FROM events e LEFT JOIN attempts a USING (attempt_id) WHERE e.rowid > ? AND "
                            "e.kind IN ('briefing', 'tool_round') ORDER BY e.rowid", (self.e_cursor,)):
            try:
                p = json.loads(r["payload"])
            except ValueError:
                p = {}
            if r["kind"] == "briefing":
                briefings.append({"session_id": r["session_id"], "tier": p.get("tier"), "agent": p.get("agent"),
                                  "at": r["recorded_at"]})
                continue
            names = [(c.get("function") or c).get("name") or "" for c in (p.get("assistant") or {}).get("tool_calls") or []]
            tool_rounds.append({"session_id": r["session_id"], "attempt_id": r["attempt_id"], "agent": r["agent"], "tier": r["tier"], "tools": names,
                                "servers": sorted({n.split("__", 1)[0] for n in names if "__" in n}), "at": r["recorded_at"]})
        e_max = db.execute("SELECT coalesce(max(rowid), 0) FROM events").fetchone()[0]
        escalations = []                         # the router's choices above the session's floor: rises and crossings
        for r in db.execute("SELECT d.rowid AS rid, d.decision_id, d.attempt_id, d.session_id, d.tier, d.reason, d.inputs, "
                            "d.created_at, a.agent, a.dialect FROM decisions d LEFT JOIN attempts a USING (attempt_id) "
                            "WHERE d.rowid > ? ORDER BY d.rowid", (self.d_cursor,)):
            self.d_cursor = r["rid"]
            try:
                floor = json.loads(r["inputs"]).get("floor")
            except (ValueError, AttributeError):
                floor = None
            if floor and self._up(floor, r["tier"]) and not self.first:
                escalations.append({"decision_id": r["decision_id"], "attempt_id": r["attempt_id"],
                                    "session_id": r["session_id"], "from_tier": floor, "to_tier": r["tier"],
                                    "reason": r["reason"], "agent": r["agent"], "dialect": r["dialect"],
                                    "at": r["created_at"]})

        now_ids = {a["attempt_id"] for a in running}
        new_ids = [a["attempt_id"] for a in new]
        gone = (self.running | set(new_ids)) - now_ids
        landed = []
        if gone:
            q = ",".join("?" * len(gone))
            landed = [dict(r) for r in db.execute(
                f"SELECT a.{', a.'.join(ROW)}, c.predicted_n, c.predicted_ms FROM attempts a LEFT JOIN costs c "
                f"USING (attempt_id) WHERE a.attempt_id IN ({q}) ORDER BY a.finished_at", tuple(gone))]
        via = {h["to_attempt"]: h["handoff_id"] for h in handoffs if h["to_attempt"]}
        started = [] if self.first else [{k: a[k] for k in ROW} | {"handoff": via.get(a["attempt_id"])} for a in new]
        if new:
            self.a_cursor = new[-1]["rid"]
        if handoffs:
            self.h_cursor = handoffs[-1]["rid"]
        self.e_cursor = e_max
        self.running = now_ids
        team = self._team(db)                    # before `first` turns: the first look reports no beams
        self.first = False
        for h in handoffs:
            h.pop("rid", None)
        titles = self._titles([a["session_id"] for a in running + landed + started + escalations] +
                              [h["session_id"] for h in handoffs])
        for a in running:
            a["tokens"] = tokens.get(a["attempt_id"], 0)
        who = self._callers(db, [a["attempt_id"] for a in running + landed + started])
        for a in running + landed + started:
            a["caller"] = who.get(a["attempt_id"])
        for a in running + landed + started:
            a["title"] = titles.get(a["session_id"], "")
        for h in handoffs + escalations:
            h["title"] = titles.get(h["session_id"], "")
        return {"version": activity.version if activity is not None else 0, "running": running, "started": started,
                "landed": landed, "handoffs": handoffs, "briefings": briefings, "escalations": escalations,
                "tool_rounds": [] if self.first else tool_rounds,
                "working": working, "team": team}

    def _team(self, db) -> dict:
        """Starweave's items in flight (by task) and the beams since the last look."""
        try:
            rows = [dict(r) for r in db.execute(
                "SELECT item_id, task_id, session_id, child_session_id, label, from_model, model, mode, state, updated_at FROM work_items "
                "WHERE state NOT IN ('done', 'failed', 'cancelled') OR item_id IN (SELECT item_id FROM work_items "
                "ORDER BY updated_at DESC LIMIT 200)")]
        except Exception:  # noqa: BLE001 - a ledger without the table (an older file, before its upgrade)
            return {"items": [], "beams": []}
        final = ("done", "failed", "cancelled")
        items, beams = [], []
        for r in rows:
            before = self.w_seen.get(r["item_id"])
            if r["state"] not in final:
                items.append(r)
            if before == r["state"] or self.first:
                continue
            if before is None:                   # given since the last look (and perhaps back already)
                beams.append({**r, "beam": "started"})
            if r["state"] in final:
                beams.append({**r, "beam": "landed"})
        self.w_seen = {r["item_id"]: r["state"] for r in rows}
        return {"items": items, "beams": beams}
