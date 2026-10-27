"""nebula/ledger.py: the ledger (docs/DESIGN.md, "The ledger"). One SQLite file, one writer.

A row exists before the thing it describes is sent: the session, its task, the attempt, the request's new events
and the request boundary are committed before the engine is called; the decision and, on a crossing, the escalated
source attempt, the handoff and the destination attempt follow, still before the call; the reply, the costs and the
outcome after it answered. Every row carries tenant_id. It is the one local owner in v1 and nothing filters on it.

Sessions are stateless clients' conversations found again. A request's session key is the X-Nebula-Session header (or the older X-Cockpit-Session)
or the request's own session field (Anthropic's metadata.user_id, which Claude Code fills with its session id; a
metadata.session_id; OpenAI's user), and its opening is the conversation's system message and first other message;
a session is one (tenant, key, opening). Within it the request's messages are matched against the recorded
transcript by content, not bytes: clients drop reasoning and regenerate tool-call ids when they send history back,
so the canonical form (nebula/events.py's canonical_message) leaves those out. Messages beyond the matched prefix
are new events; a request that drops part of the transcript (a regeneration, an edited turn) records a rewind
first. Two conversations with the same opening and no session field fall into one session, as rewinds: the header
is the fix, and `key_source` says which was used.

Event kinds: message (a client's turn), reply (an engine's), request (the boundary: the transcript's hash the
attempt was sent), rewind, briefing (a structured compaction; the next packet starts from it), handoff (a packet
sent to another tier), lifecycle (closed or reopened: explicit, by the user, or by a client that kept writing),
tool_round (an agent's tools run by the cockpit, outside the transcript a client sees).
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from nebula import migrate
from nebula.activity import Activity
from nebula.events import Done, canonical_json, canonical_message, sha256_of, stored_message, transcript_sha256

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, session_key TEXT NOT NULL, key_source TEXT NOT NULL,
  opening_key TEXT NOT NULL, lifecycle TEXT NOT NULL, revision INTEGER NOT NULL, next_seq INTEGER NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE (tenant_id, session_key, opening_key));
CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, tenant_id TEXT NOT NULL, stated_acceptance TEXT,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS attempts (
  attempt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, session_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
  tier TEXT NOT NULL, engine TEXT, model_id TEXT, dialect TEXT NOT NULL, requested_model TEXT, thinking INTEGER,
  state TEXT NOT NULL, verified_outcome TEXT NOT NULL, request_sha256 TEXT NOT NULL, tools_sha256 TEXT,
  reply_sha256 TEXT, finish_reason TEXT, error TEXT, created_at TEXT NOT NULL, finished_at TEXT, agent TEXT);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, tenant_id TEXT NOT NULL, seq INTEGER NOT NULL,
  attempt_id TEXT, kind TEXT NOT NULL, role TEXT, payload TEXT NOT NULL, content_sha256 TEXT NOT NULL,
  canonical_sha256 TEXT, completeness TEXT NOT NULL, reasoning_visibility TEXT, recorded_at TEXT NOT NULL,
  UNIQUE (session_id, seq));
CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL, session_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
  inputs TEXT NOT NULL, tier TEXT NOT NULL, thinking INTEGER, reason TEXT NOT NULL, advisory INTEGER NOT NULL,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS handoffs (
  handoff_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, tenant_id TEXT NOT NULL, from_attempt TEXT NOT NULL,
  to_attempt TEXT, to_tier TEXT NOT NULL, packet TEXT NOT NULL, packet_sha256 TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE, sent_at TEXT, acknowledged_at TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outcomes (
  outcome_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL, session_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
  kind TEXT NOT NULL, value TEXT NOT NULL, detail TEXT, recorded_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS costs (
  attempt_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
  prompt_tokens INTEGER, completion_tokens INTEGER, cached_tokens INTEGER, prompt_n INTEGER, cache_n INTEGER,
  predicted_n INTEGER, draft_n INTEGER, draft_n_accepted INTEGER, prompt_ms REAL, predicted_ms REAL,
  recorded_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_by_attempt ON events (attempt_id);
CREATE INDEX IF NOT EXISTS attempts_by_session ON attempts (session_id, created_at);
"""



def _agent_column(db) -> None:
    if "agent" not in {r["name"] for r in db.execute("PRAGMA table_info(attempts)")}:
        db.execute("ALTER TABLE attempts ADD COLUMN agent TEXT")       # a ledger from before agents


STEPS = [SCHEMA, _agent_column]         # nebula/migrate.py: one more step for each change, never an edited one

STUCK_DEFAULT = ("i can't", "i cannot", "i'm unable", "i am unable", "i don't know how", "beyond my", "not able to")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def opening_key(canonical: list[dict]) -> str:
    """The conversation's opening: its leading system message and its first other message, hashed."""
    system = canonical[0] if canonical and canonical[0].get("role") == "system" else None
    first = next((c for c in canonical if c.get("role") != "system"), None)
    return sha256_of([system, first])


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


@dataclass
class Opened:
    session_id: str
    task_id: str
    attempt_id: str
    request_sha256: str
    new_events: int
    rewound: int
    key_source: str
    first_seq: int = 0                       # the seq of this request's first event: what came before is prior


# A tool result that reports an outage of the tool's own service (the Switchyard's upstream failures): no strike against
# the model that called it.
OUTAGE = re.compile(r"(?i)(\b50[0234]\b|service unavailable|bad gateway|gateway time-?out|connection (refused|reset|aborted)|"
                    r"server ?disconnected|broken pipe|timed out|timeout|unreachable|no route to host|temporarily unavailable)")


@dataclass
class SessionState:
    """What the router reads about a session once the request's new events are in."""
    session_id: str
    current_tier: str | None = None          # the tier that answered the last completed attempt
    regenerations: int = 0                   # consecutive requests that rewound to the same point, this one included
    tool_error_streak: int = 0               # tool results at the end of the transcript that report an error
    last_reply_text: str = ""                # the last reply's content, for stuck phrases
    last_prompt_tokens: int = 0              # the last completed attempt's prompt, as the engine counted it
    pending_tool: bool = False               # the transcript ends in a call without its result
    briefing_seq: int | None = None          # the latest briefing event's seq
    transcript_seq: int = 0                  # the seq of the last message or reply in the transcript
    transcript_len: int = 0
    transcript_tiers: list[str] = field(default_factory=list)
    last_answer_at: str | None = None        # when the session's last completed attempt finished (stickiness)
    outage_errors: int = 0                   # tool errors in the streak that were outages, not the model's doing


class _Rows:
    """A statement's result, read whole while the connection's lock was held (so no cursor is stepped later, from
    another thread, while a transaction is open on the same connection)."""

    def __init__(self, cur):
        self.rows = cur.fetchall() if cur.description is not None else []
        self.lastrowid, self.rowcount, self.description = cur.lastrowid, cur.rowcount, cur.description

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def __iter__(self):
        return iter(self.rows)


class _Locked:
    """The ledger's one SQLite connection, shared by every request thread: each statement runs under one re-entrant
    lock (the same one a transaction holds from BEGIN to COMMIT), and its rows are read before the lock is let go.
    A read on the connection while another thread's transaction was mid-way is SQLite's "bad parameter or other API
    misuse" (seen as a dropped request under concurrent calls on Python 3.12)."""

    def __init__(self, conn, lock):
        self.conn, self.lock = conn, lock

    def execute(self, sql, params=()):
        with self.lock:
            return _Rows(self.conn.execute(sql, params))

    def executemany(self, sql, seq):
        with self.lock:
            return _Rows(self.conn.executemany(sql, seq))

    def executescript(self, sql):
        with self.lock:
            return self.conn.executescript(sql)

    def close(self):
        with self.lock:
            self.conn.close()

    def __getattr__(self, name):
        return getattr(self.conn, name)


class Ledger:
    def __init__(self, path: str, tenant: str = "local"):
        self.path, self.tenant = path, tenant
        self.lock = threading.RLock()
        self.activity = Activity()               # the pulse of every commit, for the page's live view
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        self.db = _Locked(conn, self.lock)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.upgraded = migrate.sqlite_upgrade(self.db, "the ledger", STEPS, path)

    def close(self) -> None:
        self.activity.close()
        self.db.close()

    @contextmanager
    def _tx(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")
        self.activity.bump()

    # --- before the engine is called
    def open_request(self, *, dialect: str, session_key: str | None, key_source: str, messages: list[dict],
                     tools, requested_model: str, thinking: bool | None, tier: str, engine: str | None,
                     agent: str | None = None) -> Opened:
        stored = [stored_message(m) for m in messages]
        canonical = [canonical_message(m) for m in messages]
        hashes = [sha256_of(c) for c in canonical]
        request_sha = transcript_sha256(canonical)
        opening = opening_key(canonical)
        key = session_key or ""
        source = key_source if session_key else "opening"
        t = now()
        with self._tx() as db:
            row = db.execute("SELECT session_id, lifecycle FROM sessions WHERE tenant_id=? AND session_key=? AND "
                             "opening_key=?", (self.tenant, key, opening)).fetchone()
            if row is not None and row["lifecycle"] == "closed":
                db.execute("UPDATE sessions SET lifecycle='active' WHERE session_id=?", (row["session_id"],))
                self._event(db, row["session_id"], None, "lifecycle", None, {"to": "active", "why": "request"}, None, t)
            if row is None:
                session_id, task_id = new_id("ses"), new_id("task")
                db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (session_id, self.tenant, key, source, opening, "active", 0, 1, t, t))
                db.execute("INSERT INTO tasks VALUES (?,?,?,?,?)", (task_id, session_id, self.tenant, None, t))
            else:
                session_id = row["session_id"]
                task_id = db.execute("SELECT task_id FROM tasks WHERE session_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                                     (session_id,)).fetchone()["task_id"]
            transcript = [h for _seq, h, _p in self._transcript(db, session_id)]
            prefix = 0
            while prefix < min(len(transcript), len(hashes)) and transcript[prefix] == hashes[prefix]:
                prefix += 1
            attempt_id = new_id("att")
            tools_sha = sha256_of(tools) if tools else None
            last = db.execute("SELECT tier, engine FROM attempts WHERE session_id=? AND state='completed' AND agent IS ? "
                              "ORDER BY created_at DESC, rowid DESC LIMIT 1", (session_id, agent)).fetchone()
            if last is not None:                 # the floor: where this agent's last turn in the session was answered
                tier, engine = last["tier"], last["engine"]
            self._attempt(db, attempt_id, task_id, session_id, tier, engine, dialect, requested_model, thinking,
                          request_sha, tools_sha, t, agent)
            first_seq = db.execute("SELECT next_seq FROM sessions WHERE session_id=?", (session_id,)).fetchone()["next_seq"]
            rewound = len(transcript) - prefix
            if rewound:
                self._event(db, session_id, attempt_id, "rewind", None, {"keep": prefix, "dropped": rewound}, None, t)
            for stored_m, canon_sha in zip(stored[prefix:], hashes[prefix:]):
                self._event(db, session_id, attempt_id, "message", stored_m.get("role"), stored_m, canon_sha, t)
            self._event(db, session_id, attempt_id, "request", None,
                        {"dialect": dialect, "requested_model": requested_model, "request_sha256": request_sha,
                         "thinking": thinking, "tier": tier}, request_sha, t)
            db.execute("UPDATE sessions SET revision=revision+1, updated_at=? WHERE session_id=?", (t, session_id))
        return Opened(session_id, task_id, attempt_id, request_sha, len(hashes) - prefix, rewound, source, first_seq)

    def decide(self, attempt_id: str, *, tier: str, thinking: bool | None, reason: str, inputs: dict,
               advisory: bool = False) -> None:
        """The router's choice for an attempt, before anything is sent."""
        with self._tx() as db:
            session_id = self._session_of(db, attempt_id)
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (new_id("dec"), attempt_id, session_id, self.tenant, canonical_json(inputs), tier,
                        None if thinking is None else int(thinking), reason, int(advisory), now()))

    def escalate(self, attempt_id: str, *, to_tier: str, to_engine: str | None, packet: dict, packet_sha256: str,
                 idempotency_key: str, thinking: bool | None, reason: str) -> tuple[str, str]:
        """The source attempt ends escalated, the handoff is recorded, and the destination attempt opens on the same
        task and the same transcript: (handoff_id, destination attempt_id), all before the packet is sent. A handoff
        with the same idempotency key (the same packet for the same request) is reused, not repeated."""
        t = now()
        with self._tx() as db:
            att = db.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if att is None:
                raise KeyError(attempt_id)
            session_id = att["session_id"]
            db.execute("UPDATE attempts SET state='escalated', error=?, finished_at=? WHERE attempt_id=?",
                       (reason, t, attempt_id))
            db.execute("INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?)",
                       (new_id("out"), attempt_id, session_id, self.tenant, "transport", "escalated",
                        canonical_json({"to_tier": to_tier, "reason": reason}), t))
            existing = db.execute("SELECT handoff_id, to_attempt FROM handoffs WHERE idempotency_key=?",
                                  (idempotency_key,)).fetchone()
            if existing is not None and existing["to_attempt"]:
                return existing["handoff_id"], existing["to_attempt"]
            handoff_id = existing["handoff_id"] if existing is not None else new_id("hand")
            dest = new_id("att")
            self._attempt(db, dest, att["task_id"], session_id, to_tier, to_engine, att["dialect"],
                          att["requested_model"], thinking, att["request_sha256"], att["tools_sha256"], t, att["agent"])
            if existing is None:
                db.execute("INSERT INTO handoffs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                           (handoff_id, session_id, self.tenant, attempt_id, dest, to_tier, canonical_json(packet),
                            packet_sha256, idempotency_key, t, None, t))
            else:
                db.execute("UPDATE handoffs SET to_attempt=?, sent_at=? WHERE handoff_id=?", (dest, t, handoff_id))
            self._event(db, session_id, attempt_id, "handoff", None,
                        {"to_tier": to_tier, "packet_sha256": packet_sha256, "idempotency_key": idempotency_key,
                         "reason": reason}, None, t)
            self._event(db, session_id, dest, "request", None,
                        {"dialect": att["dialect"], "requested_model": att["requested_model"],
                         "request_sha256": att["request_sha256"], "thinking": thinking, "tier": to_tier,
                         "via_handoff": idempotency_key}, att["request_sha256"], t)
            db.execute("UPDATE sessions SET revision=revision+1, updated_at=? WHERE session_id=?", (t, session_id))
        return handoff_id, dest

    def retier(self, attempt_id: str, tier: str, engine: str | None, thinking: bool | None) -> None:
        """The router moved the attempt on the same engine (another thinking level): its tier, before the call."""
        with self._tx() as db:
            db.execute("UPDATE attempts SET tier=?, engine=?, thinking=? WHERE attempt_id=? AND state='running'",
                       (tier, engine, None if thinking is None else int(thinking), attempt_id))

    def set_lifecycle(self, session_id: str, to: str, why: str = "user") -> None:
        """Explicit: the user closed or reopened the session. Reopening keeps every id and advances the revision."""
        if to not in ("active", "closed"):
            raise ValueError(to)
        t = now()
        with self._tx() as db:
            if db.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone() is None:
                raise KeyError(session_id)
            db.execute("UPDATE sessions SET lifecycle=?, revision=revision+1, updated_at=? WHERE session_id=?",
                       (to, t, session_id))
            self._event(db, session_id, None, "lifecycle", None, {"to": to, "why": why}, None, t)

    SESSION_TABLES = ("costs", "outcomes", "handoffs", "decisions", "events", "attempts", "tasks", "sessions")

    STUCK_AFTER_S = 7200        # a call "running" longer than this was never closed (a crash, a restart): not in flight

    def delete_sessions(self, session_ids, stuck_after_s: float | None = None) -> dict:
        """The owner deletes conversations for good: every row they own in every table (their messages, attempts,
        decisions, handoffs, outcomes, costs), in one transaction. A conversation with a call still running is not
        deleted (it is in "busy"), unless that call started more than stuck_after_s ago (it was never closed: a crash or
        a restart left it "running"); an unknown id is in "missing"."""
        import datetime as _dt
        ids = list(dict.fromkeys(str(s) for s in session_ids if s))
        out = {"deleted": [], "busy": [], "missing": []}
        since = (_dt.datetime.now(_dt.timezone.utc)
                 - _dt.timedelta(seconds=self.STUCK_AFTER_S if stuck_after_s is None else stuck_after_s)).isoformat()
        with self._tx() as db:
            for sid in ids:
                if db.execute("SELECT 1 FROM sessions WHERE session_id=?", (sid,)).fetchone() is None:
                    out["missing"].append(sid)
                    continue
                if db.execute("SELECT 1 FROM attempts WHERE session_id=? AND state='running' AND created_at>=? LIMIT 1",
                              (sid, since)).fetchone():
                    out["busy"].append(sid)
                    continue
                for table in self.SESSION_TABLES:
                    db.execute(f"DELETE FROM {table} WHERE session_id=?", (sid,))
                out["deleted"].append(sid)
        return out

    def closed_sessions(self) -> list[str]:
        return [r["session_id"] for r in self.db.execute("SELECT session_id FROM sessions WHERE lifecycle='closed'")]

    def session(self, session_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def session_by_key(self, key: str) -> dict | None:
        row = self.db.execute("SELECT * FROM sessions WHERE tenant_id=? AND session_key=? ORDER BY created_at DESC, rowid DESC "
                              "LIMIT 1", (self.tenant, key)).fetchone()
        return dict(row) if row else None

    def summaries(self) -> list[dict]:
        """One line per session for a list: title (its first user message), turns, the tier of its last answer."""
        out = []
        for s in self.db.execute("SELECT * FROM sessions ORDER BY updated_at DESC, rowid DESC"):
            conv = self._transcript(self.db, s["session_id"])
            first = next((p for _q, _h, p in conv if p.get("role") == "user"), None)
            last = self.db.execute("SELECT tier, model_id FROM attempts WHERE session_id=? AND state='completed' "
                                   "ORDER BY created_at DESC, rowid DESC LIMIT 1", (s["session_id"],)).fetchone()
            out.append({"session_id": s["session_id"], "title": text_of(first.get("content"))[:80] if first else "(empty)",
                        "lifecycle": s["lifecycle"], "updated_at": s["updated_at"], "key_source": s["key_source"],
                        "turns": sum(1 for _q, _h, p in conv if p.get("role") == "assistant"),
                        "current_tier": last["tier"] if last else None, "model_id": last["model_id"] if last else None})
        return out

    def view(self, session_id: str) -> dict | None:
        """The session as a page shows it: the transcript (rewinds applied) with each reply's attempt, decision and
        cost, and the handoffs, briefings and lifecycle changes between them, in sequence order."""
        s = self.session(session_id)
        if s is None:
            return None
        conv = {seq for seq, _h, _p in self._transcript(self.db, session_id)}
        attempts = {r["attempt_id"]: dict(r) for r in self.db.execute("SELECT * FROM attempts WHERE session_id=?", (session_id,))}
        decisions = {r["attempt_id"]: dict(r) for r in self.db.execute("SELECT * FROM decisions WHERE session_id=? "
                                                                        "ORDER BY created_at, rowid", (session_id,))}
        costs = {r["attempt_id"]: dict(r) for r in self.db.execute("SELECT * FROM costs WHERE session_id=?", (session_id,))}
        # a crossing's decision sits on the escalated source attempt; the reply is the destination's
        via = {r["to_attempt"]: r["from_attempt"] for r in self.db.execute("SELECT to_attempt, from_attempt FROM handoffs "
                                                                           "WHERE session_id=?", (session_id,))}
        items = []
        for e in self.db.execute("SELECT * FROM events WHERE session_id=? ORDER BY seq", (session_id,)):
            kind, payload = e["kind"], json.loads(e["payload"])
            if kind in ("message", "reply"):
                if e["seq"] not in conv:
                    continue
                item = {"kind": kind, "seq": e["seq"], "at": e["recorded_at"], "role": payload.get("role"),
                        "content": text_of(payload.get("content")),
                        "tool_calls": [c.get("function", c) for c in payload.get("tool_calls") or []],
                        "completeness": e["completeness"]}
                if kind == "reply":
                    a = attempts.get(e["attempt_id"]) or {}
                    d = decisions.get(e["attempt_id"]) or decisions.get(via.get(e["attempt_id"])) or {}
                    c = costs.get(e["attempt_id"]) or {}
                    item["reasoning"] = payload.get("reasoning_content") or ""
                    item["attempt"] = {k: a.get(k) for k in ("attempt_id", "tier", "model_id", "thinking", "state", "finish_reason", "agent")}
                    item["decision"] = {"reason": d.get("reason"), "inputs": json.loads(d["inputs"]) if d else None}
                    item["cost"] = {k: c.get(k) for k in ("prompt_n", "cache_n", "predicted_n", "prompt_ms", "predicted_ms")}
                items.append(item)
            elif kind == "tool_round":
                calls = [c.get("function", c) for c in (payload.get("assistant") or {}).get("tool_calls") or []]
                items.append({"kind": "tool_round", "seq": e["seq"], "at": e["recorded_at"], "attempt_id": e["attempt_id"],
                              "calls": calls, "results": [text_of(r.get("content")) for r in payload.get("results") or []]})
            elif kind == "handoff":
                items.append({"kind": "handoff", "seq": e["seq"], "at": e["recorded_at"], "to_tier": payload.get("to_tier"), "reason": payload.get("reason"),
                              "text": self._handoff_text(payload.get("idempotency_key")),
                              "handoff_id": self._handoff_id(payload.get("idempotency_key"))})
            elif kind == "briefing":
                items.append({"kind": "briefing", "seq": e["seq"], "at": e["recorded_at"], "covers_through_seq": payload.get("covers_through_seq"),
                              "tier": payload.get("tier"), "objective": payload.get("objective")})
            elif kind == "lifecycle":
                items.append({"kind": "lifecycle", "seq": e["seq"], "at": e["recorded_at"], "to": payload.get("to"), "why": payload.get("why")})
        last = self.db.execute("SELECT c.prompt_tokens, c.completion_tokens, a.tier FROM attempts a JOIN costs c USING "
                               "(attempt_id) WHERE a.session_id=? AND a.state='completed' ORDER BY a.created_at DESC, a.rowid DESC LIMIT 1",
                               (session_id,)).fetchone()
        context = {"last_prompt_tokens": (last["prompt_tokens"] or 0) + (last["completion_tokens"] or 0) if last else 0,
                   "tier": last["tier"] if last else None}
        return {"session": s, "items": items, "context": context}

    def inputs(self) -> list[dict]:
        """Per input (the attempt's dialect: chat, messages, responses, hook:<name>): attempts, sessions, the last time."""
        return [dict(r) for r in self.db.execute(
            "SELECT dialect AS input, count(*) AS attempts, count(DISTINCT session_id) AS sessions, max(created_at) AS last_at "
            "FROM attempts GROUP BY dialect ORDER BY last_at DESC")]

    def recent(self, limit: int = 40) -> dict:
        """The latest attempts, decisions and handoffs across sessions, newest first, with session titles."""
        titles = {s["session_id"]: s["title"] for s in self.summaries()}
        attempts = [dict(r) for r in self.db.execute(
            "SELECT a.*, c.prompt_n, c.cache_n, c.predicted_n, c.predicted_ms FROM attempts a LEFT JOIN costs c USING (attempt_id) "
            "ORDER BY a.created_at DESC, a.rowid DESC LIMIT ?", (limit,))]
        for a in attempts:
            a["title"] = titles.get(a["session_id"], "")
        decisions = [dict(r) for r in self.db.execute("SELECT * FROM decisions ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))]
        for d in decisions:
            d["inputs"] = json.loads(d["inputs"])
            d["title"] = titles.get(d["session_id"], "")
        handoffs = [dict(r) for r in self.db.execute("SELECT * FROM handoffs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))]
        for h in handoffs:
            packet = json.loads(h.pop("packet"))
            h["reason"] = packet["escalation_reason"]
            h["estimated_tokens"] = packet["budget"]["estimated_tokens"]
            h["title"] = titles.get(h["session_id"], "")
        return {"attempts": attempts, "decisions": decisions, "handoffs": handoffs}

    def _handoff_text(self, key: str | None) -> str:
        row = self.db.execute("SELECT packet FROM handoffs WHERE idempotency_key=?", (key,)).fetchone() if key else None
        return json.loads(row["packet"])["escalation_reason"]["text"] if row else ""

    def _handoff_id(self, key: str | None) -> str | None:
        row = self.db.execute("SELECT handoff_id FROM handoffs WHERE idempotency_key=?", (key,)).fetchone() if key else None
        return row["handoff_id"] if row else None

    def latest_handoff(self, session_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM handoffs WHERE session_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                              (session_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["packet"] = json.loads(out["packet"])
        return out

    def acknowledge(self, handoff_id: str) -> None:
        """The destination received the packet: its first event arrived."""
        with self._tx() as db:
            db.execute("UPDATE handoffs SET acknowledged_at=? WHERE handoff_id=? AND acknowledged_at IS NULL",
                       (now(), handoff_id))

    def tool_round(self, session_id: str, attempt_id: str, assistant: dict, results: list[dict]) -> None:
        """An agent's tool round, run by the cockpit: the call the engine made and what the tools answered. It is
        not part of the transcript a client sees, so it never enters the matching or a packet."""
        with self._tx() as db:
            self._event(db, session_id, attempt_id, "tool_round", "assistant", {"assistant": assistant, "results": results},
                        None, now())

    def briefing(self, session_id: str, attempt_id: str | None, payload: dict) -> None:
        """A structured compaction of the transcript through `payload['covers_through_seq']`."""
        with self._tx() as db:
            self._event(db, session_id, attempt_id, "briefing", None, payload, None, now())
            db.execute("UPDATE sessions SET revision=revision+1, updated_at=? WHERE session_id=?", (now(), session_id))

    # --- after it answered (or did not)
    def close_request(self, attempt_id: str, *, reply: dict | None, done: Done | None, completeness: str = "complete",
                      error: str | None = None, queued_s: float | None = None) -> None:
        t = now()
        with self._tx() as db:
            att = db.execute("SELECT session_id, state FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if att is None:
                raise KeyError(attempt_id)
            if att["state"] != "running":
                return
            session_id = att["session_id"]
            reply_sha = None
            if reply is not None:
                reply_sha = sha256_of(reply)
                self._event(db, session_id, attempt_id, "reply", "assistant", reply,
                            sha256_of(canonical_message(reply)), t, completeness=completeness,
                            reasoning="text" if reply.get("reasoning_content") else "unavailable")
            finish = done.finish if done is not None else None
            if done is not None:
                tm = done.timings
                db.execute("INSERT OR REPLACE INTO costs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (attempt_id, session_id, self.tenant, done.prompt_tokens, done.completion_tokens,
                            done.cached_tokens, tm.get("prompt_n"), tm.get("cache_n"), tm.get("predicted_n"),
                            tm.get("draft_n"), tm.get("draft_n_accepted"), tm.get("prompt_ms"), tm.get("predicted_ms"), t))
            state = "failed" if error else "interrupted" if completeness == "interrupted" else "completed"
            value = "error" if error else "interrupted" if completeness == "interrupted" else (finish or "stop")
            db.execute("INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?)",
                       (new_id("out"), attempt_id, session_id, self.tenant, "transport", value,
                        canonical_json({"error": error, "finish": finish, "queued_s": None if queued_s is None else round(queued_s, 3)}), t))
            db.execute("UPDATE attempts SET state=?, model_id=?, reply_sha256=?, finish_reason=?, error=?, finished_at=? "
                       "WHERE attempt_id=?",
                       (state, done.model if done is not None else None, reply_sha, finish, error, t, attempt_id))
            db.execute("UPDATE sessions SET revision=revision+1, updated_at=? WHERE session_id=?", (t, session_id))
        self.activity.finished(attempt_id)

    # --- what the router and the packet read
    def state(self, session_id: str, attempt_id: str, stuck_phrases=STUCK_DEFAULT, agent: str | None = None,
              ignore_outages: bool = False) -> SessionState:
        st = SessionState(session_id)
        events = [dict(r) for r in self.db.execute("SELECT * FROM events WHERE session_id=? ORDER BY seq", (session_id,))]
        attempts = {r["attempt_id"]: dict(r) for r in
                    self.db.execute("SELECT * FROM attempts WHERE session_id=? ORDER BY created_at, rowid", (session_id,))}
        for a in attempts.values():
            if a["state"] == "completed" and a["attempt_id"] != attempt_id and a.get("agent") == agent:
                st.current_tier = a["tier"]
                st.last_answer_at = a.get("finished_at") or a.get("created_at")
                cost = self.db.execute("SELECT prompt_tokens, completion_tokens FROM costs WHERE attempt_id=?",
                                       (a["attempt_id"],)).fetchone()
                st.last_prompt_tokens = (cost["prompt_tokens"] or 0) + (cost["completion_tokens"] or 0) if cost else 0
        conv = self._transcript(self.db, session_id)
        st.transcript_len = len(conv)
        st.transcript_seq = conv[-1][0] if conv else 0
        for e in events:
            if e["kind"] == "briefing":
                st.briefing_seq = e["seq"]
        # regenerations: this request rewound, and so did the requests before it, to the same point
        by_attempt: dict[str, list[dict]] = {}
        order: list[str] = []
        for e in events:
            if e["attempt_id"] and e["kind"] in ("rewind", "message", "request"):
                if e["attempt_id"] not in by_attempt:
                    order.append(e["attempt_id"])
                by_attempt.setdefault(e["attempt_id"], []).append(e)
        keep = None
        for aid in reversed(order):
            rewinds = [json.loads(e["payload"]) for e in by_attempt[aid] if e["kind"] == "rewind"]
            if not rewinds or (keep is not None and rewinds[0]["keep"] != keep):
                break
            keep = rewinds[0]["keep"]
            st.regenerations += 1
        # the transcript's tail: tool errors, a pending call, the last reply
        streak = 0
        for _seq, _h, payload in reversed(conv):
            if payload.get("role") != "tool":
                break
            t = text_of(payload.get("content")).strip()
            if t.lower().startswith("error"):
                if ignore_outages and OUTAGE.search(t[:4000]):   # the tool's service was down: not the model's failure
                    st.outage_errors += 1
                    continue
                streak += 1
            else:
                break
        st.tool_error_streak = streak
        if conv:
            last = conv[-1][2]
            st.pending_tool = last.get("role") == "assistant" and bool(last.get("tool_calls"))
        for _seq, _h, payload in reversed(conv):
            if payload.get("role") == "assistant":
                st.last_reply_text = text_of(payload.get("content"))
                break
        st.transcript_tiers = [a["tier"] for a in attempts.values() if a["state"] == "completed"]
        return st

    def transcript(self, session_id: str, since_seq: int = 0, until_seq: int | None = None) -> list[tuple[int, dict]]:
        """(seq, stored payload) of the transcript's messages after `since_seq` (and before `until_seq`), rewinds
        applied."""
        return [(seq, p) for seq, _h, p in self._transcript(self.db, session_id)
                if seq > since_seq and (until_seq is None or seq < until_seq)]

    def latest_briefing(self, session_id: str) -> tuple[int, dict] | None:
        row = self.db.execute("SELECT seq, payload FROM events WHERE session_id=? AND kind='briefing' ORDER BY seq DESC "
                              "LIMIT 1", (session_id,)).fetchone()
        return (row["seq"], json.loads(row["payload"])) if row else None

    def briefing_count(self, session_id: str) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM events WHERE session_id=? AND kind='briefing'", (session_id,)).fetchone()[0])

    def session_opening(self, session_id: str) -> str:
        return self.db.execute("SELECT opening_key FROM sessions WHERE session_id=?", (session_id,)).fetchone()["opening_key"]

    # --- internals
    @staticmethod
    def _session_of(db, attempt_id: str) -> str:
        row = db.execute("SELECT session_id FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        return row["session_id"]

    def _attempt(self, db, attempt_id, task_id, session_id, tier, engine, dialect, requested_model, thinking,
                 request_sha, tools_sha, t, agent=None) -> None:
        db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (attempt_id, task_id, session_id, self.tenant, tier, engine, None, dialect, requested_model,
                    None if thinking is None else int(thinking), "running", "unknown", request_sha, tools_sha,
                    None, None, None, t, None, agent))

    def _event(self, db, session_id, attempt_id, kind, role, payload, canonical_sha, t,
               completeness="complete", reasoning=None) -> None:
        seq = db.execute("SELECT next_seq FROM sessions WHERE session_id=?", (session_id,)).fetchone()["next_seq"]
        db.execute("UPDATE sessions SET next_seq=? WHERE session_id=?", (seq + 1, session_id))
        db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (new_id("evt"), session_id, self.tenant, seq, attempt_id, kind, role, canonical_json(payload),
                    sha256_of(payload), canonical_sha, completeness, reasoning, t))

    @staticmethod
    def _transcript(db, session_id) -> list[tuple[int, str, dict]]:
        """The conversation the events add up to: (seq, canonical hash, stored payload), rewinds applied."""
        conv: list[tuple[int, str, dict]] = []
        for row in db.execute("SELECT seq, kind, payload, canonical_sha256 FROM events WHERE session_id=? ORDER BY seq",
                              (session_id,)):
            if row["kind"] == "rewind":
                conv = conv[:json.loads(row["payload"])["keep"]]
            elif row["kind"] in ("message", "reply"):
                conv.append((row["seq"], row["canonical_sha256"], json.loads(row["payload"])))
        return conv

    # --- reads
    def sessions(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM sessions ORDER BY created_at, rowid")]

    def events(self, session_id: str) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM events WHERE session_id=? ORDER BY seq", (session_id,))]

    def attempts(self, session_id: str) -> dict[str, dict]:
        return {r["attempt_id"]: dict(r) for r in
                self.db.execute("SELECT * FROM attempts WHERE session_id=? ORDER BY created_at, rowid", (session_id,))}
