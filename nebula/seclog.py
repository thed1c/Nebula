"""nebula/seclog.py: the Switchyard's security log (docs/security-log.md).

An append-only record of who tried to get in and what changed who may: sign-ins and refusals, key use, keys made and
revoked, Gravity invites, joins and grants, guests refused a model, rate limits, and every Halo action and undo. Halo
reads it (through Lookup's read-only `security` tool); the owner reads it on the Assistant's Halo panel.

- **Append-only.** Its own SQLite file beside the ledger (`<ledger>.security.sqlite3`). Triggers refuse every UPDATE
  and DELETE; each row carries the sha256 of the row before it, so a row removed or changed by hand shows as a break
  in the chain (`verify()`).
- **Never a secret.** A key is named by its id or its 8-character hint, a prompt never appears, an address only as
  the connecting machine's own (loopback, a LAN address or a Gravity node id).
- **No flood.** The same event (kind, who, what) within `COLLAPSE_S` adds to the last row's count instead of a new row
  (an attacker hammering a bad key makes one row with a count, not a million rows). Key use is written once per key
  per hour.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time

from nebula import migrate

COLLAPSE_S = 60.0
KEY_USE_EVERY_S = 3600.0
SEVERITY = ("info", "notice", "warn", "alert")

# kind -> severity. Anything not listed is "info".
KINDS = {
    "auth.refused": "warn",            # a request without the key the API needs, or a wrong one
    "auth.unpaired": "warn",           # from beyond this machine with no paired token
    "auth.signin": "notice",           # the page API without a signed-in browser
    "signin.ok": "notice", "signin.refused": "warn", "signin.changed": "alert",
    "auth.bad_host": "warn",           # a Host that is not one of this Nebula's names (DNS rebinding)
    "auth.cross_site": "warn",         # another web site's page tried to use this Nebula
    "key.used": "info",                # a key's first use in the hour (by id, never the key)
    "key.made": "notice", "key.revoked": "notice",
    "caller.changed": "notice", "caller.removed": "notice",
    "guest.refused": "notice",         # a guest asked for a model not shared with it
    "guest.paused": "warn", "guest.limited": "warn",
    "gravity.invite": "notice", "gravity.join": "warn", "gravity.joined": "notice", "gravity.refused": "warn",
    "gravity.grants": "notice", "gravity.revoked": "notice", "gravity.relay": "notice",
    "share.changed": "notice",         # a model marked shared or not, a caller's ticks changed
    "server.added": "notice", "server.trust": "notice", "server.removed": "notice",
    "network.invite": "notice", "network.joined": "notice",
    "entry.added": "notice", "entry.changed": "notice", "entry.removed": "notice", "otel.settings": "notice",
    "halo.action": "alert", "halo.undo": "notice", "halo.ask": "alert", "halo.error": "warn",
    "admin.refused": "warn",
    "sessions.deleted": "notice",      # conversations deleted for good on the page (how many, never what)
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  at REAL NOT NULL,
  last_at REAL NOT NULL,
  kind TEXT NOT NULL,
  severity TEXT NOT NULL,
  who TEXT,
  what TEXT,
  detail TEXT NOT NULL,
  count INTEGER NOT NULL DEFAULT 1,
  prev_sha TEXT NOT NULL,
  sha TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS repeats (            -- a collapsed repeat: the row it adds to, its own time (append-only too)
  seq INTEGER NOT NULL,
  at REAL NOT NULL
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'the security log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'the security log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS repeats_no_update BEFORE UPDATE ON repeats BEGIN SELECT RAISE(ABORT, 'the security log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS repeats_no_delete BEFORE DELETE ON repeats BEGIN SELECT RAISE(ABORT, 'the security log is append-only'); END;
CREATE INDEX IF NOT EXISTS events_kind ON events(kind, at);
"""


def _sha(prev: str, row: tuple) -> str:
    return hashlib.sha256((prev + "\x1f" + json.dumps(row, separators=(",", ":"), sort_keys=True)).encode()).hexdigest()


class SecLog:
    def __init__(self, path: str | None):
        self.path = path or ":memory:"
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL" if path else "PRAGMA journal_mode=MEMORY")
        migrate.sqlite_upgrade(self.db, "the security log", [SCHEMA], path)
        self.lock = threading.Lock()
        self.recent: dict[tuple, tuple[int, float]] = {}        # (kind, who, what) -> (seq, when it was last written)
        self.key_seen: dict[str, float] = {}
        self.listeners: list = []                               # called with each new row (Halo wakes on them)

    def close(self):
        try:
            self.db.close()
        except Exception:  # noqa: BLE001
            pass

    # --- writing
    def add(self, kind: str, who: str | None = None, what: str | None = None, **detail) -> int:
        """One event; a repeat of the same (kind, who, what) within COLLAPSE_S is counted on the row it repeats."""
        now = time.time()
        sev = detail.pop("severity", None) or KINDS.get(kind, "info")
        key = (kind, who, what)
        with self.lock:
            last = self.recent.get(key)
            if last is not None and now - last[1] < COLLAPSE_S:
                self.db.execute("INSERT INTO repeats(seq, at) VALUES (?, ?)", (last[0], now))
                self.recent[key] = (last[0], now)
                seq, fresh = last[0], False
            else:
                prev = self.db.execute("SELECT sha FROM events ORDER BY seq DESC LIMIT 1").fetchone()
                prev_sha = prev[0] if prev else "0" * 64
                d = json.dumps(detail, ensure_ascii=False, sort_keys=True, default=str)
                row = (now, kind, sev, who, what, d)
                cur = self.db.execute("INSERT INTO events(at, last_at, kind, severity, who, what, detail, prev_sha, sha) "
                                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (now, now, kind, sev, who, what, d, prev_sha, _sha(prev_sha, row)))
                seq, fresh = cur.lastrowid, True
                self.recent[key] = (seq, now)
                if len(self.recent) > 4096:
                    cut = now - COLLAPSE_S
                    self.recent = {k: v for k, v in self.recent.items() if v[1] >= cut}
        for f in list(self.listeners):
            try:
                f({"seq": seq, "kind": kind, "severity": sev, "who": who, "what": what, "fresh": fresh})
            except Exception:  # noqa: BLE001 - a listener never stops the log
                pass
        return seq

    def key_used(self, key_id: str, who: str, address: str | None) -> None:
        """A key's use, once per key per hour (and again at once from a new address)."""
        now = time.time()
        k = f"{key_id}@{address}"
        if now - self.key_seen.get(k, 0) < KEY_USE_EVERY_S:
            return
        self.key_seen[k] = now
        self.add("key.used", who, key_id, address=address)

    # --- reading
    def _rows(self, where: str = "", args: tuple = (), limit: int = 100) -> list[dict]:
        q = ("SELECT e.seq, e.at, e.kind, e.severity, e.who, e.what, e.detail, "
             "1 + (SELECT COUNT(*) FROM repeats r WHERE r.seq = e.seq), "
             "COALESCE((SELECT MAX(r.at) FROM repeats r WHERE r.seq = e.seq), e.at) FROM events e "
             + where + " ORDER BY e.seq DESC LIMIT ?")
        with self.lock:
            rows = self.db.execute(q, args + (int(limit),)).fetchall()
        return [{"seq": r[0], "at": r[1], "kind": r[2], "severity": r[3], "who": r[4], "what": r[5],
                 "detail": json.loads(r[6] or "{}"), "count": r[7], "last_at": r[8]} for r in rows]

    def read(self, since_seq: int = 0, kinds: list[str] | None = None, who: str | None = None, min_severity: str | None = None,
             since_s: float | None = None, limit: int = 100) -> list[dict]:
        """Newest first. A row's count and last_at include its collapsed repeats."""
        cond, args = ["e.seq > ?"], [int(since_seq or 0)]
        if kinds:
            cond.append("(" + " OR ".join("e.kind = ? OR e.kind LIKE ?" for _ in kinds) + ")")
            for k in kinds:
                args += [k, k.rstrip(".*") + ".%" if k.endswith("*") else k]
        if who:
            cond.append("e.who = ?")
            args.append(who)
        if min_severity in SEVERITY:
            ok = SEVERITY[SEVERITY.index(min_severity):]
            cond.append("e.severity IN (" + ",".join("?" * len(ok)) + ")")
            args += list(ok)
        if since_s:
            cond.append("(e.at >= ? OR EXISTS (SELECT 1 FROM repeats r WHERE r.seq = e.seq AND r.at >= ?))")
            t = time.time() - float(since_s)
            args += [t, t]
        return self._rows("WHERE " + " AND ".join(cond), tuple(args), max(1, min(int(limit or 100), 1000)))

    def last_seq(self) -> int:
        with self.lock:
            r = self.db.execute("SELECT MAX(seq) FROM events").fetchone()
        return int(r[0] or 0)

    def counts(self, since_s: float = 3600) -> dict:
        """Per kind, in the last `since_s` seconds (repeats counted)."""
        t = time.time() - since_s
        with self.lock:
            rows = self.db.execute("SELECT kind, COUNT(*) FROM (SELECT e.kind FROM events e WHERE e.at >= ? UNION ALL "
                                   "SELECT e.kind FROM repeats r JOIN events e ON e.seq = r.seq WHERE r.at >= ?) GROUP BY kind",
                                   (t, t)).fetchall()
        return {k: n for k, n in rows}

    def verify(self) -> dict:
        """Walk the chain: the first row whose hash or link does not hold, or ok."""
        prev = "0" * 64
        n = 0
        with self.lock:
            cur = self.db.execute("SELECT seq, at, kind, severity, who, what, detail, prev_sha, sha FROM events ORDER BY seq")
            for seq, at, kind, sev, who, what, d, prev_sha, sha in cur:
                if prev_sha != prev or _sha(prev, (at, kind, sev, who, what, d)) != sha:
                    return {"ok": False, "broken_at": seq, "rows": n}
                prev, n = sha, n + 1
        return {"ok": True, "rows": n}


def path_for(ledger: str | None) -> str | None:
    if not ledger or ledger == ":memory:":
        return None
    return ledger + ".security.sqlite3"


# --- the page's writes that change who may do what: logged at the API, one place for all of them
import re as _re

WRITES = [
    (r"^/api/stars/api/keys$", "key.made", "api"),
    (r"^/api/stars/api/keys/([^/]+)/revoke$", "key.revoked", "api"),
    (r"^/api/stars/api$", "share.changed", "api"),
    (r"^/api/yard/callers$", "caller.changed", None),
    (r"^/api/yard/callers/([^/]+)/keys$", "key.made", "caller"),
    (r"^/api/yard/callers/([^/]+)/keys/([^/]+)/revoke$", "key.revoked", "caller"),
    (r"^/api/yard/callers/([^/]+)/remove$", "caller.removed", "caller"),
    (r"^/api/yard/node/invite$", "gravity.invite", None),
    (r"^/api/yard/node/public-relay$", "gravity.relay", None),
    (r"^/api/yard/node/join$", "gravity.joined", None),
    (r"^/api/yard/node/peers/([^/]+)/grants$", "gravity.grants", "peer"),
    (r"^/api/yard/node/peers/([^/]+)/revoke$", "gravity.revoked", "peer"),
    (r"^/api/yard/node/models/([^/]+)/risky$", "share.changed", "model"),
    (r"^/api/yard/servers/trust$", "server.trust", None),
    (r"^/api/yard/servers/install$", "server.added", None),
    (r"^/api/yard/servers/([^/]+)/uninstall$", "server.removed", "server"),
    (r"^/api/yard/network/invite$", "network.invite", None),
    (r"^/api/yard/network/accept$", "network.joined", None),
    (r"^/api/models/([^/]+)/shared$", "share.changed", "model"),
    (r"^/api/api-servers$", "entry.added", None),
    (r"^/api/api-servers/([^/]+)/keys$", "key.made", "entry"),
    (r"^/api/api-servers/([^/]+)/keys/([^/]+)/revoke$", "key.revoked", "entry"),
    (r"^/api/api-servers/([^/]+)/remove$", "entry.removed", "entry"),
    (r"^/api/api-servers/([^/]+)$", "entry.changed", "entry"),
    (r"^/api/lookup/collections/([^/]+)/share$", "share.changed", "collection"),
    (r"^/api/halo$", "halo.settings", None),
    (r"^/api/otel$", "otel.settings", None),
]
_WRITES = [(_re.compile(p), k, w) for p, k, w in WRITES]
SAFE_FIELDS = ("id", "name", "models", "class", "guest", "kind", "address", "host", "port", "fingerprint", "label",
               "shared", "tenants", "scope", "formats", "risky", "confirm", "enabled", "on")


def page_write(log: SecLog | None, path: str, req: dict, status: int) -> None:
    """A page write that changes who may do what: its kind, the subject from the path, a few safe fields of the body
    (never a key, token, secret, invite, PIN or password)."""
    if log is None:
        return
    try:
        _page_write(log, path, req, status)
    except Exception:  # noqa: BLE001 - the log never breaks the write it records
        import logging
        logging.getLogger("nebula.seclog").warning("the security log could not record %s", path, exc_info=True)


def _page_write(log: SecLog, path: str, req: dict, status: int) -> None:
    for rx, kind, what in _WRITES:
        m = rx.match(path)
        if m:
            subject = "/".join(m.groups()) if m.groups() else None
            body = {k: req.get(k) for k in SAFE_FIELDS if isinstance(req, dict) and k in req}
            log.add(kind if status < 400 else kind + ".failed", "owner", f"{what}:{subject}" if what and subject else subject,
                    status=status, request=body)
            return
