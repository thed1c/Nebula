"""nebula/replay.py: rebuild every session's transcript from its events alone, and check it against what the
attempts recorded. A briefing or a handoff event changes nothing in the transcript; its hash is checked like any. This is step 2's done-when (docs/DESIGN.md).

    python3 -m nebula.replay [--ledger nebula.sqlite3] [SESSION_ID]

For each session, in sequence order: a rewind cuts the conversation back, a message or a reply appends its
canonical form, and a request boundary must find the conversation hashing to exactly the request_sha256 the attempt
row stored before the engine was called. Every event's stored hashes are recomputed from its payload; every
completed attempt must have a reply whose hash the attempt row holds; sequence numbers must run 1, 2, 3 with no
gap. One mismatch fails the run.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from nebula.events import canonical_message, sha256_of, transcript_sha256


def verify_session(db, session_id: str) -> tuple[list[str], list[str]]:
    """(lines of what was checked, lines of what failed)."""
    notes, bad = [], []
    attempts = {r["attempt_id"]: dict(r) for r in
                db.execute("SELECT * FROM attempts WHERE session_id=?", (session_id,))}
    events = [dict(r) for r in db.execute("SELECT * FROM events WHERE session_id=? ORDER BY seq", (session_id,))]
    conv: list[dict] = []
    requests_seen: dict[str, int] = {}
    replies_seen: dict[str, int] = {}
    for n, e in enumerate(events, 1):
        if e["seq"] != n:
            bad.append(f"seq gap: event {e['event_id']} has seq {e['seq']}, expected {n}")
        payload = json.loads(e["payload"])
        if sha256_of(payload) != e["content_sha256"]:
            bad.append(f"seq {e['seq']}: content_sha256 does not match the payload")
        kind = e["kind"]
        if kind == "rewind":
            conv = conv[:payload["keep"]]
        elif kind in ("message", "reply"):
            canon = canonical_message(payload)
            if sha256_of(canon) != e["canonical_sha256"]:
                bad.append(f"seq {e['seq']}: canonical_sha256 does not match the payload")
            if kind == "reply":
                att = attempts.get(e["attempt_id"])
                replies_seen[e["attempt_id"]] = replies_seen.get(e["attempt_id"], 0) + 1
                if att is None:
                    bad.append(f"seq {e['seq']}: reply for unknown attempt {e['attempt_id']}")
                elif att["reply_sha256"] != e["content_sha256"]:
                    bad.append(f"seq {e['seq']}: the attempt's reply_sha256 differs from the reply event")
            conv.append(canon)
        elif kind in ("briefing", "handoff", "lifecycle", "tool_round"):
            if kind == "handoff" and e["attempt_id"] not in attempts:
                bad.append(f"seq {e['seq']}: handoff from unknown attempt {e['attempt_id']}")
        elif kind == "request":
            att = attempts.get(e["attempt_id"])
            requests_seen[e["attempt_id"]] = requests_seen.get(e["attempt_id"], 0) + 1
            rebuilt = transcript_sha256(conv)
            if att is None:
                bad.append(f"seq {e['seq']}: request for unknown attempt {e['attempt_id']}")
            elif not (rebuilt == payload["request_sha256"] == att["request_sha256"] == e["canonical_sha256"]):
                bad.append(f"seq {e['seq']}: the transcript rebuilt from events ({rebuilt[:12]}) is not the request "
                           f"the attempt recorded ({att['request_sha256'][:12]})")
        else:
            bad.append(f"seq {e['seq']}: unknown event kind {kind!r}")
    for aid, att in attempts.items():
        if requests_seen.get(aid, 0) != 1:
            bad.append(f"attempt {aid}: {requests_seen.get(aid, 0)} request events, expected 1")
        if att["state"] == "completed" and replies_seen.get(aid, 0) != 1:
            bad.append(f"attempt {aid}: completed with {replies_seen.get(aid, 0)} reply events")
        if att["state"] == "running":
            bad.append(f"attempt {aid}: still running (the request was never closed)")
    notes.append(f"{session_id}: {len(events)} events, {len(attempts)} attempts, "
                 f"{len(conv)} messages in the final transcript")
    return notes, bad


def verify(path: str, session_id: str | None = None) -> tuple[bool, list[str]]:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        ids = [session_id] if session_id else [r["session_id"] for r in
                                               db.execute("SELECT session_id FROM sessions ORDER BY created_at")]
        lines, ok = [], True
        for sid in ids:
            notes, bad = verify_session(db, sid)
            lines += notes
            for b in bad:
                lines.append("  FAIL " + b)
            ok = ok and not bad
        lines.append(f"{len(ids)} sessions: {'all rebuilt' if ok else 'MISMATCH'}")
        return ok, lines
    finally:
        db.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nebula.replay", description=__doc__.split("\n")[0])
    p.add_argument("--ledger", default=None, help="the ledger (default nebula.sqlite3, or an existing cockpit.sqlite3)")
    p.add_argument("session_id", nargs="?")
    a = p.parse_args(argv)
    from nebula import names
    ok, lines = verify(a.ledger or names.local_path(None, "nebula.sqlite3", "cockpit.sqlite3"), a.session_id)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
