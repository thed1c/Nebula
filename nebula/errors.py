"""nebula/errors.py: the error log the System tab shows, with a button to clear it.

A logging handler on the cockpit's loggers and Nebula's own (`nebula.*`: Halo, Starweave, the toolbox, Gravity, updates,
and serve/'s under it, whose MCP hub the cockpit runs) keeps every warning and error
in memory, newest last, up to `capacity`: a request the engine refused or failed, a hook, callback or schedule that
failed, a compaction that did not come back, an agent's tool that raised, a request handler that crashed. The page adds
its own (`add`, from POST /api/errors): a turn that failed in the room, a script error; and the engine's own problem
lines (nebula/enginelog.py). Clearing empties this log only;
the ledger keeps its record of every failed attempt.
"""
from __future__ import annotations

import collections
import logging
import threading
import traceback
from datetime import datetime, timezone

LOGGERS = ("cockpit", "nebula")            # "nebula" holds nebula._vendor.strata_serve too
REPEAT_S = 3600                           # the same warning again within an hour adds to its count


class ErrorLog(logging.Handler):
    def __init__(self, capacity: int = 500):
        super().__init__(logging.WARNING)
        self.items: collections.deque = collections.deque(maxlen=capacity)
        self.mutex = threading.Lock()
        self.seq = 0                              # grows forever: the page knows what is new by it

    def emit(self, record: logging.LogRecord) -> None:
        try:
            detail = "".join(traceback.format_exception(*record.exc_info)) if record.exc_info else None
            self.add(record.getMessage(), source=record.name, level=record.levelname.lower(), detail=detail,
                     collapse_s=REPEAT_S)       # a job that fails every few minutes is one line with its count
        except Exception:  # noqa: BLE001 - a log handler must never raise
            pass

    def add(self, message: str, source: str = "cockpit", level: str = "error", detail: str | None = None,
            collapse_s: float = 0) -> dict:
        """One entry. With `collapse_s`, the same message from the same source within that many seconds of its last time
        adds to that entry's count ("repeats", "last_at") instead of a new entry."""
        message = str(message)[:2000]
        now = datetime.now(timezone.utc)
        with self.mutex:
            if collapse_s:
                # the same message from the same source at the same level within the window adds to that entry's
                # count, and keeps the newest traceback (a warning then an error stay two entries, the error with its own)
                for item in reversed(self.items):
                    if (now - datetime.fromisoformat(item.get("last_at") or item["at"])).total_seconds() > collapse_s:
                        break               # older than the window: so is everything before it
                    if item["source"] == source and item["message"] == message and item["level"] == level:
                        item["repeats"] = item.get("repeats", 1) + 1
                        item["last_at"] = now.isoformat(timespec="seconds")
                        if detail:
                            item["detail"] = str(detail)[:8000]
                        return item
            self.seq += 1
            item = {"id": self.seq, "at": now.isoformat(timespec="seconds"), "level": level, "source": source,
                    "message": message, "detail": (detail or None) and str(detail)[:8000]}
            self.items.append(item)
            return item

    def list(self) -> dict:
        with self.mutex:
            return {"errors": list(self.items), "count": len(self.items), "last_id": self.seq}

    def count(self) -> int:
        with self.mutex:
            return len(self.items)

    def clear(self) -> int:
        with self.mutex:
            n = len(self.items)
            self.items.clear()
            return n

    def attach(self) -> "ErrorLog":
        for name in LOGGERS:
            logging.getLogger(name).addHandler(self)
        return self

    def detach(self) -> None:
        for name in LOGGERS:
            logging.getLogger(name).removeHandler(self)
