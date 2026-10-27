"""nebula/parking.py: the parking lot (Switchyard plumbing step C).

A request no ticked model can hold is parked: its handler thread waits here, visibly, with its reason; every few
seconds the lot asks again whether a model holds it now (a policy change ticked a bigger one, a model was added,
an engine came back with a larger context), and the first one that does takes it; after the configured wait it is
refused 503 `parked_too_big`. Nothing is sent to fail and nothing is truncated. `release(name)` wakes the lot early
when something changed.
"""
from __future__ import annotations

import threading
import time
import uuid


class ParkedTooBig(Exception):
    def __init__(self, message: str, item: dict):
        super().__init__(message)
        self.item = item


class Lot:
    def __init__(self, recheck_seconds: float = 3.0, wait_seconds: float = 600.0):
        self.cond = threading.Condition()
        self.recheck_seconds, self.wait_seconds = recheck_seconds, wait_seconds
        self.items: dict[str, dict] = {}
        self.history: list[dict] = []        # the last parked items that left, newest last, bounded

    def park(self, info: dict, fits_now, wait_seconds: float | None = None) -> tuple[dict, str]:
        """Block until `fits_now()` names a tier (returned with the item), or raise ParkedTooBig after the wait."""
        item = {"id": "park_" + uuid.uuid4().hex[:12], "since": time.time(), "since_mono": time.monotonic(), **info}
        wait = self.wait_seconds if wait_seconds is None else wait_seconds
        with self.cond:
            self.items[item["id"]] = item
            self.cond.notify_all()
        try:
            while True:
                tier = fits_now()
                if tier is not None:
                    item["left"] = "released"
                    item["to"] = tier
                    return item, tier
                waited = time.monotonic() - item["since_mono"]
                if waited >= wait:
                    item["left"] = "refused"
                    raise ParkedTooBig(f"parked_too_big: {item.get('reason', 'no ticked model holds it')} "
                                       f"after {int(waited)} s", item)
                with self.cond:
                    self.cond.wait(min(self.recheck_seconds, max(0.05, wait - waited)))
        finally:
            with self.cond:
                self.items.pop(item["id"], None)
                item["waited_s"] = round(time.monotonic() - item["since_mono"], 2)
                self.history.append(item)
                del self.history[:-50]
                self.cond.notify_all()

    def poke(self) -> None:
        """Something changed (a policy, a model): the parked ask again now."""
        with self.cond:
            self.cond.notify_all()

    def parked(self) -> list[dict]:
        with self.cond:
            now = time.monotonic()
            return [{k: v for k, v in it.items() if k != "since_mono"} | {"waited_s": round(now - it["since_mono"], 1)}
                    for it in self.items.values()]

    def status(self) -> dict:
        with self.cond:
            return {"parked": self.parked(), "recent": [{k: v for k, v in it.items() if k != "since_mono"} for it in self.history[-10:]],
                    "recheck_seconds": self.recheck_seconds, "wait_seconds": self.wait_seconds}
