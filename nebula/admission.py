"""nebula/admission.py: the gate at every engine, and where a turn goes when its engine is busy (Switchyard
plumbing step B: capacity routing).

One gate per engine address (several tiers may share one engine), as many slots as the engine reports. A turn
takes a slot before its engine call and gives it back after; while it waits it is served by class, then arrival:
interactive (the room, the API dialects) ahead of agent (webhooks) ahead of background (schedules). The wait is
visible (`waiting()`) and, once over, recorded on the attempt.

Expected wait at an engine for a class = (waiters served before it + held slots - slots + 1) / slots × the
engine's average call, the average learnt from finished calls. A turn whose chosen engine is busy keeps it while
the expected wait is within its class's tolerance (interactive 60 s, agent 15 s, background 5 s) or the session is
already sticky on it; past that, the first later ticked tier on another engine with a free slot takes it, else the
one with the shortest expected wait when it beats the chosen one by more than the tolerance. The choice is a
reason on the decision ("overflow:<tier>"), and a different engine is a crossing like any other.
"""
from __future__ import annotations

import itertools
import threading
import time

CLASSES = ("interactive", "agent", "background")
RANK = {"interactive": 0, "agent": 1, "background": 2}
DEFAULT_TOLERANCE = {"interactive": 60.0, "agent": 15.0, "background": 5.0}
DEFAULT_AVG_SECONDS = 5.0


class QueueTimeout(Exception):
    pass


class Gate:
    """A counting gate whose waiters are served by (rank, arrival)."""

    def __init__(self, slots: int = 1):
        self.cond = threading.Condition()
        self.slots = max(1, int(slots))
        self.held = 0
        self.queue: list[list] = []              # [rank, seq, info, since]
        self._seq = itertools.count()
        self.holding: list[dict] = []            # the calls holding a slot: their info, with "since" (epoch seconds)

    def resize(self, slots: int) -> None:
        with self.cond:
            self.slots = max(1, int(slots))
            self.cond.notify_all()

    def _head(self):
        return min(self.queue, key=lambda e: (e[0], e[1])) if self.queue else None

    def acquire(self, rank: int, info: dict, timeout: float | None = None) -> float:
        """Blocks until this call holds a slot; returns the seconds it waited."""
        entry = [rank, next(self._seq), info, time.monotonic()]
        with self.cond:
            self.queue.append(entry)
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                if self._head() is entry and self.held < self.slots:
                    self.queue.remove(entry)
                    self.held += 1
                    info["since"] = time.time()
                    self.holding.append(info)
                    self.cond.notify_all()           # the next head may be free too
                    return time.monotonic() - entry[3]
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    self.queue.remove(entry)
                    self.cond.notify_all()
                    raise QueueTimeout(f"waited {timeout:.0f} s for a slot")
                self.cond.wait(left)

    def release(self, info: dict | None = None) -> None:
        """A slot given back: the holder's own info when it is known (else the oldest holder's row goes)."""
        with self.cond:
            self.held = max(0, self.held - 1)
            if info is not None and info in self.holding:
                self.holding.remove(info)
            elif self.holding and len(self.holding) > self.held:
                self.holding.pop(0)
            self.cond.notify_all()

    def queued(self, rank: int) -> int:
        """How many of exactly this class wait now."""
        with self.cond:
            return sum(1 for e in self.queue if e[0] == rank)

    def ahead(self, rank: int) -> int:
        """Waiters a new call of this rank would be served after (the same class or better: they came first)."""
        with self.cond:
            return sum(1 for e in self.queue if e[0] <= rank)

    def free(self, rank: int) -> bool:
        with self.cond:
            return self.held < self.slots and not any(e[0] <= rank for e in self.queue)

    def status(self) -> dict:
        with self.cond:
            now = time.monotonic()
            return {"slots": self.slots, "held": self.held, "holding": [dict(h) for h in self.holding],
                    "waiting": [dict(e[2], **{"class": CLASSES[e[0]] if 0 <= e[0] < len(CLASSES) else str(e[0]),
                                              "waited_s": round(now - e[3], 1)}) for e in sorted(self.queue, key=lambda e: (e[0], e[1]))]}


class Admission:
    def __init__(self, cfg, registry, tolerance: dict | None = None, default_avg: float = DEFAULT_AVG_SECONDS):
        self.cfg, self.registry = cfg, registry
        self.tolerance = {**DEFAULT_TOLERANCE, **(tolerance or {})}
        self.default_avg = default_avg
        self.gates: dict[str, Gate] = {}
        self.avg: dict[str, float] = {}          # engine address -> EWMA seconds of a call
        self.lock = threading.Lock()
        self.refresh(cfg, registry)

    # --- the gates follow the config and the registry
    def refresh(self, cfg, registry) -> None:
        self.cfg, self.registry = cfg, registry
        with self.lock:
            for t in cfg.tiers:
                if not t.address:
                    continue
                node = registry.node(t.name) if registry is not None else None
                slots = (node or {}).get("slots") or 1
                g = self.gates.get(t.address)
                if g is None:
                    self.gates[t.address] = Gate(slots)
                elif g.slots != slots:
                    g.resize(slots)

    def gate(self, tier) -> Gate:
        with self.lock:
            g = self.gates.get(tier.address)
            if g is None:
                g = self.gates[tier.address] = Gate(1)
            return g

    # --- what a call costs and would wait
    def learn(self, tier, seconds: float) -> None:
        if seconds <= 0:
            return
        with self.lock:
            old = self.avg.get(tier.address)
            self.avg[tier.address] = seconds if old is None else old * 0.7 + seconds * 0.3

    def average(self, tier) -> float:
        with self.lock:
            return self.avg.get(tier.address, self.default_avg)

    def expected_wait(self, tier, rank: int) -> float:
        g = self.gate(tier)
        with g.cond:
            queued = sum(1 for e in g.queue if e[0] <= rank)
            calls_ahead = max(0, queued + g.held - g.slots + 1)
        return calls_ahead / g.slots * self.average(tier)

    def choose(self, ladder: list, index: int, rank: int, sticky: bool, wait_s=None) -> tuple[int, str | None]:
        """(the index in the ladder to use, the reason when it is not the one asked for). `wait_s(tier)`: that model's
        own wait (the Switchyard's capacity wait), when shorter than the class's tolerance."""
        cur = ladder[index]
        if sticky or self.gate(cur).free(rank):
            return index, None
        klass = CLASSES[rank] if 0 <= rank < len(CLASSES) else "agent"
        tol = self.tolerance.get(klass, 15.0)
        own = wait_s(cur) if wait_s is not None else None
        if own is not None:
            tol = min(tol, float(own))
        cur_wait = self.expected_wait(cur, rank)
        if cur_wait <= tol:
            return index, None
        best, best_wait = None, None
        for j in range(index + 1, len(ladder)):
            t = ladder[j]
            if t.address == cur.address:
                continue
            if self.gate(t).free(rank):
                return j, f"overflow:{t.name}"
            w = self.expected_wait(t, rank)
            if best is None or w < best_wait:
                best, best_wait = j, w
        if best is not None and best_wait + tol < cur_wait:
            return best, f"overflow:{ladder[best].name}"
        return index, None

    # --- the queue, for the page
    def waiting(self) -> list[dict]:
        out = []
        with self.lock:
            gates = list(self.gates.items())
        for address, g in gates:
            st = g.status()
            tiers = [t.name for t in self.cfg.tiers if t.address == address]
            for w in st["waiting"]:
                out.append({**w, "address": address, "tiers": tiers, "slots": st["slots"], "held": st["held"]})
        return out

    def status(self) -> dict:
        with self.lock:
            gates = list(self.gates.items())
        return {address: {**g.status(), "tiers": [t.name for t in self.cfg.tiers if t.address == address],
                          "average_s": round(self.avg.get(address, self.default_avg), 2)} for address, g in gates}


def priority_of(input_name: str | None, explicit: str | None, agent_priority: str | None) -> str:
    """The class of a turn: the input's own setting, else the agent's, else by what the input is."""
    for p in (explicit, agent_priority):
        if p in CLASSES:
            return p
    if input_name and input_name.startswith("hook:"):
        return "agent"
    return "interactive"
