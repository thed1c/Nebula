"""nebula/guard.py: the small, undoable holds Halo (or the owner) puts on a guest, checked before the Switchyard runs.

    "guard": {"paused":  {"caller:friend-app": <until, epoch s>},
              "limits":  {"peer:ab12…": {"per_min": 6, "until": <epoch s>}},
              "denied":  {"caller:friend-app": ["big-model"]}}

- **paused**: the guest's requests are refused (403) until then.
- **limits**: at most `per_min` requests a minute (429 past it) until then.
- **denied**: a model held back from that guest (its ticks or grants are untouched; undo gives it back).

Kept in the config (so a restart keeps them, and the owner can read them there); expired entries drop out by
themselves. The owner's own traffic is never held. The Switchyard's rules stay the real guard: these only tighten.
"""
from __future__ import annotations

import threading
import time
from collections import deque

OWNER = "owner"
MAX_S = 24 * 3600


class Guard:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.seen: dict[str, deque] = {}

    def _sec(self) -> dict:
        g = self.cfg.raw.setdefault("guard", {})
        for k in ("paused", "limits", "denied"):
            g.setdefault(k, {})
        now = time.time()
        g["paused"] = {t: u for t, u in g["paused"].items() if u > now}
        g["limits"] = {t: v for t, v in g["limits"].items() if v.get("until", 0) > now}
        g["denied"] = {t: m for t, m in g["denied"].items() if m}
        return g

    def save(self):
        if self.cfg.path:
            self.cfg.save()

    # --- the checks
    def check(self, tenant: str | None) -> tuple[int, str, str] | None:
        """None, or (status, message, the security log's kind) for a request from that tenant."""
        if not tenant or tenant == OWNER:
            return None
        with self.lock:
            g = self._sec()
            until = g["paused"].get(tenant)
            if until:
                return 403, f"paused until {time.strftime('%H:%M', time.localtime(until))} (Halo, or this Nebula's owner)", "guest.paused"
            lim = g["limits"].get(tenant)
            if lim:
                q = self.seen.setdefault(tenant, deque())
                now = time.time()
                while q and now - q[0] > 60:
                    q.popleft()
                if len(q) >= int(lim["per_min"]):
                    return 429, f"at most {lim['per_min']} requests a minute for now (Halo, or this Nebula's owner)", "guest.limited"
                q.append(now)
        return None

    def denied(self, tenant: str | None) -> list[str]:
        if not tenant or tenant == OWNER:
            return []
        with self.lock:
            return list(self._sec()["denied"].get(tenant) or [])

    def view(self) -> dict:
        with self.lock:
            g = self._sec()
            return {"paused": dict(g["paused"]), "limits": dict(g["limits"]), "denied": {t: list(m) for t, m in g["denied"].items()}}

    # --- the holds (each returns what undoes it)
    def _who(self, tenant: str) -> str:
        if not (str(tenant).startswith("caller:") or str(tenant).startswith("peer:")):
            raise ValueError("a guest is caller:<id> or peer:<id>; the owner is never held")
        return str(tenant)

    def pause(self, tenant: str, seconds: float) -> dict:
        t = self._who(tenant)
        with self.lock:
            g = self._sec()
            before = g["paused"].get(t)
            g["paused"][t] = time.time() + max(60, min(float(seconds), MAX_S))
        self.save()
        return {"op": "pause", "tenant": t, "before": before}

    def limit(self, tenant: str, per_min: int, seconds: float) -> dict:
        t = self._who(tenant)
        if int(per_min) < 1:
            raise ValueError("a limit lets at least one request a minute through (pause stops them all)")
        with self.lock:
            g = self._sec()
            before = g["limits"].get(t)
            g["limits"][t] = {"per_min": int(per_min), "until": time.time() + max(60, min(float(seconds), MAX_S))}
        self.save()
        return {"op": "limit", "tenant": t, "before": before}

    def deny(self, tenant: str, model: str) -> dict:
        t = self._who(tenant)
        with self.lock:
            g = self._sec()
            had = model in (g["denied"].get(t) or [])
            g["denied"][t] = list(dict.fromkeys((g["denied"].get(t) or []) + [model]))
        self.save()
        return {"op": "deny", "tenant": t, "model": model, "had": had}

    def undo(self, u: dict) -> None:
        with self.lock:
            g = self._sec()
            t = u["tenant"]
            if u["op"] == "pause":
                if u.get("before"):
                    g["paused"][t] = u["before"]
                else:
                    g["paused"].pop(t, None)
            elif u["op"] == "limit":
                if u.get("before"):
                    g["limits"][t] = u["before"]
                else:
                    g["limits"].pop(t, None)
            elif u["op"] == "deny" and not u.get("had"):
                g["denied"][t] = [m for m in g["denied"].get(t) or [] if m != u["model"]]
        self.save()
