"""nebula/registry.py: the model registry (docs/DESIGN.md, Switchyard plumbing step A).

Every tier is a node the cockpit keeps current from the engine itself: up or down (its /health, probed at most
every few seconds and again right after a failure), slots and context per slot (/props), vision (/props
modalities), its model alias; and from the config: host, power class, cost, thinking, tools. The router asks the
registry which ticked models are usable; the page reads it as /api/registry. It is read, not written, since the engines
say what they are.
"""
from __future__ import annotations

import json
import threading
import time


class Registry:
    policy = None                                    # cockpit.policy.Policy: capacity and vision overrides

    def __init__(self, cfg, engines: dict, probe_every: float = 3.0):
        self.cfg, self.engines, self.probe_every = cfg, engines, probe_every
        self.lock = threading.Lock()
        self._health: dict[str, tuple[float, bool]] = {}      # tier -> (when, up)
        self.not_ready: dict[str, str] = {}                     # tier -> why it answers but is not ready (the wrong file)

    # --- health
    def up(self, name: str, force: bool = False) -> bool:
        eng = self.engines.get(name)
        if eng is None:
            return False
        now = time.monotonic()
        with self.lock:
            cached = self._health.get(name)
            if cached and not force and now - cached[0] < self.probe_every:
                return cached[1]
        status, _ctype, _data = eng.get(getattr(eng, "health_path", "/health"))
        ok = status == 200
        expect = self.policy.expect_model(name) if (ok and self.policy is not None) else None
        if expect:                                   # ready only with the file it is meant to have loaded
            eng._props = None
            loaded = str((eng.props() or {}).get("model_path") or "")
            if expect.lower() not in loaded.lower():
                ok = False
                self.not_ready[name] = f"loaded {loaded.rsplit('/', 1)[-1] or 'an unknown file'}, expected {expect}"
            else:
                self.not_ready.pop(name, None)
        elif ok:
            self.not_ready.pop(name, None)
        with self.lock:
            self._health[name] = (now, ok)
        return ok

    def mark_down(self, name: str) -> None:
        """An engine that just failed a request is down until the next probe."""
        with self.lock:
            self._health[name] = (time.monotonic(), False)

    def usable(self, names) -> list[str]:
        """Those of the tiers, in the given order, whose engine answers."""
        return [n for n in names if self.cfg.tier(n) is not None and self.up(n)]

    # --- nodes
    def node(self, name: str) -> dict | None:
        t = self.cfg.tier(name)
        eng = self.engines.get(name)
        if t is None or eng is None:
            return None
        up = self.up(name)
        props = eng.props() if up else {}
        if not up:
            eng._props = None                            # read again when it is back: a restart may have changed it
        modalities = props.get("modalities") if isinstance(props.get("modalities"), dict) else {}
        http = bool(t.address) and t.address.startswith("http")
        slots = props.get("total_slots") if isinstance(props.get("total_slots"), int) else (1 if t.engine else None)
        if getattr(t, "mesh", None) and t.slots:                # a Gravity router takes as many as its machines do
            slots = t.slots
        vision = bool(modalities.get("vision"))
        pol = self.policy
        allowed, why = pol.allowed(name) if pol is not None else (True, None)
        return {"name": t.name, "address": t.address, "up": up,
                "kind": "command" if t.is_command else ("provider" if t.provider else ("gravity" if getattr(t, "mesh", None) else "engine")),
                "peer": t.peer, "sandbox": props.get("sandbox") if t.is_command else None,
                "why_down": None if up else (self.not_ready.get(name) or (json.loads(eng.get("/health")[2]).get("status") if t.is_command else None)),
                "model": props.get("model_alias") or t.model, "model_path": props.get("model_path"),
                "slots": pol.slots(name, slots) if pol is not None else slots, "reported_slots": slots,
                "context": eng.n_ctx(), "vision": pol.vision(name, vision) if pol is not None else vision,
                "allowed": allowed, "why_not": why,
                "thinking": t.thinking, "provider": t.provider,
                "host": getattr(t, "host", None) or (t.address.split("//", 1)[-1].split(":")[0] if http else None),
                "power_class": getattr(t, "power_class", None) or ("cloud" if _cloud(t) else "local"),
                "cost_per_mtoken": getattr(t, "cost_per_mtoken", None)}

    def nodes(self) -> list[dict]:
        return [n for n in (self.node(t.name) for t in self.cfg.tiers) if n is not None]

    def offer(self) -> list[dict]:
        """What this machine offers a paired one: its own nodes, not the ones it has from other peers (a peer
        reaches those through their own machine) and never a command tier (a CLI runs under the owner's account
        and stays inside the owner's own Nebula)."""
        return [n for n in (self.node(t.name) for t in self.cfg.tiers if not t.peer and not t.is_command) if n is not None]


def _cloud(t) -> bool:
    """A cloud model: a provider's, or a CLI on your own account (Anti-Gravity, nebula/gravity.py): chained on the map."""
    from nebula import gravity
    return bool(gravity.anti_gravity(t))
