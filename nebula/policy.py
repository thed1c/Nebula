"""nebula/policy.py: this Nebula's own routing policy, set from the page's Configure panels (shift+right-click).

Kept beside the config as `NAME.routing.json` (in memory without a config file), read by the router on every request:

    {"models": {"home-rig": {"enabled": true, "prefer": false, "priority": 5, "slots": 1, "vision": null, "thinking": null,
                             "limits": {"max_concurrent": 1, "max_output_tokens": 2048, "calls_per_hour": 60},
                             "hours": {"days": [0, 1, 2, 3, 4], "from": "08:00", "to": "22:00"}}},
     "advertise": {"<peer uuid>": ["home-rig"]}}

- `models` is keyed by this Nebula's tier names: its own models (stars) and the models it reaches on paired Nebulas
  (remote stars, `<peer>/<tier>`). `enabled` false takes a model off every route; `hours` are when it may be used (local
  time, days 0 = Monday, `to` before `from` runs past midnight); `calls_per_hour` counts the calls this Nebula sent it.
  A model that is off, out of its hours or over its hourly count is left off the ladder, and a request that names it
  is refused with the reason.
- The ladder is ordered `prefer` first, then `priority` (1 first, 9 last, 5 by default), else the config's order.
- `slots` (capacity) and `limits.max_concurrent` narrow the model's gate; `vision` overrides what its engine reports;
  `thinking` overrides its tier's thinking; `limits.max_output_tokens` caps what a call may ask of it.
- `advertise` is what each wormhole carries: per paired Nebula (its uuid), the names of this Nebula's own models it
  is offered. No entry offers everything; an empty list offers nothing. A paired Nebula is offered only those, and its
  calls to anything else are refused.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

INT_FIELDS = {"priority": (1, 9), "slots": (1, 256)}
LIMIT_FIELDS = {"max_concurrent": (1, 256), "max_output_tokens": (1, 10_000_000), "calls_per_hour": (1, 1_000_000)}
CAPS = ("vision", "tools", "structured", "long_context")
CLASSES = ("interactive", "agent", "background")
STAGES = ("everyday", "larger", "writing", "critique", "capable")   # "capable": the older name of larger
ROUTES = ("everyday", "larger", "writing", "critique")
HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class PolicyError(ValueError):
    pass


def _check_model(name: str, m) -> dict:
    if not isinstance(m, dict):
        raise PolicyError(f"{name}: a model's policy is an object")
    out = {}
    for k, v in m.items():
        if v is None:
            continue
        if k in ("enabled", "prefer", "vision", "thinking"):
            if not isinstance(v, bool):
                raise PolicyError(f"{name}.{k} is true or false")
            out[k] = v
        elif k in INT_FIELDS:
            lo, hi = INT_FIELDS[k]
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise PolicyError(f"{name}.{k} is a whole number from {lo} to {hi}")
            out[k] = v
        elif k == "limits":
            if not isinstance(v, dict):
                raise PolicyError(f"{name}.limits is an object")
            lim = {}
            for lk, lv in v.items():
                if lv is None:
                    continue
                if lk not in LIMIT_FIELDS:
                    raise PolicyError(f"{name}.limits has no {lk!r}")
                lo, hi = LIMIT_FIELDS[lk]
                if isinstance(lv, bool) or not isinstance(lv, int) or not lo <= lv <= hi:
                    raise PolicyError(f"{name}.limits.{lk} is a whole number from {lo} to {hi}")
                lim[lk] = lv
            if lim:
                out["limits"] = lim
        elif k == "hours":
            if not isinstance(v, dict):
                raise PolicyError(f"{name}.hours is an object")
            days = v.get("days", list(range(7)))
            if not isinstance(days, list) or not all(isinstance(d, int) and not isinstance(d, bool) and 0 <= d <= 6 for d in days):
                raise PolicyError(f"{name}.hours.days are 0 (Monday) to 6 (Sunday)")
            frm, to = v.get("from", "00:00"), v.get("to", "24:00")
            if not (isinstance(frm, str) and HHMM.match(frm)) or not (isinstance(to, str) and (to == "24:00" or HHMM.match(to))):
                raise PolicyError(f"{name}.hours.from and .to are HH:MM")
            out["hours"] = {"days": sorted(set(days)), "from": frm, "to": to}
        elif k == "eligible":                           # what the model is used for: a need it is not eligible for skips it
            if not isinstance(v, dict) or any(c not in CAPS or not isinstance(b, bool) for c, b in v.items()):
                raise PolicyError(f"{name}.eligible is {{{', '.join(CAPS)}: true or false}}")
            if v:
                out["eligible"] = dict(v)
        elif k == "queue":                              # its own queue: how many may wait, per class, and for how long
            if not isinstance(v, dict) or any(q not in ("max_queued", "timeout_s") for q in v):
                raise PolicyError(f"{name}.queue is {{max_queued: {{class: n}}, timeout_s}}")
            q = {}
            mq = v.get("max_queued") or {}
            if not isinstance(mq, dict) or any(c not in CLASSES or isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 64
                                               for c, n in mq.items()):
                raise PolicyError(f"{name}.queue.max_queued is {{interactive|agent|background: 0 to 64}}")
            if mq:
                q["max_queued"] = dict(mq)
            if v.get("timeout_s") is not None:
                t = v["timeout_s"]
                if isinstance(t, bool) or not isinstance(t, (int, float)) or not 5 <= t <= 3600:
                    raise PolicyError(f"{name}.queue.timeout_s is 5 to 3600 seconds")
                q["timeout_s"] = t
            if q:
                out["queue"] = q
        elif k == "wait_s":                             # how long a call may wait for this model before a substitute
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 3600:
                raise PolicyError(f"{name}.wait_s is 0 to 3600 seconds")
            out["wait_s"] = v
        elif k == "stage":                              # the stage router's tier (nebula/stage.py)
            if v not in STAGES:
                raise PolicyError(f"{name}.stage is one of everyday, larger, writing, critique")
            out["stage"] = "larger" if v == "capable" else v
        elif k == "expect_model":                       # readiness: the loaded file's name must contain this
            if not isinstance(v, str) or len(v) > 160:
                raise PolicyError(f"{name}.expect_model is a part of the model file's name")
            if v.strip():
                out["expect_model"] = v.strip()
        elif k == "schema":                             # a typed destination: its answer must match this JSON schema
            if not isinstance(v, dict) or len(json.dumps(v)) > 20000:
                raise PolicyError(f"{name}.schema is a JSON schema (an object)")
            if v:
                out["schema"] = v
        elif k == "explicit":                           # used only when a request names it: never a fallback
            if not isinstance(v, bool):
                raise PolicyError(f"{name}.explicit is true or false")
            out["explicit"] = v
        else:
            raise PolicyError(f"{name} has no {k!r}")
    return out


def check(data) -> dict:
    """The policy, validated and with nothing but its known parts; raises PolicyError."""
    if not isinstance(data, dict):
        raise PolicyError("the policy is an object")
    models = data.get("models") or {}
    adv = data.get("advertise") or {}
    if not isinstance(models, dict) or not isinstance(adv, dict):
        raise PolicyError('"models" and "advertise" are objects')
    out_models = {}
    for name, m in models.items():
        c = _check_model(str(name), m)
        if c:
            out_models[str(name)] = c
    out_adv = {}
    for peer, names in adv.items():
        if names is None:
            continue
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise PolicyError(f"advertise.{peer} is a list of model names")
        out_adv[str(peer)] = list(dict.fromkeys(names))
    return {"models": out_models, "advertise": out_adv}


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


class Policy:
    def __init__(self, path: str | None = None, clock=time.time):
        self.path = Path(path) if path else None
        self.clock = clock
        self.lock = threading.Lock()
        self.data = {"models": {}, "advertise": {}}
        self.calls: dict[str, deque] = defaultdict(deque)
        if self.path is not None and self.path.exists():
            try:
                self.data = check(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as e:
                raise PolicyError(f"{self.path}: {e}") from None

    # --- reading and writing
    def get(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.data))

    def put(self, data) -> dict:
        clean = check(data)
        with self.lock:
            self.data = clean
        if self.path is not None:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        return clean

    def model(self, name: str) -> dict:
        with self.lock:
            return dict(self.data["models"].get(name) or {})

    # --- what the router asks
    def allowed(self, name: str, now: float | None = None) -> tuple[bool, str | None]:
        """Whether the model may be used now, and if not, why."""
        m = self.model(name)
        if m.get("enabled") is False:
            return False, "disabled"
        now = self.clock() if now is None else now
        h = m.get("hours")
        if h:
            lt = time.localtime(now)
            mins, frm, to = lt.tm_hour * 60 + lt.tm_min, _minutes(h["from"]), _minutes(h["to"])
            day_ok = lt.tm_wday in h["days"]
            in_window = frm <= mins < to if frm < to else (mins >= frm or mins < to) if frm > to else True
            if frm > to and mins < to:                      # past midnight: the window opened the day before
                day_ok = (lt.tm_wday - 1) % 7 in h["days"]
            if not (day_ok and in_window):
                return False, "outside its hours"
        cap = (m.get("limits") or {}).get("calls_per_hour")
        if cap:
            with self.lock:
                q = self.calls[name]
                while q and q[0] < now - 3600:
                    q.popleft()
                if len(q) >= cap:
                    return False, f"its hourly limit ({cap})"
        return True, None

    def order(self, tiers: list) -> list:
        """The tiers, preferred first, then by priority (1 first); otherwise in the order given."""
        def key(it):
            i, t = it
            m = self.model(t.name)
            return (0 if m.get("prefer") else 1, m.get("priority", 5), i)
        return [t for _i, t in sorted(enumerate(tiers), key=key)]

    def slots(self, name: str, reported):
        m = self.model(name)
        caps = [c for c in (m.get("slots"), (m.get("limits") or {}).get("max_concurrent")) if c]
        if not caps:
            return reported
        return min(caps + ([reported] if isinstance(reported, int) and reported > 0 and not m.get("slots") else []))

    def vision(self, name: str, reported: bool) -> bool:
        v = self.model(name).get("vision")
        return reported if v is None else v

    def thinking(self, name: str):
        return self.model(name).get("thinking")

    def eligible(self, name: str, cap: str, default: bool = True) -> bool:
        return bool(self.model(name).get("eligible", {}).get(cap, default))

    def queue(self, name: str) -> dict:
        return dict(self.model(name).get("queue") or {})

    def wait_s(self, name: str):
        return self.model(name).get("wait_s")

    def stage(self, name: str) -> str:
        s = self.model(name).get("stage") or "everyday"
        return "larger" if s == "capable" else s

    def expect_model(self, name: str):
        return self.model(name).get("expect_model")

    def schema(self, name: str):
        return self.model(name).get("schema")

    def explicit(self, name: str) -> bool:
        return bool(self.model(name).get("explicit"))

    def max_output(self, name: str):
        return (self.model(name).get("limits") or {}).get("max_output_tokens")

    def note_call(self, name: str, now: float | None = None) -> None:
        with self.lock:
            self.calls[name].append(self.clock() if now is None else now)

    # --- what the wormholes carry
    def advertised(self, peer: str | None, names: list[str]) -> list[str]:
        """Of this Nebula's own model names, those offered to the paired Nebula `peer` (its uuid)."""
        with self.lock:
            allow = self.data["advertise"].get(peer) if peer else None
        return list(names) if allow is None else [n for n in names if n in allow]
