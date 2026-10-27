"""nebula/power.py: machines that sleep, and waking them for the calls that may (its own Switchyard: power).

The config's `hosts` names the machines a model can be on and how to wake each one:

    {"hosts": [{"name": "second-box", "wake": {"kind": "wol", "mac": "aa:bb:cc:dd:ee:ff", "broadcast": "192.0.2.255"},
                "wake_seconds": 120, "interactive_may_wake": true}]}

A tier is on a host by its `host` field. A host is **awake** when any of its tiers answers. When the models a request
may use are all asleep:
- an **interactive** call (the room, the API dialects; a caller's class) on a host where `interactive_may_wake` is on
  wakes it (a Wake-on-LAN packet, or the host's own `command`) and waits, visibly, up to `wake_seconds` for one of
  its tiers to answer; the decision records `woke:<host>`;
- an **agent** or **background** call never wakes a machine: it uses an awake model it may use, or waits in the queue
  or the parking lot, or is refused with the reason ("asleep: <host>").

Waking is also a button on the page (Engines › Hosts), and `POST /api/yard/hosts/<name>/wake`.
"""
from __future__ import annotations

import socket
import subprocess
import threading
import time

KINDS = ("wol", "command")


class HostError(ValueError):
    pass


def check(row: dict) -> dict:
    if not isinstance(row, dict) or not str(row.get("name") or "").strip():
        raise HostError('a host is an object with a "name"')
    bad = [k for k in row if k not in ("name", "wake", "wake_seconds", "interactive_may_wake", "note")]
    if bad:
        raise HostError(f"host {row['name']}: unknown field {bad[0]!r}")
    out = {"name": str(row["name"]).strip(), "interactive_may_wake": row.get("interactive_may_wake", True) is not False,
           "wake_seconds": float(row.get("wake_seconds") or 120)}
    w = row.get("wake")
    if w:
        if not isinstance(w, dict) or w.get("kind") not in KINDS:
            raise HostError(f"host {out['name']}: wake is {{\"kind\": \"wol\", \"mac\"}} or {{\"kind\": \"command\", \"command\"}}")
        if w["kind"] == "wol":
            mac = str(w.get("mac") or "").replace("-", ":").lower()
            if len(mac.split(":")) != 6 or not all(len(x) == 2 for x in mac.split(":")):
                raise HostError(f"host {out['name']}: a MAC address like aa:bb:cc:dd:ee:ff")
            out["wake"] = {"kind": "wol", "mac": mac, "broadcast": w.get("broadcast") or "255.255.255.255", "port": int(w.get("port") or 9)}
        else:
            if not w.get("command"):
                raise HostError(f"host {out['name']}: a wake command")
            out["wake"] = {"kind": "command", "command": w["command"]}
    if row.get("note"):
        out["note"] = str(row["note"])[:200]
    return out


def magic_packet(mac: str) -> bytes:
    return b"\xff" * 6 + bytes.fromhex(mac.replace(":", "")) * 16


class Power:
    def __init__(self, cfg, registry):
        self.cfg, self.registry = cfg, registry
        self.woken: dict[str, float] = {}               # host -> when the last wake was sent
        self.lock = threading.Lock()
        self.procs: list = []

    def hosts(self) -> list[dict]:
        return [check(h) for h in getattr(self.cfg, "hosts", None) or []]

    def host(self, name: str | None) -> dict | None:
        return next((h for h in self.hosts() if h["name"] == name), None) if name else None

    def tiers_on(self, name: str) -> list:
        return [t for t in self.cfg.tiers if getattr(t, "host", None) == name]

    def awake(self, name: str, force: bool = False) -> bool:
        return any(self.registry.up(t.name, force=force) for t in self.tiers_on(name)) if self.registry is not None else False

    def wake(self, name: str) -> dict:
        h = self.host(name)
        if h is None:
            raise HostError(f"no host {name!r}")
        w = h.get("wake")
        if not w:
            raise HostError(f"host {name} has no way to be woken (add a Wake-on-LAN MAC or a command)")
        with self.lock:
            last = self.woken.get(name, 0)
            if time.time() - last < 5:                   # one wake at a time: the others wait for it
                return {"host": name, "sent": False, "why": "a wake was sent a moment ago"}
            self.woken[name] = time.time()
        if w["kind"] == "wol":
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                s.sendto(magic_packet(w["mac"]), (w["broadcast"], w["port"]))
        else:
            cmd = w["command"]
            self.procs.append(subprocess.Popen(cmd if isinstance(cmd, list) else str(cmd), shell=not isinstance(cmd, list),
                                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                               start_new_session=True))
        return {"host": name, "sent": True, "kind": w["kind"]}

    def wait_awake(self, name: str, seconds: float, poll: float = 0.5) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            if self.awake(name, force=True):
                return True
            time.sleep(poll)
        return False

    def wake_for(self, ladder: list, interactive: bool) -> tuple[str | None, str | None]:
        """(the host woken and now awake, else None; the reason none was). Only for an interactive call."""
        asleep = []
        for t in ladder:
            h = self.host(getattr(t, "host", None))
            if h is not None and h.get("wake") and h["name"] not in asleep and not self.awake(h["name"]):
                asleep.append(h["name"])
        if not asleep:
            return None, None
        if not interactive:
            return None, "asleep: " + ", ".join(asleep) + " (only an interactive call wakes a machine)"
        for name in asleep:
            h = self.host(name)
            if not h["interactive_may_wake"]:
                continue
            self.wake(name)
            if self.wait_awake(name, h["wake_seconds"]):
                return name, None
            return None, f"{name} did not wake within {h['wake_seconds']:.0f} s"
        return None, "asleep: " + ", ".join(asleep) + " (waking for calls is off)"

    def view(self) -> list[dict]:
        out = []
        for h in self.hosts():
            out.append({**h, "awake": self.awake(h["name"]), "models": [t.name for t in self.tiers_on(h["name"])],
                        "last_wake": self.woken.get(h["name"])})
        return out
