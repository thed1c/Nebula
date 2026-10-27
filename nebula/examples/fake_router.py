"""A pretend router to watch: serves a status in the documented shape (docs/router-status-v1.md) and a caller
registry, with a call starting and landing every few seconds. No model runs; nothing it is sent is acted on.

    python3 -m nebula.examples.fake_router [--port 18300] [--every 4]
    python3 -m nebula --config nebula/examples/watch-router.json        # then open http://127.0.0.1:18194

GET /switchyard/status   the status
GET /switchyard/registry the callers ({"callers": [...]})
GET /switchyard/plan     its decision plan (docs/plan-v1.md; the example's fake-plan.json)
Anything else is answered 404 and counted, so a test can check the watcher only reads.
"""
from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CALLERS = [
    {"id": "writer-app", "name": "The writing app", "class": "interactive", "tags": ["writer-app/*"]},
    {"id": "nightly", "name": "Nightly jobs", "class": "batch", "tags": ["job.*"]},
]
CALLS = [
    {"caller": "writer-app/draft", "route": "big", "actor": "writer"},
    {"caller": "job.summaries", "route": "small", "actor": "planner"},
    {"caller": "writer-app/review", "route": "small", "actor": "reviewer"},
]
PLAN = Path(__file__).with_name("fake-plan.json")


def status(calls: list[dict], registry_url: str | None = None, waiting: list | None = None, n: int = 0) -> dict:
    """The status in the documented shape. `calls` are the calls in flight; `n` counts the calls made so far, for the
    decision counters and the latest decisions."""
    busy = {"big": 0, "small": 0}
    for c in calls:
        busy[c["route"]] += 1
    holders = {lane: [{"actor": c["actor"], "route": c["route"], "since": c.get("since")} for c in calls if c["route"] == lane]
               for lane in busy}
    out = {
        "capacity": {
            "destinations": {
                "big": {"label": "Big model", "routes": ["big"], "slots": 1, "busy": busy["big"], "queued": 0,
                        "slot_context": 131072, "avg_call_seconds": 40.0, "unavailable": None},
                "small": {"label": "Small model", "routes": ["small"], "slots": 4, "busy": busy["small"], "queued": 0,
                          "slot_context": 32768, "avg_call_seconds": 6.0, "unavailable": None},
            },
            "parking": [],
        },
        "routes": {"big": {"base_url": "http://127.0.0.1:9001/v1", "ready": True},
                   "small": {"base_url": "http://127.0.0.1:9002/v1", "ready": True}},
        "switchyard": {"actors": {"writer": "live", "planner": "live", "reviewer": "off"}},
        "in_flight_calls": calls,
        "waiting": list(waiting or []),
        "external_destinations": {"destinations": [
            {"id": "cli", "kind": "command", "enabled": True, "status": "ready", "running": 0, "queued": 0}]},
        # optional sections (docs/router-status-v1.md): lanes, policy and decisions
        "admission": {"groups": {lane: {"label": label, "size": size, "busy": busy[lane], "free": size - busy[lane],
                                        "queued": 0, "on_hold": None, "holders": holders[lane]}
                                 for lane, label, size in (("big", "Big model", 1), ("small", "Small model", 4))},
                      "stats": {"waits": n // 3, "timeouts": 0, "longest_seconds": 12.5}},
        "route_policy": {
            "queues": {"big": {"routes": ["big"], "slots": 1, "timeout_seconds": 600, "max_queued": {"background": 4}},
                       "small": {"routes": ["small"], "slots": 4, "timeout_seconds": 120, "max_queued": {}}},
            "routes": {"big": {"enabled": True, "priority": 1, "group": "big", "state": "qualified", "model": "big",
                               "configured_context": 131072, "eligible": {"vision": False, "tools": True, "thinking": True}},
                       "small": {"enabled": True, "priority": 5, "group": "small", "state": "qualified", "model": "small",
                                 "configured_context": 32768, "eligible": {"vision": True, "tools": True, "thinking": False}}},
            "counters": {"redirects": 0, "refused_ineligible": 0, "queue_full": 0, "parking_full": 0}},
    }
    if calls:                                    # the latest decisions, as the router reports them: only the last of each
        c = calls[-1]
        at = c.get("since") or time.time()
        out["capacity"]["mode"] = "live"
        out["capacity"]["counters"] = {"decisions": {"live/pinned": n}, "agree": n, "differ": 0, "parked": 0}
        out["capacity"]["last"] = {"at": at, "kind": "pick", "mode": "live", "caller": c["caller"], "actor": c["actor"],
                                   "class": "interactive", "candidates": ["big", "small"] if c["route"] == "big" else ["small"],
                                   "rank": 0, "pick": c["route"], "selected": c["route"], "actual": c["route"], "agree": True,
                                   "reason": "first ticked model with a free slot", "wait_s": 0.0, "est_tokens": 1800,
                                   "run_key": c.get("run")}
        out["switchyard"].update(mode="live", decisions={f"{c['actor']}/efficient": n}, errors=0,
                                 last={"at": at, "actor": c["actor"], "caller": c["caller"], "mode": "live", "tier": "efficient",
                                       "switchyard_order": [c["route"], "small"], "selected_route": c["route"],
                                       "routed": c["route"], "applied": c["route"], "differs": False, "ms": 0.4})
        out["admission"]["stats"]["last"] = {"id": c.get("run"), "actor": c["actor"], "route": c["route"], "group": c["route"],
                                             "label": "Big model" if c["route"] == "big" else "Small model",
                                             "caller": c["caller"], "class": "interactive", "behind": [], "since": at,
                                             "outcome": "admitted", "seconds": 0.0, "ended_at": at}
    if registry_url:
        out["capacity"]["registry"] = registry_url
    return out


class FakeRouter:
    def __init__(self, port: int = 0, every: float = 4.0, host: str = "127.0.0.1"):
        outer = self
        self.calls: list[dict] = []
        self.n = 0                                       # calls made so far
        self.other: list[tuple[str, str]] = []           # every request that was not a read of its paths
        self.lock = threading.Lock()
        self.every = every
        self.closing = threading.Event()

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/switchyard/status":
                    return self._send(outer.status())
                if self.path == "/switchyard/registry":
                    return self._send({"callers": CALLERS})
                if self.path == "/switchyard/plan":
                    return self._send(json.loads(PLAN.read_text()))
                outer.other.append(("GET", self.path))
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                outer.other.append(("POST", self.path))
                self.send_response(404)
                self.end_headers()

        self.srv = ThreadingHTTPServer((host, port), H)
        self.base = f"http://{host}:{self.srv.server_port}"
        self.url = self.base + "/switchyard/status"

    def status(self) -> dict:
        with self.lock:
            return status([dict(c) for c in self.calls], n=self.n)

    def start(self) -> "FakeRouter":
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        if self.every > 0:
            threading.Thread(target=self._traffic, daemon=True).start()
        return self

    def _traffic(self) -> None:
        for n, c in zip(itertools.count(1), itertools.cycle(CALLS)):
            with self.lock:
                self.calls = [dict(c, run=f"run-{n}", since=time.time())]
                self.n = n
            if self.closing.wait(self.every * 0.75):
                return
            with self.lock:
                self.calls = []
            if self.closing.wait(self.every * 0.25):
                return

    def close(self) -> None:
        self.closing.set()
        self.srv.shutdown()
        self.srv.server_close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="fake_router", description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=18300)
    p.add_argument("--every", type=float, default=4.0, help="seconds between calls")
    a = p.parse_args(argv)
    r = FakeRouter(a.port, a.every).start()
    print(f"fake Switchyard: {r.base}  (watch it with {{\"watch\": {{\"url\": \"{r.base}\"}}}}); Ctrl-C to stop", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        r.close()


if __name__ == "__main__":
    main()
