"""nebula/otel.py: OpenTelemetry traces, optional and off by default (docs/otel.md).

    "otel": {"enabled": false, "endpoint": "http://127.0.0.1:4318/v1/traces", "headers": {}, "service_name": "nebula"}

One span per turn, sent over OTLP/HTTP (JSON) to the collector you name, in the GenAI semantic conventions' words
where they have one. **Metadata only:** which input, tenant and class, which model and engine, the tokens, the wait at
the gates, how long it took, how it ended, the tool names a turn ran and its escalations. **Never** a prompt, a reply,
a system prompt, a tool's arguments or result, a key, or an error's text (only its status).

The endpoint is https, or http on this machine (a collector beside Nebula): metadata does not cross the network in
the clear. Spans are batched and sent from one background thread; a collector that is down loses spans, never a turn.
"""
from __future__ import annotations

import json
import logging
import queue
import secrets
import threading
import time
import urllib.parse
import urllib.request

from nebula import __version__

LOG = logging.getLogger("nebula.otel")
BATCH, EVERY_S = 64, 5.0


class OtelError(ValueError):
    pass


def check_endpoint(url: str) -> str:
    u = urllib.parse.urlparse(str(url or ""))
    if u.scheme == "https" and u.hostname:
        return url
    if u.scheme == "http" and (u.hostname in ("localhost", "::1") or str(u.hostname or "").startswith("127.")):
        return url
    raise OtelError("the collector's address is https, or http on this machine (127.0.0.1): metadata never crosses the network in the clear")


def settings(cfg) -> dict:
    o = cfg.raw.get("otel") if isinstance(cfg.raw.get("otel"), dict) else {}
    return {"enabled": bool(o.get("enabled")), "endpoint": o.get("endpoint") or "http://127.0.0.1:4318/v1/traces",
            "headers": dict(o.get("headers") or {}), "service_name": o.get("service_name") or "nebula"}


def _attr(k: str, v) -> dict | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return {"key": k, "value": {"boolValue": v}}
    if isinstance(v, int):
        return {"key": k, "value": {"intValue": str(v)}}
    if isinstance(v, float):
        return {"key": k, "value": {"doubleValue": v}}
    if isinstance(v, (list, tuple)):
        return {"key": k, "value": {"arrayValue": {"values": [{"stringValue": str(x)} for x in v]}}}
    return {"key": k, "value": {"stringValue": str(v)[:200]}}


class Exporter:
    def __init__(self, cfg, node_id: str | None = None):
        self.cfg, self.node_id = cfg, node_id
        self.q: queue.Queue = queue.Queue(maxsize=5000)
        self.sent = self.dropped = self.failed = 0
        self.last_error = None
        self.closing = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="otel", daemon=True)
        self.thread.start()

    def on(self) -> bool:
        return settings(self.cfg)["enabled"]

    def span(self, name: str, start: float, end: float, attrs: dict, ok: bool = True, status_code: int | None = None) -> None:
        if not self.on():
            return
        s = {"traceId": secrets.token_hex(16), "spanId": secrets.token_hex(8), "name": name, "kind": 2,
             "startTimeUnixNano": str(int(start * 1e9)), "endTimeUnixNano": str(int(end * 1e9)),
             "attributes": [a for a in (_attr(k, v) for k, v in attrs.items()) if a],
             "status": {"code": 1} if ok else {"code": 2, "message": f"status {status_code}" if status_code else "error"}}
        try:
            self.q.put_nowait(s)
        except queue.Full:
            self.dropped += 1

    def _loop(self):
        while not self.closing.is_set():
            batch, flushed = [], []
            try:
                batch.append(self.q.get(timeout=EVERY_S))
                while len(batch) < BATCH:
                    batch.append(self.q.get_nowait())
            except queue.Empty:
                pass
            flushed = [x for x in batch if isinstance(x, threading.Event)]   # flush() markers: sent once this batch is
            batch = [x for x in batch if not isinstance(x, threading.Event)]
            if batch:
                self._send(batch)
            for ev in flushed:
                ev.set()

    def _send(self, spans: list) -> None:
        st = settings(self.cfg)
        if not st["enabled"]:
            return
        body = {"resourceSpans": [{"resource": {"attributes": [a for a in (
            _attr("service.name", st["service_name"]), _attr("service.version", __version__),
            _attr("service.instance.id", self.node_id)) if a]},
            "scopeSpans": [{"scope": {"name": "nebula", "version": __version__}, "spans": spans}]}]}
        try:
            url = check_endpoint(st["endpoint"])
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json", **{str(k): str(v) for k, v in st["headers"].items()}})
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            self.sent += len(spans)
            self.last_error = None
        except Exception as e:  # noqa: BLE001 - a collector that is down loses spans, never a turn
            self.failed += len(spans)
            self.last_error = f"{type(e).__name__}: {e}"[:200]

    def flush(self, timeout: float = 10.0) -> bool:
        """Send everything queued so far, including a batch the sending thread already holds, and wait for it. The
        marker goes through the same queue, so it is reached only after every span put before it."""
        done = threading.Event()
        self.q.put(done)
        return done.wait(timeout)

    def view(self) -> dict:
        return {**{k: v for k, v in settings(self.cfg).items() if k != "headers"}, "headers": sorted(settings(self.cfg)["headers"]),
                "sent": self.sent, "failed": self.failed, "dropped": self.dropped, "last_error": self.last_error}

    def close(self):
        self.closing.set()


def turn_span(exporter, turn, done, completeness: str, error) -> None:
    """A finished turn as one span: metadata only (the module's promise)."""
    if exporter is None or not exporter.on():
        return
    tier = getattr(turn, "tier", None)
    d = done
    attrs = {"gen_ai.operation.name": "chat", "gen_ai.system": "nebula",
             "gen_ai.request.model": turn.req.get("model") if isinstance(getattr(turn, "req", None), dict) and isinstance(turn.req.get("model"), str) else None,
             "gen_ai.response.model": (d.model if d is not None and d.model else None),
             "gen_ai.usage.input_tokens": d.prompt_tokens if d is not None else None,
             "gen_ai.usage.output_tokens": d.completion_tokens if d is not None else None,
             "gen_ai.response.finish_reasons": [d.finish] if d is not None and d.finish else None,
             "nebula.input": getattr(turn, "dialect", None), "nebula.tier": tier.name if tier is not None else None,
             "nebula.engine": tier.address if tier is not None else None,
             "nebula.tenant": getattr(turn, "tenant", None), "nebula.guest": bool(getattr(turn, "guest", False)),
             "nebula.priority": getattr(turn, "priority", None), "nebula.caller": (getattr(turn, "caller", None) or {}).get("id"),
             "nebula.agent": turn.agent.name if getattr(turn, "agent", None) is not None else None,
             "nebula.cached_tokens": d.cached_tokens if d is not None else None,
             "nebula.queued_s": round(float(getattr(turn, "queued_s", 0) or 0), 3), "nebula.tool_rounds": getattr(turn, "rounds", 0),
             "nebula.retries": getattr(turn, "retries", 0), "nebula.completeness": completeness}
    status = getattr(error, "status", None) if error is not None and not isinstance(error, str) else None
    exporter.span("nebula.turn", getattr(turn, "t0", time.time()), time.time(), attrs, ok=not error and completeness == "complete",
                  status_code=status)
