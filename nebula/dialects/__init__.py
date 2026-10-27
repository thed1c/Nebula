"""nebula/dialects: one module per API a client speaks. Each handler takes (engine, context, request) and
returns a JsonReply or an SseReply; the server writes it. An engine or request error is raised, and the server
answers it in the dialect's own error shape (see nebula/server.py). Each handler records its request in the ledger
through nebula/record.py: the rows before the engine is called, the reply after."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Context:
    cfg: Any
    ledger: Any = None
    headers: Any = field(default_factory=dict)
    engines: Any = field(default_factory=dict)   # tier name -> Engine
    tools: Any = None                            # cockpit.agents.Tools, the MCP hub
    input: Any = None                            # the input's name when it is not a dialect ("hook:<name>")
    registry: Any = None                         # cockpit.registry.Registry
    ticks: Any = None                            # an input's own ticks (a hook's or schedule's "models"), if any
    priority: Any = None                         # an input's own class at the gates ("priority" on a hook or schedule)
    admission: Any = None                        # cockpit.admission.Admission
    lot: Any = None                              # cockpit.parking.Lot
    policy: Any = None                           # cockpit.policy.Policy, this Nebula's own routing policy
    peer: Any = None                             # the paired Nebula asking (its uuid), when one is
    mode: Any = None                             # "passthrough": the named model as it is, no escalation, overflow or retry
    caller: Any = None                           # {"id", "label"}: the registered caller and its origin label (nebula/callers.py)
    slots: Any = None                            # nebula.tenancy.Slots: each tenant's slot on a shared engine
    power: Any = None                            # nebula.power.Power: sleeping machines, woken for interactive calls
    yard: Any = None                             # nebula.engines.Engines, the engines this Nebula runs (draining ones are skipped)
    seclog: Any = None                           # nebula.seclog.SecLog: the security log (who was refused, what changed who may)
    guard: Any = None                            # nebula.guard.Guard: Halo's holds on guests (paused, limited, a model held back)
    otel: Any = None                             # nebula.otel.Exporter: a span per turn (metadata only), when turned on
    reply_headers: Any = field(default_factory=dict)   # what the turn tells the caller beside the answer (X-Nebula-Budget, …)


@dataclass
class JsonReply:
    status: int
    body: dict


@dataclass
class SseReply:
    frames: Iterator[bytes]
    headers: dict = field(default_factory=dict)   # extra response headers (the page learns its session id this way)


def framed(events, name_of, data_of):
    """SSE frames from an event generator; closing the frames closes the generator, so its finally runs
    (the recorder marks the reply interrupted) when the client goes away."""
    try:
        for e in events:
            yield sse_frame(name_of(e), data_of(e))
    finally:
        events.close()


def sse_frame(event: str | None, data) -> bytes:
    head = f"event: {event}\n".encode("utf-8") if event else b""
    return head + b"data: " + json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n\n"


def requested_model(req: dict, cfg) -> str:
    """The name the answer echoes: what the client asked for, else the first tier's."""
    name = req.get("model")
    return name if isinstance(name, str) and name else cfg.tiers[0].name
