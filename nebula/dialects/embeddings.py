"""nebula/dialects/embeddings.py: POST /v1/embeddings, pinned (the Switchyard's rule: embeddings never fall back).

Vectors from one model do not match another's, so the request goes to the model it names and nowhere else: no ladder,
no overflow, no retry on another model. The same doors stand as for a chat: the policy (off, hours, hourly count), a
guest only to models shared with guests, a model Halo held back from a guest.
"""
from __future__ import annotations

import json

from nebula import names
from nebula.dialects import JsonReply
from nebula.engine import EngineError


def embeddings(engine, ctx, req: dict):
    from nebula import tenancy
    cfg = ctx.cfg
    name = names.bare_model(req.get("model")) if isinstance(req.get("model"), str) else None
    if not name:
        return JsonReply(400, {"error": {"message": "name the embedding model: embeddings never move to another model",
                                         "type": "invalid_request_error"}})
    t = cfg.tier(name)
    if t is None or ctx.engines.get(name) is None:
        return JsonReply(404, {"error": {"message": f"no model {name!r} here", "type": "invalid_request_error"}})
    tenant, guest = tenancy.tenant_of(ctx.caller if (ctx.caller or {}).get("id") else None, ctx.peer)
    if guest and not getattr(t, "shared", False):
        return JsonReply(403, {"error": {"message": f"{name} is not shared with guests", "type": "invalid_request_error"}})
    if guest and ctx.guard is not None and name in ctx.guard.denied(tenant):
        return JsonReply(403, {"error": {"message": f"{name} is held back from you for now", "type": "invalid_request_error"}})
    if ctx.ticks and name not in ctx.ticks:
        return JsonReply(404, {"error": {"message": f"{name!r} is not offered here", "type": "invalid_request_error"}})
    if ctx.policy is not None:
        ok, why = ctx.policy.allowed(name)
        if not ok:
            return JsonReply(503, {"error": {"message": f"{name} is not to be used now: {why}", "type": "server_error"}})
        ctx.policy.note_call(name)
    body = dict(req, model=t.model or req.get("model"))
    eng = ctx.engines[name]
    try:
        r = eng._open("/v1/embeddings", body)
        out = json.loads(r.read())
    except EngineError as e:
        return JsonReply(e.status, e.body if isinstance(e.body, dict) else {"error": {"message": str(e.body), "type": "server_error"}})
    if isinstance(out, dict):
        out["model"] = name
    ctx.reply_headers["X-Nebula-Route"] = f"{name} (pinned: embeddings never move)"
    return JsonReply(200, out)
