"""nebula/dialects/chat.py: OpenAI Chat Completions.

The engine already speaks this dialect, so without an agent the request goes as it is and the engine's chunks
come back as they are: the model name the client sent is the one echoed (the engine answers with its own alias),
the engine is always asked for the usage chunk so the ledger sees every request's counts (a client that did not
ask for it does not get it), and the request and its reply are recorded. The router (nebula/turn.py) may send it
to another tier.

With an agent the cockpit runs the turn itself (its system prompt, its tools, its rounds), and the reply's events
are turned back into this dialect's chunks or completion.
"""
from __future__ import annotations

import itertools
import json

from nebula.dialects import JsonReply, SseReply, requested_model, sse_frame
from nebula.engine import EngineError
from nebula.events import Stream, chunks_from, completion_chunks, completion_from
from nebula.turn import Turn


def chat(engine, ctx, req: dict):
    cfg = ctx.cfg
    model = requested_model(req, cfg)
    body = {k: v for k, v in req.items() if k != "model"}
    messages = req.get("messages") if isinstance(req.get("messages"), list) else []
    kw = req.get("chat_template_kwargs") if isinstance(req.get("chat_template_kwargs"), dict) else {}
    thinking = None if "enable_thinking" not in kw else bool(kw["enable_thinking"])
    turn = Turn(ctx, "chat", req)
    engine, body = turn.prepare(body, messages=messages, tools=req.get("tools"), requested_model=model,
                                thinking_requested=thinking)
    options = req.get("stream_options") if isinstance(req.get("stream_options"), dict) else {}
    want_usage = bool(options.get("include_usage"))
    if turn.agent is not None:
        if not req.get("stream"):
            events = list(turn.run(engine, body))
            return JsonReply(200, completion_from(events, turn.done, model))

        def agent_frames():
            for chunk in chunks_from(turn.run(engine, body), model, lambda: turn.done, want_usage):
                yield sse_frame(None, chunk)
            yield b"data: [DONE]\n\n"

        return SseReply(agent_frames())
    gate = turn.admission.gate(turn.tier) if (turn.admission is not None and turn.tier is not None) else None
    schema = turn.policy.schema(turn.tier.name) if (turn.policy is not None and turn.tier is not None) else None
    if schema is not None:                              # a typed destination: it answers whole, and its answer is checked
        if req.get("stream"):
            turn.finish(error="a typed destination answers whole")
            return JsonReply(400, {"error": {"message": f"{turn.tier.name} answers whole (its answer is checked against its schema): "
                                                        "send stream false", "type": "invalid_request_error"}})
        body.setdefault("response_format", {"type": "json_schema", "json_schema": {"name": turn.tier.name, "schema": schema}})
    if not req.get("stream"):
        while True:
            try:
                with turn.slot(gate):
                    obj = engine.chat(body)
                break
            except EngineError as e:
                moved = turn.reroute(e, body)        # nothing was answered: the next eligible model, if any
                if moved is None:
                    turn.finish(error=e.message)
                    raise
                engine, gate = moved
        if schema is not None:
            from nebula.schema import errors
            text = ((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            try:
                bad = errors(json.loads(text), schema)
            except ValueError:
                bad = ["the answer is not JSON"]
            if bad:
                turn.finish(error="the typed answer did not match its schema: " + "; ".join(bad[:5]))
                return JsonReply(502, {"error": {"message": f"{turn.tier.name}'s answer did not match its schema: " + "; ".join(bad[:5]),
                                                 "type": "invalid_response"}})
        stream = Stream(completion_chunks(obj))
        for ev in stream:
            turn.add(ev)
        turn.finish(stream.done)
        obj["model"] = model
        return JsonReply(200, obj)
    body["stream_options"] = dict(options, include_usage=True)

    def opened():
        """The engine's chunks, from a slot held, once its first chunk came: a failure before it may move the
        request to the next eligible model (nothing has been sent to the client yet)."""
        nonlocal engine, gate
        while True:
            slot = turn.slot(gate)
            slot.__enter__()
            chunks = engine.chat_stream(body)
            try:
                head = next(chunks, None)
            except EngineError as e:
                slot.__exit__(None, None, None)
                moved = turn.reroute(e, body)
                if moved is None:
                    raise
                engine, gate = moved
                continue
            return slot, (itertools.chain([head], chunks) if head is not None else iter(()))

    def frames():
        stream = Stream()
        slot = None
        try:
            slot, chunks = opened()                  # entered: its gate is held until the stream ends
            for chunk in chunks:
                for ev in stream.feed(chunk):
                    turn.add(ev)
                if not chunk.get("choices") and "usage" in chunk and not want_usage:
                    continue
                chunk["model"] = model
                yield sse_frame(None, chunk)
            for ev in stream.end():
                turn.add(ev)
            turn.finish(stream.done)
            yield b"data: [DONE]\n\n"
        except EngineError as e:
            turn.finish(error=e.message)
            raise
        finally:
            if slot is not None:
                slot.__exit__(None, None, None)
            turn.finish(stream.done, completeness="interrupted")

    return SseReply(frames())
