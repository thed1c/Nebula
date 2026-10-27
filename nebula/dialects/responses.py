"""nebula/dialects/responses.py: OpenAI's Responses API, stateless, on serve/responses.py (lifted).

serve/responses.py already turns a Responses request into template messages, tools and kwargs and assembles the
output items and typed events from serve.frontend Events; the cockpit converts its messages for the engine, runs
the stream through cockpit.events.Stream, and feeds the Assembler. Codex CLI is the first client, as it is there.

A text format (json_object or json_schema) is passed to the engine as response_format, which enforces it with a
grammar; the Assembler holds the answer until the end and the cockpit releases it as written.
"""
from __future__ import annotations

from nebula._vendor.strata_serve import responses as R
from nebula._vendor.strata_serve.frontend import tool_choice_of

from nebula.dialects import JsonReply, SseReply, framed, requested_model
from nebula.events import engine_request, template_to_openai, template_tools_to_openai
from nebula.turn import Turn


def responses(engine, ctx, req: dict):
    cfg = ctx.cfg
    R.check_request(req)
    model = requested_model(req, cfg)
    msgs = R.input_messages(req)
    tools, names, _skipped = R.request_tools(req)
    kw = R.template_kwargs(req, {})
    fmt = R.text_format(req)
    kind, name = tool_choice_of(req.get("tool_choice"))
    choice = None
    if tools and kind == "required":
        choice = "required"
    elif tools and kind == "named":
        choice = {"type": "function", "function": {"name": name}}
    openai_messages, openai_tools = template_to_openai(msgs), template_tools_to_openai(tools)
    body = engine_request(openai_messages, openai_tools, kw,
                          max_tokens=req.get("max_output_tokens"), temperature=req.get("temperature"),
                          top_p=req.get("top_p"), response_format=fmt, tool_choice=choice)
    encrypted = "reasoning.encrypted_content" in (req.get("include") or [])
    asm = R.Assembler(req, model, 0, names, encrypted, json_mode=fmt is not None)
    thinking = False if kw.get("enable_thinking") is False else (True if kw.get("reasoning_effort") else None)
    turn = Turn(ctx, "responses", req)
    engine, body = turn.prepare(body, messages=openai_messages, tools=openai_tools, requested_model=model,
                                thinking_requested=thinking)

    def events():
        it = iter(turn.run(engine, body))
        try:
            first = next(it)                 # an engine error is raised here, before response.created went out
        except StopIteration:
            first = None
        yield from asm.start()
        if first is not None:
            yield from asm.feed(first)
        for ev in it:
            yield from asm.feed(ev)
        d = turn.done
        done = {"finish": "length" if d.finish == "length" else "stop", "completion_tokens": d.completion_tokens,
                "prompt_tokens": d.prompt_tokens, "reused": d.cache_tokens, "reasoning_tokens": 0}
        yield from asm.finish(done, lambda text, finish: text)

    if not req.get("stream"):
        return JsonReply(200, R.collect(events()))
    return SseReply(framed(events(), lambda e: e["type"], lambda e: e))
