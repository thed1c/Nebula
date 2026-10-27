"""nebula/stage.py: smart routing (the stage router): for each request, which route answers it.

    "rules": {"stage_router": {"mode": "off" | "shadow" | "live", "picker": "efficient_first" | "capable_first",
                               "threshold": 0.5, "judge": "<a model>" | "halo" | null}}

**Routes** (a model's `stage` in the routing policy): everyday (the default), larger, writing, critique ("capable" is
the older name of larger). **Halo as the judge** (`"judge": "halo"`, design decision 2026-10-21: "Halo is supposed to be entry
to chat and the smart router"): Halo's own model reads every turn and picks the route among those on its ladder, with
a few words of why; the score below is the fallback when Halo cannot answer. On the page: Off, Practice (shadow: the
pick is recorded, nothing moves), On (live).

Each model is in a tier (`"stage": "everyday" | "capable"` in the routing policy; everyday by default). A request is
scored from what it shows: a long prompt, pictures, tool results and tool errors in this turn, the same turn asked
again, a long or code-heavy question, words that ask for depth. At or over the threshold it wants the capable tier.
When the score is close to the threshold (unsure), the judge model, if one is set, is asked to decide (a short call
on the same Switchyard, the same tenant); without a judge, the picker breaks the tie (efficient_first: everyday).

- **off**: nothing happens (the default).
- **shadow**: the pick is recorded on the decision, nothing changes (the Switchyard's shadow mode, to watch first).
- **live**: the picked tier's models go first on the ladder; a model the request named itself is never moved.
"""
from __future__ import annotations

import json
import re
import time

DEEP = re.compile(r"(?i)\b(prove|proof|debug|architect|architecture|step[- ]by[- ]step|analy[sz]e|refactor|derive|"
                  r"optimi[sz]e|trade-?offs?|edge cases?|rigorous|in depth|thoroughly)\b")
TECH = re.compile(r"(?i)\b(code|bug|error|exception|traceback|python|javascript|sql|function|algorithm|compile|database|"
                  r"api|regex|prove|theorem|equation|calculate|summari[sz]e|contract|trade-?offs?|compare)\b")
CRITIQUE = re.compile(r"(?i)\b(critique|critic|review|feedback|what works|what does ?n[o']t|what doesn't|how can i improve|"
                      r"is this (good|any good)|pacing|line edit)\b")
HALO = None                         # a callable giving Halo's own engine (nebula/halo_engine.py), set by the server
ROUTE_WORDS = {"everyday": "greetings, small talk, quick facts, everyday questions and short simple tasks",
               "larger": "code and debugging, technical or step-by-step explanations, analysis, comparisons, math, long documents",
               "writing": "creative writing to produce: stories, scenes, chapters, dialogue, poems, prose in a voice",
               "critique": "feedback on a piece of creative writing the user gives (a line, a scene, a chapter): what works, what does not"}
JUDGE_SYSTEM = ("You route requests. Decide whether the request below needs a capable (larger, slower) model, or an "
                "everyday model can answer it well. Answer only JSON: {\"capable\": true|false, \"confidence\": 0..1}.")


def settings(rules) -> dict:
    s = dict(getattr(rules, "stage_router", None) or {})
    return {"mode": s.get("mode") if s.get("mode") in ("off", "shadow", "live") else "off",
            "picker": s.get("picker") if s.get("picker") in ("efficient_first", "capable_first") else "efficient_first",
            "threshold": min(1.0, max(0.0, float(s.get("threshold", 0.5)))), "judge": s.get("judge") or None,
            "unsure": min(0.5, max(0.0, float(s.get("unsure", 0.15))))}


def _text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
    return ""


def score(req: dict, est_tokens: int, long_tokens: int, state=None) -> tuple[float, list[str]]:
    """(0..1, the signals): how much this request looks like the capable tier's work."""
    msgs = [m for m in (req.get("messages") or []) if isinstance(m, dict)]
    sig, s = [], 0.0
    tail = []
    for m in reversed(msgs):
        if m.get("role") == "tool":
            tail.append(m)
        else:
            break
    if tail:
        s += 0.15 + 0.05 * min(len(tail), 4)
        sig.append(f"{len(tail)} tool result(s)")
    errs = sum(1 for m in tail if _text(m.get("content")).strip().lower().startswith("error"))
    if errs:
        s += 0.25
        sig.append(f"{errs} tool error(s)")
    if est_tokens > long_tokens:
        s += 0.3
        sig.append("a long prompt")
    if any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image")
                                                     for p in m["content"]) for m in msgs):
        s += 0.1
        sig.append("pictures")
    last = next((_text(m.get("content")) for m in reversed(msgs) if m.get("role") == "user"), "")
    if len(last) > 2000:
        s += 0.1
        sig.append("a long question")
    if last.count("```") >= 2:
        s += 0.1
        sig.append("code")
    if DEEP.search(last):
        s += 0.15
        sig.append("asks for depth")
    if state is not None and getattr(state, "regenerations", 0):
        s += 0.25
        sig.append("asked again")
    return min(1.0, round(s, 3)), sig


def ask_judge(server_engines, cfg, policy, judge: str, req: dict, timeout: float = 15.0) -> dict | None:
    """The judge's verdict ({capable, confidence, model, seconds}), or None when it cannot answer now."""
    eng = server_engines.get(judge) if server_engines else None
    t = cfg.tier(judge)
    if eng is None or t is None:
        return None
    if policy is not None:
        ok, _why = policy.allowed(judge)
        if not ok:
            return None
    last = next((_text(m.get("content")) for m in reversed(req.get("messages") or []) if isinstance(m, dict) and m.get("role") == "user"), "")
    body = {"messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": last[:4000]}],
            "max_tokens": 40, "temperature": 0, "stream": False, **({"model": t.model} if t.model else {}),
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.monotonic()
    try:
        old = eng.timeout_s
        eng.timeout_s = min(old or timeout, timeout)
        try:
            obj = eng.chat(body)
        finally:
            eng.timeout_s = old
        text = ((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        m = re.search(r"\{.*\}", text, re.S)
        d = json.loads(m.group(0)) if m else {}
        if policy is not None:
            policy.note_call(judge)
        return {"model": judge, "capable": bool(d.get("capable")), "confidence": float(d.get("confidence") or 0.5),
                "seconds": round(time.monotonic() - t0, 2)}
    except Exception:  # noqa: BLE001 - an absent judge leaves the score and the picker to decide
        return None


CREATIVE = re.compile(r"(?i)\b(story|stories|scene|chapter|poem|haiku|novel|dialogue|prose|fiction|character|plot|verse|lyrics)\b")
WRITING = re.compile(r"(?i)\b(write|continue|draft|tell|compose|create|make up)\b.*\b(story|scene|chapter|poem|haiku|novel|dialogue|opening|prose|letter|verse|lyrics)\b")


def cue_route(last: str, routes: list[str]) -> str:
    """The route a message's own words point to, when Halo cannot say: feedback on writing, writing, technical, else
    everyday."""
    if "critique" in routes and CRITIQUE.search(last) and re.search(r"[\"'“‘]", last):
        return "critique"
    if "writing" in routes and WRITING.search(last):
        return "writing"
    if "larger" in routes and (DEEP.search(last) or TECH.search(last) or last.count("```") >= 2):
        return "larger"
    return "everyday" if "everyday" in routes else routes[0]


def ask_halo(req: dict, routes: list[str], timeout: float = 12.0) -> dict | None:
    """Halo's pick among these routes ({route, why, seconds}), or None when its own model cannot answer now."""
    own = HALO() if HALO else None
    if own is None or not own.ready():
        return None
    from nebula.engine import Engine
    msgs = [m for m in (req.get("messages") or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    last = next((_text(m.get("content")) for m in reversed(msgs) if m.get("role") == "user"), "")
    before = " | ".join(_text(m.get("content"))[:200] for m in msgs[-4:-1])
    menu = "\n".join(f"- {r}: {ROUTE_WORDS.get(r, r)}" for r in routes)
    system = ("You are Halo, the router. Pick the one route that should answer the user's last message.\nRoutes:\n" + menu +
              "\nCode, bugs and technical questions are never critique: critique is only feedback on creative writing."
              "\nAnswer only JSON: {\"route\": \"<one of: " + ", ".join(routes) + ">\", "
              "\"why\": \"<what the message asks for, at most eight words>\"}")
    user = (("Earlier: " + before + "\n\n") if before else "") + "Last message: " + last[:3000]
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 80,
            "temperature": 0, "stream": False, "model": own.tier.model, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.monotonic()
    try:
        # its server's key of this start (nebula/halo_engine.py): a server that enforces it answers nothing else
        obj = Engine(own.tier.address, timeout, model=own.tier.model, context=own.tier.context,
                     api_key=getattr(own, "api_key", None)).chat(body)
        text = ((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        m = re.search(r"\{.*\}", text, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except ValueError:                                  # half-made JSON: its route and why, as written
            r = re.search(r'"route"\s*:\s*"(\w+)"', text)
            w = re.search(r'"why"\s*:\s*"([^"]{0,200})', text)
            d = {"route": r.group(1) if r else "", "why": w.group(1) if w else ""}
        route = str(d.get("route") or "").strip().lower()
        if route == "capable":
            route = "larger"
        if route not in routes:                             # a route of its own making ("debug"): read by its words
            route, d = cue_route(last, routes), {"why": f"{route or 'no route'}, read as {cue_route(last, routes)}"}
        why = " ".join(str(d.get("why") or "").split()[:10]).rstrip(".,;:") or f"looks like {route} work"
        # Halo is a small model: two plain checks keep its pick honest (both said in its why)
        deep = bool(DEEP.search(last) or TECH.search(last) or last.count("```") >= 2)
        if route == "critique" and not CRITIQUE.search(last):
            route, why = ("larger" if deep and "larger" in routes else "everyday"), why + "; not a critique of writing"
        elif route in ("writing", "everyday") and CRITIQUE.search(last) and "critique" in routes and re.search(r"[\"'“‘]", last):
            route, why = "critique", why + "; asks for feedback on writing"
        elif route == "writing" and not (WRITING.search(last) or CREATIVE.search(last)):
            route = cue_route(last, [r for r in routes if r != "writing"] or routes)
            why += f"; not creative writing, so {route}"
        elif route == "everyday" and deep and "larger" in routes:
            route, why = "larger", why + "; technical or step-by-step"
        return {"route": route, "why": why, "seconds": round(time.monotonic() - t0, 2), "model": "halo"}
    except Exception:  # noqa: BLE001 - Halo out of reach: the score decides
        return None


def ask_jev(req: dict, routes: list[str], timeout: float = 40.0) -> dict | None:
    """TypeSafe's Jev as the judge (off unless chosen in Routes; each judged turn's text goes to TypeSafe and costs): one typed choice among the routes, through the owner's own `jev` client and key."""
    import os
    import shutil
    import subprocess
    import tempfile
    exe = shutil.which("jev") or os.path.join(os.path.expanduser("~"), ".local", "bin", "jev")   # its installer's place
    if not os.path.isfile(exe):
        return None
    last = next((_text(m.get("content")) for m in reversed(req.get("messages") or []) if isinstance(m, dict) and m.get("role") == "user"), "")
    payload = {"model": "jev-latest", "state": {"request": last[:4000]},
               "questions": {"route": {"type": "choice", "instructions": "Pick the route that should answer this request.",
                                       "criteria": {r: ROUTE_WORDS.get(r, r) for r in routes}}}}
    t0 = time.monotonic()
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
        p = subprocess.run([exe, "run", f.name], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        os.unlink(f.name)
        d = json.loads(p.stdout or "{}")
        a = ((d.get("answers") or {}).get("route") or {})
        route = a.get("choice")
        if route not in routes:
            return None
        return {"route": route, "why": f"Jev, confidence {float(a.get('confidence') or 0):.2f}", "model": "jev",
                "seconds": round(time.monotonic() - t0, 2)}
    except Exception:  # noqa: BLE001 - Jev out of reach: the score decides
        return None


def pick(turn, est_tokens: int, state=None) -> dict | None:
    """The stage router's decision for this turn (None when it is off); in live mode it reorders turn.ladder."""
    st = settings(turn.rules)
    if st["mode"] == "off" or turn.policy is None or turn.pinned or turn.passthrough or not turn.ladder:
        return None
    tiers = {t.name: turn.policy.stage(t.name) for t in turn.ladder}
    if len(set(tiers.values())) < 2:                    # one tier only on this ladder: nothing to choose between
        return {"mode": st["mode"], "picked": next(iter(tiers.values())), "why": "one tier only"}
    sc, signals = score(turn.req, est_tokens, turn.rules.long_context_tokens, state)
    if st["judge"] in ("halo", "halo-own", "jev"):       # Halo (or Jev, when chosen) reads every turn and picks the route
        routes = [r for r in ("everyday", "larger", "writing", "critique") if r in set(tiers.values())]
        h = ask_jev(turn.req, routes) if st["judge"] == "jev" else ask_halo(turn.req, routes)
        if h is not None:
            out = {"mode": st["mode"], "picked": h["route"], "score": sc, "threshold": st["threshold"], "signals": signals,
                   "how": "halo" if h["model"] == "halo" else "jev", "judge": h, "why": h["why"]}
            if st["mode"] == "live":
                first = [t for t in turn.ladder if tiers[t.name] == h["route"]]
                turn.ladder = first + [t for t in turn.ladder if tiers[t.name] != h["route"]]
            return out
    unsure = abs(sc - st["threshold"]) <= st["unsure"]
    verdict = None
    if unsure and st["judge"]:
        verdict = ask_judge(turn.engines, turn.cfg, turn.policy, st["judge"], turn.req)
    if verdict is not None:
        picked = "larger" if verdict["capable"] else "everyday"
        how = "judge"
    elif unsure:
        picked = "everyday" if st["picker"] == "efficient_first" else "larger"
        how = st["picker"]
    else:
        picked = "larger" if sc >= st["threshold"] else "everyday"
        how = "score"
    out = {"mode": st["mode"], "picked": picked, "score": sc, "threshold": st["threshold"], "signals": signals, "how": how,
           "judge": verdict}
    if st["mode"] == "live":
        first = [t for t in turn.ladder if tiers[t.name] == picked]
        turn.ladder = first + [t for t in turn.ladder if tiers[t.name] != picked]
    return out
