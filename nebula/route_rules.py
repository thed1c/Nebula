"""nebula/route_rules.py: Custom routing's own rules: "if this is / is not that, then send it there".

    "rules": {"route_rules": [
      {"name": "pictures to Gemma", "if": [{"what": "pictures", "is": true}], "then": {"to": "gemma"}},
      {"name": "night: everyday", "if": [{"what": "hours", "is": true, "value": "22:00-07:00"}], "then": {"to": "everyday"}},
      {"name": "no guest code", "if": [{"what": "guest", "is": true}, {"what": "code", "is": true}],
       "then": {"refuse": "code questions are for the owner"}}]}

Each rule's conditions must all hold ("is": false turns one into "is not"); the rules are tried in order and the first
that holds decides; a request no rule matches goes on to smart routing (nebula/stage.py) as before. A model the request
named itself keeps its place: a rule may still refuse it. A rule can be switched off ("off": true).

The conditions (`what`, and the `value` some take):
  app <glob>          the caller's label (X-Nebula-Caller or the app's key name), e.g. "writer-*"
  caller <id>         the caller (nebula/callers.py)
  guest               a guest asks (a friend, a guest key)
  pictures / tools / code / asked_again
                      the request has pictures; offers tools; has a code block; is the same turn asked again
  long <tokens>       the prompt is longer than that
  words <a, b, c>     the last question has one of the words (any case)
  hours <HH:MM-HH:MM> the time here is in that span (it may cross midnight)
  named <model>       the request named that model

The ends (`then`): {"to": <a model> | "everyday" | "larger" | "writing" | "critique"} puts those first and leaves the rest as fallbacks;
{"only": <a model>} uses that model or nothing; {"refuse": <the message>} answers 403 with it.
"""
from __future__ import annotations

import fnmatch
import re
import time

WHATS = {"app": "glob", "caller": "str", "guest": None, "pictures": None, "tools": None, "code": None, "asked_again": None,
         "long": "int", "words": "str", "hours": "span", "named": "str"}
SPAN = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")


class RuleError(ValueError):
    pass


def check(rules) -> list[dict]:
    """The rules, checked and normalised (the page's Apply, and nebula/ui.py's check_rules)."""
    if not isinstance(rules, list):
        raise RuleError("route_rules is a list of rules")
    out = []
    for i, r in enumerate(rules, 1):
        if not isinstance(r, dict):
            raise RuleError(f"rule {i} is an object")
        conds = r.get("if") or []
        if not isinstance(conds, list) or not conds:
            raise RuleError(f"rule {i}: at least one condition (if)")
        cs = []
        for c in conds:
            what = c.get("what") if isinstance(c, dict) else None
            if what not in WHATS:
                raise RuleError(f"rule {i}: a condition is one of {', '.join(WHATS)}")
            kind, v = WHATS[what], c.get("value")
            if kind == "int":
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    raise RuleError(f"rule {i}: '{what}' takes a number of tokens") from None
            elif kind == "span":
                if not SPAN.match(str(v or "")):
                    raise RuleError(f"rule {i}: 'hours' is HH:MM-HH:MM, e.g. 22:00-07:00")
            elif kind in ("glob", "str"):
                v = str(v or "").strip()
                if not v:
                    raise RuleError(f"rule {i}: '{what}' needs a value")
            else:
                v = None
            cs.append({"what": what, "is": c.get("is", True) is not False, **({"value": v} if v is not None else {})})
        then = r.get("then") if isinstance(r.get("then"), dict) else {}
        ends = [k for k in ("to", "only", "refuse") if then.get(k)]
        if len(ends) != 1:
            raise RuleError(f"rule {i}: then is one of to / only / refuse")
        out.append({"name": str(r.get("name") or f"rule {i}")[:60], "if": cs, "then": {ends[0]: str(then[ends[0]])[:300]},
                    **({"off": True} if r.get("off") else {})})
    return out


def _text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
    return ""


def _in_span(span: str, now: time.struct_time | None = None) -> bool:
    m = SPAN.match(span)
    t = now or time.localtime()
    cur, a, b = t.tm_hour * 60 + t.tm_min, int(m[1]) * 60 + int(m[2]), int(m[3]) * 60 + int(m[4])
    return a <= cur < b if a <= b else (cur >= a or cur < b)


def holds(c: dict, turn, est_tokens: int, state, named: str | None) -> bool:
    what, v = c["what"], c.get("value")
    req = getattr(turn, "req", {}) or {}
    msgs = [m for m in (req.get("messages") or []) if isinstance(m, dict)]
    last = next((_text(m.get("content")) for m in reversed(msgs) if m.get("role") == "user"), "")
    caller = getattr(turn, "caller", None) or {}
    if what == "app":
        hit = fnmatch.fnmatchcase(str(caller.get("label") or ""), v)
    elif what == "caller":
        hit = caller.get("id") == v
    elif what == "guest":
        hit = bool(getattr(turn, "guest", False))
    elif what == "pictures":
        hit = any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image")
                                                              for p in m["content"]) for m in msgs)
    elif what == "tools":
        hit = bool(req.get("tools"))
    elif what == "code":
        hit = last.count("```") >= 2
    elif what == "asked_again":
        hit = bool(getattr(state, "regenerations", 0))
    elif what == "long":
        hit = est_tokens > v
    elif what == "words":
        low = last.lower()
        hit = any(w.strip().lower() and w.strip().lower() in low for w in v.split(","))
    elif what == "hours":
        hit = _in_span(v)
    elif what == "named":
        hit = named == v
    else:
        hit = False
    return hit if c.get("is", True) else not hit


def match(rules: list, turn, est_tokens: int, state, named: str | None) -> dict | None:
    for r in rules or []:
        if r.get("off"):
            continue
        if all(holds(c, turn, est_tokens, state, named) for c in r.get("if") or []):
            return r
    return None


def apply(turn, est_tokens: int, state, named: str | None) -> dict | None:
    """The first rule that holds, applied to turn.ladder ({"rule", "then", "moved"}), or None. A refusal is the caller's
    to raise (the turn's own refusal path records it)."""
    rules = list(getattr(turn.rules, "route_rules", None) or [])
    r = match(rules, turn, est_tokens, state, named)
    if r is None:
        return None
    then, out = r["then"], {"rule": r["name"], "then": r["then"], "moved": False}
    if "refuse" in then or turn.pinned or not turn.ladder:
        return out
    target = then.get("to") or then.get("only")
    if target in ("larger", "everyday", "writing", "critique") and turn.policy is not None:
        want = target
        first = [t for t in turn.ladder if turn.policy.stage(t.name) == want]
        if first:
            turn.ladder = first + [t for t in turn.ladder if t not in first]
            out["moved"] = True
    else:
        hit = [t for t in turn.ladder if t.name == target]
        if hit:
            turn.ladder = hit if "only" in then else hit + [t for t in turn.ladder if t.name != target]
            out["moved"] = True
        elif "only" in then:
            out["missing"] = target                     # its model is not on this request's list: refused below
    return out
