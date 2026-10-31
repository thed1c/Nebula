"""nebula/route_settings.py: each route's own settings, today its instructions (a system prompt of its own).

    "rules": {"routes": {"writing": {"instructions": "You write fiction. Write in full scenes."},
                         "critique": {"instructions": "Be specific: quote the line, then say what to change."}}}

The routes are smart routing's (nebula/policy.py ROUTES): everyday, larger, writing, critique. A request sent to a route
has that route's instructions put first in its system prompt, before an agent's and before the app's own (which are
kept). The same text always leads, so an engine's prompt cache keeps holding it.

Which route a request is on: the one a rule sent it to ("then": {"to": "writing"}), else the one smart routing sent it
to (live), else the route of the model that answers (its `stage` in the routing policy, everyday when none). A caller
in passthrough mode gets none: its request goes to the model as it is.

Other keys in a route's object (an adapter, for one) are kept as they are, for whoever reads them.
"""
from __future__ import annotations

ROUTES = ("everyday", "larger", "writing", "critique")
MAX_INSTRUCTIONS = 20000                            # characters, per route


class RouteError(ValueError):
    pass


def check(routes) -> dict:
    """The routes section, checked and normalised (the Routes tab's Apply). An empty instructions text is dropped, and
    a route left with nothing is dropped; other keys pass as they are."""
    if not isinstance(routes, dict):
        raise RouteError("routes is {route: its settings}")
    out = {}
    for name, d in routes.items():
        if name not in ROUTES:
            raise RouteError(f"a route is one of {', '.join(ROUTES)}")
        if not isinstance(d, dict):
            raise RouteError(f"routes.{name} is an object")
        d = dict(d)
        text = d.pop("instructions", None)
        if text is not None and not isinstance(text, str):
            raise RouteError(f"routes.{name}.instructions is text")
        text = (text or "").strip()
        if len(text) > MAX_INSTRUCTIONS:
            raise RouteError(f"routes.{name}.instructions is at most {MAX_INSTRUCTIONS} characters")
        if text:
            d["instructions"] = text
        if d:
            out[name] = d
    return out


def instructions(rules, route: str | None) -> str:
    routes = getattr(rules, "routes", None) or {}
    d = routes.get(route) if route else None
    return (d.get("instructions") or "") if isinstance(d, dict) else ""


def longest(rules) -> str:
    """The longest instructions of any route: the request is sized before its route is known."""
    texts = [instructions(rules, r) for r in ROUTES]
    return max(texts, key=len) if texts else ""


def route_of(turn, tier_name: str | None) -> str | None:
    """The route this turn is on (see the module's words), or None for a passthrough caller."""
    if getattr(turn, "passthrough", False):
        return None
    routed = getattr(turn, "routed", None)
    if routed and routed.get("moved") and routed.get("then", {}).get("to") in ROUTES:
        return routed["then"]["to"]
    st = getattr(turn, "stage", None)
    if st and st.get("mode") == "live" and st.get("picked") in ROUTES:
        return st["picked"]
    policy = getattr(turn, "policy", None)
    if policy is not None and tier_name:
        s = policy.stage(tier_name)
        return s if s in ROUTES else "everyday"
    return "everyday"


def apply(body: dict, text: str) -> None:
    """The route's instructions first in the system prompt (a system message of their own when there is none)."""
    if not text:
        return
    messages = list(body.get("messages") or [])
    if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
        messages[0] = dict(messages[0], content=text + "\n\n" + messages[0]["content"])
    elif messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), list):
        messages[0] = dict(messages[0], content=[{"type": "text", "text": text + "\n\n"}] + list(messages[0]["content"]))
    else:
        messages.insert(0, {"role": "system", "content": text})
    body["messages"] = messages
