"""nebula/route_settings.py: each route's own settings: its instructions (a system prompt of its own), its words, its kind.

    "rules": {"routes": {"writing": {"instructions": "You write fiction. Write in full scenes."},
                         "critique": {"instructions": "Be specific: quote the line, then say what to change."},
                         "pictures": {"name": "Pictures", "about": "photos and screenshots to read or describe",
                                      "kind": ["vision"], "instructions": "Describe what you see first."}}}

The routes are smart routing's: the four built in (everyday, larger, writing, critique) and any the owner adds (an id
of lowercase letters, digits and hyphens, with a name). The page shows the built-in ones as suggestions: larger, writing
and critique can be put away (`"hidden": true`), which takes them out of smart routing until they are brought back.
Everyday stays: a model with no route is on it. A model is on one route (its `stage` in the routing policy).
Smart routing (nebula/stage.py) offers Halo every route that has a model, each by its words: its `about`, else the
built-in words, else its name, with what its kind says (pictures, tools, code…). Its `kind`:

- thinking / no_thinking: a request answered on this route thinks (or does not), whatever its model's default;
- vision, tools, long, code, writing, quick: what the route is for, said to Halo with its words.

A request sent to a route has that route's instructions put first in its system prompt, before an agent's and before
the app's own (which are kept). The same text always leads, so an engine's prompt cache keeps holding it.

Which route a request is on: the one a rule sent it to ("then": {"to": "writing"}), else the one smart routing sent it
to (live), else the route of the model that answers (its `stage` in the routing policy, everyday when none). A caller
in passthrough mode gets none: its request goes to the model as it is.

Other keys in a route's object (an adapter, for one) are kept as they are, for whoever reads them.
"""
from __future__ import annotations

import re

ROUTES = ("everyday", "larger", "writing", "critique")       # built in, always there
ID = re.compile(r"^[a-z][a-z0-9-]{1,30}$")                   # a route of the owner's own
MAX_ROUTES = 16
MAX_INSTRUCTIONS = 20000                            # characters, per route
KINDS = {"thinking": "thinks before answering", "no_thinking": "answers without thinking first",
         "vision": "requests with pictures", "tools": "requests that use tools", "long": "long prompts and documents",
         "code": "code", "writing": "creative writing", "quick": "quick answers"}


class RouteError(ValueError):
    pass


def check(routes) -> dict:
    """The routes section, checked and normalised (the Routes tab's Apply). An empty instructions text is dropped, and
    a route left with nothing is dropped; other keys pass as they are."""
    if not isinstance(routes, dict):
        raise RouteError("routes is {route: its settings}")
    out = {}
    if len(routes) > MAX_ROUTES:
        raise RouteError(f"at most {MAX_ROUTES} routes")
    for name, d in routes.items():
        if name not in ROUTES and not (isinstance(name, str) and ID.fullmatch(name)):
            raise RouteError(f"a route's id is lowercase letters, digits and hyphens (2 to 31), not {name!r}")
        if not isinstance(d, dict):
            raise RouteError(f"routes.{name} is an object")
        d = dict(d)
        for k, most in (("name", 40), ("about", 400)):
            v = d.pop(k, None)
            if v is not None and not isinstance(v, str):
                raise RouteError(f"routes.{name}.{k} is text")
            v = " ".join((v or "").split())
            if len(v) > most:
                raise RouteError(f"routes.{name}.{k} is at most {most} characters")
            if v:
                d[k] = v
        if name not in ROUTES and not d.get("name"):
            raise RouteError(f"the new route {name} needs a name")
        hidden = d.pop("hidden", None)
        if hidden not in (None, False, True):
            raise RouteError(f"routes.{name}.hidden is true or false")
        if hidden:
            if name not in ROUTES or name == ROUTES[0]:
                raise RouteError(f"only a suggested route ({', '.join(ROUTES[1:])}) is put away; remove {name} instead"
                                 if name != ROUTES[0] else "everyday stays: a model with no route is on it")
            d["hidden"] = True
        kind = d.pop("kind", None)
        if kind is not None:
            if not isinstance(kind, list) or any(k not in KINDS for k in kind):
                raise RouteError(f"routes.{name}.kind is a list of {', '.join(KINDS)}")
            if "thinking" in kind and "no_thinking" in kind:
                raise RouteError(f"routes.{name}: thinking or no thinking, not both")
            kind = [k for k in KINDS if k in kind]
            if kind:
                d["kind"] = kind
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


def _routes(rules) -> dict:
    r = rules.get("routes") if isinstance(rules, dict) else getattr(rules, "routes", None)
    return r if isinstance(r, dict) else {}


def names(rules) -> list[str]:
    """Every route, in the page's order: the four built in (but those put away), then the owner's own in the order
    they were added."""
    rs = _routes(rules)
    return [r for r in ROUTES if not (isinstance(rs.get(r), dict) and rs[r].get("hidden") and r != ROUTES[0])] \
        + [r for r, d in rs.items() if r not in ROUTES and isinstance(d, dict) and ID.fullmatch(r)]


def is_route(rules, route) -> bool:
    return isinstance(route, str) and route in names(rules)


def label(rules, route: str) -> str:
    d = _routes(rules).get(route)
    return (d.get("name") if isinstance(d, dict) else None) or route


def kind(rules, route: str | None) -> list[str]:
    d = _routes(rules).get(route) if route else None
    return list(d.get("kind") or []) if isinstance(d, dict) else []


def words(rules, route: str, default: dict | None = None) -> str:
    """What the route is for, to Halo: its own words, else the built-in ones, else its name; with what its kind says."""
    d = _routes(rules).get(route)
    d = d if isinstance(d, dict) else {}
    text = d.get("about") or (default or {}).get(route) or d.get("name") or route
    extra = [KINDS[k] for k in kind(rules, route) if k not in ("thinking", "no_thinking")]
    return text + (" (for " + ", ".join(extra) + ")" if extra else "")


def thinking(rules, route: str | None):
    """True or False when the route's kind says so, else None (the model's own)."""
    k = kind(rules, route)
    return True if "thinking" in k else False if "no_thinking" in k else None


def instructions(rules, route: str | None) -> str:
    routes = getattr(rules, "routes", None) or {}
    d = routes.get(route) if route else None
    return (d.get("instructions") or "") if isinstance(d, dict) else ""


def longest(rules) -> str:
    """The longest instructions of any route: the request is sized before its route is known."""
    texts = [instructions(rules, r) for r in names(rules)]
    return max(texts, key=len) if texts else ""


def route_of(turn, tier_name: str | None) -> str | None:
    """The route this turn is on (see the module's words), or None for a passthrough caller."""
    if getattr(turn, "passthrough", False):
        return None
    routed = getattr(turn, "routed", None)
    rules = getattr(turn, "rules", None)
    if routed and routed.get("moved") and is_route(rules, routed.get("then", {}).get("to")):
        return routed["then"]["to"]
    st = getattr(turn, "stage", None)
    if st and st.get("mode") == "live" and is_route(rules, st.get("picked")):
        return st["picked"]
    policy = getattr(turn, "policy", None)
    if policy is not None and tier_name:
        s = policy.stage(tier_name)
        return s if is_route(rules, s) else "everyday"
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
