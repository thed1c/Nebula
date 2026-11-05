"""nebula/contracts.py: a Star Weaving team's contract, one page that says what the team is for and how it works.

    "rules": {"starweave": {"teams": {"code": {
        "name": "Code team", "class": "coding",              # coding, council or research
        "purpose": "Builds and fixes code with tests.", "examples": ["add a feature", "fix a failing test"],
        "routes": ["larger"],                                   # the routes it serves (Halo picks among their teams)
        "lead": "big", "helpers": ["codex"], "reviewer": "codex",
        "good_at": {"codex": "fast, careful edits"},            # a line per member, on the team sheet
        "stand_ins": {"codex": "think"},                        # who works when a member is not answering
        "others": false,                                        # whether models off the team may be given work
        "review": {"lead_reviews": true, "max_rework": 2},
        "budgets": {"max_task_tokens": 3000000, "max_background_seconds": 2700, "cloud_cap_usd": 1.0,
                    "max_depth": 2, "max_in_flight": 3},
        "tools": {"lead": "write", "helpers": "write", "reviewer": "read", "folder": "/path", "skills": ["pdf"]},
        "house_rules": "Never commit.",
        "done": {"tests": "python3 -m unittest", "review": true, "report": true},
        "escalation": {"ask_before": ["deleting files", "adding a dependency"], "warn_at": 0.8, "ask_after_rejects": 2},
        "reporting": {"sections": ["done", "files", "tests", "verdict", "left over"]}}}}}

A team has a class, which sets what it does by default and adds a rule of its own:
- **coding**: makes the change, runs the tests, has it reviewed;
- **council**: every crew member gives its own view on the question, and the lead weighs them (Halo checks each one
  was heard before the answer reaches the owner);
- **research**: finds things out, read-only, and says where each fact comes from;
- **writing**: drafts and edits long text, reviewed before it reaches the owner;
- **review**: checks finished work and changes nothing (Halo holds every role to read only);
- **planning**: turns a goal into steps, owners and risks, and changes nothing;
- **debugging**: reproduces a failure first, then fixes it with a test that proves it.

Council, research, review and planning teams change no files: every role is held to read only, whatever the tools
part says.

A team has a name and the routes it serves, so one route can have a Code team and a Research team. The key ("code")
is its id. Every part may be left out: a part left out falls back to Star Weaving's own settings (rules.starweave),
so a contract can start small. The older shape, a team per route ({"larger": {"lead", "helpers", "reviewer"}}), is
read as a team named after its route that serves that route, and changes nothing about how it works.

Who enforces what:
- Halo, in code, on every call: who is on the team (roles), that nobody reviews their own work, the budgets, which
  tools each role gets, and the done checks when the lead says it is done (it runs the test command itself).
- The lead, from its instructions: the purpose, the house rules, what to ask the owner before doing.

A task takes a snapshot of its team's contract when it starts (nebula/teamtasks.py) and keeps it with the contract's
version. Editing a contract changes new tasks only, never one already running.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading

ROUTES = ("everyday", "larger", "writing", "critique")   # built in; a team may also serve a route of the owner's own
LEVELS = ("write", "read", "none")
ROLES = ("lead", "helpers", "reviewer")
CLASSES = ("coding", "council", "research", "writing", "review", "planning", "debugging")
CLASS_NAMES = {"coding": "Coding", "council": "Council", "research": "Research", "writing": "Writing", "review": "Review",
               "planning": "Planning", "debugging": "Debugging"}
READ_ONLY_CLASSES = ("council", "research", "review", "planning")   # these change no files: Halo holds every role to read
BUDGETS = {                                     # name: (lowest, highest); the defaults are Star Weaving's (team.LIMITS)
    "max_task_tokens": (10_000, 5_000_000),
    "max_task_seconds": (60, 86_400),
    "max_background_seconds": (60, 86_400),
    "cloud_cap_usd": (0.0, 100.0),
    "max_depth": (1, 3),
    "max_in_flight": (1, 16),
    "max_hand_moves": (0, 20),
}
MAX_TEAMS = 24
MAX_HELPERS = 6
TEXT, LINE, ITEMS = 2000, 200, 8                # characters of a long text, of one line; lines in a list
TAG, TAGS = 24, 8                               # a tag's characters; tags per team
HOUSE_RULES = 4000
DEFAULT_SECTIONS = ["done", "files", "tests", "verdict", "left over"]
AUTO_TESTS = "auto"                             # "tests pass when the folder has them"
WARN_AT = 0.8


class ContractError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ the shape
def _str(v, n=LINE) -> str:
    return str(v).strip()[:n] if isinstance(v, (str, int, float)) and not isinstance(v, bool) else ""


def _lines(v, n=ITEMS) -> list[str]:
    if isinstance(v, str):
        v = [x for x in v.replace(";", "\n").splitlines()]
    return [s for s in (_str(x) for x in (v or [])) if s][:n] if isinstance(v, (list, tuple, str)) else []


def _model(v) -> str | None:
    s = _str(v)
    return s or None


def _route_id(r: str) -> bool:
    """A route a team may serve: one of the four built in, or an id of the owner's own (nebula/route_settings.py)."""
    from nebula import route_settings
    return r in ROUTES or bool(route_settings.ID.fullmatch(r))


def _tags(v) -> list[str]:
    out = []
    for t in _lines(v, TAGS * 2):
        t = " ".join(t.split())[:TAG].lower()
        if t and t not in out:
            out.append(t)
    return out[:TAGS]


def is_legacy(key: str, raw) -> bool:
    """The older shape: a team per route, keyed by the route, with no routes of its own."""
    return key in ROUTES and isinstance(raw, dict) and "routes" not in raw


def normalise(key: str, raw) -> dict:
    """A contract, read leniently (what is not understood is left out): every part present, blanks as None/[]/{}."""
    raw = raw if isinstance(raw, dict) else {}
    routes = [r for r in _lines(raw.get("routes"), ITEMS) if _route_id(r)]
    if is_legacy(key, raw):
        routes = [key]
    helpers = []
    for h in _lines(raw.get("helpers"), MAX_HELPERS):
        if h not in helpers:
            helpers.append(h)
    good = raw.get("good_at") if isinstance(raw.get("good_at"), dict) else {}
    stand = raw.get("stand_ins") if isinstance(raw.get("stand_ins"), dict) else {}
    review = raw.get("review") if isinstance(raw.get("review"), dict) else {}
    budgets = raw.get("budgets") if isinstance(raw.get("budgets"), dict) else {}
    tools = raw.get("tools") if isinstance(raw.get("tools"), dict) else {}
    done = raw.get("done") if isinstance(raw.get("done"), dict) else {}
    esc = raw.get("escalation") if isinstance(raw.get("escalation"), dict) else {}
    rep = raw.get("reporting") if isinstance(raw.get("reporting"), dict) else {}
    b = {}
    for k, (lo, hi) in BUDGETS.items():
        v = budgets.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        b[k] = max(lo, min(float(v) if k == "cloud_cap_usd" else int(v), hi))
    rework = review.get("max_rework")
    out = {
        "name": _str(raw.get("name"), 60) or key.replace("-", " ").replace("_", " ").title(),
        "class": raw.get("class") if raw.get("class") in CLASSES else "",
        "tags": _tags(raw.get("tags")),
        "purpose": _str(raw.get("purpose"), TEXT),
        "examples": _lines(raw.get("examples"), 4),
        "routes": routes,
        "lead": _model(raw.get("lead")),
        "helpers": helpers,
        "reviewer": _model(raw.get("reviewer")),
        "good_at": {str(k): _str(v) for k, v in good.items() if _str(v)},
        "stand_ins": {str(k): _str(v) for k, v in stand.items() if _str(v) and _str(v) != str(k)},
        "others": raw.get("others") if isinstance(raw.get("others"), bool) else True,
        "review": {"lead_reviews": review.get("lead_reviews") if isinstance(review.get("lead_reviews"), bool) else True,
                   "max_rework": max(0, min(int(rework), 5)) if isinstance(rework, int) and not isinstance(rework, bool) else None},
        "budgets": b,
        "tools": {**{r: tools.get(r) if tools.get(r) in LEVELS else None for r in ROLES},
                  "folder": os.path.expanduser(_str(tools.get("folder"), 1000)) or None,
                  "skills": _lines(tools.get("skills"), 16)},
        "house_rules": _str(raw.get("house_rules"), HOUSE_RULES),
        "done": {"tests": _str(done.get("tests"), 1000),
                 "review": bool(done.get("review")),
                 "report": bool(done.get("report")),
                 "council": bool(done.get("council"))},
        "escalation": {"ask_before": _lines(esc.get("ask_before")),
                       "warn_at": max(0.5, min(float(esc["warn_at"]), 0.99))
                       if isinstance(esc.get("warn_at"), (int, float)) and not isinstance(esc.get("warn_at"), bool) else WARN_AT,
                       "ask_after_rejects": max(0, min(int(esc["ask_after_rejects"]), 5))
                       if isinstance(esc.get("ask_after_rejects"), int) and not isinstance(esc.get("ask_after_rejects"), bool) else 0},
        "reporting": {"sections": _lines(rep.get("sections")) or []},
    }
    if out["tools"]["reviewer"] == "write":
        out["tools"]["reviewer"] = "read"       # a reviewer reads; it never changes the work it judges
    if out["class"] in READ_ONLY_CLASSES:
        for r in ROLES:
            if out["tools"][r] in (None, "write"):
                out["tools"][r] = "read"
    return out


def compact(c: dict) -> dict:
    """The contract as the config keeps it: the parts left blank are left out."""
    out = {"name": c["name"]}
    if c.get("class"):
        out["class"] = c["class"]
    for k in ("tags", "purpose", "examples", "routes", "lead", "helpers", "reviewer", "good_at", "stand_ins", "house_rules"):
        if c.get(k):
            out[k] = c[k]
    if c.get("others") is False:
        out["others"] = False
    rv = {k: v for k, v in c["review"].items() if not (k == "lead_reviews" and v is True) and v is not None}
    if rv:
        out["review"] = rv
    if c["budgets"]:
        out["budgets"] = dict(c["budgets"])
    tl = {k: v for k, v in c["tools"].items() if v}
    if tl:
        out["tools"] = tl
    dn = {k: v for k, v in c["done"].items() if v}
    if dn:
        out["done"] = dn
    es = {k: v for k, v in c["escalation"].items() if v and not (k == "warn_at" and v == WARN_AT)}
    if es:
        out["escalation"] = es
    if c["reporting"]["sections"]:
        out["reporting"] = {"sections": list(c["reporting"]["sections"])}
    return out


def norm(c: dict, key: str | None = None) -> dict:
    """normalise() of a contract that may already be normalised (it is idempotent); its id and version left out."""
    return normalise(key or c.get("id") or "team", {k: v for k, v in c.items() if k not in ("id", "version")})


def version(c: dict) -> str:
    """The contract's version: the same contract always has the same one, any edit gives a new one."""
    body = json.dumps(compact(norm(c)), sort_keys=True, ensure_ascii=False)
    return "v" + hashlib.sha256(body.encode()).hexdigest()[:8]


def _section(rules) -> dict:
    sw = getattr(rules, "starweave", None) if not isinstance(rules, dict) else rules.get("starweave")
    return sw if isinstance(sw, dict) else {}


def teams(rules) -> dict[str, dict]:
    """Every team's contract, read leniently, by id; the older per-route shape read as a team per route."""
    raw = _section(rules).get("teams")
    if not isinstance(raw, dict):
        return {}
    return {str(k): normalise(str(k), v) for k, v in raw.items() if isinstance(v, dict)}


def for_route(rules, route: str | None) -> list[tuple[str, dict]]:
    """The teams that serve a route, in the config's order."""
    return [(k, c) for k, c in teams(rules).items() if route and route in c["routes"]]


# ------------------------------------------------------------------------------------------------ checks on save
def check(key: str, raw, tier_names=None, route_names=None) -> dict:
    """One contract, checked as the Star Weaving page's Save checks it; the compact contract. Refuses, with the
    reason: a team with no lead, a member that is not a model here, a reviewer who is the only author, a budget out
    of range, a part of the wrong kind, a route that is not one (of `route_names`, when given)."""
    if not isinstance(raw, dict):
        raise ContractError(f"team {key}: a contract is an object")
    k = str(key or "").strip()
    if not k or len(k) > 40 or not all(ch.isalnum() or ch in "-_" for ch in k):
        raise ContractError(f"team id {key!r}: letters, digits, - and _ only, at most 40")
    for part in ("review", "budgets", "tools", "done", "escalation", "reporting", "good_at", "stand_ins"):
        if raw.get(part) is not None and not isinstance(raw[part], dict):
            raise ContractError(f"team {k}: {part} is an object")
    for part in ("helpers", "examples", "routes", "tags"):
        if raw.get(part) is not None and not isinstance(raw[part], (list, str)):
            raise ContractError(f"team {k}: {part} is a list")
    if raw.get("class") not in (None, "", *CLASSES):
        raise ContractError(f"team {k}: the class is one of {', '.join(CLASSES)}")
    routes = raw.get("routes") if not is_legacy(k, raw) else [k]
    known = list(route_names) if route_names is not None else list(ROUTES)   # the owner's own, when the caller knows them
    for r in _lines(routes, ITEMS):
        if not _route_id(r) or r not in known:
            raise ContractError(f"team {k}: {r!r} is not a route; the routes are {', '.join(known)}")
    for name, (lo, hi) in BUDGETS.items():
        v = (raw.get("budgets") or {}).get(name)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise ContractError(f"team {k}: {name} is a number from {lo:g} to {hi:g}")
    for role in ROLES:
        v = (raw.get("tools") or {}).get(role)
        if v not in (None, "", *LEVELS):
            raise ContractError(f"team {k}: tools for the {role} are write, read or none")
    rw = (raw.get("review") or {}).get("max_rework")
    if rw is not None and (isinstance(rw, bool) or not isinstance(rw, int) or not 0 <= rw <= 5):
        raise ContractError(f"team {k}: rework rounds are 0 to 5")
    f = (raw.get("tools") or {}).get("folder")
    if f and not os.path.isabs(os.path.expanduser(str(f))):
        raise ContractError(f"team {k}: the folder is a full path")
    c = normalise(k, raw)
    if not c["lead"]:
        raise ContractError(f"team {k}: every team has a lead")
    if c["lead"] in c["helpers"]:
        raise ContractError(f"team {k}: {c['lead']} leads; it is not also a helper")
    members = [c["lead"], *c["helpers"], *([c["reviewer"]] if c["reviewer"] else []), *c["stand_ins"].values()]
    if tier_names is not None:
        names = set(tier_names)
        for m in members:
            if m not in names:
                raise ContractError(f"team {k}: there is no model {m!r} here")
    authors = {c["lead"], *c["helpers"]}
    if c["reviewer"] and authors == {c["reviewer"]}:
        raise ContractError(f"team {k}: {c['reviewer']} would review only its own work; pick another reviewer or add a helper")
    if c["done"]["council"] and not c["helpers"]:
        raise ContractError(f"team {k}: a council needs a crew to hear from: add a helper")
    if c["done"]["review"] and not c["reviewer"] and not (c["review"]["lead_reviews"] and c["helpers"]):
        raise ContractError(f"team {k}: done needs an approved review, so the team needs a reviewer")
    return compact(c)


def check_all(raw, tier_names=None, route_names=None) -> dict:
    """The teams section, checked: {id: compact contract}."""
    if not isinstance(raw, dict):
        raise ContractError("starweave.teams is {team id: its contract}")
    if len(raw) > MAX_TEAMS:
        raise ContractError(f"at most {MAX_TEAMS} teams")
    return {str(k): check(str(k), v, tier_names, route_names) for k, v in raw.items()}


def merge_route_teams(existing: dict, route_teams: dict) -> dict:
    """The page's older per-route edit ({route: {lead, helpers, reviewer}}) applied to named contracts: the first team
    serving each route takes its members, a route with no team gets one named after it. The rest of every contract
    stays as it was."""
    out = {k: copy.deepcopy(v) for k, v in (existing or {}).items()}
    for route, m in (route_teams or {}).items():
        if not isinstance(m, dict):
            continue
        key = next((k for k, v in out.items() if route in normalise(k, v)["routes"]), None)
        if key is None:
            key = route if route not in out else f"{route}-team"
            out[key] = {"name": route.title(), "routes": [route]}
        elif is_legacy(key, out[key]):
            out[key] = {**out[key], "routes": [route]}
        out[key] = {**out[key], "lead": m.get("lead"), "helpers": [h for h in m.get("helpers") or [] if h],
                    "reviewer": m.get("reviewer")}
    return out


def all_legacy(raw) -> bool:
    return isinstance(raw, dict) and bool(raw) and all(is_legacy(str(k), v) for k, v in raw.items())


# ------------------------------------------------------------------------------------------------ templates
TEMPLATES = {                                   # one per class: what a new team of that class starts with
    "coding": {"name": "Coding team", "class": "coding", "purpose": "Builds and fixes code, with its tests.",
               "examples": ["add a feature", "fix a failing test", "refactor a module"],
               "others": False, "review": {"max_rework": 2},
               "budgets": {"max_task_tokens": 3_000_000, "max_background_seconds": 2700, "cloud_cap_usd": 1.0, "max_depth": 2,
                           "max_in_flight": 3},
               "tools": {"lead": "write", "helpers": "write", "reviewer": "read"},
               "house_rules": "Never commit. Keep changes small and say why each one is needed.",
               "done": {"tests": AUTO_TESTS, "review": True, "report": True},
               "escalation": {"ask_before": ["deleting files", "adding a dependency"], "ask_after_rejects": 2},
               "reporting": {"sections": DEFAULT_SECTIONS}},
    "council": {"name": "Council", "class": "council",
                "purpose": "Weighs a hard question: every member gives its own view, and the lead weighs them into one answer.",
                "examples": ["which design should we pick", "is this plan sound", "what are we missing"],
                "others": False, "review": {"max_rework": 0},
                "budgets": {"max_task_tokens": 1_000_000, "max_background_seconds": 1800, "max_depth": 1, "max_in_flight": 4},
                "tools": {"lead": "read", "helpers": "read", "reviewer": "read"},
                "house_rules": "Each member answers on its own, without seeing the others. Say where the members agree and where they differ.",
                "done": {"council": True, "report": True},
                "reporting": {"sections": ["answer", "where they agree", "where they differ"]}},
    "writing": {"name": "Writing team", "class": "writing", "purpose": "Writes and edits long text: documents, stories, release notes.",
                "examples": ["draft a README", "edit a chapter", "write release notes"],
                "others": False, "review": {"max_rework": 2},
                "budgets": {"max_task_tokens": 1_500_000, "max_background_seconds": 1800, "max_depth": 2, "max_in_flight": 2},
                "tools": {"lead": "write", "helpers": "write", "reviewer": "read"},
                "house_rules": "Keep the owner's voice. Plain words, short sentences.",
                "done": {"review": True, "report": True},
                "escalation": {"ask_before": ["changing the meaning of a section"], "ask_after_rejects": 2},
                "reporting": {"sections": ["done", "files", "verdict", "left over"]}},
    "review": {"name": "Review team", "class": "review", "purpose": "Checks work that is already done and says what to fix.",
               "examples": ["review a patch", "check a document against its brief"],
               "others": False, "review": {"max_rework": 0},
               "budgets": {"max_task_tokens": 500_000, "max_background_seconds": 1200, "max_depth": 1, "max_in_flight": 2},
               "tools": {"lead": "read", "helpers": "read", "reviewer": "read"},
               "house_rules": "Quote the part, then say what is wrong. Change nothing.",
               "done": {"report": True},
               "reporting": {"sections": ["verdict", "blocking issues", "followups"]}},
    "planning": {"name": "Planning team", "class": "planning", "purpose": "Turns a goal into a plan: steps, who does each, risks.",
                 "examples": ["plan the 0.2 release", "break this feature into parts", "what order should we do this in"],
                 "others": True, "review": {"max_rework": 1},
                 "budgets": {"max_task_tokens": 800_000, "max_background_seconds": 1200, "max_depth": 2, "max_in_flight": 3},
                 "tools": {"lead": "read", "helpers": "read", "reviewer": "read"},
                 "house_rules": "Each step says who does it and how we know it is done. Change nothing.",
                 "done": {"report": True},
                 "escalation": {"ask_before": ["choosing between goals"]},
                 "reporting": {"sections": ["goal", "steps", "risks", "open questions"]}},
    "debugging": {"name": "Debugging team", "class": "debugging", "purpose": "Finds why something fails and fixes it, with a test that proves it.",
                  "examples": ["this test fails since yesterday", "why does the server crash on start", "fix this error"],
                  "others": False, "review": {"max_rework": 2},
                  "budgets": {"max_task_tokens": 2_000_000, "max_background_seconds": 2700, "cloud_cap_usd": 1.0, "max_depth": 2,
                              "max_in_flight": 2},
                  "tools": {"lead": "write", "helpers": "write", "reviewer": "read"},
                  "house_rules": "Reproduce the failure first and show it. Fix the cause, not the symptom. Never commit.",
                  "done": {"tests": AUTO_TESTS, "review": True, "report": True},
                  "escalation": {"ask_before": ["deleting files", "changing a test to make it pass"], "ask_after_rejects": 2},
                  "reporting": {"sections": ["cause", "fix", "files", "tests", "left over"]}},
    "research": {"name": "Research team", "class": "research", "purpose": "Finds things out and reports with sources.",
                 "examples": ["compare three libraries", "read a paper and sum it up", "find why a test fails"],
                 "others": True, "review": {"max_rework": 1},
                 "budgets": {"max_task_tokens": 1_000_000, "max_background_seconds": 1800, "max_depth": 2, "max_in_flight": 4},
                 "tools": {"lead": "read", "helpers": "read", "reviewer": "read"},
                 "house_rules": "Say where each fact comes from. Say what you could not check.",
                 "done": {"report": True},
                 "escalation": {"ask_after_rejects": 2},
                 "reporting": {"sections": ["answer", "sources", "left over"]}},
}
CLASS_TEXT = {                                  # what the lead reads for its class
    "coding": "This is a coding team: make the change, run the tests, and have the change reviewed before you answer.",
    "council": ("This is a council: give the question to every member of your crew (team__delegate, one item each, the "
                "same question in full), so each answers on its own. Then weigh their views and answer: where they agree, "
                "where they differ, and your call. Halo checks that every member was heard."),
    "research": "This is a research team: find things out, say where each fact comes from, and say what you could not check.",
    "writing": "This is a writing team: draft it, have the draft reviewed, and fix what the review finds before you answer.",
    "review": "This is a review team: check the work and say what to fix, quoting each part. You change nothing.",
    "planning": "This is a planning team: turn the goal into steps, who does each, how each is checked, and the risks. You change nothing.",
    "debugging": ("This is a debugging team: reproduce the failure first and show it, find the cause, fix it, and prove "
                  "the fix with a test that failed before and passes now."),
}


def template(name: str, lead: str | None = None, helpers=(), reviewer: str | None = None, routes=(), models=()) -> dict:
    """A draft contract from a template, with the members filled in: the owner adjusts it, then saves it. Given the
    models here, a part the template needs is filled from them, so the draft saves as it comes: a reviewer other
    than the lead when done needs a review, and a crew of two when it is a council."""
    t = TEMPLATES.get(name)
    if t is None:
        raise ContractError(f"no template {name!r}; the templates are {', '.join(TEMPLATES)}")
    out = copy.deepcopy(t)
    helpers, others = [h for h in helpers if h and h != lead], [m for m in models if m and m != lead]
    if (out.get("done") or {}).get("council") and not helpers:
        helpers = others[:2]
    if (out.get("done") or {}).get("review") and not reviewer and others:
        reviewer = next((m for m in others if m not in helpers), others[0])
    out.update({"lead": lead, "helpers": list(helpers), "reviewer": reviewer, "routes": list(routes)})
    return {k: v for k, v in out.items() if v not in (None, [])}


# ------------------------------------------------------------------------------------------------ the first contract
FIRST_PURPOSE = {
    "everyday": "Everyday work with a few parts: quick answers, small edits, short checks.",
    "larger": "Builds and fixes code with tests; deep reasoning and long documents.",
    "writing": "Writes and edits long text.",
    "critique": "Reviews and critiques work.",
}


def first_contracts(sw: dict) -> dict:
    """Today's per-route teams as named contracts, with today's numbers as each team's own budgets. One change: a team
    whose reviewer is its own lead (or has none) gets its first helper as the reviewer, so review is independent, and
    done needs that review (and the tests, when the folder has them)."""
    from nebula import team as T
    out = {}
    for route, m in (sw.get("teams") or {}).items():
        if not is_legacy(str(route), m):
            out[route] = m
            continue
        lead, helpers, rev = m.get("lead"), [h for h in m.get("helpers") or [] if h and h != m.get("lead")], m.get("reviewer")
        if (not rev or rev == lead) and helpers:
            rev = helpers[0]
        budgets = {}
        for k in BUDGETS:
            default = T.LIMITS[k][0]
            v = sw.get(k, default)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                budgets[k] = v
        c = {"name": f"{route.title()} team", "class": "coding" if route == "larger" else "", "purpose": FIRST_PURPOSE.get(route, ""), "routes": [route], "lead": lead,
             "helpers": helpers, "reviewer": rev if rev and rev != lead else None, "budgets": budgets,
             "tools": {"lead": "write", "helpers": "write", "reviewer": "read"},
             "review": {"max_rework": int(sw.get("max_rework", T.LIMITS["max_rework"][0]))},
             "done": {"tests": AUTO_TESTS, "review": bool(rev and rev != lead)},
             "escalation": {"ask_after_rejects": 2}}
        good = {h: sw["good_at"][h] for h in [lead, *helpers, rev] if h and isinstance(sw.get("good_at"), dict) and sw["good_at"].get(h)}
        if good:
            c["good_at"] = good
        out[route] = compact(normalise(route, c))
    return out


# ------------------------------------------------------------------------------------------------ a task's contract
_SNAP: dict[str, dict] = {}                     # task id -> its contract snapshot (immutable once the task started)
_SNAP_LOCK = threading.Lock()


def snapshot(key: str, c: dict) -> dict:
    c = norm(c, key)
    c["id"] = key
    c["version"] = version(c)
    return c


def remember(task_id: str, c: dict | None) -> None:
    if c is None:
        return
    with _SNAP_LOCK:
        if len(_SNAP) > 1000:
            for k in list(_SNAP)[:200]:
                _SNAP.pop(k, None)
        _SNAP[task_id] = c


def _task_id(turn) -> str | None:
    ctx = getattr(turn, "team_ctx", None) or {}
    if ctx.get("task_id", "").startswith("task:"):
        return ctx["task_id"][5:]
    from nebula import names
    given = names.header(getattr(turn, "headers", None), "Task")
    return str(given) if given else None


def of_turn(turn) -> dict | None:
    """The contract this turn's team task runs under (its snapshot), or None outside a team task."""
    tid = _task_id(turn)
    if not tid:
        return None
    with _SNAP_LOCK:
        if tid in _SNAP:
            return _SNAP[tid]
    led = getattr(getattr(turn, "cfg", None), "ledger", None)
    if not led:
        return None
    try:
        with open(f"{led}.tasks.json", encoding="utf-8") as f:
            for t in json.load(f).get("tasks") or []:
                if t.get("id") == tid and isinstance(t.get("contract"), dict):
                    remember(tid, t["contract"])
                    return t["contract"]
    except (OSError, ValueError):
        return None
    return None


def role_of(turn, c: dict | None = None) -> str | None:
    """lead, helpers or reviewer: this turn's role on its task's team."""
    c = c if c is not None else of_turn(turn)
    if c is None:
        return None
    ctx = getattr(turn, "team_ctx", None) or {}
    if not ctx:
        return "lead"
    return "reviewer" if ctx.get("mode") == "review" else "helpers"


def tools_level(turn) -> str | None:
    """write, read or none for this turn's role, as its contract says; None when the contract says nothing."""
    c = of_turn(turn)
    if c is None:
        return None
    role = role_of(turn, c)
    lv = c["tools"].get(role)
    if role == "reviewer" and lv == "write":
        lv = "read"
    return lv


def members(c: dict) -> list[str]:
    return [m for m in [c.get("lead"), *c.get("helpers", []), c.get("reviewer")] if m]


def limits(base: dict, c: dict | None) -> dict:
    """Star Weaving's limits with the contract's budgets over them (and its rework rounds)."""
    out = dict(base)
    if c is None:
        return out
    out.update(c.get("budgets") or {})
    if (c.get("review") or {}).get("max_rework") is not None:
        out["max_rework"] = c["review"]["max_rework"]
    return out


def skills(turn) -> list[str]:
    c = of_turn(turn)
    return list((c or {}).get("tools", {}).get("skills") or [])


# ------------------------------------------------------------------------------------------------ words
def _mins(s) -> str:
    s = int(s)
    return f"{s // 3600} h {s % 3600 // 60} min".replace(" 0 min", "") if s >= 3600 else f"{max(1, s // 60)} min"


def _tokens(n) -> str:
    n = int(n)
    return f"{n / 1_000_000:g}M" if n >= 1_000_000 else f"{n // 1000}K"


def plain(c: dict, base: dict | None = None) -> list[str]:
    """The contract in plain sentences, for its card on the Star Weaving page."""
    c = norm(c)
    lim = limits(base or {}, c)
    out = []
    name = c["name"]
    if c["purpose"]:
        out.append(f"The {name} {c['purpose'][0].lower() + c['purpose'][1:]}" if not c["purpose"].lower().startswith(name.lower())
                   else c["purpose"])
    who = [f"{c['lead']} leads"] if c["lead"] else ["No lead yet"]
    if c["helpers"]:
        who.append(", ".join(c["helpers"]) + (" helps" if len(c["helpers"]) == 1 else " help"))
    if c["reviewer"]:
        who.append(f"{c['reviewer']} reviews")
    out.append(", ".join(who) + "." + ("" if c["others"] else " Only they work on it."))
    if c["class"] == "council":
        out.append(f"A council: every crew member gives its own view, and {c['lead'] or 'the lead'} weighs them.")
    if c["routes"]:
        out.append("Serves " + ", ".join(c["routes"]) + ".")
    b = []
    if "max_task_tokens" in lim:
        b.append(f"{_tokens(lim['max_task_tokens'])} tokens")
    if "max_background_seconds" in lim:
        b.append(_mins(lim["max_background_seconds"]))
    if lim.get("cloud_cap_usd"):
        b.append(f"${lim['cloud_cap_usd']:g} of cloud on its pieces")
    if b:
        out.append("Up to " + ", ".join(b) + ".")
    d = []
    if c["done"]["tests"]:
        d.append("the tests pass" if c["done"]["tests"] == AUTO_TESTS else f"`{c['done']['tests']}` passes")
    if c["done"]["review"]:
        d.append(f"{c['reviewer'] or 'the reviewer'} approves")
    if c["done"]["council"]:
        d.append("every crew member was heard")
    if c["done"]["report"]:
        d.append("the report has its parts")
    out.append(("Done when " + " and ".join(d) + ".") if d else "Done when the lead says so.")
    if c["escalation"]["ask_before"]:
        out.append("Asks you before " + ", ".join(c["escalation"]["ask_before"]) + ".")
    if c["house_rules"]:
        out.append(c["house_rules"].splitlines()[0][:160])
    return out


def lead_text(c: dict, base: dict | None = None) -> str:
    """The contract as the lead reads it, in its task prompt."""
    lim = limits(base or {}, c)
    lines = [f"Your team's contract ({c['name']}, {c.get('version') or version(c)}):"]
    if c.get("class"):
        lines.append(f"- Class: {CLASS_NAMES[c['class']]}. {CLASS_TEXT[c['class']]}")
    if c["purpose"]:
        lines.append(f"- Purpose: {c['purpose']}")
    roles = [f"you ({c['lead']}) lead"]
    for h in c["helpers"]:
        roles.append(f"{h} helps" + (f" ({c['good_at'][h]})" if c["good_at"].get(h) else ""))
    if c["reviewer"]:
        roles.append(f"{c['reviewer']} reviews")
    lines.append("- Roles: " + "; ".join(roles) + "." + ("" if c["others"] else " Only these models may be given work."))
    rv = "Every part a teammate did is reviewed by someone other than its author; never review your own draft."
    if not c["review"]["lead_reviews"]:
        rv += " You do not approve your team's work yourself: the reviewer does."
    if lim.get("max_rework") is not None:
        rv += f" At most {lim['max_rework']} rework rounds per part."
    lines.append("- Review: " + rv)
    b = []
    if "max_task_tokens" in lim:
        b.append(f"{_tokens(lim['max_task_tokens'])} tokens")
    if "max_background_seconds" in lim:
        b.append(_mins(lim["max_background_seconds"]))
    if lim.get("cloud_cap_usd") is not None:
        b.append(f"${lim['cloud_cap_usd']:g} cloud on pieces")
    if "max_in_flight" in lim:
        b.append(f"{lim['max_in_flight']} parts at once")
    if b:
        lines.append("- Budget: " + ", ".join(b) + ". Halo stops the work at the budget; at "
                     f"{int(c['escalation']['warn_at'] * 100)}% start putting the results together.")
    done = []
    if c["done"]["tests"]:
        done.append("the tests pass (Halo runs " + ("them" if c["done"]["tests"] == AUTO_TESTS else f"`{c['done']['tests']}`")
                    + " itself in the sandbox)")
    if c["done"]["review"]:
        done.append("a review approved the work (team__review)")
    if c["done"]["council"]:
        done.append("every member of your crew gave its view")
    if c["done"]["report"]:
        done.append("your answer has each report part")
    if done:
        lines.append("- Done means: " + "; ".join(done) + ". Halo checks this when you answer and sends the task back "
                     "with what failed.")
    if c["escalation"]["ask_before"]:
        lines.append("- Ask the owner (team__ask_owner) before: " + "; ".join(c["escalation"]["ask_before"]) + ".")
    if c["house_rules"]:
        lines.append("- House rules: " + c["house_rules"])
    return "\n".join(lines)


def report_sections(c: dict | None) -> list[str]:
    if c is None:
        return []
    return list(c["reporting"]["sections"] or (DEFAULT_SECTIONS if c["done"]["report"] else []))


def member_text(c: dict | None) -> str:
    """What every teammate reads besides its task: the team's house rules."""
    if not c or not c.get("house_rules"):
        return ""
    return f"\n\nYour team's house rules ({c['name']}): {c['house_rules']}"


# ------------------------------------------------------------------------------------------------ done checks
def missing_sections(text: str, sections: list[str]) -> list[str]:
    low = (text or "").lower()
    return [s for s in sections if s.lower() not in low]


def approved(items: list[dict]) -> bool:
    """Whether a review in the task approved work: a finished review item whose verdict says approve."""
    for i in items:
        if i.get("mode") != "review" or i.get("state") != "done":
            continue
        try:
            v = json.loads(i.get("result") or "")
        except ValueError:
            continue
        if isinstance(v, dict) and v.get("verdict") == "approve":
            return True
    return False


def unheard(c: dict, items: list[dict], down=lambda m: False) -> list[str]:
    """A council's crew members who gave no view in the task (a finished ask or item); a member that is down counts
    as heard when its stand-in gave one."""
    heard = {i.get("model") for i in items if i.get("mode") in ("ask", "execute") and i.get("state") == "done"}
    out = []
    for m in c.get("helpers") or []:
        stand = (c.get("stand_ins") or {}).get(m)
        if m in heard or (stand and stand in heard and down(m)):
            continue
        out.append(m)
    return out


def has_tests(folder: str | None) -> bool:
    if not folder or not os.path.isdir(folder):
        return False
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv"}
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in skip]
        if any(f.startswith("test") and f.endswith(".py") for f in files):
            return True
    return False


def test_command(c: dict, folder: str | None) -> str | None:
    """The command Halo runs for the done check, or None when there is none to run."""
    t = c["done"]["tests"]
    if not t:
        return None
    if t == AUTO_TESTS:
        return "python3 -m unittest discover" if has_tests(folder) else None
    return t


def convert_file(path: str) -> str:
    """The switch's one step: a config file whose teams are the older per-route shape gets its first contracts. The
    file is copied aside first (<file>.before-contracts.bak). Says what it did."""
    from pathlib import Path
    import shutil
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    sw = (d.get("rules") or {}).get("starweave")
    if not isinstance(sw, dict) or not all_legacy(sw.get("teams")):
        return "nothing to convert: the teams are contracts already (or there are none)"
    bak = f"{path}.before-contracts.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    new = first_contracts(sw)
    d["rules"]["starweave"] = {**sw, "teams": new}
    tmp = path + ".tmp"
    Path(tmp).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    lines = [f"converted {len(new)} team(s); the old file is at {bak}"]
    for k, c in new.items():
        lines += [f"[{k}]"] + ["  " + x for x in plain(normalise(k, c))]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != "--convert":
        sys.exit("usage: python3 -m nebula.contracts --convert <nebula.json>")
    print(convert_file(sys.argv[2]))
