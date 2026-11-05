"""nebula/team.py: Starweave, the models working as one team behind Halo: its settings, the team tools each turn is
offered, the team sheet, and Halo's rulings. The work itself runs in nebula/teamwork.py.

    "rules": {"starweave": {"mode": "off" | "practice" | "on", "max_hand_moves": 4, "max_depth": 2, "max_in_flight": 4,
                            "max_task_tokens": 400000, "max_task_seconds": 1200, "max_background_seconds": 7200,
                            "cloud_cap_usd": 1.0, "good_at": {"big": "long code and analysis"}}}

Starweave rides on the Switchyard and replaces none of it: every team move is an ordinary Switchyard move, through
the same ladder, gates, policy and ledger. In phase 1 the model answering a turn may move the work itself:

- `team__hand_up(to, why)`: this turn is beyond it. The larger model continues THIS answer: the turn crosses to it
  mid-stream (a handoff packet, as any crossing), and the client sees one answer that the larger model finishes.
- `team__hand_down(to, why)`: the hard part is done. It finishes its answer; the NEXT turn starts on the smaller
  model, from a briefing (the secretary's compaction, Halo's when it is the secretary), so the smaller model does
  not read the whole history again.

Halo referees every move before it is made (`check`): the target must be on this request's ladder in that
direction, the request must not have named its model, the session must be under its hand-move budget, and a move
that undoes the last two (A to B, B to A, A to B again) is refused. A refused move answers the model with the reason,
and it carries on. Every call, allowed or refused, practice or live, is a `team_action` event in the ledger.

Phase 2: the lead gives work to teammates and asks them, and every member may leave a note:

- `team__delegate(items, wait)`: work items, each to a teammate, run in parallel (at class "agent", never ahead of
  your own chat), each in a child session of its own; `depends_on` holds an item until the ones it names are done.
  `wait` all (the default) or any returns results as they finish, none returns at once.
- `team__collect(ids, wait)`: the items' results so far, whole.
- `team__ask(to, question)`: one short answer from a teammate (512 tokens at most).
- `team__note(text)`: the task's shared memory; every later teammate's prompt carries the task's notes.

Who gets which: the lead (the turn you started) all of them, plus the hand moves; a teammate working an item
ask and note (it does not start a team of its own); one asked at the last depth only note. Halo rules on each call
first: the depth, the loop guard (never back to a model already up the chain, never the same task twice to the same
model), the task's token, time and cloud-spend budgets, and the teammates the request may use (the routing policy,
the ticks, the registry). Teammates see only what their item needs: the user's request (its first 2,000 characters),
their task, the team's notes and the results they depend on, never the conversation.

- **off** (the default): no team tools are offered.
- **practice**: the tools are offered and each call is ruled and recorded, but nothing moves: the model is told the
  move was recorded and answers itself (smart routing's shadow, for Starweave).
- **on**: allowed moves are made.

Team tools are offered only to the owner's own requests (never a guest's or a paired Nebula's), never on a
passthrough or a request that named its model, only when the answering model may use tools (the routing policy's
`tools` capability) and the ladder has somewhere to move to.
"""
from __future__ import annotations

import json

MODES = ("off", "practice", "on")
TAG, TAGS = 24, 8                   # a model's tags (Star Weaving's crew): characters each, and how many
PREFIX = "team__"
HAND_UP, HAND_DOWN = "team__hand_up", "team__hand_down"
DELEGATE, COLLECT, ASK, NOTE = "team__delegate", "team__collect", "team__ask", "team__note"
REVIEW, ASK_OWNER = "team__review", "team__ask_owner"
MOVES = (HAND_UP, HAND_DOWN)
WORK = (DELEGATE, COLLECT, ASK, REVIEW, ASK_OWNER, NOTE)
WAITS = (DELEGATE, COLLECT, ASK, REVIEW)   # calls that wait on teammates: the lead's engine slot is given back meanwhile
MAX_OPTIONS, MAX_OWNER_ASKS = 4, 3          # an owner card's buttons; asks waiting on the owner per task
VERDICT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdict", "summary", "blocking_issues", "followups"],
                  "properties": {"verdict": {"type": "string", "enum": ["approve", "reject"]}, "summary": {"type": "string"},
                                 "blocking_issues": {"type": "array", "items": {"type": "string"}},
                                 "followups": {"type": "array", "items": {"type": "string"}}}}
DEFAULT_MAX_HAND_MOVES = 4
MAX_ROUNDS = 8                      # a turn without an agent: how many tool rounds team calls may take
MAX_ITEMS_PER_CALL = 8
MAX_NOTE = 1000                     # characters
ASK_TOKENS = 512
RESULT_CHARS = 6000                 # about 1,500 tokens: a result as the lead reads it from delegate (collect: whole)
REQUEST_CHARS = 2000                # the user's request as a teammate reads it
LIMITS = {                          # name: (default, lowest, highest), each a whole number but the cloud cap
    "max_hand_moves": (DEFAULT_MAX_HAND_MOVES, 0, 20),
    "max_depth": (2, 1, 3),
    "max_in_flight": (4, 1, 16),
    "max_task_tokens": (400_000, 10_000, 5_000_000),
    "max_task_seconds": (1200, 60, 86_400),
    "max_background_seconds": (7200, 60, 86_400),
    "cloud_cap_usd": (1.0, 0.0, 100.0),
    "max_rework": (2, 0, 5),
}
MODE_TEXT = {                       # one fixed instruction per turn mode, after the route's own, so caches hold
    "execute": ("You are a teammate on a Nebula team. Do the task below and return only the result, complete and ready "
                "for the lead to use. You do not see the lead's conversation: everything you have is below."),
    "ask": ("You are a teammate on a Nebula team. Answer the lead's question below in a few sentences, directly. You do "
            "not see the lead's conversation."),
    "review": ("You are the reviewer on a Nebula team. Check the work below against the criteria and the user's request. "
               "Return the verdict JSON only: approve, or reject with at least one blocking issue (each one specific: "
               "quote the part, say what is wrong). Followups are nice-to-haves that do not block."),
    "rework": ("You are a teammate on a Nebula team. A reviewer rejected your work. Fix only the blocking issues listed "
               "below, keep everything else as it was, and return the whole corrected result."),
}


class TeamError(ValueError):
    pass


def settings(rules) -> dict:
    s = dict(getattr(rules, "starweave", None) or {})
    out = {"mode": s.get("mode") if s.get("mode") in MODES else "off"}
    for k, (default, lo, hi) in LIMITS.items():
        try:
            v = float(s.get(k, default)) if k == "cloud_cap_usd" else int(s.get(k, default))
        except (TypeError, ValueError):
            v = default
        out[k] = max(lo, min(v, hi))
    out["reviewer"] = s.get("reviewer") if isinstance(s.get("reviewer"), str) and s.get("reviewer") else None
    good = s.get("good_at") if isinstance(s.get("good_at"), dict) else {}
    out["good_at"] = {str(k): str(v)[:200] for k, v in good.items() if isinstance(v, str) and v.strip()}
    tags = s.get("tags") if isinstance(s.get("tags"), dict) else {}
    out["tags"] = {str(k): [str(t)[:TAG] for t in v if isinstance(t, str) and t.strip()][:TAGS]
                   for k, v in tags.items() if isinstance(v, list) and v}
    from nebula import contracts
    out["teams"] = contracts.teams(rules)        # {team id: its contract} (nebula/contracts.py)
    return out


def contract(turn) -> dict | None:
    """The contract this turn's team task runs under (nebula/contracts.py), or None outside a team task."""
    from nebula import contracts
    return contracts.of_turn(turn)


def limits(turn) -> dict:
    """Star Weaving's settings with this task's contract's budgets over them: what Halo holds this turn to."""
    from nebula import contracts
    return contracts.limits(settings(turn.rules), contract(turn))


def check_settings(v) -> dict:
    """The starweave section, checked and normalised (the Routes tab's Apply)."""
    if not isinstance(v, dict):
        raise TeamError("starweave is an object")
    if v.get("mode", "off") not in MODES:
        raise TeamError("starweave.mode is off, practice or on")
    out = {"mode": v.get("mode", "off")}
    for k, (_default, lo, hi) in LIMITS.items():
        if k not in v or v[k] is None:
            continue
        m = v[k]
        if isinstance(m, bool) or not isinstance(m, (int, float)) or not lo <= m <= hi:
            what = "a number" if k == "cloud_cap_usd" else "a whole number"
            raise TeamError(f"starweave.{k} is {what} from {lo:g} to {hi:g}")
        out[k] = float(m) if k == "cloud_cap_usd" else int(m)
    if v.get("good_at") is not None:
        g = v["good_at"]
        if not isinstance(g, dict) or not all(isinstance(x, str) for x in g.values()):
            raise TeamError("starweave.good_at is {model: a short line of what it is good at}")
        g = {str(k): x.strip()[:200] for k, x in g.items() if x.strip()}
        if g:
            out["good_at"] = g
    if v.get("tags") is not None:            # a model's tags, set on the Models tab: shown on Star Weaving's crew
        g = v["tags"]
        if not isinstance(g, dict) or not all(isinstance(x, list) and all(isinstance(t, str) for t in x) for x in g.values()):
            raise TeamError("starweave.tags is {model: [tag, …]}")
        clean = {}
        for k, x in g.items():
            seen = []
            for t in x:
                t = " ".join(t.split()).lower()
                if len(t) > TAG:
                    raise TeamError(f"a tag is at most {TAG} characters: {t[:40]!r}")
                if t and t not in seen:
                    seen.append(t)
            if len(seen) > TAGS:
                raise TeamError(f"{k}: at most {TAGS} tags")
            if seen:
                clean[str(k)] = seen
        if clean:
            out["tags"] = clean
    if v.get("reviewer") not in (None, ""):
        if not isinstance(v["reviewer"], str):
            raise TeamError("starweave.reviewer is a model's name")
        out["reviewer"] = v["reviewer"]
    if v.get("teams") is not None:
        from nebula import contracts
        t = v["teams"]
        if not isinstance(t, dict):
            raise TeamError("starweave.teams is {team id: its contract}")
        for k, m in t.items():
            if not isinstance(m, dict):
                raise TeamError(f"starweave.teams.{k} is a contract (an object)")
            helpers = m.get("helpers") or []
            if not isinstance(helpers, list) or not all(isinstance(h, str) for h in helpers):
                raise TeamError(f"starweave.teams.{k}.helpers is a list of model names")
        if contracts.all_legacy(t):         # the older per-route shape, as the page's team cards still send it
            clean = {}
            for route, m in t.items():
                clean[route] = {"lead": m.get("lead") if isinstance(m.get("lead"), str) else None,
                                "helpers": [h for h in m.get("helpers") or [] if h][:contracts.MAX_HELPERS],
                                "reviewer": m.get("reviewer") if isinstance(m.get("reviewer"), str) else None}
        else:
            try:                            # a route's empty card (no lead) is no team
                clean = {k: contracts.check(k, m) for k, m in t.items()
                         if m.get("lead") or not contracts.is_legacy(k, m)}
            except contracts.ContractError as e:
                raise TeamError(str(e)) from None
        if clean:
            out["teams"] = clean
    return out


def is_team(name: str | None) -> bool:
    return isinstance(name, str) and name.startswith(PREFIX)


def arguments_of(call) -> dict:
    args = getattr(call, "arguments", None)
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            args = {}
    return args if isinstance(args, dict) else {}


YES, NO = ("on", "1", "true", "yes"), ("off", "none", "0", "false", "no")


def agent_work(turn) -> bool:
    """Whether Nebula's own tools (the team's, the toolbox, skills) go on this call: agent work only. That is Halo's
    chats (the page's chat, Halo's own turns, the assistant agent), a team task and its work items, and a caller that
    asks for them: X-Nebula-Tools: on, or its label in rules.toolbox.callers. A plain API call gets none (a Harness check
    went into 60 tool rounds and 2.2M tokens); X-Nebula-Tools: off refuses them on any call."""
    from nebula import names
    from nebula import toolbox
    ask = str(names.header(getattr(turn, "headers", None), "Tools") or "").strip().lower()
    if ask in NO:
        return False
    if ask in YES or getattr(turn, "team_ctx", None):
        return True
    label = (getattr(turn, "caller", None) or {}).get("label")
    if label == "starweave" and names.header(turn.headers, "Task"):
        return True                                     # a team task's lead (nebula/teamtasks.py)
    if getattr(turn, "dialect", None) in ("room", "halo") or getattr(getattr(turn, "agent", None), "name", None) == "assistant":
        return True
    return bool(label) and label in toolbox.settings(turn.rules)["callers"]


def _team_ok(turn) -> bool:
    """Whether this turn's model may be on a team at all."""
    s = settings(turn.rules)
    if s["mode"] == "off" or turn.ledger is None or turn.opened is None:
        return False
    if not agent_work(turn):
        return False
    if turn.tenant != "owner" or turn.guest or turn.peer is not None or turn.passthrough:
        return False
    if turn.tier is None:
        return False
    if turn.policy is not None and not turn.policy.eligible(turn.tier.name, "tools", True):
        return False
    if turn.policy is not None and turn.policy.schema(turn.tier.name) is not None:
        return False                    # a typed destination answers whole, checked against its schema
    return True


def depth_of(turn) -> int:
    return int((getattr(turn, "team_ctx", None) or {}).get("depth") or 0)


def chain_of(turn) -> list[str]:
    """The models up this turn's chain, itself last: a work item never goes back to one of them."""
    ctx = getattr(turn, "team_ctx", None) or {}
    return list(ctx.get("chain") or []) + ([turn.tier.name] if turn.tier is not None else [])


def offered(turn) -> bool:
    """Whether this turn's model is offered the hand moves: the lead only, never on a request that named its model."""
    if not _team_ok(turn) or getattr(turn, "team_ctx", None) or turn.pinned:
        return False
    return len(turn.ladder) >= 2


def route_team(turn) -> dict:
    """This turn's team: its task's contract (the snapshot the task took), else the first team serving the route this
    turn is on (nebula/contracts.py). {"lead", "helpers", "reviewer", "contract"} or {}."""
    from nebula import contracts
    c = contract(turn)
    if c is None:
        if turn.tier is None:
            return {}
        served = contracts.for_route(turn.rules, _route_word(turn, turn.tier.name) or "everyday")
        if not served:
            return {}
        c = served[0][1]
    return {"lead": c.get("lead"), "helpers": list(c.get("helpers") or []), "reviewer": c.get("reviewer"), "contract": c}


def _on_contract(turn) -> set | None:
    """The models a team task's contract lets work on it (its members, and a member's stand-in while the member is
    not answering); None when anyone in the pool may (no contract, or one that lets others help)."""
    c = contract(turn)
    if c is None or c.get("others", True):
        return None
    reg = getattr(turn, "registry", None)
    out = {m for m in [c.get("lead"), *(c.get("helpers") or []), c.get("reviewer")] if m}
    for m, stand in (c.get("stand_ins") or {}).items():
        if m in out and reg is not None and not reg.up(m):
            out.add(stand)
    return out


def teammates(turn) -> list:
    """The tiers this turn may give work to: the request's own pool (the ticks, the policy, the registry), less the
    models up its chain; the route's accepted helpers first, then its reviewer, then the rest in the pool's order. A
    priced model past the task's cloud cap is left to Halo's check. A command tier (Codex, Claude Code: the owner's own
    account, its own tools) joins only when the owner named it: an accepted helper or reviewer, or the reviewer setting;
    so does a model the policy uses only when a request names it."""
    chain = set(chain_of(turn))
    rt = route_team(turn)
    first = list(rt.get("helpers") or []) + ([rt["reviewer"]] if rt.get("reviewer") else [])
    allowed = _on_contract(turn)
    named = set(first) | {settings(turn.rules).get("reviewer")} | (allowed or set())
    policy = getattr(turn, "policy", None)
    pool = [t for t in getattr(turn, "team_pool", None) or [] if t.name not in chain
            and (allowed is None or t.name in allowed)
            and (not (t.is_command or (policy is not None and policy.explicit(t.name))) or t.name in named)]
    rank = {n: i for i, n in enumerate(first)}
    return sorted(pool, key=lambda t: rank.get(t.name, len(first)))


def work_offered(turn) -> list[str]:
    """The work tools this turn's model is offered: the lead all four, a teammate ask and note, the last depth note."""
    if not _team_ok(turn):
        return []
    depth, s = depth_of(turn), limits(turn)
    if (getattr(turn, "team_ctx", None) or {}).get("mode") in ("review", "rework"):
        return []                       # a reviewer answers with its verdict only; a rework fixes, nothing more
    mates = bool(teammates(turn))
    if depth == 0:
        return ([DELEGATE, COLLECT, ASK] if mates else []) + ([REVIEW] if reviewer_of(turn, any_item=True) else []) + [ASK_OWNER, NOTE]
    if depth < s["max_depth"]:
        return ([ASK] if mates else []) + [NOTE]
    return [NOTE]


def _route_word(turn, tier_name: str) -> str:
    from nebula import route_settings
    try:
        return route_settings.route_of(turn, tier_name) or ""
    except Exception:  # noqa: BLE001 - a word for a tool's description only
        return ""


def _targets(turn, up: bool) -> list[str]:
    names = [t.name for t in turn.ladder]
    if turn.tier is None or turn.tier.name not in names:
        return []
    i = names.index(turn.tier.name)
    return names[i + 1:] if up else list(reversed(names[:i]))


def reviewer_of(turn, to: str | None = None, item: dict | None = None, any_item: bool = False):
    """Who reviews for this turn: the one it named, else the reviewer setting, else the route's accepted team's, else
    the first teammate on the critique route. Never a model up its chain, except the lead's own model on an item a
    teammate did (`item`, or `any_item` for whether one could be reviewed); never the model that did the item. None
    when there is none."""
    mates = {t.name: t for t in teammates(turn)}
    c = contract(turn)
    lead_reviews = c is None or (c.get("review") or {}).get("lead_reviews", True)
    if (item is not None or any_item) and depth_of(turn) == 0 and turn.tier is not None and lead_reviews \
            and (item is None or item.get("model") != turn.tier.name):
        mates.setdefault(turn.tier.name, turn.tier)          # the lead may check its team's work itself
    if to:
        return mates.get(to)
    if item is not None:
        mates.pop(item.get("model"), None)
    s = settings(turn.rules)
    order = (s.get("reviewer"), route_team(turn).get("reviewer"))
    if c is not None:                   # a task's contract names its reviewer first
        order = (route_team(turn).get("reviewer"), s.get("reviewer"))
    for name in order:
        if name and name in mates:
            return mates[name]
    if turn.policy is not None:
        for t in mates.values():
            if t is not turn.tier and turn.policy.stage(t.name) == "critique":
                return t
    return None


def sheet_line(turn, t) -> str:
    """One teammate on the team sheet: what it is good at, what it can do, its room and whether it is free."""
    s = settings(turn.rules)
    bits = [_route_word(turn, t.name) or "everyday"]
    rt = route_team(turn)
    good = dict(s["good_at"], **((rt.get("contract") or {}).get("good_at") or {}))
    if t.name in (rt.get("helpers") or []):
        bits.append("a helper on your team" if contract(turn) is not None else "on this route's team")
    elif t.name == rt.get("reviewer"):
        bits.append("your team's reviewer" if contract(turn) is not None else "this route's reviewer")
    elif t.name in (((rt.get("contract") or {}).get("stand_ins")) or {}).values():
        bits.append("standing in for a member who is not answering")
    if good.get(t.name):
        bits.append(good[t.name])
    if turn.policy is not None:
        caps = [c for c in ("vision", "tools", "long_context") if turn.policy.eligible(t.name, c, c == "tools")]
        if caps:
            bits.append(", ".join(caps))
    try:
        n = turn._engine(t).n_ctx()
        if n:
            bits.append(f"{n // 1024}K context")
    except Exception:  # noqa: BLE001 - a line of a tool's description only
        pass
    adm = getattr(turn, "admission", None)
    if adm is not None and t.address:
        from nebula.admission import RANK
        bits.append("free now" if adm.gate(t).free(RANK["agent"]) else f"busy, about {adm.expected_wait(t, RANK['agent']):.0f} s wait")
    price = getattr(t, "cost_per_mtoken", None)
    bits.append(f"cloud, ${price:g} per million tokens" if price else "local")
    return f"- {t.name}: " + "; ".join(bits)


def _work_specs(turn, names: list[str]) -> list[dict]:
    mates = [t.name for t in teammates(turn)]
    sheet = "\n".join(sheet_line(turn, t) for t in teammates(turn))
    out = []
    for name in names:
        if name == DELEGATE:
            desc = ("Give parts of this task to teammates. They work in parallel while you wait (wait all, the default), "
                    "and their results come back to you to put into one answer. Each teammate sees only its own task, "
                    "the user's request and the team's notes, never this conversation: write each task in full. Use it "
                    "for parts that are separate and worth a model's own attention; do small things yourself. "
                    "Your team:\n" + sheet)
            params = {"type": "object", "properties": {
                "items": {"type": "array", "maxItems": MAX_ITEMS_PER_CALL, "items": {"type": "object", "properties": {
                    "id": {"type": "string", "description": "your short name for this item, e.g. outline-check"},
                    "to": {"type": "string", "enum": mates},
                    "task": {"type": "string", "description": "the task, in full"},
                    "skills": {"type": "array", "items": {"type": "string"},
                               "description": "skill names attached to this item's turn"},
                    "depends_on": {"type": "array", "items": {"type": "string"},
                                   "description": "ids of items whose results this one needs first"},
                    "folder": {"type": "string", "description": "a folder inside yours for this part (the toolbox): the "
                               "teammate works there and nowhere else"},
                    "continue": {"type": "string", "description": "a finished item's id on the same teammate, from this "
                                 "task or an earlier one in this chat: Codex or Claude Code goes on in that item's own "
                                 "session (it remembers its work), so the task needs only what is new"}},
                    "required": ["id", "to", "task"]}},
                "wait": {"type": "string", "enum": ["all", "any", "none"]}}, "required": ["items"]}
        elif name == COLLECT:
            desc = "The results of items you delegated, whole (every item of this task when ids is left out)."
            params = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}},
                                                       "wait": {"type": "string", "enum": ["all", "any", "none"]}}}
        elif name == ASK:
            desc = ("Ask a teammate one short question and get a short answer (a fact, a check, a second opinion). It "
                    "does not see this conversation: put what it needs in the question. Your team:\n" + sheet)
            params = {"type": "object", "properties": {"to": {"type": "string", "enum": mates},
                                                       "question": {"type": "string"}}, "required": ["to", "question"]}
        elif name == REVIEW:
            rv = reviewer_of(turn) or reviewer_of(turn, any_item=True)
            desc = ("Have the reviewer check work before you answer: an item you delegated (by its id; when it is rejected "
                    f"it goes back to its teammate for fixes, at most {limits(turn)['max_rework']} times), or your "
                    "own draft (work). Returns the verdict: approve, or reject with the blocking issues. Reviewer: "
                    + (rv.name if rv is not None else "none") + ".")
            params = {"type": "object", "properties": {
                "item": {"type": "string", "description": "the id of an item you delegated"},
                "work": {"type": "string", "description": "or the work itself, in full"},
                "criteria": {"type": "string", "description": "what it must meet"},
                "to": {"type": "string", "enum": mates, "description": "another reviewer than the usual one"}},
                "required": ["criteria"]}
        elif name == ASK_OWNER:
            desc = ("Ask the owner (the person this Nebula belongs to) about something only they can decide: an action "
                    "that cannot be undone, a choice of goal, a missing fact. They get a card with your options; this "
                    "does not wait. Carry on with what does not depend on it, and say in your answer what waits on them.")
            params = {"type": "object", "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_OPTIONS}}, "required": ["question"]}
        else:
            desc = ("Leave a note for the whole team on this task: a fact, a decision or a constraint every teammate "
                    "should know. Short.")
            params = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
    return out


def tool_specs(turn) -> list[dict]:
    """The team tools for this turn's model: hand_up to the models above it on the ladder, hand_down to those below
    (the lead's), then the work tools its depth allows."""
    return (_move_specs(turn) if offered(turn) else []) + _work_specs(turn, work_offered(turn))


def _move_specs(turn) -> list[dict]:
    out = []
    for up, name in ((True, HAND_UP), (False, HAND_DOWN)):
        targets = _targets(turn, up)
        if not targets:
            continue
        listing = ", ".join(f"{n} ({_route_word(turn, n)})" if _route_word(turn, n) else n for n in targets)
        if up:
            desc = ("Hand this answer to a larger model when the request is beyond you: hard reasoning, long or tricky "
                    "code, deep analysis, a long document. The larger model continues this same answer for the user; "
                    "do not call it for simple requests. Models above you: " + listing + ".")
        else:
            desc = ("After you have done the hard part and what remains is simple, hand the next turns to a smaller, "
                    "faster model. Finish your answer after calling it. Models below you: " + listing + ".")
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": {
            "type": "object", "properties": {"to": {"type": "string", "enum": targets, "description": "the model to hand to"},
                                             "why": {"type": "string", "description": "one short sentence: why"}},
            "required": ["to", "why"]}}})
    return out


def offer(turn, body: dict) -> list[str]:
    """Adds the team tools to the engine's body when this turn is offered them; the names offered."""
    if body.get("tool_choice") == "none" or not _team_ok(turn):
        return []
    specs = tool_specs(turn)
    have = {t.get("function", {}).get("name") for t in body.get("tools") or [] if isinstance(t, dict)}
    specs = [t for t in specs if t["function"]["name"] not in have]
    if specs:
        body["tools"] = list(body.get("tools") or []) + specs
    return [t["function"]["name"] for t in specs]


def check(turn, name: str, args: dict) -> tuple[bool, str, object]:
    """Halo's ruling on a hand move: (allowed, the reason, the target tier)."""
    s = limits(turn)
    up = name == HAND_UP
    to = str(args.get("to") or "").strip()
    if name not in (HAND_UP, HAND_DOWN):
        return False, f"there is no team tool {name!r}", None
    if turn.pinned:
        return False, "the request named its model; it is not moved", None
    targets = _targets(turn, up)
    if to not in targets:
        where = "above" if up else "below"
        return False, (f"{to or 'that model'} is not {where} you on this request's ladder"
                       + (f"; you may hand to {', '.join(targets)}" if targets else f"; there is none {where} you")), None
    tier = next(t for t in turn.ladder if t.name == to)
    if turn.registry is not None and not turn.registry.up(to):
        return False, f"{to} is not answering now; carry on yourself", None
    moves = turn.ledger.team_moves(turn.opened.session_id)
    live = [m for m in moves if m.get("ruling") == "allowed" and m.get("mode") == "on"]
    if s["mode"] == "on" and len(live) >= s["max_hand_moves"]:
        return False, f"this conversation already used its {s['max_hand_moves']} hand moves; carry on yourself", None
    me = turn.tier.name
    if len(live) >= 2 and (live[-1]["from"], live[-1]["to"]) == (live[-2]["to"], live[-2]["from"]) \
            and (me, to) == (live[-2]["from"], live[-2]["to"]):
        return False, f"the work already went back and forth between {me} and {to}; carry on yourself", None
    if up:
        from nebula.sizing import holds
        if not holds(turn._context_of(tier), getattr(turn, "est", 0), turn.max_tokens, turn.rules.output_reserve_tokens):
            return False, f"{to} cannot hold this conversation; carry on yourself", None
    return True, "allowed", tier


def record(turn, name: str, args: dict, allowed: bool, reason: str, extra: dict | None = None) -> dict:
    s = settings(turn.rules)
    action = name[len(PREFIX):] if is_team(name) else name
    if name == DELEGATE:
        items = [i for i in args.get("items") or [] if isinstance(i, dict)]
        to = ",".join(str(i.get("to") or "") for i in items)
        why = "; ".join(str(i.get("task") or "")[:120] for i in items)
        extra = {"items": [{"id": str(i.get("id") or ""), "to": str(i.get("to") or "")} for i in items], **(extra or {})}
    elif name == REVIEW:
        rv = reviewer_of(turn, str(args.get("to") or "").strip() or None)
        to = rv.name if rv is not None else str(args.get("to") or "")
        why = (f"item {args['item']}: " if args.get("item") else "") + str(args.get("criteria") or "")
    else:
        to = str(args.get("to") or "")
        why = str(args.get("why") or args.get("question") or args.get("text") or "")
    payload = {"action": action, "from": turn.tier.name if turn.tier else None, "to": to, "why": why[:500],
               "mode": s["mode"], "ruling": "allowed" if allowed else "refused", "reason": reason, **(extra or {})}
    c = contract(turn)
    if c is not None:                   # the contract it was ruled under (a task card shows its refusals)
        payload["contract"] = {"team": c.get("id"), "version": c.get("version")}
    turn.ledger.team_action(turn.opened.session_id, turn.attempt_id, payload)
    return payload


WORK_DONE_STATES = ("done", "failed", "cancelled")


def earlier_item(turn, cont: str, done: list[dict] | None = None) -> dict | None:
    """The item a delegated item's `continue` names: this task's item by its id; else an earlier task's item, by
    its full item id or by its id in the same chat (the newest with that id). Executed items only, the owner's own."""
    from nebula import teamwork
    cont = (cont or "").strip()
    if not cont or turn.ledger is None:
        return None
    task_id, session_id = teamwork.task_of(turn)
    if done is None:
        done = turn.ledger.work_items(task_id=task_id)
    prev = next((i for i in done if i.get("label") == cont and i["mode"] == "execute"), None)
    if prev is not None:
        return prev
    row = turn.ledger.work_item(cont)
    if row is not None and row["mode"] == "execute" and row.get("tenant_id") == turn.ledger.tenant:
        return row
    earlier = [i for i in turn.ledger.work_items(session_id=session_id)
               if i.get("label") == cont and i["mode"] == "execute" and i["task_id"] != task_id]
    return earlier[-1] if earlier else None


def check_work(turn, name: str, args: dict) -> tuple[bool, str]:
    """Halo's ruling on a work call (delegate, ask, collect, note): (allowed, the reason)."""
    from nebula import teamwork
    s = limits(turn)
    if name not in work_offered(turn):
        return False, f"{name} is not yours to call here"
    if name == NOTE:
        text = str(args.get("text") or "").strip()
        if not text:
            return False, "the note is empty"
        return (True, "allowed") if len(text) <= MAX_NOTE else (False, f"a note is at most {MAX_NOTE} characters")
    if name == COLLECT:
        return True, "allowed"
    if name == ASK_OWNER:
        q = str(args.get("question") or "").strip()
        opts = args.get("options") or []
        if not q:
            return False, "the question is empty"
        if not isinstance(opts, list) or len(opts) > MAX_OPTIONS or not all(isinstance(o, str) and 0 < len(o.strip()) <= 80 for o in opts):
            return False, f"options are up to {MAX_OPTIONS} short choices"
        waiting = [i for i in turn.ledger.work_items(task_id=teamwork.task_of(turn)[0])
                   if i["mode"] == "ask_owner" and i["state"] == "awaiting_owner"]
        if len(waiting) >= MAX_OWNER_ASKS:
            return False, f"{len(waiting)} questions already wait on the owner in this task; carry on and say what waits"
        return True, "allowed"
    mates = {t.name: t for t in teammates(turn)}
    task_id = teamwork.task_of(turn)[0]
    done = turn.ledger.work_items(task_id=task_id)
    spent = teamwork.capped_tokens(turn, done)
    if spent >= s["max_task_tokens"]:
        return False, f"this task used its {s['max_task_tokens']} tokens; put the results together and answer now"
    if teamwork.elapsed(task_id) >= teamwork.time_limit(turn):
        return False, "this task is out of time; put the results together and answer now"
    cloud = teamwork.cloud_spent(turn, done)
    if name == REVIEW:
        label = str(args.get("item") or "").strip()
        item = next((i for i in done if i.get("label") == label and i["mode"] == "execute"), None) if label else None
        if label and item is None:
            return False, f"no delegated item {label} in this task"
        to = str(args.get("to") or "").strip() or None
        rv = reviewer_of(turn, to, item=item)
        if rv is None:
            if not to and not label and reviewer_of(turn, any_item=True) is not None:
                return False, "you are the reviewer here: check your own draft yourself, or name a teammate (to)"
            if not to and not label and contract(turn) is not None and not lead_may_review(turn):
                return False, "under your team's contract the reviewer checks the work, not you: name a teammate (to)"
            return False, ("there is no reviewer: set one under Routes › Starweave, or name a teammate (to)"
                           if not to else f"{to} is not on your team now")
        if not str(args.get("criteria") or "").strip():
            return False, "say what the work must meet (criteria)"
        if label:
            if item["state"] != "done":
                return False, f"item {label} is {item['state']}; review it once it is done (team__collect)"
            if item["model"] == rv.name:
                return False, f"{rv.name} did item {label}; another model reviews it"
        elif not str(args.get("work") or "").strip():
            return False, "give the item's id or the work itself"
        if getattr(rv, "cost_per_mtoken", None) and cloud >= s["cloud_cap_usd"]:
            return False, f"cloud cap reached for this task (${s['cloud_cap_usd']:g})"
        return True, "allowed"
    if name == ASK:
        wanted = [{"to": str(args.get("to") or ""), "task": str(args.get("question") or "")}]
    else:
        wanted = [i for i in args.get("items") or [] if isinstance(i, dict)]
        if not wanted:
            return False, "no items: give each one an id, a teammate (to) and its task"
        if len(wanted) > MAX_ITEMS_PER_CALL:
            return False, f"at most {MAX_ITEMS_PER_CALL} items in one call"
    have = {i["label"] for i in done if i.get("label")}
    seen: set[str] = set()
    for n, item in enumerate(wanted, 1):
        to, task = str(item.get("to") or "").strip(), str(item.get("task") or "").strip()
        label = str(item.get("id") or "").strip() if name == DELEGATE else "ask"
        who = f"item {label or n}"
        if to in chain_of(turn):
            return False, f"{who}: {to} is already up this task's chain; give it to another teammate or do it yourself"
        if to not in mates:
            names_ = ", ".join(mates) or "none"
            return False, f"{who}: {to or 'that model'} is not on your team now; your team: {names_}"
        if turn.registry is not None and not turn.registry.up(to):
            return False, f"{who}: {to} is not answering now"
        if not task:
            return False, f"{who}: the task is empty"
        if any(i["model"] == to and i["task_text"].strip() == task for i in done):
            return False, f"{who}: {to} already had this same task in this task; use its result (team__collect)"
        if getattr(mates[to], "cost_per_mtoken", None) and cloud >= s["cloud_cap_usd"]:
            return False, f"{who}: cloud cap reached for this task (${s['cloud_cap_usd']:g})"
        if name == DELEGATE:
            skills = item.get("skills") or []
            if not isinstance(skills, list) or not all(isinstance(s, str) and s for s in skills):
                return False, f"{who}: skills must be a list of skill names"
            if not label:
                return False, f"item {n}: give it an id"
            if label in seen or label in have:
                return False, f"item id {label} is used twice"
            for d in item.get("depends_on") or []:
                if str(d) not in seen and str(d) not in have:
                    return False, f"item {label}: depends_on {d}, which is not an item before it"
            sub_folder = str(item.get("folder") or "").strip()
            if sub_folder:
                from nebula import toolbox
                if not toolbox.mode_of(turn):
                    return False, f"item {label}: there is no toolbox folder of yours to give a part of"
                try:
                    toolbox.resolve(toolbox.folder_of(turn), sub_folder)
                except toolbox.BoxError as e:
                    return False, f"item {label}: {e}"
            cont = str(item.get("continue") or "").strip()
            if cont:
                prev = earlier_item(turn, cont, done)
                if prev is None:
                    return False, (f"item {label}: continue {cont}, which is not an earlier item of this task or "
                                   "of an earlier task in this chat")
                if prev["model"] != to:
                    return False, f"item {label}: {cont} was {prev['model']}'s; only that teammate can go on with it"
                if not prev.get("native_session"):
                    return False, (f"item {label}: {cont} has no session to go on with (only Codex and Claude Code "
                                   "keep one, once their item has run)")
                if prev["state"] not in WORK_DONE_STATES:
                    return False, f"item {label}: {cont} is still working; continue it once it is done"
            seen.add(label)
    return True, "allowed"


def lead_may_review(turn) -> bool:
    c = contract(turn)
    return c is None or (c.get("review") or {}).get("lead_reviews", True)


_WARNED: set[str] = set()               # tasks Halo has told they are near their budget


def budget_note(turn) -> str:
    """Once per task under a contract, when its tokens or time pass the contract's warn mark (80%): a line the lead
    reads after its call. Recorded as a team_action (budget_warning)."""
    from nebula import teamwork
    c = contract(turn)
    if c is None or depth_of(turn) or turn.ledger is None or turn.opened is None:
        return ""
    task_id, _sid = teamwork.task_of(turn)
    if task_id in _WARNED:
        return ""
    s = limits(turn)
    items = turn.ledger.work_items(task_id=task_id)
    used_t = teamwork.capped_tokens(turn, items) / max(1, s["max_task_tokens"])
    used_s = teamwork.elapsed(task_id) / max(1.0, teamwork.time_limit(turn))
    used_c = teamwork.cloud_spent(turn, items) / s["cloud_cap_usd"] if s["cloud_cap_usd"] else 0.0
    used = max(used_t, used_s, used_c)
    mark = (c.get("escalation") or {}).get("warn_at") or 0.8
    if used < mark:
        return ""
    _WARNED.add(task_id)
    what = "tokens" if used == used_t else "time" if used == used_s else "cloud budget"
    record(turn, "budget_warning", {"why": f"{int(used * 100)}% of the task's {what}"}, True, "warning")
    return (f"\n\n(Halo: this task has used {int(used * 100)}% of its {what} under your team's contract. Start putting "
            "the results together; at 100% Halo stops new work.)")


def answer(name: str, args: dict, allowed: bool, reason: str, mode: str) -> str:
    """What the model reads back from its team call."""
    to = str(args.get("to") or "")
    if not allowed:
        return f"Not moved: {reason}."
    if mode == "practice":
        return f"Practice mode: the move to {to} was recorded but not made. Carry on and answer yourself."
    if name == HAND_DOWN:
        return f"Noted: after this answer, {to} takes the next turns. Finish your answer now."
    return f"Handed to {to}."


def practice_text(name: str) -> str:
    if name == NOTE:
        return "Practice mode: the note was recorded but not shared. Carry on."
    if name == COLLECT:
        return "Practice mode: nothing was delegated, so there is nothing to collect. Do the work yourself."
    if name == REVIEW:
        return "Practice mode: skipped (practice), no review was run. Check the work yourself and carry on."
    if name == ASK_OWNER:
        return "Practice mode: the question was recorded but the owner was not asked. Say in your answer what you would ask."
    return "Practice mode: skipped (practice), the call was recorded but not run. Do this part yourself and carry on."


def reason_text(frm: str, why: str, partial: str) -> str:
    """The crossing's reason, as the larger model reads it in the packet."""
    text = f"{frm} handed this turn to you: {why or 'it needs a larger model'}."
    partial = (partial or "").strip()
    if partial:
        text += (" The user has already seen the start of the answer below; continue from where it stops, without "
                 "repeating it:\n" + partial[-2000:])
    return text
