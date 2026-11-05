"""nebula/teamwork.py: Starweave's work, run (phase 2). nebula/team.py offers the tools and rules on each call; this
runs the calls Halo allowed.

A work item is one teammate's part of a task, a row in the ledger's `work_items` from the moment it is given. It runs
as an ordinary turn (nebula/turn.py) in a thread of its own: the item's model named (so it is never moved), class
"agent" (your own chat always goes first at every gate), in a child session of its own (key `team:<item_id>`), with
the turn mode's one fixed instruction and the prompt the item needs:

    the user's request (its first 2,000 characters), the item's task, the team's notes so far, and the results of
    the items it depends on; never the conversation.

At most `max_in_flight` items of a task run at once; the others wait their turn. An item that depends on others waits
for them first, and fails when one of them failed. The lead's engine slot is given back while it waits on its team
(turn.run), so a teammate on the same engine is never stuck behind the lead that waits for it.

A task is the request the lead answers (its first attempt), or every request carrying the same X-Nebula-Task. Its
budgets are the task's: tokens (every item's prompt and answer), time since its first item, and cloud spend (items on
a priced model, at its price).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
import weakref

from nebula import names, team, toolbox
from nebula.events import assemble_reply
from nebula.ledger import WORK_FINAL, text_of

LOG = logging.getLogger("nebula.teamwork")

_COND = threading.Condition()           # notified whenever an item ends
_ENDED: set[str] = set()                # items that ended in this process
_SEMS: dict[str, threading.Semaphore] = {}
_STARTED: dict[str, float] = {}         # task -> when its first item was given (monotonic)
_LIVE: dict[str, tuple[threading.Thread, object]] = {}     # item -> (its thread, its ledger), while it runs
_CLOSING: "weakref.WeakSet" = weakref.WeakSet()   # ledgers whose Nebula is closing: no new item starts on them
STOPPED = "Nebula stopped while this piece of work ran (a restart or a quit); hand it out again"
UNFINISHED = ("ready", "running", "in_review", "rework")     # an item a thread of this process works on


def task_of(turn) -> tuple[str, str]:
    """(the task's id, the lead's session id) for any member of the team."""
    ctx = getattr(turn, "team_ctx", None)
    if ctx:
        return ctx["task_id"], ctx["session_id"]
    if not turn.team_task:
        given = names.header(turn.headers, "Task")
        turn.team_task = f"task:{given}" if given else turn.opened.attempt_id
    return turn.team_task, turn.opened.session_id


def elapsed(task_id: str) -> float:
    with _COND:
        t0 = _STARTED.get(task_id)
    return 0.0 if t0 is None else time.monotonic() - t0


def time_limit(turn) -> float:
    s = team.limits(turn)
    return s["max_background_seconds"] if turn.priority == "background" else s["max_task_seconds"]


def cloud_spent(turn, items: list[dict]) -> float:
    """Dollars the task's items spent on priced models."""
    out = 0.0
    for i in items:
        t = turn.cfg.tier(i["model"]) if hasattr(turn.cfg, "tier") else None
        price = getattr(t, "cost_per_mtoken", None) if t is not None else None
        if price:
            out += (i["tokens"] or 0) * float(price) / 1_000_000
    return out


_PRICED: dict[str, dict] = {}          # task -> {priced item running now: the dollars set aside for it}
MIN_PIECE_TOKENS = 1_000                # a priced piece starts only with at least this many tokens of budget


def _price(turn, model: str) -> float:
    t = turn.cfg.tier(model) if hasattr(turn.cfg, "tier") else None
    return float(getattr(t, "cost_per_mtoken", None) or 0) if t is not None else 0.0


def spent_usd(turn, items: list[dict]) -> float:
    """Dollars the task's priced pieces have used, whatever state each is in now (a piece under review or being
    reworked has used its tokens all the same)."""
    return cloud_spent(turn, [i for i in items if (i.get("tokens") or 0) > 0])


def admit_priced(turn, task_id: str, item_id: str, model: str) -> tuple[str | None, int | None]:
    """A priced (cloud) piece starts with a budget of its own: what is left of the task's cloud cap (spent and set
    aside for the pieces running now taken off), shared among the pieces that may still run at once. The budget is
    set aside while it runs and its turn stops at it (its token ceiling: run_child, the turn's rounds), so the
    pieces together stay within the cap; one that would get under MIN_PIECE_TOKENS waits for running pieces to free
    theirs, or is refused when none runs. (None, its tokens) when it may start; else (the reason, None)."""
    lim = team.limits(turn)
    cap, price = lim["cloud_cap_usd"], _price(turn, model)
    if not price:
        return None, None
    deadline = time.monotonic() + max(0.0, time_limit(turn) - elapsed(task_id))
    while True:
        spent = spent_usd(turn, turn.ledger.work_items(task_id=task_id))
        with _COND:
            live = _PRICED.setdefault(task_id, {})
            free = cap - spent - sum(live.values())
            budget = free / max(1, lim["max_in_flight"] - len(live))
            tokens = int(budget / price * 1_000_000) if budget > 0 else 0
            if tokens >= MIN_PIECE_TOKENS:
                live[item_id] = budget
                return None, tokens
            if not live:
                return f"cloud cap reached for this task (${cap:g}: ${spent:.2f} spent)", None
            if time.monotonic() >= deadline:
                return "the task ran out of time before this priced piece could start", None
            _COND.wait(1.0)             # pieces running now: what they do not use comes back when they end


def _priced_done(task_id: str, item_id: str) -> None:
    with _COND:
        live = _PRICED.get(task_id)
        if live is not None:
            live.pop(item_id, None)
            if not live:
                _PRICED.pop(task_id, None)
        _COND.notify_all()


def free_command(turn, model: str) -> bool:
    """A command tier with no price (Codex, Claude Code on a subscription): its tokens are not this machine's nor
    billed per token, so the task's token cap leaves them out (its time limit still holds)."""
    t = turn.cfg.tier(model) if hasattr(turn.cfg, "tier") else None
    return t is not None and t.is_command and not getattr(t, "cost_per_mtoken", None)


def capped_tokens(turn, items: list[dict]) -> int:
    """The task's tokens that count toward its cap: every item's but a free command tier's."""
    return sum(i["tokens"] or 0 for i in items if not free_command(turn, i["model"]))


KEEP_TASKS = 500                        # tasks remembered in this process (their items stay in the ledger)


def _sem(turn, task_id: str) -> threading.Semaphore:
    with _COND:
        if task_id not in _SEMS and len(_STARTED) >= KEEP_TASKS:     # the oldest tasks' in-process state goes
            for old in sorted(_STARTED, key=_STARTED.get)[:KEEP_TASKS // 5]:
                _STARTED.pop(old, None)
                _SEMS.pop(old, None)
            _ENDED.clear()              # an ended item is read from the ledger after this
        if task_id not in _SEMS:
            _SEMS[task_id] = threading.Semaphore(team.limits(turn)["max_in_flight"])
            _STARTED.setdefault(task_id, time.monotonic())
        return _SEMS[task_id]


def _end(item_id: str) -> None:
    with _COND:
        _ENDED.add(item_id)
        _COND.notify_all()


def _ended(ledger, item_id: str) -> bool:
    with _COND:
        if item_id in _ENDED:
            return True
    row = ledger.work_item(item_id)
    return row is None or row["state"] in WORK_FINAL


# --- one call each
def call(turn, name: str, args: dict) -> str:
    """Runs an allowed work call (live); what the model reads back."""
    if name == team.NOTE:
        task_id, session_id = task_of(turn)
        turn.ledger.team_note(session_id, None if turn.team_ctx else turn.attempt_id, turn.tier.name,
                              str(args.get("text") or "").strip())
        return "Noted for the team."
    if name == team.ASK:
        item = _give(turn, label=None, to=str(args["to"]).strip(), task=str(args["question"]).strip(), mode="ask", deps=[])
        _wait(turn, [item], "all")
        row = turn.ledger.work_item(item)
        if row["state"] != "done":
            return f"{args['to']} could not answer: {row.get('error') or row['state']}"
        return _cut(row["result"] or "", team.ASK_TOKENS * 4, None)
    if name == team.DELEGATE:
        given = []
        items = list(args.get("items") or [])
        sessions = [continued_session(turn, i) for i in items]   # looked up before this call adds its own items
        for i, session in zip(items, sessions):
            to = str(i["to"]).strip()
            given.append(_give(turn, label=str(i["id"]).strip(), to=to, task=str(i["task"]).strip(),
                               mode="execute", deps=[str(d) for d in i.get("depends_on") or []],
                               session=session,
                               folder=toolbox.item_folder(turn, str(i.get("folder") or "").strip() or None, to)
                               if toolbox.settings(turn.rules)["mode"] == "on" else None,
                               skills=i.get("skills") or []))
        wait = args.get("wait") if args.get("wait") in ("all", "any", "none") else "all"
        if wait == "none":
            rows = [turn.ledger.work_item(i) for i in given]
            return ("Started: " + ", ".join(f"{r['label']} ({r['model']})" for r in rows) +
                    ". Call team__collect when you need their results.")
        _wait(turn, given, wait)
        return results_text(turn, [turn.ledger.work_item(i) for i in given], whole=False)
    if name == team.REVIEW:
        return review(turn, args)
    if name == team.ASK_OWNER:
        q = str(args.get("question") or "").strip()
        opts = [str(o).strip() for o in args.get("options") or [] if str(o).strip()]
        task_id, session_id = task_of(turn)
        item = turn.ledger.add_work_item(task_id=task_id, session_id=session_id, model="owner", mode="ask_owner", task_text=q,
                                         label=f"owner-{sum(1 for i in turn.ledger.work_items(task_id=task_id) if i['mode'] == 'ask_owner') + 1}",
                                         from_model=turn.tier.name, chain=team.chain_of(turn))
        turn.ledger.update_work_item(item, state="awaiting_owner", verdict={"options": opts, "answer": None, "delivered": False})
        return ("Asked the owner; their answer comes later (a card in their chat). Carry on with everything that does not "
                "need it, and say in your answer what waits on them.")
    if name == team.COLLECT:
        task_id, _sid = task_of(turn)
        rows = turn.ledger.work_items(task_id=task_id)
        wanted = [str(x) for x in args.get("ids") or []]
        if wanted:
            rows = [r for r in rows if r["label"] in wanted or r["item_id"] in wanted]
        rows = [r for r in rows if r["mode"] == "execute" and r["parent_item_id"] == (turn.team_ctx or {}).get("item_id")]
        if not rows:
            return "No delegated items in this task" + (f" named {', '.join(wanted)}" if wanted else "") + "."
        wait = args.get("wait") if args.get("wait") in ("all", "any", "none") else "all"
        if wait != "none":
            _wait(turn, [r["item_id"] for r in rows], wait)
        return results_text(turn, [turn.ledger.work_item(r["item_id"]) for r in rows], whole=True)
    raise ValueError(f"no team work call {name}")


def continued_session(turn, item: dict) -> str | None:
    """The native session a delegated item continues (its `continue`: an earlier item's id, of this task or an
    earlier one in the same chat, on the same command tier), or None."""
    label = str(item.get("continue") or "").strip()
    if not label:
        return None
    prev = team.earlier_item(turn, label)
    return prev.get("native_session") if prev and prev["model"] == str(item.get("to") or "").strip() else None


def review(turn, args: dict) -> str:
    """team__review: the reviewer's verdict on an item (with the rework loop) or on the lead's own work."""
    s = team.limits(turn)
    criteria = str(args.get("criteria") or "").strip()
    label = str(args.get("item") or "").strip()
    to = str(args.get("to") or "").strip() or None
    if not label:
        rv = team.reviewer_of(turn, to)
        v = _review_once(turn, rv.name, str(args.get("work") or ""), criteria, None)
        return verdict_text(v)
    task_id, _sid = task_of(turn)
    item = next(i for i in turn.ledger.work_items(task_id=task_id) if i.get("label") == label and i["mode"] == "execute")
    rv = team.reviewer_of(turn, to, item=item)
    verdicts = []
    for rnd in range(s["max_rework"] + 1):
        turn.ledger.update_work_item(item["item_id"], state="in_review")
        v = _review_once(turn, rv.name, turn.ledger.work_item(item["item_id"])["result"] or "", criteria, item)
        verdicts.append(v)
        turn.ledger.update_work_item(item["item_id"], verdict=v)
        if v.get("verdict") == "approve":
            turn.ledger.update_work_item(item["item_id"], state="done")
            reworks = f" after {rnd} rework{'s' if rnd != 1 else ''}" if rnd else ""
            return f"[{label}] approved by {rv.name}{reworks}.\n" + verdict_text(v) + "\n\nThe approved result:\n" + \
                _cut(turn.ledger.work_item(item["item_id"])["result"] or "", team.RESULT_CHARS, label)
        if v.get("verdict") != "reject" or rnd == s["max_rework"]:
            break
        turn.ledger.update_work_item(item["item_id"], state="rework", rework_count=rnd + 1)
        fixed = _rework(turn, item, v)
        if fixed is None:
            break
        turn.ledger.update_work_item(item["item_id"], result=fixed)
    turn.ledger.update_work_item(item["item_id"], state="done")
    asked = _ask_owner_on_rejects(turn, item, rv.name, verdicts)
    return (f"[{label}] not approved by {rv.name} after {len(verdicts) - 1} rework{'s' if len(verdicts) != 2 else ''}. "
            + (asked or "Decide: accept it as it is, fix it yourself, hand up, or ask the owner.") + "\n\n" +
            "\n\n".join(f"Verdict {n}:\n" + verdict_text(v) for n, v in enumerate(verdicts, 1)))


def _ask_owner_on_rejects(turn, item: dict, reviewer: str, verdicts: list[dict]) -> str:
    """Under a contract that says so (escalation.ask_after_rejects): the same part rejected that many times, Halo asks
    the owner itself (a card, as team__ask_owner makes). What the lead reads, or "" when Halo did not ask."""
    c = team.contract(turn)
    n = int(((c or {}).get("escalation") or {}).get("ask_after_rejects") or 0)
    rejects = sum(1 for v in verdicts if v.get("verdict") == "reject")
    if c is None or not n or rejects < n:
        return ""
    task_id, session_id = task_of(turn)
    waiting = [i for i in turn.ledger.work_items(task_id=task_id) if i["mode"] == "ask_owner" and i["state"] == "awaiting_owner"]
    if len(waiting) >= team.MAX_OWNER_ASKS:
        return ""
    last = (verdicts[-1].get("blocking_issues") or [""])[0][:200] if verdicts else ""
    q = (f"{reviewer} rejected part {item.get('label')} ({item.get('model')}) {rejects} times"
         + (f", last for: {last}" if last else "") + ". What should the team do?")
    opts = ["Accept it as it is", "Keep fixing it", "Stop the task"]
    owner = turn.ledger.add_work_item(task_id=task_id, session_id=session_id, model="owner", mode="ask_owner", task_text=q,
                                      label=f"owner-{sum(1 for i in turn.ledger.work_items(task_id=task_id) if i['mode'] == 'ask_owner') + 1}",
                                      from_model="halo", chain=team.chain_of(turn))
    turn.ledger.update_work_item(owner, state="awaiting_owner", verdict={"options": opts, "answer": None, "delivered": False})
    team.record(turn, team.ASK_OWNER, {"question": q}, True, f"rejected {rejects} times, as the contract says",
                {"by": "halo"})
    return ("Halo asked the owner what to do, as your team's contract says after this many rejections. Carry on with "
            "what does not depend on this part, and say in your answer that it waits on them.")


def verdict_text(v: dict) -> str:
    out = f"verdict: {v.get('verdict')}\nsummary: {v.get('summary') or ''}"
    if v.get("blocking_issues"):
        out += "\nblocking issues:\n" + "\n".join(f"- {b}" for b in v["blocking_issues"])
    if v.get("followups"):
        out += "\nfollowups:\n" + "\n".join(f"- {f}" for f in v["followups"])
    if v.get("error"):
        out += f"\n({v['error']})"
    return out


def _review_once(turn, reviewer: str, work: str, criteria: str, item: dict | None) -> dict:
    text = (f"Criteria:\n{criteria}\n\n" + (f"The task it was given:\n{item['task_text']}\n\n" if item else "") + f"The work:\n{work}")
    where = (turn.ledger.work_item(item["item_id"]).get("folder") if item else None) or \
        (toolbox.folder_of(turn) if toolbox.mode_of(turn) else None)
    rid = _give(turn, label=None, to=reviewer, task=text, mode="review", deps=[],
                parent=item["item_id"] if item else None, folder=where)
    _wait(turn, [rid], "all")
    row = turn.ledger.work_item(rid)
    if row["state"] != "done":
        return {"verdict": "error", "summary": "", "blocking_issues": [], "followups": [],
                "error": f"the review did not finish: {row.get('error') or row['state']}"}
    try:
        return json.loads(row["result"] or "")
    except ValueError:
        return {"verdict": "error", "summary": "", "blocking_issues": [], "followups": [], "error": "the verdict could not be read"}


def _rework(turn, item: dict, verdict: dict) -> str | None:
    text = (f"Your task was:\n{item['task_text']}\n\nYour result:\n{turn.ledger.work_item(item['item_id'])['result'] or ''}"
            "\n\nBlocking issues to fix:\n" + "\n".join(f"- {b}" for b in verdict.get("blocking_issues") or []))
    wid = _give(turn, label=None, to=item["model"], task=text, mode="rework", deps=[], parent=item["item_id"],
                session=turn.ledger.work_item(item["item_id"]).get("native_session"),
                folder=turn.ledger.work_item(item["item_id"]).get("folder"))
    _wait(turn, [wid], "all")
    row = turn.ledger.work_item(wid)
    return row["result"] if row["state"] == "done" else None


def results_text(turn, rows: list[dict], whole: bool) -> str:
    out = []
    for r in rows:
        head = f"[{r['label'] or r['item_id']}] {r['model']}: "
        if r["state"] == "done":
            out.append(head + "done\n" + (r["result"] if whole else _cut(r["result"] or "", team.RESULT_CHARS, r["label"])))
        elif r["state"] in WORK_FINAL:
            out.append(head + f"{r['state']}: {r.get('error') or 'no result'}")
        else:
            out.append(head + "still working; call team__collect later")
    return "\n\n".join(out)


def _cut(text: str, chars: int, label: str | None) -> str:
    text = text or ""
    if len(text) <= chars:
        return text
    more = f" (cut; team__collect with id {label} gives it whole)" if label else " (cut)"
    return text[:chars] + "…" + more


# --- giving an item, and running it
def _give(turn, *, label: str | None, to: str, task: str, mode: str, deps: list[str], parent: str | None = None,
          session: str | None = None, folder: str | None = None, skills=()) -> str:
    task_id, session_id = task_of(turn)
    rows = {r["label"]: r["item_id"] for r in turn.ledger.work_items(task_id=task_id) if r.get("label")}
    dep_ids = [rows[d] for d in deps if d in rows]
    if mode in ("ask", "review", "rework"):
        label = f"{mode}-{sum(1 for k in rows if k.startswith(mode + '-')) + 1}"
    parent = parent or (turn.team_ctx or {}).get("item_id")
    item_id = turn.ledger.add_work_item(task_id=task_id, session_id=session_id, model=to, mode=mode, task_text=task,
                                        label=label, from_model=turn.tier.name, parent_item_id=parent,
                                        depends_on=dep_ids, chain=team.chain_of(turn), skills=skills)
    if session:                         # a command tier continues this native session (nebula/runs.py)
        turn.ledger.update_work_item(item_id, native_session=session)
    if folder:                          # the toolbox folder it works in (nebula/toolbox.py)
        turn.ledger.update_work_item(item_id, folder=folder)
    sem = _sem(turn, task_id)
    with _COND:
        closing = turn.ledger in _CLOSING
        if not closing:
            th = threading.Thread(target=_run_item, args=(turn, item_id, sem), name=f"team-{item_id}", daemon=True)
            _LIVE[item_id] = (th, turn.ledger)
    if closing:
        turn.ledger.update_work_item(item_id, state="failed", error=STOPPED)
        _end(item_id)
    else:
        th.start()
    return item_id


def _wait(turn, item_ids: list[str], how: str) -> None:
    """Until all (or any) of the items ended, or the task's time is up."""
    task_id, _sid = task_of(turn)
    left = max(0.0, time_limit(turn) - elapsed(task_id))
    deadline = time.monotonic() + left
    pick = all if how == "all" else any
    with _COND:
        while not pick(_ended(turn.ledger, i) for i in item_ids):
            rest = deadline - time.monotonic()
            if rest <= 0:
                return
            _COND.wait(min(rest, 1.0))


def _request_text(turn) -> str:
    """The user's request as teammates read it: the lead's last user message, its first 2,000 characters."""
    ctx = getattr(turn, "team_ctx", None)
    if ctx:
        return ctx.get("request") or ""
    for m in reversed(turn.req.get("messages") or []):
        if m.get("role") == "user":
            return text_of(m.get("content"))[:team.REQUEST_CHARS]
    return ""


def _prompt(turn, row: dict) -> str:
    _task, session_id = task_of(turn)
    parts = []
    req = _request_text(turn)
    if req:
        parts.append("The user's request (for context):\n" + req)
    head = {"ask": "The lead's question:\n", "review": "", "rework": ""}.get(row["mode"], "Your task:\n")
    parts.append(head + row["task_text"])
    notes = turn.ledger.team_notes(session_id)
    if notes:
        parts.append("Team notes:\n" + "\n".join(f"- {n.get('text')} ({n.get('by')})" for n in notes))
    deps = [turn.ledger.work_item(d) for d in row["depends_on"]]
    if deps:
        parts.append("Results you build on:\n" + "\n\n".join(f"[{d['label']}] {d['model']}:\n{d['result'] or ''}" for d in deps))
    return "\n\n".join(parts)


def _run_item(lead, item_id: str, sem: threading.Semaphore) -> None:
    ledger = lead.ledger
    try:
        row = ledger.work_item(item_id)
        task_id, _sid = task_of(lead)
        left = max(0.0, time_limit(lead) - elapsed(task_id))
        with _COND:                     # its dependencies first, without holding a place in flight
            deadline = time.monotonic() + left
            while not all(_ended(ledger, d) for d in row["depends_on"]):
                rest = deadline - time.monotonic()
                if rest <= 0:
                    break
                _COND.wait(min(rest, 1.0))
        failed = [d for d in (ledger.work_item(x) for x in row["depends_on"]) if d["state"] != "done"]
        if failed:
            ledger.update_work_item(item_id, state="failed", error=f"it depends on {failed[0]['label']}, which "
                                    f"{'did not finish' if failed[0]['state'] not in WORK_FINAL else failed[0]['state']}")
            return
        if not sem.acquire(timeout=max(0.0, time_limit(lead) - elapsed(task_id))):
            ledger.update_work_item(item_id, state="failed", error="the task ran out of time before it could start")
            return
        priced = bool(_price(lead, row["model"]))
        try:
            if priced:
                why, budget = admit_priced(lead, task_id, item_id, row["model"])
                if why:
                    ledger.update_work_item(item_id, state="failed", error=why)
                    return
                row = dict(row, token_budget=budget)
            ledger.update_work_item(item_id, state="running")
            text, tokens, child_session = run_child(lead, row)
            ledger.update_work_item(item_id, state="done", result=text, tokens=tokens, child_session_id=child_session)
        finally:
            if priced:
                _priced_done(task_id, item_id)
            sem.release()
    except CancelledRun:
        ledger.update_work_item(item_id, state="cancelled", error="cancelled by the owner")
    except Exception as e:  # noqa: BLE001 - an item's failure is the lead's to read
        if ledger in _CLOSING:          # the ledger closed under it: shutdown() has marked it stopped
            return
        LOG.warning("starweave: work item %s failed: %s", item_id, e)
        try:
            ledger.update_work_item(item_id, state="failed", error=str(getattr(e, "message", None) or e)[:500])
        except Exception:  # noqa: BLE001
            pass
    finally:
        with _COND:
            _LIVE.pop(item_id, None)
        _end(item_id)


def shutdown(ledger, wait: float = 3.0) -> list[str]:
    """Before the ledger closes: no new item starts on it, the running ones get `wait` seconds to finish, and the ones
    still running are marked stopped (their answer would come after the ledger closed: lost, and the item left
    "running" for ever). The items it marked."""
    if ledger is None:
        return []
    with _COND:
        _CLOSING.add(ledger)
        live = [(i, th) for i, (th, led) in _LIVE.items() if led is ledger]
    deadline = time.monotonic() + wait
    for _i, th in live:
        th.join(max(0.0, deadline - time.monotonic()))
    stopped = []
    for item_id, th in live:
        if th.is_alive():
            try:
                ledger.update_work_item(item_id, state="failed", error=STOPPED)
                stopped.append(item_id)
            except Exception:  # noqa: BLE001 - a ledger already gone
                pass
            _end(item_id)
    if stopped:
        LOG.warning("starweave: %d work item(s) were still running when Nebula closed: marked stopped", len(stopped))
    return stopped


def prompt_tokens(messages: list) -> int:
    """About how many tokens a prompt is (a third of its characters, a little over the usual four: a budget errs on
    the safe side)."""
    return sum(len(text_of(m.get("content"))) for m in messages) // 3 + 16 * len(messages)


def run_child(lead, row: dict) -> tuple[str, int, str | None]:
    """One item's turn, on its model, in its child session: (the answer, tokens it used, the child session). A
    command tier's item runs detached instead (run_detached)."""
    tier = lead.cfg.tier(row["model"]) if hasattr(lead.cfg, "tier") else None
    if detachable(tier):
        return run_detached(lead, row, tier)
    task_id, session_id = task_of(lead)
    prompt = _prompt(lead, row)
    team_ctx = {"task_id": task_id, "session_id": session_id, "item_id": row["item_id"],
                "depth": team.depth_of(lead) + 1, "chain": list(row["chain"]), "request": _request_text(lead), "mode": row["mode"],
                "folder": row.get("folder"),
                "skills": list(row.get("skills") or []), "token_budget": row.get("token_budget")}
    ctx = dataclasses.replace(lead.ctx, headers={}, input="team", ticks=None, mode=None, reply_headers={},
                              priority="background" if lead.priority == "background" else "agent", team=team_ctx)
    from nebula import contracts
    house = contracts.member_text(team.contract(lead))
    messages = [{"role": "system", "content": team.MODE_TEXT[row["mode"] if row["mode"] in team.MODE_TEXT else "execute"] + house},
                {"role": "user", "content": prompt}]
    req = {"model": row["model"], "messages": messages}
    if row["mode"] == "ask":
        req["max_tokens"] = team.ASK_TOKENS
    budget = row.get("token_budget")
    if budget:                                  # a priced piece: its answer fits what is left of its budget
        room = budget - prompt_tokens(messages)
        if room < 64:
            raise ValueError("cloud cap: this piece's budget does not cover its own prompt")
        req["max_tokens"] = min(req.get("max_tokens") or room, room)
    if row["mode"] == "review":
        req["response_format"] = {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": team.VERDICT_SCHEMA}}
    text, tokens, child = _one_turn(ctx, row["model"], req)
    if row["mode"] == "review":                  # the verdict, checked; asked again once when it does not hold
        v = _verdict(text)
        if v is None:
            again = dict(req, messages=messages + [{"role": "assistant", "content": text},
                                                   {"role": "user", "content": "Return the verdict JSON only, as the schema says: "
                                                    "a reject needs at least one blocking issue."}])
            if budget:
                room = budget - tokens - prompt_tokens(again["messages"])
                if room < 64:
                    raise ValueError("cloud cap: no budget left to ask the reviewer again")
                again["max_tokens"] = room
            text2, tokens2, _c = _one_turn(dataclasses.replace(ctx, team=dict(team_ctx)), row["model"], again)
            tokens += tokens2
            v = _verdict(text2)
        if v is None:
            raise ValueError("the reviewer's verdict could not be read")
        text = json.dumps(v)
    return text, tokens, child


# --- a command tier's item: a detached run (nebula/runs.py, AutoSwarm v2's runner)
def detachable(tier) -> bool:
    """A command tier on this machine (not in an OpenShell sandbox: its calls stay there, under its timeout): Codex,
    Claude Code, or the owner's own command that reads its prompt on stdin."""
    if tier is None or not tier.is_command or tier.sandbox or tier.peer:
        return False
    if tier.cli:
        return tier.cli in ("codex", "claude")
    return "{prompt_file}" not in str(tier.command) and "{output_file}" not in str(tier.command)


def _base_dir(lead) -> str | None:
    return getattr(lead.cfg, "base_dir", None)


def _resume_text(row: dict) -> str:
    """What a continued session is told: only what is new, since the CLI keeps its own history."""
    if row["mode"] == "rework":
        issues = row["task_text"].split("Blocking issues to fix:", 1)
        return ("The reviewer did not approve your result. Fix only these blocking issues, then reply with your corrected "
                "result and what you changed.\n\nBlocking issues to fix:" + issues[1]) if len(issues) == 2 else row["task_text"]
    return "A follow-up from the lead:\n" + row["task_text"]


def run_detached(lead, row: dict, tier) -> tuple[str, int, str | None]:
    """The item as a run of its own: started detached, waited on with no deadline of its own (the lead's wait has the
    task's time; a run left running is recorded when it returns, and team__collect reads it then)."""
    from nebula import runs
    from nebula.commands import DEFAULT_SLOTS
    base = _base_dir(lead)
    session = row.get("native_session")
    from nebula import contracts
    mode_text = team.MODE_TEXT[row["mode"] if row["mode"] in team.MODE_TEXT else "execute"] + contracts.member_text(team.contract(lead))
    review = row["mode"] == "review"
    if review:                                       # a CLI has no response_format: the schema is in the instructions
        mode_text += "\n\n" + REVIEW_JSON
    prompt = _resume_text(row) if session else _prompt(lead, row)
    if tier.cli == "codex" and not session:          # codex has no system flag: the instructions lead the prompt
        prompt = f"System instructions:\n{mode_text}\n\n{prompt}"
    cwd = row.get("folder") or tier.cwd or names.local_path(base, "nebula-work", "cockpit-work")
    writes = bool(getattr(tier, "writes", False) and tier.cwd) and not review     # a reviewer reads, it does not change

    slots = int(tier.slots or DEFAULT_SLOTS)
    while sum(1 for r in runs.runs(base, limit=KEEP_TASKS) if r["tier"] == tier.name and r["state"] == "running") >= slots:
        time.sleep(1.0)                              # the tier's slots, across every task (and Nebula restarts)
    argv, how = runs.run_form(tier.cli, tier.command, tier.model, None if tier.cli == "codex" else mode_text, writes,
                              session, runs.ANSWER)
    run_id = runs.start(base, tier=tier.name, cli=tier.cli, argv=argv, how=how, cwd=cwd, prompt=prompt, session=session,
                        label=row.get("label"), item_id=row["item_id"])
    lead.ledger.update_work_item(row["item_id"], run_id=run_id)
    text, tokens, child = collect_run(lead.ledger, base, row["item_id"], run_id, prompt)
    if not review:
        return text, tokens, child
    v = _verdict(text)
    sid = lead.ledger.work_item(row["item_id"]).get("native_session")
    if v is None and sid:                            # asked again once, in the same session
        argv, how = runs.run_form(tier.cli, tier.command, tier.model, None, False, sid, runs.ANSWER)
        again = "Return the verdict JSON only, as the schema says: " + REVIEW_JSON
        run2 = runs.start(base, tier=tier.name, cli=tier.cli, argv=argv, how=how, cwd=cwd, prompt=again, session=sid,
                          label=row.get("label"), item_id=row["item_id"])
        lead.ledger.update_work_item(row["item_id"], run_id=run2)
        text2, tokens2, _c = collect_run(lead.ledger, base, row["item_id"], run2, again)
        tokens += tokens2
        v = _verdict(text2)
    if v is None:
        raise ValueError("the reviewer's verdict could not be read")
    return json.dumps(v), tokens, child


REVIEW_JSON = ('Answer with one JSON object and nothing else: {"verdict": "approve" or "reject", "summary": "…", '
               '"blocking_issues": ["…"], "followups": ["…"]}. A reject needs at least one blocking issue.')


def collect_run(ledger, base: str | None, item_id: str, run_id: str, prompt: str = "") -> tuple[str, int, str | None]:
    """Waits for a run and reads its result: (the answer, tokens, None). Raises when it did not complete."""
    from nebula import runs
    info = runs.wait(base, [run_id])[0]
    res = info.get("result") or {}
    if info.get("session"):
        ledger.update_work_item(item_id, native_session=info["session"])
    if info["state"] == "interrupted":
        raise RuntimeError("the run stopped without a result (its runner is gone)")
    if res.get("outcome") == "cancelled":
        raise CancelledRun("cancelled")
    if res.get("outcome") != "completed":
        raise RuntimeError(res.get("error") or "the run failed")
    text = res.get("content") or ""
    u = res.get("usage") or {}
    tokens = uncached_tokens(u) if u else (len(prompt) + len(text)) // 4
    return text, tokens, None


def uncached_tokens(u: dict) -> int:
    """A run's tokens without the input its CLI read from cache (nebula/runs.py gives prompt_tokens with the cached
    ones inside, as Codex reports them; Claude Code's cache reads are outside its input already)."""
    return max(0, int(u.get("prompt_tokens") or 0) - int(u.get("cached_tokens") or 0)) + int(u.get("completion_tokens") or 0)


class CancelledRun(RuntimeError):
    pass


def recover(srv) -> int:
    """At start: items whose detached runs outlived the Nebula that started them are collected when they return; an
    item this process's threads would have worked on (no detached run) is marked stopped, as no thread works on it now."""
    ledger = getattr(srv, "ledger", None)
    if ledger is None:
        return 0
    base = getattr(srv.cfg, "base_dir", None)
    stale = ledger.stale_items(UNFINISHED)
    for r in stale:
        ledger.update_work_item(r["item_id"], state="failed", error=STOPPED)
    if stale:
        LOG.warning("starweave: %d work item(s) were left unfinished by the last Nebula: marked stopped", len(stale))
    rows = ledger.running_runs()

    def finish(r):
        # a detached run outlives this Nebula too: if it closes first, the item stays "running" with its run, and
        # the next start collects it again (no write into a closed ledger)
        try:
            try:
                text, tokens, _c = collect_run(ledger, base, r["item_id"], r["run_id"])
                state, extra = "done", {"result": text, "tokens": tokens}
            except CancelledRun:
                state, extra = "cancelled", {"error": "cancelled"}
            except Exception as e:  # noqa: BLE001
                state, extra = "failed", {"error": str(e)[:500]}
            if ledger not in _CLOSING:
                ledger.update_work_item(r["item_id"], state=state, **extra)
        except Exception:  # noqa: BLE001 - the ledger closed between the check and the write
            if ledger not in _CLOSING:
                raise
        finally:
            _end(r["item_id"])
    for r in rows:
        threading.Thread(target=finish, args=(r,), name=f"team-recover-{r['item_id']}", daemon=True).start()
    return len(rows)


def _one_turn(ctx, model: str, req: dict) -> tuple[str, int, str | None]:
    from nebula.turn import Turn
    turn = Turn(ctx, "chat", req)
    body = {k: v for k, v in req.items() if k != "model"}
    engine, body = turn.prepare(body, messages=req["messages"], tools=None, requested_model=model, thinking_requested=None)
    events = list(turn.run(engine, body))
    reply = assemble_reply(events) or {}
    tokens = (turn.done.prompt_tokens or 0) + (turn.done.completion_tokens or 0)
    child = turn.opened.session_id if turn.opened is not None else None
    return text_of(reply.get("content")).strip(), tokens, child


def _verdict(text: str) -> dict | None:
    """A verdict that holds: approve, or reject with at least one blocking issue."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        v = json.loads(t)
    except ValueError:
        a, b = t.find("{"), t.rfind("}")        # a CLI's verdict with words around it
        try:
            v = json.loads(t[a:b + 1]) if 0 <= a < b else None
        except ValueError:
            return None
    if not isinstance(v, dict) or v.get("verdict") not in ("approve", "reject"):
        return None
    issues = [str(b) for b in v.get("blocking_issues") or [] if str(b).strip()]
    if v["verdict"] == "reject" and not issues:
        return None
    return {"verdict": v["verdict"], "summary": str(v.get("summary") or ""), "blocking_issues": issues,
            "followups": [str(f) for f in v.get("followups") or [] if str(f).strip()]}
