"""nebula/teamtasks.py: a team task, the owner's "Halo, I want to do this task" made real.

The owner tells Halo a task in the Event Horizon chat. Halo answers small things itself or lets routing pick the model;
work with several parts, code and its tests, or work that wants a review, it hands to a team (its `starweave` tools,
below). A team task is then a thing of its own:

    {"id", "goal", "team", "route", "lead", "helpers", "reviewer", "folder", "state": running | done | failed,
     "started_at", "finished_at", "session_id" (the lead's conversation), "result", "error", "origin" (Halo's chat),
     "contract" (the snapshot of the team's contract it runs under), "contract_version", "checks", "sent_back"}

kept in `<ledger>.tasks.json`. It runs in the background behind the owner's chats: one request to the route's team lead
(rules.starweave.teams), at the background class, with `X-Nebula-Task: <id>` so every work item, review and Codex run
is that task's (nebula/teamwork.py), in its own conversation `starweave-task:<id>`. When it ends Halo says so in the
Mission Log (Halo.notify), with the result. GET /api/starweave/tasks lists them; POST starts one.

The task runs under its team's contract (nebula/contracts.py), a snapshot taken when it starts: an edit to the contract
changes the next task, never this one. When the lead answers, Halo runs the contract's done checks itself (an approved
review, the test command in the folder's sandbox, the report's parts). A check that fails sends the task back to the
lead with what failed, in the same conversation, up to the contract's rework rounds (at least once); after that the
task ends as failed, with the lead's last answer and the checks kept on it.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import weakref

LOG = logging.getLogger("nebula.teamtasks")
MAX_KEPT = 200
GOAL_CHARS = 8000

_LOCK = threading.Lock()
_LIVE: dict[str, tuple[threading.Thread, object]] = {}     # task -> (its thread, its server), while it runs
_CLOSING: "weakref.WeakSet" = weakref.WeakSet()             # servers closing: a late answer no longer writes
STOPPED = "Nebula stopped while this task ran (a restart or a quit). Start it again from its card."


class TaskError(ValueError):
    pass


def _path(server) -> str | None:
    led = getattr(server.cfg, "ledger", None)
    return f"{led}.tasks.json" if led else None


def _load(server) -> list[dict]:
    p = _path(server)
    if not p:
        return list(getattr(server, "_team_tasks", []))
    try:
        with open(p, encoding="utf-8") as f:
            return list(json.load(f).get("tasks") or [])
    except (OSError, ValueError):
        return []


def _save(server, tasks: list[dict]) -> None:
    p = _path(server)
    tasks = tasks[-MAX_KEPT:]
    if not p:
        server._team_tasks = tasks
        return
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"tasks": tasks}, f)
    os.replace(tmp, p)


def _update(server, task_id: str, _closing_ok: bool = False, **fields) -> dict | None:
    if server in _CLOSING and not _closing_ok:   # Nebula is closing: shutdown() has said what became of the task
        return None
    with _LOCK:
        tasks = _load(server)
        for t in tasks:
            if t["id"] == task_id:
                t.update(fields)
                _save(server, tasks)
                return dict(t)
    return None


def tasks(server, task_id: str | None = None) -> list[dict]:
    """Newest first, each with its work items' states (from the ledger) beside it."""
    out = list(reversed(_load(server)))
    if task_id:
        out = [t for t in out if t["id"] == task_id]
    led = getattr(server, "ledger", None)
    for t in out:
        items = led.work_items(task_id=f"task:{t['id']}") if led is not None else []
        t["items"] = [{k: w.get(k) for k in ("item_id", "label", "model", "mode", "state", "tokens", "error")} for w in items]
        t["refusals"] = refusals(led, t.get("session_id")) if t.get("contract") else []
    return out


def refusals(led, session_id: str | None) -> list[dict]:
    """The moves Halo refused in a task's conversation: [{"action", "from", "reason"}] ("refused 2 moves under the
    contract")."""
    if led is None or not session_id:
        return []
    out = []
    try:
        for e in led.events(session_id):
            if e.get("kind") != "team_action":
                continue
            p = json.loads(e.get("payload") or "{}")
            if p.get("ruling") == "refused":
                out.append({"action": p.get("action"), "from": p.get("from"), "reason": p.get("reason")})
    except Exception:  # noqa: BLE001 - a line on a card
        return out
    return out


def team_for(cfg, route: str | None = None, lead: str | None = None, team: str | None = None) -> dict:
    """The team a task runs with (rules.starweave.teams, nebula/contracts.py): the one named (by its id or its name),
    else the first team serving the route, else the one whose lead is `lead`, else the only team there is.
    {"team", "route", "lead", "helpers", "reviewer", "contract"}: the contract is the snapshot the task keeps."""
    from nebula import contracts
    teams = contracts.teams(cfg.rules or {})
    k = None
    if team:
        want = str(team).strip().lower()
        k = next((x for x, c in teams.items() if x.lower() == want or c["name"].lower() == want), None)
        if k is None:
            raise TaskError(f"no team {team!r}; the teams are {', '.join(c['name'] for c in teams.values()) or 'none yet'}")
    elif route and contracts.for_route(cfg.rules or {}, route):
        k = contracts.for_route(cfg.rules or {}, route)[0][0]
    elif lead:
        k = next((x for x, c in teams.items() if c.get("lead") == lead), None)
    elif len(teams) == 1:
        k = next(iter(teams))
    if k is None:
        if lead and cfg.tier(lead) is not None:
            return {"team": None, "route": route, "lead": lead, "helpers": [], "reviewer": None, "contract": None}
        raise TaskError("no team to give it to: set up a team under Star Weaving (or name its lead)")
    c = teams[k]
    if not c.get("lead") or cfg.tier(c["lead"]) is None:
        raise TaskError(f"the {c['name']} has no lead model here")
    snap = contracts.snapshot(k, c)
    return {"team": k, "route": route if route in c["routes"] else (c["routes"][0] if c["routes"] else None),
            "lead": c["lead"], "helpers": list(c["helpers"]), "reviewer": c["reviewer"], "contract": snap}


def _contract_line(cfg, task: dict) -> str:
    from nebula import contracts, team as T
    from nebula.router import Rules
    c = task.get("contract")
    return contracts.lead_text(c, T.settings(Rules.from_dict(cfg.rules or {}))) if c else ""


def _prompt(task: dict, cfg=None) -> str:
    from nebula import contracts
    who = []
    if task.get("helpers"):
        who.append("helpers: " + ", ".join(task["helpers"]))
    if task.get("reviewer"):
        who.append("reviewer: " + task["reviewer"])
    lines = [f"Team task {task['id']} from the owner, given to you as the lead of your team ({'; '.join(who) or 'no helpers'}).",
             "", "The task:", task["goal"], ""]
    if task.get("folder"):
        lines += [f"Work in the folder {task['folder']}: your toolbox (box__list, box__read, box__search, box__write, box__patch,"
                  " box__run) works there, and commands run in its sandbox. Do the work yourself, run the tests, and do not commit.", ""]
    if task.get("contract") and cfg is not None:
        lines += [_contract_line(cfg, task), ""]
    lines += ["Plan it, then do it part by part. Give a teammate work with team__delegate when that is faster or it is better at it;",
              "have the work reviewed with team__review before you call it done (a teammate reviews: never review your own draft);",
              "fix what a review finds. Ask the owner with team__ask_owner only for what only they can decide.",
              "If a tool you need is missing or a call is refused, say exactly what and stop: never say something was done that was not.",
              ""]
    sections = contracts.report_sections(task.get("contract"))
    if sections:
        lines.append("Answer with a part for each of: " + ", ".join(sections) + " (each under its own heading).")
    else:
        lines.append("Answer with: what was done, the files changed, the test results, the review's verdict, and anything left.")
    return "\n".join(lines)


def start(server, goal: str, *, route: str | None = None, lead: str | None = None, folder: str | None = None,
          origin: str | None = None, run: bool = True, team: str | None = None) -> dict:
    """A team task, started in the background; the task (state "running")."""
    from nebula import team as T, toolbox
    from nebula.router import Rules
    cfg = server.cfg
    goal = str(goal or "").strip()
    if not goal:
        raise TaskError("say what the task is")
    if len(goal) > GOAL_CHARS:
        raise TaskError(f"describe the task in at most {GOAL_CHARS} characters (put the rest in a file in its folder)")
    rules = Rules.from_dict(cfg.rules)
    if T.settings(rules)["mode"] != "on":
        raise TaskError("Star Weaving is not on: turn it on under Star Weaving to give work to a team")
    from nebula import contracts
    team_ = team_for(cfg, route, lead, team)
    c = team_.get("contract")
    tb = toolbox.settings(rules)
    if folder:
        folder = os.path.realpath(os.path.expanduser(str(folder)))
        if not os.path.isdir(folder):
            raise TaskError(f"no folder {folder}")
    elif c and c["tools"].get("folder"):
        folder = os.path.realpath(c["tools"]["folder"])
        if not os.path.isdir(folder):
            raise TaskError(f"the {c['name']}'s folder {folder} is not there")
    elif tb["mode"] == "on":
        folder = tb["folders"].get(team_["lead"]) or tb["folder"]
    task = {"id": secrets.token_hex(5), "goal": goal, **team_, "contract_version": c.get("version") if c else None,
            "folder": folder, "state": "running", "started_at": time.time(), "finished_at": None, "session_id": None,
            "result": None, "error": None, "origin": origin, "checks": [], "sent_back": 0}
    contracts.remember(task["id"], c)
    with _LOCK:
        all_ = _load(server)
        all_.append(task)
        _save(server, all_)
    if run:
        th = threading.Thread(target=_run, args=(server, dict(task)), name=f"team-task-{task['id']}", daemon=True)
        with _LOCK:
            _LIVE[task["id"]] = (th, server)
        th.start()
    return task


def _ask_lead(server, task: dict, messages: list[dict]):
    from nebula.dialects import Context
    from nebula.dialects.chat import chat
    cfg = server.cfg
    req = {"model": task["lead"], "stream": False, "messages": messages}
    headers = {"X-Nebula-Task": task["id"], "X-Nebula-Session": f"starweave-task:{task['id']}", "X-Nebula-Caller": "starweave"}
    ctx = Context(cfg, server.ledger, headers, server.engines, server.tools, input="api",
                  caller={"id": None, "label": "starweave", "guest": False}, priority="background",
                  registry=server.registry, admission=server.admission, lot=server.lot, policy=server.policy,
                  yard=server.yard, power=server.power, slots=server.slots, seclog=server.seclog,
                  guard=server.guard, otel=server.otel)
    return chat(server.engines.get(task["lead"]) or server.engine, ctx, req)


def _run(server, task: dict) -> None:
    try:
        _run_task(server, task)
    finally:
        with _LOCK:
            _LIVE.pop(task["id"], None)


def shutdown(server, wait: float = 3.0) -> list[str]:
    """Before Nebula closes: running team tasks get `wait` seconds, then the ones still running are marked stopped
    (otherwise each would stay "running" on its card for ever). The tasks it marked."""
    _CLOSING.add(server)
    with _LOCK:
        live = [(t, th) for t, (th, srv) in _LIVE.items() if srv is server]
    deadline = time.monotonic() + wait
    for _t, th in live:
        th.join(max(0.0, deadline - time.monotonic()))
    stopped = [t for t, th in live if th.is_alive()]
    for t in stopped:
        _update(server, t, _closing_ok=True, state="failed", finished_at=time.time(), error=STOPPED)
    return stopped


def recover(server) -> int:
    """At start: a task the last Nebula left running is marked stopped (no thread runs it now)."""
    with _LOCK:
        tasks = _load(server)
        left = [t for t in tasks if t.get("state") == "running"]
        for t in left:
            t.update(state="failed", finished_at=time.time(), error=STOPPED)
        if left:
            _save(server, tasks)
    return len(left)


def _run_task(server, task: dict) -> None:
    from nebula.dialects import JsonReply
    cfg = server.cfg
    messages = [{"role": "user", "content": _prompt(task, cfg)}]
    sid, checks, sent_back = None, [], 0
    c = task.get("contract")
    rounds = max(1, int(((c or {}).get("review") or {}).get("max_rework") or 0)) if c else 0
    try:
        while True:
            reply = _ask_lead(server, task, messages)
            led = server.ledger.session_by_key(f"starweave-task:{task['id']}") if server.ledger is not None else None
            sid = led["session_id"] if led else sid
            if not (isinstance(reply, JsonReply) and reply.status == 200):
                err = (getattr(reply, "body", None) or {}).get("error", {}) if isinstance(getattr(reply, "body", None), dict) else {}
                done = _update(server, task["id"], state="failed", finished_at=time.time(), session_id=sid, checks=checks,
                               sent_back=sent_back,
                               error=str(err.get("message") if isinstance(err, dict) else err or "the lead did not answer")[:2000])
                break
            text = ((reply.body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            checks = done_checks(server, task, text, sid)
            failed = [k for k in checks if k["ok"] is False]
            if not failed:
                done = _update(server, task["id"], state="done", finished_at=time.time(), session_id=sid, result=text[:20000],
                               checks=checks, sent_back=sent_back)
                break
            if sent_back >= rounds:
                done = _update(server, task["id"], state="failed", finished_at=time.time(), session_id=sid, result=text[:20000],
                               checks=checks, sent_back=sent_back,
                               error="Not done under the team's contract: " + "; ".join(f"{k['check']}: {k['detail']}" for k in failed)[:2000])
                break
            sent_back += 1
            _update(server, task["id"], checks=checks, sent_back=sent_back, session_id=sid)
            messages = messages + [{"role": "assistant", "content": text},
                                   {"role": "user", "content": sent_back_text(failed, sent_back, rounds)}]
    except Exception as e:  # noqa: BLE001 - said on the task and in the Mission Log
        if server in _CLOSING:          # Nebula closed under it (its ledger with it): marked stopped already
            return
        LOG.warning("team task %s failed: %s", task["id"], e)
        done = _update(server, task["id"], state="failed", finished_at=time.time(), error=f"{type(e).__name__}: {e}"[:2000],
                       checks=checks, sent_back=sent_back)
    halo = getattr(server, "halo", None)
    if halo is not None and done is not None and server not in _CLOSING:
        ok = done["state"] == "done"
        extra = ""
        if done.get("checks"):
            extra = "\n\nChecks: " + "; ".join(f"{k['check']} {'passed' if k['ok'] else 'skipped' if k['ok'] is None else 'failed'}"
                                             for k in done["checks"])
        halo.notify(f"Team task {'done' if ok else 'stopped'}: {done['goal'][:80]}",
                    ((done.get("result") if ok else done.get("error")) or "") + extra, task_id=done["id"],
                    session_id=done.get("session_id"), kind="done" if ok else "failed")


def sent_back_text(failed: list[dict], n: int, rounds: int) -> str:
    return ("Halo checked your answer against your team's contract, and it is not done yet:\n"
            + "\n".join(f"- {k['check']}: {k['detail']}" for k in failed)
            + f"\n\nFix this (send-back {n} of {rounds}), then answer again in full. If it cannot be fixed, say exactly why.")


def done_checks(server, task: dict, text: str, session_id: str | None = None) -> list[dict]:
    """The contract's done checks on the lead's answer: [{"check", "ok" (True, False, or None: not checked), "detail"}].
    Each check is a team_action event (done_check) in the lead's conversation."""
    from nebula import contracts
    c = task.get("contract")
    if not c:
        return []
    out = []
    led = getattr(server, "ledger", None)
    items = led.work_items(task_id=f"task:{task['id']}") if led is not None else []
    if c["done"].get("review"):
        ok = contracts.approved(items)
        out.append({"check": "review", "ok": ok,
                    "detail": "a review approved the work" if ok else "no review approved the work: have it reviewed with team__review"})
    if c["done"].get("council"):
        reg = getattr(server, "registry", None)
        miss = contracts.unheard(c, items, (lambda m: not reg.up(m)) if reg is not None else (lambda m: False))
        out.append({"check": "council", "ok": not miss,
                    "detail": "every crew member gave its view" if not miss else
                    "not heard from: " + ", ".join(miss) + " (give each the question with team__delegate)"})
    if c["done"].get("tests"):
        out.append(_tests_check(server, task, c))
    if c["done"].get("report"):
        miss = contracts.missing_sections(text, contracts.report_sections(c))
        out.append({"check": "report", "ok": not miss,
                    "detail": "the report has each part" if not miss else "the report has no part for: " + ", ".join(miss)})
    if led is not None and session_id:
        for k in out:
            try:
                led.team_action(session_id, None, {"action": "done_check", "from": "halo", "to": task.get("lead"),
                                                   "why": k["check"], "mode": "on", "ruling": {True: "passed", False: "failed",
                                                                                              None: "skipped"}[k["ok"]],
                                                   "reason": k["detail"][:500],
                                                   "contract": {"team": c.get("id"), "version": c.get("version")}})
            except Exception:  # noqa: BLE001 - a record beside the check, never the check itself
                pass
    return out


def _tests_check(server, task: dict, c: dict) -> dict:
    from types import SimpleNamespace
    from nebula import contracts, toolbox
    from nebula.router import Rules
    folder = task.get("folder")
    cmd = contracts.test_command(c, folder)
    if cmd is None:
        return {"check": "tests", "ok": None, "detail": "the folder has no tests" if folder else "the task has no folder"}
    if not folder or not os.path.isdir(folder):
        return {"check": "tests", "ok": None, "detail": "the task has no folder to run the tests in"}
    shim = SimpleNamespace(rules=Rules.from_dict(server.cfg.rules or {}), cfg=server.cfg)
    try:
        box = toolbox.box_for(shim, folder)
        why = box.why_not()
        if why:
            return {"check": "tests", "ok": None, "detail": f"the tests could not run: commands run only in OpenShell, and {why}"}
        code, out = box.run(cmd, toolbox.settings(shim.rules)["run_timeout_s"])
    except Exception as e:  # noqa: BLE001 - said on the check
        return {"check": "tests", "ok": None, "detail": f"the tests could not run: {e}"[:500]}
    if code == 5 and c["done"]["tests"] == contracts.AUTO_TESTS:
        return {"check": "tests", "ok": None, "detail": "no tests were found to run"}
    tail = (out or "")[-1500:]
    if code == 0:
        return {"check": "tests", "ok": True, "detail": f"`{cmd}` passed"}
    return {"check": "tests", "ok": False, "detail": f"`{cmd}` failed (exit code {code}):\n{tail}"}


class StarweaveTools:
    """Halo's team tools in the owner's own chat (the Assistant): start a team task, see how they go."""
    status, error = "ready", None

    def __init__(self, server):
        self.server = server

    @property
    def tools(self) -> list[dict]:
        """start_task names the teams there are, each with its purpose (its contract), so the team is picked by what it
        is for; tasks says how they go."""
        from nebula import contracts
        teams = contracts.teams(getattr(self.server.cfg, "rules", None) or {})
        listing = "; ".join(f"{k}: {c['purpose'] or c['name']}" + (f" (e.g. {', '.join(c['examples'][:2])})" if c["examples"] else "")
                            for k, c in teams.items())
        props = {
            "goal": {"type": "string", "description": "the task, complete: what to do, where, how it is checked"},
            "team": {"type": "string", "description": "the team that takes it, by what it is for: " + (listing or "none yet")},
            "route": {"type": "string", "description": "or the route whose team takes it (everyday, larger, writing, critique)"},
            "folder": {"type": "string", "description": "the folder to work in (default: the team's folder, else the lead's)"},
            "why": {"type": "string", "description": "one line for the owner: why a team"}}
        if teams:
            props["team"]["enum"] = list(teams)
        return [
            {"name": "start_task",
             "description": "Give a task to a team (Star Weaving) to run in the background: for work with several parts, code and "
                            "its tests, or anything that should be reviewed. Not for a question you can answer, or one model can. "
                            "Pick the team whose purpose fits.",
             "inputSchema": {"type": "object", "properties": props, "required": ["goal", "why"]}},
            {"name": "tasks", "description": "The team tasks, newest first: state, team, parts, result. Give an id for one.",
             "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}}},
        ]

    def call(self, tool: str, a: dict) -> str:
        from nebula.engine import CALL_TENANT
        if CALL_TENANT.get() != "owner":
            raise PermissionError("team tasks are the owner's")
        if tool == "start_task":
            t = start(self.server, str(a.get("goal") or ""), route=a.get("route") or None, folder=a.get("folder") or None,
                      team=a.get("team") or None)
            team = ", ".join(x for x in [f"{t['lead']} leads", *(f"{h} helps" for h in t["helpers"]),
                                          f"{t['reviewer']} reviews" if t.get("reviewer") else ""] if x)
            name = (t.get("contract") or {}).get("name")
            return (f"Started team task {t['id']}" + (f" with the {name}" if name else "")
                    + f" ({team}{'; in ' + t['folder'] if t.get('folder') else ''}). It runs in the "
                    "background; its result will show in the Mission Log, and the tasks tool says how it goes.")
        if tool == "tasks":
            rows = tasks(self.server, str(a.get("id") or "") or None)[:10]
            if not rows:
                return "no team tasks yet"
            out = []
            for t in rows:
                parts = ", ".join(f"{w['label'] or w['model']} {w['state']}" for w in t["items"]) or "no parts yet"
                out.append(f"{t['id']} [{t['state']}] {t['goal'][:120]} · {t['lead']} leads · {parts}"
                           + (f"\nresult: {t['result'][:1500]}" if t.get("result") and a.get("id") else "")
                           + (f"\nerror: {t['error']}" if t.get("error") else ""))
            return "\n".join(out)
        raise TaskError(f"no tool {tool}")


SYSTEM_TEAMS = ("\n\nYou can give work to a team (Star Weaving): starweave__start_task. Answer questions and small things "
                "yourself, as before. When the owner asks for real work with several parts, code and its tests, or "
                "something that should be reviewed, start a team task with the goal written out in full, then tell the "
                "owner in a line which team took it and why. starweave__tasks says how tasks are going.")
