"""nebula/runs.py: a command tier's long work, run detached (AutoSwarm v2's runner, rewritten for Nebula).

A Starweave work item for Codex or Claude Code can take an hour. It is not one synchronous engine call with a deadline:
each item is a **run**, a folder of small files and a runner process of its own that outlives the call that started it
(and Nebula itself):

    nebula-runs/<run_id>/
        task.json      what to run: the argv, its folder, the prompt, the CLI, the native session it continues
        handle.json    the runner's pid (and its start time, so a reused pid is not mistaken for it), the child's pid
        stdout.log, stderr.log
        result.json    written once, at the end: outcome (completed, failed, cancelled), the answer, the native
                       session id, the usage, the exit code, the seconds
        cancel.json    asks the runner to stop its child

`start` writes task.json and starts `python -m nebula.runs <folder>` in a session of its own; `inspect` reads the
folder (returned, running, or interrupted: the runner is gone and wrote no result); `wait` polls for result.json with
no deadline of its own (the caller's task time is the only limit, and a run left behind is still collected later);
`cancel` writes cancel.json. Like AutoSwarm, no retry and no deadline is added here.

**Native sessions.** Codex (`codex exec --json`, its `thread.started` event) and Claude Code (`--output-format json`,
its `session_id`) say which session they ran in. A follow-up or a rework continues that session (`codex exec ...
resume <id>`, `claude --resume <id>`) with only the new instructions, as AutoSwarm resumes a child by its native id:
the CLI keeps its own history, so nothing is replayed. One run per native session at a time.

The child gets this Nebula's environment less the variables that would make it think it is inside another agent's
session (CODEX_THREAD_ID, CLAUDE_SESSION_ID, CLAUDECODE), as AutoSwarm's runner does. A sandboxed command tier
(OpenShell) is not run this way: its calls stay in its sandbox, synchronous, under its own timeout.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

KEEP_RUNS = 200                         # finished runs kept; older ones are removed when a new one starts
POLL_S = 0.2
STRIP_ENV = ("CODEX_THREAD_ID", "CLAUDE_SESSION_ID", "CLAUDECODE", "NEBULA_RUN_PARENT")
TAIL = 600
ANSWER = "{answer}"                     # in a run's argv: its answer file, in its own folder


class RunError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runs_dir(base_dir: str | None) -> str:
    from nebula import names
    return names.local_path(base_dir, "nebula-runs", "nebula-runs")


def read_json(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path: str, value) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path), dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ------------------------------------------------------------------------------------------------ the CLI forms
def run_form(cli: str | None, command, model: str | None, system: str | None, writes: bool, session: str | None,
             output_file: str) -> tuple[list[str], str]:
    """(argv with the prompt on stdin, how the answer is read: "codex" | "claude" | "stdout") for a detached run.
    Unlike the chat form (nebula/commands.py), the session is kept so that it can be continued."""
    if cli == "codex":
        argv = ["codex", "exec", "--json", "--skip-git-repo-check", "-s", "workspace-write" if writes else "read-only",
                "--color", "never", "-o", output_file]
        if model:
            argv += ["-m", model]
        return argv + (["resume", session, "-"] if session else ["-"]), "codex"
    if cli == "claude":
        argv = ["claude", "-p", "--output-format", "json"]
        if writes:
            argv += ["--permission-mode", "acceptEdits"]
        if model:
            argv += ["--model", model]
        if system and not session:
            argv += ["--system-prompt", system]
        if session:
            argv += ["--resume", session]
        return argv, "claude"
    if cli:
        raise RunError(f"{cli} has no detached form")
    from nebula import cmdline
    words = cmdline.split(command) if isinstance(command, str) else [str(w) for w in (command or [])]
    if not words:
        raise RunError("a command tier needs a cli or a command")
    return [w.replace("{model}", model or "") for w in words], "stdout"


def parse_codex(stdout: str, answer_file: str) -> tuple[str, str | None, dict | None]:
    """(answer, thread id, usage) from `codex exec --json`'s events and its -o file."""
    session, usage, last = None, None, None
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        session = ev.get("thread_id") or ev.get("session_id") or (ev.get("msg") or {}).get("session_id") or session
        if ev.get("type") == "turn.completed" and isinstance(ev.get("usage"), dict):
            u = ev["usage"]
            usage = {"prompt_tokens": int(u.get("input_tokens") or 0), "completion_tokens": int(u.get("output_tokens") or 0),
                     "cached_tokens": int(u.get("cached_input_tokens") or 0)}
        item = ev.get("item") if isinstance(ev.get("item"), dict) else None
        if ev.get("type") == "item.completed" and item and item.get("type") in ("agent_message", "assistant_message"):
            last = item.get("text") or last
    text = None
    try:
        with open(answer_file, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        pass
    return (text if text and text.strip() else (last or "")).strip(), session, usage


def parse_claude(stdout: str) -> tuple[str, str | None, dict | None]:
    """(answer, session id, usage) from Claude Code's --output-format json."""
    try:
        obj = json.loads(stdout)
    except ValueError:
        return stdout.strip(), None, None
    if isinstance(obj, list):
        obj = next((o for o in reversed(obj) if isinstance(o, dict) and o.get("type") == "result"), obj[-1] if obj else {})
    if not isinstance(obj, dict):
        return stdout.strip(), None, None
    u = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
    cached = int(u.get("cache_read_input_tokens") or 0) if u else 0     # outside input_tokens in Claude's count
    usage = {"prompt_tokens": int(u.get("input_tokens") or 0) + cached, "completion_tokens": int(u.get("output_tokens") or 0),
             "cached_tokens": cached} if u else None
    text = obj.get("result") if isinstance(obj.get("result"), str) else ""
    return text.strip(), obj.get("session_id"), usage


# ------------------------------------------------------------------------------------------------ is it alive
def _start_time(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _alive(handle: dict | None) -> bool:
    if not handle or not handle.get("pid"):
        return False
    pid = int(handle["pid"])
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    st = _start_time(pid)
    if st is not None and handle.get("start") and st != handle["start"]:
        return False                     # the pid was reused by another process
    if st is not None:                   # a finished runner the starter has not reaped yet is not running
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
                if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return False
        except (OSError, IndexError):
            pass
    return True


# ------------------------------------------------------------------------------------------------ the caller's side
def start(base_dir: str | None, *, tier: str, cli: str | None, argv: list[str], how: str, cwd: str, prompt: str,
          session: str | None = None, label: str | None = None, item_id: str | None = None, env: dict | None = None) -> str:
    """Starts a run detached; its id once the runner is up."""
    root = runs_dir(base_dir)
    os.makedirs(root, exist_ok=True)
    prune(base_dir)
    if session:
        for r in runs(base_dir):
            if r.get("session") == session and r["state"] == "running":
                raise RunError(f"the session {session} is still working on {r['label'] or r['run_id']}; wait for it or cancel it")
    run_id = "run_" + uuid.uuid4().hex[:16]
    folder = os.path.join(root, run_id)
    os.makedirs(folder, mode=0o700)
    if not os.path.isdir(cwd):
        os.makedirs(cwd, exist_ok=True)
    answer = os.path.join(folder, "answer.txt")
    argv = [answer if a == ANSWER else a for a in argv]
    task = {"run_id": run_id, "tier": tier, "cli": cli, "argv": argv, "how": how, "cwd": cwd, "prompt": prompt,
            "session": session, "label": label, "item_id": item_id, "answer_file": answer,
            "env": dict(env or {}), "created_at": now()}
    write_json(os.path.join(folder, "task.json"), task)
    with open(os.path.join(folder, "runner.log"), "ab") as log:
        p = subprocess.Popen([sys.executable, "-m", "nebula.runs", folder], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             env=_runner_env(), start_new_session=True, close_fds=True)
    end = time.monotonic() + 30
    while not os.path.exists(os.path.join(folder, "handle.json")) and not os.path.exists(os.path.join(folder, "result.json")):
        if p.poll() is not None:
            raise RunError(f"the runner stopped before it started: see {os.path.join(folder, 'runner.log')}")
        if time.monotonic() > end:
            raise RunError("the runner did not start in 30 s")
        time.sleep(0.02)
    threading.Thread(target=p.wait, daemon=True).start()          # reaped when it ends; it does not need us
    return run_id


def _runner_env() -> dict:
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["NEBULA_RUN_PYTHONPATH"] = env.get("PYTHONPATH", "")          # the child gets the owner's own back
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def folder_of(base_dir: str | None, run_id: str) -> str:
    if not run_id.startswith("run_") or os.sep in run_id or "/" in run_id:
        raise RunError(f"no run {run_id}")
    return os.path.join(runs_dir(base_dir), run_id)


def inspect(base_dir: str | None, run_id: str) -> dict:
    folder = folder_of(base_dir, run_id)
    task = read_json(os.path.join(folder, "task.json"))
    if task is None:
        raise RunError(f"no run {run_id}")
    handle = read_json(os.path.join(folder, "handle.json"), {})
    result = read_json(os.path.join(folder, "result.json"))
    if result is not None:
        state = "returned"
    elif _alive(handle):
        state = "running"
    else:
        result = read_json(os.path.join(folder, "result.json"))     # it may have finished between the two reads
        state = "returned" if result is not None else "interrupted"
    return {"run_id": run_id, "tier": task.get("tier"), "cli": task.get("cli"), "label": task.get("label"),
            "item_id": task.get("item_id"), "state": state, "outcome": (result or {}).get("outcome"),
            "session": (result or {}).get("session") or handle.get("session") or task.get("session"),
            "resumed": task.get("session"), "cwd": task.get("cwd"), "created_at": task.get("created_at"),
            "started_at": handle.get("started_at"), "finished_at": (result or {}).get("finished_at"),
            "seconds": (result or {}).get("seconds"), "usage": (result or {}).get("usage"),
            "error": (result or {}).get("error") or (None if state != "interrupted" else "the runner stopped without a result"),
            "cancel_asked": os.path.exists(os.path.join(folder, "cancel.json")), "result": result}


def wait(base_dir: str | None, run_ids: list[str], *, any_result: bool = False, until: float | None = None,
         stop=None) -> list[dict]:
    """Until all (or any) of the runs returned or were interrupted, or `until` (monotonic) passed, or `stop()` is
    true. No deadline of its own."""
    while True:
        infos = [inspect(base_dir, r) for r in run_ids]
        done = [i for i in infos if i["state"] != "running"]
        if (any_result and done) or len(done) == len(infos):
            return infos
        if (until is not None and time.monotonic() >= until) or (stop is not None and stop()):
            return infos
        time.sleep(POLL_S)


def cancel(base_dir: str | None, run_id: str) -> dict:
    folder = folder_of(base_dir, run_id)
    info = inspect(base_dir, run_id)
    if info["state"] == "running":
        write_json(os.path.join(folder, "cancel.json"), {"run_id": run_id, "at": now()})
        return dict(info, cancel_asked=True)
    return info


def runs(base_dir: str | None, limit: int = 50) -> list[dict]:
    """The latest runs, newest first."""
    root = runs_dir(base_dir)
    try:
        names_ = [n for n in os.listdir(root) if n.startswith("run_")]
    except OSError:
        return []
    names_.sort(key=lambda n: os.path.getmtime(os.path.join(root, n)), reverse=True)
    out = []
    for n in names_[:limit]:
        try:
            out.append(inspect(base_dir, n))
        except RunError:
            continue
    return out


def prune(base_dir: str | None, keep: int = KEEP_RUNS) -> int:
    """Removes the oldest finished runs past `keep` (the drive is not filled with old runs)."""
    root = runs_dir(base_dir)
    try:
        names_ = [n for n in os.listdir(root) if n.startswith("run_")]
    except OSError:
        return 0
    if len(names_) <= keep:
        return 0
    names_.sort(key=lambda n: os.path.getmtime(os.path.join(root, n)))
    gone = 0
    for n in names_[:len(names_) - keep]:
        folder = os.path.join(root, n)
        if os.path.exists(os.path.join(folder, "result.json")) or not _alive(read_json(os.path.join(folder, "handle.json"), {})):
            shutil.rmtree(folder, ignore_errors=True)
            gone += 1
    return gone


# ------------------------------------------------------------------------------------------------ the runner
def _stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait()


def execute(folder: str) -> int:
    """The runner: one run, its child process, its result. Runs as `python -m nebula.runs <folder>`."""
    task = read_json(os.path.join(folder, "task.json"))
    if task is None:
        return 2
    handle = {"pid": os.getpid(), "start": _start_time(os.getpid()), "started_at": now(), "session": task.get("session")}
    write_json(os.path.join(folder, "handle.json"), handle)
    result = {"run_id": task["run_id"], "outcome": "failed", "content": "", "session": task.get("session"), "usage": None,
              "exit_code": None, "error": None, "seconds": None, "finished_at": None}
    started = time.monotonic()
    proc = None
    try:
        if os.path.exists(os.path.join(folder, "cancel.json")):
            result["outcome"] = "cancelled"
            return 0
        env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV and k not in ("PYTHONPATH", "NEBULA_RUN_PYTHONPATH")}
        if os.environ.get("NEBULA_RUN_PYTHONPATH"):
            env["PYTHONPATH"] = os.environ["NEBULA_RUN_PYTHONPATH"]
        env.update({"NO_COLOR": "1", "TERM": "dumb", "NEBULA_RUN_ID": task["run_id"]}, **(task.get("env") or {}))
        argv = list(task["argv"])
        if argv and not os.path.isabs(argv[0]):
            argv[0] = shutil.which(argv[0]) or argv[0]
        out_path, err_path = os.path.join(folder, "stdout.log"), os.path.join(folder, "stderr.log")
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            proc = subprocess.Popen(argv, cwd=task["cwd"], env=env, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                    start_new_session=True)
            handle["child_pid"] = proc.pid
            write_json(os.path.join(folder, "handle.json"), handle)
            try:
                proc.stdin.write((task.get("prompt") or "").encode("utf-8"))
                proc.stdin.close()
            except OSError:
                pass
            cancelled = False
            while proc.poll() is None:
                if not cancelled and os.path.exists(os.path.join(folder, "cancel.json")):
                    cancelled = True
                    _stop(proc)
                    break
                time.sleep(POLL_S)
        result["exit_code"] = proc.returncode
        with open(out_path, encoding="utf-8", errors="replace") as f:
            stdout = f.read()
        with open(err_path, encoding="utf-8", errors="replace") as f:
            stderr = f.read()
        how = task.get("how")
        if how == "codex":
            text, session, usage = parse_codex(stdout, task["answer_file"])
        elif how == "claude":
            text, session, usage = parse_claude(stdout)
        else:
            text, session, usage = stdout.strip(), None, None
        result.update(content=text, session=session or task.get("session"), usage=usage)
        if cancelled:
            result["outcome"] = "cancelled"
        elif proc.returncode == 0:
            result["outcome"] = "completed"
        else:
            result["error"] = f"exited {proc.returncode}: {(stderr or stdout).strip()[-TAIL:]}"
    except BaseException as e:  # noqa: BLE001 - the result says what went wrong; the caller reads it
        result["error"] = str(e)[:TAIL]
        if proc is not None and proc.poll() is None:
            _stop(proc)
    finally:
        result["seconds"] = round(time.monotonic() - started, 2)
        result["finished_at"] = now()
        write_json(os.path.join(folder, "result.json"), result)
    return 0


if __name__ == "__main__":
    from nebula.winproc import hide_consoles
    hide_consoles()                                    # Windows: Codex or Claude Code opens no console window
    sys.exit(execute(sys.argv[1]))
