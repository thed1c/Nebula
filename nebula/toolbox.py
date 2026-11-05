"""nebula/toolbox.py: the workspace toolbox, offered to every model this Nebula runs (Routes › Toolbox).

A model with the toolbox works in **one folder**: it lists, searches, reads, writes and patches files there, and runs
commands (the tests) in it **inside OpenShell only**. There is no fallback to the host: without OpenShell, `box__run` is
not offered and the reason is shown on the Routes page. Files never leave the folder: every path is resolved and
checked against it, symbolic links included.

Which folder:
- the lead works in the model's own folder (`toolbox.folders[<model>]`), else the toolbox's folder (`toolbox.folder`);
- a Starweave work item works in the folder its lead named for it (`folder`, inside the lead's own), else in its
  model's own folder; a reviewer reads (list, search, read) but does not write or run.

Who gets it: the owner's own requests (never a guest's, a paired Nebula's, or a passthrough), on every model this
Nebula runs, unless the Routes page leaves a model out (`toolbox.off`). Codex and Claude Code have tools of their own
and are not offered it.

Every call is in the ledger: the tool round (what the model asked, what it read back) as any agent tool's, and a
`box_action` event (the tool, the path or command, whether it worked, the bytes or the exit code), as the team tools
record theirs. The chat's team line counts them ("edited 3 files · ran 2 commands").

Commands run in a kept OpenShell sandbox per folder (`nebula-box-<hash>`): the folder mounted read-write at
/sandbox/work, the owner's own user, no network, the image the Routes page names (python:3.12-slim by default).
"""
from __future__ import annotations

import errno
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import threading

LOG = logging.getLogger("nebula.toolbox")

PREFIX = "box__"
LIST, READ, WRITE, PATCH, SEARCH, RUN = (PREFIX + n for n in ("list", "read", "write", "patch", "search", "run"))
READS = (LIST, READ, SEARCH)
TOOLS = (LIST, READ, WRITE, PATCH, SEARCH, RUN)
MODES = ("off", "on")
READ_CHARS = 60_000                     # one read at most (a model asks for a range of lines for more)
LINE_CHARS = 4_000                      # one line at most, in a read or a search (a longer one is cut)
WRITE_BYTES = 2_000_000
LIST_ENTRIES = 400
SEARCH_HITS = 120
RUN_OUT = 12_000                        # the tail of a command's output the model reads
RUN_KEEP = 65_536                       # what the sandbox sends back of a command's output at most (its end)
EXIT_MARK = "__nebula_exit__"
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
DEFAULT_IMAGE = "python:3.12-slim"
WORKDIR = "/sandbox/work"
DEPS = "/sandbox/deps"                  # the packages copy, read-only: what Nebula imports beyond the standard library


class BoxError(ValueError):
    pass


def settings(rules) -> dict:
    s = dict(getattr(rules, "toolbox", None) or {})
    try:
        timeout = max(10, min(int(s.get("run_timeout_s", 900)), 7200))
    except (TypeError, ValueError):
        timeout = 900
    folders = s.get("folders") if isinstance(s.get("folders"), dict) else {}
    try:
        rounds = max(4, min(int(s.get("max_rounds", 60)), 400))
    except (TypeError, ValueError):
        rounds = 60
    return {"mode": s.get("mode") if s.get("mode") in MODES else "off", "max_rounds": rounds,
            "folder": os.path.expanduser(str(s["folder"])) if s.get("folder") else None,
            "folders": {str(k): os.path.expanduser(str(v)) for k, v in folders.items() if v},
            "off": [str(x) for x in s.get("off") or []],
            "callers": [str(x) for x in s.get("callers") or []],     # caller labels whose plain calls get the tools
            "image": str(s.get("image") or DEFAULT_IMAGE), "run_timeout_s": timeout,
            "packages": "off" if s.get("packages") == "off" else "nebula"}


# ------------------------------------------------------------------------------------------------ the packages
def _declared() -> list[str]:
    """The top-level modules of Nebula's own dependencies on this system (pyproject.toml's, markers checked)."""
    import importlib.metadata as md
    reqs = []
    pyproject = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml")
    try:
        import tomllib
        with open(pyproject, "rb") as f:
            reqs = list((tomllib.load(f).get("project") or {}).get("dependencies") or [])
    except (OSError, ValueError, ImportError):
        try:
            reqs = list(md.distribution("nebula-eh").requires or [])
        except md.PackageNotFoundError:
            reqs = ["jinja2"]
    try:
        from packaging.requirements import Requirement
    except ImportError:
        Requirement = None
    by_dist: dict[str, list[str]] = {}
    for mod, dists in md.packages_distributions().items():
        for d in dists:
            by_dist.setdefault(d.lower().replace("_", "-"), []).append(mod)
    out = []
    for r in reqs:
        if Requirement is not None:
            req = Requirement(r)
            if req.marker is not None and not req.marker.evaluate():
                continue
            name = req.name
        else:
            if ";" in r:
                continue
            name = re.split(r"[<>=!~\[ ]", r, 1)[0]
        out += [m for m in by_dist.get(name.lower().replace("_", "-"), []) if not m.startswith("_")]
    return sorted(set(out))


def _copy_module(name: str, dest: str) -> bool:
    import importlib.util
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin or spec.origin in ("built-in", "frozen"):
        return False
    if spec.submodule_search_locations:
        src = list(spec.submodule_search_locations)[0]
        shutil.copytree(src, os.path.join(dest, name), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(spec.origin, os.path.join(dest, os.path.basename(spec.origin)))
    return True


def packages_copy(base_dir: str | None) -> str | None:
    """The folder each sandbox mounts read-only at /sandbox/deps: a copy (never Nebula's own folders, which Podman
    would take over on a first mount) of the packages Nebula needs beyond the standard library, with what they import
    in turn (found by importing each in a bare interpreter), refreshed when their versions change. None when there is
    nothing to copy."""
    import importlib.metadata as md
    import sys
    want = _declared()
    if not want:
        return None
    dest = os.path.join(base_dir or os.path.expanduser("~"), "nebula-box-packages")
    stamp = os.path.join(dest, ".nebula-packages.json")
    def version(m):
        try:
            return md.version((md.packages_distributions().get(m) or [m])[0])
        except md.PackageNotFoundError:
            return "?"
    try:
        with open(stamp, encoding="utf-8") as f:
            have = json.load(f)
        got = have.get("modules", {})
        if have.get("python") == sys.version.split()[0] and set(want) <= set(got) and all(got[m] == version(m) for m in got):
            return dest
    except (OSError, ValueError):
        pass
    os.makedirs(dest, exist_ok=True)
    mods, todo = {}, list(want)
    for _ in range(40):
        while todo:
            m = todo.pop()
            if m not in mods and _copy_module(m, dest):
                mods[m] = version(m)
        probe = ("import sys; sys.path = [sys.argv[1]] + [p for p in sys.path if not p.rstrip('/').endswith(('site-packages', "
                 "'dist-packages'))]\nimport importlib\nfor m in sys.argv[2:]:\n  try: importlib.import_module(m)\n"
                 "  except ModuleNotFoundError as e: print(e.name.split('.')[0])")
        out = subprocess.run([sys.executable, "-I", "-S", "-c", probe, dest, *mods], capture_output=True, text=True, timeout=60)
        missing = [x for x in out.stdout.split() if x and x not in mods]
        if not missing:
            break
        todo = missing
    with open(stamp, "w", encoding="utf-8") as f:
        json.dump({"python": sys.version.split()[0], "modules": mods}, f)
    return dest


_PACKAGES: dict[str, tuple[float, list[str]]] = {}
PACKAGES_RECHECK_S = 600                # looking up the installed packages takes seconds: at most every 10 minutes


def package_dirs(s: dict, base_dir: str | None = None) -> list[str]:
    import time
    if s.get("packages") == "off":
        return []
    key = base_dir or ""
    hit = _PACKAGES.get(key)
    if hit and time.monotonic() - hit[0] < PACKAGES_RECHECK_S:
        return hit[1]
    try:
        d = packages_copy(base_dir)
    except Exception as e:  # noqa: BLE001 - a sandbox without the copy still runs; the page says why
        LOG.warning("toolbox: the packages copy could not be made: %s", e)
        d = None
    _PACKAGES[key] = (time.monotonic(), [d] if d else [])
    return _PACKAGES[key][1]


def check_settings(raw) -> None:
    """Refuses settings that cannot work (the Routes page's Apply)."""
    if raw in (None, {}):
        return
    if not isinstance(raw, dict):
        raise BoxError("toolbox settings are an object")
    if raw.get("mode") not in (None, *MODES):
        raise BoxError(f"toolbox mode is one of {', '.join(MODES)}")
    if raw.get("callers") is not None and (not isinstance(raw["callers"], list) or not all(isinstance(x, str) for x in raw["callers"])):
        raise BoxError("toolbox callers is a list of caller labels")
    if raw.get("packages") not in (None, "nebula", "off"):
        raise BoxError('toolbox packages is "nebula" (a copy of what Nebula imports) or "off"')
    for k in ("folder",):
        if raw.get(k) and not os.path.isabs(os.path.expanduser(str(raw[k]))):
            raise BoxError("the toolbox folder is a full path")
    for m, f in (raw.get("folders") or {}).items():
        if f and not os.path.isabs(os.path.expanduser(str(f))):
            raise BoxError(f"{m}'s toolbox folder is a full path")


# ------------------------------------------------------------------------------------------------ who, and where
def folder_of(turn) -> str | None:
    """The folder this turn works in, or None (no toolbox for it)."""
    s = settings(turn.rules)
    ctx = getattr(turn, "team_ctx", None) or {}
    if ctx.get("folder"):
        return ctx["folder"]
    name = turn.tier.name if turn.tier is not None else None
    return s["folders"].get(name) or s["folder"]


def mode_of(turn) -> str:
    """"write" (every tool), "read" (a reviewer: list, search, read), or "" (none)."""
    s = settings(turn.rules)
    if s["mode"] != "on" or turn.ledger is None or turn.opened is None or turn.tier is None:
        return ""
    if turn.tenant != "owner" or turn.guest or turn.peer is not None or turn.passthrough:
        return ""
    from nebula.team import agent_work
    if not agent_work(turn):                          # agent work only: a plain API call gets no toolbox
        return ""
    if turn.tier.is_command or turn.tier.name in s["off"]:
        return ""
    if turn.policy is not None and not turn.policy.eligible(turn.tier.name, "tools", True):
        return ""
    if turn.policy is not None and turn.policy.schema(turn.tier.name) is not None:
        return ""
    f = folder_of(turn)
    if not f or not os.path.isdir(f):
        return ""
    from nebula import contracts
    level = contracts.tools_level(turn)               # a team task's contract: what this role gets (write, read, none)
    if level == "none":
        return ""
    if level == "read" or (getattr(turn, "team_ctx", None) or {}).get("mode") == "review":
        return "read"
    return "write"


def item_folder(turn, sub: str | None, model: str) -> str | None:
    """A work item's folder: the one its lead named (inside the lead's own), else its model's own folder."""
    s = settings(turn.rules)
    if sub:
        base = folder_of(turn)
        if not base:
            raise BoxError("you have no toolbox folder to give a part of")
        path = resolve(base, sub)
        os.makedirs(path, exist_ok=True)
        return path
    return s["folders"].get(model) or None


def resolve(folder: str, path: str | None) -> str:
    """The real path of `path` inside `folder`; refuses one outside it (.., an absolute path, a link out)."""
    root = os.path.realpath(folder)
    p = str(path or ".").strip() or "."
    if p.startswith("~") or "\x00" in p:
        raise BoxError(f"{p}: a path inside your folder, such as src/app.py")
    if os.path.isabs(p) and not (p == root or p.startswith(root + os.sep)):
        p = p.lstrip("/")                   # "/src/app.py" means the folder's src/app.py
    full = os.path.realpath(os.path.join(root, p))
    if full != root and not full.startswith(root + os.sep):
        raise BoxError(f"{p} is outside your folder")
    return full


def _inside(folder: str, path: str) -> str | None:
    """`path`'s real path when it is inside `folder` (links followed and checked), else None: a walk of the folder
    yields links too, and a link a command made (to ~/.ssh, say) must not be read through."""
    root, real = os.path.realpath(folder), os.path.realpath(path)
    return real if real == root or real.startswith(root + os.sep) else None


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _open_in(folder: str, real: str, flags: int, make_dirs: bool = False) -> int:
    """A file descriptor for `real` (a checked real path inside `folder`), opened one folder at a time from the
    folder itself, never following a link on the way. A check of the path and an open of the same name later can be
    told apart: a command may swap a folder on the way for a link in between (to ~/.ssh, say); here every step is
    opened from the one before it, so a swapped step is refused (ELOOP / ENOTDIR) instead of followed. On a system
    without these flags (Windows) it falls back to the plain open of the checked path."""
    if not (_NOFOLLOW and _DIRECTORY and os.open in os.supports_dir_fd):
        if make_dirs:
            os.makedirs(os.path.dirname(real), exist_ok=True)
        return os.open(real, flags, 0o644)
    root = os.path.realpath(folder)
    parts = [x for x in os.path.relpath(real, root).split(os.sep) if x not in ("", ".")]
    if not parts or ".." in parts:
        raise BoxError(f"{rel(folder, real)}: not a file inside your folder")
    fd = os.open(root, os.O_RDONLY | _DIRECTORY)
    try:
        for part in parts[:-1]:
            try:
                nxt = os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if not make_dirs:
                    raise
                os.mkdir(part, 0o755, dir_fd=fd)
                nxt = os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return os.open(parts[-1], flags | _NOFOLLOW, 0o644, dir_fd=fd)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            raise BoxError(f"{rel(folder, real)}: a link on the way, not followed") from None
        raise
    finally:
        os.close(fd)


def _open_text(folder: str, real: str, errors: str = "strict"):
    """A checked file inside the folder, opened for reading through the folder itself (_open_in)."""
    return os.fdopen(_open_in(folder, real, os.O_RDONLY), encoding="utf-8", errors=errors)


def _write_file(folder: str, real: str, data: bytes) -> None:
    """Writes `data` to a checked file inside the folder, through the folder itself (_open_in). A file with more than
    one name (a hard link) is refused: writing it would change the same file wherever its other names are."""
    fd = _open_in(folder, real, os.O_WRONLY | os.O_CREAT, make_dirs=True)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise BoxError(f"{rel(folder, real)} is not a plain file")
        if st.st_nlink > 1:
            raise BoxError(f"{rel(folder, real)} has other names on this machine (a hard link): it is not written")
        os.ftruncate(fd, 0)
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
    finally:
        os.close(fd)


def _lines(f, limit: int = None):
    """A text file's lines, a very long one cut at `limit` characters and marked " …(line cut)" (the rest of it read
    and dropped piece by piece): a read keeps only what it shows, however long a line is."""
    limit = limit or LINE_CHARS
    while True:
        line = f.readline(limit)
        if not line:
            return
        if len(line) >= limit and not line.endswith("\n"):
            while True:                         # the rest of a too-long line, dropped
                more = f.readline(limit)
                if not more or more.endswith("\n"):
                    break
            line += " …(line cut)\n"
        yield line


def rel(folder: str, full: str) -> str:
    return os.path.relpath(full, os.path.realpath(folder)) or "."


# ------------------------------------------------------------------------------------------------ the tools
def specs(mode: str, can_run: bool, folder: str) -> list[dict]:
    where = f"Your folder: {os.path.basename(folder.rstrip(os.sep)) or folder}. Paths are inside it, such as src/app.py."

    def fn(name, desc, props, req=()):
        return {"type": "function", "function": {"name": name, "description": desc + " " + where,
                                                 "parameters": {"type": "object", "properties": props, "required": list(req)}}}
    out = [
        fn(LIST, "List the files and folders under a path (default: the whole folder, a few levels deep).",
           {"path": {"type": "string"}, "depth": {"type": "integer", "description": "levels to go down (default 3)"}}),
        fn(SEARCH, "Search the files for a regular expression; returns path:line: text for each hit.",
           {"pattern": {"type": "string"}, "path": {"type": "string", "description": "a file or folder (default: all)"},
            "glob": {"type": "string", "description": "only files matching this, such as *.py"}}, ("pattern",)),
        fn(READ, "Read a file, numbered by line. For a long file, read a range.",
           {"path": {"type": "string"}, "start": {"type": "integer", "description": "first line (default 1)"},
            "lines": {"type": "integer", "description": "how many lines (default: as many as fit)"}}, ("path",)),
    ]
    if mode == "write":
        out += [
            fn(WRITE, "Write a whole file (made, or replaced), creating its folders.",
               {"path": {"type": "string"}, "content": {"type": "string"}}, ("path", "content")),
            fn(PATCH, "Change a file in place: each edit replaces one exact piece of its text (it must occur exactly "
               "once) with new text. Read the file first; copy the old text exactly, indentation included.",
               {"path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object", "properties": {
                   "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["old", "new"]}}}, ("path", "edits")),
        ]
        if can_run:
            out.append(fn(RUN, "Run a shell command in your folder, in a sandbox with no network (OpenShell), such as "
                          "the tests: `python3 -m unittest discover -s tests`. Returns the exit code and the end of its output.",
                          {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}, ("command",)))
    return out


def offer(turn, body: dict) -> list[str]:
    """Adds the toolbox to the engine's body when this turn gets it; the names offered."""
    if body.get("tool_choice") == "none":
        return []
    mode = mode_of(turn)
    if not mode:
        return []
    folder = folder_of(turn)
    can_run = mode == "write" and box_for(turn, folder).why_not() is None
    have = {t.get("function", {}).get("name") for t in body.get("tools") or [] if isinstance(t, dict)}
    out = [t for t in specs(mode, can_run, folder) if t["function"]["name"] not in have]
    if out:
        body["tools"] = list(body.get("tools") or []) + out
    return [t["function"]["name"] for t in out]


def is_box(name: str) -> bool:
    return str(name or "").startswith(PREFIX)


def call(turn, name: str, args: dict) -> str:
    """Runs a toolbox call; what the model reads back. Recorded in the ledger either way."""
    folder = folder_of(turn)
    mode = mode_of(turn)
    payload = {"tool": name[len(PREFIX):], "from": turn.tier.name if turn.tier else None, "folder": folder}
    try:
        if not mode:
            raise BoxError("the toolbox is not yours to use here")
        if name not in READS and mode != "write":
            raise BoxError("a reviewer reads; it does not change the files")
        text, extra = _call(turn, folder, name, args if isinstance(args, dict) else {})
        payload.update(ok=True, **extra)
        return text
    except BoxError as e:
        payload.update(ok=False, error=str(e)[:300])
        return f"Not done: {e}"
    except Exception as e:  # noqa: BLE001 - the model reads what went wrong
        LOG.warning("toolbox: %s failed: %s", name, e)
        payload.update(ok=False, error=str(e)[:300])
        return f"Not done: {e}"
    finally:
        try:
            turn.ledger.box_action(turn.opened.session_id, turn.attempt_id, payload)
        except Exception:  # noqa: BLE001
            pass


def _call(turn, folder: str, name: str, a: dict) -> tuple[str, dict]:
    if name == LIST:
        base = resolve(folder, a.get("path"))
        depth = max(1, min(int(a.get("depth") or 3), 8))
        rows = []
        for root, dirs, files in os.walk(base):
            d = rel(base, root).count(os.sep) + (0 if root == base else 1)
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            if d >= depth:
                dirs[:] = []
            for x in dirs:
                rows.append(rel(folder, os.path.join(root, x)) + "/")
            for x in sorted(files):
                p = os.path.join(root, x)
                if _inside(folder, p) is None:
                    rows.append(f"{rel(folder, p)}  (a link out of the folder: not followed)")
                    continue
                try:
                    rows.append(f"{rel(folder, p)}  ({os.path.getsize(p):,} bytes)")
                except OSError:
                    continue
            if len(rows) >= LIST_ENTRIES:
                rows = rows[:LIST_ENTRIES] + [f"… more (list a smaller path)"]
                break
        return "\n".join(sorted(rows)) or "(empty)", {"path": rel(folder, base), "entries": len(rows)}
    if name == READ:
        p = resolve(folder, a.get("path"))
        if not os.path.isfile(p):
            raise BoxError(f"{a.get('path')}: no such file")
        start = max(1, int(a.get("start") or 1))
        want = int(a.get("lines") or 0)
        stop = start + want if want > 0 else None   # the first line not asked for
        out, size, total, cut_at = [], 0, 0, None
        with _open_text(folder, p, errors="replace") as f:
            for n, line in enumerate(_lines(f), 1):    # streamed: only the lines asked for are kept
                total = n
                if n < start or cut_at is not None or (stop is not None and n >= stop):
                    continue
                row = f"{n:>5}  {line.rstrip(chr(13) + chr(10))}"
                if size + len(row) > READ_CHARS:
                    cut_at = n
                    continue
                out.append(row)
                size += len(row) + 1
        if cut_at is not None:
            out.append(f"… cut at line {cut_at - 1} of {total}: read from line {cut_at} for more")
        return ("\n".join(out) if out else f"(lines {start}+ are past its end: it has {total})"), \
            {"path": rel(folder, p), "lines": total}
    if name == SEARCH:
        try:
            rx = re.compile(str(a.get("pattern") or ""))
        except re.error as e:
            raise BoxError(f"the pattern does not read as a regular expression: {e}") from None
        base = resolve(folder, a.get("path"))
        glob = str(a.get("glob") or "")
        hits = []
        for p in ([base] if os.path.isfile(base) else _files(base)):
            if glob and not fnmatch.fnmatch(os.path.basename(p), glob):
                continue
            real = _inside(folder, p)               # a link that leads out of the folder is not followed
            if real is None:
                continue
            try:
                if os.path.getsize(real) > WRITE_BYTES:
                    continue
                with _open_text(folder, real) as f:
                    for n, line in enumerate(_lines(f), 1):
                        if rx.search(line):
                            hits.append(f"{rel(folder, p)}:{n}: {line.rstrip()[:300]}")
                            if len(hits) >= SEARCH_HITS:
                                break
            except (OSError, UnicodeDecodeError):
                continue
            if len(hits) >= SEARCH_HITS:
                hits.append("… more hits (narrow the search)")
                break
        return "\n".join(hits) or "no matches", {"pattern": str(a.get("pattern"))[:200], "hits": len(hits)}
    if name in (WRITE, PATCH):
        return _write(folder, name, a)
    if name == RUN:
        cmd = str(a.get("command") or "").strip()
        if not cmd:
            raise BoxError("the command is empty")
        s = settings(turn.rules)
        timeout = max(5, min(int(a.get("timeout_s") or s["run_timeout_s"]), s["run_timeout_s"]))
        box = box_for(turn, folder)
        why = box.why_not()
        if why:
            raise BoxError(f"commands run only in OpenShell, and {why}")
        code, out = box.run(cmd, timeout)
        tail = out[-RUN_OUT:]
        return (f"exit code {code}\n" + ("… (the start is cut)\n" if len(out) > RUN_OUT else "") + tail), \
            {"command": cmd[:300], "exit_code": code}
    raise BoxError(f"no toolbox tool {name}")


def _write(folder: str, name: str, a: dict) -> tuple[str, dict]:
    """box__write and box__patch, under the folder's lock (never while a command has the folder)."""
    with folder_lock(folder):
        return _write_now(folder, name, a)


def _write_now(folder: str, name: str, a: dict) -> tuple[str, dict]:
    if name == WRITE:
        p = resolve(folder, a.get("path"))
        content = str(a.get("content") if a.get("content") is not None else "")
        data = content.encode("utf-8")
        if len(data) > WRITE_BYTES:
            raise BoxError(f"at most {WRITE_BYTES:,} bytes in one file")
        if os.path.isdir(p):
            raise BoxError(f"{a.get('path')} is a folder")
        new = not os.path.exists(p)
        _write_file(folder, p, data)
        return f"{'Made' if new else 'Wrote'} {rel(folder, p)} ({len(data):,} bytes).", {"path": rel(folder, p), "bytes": len(data), "new": new}
    if name == PATCH:
        p = resolve(folder, a.get("path"))
        if not os.path.isfile(p):
            raise BoxError(f"{a.get('path')}: no such file (box__write makes one)")
        if os.path.getsize(p) > WRITE_BYTES:
            raise BoxError(f"{a.get('path')} is over {WRITE_BYTES:,} bytes: rewrite it with box__write instead")
        with _open_text(folder, p) as f:
            text = f.read(WRITE_BYTES + 1)
        edits = [e for e in a.get("edits") or [] if isinstance(e, dict)]
        if not edits:
            raise BoxError("no edits")
        for n, e in enumerate(edits, 1):
            old, new = str(e.get("old") or ""), str(e.get("new") if e.get("new") is not None else "")
            count = text.count(old) if old else 0
            if count != 1:
                raise BoxError(f"edit {n}: its old text occurs {count} times in {rel(folder, p)}; it must occur exactly once "
                               "(copy it exactly from box__read, with a line more around it). Nothing was changed.")
            text = text.replace(old, new, 1)
        data = text.encode("utf-8")
        if len(data) > WRITE_BYTES:
            raise BoxError(f"the patched file would be over {WRITE_BYTES:,} bytes. Nothing was changed.")
        _write_file(folder, p, data)
        return f"Changed {rel(folder, p)}: {len(edits)} edit{'s' if len(edits) != 1 else ''}.", {"path": rel(folder, p), "edits": len(edits)}
    raise BoxError(f"no toolbox tool {name}")


def _files(base: str):
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(files):
            yield os.path.join(root, f)


# ------------------------------------------------------------------------------------------------ the sandbox
class WorkBox:
    """A folder's kept OpenShell sandbox: the folder read-write at /sandbox/work, the owner's user, no network."""

    def __init__(self, folder: str, image: str = DEFAULT_IMAGE, base_dir: str | None = None, exe: str | None = None,
                 packages: list[str] | None = None):
        self.folder, self.image, self.base_dir, self.exe = os.path.realpath(folder), image, base_dir, exe
        self.packages = list(packages or [])
        # OpenShell takes sandbox names of at most 19 characters: "nebula-box-" and 8 hex digits. With packages the
        # name differs: a sandbox's mounts are fixed when it starts, so mounts of its own mean a sandbox of its own.
        self.name = "nebula-box-" + hashlib.sha256("|".join([self.folder, *self.packages]).encode()).hexdigest()[:8]
        self.lock = threading.Lock()
        self.ready = False
        self.lender: str | None = None   # rootless Podman: the podman that lends the folder to the sandbox's user

    def why_not(self) -> str | None:
        from nebula import openshell
        if openshell.binary(self.exe) is None:
            return "OpenShell is not installed on this machine"
        return openshell.why_not(self.exe)

    def _volume(self, tool: str, folder: str | None = None, prefix: str = "nebula-box-") -> str:
        from nebula import openshell
        folder = folder or self.folder
        vol = prefix + hashlib.sha256(folder.encode()).hexdigest()[:12]
        argv = openshell.volume_argv(folder, vol)
        check = [tool, "volume", "exists" if os.path.basename(tool) == "podman" else "inspect", vol]
        if subprocess.run(check, capture_output=True).returncode != 0:
            subprocess.run([tool] + argv[1:], capture_output=True, check=True)
        return vol

    def ensure(self) -> None:
        from nebula import openshell
        with self.lock:
            if self.ready:
                return
            driver = openshell.gateway_driver(self.exe) or "docker"
            tool = shutil.which("podman" if driver == "podman" else "docker") or (os.environ.get("NEBULA_DOCKER") if driver != "podman" else None)
            if not tool:
                raise BoxError(f"the OpenShell gateway runs on {driver}, which is not installed for this user")
            vol = self._volume(tool)
            pol_dir = os.path.join(self.base_dir or os.path.expanduser("~"), "nebula-policies")
            os.makedirs(pol_dir, exist_ok=True)
            pol = os.path.join(pol_dir, f"{self.name}.yaml")
            with open(pol, "w", encoding="utf-8") as f:
                f.write(openshell.policy_text(os.getuid()))      # the owner's own user: what it writes stays the owner's
            mounts = {"mounts": [{"type": "volume", "source": vol, "target": WORKDIR, "read_only": False}] +
                      [{"type": "volume", "source": self._volume(tool, d, "nebula-pkgs-"), "target": DEPS, "read_only": True}
                       for d in self.packages[:1]]}
            argv = ["sandbox", "create", "--name", self.name, "--from", self.image, "--policy", pol, "--detach",
                    "--driver-config-json", json.dumps({driver: mounts}, separators=(",", ":")), "--", "sleep", "infinity"]
            try:
                openshell.run(argv, self.exe, timeout=600)
            except openshell.OpenShellError as e:
                if "already exists" not in str(e):
                    raise BoxError(f"the sandbox could not start: {e}") from None
            # Rootless Podman maps the sandbox's user (the owner's uid inside) to a sub-uid on the host, and OpenShell
            # runs no sandbox as root (which would map to the owner): files a command writes, and the folder itself
            # at mount, would belong to that sub-uid. So the folder is lent to it for each command and given back
            # after (podman unshare, no sudo); Nebula's own writes there wait meanwhile (folder_lock).
            self.lender = tool if driver == "podman" and os.getuid() != 0 else None
            with folder_lock(self.folder):
                self._give_back()
            self.ready = True

    def _chown(self, ids: str) -> None:
        p = subprocess.run([self.lender, "unshare", "chown", "-R", "-h", ids, self.folder], capture_output=True, text=True)
        if p.returncode != 0:
            raise BoxError(f"the folder's owner could not be set ({ids} in Podman's user namespace): {p.stderr.strip()[:300]}")

    def _lend(self) -> None:
        if self.lender:
            self._chown(f"{os.getuid()}:{os.getgid()}")     # the sandbox's user as the sandbox sees it

    def _give_back(self) -> None:
        if self.lender:
            self._chown("0:0")                              # root in Podman's namespace is the owner on the host
            for d in self.packages[:1]:                     # a first mount hands the packages copy's top folder over too
                subprocess.run([self.lender, "unshare", "chown", "-h", "0:0", d], capture_output=True)

    def script(self, timeout: int) -> str:
        """The shell line the sandbox runs, the command itself passed as $1 (never pasted into it). The sandbox stops the
        command when its time is up and kills what it started (killing the openshell client here would leave it running
        in the sandbox after the folder is given back), and keeps only the end of a long output (a command that prints
        gigabytes stays a few kilobytes here). Its exit code comes back as the last line."""
        env = f'export PYTHONPATH="{DEPS}${{PYTHONPATH:+:$PYTHONPATH}}"; ' if self.packages else ""
        return (f'{env}cd {WORKDIR} || exit 1; '
                f'if command -v timeout >/dev/null 2>&1 && command -v tail >/dev/null 2>&1; then '
                f'{{ timeout -k 5 {timeout} sh -c "$1" 2>&1; printf "\\n{EXIT_MARK} %s\\n" "$?"; }} | tail -c {RUN_KEEP}; '
                f'else echo "this sandbox image has no timeout or tail (coreutils): commands are not run without them" >&2; '
                f'printf "\\n{EXIT_MARK} 125\\n"; fi')

    def run(self, command: str, timeout: float) -> tuple[int, str]:
        from nebula import openshell
        self.ensure()
        exe = openshell.binary(self.exe)
        t = max(1, int(timeout))
        argv = [exe, "sandbox", "exec", "-n", self.name, "--", "sh", "-c", self.script(t), "nebula-box", command]
        with folder_lock(self.folder):
            try:
                self._lend()                        # inside the try: a lend that fails half way is given back too
                p = subprocess.run(argv, capture_output=True, timeout=t + 20)
            except subprocess.TimeoutExpired as e:
                out = (e.stdout or b"") + (e.stderr or b"")
                return 124, out[-RUN_KEEP:].decode("utf-8", "replace") + f"\n(stopped after {t} s)"
            finally:
                self._give_back()
        out = (p.stdout + (b"\n" + p.stderr if p.stderr else b""))[-RUN_KEEP:].decode("utf-8", "replace")
        code = p.returncode
        m = re.search(r"\n?%s (\d+)\n?\s*$" % EXIT_MARK, out)
        if m:
            code, out = int(m.group(1)), out[:m.start()]
        if code in (124, 137) and m:
            out += f"\n(stopped after {t} s)"
        return code, out


_BOXES: dict[str, WorkBox] = {}
_BOX_LOCK = threading.Lock()
_FOLDER_LOCKS: dict[str, "_FolderLock"] = {}


class _FolderLock:
    """One folder's lock: held alone (a command or a write in it), or shared (by a command or a write in a folder
    inside it). The holder alone may take it again (re-entrant)."""

    def __init__(self):
        self.cond = threading.Condition()
        self.shared, self.owner, self.depth = 0, None, 0

    def acquire(self, alone: bool = True) -> None:
        me = threading.get_ident()
        with self.cond:
            if self.owner == me:
                self.depth += 1
                return
            while self.owner is not None or (alone and self.shared):
                self.cond.wait()
            if alone:
                self.owner, self.depth = me, 1
            else:
                self.shared += 1

    def release(self, alone: bool = True) -> None:
        me = threading.get_ident()
        with self.cond:
            if self.owner == me:
                self.depth -= 1
                if not self.depth:
                    self.owner = None
            else:
                self.shared -= 1
            self.cond.notify_all()


class _Locks:
    """The locks a folder takes, always in path order: the folders around it shared, its own and the ones inside it
    alone. A folder and one inside it never run at once; two side by side do."""

    def __init__(self, locks: list):
        self.locks = locks                     # [(lock, alone)]

    def __enter__(self):
        for lock, alone in self.locks:
            lock.acquire(alone)
        return self

    def __exit__(self, *_exc):
        for lock, alone in reversed(self.locks):
            lock.release(alone)


def folder_lock(folder: str) -> _Locks:
    """A folder's lock: a command in its sandbox and Nebula's own writes there never overlap (while a command runs on
    rootless Podman the folder is lent to the sandbox's user, and the owner's writes would be refused). A folder inside
    another (a teammate's, in its lead's) lends some of the same files: it shares the folders around it and has the
    ones inside it alone, in path order, so neither runs while the other does, and two side by side still do."""
    key = os.path.realpath(folder)
    with _BOX_LOCK:
        _FOLDER_LOCKS.setdefault(key, _FolderLock())
        out = []
        for k in sorted(_FOLDER_LOCKS):
            if key.startswith(k + os.sep):
                out.append((_FOLDER_LOCKS[k], False))      # around it: shared
            elif k == key or k.startswith(key + os.sep):
                out.append((_FOLDER_LOCKS[k], True))       # itself and inside it: alone
        return _Locks(out)


def box_for(turn, folder: str):
    """The folder's sandbox (one per folder, kept)."""
    s = settings(turn.rules)
    base = getattr(turn.cfg, "base_dir", None)
    packages = package_dirs(s, base)
    key = "|".join([os.path.realpath(folder), s["image"], *packages])
    with _BOX_LOCK:
        if key not in _BOXES:
            _BOXES[key] = WORKBOX(folder, s["image"], base, packages=packages)
        return _BOXES[key]


WORKBOX = WorkBox                       # tests put a fake here


def view(cfg, rules) -> dict:
    """The Routes page's Toolbox panel: the settings, whether commands can run, and why not."""
    s = settings(rules)
    from nebula import openshell
    why = None
    try:
        why = "OpenShell is not installed on this machine" if openshell.binary() is None else openshell.why_not()
    except Exception as e:  # noqa: BLE001
        why = str(e)
    folders = {"": s["folder"], **s["folders"]}
    return {**s, "can_run": why is None, "why_not": why, "package_modules": _declared() if s["packages"] != "off" else [],
            "folder_ok": {k or "default": bool(v and os.path.isdir(v)) for k, v in folders.items()},
            "models": [t.name for t in cfg.tiers if not t.is_command and not t.peer]}
