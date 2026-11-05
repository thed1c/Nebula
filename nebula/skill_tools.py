"""Attach skill summaries to turns and execute explicit skill tools."""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import tarfile
import threading
from pathlib import Path

from nebula import openshell, route_settings, team
from nebula.skills import MAX_BODY_BYTES, ENV_RE, NAME_RE, SkillCatalog, SkillError, check_skill, parse_skill

LOAD = "skill__load"
READ = "skill__read"
RUN = "skill__run"
SUGGEST = "skill__suggest"
_RUN_LOCK = threading.Lock()
_SANDBOX = None
_EXIT_MARK = "__nebula_exit__"
_OUTPUT_CAP = 16 * 1024
_TIMEOUT = 30
_SANDBOX_DIR = "/tmp/nebula-skills"


def settings(rules) -> dict:
    raw = rules if isinstance(rules, dict) else (getattr(rules, "skills", None) or {})
    return {"mode": raw.get("mode", "off"), "match": raw.get("match", "on"),
            "top_k": raw.get("top_k", 3), "routes": raw.get("routes") or {},
            "off": raw.get("off") or [], "allow_env": raw.get("allow_env") or {},
            "hosts": raw.get("hosts") or {}, "dirs": raw.get("dirs") or [],
            "image": raw.get("image") or "ubuntu:24.04"}


def _spec(name: str, description: str, properties: dict, required=()) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)}}}


def _sandbox(turn):
    global _SANDBOX
    if _SANDBOX is None:
        from nebula.sandbox import Sandbox
        _SANDBOX = Sandbox("skills", {"image": settings(turn.rules)["image"], "hosts": [],
                                      "openshell": os.environ.get("NEBULA_OPENSHELL") or "openshell"},
                           base_dir=turn.cfg.base_dir, timeout_s=_TIMEOUT)
    return _SANDBOX


def owner_turn(turn) -> bool:
    """The owner's own request: not a guest's, a paired Nebula's or a passthrough. Skills carry the owner's allowed
    environment values into their scripts, so nobody else is offered them or may call them."""
    return getattr(turn, "tenant", None) == "owner" and not getattr(turn, "guest", False) \
        and getattr(turn, "peer", None) is None and not getattr(turn, "passthrough", False)


def offer(turn, body: dict) -> list[str]:
    """Offer metadata only; the body is available solely through skill__load."""
    turn.skill_catalog = None
    turn.attached_skills = []
    turn.skill_offered = []
    cfg = settings(turn.rules)
    if cfg["mode"] != "on" or body.get("tool_choice") == "none":
        return []
    if not owner_turn(turn):                          # the owner's own requests only, as the toolbox and the team
        return []
    from nebula.team import agent_work
    if not agent_work(turn):                          # agent work only (nebula/team.py)
        return []
    from nebula import contracts
    level = contracts.tools_level(turn)               # a team task's contract: what this role gets (write, read, none)
    if level == "none":
        return []
    may_run = level != "read" and (getattr(turn, "team_ctx", None) or {}).get("mode") != "review"
    library = getattr(turn.ctx, "library", None)
    catalog = SkillCatalog(turn.cfg.base_dir or os.getcwd(), cfg, lookup_on=library is not None,
                           lookup_settings=(getattr(turn.cfg, "raw", {}) or {}).get("lookup"),
                           lookup_library=library)
    turn.skill_catalog = catalog
    attached = []
    route_names = cfg["routes"].get(turn.route) or []
    names = list(route_names) + list(getattr(turn.agent, "skills", None) or []) + list((turn.team_ctx or {}).get("skills") or []) \
        + contracts.skills(turn)                      # the skills a team task's contract gives its members
    existing = {t.get("function", {}).get("name") for t in body.get("tools") or [] if isinstance(t, dict)}
    for name in names:
        skill = catalog.get(name)
        if skill is not None and skill.name not in [s.name for s in attached] and skill.accepts_tools(existing):
            attached.append(skill)
    user = next((m for m in reversed(body.get("messages") or []) if m.get("role") == "user"), None)
    from nebula.ledger import text_of
    matched = catalog.match(text_of(user.get("content")) if user else "", top_k=cfg["top_k"], tools=existing) \
        if cfg["match"] == "on" else []
    skills = list(attached)
    skills.extend(s for s in matched if s.name not in {x.name for x in skills})
    turn.attached_skills = [s.name for s in skills]
    if skills:
        note = "Available skills (load one only when needed):\n" + "\n".join(f"- {s.name}: {s.description}" for s in skills)
        route_settings.apply(body, note)
    specs = [_spec(LOAD, "Load one available skill's instructions and file list.",
                   {"name": {"type": "string"}}, ["name"]),
             _spec(READ, "Read a text file inside an available skill folder (64 KB maximum).",
                   {"name": {"type": "string"}, "file": {"type": "string"}}, ["name", "file"])] if skills else []
    if may_run and any(catalog.get(name).scripts for name in turn.attached_skills) and openshell.why_not() is None:
        specs.append(_spec(RUN, "Run a skill script inside the OpenShell skills sandbox.",
                           {"name": {"type": "string"}, "script": {"type": "string"},
                            "args": {"type": "array", "items": {"type": "string"}}, "stdin": {"type": "string"}},
                           ["name", "script"]))
    if not turn.team_ctx and team.settings(turn.rules)["mode"] == "on" and team._team_ok(turn):
        specs.append(_spec(SUGGEST, "Write a suggested skill for the owner to accept or dismiss.",
                           {"name": {"type": "string"}, "description": {"type": "string"},
                            "body": {"type": "string"}}, ["name", "description", "body"]))
    specs = [s for s in specs if s["function"]["name"] not in existing]
    if specs:
        body["tools"] = list(body.get("tools") or []) + specs
    turn.skill_offered = [s["function"]["name"] for s in specs]
    return turn.skill_offered


def _skill(turn, name: str):
    if name not in turn.attached_skills or turn.skill_catalog is None:
        raise ValueError(f"skill {name!r} is not available on this turn")
    skill = turn.skill_catalog.get(name)
    if skill is None:
        raise ValueError(f"skill {name!r} is off or unavailable")
    return skill


def _inside(folder: str, file: str) -> Path:
    if not isinstance(file, str) or not file or Path(file).is_absolute():
        raise ValueError("file must be a relative path inside the skill folder")
    root = Path(folder).resolve()
    path = (root / file).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError("file must stay inside the skill folder")
    return path


def _suggest(turn, args: dict) -> str:
    name, description, body = (args.get(k) for k in ("name", "description", "body"))
    if not all(isinstance(x, str) for x in (name, description, body)) or not NAME_RE.fullmatch(name or ""):
        raise ValueError("a valid skill name, description and body are required")
    folder = turn.skill_catalog.root / ".suggested" / name
    errors, _warnings = check_skill({"name": name, "description": description}, body, folder)
    if errors:
        raise ValueError("; ".join(errors))
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(exist_ok=False)
    text = "---\nname: " + name + "\ndescription: " + json.dumps(description, ensure_ascii=False) + "\n---\n" + body
    (folder / "SKILL.md").write_text(text, encoding="utf-8")
    return f"Suggested skill {name} was saved for the owner."


def _archive(skill) -> bytes:
    root = Path(skill.folder)
    stream = io.BytesIO()
    total = 0
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for file in ["SKILL.md", *skill.files]:
            path = _inside(skill.folder, file)
            size = path.stat().st_size
            total += size
            if total > 8 * 1024 * 1024:
                raise ValueError("skill files exceed the 8 MB sandbox copy limit")
            info = tarfile.TarInfo(file)
            info.size = size
            info.mode = 0o700 if file in skill.scripts else 0o600
            with path.open("rb") as src:
                archive.addfile(info, src)
    return stream.getvalue()


def _run(turn, skill, args: dict) -> str:
    script = args.get("script")
    argv = args.get("args") or []
    stdin = args.get("stdin") or ""
    if script not in skill.scripts or not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise ValueError("script must be one of this skill's scripts and args must be strings")
    if not isinstance(stdin, str) or len(stdin.encode("utf-8")) > 1024 * 1024:
        raise ValueError("stdin exceeds 1 MB")
    _inside(skill.folder, script)
    if openshell.why_not(max_age_s=0) is not None:
        raise ValueError("OpenShell is unavailable; no script was run")
    cfg = settings(turn.rules)
    with _RUN_LOCK:
        box = _sandbox(turn)
        box.spec["hosts"] = cfg["hosts"].get(skill.name) or []
        box.ensure()
        policy = box.apply_policy()
        if not policy.get("applied"):
            raise ValueError("OpenShell could not apply the skill's network policy")
        target = f"{_SANDBOX_DIR}/{skill.name}"
        rc, _out, err = box.exec(["mkdir", "-p", target], timeout=_TIMEOUT)
        if rc:
            raise ValueError(err.decode("utf-8", "replace")[-300:])
        rc, _out, err = box.exec(["tar", "-x", "-C", target], stdin=_archive(skill), timeout=_TIMEOUT)
        if rc:
            raise ValueError(err.decode("utf-8", "replace")[-300:])
        allowed = set(cfg["allow_env"].get(skill.name) or [])
        declared = {row["name"] for row in skill.environment}
        env = {key: os.environ[key] for key in allowed & declared if key in os.environ}
        path = f"{target}/{script}"
        command = ["python3", path] if script.endswith(".py") else ["sh", path] if script.endswith(".sh") else [path]
        # only the end of what it prints comes back from the sandbox (a script that prints gigabytes stays small here),
        # its exit code as the last line
        bounded = ["sh", "-c", f'{{ "$@" 2>&1; printf "\\n{_EXIT_MARK} %s\\n" "$?"; }} | tail -c {_OUTPUT_CAP}', "nebula-skill"]
        rc, out, err = box.exec(bounded + command + argv, stdin=stdin.encode("utf-8"), env=env, timeout=_TIMEOUT)
        text = (out + err)[-_OUTPUT_CAP:].decode("utf-8", "replace")
        m = re.search(r"\n?%s (\d+)\n?\s*$" % _EXIT_MARK, text)
        if m:
            rc, text = int(m.group(1)), text[:m.start()]
        return f"exit {rc}\n" + text


def call(turn, name: str, args: dict) -> str:
    action = name.removeprefix("skill__")
    skill_name = str(args.get("name") or "") if isinstance(args, dict) else ""
    ok = False
    try:
        if not isinstance(args, dict):
            raise ValueError("skill arguments must be an object")
        if not owner_turn(turn):
            raise ValueError("skills are the owner's own")
        if name == SUGGEST:
            result = _suggest(turn, args)
        else:
            skill = _skill(turn, skill_name)
            if name == LOAD:
                folder = f"{_SANDBOX_DIR}/{skill.name}"
                result = skill.body.replace("${HERMES_SKILL_DIR}", folder).replace("${SKILL_DIR}", folder)
                result += "\nFiles: " + ", ".join(["SKILL.md", *skill.files])
            elif name == READ:
                path = _inside(skill.folder, args.get("file"))
                if path.stat().st_size > MAX_BODY_BYTES:
                    raise ValueError("file exceeds 64 KB")
                result = path.read_text(encoding="utf-8")
            elif name == RUN:
                if RUN not in (getattr(turn, "skill_offered", None) or ()):
                    raise ValueError("skill scripts are not offered on this turn")
                result = _run(turn, skill, args)
            else:
                raise ValueError(f"unknown skill tool {name}")
        ok = True
        return result
    except (OSError, UnicodeError, ValueError) as e:
        return f"Not done: {e}."
    finally:
        if turn.ledger is not None and turn.opened is not None:
            turn.ledger.skill_event(turn.opened.session_id, turn.attempt_id,
                                    {"action": action, "name": skill_name, "ok": ok})


# ---------------------------------------------------------------- the Routes page (GET /api/skills, POST /api/skills/*)

def rules_of(cfg) -> dict:
    """The rules.skills block as the config holds it."""
    return dict((getattr(cfg, "rules", None) or {}).get("skills") or {})


def _names(value, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise ValueError(f"{where} must be a list of names")
    return list(dict.fromkeys(value))


def _base(cfg) -> Path:
    return Path(getattr(cfg, "base_dir", None) or os.getcwd())


def save_settings(cfg, req: dict) -> dict:
    """The page's switches, checked: the rules.skills block to keep. What the request does not mention stays."""
    raw = rules_of(cfg)
    s = settings(raw)
    if "mode" in req:
        if req["mode"] not in ("off", "on"):
            raise ValueError('skills mode is "off" or "on"')
        s["mode"] = req["mode"]
    if "match" in req:
        if req["match"] not in ("off", "on"):
            raise ValueError('skills matching is "off" or "on"')
        s["match"] = req["match"]
    if "top_k" in req:
        try:
            s["top_k"] = max(1, min(3, int(req["top_k"])))
        except (TypeError, ValueError):
            raise ValueError("top_k is a number from 1 to 3") from None
    if "off" in req:
        s["off"] = _names(req["off"], "off")
    if "routes" in req:
        if not isinstance(req["routes"], dict):
            raise ValueError("routes must be a map of route name to skill names")
        s["routes"] = {str(r): _names(v, f"routes.{r}") for r, v in req["routes"].items()}
    if "repos" in req:                                  # the skills repositories the Skills page searches
        from nebula.skill_hub import REPO
        if not isinstance(req["repos"], list) or not all(isinstance(r, str) and REPO.fullmatch(r.strip()) for r in req["repos"]):
            raise ValueError("repos is a list of owner/name")
        s["repos"] = list(dict.fromkeys(r.strip() for r in req["repos"]))
    if "allow_env" in req:
        if not isinstance(req["allow_env"], dict):
            raise ValueError("allow_env must be a map of skill name to variable names")
        kept = {}
        for name, variables in req["allow_env"].items():
            for var in _names(variables, f"allow_env.{name}"):
                if not ENV_RE.fullmatch(var):
                    raise ValueError(f"{var} is not an environment variable name")
            kept[str(name)] = list(dict.fromkeys(variables))
        s["allow_env"] = kept
    return {**raw, **{k: s[k] for k in ("mode", "match", "top_k", "routes", "off", "allow_env") + (("repos",) if "repos" in s else ())}}


def switch_off(cfg, name: str) -> None:
    """Puts a skill on the Off list (and saves the config): a skill the safety scan warned about stays Off until the
    owner has read the warnings and switched it on."""
    off = list(settings(rules_of(cfg))["off"])
    if name in off:
        return
    rules = save_settings(cfg, {"off": off + [name]})
    cfg.rules = {**(getattr(cfg, "rules", None) or {}), "skills": rules}
    raw = getattr(cfg, "raw", None)
    if isinstance(raw, dict):
        raw["rules"] = {**(raw.get("rules") or {}), "skills": rules}
    if getattr(cfg, "path", None):
        cfg.save()


def view(cfg) -> dict:
    """What the Skills section shows: the switches, every skill with its routes, needs and warnings, and the
    suggestions waiting for the owner. Nothing is embedded: the page never matches."""
    from nebula import skill_hub
    skill_hub.restore_interrupted(cfg)          # an update that stopped half way: the skill as it was
    s = settings(rules_of(cfg))
    catalog = SkillCatalog(_base(cfg), s)
    routes = {str(r): [str(n) for n in (names or [])] for r, names in (s["routes"] or {}).items()}
    skills, suggested = [], []
    for skill in catalog.entries():
        row = skill.as_dict()
        row["body"] = skill.body
        if skill.source != "yours":
            suggested.append(row)
            continue
        from nebula import skill_hub
        row["origin"] = (skill_hub.origin(Path(skill.folder)) or {}).get("url")      # where an update comes from
        row["routes"] = sorted(r for r, names in routes.items() if skill.name in names)
        row["allow_env"] = sorted(s["allow_env"].get(skill.name) or [])
        skills.append(row)
    return {"dir": str(catalog.root),
            "settings": {"mode": s["mode"], "match": s["match"], "top_k": s["top_k"],
                         "off": list(s["off"]), "routes": routes, "allow_env": s["allow_env"]},
            "skills": skills, "suggested": suggested, "repos": __import__("nebula.skill_hub").skill_hub.repos(cfg)}


def _env_needs(metadata: dict) -> str | None:
    """One message when required_environment_variables is not a list of objects with a name. The catalogue reads them
    row by row (skills.py's _environment), so a `required_environment_variables: 5` would be written to the folder and
    then break every later read of the skills; the paste is refused before anything is written. A skill that does not
    name the key at all needs nothing; one whose value is there but is not a list (a `: null`) is refused, as
    skills.py's _read already counts that as an error."""
    if "required_environment_variables" not in metadata:
        return None
    raw = metadata["required_environment_variables"]
    if not isinstance(raw, list):
        return "required_environment_variables must be a list of objects with a name"
    for item in raw:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not ENV_RE.fullmatch(name):
            return "each required_environment_variables entry must be an object with a name"
    return None


def add_skill(cfg, text: str) -> str:
    """A pasted SKILL.md becomes nebula-skills/<name>/SKILL.md; nothing is written when it does not check out."""
    try:
        metadata, body = parse_skill(str(text or ""))
    except SkillError as e:
        raise ValueError(str(e)) from None
    name = metadata.get("name")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError("the frontmatter needs a name of lowercase letters, digits and hyphens")
    folder = _base(cfg) / "nebula-skills" / name
    errors, warnings = check_skill(metadata, body, folder)
    needs = _env_needs(metadata)
    if needs:
        errors.append(needs)
    if errors:
        raise ValueError("; ".join(errors))
    if folder.exists():
        raise ValueError(f"the skill {name} is already there")
    if warnings:                                # Off before it is there at all (its warnings are on the page)
        switch_off(cfg, name)
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    (folder / "SKILL.md").write_text(str(text), encoding="utf-8")
    return name


def _suggested(cfg, name) -> Path:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError("a valid skill name is required")
    folder = _base(cfg) / "nebula-skills" / ".suggested" / name
    if not folder.is_dir():
        raise ValueError(f"no suggested skill {name}")
    return folder


def accept_suggested(cfg, name: str) -> str:
    """The owner accepts: the folder moves into nebula-skills/, which is where the turns read from."""
    folder = _suggested(cfg, name)
    target = folder.parent.parent / name
    if not (folder / "SKILL.md").is_file():
        raise ValueError(f"the suggested skill {name} has no SKILL.md")
    if target.exists():
        raise ValueError(f"the skill {name} is already there: use it as it is, or remove it first")
    shutil.move(str(folder), str(target))
    return name


def dismiss_suggested(cfg, name: str) -> str:
    """The owner says no: the suggested folder is gone and nothing else is touched."""
    shutil.rmtree(_suggested(cfg, name))
    return name
