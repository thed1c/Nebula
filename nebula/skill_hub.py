"""nebula/skill_hub.py: skills from a link or from a skills repository, and their updates (the Skills page).

    POST /api/skills/link    {url}            a skill from a link: a GitHub folder or SKILL.md (github.com/<owner>/<repo>
                                              /tree|blob/<ref>/<path>), or any https link to a SKILL.md
    POST /api/skills/search  {q}              the skills in your skills repositories (rules.skills.repos, by default
                                              anthropics/skills) whose folder matches the words
    POST /api/skills/check                    for every skill that came from a link: is there a newer one there?
    POST /api/skills/update  {name} | {all}   that skill (or every one with a newer version) fetched again
    POST /api/skills/remove  {name}           a skill of yours deleted (its folder)

A skill from a link is fetched whole (its SKILL.md and the files beside it, at most MAX_FILES and MAX_BYTES), checked
the same way as a pasted one (skills.check_skill: its name, its description, what it asks to run), and written to
nebula-skills/<name>/ only when it checks out. Where it came from is kept beside it (.origin.json), so an update knows
where to look; an update is checked again before anything is replaced. Nothing here runs a skill: its scripts still
run only in the skills sandbox (skill_tools), and a skill is used only once it is on (the Skills page's switch).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from nebula.skills import NAME_RE, check_skill, parse_skill

DEFAULT_REPOS = ["anthropics/skills"]
MAX_FILES = 60
MAX_BYTES = 4 << 20
ORIGIN = ".origin.json"
GITHUB = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/(?:tree|blob)/([^/]+)(?:/(.*?))?)?/?$")
RAW = re.compile(r"^https://raw\.githubusercontent\.com/([\w.-]+)/([\w.-]+)/([^/]+)/(.+)$")
REPO = re.compile(r"^[\w.-]+/[\w.-]+$")
_CACHE: dict[str, tuple[float, object]] = {}
_LOCK = threading.Lock()
TREE_TTL = 1800.0


class HubError(ValueError):
    pass


def _base(cfg) -> Path:
    return Path(getattr(cfg, "base_dir", None) or os.getcwd()) / "nebula-skills"


def repos(cfg) -> list[str]:
    raw = ((getattr(cfg, "rules", None) or {}).get("skills") or {}).get("repos")
    return [r for r in (raw if isinstance(raw, list) else DEFAULT_REPOS) if isinstance(r, str) and REPO.fullmatch(r)]


def _get(url: str, timeout: float = 20, limit: int = MAX_BYTES) -> bytes:
    if not url.startswith("https://"):
        raise HubError("only https links")
    headers = {"User-Agent": "nebula-skills", "Accept": "application/vnd.github+json" if "api.github.com" in url else "*/*"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok and urllib.parse.urlparse(url).hostname in ("api.github.com", "raw.githubusercontent.com"):
        headers["Authorization"] = f"Bearer {tok}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            data = r.read(limit + 1)
    except urllib.error.HTTPError as e:
        raise HubError(f"{url}: {e.code} {'not found' if e.code == 404 else e.reason}") from None
    except OSError as e:
        raise HubError(f"{url}: {e}") from None
    if len(data) > limit:
        raise HubError(f"{url}: larger than {limit // 1024} KB")
    return data


def _cached(key: str, make, ttl: float = TREE_TTL):
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    val = make()
    with _LOCK:
        _CACHE[key] = (time.time(), val)
    return val


def _branch(repo: str) -> str:
    return _cached("branch:" + repo, lambda: json.loads(_get(f"https://api.github.com/repos/{repo}")).get("default_branch") or "main")


def _tree(repo: str, ref: str, fresh: bool = False) -> list[dict]:
    def make():
        v = json.loads(_get(f"https://api.github.com/repos/{repo}/git/trees/{urllib.parse.quote(ref, safe='')}?recursive=1", limit=32 << 20))
        return [x for x in v.get("tree") or [] if x.get("type") == "blob"]
    return make() if fresh else _cached(f"tree:{repo}@{ref}", make)


def where(url: str) -> dict:
    """A link, as what to fetch: {"repo", "ref", "folder"} on GitHub, else {"url"} (one SKILL.md)."""
    url = str(url or "").strip()
    m = GITHUB.match(url) or RAW.match(url)
    if m:
        repo, ref, path = f"{m.group(1)}/{m.group(2)}", m.group(3), (m.group(4) or "").strip("/")
        if path.endswith("SKILL.md"):
            path = path[:-len("SKILL.md")].strip("/")
        return {"repo": repo, "ref": ref, "folder": urllib.parse.unquote(path)}
    if url.startswith("https://"):
        return {"url": url}
    raise HubError("a link starts with https:// (a GitHub folder, or a SKILL.md)")


def _safe(rel: str) -> bool:
    p = Path(rel)
    return bool(rel) and not p.is_absolute() and ".." not in p.parts and not any(x.startswith(".") for x in p.parts)


def fetch(loc: dict) -> tuple[dict[str, bytes], str]:
    """The skill's files {relative path: bytes} and its version (a hash of what was fetched)."""
    if "url" in loc:
        data = _get(loc["url"], limit=256 << 10)
        return {"SKILL.md": data}, hashlib.sha256(data).hexdigest()[:16]
    repo, folder = loc["repo"], loc["folder"]
    ref = loc.get("ref") or _branch(repo)
    blobs = _tree(repo, ref, fresh=True)
    pre = folder + "/" if folder else ""
    mine = [b for b in blobs if b["path"].startswith(pre) and _safe(b["path"][len(pre):])]
    if not any(b["path"] == pre + "SKILL.md" for b in mine):
        raise HubError(f"no SKILL.md in {repo}/{folder or '(top)'}")
    if len(mine) > MAX_FILES:
        raise HubError(f"{repo}/{folder}: more than {MAX_FILES} files")
    if sum(int(b.get("size") or 0) for b in mine) > MAX_BYTES:
        raise HubError(f"{repo}/{folder}: larger than {MAX_BYTES >> 20} MB")
    files = {}
    for b in mine:
        raw = f"https://raw.githubusercontent.com/{repo}/{urllib.parse.quote(ref, safe='')}/{urllib.parse.quote(b['path'])}"
        files[b["path"][len(pre):]] = _get(raw)
    version = hashlib.sha256("".join(sorted(b["path"] + b.get("sha", "") for b in mine)).encode()).hexdigest()[:16]
    return files, version


def install(cfg, url: str, replace: str | None = None, warnings: list | None = None) -> str:
    """The skill at the link, checked and written; with replace (its name), the one there is replaced (an update). The
    safety scan's warnings (lines that look like prompt injection, say) go into `warnings` when a list is given; a
    skill (new or updated) that the scan warned about is put Off before its folder is in place, until the owner has
    read them and switched it on."""
    loc = where(url)
    files, version = fetch(loc)
    try:
        text = files["SKILL.md"].decode("utf-8")
    except UnicodeDecodeError:
        raise HubError("its SKILL.md is not text") from None
    try:
        metadata, body = parse_skill(text)
    except ValueError as e:
        raise HubError(str(e)) from None
    name = metadata.get("name")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise HubError("its SKILL.md needs a name of lowercase letters, digits and hyphens")
    if replace is not None and name != replace:
        raise HubError(f"the link now holds the skill {name}, not {replace}")
    base = _base(cfg)
    target = base / name
    if replace is None and target.exists():
        raise HubError(f"the skill {name} is already there")
    base.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".fetch-", dir=base)) / name
    try:
        for rel, data in files.items():
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        errors, warned = check_skill(metadata, body, tmp)
        if warnings is not None:
            warnings.extend(warned)
        from nebula import skill_tools
        if skill_tools._env_needs(metadata):
            errors.append(skill_tools._env_needs(metadata))
        if errors:
            raise HubError("; ".join(errors))
        if warned:                          # Off before it is there: never on for a moment with what was warned about
            skill_tools.switch_off(cfg, name)
        (tmp / ORIGIN).write_text(json.dumps({"url": str(url).strip(), "version": version, "fetched_at": time.time()}), encoding="utf-8")
        old = None
        if target.exists():                 # the update swaps folders: the old one is moved aside, then removed
            old = tmp.parent / (name + ".old")
            os.replace(target, old)
        try:
            os.replace(tmp, target)
        except OSError:
            if old is not None:
                os.replace(old, target)     # the old skill stays as it was
            raise
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    return name


def origin(folder: Path) -> dict | None:
    try:
        o = json.loads((Path(folder) / ORIGIN).read_text(encoding="utf-8"))
        return o if isinstance(o, dict) and o.get("url") else None
    except (OSError, ValueError):
        return None


def search(cfg, q: str, limit: int = 40) -> dict:
    """The skills in your skills repositories whose folder matches every word of q (all of them for no words)."""
    words = [w for w in re.findall(r"[a-z0-9]+", str(q or "").lower())]
    have = {p.name for p in _base(cfg).iterdir() if p.is_dir()} if _base(cfg).is_dir() else set()
    out, errors = [], []
    for repo in repos(cfg):
        try:
            ref = _branch(repo)
            for b in _tree(repo, ref):
                if not b["path"].endswith("/SKILL.md") and b["path"] != "SKILL.md":
                    continue
                folder = b["path"][:-len("SKILL.md")].strip("/")
                hay = folder.lower().replace("/", " ").replace("-", " ").replace("_", " ")
                if all(w in hay for w in words):
                    name = folder.rsplit("/", 1)[-1] or repo.split("/")[1]
                    out.append({"name": name, "repo": repo, "folder": folder, "installed": name in have,
                                "url": f"https://github.com/{repo}/tree/{ref}/{folder}".rstrip("/")})
        except HubError as e:
            errors.append(str(e))
    return {"results": out[:limit], "more": max(0, len(out) - limit), "repos": repos(cfg), "errors": errors}


def check(cfg) -> dict:
    """{name: {"url", "update": bool} or {"url", "error"}} for every skill of yours that came from a link."""
    out = {}
    base = _base(cfg)
    for folder in sorted(base.iterdir()) if base.is_dir() else []:
        o = origin(folder) if folder.is_dir() and not folder.name.startswith(".") else None
        if not o:
            continue
        try:
            _, version = fetch(where(o["url"]))
            out[folder.name] = {"url": o["url"], "update": version != o.get("version")}
        except HubError as e:
            out[folder.name] = {"url": o["url"], "error": str(e)}
    return out


def update(cfg, name: str, warnings: list | None = None) -> str:
    folder = _base(cfg) / str(name)
    o = origin(folder) if NAME_RE.fullmatch(str(name)) else None
    if not o:
        raise HubError(f"{name} did not come from a link: nothing to update it from")
    return install(cfg, o["url"], replace=str(name), warnings=warnings)


def restore_interrupted(cfg) -> list[str]:
    """An update moves the old folder aside (.fetch-*/<name>.old), then the new one into place: if Nebula stopped
    between the two, the skill is put back as it was. The names restored."""
    base, out = _base(cfg), []
    if not base.is_dir():
        return out
    for tmpdir in base.glob(".fetch-*"):
        for old in tmpdir.glob("*.old"):
            target = base / old.name[:-4]
            if NAME_RE.fullmatch(target.name) and not target.exists():
                os.replace(old, target)
                out.append(target.name)
        if not any(tmpdir.glob("*.old")):
            shutil.rmtree(tmpdir, ignore_errors=True)
    return out


def remove(cfg, name: str) -> str:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise HubError("a valid skill name is required")
    folder = _base(cfg) / name
    if not folder.is_dir():
        raise HubError(f"no skill {name}")
    shutil.rmtree(folder)
    return name
