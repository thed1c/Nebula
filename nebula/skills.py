"""Local SKILL.md catalog and task matching.

The catalog reads metadata without loading a skill's body into a model turn. The
body is retained for an explicit load by the turn's skill tools. Lookup's own
embedder can be passed in; no embedding service is started by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform as host_platform
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

MAX_BODY_BYTES = 64 * 1024
NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
WORDS = re.compile(r"[a-z0-9]+")
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of",
             "on", "or", "the", "this", "to", "use", "with", "you", "your"}
INJECTION = re.compile(
    r"\b(?:ignore|disregard|override)\b.{0,50}\b(?:previous|prior|system|developer|instructions|rules)\b"
    r"|\byou are now\b|\breveal (?:the |your )?(?:system|developer) (?:prompt|message)\b",
    re.I,
)
EXFILTRATION = re.compile(
    r"\b(?:curl|wget|requests\.post|httpx\.post|upload|exfiltrat|send)\b.{0,100}"
    r"(?:secret|token|credential|api[_ -]?key|private[_ -]?key|\.ssh|\.env)"
    r"|(?:secret|token|credential|api[_ -]?key|private[_ -]?key|\.ssh|\.env).{0,100}"
    r"\b(?:curl|wget|requests\.post|httpx\.post|upload|exfiltrat|send)\b",
    re.I,
)


class SkillError(ValueError):
    """Malformed SKILL.md frontmatter."""


def _next_content(lines: list[str], index: int) -> int:
    while index < len(lines) and (not lines[index].strip() or lines[index].lstrip().startswith("#")):
        index += 1
    return index


def _indent(line: str) -> int:
    if "\t" in line[:len(line) - len(line.lstrip())]:
        raise SkillError("tabs are not supported for YAML indentation")
    return len(line) - len(line.lstrip(" "))


def _key_value(line: str) -> tuple[str, str] | None:
    quote = None
    for i, c in enumerate(line):
        if c in "\"'":
            if quote == c:
                quote = None
            elif quote is None:
                quote = c
        elif c == ":" and quote is None and (i + 1 == len(line) or line[i + 1].isspace()):
            key = line[:i].strip()
            if key:
                return key, line[i + 1:].strip()
            break
    return None


def _strip_comment(text: str) -> str:
    quote = None
    escaped = False
    for i, c in enumerate(text):
        if escaped:
            escaped = False
        elif c == "\\" and quote == '"':
            escaped = True
        elif c in "\"'":
            if quote == c:
                quote = None
            elif quote is None:
                quote = c
        elif c == "#" and quote is None and (i == 0 or text[i - 1].isspace()):
            return text[:i].rstrip()
    return text.strip()


def _inline_items(text: str) -> list[str]:
    items, start, quote, depth, escaped = [], 0, None, 0, False
    for i, c in enumerate(text):
        if escaped:
            escaped = False
        elif c == "\\" and quote == '"':
            escaped = True
        elif c in "\"'":
            if quote == c:
                quote = None
            elif quote is None:
                quote = c
        elif quote is None:
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            elif c == "," and depth == 0:
                items.append(text[start:i].strip())
                start = i + 1
    if quote or depth:
        raise SkillError("unclosed inline list")
    items.append(text[start:].strip())
    return items


def _scalar(text: str):
    text = _strip_comment(text)
    if text.startswith("[") and text.endswith("]"):
        inside = text[1:-1].strip()
        return [_scalar(x) for x in _inline_items(inside)] if inside else []
    if len(text) >= 2 and text[0] == text[-1] == '"':
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise SkillError(f"invalid quoted string: {e.msg}") from None
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1].replace("''", "'")
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if text.lower() in ("null", "~"):
        return None
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?", text):
        return float(text)
    return text


def _block_scalar(lines: list[str], index: int, parent_indent: int, folded: bool) -> tuple[str, int]:
    kept = []
    while index < len(lines):
        line = lines[index]
        if line.strip() and _indent(line) <= parent_indent:
            break
        kept.append(line)
        index += 1
    content_indent = min((_indent(x) for x in kept if x.strip()), default=parent_indent + 1)
    values = [x[content_indent:] if x.strip() else "" for x in kept]
    while values and values[-1] == "":
        values.pop()
    if not values:
        return "", index
    if not folded:
        return "\n".join(values) + "\n", index
    out = values[0]
    for previous, current in zip(values, values[1:]):
        out += (" " if previous and current else "\n") + current
    return out + "\n", index


def _value(lines: list[str], index: int, parent_indent: int, raw: str):
    raw = _strip_comment(raw)
    if raw in ("|", ">"):
        return _block_scalar(lines, index, parent_indent, raw == ">")
    if raw:
        return _scalar(raw), index
    child = _next_content(lines, index)
    if child < len(lines) and _indent(lines[child]) > parent_indent:
        return _block(lines, child, _indent(lines[child]))
    return None, index


def _mapping(lines: list[str], index: int, indent: int) -> tuple[dict, int]:
    result = {}
    while True:
        index = _next_content(lines, index)
        if index >= len(lines) or _indent(lines[index]) < indent:
            return result, index
        if _indent(lines[index]) != indent or lines[index].lstrip().startswith("- "):
            raise SkillError(f"unexpected indentation on frontmatter line {index + 1}")
        pair = _key_value(lines[index].strip())
        if pair is None:
            raise SkillError(f"expected key: value on frontmatter line {index + 1}")
        key, raw = pair
        if key in result:
            raise SkillError(f"duplicate frontmatter key {key!r}")
        result[key], index = _value(lines, index + 1, indent, raw)


def _sequence(lines: list[str], index: int, indent: int) -> tuple[list, int]:
    result = []
    while True:
        index = _next_content(lines, index)
        if index >= len(lines) or _indent(lines[index]) < indent:
            return result, index
        line = lines[index]
        if _indent(line) != indent or not (line.strip() == "-" or line.lstrip().startswith("- ")):
            raise SkillError(f"expected list item on frontmatter line {index + 1}")
        raw = line.lstrip()[1:].strip()
        pair = _key_value(raw)
        if pair is None:
            value, index = _value(lines, index + 1, indent, raw)
        else:
            key, first = pair
            value = {}
            value[key], index = _value(lines, index + 1, indent + 2, first)
            next_line = _next_content(lines, index)
            if next_line < len(lines) and _indent(lines[next_line]) > indent:
                more, index = _mapping(lines, next_line, _indent(lines[next_line]))
                if key in more:
                    raise SkillError(f"duplicate frontmatter key {key!r}")
                value.update(more)
        result.append(value)


def _block(lines: list[str], index: int, indent: int):
    if lines[index].lstrip().startswith("- ") or lines[index].strip() == "-":
        return _sequence(lines, index, indent)
    return _mapping(lines, index, indent)


def parse_skill(text: str) -> tuple[dict, str]:
    """Return all frontmatter keys and the unchanged markdown body."""
    lines = text.lstrip("\ufeff").splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise SkillError("SKILL.md must start with --- frontmatter")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise SkillError("SKILL.md is missing the closing ---")
    header = [x.rstrip("\r\n") for x in lines[1:end]]
    start = _next_content(header, 0)
    if start == len(header):
        metadata = {}
    else:
        if _indent(header[start]) != 0:
            raise SkillError("frontmatter must begin at column 1")
        metadata, index = _block(header, start, 0)
        if not isinstance(metadata, dict) or _next_content(header, index) != len(header):
            raise SkillError("frontmatter must be a mapping")
    return metadata, "".join(lines[end + 1:])


def _host_name() -> str:
    return host_platform.system().lower()


def _platform_name(name: str) -> str:
    return {"macos": "darwin", "osx": "darwin", "win32": "windows"}.get(name.lower(), name.lower())


def _platform_names(metadata: dict) -> list[str]:
    raw = metadata.get("platforms")
    return [_platform_name(str(x)) for x in raw] if isinstance(raw, list) else []


def scan_warnings(body: str) -> list[str]:
    warnings = []
    for line_no, line in enumerate(body.splitlines(), 1):
        if INJECTION.search(line):
            warnings.append(f"line {line_no}: possible prompt injection: {line.strip()[:120]}")
        if EXFILTRATION.search(line):
            warnings.append(f"line {line_no}: possible exfiltration: {line.strip()[:120]}")
    return warnings


def check_skill(metadata: dict, body: str, folder: str | Path, platform: str | None = None) -> tuple[list[str], list[str]]:
    """Validate the documented shape; suspicious prose is advisory only."""
    errors, warnings = [], scan_warnings(body)
    name = metadata.get("name")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name) or len(name) > 64:
        errors.append("name must be 1-64 lowercase letters, digits or hyphens")
    elif name != Path(folder).name:
        errors.append("name must match the folder name")
    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip() or "\n" in description or len(description) > 1024:
        errors.append("description must be one nonempty line of at most 1024 characters")
    raw_platforms = metadata.get("platforms")
    if raw_platforms is not None:
        if not isinstance(raw_platforms, list) or not raw_platforms or not all(isinstance(x, str) and x for x in raw_platforms):
            errors.append("platforms must be a nonempty list of names")
        elif _platform_name(platform or _host_name()) not in _platform_names(metadata) and "any" not in _platform_names(metadata):
            warnings.append(f"not available on {(platform or _host_name()).lower()}")
    if len(body.encode("utf-8")) > MAX_BODY_BYTES:
        errors.append("body exceeds 64 KB")
    return errors, warnings


@dataclass
class Skill:
    name: str
    description: str
    tags: list[str]
    folder: str
    source: str
    scripts: list[str]
    files: list[str]
    environment: list[dict]
    warnings: list[str]
    on: bool
    metadata: dict = field(default_factory=dict)
    body: str = ""
    sha: str = ""
    errors: list[str] = field(default_factory=list)

    def as_dict(self, include_body: bool = False) -> dict:
        row = {key: getattr(self, key) for key in
               ("name", "description", "tags", "folder", "source", "scripts", "files", "environment",
                "warnings", "on", "errors")}
        if include_body:
            row["body"] = self.body
        return row

    def accepts_tools(self, tools) -> bool:
        nested = self.metadata.get("metadata")
        hermes = (nested.get("hermes") or {}) if isinstance(nested, dict) else {}
        if not isinstance(hermes, dict):
            return True
        offered = set(tools or ())
        required = set(hermes.get("requires_tools") or ())
        fallback = set(hermes.get("fallback_for_tools") or ())
        return required <= offered and not fallback.intersection(offered)


def _environment(metadata: dict, environ: dict) -> list[dict]:
    result = []
    for item in metadata.get("required_environment_variables") or []:
        row = {"name": item} if isinstance(item, str) else dict(item) if isinstance(item, dict) else {}
        name = row.get("name")
        if isinstance(name, str) and ENV_RE.fullmatch(name) and not any(x["name"] == name for x in result):
            row["set"] = name in environ
            result.append(row)
    return result


def _terms(text: str) -> set[str]:
    return set(WORDS.findall(text.lower())) - STOPWORDS


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    norm = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


class SkillCatalog:
    """Read skill roots; match active skills with Lookup or word overlap."""

    def __init__(self, data_dir: str | Path, settings: dict | None = None, *, embedder=None,
                 lookup_on: bool | None = None, lookup_settings: dict | None = None, lookup_library=None,
                 platform: str | None = None, environ: dict | None = None):
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "nebula-skills"
        self.settings = dict(settings or {})
        self.embedder = embedder
        self.lookup_library = lookup_library
        self.lookup_settings = {**dict(getattr(lookup_library, "settings", {}) or {}), **dict(lookup_settings or {})}
        self.lookup_on = self.lookup_settings.get("library", True) is not False if lookup_on is None else lookup_on
        self.platform = platform or _host_name()
        self.environ = os.environ if environ is None else environ
        self.skills: list[Skill] = []
        self.refresh()

    def _roots(self) -> list[Path]:
        roots = [self.root]
        for raw in self.settings.get("dirs") or []:
            path = Path(raw)
            roots.append(path if path.is_absolute() else self.data_dir / path)
        return roots

    def refresh(self) -> list[Skill]:
        rows, seen = [], set()
        for root in self._roots():
            if not root.is_dir():
                continue
            for folder in sorted(root.iterdir()):
                if not folder.is_dir() or folder.is_symlink() or folder.name.startswith("."):
                    continue
                skill = self._read(folder, "yours")
                if skill and skill.name not in seen:
                    rows.append(skill)
                    seen.add(skill.name)
        suggested = self.root / ".suggested"
        if suggested.is_dir():
            for folder in sorted(suggested.iterdir()):
                if folder.is_dir() and not folder.is_symlink() and not folder.name.startswith("."):
                    skill = self._read(folder, "suggested")
                    if skill:
                        rows.append(skill)
        self.skills = rows
        return rows

    def _read(self, folder: Path, source: str) -> Skill | None:
        path = folder / "SKILL.md"
        if not path.is_file() or path.is_symlink():
            return None
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        errors = []
        try:
            metadata, body = parse_skill(raw.decode("utf-8"))
        except (SkillError, UnicodeError) as e:
            metadata, body = {}, ""
            errors.append(f"invalid SKILL.md: {e}")
        checked, warnings = check_skill(metadata, body, folder, self.platform)
        errors.extend(checked)
        if len(raw) > 128 * 1024:
            errors.append("SKILL.md exceeds 128 KB")
        if not isinstance(metadata.get("required_environment_variables", []), list):
            errors.append("required_environment_variables must be a list")
        nested = metadata.get("metadata")
        hermes = (nested.get("hermes") or {}) if isinstance(nested, dict) else {}
        tags = hermes.get("tags") or metadata.get("tags") or [] if isinstance(hermes, dict) else []
        tags = [str(x) for x in tags] if isinstance(tags, list) else []
        files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                       if p.is_file() and not p.is_symlink() and p.name != "SKILL.md")
        scripts = [p for p in files if p.startswith("scripts/")]
        name = metadata.get("name") if isinstance(metadata.get("name"), str) else folder.name
        available = not any(x.startswith("not available on ") for x in warnings)
        on = not errors and available and source == "yours" and name not in (self.settings.get("off") or [])
        return Skill(name, str(metadata.get("description") or ""), tags, str(folder), source,
                     scripts, files, _environment(metadata, self.environ), warnings, on, metadata, body, sha, errors)

    def entries(self) -> list[Skill]:
        return list(self.skills)

    def list(self) -> list[dict]:
        return [skill.as_dict() for skill in self.skills]

    def get(self, name: str, *, include_off: bool = False) -> Skill | None:
        return next((s for s in self.skills if s.name == name and (include_off or s.on)), None)

    def _cache_path(self) -> Path:
        return self.root / ".embedding-cache.json"

    def _default_embedder(self):
        """Use the running library's model, or its configured local embedder."""
        library = self.lookup_library
        if library is not None and getattr(library, "embedder", None) is not None:
            self.embedder = library.embedder
            return self.embedder
        from nebula.rag import embed as lookup_embed  # lazy: Lookup is optional

        embed = self.lookup_settings.get("embed") or {}
        if embed.get("tier"):
            engine_of = getattr(library, "engine_of", None)
            engine = engine_of(embed["tier"]) if callable(engine_of) else None
            if engine is None:
                return None
            self.embedder = lookup_embed.EngineEmbedder(engine, embed["tier"], embed.get("model"))
            return self.embedder
        data = Path(self.lookup_settings.get("data_dir") or self.data_dir / "nebula-lookup")
        models_dir = getattr(library, "models_dir", None)
        models = (models_dir() if callable(models_dir) else
                  self.lookup_settings.get("models_dir") or os.environ.get("FASTEMBED_CACHE_PATH") or data / "models")
        self.embedder = lookup_embed.FastEmbedder(embed.get("model") or lookup_embed.DEFAULT_EMBED, cache_dir=str(models))
        return self.embedder

    def _vectors(self, skills: list[Skill], embedder) -> dict[str, list[float]]:
        model = str(getattr(embedder, "name", type(embedder).__name__))
        try:
            cache = json.loads(self._cache_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        vectors = cache.get("vectors", {}) if cache.get("model") == model else {}
        vectors = {k: v for k, v in vectors.items() if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)}
        missing = {s.sha: s.description for s in skills if s.sha not in vectors}
        if missing:
            embedded = embedder.embed(list(missing.values()))
            if len(embedded) != len(missing):
                raise SkillError("Lookup returned the wrong number of skill embeddings")
            vectors.update({sha: list(map(float, vector)) for sha, vector in zip(missing, embedded)})
            self.root.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".embedding-cache-", dir=self.root)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump({"model": model, "vectors": vectors}, f)
                os.replace(temporary, self._cache_path())
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return vectors

    def match(self, message: str, *, top_k: int = 3, threshold: float | None = None,
              tools=(), embedder=None, lookup_on: bool | None = None) -> list[Skill]:
        """Return at most three active skills, ordered by relevance to the last user message."""
        limit = min(3, max(0, int(top_k)))
        if not limit or not message.strip():
            return []
        skills = [s for s in self.skills if s.on and s.accepts_tools(tools)]
        if not skills:
            return []
        use_lookup = self.lookup_on if lookup_on is None else lookup_on
        scored = []
        if use_lookup:
            try:
                model = embedder or self.embedder or self._default_embedder()
                if model is not None:
                    vectors = self._vectors(skills, model)
                    query = model.embed_query(message)
                    scored = [(_cosine(query, vectors[s.sha]), s) for s in skills]
            except Exception:  # noqa: BLE001 - unavailable Lookup must not stop local skill matching
                scored = []
        if not scored:
            query_words = _terms(message)
            scored = [(len(query_words & _terms(" ".join([s.name, s.description, *s.tags]))) / len(query_words)
                       if query_words else 0.0, s) for s in skills]
            floor = 0.2 if threshold is None else threshold
        else:
            floor = 0.25 if threshold is None else threshold
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [skill for score, skill in scored if score >= floor and score > 0][:limit]


Catalog = SkillCatalog
