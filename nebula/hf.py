"""nebula/hf.py: a Hugging Face link in, a model file on disk out (Deploy your own).

    resolve(url)            -> {"repo", "revision", "files": [{"path", "size", "sha256", "quant", "kind"}], "chosen", "mmproj"}
    Pulls(models_dir).start(repo, [paths], revision)  -> a pull id; .status(id) has bytes, total, state per file

A link is a repository (`https://huggingface.co/<owner>/<repo>`, optionally `/tree/<rev>`) or one file
(`.../blob/<rev>/<path>` or `.../resolve/<rev>/<path>`). The repository's GGUF files are listed from the Hub's tree API
with their sizes and LFS sha256; they are grouped by quantization (Q4_K_M, Q8_0, ...); projector files (`mmproj*`)
are found and offered with the model. A file of several shards (`-00001-of-00003.gguf`) is one choice.

Downloads go to `<models_dir>/<owner>/<repo>/<path>`, resume from a partial `.part` file with an HTTP Range, and are
checked against the LFS sha256 before they are renamed into place. A gated or private repository needs a token: it is
read from Nebula's own key file (`nebula-keys/huggingface.token`, 0600), never from or into the config. The Hub's
address is `https://huggingface.co`, or `HF_ENDPOINT` as huggingface_hub reads it (a mirror, or a test server).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

QUANT = re.compile(r"(?i)(?:^|[-_.])((?:I?Q\d(?:_[A-Z0-9]+)*)|BF16|F16|F32|MXFP4(?:_MOE)?)(?=[-_.]|$)")
SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


RETRIES = 6                                       # a broken-off download resumes this many times (Range), pausing 2 s .. 60 s

class HfError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def parse(url: str) -> dict:
    """{"repo", "revision", "path"} of a Hub link (path None for a whole repository)."""
    u = urllib.parse.urlparse(url.strip())
    if u.scheme not in ("http", "https") or not u.netloc:
        if re.match(r"^[\w.-]+/[\w.-]+$", url.strip()):                   # owner/repo
            return {"repo": url.strip(), "revision": "main", "path": None}
        raise HfError(400, "a Hugging Face link: https://huggingface.co/<owner>/<repo>[/blob/<rev>/<file>.gguf]")
    parts = [urllib.parse.unquote(x) for x in u.path.strip("/").split("/") if x]
    if parts and parts[0] in ("models",):
        parts = parts[1:]
    if len(parts) < 2:
        raise HfError(400, "the link names no repository (owner/name)")
    repo = f"{parts[0]}/{parts[1]}"
    if len(parts) >= 4 and parts[2] in ("blob", "resolve", "tree"):
        rev = parts[3]
        path = "/".join(parts[4:]) or None
        if parts[2] == "tree":
            path = None
        return {"repo": repo, "revision": rev, "path": path}
    return {"repo": repo, "revision": "main", "path": None}


def _headers(token: str | None) -> dict:
    h = {"User-Agent": "nebula"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get_json(url: str, token: str | None):
    req = urllib.request.Request(url, headers=_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise HfError(e.code, "this repository is gated or private: add a Hugging Face token (it is kept in Nebula's "
                                  "key file only) and accept the model's terms on its page") from None
        if e.code == 404:
            raise HfError(404, "no such repository or revision on the Hub") from None
        raise HfError(502, f"the Hub answered {e.code}") from None
    except (OSError, ValueError) as e:
        raise HfError(502, f"the Hub cannot be reached: {e}") from None


def quant_of(path: str) -> str | None:
    name = os.path.basename(path)
    m = None
    for m in QUANT.finditer(name.rsplit(".", 1)[0]):
        pass
    return m.group(1).upper() if m else None


def resolve(url: str, token: str | None = None) -> dict:
    """The repository's GGUF choices for a link: one per quantization (shards together), projectors apart."""
    ref = parse(url)
    rev = urllib.parse.quote(ref["revision"], safe="")
    tree = _get_json(f"{endpoint()}/api/models/{ref['repo']}/tree/{rev}?recursive=true", token)
    files = []
    for e in tree if isinstance(tree, list) else []:
        p = e.get("path") or ""
        if e.get("type") != "file" or not p.lower().endswith(".gguf"):
            continue
        lfs = e.get("lfs") or {}
        base = os.path.basename(p).lower()
        kind = "mmproj" if base.startswith("mmproj") else "mtp" if base.startswith("mtp-") else "model"   # a drafter is not a model
        files.append({"path": p, "size": int(lfs.get("size") or e.get("size") or 0), "sha256": lfs.get("oid"),
                      "quant": quant_of(p), "kind": kind})
    if not files:
        raise HfError(404, "no GGUF files in this repository (Nebula's engines read GGUF)")
    choices: dict = {}
    for f in files:
        if f["kind"] != "model":
            continue
        m = SHARD.search(f["path"])
        key = SHARD.sub(".gguf", f["path"]) if m else f["path"]
        c = choices.setdefault(key, {"id": key, "quant": f["quant"], "files": [], "size": 0})
        c["files"].append(f["path"])
        c["size"] += f["size"]
    for c in choices.values():
        c["files"].sort()
    projectors = [f for f in files if f["kind"] == "mmproj"]
    chosen = None
    if ref["path"]:
        want = SHARD.sub(".gguf", ref["path"]) if SHARD.search(ref["path"]) else ref["path"]
        chosen = want if want in choices else None
        if chosen is None and not any(f["path"] == ref["path"] for f in projectors):
            raise HfError(404, f"no GGUF file {ref['path']} in {ref['repo']}")
    elif len(choices) == 1:
        chosen = next(iter(choices))
    best_proj = sorted(projectors, key=lambda f: (0 if (f["quant"] or "") in ("F16", "BF16") else 1, f["size"]))
    return {"repo": ref["repo"], "revision": ref["revision"], "files": files,
            "choices": sorted(choices.values(), key=lambda c: c["size"]), "chosen": chosen,
            "mmproj": best_proj[0]["path"] if best_proj else None}


class Pulls:
    """Downloads in the background, one thread per pull; their progress for the page."""

    def __init__(self, models_dir: str, token_path: str | None = None):
        self.models_dir, self.token_path = models_dir, token_path
        self.pulls: dict[str, dict] = {}
        self.lock = threading.Lock()

    # the token: Nebula's own key file, never the config
    def token(self) -> str | None:
        if not self.token_path:
            return None
        try:
            return Path(self.token_path).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def set_token(self, token: str | None) -> bool:
        if not self.token_path:
            raise HfError(400, "this Nebula has no config file, so no key file for a token")
        if not token:
            Path(self.token_path).unlink(missing_ok=True)
            return False
        os.makedirs(os.path.dirname(self.token_path), mode=0o700, exist_ok=True)
        fd = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token.strip() + "\n")
        return True

    def target(self, repo: str, path: str) -> str:
        parts = repo.split("/") + path.split("/")
        if any(x in ("", ".", "..") or "\\" in x for x in parts) or len(repo.split("/")) != 2:
            raise HfError(400, f"not a file in a repository: {repo}/{path}")
        p = Path(self.models_dir, *parts).resolve()
        if Path(self.models_dir).resolve() not in p.parents:
            raise HfError(400, "a file outside the models folder")
        return str(p)

    def start(self, repo: str, paths: list[str], revision: str = "main", sha256: dict | None = None,
              sizes: dict | None = None) -> str:
        if not paths:
            raise HfError(400, "nothing to pull")
        need = sum(int((sizes or {}).get(p) or 0) for p in paths)
        os.makedirs(self.models_dir, exist_ok=True)
        free = __import__("shutil").disk_usage(self.models_dir).free
        if need and need > free:
            raise HfError(507, f"needs {need / 2**30:.1f} GB, the disk has {free / 2**30:.1f} GB free")
        pid = uuid.uuid4().hex[:12]
        files = [{"path": p, "target": self.target(repo, p), "bytes": 0, "total": int((sizes or {}).get(p) or 0),
                  "sha256": (sha256 or {}).get(p), "state": "waiting", "error": None} for p in paths]
        with self.lock:
            self.pulls[pid] = {"id": pid, "repo": repo, "revision": revision, "files": files, "state": "running",
                               "started_at": time.time(), "cancel": False}
        threading.Thread(target=self._run, args=(pid,), name=f"hf-pull-{pid}", daemon=True).start()
        return pid

    def status(self, pid: str | None = None):
        with self.lock:
            if pid is None:
                return [self._view(p) for p in self.pulls.values()]
            if pid not in self.pulls:
                raise HfError(404, f"no pull {pid}")
            return self._view(self.pulls[pid])

    @staticmethod
    def _view(p: dict) -> dict:
        done = sum(f["bytes"] for f in p["files"])
        total = sum(f["total"] for f in p["files"])
        return {"id": p["id"], "repo": p["repo"], "state": p["state"], "bytes": done, "total": total,
                "files": [{k: f[k] for k in ("path", "target", "bytes", "total", "state", "error")} for f in p["files"]]}

    def cancel(self, pid: str) -> dict:
        with self.lock:
            if pid in self.pulls:
                self.pulls[pid]["cancel"] = True
        return self.status(pid)

    def _run(self, pid: str) -> None:
        p = self.pulls[pid]
        token = self.token()
        for f in p["files"]:
            if p["cancel"]:
                break
            try:
                self._one(p, f, token)
            except Exception as e:  # noqa: BLE001 - shown on the page
                f["state"], f["error"] = "failed", str(e)[:300]
                p["state"] = "failed"
                return
        p["state"] = "cancelled" if p["cancel"] else "done"

    def _fetch(self, p: dict, f: dict, url: str, part: str, token: str | None) -> bool:
        """Read the file into `part` from the bytes it already has; True when a pause stopped it."""
        have = os.path.getsize(part) if os.path.exists(part) else 0
        h = _headers(token)
        if have:
            h["Range"] = f"bytes={have}-"
        f["state"] = "downloading"
        req = urllib.request.Request(url, headers=h)
        try:
            r = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:                       # already whole
                return False
            if e.code in (401, 403):
                raise HfError(e.code, "gated or private: a Hugging Face token is needed") from None
            if e.code >= 500 or e.code == 429:                   # the Hub busy for a moment: retried
                raise ConnectionError(f"the Hub answered {e.code}") from None
            raise HfError(502, f"the Hub answered {e.code} for {f['path']}") from None
        with r:
            if have and r.status != 206:                     # the server ignored the range: start over
                have = 0
            total = int(r.headers.get("Content-Length") or 0) + have
            f["total"] = f["total"] or total
            with open(part, "ab" if have else "wb") as out:
                f["bytes"] = have
                while True:
                    if p["cancel"]:
                        f["state"] = "paused"
                        return True
                    chunk = r.read(1 << 18)                  # small enough that a pause lands within a moment
                    if not chunk:
                        break
                    out.write(chunk)
                    f["bytes"] += len(chunk)
        if f["total"] and os.path.getsize(part) < f["total"]:   # the stream ended early without an error
            raise ConnectionError(f"the stream ended at {os.path.getsize(part)} of {f['total']} bytes")
        return False

    def _one(self, p: dict, f: dict, token: str | None) -> None:
        target = f["target"]
        if os.path.exists(target) and (not f["total"] or os.path.getsize(target) == f["total"]):
            f["bytes"], f["state"] = os.path.getsize(target), "done (already here)"
            return
        os.makedirs(os.path.dirname(target), exist_ok=True)
        part = target + ".part"
        rev = urllib.parse.quote(p["revision"], safe="")
        url = f"{endpoint()}/{p['repo']}/resolve/{rev}/{urllib.parse.quote(f['path'])}"
        # a dropped connection or a stalled read mid-file (a 30 GB file is read for many minutes) picks up from the bytes
        # already on disk (Range), up to RETRIES times with a growing pause; the Hub's own refusals are not retried
        for attempt in range(RETRIES + 1):
            try:
                if self._fetch(p, f, url, part, token):
                    return                                   # paused
                break
            except HfError:
                raise
            except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
                if attempt == RETRIES or p["cancel"]:
                    raise HfError(502, f"{f['path']}: the download kept breaking off ({e}); pull again to resume") from None
                f["state"] = f"retrying ({attempt + 1}/{RETRIES}): {str(e)[:80]}"
                time.sleep(min(60, 2 ** (attempt + 1)))
        if f["sha256"]:
            f["state"] = "checking"
            h = hashlib.sha256()
            with open(part, "rb") as src:
                for chunk in iter(lambda: src.read(1 << 22), b""):
                    h.update(chunk)
            if h.hexdigest() != f["sha256"]:
                os.unlink(part)
                raise HfError(502, f"{f['path']}: the checksum does not match (the download was removed; pull again)")
        os.replace(part, target)
        f["state"] = "done"


# --- what a repository's GGUF is, before anything is downloaded
_GGUF_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}   # type -> bytes (8 string, 9 array)


def gguf_architecture(head: bytes) -> str | None:
    """general.architecture from the first bytes of a GGUF file (its header and metadata), or None when it is not in them."""
    import struct
    if head[:4] != b"GGUF":
        return None
    pos = 4 + 4 + 8 + 8                                  # magic, version, tensor count, key count

    def take(n):
        nonlocal pos
        if pos + n > len(head):
            raise EOFError
        b = head[pos:pos + n]
        pos += n
        return b

    def string():
        return take(struct.unpack("<Q", take(8))[0])

    def skip(t):
        if t == 8:
            string()
        elif t == 9:
            et, n = struct.unpack("<I", take(4))[0], struct.unpack("<Q", take(8))[0]
            for _ in range(n):
                skip(et)
        else:
            take(_GGUF_SCALAR[t])
    try:
        n_kv = struct.unpack("<Q", head[16:24])[0]
        for _ in range(n_kv):
            key = string().decode("utf-8", "replace")
            t = struct.unpack("<I", take(4))[0]
            if key == "general.architecture" and t == 8:
                return string().decode("utf-8", "replace")
            skip(t)
    except (EOFError, KeyError, struct.error):
        return None
    return None


def remote_architecture(repo: str, revision: str, path: str, token: str | None = None, limit: int = 4 << 20) -> str | None:
    """A Hub GGUF's general.architecture, read from its first bytes (a range request: nothing else is downloaded)."""
    url = f"{endpoint()}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(path)}"
    req = urllib.request.Request(url, headers={**_headers(token), "Range": f"bytes=0-{limit - 1}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            head = r.read(limit)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise HfError(e.code, "this repository is gated or private: add a Hugging Face token (it is kept in Nebula's "
                                  "key file only) and accept the model's terms on its page") from None
        raise HfError(502, f"the Hub answered {e.code}") from None
    except OSError as e:
        raise HfError(502, f"the Hub cannot be reached: {e}") from None
    return gguf_architecture(head)
