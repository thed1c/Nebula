"""nebula/hfcache.py: the GGUF models already on this computer in the Hugging Face cache, offered without a download.

The cache is where huggingface_hub keeps what it downloaded: `$HF_HUB_CACHE`, `$HF_HOME/hub` and
`$XDG_CACHE_HOME/huggingface/hub` (else `~/.cache/huggingface/hub`), or only `$NEBULA_HF_CACHE`; more in the config:

    "hf_cache_dirs": ["<another cache folder>"]

Each repository is `models--<owner>--<repo>/snapshots/<revision>/<file>` (links into `blobs/`, named by the file's
sha256). Nebula only reads it: an engine runs the file where it is, and a sandboxed engine sees the cache read-only.
A file still downloading (`.incomplete`) or a broken link is skipped; a model of several shards is one choice (its first
shard); a projector (`mmproj*`) goes with the model in its folder.
"""
from __future__ import annotations

import os
import re
import threading
import time

from nebula.hf import QUANT

SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
_CACHE: dict = {"at": 0.0, "key": None, "models": []}
_LOCK = threading.Lock()


def roots(cfg=None) -> list[str]:
    """The cache folders that exist here, each once."""
    env = os.environ
    if "NEBULA_HF_CACHE" in env:                          # these folders only (several: separated as in PATH)
        cands = [x for x in env["NEBULA_HF_CACHE"].split(os.pathsep) if x]
    else:
        cands = [env.get("HF_HUB_CACHE"), os.path.join(env["HF_HOME"], "hub") if env.get("HF_HOME") else None,
                 os.path.join(env.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "huggingface", "hub")]
    cands += list(((getattr(cfg, "raw", None) or {}).get("hf_cache_dirs") or []))
    out, seen = [], set()
    for c in cands:
        if not c:
            continue
        r = os.path.realpath(os.path.expanduser(str(c)))
        if r not in seen and os.path.isdir(r):
            seen.add(r); out.append(r)
    return out


def _quant(name: str) -> str | None:
    m = QUANT.search(os.path.basename(name).rsplit(".", 1)[0])
    return m.group(1).upper() if m else None


def scan_root(root: str) -> list[dict]:
    out = []
    for d in sorted(os.listdir(root)):
        if not d.startswith("models--"):
            continue
        repo = d[len("models--"):].replace("--", "/", 1)
        snaps = os.path.join(root, d, "snapshots")
        if not os.path.isdir(snaps):
            continue
        for rev in sorted(os.listdir(snaps)):
            base = os.path.join(snaps, rev)
            files = []
            for dp, _dn, fn in os.walk(base):
                for f in fn:
                    if f.lower().endswith(".gguf"):
                        files.append(os.path.relpath(os.path.join(dp, f), base))
            projectors = [f for f in files if os.path.basename(f).lower().startswith("mmproj")]
            for f in sorted(files):
                if f in projectors:
                    continue
                sm = SHARD.search(f)
                if sm and sm.group(1) != "00001":
                    continue                                      # a later shard: its first one stands for it
                p = os.path.join(base, f)
                try:
                    if not os.path.exists(p) or os.path.realpath(p).endswith(".incomplete"):
                        continue
                    size = os.path.getsize(p)
                    if sm:                                        # every shard here, or it is not usable
                        n = int(sm.group(2))
                        parts = [os.path.join(base, SHARD.sub(f"-{i:05d}-of-{n:05d}.gguf", f)) for i in range(1, n + 1)]
                        if not all(os.path.exists(x) for x in parts):
                            continue
                        size = sum(os.path.getsize(x) for x in parts)
                except OSError:
                    continue
                folder = os.path.dirname(f)
                mm = next((x for x in projectors if os.path.dirname(x) == folder), None)
                out.append({"id": f"{repo}@{rev[:8]}:{f}", "repo": repo, "revision": rev, "file": f, "path": p,
                            "size": size, "quant": _quant(f), "shards": int(sm.group(2)) if sm else 1,
                            "mmproj": os.path.join(base, mm) if mm else None, "root": root})
    return out


def models(cfg=None, max_age_s: float = 60.0) -> list[dict]:
    """Every usable GGUF in the cache folders (scanned at most once a minute)."""
    rs = roots(cfg)
    with _LOCK:
        if _CACHE["key"] == rs and time.time() - _CACHE["at"] < max_age_s:
            return list(_CACHE["models"])
    found = []
    for r in rs:
        try:
            found += scan_root(r)
        except OSError:
            continue
    with _LOCK:
        _CACHE.update(at=time.time(), key=rs, models=found)
    return list(found)


def find(cfg, repo: str, revision: str | None, file: str) -> str | None:
    """A catalog file already in the cache at its pinned revision (a moving one, "main", matches any)."""
    for r in roots(cfg):
        base = os.path.join(r, "models--" + repo.replace("/", "--"), "snapshots")
        if not os.path.isdir(base):
            continue
        revs = [revision] if revision and revision != "main" else sorted(os.listdir(base))
        for rev in revs:
            p = os.path.join(base, rev, file)
            if os.path.isfile(p) and not os.path.realpath(p).endswith(".incomplete"):
                return p
    return None


def root_of(cfg, path: str) -> str | None:
    """The cache folder a path is in (its links resolve inside it), else None."""
    ap = os.path.abspath(path)
    return next((r for r in roots(cfg) if ap.startswith(r + os.sep)), None)


def label(m: dict) -> str:
    gb = m["size"] / 1e9
    return f"{m['repo'].split('/')[-1]} · {m['quant'] or os.path.basename(m['file'])} · {gb:.1f} GB" + (" · reads pictures" if m.get("mmproj") else "")
