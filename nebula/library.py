"""nebula/library.py: the model files on this computer, for the Models page's cards.

    GET  /api/yard/library                    every model file Nebula can see: in its own models folder, in the folders
                                              you added, and in your Hugging Face cache; each with the engines that load
                                              it (a card is solid while one of them runs it) and the folders themselves
    POST /api/yard/library/folders            {add: path} | {remove: path}: another folder to look in (an external
                                              drive, a second disk); kept in the config as "model_dirs"
    POST /api/yard/library/download-to        {path}: where downloads go from now on (Nebula's own models folder, or a
                                              folder you added); kept in the config as "download_dir"
    POST /api/yard/library/delete             {path}: deletes a model file (every shard of it) from Nebula's own folder or
                                              a folder you added; never one an engine names (remove or change that engine
                                              first), never in the Hugging Face cache, never anything that is not a model

A model is a .gguf file. A model of several shards is one model (its first shard stands for it, its size is all of
them); a projector (mmproj*) goes with the model in its folder and is not a model of its own. A file still downloading
(.part) is not a model yet. Nebula moves nothing; it deletes a model file only when you ask it to (delete_file).
"""
from __future__ import annotations

import os
import re
import shutil

from nebula import hfcache

SHARD = hfcache.SHARD
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".cache", "blobs"}
MAX_FILES = 4000                                  # a folder of a whole drive: stop counting long before it hurts


class LibraryError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _abs(cfg, p: str) -> str:
    p = os.path.expanduser(str(p))
    return os.path.realpath(p if os.path.isabs(p) else os.path.join(cfg.base_dir or os.getcwd(), p))


def own_dir(cfg) -> str:
    return _abs(cfg, (cfg.raw or {}).get("models_dir") or "models")


def extra_dirs(cfg) -> list[str]:
    return [_abs(cfg, d) for d in ((cfg.raw or {}).get("model_dirs") or []) if str(d).strip()]


def download_dir(cfg) -> str:
    """Where a download goes: the folder picked on the Models page if it is still one of the library's, else Nebula's own."""
    d = (cfg.raw or {}).get("download_dir")
    if d:
        a = _abs(cfg, d)
        if a in [own_dir(cfg)] + extra_dirs(cfg):
            return a
    return own_dir(cfg)


def folders(cfg) -> list[dict]:
    out, seen = [], set()
    dl = download_dir(cfg)
    for kind, d in [("nebula", own_dir(cfg))] + [("added", x) for x in extra_dirs(cfg)] + [("hf", x) for x in hfcache.roots(cfg)]:
        if d in seen:
            continue
        seen.add(d)
        free = None
        try:
            if os.path.isdir(d):
                free = shutil.disk_usage(d).free
        except OSError:
            pass
        out.append({"path": d, "kind": kind, "exists": os.path.isdir(d), "free": free, "downloads": d == dl and kind != "hf"})
    return out


def _quant(name: str) -> str | None:
    return hfcache._quant(name)


def scan_dir(d: str) -> list[dict]:
    """The models in a plain folder (and its subfolders)."""
    if not os.path.isdir(d):
        return []
    found, n = [], 0
    for dp, dn, fn in os.walk(d):
        dn[:] = sorted(x for x in dn if x not in SKIP_DIRS and not x.startswith("models--") and not x.startswith("."))
        ggufs = sorted(f for f in fn if f.lower().endswith(".gguf"))
        projectors = [f for f in ggufs if f.lower().startswith("mmproj")]
        for f in ggufs:
            n += 1
            if n > MAX_FILES:
                return found
            if f in projectors:
                continue
            sm = SHARD.search(f)
            if sm and sm.group(1) != "00001":
                continue
            p = os.path.join(dp, f)
            try:
                size = os.path.getsize(p)
                if sm:
                    k = int(sm.group(2))
                    parts = [os.path.join(dp, SHARD.sub(f"-{i:05d}-of-{k:05d}.gguf", f)) for i in range(1, k + 1)]
                    if not all(os.path.exists(x) for x in parts):
                        continue
                    size = sum(os.path.getsize(x) for x in parts)
            except OSError:
                continue
            rel = os.path.relpath(p, d)
            parts_ = rel.split(os.sep)
            repo = "/".join(parts_[:2]) if len(parts_) >= 3 else None      # Nebula's own downloads: <owner>/<repo>/<file>
            found.append({"path": p, "file": f, "repo": repo, "size": size, "quant": _quant(f),
                          "shards": int(sm.group(2)) if sm else 1, "mmproj": os.path.join(dp, projectors[0]) if projectors else None,
                          "folder": d})
    return found


def _engine_files(cfg, engines) -> dict:
    """realpath of a model file -> the engines whose spec names it, with whether each runs now."""
    out: dict = {}
    for st in engines or []:
        spec = st.get("spec") or {}
        m = spec.get("model")
        if not m or not isinstance(m, str):
            continue
        p = _abs(cfg, m)
        out.setdefault(p, []).append({"engine": st.get("name") or spec.get("name"), "state": st.get("state"),
                                      "models": st.get("tiers") or []})
    return out


def view(cfg, engines=None, pulls=None) -> dict:
    """The library: the folders, every model file in them (once each), what loads it, and the downloads under way."""
    files, seen = [], set()
    for d in [own_dir(cfg)] + extra_dirs(cfg):
        for m in scan_dir(d):
            rp = os.path.realpath(m["path"])
            if rp in seen:
                continue
            seen.add(rp)
            files.append({**m, "source": "nebula" if d == own_dir(cfg) else "added"})
    for m in hfcache.models(cfg):
        rp = os.path.realpath(m["path"])
        if rp in seen:
            continue
        seen.add(rp)
        files.append({"path": m["path"], "file": m["file"], "repo": m["repo"], "size": m["size"], "quant": m["quant"],
                      "shards": m["shards"], "mmproj": m["mmproj"], "folder": m["root"], "source": "hf"})
    # a model file an engine on this computer names, wherever it is: Nebula's own too, even outside the folders above
    extra_folders = []
    for st in engines or []:
        spec = st.get("spec") or {}
        m = spec.get("model")
        if not m or not isinstance(m, str) or spec.get("machine") or not m.lower().endswith(".gguf"):
            continue
        p = _abs(cfg, m)
        if p in seen or not os.path.isfile(p):
            continue
        seen.add(p)
        size, shards, sm = os.path.getsize(p), 1, SHARD.search(os.path.basename(p))
        if sm:
            shards = int(sm.group(2))
            parts = [os.path.join(os.path.dirname(p), SHARD.sub(f"-{i:05d}-of-{shards:05d}.gguf", os.path.basename(p))) for i in range(1, shards + 1)]
            size = sum(os.path.getsize(x) for x in parts if os.path.exists(x))
        d = os.path.dirname(p)
        projectors = sorted(x for x in os.listdir(d) if x.lower().startswith("mmproj") and x.lower().endswith(".gguf")) if os.path.isdir(d) else []
        files.append({"path": p, "file": os.path.basename(p), "repo": None, "size": size, "quant": _quant(os.path.basename(p)),
                      "shards": shards, "mmproj": os.path.join(d, projectors[0]) if projectors else None, "folder": d, "source": "engine"})
        if d not in extra_folders:
            extra_folders.append(d)
    used = _engine_files(cfg, engines)
    for f in files:
        by = used.get(os.path.realpath(f["path"])) or used.get(_abs(cfg, f["path"])) or []
        f["engines"] = by
        f["loaded"] = any(e.get("state") in ("ready", "loading", "starting", "draining", "running elsewhere") for e in by)
        f["name"] = re.sub(r"(?i)(-\d{5}-of-\d{5})?\.gguf$", "", f["file"])
    files.sort(key=lambda f: (not f["loaded"], not f["engines"], f["name"].lower()))
    fl = folders(cfg)
    fl += [{"path": d, "kind": "engine", "exists": True, "free": None, "downloads": False} for d in extra_folders if d not in [x["path"] for x in fl]]
    return {"folders": fl, "files": files, "download_dir": download_dir(cfg),
            "pulls": [p for p in (pulls or []) if p.get("state") in ("running", "paused", "failed", "cancelled")]}


def add_folder(cfg, path: str) -> list[str]:
    if not path or not str(path).strip():
        raise LibraryError(400, "which folder?")
    a = _abs(cfg, path)
    if not os.path.isdir(a):
        raise LibraryError(404, f"no folder at {a}")
    if a == os.path.realpath(os.sep) or a == os.path.realpath(os.path.expanduser("~")):
        raise LibraryError(400, "pick the folder the models are in, not a whole drive or your home folder")
    raw = cfg.raw.setdefault("model_dirs", [])
    if a not in [_abs(cfg, x) for x in raw] and a != own_dir(cfg):
        raw.append(a)
    return extra_dirs(cfg)


def remove_folder(cfg, path: str) -> list[str]:
    a = _abs(cfg, path)
    cfg.raw["model_dirs"] = [x for x in (cfg.raw.get("model_dirs") or []) if _abs(cfg, x) != a]
    if not cfg.raw["model_dirs"]:
        cfg.raw.pop("model_dirs")
    if (cfg.raw.get("download_dir") and _abs(cfg, cfg.raw["download_dir"]) == a):
        cfg.raw.pop("download_dir")
    return extra_dirs(cfg)


def set_download_dir(cfg, path: str) -> str:
    a = _abs(cfg, path)
    if a == own_dir(cfg):
        cfg.raw.pop("download_dir", None)
    elif a in extra_dirs(cfg):
        cfg.raw["download_dir"] = a
    else:
        raise LibraryError(400, "add the folder first: downloads go to Nebula's own folder or one you added")
    return download_dir(cfg)


def delete_file(cfg, path: str, engines=None) -> dict:
    """Delete one model file (all its shards) that the library shows, from Nebula's folder or one you added."""
    if not path or not str(path).strip():
        raise LibraryError(400, "which file?")
    want = os.path.realpath(_abs(cfg, path))
    f = next((x for x in view(cfg, engines)["files"] if os.path.realpath(x["path"]) == want), None)
    if f is None:
        raise LibraryError(404, "not a model file Nebula knows: only the files on the Models page can be deleted")
    if f["source"] == "hf":
        raise LibraryError(400, "that file is in your Hugging Face cache: Nebula leaves the cache alone (huggingface-cli delete-cache)")
    if f["engines"]:
        names = ", ".join(sorted({str(e.get("engine")) for e in f["engines"]}))
        raise LibraryError(409, f"an engine uses it ({names}): remove that engine, or give it another file, first")
    parts = [f["path"]]
    sm = SHARD.search(f["file"])
    if sm:
        k = int(sm.group(2))
        parts = [os.path.join(os.path.dirname(f["path"]), SHARD.sub(f"-{i:05d}-of-{k:05d}.gguf", f["file"])) for i in range(1, k + 1)]
    freed = 0
    for x in parts:
        if not x.lower().endswith(".gguf") or not os.path.isfile(x):
            continue
        freed += os.path.getsize(x)
        os.remove(x)
    d, root = os.path.realpath(os.path.dirname(f["path"])), os.path.realpath(f.get("folder") or "")
    if root and d != root and d.startswith(root + os.sep) and not os.listdir(d):
        os.rmdir(d)  # the model's own folder, now empty; never the library folder itself
    return {"deleted": f["name"], "files": len(parts), "freed": freed}
