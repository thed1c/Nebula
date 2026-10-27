"""nebula/browse.py: the page's folder picker (Browse…): the folders in a folder on this machine, never a file's contents.

    GET /api/browse?path=<a folder>&hidden=1   -> {path, parent, dirs: [{name, path}], readable, places: [{name, path}]}

The page API's own rules apply (this machine, a signed-in browser once sign-in is set). `readable` counts the files
right there that Lookup can read, so the picker can say "12 documents here". Hidden folders and Lookup's own skips
(.git, node_modules, …) are left out unless asked for.
"""
from __future__ import annotations

import os

MAX_DIRS = 500


class BrowseError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def places() -> list[dict]:
    """Where a picker starts: home, its usual folders, the whole disk, and mounted drives."""
    home = os.path.expanduser("~")
    out = [{"name": "Home", "path": home}]
    for n in ("Documents", "Desktop", "Downloads"):
        p = os.path.join(home, n)
        if os.path.isdir(p):
            out.append({"name": n, "path": p})
    out.append({"name": "This computer", "path": os.path.abspath(os.sep)})
    for base in ("/mnt", "/media", os.path.join("/media", os.path.basename(home)), "/Volumes"):
        try:
            for e in sorted(os.scandir(base), key=lambda e: e.name.lower()):
                if e.is_dir() and not e.name.startswith(".") and os.path.basename(home) != e.name:
                    out.append({"name": e.name, "path": e.path})
        except OSError:
            continue
    return out


def listing(path: str | None, hidden: bool = False) -> dict:
    from pathlib import Path
    from nebula.rag.ingest import SKIP_DIRS, wanted
    p = os.path.realpath(os.path.expanduser(path or "~"))
    if not os.path.isdir(p):
        raise BrowseError(404, f"{path} is not a folder on this machine")
    dirs, n = [], 0
    try:
        entries = sorted(os.scandir(p), key=lambda e: e.name.lower())
    except PermissionError:
        raise BrowseError(403, f"this machine does not let Nebula read {p}") from None
    for e in entries:
        try:
            if e.is_dir():
                if not hidden and (e.name.startswith(".") or e.name in SKIP_DIRS):
                    continue
                if len(dirs) < MAX_DIRS:
                    dirs.append({"name": e.name, "path": e.path})
            elif e.is_file() and wanted(Path(e.name)):
                n += 1
        except OSError:
            continue
    parent = os.path.dirname(p)
    return {"path": p, "parent": parent if parent != p else None, "dirs": dirs, "more": len(dirs) >= MAX_DIRS,
            "readable": n, "places": places()}
