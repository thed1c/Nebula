"""nebula/pieces.py: the pieces Nebula downloads only when they are needed (nebula/catalog/pieces.json).

The installers carry what everyone needs and nothing else (the owner's 2026-10-27 word: "keep the downloads as small as
possible"); a piece that only some machines or some models need is a release file of its own, downloaded by whoever
needs it (Halo on a machine without an NVIDIA card, a guided setup) and said plainly: what it is, how big, why.

    pin("impulse-portable")            this platform's file (url, sha256, bytes), or None while it is not published
    find("impulse-portable", base)     its program when it is already here (<config folder>/nebula-<piece>/)
    fetch("impulse-portable", base)    downloaded, sha256 checked, unpacked (a half download leaves nothing behind)
    say("impulse-portable")            the sentence a page or a setup shows before it downloads
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

CATALOG = Path(__file__).resolve().parent / "catalog" / "pieces.json"


def platform_key() -> str:
    return "windows-x86_64" if os.name == "nt" else "linux-x86_64"


def _piece(name: str) -> tuple[dict, dict]:
    m = json.loads(CATALOG.read_text())
    if name not in m["pieces"]:
        raise KeyError(f"no piece {name!r}")
    return m, m["pieces"][name]


def _platform(p: dict) -> str:
    """The piece's file for this platform, or its one file for every platform ("any": data, such as Halo's adapters)."""
    files = p.get("files") or {}
    return platform_key() if platform_key() in files or "any" not in files else "any"


def pin(name: str) -> dict | None:
    m, p = _piece(name)
    plat = _platform(p)
    f = (p.get("files") or {}).get(plat) or {}
    if not f.get("sha256"):
        return None
    ver = p.get("version") or m["version"]              # a piece published with a later release names its own
    url = (m["url"].replace("{version}", ver).replace("{piece}", name)
           .replace("{platform}", plat).replace("{ext}", f["ext"]))
    return {"url": url, "sha256": f["sha256"], "bytes": int(f.get("bytes") or 0), "ext": f["ext"], "version": ver,
            "label": p.get("label") or name, "why": p.get("why") or ""}


def folder(name: str, base_dir: str | None) -> Path:
    return Path(base_dir or os.getcwd(), "nebula-" + name)


def program(name: str) -> str:
    """The file that says the piece is here: its program (.exe on Windows), or a data piece's file as named."""
    _m, p = _piece(name)
    return p["program"] + (".exe" if os.name == "nt" and not p.get("data") else "")


def find(name: str, base_dir: str | None) -> str | None:
    exe = folder(name, base_dir) / program(name)
    return str(exe) if exe.is_file() else None


def say(name: str) -> str:
    p = pin(name)
    if p is None:
        _m, x = _piece(name)
        return f"{x.get('label') or name} is not published for {platform_key()} yet"
    return (f"downloads {p['label']} ({max(1, p['bytes'] // 2**20)} MB): {p['why']}; it is not in the installer, "
            "to keep the download small")


def fetch(name: str, base_dir: str | None, progress=None) -> str:
    """The piece, downloaded into the config's folder and checked against its sha256; its program's path."""
    p = pin(name)
    if p is None:
        raise RuntimeError(say(name))
    base = Path(base_dir or os.getcwd())
    base.mkdir(parents=True, exist_ok=True)
    dest = folder(name, base_dir)
    with tempfile.TemporaryDirectory(dir=base) as d:
        arc = Path(d, f"{name}.{p['ext']}")
        h, got = hashlib.sha256(), 0
        req = urllib.request.Request(p["url"], headers={"User-Agent": "nebula"})
        with urllib.request.urlopen(req, timeout=120) as r, open(arc, "wb") as out:
            total = int(r.headers.get("Content-Length") or p["bytes"] or 0)
            for b in iter(lambda: r.read(1 << 20), b""):
                out.write(b)
                h.update(b)
                got += len(b)
                if progress:
                    progress(got, total)
        if h.hexdigest() != p["sha256"]:
            raise RuntimeError(f"{p['label']}'s download does not match its sha256: refused")
        new = Path(d, "unpacked")
        new.mkdir()
        if p["ext"] == "zip":
            with zipfile.ZipFile(arc) as z:
                z.extractall(new)
        else:
            with tarfile.open(arc, "r:*") as t:                 # tar.gz, or tar.xz (Impulse (L): its CUDA libraries)
                t.extractall(new, filter="data")
        if not (new / program(name)).is_file():
            raise RuntimeError(f"{p['label']}'s download has no {program(name)}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(new), str(dest))
    return str(dest / program(name))
