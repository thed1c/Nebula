"""nebula/updates.py: is there a newer Nebula? A signed feed, read only when the owner chose so, and the update itself
when it can be (docs/updates.md).

    "updates": {"mode": "demand", "feed": "https://raw.githubusercontent.com/thed1c/nebula-updates/main/latest.json"}

mode (design decision 2026-10-25: "I only want that at start-up, and opt-in for 'on-launch' or on demand"):
  demand  the default: only when the owner presses "Look now" on the page; no request before that
  launch  once, a minute after Nebula starts, and never again until the next start (plus "Look now")
  off     never: no network at all, and no "Look now"
There is no timer beyond the one look at launch. The first release's "ask" (it looked every day) reads as "demand":
nobody chose to be looked for. Nothing is ever installed without the owner's click.

The feed is a small JSON file, `latest.json`, and beside it `latest.json.sig`: the base64 of an Ed25519 signature over
the file's exact bytes, made with the key whose public half is PUBLIC_KEY below (greenfield/sign_feed.py makes both).
A feed that is unsigned, badly signed or malformed is ignored (logged only): the page never shows it.

    {"version": "0.1.3", "notes": "...", "published": "2026-10-25",
     "files": {"linux-x86_64": {"url": "...", "sha256": "..."}, "windows-x86_64": {"url": "...", "sha256": "..."}},
     "release_page": "https://github.com/thed1c/Nebula/releases/tag/v0.1.3"}

Two ways to the new version:
  notify   the page's button opens the release page (while the releases are private, the only way)
  install  only for a Nebula the one-click installer put in place, and only when its file downloads without signing in:
           the file is fetched to a temporary folder, checked against its sha256, and its installer run quietly as this
           user (Linux: the .run with --quiet; Windows: Setup.exe /S, per-user) once no turn is in flight; then Nebula
           opens again. Never root, never an administrator.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from nebula import __version__

LOG = logging.getLogger("nebula.updates")
FEED = "https://raw.githubusercontent.com/thed1c/nebula-updates/main/latest.json"
PUBLIC_KEY = "5e629a04d650d6af42e13ba3c1e39c832437c80f1471364a23086483a49f7447"      # Ed25519, hex (greenfield/sign_feed.py --keygen prints it); its private half is never in the repo
FIRST_S = 60.0                     # "launch": the one look, this long after the start
MAX_FEED = 64 * 1024
MODES = ("demand", "launch", "off")
OLD_MODES = {"ask": "demand"}      # the first release's daily look: nobody opted in, so it is the default now


def settings(cfg) -> dict:
    u = cfg.raw.get("updates") if isinstance(cfg.raw.get("updates"), dict) else {}
    mode = OLD_MODES.get(u.get("mode"), u.get("mode"))
    mode = mode if mode in MODES else "demand"
    return {"mode": mode, "feed": os.environ.get("NEBULA_UPDATES_FEED") or u.get("feed") or FEED}


def version_tuple(v: str) -> tuple[int, ...]:
    """"0.1.10" -> (0, 1, 10): compared as numbers, so 0.1.10 is newer than 0.1.9. A leading "v" is allowed."""
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)", str(v or "").strip())
    if not m:
        raise ValueError(f"not a version: {v!r}")
    parts = [int(x) for x in m.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:         # 0.2 == 0.2.0
        parts.pop()
    return tuple(parts)


def is_newer(latest: str, current: str = __version__) -> bool:
    return version_tuple(latest) > version_tuple(current)


# --- Ed25519 (RFC 8032, section 5.1), verify only, plain Python: one check a day, so speed does not matter
_P = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _add(a, b):
    A = (a[1] - a[0]) * (b[1] - b[0]) % _P
    B = (a[1] + a[0]) * (b[1] + b[0]) % _P
    C = 2 * a[3] * b[3] * _D % _P
    D = 2 * a[2] * b[2] % _P
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _P, G * H % _P, F * G % _P, E * H % _P)


def _mul(s: int, pt):
    q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            q = _add(q, pt)
        pt = _add(pt, pt)
        s >>= 1
    return q


def _same(a, b) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _x_of(y: int, sign: int):
    if y >= _P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P:
        return None
    return _P - x if (x & 1) != sign else x


def _point(b: bytes):
    if len(b) != 32:
        return None
    y = int.from_bytes(b, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _x_of(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def _bytes(pt) -> bytes:
    zi = pow(pt[2], _P - 2, _P)
    x, y = pt[0] * zi % _P, pt[1] * zi % _P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


_GY = 4 * pow(5, _P - 2, _P) % _P
_G = (_x_of(_GY, 0), _GY, 1, _x_of(_GY, 0) * _GY % _P)


def _h(*parts: bytes) -> int:
    return int.from_bytes(hashlib.sha512(b"".join(parts)).digest(), "little")


def verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    if len(public_key) != 32 or len(signature) != 64:
        return False
    a, r = _point(public_key), _point(signature[:32])
    s = int.from_bytes(signature[32:], "little")
    if a is None or r is None or s >= _L:
        return False
    k = _h(signature[:32], public_key, message) % _L
    return _same(_mul(s, _G), _add(r, _mul(k, a)))


def _secret(seed: bytes) -> tuple[int, bytes]:
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_of(seed: bytes) -> bytes:
    """The public key of a 32-byte private seed (the build side and the tests; Nebula itself only verifies)."""
    return _bytes(_mul(_secret(seed)[0], _G))


def sign(seed: bytes, message: bytes) -> bytes:
    """An Ed25519 signature (the build side's fallback when `cryptography` is missing, and the tests)."""
    a, prefix = _secret(seed)
    pub = _bytes(_mul(a, _G))
    r = _h(prefix, message) % _L
    rb = _bytes(_mul(r, _G))
    return rb + ((r + _h(rb, pub, message) % _L * a) % _L).to_bytes(32, "little")


# --- the feed
def platform_key() -> str | None:
    import platform
    arch = platform.machine().lower().replace("amd64", "x86_64")
    if arch != "x86_64":
        return None
    return "windows-x86_64" if os.name == "nt" else ("linux-x86_64" if sys.platform.startswith("linux") else None)


def read_feed(body: bytes, sig: bytes, public_key: str = None) -> dict | None:
    """The feed when its signature checks out and it is well formed, else None (and why, in the log)."""
    key = public_key or PUBLIC_KEY
    try:
        pub = bytes.fromhex(key)
        signature = base64.b64decode(sig.strip(), validate=True)
    except ValueError:
        LOG.warning("updates: the feed's signature (or the key) is not readable: ignored")
        return None
    if not verify(pub, body, signature):
        LOG.warning("updates: the feed's signature does not check out: ignored")
        return None
    try:
        feed = json.loads(body)
        version_tuple(feed["version"])
        page = str(feed.get("release_page") or "")
        if not page.startswith("https://"):
            raise ValueError("release_page is not https")
        files = feed.get("files") or {}
        for f in files.values():
            if not str(f["url"]).startswith("https://") or not re.fullmatch(r"[0-9a-f]{64}", str(f["sha256"])):
                raise ValueError("a file without an https url or a sha256")
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        LOG.warning("updates: the feed is signed but malformed (%s): ignored", e)
        return None
    return feed


def _get(url: str, limit: int = MAX_FEED, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": f"nebula/{__version__}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read(limit + 1)
    if len(body) > limit:
        raise ValueError(f"{url}: larger than {limit} bytes")
    return body


def downloadable(url: str, timeout: float = 15.0) -> bool:
    """True when the file downloads without signing in (a public release); a private one answers 404 to us."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": f"nebula/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def install_dir() -> Path | None:
    """Where the one-click installer put this Nebula, when it did (design decision: a Nebula run from a checkout or a
    pip install is never replaced by an installer; it is told about the update only)."""
    py = Path(sys.executable).resolve()
    if os.name == "nt":
        home = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Programs" / "Nebula EH"
    else:
        home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "nebula-eh"
    try:
        py.relative_to(home.resolve() / "python")
    except ValueError:
        return None
    return home


def busy(server) -> bool:
    """A turn in flight, or a call waiting at a gate: the update waits for it."""
    ledger = getattr(server, "ledger", None)
    if ledger is not None:
        with ledger.lock:
            if ledger.db.execute("SELECT 1 FROM attempts WHERE state='running' LIMIT 1").fetchone():
                return True
    adm = getattr(server, "admission", None)
    return bool(adm is not None and adm.waiting())


class Updates:
    """The looks for a newer Nebula: one at launch when the owner chose "launch", else only on the page's "Look now";
    the last answer kept on disk."""

    def __init__(self, server, first_s: float = FIRST_S):
        self.server, self.cfg = server, server.cfg
        self.first_s = first_s
        self.closing = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.installing = None              # None, or what the update is doing: downloading, waiting, installing, failed: ...
        self.state = self._load()

    # the last answer, beside the config (NAME.updates.json)
    def _cache_path(self) -> Path | None:
        p = getattr(self.cfg, "path", None)
        return Path(p).with_suffix(".updates.json") if p else None

    def _load(self) -> dict:
        p = self._cache_path()
        try:
            st = json.loads(p.read_text()) if p and p.is_file() else {}
        except (OSError, ValueError):
            st = {}
        return st if isinstance(st, dict) else {}

    def _save(self) -> None:
        p = self._cache_path()
        if p is None:
            return
        try:
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=1))
            os.replace(tmp, p)
        except OSError as e:
            LOG.info("updates: the last answer was not kept: %s", e)

    def start(self) -> "Updates":
        if settings(self.cfg)["mode"] == "launch":       # demand and off: no thread, no request
            self.thread = threading.Thread(target=self._at_launch, name="updates", daemon=True)
            self.thread.start()
        return self

    def close(self) -> None:
        self.closing.set()

    def _at_launch(self) -> None:
        if self.closing.wait(self.first_s) or settings(self.cfg)["mode"] != "launch":   # switched since the start
            return
        try:
            self.check()
        except Exception:  # noqa: BLE001 - a failed look waits for the next start or "Look now", never a crash
            LOG.exception("updates: the check failed")

    def check(self) -> dict:
        """Read the feed now (the look at launch, or the page's "Look now"). With mode off, nothing is fetched."""
        st = settings(self.cfg)
        if st["mode"] == "off":
            return self.view()
        now = time.time()
        try:
            feed = read_feed(_get(st["feed"]), _get(st["feed"] + ".sig", 1024))
        except (urllib.error.URLError, OSError, ValueError) as e:
            LOG.info("updates: the feed was not reached: %s", e)
            with self.lock:
                self.state.update(last_checked=now, error="not reached")
                self._save()
            return self.view()
        with self.lock:
            self.state.update(last_checked=now, error=None if feed else "not signed")
            if feed:
                f = (feed.get("files") or {}).get(platform_key() or "")
                public = bool(f and install_dir() is not None and downloadable(f["url"]))
                self.state.update(latest=str(feed["version"]).lstrip("v"), notes=str(feed.get("notes") or "")[:4000],
                                  published=feed.get("published"), release_page=feed["release_page"],
                                  file=f if public else None)
            self._save()
        return self.view()

    def view(self) -> dict:
        st = settings(self.cfg)
        with self.lock:
            s = dict(self.state)
        latest = s.get("latest")
        try:
            newer = bool(latest) and is_newer(latest)
        except ValueError:
            newer = False
        return {"mode": st["mode"], "current": __version__, "latest": latest, "newer": newer,
                "notes": s.get("notes") or "", "published": s.get("published"),
                "release_page": s.get("release_page") if newer else None,
                "can_install": newer and bool(s.get("file")) and install_dir() is not None,
                "installing": self.installing, "last_checked": s.get("last_checked"), "error": s.get("error")}

    def install(self) -> dict:
        """The page's button. Notify (the release page to open) unless this Nebula can update itself; then the update
        runs on its own thread and the page follows `installing`."""
        v = self.view()
        if not v["newer"]:
            return {**v, "action": "none"}
        if not v["can_install"]:
            return {**v, "action": "open", "open": v["release_page"]}
        with self.lock:
            if self.installing and not self.installing.startswith("failed"):
                return {**v, "action": "installing"}
            self.installing = "downloading"
            f = dict(self.state["file"])
        threading.Thread(target=self._install, args=(f,), name="update-install", daemon=True).start()
        return {**self.view(), "action": "installing"}

    def _install(self, f: dict) -> None:
        try:
            work = Path(tempfile.mkdtemp(prefix="nebula-update-"))
            name = f["url"].rsplit("/", 1)[-1].split("?")[0] or ("Setup.exe" if os.name == "nt" else "nebula.run")
            dest = fetch_checked(f["url"], work / name, f["sha256"])
            self.installing = "waiting"            # for the turns in flight: never in the middle of one
            while busy(self.server):
                if self.closing.wait(5):
                    return
            self.installing = "installing"
            run_installer(dest, self.cfg, install_dir())
        except Exception as e:  # noqa: BLE001 - said on the page; the running Nebula carries on
            LOG.warning("updates: the update did not run: %s", e)
            self.installing = f"failed: {str(e)[:200]}"
            return
        if os.name == "nt":                         # Setup needs Nebula's files free: Nebula steps out, the helper reopens it
            threading.Timer(2.0, _exit_for_update, [self.server]).start()


def fetch_checked(url: str, dest: Path, sha256: str) -> Path:
    """The file, downloaded beside itself as .part, renamed only when its sha256 matches (else removed and refused)."""
    part = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": f"nebula/{__version__}"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as out:
        for b in iter(lambda: r.read(1 << 20), b""):
            out.write(b)
            h.update(b)
    if h.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: the download does not match its sha256: refused")
    os.replace(part, dest)
    return dest


LINUX_HELPER = """#!/bin/sh
# Nebula's update (nebula/updates.py): the new installer, quietly, as this user; then this Nebula steps out and opens again.
"$1" --quiet || { echo "the installer failed: Nebula keeps running as it was"; exit 1; }
kill -INT "$2" 2>/dev/null
for i in $(seq 1 60); do kill -0 "$2" 2>/dev/null || break; sleep 1; done
kill -0 "$2" 2>/dev/null && kill -TERM "$2"
sleep 1
exec "$3/nebula-open"
"""

WINDOWS_HELPER = """@echo off
rem Nebula's update (nebula/updates.py): waits for Nebula to close, runs Setup quietly (per user), opens Nebula again.
:wait
tasklist /FI "PID eq %~2" 2>nul | find "%~2" >nul && (timeout /t 1 /nobreak >nul & goto wait)
"%~1" /S
cd /d "%~4"
start "" "%~3\\python\\pythonw.exe" -m nebula.app --config "%~5"
"""


def run_installer(path: Path, cfg, home: Path | None) -> None:
    """Starts the helper that runs the installer and reopens Nebula; it outlives this process. No sudo, ever."""
    if home is None:
        raise RuntimeError("this Nebula was not put in place by the installer")
    work = path.parent
    if os.name == "nt":
        helper = work / "nebula-update.cmd"
        helper.write_text(WINDOWS_HELPER)
        flags = 0x00000008 | 0x00000200 | 0x08000000     # DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW
        subprocess.Popen(["cmd", "/c", str(helper), str(path), str(os.getpid()), str(home),
                          cfg.base_dir or str(work), cfg.path or ""], creationflags=flags, close_fds=True)
        return
    path.chmod(0o755)
    helper = work / "nebula-update.sh"
    helper.write_text(LINUX_HELPER)
    log = open(work / "nebula-update.log", "ab")
    subprocess.Popen(["sh", str(helper), str(path), str(os.getpid()), str(home)], stdin=subprocess.DEVNULL,
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)


def _exit_for_update(server) -> None:
    """Windows: Nebula closes so Setup can replace its files (the helper waits for this process, then reopens it)."""
    t = threading.Thread(target=server.shutdown, daemon=True)
    t.start()
    t.join(10)
    try:
        server.server_close()
    finally:
        os._exit(0)


def set_mode(cfg, mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(f"updates.mode is one of {', '.join(MODES)}")
    u = dict(cfg.raw.get("updates") or {})
    u["mode"] = mode
    cfg.raw["updates"] = u
    if cfg.path:
        cfg.save()
    return settings(cfg)

