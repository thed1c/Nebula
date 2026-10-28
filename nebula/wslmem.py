"""nebula/wslmem.py: let Nebula under WSL2 use more of the PC's memory, only when the owner clicks for it.

WSL2 gives Linux half of the PC's RAM unless `%UserProfile%\\.wslconfig` says otherwise (`[wsl2]` `memory=`). A model
that keeps its experts in RAM (Flint) may not fit in that half on a 32 GB PC. When a start is refused for RAM under WSL,
the page offers "Let Nebula use more memory":

- plan(): the .wslconfig path (the Windows profile, found from inside WSL: `cmd.exe /c echo %UserProfile%` + `wslpath`),
  the PC's total RAM (PowerShell's Win32_ComputerSystem), its `memory=` now, and the new value: the total less 8 GB, at
  least half of it.
- apply(): a backup of .wslconfig beside it first, then only the `memory` key under `[wsl2]` changed (the section added
  when missing); every other line kept as it was.
- undo(): the backup put back (no file before: the one Nebula wrote is removed).
- restart(): `wsl --shutdown`, which closes every WSL program, Nebula too; only with its own second yes (confirm). Nebula
  starts again the next time it is opened.

Nothing here runs on start-up or from the installer: only the four POST /api/wsl/memory* calls of the page.
NEBULA_FAKE_WINPROFILE (a folder) and NEBULA_FAKE_WIN_RAM_MB stand in for Windows in tests.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

KEEP_FOR_WINDOWS_MB = 8192


class WslMemError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _run(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def profile_dir() -> Path:
    """The Windows user's profile folder, as a Linux path."""
    fake = os.environ.get("NEBULA_FAKE_WINPROFILE")
    if fake:
        return Path(fake)
    win = _run(["cmd.exe", "/c", "echo %UserProfile%"]).splitlines()[-1:] or [""]
    win = win[0].strip()
    if not win or "%" in win:
        raise WslMemError(409, "this is not WSL, or Windows' profile folder could not be found")
    lin = _run(["wslpath", "-u", win])
    if not lin:
        raise WslMemError(409, "the Windows profile folder could not be turned into a Linux path (wslpath)")
    return Path(lin)


def windows_total_mb() -> int:
    """The PC's RAM (Windows' view, not WSL's share), in MB."""
    fake = os.environ.get("NEBULA_FAKE_WIN_RAM_MB")
    if fake:
        return int(fake)
    out = _run(["powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
    try:
        return int(out.splitlines()[-1]) // 2**20
    except (ValueError, IndexError):
        raise WslMemError(409, "Windows' total RAM could not be read (PowerShell)") from None


def _memory_now(text: str) -> str | None:
    sec = None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            sec = s[1:-1].strip().lower()
        elif sec == "wsl2" and re.match(r"memory\s*=", s, re.I):
            return s.split("=", 1)[1].strip()
    return None


def _set_memory(text: str, value: str) -> str:
    """The text with `memory=value` under [wsl2]: the key replaced where it is, else added at the section's top, else the
    section added at the end. Every other line as it was."""
    lines = text.splitlines()
    sec, at, done = None, None, False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            sec = s[1:-1].strip().lower()
            if sec == "wsl2" and at is None:
                at = i
        elif sec == "wsl2" and re.match(r"memory\s*=", s, re.I):
            lines[i] = f"memory={value}"
            done = True
    if not done:
        if at is not None:
            lines.insert(at + 1, f"memory={value}")
        else:
            if lines and lines[-1].strip():
                lines.append("")
            lines += ["[wsl2]", f"memory={value}"]
    return "\n".join(lines) + "\n"


def plan() -> dict:
    prof = profile_dir()
    total = windows_total_mb()
    gb = max((total - KEEP_FOR_WINDOWS_MB) // 1024, (total // 2 + 1023) // 1024)
    path = prof / ".wslconfig"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    return {"path": str(path), "exists": path.is_file(), "windows_total_gb": round(total / 1024), "memory_now": _memory_now(text),
            "memory_new": f"{gb}GB", "restart_needed": True,
            "about": f"Nebula will let WSL use {gb} GB of the PC's {round(total / 1024)} GB (Windows keeps the rest). "
                     f".wslconfig is backed up first and only its memory line changes. WSL must restart for it to count: "
                     f"that closes every WSL program, Nebula too."}


def apply() -> dict:
    p = plan()
    path = Path(p["path"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f".wslconfig.nebula-backup-{stamp}")
    if path.is_file():
        shutil.copy2(path, backup)
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        backup.write_text("", encoding="utf-8")           # an empty marker: there was no .wslconfig before
        text = ""
    path.write_text(_set_memory(text, p["memory_new"]), encoding="utf-8")
    return {**p, "applied": True, "backup": str(backup), "had_file": p["exists"],
            "next": "Restart WSL to use it (wsl --shutdown closes every WSL program, Nebula too); open Nebula again after."}


def undo(backup: str, had_file: bool) -> dict:
    b = Path(backup)
    if not b.name.startswith(".wslconfig.nebula-backup-") or not b.is_file():
        raise WslMemError(404, "that backup is not there")
    path = b.with_name(".wslconfig")
    if had_file:
        shutil.copy2(b, path)
    elif path.is_file():
        path.unlink()
    b.unlink()
    return {"undone": True, "path": str(path), "memory_now": _memory_now(path.read_text(encoding="utf-8")) if path.is_file() else None,
            "next": "Restart WSL (wsl --shutdown) for the old value to count."}


def restart(confirm: bool) -> dict:
    if confirm is not True:
        raise WslMemError(409, "restarting WSL closes every WSL program, Nebula too: it needs its own yes (confirm)")
    if os.environ.get("NEBULA_FAKE_WINPROFILE"):
        return {"restarting": True, "fake": True}
    subprocess.Popen(["wsl.exe", "--shutdown"], start_new_session=True)
    return {"restarting": True}
