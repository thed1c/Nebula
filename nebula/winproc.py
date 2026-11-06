"""nebula/winproc.py: on Windows, no console window opens for the programs Nebula starts.

On Windows Nebula runs without a console (the shortcut starts pythonw.exe -m nebula.app; the desktop window starts
the server with Electron's windowsHide), so every console program it starts (nvidia-smi, wsl.exe, podman, openshell,
git, a skill's script, Codex) got a console window of its own, open while it ran: with the card readings, the
OpenShell list and the toolbox, one flashed every few seconds. The engines were started hidden already
(nebula/app/engine.py). `hide_consoles()` (called by nebula.app, python -m nebula, make_server and the detached
runner) makes that the default for every subprocess in the process: CREATE_NO_WINDOW is added unless the caller asked
for a console of its own (CREATE_NEW_CONSOLE) or none (DETACHED_PROCESS). Nothing changes on Linux or macOS.
"""
from __future__ import annotations

import os
import subprocess

CREATE_NO_WINDOW = 0x08000000
OWN_CONSOLE = 0x00000010 | 0x00000008          # CREATE_NEW_CONSOLE, DETACHED_PROCESS: the caller decided


def flags(creationflags: int | None) -> int:
    """The creation flags a child gets: CREATE_NO_WINDOW added, unless the caller chose its console itself."""
    f = int(creationflags or 0)
    return f if f & OWN_CONSOLE else f | CREATE_NO_WINDOW


def hide_consoles() -> bool:
    """Every subprocess.Popen in this process (and subprocess.run, check_output, ... which use it) starts hidden.
    Windows only; once per process. True when it is in place."""
    if os.name != "nt":
        return False
    if getattr(subprocess.Popen, "_nebula_hidden", False):
        return True
    init = subprocess.Popen.__init__

    def hidden(self, *args, **kw):
        if len(args) < 14:                         # creationflags given by position (the 14th): left as it is
            kw["creationflags"] = flags(kw.get("creationflags"))
        init(self, *args, **kw)

    subprocess.Popen.__init__ = hidden
    subprocess.Popen._nebula_hidden = True
    return True
