"""nebula/privdir.py: a folder only this user reads (keys, tokens, sign-in, a sandbox's state).

On Linux and macOS: mode 0700. On Windows: a plain folder, because the user's profile it lives in is already theirs
alone. Python 3.12 turns mode 0o700 there into an owner-only access list, and a Nebula run as administrator owns what
it creates as the Administrators group, which locked the user out of Nebula's own keys afterwards (a Windows machine
whose Nebula was once opened as administrator, 2026-11-01).
"""
from __future__ import annotations

import os


def private_dir(path: str) -> None:
    if os.name == "nt":
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, mode=0o700, exist_ok=True)
