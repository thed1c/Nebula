"""nebula/cmdline.py: a command line in words, on Linux and Windows alike.

Linux and macOS: POSIX shell rules (shlex). Windows: a backslash is part of a path (C:\\Users\\…\\python.exe), never an
escape, and a quoted word loses its quotes ("a model.gguf" is one word, without the quotes).
"""
from __future__ import annotations

import os
import shlex


def split(command: str) -> list[str]:
    if os.name != "nt":
        return shlex.split(command)
    return [w[1:-1] if len(w) > 1 and w[0] == w[-1] and w[0] in "'\"" else w for w in shlex.split(command, posix=False)]
