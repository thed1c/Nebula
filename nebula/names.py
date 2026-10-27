"""nebula/names.py: Nebula's public names, and the older "cockpit" ones it still accepts.

Clients name a tier or an agent as "nebula/<name>" (or the bare name), send X-Nebula-* headers, and find the ledger
in nebula.sqlite3 and the key files in nebula-keys/. Before the split from the engine's repository these were
"cockpit/<name>", X-Cockpit-* and cockpit.*: those are still read, so an existing install and its clients keep
working, and new ones are written.
"""
from __future__ import annotations

import os

MODEL_PREFIX = "nebula/"
OLD_MODEL_PREFIXES = ("nebula/", "cockpit/")
SERVED_MODEL = "nebula"


def model(name: str | None = None) -> str:
    """The model field that names a tier or an agent: "nebula/<name>", or "nebula" for none."""
    return f"{MODEL_PREFIX}{name}" if name else SERVED_MODEL


def bare_model(value: str) -> str:
    """A model field without its "nebula/" (or older "cockpit/") prefix."""
    for p in OLD_MODEL_PREFIXES:
        if value.startswith(p):
            return value[len(p):]
    return value


def header(headers, name: str):
    """X-Nebula-<name>, else the older X-Cockpit-<name>; None without headers or either."""
    if headers is None:
        return None
    return headers.get(f"X-Nebula-{name}") or headers.get(f"X-Cockpit-{name}")


def set_header(headers: dict, name: str, value) -> None:
    """Set X-Nebula-<name> on a dict of headers (dropping an older X-Cockpit-<name>), or drop both for None."""
    headers.pop(f"X-Cockpit-{name}", None)
    if value is None:
        headers.pop(f"X-Nebula-{name}", None)
    else:
        headers[f"X-Nebula-{name}"] = value


def local_path(base_dir: str | None, new: str, old: str) -> str:
    """A file or folder beside the config: the new name, unless only the older one exists there already."""
    base = base_dir or os.getcwd()
    n, o = os.path.join(base, new), os.path.join(base, old)
    return o if os.path.exists(o) and not os.path.exists(n) else n
