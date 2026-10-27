"""nebula/migrate.py: what Nebula keeps on disk carries its version, so an update never breaks an install.

- **Each SQLite file** (the ledger, the security log, Lookup's catalog) keeps its version in `PRAGMA user_version`.
  Its steps are a list: the first is the first schema, each later one changes it. A file at an older version is
  copied aside first (`<file>.v<N>.bak`), then brought up step by step, each step in one transaction with its version.
- **The config file** keeps `"config_version"`; a missing one is version 1. A file at an older version is copied
  aside (`nebula.json.v<N>.bak`), brought up by `CONFIG_STEPS`, and written back.
- **A file from a newer Nebula** is refused, never opened and never changed: going back to an older Nebula after an
  update would otherwise corrupt it. The message says to update Nebula, or to use the backup the update left.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path


class SchemaError(ValueError):
    pass


def _backup_name(path: str, version: int) -> str:
    out, n = f"{path}.v{version}.bak", 2
    while os.path.exists(out):
        out, n = f"{path}.v{version}.{n}.bak", n + 1
    return out


def _has_tables(conn) -> bool:
    return conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0] > 0


def sqlite_upgrade(conn, name: str, steps: list, path: str | None = None) -> dict:
    """Bring a SQLite file to len(steps). A step is an SQL script, or a function given the connection. Returns
    {"from", "to", "backup"}."""
    have, want = int(conn.execute("PRAGMA user_version").fetchone()[0]), len(steps)
    if have > want:
        raise SchemaError(f"{name} ({path or 'in memory'}) was written by a newer Nebula (its version {have}, this "
                          f"Nebula reads up to {want}): update Nebula, or put back the copy an update left beside it")
    out = {"from": have, "to": want, "backup": None}
    if have == want:
        return out
    if path and path != ":memory:" and _has_tables(conn):        # a file in use: copied aside before it changes
        out["backup"] = _backup_name(path, have)
        dest = sqlite3.connect(out["backup"])
        try:
            raw = getattr(conn, "conn", conn)
            raw.backup(dest)
        finally:
            dest.close()
    for i in range(have, want):
        step = steps[i]
        conn.execute("BEGIN IMMEDIATE")
        try:
            if callable(step):
                step(conn)
            else:
                for stmt in _statements(step):
                    conn.execute(stmt)
            conn.execute(f"PRAGMA user_version = {i + 1}")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    return out


def _statements(script: str) -> list[str]:
    """An SQL script's statements (sqlite3.complete_statement finds where each ends), so a step runs inside the one
    transaction executescript would otherwise commit."""
    out, cur = [], ""
    for line in script.splitlines(keepends=True):
        cur += line
        if sqlite3.complete_statement(cur):
            if cur.strip().rstrip(";").strip():
                out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


# --- the config file
CONFIG_VERSION = 1
CONFIG_STEPS: dict = {}            # {from version: function(dict) -> dict}, for CONFIG_VERSION > 1


def config_version(d: dict) -> int:
    try:
        return int(d.get("config_version") or 1)
    except (TypeError, ValueError):
        raise SchemaError(f"\"config_version\" is {d.get('config_version')!r}: a whole number") from None


def check_config(d: dict, path: str | None = None) -> None:
    v = config_version(d)
    if v > CONFIG_VERSION:
        raise SchemaError(f"the config {path or ''} was written by a newer Nebula (config_version {v}, this Nebula "
                          f"reads up to {CONFIG_VERSION}): update Nebula, or put back the copy an update left beside it")


def upgrade_config(d: dict, path: str | None = None) -> tuple[dict, str | None]:
    """(the config at CONFIG_VERSION, the backup's path when one was made). Writes the file back when it changed."""
    check_config(d, path)
    v = config_version(d)
    if v == CONFIG_VERSION:
        return d, None
    backup = None
    if path:
        backup = _backup_name(path, v)
        shutil.copy2(path, backup)
    while v < CONFIG_VERSION:
        d = CONFIG_STEPS[v](dict(d))
        v += 1
    d["config_version"] = CONFIG_VERSION
    if path:
        tmp = path + ".tmp"
        Path(tmp).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    return d, backup


def read_config(path: str) -> dict:
    """A config file, checked and brought to this Nebula's version."""
    return upgrade_config(json.loads(Path(path).read_text(encoding="utf-8")), path)[0]
