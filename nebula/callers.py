"""nebula/callers.py: who is asking (its own Switchyard's caller registry).

A caller is an app, an agent framework or another system that uses this Nebula's API. The config's `callers`:

    {"callers": [{"id": "writer-app", "name": "The writing app", "class": "interactive",
                  "tags": ["writer/*"], "models": ["big", "small"], "keys": [{"id", "label", "sha256", "hint", "created_at"}]}]}

- **The key names the client.** A request that presents one of a caller's keys (`Authorization: Bearer …` or
  `x-api-key`) is that caller. Its keys are kept as hashes only; a key is shown once, when it is made.
- **The label names the origin inside it.** `X-Nebula-Caller: <label>` (free text, e.g. `writer/draft`) is recorded on
  every decision. One client can carry several origins (several agents behind one app) under one key.
- Without a key, a label that matches a caller's tag globs picks that caller (callers on this machine that do not
  authenticate); the key always wins over the label.
- `mode` "passthrough": the model the request names (else the caller's first) is used as it is: no escalation, no
  overflow to another model, no retry; "composite" (the default) is the whole routing below.
- What a caller is for: its `models` are the job's ticks (the models it may use, in order: the first with a free slot
  wins), its `class` its place at the gates (interactive, agent, background). Both appear in the Plan tab as its job.
"""
from __future__ import annotations

import fnmatch
import hashlib
import hmac
import re
import secrets
import time

CLASSES = ("interactive", "agent", "background")
FIELDS = ("id", "name", "class", "tags", "models", "keys", "note", "mode", "guest")
MODES = ("composite", "passthrough")
ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")


class CallerError(ValueError):
    pass


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def check(row: dict, tiers: list[str] | None = None) -> dict:
    if not isinstance(row, dict):
        raise CallerError("a caller is an object")
    bad = [k for k in row if k not in FIELDS]
    if bad:
        raise CallerError(f"caller: unknown field {bad[0]!r}")
    cid = str(row.get("id") or "").strip()
    if not ID.match(cid):
        raise CallerError("a caller id is lowercase letters, digits, '.', '_' or '-' (up to 48)")
    out = {"id": cid, "name": str(row.get("name") or cid)[:80]}
    if row.get("class") is not None:
        if row["class"] not in CLASSES:
            raise CallerError(f"caller {cid}: class is one of {', '.join(CLASSES)}")
        out["class"] = row["class"]
    tags = row.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) and t.strip() for t in tags):
        raise CallerError(f"caller {cid}: tags is a list of label globs")
    if tags:
        out["tags"] = [t.strip() for t in tags]
    models = row.get("models") or []
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models) or len(set(models)) != len(models):
        raise CallerError(f"caller {cid}: models is a list of model names, each once")
    if tiers is not None:
        unknown = [m for m in models if m not in tiers]
        if unknown:
            raise CallerError(f"caller {cid}: no model named {unknown[0]!r}")
    if models:
        out["models"] = list(models)
    if row.get("keys"):
        out["keys"] = [dict(k) for k in row["keys"] if isinstance(k, dict) and k.get("sha256")]
    if row.get("mode") not in (None, "composite"):
        if row["mode"] not in MODES:
            raise CallerError(f"caller {cid}: mode is one of {', '.join(MODES)}")
        out["mode"] = row["mode"]
    if row.get("note"):
        out["note"] = str(row["note"])[:300]
    if row.get("guest"):
        out["guest"] = True                      # someone else's app: only shared models, its own cache (nebula/tenancy.py)
    return out


def all_of(cfg) -> list[dict]:
    return list(getattr(cfg, "callers", None) or [])


def get(cfg, cid: str) -> dict | None:
    return next((c for c in all_of(cfg) if c.get("id") == cid), None)


def put(cfg, row: dict) -> dict:
    """Add or change a caller (its keys are kept: they change only through make_key/revoke)."""
    c = check(row, [t.name for t in cfg.tiers])
    old = get(cfg, c["id"])
    if old is not None and old.get("keys"):
        c["keys"] = old["keys"]
    cfg.callers = [c if x.get("id") == c["id"] else x for x in all_of(cfg)] if old is not None else all_of(cfg) + [c]
    return c


def remove(cfg, cid: str) -> bool:
    left = [c for c in all_of(cfg) if c.get("id") != cid]
    changed = len(left) != len(all_of(cfg))
    cfg.callers = left
    return changed


def make_key(cfg, cid: str, label: str = "") -> tuple[dict, str]:
    c = get(cfg, cid)
    if c is None:
        raise CallerError(f"no caller {cid!r}")
    key = "nbc_" + secrets.token_urlsafe(32)
    row = {"id": secrets.token_hex(4), "label": (label or "key")[:60], "sha256": _hash(key),
           "hint": key[:8] + "…" + key[-4:], "created_at": round(time.time())}
    c["keys"] = list(c.get("keys") or []) + [row]
    return row, key


def revoke_key(cfg, cid: str, key_id: str) -> bool:
    c = get(cfg, cid)
    if c is None:
        return False
    keys = list(c.get("keys") or [])
    left = [k for k in keys if k.get("id") != key_id]
    c["keys"] = left
    if not left:
        c.pop("keys", None)
    return len(left) != len(keys)


def by_key(cfg, presented: str | None) -> dict | None:
    if not presented:
        return None
    h = _hash(presented)
    for c in all_of(cfg):
        if any(hmac.compare_digest(k.get("sha256", ""), h) for k in c.get("keys") or []):
            return c
    return None


def by_label(cfg, label: str | None) -> dict | None:
    if not label:
        return None
    for c in all_of(cfg):
        if any(fnmatch.fnmatchcase(label, g) for g in c.get("tags") or []):
            return c
    return None


def identify(cfg, headers, presented_key: str | None) -> tuple[dict | None, str | None]:
    """(the caller, the origin label) of a request: the key first, then the label's tag globs."""
    from nebula import names
    label = names.header(headers, "Caller")
    label = label.strip()[:120] if isinstance(label, str) and label.strip() else None
    c = by_key(cfg, presented_key) or by_label(cfg, label)
    return c, label


def view(cfg) -> list[dict]:
    """For the page: everything but the key hashes."""
    out = []
    for c in all_of(cfg):
        out.append({**{k: v for k, v in c.items() if k != "keys"},
                    "keys": [{k: v for k, v in key.items() if k != "sha256"} for key in c.get("keys") or []]})
    return out
