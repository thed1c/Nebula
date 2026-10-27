"""nebula/tenancy.py: who a request is for, and keeping tenants apart on the engines they share (docs/sharing.md).

**The tenant** of a request is its caller (`caller:<id>`, named by its key or its label: the API Node) or its paired
peer (`peer:<uuid>`: Gravity). The owner's own traffic (the page, the Assistant, keyless local calls) is one tenant,
`owner`. A **guest** is a paired peer, or a caller marked `"guest": true` (someone else's app); a guest may only use
models marked `"shared": true`.

**No engine has a cache salt** (llama.cpp, Impulse, Strata: the cached prompt of one request is reused by the next
one that shares its prefix, which shows in the time to the first token). On a model marked `shared`, every request is
kept to its tenant's own cache:
- `cache_isolation: "slots"` (llama.cpp and Strata engines Nebula runs): each tenant is pinned to a slot of its own
  (`id_slot`); a slot that changes tenant is erased first (`POST /slots/N?action=erase`), so nothing cached by one
  tenant is ever served to another, and a tenant's own next request still reuses its prefix;
- `cache_isolation: "no-cache"` (the default elsewhere, Impulse today): a guest's request is sent with
  `cache_prompt: false`, so it can never hit anyone's cache.
Sessions are kept per tenant too: two tenants whose conversations open the same way never share one.
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.error
import urllib.request

LOG = logging.getLogger("nebula.tenancy")
OWNER = "owner"
MODES = ("slots", "no-cache")


def tenant_of(caller: dict | None, peer: str | None) -> tuple[str, bool]:
    """(the tenant, whether it is a guest)."""
    if peer and not str(peer).startswith("sandbox:"):    # a sandboxed CLI of the owner's own is the owner
        return f"peer:{peer}", True
    if caller and caller.get("id"):
        return f"caller:{caller['id']}", bool(caller.get("guest"))
    return OWNER, False


def mode_of(tier) -> str:
    m = getattr(tier, "cache_isolation", None)
    return m if m in MODES else "no-cache"


class Slots:
    """Per engine address: which tenant each slot last served, and each tenant's slot."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last: dict[str, dict[int, str]] = {}           # address -> slot -> tenant whose prompt it holds
        self.mine: dict[str, dict[str, int]] = {}           # address -> tenant -> its slot
        self.used: dict[str, dict[int, float]] = {}         # address -> slot -> when it was last given out

    def assign(self, address: str, slots: int, tenant: str) -> tuple[int, bool]:
        """(the tenant's slot, whether it held another tenant's prompt and must be erased first)."""
        n = max(1, int(slots or 1))
        with self.lock:
            last, mine, used = (self.last.setdefault(address, {}), self.mine.setdefault(address, {}),
                                self.used.setdefault(address, {}))
            s = mine.get(tenant)
            if s is None or s >= n:
                free = [i for i in range(n) if i not in last]
                if free:
                    s = free[0]
                else:                                     # every slot holds someone's prompt: the least recently used
                    s = min(range(n), key=lambda i: used.get(i, 0))
                for t, i in list(mine.items()):
                    if i == s:
                        del mine[t]
                mine[tenant] = s
            dirty = last.get(s) not in (None, tenant)
            last[s], used[s] = tenant, time.time()
            return s, dirty


def erase(engine, slot: int) -> bool:
    """Erase a slot's cached prompt on a llama.cpp-style server."""
    url = engine.base.rstrip("/") + f"/slots/{int(slot)}?action=erase"
    h = {"Content-Type": "application/json", **{k: str(v) for k, v in (engine.headers or {}).items()}}
    if getattr(engine, "api_key", None):
        h["Authorization"] = f"Bearer {engine.api_key}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=b"{}", headers=h, method="POST"), timeout=10) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError) as e:
        LOG.warning("could not erase slot %s at %s: %s", slot, engine.base, e)
        return False


def isolate(body: dict, tier, engine, tenant: str, guest: bool, slots: Slots | None) -> str | None:
    """Keep this request to its tenant's own cache on a shared model. Returns what was done, for the decision."""
    if not getattr(tier, "shared", False):
        return None
    if mode_of(tier) == "slots" and slots is not None and tier.address:
        n = getattr(tier, "slots", None) or 1
        s, dirty = slots.assign(tier.address, n, tenant)
        if dirty and not erase(engine, s):
            body["cache_prompt"] = False                  # could not erase: never serve the other tenant's prefix
            body["id_slot"] = s
            return f"slot {s} (another tenant's, not erased: no cache)"
        body["id_slot"] = s
        return f"slot {s}" + (" (erased: it held another tenant's prompt)" if dirty else "")
    if tenant != OWNER:
        body["cache_prompt"] = False
        return "no cache (this engine cannot keep tenants apart)"
    return None
