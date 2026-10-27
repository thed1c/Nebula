"""nebula/rag/tools.py: Lookup's own tools, served in Nebula's process (nebula/agents.py's `local` servers) and over
MCP at POST /mcp (nebula/rag/mcp.py).

- **library**: search the collections the asking tenant may see, read round a result, list them. Answers carry
  citations: `[n] title, page N (heading)`.
- **security**: the security log (nebula/seclog.py), read only, for the owner and Halo alone; a guest is refused.

The tenant is the turn's (nebula.engine.CALL_TENANT), or the MCP caller's key.
"""
from __future__ import annotations

import json
import time

from nebula.engine import CALL_TENANT

OWNER = "owner"


class LibraryTools:
    tools = [
        {"name": "search", "description": "Search the document collections for passages about a question. Returns numbered "
         "passages with where each is from (file, page, heading); cite them as [n] in the answer.",
         "inputSchema": {"type": "object", "properties": {
             "query": {"type": "string", "description": "what to look for, in plain words"},
             "collections": {"type": "array", "items": {"type": "string"}, "description": "collection ids (default: all you may see)"},
             "limit": {"type": "integer", "description": "how many passages (default 8, at most 30)"}}, "required": ["query"]}},
        {"name": "read_passage", "description": "Read more round a passage a search returned: its chunk with its neighbours.",
         "inputSchema": {"type": "object", "properties": {"chunk": {"type": "integer"}, "around": {"type": "integer"}},
                         "required": ["chunk"]}},
        {"name": "collections", "description": "The document collections you may search, with how many documents each holds.",
         "inputSchema": {"type": "object", "properties": {}}},
    ]

    def __init__(self, library, tenant_of=None):
        self.library = library
        self.tenant_of = tenant_of or (lambda: CALL_TENANT.get())

    @property
    def status(self) -> str:
        return {"ready": "ready", "starting": "starting"}.get(self.library.state, "error")

    @property
    def error(self):
        return self.library.why

    def call(self, tool: str, a: dict) -> str:
        tenant = self.tenant_of()
        if tool == "search":
            rows = self.library.search(str(a.get("query") or ""), tenant, a.get("collections") or None, a.get("limit") or 8)
            if not rows:
                return "no passages found (or no collection you may search)"
            out = []
            for r in rows:
                where = r["title"] + (f", page {r['page']}" if r.get("page") else "") + (f" ({r['heading']})" if r.get("heading") else "")
                out.append(f"[{r['n']}] {where} — {r['collection_name']} · chunk {r['chunk']}\n{r['text']}")
            return "\n\n".join(out)
        if tool == "read_passage":
            p = self.library.passage(int(a.get("chunk")), tenant, int(a.get("around") or 1))
            return f"{p['title']}" + (f", page {p['page']}" if p.get("page") else "") + f"\n\n{p['text']}"
        if tool == "collections":
            v = self.library.view(tenant)
            return "\n".join(f"{c['id']}: {c['name']} ({c['indexed']} documents)" for c in v["collections"]) or "no collections"
        raise ValueError(f"no tool {tool}")


class SecurityTools:
    tools = [
        {"name": "read_log", "description": "Read the security log (newest first): refusals, key use, Gravity joins and "
         "grants, guests refused, sharing changes, Halo's own actions. Read only.",
         "inputSchema": {"type": "object", "properties": {
             "since_seq": {"type": "integer", "description": "only rows after this one"},
             "kinds": {"type": "array", "items": {"type": "string"}, "description": "e.g. [\"auth.refused\", \"gravity.*\"]"},
             "min_severity": {"type": "string", "enum": ["info", "notice", "warn", "alert"]},
             "since_minutes": {"type": "number"}, "who": {"type": "string"},
             "limit": {"type": "integer", "description": "default 50"}}}},
        {"name": "counts", "description": "How many events of each kind in the last N minutes.",
         "inputSchema": {"type": "object", "properties": {"minutes": {"type": "number"}}}},
    ]
    status, error = "ready", None

    def __init__(self, seclog, tenant_of=None):
        self.log = seclog
        self.tenant_of = tenant_of or (lambda: CALL_TENANT.get())

    def call(self, tool: str, a: dict) -> str:
        if self.tenant_of() not in (OWNER, None):
            raise PermissionError("the security log is the owner's")
        if tool == "read_log":
            rows = self.log.read(since_seq=int(a.get("since_seq") or 0), kinds=a.get("kinds") or None, who=a.get("who") or None,
                                 min_severity=a.get("min_severity") or None,
                                 since_s=float(a["since_minutes"]) * 60 if a.get("since_minutes") else None,
                                 limit=max(1, min(int(a.get("limit") or 50), 200)))
            for r in rows:
                r["at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["at"]))
                r["last_at"] = time.strftime("%H:%M:%S", time.localtime(r["last_at"]))
            return json.dumps(rows, ensure_ascii=False)
        if tool == "counts":
            return json.dumps(self.log.counts(float(a.get("minutes") or 60) * 60))
        raise ValueError(f"no tool {tool}")


def local_servers(server) -> dict:
    """Nebula's own tool servers, as configured in mcp_servers (`{"builtin": "library"}`, `{"builtin": "security"}`)."""
    out = {}
    for name, c in (server.cfg.mcp_servers or {}).items():
        if not isinstance(c, dict) or c.get("disabled") is True:
            continue
        if c.get("builtin") == "library" and getattr(server, "library", None) is not None:
            out[name] = LibraryTools(server.library)
        elif c.get("builtin") == "security" and getattr(server, "seclog", None) is not None:
            out[name] = SecurityTools(server.seclog)
    if getattr(server, "halo", None) is not None:          # Halo's own tools: only its turn runs them (nebula/halo.py)
        from nebula.halo import HaloTools
        out["halo"] = HaloTools(server)
    return out
