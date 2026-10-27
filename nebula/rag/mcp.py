"""nebula/rag/mcp.py: Lookup's library as an MCP server over HTTP (POST /mcp, JSON-RPC 2.0, a JSON answer per
request), for MCP clients outside Nebula (a desktop assistant, an IDE). The API's key rules apply; the key names the
tenant, so a guest's client sees only the collections shared with it. The security log is never offered here.
"""
from __future__ import annotations

import json

from nebula import __version__
from nebula.rag.tools import LibraryTools


def handle(server, tenant: str | None, raw: bytes) -> tuple[int, dict | None]:
    try:
        msg = json.loads(raw or b"{}")
    except ValueError:
        return 400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "not JSON"}}
    if not isinstance(msg, dict):
        return 400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "one request per POST"}}
    if "id" not in msg:
        return 202, None                                     # a notification: nothing to answer
    rid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    lib = getattr(server, "library", None)
    tools = LibraryTools(lib, tenant_of=lambda: tenant or "owner") if lib is not None else None
    if method == "initialize":
        res = {"protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
               "serverInfo": {"name": "nebula-lookup", "version": __version__}}
    elif method == "ping":
        res = {}
    elif method == "tools/list":
        res = {"tools": LibraryTools.tools if tools is not None else []}
    elif method == "tools/call":
        if tools is None:
            res = {"content": [{"type": "text", "text": getattr(server, "library_why", None) or "the library is off"}], "isError": True}
        else:
            try:
                res = {"content": [{"type": "text", "text": tools.call(str(params.get("name")), params.get("arguments") or {})}],
                       "isError": False}
            except Exception as e:  # noqa: BLE001 - the client reads the error
                res = {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
    else:
        return 200, {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no method {method}"}}
    return 200, {"jsonrpc": "2.0", "id": rid, "result": res}
