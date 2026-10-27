"""The stdio MCP loop the built-in Lookup servers share: initialize, tools/list, tools/call, nothing else."""
from __future__ import annotations

import json
import sys


def text(t: str, error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": t}], "isError": error}


def serve(name: str, tools: list[dict], call) -> None:
    """Answer requests on stdin until it closes. `call(tool, arguments)` returns the text, or raises to report an error."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if "id" not in msg:
            continue                                          # notifications: nothing to answer
        rid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if method == "initialize":
            res = {"protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
                   "serverInfo": {"name": name, "version": "0.1"}}
        elif method == "tools/list":
            res = {"tools": tools}
        elif method == "tools/call":
            try:
                res = text(call(params.get("name"), params.get("arguments") or {}))
            except Exception as e:  # noqa: BLE001 - the model reads the error and can react
                res = text(f"{type(e).__name__}: {e}", error=True)
        elif method == "ping":
            res = {}
        else:
            out = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no method {method}"}}
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}) + "\n")
        sys.stdout.flush()
