"""nebula/agents.py: which agent a request runs as, what it adds to the engine's body, and its tools.

The tools come from serve/mcp.py's McpHub (lifted): every configured MCP server's tools, named `<server>__<tool>`,
offered to the engine as function tools and run here when the engine calls them. An agent lists the servers whose
tools it may use. The hub starts with the cockpit and is shared by every turn.
"""
from __future__ import annotations

import logging
from pathlib import Path
from nebula import names

LOG = logging.getLogger("cockpit.agents")


class Tools:
    """The MCP hub (the configured servers, each its own process or URL) and Nebula's own in-process tool servers
    (Lookup's library, the security log: `local`, name -> an object with `tools` (MCP tool dicts) and
    `call(tool, arguments) -> str`). Both name their tools `<server>__<tool>`."""

    def __init__(self, servers: dict, settings: dict | None = None, local: dict | None = None):
        self.hub = None
        self.local = dict(local or {})
        if servers:
            from nebula._vendor.strata_serve.mcp import McpHub
            self.hub = McpHub(servers, settings or {})
            self.hub.start(wait=False)

    def _local_tools(self, names) -> list[dict]:
        out = []
        for n in names:
            srv = self.local.get(n)
            for t in (srv.tools if srv is not None else []):
                out.append({"type": "function", "function": {"name": f"{n}__{t['name']}"[:64], "description": t.get("description", ""),
                                                              "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}}})
        return out

    def openai_tools(self, server_names) -> list[dict]:
        """The function tools of those servers, for the engine's body."""
        if not server_names:
            return []
        out = self._local_tools([n for n in server_names if n in self.local])
        if self.hub is not None and any(n not in self.local for n in server_names):
            self.hub.wait(self.hub.settings.get("start_timeout_s", 120.0))
            wanted = tuple(f"{n}__" for n in server_names if n not in self.local)
            out += [t for t in self.hub.openai_tools() if t["function"]["name"].startswith(wanted)]
        return out

    def _local_of(self, name: str):
        server, _, tool = name.partition("__")
        srv = self.local.get(server)
        if srv is not None and any(t["name"] == tool for t in srv.tools):
            return srv, tool
        return None, None

    def owns(self, name: str) -> bool:
        return self._local_of(name)[0] is not None or (self.hub is not None and name in self.hub.routes())

    def call(self, name: str, arguments: dict) -> str:
        """The tool's text, or an "error: ..." text the model can read (serve/mcp.py's contract)."""
        srv, tool = self._local_of(name)
        if srv is not None:
            try:
                return srv.call(tool, arguments if isinstance(arguments, dict) else {})
            except Exception as e:  # noqa: BLE001 - the model reads the error and can react
                return f"error: {e}"
        r = self.hub.call(name, arguments if isinstance(arguments, dict) else {})
        return r.get("text") or ("" if r.get("ok") else "error: the tool answered nothing")

    def status(self) -> dict:
        st = dict(self.hub.status()) if self.hub is not None else {"servers": [], "tools": 0}
        for n, srv in self.local.items():
            st["servers"] = list(st.get("servers") or []) + [{
                "name": n, "transport": "nebula", "status": getattr(srv, "status", "ready"), "error": getattr(srv, "error", None),
                "tools": [{"name": f"{n}__{t['name']}", "tool": t["name"], "description": t.get("description", "")[:300]} for t in srv.tools]}]
            st["tools"] = int(st.get("tools") or 0) + len(srv.tools)
        return st

    def close(self) -> None:
        if self.hub is not None:
            self.hub.close()


def select_agent(cfg, req: dict, headers):
    """The X-Nebula-Agent header, else a model field naming an agent ("nebula/<name>" or the name), else the
    default agent, else None."""
    name = names.header(headers, "Agent")
    if name:
        a = cfg.agent(str(name))
        if a is None:
            raise ValueError(f"no agent named {name!r}")
        return a
    model = req.get("model")
    if isinstance(model, str):
        bare = names.bare_model(model)
        a = cfg.agent(bare)
        if a is not None:
            return a
    return cfg.default_agent()


def system_text(agent, base_dir: str | None) -> str:
    if agent.system_file:
        path = Path(agent.system_file)
        if not path.is_absolute() and base_dir:
            path = Path(base_dir) / path
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            LOG.warning("agent %s: system_file %s cannot be read: %s", agent.name, path, e)
    return agent.system or ""


def apply_agent(body: dict, agent, system: str, tools: list[dict]) -> None:
    """The agent's system prompt first, its tools beside the request's, its output shape when the request has none."""
    if system:
        messages = list(body.get("messages") or [])
        if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
            messages[0] = dict(messages[0], content=system + "\n\n" + messages[0]["content"])
        else:
            messages.insert(0, {"role": "system", "content": system})
        body["messages"] = messages
    if tools:
        have = {t.get("function", {}).get("name") for t in body.get("tools") or [] if isinstance(t, dict)}
        body["tools"] = list(body.get("tools") or []) + [t for t in tools if t["function"]["name"] not in have]
    if agent.output_schema and not body.get("response_format"):
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": agent.name, "strict": True,
                                                                           "schema": agent.output_schema}}
