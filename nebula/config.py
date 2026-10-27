"""nebula/config.py: what the cockpit is told at start.

Tiers are the spec's (docs/DESIGN.md, "The router"). Step 1 uses only the first tier's engine; the router
that chooses between them is step 3, and the fields are here so a config written now stays valid then.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from nebula import migrate


@dataclass
class Tier:
    name: str
    engine: str | None = None                 # base URL of a strata-lite-server; None for a provider tier
    thinking: bool | None = None              # None: as the request asks; False: off; True: on
    reasoning_budget_tokens: int | None = None
    provider: str | None = None               # "openai-compatible" for a remote tier
    base_url: str | None = None
    model: str | None = None                  # the model name a provider tier is asked for
    api_key_env: str | None = None            # the environment variable holding its key
    api_key_file: str | None = None           # a file holding its key (0600), written by the page's add-model form
    context: int | None = None                # its context size, when /props cannot say (a provider tier)
    host: str | None = None                   # registry metadata: where it runs
    power_class: str | None = None            # "local", "second-box", "cloud": what waking or paying it means
    cost_per_mtoken: float | None = None      # registry metadata: the price of a million tokens, if any
    cli: str | None = None                    # a command tier: one of the CLIs nebula/commands.py knows (claude, codex, grok)
    command: list | str | None = None         # or the owner's own command, with the prompt on its stdin (plan step F)
    cwd: str | None = None                    # where the command runs (default: nebula-work beside the config)
    slots: int | None = None                  # how many may run at once (default 2)
    sandbox: dict | None = None               # run the command in an OpenShell sandbox: {"image", "env", "providers", "policy", "model_via_nebula"} (step O1)
    peer: str | None = None                   # the uuid of the paired machine this tier came from, when it did
    headers: dict | None = None               # sent on every request to it; "{input}" in a value is the call's input (api, assistant, ...)
    risky: bool = False                       # the owner's mark: sharing it over Gravity warns both ends (nebula/gravity.py)
    not_guaranteed_safe: bool = False         # a peer's model its owner shared with a warning (shown here)
    shared: bool = False                      # guests may use it; every tenant is kept to its own cache (nebula/tenancy.py)
    cache_isolation: str | None = None        # "slots" (a slot per tenant, erased on change) or "no-cache" (guests: no cache)
    mesh: str | None = None                   # "gravity": a Gravity upstream (a router of other machines; nebula/pairlocal.py)
    timeout_s: float | None = None            # this route's own deadline for an engine call (default: engine_timeout_s)
    managed: str | None = None                # the engine this Nebula runs for it (an `engines` name, nebula/engines.py)
    health_path: str | None = None            # its health check (default /health; /v1/models for a router on a socket or a
                                              # router's HTTP caller address, where the catalog is the health)

    @property
    def is_command(self) -> bool:
        return bool(self.cli or self.command)

    @property
    def address(self) -> str | None:
        """Where its requests go: the engine, the provider's base URL (its /v1 stripped, as the client adds it), or
        `cli:<binary>` for a command tier."""
        if self.engine:
            return self.engine
        if self.base_url:
            return self.base_url.rstrip("/").removesuffix("/v1")
        if self.cli:
            return f"cli:{self.cli}"
        if self.command:
            words = self.command.split() if isinstance(self.command, str) else list(self.command)
            return f"cli:{words[0]}" if words else None
        return None

    @staticmethod
    def from_dict(d: dict) -> "Tier":
        if not isinstance(d, dict) or not d.get("name"):
            raise ValueError('a tier is an object with a "name"')
        known = {k: d.get(k) for k in Tier.__dataclass_fields__ if k in d}
        return Tier(**known)


@dataclass
class Agent:
    """A named process (docs/DESIGN.md, step 5): its own system prompt, tools, floor and ceiling on the
    ladder, thinking, and an output shape. A request picks one by model name ("nebula/<agent>"), by the
    X-Nebula-Agent header, or by the input's default. An `event` agent (the secretary) runs on the cockpit's own
    events, not on a user's turn."""
    name: str
    system: str | None = None
    system_file: str | None = None           # read relative to the config file
    tier: str | None = None                  # its floor; the first tier when None
    ceiling: str | None = None               # the highest tier the router may pick for it
    thinking: bool | None = None
    output_schema: dict | None = None        # a JSON schema the engine enforces on its answers
    tools: list = field(default_factory=list)   # names from "mcp_servers": their tools, run by the cockpit
    models: list = field(default_factory=list)  # the ticks: the tiers it may use, in order; empty = tier/ceiling on the ladder
    priority: str | None = None               # its class at the gates: interactive, agent, background; None = by the input
    default: bool = False
    event: bool = False
    max_rounds: int = 8                      # tool rounds in one turn

    @staticmethod
    def from_dict(d: dict) -> "Agent":
        if not isinstance(d, dict) or not d.get("name"):
            raise ValueError('an agent is an object with a "name"')
        known = {k: d.get(k) for k in Agent.__dataclass_fields__ if k in d}
        if "class" in d and "priority" not in known:
            known["priority"] = d["class"]
        for k in ("tools", "models"):
            if known.get(k) is None:
                known.pop(k, None)
        return Agent(**known)


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8090
    tiers: list[Tier] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)   # model names /v1/models lists beside the tiers' names
    engine_timeout_s: float = 600.0
    ledger: str | None = "nebula.sqlite3"             # the ledger's SQLite file (an existing cockpit.sqlite3 is kept); None keeps none
    tenant: str = "local"                             # tenant_id on every ledger row
    rules: dict = field(default_factory=dict)         # the router's thresholds (nebula/router.py Rules)
    compaction_fraction: float = 0.75                 # compact when a reply's prompt passed this share of the context
    compaction_full_every: int = 5                    # every Nth briefing of a session reads the whole conversation, not the last briefing + what came after (0: never)
    monitor: bool = True                              # sample the hardware for the page's monitor (serve/telemetry.py)
    agents: list = field(default_factory=list)        # Agent rows
    mcp_servers: dict = field(default_factory=dict)   # name -> MCP server config (serve/mcp.py's shape)
    mcp_settings: dict = field(default_factory=dict)
    secretary: str | None = None                      # the event agent compaction runs as; the first `event` agent when None
    default_models: list = field(default_factory=list)   # the ticks for a request with no agent; empty = the whole ladder
    lan: dict = field(default_factory=dict)           # {"decided", "discoverable", "name", "uuid", "peers": {uuid: {...}}} (nebula/lan.py)
    hooks: list = field(default_factory=list)         # webhook rows (nebula/hooks.py)
    schedules: list = field(default_factory=list)     # scheduled runs (nebula/hooks.py Scheduler)
    engine_logs: list = field(default_factory=list)   # {"tier", "file"|"command"}: engine logs to follow (nebula/enginelog.py)
    engines: list = field(default_factory=list)       # the engines this Nebula runs itself (nebula/engines.py)
    hosts: list = field(default_factory=list)         # machines a model can be on, and how to wake them (nebula/power.py)
    callers: list = field(default_factory=list)       # who uses its API: key, label tags, ticks, class (nebula/callers.py)
    watch: dict = field(default_factory=dict)         # optional: another Switchyard to draw by its URL, read only (nebula/switchyard.py)
    api: dict = field(default_factory=dict)           # the API Server star: require_key, keys (hashes), route (nebula/stars.py)
    assistant: dict = field(default_factory=dict)     # the Assistant star: route, system, tools (nebula/stars.py)
    base_dir: str | None = None                       # the config file's directory, for system_file
    path: str | None = None                           # the config file itself, for save()
    raw: dict = field(default_factory=dict)           # the file as loaded; save() rewrites only the sections it owns

    @property
    def engine(self) -> str:
        """The first tier's engine: where a request goes before the router has a reason to move it."""
        for t in self.tiers:
            if t.address:
                return t.address
        raise ValueError("no tier has an engine URL")

    def tier(self, name: str):
        for t in self.tiers:
            if t.name == name:
                return t
        return None

    def agent(self, name: str | None):
        for a in self.agents:
            if a.name == name:
                return a
        return None

    def default_agent(self):
        return next((a for a in self.agents if a.default), None)

    def secretary_agent(self):
        if self.secretary:
            return self.agent(self.secretary)
        return next((a for a in self.agents if a.event), None)

    @property
    def model_names(self) -> list[str]:
        return [t.name for t in self.tiers] + [a for a in self.aliases if a not in {t.name for t in self.tiers}]

    def to_dict(self) -> dict:
        """The file to write: what was loaded, with the sections the page edits replaced."""
        def tier_dict(t):
            return {k: v for k, v in t.__dict__.items() if v is not None and v is not False and v != []}

        def agent_dict(a):
            out = {k: v for k, v in a.__dict__.items() if v is not None and v is not False and v != []}
            if a.max_rounds == 8:
                out.pop("max_rounds", None)
            return out
        d = dict(self.raw)
        d["config_version"] = migrate.CONFIG_VERSION
        d["tiers"] = [tier_dict(t) for t in self.tiers]
        d["agents"] = [agent_dict(a) for a in self.agents if not getattr(a, "builtin", False)]
        for k in ("api", "assistant"):
            v = getattr(self, k)
            if v:
                d[k] = dict(v)
            else:
                d.pop(k, None)
        if self.mcp_servers:
            d["mcp_servers"] = dict(self.mcp_servers)
            d.pop("mcpServers", None)
        if self.default_models:
            d["default_models"] = list(self.default_models)
        else:
            d.pop("default_models", None)
        if self.lan:
            d["lan"] = dict(self.lan)
        if self.engines:
            d["engines"] = [dict(e) for e in self.engines]
        else:
            d.pop("engines", None)
        if self.callers:
            d["callers"] = [dict(c) for c in self.callers]
        else:
            d.pop("callers", None)
        if self.hosts:
            d["hosts"] = [dict(h) for h in self.hosts]
        else:
            d.pop("hosts", None)
        return d

    def save(self) -> str:
        """Write the config file (atomically) and return its path. Needs `path`."""
        if not self.path:
            raise ValueError("this configuration was not read from a file: nothing to save to")
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1, ensure_ascii=False)
            f.write("\n")
        import os
        os.replace(tmp, self.path)
        return self.path

    @staticmethod
    def from_dict(d: dict) -> "Config":
        migrate.check_config(d)
        cfg = Config()
        cfg.raw = {k: v for k, v in d.items()}
        for k in ("host", "port", "engine_timeout_s", "ledger", "tenant", "rules", "compaction_fraction", "compaction_full_every", "monitor"):
            if k in d:
                setattr(cfg, k, d[k])
        cfg.tiers = [Tier.from_dict(t) for t in d.get("tiers") or []]
        cfg.aliases = [str(a) for a in d.get("aliases") or []]
        cfg.agents = [Agent.from_dict(a) for a in d.get("agents") or []]
        cfg.mcp_servers = dict(d.get("mcp_servers") or d.get("mcpServers") or {})
        cfg.mcp_settings = dict(d.get("mcp_settings") or {})
        cfg.secretary = d.get("secretary")
        cfg.lan = dict(d.get("lan") or {})
        cfg.hooks = [dict(h) for h in d.get("hooks") or []]
        cfg.schedules = [dict(x) for x in d.get("schedules") or []]
        cfg.engines = [dict(e) for e in d.get("engines") or []]
        from nebula.engines import check_spec
        for e in cfg.engines:
            check_spec(e)
        from nebula import power as PW
        cfg.hosts = [PW.check(h) for h in d.get("hosts") or []]
        from nebula import callers as C
        cfg.callers = [C.check(c, [t.get("name") for t in d.get("tiers") or []] or None) for c in d.get("callers") or []]
        cfg.watch = dict(d.get("watch") or {})
        cfg.api = dict(d.get("api") or {})
        cfg.assistant = dict(d.get("assistant") or {})
        cfg.engine_logs = [dict(x) for x in d.get("engine_logs") or []]
        for x in cfg.engine_logs:
            if not x.get("tier") or not (x.get("file") or x.get("command")):
                raise ValueError('an engine_logs entry is {"tier", "file"} or {"tier", "command"}')
        for h in cfg.hooks + cfg.schedules:
            if not h.get("name"):
                raise ValueError('a hook or schedule is an object with a "name"')
        names = [t.name for t in cfg.tiers]
        for a in cfg.agents:
            if a.name in names:
                raise ValueError(f"agent {a.name!r} has a tier's name: tiers and agents share the model-name space")
        cfg.default_models = [str(m) for m in d.get("default_models") or []]
        for owner, ticks in [("default_models", cfg.default_models)] + [(f"agent {a.name}", a.models) for a in cfg.agents] \
                + [(f"hook {h.get('name')}", h.get("models") or []) for h in cfg.hooks] \
                + [(f"schedule {x.get('name')}", x.get("models") or []) for x in cfg.schedules]:
            unknown = [m for m in ticks if m not in names]
            if unknown:
                raise ValueError(f"{owner}: ticked models {unknown} are not tiers")
            if len(set(ticks)) != len(ticks):
                raise ValueError(f"{owner}: a model is ticked twice")
        return cfg

    @staticmethod
    def from_args(argv=None) -> "Config":
        p = argparse.ArgumentParser(prog="nebula", description="Nebula: the server above a local engine.")
        p.add_argument("--config", help="a JSON file with host, port, tiers and aliases (see docs/DESIGN.md)")
        p.add_argument("--engine", help="the engine's base URL (one tier named 'local'); default http://127.0.0.1:8080")
        p.add_argument("--host")
        p.add_argument("--port", type=int)
        p.add_argument("--alias", action="append", default=[], help="another model name to list and answer to")
        p.add_argument("--engine-timeout", type=float, help="seconds to wait on the engine (default 600)")
        p.add_argument("--ledger", help="the ledger's SQLite file (default nebula.sqlite3 in the current directory, or an existing cockpit.sqlite3)")
        p.add_argument("--no-ledger", action="store_true", help="keep no ledger")
        p.add_argument("--tenant", help="tenant_id written on every ledger row (default local)")
        a = p.parse_args(argv)
        cfg = Config.from_dict(migrate.read_config(a.config)) if a.config else Config()
        if a.config:
            cfg.base_dir = str(Path(a.config).resolve().parent)
            cfg.path = str(Path(a.config).resolve())
        # the one-engine default only for a bare start: a config that lists its tiers (even none), runs its own engines
        # or is on a network (a model server, a household's main) gets no guessed engine on :8080
        planned = "tiers" in cfg.raw or cfg.engines or (cfg.raw.get("network") or {}).get("enabled")
        if a.engine or (not cfg.tiers and not cfg.watch and not planned):
            cfg.tiers.insert(0, Tier(name="local", engine=a.engine or "http://127.0.0.1:8080"))
        if a.host:
            cfg.host = a.host
        if a.port:
            cfg.port = a.port
        if a.engine_timeout:
            cfg.engine_timeout_s = a.engine_timeout
        cfg.aliases += [x for x in a.alias if x not in cfg.aliases]
        if a.ledger:
            cfg.ledger = os.path.abspath(a.ledger)          # a command line's path is from where it ran
        if a.no_ledger:
            cfg.ledger = None
        if a.tenant:
            cfg.tenant = a.tenant
        return cfg
