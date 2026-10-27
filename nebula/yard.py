"""nebula/yard.py: the page's API for this Nebula's own Switchyard (docs/switchyard.md).

    GET  /api/yard/engines                    the managed engines (state, fit, command, version) and this machine's cards
    GET  /api/yard/engines/<name>             one engine, with its deployment history
    POST /api/yard/engines                    add or change an engine (a new deployment version)
    POST /api/yard/engines/<name>/<action>    start | stop | drain | restart | verify | undo | remove
    POST /api/yard/engines/<name>/route       make it a model (a tier on its address), unless one is already there
    GET  /api/yard/packages                   the engine packages, each one's settings (the Edit form) and binary/version
    POST /api/yard/raw                        {engine, text}: the Raw tab's command line read back into settings (kept
                                              verbatim when unknown; Nebula's own hardening locked), the diff, the line
    POST /api/yard/preview                    {engine}: the command line it would run, its fit, or why it cannot run
    POST /api/yard/hf/resolve                 {url}: a Hugging Face link's GGUF choices, sizes, projector, disk and fit
    POST /api/yard/hf/pull                    {repo, revision, files}: start a download; GET /api/yard/hf/pulls follows it
    POST /api/yard/hf/pulls/<id>/cancel       pause it (pulling again resumes)
    POST /api/yard/hf/token                   {token}: kept in nebula-keys/huggingface.token (0600), never shown again
    GET  /api/yard/openshell                  OpenShell here: gateways, sandboxes, providers (read only)
    GET  /api/yard/openshell/profile          ?port=: the provider profile that lets a sandbox reach only this Nebula
    GET  /api/yard/openshell/sandboxes/<name>/policy   a sandbox's effective policy (shown, never written from here)
    POST /api/yard/openshell/sandboxes/<name>/delete   stop a sandbox Nebula started (an engine's)
    GET  /api/yard/network                    model servers on your network: this machine's role, the members, the models
    POST /api/yard/network/on {role} | /off   main (where you work) or server (a headless model server)
    POST /api/yard/network/invite {address}   (main) invite a server: its PIN; /state {inviteId}; /adopt (its models here)
    POST /api/yard/network/accept {pin}       (server) answer the main Nebula's invite; /engine: this Nebula as its engine
    POST /api/yard/servers/scan {host, port}  Add a server (SSH): its host keys, to confirm; /trust {host, port, fingerprint}
    POST /api/yard/servers/preflight          {host, port, user, key_file}: what it is, the exact commands, and `needs`:
                                              what an administrator should do there or here (fixed tasks)
    POST /api/yard/servers/admin              {host, port, user, key_file, tasks, password, local_password}: those tasks
                                              with sudo (the passwords on sudo's stdin once; never kept or logged)
    POST /api/yard/servers/install            {..., name}: a model machine: PAIR's node, the node keeper and OpenShell
                                              (rootless Podman) there, no Nebula; joined to this machine's network (mTLS),
                                              its gateway registered here; /servers/<name>/uninstall
    GET  /api/yard/servers                    the servers added this way
    GET  /api/yard/node                       the Gravity node (Nebula to Nebula over iroh): id, peers, grants
    POST /api/yard/node/on | /off             turn it on (a key of its own) or off; /relays {urls}: the gateway
    POST /api/yard/node/invite {kind, grants?, confirm?}  a one-time invite (friend | owner: one of your own machines);
                                              a friend's may carry the models it gets on joining; /invites/<id>/cancel
    POST /api/yard/node/join {invite}         join the Nebula that made it
    POST /api/yard/node/peers/<id>/grants {models} | /revoke | /adopt (its granted models as models here)
    *    /api/yard/node/peers/<id>/admin/<path>   an owner machine's /api/yard/<path>, over its admin ALPN
    GET  /api/yard/gravity                    the Gravity routers on this machine (loopback only) and the ones adopted
    POST /api/yard/gravity/adopt              {address, models}: one model per Gravity model, routed like any other
    GET  /api/yard/hosts                      the machines that sleep: awake or not, their models, how they wake
    POST /api/yard/hosts                      add or change one; POST /api/yard/hosts/<name>/wake | /remove
    GET  /api/yard/lanes | decisions | stats | status   this Nebula's own Switchyard (nebula/ownyard.py)
    POST /api/yard/deploy                     {engine, start, route}: Deploy your own's last step (add, start, make a model)
    GET  /api/yard/callers                    the registered callers (no key hashes)
    POST /api/yard/callers                    add or change one {id, name, class, tags, models}
    POST /api/yard/callers/<id>/keys          {label}: a new key, shown this once
    POST /api/yard/callers/<id>/keys/<key>/revoke
    POST /api/yard/callers/<id>/remove

Every write is saved to the config at once and the models are reloaded; nothing here talks to anything but this
Nebula's own engines.
"""
from __future__ import annotations

import json
import os
import re
import time

from nebula import engines as E


class YardError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _save(server) -> str | None:
    cfg = server.cfg
    return cfg.save() if cfg.path else None


def models_dir(server) -> str:
    cfg = server.cfg
    d = (cfg.raw or {}).get("models_dir") or "models"
    return d if os.path.isabs(d) else os.path.join(cfg.base_dir or os.getcwd(), d)


def pulls(server):
    if getattr(server, "pulls", None) is None:
        from nebula import hf, names
        cfg = server.cfg
        keys = names.local_path(cfg.base_dir, "nebula-keys", "cockpit-keys") if cfg.path else None
        server.pulls = hf.Pulls(models_dir(server), os.path.join(keys, "huggingface.token") if keys else None)
    return server.pulls


def packages_view() -> dict:
    from nebula import packages as P
    out = []
    for name, p in P.PACKAGES.items():
        found = P.locate(name)
        out.append({"name": name, **p, "found": found, "schema": P.schema(name)})
    return {"packages": out, "groups": P.GROUPS}


def free_port(server, start: int = 18300) -> int:
    used = {int(e.get("port") or 0) for e in server.cfg.engines} | {server.server_port}
    import socket
    for port in range(start, start + 500):
        if port in used:
            continue
        with socket.socket() as so:
            try:
                so.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise YardError(503, "no free port")


def get(server, path: str, query: str = "") -> dict | None:
    yard = getattr(server, "yard", None)
    if yard is None:
        return None
    if path == "/api/yard/packages":
        return packages_view()
    if path in ("/api/yard/lanes", "/api/yard/decisions", "/api/yard/stats", "/api/yard/status"):
        from nebula import ownyard
        if path == "/api/yard/status":
            return ownyard.status(server)
        if path == "/api/yard/decisions":
            return ownyard.decisions(server)
        if path == "/api/yard/stats":
            return ownyard.stats(server)
        from nebula.switchyard import SwitchyardWatch           # the same reader as a watched router's lanes
        w = SwitchyardWatch({"url": "self"}, fetch=lambda: ownyard.status(server))
        w.poll_once()
        return {"lanes": w.lanes(), "read_at": w.summary()["read_at"]}
    if path.startswith("/api/yard/openshell"):
        from nebula import openshell
        try:
            if path == "/api/yard/openshell":
                return openshell.panel()
            if path == "/api/yard/openshell/profile":
                port = server.server_port
                return {"profile": openshell.profile(port), "steps": openshell.provider_steps(port),
                        "note": "give the provider a key from the Callers tab: the sandbox's calls are that caller's"}
            if path.startswith("/api/yard/openshell/sandboxes/") and path.endswith("/policy"):
                return openshell.policy(path[len("/api/yard/openshell/sandboxes/"):-len("/policy")])
        except openshell.OpenShellError as e:
            raise YardError(503, str(e)) from None
    if path == "/api/yard/servers":
        return {"servers": list(server.cfg.raw.get("servers") or [])}
    if path == "/api/yard/network":
        n = getattr(server, "network", None)
        return n.view() if n is not None else {"on": False, "role": (server.cfg.raw.get("network") or {}).get("role")}
    if path == "/api/yard/node/models":
        from nebula import gravity
        return {"models": [{"name": t.name, "shared": bool(getattr(t, "shared", False)), "anti_gravity": gravity.anti_gravity(t),
                            "needs_care": gravity.needs_care(t, getattr(server, "yard", None)), "not_guaranteed_safe": bool(getattr(t, "not_guaranteed_safe", False)),
                            "mesh": getattr(t, "mesh", None)} for t in server.cfg.tiers]}
    if path == "/api/yard/node":
        g = getattr(server, "gravity", None)
        if g is None:
            from nebula import gravity
            ok, why = gravity.available()
            return {"on": False, "available": ok, "why": why}
        return {"on": True, **g.view()}
    if path.startswith("/api/yard/node/peers/") and "/admin/" in path:
        return node_admin(server, path, "GET", None)
    if path == "/api/yard/gravity":
        from nebula import pairlocal
        ports = server.cfg.raw.get("gravity_local_ports")
        here = pairlocal.find([(int(p), "openai") for p in ports] if ports else None)
        return {"here": here, "adopted": [{"name": t.name, "address": t.address, "model": t.model}
                                          for t in server.cfg.tiers if getattr(t, "mesh", None) == "gravity"]}
    if path == "/api/yard/hosts":
        return {"hosts": server.power.view() if getattr(server, "power", None) else []}
    if path == "/api/yard/callers":
        from nebula import callers
        return {"callers": callers.view(server.cfg), "classes": list(callers.CLASSES), "models": [t.name for t in server.cfg.tiers]}
    if path == "/api/yard/hf/pulls":
        return {"pulls": pulls(server).status(), "models_dir": models_dir(server), "token": bool(pulls(server).token())}
    if path == "/api/yard/engines":
        from nebula import machines
        return {"engines": yard.status(), "cards": E.cards() + machines.cards(server.cfg), "kinds": list(E.KINDS),
                "cache_types": list(E.CACHE_BYTES), "machines": [s["name"] for s in server.cfg.raw.get("servers") or [] if s.get("gateway")]}
    if path.startswith("/api/yard/engines/"):
        name = path[len("/api/yard/engines/"):].strip("/")
        try:
            return {**yard.status(name), "history": yard.history(name)}
        except KeyError:
            raise YardError(404, f"no engine named {name!r}") from None
    return None


def post(server, path: str, req: dict) -> dict | None:
    yard = getattr(server, "yard", None)
    if yard is None:
        return None
    from nebula import callers, hf, power
    try:
        if path.startswith("/api/yard/openshell/sandboxes/") and path.endswith("/delete"):
            from nebula import openshell
            name = path[len("/api/yard/openshell/sandboxes/"):-len("/delete")]
            if not openshell.is_engine_sandbox(name):
                raise YardError(403, "only sandboxes Nebula started for its engines are stopped from here")
            try:
                openshell.delete(name)
            except openshell.OpenShellError as e:
                raise YardError(503, str(e)) from None
            return {"deleted": name}
        if path.startswith("/api/yard/node"):
            return node_post(server, path, req)
        if path.startswith("/api/yard/network"):
            return network_post(server, path, req)
        if path.startswith("/api/yard/servers"):
            return servers_post(server, path, req)
        if path == "/api/yard/gravity/adopt":
            return {**gravity_adopt(server, req), "saved": _save(server)}
        if path == "/api/yard/hosts":
            h = power.check(req.get("host") if isinstance(req.get("host"), dict) else req)
            server.cfg.hosts = [x for x in server.cfg.hosts if x.get("name") != h["name"]] + [h]
            return {"host": h["name"], "saved": _save(server)}
        if path.startswith("/api/yard/hosts/"):
            name, _, action = path[len("/api/yard/hosts/"):].strip("/").partition("/")
            if action == "wake":
                return server.power.wake(name)
            if action == "remove":
                server.cfg.hosts = [x for x in server.cfg.hosts if x.get("name") != name]
                return {"removed": name, "saved": _save(server)}
            raise YardError(404, f"no host action {action!r}")
        if path == "/api/yard/callers":
            c = callers.put(server.cfg, req.get("caller") if isinstance(req.get("caller"), dict) else req)
            return {"caller": c["id"], "saved": _save(server)}
        if path.startswith("/api/yard/callers/"):
            cid, _, rest = path[len("/api/yard/callers/"):].strip("/").partition("/")
            if callers.get(server.cfg, cid) is None:
                raise YardError(404, f"no caller {cid!r}")
            if rest == "keys":
                row, key = callers.make_key(server.cfg, cid, str(req.get("label") or ""))
                return {"key": key, "row": {k: v for k, v in row.items() if k != "sha256"}, "saved": _save(server),
                        "note": "the key is shown this once; keep it with the app"}
            if rest.startswith("keys/") and rest.endswith("/revoke"):
                ok = callers.revoke_key(server.cfg, cid, rest[len("keys/"):-len("/revoke")])
                return {"revoked": ok, "saved": _save(server)}
            if rest == "remove":
                return {"removed": callers.remove(server.cfg, cid), "saved": _save(server)}
            raise YardError(404, f"no caller action {rest!r}")
        if path == "/api/yard/raw":
            return raw_post(server, req)
        if path == "/api/yard/preview":
            spec = dict(req.get("engine") or req)
            spec.setdefault("port", 1)
            s = E.check_spec(spec)
            return {"command": E.command_of(s, server.cfg.base_dir), "fit": E.fit(s, server.cfg.base_dir), "spec": s}
        if path == "/api/yard/hf/resolve":
            r = hf.resolve(str(req.get("url") or ""), pulls(server).token())
            import shutil
            md = models_dir(server)
            os.makedirs(md, exist_ok=True)
            r["disk_free"] = shutil.disk_usage(md).free
            cards = E.cards()
            best = max((c["memory_mb"] - c["used_mb"] for c in cards), default=None)
            for c in r["choices"]:
                c["fits_disk"] = c["size"] <= r["disk_free"]
                c["fits_card"] = None if best is None else c["size"] / 2**20 + E.OVERHEAD_MB <= best
            r["best_card_free_mb"] = best
            return r
        if path == "/api/yard/hf/pull":
            files = [str(f) for f in req.get("files") or []]
            meta = {f["path"]: f for f in req.get("meta") or []}
            pid = pulls(server).start(str(req.get("repo") or ""), files, str(req.get("revision") or "main"),
                                      sha256={k: v.get("sha256") for k, v in meta.items()},
                                      sizes={k: v.get("size") for k, v in meta.items()})
            return pulls(server).status(pid)
        if path.startswith("/api/yard/hf/pulls/") and path.endswith("/cancel"):
            return pulls(server).cancel(path[len("/api/yard/hf/pulls/"):-len("/cancel")])
        if path == "/api/yard/hf/token":
            return {"saved": pulls(server).set_token(req.get("token") or None)}
        if path == "/api/yard/deploy":
            spec = dict(req.get("engine") or {})
            if not spec.get("port"):
                spec["port"] = free_port(server)
            out = yard.apply(spec)
            out["saved"] = _save(server)
            name = E.check_spec(spec)["name"]
            if req.get("route", True):
                out["route"] = route_for(server, name, {"tier": req.get("tier") or name})
                _save(server)
            if req.get("start", True):
                out["engine"] = yard.start(name, wait=bool(req.get("wait")))
            return out
        if path == "/api/yard/engines":
            out = yard.apply(req.get("engine") if isinstance(req.get("engine"), dict) else req)
            sync_tiers(server, out["name"])
            out["saved"] = _save(server)
            return out
        if path.startswith("/api/yard/engines/"):
            name, _, action = path[len("/api/yard/engines/"):].strip("/").partition("/")
            yard.spec(name)                                    # KeyError: no such engine
            if action == "start":
                return yard.start(name, wait=bool(req.get("wait")))
            if action == "stop":
                return yard.stop(name)
            if action == "drain":
                return yard.drain(name, timeout_s=float(req.get("timeout_s") or 600), then_stop=req.get("then_stop", True) is not False)
            if action == "restart":
                return yard.restart(name, wait=bool(req.get("wait")))
            if action == "verify":
                return yard.verify(name)
            if action == "undo":
                out = yard.undo(name)
                out["saved"] = _save(server)
                return out
            if action == "remove":
                out = yard.remove(name)
                out["saved"] = _save(server)
                return out
            if action == "route":
                return {**route_for(server, name, req), "saved": _save(server)}
            raise YardError(404, f"no engine action {action!r}")
    except KeyError as e:
        raise YardError(404, f"no engine named {e.args[0]!r}") from None
    except E.EngineSpecError as e:
        raise YardError(400, str(e)) from None
    except hf.HfError as e:
        raise YardError(e.status, str(e)) from None
    except callers.CallerError as e:
        raise YardError(400, str(e)) from None
    except power.HostError as e:
        raise YardError(400, str(e)) from None
    return None


def node_post(server, path: str, req: dict) -> dict:
    from nebula import gravity
    g = getattr(server, "gravity", None)
    try:
        if path == "/api/yard/node/on":
            sec = server.cfg.raw.setdefault("gravity", {})
            sec["enabled"] = True
            if req.get("name"):
                sec["name"] = str(req["name"])[:60]
            if g is None:
                server.gravity = g = gravity.GravityNode(server).start()
            _save(server)
            return {"on": True, **g.view()}
        if path == "/api/yard/node/off":
            server.cfg.raw.setdefault("gravity", {})["enabled"] = False
            if g is not None:
                g.close()
                server.gravity = None
            _save(server)
            return {"on": False}
        if g is None:
            raise YardError(409, "the Gravity node is off: turn it on first")
        if path == "/api/yard/node/relays":
            urls = [u for u in req.get("urls") or [] if isinstance(u, str) and u.startswith(("http://", "https://"))]
            g.section()["relays"] = urls
            _save(server)
            g.close()
            server.gravity = gravity.GravityNode(server).start()       # a new endpoint with the new relays
            return server.gravity.view()
        if path == "/api/yard/node/public-relay":                   # iroh's public relays: on the owner's word only
            g.section()["public_relay"] = bool(req.get("on"))
            _save(server)
            g.close()
            server.gravity = gravity.GravityNode(server).start()       # a new endpoint with the new relay mode
            return server.gravity.view()
        if path.startswith("/api/yard/node/models/") and path.endswith("/risky"):
            t = server.cfg.tier(path[len("/api/yard/node/models/"):-len("/risky")])
            if t is None:
                raise YardError(404, "no such model")
            t.risky = bool(req.get("risky", True))
            _save(server)
            return {"name": t.name, "risky": t.risky}
        if path == "/api/yard/node/gateway":
            return gateway_deploy(server, g, req)
        if path == "/api/yard/node/gateway/remove":
            from nebula import gateway
            gw = g.section().pop("gateway", None) or {}
            if gw.get("managed"):
                gateway.remove()
            g.section()["relays"] = [r for r in g.section()["relays"] if r != gw.get("url")]
            _save(server)
            return {"removed": bool(gw)}
        if path == "/api/yard/node/invite":
            return g.invite(req.get("kind") or "friend", float(req.get("ttl_s") or 3600), req.get("for"),
                            req.get("grants") or None, req.get("confirm") or None)
        if path.startswith("/api/yard/node/invites/") and path.endswith("/cancel"):
            return {"cancelled": g.cancel_invite(path[len("/api/yard/node/invites/"):-len("/cancel")])}
        if path == "/api/yard/node/join":
            return {"peer": {k: v for k, v in g.join(str(req.get("invite") or "")).items() if k != "addr"}}
        if path.startswith("/api/yard/node/peers/"):
            pid, _, action = path[len("/api/yard/node/peers/"):].partition("/")
            if action.startswith("admin/"):
                return node_admin(server, path, "POST", req)
            if action == "grants":
                try:
                    return g.set_grants(pid, [str(m) for m in req.get("models") or []], [str(m) for m in req.get("confirm") or []])
                except gravity.GravityError as e:
                    if getattr(e, "needs_confirm", None):
                        return {"needs_confirm": e.needs_confirm, "warning": str(e), "granted": False}
                    raise
            if action == "revoke":
                return {"revoked": g.revoke(pid)}
            if action == "adopt":
                return g.adopt(pid)
        raise YardError(404, f"no such path: {path}")
    except gravity.GravityError as e:
        raise YardError(e.status, str(e)) from None


def gateway_deploy(server, g, req: dict) -> dict:
    """Deploy the gateway on this machine (Docker), or, with "elsewhere", only say how (the commands for a server
    with a public address); then this node uses it, and its invites carry it."""
    from nebula import gateway, gravity
    hostname = (str(req.get("hostname") or "").strip() or None)
    email = str(req.get("email") or "").strip() or None
    port = int(req.get("port") or 3340)
    conf = os.path.join(server.cfg.base_dir or os.getcwd(), "nebula-gateway")
    members = [g.node_id()] + [p["id"] for p in g.peers()]
    if req.get("elsewhere"):
        os.makedirs(conf, exist_ok=True)
        with open(os.path.join(conf, "relay.toml"), "w", encoding="utf-8") as f:
            f.write(gateway.config(members, hostname, email, port))
        return {"commands": gateway.commands("/etc/nebula-gateway", hostname, port), "config": gateway.config(members, hostname, email, port),
                "then": "copy relay.toml to /etc/nebula-gateway on that host, run the two commands, then give its URL here"}
    try:
        out = gateway.deploy(conf, members, hostname, email, port)
    except gateway.GatewayError as e:
        raise YardError(503, str(e)) from None
    url = req.get("public_url") or out["url"]
    g.section()["gateway"] = {"managed": True, "dir": conf, "hostname": hostname, "email": email, "port": port, "url": url}
    g.section()["relays"] = [url]
    _save(server)
    g.close()
    server.gravity = gravity.GravityNode(server).start()               # an endpoint that uses it
    return {"url": url, **server.gravity.view()}


def network_post(server, path: str, req: dict) -> dict:
    from nebula import network, pairnet
    n = getattr(server, "network", None)
    try:
        if path == "/api/yard/network/on":
            sec = server.cfg.raw.setdefault("network", {})
            sec["enabled"] = True
            if req.get("role") in ("main", "server"):
                sec["role"] = req["role"]
            if n is None:
                server.network = n = network.Network(server).start()
                if getattr(server, "yard", None) is not None:
                    server.yard.network = n
            _save(server)
            return n.view()
        if path == "/api/yard/network/off":
            server.cfg.raw.setdefault("network", {})["enabled"] = False
            if n is not None:
                n.close()
                server.network = None
                if getattr(server, "yard", None) is not None:
                    server.yard.network = None
            _save(server)
            return {"on": False}
        if n is None:
            raise YardError(409, "the network is off: turn it on first")
        if path == "/api/yard/network/invite":
            return n.invite(str(req.get("address") or ""))
        if path == "/api/yard/network/state":
            return n.invite_state(str(req.get("inviteId") or ""))
        if path == "/api/yard/network/adopt":
            return n.adopt()
        if path == "/api/yard/network/accept":
            return n.accept(str(req.get("pin") or ""), float(req.get("timeout_s") or 120))
        if path == "/api/yard/network/engine":
            return n.start_engine()
        if path == "/api/yard/network/remove":           # a member of your network out (a machine gone, or a stale entry)
            n._need()
            return n.broker.call("nodes:remove", {"nodeUuid": str(req.get("nodeUuid") or req.get("nodeId") or "")}, timeout=60)
        raise YardError(404, f"no such path: {path}")
    except network.NetworkError as e:
        raise YardError(e.status, str(e)) from None
    except pairnet.PairError as e:
        raise YardError(502, str(e)) from None


def _known_hosts(server) -> str:
    from nebula import names
    return os.path.join(names.local_path(server.cfg.base_dir, "nebula-keys", "cockpit-keys"), "known_hosts")


def servers_post(server, path: str, req: dict) -> dict:
    """Add a server (SSH): nebula/sshinstall.py does the SSH part; the joining is the network's (PAIR's mutual TLS)."""
    from nebula import gravity, network, pairnet, sshinstall
    kh = _known_hosts(server)
    try:
        if path == "/api/yard/servers/scan":
            return {"keys": [{k: v for k, v in x.items() if k != "line"} for x in sshinstall.scan(str(req["host"]), int(req.get("port") or 22))]}
        if path == "/api/yard/servers/trust":
            return sshinstall.trust(kh, str(req["host"]), int(req.get("port") or 22), str(req["fingerprint"]))
        t = None
        if path in ("/api/yard/servers/preflight", "/api/yard/servers/install", "/api/yard/servers/admin"):
            t = sshinstall.Target(str(req.get("host") or ""), str(req.get("user") or ""), int(req.get("port") or 22),
                                  req.get("key_file") or None, kh)
        if path == "/api/yard/servers/preflight":
            if req.get("fingerprint"):                  # the key just confirmed in the wizard, not kept yet
                import tempfile
                with tempfile.TemporaryDirectory() as d:
                    t.known_hosts = sshinstall.confirmed_known_hosts(t.host, t.port, str(req["fingerprint"]), d)
                    return sshinstall.preflight(t)
            return sshinstall.preflight(t)
        if path == "/api/yard/servers/admin":
            return sshinstall.run_admin(t, [str(x) for x in req.get("tasks") or []], str(req.get("password") or ""),
                                        str(req.get("local_password") or ""))
        if path == "/api/yard/servers/install":
            name = re.sub(r"[^A-Za-z0-9 ._-]", "", str(req.get("name") or req["host"]))[:40] or t.host
            if getattr(server, "network", None) is None:              # this Nebula: the main, its PAIR on
                network_post(server, "/api/yard/network/on", {"role": "main"})
            if server.network.broker is None:
                raise YardError(503, server.network.why or "this machine's network is off")
            pair_bin = server.network.broker.bin_dir
            from nebula import openshell
            osh = openshell.binary()
            out = sshinstall.install(t, pair_bin, os.path.dirname(os.path.realpath(osh)) if osh else None)
            steps = [f"the node pieces on {t.host}: the network's node programs, the node keeper, OpenShell on Podman (your services there)"]
            address = _address(t)
            joined = server.network.join(t, address)
            steps.append("joined your network (mutual TLS)" + (" (it already was)" if joined.get("already") else ""))
            gw, gw_port = _register_gateway(server, t, name)
            steps.append(f"its OpenShell gateway registered here as {gw} (through an SSH tunnel on 127.0.0.1:{gw_port})")
            row = {"name": name, "host": t.host, "port": t.port, "user": t.user, "address": address, "gateway": gw, "gateway_port": gw_port,
                   "node": (joined.get("node") or {}).get("nodeId"), "added_at": time.time()}
            server.cfg.raw["servers"] = [s for s in server.cfg.raw.get("servers") or [] if s.get("name") != name] + [row]
            _save(server)
            return {"server": row, "steps": steps, "log": out["log"]}
        if path.startswith("/api/yard/servers/") and path.endswith("/uninstall"):
            name = path[len("/api/yard/servers/"):-len("/uninstall")]
            row = next((s for s in server.cfg.raw.get("servers") or [] if s.get("name") == name), None)
            if row is None:
                raise YardError(404, f"no server {name}")
            t = sshinstall.Target(row["host"], row["user"], row["port"], req.get("key_file") or None, kh)
            out = sshinstall.uninstall(t)
            if row.get("node") and getattr(server, "network", None) is not None and server.network.broker is not None:
                try:                                       # and out of your network's members
                    server.network.broker.call("nodes:remove", {"nodeId": row["node"]}, timeout=60)
                except pairnet.PairError:
                    pass
            if row.get("gateway"):
                from nebula import machines
                machines.close_tunnel(name)
                _unregister_gateway(row["gateway"])
            server.cfg.raw["servers"] = [s for s in server.cfg.raw.get("servers") or [] if s.get("name") != name]
            _save(server)
            return {"removed": name, "log": out["log"]}
        raise YardError(404, f"no such path: {path}")
    except sshinstall.SshError as e:
        raise YardError(502, str(e)) from None
    except (gravity.GravityError, network.NetworkError) as e:
        raise YardError(e.status, str(e)) from None
    except TimeoutError:
        raise YardError(504, f"{req.get('host')} did not answer in time: is a firewall between the two machines? "
                             "(the administrator step can let this computer through)") from None
    except KeyError as e:
        raise YardError(400, f"{e.args[0]} is needed") from None


def _address(t) -> str:
    """The model machine's address on your network (PAIR's invite goes there): what SSH reached, as an IP."""
    import socket
    from nebula import sshinstall
    host = sshinstall.ssh_config(t.host, t.port).get("hostname") or t.host
    try:
        return socket.gethostbyname(host)
    except OSError:
        return host


def _gateway_name(name: str) -> str:
    return "nebula-" + (re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")[:24] or "server")


def _register_gateway(server, t, name: str) -> tuple[str, int]:
    """The model machine's OpenShell gateway, for this machine's `openshell`: its client certificate kept beside this
    machine's own gateways, reached through Nebula's own SSH tunnel to it (a local port of its own: the gateway there
    listens on its loopback only); the gateway this machine uses by default stays as it was."""
    import subprocess
    from nebula import machines, openshell, prereqs, sshinstall
    gw = _gateway_name(name)
    exe = openshell.binary()
    if not exe:
        raise YardError(503, "OpenShell is not installed on this computer (its `openshell` manages the gateway there)")
    taken = [int(s.get("gateway_port") or 0) for s in server.cfg.raw.get("servers") or [] if s.get("name") != name]
    port = machines.free_port(taken)
    conf = str(prereqs._xdg("XDG_CONFIG_HOME", ".config") / "openshell")
    active = os.path.join(conf, "active_gateway")
    before = open(active, encoding="utf-8").read() if os.path.isfile(active) else None
    mtls = os.path.join(conf, "gateways", gw, "mtls")
    subprocess.run([exe, "gateway", "remove", gw], capture_output=True, timeout=60, stdin=subprocess.DEVNULL)   # added again
    for fname, text in sshinstall.gateway_certs(t).items():
        os.makedirs(mtls, mode=0o700, exist_ok=True)
        fd = os.open(os.path.join(mtls, fname), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
    dest = f"{t.user}@{t.host}" + ("" if t.port == 22 else f":{t.port}")
    p = subprocess.run([exe, "gateway", "add", "--remote", dest, "--name", gw, f"https://127.0.0.1:{port}"],
                       capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    if p.returncode != 0 and "already" not in (p.stdout + p.stderr).lower():
        raise YardError(502, f"its OpenShell gateway could not be registered here: {(p.stderr or p.stdout).strip()[-300:]}")
    if before is not None:
        with open(active, "w", encoding="utf-8") as f:
            f.write(before)
    elif os.path.isfile(active):
        os.remove(active)
    return gw, port


def _unregister_gateway(gw: str) -> None:
    import subprocess
    from nebula import openshell
    exe = openshell.binary()
    if exe:
        subprocess.run([exe, "gateway", "remove", gw], capture_output=True, timeout=60, stdin=subprocess.DEVNULL)


def node_admin(server, path: str, method: str, body):
    """An owner machine's /api/yard, over the Gravity admin ALPN (through the loopback bridge)."""
    import urllib.request
    import urllib.error
    g = getattr(server, "gravity", None)
    if g is None or g.bridge is None:
        raise YardError(409, "the Gravity node is off")
    pid, _, rest = path[len("/api/yard/node/peers/"):].partition("/admin/")
    p = g.peer(pid)
    if p is None or p.get("role") != "admin":
        raise YardError(403, "only a machine you installed (joined by an owner invite) is administered from here")
    url = f"{g.bridge.url}/admin/{pid}/api/yard/{rest}"
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, method=method,
                                                           headers={"Content-Type": "application/json"}), timeout=900) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error")
        except ValueError:
            msg = None
        raise YardError(e.code, (msg or {}).get("message") if isinstance(msg, dict) else str(msg or e)) from None


def gravity_adopt(server, req: dict) -> dict:
    """A Gravity router on this machine as models: one tier per model it offers (a request names one exactly)."""
    import re
    from nebula import pairlocal
    from nebula.config import Tier
    from nebula.server import reload_models
    addr = str(req.get("address") or "").rstrip("/")
    if not re.match(r"^http://(127\.0\.0\.1|localhost|\[::1\]):\d+$", addr):
        raise YardError(400, "a Gravity router on this machine: http://127.0.0.1:<port>")
    if not pairlocal.is_router(addr):
        raise YardError(400, f"no Gravity router answers at {addr}")
    offered = pairlocal.models(addr)
    want = [m for m in (req.get("models") or offered) if m in offered]
    if not want:
        raise YardError(400, "it offers no models yet: add a model on one of its machines first")
    cfg, made = server.cfg, []
    for m in want:
        name = "gravity-" + re.sub(r"[^a-z0-9._-]+", "-", m.lower()).strip("-")[:48]
        if cfg.tier(name) is not None:
            continue
        cfg.tiers.append(Tier(name=name, engine=addr, model=m, mesh="gravity", health_path="/v1/models",
                              slots=int(req.get("slots") or 8)))   # the router spreads them over its machines
        made.append(name)
    reload_models(server)
    return {"made": made, "offered": offered}


COMMAND_FIELDS = ("binary", "model", "port", "context", "slots", "options", "flags", "cache_type", "host")


def raw_post(server, req: dict) -> dict:
    """The Raw tab: a command line → the engine's settings (merged onto its deployment), what changes, and the line it
    would really run (Nebula's hardening put back)."""
    from nebula import packages as P
    eng = req.get("engine")
    base = dict(eng) if isinstance(eng, dict) else server.yard.spec(str(eng))
    kind = base.get("kind") or "impulse"
    if kind == "command":
        raise YardError(400, "a command engine is its command: edit it in the form")
    r = P.parse_command(kind, str(req.get("text") or ""))
    new = {k: v for k, v in base.items() if k not in COMMAND_FIELDS}
    new.update(r["spec"])
    new.setdefault("port", base.get("port"))
    # the line spells out what the deployment leaves to defaults: the same thing, not a change
    if not base.get("binary") and new.get("binary") == P.PACKAGES.get(kind, {}).get("binary"):
        new.pop("binary", None)
    if base.get("model") and new.get("model") and not os.path.isabs(base["model"]) and \
            os.path.join(server.cfg.base_dir or "", base["model"]) == new["model"]:
        new["model"] = base["model"]
    if not base.get("slots") and new.get("slots") == 1:
        new.pop("slots", None)
    out = {"spec": new, "locked": r["locked"], "warnings": r["warnings"], "invalid": list(r["invalid"])}
    try:
        s = E.check_spec(new)
        out["spec"] = s
        out["command"] = E.command_of(s, server.cfg.base_dir, E.key_path(s, server.cfg))
        old = E.check_spec(base) if base.get("name") else {}
        out["diff"] = P.diff_specs({k: v for k, v in old.items() if k in COMMAND_FIELDS},
                                   {k: v for k, v in s.items() if k in COMMAND_FIELDS})
    except E.EngineSpecError as e:
        out["invalid"].append(str(e))
    return out


def route_for(server, name: str, req: dict) -> dict:
    """A tier on a managed engine's address: the model a request names, routed like any other."""
    from nebula.config import Tier
    from nebula.server import reload_models
    cfg, yard = server.cfg, server.yard
    s = yard.spec(name)
    have = [t for t in cfg.tiers if t.managed == name]
    if have:
        return {"tier": have[0].name, "created": False}
    tier_name = str(req.get("tier") or name).strip()
    if not tier_name or any(c in tier_name for c in " /\\"):
        raise YardError(400, "a model needs a short name without spaces or slashes")
    if cfg.tier(tier_name) is not None or cfg.agent(tier_name) is not None:
        raise YardError(409, f"the name {tier_name!r} is taken")
    cfg.tiers.append(Tier(name=tier_name, engine=yard.address(name), managed=name,
                          thinking=req.get("thinking") if isinstance(req.get("thinking"), bool) else None,
                          context=int(s["context"]) if s.get("context") else None, slots=int(s.get("slots") or 1)))
    sync_tiers(server, name)
    return {"tier": tier_name, "created": True}


def sync_tiers(server, name: str) -> None:
    """A managed engine's models follow it: its address, its slots, the key only Nebula holds, and whether guests
    share it (and how tenants are kept apart on it: nebula/tenancy.py)."""
    from nebula import packages
    from nebula.server import reload_models
    cfg, yard = server.cfg, server.yard
    try:
        s = yard.spec(name)
    except KeyError:
        return
    for t in cfg.tiers:
        if getattr(t, "managed", None) != name:
            continue
        t.engine, t.slots = yard.address(name), int(s.get("slots") or 1)
        if s.get("machine"):                       # on a model machine: through your network, by its name
            t.model, t.mesh, t.health_path = name, "network", "/v1/models"
        kp = None if s.get("machine") else E.key_path(s, cfg)
        if kp:
            E.ensure_key(kp)
            t.api_key_file = kp
        t.shared = bool(s.get("guests"))
        t.cache_isolation = packages.PACKAGES[s["kind"]].get("isolation") if s["kind"] in packages.PACKAGES else "no-cache"
    reload_models(server)
