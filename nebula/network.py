"""nebula/network.py: model machines on your network (docs/network.md).

One Nebula, many machines. This machine runs the Personal AI Router's backend (internal: nebula/pairnet.py) and
drives it; a model machine runs only that router's node programs and the node keeper (nebula/nodekeeper.py), set up
over SSH by Add a server (nebula/sshinstall.py), and never a Nebula. Users see "model machines on your network".

- **This Nebula** founds the cluster and joins machines to it: it invites the machine's address, and that machine's
  keeper (reached over SSH, the same JSON-RPC as the broker's) answers with the PIN. Models served there are routed
  through this machine's router to it over mTLS (`net-<model>`, or an engine deployed there from Engines).
- **Who a request is for** is decided here, before it leaves this machine: a model machine runs engines only.
- **No plain HTTP between machines:** a model at another machine's http:// address is refused; it is on a model
  machine (mTLS) or a Gravity peer (iroh), never an open port.
- The older shape (`role: "server"`, a headless Nebula answering as its router's engine, nebula/pairface.py) is kept
  for machines set up that way; Add a server no longer makes it.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import urllib.request

from nebula import pairnet


class NetworkError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Network:
    def __init__(self, server):
        self.server, self.cfg = server, server.cfg
        self.broker: pairnet.Broker | None = None
        self.face = None
        self.why: str | None = None

    def section(self) -> dict:
        n = self.cfg.raw.setdefault("network", {})
        n.setdefault("role", "main")
        if not n.get("secret"):
            n["secret"] = secrets.token_urlsafe(32)
        return n

    def save(self):
        if self.cfg.path:
            self.cfg.save()

    def start(self) -> "Network":
        sec = self.section()
        bin_dir = pairnet.find_bin(sec.get("pair_bin"))
        if bin_dir is None:
            self.why = "the router's backend is not installed here (NEBULA_PAIR_BIN or network.pair_bin)"
            return self
        data = sec.get("data_dir") or os.path.join(self.cfg.base_dir or os.getcwd(), "nebula-network")
        self.broker = pairnet.Broker(bin_dir, data).start()
        if sec["role"] == "server":                      # this machine's models, for the household, through Nebula
            from nebula.pairface import Face
            self.face = Face(self.server, int(sec.get("face_port") or 8081), secret=sec["secret"])
        return self

    def close(self):
        if self.face is not None:
            self.face.close()
        if self.broker is not None:
            self.broker.stop()

    def _need(self):
        if self.broker is None:
            raise NetworkError(503, self.why or "the network is off")

    # --- the main Nebula
    def found(self) -> dict:
        self._need()
        me = self.broker.call("cluster:get-node-id")
        if not me.get("clusterId"):
            me = {**me, **self.broker.call("cluster:create")}
        return me

    def invite(self, address: str) -> dict:
        """Invite a server at `address`: the PIN is what the server answers with (over SSH or the admin link)."""
        self._need()
        if not re.match(r"^[A-Za-z0-9.:\-\[\]]+$", address or ""):
            raise NetworkError(400, "an address (an IP or a host name on your network)")
        self.found()
        inv = self.broker.call("cluster:invite-node", {"address": address}, timeout=60)
        return {"inviteId": inv.get("inviteId"), "pin": inv.get("pin"), "state": inv.get("state")}

    def node(self, t) -> pairnet.Broker:
        """A model machine's PAIR, through its node keeper over SSH (nebula/nodekeeper.py): the same JSON-RPC."""
        from nebula import sshinstall
        return pairnet.Broker("", "", argv=sshinstall.keeper_argv(t)).start()

    def join(self, t, address: str, timeout: float = 120) -> dict:
        """A model machine into this Nebula's cluster: this machine invites it, its keeper answers with the PIN."""
        import time
        self._need()
        mine = self.found().get("clusterId")
        node = self.node(t)
        try:
            there = node.call("cluster:get-node-id", timeout=60)
            if mine and there.get("clusterId") == mine:
                return {"node": there, "already": True}
            inv = self.invite(address)
            n = node.wait_note("cluster:invite-received", timeout=90)
            node.call("cluster:respond-to-invite", {"inviteId": n["inviteId"], "accept": True, "pin": str(inv["pin"])}, timeout=60)
            state, end = {}, time.time() + timeout
            while time.time() < end:
                state = self.invite_state(inv["inviteId"])
                if state.get("state") in ("paired", "failed", "declined", "expired"):
                    break
                time.sleep(2)
            if state.get("state") != "paired":
                raise NetworkError(502, f"the pairing did not complete ({state.get('state') or 'no answer'})")
            return {"node": node.call("cluster:get-node-id", timeout=60)}
        except pairnet.PairError as e:
            raise NetworkError(502, f"{t.host}: {e}") from None
        finally:
            node.stop()

    def node_call(self, t, method: str, params: dict | None = None, timeout: float = 60) -> dict:
        """One call to a model machine's keeper (keeper:set-engines, keeper:status) or its PAIR."""
        node = self.node(t)
        try:
            return node.call(method, params, timeout=timeout)
        except pairnet.PairError as e:
            raise NetworkError(502, f"{t.host}: {e}") from None
        finally:
            node.stop()

    def invite_state(self, invite_id: str) -> dict:
        self._need()
        return self.broker.call("cluster:invite-status", {"inviteId": invite_id})

    def members(self) -> list[dict]:
        if self.broker is None:
            return []
        try:
            return self.broker.call("nodes:get-initial").get("nodes") or []
        except pairnet.PairError:
            return []

    def offered(self) -> list[str]:
        """The models the household's servers offer, as this machine's router sees them (its llama.cpp facade)."""
        port = int(self.section().get("router_port") or 8080)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
                return [m["id"] for m in json.loads(r.read()).get("data") or [] if m.get("id")]
        except (OSError, ValueError):
            return []

    def adopt(self) -> dict:
        """The household's models as models here: `net-<model>`, through this machine's router."""
        from nebula.config import Tier
        from nebula.server import reload_models
        port = int(self.section().get("router_port") or 8080)
        made = []
        for m in self.offered():
            name = "net-" + re.sub(r"[^a-z0-9._-]+", "-", m.lower()).strip("-")[:44]
            if self.cfg.tier(name) is None:
                self.cfg.tiers.append(Tier(name=name, engine=f"http://127.0.0.1:{port}", model=m, mesh="network",
                                           health_path="/v1/models", slots=4))
                made.append(name)
        reload_models(self.server)
        self.save()
        return {"made": made}

    # --- a server
    def accept(self, pin: str, timeout: float = 120) -> dict:
        """The invite from the household's main Nebula, answered with its PIN."""
        self._need()
        n = self.broker.wait_note("cluster:invite-received", timeout=timeout)
        return self.broker.call("cluster:respond-to-invite", {"inviteId": n["inviteId"], "accept": True, "pin": str(pin)}, timeout=60)

    def start_engine(self) -> dict:
        """This Nebula, adopted by the router as its engine (so requests over mTLS land here)."""
        self._need()
        return self.broker.call("engine:start", {"engine": "llamacpp"}, timeout=60)

    def view(self) -> dict:
        sec = self.section()
        return {"on": self.broker is not None, "why": self.why, "role": sec.get("role"), "members": self.members(),
                "models": [t.name for t in self.cfg.tiers if getattr(t, "mesh", None) == "network"]}
