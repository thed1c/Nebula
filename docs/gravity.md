# Gravity: Nebula to Nebula

Every Nebula has two doors: the **API Node** (HTTP, for apps and agents, a key per caller) and the **Gravity node**
(`nebula/gravity.py`, for other Nebulas: a friend's, your laptop, a server you installed), over iroh QUIC
(`pip install "nebula-eh[gravity]"`). The Gravity tab turns it on; it is off until you do.

- **A node is its key.** `nebula-keys/gravity.key` (0600); its public half is the node's id.
- **Nothing is found on the network:** no mDNS, no public DNS, no DHT. A Nebula joins only with an **invite** you
  make (Gravity › Invite a friend / Invite one of my machines): a one-time ticket with this node's addresses, your
  gateway's address if you have one, and a secret. It works once and expires.
- **On your network the two connect directly; elsewhere through your gateway**, your own iroh relay (Gravity ›
  Gateway). Public relays are never used. A relay only forwards encrypted traffic it cannot read.
- **A peer gets nothing until you tick a model for it**, and only models shared with guests (docs/sharing.md): at your
  Nebula it is its own tenant, with its own cache and its own history.
- **Anti-Gravity:** a cloud model (your API key, your bill, the provider's terms) or a CLI on your own account can
  never be ticked for a peer; its tick is greyed out with the reason, and the API refuses it too.
- **Local models that need care** (you marked it, or Nebula does not run its engine and so cannot lock it down, or it
  is a router of other machines) can be shared, but ticking one asks you to confirm a warning, and the peer sees the
  model as **not guaranteed safe**. Models on engines Nebula runs share by the tick alone.
- **"Use its models"** adds the models a peer granted you as models here, routed, queued and recorded like any other.
- **Your own machines** (joined with "Invite one of my machines") can be managed from the machine that joined: its
  engines, deploy your own, its settings (`/api/yard` over the owner-only admin ALPN). A friend can never do that.
- **Remove** forgets a peer's key and closes its connections; every request is checked anyway.

The ALPNs: `gravity/enroll/1` (joining), `gravity/models/1` (the models granted to the asking key),
`gravity/load/1` (pending work), `gravity/infer/1` (a request and its stream), `gravity/admin/1` (owner machines).
An ALPN with no handler is refused at the socket.

## The gateway (`nebula/gateway.py`)

Gravity › Gateway › **Deploy here** runs iroh's own relay server in a container on this machine (Docker): plain HTTP
on 3340 without a hostname, HTTPS with a Let's Encrypt certificate with one. **Deploy elsewhere…** writes its config and
gives the two commands for a server with a public address. Nebula writes who may use it: an allowlist of your members'
keys, rewritten when a peer joins or is removed. Your node then uses it, and the invites you make carry it.
`greenfield/gatewaynet.sh` proves it: two Nebulas on separate networks with no route out meet only through it, and a
Nebula not on its allowlist cannot join.
