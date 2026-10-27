# Halo

Halo is Halo's guardian. It is on by default; Halo's panel turns it off and on.

## What it does

- It **watches the security log** ([security-log.md](security-log.md)). When a guest does something worth a look
  (repeated refusals, someone joining over Gravity, a guest probing models it was not given), it wakes, waits a few
  seconds for a burst to land, and looks once.
- It **thinks with Halo's model** (or its own: the model picker on its panel), through the Switchyard like
  any turn, in the interactive class at the gates, so it is not stuck behind background work. Its turns are in the
  Ledger as `halo`.
- It **reads the log** with Lookup's read-only `security` tool, then **acts only through small tools that undo**:

| Tool | What it does | Undo |
|---|---|---|
| `rate_limit` | at most N requests a minute from one guest, for a while | lifts it |
| `pause_guest` | refuses one guest's requests for up to a day | lifts it |
| `untick_share` | holds one model back from one guest | gives it back |
| `revoke_key` | revokes an API Server or caller key that is clearly abused | restores the same key |
| `ask_owner` | anything bigger, or anything it is unsure of | — |

- It **never touches the owner's own traffic** or the owner's own machines, never shares or opens anything, and never
  loosens a rule. The Switchyard's own rules stay the real guard.
- **Everything it does is in the security log** (`halo.action`, `halo.ask`, `halo.undo`), and on its panel with the
  reason and an **Undo** button.

## The ring

The ring round Halo on the map, and on its panel:
- **glow**: calm
- **pulse**: looking at something
- **red**: it acted, or it asks you. It stays red until you press *Seen*.

## Where its holds live

The holds it places are kept in the config's `"guard"` section (paused, limits, denied), so a restart keeps them and
you can read them there. They expire by themselves. Its records (actions, asks, what it has read) are kept beside
the ledger (`<ledger>.halo.json`).

## Settings

`"assistant": {"halo": {"on": true, "route": {"tier": ...}, "settle_s": 3, "every_s": 300}}`. `route` is its own
model (empty: Halo's), `settle_s` how long it lets a burst land, and `every_s` how often it looks anyway.
API: `GET /api/halo`; `POST /api/halo {on, route}`, `/api/halo/actions/<id>/undo`, `/api/halo/seen`, `/api/halo/check`.

## Halo's own model

Halo can run its own small model instead of the Assistant's (`nebula/halo_model.py`, `nebula/catalog/halo.json`): Liquid AI's
**LFM2.5-1.2B-Instruct**, the makers' QAD Q4_0 file (0.7 GB), fetched at first run from Liquid's repository at a pinned revision
and checked against its sha256, plus two LoRA adapters that ship with Nebula in `nebula/assets/halo/` (F16, LFM Open License
v1.0; what they change is in that folder's README):

- **Standard**: the base + the Halo adapter (the guard, the router judge, handoff briefings).
- **Uncensored**: the same + the Uncensored adapter. Both are always loaded; the setting is the second adapter's scale (0 or 1),
  set on the running server (`/lora-adapters`), so switching needs no reload.

**Where it runs** (the owner's order): a spare GPU (a card with no engine), else spare VRAM on an engine's card (only what stays
free after the engines have sized their caches, plus a margin), else the CPU, always warm with capped threads and `nice`, using
no CPU when idle. On the CPU a second one-slot server is Halo's router lane, so a long handoff read never holds a routing call.
