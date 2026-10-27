# The security log

Nebula keeps an append-only record of who tried to get in and what changed who may do what (`nebula/seclog.py`).
Halo reads it through Lookup's read-only `security` tool, and the owner reads it on Halo's panel or at
`GET /api/security`.

## What is recorded

| Kind | When |
|---|---|
| `auth.refused` | a request to the API without the key it needs, or with a wrong one |
| `auth.unpaired` | a request from another machine with no paired token |
| `auth.bad_host` | a request under a name that is not this Nebula's (DNS rebinding); add a name you use on purpose to `"allowed_hosts"` |
| `auth.cross_site` | another web site's page tried to use this Nebula (writes, the page's API, the model API) |
| `key.used` | a key's first use in the hour, and again from a new address (by its id, never the key) |
| `key.made`, `key.revoked` | an API Server or caller key made or revoked on the page |
| `caller.changed`, `caller.removed` | a caller added, changed or removed |
| `sessions.deleted` | conversations deleted for good on the page (how many; never which, nor what they said) |
| `guest.refused` | a guest asked for a model not shared with it (an app's key, or a Gravity peer) |
| `gravity.invite`, `gravity.join`, `gravity.joined`, `gravity.refused`, `gravity.grants`, `gravity.revoked` | Gravity invites made, someone joining by one, this Nebula joining another, an invite refused, models ticked for a peer, a peer removed |
| `share.changed` | a model marked shared, an entry point or a collection's sharing changed |
| `server.*`, `network.*` | a model server added, trusted, removed; the household network's invites |
| `halo.action`, `halo.undo`, `halo.ask` | everything Halo does, its undo, and what it asks the owner |

Each row has a time, a kind, a severity (`info`, `notice`, `warn`, `alert`), who (an address, `owner`,
`caller:<id>` or `peer:<id>`), what it was about, and a few details. It never holds a key, token, invite secret,
PIN, password or prompt.

## How it is kept

- **Its own file** beside the ledger: `<ledger>.security.sqlite3` (in memory when Nebula runs without a ledger).
- **Append-only.** Triggers refuse every `UPDATE` and `DELETE`.
- **A hash chain.** Each row carries the sha256 of the row before it. `GET /api/security/verify` walks the chain and
  names the first row that was changed or removed by hand.
- **No flood.** The same event (kind, who, what) within a minute is counted on the row it repeats, so a script
  hammering a bad key makes one row with a count. Key use is written once per key per hour.

## Reading it

`GET /api/security?since=SEQ&kinds=gravity.*,auth.refused&who=caller:x&min=warn&since_s=3600&limit=100` returns
the newest first, each with its count and last time, plus the counts per kind in the last hour.
