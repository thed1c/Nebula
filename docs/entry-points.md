# API entry points

The API Server is the door apps use to reach your models. Nebula's own port is the first door ("This PC"). You can
add more from the page (the **Add an API entry point** wizard, or the API Server's panel), one for each kind of client:
*House LAN* for the TV and the tablets, *Kids* with two small models, *Home Assistant* in Ollama's style. On the map,
each one is a Quasar on the left edge.

Each entry point has:
- **A scope.**
  - *This machine*: it listens on loopback only.
  - *Your home network*: it answers private addresses only (`10/8`, `172.16/12`, `192.168/16`, link-local, IPv6 ULA)
    and always needs a key. It refuses public addresses and carrier-grade NAT or VPN ranges such as `100.64/10`.
  - Never outside: the only door from outside is Gravity.
- **Its own keys.** Every entry point is a caller of its own (`entry-<id>`), so a request with one of its keys is that
  caller's tenant, as any caller's is. A key for another entry point, a caller or the main API does not open it: such
  a call bounces off (401). The bounce goes into the security log and shows on its Quasar.
- **Its model list.** Only these models are offered here (`/v1/models` and `/api/tags` list only them). A request that
  names another model is refused (404), not quietly answered by a different one. An empty list offers all models.
- **Its API styles.**
  - **OpenAI:** `/v1/chat/completions`, `/v1/responses` and `/v1/models`.
  - **Anthropic:** `/v1/messages`.
  - **Ollama:** `/api/chat`, `/api/generate`, `/api/tags` and `/api/version`, streaming newline-delimited JSON as Ollama
    does. Ollama's paths exist only on an entry point, since on Nebula's own port `/api/` is the page.
- **Guest, optionally.** A guest entry point gets only models shared with guests, each with its own cache.

All of them feed the same Switchyard, with the same routing, gates, ledger, Halo and guard.

On your network an entry point speaks plain HTTP, as Ollama does, so give each device its own key.

API: `GET /api/api-servers` (every entry point, its counters for the map); `POST /api/api-servers` {name, scope, port,
formats, models, guest, require_key}; `POST /api/api-servers/<id>` (change it), `/<id>/keys` {label} (a key, shown
once), `/<id>/keys/<key>/revoke`, `/<id>/remove`.
