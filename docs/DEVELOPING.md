# Developing Nebula

The user-facing guide is the [README](../README.md). This page is for building and running Nebula from source,
the config file and the API.

Your own AI environment above the engines on your machine: one server in front of local engines (Impulse,
llama.cpp, or any OpenAI-compatible server), cloud providers and CLIs, with a router, agents, a ledger of every
decision, and a live star-map page. In the code it is still called "the cockpit", its working name.

**Tested on Linux only** (Python 3.10+). Windows and macOS have not been run yet.

Names: clients use `nebula/<name>` for a model, `X-Nebula-*` headers, and Nebula keeps `nebula.sqlite3` and
`nebula-keys/`. The older `cockpit/<name>`, `X-Cockpit-*`, `cockpit.sqlite3` and `cockpit-keys/` from before the
rename are still accepted.
[docs/DESIGN.md](DESIGN.md) is the design.

The gateway: OpenAI Chat Completions, Anthropic Messages and OpenAI Responses in, the engine's `/v1/chat/completions`
out. Standard library plus Strata's translators, vendored in `nebula/_vendor/strata_serve/` (jinja2 comes with them).

Install and run, from the repository root, against a running engine:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[monitor]"   # [monitor] adds psutil for the This machine panel
pip install "nebula-eh[monitor] @ git+https://github.com/thed1c/Nebula"   # or straight from GitHub (the package is nebula-eh)
python3 -m nebula --engine http://127.0.0.1:8080 --port 8090
```

Then point apps at `http://127.0.0.1:8090/v1` (OpenAI), `http://127.0.0.1:8090` as `ANTHROPIC_BASE_URL` (Claude
Code), or `/v1/responses` (Codex CLI). `--alias NAME` adds a model name `/v1/models` lists and the answers echo;
`--config FILE` reads the tiers, host, port and aliases from JSON.

Every request is recorded in the ledger, `nebula.sqlite3` in the current directory (`--ledger PATH` moves it,
`--no-ledger` keeps none): the session, the new messages and the request boundary before the engine is called,
the reply, the costs and the outcome after. A client that wants its conversations kept apart sends an
`X-Nebula-Session` header; Claude Code's `metadata.user_id` is read the same way; otherwise a session is found by
the conversation's opening. Rebuild every transcript from the events and check it against what each request
recorded:

```bash
python3 -m nebula.replay --ledger nebula.sqlite3
```

The cockpit's own page is at `http://127.0.0.1:8090/`: sessions on the left, the transcript with each answer's
tier, model and thinking, a "why?" for the router's reason and the call's counts, an "escalate" on every answer, close
and reopen, and a monitor of the card and each tier's engine. Its API is under `/api` (`nebula/ui.py`). The home moves with
what runs: `/api/stream` sends a server-sent event whenever the ledger commits or a reply streams (`/api/live` is the
same as one read), and each turn travels from its input through the router to its tier's lane and back.

Tiers and routing (`DESIGN.md`, "The router"): a config file lists the ladder, and the router picks the
lowest tier the turn's signals allow, never dropping back on its own. A request names a tier with `model:
"nebula/big"` or asks for one step up with an `X-Nebula-Escalate: 1` header; repeated regenerations, consecutive
tool errors and a stuck reply raise it too. A crossing to another engine carries a packet built from the ledger.

```json
{"port": 8090, "ledger": "nebula.sqlite3",
 "tiers": [{"name": "fast",  "engine": "http://127.0.0.1:8080", "thinking": false},
           {"name": "think", "engine": "http://127.0.0.1:8080", "thinking": true},
           {"name": "big",   "engine": "http://second-box:8080"},
           {"name": "cloud", "provider": "openai-compatible", "base_url": "https://api.example.com/v1",
            "model": "some-model", "api_key_env": "CLOUD_API_KEY", "context": 200000}],
 "rules": {"regenerations_to_rise": 1, "regenerations_to_cross": 2, "tool_errors_to_rise": 2, "tool_errors_to_cross": 4}}
```

Agents (`DESIGN.md`, step 5): a named process with its own system prompt, tools from MCP servers, a
floor and a ceiling on the ladder, thinking and an output shape. Pick one with `model: "nebula/<agent>"`, the
`X-Nebula-Agent` header, or the page's picker; `"default": true` makes it the one a request without one gets. An
`"event": true` agent is the secretary: compaction runs as it. MCP servers use Strata's MCP config shape (`nebula/_vendor/strata_serve/mcp.py`).

```json
{"agents": [{"name": "assistant", "system": "Answer plainly.", "tier": "fast", "default": true},
            {"name": "researcher", "system_file": "prompts/researcher.md", "tools": ["web"], "tier": "think", "ceiling": "big"},
            {"name": "secretary", "system": "You keep the conversation's notes.", "tier": "fast", "event": true}],
 "mcp_servers": {"web": {"command": "npx", "args": ["-y", "@some/mcp-server"]}}}
```

Models and policy (Switchyard plumbing step A): the Models screen on the page shows the server's URL and the base
URL per dialect to paste into another app, a direct model name per tier and agent (`nebula/<name>`), the registry
of every model (up, slots, context, vision), an add-model form (a local engine by URL, or an OpenAI-compatible
provider or gateway by base URL, model id and key; probed first, the key kept in a 0600 file beside the config),
and the policy: the models each agent may use, saved as its `models` ticks. In the config:

```json
{"tiers": [{"name": "cloud", "provider": "openai-compatible", "base_url": "https://api.example.com/v1", "model": "some-model",
            "api_key_file": "nebula-keys/cloud.key", "context": 200000, "power_class": "cloud", "cost_per_mtoken": 2.5}],
 "default_models": ["fast", "think"],
 "agents": [{"name": "researcher", "system": "...", "models": ["think", "cloud"]}]}
```

Queues (Switchyard plumbing step B): every engine has a gate with as many slots as it reports; waiting turns are
served by class then arrival, interactive (the room, the API dialects) first, then agent (webhooks), then background
(schedules). Set a hook's, a schedule's or an agent's class with `"priority"`. A turn whose engine is busy overflows
to a later ticked tier on another engine once its expected wait passes the class's tolerance
(`"rules": {"tolerance_seconds": {"interactive": 60, "agent": 15, "background": 5}}`). `/api/admission` shows the gates.

The parking lot (Switchyard plumbing step C): a request no ticked model can hold (its size, counted with the
engine's tokenizer or estimated, plus the output it asks for, against each model's context) is parked, visible on
the System tab and at `/api/parking`, re-checked every few seconds and when the policy or the models change, and
refused after `"rules": {"parking_wait_seconds": 600}`. Nothing is truncated and nothing is sent to fail.

This machine (step G1): Nebula looks around at every launch and on the Models screen's Network panel: engines
already running here at their usual ports (adopt one as a tier in a click), and the CLIs installed here with the line
that connects each to Nebula. Grok, Claude Code and Codex installed here can be offered as models ("offer as a model"
in the launch dialog or Network; a command of your own under Add a model → a CLI): the conversation goes to the CLI as
a prompt and its output is the answer. A CLI runs under your account, so it stays inside your own Nebula.
LAN discovery and pairing with other Nebulas come in a later release.
With NVIDIA's OpenShell installed (`curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh |
sh`, plus Docker), tick "run it in an OpenShell sandbox" and the CLI runs in a container that reaches nothing but
what its policy allows: its own vendor, this Nebula, and the hosts you allow (the add-model form's "hosts", or
System → Sandboxes). A connection it tried and was refused shows there as an ask, to approve or reject. With
`"model_via_nebula": true` on the tier's `sandbox`, the CLI's own model calls come back to this Nebula with a token
of their own.

The core stars: three small gold stars circle the router, and a tap opens each one's panel.

- **API Server**: the addresses to paste into apps (OpenAI, Anthropic, Responses), the Claude Code, Codex and OpenAI
  SDK lines, call counts, the model (Nebula routes each request, or one dedicated model), and a pass-key: when it is
  required, every `/v1` call needs one of its keys (`Authorization: Bearer …` or `x-api-key`), from this machine too.
  A key is shown once; Nebula keeps only its hash. Saved in the config's `"api"`.
- **Halo**: Nebula's chat (the Conversation widget), its model (Nebula routes, or a dedicated one) and its
  tools: every Lookup server that is on. It runs as the built-in agent `assistant`. Saved in `"assistant"`.
- **Lookup**: the MCP servers: list, add (a command for stdio or an http(s) URL), turn on or off, remove; their tools
  and state. Two built in, off until you turn them on: **Documents** (searches and reads the folders you add) and
  **Web search** (Brave Search with a key from the environment variable you name, else DuckDuckGo's keyless page).

A tier can be any OpenAI-compatible server on this machine (or a provider over https), with `"headers"` it sends on
every call, where `{input}` is the call's input (`api`, `assistant`, `hook`, ...). A router in front of several models
is the same with a path prefix; `"health_path": "/v1/models"` makes its catalog the health check, since a router's
caller address has no `/health` (an `api_key` or a header carries a caller token when the router asks for one):

```json
{"tiers": [{"name": "router", "engine": "http://127.0.0.1:8770/model", "health_path": "/v1/models",
            "headers": {"X-Caller": "nebula.{input}"}}]}
```

A dedicated model's choices are the tiers and, for a tier in front of a router, the models its `/v1/models` lists.
A config can both watch another Switchyard (`"watch"`, optional) and serve through tiers: the map draws the watched
Switchyard, the chat and the API answer through the tiers.

Inputs beyond chat clients (step 6): webhooks and schedules start an agent's turn. `POST /hooks/<name>` with the
hook's secret (an `X-Nebula-Secret` header, a bearer token, or an `X-Hub-Signature-256` HMAC); the reply comes back
as JSON, or is posted to the hook's `callback`. `session` is `per-call`, `per-hook`, or `field:<path>` (one session
per issue number, say); `message` is a template over the payload; `tier` names the tier its turns ask for, or with
`field:<path>` takes it from the payload per call.

```json
{"hooks": [{"name": "issues", "secret": "env:ISSUES_SECRET", "agent": "researcher", "session": "field:issue.number",
            "message": "Issue {issue.number}: {issue.title}\n\n{issue.body}", "callback": "https://example.com/reply"}],
 "schedules": [{"name": "nightly", "every": "24h", "agent": "assistant", "message": "Summarize today's sessions.", "session": "per-hook"}]}
```

```bash
python3 -m nebula --config nebula.json
```

The two CLIs through it (any model name works; each CLI warns that it does not
know the name, and goes on):

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8090 ANTHROPIC_AUTH_TOKEN=nebula claude -p --model local "hello"
```

```bash
codex exec -c 'model_providers.nebula.name="nebula"' -c 'model_providers.nebula.base_url="http://127.0.0.1:8090/v1"' -c 'model_providers.nebula.wire_api="responses"' -c 'model_provider="nebula"' -m local "hello"
```

Watching another Switchyard by URL, read only (optional, off unless the config has a `watch`; `nebula/switchyard.py`):
`"watch": {"url": "http://host:port"}` draws that Switchyard on the page from its `/switchyard/status` (the shape in
[docs/router-status-v1.md](router-status-v1.md); another Nebula serves it, and so can any router). Its
destinations become the models, its actors the ring, its callers (`/switchyard/registry`) the sources, its calls in
flight the lights, and its plan (`/switchyard/plan`) shows read only beside this Nebula's own on the Plan tab. Its
name and actors are the ones it reports; nothing about it is configured here, nothing is read from disk, and nothing
is ever sent to it. With no models of its own, this Nebula's chat, API and model writes answer 409.

```bash
python3 -m nebula.examples.fake_router &                            # a pretend Switchyard on :18300
python3 -m nebula --config nebula/examples/watch-router.json        # http://127.0.0.1:18194
```

The cockpit as a desktop application (`nebula/app/`): a native window on the same page, the server started by the app,
and optionally the engine too, its load shown in the window until it answers and stopped when the window closes.

```bash
pip install pywebview
python3 -m nebula.app --config nebula.json --install-shortcut
python3 -m nebula.app --config nebula.json
```

```json
{"app": {"title": "Nebula", "prevent_sleep": false,
         "engine": {"tier": "fast", "command": ["/path/to/impulse-server", "-m", "model.gguf", "--port", "8080"],
                    "ready_timeout_s": 900, "stop_timeout_s": 30, "log": "engine.log"}}}
```

The map zooms (the wheel, or the buttons; Fit shows everything) and pans (drag empty space). Right-click a star for
Hide; the Widgets menu lists hidden stars to show again.

Configure (shift+right-click on the map; a right-click alone is colours): this Nebula's routing policy
(`nebula/policy.py`, kept as `NAME.routing.json`, `GET`/`POST /api/routing`), read by the router on every request.
On a model star: on/off and thinking. On its stream: priority, prefer, capacity and vision overrides. A watching Nebula shows the watched router's own values, read only.

Nebula's Electron window (`nebula/desktop/`), drawn on the GPU by Chromium where WebKitGTK was too slow for the
stage. It attaches to a cockpit already answering at the config's address, or starts one with the config and stops
it on close. The Python it starts is `--python`, else `NEBULA_PYTHON`, else `~/.local/share/nebula/venv`, else
`python3`. Node is needed only to install Electron.

```bash
cd nebula/desktop && npm ci
./node_modules/.bin/electron . --config ~/.config/nebula/nebula.json
```

The System tab's error log keeps the cockpit's warnings and errors and the engine's own (`nebula/enginelog.py`). For
an engine the cockpit did not start, name where its log is:

```json
{"engine_logs": [{"tier": "fast", "command": ["journalctl", "--user-unit", "my-engine.service", "-f", "-n", "0", "-o", "cat"]},
                 {"tier": "big", "file": "/var/log/impulse/big.log"}]}
```

Tests, without an engine:

```bash
python3 -m unittest discover -s nebula/tests -t .
```

Parity with the engine, through the cockpit, with Impulse's checker (the engine must have been started with
`--alias reference`, or the cockpit with `--alias reference`):

```bash
python3 impulse/tests/api_parity.py http://127.0.0.1:8090
```

Examples: `nebula/examples/local-engine.json` (one local engine, four example agents),
`nebula/examples/watch-router.json` and `nebula/examples/fake_router.py` (watching another Switchyard).

## Licence

MIT, see [LICENSE](../LICENSE). `nebula/_vendor/strata_serve/` is code from [Strata](https://github.com/Niko1221/Strata)
(MIT, its own notice in that folder; provenance in its `UPSTREAM` file), see [NOTICE](../NOTICE).

## Building the Linux installer (on your own machine, no GitHub Actions)

Everything builds locally with Docker; nothing here uses GitHub's minutes.

1. **CUDA** (no install, no root): `python3 greenfield/cuda_redist.py --platform linux-x86_64 --out <cuda>`
2. **Impulse**, on Ubuntu 22.04 so it runs on glibc 2.35 and newer: build `impulse/` with that CUDA
   (`-DIMPULSE_PORTABLE=ON -DIMPULSE_FLASHNEXT=ON`; its default CUDA list is every generation from RTX 20 (75) to
   RTX 50 (120a), plus 80, 90 and 100; the V100 (70) needs a separate build with CUDA 12.x, which CUDA 13 dropped;
   `IMPULSE_FLASHNEXT` adds impulse-fx and the Flash-Next engine library), then, in the same Ubuntu 22.04 container (so
   the libgomp it copies needs no newer glibc), `python3 greenfield/impulse_dist.py --platform linux --build <build>
   --cuda <cuda> --out <impulse-dist> --require-fx`.
3. **llama-server** for Halo's fallback: `greenfield/halo_server_build.sh` in a clean Ubuntu 22.04 container.
4. **The network's node programs** (NVIDIA Personal-AI-Router's services): clone it and run `services/build.sh` in a `golang:1.25` container
   with `CGO_ENABLED=0 GOFLAGS=-buildvcs=false`.
5. **The installer**:

   ```bash
   python3 greenfield/package.py --platform linux --impulse <impulse-dist> --halo-server <llama.cpp build>/bin \
     --pair-bin <pair>/services/build/bin --pair-src <pair> --out dist
   ```

   The result (`dist/nebula-eh-<version>-linux-x86_64.run`, about 212 MB) installs from the internet: its own Python,
   Nebula's packages at the versions resolved here (`constraints.txt`), then `python -m nebula.prereqs` (CUDA's libraries
   from NVIDIA's pip packages, Lookup's Qdrant and models, docling's models, OpenShell, Docker images). `--offline`
   builds a self-contained installer instead.

Test it the way a user meets it: `docker run --rm [--gpus all] -v $PWD/dist:/d ubuntu:22.04` and run the `.run` as an
ordinary user inside.
