# Nebula's design: a local engine as the base of an assistant

**Status: built.** This is the design Nebula was built from (in the code it is still called "the cockpit", its working
name; the package is `nebula/`). The build diary that followed it stays with the engine fork it was written in; the
README says how to run every piece. Numbers quoted here come from the pages that measured them.

The cockpit is a new server above the Strata-Lite engine. It gives one person, on one PC, an assistant that picks how
hard the model thinks, escalates to a bigger model when the local one is stuck, carries the conversation across that
escalation without starting over, and keeps a record of every decision it made. Later it becomes a platform other people
run without authoring a project. Strata's chat page stays what it is: a window onto one model.

## The stable line

The engine's HTTP API is the line. Below it: `strata-lite-server`, its ports, the expert cache, MTP, the KV pager, the
installer's builds. That part keeps tracking upstream Strata and llama.cpp as it does today. Above it: the cockpit, written
fresh, in its own package. The cockpit talks to the engine only through what `lite/docs/API.md` lists:

| endpoint | the cockpit uses it for |
|---|---|
| `POST /v1/chat/completions` (streaming, `reasoning_content`, `cache_prompt`, `chat_template_kwargs`) | every model turn |
| `GET /health`, `GET /props`, `GET /slots` | is the engine up, which model, is the one slot busy |
| `GET /metrics`, `timings` and `usage` in each response | the ledger's cost columns |

Rule: the cockpit never imports from `strata/` or `lite/`, and never reads the engine's files. A second engine (another
box, or a second process once the box has the VRAM) is the same API at another address.

## What shapes the design (measured, or stated by the engine's docs)

- **One request at a time** per engine (`lite/docs/RUN.md`). Anything the cockpit wants from the model, a user turn, a
  summary, a routing classifier, competes for that one slot.
- **One model per process.** Changing model means loading another one. Strata's llama-swap page measured 80 s and 105 s
  to load Qwen3.8-Flash-Next from a SATA SSD; Strata-Lite's own load time has not been measured for this page. Either
  way it is minutes, not seconds, and the prompt cache is lost.
- **The prompt cache lives in the process.** A conversation that moves to another engine is read again from the start.
  At 157k tokens that read took 234 s on an RTX 3060 and 159 s on an RTX 5070 (README grid, 262k context).
- **Thinking is a request parameter, not a model.** For Gemma 4 `chat_template_kwargs: {"enable_thinking": false}` turns
  it off; thoughts arrive as `reasoning_content`. Switching costs nothing and keeps the cache when the toggle sits at the
  end of the prompt (Strata's `effort_position: end`).
- **The first reply after a long read is 13 to 27 % slower** while the cache refills (README grid). A routing decision
  that moves a long conversation pays this on arrival.

These four facts decide the routing model: on one card, the free axes are *whether and how long to think*; a *bigger
model* is a tier crossing to another engine, and the crossing has to carry a self-contained packet because nothing else
crosses with it.

## Seams

| seam | owner | what it is |
|---|---|---|
| Inference | Strata-Lite engine, unchanged | the API above |
| Gateway | cockpit, lifted from `serve/` | OpenAI, Anthropic and Responses dialects in; engine chat completions out |
| Stream parsing | cockpit, lifted from `serve/frontend.py` | `reasoning_content` from the engine where it separates thinking itself; `OutputParser` for tool calls written as template tags, and for engines that do not separate thinking |
| Tools | cockpit, lifted from `serve/mcp.py` | `McpHub`: stdio and Streamable HTTP MCP servers, tools named `server__tool`, failures as `error:` results |
| Ledger | cockpit, new | SQLite: sessions, tasks, attempts, events, decisions, handoffs, outcomes |
| Router | cockpit, new | picks tier and thinking per turn, writes a decision row, triggers escalation |
| Continuity | cockpit, new | event capture, structured compaction, the handoff packet |
| UI | cockpit, new | the hot rod's own page: model, tier, thinking, escalation state, an escalate button, session list, monitor |
| Host | cockpit, lifted | the installer, `serve/keepawake.py`, `serve/winjob.py`, idle unload |

### Lifted from `serve/`, and how

Now that Nebula is its own repository, the modules it uses are vendored, unchanged apart from their imports of each
other, in `nebula/_vendor/strata_serve/` (provenance in its `UPSTREAM` file; MIT, Strata's notice kept). The design
was written while Nebula lived in the engine's repository and imported them; the table is that design:

| module | lift | note |
|---|---|---|
| `serve/frontend.py` | `openai_to_messages`, `anthropic_to_messages`, `effort_kwargs`, `OutputParser` | the parser's tags are Strata's Qwen3.8 template's; confirm per engine model which tags the template emits before relying on it |
| `serve/responses.py` | the Responses item and event assembler | Codex CLI is its first client; keep it so Codex plugs into the cockpit unchanged |
| `serve/mcp.py` | `McpHub` whole | it already contains its children with `winjob.contain` |
| `serve/replicas.py` | the cost model only: prompt tokens to re-read plus a load constant per running request | generalized from identical replicas to engines of different models, this is the router's cost of moving a conversation |
| `serve/telemetry.py` | the sampler, for the UI's monitor | NVML through ctypes, no pip package |
| `serve/keepawake.py`, `serve/winjob.py` | whole | Windows lifetime and sleep |
| `setup.py`, `setup_lite.py`, `START-HERE.bat`, `setup.sh` | the install path | the cockpit becomes what the start script starts; the engine becomes a child it contains |

Not lifted: `serve/server.py` (376 KB, one model, one chat window; its tests are 286 KB), the chat page, the Monitor tab,
`serve/runconfig.py`'s settings page.

## The ledger

One SQLite file, one writer, the harness's transaction habit: a row exists before the thing it describes is sent. Every
row carries `tenant_id`. It is always the one local owner in v1 and nothing filters on it yet; it is there so that the
platform version adds a filter instead of a migration.

| table | one row per | columns that matter |
|---|---|---|
| `sessions` | continuing conversation | `session_id`, `tenant_id`, `lifecycle` (active, closed), `created_at`, `revision` |
| `tasks` | objective within a session | `task_id`, `session_id`, stated acceptance if any |
| `attempts` | one model's go at a task | `attempt_id`, `task_id`, `tier`, `engine`, `model_id` from `/props`, `thinking`, `state` (running, completed, escalated, failed, cancelled), `verified_outcome` (unknown by default) |
| `events` | committed message, tool call, tool result, user correction, control record | `event_id`, `session_id`, `seq` (per session, gapless), `attempt_id` where applicable, `kind`, `content` or a reference, `content_sha256`, `completeness` (complete, interrupted), `reasoning_visibility` (text, summary, unavailable) |
| `decisions` | router choice | `decision_id`, `attempt_id`, inputs the router saw, chosen tier and thinking, reason code, `advisory` flag |
| `handoffs` | packet sent across a tier | `handoff_id`, from and to `attempt_id`, `packet` JSON, `packet_sha256`, `idempotency_key`, `sent_at`, `acknowledged_at` |
| `outcomes` | what happened to an attempt | transport result, worker claim, user signal (regenerate, rephrase, escalate pressed, thumbs), verifier result where one exists |
| `costs` | each engine call | `prompt_n`, `predicted_n`, `draft_n_accepted`, seconds, from `timings` and `usage` |

The vocabulary and the invariants are the secretary investigation's (`CONTRACTS.md` there, 2026-10-15): sequence
coverage, content hashes, escalated is not failed, unknown stays unknown, a correction appends and never rewrites. What
that design calls `workspace_id` and `project_id` collapses to `tenant_id` here; its notebook revision and writer fence
become the single SQLite writer and `sessions.revision`.

## The router

Tiers are rows in the config, not code:

```json
"tiers": [
  {"name": "fast",  "engine": "http://127.0.0.1:8080", "thinking": false},
  {"name": "think", "engine": "http://127.0.0.1:8080", "thinking": true, "reasoning_budget_tokens": 4096},
  {"name": "big",   "engine": "http://second-box:8080"},
  {"name": "cloud", "provider": "openai-compatible", "base_url": "...", "model": "...", "api_key_env": "..."}
]
```

`fast` and `think` are the same engine and the same prompt cache. Moving between them is a request parameter. `big` and
`cloud` are tier crossings and go through a handoff packet.

**Per turn, the router picks the lowest tier whose conditions hold.** v1 conditions are rules, written down and logged,
with no learning:

| go up when | signal source |
|---|---|
| the user pressed escalate | UI; the strongest and cleanest label |
| the user regenerated or rephrased the same ask | ledger: two consecutive user turns with high overlap |
| the attempt reported it was stuck or asked for a capability it lacks | the model's own text, matched by the parser's events, logged as a claim |
| a tool call failed N times in a row | events |
| a verifier failed, where the task has one | outcomes |
| the request names a tier (`model: "cockpit/big"`) | gateway |

Each choice writes a `decisions` row with the signals it saw. The router never reads `outcomes` to change a rule in
v1; it only produces the rows a later evaluation needs. That evaluation is the secretary investigation's `VALIDATION.md`,
section 3: replay with the features known at decision time, denominators from assignments not hops, and no benefit claim
without a matched comparison.

**Cost of moving.** Before a crossing, the router computes the replicas cost: tokens the destination would read from
the start, plus the load constant if the destination is busy. The UI shows it ("moving this conversation re-reads 41k
tokens, about 30 s on the big box"). Over a configured ceiling the router asks before moving.

## Continuity and the handoff

**Capture is deterministic.** Every completed message, tool call, tool result, and user turn is an `events` row at the
moment it commits, with its sequence number. Interrupted streams are rows marked `interrupted`. Reasoning text is stored
where the engine exposes it and marked `unavailable` where it does not. No model reads the stream to write notes in v1:
the one slot belongs to the user.

**Compaction becomes structured.** Strata's chat page summarizes at three quarters of the context. The cockpit does the
same at the same point, but asks for a fixed shape instead of prose: objective, constraints the user stated (quoted),
what has been tried and what each try showed, what is open, pending tool effects. That shape is stored as an event and
is the `briefing` of the next packet. It is the only model-written summary in v1, and it runs on the user's slot
because it runs in the user's turn.

**The packet** is built from the ledger, not from the model:

```json
{
  "handoff_id": "...", "session_id": "...", "task_id": "...",
  "from_attempt": "...", "to_tier": "big",
  "original_task": "the user's first message of the task, verbatim",
  "constraints": ["quoted user statements"],
  "escalation_reason": {"source": "user_button | rule:<name> | model_claim", "text": "verbatim"},
  "briefing": {"...": "the last structured compaction, or null"},
  "events_since_briefing": ["every committed event after the briefing's seq, in order, content included"],
  "tool_state": {"pending": [], "uncertain": []},
  "coverage": {"through_seq": 57, "gaps": []},
  "budget": {"fits": true, "destination_context": 262144}
}
```

Hash the packet; the idempotency key is the hash plus the handoff id; a retry sends the same bytes. A packet with
pending or uncertain tool effects is not sent: the router waits for the boundary or marks the effect unknown and says so
in the packet. A packet that does not fit the destination's context is not sent either; the UI explains which.

**Safe boundary.** Escalation happens only between tool calls. The source attempt is marked `escalated` before the
packet is sent; it issues no further tool calls after that mark. The destination's reply is a new attempt under the same
task.

## The UI

Requirements, not a design:

- Every answer shows which tier and model wrote it and whether thinking was on. Escalation is shown as a move, never as
  a failure.
- An escalate button on every assistant turn. It is the strongest label the ledger gets.
- The router's reason for its choice is one click away, read from the `decisions` row.
- Sessions list, resume, close. Closing is explicit; a closed browser tab closes nothing.
- A monitor from `serve/telemetry.py`: GPU, VRAM, the engine's slot, the current context size against the KV pager.
- The same page works against one engine or several; adding a tier is editing the config.


## Not in v1, on purpose

- A secretary model reading the stream continuously. It would take the user's slot; see step 5.
- Embeddings and semantic search over sessions. Full-text search over `events` first.
- Learned routing. Rules plus a complete ledger first; the evaluation decides.
- Model swapping on one card as a routing move. Minutes of load and a lost cache make it a user action, not a router one.
- Multi-tenant enforcement. The column is there; the filter and the auth are platform work.
- Any change to the engine's API.

## Open questions for the first build

1. ~~Does `cockpit/` live in the engine's repository or in its own?~~ Its own: this repository. The stable line made
   the move a directory move plus the vendored `serve/` modules.
2. Which template tags do the Gemma 4 and Qwen3.6 GGUFs emit for tool calls, and does the engine already separate
   thinking for both? `OutputParser` is lifted on the answer, not before.
3. The `cloud` tier: which provider first, and does its API expose reasoning text, a summary, or nothing? The
   `reasoning_visibility` column records the answer per event.
4. The second box: plain HTTP to another `strata-lite-server`, or a router's front (a broker in front of several engines) with its admission and
   decision receipts? Plain HTTP first; the front's receipt shape informs the `handoffs` columns either way.
