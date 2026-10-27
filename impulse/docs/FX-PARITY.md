# impulse-fx and serve/server.py: what the reference engine's callers get (One Engine P2)

A Flash-Next pack runs in `impulse-fx` (`impulse-server -m <pack>` execs it, `src/fx/`). Since P2 it answers the reference engine's
callers as Strata's Python server `impulse/flashnext/serve/server.py` does, so P3 can swap the reference engine's image without a
caller noticing. This page lists every request field, endpoint and behaviour server.py has, which of them the reference engine's
callers use, and what impulse-fx does with each. Branch `one-engine-p2`, 2026-10-25.

## The reference

- **server.py here** is `impulse/flashnext/serve/` (the numa-always tree; byte-identical to the test server's
  `strata-nvfp4-v0.1.44.1-nvfp4.2-numa/serve/server.py`). Its 325 tests in `serve/test_*.py` are the reference for edge
  cases.
- **The reference engine's image runs an older Strata server.** `~/nebula-images/reference/strata-serve` (copied to `/opt/strata-serve`
  in `nebula-flashnext:*`) is 5,895 lines against 7,411 here: it has no EOS literal (`stop_token_text`), no `</tool_call>` /
  `<tool_call>` / `<tool_response>` literal marks (#2026), no `_check_parts` (a `refusal` part), no stable tool order, no
  request sampling type checks, no `logit_bias`, no `uncensored` alias, no non-finite retry (#879). impulse-fx follows the
  reference, so these are the differences callers can see when P3 swaps the image (most only on odd inputs). The one that
  shows on ordinary answers is the EOS literal: where the model writes `<|im_end|>` after a space, an opening quote or a
  backtick, or inside a tool argument, the reference (and impulse-fx) writes the token's text and goes on; today's image
  ends the answer there. `"stop_token_text": false` in the run config gives today's behaviour (a decision for the owner).
  Gate P2's arm L replays the callers' shapes on the image's server code to put the rest on record.

## The callers it was checked against (2026-10-25)

- **Nebula** (`nebula/engine.py`, `turn.py`, `dialects/chat.py`): only `POST /v1/chat/completions` (Anthropic and Responses
  requests are translated to chat first); `stream` false or true, `stream_options.include_usage` when streaming; `model`;
  `chat_template_kwargs.enable_thinking` and, when it clamps the output, `chat_template_kwargs.thinking_budget` and
  `reasoning_budget_tokens`; `max_tokens`; `response_format` json_schema (typed tiers); `id_slot` / `cache_prompt` only for
  other tenants. It reads `choices[].message|delta` (`content`, `reasoning_content`, `tool_calls`), `finish_reason`, `usage`
  (incl. `prompt_tokens_details.cached_tokens`) and `timings`. Health: `GET /v1/models` (the reference engine's tier `health_path`),
  `GET /health` 200 at start, then a 1-token chat. `GET /props` (`default_generation_settings.n_ctx`, `total_slots`,
  `modalities.vision`, `model_alias`, `model_path`), `GET /slots` (`is_processing`). `POST /tokenize` is tried for sizing
  and remembered as absent after a 404. No `/metrics`, no `/v1/status`.
- **An agent router** (role agents with tools): `{model, messages, tools, tool_choice: "auto"}` plus a generation policy
  `chat_template_kwargs: {enable_thinking: true}`, `reasoning_format: "deepseek"`, `reasoning_budget_tokens: 8192`,
  `temperature: 0.2`; a long-form writer `{max_tokens: 20000, temperature: 0.4}`; picture checks `{model: <alias>, messages
  with image_url data: URLs (labelled png / jpeg; may be WebP bytes), max_tokens, stream: false}`.
- **An app harness**: `{model, messages, stream, chat_template_kwargs.enable_thinking}` through Nebula; no tools, formats or
  pictures.
- **Monitors**: a watchdog's probe: `GET /slots`, a 1-token chat with `temperature: 0`, thinking off; a health script that
  greps `/health` for `"loaded": true`; a speed probe that reads `timings.predicted_per_second`.

Used by nobody found: `logprobs` / `top_logprobs` / `n_probs` (server.py has none either), `stop`, `top_p` / `top_k` /
`min_p`, `seed`, `n`, any `strata_*` field, `return_progress`, slot save/restore, `/v1/messages`, `/v1/responses`,
`/metrics`, `/v1/status`.

## Requests: POST /v1/chat/completions

| field / behaviour | server.py | impulse-fx P1 | impulse-fx P2 | used by |
|---|---|---|---|---|
| `messages` (system, developer → system, user, assistant, tool; content string or parts) | `openai_to_messages` | llama.cpp's oaicompat parse | the same function (`fx_front.cpp`), JSON-string messages (#460), refusal parts read as text, unknown parts left out | all |
| late `system` / `developer` messages | turned into `user` in place | template error | as server.py | role agents (hooks) |
| the template | the pack's `chat_template.jinja` (= the dense GGUF's) through jinja2, completed empty assistant turns dropped (#843) | llama.cpp's template path | the GGUF's template through llama.cpp's Jinja engine, with Python's float repr and `str.strip()` (`llama-hooks/fx-jinja-python.cmake`), the same turn filter | all |
| literal tags in messages (`<think>`, `</think>`, vision markers, tool tags, control tokens' text) | encoded as text (#537, #554, #2026), an assistant's leading think block kept, a wrapped `<tool_response>` user turn kept | parsed as specials | as server.py: the same marks, the spans encoded as text | any |
| tokenizer | Strata's Python BPE on the pack's tokenizer | llama.cpp's | Strata's rules on llama.cpp's vocabulary: literals leftmost-longest (CONTROL only with parse_special), stretches without a literal through llama.cpp's BPE (gate P1a), stretches with a quoted literal through the same BPE merges in `fx_front.cpp` | all |
| `tools` (OpenAI or bare), checks, stable order of a known set | `_tool_list`, `_stable_tool_order` (64 sets) | llama.cpp | as server.py | role agents |
| assistant `tool_calls` history (`arguments` as JSON text, object, malformed, other) | `tool_arguments` (#510) | llama.cpp | as server.py | role agents |
| `tool_choice` (`auto`, `none`, `required`, named, Anthropic forms) | `tool_choice_of`, `forced_call` (the call opened in the prompt or after the thinking) | ignored | as server.py | role agents (`auto`) |
| `reasoning_effort`, `reasoning.effort`, `chat_template_kwargs.{enable_thinking, reasoning_effort}` | `effort_kwargs`; enable_thinking false wins | enable_thinking only | as server.py | agent router, Nebula, app harness |
| `chat_template_kwargs.thinking_budget`, `reasoning_format` | ignored | ignored | ignored (as server.py) | Nebula, agent router |
| `reasoning_budget_tokens` (#123) | wrap-up text and `</think>` at the budget, capped to leave room for the answer (#984) | no | as server.py | agent router (8192), Nebula clamps |
| `max_tokens` / `max_completion_tokens` (0 / -1: the rest; int() of a string) | 400 when prompt + max + 8 > context (never truncated; `fit_max_tokens`) | clamped | as server.py | all |
| `temperature`, `top_p`, `top_k`, `min_p`, penalties, `seed`; numbers as strings; `repeat_penalty` / `repeat_last_n` | `check_request_sampling`, `sampling_keys` (Python's float repr) | the keys, `%.9g` | as server.py | agent router / writer temperatures |
| `strata_tune`, `strata_checkpoint`, `experimental_speed_projection` / `uncensored`, `logit_bias` (engine INFO logit_bias=1) | engine keys | some | as server.py | none |
| `strata_prefix` (#R1, a pinned shared prefix) | `pin=N` | no | **not ported**: ignored (server.py's own answer when it cannot mark a prefix; only the engine's caching changes) | none |
| `stop` / `stop_sequences` | `StopMatcher` on the answer's text (not across thinking or calls), STOP sent | llama-server's stop | as server.py | none |
| `response_format` json_object / json_schema | directive in the system turn; the answer checked as one JSON object (the image has no `jsonschema`), canonical JSON; 502 `structured_output_failed`; buffered when streamed | llama.cpp grammar path | as server.py | Nebula typed tiers |
| pictures: `image_url` / `input_image` / Anthropic `image` parts | `strata-vision` child (ENC), cache by hash, `GENI` with the combined file and `img_keys` (#2228); data: URLs, files, public http(s) URLs; Pillow for formats stb cannot read and EXIF-turned JPEGs | 400 | as server.py through `fx_vision.cpp`: data: URLs and local files; other formats converted by the same Pillow code (`--fx-image-python`, the reference engine's image has Pillow 12.3.0); **http(s) URLs refused** (no caller sends one) | pipeline |
| tool-result pictures that cannot be read | a text note in their place | 400 | as server.py | none |
| `stream`, `stream_options` | SSE `data:` chunks, usage and timings always in the last chunk, `: keep-alive` every 10 s while reading or queued (#1619) | llama-server's chunks | as server.py | Nebula, speed probe |
| `return_progress` (llama.cpp's prompt_progress chunks) | yes | no | **not ported**: plain keep-alives | none |
| `strata_mcp` (the web app's MCP tools) | only with MCP servers in the config (the reference engine has none: ignored) | no | ignored (as server.py without MCP servers); with `response_format` a 400 as there | none |
| `n`, `logprobs` | ignored / absent | — | ignored / absent | none |

## Answers

| behaviour | server.py | impulse-fx P2 |
|---|---|---|
| reasoning split (`reasoning_content`), the answer's leading newlines, a quoted `</think>` | `OutputParser` | the same state machine (`fx_front.cpp`); parity: tests/fx_front_parity.py |
| tool calls: the template's XML, values JSON or text per schema, Python `True`/`False` where a boolean is allowed, `</parameter>` inside values, unclosed calls, calls without `</tool_call>`, calls inside the thinking (#804/#1058), code fences and inline code | yes, streamed as pieces (name first) | the same; ids `call_` + 24 hex |
| undeclared call names | sent as calls (`tool_name_repair` off); repaired / text with it on | the same; `"tool_name_repair": true` in the run config works |
| `tool_call_recovery` (bare `<function=`, JSON calls, batches) | opt-in | **not ported**: the run config's `true` is refused at start (the reference engine: off) |
| EOS literal (`stop_token_text`, on by default) | the token's text where no turn ends, another engine pass | the same (two passes, `owed` piece, example turns) |
| repeat guard (#606, 256 of one token) | ends the reply as `length` | the same (no engine restart: in-process) |
| non-finite logits (#879) | one silent retry before the first token, else 503 / `length` | the same |
| finish_reason | `stop`, `length`, `tool_calls` (all calls whole), `cancel` → `stop` | the same |
| `usage`, `timings` (llama.cpp names, drafts) | from the engine's DONE lines, summed over passes | the same |
| model name in the answer | the request's when it is an alias, else `model_name` | the same (`model_name`, `aliases` from the run config; `--alias` overrides) |
| errors | 400 `invalid_request_error` (message texts), 502 structured, 503 engine | the same statuses and types; a few 400 message texts differ (malformed shapes) |

## Endpoints

| endpoint | server.py | impulse-fx P2 |
|---|---|---|
| `GET /health` | 200 `{status, max_context, model, images, api_key, image_generation, loaded, service: "strata"}`; not listening until the engine is ready | the same once ready; 503 "Loading model" before (Nebula polls until 200) |
| `GET /v1/models`, `/models` | the model plus each alias (`alias_of`), `meta.n_ctx`, `architecture.input_modalities` | the same |
| `GET /props` | `default_generation_settings.{n_ctx, params}`, `total_slots`, `model_alias`, `chat_template`, `chat_template_caps`, `modalities.vision`, `model_path`, `build_info`; `?model=` checked | the same, plus `impulse.engine_info` (the gates read it) |
| `GET /slots` | one slot: `is_processing`, `n_prompt_tokens` | the same |
| `POST /tokenize`, `/detokenize` | absent (404) | absent unless `--fx-tokenize-endpoints` (Nebula would start using it) |
| `POST /completion`, `/completions` | absent | llama-server's raw-id completion (the gates' path), not used by callers |
| `/v1/messages`, `/v1/messages/count_tokens`, `/v1/responses`, `/v1/images/generations`, `/metrics`, `/metrics/prometheus`, `/status`, `/v1/status`, `/settings`, `/config`, `/mcp`, `/load`, `/unload`, `/v1/vram`, `/slots/N?action=save|restore`, the web app | yes | **not ported** (404): no caller uses them (Nebula translates Anthropic and Responses to chat; slot files need `slot_save_path`, which the reference engine's config has not) |

## Engine-facing behaviour

| behaviour | server.py | impulse-fx P2 |
|---|---|---|
| one request at a time (`parallel` 1) | FIFO; queued requests send keep-alives and give their place up when the client leaves | the same |
| RoPE past the trained context | `rope_fallback_args`: setup's `rope_auto` dynamic, yarn where dynamic cannot run (several GPUs, batch slots) | `"rope": {"type", "factor", "orig_ctx"}` in the run config (none / linear / yarn / dynamic / ntk) and the automatic pick for `--max-context` above 262144: dynamic yarn (the engine now runs it on a layer split and with batch slots, engine-wide), static yarn with factor context / 262144 with pipelined slot groups, `--batch-mtp`, q4 K/V or `--pipeline-windows` (fx_server.cpp `rope_args`); the args' own `--rope-scaling` wins |
| several at once (`parallel` N > 1: `--batch N`) | solo when alone, promoted into a batch slot when a second request comes (BGEN), back to solo when left alone; long prompts alone (BGEN-LONG); BYIELD; slot picked by held prefix / LRU / pipeline groups | the same (src/fx/fx_slots.hpp, also `--parallel N` / `-np N`), admissions and slot waits in arrival order; `/props` total_slots and `/slots` the engine's slots; the startup log says what a slot costs the expert cache (INFO batch_slot_mib). Not here yet: BYIELD (a long read giving way to a short prompt) |
| the GEN line | `GEN <max_new><keys> <ids>` / `GENI ... <file> <ids>` | byte for byte the same (tests: heads and ids) |
| conversation cache / prefix reuse | the engine's own (conversation cache, checkpoints): the server sends the whole prompt every turn; `park` / `BGEN` only with `--batch` | the same (no `--batch`) |
| STOP | when the consumer stops early: a stop string, a stop token (also after the engine's own EOS), a client that left (polled every 0.5 s, non-streamed too, #430), a wrap-up | the same |
| a silent engine (#481) | `engine_silence_s` (300): the engine killed, the next request restarts it | the same allowance (PP-chunk based); the request ends 503 and impulse-fx exits so its supervisor restarts it (the engine is in-process). The CPU/IO stall check (#1317, psutil) is not ported |
| the engine ends | restart on the next request | impulse-fx exits (as P1) |
| the engine log (run config `"log"`) | appended (never truncated); one `[strata] <time> engine started: <exe> --serve <args>` line per start; the engine's stderr and the vision helper's go there; the server's own lines go to stdout only | the same file, appended, the same start line (`<exe>` = libimpulse_fx); the engine's, the vision helper's and llama.cpp's stderr go there only; the front's own lines go to the log **and** to the console (its original stderr). Without `"log"` everything stays on stderr (`src/fx/fx_log.hpp`) |
| config keys that change answers and are not here | | refused at start: `reasoning_loop_recovery`, `reasoning_close_retry`, `env_to_user`, `effort_position: end`, `replicas`, `sampling`, `reasoning_effort`, `slot_save_path`, `tool_call_recovery`, `vision.elastic`, `vision.cuda_device` (the reference engine sets none) |
| Chat settings shared with apps (`*.shared-settings.json`) | applied | not read (the reference engine's has none) |
| Host / Origin checks, CORS, API key | yes | API key (`--api-key`, Bearer or X-Api-Key); no Host/Origin checks (impulse-fx listens on loopback in its sandbox) |

## Tests

- `tests/fx_front_parity.py` (the dev machine, CPU): server.py's own functions against `impulse-fx --fx-front-test` on the reference engine's
  tokenizer and template. 589 requests (557 accepted, 32 refused by both; 349,513 prompt tokens) - token ids, GEN heads,
  max_new, thinking, forced openings: 0 mismatches. 1,914 parser feeds (22 outputs x thinking x tools x repair x 5 ways of
  cutting x 2 finishes, plus the 37 recorded #804/#1058 specimens): 0 mismatches. 27 scripted engine runs
  (Service._run + openai_collect: answers and every GEN head): 0 mismatches.
- `tests/fx_server_test.py` (mock engine): the endpoints, the reasoning split, tool calls whole and streamed, an EOS literal's
  second pass, response_format, STOP when a client leaves (streamed and not), pictures through a stand-in strata-vision
  (GENI, pads, a WebP converted by Pillow), a silent engine.
- Gate P2 (a two-GPU test host, its own gate script): prompt parity on that host's files, speed, pictures, the replay
  A vs B, a 45-minute soak, the b2 rerun.
