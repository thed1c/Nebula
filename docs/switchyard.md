# Nebula's own Switchyard

Nebula runs its own model servers, knows who is asking, chooses where each request goes and shows all of it. Every
part is in the page (the Engines, Callers and Plan tabs, the Lanes and Decisions widgets) and in the API under
`/api/yard/` (`nebula/yard.py`). Nothing here needs another router; a router Nebula watches is shown the same way.

## Engines: the servers Nebula runs (`nebula/engines.py`)

An engine is a model server Nebula starts and stops, from the config's `engines`:

```json
{"engines": [{"name": "small", "kind": "llama.cpp", "model": "models/small.gguf", "port": 18300,
              "context": 8192, "slots": 2, "card": "GPU-…", "options": {"gpu_layers": 999, "thinking": false},
              "autostart": true}]}
```

- **Packages** (`kind`, `nebula/packages.py`): `llama.cpp` (`llama-server`), `impulse` (`impulse-server`; Flash-Next
  packs run on its impulse-fx), or `command` (any command line). Strata's own package was removed 2026-10-25 (the
  owner's word); a saved `strata` engine loads as `impulse`. Each package declares every setting it takes, and the
  Edit form is drawn from that list: the vision projector, LoRA, MTP, KV cache types, RoPE/YaRN, GPU layers and split,
  flash attention, batch sizes, threads, drafting, thinking and its budget, sampling defaults, the chat template,
  the server's own API key. The command line it runs is shown under the form as it changes.
- **Deployments:** every change is a new version kept beside the config (`NAME.deployments.json`); **Undo** goes back
  one. A running engine keeps its old version until it is restarted.
- **Start, Stop, Drain, Restart, Verify.** Drain sends an engine nothing new (routing skips it) and stops it when
  its running calls are done. Verify checks it answers and that `/props` names the deployment's model file. An
  engine that already answers at its address is attached, never started twice.
- **Cards:** an engine can be pinned to a GPU (UUID or index, from `nvidia-smi`). A start on a card the machine does
  not have is refused, and so is one the fit estimate says will not fit (the GGUF header gives the KV cache size).
- **Make it a model:** an engine becomes a model (a tier) with one click, and is then routed like any other.

### Raw

Edit › Raw shows the engine's whole command line as text. "Read into the settings" turns it back into the toggles it
knows; flags without a toggle are kept exactly as written; nothing is dropped. Flags Nebula sets itself (this machine
only, the key only Nebula holds, the engine's own pages off, the cache kept per guest) are locked: a raw edit cannot
change them, and they are put back on every start, with a note saying why. "Show the changes" lists what differs;
saving is a new deployment, so Undo takes it back.

### Deploy your own

Models ▾ › Add a model › **Deploy your own** (or the Engines tab): pick a package, paste a Hugging Face link (a
repository or one `.gguf` file) or give a file on this machine, choose the settings, choose where it runs, Deploy.
The link is resolved to the repository's GGUF files grouped by quantization, with sizes against the disk and the
cards; a projector (`mmproj`) is found and pulled with the model. Downloads resume, and are checked against the
Hub's sha256. A gated or private model needs a Hugging Face token: it is kept only in `nebula-keys/huggingface.token`
(0600), never in the config. `HF_ENDPOINT` points the pulls at a mirror.

## Callers: who is asking (`nebula/callers.py`)

A caller is an app or agent framework that uses the API: `{"id", "name", "class", "tags", "models", "mode"}`.

- **The key names the client.** A caller's keys (made in the Callers tab, shown once, kept as hashes) identify it on
  `Authorization: Bearer …` or `x-api-key`; with the API Server requiring a key, a caller's key opens it.
- **The label names the origin inside it.** `X-Nebula-Caller: <label>` is recorded on every decision. Without a key,
  a label matching a caller's tag globs picks that caller.
- A caller's `models` are its job's ticks (in order) and its `class` its place at the gates. Both are its row in the
  Plan, and Apply writes them.
- `mode: "passthrough"` uses exactly the model it names (else its first tick): no escalation, overflow or retry.

## Routing: one explainable choice (`nebula/turn.py`)

Each request goes through the same stages, and every decision records them: the caller's **ticks** → what is
**allowed** (the policy's on/off, hours and counts; engines being drained) → what **answers** (health) → what
**fits** (the prompt and its answer against each model's context; nothing does: the parking lot) → the **choice**
(the ladder's floor and escalation, then the gates: a free slot, or the expected wait within the class's tolerance,
else the next ticked model). An engine that fails before it answered anything moves the request to the next eligible
model once (`rules.retries_before_first_byte`), recorded as a decision of its own. A model the request named itself
is never moved in silence. A route (tier) can have its own deadline, `timeout_s`.

## Machines that sleep (`nebula/power.py`)

`hosts` lists the machines a model can be on and how to wake them (`{"kind": "wol", "mac", "broadcast"}` or
`{"kind": "command", "command"}`); a tier names its machine with `host`. A machine is awake when one of its models
answers. When every model a request may use is asleep, an **interactive** call wakes one machine (if
`interactive_may_wake`) and waits up to `wake_seconds` for it, recorded as `woke:<host>`; an agent or background call
never wakes a machine: it uses an awake model, waits, or is refused with the reason. Engines › Machines shows each
machine and a Wake button.

## OpenShell (`nebula/openshell.py`)

With NVIDIA OpenShell installed, Engines › OpenShell shows its gateways, its sandboxes and each sandbox's effective
policy (read only: policies are never written from Nebula).

- **An engine in a sandbox:** an engine's Where = OpenShell (`"where": "openshell"`, an `image`; llama.cpp has a
  default one; Impulse's is built here from its `impulse-server`, `nebula-engine-impulse-server:<sha256>`) runs in a
  sandbox (`nebeng-<name>`, within OpenShell's 19 characters): its cards passed in (NVIDIA CDI; several for a split
  model), its `env` settings, its `cpus` pinning (`taskset`; binding memory to a NUMA node is refused in a sandbox),
  the models folder mounted read-only at `/models` (a named Docker volume, `nebula-models`), a policy that allows
  nothing out, and its port exposed through the gateway, which is where Nebula routes it. Its model must be in the
  models folder. Stop deletes the sandbox. Impulse takes no key, so on this machine its gateway address is as open
  as a local engine's port.
- **This Nebula as a sandbox's model provider:** "Use this Nebula from a sandbox" gives a provider profile that
  reaches only this Nebula (`host.openshell.internal:<port>`) with a caller's key as its credential, and the commands
  to import it and attach it. The sandbox's calls are then that caller's (its models, its own cache).

## Gravity on this machine (`nebula/pairlocal.py`)

A Gravity router installed on the same machine (it sends each request to one of its machines that has the exact
model, the least busy first, and retries on another before anything was answered) is found on loopback only and
shown under Engines › Gravity on this machine. **Adopt** makes one model per model it offers (`gravity-<model>`,
`mesh: "gravity"`), routed like any other; nothing is looked for on the network.

## Seeing it

- **Lanes:** each model's slots, the calls holding them (with the caller and label), the waits and the routes.
- **Decisions:** every decision with its stages, retries marked.
- **Stats** (Engines and Callers tabs): per model and per caller over 10 minutes, 24 hours and 7 days: calls,
  latency and queue wait, prefill and decode seconds, tokens per second, prompt tokens, and prompt-size bands.

## Being watched

`GET /switchyard/status` answers in the [router-status-v1](router-status-v1.md) shape, `GET /switchyard/plan` with
the plan and `GET /switchyard/registry` with the callers, behind the API's key when it requires one. Another Nebula
watches this one with:

```json
"watch": {"url": "http://host:8090"}
```
