# Sharing your models

Nebula can let other people use your models: someone else's app with a key of its own (the API Node), or a friend's
Nebula you paired with. Every one of them is a **tenant**; your own use (the page, Halo, your own apps) is
one more. `nebula/tenancy.py` keeps them apart.

## What a guest can reach

- **Only Nebula's API**, never an engine. Engines Nebula runs listen on this machine only, with a random key only
  Nebula holds (`nebula-keys/engine-<name>.key`, 0600; `--api-key-file` for llama.cpp;
  Impulse has no key yet, so loopback is its only wall). Their own chat page is off (`--no-webui`), and their `/slots`
  page is off unless guests share them (Nebula erases slots through it).
- **Only models shared with guests** (`"shared": true` on the model; an engine's `"guests": true` makes its models
  shared). A guest naming any other model is refused ("not shared with guests").
- **Only its own history.** A paired Nebula's token opens the model API (`/v1/*`), `/health` and its own
  `/switchyard/status`, never the page's API (your sessions, transcripts, decisions, stats, settings). A caller's key
  sees only its own calls, its own plan row and its own entry in the callers list. Your conversations with Halo are
  never served to anyone. Sessions are per tenant: two tenants whose conversations open the same way never share one.

## The cache

No engine (llama.cpp, Impulse) has a per-tenant cache salt: a cached prompt is reused by the next request
that shares its prefix, and the time to its first token shows it. On a shared model Nebula keeps each tenant to its
own cache:

- **A slot each** (`cache_isolation: "slots"`, llama.cpp engines): each tenant's requests go to its own slot
  (`id_slot`); a slot that changes tenant is erased first, so nothing cached for one tenant is ever served to another,
  while a tenant's own next request still reuses its prefix. A shared llama.cpp engine starts with `--cache-ram 0
  --no-cache-idle-slots -sps 0`, so no prompt is kept across slots.
- **No cache for guests** (`cache_isolation: "no-cache"`, the default elsewhere, Impulse today): a guest's
  request is sent with `cache_prompt: false`, so it can never hit a cached prefix.

Each decision records what was done ("slot 1", "slot 0 (erased: it held another tenant's prompt)", "no cache").

## Marking a guest

A caller with `"guest": true` (the Callers tab: "someone else's app") is a guest. A paired Nebula is always one.
Your own apps are callers without it: they are kept apart from guests on shared models, and may use every model.
