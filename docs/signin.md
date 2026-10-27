# Signing in, keys and sandboxes (secure by default)

As built:

| | Default | Notes |
|---|---|---|
| The page | a passkey or a PIN | Asked for on first open. Until one is set, the page asks for one before anything else. |
| The API on this machine (`/v1`) | no key | Apps on this PC need no key. `"api": {"require_key": true}` (the API Server star) turns one on; then revoking the last key keeps the API closed. |
| API entry points on your network | a key, encouraged | One opened without a key works, and shows a warning. |
| Engines Nebula runs | in an OpenShell sandbox | Without OpenShell, or when the sandbox cannot start, the engine runs as a plain process with a warning. A model shared from it carries the warning to the other side. |

## The page

- **PIN**: 4 to 12 digits, kept only as an scrypt hash. Wrong tries are counted per address: after 5 the next one waits
  15 s, doubling up to an hour. Every sign-in, refusal and change is in the security log.
- **Passkey** (WebAuthn, ES256): your fingerprint, face, phone or security key. Nebula keeps the public key only and
  checks each sign-in's signature, its one-time challenge, the page's origin, that you were verified, and the counter.
  Passkeys need the page opened by a name: http://localhost:&lt;port&gt;/, not http://127.0.0.1:&lt;port&gt;/.
- **Sessions**: a cookie named for this port (HttpOnly, SameSite=Strict), good for 12 hours; System › Sign-in signs out.
- **Nebula's own callers** (Gravity's forwards, the household link, `nebula invite`) carry the server's internal token
  (`X-Nebula-Internal`). Commands on this machine read it from `nebula-keys/admin.token` (owner-only, rewritten at
  every start).

## Engines in sandboxes

An engine with no `"where"` goes into an OpenShell sandbox when this machine can: OpenShell installed, its gateway
running, and a sandbox image for the engine's package (llama.cpp's is pulled; Impulse's is built on this machine from
its own `impulse-server` and the libraries it loads, once per build, on `ubuntu:24.04`; Strata has none yet). The
sandbox's policy: no network, the system and the models read-only, an unprivileged user (65534). `"where": "openshell"` insists (it fails rather than run unsandboxed);
`"where": "local"` is the owner's choice to run without one (still shown with a warning).

The models folder is mounted into the sandbox and the engine's card picked through OpenShell's driver config, which a
gateway admin enables (`allow_driver_config = true` under `[openshell.drivers.docker]` in
`~/.config/openshell/gateway.toml`); admission stays on, so only volumes carrying the approval labels attach.

**Its own key**: a sandboxed engine whose package takes a key (llama.cpp) gets the same random key Nebula calls it
with, in a folder of its own that the sandbox sees read-only at `/keys`; it runs as the owner's own user (never root),
which is what can read that file. The gateway passes Nebula's `Authorization` through to it (`bearer-passthrough`), so
a program on this machine that finds its gateway address is refused (proved 2026-10-20 on the RTX 3060: Nebula's call
200, a direct call without the key or with a wrong one 401).

