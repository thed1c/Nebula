# Security

## Reporting a vulnerability

Please report security problems privately, through GitHub's **Report a vulnerability** button on this repository's
**Security** tab, not in a public issue. Say what you found, how to reproduce it, and which version (`nebula --version`,
or the installer's name). You will get an answer; a fix ships as a new release, and the release notes name the
problem once it is fixed.

Supported: the latest release. Older releases get no fixes; update from the page's **Update ready** pill or the
System page's **Look now**.

## What Nebula protects, and how

- **The page and its API** answer this machine only unless you turn on the API Node or pairing. Every request carries a
  key; the page's own key and each caller's key live in `nebula-keys/` (0600) beside the config, never in the config
  or the logs.
- **Engines** Nebula starts listen on loopback only, with a key only Nebula holds where the engine supports one, and run
  in an OpenShell sandbox on Linux when Docker is there (docs/sharing.md: what a guest can and cannot reach).
- **Guests** (other people's apps, a paired Nebula) see only models shared with them, only their own history, and never
  a cached prompt of someone else's (docs/sharing.md).
- **Pairing** (docs/gravity.md) is by invite only: nothing is discovered on the network, and a peer gets nothing until
  you tick a model for it.
- **Updates** are signed: Nebula installs an update only when its feed's Ed25519 signature matches the key built into
  `nebula/updates.py` and the download matches the feed's sha256, and only after your click (docs/updates.md).
- **The security log** (System page) records refused requests, keys made, used and revoked, refused guests and peers,
  and OpenTelemetry changes (docs/security-log.md).

What leaves your machine, and when, is in [PRIVACY.md](PRIVACY.md).

Impulse's Flash-Next engine (`impulse/flashnext/`) carries its own notes for its standalone server in
`impulse/flashnext/SECURITY.md`.
