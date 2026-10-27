# Updates and what is kept on disk

Everything Nebula keeps on disk carries its version (nebula/migrate.py), so an update never breaks an install.

| File | Its version |
|---|---|
| the config (`nebula.json`) | `"config_version"` (missing = 1) |
| the ledger, the security log, Lookup's catalog (SQLite) | `PRAGMA user_version` |

- **An older file** is copied aside first (`<file>.v<N>.bak`), then brought up step by step; each SQLite step runs in
  one transaction with its version, so a failed step leaves the file as it was.
- **A file from a newer Nebula** is refused and left untouched: Nebula says to update, or to put back the copy the
  update left beside it. Going back to an older Nebula never corrupts what a newer one wrote.
- **For developers:** a schema change is one more step at the end of the file's `STEPS` (or `CONFIG_STEPS` with
  `CONFIG_VERSION` raised by one), never an edit of an old step.

## How Nebula updates

Nebula looks for a newer version only when you choose (`nebula/updates.py`; design decision 2026-10-25: "I only want
that at start-up, and opt-in for 'on-launch' or on demand"). When one is out, a small **Update ready** pill shows in the
page's header; a click shows the version, what is new, and one button. Nothing is ever installed without that click.
On the System page, **Updates** has the choice and **Look now**.

    "updates": {"mode": "demand", "feed": "https://raw.githubusercontent.com/thed1c/nebula-updates/main/latest.json"}

- **mode**, three choices, and no timer beyond the one look at launch:
  - `"demand"`, **Only when I ask** (the default): no request until you press **Look now**.
  - `"launch"`, **Check when Nebula opens**: one look a minute after Nebula starts, none after (and **Look now**).
  - `"off"`, **Off**: never; no network at all, and no **Look now**.

  The first release's `"ask"` (a look every day) reads as `"demand"`: nobody chose to be looked for. No config
  version bump: an older Nebula reading `"demand"` or `"launch"` falls back to its own default.
- **The feed** is `latest.json` with `latest.json.sig` beside it: an Ed25519 signature over the file's exact bytes.
  Nebula carries the public key (`PUBLIC_KEY`) and believes only a feed it signed; an unsigned, badly signed or
  malformed feed is ignored and logged. The last answer is kept beside the config (`NAME.updates.json`).
- **The button.** While the releases are private it is **Get update**: it opens the release page. It becomes
  **Update now** only for a Nebula the one-click installer put in place, and only when the file downloads without
  signing in. Then Nebula downloads it to a temporary folder, checks its sha256 (a mismatch is refused), waits until no
  reply is in flight, and runs the installer quietly as you (Linux: the `.run` with `--quiet`; Windows: `Setup.exe /S`,
  per user, no administrator). Nebula closes and opens again. A Nebula run from a checkout or a pip install is only
  told.
- **What Nebula needs beside itself** follows its pins too: running `python -m nebula.prereqs` again replaces the
  Qdrant Nebula installed when its pinned version moved, and fetches Halo's model again when its file is not the
  pinned sha256 (a Qdrant or a model file of your own is left as it is).

### Publishing a release to the feed

1. Build the installers (`greenfield/package.py`), and attach them to the GitHub release `v<version>`.
2. Sign the feed with the release key (never in the repo): `--key <path>`, or `NEBULA_UPDATE_KEY`, else
   `~/.config/nebula-release/nebula-update-ed25519`:

       python3 greenfield/sign_feed.py --version 0.1.3 --notes "What changed, in plain words" \
         --file linux-x86_64=dist/nebula-eh-0.1.3-linux-x86_64.run=https://github.com/thed1c/Nebula/releases/download/v0.1.3/nebula-eh-0.1.3-linux-x86_64.run \
         --file windows-x86_64=dist/Nebula-EH-Setup-0.1.3.exe=https://github.com/thed1c/Nebula/releases/download/v0.1.3/Nebula-EH-Setup-0.1.3.exe \
         --out ../nebula-updates

   It reads each file's sha256 from the file itself, refuses a key that is not the one Nebula trusts, and writes
   `latest.json` and `latest.json.sig`.
3. Commit both to the public `thed1c/nebula-updates` repository. Each Nebula sees it at its next start ("launch") or its next **Look now**.

A new key (only if the old one is lost: every Nebula already out trusts the old one) is
`python3 greenfield/sign_feed.py --keygen`; put the public key it prints into `PUBLIC_KEY` and ship that release by hand.
