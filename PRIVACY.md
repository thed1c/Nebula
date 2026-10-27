# Privacy: what leaves your machine

Nebula runs your models on your own hardware. Your prompts, replies, files and history stay on your machine. **Nebula
has no telemetry:** it sends no usage data, crash reports or analytics anywhere. ("Telemetry" in the code is the
page's local hardware monitor, read and shown on this machine only.)

Nebula reaches the internet only for these, and only when they happen:

| When | What is fetched or sent | To |
|---|---|---|
| Installing, and `python -m nebula.prereqs` | Downloads only: Python and Nebula's packages, CUDA's libraries, Qdrant, OpenShell, Docker images, Halo's and Lookup's models | PyPI, NVIDIA, GitHub, Docker registries, Hugging Face |
| You add or download a model | The model's files (downloads only) | Hugging Face, or the address you give |
| The update check, **only** on demand (**Look now**) or at launch if you chose "on launch" (default: only when you ask) | One request for the signed feed (`latest.json` and its signature); the installer when you click **Update** | GitHub (raw.githubusercontent.com, github.com) |
| You use a cloud model (Anti-Gravity) | That conversation, to the provider you set up | Your provider, on your API key and their terms |
| You pair with another Nebula through a relay (optional) | End-to-end encrypted traffic between the two Nebulas; the relay cannot read it | Your own gateway, or iroh's public relays only if you turn on "public relay" |
| You turn on OpenTelemetry (optional, off by default) | Metadata only (model, token counts, timings), never a prompt or reply | The collector you name |

Nothing else is sent. Pairing finds no one on the network by itself (no mDNS, DNS discovery or DHT): another Nebula is
reached only through an invite you made. Turning OpenTelemetry on or off, or changing its collector, is written to the
security log (System page).

See [SECURITY.md](SECURITY.md) for how to report a security problem.
