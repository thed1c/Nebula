# Wizards

Guided steps for what you add to Nebula. They use plain words and fill in their defaults, with Back and Next, a summary
before anything changes, and Undo afterwards. Open them from the **+** menu at the top of the page, or from the empty
place they fill.

| Wizard | Offered from | What it does |
|---|---|---|
| Your first model | the map, while there is no model | downloads a small starter model sized to your GPU and runs it here, or uses a server already running here, or a cloud API; then sends it a test message |
| Deploy your own model | Engines (empty), Models | a Hugging Face link (or a file here), its size, the engine and its settings; downloads, starts, test message |
| Add a model server (SSH) | Models | another machine on your network: confirm its host key, install Nebula there, join and pair it |
| Share with a friend | Gravity (no peers) | the models a friend may use; a one-time invite that gives them on joining; never a cloud model, care warnings to confirm |
| Deploy a gateway | Gravity (no gateway) | your own relay, on this machine or with the commands for a server |
| New Lookup collection | Lookup (no collections) | a folder to index, kept current, shared with no one unless you say |

The API Server's entry points get their own wizard with the map's Quasars.

## How they keep their promises

- **Cancel changes nothing.** Every step up to the summary only reads: its choices (a link's sizes, a machine's host
  keys) and the summary itself (`POST /api/wizards/<name>/plan`). The tests check the config file is byte for byte
  the same afterwards.
- **Apply is a job** (`POST /api/wizards/<name>/apply`, then `GET /api/wizards/jobs/<id>`). It does its work through
  this Nebula's own page API, with the same checks and the same security log, and remembers how to take back each
  step.
- **A step that fails takes back the ones before it**, so a wizard that does not finish leaves nothing behind (a
  half-finished download included).
- **Undo** (`POST /api/wizards/jobs/<id>/undo`) takes back everything a finished wizard did. A model you downloaded is
  kept: it is your file, and the summary says where it went.
