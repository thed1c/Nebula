# Changelog

## 0.2.0.1

**Windows: no more console windows flashing.** Nebula's server runs without a console on Windows, so each program it
started (the graphics card readings, WSL and Podman, OpenShell's sandbox list, the toolbox's commands, a skill's script,
Codex) opened a console window of its own, open while the program ran; in 0.2.0 that was one every few seconds. Every
program Nebula starts on Windows now runs hidden, as the engines already did. Nothing changes on Linux.

## 0.2.0

**Star Weaving: your models as one team.** Ask once and Halo picks the model that leads the task. With Teamwork on,
the lead may hand parts of the work to its team, ask a teammate, hand the work up to a larger model, and send it for
review and rework; you get one answer back, with a team line saying who did what. Halo referees each move (a model
that is down, too small or past the limits is refused, with the reason) and asks you when a team has a question only
you can answer. Halo suggests teams from your models; each team has a contract (who leads, helps and reviews, what
"done" means, its limits) and a card on the Star Weaving page with its missions. Every task has limits you can change:
teammates at once, depth, tokens, 20 minutes in a chat or 2 hours in the background, rework rounds, and US $1 on cloud
models, held per piece of work. A task can start from the chat ("Halo, I want to do this task"). Off by default. A
restart marks running team work stopped instead of losing it, and closing the window leaves your engines running.

**Skills.** A skill (a folder with a SKILL.md) from a link or from a skills repository: checked before it is written
(its name, what it asks to run, prompt-injection and data-leak patterns), switched off when the check warns, used only
once you turn it on and only for a task it fits, its scripts run in a sandbox. Skills can be attached to routes, and
updates are checked again before anything is replaced.

**The toolbox.** A model can work in one folder you pick for it (Models page, per model): list, search, read, write
and patch files there, and run commands inside an OpenShell sandbox with no network. Every path is checked against the
folder (links and swapped folders included); each call is in the ledger. Teamwork, skills and the toolbox are the
owner's only: never offered to a guest, a friend, another Nebula, or a plain API call.

**A new layout.** Four tabs: Navigation (Star Calling, Star Weaving, Gravity), Captain's Log, Engineering (Engines,
Models, Skills) and Control (Halo, Appearance, System). Star Calling shows how a request finds its model: a row per
route, models left to right, routes of your own with a kind each (thinking, pictures, tools, code), Any task first,
and smart routing as Off or On with a profile (Just watch, Halo picks, Save power, Balanced, Best answers, Custom).
Engines is machines and engines; Models is the files and cloud models. Panels are widgets you can resize, fold and
move. Gravity's page has Constellation (the gateway) on top and your friends below, each with a Block. Errors,
OpenTelemetry and a new syslog forwarder (off by default; UDP, TCP or TLS) are on Captain's Log. The map rests when
nothing happens, with an Effects menu (animations full, calm or off, and each effect), and the window at rest uses a
fraction of the CPU it did. Text is readable on every page; helper text shows on hover.

**Halo.** Halo is the front door: it starts team tasks from the chat, and a chat opens only for an alert or a
question. Halo's own page holds its personality, instructions, version and Lookup. Halo can run Liquid AI's
LFM3-Omni-2B if you pick it (Halo > Version > Omni, with an NVIDIA card): the model and Nebula's adapters for it
(about 30 MB) download the first time.

**An older and a newer NVIDIA card together.** Impulse (L) now carries kernels for every card from Maxwell to
Blackwell (CUDA 12.9), so on a PC with, say, a GTX 1080 Ti beside an RTX 3070, Run a model asks which graphics card:
Nebula's pick (the newer card, on Impulse's main build), the older card, or all of them on Impulse (L). The Engines page
can move a model between them and offers the download too (about 588 MB on Linux, 704 MB on Windows; never in the
installer). Impulse (L) is now offered on any PC with an NVIDIA card.

**Gravity.** A friend sees each of your models by its own name, and a friend's calls show as pulses through the
wormhole. Peers get a random two-word name, new each time they come online; away peers leave the map and say why;
dead dials stop. A friend's calls reach only the model API (/v1/...). A model on an engine Nebula deployed to one of
your own servers shares without the care warning.

**Fixes and safety.** Lookup reads only public web addresses (redirects and DNS answers checked); a bad Content-Length
is refused at once; the error log hears every part of Nebula and folds repeats; the Chats box shows your own chats
only (API calls stay in the ledger); a plain API call is never offered Nebula's own tools; the engine monitor asks
every engine at once, so one that does not answer no longer holds the page. New: "No PIN on this computer" (System > Sign-in):
the page opens without a PIN on this computer only; other devices still sign in (off by default).

Not in this release: the LFM2.5 8B fix for experts kept in RAM (it changes Impulse itself, so it comes with the next
full release); teams across Gravity.

## 0.1.7

**Impulse (L): an optional engine for older NVIDIA cards** (Maxwell, Pascal, Volta: GTX 900 / 10 series, Titan X / Xp /
V, Tesla M / P / V100), downloaded only on a machine that has one. Impulse's main build needs an RTX 20 or newer (the
CUDA version it is built with dropped the older cards), so such a card was shown as CPU only. Now Run a model offers
Impulse (L) on a computer whose NVIDIA cards are all older ones: it says what it is and how big (about 471 MB on Linux,
586 MB on Windows) and downloads only when you go ahead; it is never in the installer. The model then runs on that
card, and Halo uses it too once it is there. It needs NVIDIA driver 525.60.13 or newer (528.33 on Windows) and says so
when the driver is older. One model runs on cards of one kind: an older card and an RTX card together are refused, with
the reason. Older cards still (Kepler: GTX 600 / 700, Tesla K80) stay on the CPU, and Flash-Next stays on RTX 30 and
newer. On Windows the Safe sandbox carries Impulse's main build only, so a model on Impulse (L) runs outside the
sandbox, and Nebula says so.

## 0.1.6.3

**LFM2.5 8B in the model list.** Liquid AI's LFM2.5-8B-A1B, the smallest and quickest model on offer, is now in
first run and Deploy your own: Liquid AI's own Q6_K file (7.0 GB, pinned and checked against its hash), for a 6 GB
card and 16 GB of RAM, and offered to friends. Uncensored picks gaston-parravicini's abliterated build of the same
model (Q6_K, pinned and hash-checked). Impulse runs it on the same engine as LFM2.5 24B. On a computer too small for
Gemma 4 26B and LFM2.5 24B it is the model Nebula recommends. Text only.

**Windows: updates are small.** Setup used to delete Nebula's whole Python folder every time and download Python and
every package again (the CUDA libraries and torch among them), so a small fix was a full reinstall. Setup now keeps
Python and Nebula's packages when Python's version did not change, and pip fetches only the packages whose version
moved. If they do not install over the Python already there, Setup starts once more with a fresh Python. The Safe
sandbox keeps its Impulse image when Impulse did not change, instead of downloading Impulse for Linux (about 500 MB) and
building the image again, and Setup no longer asks the sandbox question again once a sandbox is in place. The release
feed now carries Windows' small update (Nebula's own code alone) as it does Linux's, and the feed signer refuses a release
that leaves a platform's small update out without saying so.

**Windows: the update shows in front.** The Update button's Setup used to run hidden: Nebula closed and nothing
showed until it opened again. Setup now opens in front of the other windows with no pages to click, shows what it is
doing, and closes by itself when it is done; then Nebula opens again. A small update shows a small "Updating Nebula EH"
window of its own until Nebula opens again. An update never asks the Safe sandbox question: the choice made at the
install stands.

**Model pickers work again.** Picking a model in the Assistant's picker, a Star's route or Halo's own picker failed
with "no tier named '…'": the picker's value carried a character the page could not keep. It carries the tier and the
model as plain text now.

## 0.1.6.2

**Windows: Setup closes Nebula first.** Setup over a Nebula that was still open left part of its Python in place (the
files Nebula had open), unpacked a new Python over them and failed while installing Nebula's packages, with a message
about the internet connection. Setup now closes Nebula and the programs it started (found by where they are installed,
so other programs are left alone) before it changes anything. If a Nebula it cannot close is still running (one started
as administrator), Setup stops at once, before changing anything, and says so. When files in use stop an install, the
message says that instead of blaming the connection. Setup's "Open Nebula EH" now opens Nebula as you even when Setup
itself ran as administrator, and the uninstaller closes Nebula the same way.

**Windows: a Nebula once opened as administrator no longer locks you out of your own settings.** Nebula kept its keys
and a few other folders readable by their owner only; a Nebula opened as administrator (Setup's "Open Nebula EH" after
running Setup as administrator) made the Administrators group their owner, and the Nebula you open normally could not
read them afterwards (Halo, for one, did not start). These folders are now plain folders in your own profile, which is
yours alone already. A folder an earlier run locked stays locked until it is given back (see the release notes).

**Older NVIDIA cards are said as such.** Impulse's engines need a card of compute capability 7.5 or newer (RTX 20,
GTX 16 and newer). An older card (a GTX 10, for one) is now shown as "older than this engine build: CPU only", is never
picked for an engine, Halo or a guided setup, and an engine set to it stops at once with that reason instead of a
generic failure. Models still run on the CPU there. An engine build for older cards is planned as an optional download.

**LFM2.5 8B in the model list.** Liquid AI's LFM2.5-8B-A1B, the smallest and quickest model on offer, is now in
first run and Deploy your own: Liquid AI's own Q6_K file (7.0 GB, pinned and checked against its hash), for a 6 GB
card and 16 GB of RAM, and offered to friends. Impulse runs it on the same engine as LFM2.5 24B. On a computer too
small for Gemma 4 26B and LFM2.5 24B it is the model Nebula recommends. Text only, standard build only.

**Add a server: engines start on machines with Podman 5.** On a model machine whose Podman is version 5 (Ubuntu 25.10
and later), an engine's sandbox failed at once: NVIDIA's description of the cards for containers (CDI) carries the
video and render groups, Podman 5 hands them to the container, and OpenShell then refuses the sandbox ("supplementary
groups ... do not match"). Podman 4 never read those groups, which is why only newer machines hit it. Adding a server
now gives that user's Podman its own copy of the card description without the groups, and writes it again whenever
NVIDIA's own description changes (a driver update, a card moved). Removing the server removes it too.

## 0.1.6.1

**Windows: the Update button works.** On Windows, Nebula's Update closed Nebula, but its helper stopped before putting
the new version in place, so Nebula did not come back. The helper now runs to the end and reopens Nebula. Windows
installs of 0.1.5 and 0.1.6 are never offered an update through the button (theirs would close Nebula for nothing): it
opens the release page, and their Setup installs 0.1.6.1. From 0.1.6.1 on, Windows updates come through the button.

## 0.1.6

**Windows: Nebula's window opens again.** On Windows, 0.1.5 closed at once, before its window appeared: the window's
icon was an SVG, and Windows' window library reads only .ico files. Its window now uses the .ico. If you have 0.1.5 on
Windows, install 0.1.6 with its Setup (Nebula's own Update button is in the window that did not open).

**Downloads say why they failed, and Setup moves on.** When a model cannot be downloaded, Setup and Nebula now say why
in a few plain words: no internet, a DNS filter or firewall that blocks Hugging Face, a proxy, an antivirus that
inspects HTTPS, or a server that is busy. When Hugging Face cannot be reached at all, Setup says so once and goes on in
seconds instead of retrying for minutes; Nebula fetches Halo's and Lookup's models by itself later. A slow or unsteady
connection that keeps moving is never given up on. On Windows and macOS, downloads trust the certificates the system
trusts, as the installer's pip does. When one of Setup's downloads fails, its reason and the full error are kept in
`prereqs-errors.log` in Nebula's settings folder (replaced on each run, and removed once every step goes through).

**Instructions for each route.** The Routes tab has a box for each route (Everyday, Larger, Writing, Critique). A
request sent to a route gets that route's instructions at the top of its system prompt, before an agent's and the
app's own, which are kept. The route is the one a rule or smart routing sent it to, else the answering model's own.
Callers in passthrough mode get none. Each reply's "why?" says which route's instructions it carried.

The small update: a 0.1.5 install takes 0.1.6 as Nebula's code alone (Impulse and the rest stay as they are).

## 0.1.5

**Halo upgraded to LFM3.** With an NVIDIA card, Halo, Nebula's helper model, now runs Liquid AI's LFM3-1.7B with new
adapters (Halo v3.1): a better router judge, handoff briefings that invent nothing, and the Uncensored option, at about
1.5x the time per call of Halo v2. Its licence (LFM Open License v1.0) ships with Nebula.
Without an NVIDIA card (a Radeon through Vulkan, or the CPU), Halo stays on v2 (LFM2.5-1.2B) until LFM3 runs there.
Each machine downloads only the model it runs (1.0 GB for LFM3, 0.7 GB for LFM2.5) and says which and why. After an
update, Halo v2 keeps running while LFM3 downloads, then Halo moves to it; if the download fails, Halo v2 stays.
Impulse runs LFM3 natively (its sliding-window layers in a cache of their own), and its server takes llama-server's
`--reasoning` and `--chat-template-kwargs`.

## 0.1.4

**Smaller downloads.** The installers no longer carry llama-server: Halo, Nebula's helper model, runs on Impulse on
every machine now, Windows included. A machine without an NVIDIA card gets a small Impulse portable (CPU + Vulkan,
about 25 MB) the first time Halo needs it. Nebula says what it downloads, how big, and why: to keep the downloads as
small as possible. Each such piece is pinned and checked against its checksum.

**AMD Radeon and other cards.** Without an NVIDIA card, models now run on Impulse portable through Vulkan: a Radeon
(or an Intel or older NVIDIA card) takes as much of the model as fits, the rest stays in RAM, and Run a model says
which before anything downloads. Flint runs there too. They run directly on the computer, not in a sandbox (a sandbox
does not reach the graphics card), and the preview says so. Qwen3.8 Flash-Next still needs an NVIDIA card, and says so.

**Small updates.** When only Nebula's own code changed, the Update button downloads just that (a few MB) instead of
the whole installer.

**Qwen3.8 Flash-Next as a guided setup.** Run a model › Qwen3.8 Flash-Next asks for a Hugging Face repository,
checks that it really is a Flash-Next model, shows its sizes and which of them fit this computer, and runs it with
Impulse in a sandbox. Impulse's Flash-Next engine is a separate download (about 105 MB), fetched only when you set one
up, and the setup says so first. An existing Flash-Next pack on disk works too (Advanced). Guided setups now give
their model a graphics card (the one with the most free memory, unless you pick).

**Flint, a coding model, in the model list.** Recursive's Flint 40B (Apache 2.0) writes and fixes code, calls tools and
thinks before it answers when it needs to. Two sizes: **Q4** (25 GB, about 27 GB of free RAM for its experts) and **Q6**
(36 GB, about 38 GB of free RAM; the closest to the full model, the best for coding). It starts with a 32K context
(64K on its Edit form, about 3 GB more RAM). Both run on an 8 GB card and up;
Nebula says how much RAM it needs before it downloads, and will not start it when the RAM is not there. A computer
with two cards that hold all of Q4 gets it all on the cards, about twice as fast. Each also comes as an **uncensored**
build (dima6312's abliterated Flint) under Uncensored. Its own sampling (temperature 1.0, top-p 0.95, top-k 20) is set
for you, and each download is checked against the file we tested.

**Answers appear as they are written.** With Impulse, a streamed answer used to arrive all at once at the end; now the
thinking and the answer show word by word (Flint's thinking starts in under half a second), and stopping a reply frees
the model at once.

**WSL: let Nebula use more memory.** Windows gives WSL half of the PC's RAM. When a model does not fit in that half,
Nebula offers to raise it (in .wslconfig, backed up first, only its memory line, with Undo); restarting WSL for it
asks its own yes.

**Fixes**
- Conversations: "Delete all conversations" deletes the open ones too, not only the closed ones.
- Editing a model: Save, Save and restart, and Save as a copy (not only "Save as a new deployment").
- The OpenShell panel no longer breaks when OpenShell lists its sandboxes in a different shape.
- Gravity: a peer whose address changes is still found; the invite says how a friend reaches you and that the public
  relay is a tick both sides need; a join that fails names the relay.
- A Flash-Next preview says plainly when it will run directly on this computer, not in a sandbox (no sandbox there).

## 0.1.3.1

**Fixes**
- Old conversations can be deleted: one at a time with the ✕ beside it, or all the closed ones at once. Nebula asks
  first, and keeps a conversation whose reply is still on the way.
- Gravity: the ticks (Use the public relay, and the models a peer may use) light up with a check mark when they are on.
- Gravity: Invite a friend and Invite one of my machines have a Copy invite button that copies just the invite.

## 0.1.3

**One engine for every model.** Impulse now runs every lane: the Flash-Next packs and the GGUF models start from the same
engine, so there is one program to install and update.

**Long context, if you want it (off by default).** A Flash-Next model can now read up to 512K or 1M tokens of context
instead of 256K. Turn it on per model under **Long context: off / 512K / 1M**. Up to 256K nothing changes at all; past
it the model stretches its position encoding as the conversation grows (dynamic YaRN). Each request pays for it with a
somewhat smaller expert cache, so answers can come a little slower and differ slightly from the 256K setting. In our
tests a model at 512K found a passkey hidden in a 300,000-token document and wrote at about the same speed.

**More than one request at once, if you want it (off by default).** A model can answer several people at the same time
(**parallel** per model) instead of one after another. The second person no longer waits for the first; total speed
stays about the same or drops a little, depending on the model and the cards. A request that runs alone keeps its full
speed, including speculative decoding (MTP); requests that share the engine decode one token at a time. With long context
on a two-card setup, Nebula picks the engine mode that supports it by itself. Answers are consistent but not always
identical, character for character, to the same request run alone.

**Updates only when you ask.** Nebula can look for a new version and tell you on the page ("Update ready"); it is
opt-in, there is no daily timer, and the default is "Only when I ask". Update feeds are signed.

**Fixes**
- After a restart, Nebula takes back a model it was already running in a sandbox on another machine instead of showing it
  as stopped.
- Halo's router sends its own key on every call, so Halo answers only Nebula.

**Experimental (not tested on real hardware yet)**
- Spreading one model's experts over two machines over the network (TCP, with encryption and authentication) and an
  RDMA transport for faster links. They are built and tested on one machine only; leave them off unless you want to try
  them.
