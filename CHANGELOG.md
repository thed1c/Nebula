# Changelog

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
