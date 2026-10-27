# Nebula: Event Horizon

**Deploy your own local AI on your own computer. Secure it, share it, scale it with friends, only if you want to.**

**Nebula: Event Horizon** (Nebula EH for short, or just Nebula) is a desktop app that puts AI models on your graphics card for you. You don't need to know what a
"server" or a "quantization" is. You open Nebula, pick a model from a short list we've tested, and start chatting.
Everything runs on your machine, and nothing leaves it unless you say so.

**Deploy your own and join us.**

![Nebula's map: Halo answers on a light model, then one click escalates the same question to a larger model, all on your graphics card](docs/media/nebula.webp)

> **Nebula is in testing.** Things may be rough around the edges. If something goes wrong, tell us (see [Getting help](#getting-help)). That's the whole point of testing.

**New in 0.1.3:** one engine (Impulse) behind everything, Impulse Flash-Next for very large models (manual setup), opt-in long
context up to 1M tokens, several requests at once, and updates that only happen when you say so.

---

## Built for two things: easy and private

### Easy

- **No setup knowledge needed.** No terminals after install, no config files, no jargon. Nebula picks sensible
  settings for your graphics card.
- **Tested models, sized for you.** You choose from a short list we've tried ourselves, not thousands of files.
- **Guided steps for everything.** Adding a model, sharing with a friend or indexing a folder is a wizard with plain
  words, a summary before anything happens, and **Undo** after.
- **Cancel really cancels.** Backing out of a wizard changes nothing.
- **One download, one install.** A small installer brings the rest itself: Nebula, the Impulse engine, Halo's own
  model, Lookup and its document reading. No administrator, nothing to compile.

### Private

Your AI runs on **your** computer. Your chats, your documents and your history stay there.

- **No account, no sign-up, no tracking.** Nebula doesn't phone home. It sends no usage data, no analytics and no
  crash reports. It doesn't even look for updates unless you ask it to.
- **Your conversations stay local.** The Captain's Log, Halo's chats and Lookup's index all live in files on your
  machine.
- **Your documents never go to the cloud.** Lookup reads and searches them on your computer.
- **Locked by default.** A PIN or passkey guards your Nebula, each model runs in a sandbox with no internet access
  (when NVIDIA OpenShell is installed; otherwise Nebula tells you),
  and websites can't send it commands.
- **Nobody can find you.** Nebula never advertises itself on your network or the internet. Friends connect only with
  a one-time invite from you, and they only see the models you tick.

**What does leave your computer**, and only when you do it:

| When | What goes out |
|---|---|
| You install Nebula | Downloads only, each checked against its known fingerprint where its maker publishes one: Python, Nebula's packages (from PyPI), Halo's own model and Lookup's search and document-reading models (from Hugging Face), Lookup's database program (Qdrant), NVIDIA OpenShell, and with an NVIDIA card, CUDA's libraries from NVIDIA. On Windows with the Safe sandbox: Ubuntu for WSL2 and Impulse's Linux build. |
| You download a model | A download from Hugging Face, where the models live. |
| You look for an update | One request to GitHub for a small signed file that names the newest version. Only when you press **Look now**, or once at start-up if you turned that on. |
| You turn on web search | Your search words, to Brave Search or DuckDuckGo. |
| You add a cloud model (Anti-Gravity) | Whatever you send that model, to the company that runs it. Nebula marks these clearly. |
| You share with a friend (Gravity) | Their requests and your model's answers, encrypted. They go directly on your home network, otherwise through a relay you run yourself, or through iroh's free public relays if you switch that on. A relay only passes along traffic it can't read. |

That's the whole list. If none of these are on, Nebula works with no internet at all.

---

## What you need

- A computer running **Linux** (we test on Ubuntu 22.04 and 24.04), or **Windows 10** (21H2 or LTSC 2021 or later)
  or **Windows 11**, 64-bit Intel or AMD.
- An **NVIDIA graphics card** with its driver installed. We test on the RTX 30, 40 and 50 series. Impulse is also
  built for the RTX 20 series and data center cards like the A100 and H100, but we test those less. 8 GB of video
  memory runs the light model, 12 GB the all-rounder and 16 GB the biggest. Without an NVIDIA card, Halo and Lookup
  still run.
- **An internet connection** while you install and while you download your first model.
- About 5 GB of disk space for Nebula itself, and more for models. They are big: a typical one is 5 to 20 GB.
- On Linux, `curl` or `wget` (most Linux desktops have one; if not: `sudo apt install curl`).
- **For the sandbox on Linux (optional):** Docker or Podman, so each model runs in its own NVIDIA OpenShell sandbox.
  The installer tells you the command if it is missing; without it, models run outside a sandbox and Nebula says so.
- **For the sandbox on Windows:** virtualization switched on in your computer's BIOS. Setup does the rest.

**Not yet:** AMD and Intel graphics cards for models (AMD is planned for the next release; Halo alone can already use
a Radeon), and Macs.

## Install

**On Linux:**

1. Download **`nebula-eh-0.1.3-linux-x86_64.run`** from the [Releases page](../../releases).
2. Open a terminal in the folder where it downloaded and run:

   ```bash
   sh nebula-eh-0.1.3-linux-x86_64.run
   ```

   You don't need `sudo`. Nebula installs just for you, in your home folder. It fetches the rest as it goes (Python,
   its packages, Halo's model, Lookup's models, OpenShell) and tells you each step; it takes a few minutes. If
   something needs your computer's administrator (Docker, the NVIDIA driver), it says exactly what to run.
3. Nebula opens by itself. Next time, open **Nebula EH** from your app menu.

**On Windows:**

1. Download **`Nebula-EH-Setup-0.1.3.exe`** from the [Releases page](../../releases).
2. Double-click it. It installs just for you and fetches the rest as it goes; it takes a few minutes.
3. Setup asks **"Run your models in a Safe sandbox?"** Say **Yes** (it's the default). Each model then runs in its
   own container inside its own small Linux (Ubuntu 24.04 under WSL2), just like on Linux. If WSL2 isn't on yet,
   Windows asks for an administrator once and may need one restart; the sandbox then finishes by itself the next time
   you sign in. It downloads about 350 MB (Ubuntu plus Impulse's Linux build).
4. Open **Nebula EH** from the Start menu or the desktop icon.

Uninstalling Nebula also removes its sandbox. WSL itself, your models and your settings stay.

That's it.

## Your first five minutes

1. **Make a PIN or a passkey.** The first time Nebula opens, it asks you to pick a PIN (4 to 12 digits) or use a
   passkey (your fingerprint, face, phone or security key). This keeps other people, and websites, from using
   your Nebula.
2. **Get your first model.** Nebula shows a short list of models we've tested, sized for your graphics card. Each
   one says whether it is **everyday** (quick, light) or **larger** (smarter, slower, needs more memory), and
   whether it is **Standard** or **Uncensored** (Standard is picked for you unless you change it). Pick one and
   press go. Nebula downloads it and starts it on the
   Impulse engine.
3. **Say hello.** Nebula sends a test message to make sure everything works, then you're in. Click
   **Halo** to chat.

If you change your mind about anything a wizard did, there's an **Undo**. Pressing **Cancel** changes nothing.

## Finding your way around

Nebula draws your setup as a little space map. Everything on it is something you can click.

| On the map | What it is |
|---|---|
| **The black hole** | Your Nebula, in the middle. It glows amber when your graphics card runs hot (red when very hot); hover for each card. Right-click it for the quick menu, including any models you've hidden. |
| **Model stars** | The AI models you've set up. |
| **Halo** (the star with rings) | Where you chat. Its ring is a guard that watches for trouble and is on by default (see below). You can rename it. |
| **Lookup** (the **Dyson Sphere**, a swarm of silver satellites) | Lets Halo read your own documents and search the web. |
| **API Server** (the **Quasars**) | Doors for other apps on your computer or home network to use your models. |
| **Gravity** | The door to friends' Nebulas, over the internet. Off until you turn it on. |
| **Anti-Gravity** (stars in iron chains) | Cloud models like the ones you pay a company for. Optional. They're chained because they can never be shared. |
| **Friends' systems** | Each friend you've joined, with their models round them. Ones they share with you are free, with a route. Ones they don't are chained. |

**Mouse basics:** drag to move things, scroll to zoom, right-click a star for its menu.

Along the top are the tabs:

| Tab | What's in it |
|---|---|
| **Navigation** | Who's asking (your apps, your friends) and which models each one may use. |
| **Engineering** | Your engines and models, plus **System** (errors, sandboxes, sign-in). |
| **Captain's Log** | Every conversation and every choice Nebula made, to look back on. |
| **Gravity** | Sharing with friends and your own other machines. |
| **Widgets** | Show or hide the panels on the map. |

### Traffic control

At the heart of Nebula is its traffic control, the event horizon every message passes through. When a message comes
in, Nebula picks which model answers, keeps a queue when a model is busy, and writes every conversation and choice in
the **Captain's Log**, so you can look back at what happened. You steer it from **Navigation** and tune your engines
in **Engineering**.

**Smart routing** is one plain switch:

- **Off**: you always get the model you picked.
- **Practice**: Nebula shows what it *would* have picked, but doesn't change anything. Good for getting a feel.
- **On**: **Halo** reads each message and sends it down a route: **everyday** (quick questions and chat), **larger**
  (code, analysis, step-by-step work), **writing** (stories, scenes, poems) or **critique** (feedback on your
  writing). Every reply says which model answered and why. You choose which models are on each route, and in what
  order. Cloud models are only used when you ask for one by name.

### Halo

**Halo** is where you chat. (That's its name out of the box; you can change it.) Its rings keep watch when you
share your models:

- a soft **glow** means all is calm,
- a **pulse** means it's looking at something,
- **orange** means trouble is brewing,
- **red** means it stepped in (like slowing down a guest who was misbehaving) or wants to ask you.

It only takes small steps that you can undo with one button. Anything bigger, it asks you first. It never touches
your own use.

### Lookup

Give Lookup a folder (**+ → New Lookup collection**) and Halo can search your documents and tell you which
file and page each answer came from. Your documents stay on your computer. Lookup can also do web searches, if you
switch that on.

### Using Nebula from other apps

Most AI apps can talk to Nebula instead of a cloud service. Click the **API Server** to copy the address (for apps
on this computer it's `http://localhost:8090/v1`). It works with apps that speak the OpenAI, Anthropic or Ollama
style. To reach it from a phone or TV on your home network, add an entry point there. Each one gets its own key.

## Sharing with friends (Gravity)

This is the "join us" part.

> **Gravity is still in testing.** It works, and it's encrypted end to end, but it has had far less real-world use
> than the rest of Nebula. Share with people you trust while we shake it out.

![Gravity, two Nebulas side by side: you share your Novel Writer from the Gravity tab, a friend opens the one-time invite and joins your constellation, then escalates a story request to your Novel Writer across Gravity](docs/media/gravity.webp)

When you're ready:

1. Open **Gravity** and choose **Share with a friend**.
2. Tick the models your friend may use.
3. Send them the one-time invite. It works once, then expires.

Good to know:

- **A friend gets nothing until you tick it.** They can see the names of your models, but only the ones you
  ticked are usable. The rest show up chained on their map. They never see your chats, files or settings.
- **You choose per friend.** Each friend gets their own ticks, so you can share a model with one friend and not
  another.
- **Invites are one to one.** Each invite joins one friend to you. If two of your friends want to share with each
  other, one of them invites the other.
- **Nobody finds you by accident.** Nebula never advertises itself. Only someone with your invite can connect.
- **Cloud models are never shared.** They're on your bill and your account, so their tick is greyed out.
- Some models come with a warning (for example, one Nebula can't lock down). You can still share them, but you're
  asked to confirm, and your friend sees the warning too.

![Three Nebulas: Friend A invites a third friend, Friend C. C has the same Coder model as A, so C shares it with Friend B but not with A. On A's map C's Coder is chained; on B's map the same Coder is free, with a route](docs/media/gravity-three.webp)

You can also join your own machines this way, like a second PC with a spare graphics card, and run them all from
one Nebula.

## Local, isolated, containerized

Those three words are how Nebula is built.

**Local.** Your models run on your own graphics cards. Your chats, your documents, your settings and your keys are
files on your own computer. Once a model is downloaded, Nebula works with the internet unplugged.

**Isolated.** Every model runs in its own locked box (where a box isn't possible, Nebula tells you; see
[Safe by default](#safe-by-default)), and the box gets nothing it doesn't need:

- no network at all, so a model can't send anything anywhere,
- the system and the model files are read-only, so it can't change them,
- one small scratch folder of its own, and it never runs as the administrator (root).

People are kept apart too. Each friend gets their own key and their own ticks, sees only what you ticked, and never
sees your chats, files or settings. Cloud models never leave your account.

**Containerized.** Each box is a container, run by NVIDIA OpenShell with Docker or Podman (on Windows, Podman inside
Nebula's own small Linux under WSL2). Impulse's container is
built on your computer from the Impulse that came with Nebula, llama.cpp's comes from its makers' official image, and
any engine you already have as a container image runs the same way. Stopping a model deletes its box. **Add a
server** puts the same boxes on your other computers.

## Safe by default

You don't have to set any of this up. It's already on.

- **Sign-in:** a PIN or passkey protects your Nebula's page.
- **Sandboxed engines:** Nebula runs each model inside a locked box (NVIDIA OpenShell) with no internet access and
  read-only files. If OpenShell isn't on your computer, the model still runs, and Nebula shows you a warning. If you
  share that model, your friend sees the warning too.
- **Models on your other computers.** **Add a server** puts another Linux computer with NVIDIA cards to work for your
  Nebula: it sets up only what runs models there (no second Nebula), asks for that computer's password once in the
  page for the parts that need an administrator, and then its cards show up in **Engines** next to your own. Its
  models run in the same locked boxes, and everything between the two computers is encrypted.
- **Any engine in a box.** Besides Impulse, Nebula can run llama.cpp's own engine, or any engine you have in a
  container image, the same protected way.
- **No key needed on this computer.** Apps on your own PC can use your models straight away. On your home network,
  Nebula strongly encourages a key and warns you if you skip it.
- **Websites can't drive it.** Nebula only takes commands from its own page.
- **Safe updates.** Before an update changes your saved settings or history, Nebula makes a backup copy.

Where it's less locked down, Nebula tells you, and here it is in one place:

- **No sandbox on Linux without Docker or Podman, or on Windows if you said No** (or a setup step failed). The model
  still runs, as a normal program on your computer, with a warning in Nebula (and on your friend's side if you share
  it). **Engineering → System** says why.
- **On Windows, the container is the whole sandbox.** Models are read-only to it and its port is reachable from this
  PC only, but OpenShell's network rules don't run on Windows yet, so the "no network at all" rule isn't enforced
  there.
- **Your home network uses plain HTTP**, like Ollama does. A key per device stops strangers using it, but anyone on
  your network who can watch the traffic could read it.
- **Friends don't get Impulse's prompt memory.** To keep guests apart, a friend's request on an Impulse model starts
  fresh every time. That's safe, but long conversations from friends are slower than your own.

## Updates

Nebula never updates itself behind your back. In **Engineering → System → Updates** you pick one of three:

- **Only when I ask** (the default): Nebula never looks until you press **Look now**.
- **Check when Nebula opens**: one look, a minute after it starts, and none after that.
- **Off**: never, and the **Look now** button goes away.

When there is a new version, a small **Update ready** shows at the top. Nebula only believes an update notice that
we signed, and checks every download against its fingerprint. Nothing installs until you click. While the project's
releases are private, the button just opens the release page for you.

## Impulse: one engine

Nebula comes with **Impulse 0.1.1**, our own engine, built in. It is now the one engine behind everything Nebula runs
on an NVIDIA card: every tested model, Halo and the big setups below all run on the same Impulse. Before we moved our
own setups over to it, we checked that each one gives exactly the same answers as the engine it replaced.

**How fast?** On the same graphics card, in our tests, Impulse reads long prompts 2 to 7 times faster than llama.cpp
and writes answers 1.1 to 1.8 times faster. To be fair about it: we ran llama.cpp with its own new expert cache
switched off, so today the gap may be smaller. We haven't re-checked that yet.

**Other engines.** llama.cpp's own server is still there as the reference: it runs any GGUF model you bring, and
Nebula uses it by itself if Impulse can't run on your card. Any engine you have in a container image works too, from
the **Engineering** tab.

## Advanced: big setups, CPU and memory

Everything about your computer's CPU and memory (RAM) lives here, because you don't need it for the easy path. These
are for people with a lot of memory and more than one graphics card. They are not in the easy model list.

- **Impulse Flash-Next** runs very large expert models (Qwen3.8 Flash-Next packs) split across your graphics cards
  and your computer's memory. **In 0.1.3 it's a manual setup only**, with no one-click wizard yet (that's planned for
  0.1.4). You bring the model pack, build its container image (see
  [impulse/flashnext/nebula-image/README.md](impulse/flashnext/nebula-image/README.md)), and define the engine
  yourself in **Engineering**.
  - **What it needs:** Linux with Docker or Podman, NVIDIA cards, a CPU with AVX2 (AVX-512 makes it faster), and a
    lot of RAM: the model's experts alone take about 64 GB.
  - **What we run it on:** two NVIDIA RTX 50 series cards (12 GB and 16 GB) and two server CPUs. That gives about
    100 tokens a second writing and about 2,000 a second reading. Smaller machines haven't been tried.
- **Long context** (Flash-Next only) is an opt-in, **off by default**. In **Engineering**, a Flash-Next model can be set to
  512K or 1M tokens instead of 262K. Nebula only stretches the model once a conversation grows past 262K, and it found
  hidden test words at every length we tried up to 1M. It costs some speed on every request, even short ones. In our
  tests, 512K read about 5% slower and wrote 1% to 8% slower; 1M read about 25% slower and wrote about 15% slower.
  Answers under 262K stay effectively the same, but not bit-for-bit, because the bigger memory leaves a smaller expert
  cache and a little more work moves to the CPU.
- **Several requests at once** (Flash-Next) is an opt-in, **off by default** (one at a time). You can turn on up to 8 at once.
  Honestly: in our tests two at once gave less total speed than one at a time (10% to 15% less), and the answers aren't
  yet bit-for-bit the same as one at a time. Each extra slot also takes video memory from the expert cache.
- **Several slots on the regular models.** Impulse's **Parallel slots** setting works, but when several requests (or
  one request with drafting) land together, part of each step skips the graphics card's expert cache and runs on the
  CPU. More slots can end up slower. We haven't measured by how much yet.

## Known limits

We'd rather you hear it from us:

- **NVIDIA only** for models in this release. Mac isn't supported.
- **In the code, but switched off and never tried between two real computers:** sending expert work to another
  computer over the network, splitting one model's layers across two computers, and the fast RDMA network link.
  They're there for the future; Nebula doesn't use them.
- **Tested models only, out of the box.** The short list is what we've checked. You can add other GGUF models
  yourself, and llama.cpp runs any of them, but we haven't tried them.

## Getting help

- Something broke? Open **Engineering → System**. It has an error log you can copy.
- Tell us in an [issue](../../issues), with what you did and what you saw. A screenshot helps.

## For developers

Building from source, the config file, the full API and how the pieces fit together are in
[docs/DEVELOPING.md](docs/DEVELOPING.md) and the rest of [docs/](docs/).

## Licence

MIT, see [LICENSE](LICENSE). Nebula and Impulse include code from other open-source projects, each under its own
licence; see [NOTICE](NOTICE).

## Buy me some compute

Nebula is free and open source, and it always will be. If it's useful to you, you can buy me some compute.

<p align="center"><a href="https://buymeacoffee.com/thed1c"><img src="https://img.shields.io/badge/Buy%20me%20some%20compute-FFDD00?style=for-the-badge&amp;logo=buymeacoffee&amp;logoColor=black" alt="Buy me some compute" height="50"></a></p>
