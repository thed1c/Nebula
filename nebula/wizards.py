"""nebula/wizards.py: the guided flows (docs/wizards.md). Plain words, defaults filled in, back and next, a summary
before anything changes, and an Undo after.

Each wizard is steps of fields (the page draws them), then:

    POST /api/wizards/<name>/options {field, answers}   choices that depend on earlier answers (a link's quants, …)
    POST /api/wizards/<name>/plan    {answers}          checks them and says what will happen; changes NOTHING
    POST /api/wizards/<name>/apply   {answers}          does it, in the background: a job with its steps
    GET  /api/wizards/jobs/<id>                         the job: running | done | failed, its steps, its result
    POST /api/wizards/jobs/<id>/undo                    takes back everything it did

A job runs its work as calls to this Nebula's own page API (the same checks, the same security log), and remembers
how to take back each one. A step that fails takes back the ones before it: a wizard that does not finish leaves
nothing behind. Cancel, before Apply, has nothing to take back.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time


class WizardError(ValueError):
    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status, self.extra = status, extra


def _call(server, path: str, req: dict | None = None, get: bool = False) -> dict:
    """This Nebula's own page API, as the page would call it (its checks; the security log sees the write)."""
    from nebula import seclog, ui
    from nebula.dialects import Context, JsonReply
    if get:
        status, _ctype, data = ui.get(server, path, "", {})
        body = json.loads(data or b"{}")
        if status >= 400:
            raise WizardError(status, str(body.get("error") or body))
        return body
    ctx = Context(server.cfg, server.ledger, {}, server.engines, server.tools, registry=server.registry,
                  admission=server.admission, lot=server.lot, policy=server.policy, yard=server.yard, power=server.power,
                  slots=server.slots, seclog=server.seclog, guard=server.guard, otel=server.otel)
    try:
        reply = ui.post(server, path, server.engine, ctx, dict(req or {}))
    except ui.UiError as e:
        seclog.page_write(server.seclog, path, req or {}, e.status)
        raise WizardError(e.status, str(e)) from None
    if not isinstance(reply, JsonReply):
        raise WizardError(500, f"{path} answered no JSON")
    seclog.page_write(server.seclog, path, req or {}, reply.status)
    if reply.status >= 400:
        raise WizardError(reply.status, str((reply.body.get("error") or {}).get("message") if isinstance(reply.body.get("error"), dict)
                                            else reply.body.get("error") or reply.body))
    return reply.body


def _need(a: dict, *keys):
    for k in keys:
        v = a.get(k)
        if v is None or (isinstance(v, str) and not v.strip()) or v == []:
            raise WizardError(400, f"{k.replace('_', ' ')} is needed")


def _free_name(server, base: str) -> str:
    import re
    base = re.sub(r"[^a-z0-9._-]+", "-", base.lower()).strip("-")[:40] or "model"
    names = {t.name for t in server.cfg.tiers} | {e.get("name") for e in server.cfg.engines}
    n, out = 2, base
    while out in names:
        out, n = f"{base}-{n}", n + 1
    return out


# ------------------------------------------------------------------------------------------------ the wizards
class Wizard:
    name = title = about = ""
    empty = None                     # the empty state it is offered from: models | engines | servers | entry | friends | gateway | collections
    steps: list = []

    def defaults(self, server) -> dict:
        return {}

    def options(self, server, field: str, a: dict) -> list:
        return []

    def plan(self, server, a: dict) -> dict:
        raise NotImplementedError

    def apply(self, server, a: dict, job) -> dict:
        raise NotImplementedError

    def describe(self, server) -> dict:
        return {"name": self.name, "title": self.title, "about": self.about, "empty": self.empty, "steps": self.steps,
                "defaults": self.defaults(server)}


VARIANT_FIELD = {"key": "variant", "label": "", "type": "choice", "options": [
    {"value": "standard", "label": "Standard: the makers' own release (it declines some requests)"},
    {"value": "uncensored", "label": "Uncensored: answers anything (for adults)"}]}


def _catalog_options(server, friends: bool = False, variant: str | None = None, ask_repo: bool = True) -> list:
    """The catalog as the page's model cards: the recommended one first, what each needs, why one does not fit. A card
    that asks for its file (ask_repo) only where the setup has that step (Run a model, not the one-click first run)."""
    from nebula import catalog as C
    from nebula import yard
    o = C.offer(yard.models_dir(server), friends, variant)
    rows = sorted((m for m in o["models"] if ask_repo or not m.get("ask_repo")),
                  key=lambda m: (m["id"] != o["recommended"], bool(m["lacks"])))
    return [{"value": m["id"], "label": m["name"], "about": m["about"], "maker": m.get("maker"),
             "size": "you choose the file and its size" if m.get("ask_repo") else
                     "on this computer: no download" if _cached_files(server, m) is not None else f"{m['download_gb']:.0f} GB download",
             "note": m.get("content"), "recommended": m["id"] == o["recommended"], "lacks": m["lacks"], "disabled": False,
             "engine_ok": o["impulse"], "engine_why": o["impulse_why"]} for m in rows]


def _cached_files(server, m: dict) -> dict | None:
    """A catalog model's files already in the Hugging Face cache at its pinned revision ({file: path}); None when its
    main file is not there (the drafter or the projector may still be downloaded)."""
    from nebula import hfcache
    if m.get("ask_repo"):                               # its file is the owner's choice, not the catalog's
        return None
    main = hfcache.find(server.cfg, m["repo"], m.get("revision"), m["file"])
    if main is None:
        return None
    out = {m["file"]: main}
    for k in ("mtp", "mmproj"):
        if m.get(k):
            p = hfcache.find(server.cfg, m["repo"], m.get("revision"), m[k])
            if p:
                out[m[k]] = p
    for f in m.get("shards") or []:                    # a model split over several files: the others beside the first
        p = hfcache.find(server.cfg, m["repo"], m.get("revision"), f)
        if p:
            out[f] = p
    return out


def _cache_options(server) -> list:
    from nebula import hfcache
    return [{"value": m["id"], "label": hfcache.label(m)} for m in hfcache.models(server.cfg)]


def _cached(server, cid: str) -> dict:
    from nebula import hfcache
    m = next((x for x in hfcache.models(server.cfg) if x["id"] == cid), None)
    if m is None:
        raise WizardError(404, "that model is no longer in the Hugging Face cache")
    return m


def _cached_plan(server, a: dict) -> dict:
    from nebula import hfcache
    m = _cached(server, a["cached"])
    pkg, binary, why = _engine_for(server, {k: v for k, v in a.items() if k != "package"} if not a.get("advanced") else a)
    if not binary:
        raise WizardError(409, why or f"{pkg} isn't on this computer")
    size = None
    if _is_portable(server, binary):
        try:
            size = os.path.getsize(m["path"]) / 2**20
        except (OSError, KeyError):
            pass
    return {"lines": [f"Use {hfcache.label(m)} from your Hugging Face cache (no download; Nebula only reads it)",
                      *_portable_lines(server, binary),
                      _engine_line(server, pkg, binary) + (", reading pictures" if m.get("mmproj") else ""),
                      "Send it a hello to check it answers"],
            "warnings": ([why] if why else []) + (_vulkan_fit(server, size, a) if _is_portable(server, binary) else [])}


def _cached_deploy(server, job, a: dict) -> dict:
    m = _cached(server, a["cached"])
    pkg, binary, _why = _engine_for(server, {k: v for k, v in a.items() if k != "package"} if not a.get("advanced") else a)
    name = a.get("name") or _free_name(server, m["repo"].split("/")[-1].lower().replace("-gguf", "")[:24])
    opts = {"mmproj": True, "mmproj_file": m["mmproj"]} if m.get("mmproj") else {}
    card = _cards_for(a)
    job.deploy({"name": name, "kind": pkg, **_portable_spec(server, job, pkg, binary, a), "model": m["path"], "context": int(a.get("context") or 8192),
                "slots": 1, "options": opts, **({"card": card} if card else {}), **({"cwd": a["cwd"]} if a.get("cwd") else {})})
    return {"model": name, "test": job.test(name)}


PORTABLE = "impulse-portable"   # _engine_for's program while Impulse portable is still to be downloaded (nebula/pieces.py)
FLASHNEXT_NEEDS = ("Qwen3.8 Flash-Next runs on Impulse with an NVIDIA graphics card (RTX 30, 40 or 50) on Linux; this "
                   "computer has none, so Nebula cannot run it here")


def _engine_for(server, a: dict) -> tuple[str, str | None, str | None]:
    """(the package, its program, why Impulse cannot run here): Impulse where it runs (an NVIDIA card); else Impulse
    portable (llama.cpp's server with Impulse's hooks) on a Radeon or another card through Vulkan, else on the CPU -
    downloaded first when it is not here (PORTABLE); else a llama-server of the owner's."""
    from nebula import catalog as C
    from nebula import packages as P
    runs, why = C.impulse_runs_here()
    if a.get("package") and (runs or a["package"] != "impulse"):    # Advanced: chosen by hand (Impulse where it runs)
        return a["package"], a.get("binary") or P.locate(a["package"], server.cfg.base_dir), None
    if runs:
        b = a.get("binary") or P.locate("impulse", server.cfg.base_dir)
        return "impulse", b, None if b else "Impulse isn't on this computer yet"
    from nebula import halo_engine as HE, pieces
    port = HE.find_portable(server.cfg.base_dir)
    if port or pieces.pin("impulse-portable") is not None:
        return "llama.cpp", port or PORTABLE, f"{why}: Nebula runs it with Impulse portable instead, {_vulkan_where(server, a)}"
    b = a.get("binary") or P.locate("llama.cpp", server.cfg.base_dir)
    return "llama.cpp", b, why + (": Nebula uses llama.cpp instead" if b else ", and llama.cpp isn't on this computer either")


def _is_portable(server, binary) -> bool:
    from nebula import halo_engine as HE
    return binary == PORTABLE or (binary is not None and binary == HE.find_portable(server.cfg.base_dir))


def _vulkan_card(server, a: dict | None = None) -> dict | None:
    """The card Impulse portable runs a model on: the one picked (Advanced), else Nebula's pick (catalog.vulkan_pick)."""
    from nebula import catalog as C
    return C.vulkan_pick(C.vulkan_cards(server.cfg.base_dir), (a or {}).get("vk_card") or None)


def _vulkan_where(server, a: dict | None = None) -> str:
    """Where Impulse portable runs a model here, in a few words."""
    from nebula import halo_engine as HE
    c = _vulkan_card(server, a)
    if c:
        return f"on the {c['name']} through Vulkan"
    if HE.find_portable(server.cfg.base_dir):
        return "on the CPU (no graphics card answers through Vulkan here)"
    return "on a Radeon or another graphics card through Vulkan, else on the CPU"


def _engine_line(server, pkg: str, binary) -> str:
    if _is_portable(server, binary):
        return ("Run it with Impulse portable, directly on this computer (not in a sandbox: a sandbox does not reach the "
                "graphics card)")
    return f"Run it on this computer with {'Impulse 0.1.1' if pkg == 'impulse' else pkg}"


def _portable_lines(server, binary) -> list:
    """The download Impulse portable needs first, said plainly (nothing when it is here)."""
    from nebula import pieces
    return [pieces.say("impulse-portable")[0].upper() + pieces.say("impulse-portable")[1:]] if binary == PORTABLE else []


def _vulkan_fit(server, size_mb: float | None, a: dict | None = None) -> list:
    """Impulse portable's fit, in plain words: all on the card, or how much goes on it and how much stays in RAM
    (llama.cpp's --fit places the layers, and a mixture of experts' experts, by the card's free memory)."""
    from nebula import engines as E, halo_engine as HE
    c = _vulkan_card(server, a)
    ram = E.ram_free_mb()
    if not c:
        if HE.find_portable(server.cfg.base_dir):
            return ["No graphics card answers through Vulkan here: it runs on the CPU, from RAM (slower than on a card)."]
        return ["Impulse portable puts as much of it as fits on the graphics card (through Vulkan) and keeps the rest in RAM."]
    if not size_mb:
        return [f"It goes on the {c['name']} ({c['free_mb'] / 1024:.1f} GB free) as far as it fits; the rest stays in RAM."]
    room = max(0, c["free_mb"] - 1024)                           # --fit keeps 1 GB of the card free
    if size_mb <= room:
        return [f"It fits on the {c['name']} ({c['free_mb'] / 1024:.1f} GB free)."]
    rest = size_mb - room
    msg = (f"About {room / 1024:.1f} GB of it goes on the {c['name']} and the rest, about {rest / 1024:.1f} GB, stays in "
           "RAM: it runs, more slowly than all on a card.")
    if ram is not None and rest > ram:
        msg += f" This computer has {ram / 1024:.1f} GB of RAM free now, so it may not start: close other programs first."
    return [msg]


def _portable_spec(server, job, pkg: str, binary, a: dict | None = None) -> dict:
    """The deployment's program: Impulse portable downloaded now when it is not here (said first), and run directly on
    this computer (a sandbox does not reach the graphics card); any other program as it is."""
    if not _is_portable(server, binary):
        return {"binary": binary}
    if binary == PORTABLE:
        from nebula import pieces
        job.say(pieces.say("impulse-portable"), "running")
        binary = pieces.fetch("impulse-portable", server.cfg.base_dir,
                              lambda got, total: job.steps[-1].__setitem__("progress", [got, total]))
        job.steps[-1]["state"] = "done"
    c = _vulkan_card(server, a)                          # one card, by its name (found again at each start)
    return {"binary": binary, "where": "local", **({"vulkan_card": c["name"]} if c else {}),
            "note": f"Impulse portable (no NVIDIA card here): {'the ' + c['name'] + ' through Vulkan' if c else 'the CPU'}"}


# --- a catalog card that asks for its file (Qwen3.8 Flash-Next): a Hugging Face repository the owner picks
def _ask_repo(a: dict) -> dict | None:
    from nebula import catalog as C
    m = C.get(a.get("model") or "") if a.get("model") else None
    return m if m and m.get("ask_repo") else None


def _size_name(c: dict) -> str:
    """A size as the repository names it: its folder (UD-Q4_K_XL) when its files are in one, else its quantization."""
    return c["id"].split("/", 1)[0] if "/" in c["id"] else os.path.basename(c["id"]).rsplit(".gguf", 1)[0]


def _gb(b: int) -> str:
    return f"{b / 2**30:.0f} GB"


def _repo_lacks(n: dict, have: dict) -> list[str]:
    """What this computer lacks for it, in GB as the rest of the page says them: what it needs rounded up, what this
    computer has rounded down (so "needs 177 GB, 177 GB free" never happens)."""
    import math
    out = []
    for k, what in (("gpu_mb", "a graphics card with {} GB"), ("ram_mb", "{} GB of memory"), ("disk_mb", "{} GB of free disk")):
        need, got = n.get(k, 0), have.get(k.replace("_mb", "_mb"), 0)
        if k != "gpu_mb" and not got:
            continue                                     # not known here: not said
        if got < need:
            out.append(what.format(math.ceil(need / 1024)) + (f" (this computer has {math.floor(got / 1024)} GB"
                                                                + (" free)" if k == "disk_mb" else ")") if got else " (none found)"))
    return out


def _repo_needs(size: int) -> dict:
    """What a Flash-Next file of this size needs here: its experts stay in RAM (the file plus ~8 GB), the file on disk
    (plus ~2 GB), and a card of 12 GB for the rest."""
    mb = size // 2**20
    return {"gpu_mb": 12000, "ram_mb": mb + 8192, "disk_mb": mb + 2048}


def _repo_choice(server, m: dict, a: dict) -> dict:
    """The repository the owner pasted, the size they picked, checked: a Flash-Next file (read from its first bytes, not
    downloaded), its total size, and whether it fits this computer (else the smaller sizes that do)."""
    from nebula import catalog as C, hf, yard
    link = str(a.get("fn_link") or "").strip()
    if not link:
        raise WizardError(400, f"Paste a link to a Hugging Face repository with a {m['name']} GGUF file "
                               f"(for example {m['example_repo']}).")
    r = _call(server, "/api/yard/hf/resolve", {"url": link})
    choices = r.get("choices") or []
    pick = a.get("fn_quant") or r.get("chosen") or (choices[0]["id"] if len(choices) == 1 else None)
    c = next((x for x in choices if x["id"] == pick or _size_name(x).upper() == str(pick or "").upper()
              or (x.get("quant") or "").upper() == str(pick or "").upper()), None)
    first = c or (choices[0] if choices else None)          # what it is: read once, from the first size's first file
    if first is None:
        raise WizardError(404, "This repository has no GGUF model files.")
    try:
        arch = hf.remote_architecture(r["repo"], r.get("revision") or "main", first["files"][0], yard.pulls(server).token())
    except hf.HfError as e:
        raise WizardError(e.status, str(e)) from None
    if arch not in (m.get("architectures") or []):
        raise WizardError(409, f"This file is not {m['name']}: it says it is {arch or 'something Nebula cannot read'}, and "
                               f"{m['name']} files say {' or '.join(m['architectures'])}. Pick a {m['name']} repository, "
                               "or use one of the other cards for this model.")
    if c is None:
        raise WizardError(400, f"This repository has {len(choices)} sizes. Pick one: "
                               + ", ".join(f"{_size_name(x)} ({_gb(x['size'])})" for x in choices) + ".")
    have = C.machine(yard.models_dir(server))
    lacks = _repo_lacks(_repo_needs(c["size"]), have)
    if lacks:
        smaller = [x for x in choices if x["size"] < c["size"] and not _repo_lacks(_repo_needs(x["size"]), have)]
        raise WizardError(409, f"{_size_name(c)} ({_gb(c['size'])}) will not fit this computer: it needs "
                               + "; ".join(lacks) + ". " + (("Smaller sizes in this repository that fit: " +
                               ", ".join(f"{_size_name(x)} ({_gb(x['size'])})" for x in sorted(smaller, key=lambda x: -x["size"])) + ".")
                               if smaller else "None of this repository's sizes fit this computer."))
    return {"repo": r["repo"], "revision": r.get("revision") or "main", "choice": c, "arch": arch, "needs": _repo_needs(c["size"])}


def _fn_placement(server) -> tuple[str, list]:
    """Where a Flash-Next engine will run, as the preview says it, and its warnings: in a sandbox only when one is
    there to run it in (an OpenShell gateway running; on Windows the Safe sandbox), else plainly on this computer."""
    from nebula import engines, flashnext
    outside = ["it runs outside a sandbox: it can read what your user can"]
    if flashnext.PLACEMENT != "sandbox":
        return "directly on this computer (not in a sandbox)", outside
    path = getattr(getattr(server, "cfg", None), "path", None)
    where, why = engines.sandbox_plan({"kind": "impulse"}, conf_dir=os.path.dirname(os.path.abspath(path)) if path else None)
    if where == "openshell":
        return "in a sandbox, like your other models", []
    return f"directly on this computer, not in a sandbox ({why})", outside


def _repo_lines(server, m: dict, a: dict) -> tuple[list, list]:
    from nebula import flashnext
    x = _repo_choice(server, m, a)
    c, n = x["choice"], x["needs"]
    files = len(c["files"])
    where, warn = _fn_placement(server)
    return ([f"Download {_size_name(c)} from {x['repo']} ({_gb(c['size'])}" + (f", {files} files" if files > 1 else "")
             + "), each file checked against the checksum Hugging Face publishes for it",
             f"It is {m['name']}: its file says {x['arch']}",
             f"It needs about {-(-n['ram_mb'] // 1024)} GB of memory, {-(-n['disk_mb'] // 1024)} GB of free disk and a graphics card "
             f"with {-(-n['gpu_mb'] // 1024)} GB: this computer has them",
             "Run it with Impulse " + where,
             "Send it a hello to check it answers"], warn)


def _deploy_repo(server, job, a: dict, m: dict) -> dict:
    x = _repo_choice(server, m, a)
    path = job.pull(a["fn_link"], x["choice"]["id"])
    name = a.get("name") or _free_name(server, m["id"])
    card = _cards_for(a)
    from nebula import flashnext
    job.deploy({"name": name, "kind": "impulse", "model": path, "context": int(m.get("context") or 32768), "slots": 1,
                "options": {}, **({"card": card} if card else {}), **({"where": "local"} if flashnext.PLACEMENT == "local" else {})})
    return {"model": name, "test": job.test(name)}


def _cards_for(a: dict) -> str | None:
    """The card(s) a guided setup gives its model: the ones the answers name ("cards": UUIDs or indexes), else this
    machine's card with the most free memory. Without one, a sandbox would have no GPU, and an engine run here every card."""
    if a.get("cards"):
        return ",".join(str(c) for c in a["cards"])
    from nebula import engines
    try:
        have = engines.cards()
    except Exception:  # noqa: BLE001
        have = []
    if not have:
        return None
    best = max(have, key=lambda c: int(c.get("memory_mb", 0)) - int(c.get("used_mb", 0)))
    return str(best.get("uuid") or best.get("index"))


def _deploy_lines(server, a: dict, m: dict) -> tuple[list, list]:
    pkg, binary, why = _engine_for(server, a)
    here = _cached_files(server, m)
    whole = here is not None and all(f in here for f in m.get("shards") or [])
    lines = [f"Use {m['name']} from your Hugging Face cache (no download)" if whole else
             f"Download {m['name']} ({m['download_gb']:.0f} GB" + (f", {1 + len(m['shards'])} files" if m.get("shards") else "")
             + ") into Nebula's models folder",
             *_portable_lines(server, binary),
             _engine_line(server, pkg, binary) + (", reading pictures" if m.get("mmproj") else ""),
             "Send it a hello to check it answers"]
    from nebula import catalog as C
    lay = C.layout(m) if pkg == "impulse" and not a.get("cards") else None
    if lay and "," in lay["card"]:
        lines.insert(2, f"Use {len(lay['card'].split(','))} cards together: {lay['about']}")
    warn = []
    ram = (m.get("needs") or {}).get("ram_free_mb")
    if ram and not (lay and int((lay.get("options") or {}).get("n_cpu_moe", -1)) == 0):   # its experts stay in RAM
        from nebula import engines as E
        have = E.ram_free_mb()
        warn.append(f"It keeps its experts in RAM: it needs about {round(ram / 1024)} GB of free RAM"
                    + (f" (this computer has {round(have / 1024)} GB free now)." if have is not None else ".")
                    + (" Close other programs first, or it will not start." if have is not None and have < ram else "")
                    + (" Nebula runs in WSL, which gets half of the PC's RAM by default: \"Let Nebula use more memory\" below "
                       "raises that (in .wslconfig, backed up first)." if have is not None and have < ram and E.in_wsl() else ""))
    if why and not binary:
        raise WizardError(409, why + (". The Impulse download comes with its first release; until then, put impulse-server on this "
                                      "computer (or set NEBULA_IMPULSE_BIN) and try again." if pkg == "impulse" else "."))
    if why:
        warn.append(why)
    lacks = C.fits(m, C.machine())
    if _is_portable(server, binary):                    # a smaller card is no stop: what does not fit on it stays in RAM
        lacks = [x for x in lacks if not x.startswith("a graphics card")]
        warn += _vulkan_fit(server, (m.get("download_gb") or 0) * 1024 or None, a)
    if lacks:
        warn.append("It may not run well here: it wants " + "; ".join(lacks) + ".")
    if m.get("content"):
        warn.append(f"{m['name']} is an {m['content']}.")
    return lines, warn


def _deploy_catalog(server, job, a: dict, mid: str) -> dict:
    from nebula import catalog as C
    m = C.get(mid, a.get("variant"))
    if m is None:
        raise WizardError(404, f"no model {mid!r} in the catalog")
    pkg, binary, why = _engine_for(server, a)
    if not binary:
        raise WizardError(409, why or "the engine is not on this computer")
    files = [m["file"]] + list(m.get("shards") or []) + [m[k] for k in ("mtp", "mmproj") if m.get(k) and not (k == "mtp" and m.get("mtp_repo"))]
    here = _cached_files(server, m) or {}                          # already in the Hugging Face cache: not downloaded
    missing = [f for f in files if f not in here]
    sha = m.get("sha256")                                          # one file's, or {file name: sha256} for a split model
    pins = ({f: sha[os.path.basename(f)] for f in missing if os.path.basename(f) in sha} if isinstance(sha, dict)
            else {m["file"]: sha}) if sha else None
    paths = job.pull_files(m["repo"], m.get("revision") or "main", missing, pins) if missing else {}
    if m.get("mtp") and m.get("mtp_repo"):                         # its drafter lives in another repository
        from nebula import hfcache
        got = hfcache.find(server.cfg, m["mtp_repo"], None, m["mtp"])
        paths[m["mtp"]] = got or job.pull_files(m["mtp_repo"], "main", [m["mtp"]])[m["mtp"]]
    paths = {**paths, **here, **{os.path.basename(f): p for f, p in here.items()}}
    name = a.get("name") or _free_name(server, m["id"])
    opts = {}
    if m.get("mtp") and pkg == "impulse":
        opts["mtp"] = paths[m["mtp"]]
    if m.get("mmproj"):
        opts["mmproj"], opts["mmproj_file"] = True, paths[m["mmproj"]]
    opts.update(m.get("sampling") or {})                           # the model card's sampling (Flint), shown on the Edit form
    # the cards: the ones the answers name; else a layout of the model's that this machine takes (two cards that hold it
    # all: everything on them); else the card with the most free memory
    lay = C.layout(m) if pkg == "impulse" and not a.get("cards") else None
    if lay:
        opts.update(lay["options"])
    card = lay["card"] if lay else _cards_for(a)
    job.deploy({"name": name, "kind": pkg, **_portable_spec(server, job, pkg, binary, a), "model": paths[m["file"]], "context": int(m.get("context") or 32768),
                "slots": 1, "options": opts, **({"card": card} if card else {}), **({"cwd": a["cwd"]} if a.get("cwd") else {})})
    return {"model": name, "test": job.test(name)}


class FirstRun(Wizard):
    name, title, empty = "first-run", "Get started", "models"
    about = "Nebula runs AI models on your own computer, and lets you share them with friends."
    steps = [
        {"id": "how", "title": "What would you like to do?", "fields": [
            {"key": "source", "label": "", "type": "choice", "options": [
                {"value": "here", "label": "Run AI on this computer (recommended)"},
                {"value": "join", "label": "Join a friend's Nebula (you have an invite)"},
                {"value": "other", "label": "Other ways (for experts)"}]}]},
        {"id": "model", "title": "Pick your model", "when": {"source": "here"}, "fields": [
            VARIANT_FIELD,
            {"key": "model", "label": "", "type": "cards", "options_from": "catalog",
             "help": "Nebula picked the one that suits this computer. The others are under “Other models”."},
            {"key": "cached", "label": "Or one already on this computer (your Hugging Face cache): no download", "type": "select", "options_from": "hfcache"},]},
        {"id": "invite", "title": "Your friend's invite", "when": {"source": "join"}, "fields": [
            {"key": "invite", "label": "Paste the invite your friend sent you", "type": "textarea", "placeholder": "nebula-invite:…"}]},
        {"id": "other", "title": "Other ways", "when": {"source": "other"}, "fields": [
            {"key": "other", "label": "", "type": "choice", "options": [
                {"value": "running", "label": "Use a model server already running on this computer"},
                {"value": "provider", "label": "Use a cloud AI service (OpenAI-compatible)"}]}]},
        {"id": "running", "title": "The running server", "when": {"source": "other", "other": "running"}, "fields": [
            {"key": "url", "label": "Server", "type": "select", "options_from": "running"}]},
        {"id": "provider", "title": "The cloud service", "when": {"source": "other", "other": "provider"}, "fields": [
            {"key": "base_url", "label": "Its address", "type": "text", "placeholder": "https://api.example.com/v1"},
            {"key": "api_key", "label": "Its key", "type": "password", "help": "kept in a file only Nebula reads"},
            {"key": "model", "label": "Model", "type": "select", "options_from": "provider_models"}]},
    ]

    def defaults(self, server):
        from nebula import catalog as C
        return {"source": "here", "other": "running", "variant": C.default_variant()}

    def options(self, server, field, a):
        if field == "catalog":
            return _catalog_options(server, variant=a.get("variant"), ask_repo=False)
        if field == "hfcache":
            return [{"value": "", "label": "none: the model picked above"}] + _cache_options(server)
        if field == "running":
            got = _call(server, "/api/lan/engines", get=True)
            return [{"value": e["url"], "label": f"{e.get('name') or e['url']} at {e['url']}"} for e in got if not e.get("added")]
        if field == "provider_models":
            if not a.get("base_url"):
                return []
            got = _call(server, "/api/providers/models", {"base_url": a["base_url"], "api_key": a.get("api_key")})
            return [{"value": m, "label": m} for m in got.get("models") or []]
        return []

    def plan(self, server, a):
        from nebula import catalog as C
        src = a.get("source") or "here"
        if src == "here" and a.get("cached"):
            return _cached_plan(server, a)
        if src == "here":
            m = C.get(a.get("model") or "", a.get("variant"))
            if m is None:
                raise WizardError(400, "pick a model")
            lines, warn = _deploy_lines(server, a, m)
            return {"lines": lines, "warnings": warn}
        if src == "join":
            return Join().plan(server, a)
        if a.get("other") == "running":
            _need(a, "url")
            return {"lines": [f"Use the server at {a['url']}", "Send it a hello to check it answers"], "warnings": []}
        _need(a, "base_url", "model")
        if not str(a["base_url"]).startswith("https://"):
            raise WizardError(400, "a cloud service's address starts with https://")
        return {"lines": [f"Use {a['model']} at {a['base_url']}", "Its key goes into a file only Nebula reads", "Send it a hello"],
                "warnings": ["A cloud model is never shared with friends."]}

    def apply(self, server, a, job):
        src = a.get("source") or "here"
        if src == "join":
            return Join().apply(server, a, job)
        return {**self._apply_model(server, a, job), **self._first_key(server, job)}

    @staticmethod
    def _first_key(server, job) -> dict:
        """Apps on this computer need a key (design decision, 2026-10-20): the first one, made here and shown once."""
        from nebula import stars
        if not stars.requires_key(server.cfg) or (server.cfg.api or {}).get("keys"):
            return {}
        k = job.do("make the first key for apps on this computer", "/api/stars/api/keys", {"label": "apps on this computer"})
        return {"api_key": k["key"]}

    def _apply_model(self, server, a, job):
        src = a.get("source") or "here"
        if src == "here" and a.get("cached"):
            return _cached_deploy(server, job, a)
        if src == "here":
            return _deploy_catalog(server, job, a, a["model"])
        if a.get("other") == "running":
            name = a.get("name") or _free_name(server, "local")
            job.do(f"add “{name}”", "/api/models", {"name": name, "kind": "engine", "url": a["url"]},
                   undo=("/api/models/" + name + "/remove", {}))
        else:
            name = a.get("name") or _free_name(server, a["model"])
            job.do(f"add “{name}”", "/api/models", {"name": name, "kind": "provider", "base_url": a["base_url"],
                                                      "model": a["model"], "api_key": a.get("api_key")},
                   undo=("/api/models/" + name + "/remove", {}))
        return {"model": name, "test": job.test(name)}


class DeployOwn(Wizard):
    name, title, empty = "deploy", "Run a model on this computer", "engines"
    about = "Pick a model and Nebula downloads it, starts it with Impulse and checks that it answers."
    steps = [
        {"id": "model", "title": "Pick a model", "fields": [
            VARIANT_FIELD,
            {"key": "model", "label": "", "type": "cards", "options_from": "catalog"},
            {"key": "cached", "label": "Or one already on this computer (your Hugging Face cache): no download", "type": "select", "options_from": "hfcache"},
            {"key": "advanced", "label": "Advanced: a model from Hugging Face, another engine", "type": "checkbox"}]},
        {"id": "fn", "title": "Qwen3.8 Flash-Next: which file?", "when": {"model": "qwen38-flashnext", "advanced": False}, "fields": [
            {"key": "fn_link", "label": "A link to a Hugging Face repository with a Qwen3.8 Flash-Next GGUF file", "type": "text",
             "placeholder": "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF",
             "help": "Nebula lists its sizes, checks that it is Flash-Next and that it fits this computer before anything downloads."},
            {"key": "fn_quant", "label": "Size", "type": "select", "options_from": "fn_quant"}]},
        {"id": "link", "title": "Advanced", "when": {"advanced": True}, "fields": [
            {"key": "link", "label": "A Hugging Face link (or a .gguf file, or a Flash-Next pack folder, on this computer)", "type": "text",
             "placeholder": "https://huggingface.co/owner/repo-GGUF"},
            {"key": "quant", "label": "Size", "type": "select", "options_from": "quant"},
            {"key": "package", "label": "Engine", "type": "select", "options_from": "package"},
            {"key": "vk_card", "label": "Graphics card (without an NVIDIA card: through Vulkan)", "type": "select", "options_from": "vk_card"},
            {"key": "context", "label": "Context (tokens)", "type": "number"},
            {"key": "thinking", "label": "Thinking before answering", "type": "checkbox"}]},
    ]

    def defaults(self, server):
        from nebula import catalog as C
        return {"advanced": False, "package": "impulse", "context": 8192, "thinking": False, "variant": C.default_variant()}

    def options(self, server, field, a):
        if field == "catalog":
            return _catalog_options(server, variant=a.get("variant"))
        if field == "hfcache":
            return [{"value": "", "label": "none: the model picked above"}] + _cache_options(server)
        if field == "package":
            from nebula import packages as P
            return [{"value": k, "label": v["label"]} for k, v in P.PACKAGES.items()]
        if field == "vk_card":                               # Impulse portable's cards (no NVIDIA card here)
            from nebula import catalog as C
            vk = C.vulkan_cards(server.cfg.base_dir)
            pick = C.vulkan_pick(vk)
            return ([{"value": "", "label": f"Nebula's pick: {pick['name']}" if pick else "Nebula's pick"}] +
                    [{"value": c["name"], "label": f"{c['name']} · {c['free_mb'] / 1024:.1f} GB free"
                      + (" (the processor's graphics: shares the RAM)" if c.get("integrated") else "")} for c in vk])
        if field == "fn_quant":                              # the sizes of the repository pasted, and which fit here
            link = str(a.get("fn_link") or "").strip()
            if not link:
                return []
            from nebula import catalog as C, yard
            r = _call(server, "/api/yard/hf/resolve", {"url": link})
            have = C.machine(yard.models_dir(server))
            return [{"value": c["id"], "label": f"{_size_name(c)} · {_gb(c['size'])}"
                     + ("" if not _repo_lacks(_repo_needs(c["size"]), have) else " · too big for this computer")}
                    for c in r.get("choices") or []]
        if field == "quant":
            m = str(a.get("link") or "").strip()
            if not m or os.path.isfile(os.path.expanduser(m)) or os.path.isdir(os.path.expanduser(m)):
                return []
            r = _call(server, "/api/yard/hf/resolve", {"url": m})
            return [{"value": c["id"], "label": f"{c['quant'] or c['id']} · {c['size'] / 2**30:.1f} GB"} for c in r.get("choices") or []]
        return []

    def plan(self, server, a):
        from nebula import catalog as C
        if not a.get("advanced") and a.get("cached"):
            return _cached_plan(server, a)
        if not a.get("advanced"):
            m = C.get(a.get("model") or "", a.get("variant"))
            if m is None:
                raise WizardError(400, "pick a model")
            if m.get("ask_repo") and not C.impulse_runs_here()[0]:
                raise WizardError(409, FLASHNEXT_NEEDS)
            lines, warn = _repo_lines(server, m, a) if m.get("ask_repo") else _deploy_lines(server, a, m)
            return {"lines": lines, "warnings": warn}
        _need(a, "link")
        from nebula import flashnext
        if flashnext.is_pack(os.path.expanduser(str(a["link"]))):
            if not C.impulse_runs_here()[0]:
                raise WizardError(409, FLASHNEXT_NEEDS.replace("Qwen3.8 Flash-Next runs", "A Flash-Next pack runs"))
            if a.get("image"):                                 # the reference engine's way: a container image of its own
                return {"lines": [f"Use the Flash-Next pack in {a['link']}",
                                  f"Run it with {flashnext.LABEL} in a sandbox, from the image {a['image']}",
                                  "Send it a hello to check it answers"],
                        "warnings": ["the image is built on this computer from impulse/flashnext/nebula-image (its README)"]}
            from nebula import pieces
            if pieces.find("impulse-flashnext", server.cfg.base_dir) is None and pieces.pin("impulse-flashnext") is None:
                raise WizardError(409, pieces.say("impulse-flashnext"))
            have = pieces.find("impulse-flashnext", server.cfg.base_dir) is not None
            return {"lines": [f"Use the Flash-Next pack in {a['link']}",
                              f"{flashnext.LABEL}: already on this computer" if have else
                              pieces.say("impulse-flashnext")[0].upper() + pieces.say("impulse-flashnext")[1:],
                              "Run it with Impulse " + _fn_placement(server)[0],
                              "Send it a hello to check it answers"],
                    "warnings": _fn_placement(server)[1]}
        local = os.path.isfile(os.path.expanduser(str(a["link"])))
        if not local:
            _need(a, "quant")
        pkg, binary, why = _engine_for(server, a)
        if not binary:
            raise WizardError(409, why or f"{pkg} isn't on this computer")
        warn = []
        if _is_portable(server, binary):
            size = None
            if local:
                size = os.path.getsize(os.path.expanduser(str(a["link"]))) / 2**20
            else:
                r = _call(server, "/api/yard/hf/resolve", {"url": str(a["link"]).strip()})
                c = next((c for c in r.get("choices") or [] if c["id"] == a["quant"]), None)
                size = c["size"] / 2**20 if c else None
            warn = [why] + _vulkan_fit(server, size, a)
        return {"lines": ([f"Use {a['link']}"] if local else [f"Download {a['quant']} from {a['link']}"]) + _portable_lines(server, binary) +
                [(_engine_line(server, pkg, binary) if _is_portable(server, binary) else f"Run it with {pkg}")
                 + f", context {int(a.get('context') or 8192):,} tokens", "Send it a hello to check it answers"],
                "warnings": warn}

    def apply(self, server, a, job):
        if not a.get("advanced") and a.get("cached"):
            return _cached_deploy(server, job, a)
        from nebula import catalog as C
        if not a.get("advanced") and _ask_repo(a):
            if not C.impulse_runs_here()[0]:
                raise WizardError(409, FLASHNEXT_NEEDS)
            return _deploy_repo(server, job, a, _ask_repo(a))
        if not a.get("advanced"):
            return _deploy_catalog(server, job, a, a["model"])
        from nebula import flashnext
        pack = flashnext.pack_files(os.path.expanduser(str(a["link"])))
        if pack is not None and not a.get("image") and not C.impulse_runs_here()[0]:
            raise WizardError(409, FLASHNEXT_NEEDS.replace("Qwen3.8 Flash-Next runs", "A Flash-Next pack runs"))
        if pack is not None and not a.get("image"):            # a Flash-Next pack on Impulse: impulse-fx is a piece of its own
            from nebula import pieces
            if pieces.find("impulse-flashnext", server.cfg.base_dir) is None:
                job.say(pieces.say("impulse-flashnext"), "running")
                pieces.fetch("impulse-flashnext", server.cfg.base_dir,
                             lambda got, total: job.steps[-1].__setitem__("progress", [got, total]))
                job.steps[-1]["state"] = "done"
            name = a.get("name") or _free_name(server, os.path.basename(pack["dir"].rstrip("/"))[:24])
            card = _cards_for(a)
            job.deploy({"name": name, "kind": "impulse", "model": pack["dir"], "context": int(a.get("context") or 32768),
                        "slots": 1, "options": {}, **({"card": card} if card else {}),
                        **({"where": "local"} if flashnext.PLACEMENT == "local" else {})})
            return {"model": name, "test": job.test(name)}
        if pack is not None:                                   # with an image: the reference engine's way
            from nebula import yard
            name = a.get("name") or _free_name(server, os.path.basename(pack["dir"].rstrip("/"))[:24])
            spec = flashnext.engine_spec(name, machine=a.get("machine"), folder=os.path.dirname(pack["dir"].rstrip("/")),
                                         files=pack, port=yard.free_port(server), cards=list(a.get("cards") or []),
                                         image=a.get("image") or flashnext.IMAGE, numa_nodes=a.get("numa_nodes"))
            job.deploy(spec)
            return {"model": name, "test": job.test(name)}
        name = a.get("name") or _free_name(server, os.path.basename(str(a.get("quant") or a["link"])).replace(".gguf", ""))
        path = os.path.expanduser(str(a["link"]))
        if not os.path.isfile(path):
            path, _mm = job.pull(a["link"], a["quant"])
        pkg, binary, _why = _engine_for(server, a)
        card = _cards_for(a)
        job.deploy({"name": name, "kind": pkg, **_portable_spec(server, job, pkg, binary, a), "model": path, "context": int(a.get("context") or 8192),
                    "slots": 1, **({"card": card} if card else {}), "options": {"thinking": bool(a.get("thinking"))}, **({"cwd": a["cwd"]} if a.get("cwd") else {})})
        return {"model": name, "test": job.test(name)}


class Join(Wizard):
    name, title, empty = "join", "Join a friend's Nebula", "friends"
    about = "Your friend sent you an invite. Join, and you can use the models they share with you; your own chats stay yours."
    steps = [
        {"id": "invite", "title": "Your friend's invite", "fields": [
            {"key": "invite", "label": "Paste the invite your friend sent you", "type": "textarea", "placeholder": "nebula-invite:…"}]},
        {"id": "also", "title": "Your own model too?", "fields": [
            {"key": "also_run", "label": "Also run a model on this computer (you can share it back later)", "type": "checkbox"},
            dict(VARIANT_FIELD, when_field="also_run"),
            {"key": "model", "label": "", "type": "cards", "options_from": "friend_catalog", "when_field": "also_run"}]},
    ]

    def defaults(self, server):
        return {"also_run": False}

    def options(self, server, field, a):
        if field == "friend_catalog":
            return _catalog_options(server, friends=True, variant=a.get("variant"))
        return []

    def _who(self, invite: str) -> str:
        from nebula import gravity
        try:
            inv = gravity.GravityNode.parse_invite(str(invite or ""))
        except gravity.GravityError as e:
            raise WizardError(400, "that does not look like a Nebula invite: " + str(e)) from None
        if inv.get("kind") != "friend":
            raise WizardError(400, "that invite is for one of its owner's own machines, not for a friend")
        return str(inv.get("name") or "your friend")

    def plan(self, server, a):
        from nebula import catalog as C
        from nebula import gravity
        _need(a, "invite")
        who = self._who(a["invite"])
        ok, why = gravity.available()
        if not ok:
            raise WizardError(503, why)
        lines = []
        if getattr(server, "gravity", None) is None:
            lines.append("Turn on Gravity (how Nebulas reach each other, privately)")
        lines += [f"Join {who}'s Nebula", f"Add the models {who} shares with you", "Send one a hello to check it answers"]
        warn = []
        if a.get("also_run") and a.get("model"):
            m = C.get(a["model"])
            if m is None or not m.get("friends"):
                raise WizardError(400, "pick one of the models offered here")
            more, w = _deploy_lines(server, a, m)
            lines += more
            warn += w
        return {"lines": lines, "warnings": warn}

    def apply(self, server, a, job):
        who = self._who(a["invite"])
        if getattr(server, "gravity", None) is None:
            job.do("turn on Gravity", "/api/yard/node/on", {}, undo=("/api/yard/node/off", {}))
        peer = job.do(f"join {who}'s Nebula", "/api/yard/node/join", {"invite": a["invite"]})["peer"]
        job.undos.append((f"/api/yard/node/peers/{peer['id']}/revoke", {}, f"undo: leave {who}'s Nebula"))
        made = job.do(f"add the models {who} shares", f"/api/yard/node/peers/{peer['id']}/adopt", {}).get("made") or []
        for n in made:
            job.undos.append((f"/api/models/{n}/remove", {}, f"undo: remove {n}"))
        out = {"joined": who, "models": made}
        if made:
            out["test"] = job.test(made[0])
        if a.get("also_run") and a.get("model"):
            out["own"] = _deploy_catalog(server, job, a, a["model"])
        return out


class AddServer(Wizard):
    name, title, empty = "add-server", "Add a model server (SSH)", "servers"
    about = ("Use another machine's cards on your network (a box with a GPU and no screen, say). Nebula sets it up over SSH "
             "with only what runs models (no Nebula there); its cards then appear under Engines like this computer's, and "
             "its models are reached over mutual TLS.")
    steps = [
        {"id": "where", "title": "Which machine", "fields": [
            {"key": "host", "label": "Its address or name on your network", "type": "text", "placeholder": "192.0.2.10"},
            {"key": "port", "label": "SSH port", "type": "number"},
            {"key": "user", "label": "Your user there", "type": "text"},
            {"key": "key_file", "label": "Your SSH key file (empty: the one your SSH settings use for it)", "type": "text",
             "help": "a name from your SSH settings works as the address too"}]},
        {"id": "trust", "title": "Is this the right machine?", "fields": [
            {"key": "fingerprint", "label": "Its host key", "type": "select", "options_from": "fingerprints",
             "help": "compare it with the one the machine shows (ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub); a mismatch is refused later"}]},
        {"id": "admin", "title": "What needs an administrator", "fields": [
            {"key": "admin", "label": "Nebula checked both machines. Ticked, these run with sudo before it installs:",
             "type": "multi", "options_from": "admin_needs"},
            {"key": "sudo_password", "label": "Your password on that machine (for sudo)", "type": "password",
             "help": "used once for the ticked tasks there, then forgotten: never saved or logged"},
            {"key": "local_sudo_password", "label": "Your password on this computer (only for tasks marked \u201con this computer\u201d)",
             "type": "password", "help": "used once, then forgotten"}]},
        {"id": "name", "title": "Name it", "fields": [
            {"key": "name", "label": "Name", "type": "text"}]},
    ]

    def defaults(self, server):
        return {"port": 22, "user": os.environ.get("USER") or ""}

    def options(self, server, field, a):
        if field == "admin_needs":
            _need(a, "host", "user", "fingerprint")
            pf = _call(server, "/api/yard/servers/preflight", dict(_target(a), fingerprint=a["fingerprint"]))
            return [{"value": n["id"], "label": n["words"], "command": n.get("command"),   # the command stays out of sight (design decision 10-22)
                     "checked": True} if n.get("id") else {"value": "", "label": n["words"], "disabled": True}
                    for n in pf.get("needs") or []]
        if field == "fingerprints":
            _need(a, "host")
            got = _call(server, "/api/yard/servers/scan", {"host": a["host"], "port": int(a.get("port") or 22)})
            return [{"value": k["fingerprint"], "label": f"{k['type']} {k['fingerprint']}"} for k in got.get("keys") or []]
        return []

    def plan(self, server, a):
        _need(a, "host", "user", "fingerprint")
        name = a.get("name") or a["host"]
        admin = [x for x in a.get("admin") or [] if x]
        lines = [f"Trust {a['host']}'s host key {a['fingerprint']}"]
        if admin:
            here = [x for x in admin if x.endswith("-here")]
            if len(admin) > len(here) and not a.get("sudo_password"):
                raise WizardError(400, f"your password on {a['host']} is needed for the ticked tasks there")
            if here and not a.get("local_sudo_password"):
                raise WizardError(400, "this computer's password is needed for the ticked task here")
            lines.append(f"As administrator (sudo): {', '.join(admin)} (not taken back by Undo)")
        return {"lines": lines + [
                          f"Copy the model-machine pieces there over SSH as {a['user']} (no Nebula): the network's node "
                          "programs, the node keeper and OpenShell, run as your services there",
                          f"Join “{name}” to your network (mutual TLS) and register its OpenShell gateway here",
                          "Its cards then appear under Engines: deploy models onto them from there"],
                "warnings": ["SSH is how this computer manages it (install, engines, removal); your models travel over mutual TLS."]}

    def apply(self, server, a, job):
        name = a.get("name") or a["host"]
        job.do("trust its host key", "/api/yard/servers/trust", {"host": a["host"], "port": int(a.get("port") or 22),
                                                                 "fingerprint": a["fingerprint"]},
               undo=lambda: _untrust(server, a["host"], int(a.get("port") or 22)))
        admin = [x for x in a.get("admin") or [] if x]
        if admin:
            job.do(f"as administrator: {', '.join(admin)}", "/api/yard/servers/admin",
                   dict(_target(a), tasks=admin, password=a.get("sudo_password") or "", local_password=a.get("local_sudo_password") or ""))
        out = job.do(f"set up {a['host']} as a model machine", "/api/yard/servers/install",
                     {"host": a["host"], "port": int(a.get("port") or 22), "user": a["user"], "key_file": a.get("key_file") or None,
                      "name": name},
                     undo=(f"/api/yard/servers/{name}/uninstall", {"key_file": a.get("key_file") or None}))
        return {"server": name, "steps": out.get("steps")}


def _target(a: dict) -> dict:
    return {"host": a["host"], "port": int(a.get("port") or 22), "user": a["user"], "key_file": a.get("key_file") or None}


def _untrust(server, host: str, port: int):
    from nebula import yard
    kh = yard._known_hosts(server)
    try:
        lines = open(kh, encoding="utf-8").read().splitlines()
    except OSError:
        return
    tag = host if port == 22 else f"[{host}]:{port}"
    keep = [ln for ln in lines if not ln.split(" ", 1)[0].split(",")[0] == tag]
    with open(kh, "w", encoding="utf-8") as f:
        f.write("\n".join(keep) + ("\n" if keep else ""))


class ShareFriend(Wizard):
    name, title, empty = "share", "Share with a friend", "friends"
    about = ("Let a friend's Nebula use some of your local models over Gravity. They get a one-time invite; your "
             "sessions stay private, each guest gets its own cache. Cloud models are never shared.")
    steps = [
        {"id": "who", "title": "Who", "fields": [{"key": "friend", "label": "Your friend's name (for you)", "type": "text"}]},
        {"id": "models", "title": "Which models they may use", "fields": [
            {"key": "models", "label": "Models", "type": "multi", "options_from": "shareable"}]},
    ]

    def defaults(self, server):
        return {"models": []}

    def options(self, server, field, a):
        if field == "shareable":
            from nebula import gravity
            out = []
            for t in server.cfg.tiers:
                ag = gravity.anti_gravity(t)
                out.append({"value": t.name, "label": t.name + (" — a cloud model: never shared (Anti-Gravity)" if ag else
                                                                 " — needs care: " + gravity.needs_care(t) if gravity.needs_care(t) else ""),
                            "disabled": bool(ag), "care": gravity.needs_care(t) if not ag else None})
            return out
        return []

    def plan(self, server, a):
        _need(a, "friend", "models")
        from nebula import gravity
        lines, warnings, confirm = [], [], []
        g = getattr(server, "gravity", None)
        if g is None:
            ok, why = gravity.available()
            if not ok:
                raise WizardError(503, why)
            lines.append("Turn Gravity on (this Nebula gets its own key; nothing is announced)")
        for m in a["models"]:
            t = server.cfg.tier(m)
            if t is None:
                raise WizardError(404, f"no model {m}")
            if gravity.anti_gravity(t):
                raise WizardError(400, f"{m} is a cloud model: it is never shared (Anti-Gravity)")
            if not getattr(t, "shared", False):
                lines.append(f"Offer {m} to guests (each guest its own cache)")
            care = gravity.needs_care(t)
            if care:
                warnings.append(f"{m}: sharing it may not be safe ({care}). Your friend is told too.")
                confirm.append(m)
        lines.append(f"Make a one-time invite for {a['friend']} that gives them {', '.join(a['models'])} when they join")
        return {"lines": lines, "warnings": warnings, "confirm": confirm}

    def apply(self, server, a, job):
        if getattr(server, "gravity", None) is None:
            job.do("turn Gravity on", "/api/yard/node/on", {}, undo=("/api/yard/node/off", {}))
        for m in a["models"]:
            if not getattr(server.cfg.tier(m), "shared", False):
                job.do(f"offer {m} to guests", f"/api/models/{m}/shared", {"shared": True}, undo=(f"/api/models/{m}/shared", {"shared": False}))
        inv = job.do("make the invite", "/api/yard/node/invite", {"kind": "friend", "for": a["friend"], "grants": a["models"],
                                                                   "confirm": a.get("confirm") or []})
        job.undos.append(("/api/yard/node/invites/" + inv["id"] + "/cancel", {}, f"cancel the invite for {a['friend']}"))
        return {"invite": inv["invite"], "note": inv["note"], "reach": inv.get("reach"), "reach_note": inv.get("reach_note"), "expires_in_s": inv["expires_in_s"]}


class Gateway(Wizard):
    name, title, empty = "gateway", "Deploy a gateway", "gateway"
    about = ("A gateway lets your friends' Nebulas and your phone reach yours from anywhere without opening a port: "
             "an iroh relay of your own that only your members may use.")
    steps = [
        {"id": "where", "title": "Where it runs", "fields": [
            {"key": "where", "label": "Where", "type": "choice", "options": [
                {"value": "here", "label": "On this machine (Docker)"},
                {"value": "elsewhere", "label": "On a server with a public address (Nebula gives you the commands)"}]},
            {"key": "hostname", "label": "Its public name (for HTTPS; empty: plain, for your network)", "type": "text"},
            {"key": "email", "label": "Your email (for the HTTPS certificate)", "type": "text"},
            {"key": "port", "label": "Port", "type": "number"}]},
    ]

    def defaults(self, server):
        return {"where": "here", "port": 3340}

    def plan(self, server, a):
        lines = []
        if getattr(server, "gravity", None) is None:
            lines.append("Turn Gravity on")
        if a.get("where") == "elsewhere":
            lines.append("Write its config and show the commands to run on your server")
        else:
            lines.append(f"Start the gateway container on this machine, port {int(a.get('port') or 3340)}")
            lines.append("Use it for this Nebula; new invites carry it")
        return {"lines": lines, "warnings": ["Only this Nebula and its members may use it (an allowlist of their keys)."]}

    def apply(self, server, a, job):
        if getattr(server, "gravity", None) is None:
            job.do("turn Gravity on", "/api/yard/node/on", {}, undo=("/api/yard/node/off", {}))
        body = {"hostname": a.get("hostname") or None, "email": a.get("email") or None, "port": int(a.get("port") or 3340),
                "elsewhere": a.get("where") == "elsewhere"}
        out = job.do("deploy the gateway", "/api/yard/node/gateway", body,
                     undo=None if body["elsewhere"] else ("/api/yard/node/gateway/remove", {}))
        return out


class Collection(Wizard):
    name, title, empty = "collection", "New Lookup collection", "collections"
    about = "Point Lookup at a folder: its documents are indexed on this machine, and Halo answers from them with citations."
    steps = [
        {"id": "what", "title": "Which folder", "fields": [
            {"key": "folder", "label": "A folder on this machine", "type": "folder", "placeholder": "Browse… or type a folder"},
            {"key": "name", "label": "Name", "type": "text"},
            {"key": "watch", "label": "Keep it current (new and changed files are read again)", "type": "checkbox"}]},
        {"id": "share", "title": "Who else may search it", "fields": [
            {"key": "shared", "label": "Guests (none by default)", "type": "multi", "options_from": "guests"}]},
    ]

    def defaults(self, server):
        return {"watch": True, "shared": []}

    def options(self, server, field, a):
        if field == "guests":
            from nebula import callers
            out = [{"value": f"caller:{c['id']}", "label": c.get("name") or c["id"]} for c in callers.all_of(server.cfg) if c.get("guest")]
            g = getattr(server, "gravity", None)
            out += [{"value": f"peer:{p['id']}", "label": p.get("name") or p["id"][:8]} for p in (g.peers() if g else [])
                    if p.get("role") == "friend"]
            return out
        return []

    def plan(self, server, a):
        _need(a, "folder")
        if getattr(server, "library", None) is None:
            raise WizardError(503, getattr(server, "library_why", None) or "Lookup's library is off")
        p = os.path.expanduser(a["folder"])
        if not os.path.exists(p):
            raise WizardError(400, f"{a['folder']} does not exist on this machine")
        from nebula.rag import ingest
        n = sum(1 for _ in ingest.walk([p]))
        name = a.get("name") or os.path.basename(os.path.abspath(p)) or "Documents"
        return {"lines": [f"Make the collection “{name}” from {p} ({n} file{'s' if n != 1 else ''} it can read)",
                          "Index it on this machine" + (" and keep it current" if a.get("watch", True) else ""),
                          ("Share it with " + ", ".join(a["shared"])) if a.get("shared") else "Share it with no one"],
                "warnings": [] if n else ["Lookup found no file it reads there yet."]}

    def apply(self, server, a, job):
        name = a.get("name") or os.path.basename(os.path.abspath(os.path.expanduser(a["folder"]))) or "Documents"
        c = job.do(f"make the collection “{name}”", "/api/lookup/collections",
                   {"name": name, "folders": [a["folder"]], "watch": bool(a.get("watch", True))})
        job.undos.append((f"/api/lookup/collections/{c['id']}/remove", {}, f"remove the collection “{name}”"))
        if a.get("shared"):
            job.do("share it", f"/api/lookup/collections/{c['id']}/share", {"tenants": a["shared"]})
        return {"collection": c["id"], "name": name}


class EntryPoint(Wizard):
    name, title, empty = "entry", "Add an API entry point", "entry"
    about = ("Another door for apps into the same Event Horizon: on this machine or on your home network (never outside), "
             "with its own keys, the models it offers and the API styles it speaks. Each one is a Quasar on the map.")
    steps = [
        {"id": "who", "title": "Who it is for", "fields": [
            {"key": "name", "label": "Name", "type": "text", "placeholder": "House LAN, Kids, Home Assistant…"},
            {"key": "scope", "label": "Who can reach it", "type": "choice", "options": [
                {"value": "loopback", "label": "Apps on this machine only"},
                {"value": "lan", "label": "Devices on your home network"}]}]},
        {"id": "what", "title": "What it offers", "fields": [
            {"key": "models", "label": "Models (none ticked: all of them)", "type": "multi", "options_from": "models"},
            {"key": "formats", "label": "API styles", "type": "multi", "options": [
                {"value": "openai", "label": "OpenAI (most apps)"}, {"value": "anthropic", "label": "Anthropic (Claude apps)"},
                {"value": "ollama", "label": "Ollama (Open WebUI, Home Assistant, …)"}]},
            {"key": "guest", "label": "Treat it as a guest: only models shared with guests, its own cache", "type": "checkbox"}]},
        {"id": "how", "title": "Port and key", "fields": [
            {"key": "port", "label": "Port", "type": "number"},
            {"key": "make_key", "label": "Require a key, and make its first one now (shown once; recommended)", "type": "checkbox"}]},
    ]

    def defaults(self, server):
        used = {server.server_port} | {e.get("port") for e in server.entries.specs()}
        port = next(p for p in range(18310, 18400) if p not in used)
        return {"scope": "loopback", "formats": ["openai"], "models": [], "port": port, "make_key": True, "guest": False}

    def options(self, server, field, a):
        if field == "models":
            return [{"value": t.name, "label": t.name + (" (cloud)" if t.provider else "")} for t in server.cfg.tiers]
        return []

    def plan(self, server, a):
        _need(a, "name", "formats")
        from nebula.entries import EntryError
        try:
            c = server.entries._check({**a, "require_key": bool(a.get("make_key", True))})
        except EntryError as e:
            raise WizardError(e.status, str(e)) from None
        where = "apps on this machine" if c["scope"] == "loopback" else "devices on your home network (private addresses only)"
        lines = [f"Open the entry point “{c['name']}” on port {c['port']} for {where}",
                 "Offer " + (", ".join(a["models"]) if a.get("models") else "every model") + " in the " + ", ".join(c["formats"]) + " style" + ("s" if len(c["formats"]) > 1 else ""),
                 "Calls without its key bounce off" if c["require_key"] else "Calls need no key"]
        if a.get("make_key", True):
            lines.append("Make its first key (shown once)")
        warn = ["On your network it speaks plain HTTP, like Ollama does: give each device its own key."] if c["scope"] == "lan" else []
        if not c["require_key"]:
            warn.append("No key: anyone on your network can use what it offers." if c["scope"] == "lan" else "No key: any app on this machine can use it.")
        return {"lines": lines, "warnings": warn}

    def apply(self, server, a, job):
        e = job.do(f"open the entry point “{a['name']}”", "/api/api-servers",
                   {"name": a["name"], "scope": a.get("scope") or "loopback", "port": a.get("port"), "formats": a["formats"],
                    "models": a.get("models") or [], "guest": bool(a.get("guest")),
                    "require_key": bool(a.get("make_key", True))})
        job.undos.append((f"/api/api-servers/{e['id']}/remove", {}, f"undo: close “{a['name']}”"))
        out = {"entry": e["id"], "port": e["port"], "listening": e["listening"], "error": e.get("error")}
        if e.get("error"):
            raise WizardError(409, e["error"])
        if a.get("make_key", True):
            k = job.do("make its key", f"/api/api-servers/{e['id']}/keys", {"label": a["name"]})
            out["key"] = k["key"]
        return out


class Routing(Wizard):
    name, title, empty = "routing", "Set up routing", "routing"
    about = ("How Nebula picks a model for each request: what matters most to you, which of your models are the larger "
             "(bigger, slower) ones, and which goes first. Everything here can be changed later on the Routes tab.")
    GOALS = {"power": ("Save power", "Most requests go to everyday models; only clearly hard ones go to a larger one",
                       {"mode": "live", "picker": "efficient_first", "threshold": 0.7}),
             "balanced": ("Balanced", "Easy requests go to everyday models and hard ones to larger ones",
                          {"mode": "live", "picker": "efficient_first", "threshold": 0.5}),
             "best": ("Best answers", "Anything not plainly easy goes to a larger model",
                      {"mode": "live", "picker": "capable_first", "threshold": 0.3}),
             "watch": ("Just watch first", "Nothing changes yet: each request's pick is only written down, so you can see what it would do",
                       {"mode": "shadow", "picker": "efficient_first", "threshold": 0.5}),
             "off": ("Keep it simple", "Every request goes down the list in order; nothing is picked by how hard it looks",
                     {"mode": "off"})}
    steps = [
        {"id": "goal", "title": "What matters most?", "fields": [
            {"key": "goal", "label": "", "type": "choice", "options": [
                {"value": k, "label": f"{v[0]}: {v[1]}"} for k, v in GOALS.items()]}]},
        {"id": "models", "title": "Your models", "fields": [
            {"key": "capable", "label": "Which are the larger ones (bigger, slower, better at hard questions)? The rest are everyday models.",
             "type": "multi", "options_from": "models"},
            {"key": "first", "label": "Which goes first for apps that do not name a model", "type": "select", "options_from": "first"}]},
        {"id": "busy", "title": "When it is busy", "fields": [
            {"key": "busy", "label": "", "type": "choice", "options": [
                {"value": "live", "label": "Use another free model at once (recommended)"},
                {"value": "off", "label": "Wait for the first choice"}]}]},
    ]

    def defaults(self, server):
        from nebula import stage
        sr = stage.settings(_Rules(server.cfg))
        goal = next((k for k, v in self.GOALS.items() if v[2].get("mode") == sr["mode"] and
                     (sr["mode"] == "off" or (v[2]["picker"] == sr["picker"] and abs(v[2]["threshold"] - sr["threshold"]) < 1e-6))), "balanced")
        pol = server.policy.get().get("models") if getattr(server, "policy", None) is not None else {}
        local = [t.name for t in server.cfg.tiers if not t.provider]
        return {"goal": goal, "capable": [m for m in local if (pol.get(m) or {}).get("stage") in ("capable", "larger")],
                "first": next((m for m in local if (pol.get(m) or {}).get("prefer")), local[0] if local else ""),
                "busy": "live" if (server.cfg.rules or {}).get("capacity_mode", "live") != "off" else "off"}

    def options(self, server, field, a):
        if field == "models":
            return [{"value": t.name, "label": t.name + (f" · {t.context // 1024}K context" if t.context else "")}
                    for t in server.cfg.tiers if not t.provider]
        if field == "first":
            return [{"value": t.name, "label": t.name + (" (cloud)" if t.provider else "")} for t in server.cfg.tiers]
        return []

    def plan(self, server, a):
        if not server.cfg.tiers:
            raise WizardError(409, "add a model first: there is nothing to route yet")
        goal = self.GOALS.get(a.get("goal") or "balanced")
        if goal is None:
            raise WizardError(400, "pick what matters most")
        local = [t.name for t in server.cfg.tiers if not t.provider]
        cap = [m for m in a.get("capable") or [] if m in local]
        lines = [f"{goal[0]}: {goal[1].lower()}",
                 ("Larger: " + ", ".join(cap)) if cap else "Every model is an everyday model",
                 ("Everyday: " + ", ".join(m for m in local if m not in cap)) if len(cap) < len(local) else "No everyday model",
                 f"{a.get('first') or local[0] if local else '—'} goes first for apps that do not name a model",
                 "When the first choice is busy: " + ("another free model takes it" if a.get("busy", "live") != "off" else "wait for it")]
        warn = []
        if goal[2]["mode"] in ("live", "shadow") and (not cap or len(cap) == len(local)):
            warn.append("Smart routing needs both kinds: mark some models larger and leave some everyday, or it has nothing to choose between.")
        return {"lines": lines, "warnings": warn}

    def apply(self, server, a, job):
        import copy
        cur = _call(server, "/api/plan", get=True)
        old_pol, old_rules = copy.deepcopy(cur.get("policy") or {}), cur.get("rules") or {}
        local = [t.name for t in server.cfg.tiers if not t.provider]
        cap = set(a.get("capable") or [])
        pol = copy.deepcopy(old_pol)
        for t in server.cfg.tiers:
            p = dict(pol.get(t.name) or {})
            if t.name in local:
                p["stage"] = "larger" if t.name in cap else ("everyday" if p.get("stage") in (None, "capable", "larger", "everyday") else p["stage"])
            if t.name == a.get("first"):
                p["prefer"] = True
            else:
                p.pop("prefer", None)
            pol[t.name] = p
        goal = self.GOALS.get(a.get("goal") or "balanced")
        rules = {"stage_router": {**(old_rules.get("stage_router") or {}), **goal[2]},
                 "capacity_mode": "off" if a.get("busy") == "off" else "live"}
        undo_rules = {"stage_router": old_rules.get("stage_router") or {"mode": "off"}, "capacity_mode": old_rules.get("capacity_mode") or "live"}
        job.do("save the routing", "/api/plan/apply", {"policy": pol, "rules": rules},
               undo=("/api/plan/apply", {"policy": old_pol, "rules": undo_rules}))
        return {"routing": goal[0]}


class _Rules:
    """The config's rules (a plain dict) as nebula/stage.py reads them."""
    def __init__(self, cfg):
        self.stage_router = (cfg.rules or {}).get("stage_router")


WIZARDS = {w.name: w for w in (FirstRun(), DeployOwn(), Routing(), Join(), AddServer(), EntryPoint(), ShareFriend(), Gateway(), Collection())}


# ------------------------------------------------------------------------------------------------ jobs
class Job:
    def __init__(self, server, wizard: Wizard, answers: dict):
        self.id = secrets.token_hex(5)
        self.server, self.wizard, self.answers = server, wizard, answers
        self.state, self.steps, self.result, self.error = "running", [], None, None
        self.undos: list = []                 # (path, body, words) or (callable, None, words), in the order done
        self.undone = False
        self.started = time.time()

    def say(self, words: str, state: str = "done"):
        self.steps.append({"what": words, "state": state, "at": time.time()})

    def do(self, words: str, path: str, body: dict, undo=None):
        self.say(words, "running")
        out = _call(self.server, path, body)
        self.steps[-1]["state"] = "done"
        if undo is not None:
            self.undos.append((undo[0], undo[1], "undo: " + words) if isinstance(undo, tuple) else (undo, None, "undo: " + words))
        return out

    def pull(self, url: str, quant: str, mmproj: bool = False):
        """A Hugging Face download, waited for; the quant matched in the repository. Returns the model's path (and
        the projector's, with `mmproj`)."""
        r = _call(self.server, "/api/yard/hf/resolve", {"url": url})
        choice = next((c for c in r.get("choices") or [] if c["id"] == quant or (c.get("quant") or "").upper() == str(quant).upper()), None)
        if choice is None:
            raise WizardError(404, f"no {quant} in {r.get('repo')}")
        files = list(choice["files"]) + ([r["mmproj"]] if mmproj and r.get("mmproj") else [])
        meta = [f for f in r.get("files") or [] if f["path"] in files]
        pull = self.do(f"download {choice.get('quant') or choice['id']} from {r['repo']}", "/api/yard/hf/pull",
                       {"repo": r["repo"], "revision": r.get("revision") or "main", "files": files, "meta": meta})
        pid = pull.get("id")
        self.undos.append((lambda: _drop_partial(self.server, pid), None, "undo: stop the download and drop what was not finished"))
        end = time.time() + 6 * 3600
        while time.time() < end:
            st = next((p for p in _call(self.server, "/api/yard/hf/pulls", get=True).get("pulls") or [] if p.get("id") == pid), None)
            if st is None or st.get("state") == "done":
                break
            if st.get("state") in ("failed", "error", "paused", "cancelled"):
                raise WizardError(502, "the download stopped: " + (st.get("error") or next((f.get("error") for f in st.get("files") or [] if f.get("error")), None) or st.get("state")))
            self.steps[-1]["progress"] = [st.get("bytes"), st.get("total")]
            time.sleep(0.5)
        self.undos.pop()                                    # finished: the file is kept (it is yours; Undo leaves it)
        targets = {os.path.basename(f["target"]): f["target"] for f in pull.get("files") or []}
        model = targets.get(os.path.basename(choice["files"][0]))
        mm = targets.get(os.path.basename(r["mmproj"])) if mmproj and r.get("mmproj") else None
        return model, mm

    def pull_files(self, repo: str, revision: str, files: list[str], pins: dict | None = None) -> dict:
        """Named files from a Hugging Face repository (at a pinned revision), waited for: {file: its path here}. `pins`
        ({file: sha256}, the catalog's): a file whose LFS sha256 on the Hub is another is refused before the download."""
        r = _call(self.server, "/api/yard/hf/resolve", {"url": f"https://huggingface.co/{repo}" + (f"/tree/{revision}" if revision and revision != "main" else "")})
        meta = [f for f in r.get("files") or [] if f["path"] in files]
        missing = [f for f in files if f not in {m["path"] for m in meta}]
        if missing:
            raise WizardError(404, f"{repo} no longer has {missing[0]}")
        for f in meta:
            want = (pins or {}).get(f["path"])
            if want and f.get("sha256") != want:
                raise WizardError(409, f"{repo}'s {f['path']} is not the file the catalog names (sha256 {str(f.get('sha256'))[:12]}…, "
                                       f"the catalog's {want[:12]}…)")
        gb = sum(m.get("size") or 0 for m in meta) / 1e9
        pull = self.do(f"download {gb:.1f} GB from {repo.split('/')[-1]}", "/api/yard/hf/pull",
                       {"repo": repo, "revision": revision or "main", "files": files, "meta": meta})
        self._wait(pull.get("id"))
        return {os.path.basename(f["path"]): f["target"] for f in pull.get("files") or []} | \
               {f["path"]: f["target"] for f in pull.get("files") or []}

    def _wait(self, pid):
        self.undos.append((lambda: _drop_partial(self.server, pid), None, "undo: stop the download and drop what was not finished"))
        end = time.time() + 12 * 3600
        while time.time() < end:
            st = next((p for p in _call(self.server, "/api/yard/hf/pulls", get=True).get("pulls") or [] if p.get("id") == pid), None)
            if st is None or st.get("state") == "done":
                break
            if st.get("state") in ("failed", "error", "paused", "cancelled"):
                raise WizardError(502, "the download stopped: " + (st.get("error") or next((f.get("error") for f in st.get("files") or [] if f.get("error")), None) or st.get("state")))
            self.steps[-1]["progress"] = [st.get("bytes"), st.get("total")]
            time.sleep(0.5)
        self.undos.pop()                                    # finished: the files are kept (Undo leaves them)

    def deploy(self, spec: dict):
        name = spec["name"]
        self.undos.append((f"/api/models/{name}/remove", {}, f"undo: remove the model {name}"))
        self.undos.append((f"/api/yard/engines/{name}/remove", {}, f"undo: stop and remove the engine {name}"))
        out = self.do(f"start {name}", "/api/yard/deploy", {"engine": spec, "wait": True})
        if (out.get("engine") or {}).get("state") not in ("ready", "running"):
            raise WizardError(502, f"{name} did not start: {(out.get('engine') or {}).get('error') or (out.get('engine') or {}).get('state')}")
        return out

    def test(self, model: str) -> dict:
        """A first message through the Switchyard, as an app would send it."""
        from nebula.dialects import Context, JsonReply
        from nebula.dialects.chat import chat
        srv = self.server
        self.say("send it a test message", "running")
        ctx = Context(srv.cfg, srv.ledger, {"X-Nebula-Caller": "wizard/test"}, srv.engines, srv.tools, input="wizard",
                      caller={"id": None, "label": "wizard/test", "guest": False}, registry=srv.registry, admission=srv.admission,
                      lot=srv.lot, policy=srv.policy, yard=srv.yard, power=srv.power, slots=srv.slots, seclog=srv.seclog, guard=srv.guard, otel=srv.otel)
        try:
            # room for a model that thinks before it answers (Flint decides by itself; 64 tokens were all thinking)
            r = chat(srv.engine, ctx, {"model": model, "stream": False, "max_tokens": 1024,
                                       "messages": [{"role": "user", "content": "Say hello to me in one short sentence."}]})
            msg = ((r.body.get("choices") or [{}])[0].get("message") or {}) if isinstance(r, JsonReply) else {}
            text = msg.get("content") or ("(thinking) " + msg["reasoning_content"][-200:] if msg.get("reasoning_content") else None)
            self.steps[-1]["state"] = "done"
            return {"ok": bool(text), "reply": (text or "")[:400]}
        except Exception as e:  # noqa: BLE001 - the model is added; the test says what went wrong
            self.steps[-1]["state"] = "failed"
            return {"ok": False, "error": str(e)[:300]}

    def run(self):
        try:
            self.result = self.wizard.apply(self.server, self.answers, self)
            self.state = "done"
        except Exception as e:  # noqa: BLE001 - every step before it is taken back
            self.error = f"{e}" if isinstance(e, WizardError) else f"{type(e).__name__}: {e}"
            if self.steps and self.steps[-1]["state"] == "running":
                self.steps[-1]["state"] = "failed"
            self.rollback()
            self.state = "failed"
        finally:
            self.forget()

    def forget(self):
        """Passwords are for the job's own steps: gone from its answers once it has run (Undo needs none)."""
        for k in [k for k in self.answers if "password" in k]:
            self.answers.pop(k, None)

    def rollback(self):
        for path, body, words in reversed(self.undos):
            try:
                path() if callable(path) else _call(self.server, path, body)
                self.say(words)
            except WizardError as e:
                if e.status == 404:                         # already gone (it never got that far)
                    continue
                self.say(f"{words} — could not: {e}", "failed")
            except Exception as e:  # noqa: BLE001
                self.say(f"{words} — could not: {e}", "failed")
        self.undos = []
        self.undone = True

    def view(self) -> dict:
        return {"id": self.id, "wizard": self.wizard.name, "state": self.state, "steps": self.steps, "result": self.result,
                "error": self.error, "can_undo": self.state == "done" and not self.undone and bool(self.undos), "undone": self.undone}


def _drop_partial(server, pid: str) -> None:
    """A download stopped half way: cancelled, and its unfinished files removed."""
    try:
        v = _call(server, f"/api/yard/hf/pulls/{pid}/cancel", {})
    except WizardError:
        v = {}
    for f in v.get("files") or []:
        if f.get("state") != "done" and f.get("target"):
            for p in (f["target"], f["target"] + ".part"):
                try:
                    os.remove(p)
                except OSError:
                    pass


class Wizards:
    def __init__(self, server):
        self.server = server
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def list(self) -> list:
        return [{"name": w.name, "title": w.title, "about": w.about, "empty": w.empty} for w in WIZARDS.values()]

    def get(self, name: str) -> Wizard:
        w = WIZARDS.get(name)
        if w is None:
            raise WizardError(404, f"no wizard {name!r}")
        return w

    def apply(self, name: str, answers: dict, wait: bool = False) -> dict:
        w = self.get(name)
        plan = w.plan(self.server, answers)                       # checked again: the answers may have been changed
        unconfirmed = [m for m in plan.get("confirm") or [] if m not in (answers.get("confirm") or [])]
        if unconfirmed:
            raise WizardError(409, "confirm: " + ", ".join(unconfirmed), needs_confirm=unconfirmed)
        job = Job(self.server, w, answers)
        with self.lock:
            self.jobs[job.id] = job
        t = threading.Thread(target=job.run, name=f"wizard-{name}", daemon=True)
        t.start()
        if wait:
            t.join()
        return job.view()

    def job(self, jid: str) -> Job:
        j = self.jobs.get(jid)
        if j is None:
            raise WizardError(404, f"no job {jid}")
        return j

    def undo(self, jid: str) -> dict:
        j = self.job(jid)
        if j.state != "done" or j.undone:
            raise WizardError(409, "nothing to undo")
        j.rollback()
        return j.view()


def get(server, path: str):
    wz = server.wizards
    if path == "/api/wizards":
        return {"wizards": wz.list()}
    if path.startswith("/api/wizards/jobs/"):
        return wz.job(path[len("/api/wizards/jobs/"):]).view()
    if path.startswith("/api/wizards/"):
        return wz.get(path[len("/api/wizards/"):]).describe(server)
    return None


def post(server, path: str, req: dict):
    wz = server.wizards
    if path.startswith("/api/wizards/jobs/") and path.endswith("/undo"):
        return wz.undo(path[len("/api/wizards/jobs/"):-len("/undo")])
    name, _, action = path[len("/api/wizards/"):].partition("/")
    answers = req.get("answers") if isinstance(req.get("answers"), dict) else {}
    w = wz.get(name)
    if action == "options":
        return {"options": w.options(server, str(req.get("field") or ""), answers)}
    if action == "plan":
        return w.plan(server, answers)
    if action == "apply":
        return wz.apply(name, answers, wait=bool(req.get("wait")))
    raise WizardError(404, f"no wizard action {action!r}")
