"""nebula/engines.py: the engines this Nebula runs itself (its own Switchyard: engine lifecycle).

An engine is an Impulse (impulse-server) or llama.cpp server Nebula starts and stops, described in the config's
`engines` list. Its **deployment** is that description; every change makes a new version, the earlier ones are kept
beside the config (`NAME.deployments.json`) and Undo goes back one.

    {"engines": [{"name": "small", "kind": "impulse", "binary": "impulse-server", "model": "models/small.gguf",
                  "host": "127.0.0.1", "port": 8080, "context": 32768, "slots": 2, "cache_type": "q8_0",
                  "flags": ["--flash-attn", "on"], "card": "GPU-…", "env": {}, "autostart": false}]}

- `kind` "impulse" and "llama.cpp" build the command from the fields (`binary` is a path, or a list such as an
  interpreter and its script) (`-m`, `--host`, `--port`, `-c` = context ×
  slots, `-np` = slots, `-ctk/-ctv` = cache_type, then `flags`); "command" runs `command` as given.
- `cpus` (optional) pins the engine to those cores ("9-17,45-53", the cores beside its cards on a two-socket
  machine): it runs under `taskset`, in a sandbox too. Memory cannot be bound to a NUMA node in a sandbox (OpenShell
  refuses set_mempolicy); pinned threads mostly allocate near their cores anyway.
- `numa` (optional, "auto" or "off"): in a sandbox, when the CPUs the engine may use (its `cpus`, else every CPU) lie on
  two or more sockets, Nebula reads the machine's layout and passes it as IMPULSE_NUMA_NODES, so the engine's CPU
  expert pool splits per socket even where the sandbox hides /sys (Impulse, Strata numa-always). "off" passes
  STRATA_NUMA_POOL=0 instead. Give a multi-socket engine both sockets' CPUs (no `cpus`, or a list on both) to use it.
- `card` is a GPU UUID or index, or several comma separated ("GPU-…,GPU-…", for a model split across cards with the
  tensor_split option); it becomes CUDA_VISIBLE_DEVICES. A start refuses a card this machine does not have.
- `context` is per slot. The fit estimate reads the GGUF header (layers, KV heads, head size) for the KV cache and
  adds the file's size for the weights; a start refuses an engine that cannot fit the card's free memory (its cards'
  together, for several).
- Lifecycle: start, stop, **drain** (routing sends it nothing new; the running calls finish, then it stops), restart,
  verify (it answers, and `/props` names the deployment's model file). An engine that already answers at its
  address is attached, never started twice.

Cards come from `nvidia-smi`; with `NEBULA_FAKE_CARDS` (a JSON list of {"uuid", "index", "name", "memory_mb",
"used_mb"}) they come from there instead, for machines without a GPU (tests, the greenfield container).
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path

from nebula.app.engine import EngineProcess, health

KINDS = ("impulse", "llama.cpp", "command")        # the first two are packages (nebula/packages.py)
REMOVED_KINDS = {"strata": "impulse"}             # the owner, 2026-10-25: "Yes. Remove it." (Strata's package); its engines load as Impulse
FIELDS = ("name", "kind", "binary", "model", "host", "port", "context", "slots", "cache_type", "flags", "card", "cpus", "env",
          "command", "cwd", "autostart", "ready_timeout_s", "stop_timeout_s", "note", "options", "source", "where", "guests", "image", "machine", "folder", "memory", "cpu_limit", "numa",
          "long_context", "vulkan_card")
# A Flash-Next engine's opt-in long context (impulse-fx --long-context: dynamic YaRN past the trained 262,144 tokens).  Off
# when absent - the default for every engine; nebula/flashnext.py long_context_cost says what each size costs.
LONG_CONTEXT = ("512k", "768k", "1m")
WHERE = ("local", "openshell")      # none given: in a sandbox when this machine can, else on it with a warning
CACHE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
               "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
OVERHEAD_MB = 600                                 # compute buffers and the CUDA context, roughly


class EngineSpecError(ValueError):
    pass


def is_fx_command(command) -> bool:
    """A command that runs impulse-fx (Impulse's server for a Flash-Next pack)."""
    words = command.split() if isinstance(command, str) else [str(x) for x in (command or [])]
    return any(os.path.basename(w) == "impulse-fx" for w in words[:3])


def check_spec(spec: dict) -> dict:
    """A deployment, checked and normalised (unknown fields refused, so a typo is not silently ignored)."""
    if not isinstance(spec, dict) or not str(spec.get("name") or "").strip():
        raise EngineSpecError('an engine is an object with a "name"')
    bad = [k for k in spec if k not in FIELDS]
    if bad:
        raise EngineSpecError(f"engine {spec['name']}: unknown field {bad[0]!r}")
    s = {k: copy.deepcopy(v) for k, v in spec.items()}
    s["kind"] = s.get("kind") or ("command" if s.get("command") else "impulse")
    if s["kind"] in REMOVED_KINDS:                       # a saved engine of a removed package: loads, as its successor
        old = s["kind"]
        s["kind"] = REMOVED_KINDS[old]
        s.pop("binary", None)                            # the old server's program: Impulse finds its own
        if isinstance(s.get("options"), dict):           # settings only the old package had
            s["options"] = {k: v for k, v in s["options"].items() if k not in ("vram_reserve_mib", "adapt_async")}
        note = f"moved from the removed {old} package to {s['kind']} (2026-10-25)"
        s["note"] = f"{s['note']} | {note}" if s.get("note") else note
        logging.getLogger("nebula.engines").warning("engine %s: %s", s["name"], note)
    if s["kind"] not in KINDS:
        raise EngineSpecError(f"engine {s['name']}: kind is one of {', '.join(KINDS)}")
    if s.get("long_context") in ("", None, "off", False):
        s.pop("long_context", None)
    elif s.get("long_context") not in LONG_CONTEXT:
        raise EngineSpecError(f"engine {s['name']}: long_context is off, {', '.join(LONG_CONTEXT)}")
    elif not (s["kind"] == "command" and is_fx_command(s.get("command"))):
        raise EngineSpecError(f"engine {s['name']}: long_context is for a Flash-Next engine (impulse-fx)")
    if s["kind"] == "command":
        if not s.get("command"):
            raise EngineSpecError(f"engine {s['name']}: a command engine needs a command")
    elif not s.get("model"):
        raise EngineSpecError(f"engine {s['name']}: an {s['kind']} engine needs a model file")
    elif s.get("binary") is not None and not (isinstance(s["binary"], str) or (isinstance(s["binary"], list) and all(isinstance(x, str) for x in s["binary"]))):
        raise EngineSpecError(f"engine {s['name']}: binary is a path, or a list of words (an interpreter and its script)")
    try:
        s["port"] = int(s.get("port") or 0)
    except (TypeError, ValueError):
        raise EngineSpecError(f"engine {s['name']}: port is a number") from None
    if not 0 < s["port"] < 65536:
        raise EngineSpecError(f"engine {s['name']}: a port between 1 and 65535 is needed")
    for k, lo in (("context", 256), ("slots", 1)):
        if s.get(k) is not None:
            try:
                s[k] = int(s[k])
            except (TypeError, ValueError):
                raise EngineSpecError(f"engine {s['name']}: {k} is a number") from None
            if s[k] < lo:
                raise EngineSpecError(f"engine {s['name']}: {k} is at least {lo}")
    if s.get("cpus") is not None and not re.fullmatch(r"\d+(-\d+)?(,\d+(-\d+)?)*", str(s["cpus"]).replace(" ", "")):
        raise EngineSpecError(f"engine {s['name']}: cpus is a list of cores, such as 0-7 or 9-17,45-53")
    if s.get("cpus") is not None:
        s["cpus"] = str(s["cpus"]).replace(" ", "")
    if s.get("numa") is not None and s["numa"] not in ("auto", "off"):
        raise EngineSpecError(f"engine {s['name']}: numa is auto (split the CPU expert pool per socket when the engine's "
                              "CPUs span two) or off")
    if s.get("cache_type") and s["cache_type"] not in CACHE_BYTES:
        raise EngineSpecError(f"engine {s['name']}: cache_type is one of {', '.join(CACHE_BYTES)}")
    if s.get("flags") is not None and not (isinstance(s["flags"], list) and all(isinstance(f, (str, int, float)) for f in s["flags"])):
        raise EngineSpecError(f"engine {s['name']}: flags is a list of words")
    s["host"] = s.get("host") or "127.0.0.1"
    if s["host"] not in ("127.0.0.1", "localhost") and s["kind"] != "command":
        raise EngineSpecError(f"engine {s['name']}: an engine Nebula runs listens on this machine only (127.0.0.1); "
                              "others reach it through Nebula")
    if s.get("memory") is not None and not re.match(r"^\d+(\.\d+)?(Ki|Mi|Gi|Ti|K|M|G|T)?$", str(s["memory"])):
        raise EngineSpecError(f"engine {s['name']}: memory is a size such as 64Gi (its sandbox's limit)")
    container = s["kind"] == "command" and bool(s.get("image"))    # a container engine: any image and its command, protected
    if container and s.get("folder") and not os.path.isabs(str(s["folder"])):
        raise EngineSpecError(f"engine {s['name']}: folder is a full path (mounted read-only at /models)")
    if s.get("machine"):                                   # on a model machine (nebula/machines.py): a sandbox there
        if s["kind"] == "command" and not container:
            raise EngineSpecError(f"engine {s['name']}: a command engine runs on this machine only, unless it has an image")
        if not container and not os.path.isabs(str(s.get("model") or "")):
            raise EngineSpecError(f"engine {s['name']}: on {s['machine']}, the model is that machine's file (a full path there)")
        s["machine"] = str(s["machine"])
        s["where"] = "openshell"
    if s.get("where") is None:
        s.pop("where", None)                               # chosen when it starts (sandbox_plan)
    elif s["where"] not in WHERE:
        raise EngineSpecError(f"engine {s['name']}: where is one of {', '.join(WHERE)}")
    elif s["where"] == "local":
        pass                                               # the owner's choice: no sandbox (shown with a warning)
    elif s["kind"] == "command" and not container:
        raise EngineSpecError(f"engine {s['name']}: a command engine runs on this machine only, unless it has an image")
    elif not container:
        from nebula import packages
        pk = packages.PACKAGES[s["kind"]]
        if not (s.get("image") or pk.get("image") or pk.get("image_from_binary")):
            raise EngineSpecError(f"engine {s['name']}: an engine in a sandbox needs the image it runs from (image)")
    if s["kind"] != "command":
        from nebula import packages
        try:
            s["options"] = packages.check(s["kind"], s.get("options") or {})
        except packages.OptionError as e:
            raise EngineSpecError(f"engine {s['name']}: {e}") from None
        if not s["options"]:
            s.pop("options", None)
    return s


def sandbox_plan(spec: dict, exe: str | None = None, conf_dir: str | None = None) -> tuple[str, str | None]:
    """("openshell", None), or ("local", why it runs without a sandbox). Engines go into
    OpenShell sandboxes by default (on Windows: a Podman container in WSL2, nebula/wslbox.py); without one they still
    run, and they and what is shared from them carry a warning."""
    from nebula import openshell, packages
    if spec.get("kind") == "command" and not spec.get("image"):
        return "local", "a command engine runs as you, on this machine"
    if spec.get("kind") == "command":
        return "openshell", None                         # a container engine: its image, always in a sandbox
    if spec.get("where") == "local":
        return "local", "you chose to run it without a sandbox"
    if spec.get("where") == "openshell":                   # insisted on: in a sandbox, or not at all
        return "openshell", None
    pk = packages.PACKAGES.get(spec.get("kind")) or {}
    if not (spec.get("image") or pk.get("image") or pk.get("image_from_binary")):
        return "local", f"there is no sandbox image for {packages.PACKAGES.get(spec.get('kind'), {}).get('label') or spec.get('kind')} yet"
    if os.name == "nt":                                    # Windows: WSL2 + Podman, when Setup made it
        from nebula import wslbox
        why = wslbox.why_not(conf_dir)
        if not why and pk.get("image_from_binary") and not spec.get("image") and not wslbox.image_for(conf_dir, spec.get("kind")):
            why = f"the Safe sandbox has no image for {pk.get('label') or spec.get('kind')}: run Setup again"
        return ("local", why) if why else ("openshell", None)
    why = openshell.why_not(exe)
    return ("local", why) if why else ("openshell", None)


def address_of(spec: dict) -> str:
    return f"http://{spec.get('host') or '127.0.0.1'}:{spec['port']}"


def key_path(spec: dict, cfg) -> str | None:
    """The file holding the random key only Nebula knows for this engine (when its package takes one)."""
    from nebula import names, packages
    if spec.get("kind") == "command" or not packages.PACKAGES[spec["kind"]].get("api_key") or not getattr(cfg, "path", None):
        return None
    return os.path.join(names.local_path(cfg.base_dir, "nebula-keys", "cockpit-keys"), f"engine-{spec['name']}.key")


def ensure_key(path: str) -> str:
    import secrets
    try:
        k = open(path, encoding="utf-8").read().strip()
        if k:
            return k
    except OSError:
        pass
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    k = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(k + "\n")
    return k


def flashnext_piece(spec: dict, base_dir: str | None) -> list[str] | None:
    """A Flash-Next pack on Impulse runs in impulse-fx, which impulse-server runs beside itself: that piece's files
    (nebula/pieces.py, downloaded by the Flash-Next setup), or None for any other model. A pack without the piece is
    refused, saying what to download."""
    if spec.get("kind") != "impulse" or not spec.get("model"):
        return None
    from nebula import flashnext, pieces
    model = str(spec["model"])
    model = model if os.path.isabs(model) else os.path.join(base_dir or os.getcwd(), model)
    root = model if os.path.isdir(model) else os.path.dirname(model)
    if not flashnext.is_pack(root):
        return None
    exe = pieces.find("impulse-flashnext", base_dir)
    if exe is None:
        raise EngineSpecError(f"engine {spec['name']}: a Flash-Next pack runs on Impulse Flash-Next, which is not here yet "
                              f"(Run a model › a Flash-Next pack: it {pieces.say('impulse-flashnext')})")
    d = os.path.dirname(exe)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and not f.startswith("LICENSE"))


def _pack_of(model: str) -> dict | None:
    """A Flash-Next pack's files (flashnext.pack_files) when `model` is one (its folder or its dense GGUF), else None."""
    from nebula import flashnext
    return flashnext.pack_files(model if os.path.isdir(model) else os.path.dirname(model))


def _piece_profile(base_dir: str | None) -> str | None:
    """The expert profile the Impulse Flash-Next piece carries beside impulse-fx (nebula/pieces.py), if it is here."""
    from nebula import flashnext, pieces
    exe = pieces.find("impulse-flashnext", base_dir)
    p = exe and os.path.join(os.path.dirname(exe), flashnext.PIECE_PROFILE)
    return p if p and os.path.isfile(p) else None


def command_of(spec: dict, base_dir: str | None = None, key_file: str | None = None) -> list[str]:
    """The argv a deployment runs: its settings, then the package's hardening (its chat page off; its /slots page off
    unless guests use it; no prompt cache across slots when they do; the key only Nebula holds), then the extra flags."""
    if spec["kind"] == "command":
        c = spec["command"]
        argv = c.split() if isinstance(c, str) else [str(x) for x in c]
        if spec.get("long_context") in LONG_CONTEXT and "--long-context" not in argv:
            at = argv.index("--fx-args") if "--fx-args" in argv else len(argv)   # (--fx-args takes the rest of the line)
            argv[at:at] = ["--long-context", spec["long_context"]]   # opt-in (off when absent): dynamic YaRN past 262,144
        return argv
    model = spec["model"]
    if base_dir and not os.path.isabs(model):
        model = str(Path(base_dir, model))
    slots = int(spec.get("slots") or 1)
    from nebula import packages
    b = spec.get("binary") or packages.locate(spec["kind"], base_dir) or packages.PACKAGES[spec["kind"]]["binary"]
    binary = list(b) if isinstance(b, list) else [str(b)]
    argv = binary + ["-m", model, "--host", spec["host"], "--port", str(spec["port"])]
    fx_files = spec.get("fx_files") or (_pack_of(model) if spec["kind"] == "impulse" else None)
    if spec["kind"] == "impulse" and fx_files:
        # A Flash-Next pack: impulse-server hands it to impulse-fx, which takes its own words (no -c/-np/--no-webui;
        # the context comes from the pack, and impulse-fx has no chat page) and a run config naming all the pack's files
        from nebula import flashnext
        files = fx_files
        gpus = len([c for c in str(spec.get("card") or "").split(",") if c.strip()]) or 1
        profile = files.get("profile") or _piece_profile(base_dir)
        cfg = flashnext.local_config({k: v for k, v in files.items() if k != "profile"}, name=spec["name"].lower(),
                                     port=int(spec["port"]), gpus=gpus, parallel=slots, profile=profile)
        argv += ["--fx-config-json", json.dumps(cfg, separators=(",", ":"))]
        if spec.get("long_context") in LONG_CONTEXT:
            argv += ["--long-context", spec["long_context"]]
        argv += [str(f) for f in spec.get("flags") or []]
        return (["taskset", "-c", spec["cpus"]] if spec.get("cpus") else []) + argv
    if spec.get("context"):
        argv += ["-c", str(int(spec["context"]) * slots)]
    argv += ["-np", str(slots)]
    opts = dict(spec.get("options") or {})
    if spec.get("cache_type"):                    # the short form: both caches the same, unless the options say otherwise
        opts.setdefault("cache_type_k", spec["cache_type"])
        opts.setdefault("cache_type_v", spec["cache_type"])
    for k in ("mmproj_file", "lora", "mtp", "draft_model", "chat_template_file"):
        if opts.get(k) and base_dir and not os.path.isabs(opts[k]):
            opts[k] = str(Path(base_dir, opts[k]))
    try:
        argv += packages.args(spec["kind"], opts, model, binary)
    except packages.OptionError as e:
        raise EngineSpecError(f"engine {spec['name']}: {e}") from None
    pk = packages.PACKAGES[spec["kind"]]
    argv += list(pk.get("harden") or []) + list(pk.get("shared" if spec.get("guests") else "harden_private") or [])
    if key_file and pk.get("api_key"):
        if pk["api_key"] == "--api-key-file":
            argv += ["--api-key-file", key_file]
        else:
            argv += [pk["api_key"], "@" + key_file]          # shown as the file; the key itself is filled in at start
    if spec.get("vulkan_card"):                      # Impulse portable on one card through Vulkan, by its name (its number
        from nebula import catalog as C              # changes when another card comes or goes): found now
        vk = next((c for c in C.vulkan_cards(base_dir) if c["name"] == spec["vulkan_card"]), None)
        if vk is None:
            raise EngineSpecError(f"engine {spec['name']}: the card {spec['vulkan_card']} does not answer through Vulkan now")
        argv += ["--device", vk["id"]]
    argv += [str(f) for f in spec.get("flags") or []]
    return (["taskset", "-c", spec["cpus"]] if spec.get("cpus") else []) + argv


# --- cards
def cards() -> list[dict]:
    """This machine's GPUs: {"uuid", "index", "name", "memory_mb", "used_mb"}; [] without nvidia-smi."""
    fake = os.environ.get("NEBULA_FAKE_CARDS")
    if fake:
        try:
            return [dict(c) for c in json.loads(fake)]
        except ValueError:
            return []
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=uuid,index,name,memory.total,memory.used,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            try:
                rows.append({"uuid": p[0], "index": int(p[1]), "name": p[2], "memory_mb": int(float(p[3])), "used_mb": int(float(p[4])),
                             **({"compute_cap": p[5]} if len(p) > 5 else {})})
            except ValueError:
                continue
    return rows


def cards_of(spec: dict, have: list[dict]) -> list[dict]:
    """The engine's cards, in its order ({"missing": …} for one this machine does not have); [] when none is chosen."""
    want = spec.get("card")
    if want is None or want == "":
        return []
    out = []
    for w in (x.strip() for x in str(want).split(",") if x.strip()):
        out.append(next((c for c in have if w in (c.get("uuid"), str(c.get("index")))), None) or {"missing": w})
    return out


def card_of(spec: dict, have: list[dict]) -> dict | None:
    """The engine's first card (its only one, usually)."""
    cs = cards_of(spec, have)
    return next((c for c in cs if c.get("missing")), cs[0]) if cs else None


# --- the fit estimate (GGUF header)
_GGUF_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def gguf_meta(path: str, wanted=("block_count", "attention.head_count_kv", "attention.head_count", "attention.key_length",
                                 "attention.value_length", "embedding_length", "general.architecture", "expert_count",
                                 "attention.sliding_window", "attention.sliding_window_pattern", "attention.key_length_swa",
                                 "attention.value_length_swa", "attention.shared_kv_layers")) -> dict:
    """The few metadata values the fit estimate needs, read from a GGUF file's header (a short array of numbers, such
    as a per-layer head count or sliding-window pattern, is read as a list; other arrays are skipped)."""
    out: dict = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version < 2:
            raise ValueError("GGUF version 1 is not read")

        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                if et in _GGUF_TYPES and n <= 4096:
                    fmt = "<" + _GGUF_TYPES[et] * n
                    return list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
                if et in _GGUF_TYPES:
                    f.seek(n * struct.calcsize("<" + _GGUF_TYPES[et]), 1)
                else:
                    for _ in range(n):
                        rval(et)
                return None
            fmt = "<" + _GGUF_TYPES[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        for _ in range(n_kv):
            key = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            v = rval(t)
            if key == "general.architecture" or any(key.endswith("." + w) for w in wanted):
                out[key.split(".", 1)[1] if key != "general.architecture" else key] = v
    return out


def _kv_mb_per_layer(spec: dict, m: dict, layers: int) -> int | None:
    """The KV cache for a model whose layers differ: per-layer KV heads, sliding-window layers (they keep only their
    window per slot), and the last layers sharing an earlier layer's cache (they keep none)."""
    heads = m.get("attention.head_count_kv")
    heads = heads if isinstance(heads, list) else [heads or m.get("attention.head_count")] * layers
    swa = m.get("attention.sliding_window_pattern")
    swa = [bool(x) for x in swa] if isinstance(swa, list) else [False] * layers
    if len(heads) != layers or len(swa) != layers or not all(isinstance(h, int) for h in heads):
        return None
    head = m.get("attention.key_length") or ((m.get("embedding_length") or 0) // (m.get("attention.head_count") or 1) or None)
    vhead = m.get("attention.value_length") or head
    head_swa, vhead_swa = m.get("attention.key_length_swa") or head, m.get("attention.value_length_swa") or vhead
    window = int(m.get("attention.sliding_window") or 0)
    shared = int(m.get("attention.shared_kv_layers") or 0)
    o = spec.get("options") or {}
    kb = CACHE_BYTES.get(o.get("cache_type_k") or spec.get("cache_type") or "f16", 2.0)
    vb = CACHE_BYTES.get(o.get("cache_type_v") or spec.get("cache_type") or "f16", 2.0)
    ctx, slots = int(spec.get("context") or 4096), int(spec.get("slots") or 1)
    total = 0.0
    for i in range(layers - shared):
        if not heads[i]:
            continue
        if swa[i] and window:
            total += heads[i] * (head_swa * kb + vhead_swa * vb) * min(window + 512, ctx) * slots
        else:
            total += heads[i] * (head * kb + vhead * vb) * ctx * slots
    return round(total / 2**20)


def gguf_tensor_bytes(path: str) -> dict:
    """A GGUF's tensor bytes split as Impulse places them: {"experts": the routed experts' (`*_exps*`, which stay in RAM
    with a cache of them on the card), "other": everything else (on the card), "expert_layers": the layers with experts}.
    Each tensor's size is the distance to the next one's offset (no table of quant formats needed)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version < 2:
            raise ValueError("GGUF version 1 is not read")
        rstr = lambda: f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
        align = 32

        def skip(t):
            if t == 8:
                rstr()
            elif t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et in _GGUF_TYPES:
                    f.seek(n * struct.calcsize("<" + _GGUF_TYPES[et]), 1)
                else:
                    for _ in range(n):
                        skip(et)
            else:
                f.seek(struct.calcsize("<" + _GGUF_TYPES[t]), 1)

        for _ in range(n_kv):
            key = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment" and t == 4:
                (align,) = struct.unpack("<I", f.read(4))
            else:
                skip(t)
        tensors = []
        for _ in range(n_tensors):
            name = rstr()
            (nd,) = struct.unpack("<I", f.read(4))
            f.seek(8 * nd + 4, 1)                                # dims, type
            (off,) = struct.unpack("<Q", f.read(8))
            tensors.append((off, name))
        data = (f.tell() + align - 1) // align * align
    end = os.path.getsize(path) - data
    tensors.sort()
    out = {"experts": 0, "other": 0, "expert_layers": set()}
    for i, (off, name) in enumerate(tensors):
        size = (tensors[i + 1][0] if i + 1 < len(tensors) else end) - off
        if "_exps" in name:
            out["experts"] += size
            out["expert_layers"].add(name.split(".")[1] if name.startswith("blk.") else name)
        else:
            out["other"] += size
    out["expert_layers"] = len(out["expert_layers"])
    return out


RAM_OVERHEAD_MB = 3072                            # the engine's own RAM beside the experts (pool buffers, the scheduler's)
MIN_SLOTS = 8                                     # Impulse's smallest expert cache per layer (kMinSlots)


def ram_free_mb() -> int | None:
    """RAM free for a new engine (MemAvailable), in MB; NEBULA_FAKE_RAM_FREE_MB for tests."""
    if os.environ.get("NEBULA_FAKE_RAM_FREE_MB"):
        return int(os.environ["NEBULA_FAKE_RAM_FREE_MB"])
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) // 1024
    except OSError:
        pass
    return None


def in_wsl() -> bool:
    """Linux under Windows' WSL2 (its RAM is a share of the PC's: half by default); NEBULA_FAKE_WSL=1 for tests."""
    if os.environ.get("NEBULA_FAKE_WSL") is not None:
        return os.environ["NEBULA_FAKE_WSL"] == "1"
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _fit_experts_in_ram(spec: dict, model: str, m: dict, out: dict, have: list[dict]) -> dict:
    """Flint on Impulse: on the card the weights that are not routed experts, the KV cache for the context (every layer
    keeps every cell, the sliding layers too, until the compact window cache) and the compute buffers; in RAM every
    routed expert, with a cache of them in the card's memory that is left (at least 8 per layer). With `n_cpu_moe` 0 on
    several cards (all on the cards) the experts go on the cards instead and RAM needs none of them. RAM also holds twice the
    KV cache for the prompt cache (conversations parked in RAM)."""
    try:
        tb = gguf_tensor_bytes(model)
    except (OSError, ValueError, struct.error) as e:
        out["why"] = f"the model file cannot be read: {e}"
        return out
    o = spec.get("options") or {}
    mb = lambda b: round(b / 2**20)
    layers, kv_heads = m.get("block_count") or 0, m.get("attention.head_count_kv") or m.get("attention.head_count") or 0
    head = m.get("attention.key_length") or 128
    vhead = m.get("attention.value_length") or head
    kt = o.get("cache_type_k") or spec.get("cache_type") or "q4_0"    # Impulse's own default for this engine
    vt = o.get("cache_type_v") or spec.get("cache_type") or "q4_0"
    tokens = int(spec.get("context") or 4096) * int(spec.get("slots") or 1)
    out["kv_mb"] = mb(layers * kv_heads * (head * CACHE_BYTES.get(kt, 2.0) + vhead * CACHE_BYTES.get(vt, 2.0)) * tokens)
    all_on_cards = int(o.get("n_cpu_moe", -1)) == 0
    per_slot = tb["experts"] / max(1, (m.get("expert_count") or 1) * max(1, tb["expert_layers"]))   # one expert, one layer
    min_cache = 0 if all_on_cards else MIN_SLOTS * per_slot * tb["expert_layers"]
    out["weights_mb"] = mb(tb["other"] + (tb["experts"] if all_on_cards else 0))
    out["need_mb"] = out["weights_mb"] + out["kv_mb"] + OVERHEAD_MB + mb(min_cache)
    # the prompt cache keeps conversations' KV in RAM too (measured: the locked RAM grows ~5 GB from 8K to 64K)
    out["ram_need_mb"] = (0 if all_on_cards else mb(tb["experts"]) + RAM_OVERHEAD_MB) + 2 * out["kv_mb"]
    out["ram_free_mb"] = ram_free_mb()
    if out["ram_need_mb"] and out["ram_free_mb"] is not None and out["ram_free_mb"] < out["ram_need_mb"]:
        out["fits"] = False
        out["why"] = (f"needs about {round(out['ram_need_mb'] / 1024)} GB of free RAM for its experts, and this computer has "
                      f"{round(out['ram_free_mb'] / 1024)} GB free")
        if in_wsl():   # WSL2 gives Linux half the PC's RAM unless told otherwise; the page offers to raise it (wslmem.py)
            out["wsl_fix"] = True
            out["why"] += (". Nebula runs in WSL, which gets half of the PC's RAM by default: raise it with a memory= line under "
                           "[wsl2] in %UserProfile%\\.wslconfig (for example memory=48GB), run wsl --shutdown, and start Nebula again")
        return out
    cs = cards_of(spec, have)
    gone = next((c for c in cs if c.get("missing")), None)
    if gone:
        out["card"], out["fits"], out["why"] = gone["missing"], False, f"this machine has no card {gone['missing']}"
        return out
    cs = cs or sorted(have, key=lambda c: int(c.get("memory_mb", 0)) - int(c.get("used_mb", 0)), reverse=True)[:1]
    if not cs:
        out["why"] = "no card here"
        return out
    out["card"] = ",".join(str(c.get("uuid")) for c in cs)
    out["free_mb"] = sum(int(c.get("memory_mb", 0)) - int(c.get("used_mb", 0)) for c in cs)
    out["fits"] = out["need_mb"] <= out["free_mb"]
    if not all_on_cards:   # what is left on the card for the expert cache, as a share of all the experts
        left = max(0, out["free_mb"] - out["need_mb"] + mb(min_cache))
        out["expert_cache_mb"] = left
        out["expert_cache_share"] = round(min(1.0, left / max(1, mb(tb["experts"]))), 2)
    where = "the card has" if len(cs) == 1 else f"its {len(cs)} cards have"
    out["why"] = None if out["fits"] else (f"needs about {out['need_mb']} MB on the card for this context "
                                           f"({out['kv_mb']} MB of it the KV cache), {where} {out['free_mb']} MB free")
    return out


def fit(spec: dict, base_dir: str | None = None, have: list[dict] | None = None) -> dict:
    """{"weights_mb", "kv_mb", "need_mb", "card", "free_mb", "fits", "why"}: what the deployment needs against its card."""
    out: dict = {"weights_mb": None, "kv_mb": None, "need_mb": None, "card": None, "free_mb": None, "fits": None, "why": None}
    if spec.get("kind") == "command":
        out["why"] = "a command engine: its size is not known"
        return out
    model = spec.get("model") or ""
    if base_dir and model and not os.path.isabs(model):
        model = str(Path(base_dir, model))
    try:
        out["weights_mb"] = round(os.path.getsize(model) / 2**20)
        m = gguf_meta(model)
    except (OSError, ValueError, struct.error) as e:
        out["why"] = f"the model file cannot be read: {e}"
        return out
    if spec.get("kind") == "impulse" and m.get("general.architecture") == "flint":
        return _fit_experts_in_ram(spec, model, m, out, cards() if have is None else have)
    layers = m.get("block_count")
    if layers and (isinstance(m.get("attention.head_count_kv"), list) or isinstance(m.get("attention.sliding_window_pattern"), list)):
        out["kv_mb"] = _kv_mb_per_layer(spec, m, layers)        # Gemma 4 and kin: per-layer heads, sliding-window layers
        if out["kv_mb"] is None:
            out["why"] = "its per-layer attention settings are not ones this estimate reads, so its card need is not estimated"
            return out
    if out["kv_mb"] is not None:
        layers = None                                          # done above
    if not layers and out["kv_mb"] is None:
        layers = m.get("block_count")
    kv_heads = m.get("attention.head_count_kv") or m.get("attention.head_count")
    head = m.get("attention.key_length") or ((m.get("embedding_length") or 0) // (m.get("attention.head_count") or 1) or None)
    vhead = m.get("attention.value_length") or head
    if layers and kv_heads and head:
        o = spec.get("options") or {}
        kt = o.get("cache_type_k") or spec.get("cache_type") or "f16"
        vt = o.get("cache_type_v") or spec.get("cache_type") or "f16"
        per_token = layers * kv_heads * (head * CACHE_BYTES.get(kt, 2.0) + vhead * CACHE_BYTES.get(vt, 2.0))
        tokens = int(spec.get("context") or 4096) * int(spec.get("slots") or 1)
        out["kv_mb"] = round(per_token * tokens / 2**20)
    if spec.get("kind") == "impulse" and (m.get("expert_count") or 0) > 0:
        out["why"] = ("a mixture-of-experts model on Impulse: its experts stay in RAM, so what its card needs is not "
                      "estimated")
        return out
    for k in ("mmproj_file", "lora", "mtp", "draft_model"):     # the files it loads beside the model
        f = (spec.get("options") or {}).get(k)
        if isinstance(f, str) and f:
            try:
                out["weights_mb"] += round(os.path.getsize(f if os.path.isabs(f) or not base_dir else str(Path(base_dir, f))) / 2**20)
            except OSError:
                pass
    out["need_mb"] = out["weights_mb"] + (out["kv_mb"] or 0) + OVERHEAD_MB
    cs = cards_of(spec, cards() if have is None else have)
    if not cs:
        out["why"] = "no card chosen"
        return out
    gone = next((c for c in cs if c.get("missing")), None)
    if gone:
        out["card"], out["fits"], out["why"] = gone["missing"], False, f"this machine has no card {gone['missing']}"
        return out
    out["card"] = ",".join(str(c.get("uuid")) for c in cs)
    out["free_mb"] = sum(int(c.get("memory_mb", 0)) - int(c.get("used_mb", 0)) for c in cs)
    out["fits"] = out["need_mb"] <= out["free_mb"]
    where = "the card has" if len(cs) == 1 else f"its {len(cs)} cards have"
    out["why"] = None if out["fits"] else f"needs about {out['need_mb']} MB, {where} {out['free_mb']} MB free"
    return out


# --- the managed engines
class Engines:
    """The config's engines, their processes and their deployment history."""

    def __init__(self, cfg, admission=None):
        self.cfg, self.admission = cfg, admission
        self.procs: dict[str, EngineProcess] = {}
        self.unsandboxed: dict[str, str] = {}               # name -> why its engine runs without a sandbox (now)
        self.draining: dict[str, float] = {}                 # name -> when the drain started
        self.lock = threading.RLock()
        self.on_change = None                               # called after a change (the server reloads its models)
        self.network = None                                 # model machines on your network (nebula/network.py), when on
        # Every engine process is started from this one thread, which lives as long as Nebula: an engine dies with
        # the thread that started it (PR_SET_PDEATHSIG, nebula/app/engine.py), and a request's thread ends at once.
        import queue
        self._launches: queue.Queue = queue.Queue()
        threading.Thread(target=self._launcher, name="engine-launcher", daemon=True).start()

    def _launcher(self) -> None:
        while True:
            p, done = self._launches.get()
            try:
                p.start()
            finally:
                done.set()

    def _launch(self, p: EngineProcess, wait_s: float = 60) -> threading.Event:
        done = threading.Event()
        self._launches.put((p, done))
        done.wait(wait_s)
        return done

    # specs and history
    def specs(self) -> list[dict]:
        return [check_spec(s) for s in self.cfg.engines]

    def spec(self, name: str) -> dict:
        for s in self.cfg.engines:
            if s.get("name") == name:
                return check_spec(s)
        raise KeyError(name)

    def history_path(self) -> Path | None:
        return Path(self.cfg.path).with_suffix(".deployments.json") if self.cfg.path else None

    def history(self, name: str | None = None) -> dict:
        p = self.history_path()
        data = getattr(self, "_mem_history", {})
        if p is not None:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
        return data.get(name, []) if name else data

    def _write_history(self, data: dict) -> None:
        p = self.history_path()
        if p is None:
            self._mem_history = data
            return
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def apply(self, spec: dict) -> dict:
        """Add or change an engine: a new deployment version, the previous one kept. A running engine keeps running
        the old version until it is restarted (the page says so)."""
        s = check_spec(spec)
        command_of(s, self.cfg.base_dir)                    # a deployment that cannot be run is refused here, with why
        with self.lock:
            others = [e for e in self.cfg.engines if e.get("name") != s["name"]]
            for e in others:
                if int(e.get("port") or 0) == s["port"] and (e.get("host") or "127.0.0.1") == s["host"]:
                    raise EngineSpecError(f"engine {e['name']} already uses {s['host']}:{s['port']}")
            hist = self.history()
            versions = hist.setdefault(s["name"], [])
            old = next((e for e in self.cfg.engines if e.get("name") == s["name"]), None)
            if old is not None and check_spec(old) == s:
                return {"name": s["name"], "version": len(versions), "changed": False}
            versions.append({"version": len(versions) + 1, "at": time.time(), "spec": s})
            self._write_history(hist)
            self.cfg.engines = others + [s] if old is None else [s if e.get("name") == s["name"] else e for e in self.cfg.engines]
        self._changed()
        return {"name": s["name"], "version": len(versions), "changed": True,
                "restart_needed": self.state(s["name"]) in ("ready", "loading", "starting")}

    def undo(self, name: str) -> dict:
        """Back to the previous deployment (that previous one becomes the newest version again)."""
        with self.lock:
            hist = self.history()
            versions = hist.get(name) or []
            if len(versions) < 2:
                raise EngineSpecError(f"engine {name}: no earlier deployment to go back to")
            prev = versions[-2]["spec"]
        out = self.apply(prev)
        out["undone_to"] = len(self.history(name)) - 2
        return out

    def remove(self, name: str) -> dict:
        self.stop(name)
        with self.lock:
            self.cfg.engines = [e for e in self.cfg.engines if e.get("name") != name]
        self._changed()
        return {"removed": name}

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    # lifecycle
    def state(self, name: str) -> str:
        p = self.procs.get(name)
        if name in self.draining:
            return "draining"
        if p is None:
            try:
                sp = self.spec(name)
                if sp.get("machine") or sandbox_plan(sp)[0] == "openshell":
                    return "stopped"                  # its port here may be someone else's (a door): only its sandbox counts
                return "running elsewhere" if health(address_of(self.spec(name)), timeout=1.0) == 200 else "stopped"
            except KeyError:
                return "unknown"
        if p.state in ("ready", "loading", "starting") and p.proc is not None and p.proc.poll() is not None:
            return "failed"
        if p.state == "ready" and p.proc is None:
            self._recheck_sandbox(name, p)
        return p.state

    SANDBOX_RECHECK_S = 30.0

    def _recheck_sandbox(self, name: str, p) -> None:
        """A sandboxed engine (no process here) said ready once; its sandbox can die later without telling Nebula (its
        program killed, the gateway's main process gone: phase Error). At most every SANDBOX_RECHECK_S, ask the sandbox;
        gone, it is "failed" with the reason (routing already skips it: its /health fails at the gateway)."""
        now = time.monotonic()
        seen = getattr(p, "_rechecked", 0.0)
        if now - seen < self.SANDBOX_RECHECK_S:
            return
        p._rechecked = now
        exited = getattr(p, "exited", None)
        if not callable(exited):
            return

        def check():                                   # in the background: the page's status never waits on the CLI
            try:
                why = exited()
            except Exception:  # noqa: BLE001 - a check that cannot run says nothing
                why = None
            if why and p.state == "ready" and self.procs.get(name) is p:
                p.state, p.error = "failed", f"its sandbox stopped after it was ready: {why}"
                if hasattr(p, "_say"):
                    p._say(f"[nebula] {p.error}")
                logging.getLogger("nebula.engines").warning("engine %s: %s", name, p.error)
                self._changed()
        threading.Thread(target=check, name=f"recheck-{name}", daemon=True).start()

    def start(self, name: str, wait: bool = False) -> dict:
        s = self.spec(name)
        if s.get("machine"):
            return self._start_on_machine(s, wait)
        have = cards()
        cs = cards_of(s, have)
        gone = next((c for c in cs if c.get("missing")), None)
        if gone:
            raise EngineSpecError(f"engine {name}: this machine has no card {gone['missing']}")
        f = fit(s, self.cfg.base_dir, have)
        if f.get("fits") is False:
            raise EngineSpecError(f"engine {name}: {f['why']}")
        with self.lock:
            p = self.procs.get(name)
            if p is not None and p.state in ("starting", "loading", "ready", "attached") and (p.proc is None or p.proc.poll() is None):
                return self.status(name)
            env = dict(s.get("env") or {})
            if cs:
                env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(c.get("uuid") or c.get("index")) for c in cs)
            fx = flashnext_piece(s, self.cfg.base_dir)          # a Flash-Next pack run here: impulse-fx from its piece
            if fx:
                env.setdefault("IMPULSE_FX_SERVER", next(f for f in fx if os.path.basename(f).startswith("impulse-fx")))
            log = str(Path(self.cfg.path).with_name(f"engine-{name}.log")) if self.cfg.path else None
            where, why = sandbox_plan(s, conf_dir=self.conf_dir())
            if where == "openshell":
                try:
                    p = self._sandboxed(s, cs)
                except EngineSpecError as e:
                    if s.get("where") == "openshell":
                        raise
                    p, where, why = None, "local", str(e).split(": ", 1)[-1]
            if where == "openshell":
                self.procs[name] = p
                self.draining.pop(name, None)
                launched = self._launch(p, 900)                # a sandbox (and its image) can take minutes
                if p.state == "failed" and s.get("where") != "openshell":   # not insisted on: it runs here, warned
                    err = p.error or "unknown"
                    try:
                        p.stop()                                   # its failed sandbox is not left behind
                    except Exception:  # noqa: BLE001
                        pass
                    why = ("the OpenShell gateway does not let Nebula mount the models or pick a card yet (allow_driver_config)"
                           if "allow_driver_config" in err else "its sandbox did not start: " + err[:200])
                    where = "local"
            if where == "openshell":
                self.unsandboxed.pop(name, None)
                if not launched.is_set():
                    p.error = "its sandbox is still being made"
                if p.state == "loading":
                    t = threading.Thread(target=self._ready, args=(name, p), name=f"engine-{name}", daemon=True)
                    t.start()
                    if wait:
                        t.join(timeout=float(s.get("ready_timeout_s") or 900) + 5)
                return self.status(name)
            self.unsandboxed[name] = why or "not in a sandbox"
            import logging
            logging.getLogger("nebula.engines").warning("engine %s runs without a sandbox: %s", name, why)
            kp = key_path(s, self.cfg)
            argv = command_of(s, self.cfg.base_dir, kp)
            if kp:
                key = ensure_key(kp)
                argv = [key if a == "@" + kp else a for a in argv]   # a package that takes the key itself, not a file
            p = EngineProcess(argv, address_of(s), cwd=s.get("cwd") or self.cfg.base_dir, env=env,
                              ready_timeout_s=float(s.get("ready_timeout_s") or 900), stop_timeout_s=float(s.get("stop_timeout_s") or 30),
                              log_path=log)
            self.procs[name] = p
            self.draining.pop(name, None)
            self._launch(p)
        if p.state == "loading":
            t = threading.Thread(target=self._ready, args=(name, p), name=f"engine-{name}", daemon=True)
            t.start()
            if wait:
                t.join(timeout=float(s.get("ready_timeout_s") or 900) + 5)
        return self.status(name)

    def model_folders(self) -> list[str]:
        """Other folders of models a sandboxed engine may read (config `model_folders`), each mounted read-only."""
        return [os.path.realpath(str(f)) for f in (self.cfg.raw or {}).get("model_folders") or [] if os.path.isdir(str(f))]

    def conf_dir(self) -> str | None:
        return os.path.dirname(os.path.abspath(self.cfg.path)) if self.cfg.path else self.cfg.base_dir

    def models_dir(self) -> str:
        d = (self.cfg.raw or {}).get("models_dir") or "models"
        return d if os.path.isabs(d) else os.path.join(self.cfg.base_dir or os.getcwd(), d)

    def _sandboxed(self, s: dict, cs: list[dict]):
        """The engine in an OpenShell sandbox (nebula/openshell.py): the model must be in the models folder, which the
        sandbox sees read-only at /models. A package with no image to pull (Impulse) gets one built here from its
        program (openshell.engine_image), which then runs from /opt/engine."""
        from nebula import hfcache, openshell
        cs = cs or []
        md = os.path.realpath(self.models_dir())
        extra: list = []

        def inside(path: str, what: str) -> str:
            """Where the sandbox sees a file: the models folder at /models, a Hugging Face cache at /hf (read-only;
            the cache's own links resolve inside it, so the cache folder is mounted, not the file)."""
            ap = os.path.abspath(path if os.path.isabs(path) else os.path.join(self.cfg.base_dir or ".", path))
            if os.path.realpath(ap).startswith(md + os.sep):
                return "/models/" + os.path.relpath(os.path.realpath(ap), md)
            for i, folder in enumerate(self.model_folders()):    # the owner's other model folders, read-only
                if os.path.realpath(ap).startswith(folder + os.sep):
                    target = f"/folders/{i}"
                    if (folder, target) not in extra:
                        extra.append((folder, target))
                    return target + "/" + os.path.relpath(os.path.realpath(ap), folder)
            root = hfcache.root_of(self.cfg, ap)
            if root is not None:
                if extra and extra[0][0] != root:
                    raise EngineSpecError(f"engine {s['name']}: its files are in two Hugging Face caches: put them in one")
                if (root, "/hf") not in extra:
                    extra.append((root, "/hf"))
                return "/hf/" + os.path.relpath(ap, root)
            raise EngineSpecError(f"engine {s['name']}: a sandboxed engine's {what} must be in the models folder ({md}), "
                                  "one of model_folders, or a Hugging Face cache")
        wsl = os.name == "nt"                                  # Windows: a Podman container in WSL2 (nebula/wslbox.py)
        # in a container its port is published from the container's own network, so it listens on all of them there
        inner = dict(s, model=inside(s["model"], "model"), host="0.0.0.0" if wsl else "127.0.0.1")
        opts = dict(inner.get("options") or {})
        for k in ("mmproj_file", "mtp", "draft_model", "lora", "chat_template_file"):
            if isinstance(opts.get(k), str) and opts[k]:
                opts[k] = inside(opts[k], k)
        if opts:
            inner["options"] = opts
        inner.pop("where", None)
        from nebula import packages
        pk = packages.PACKAGES.get(s["kind"], {})
        ib = pk.get("image_binary")
        if ib and not s.get("image") and not s.get("binary"):   # the package's own image: its binary where the image has it
            inner["binary"] = ib
        build_from, image, extras = None, None, []
        if wsl:
            from nebula import wslbox
            image = s.get("image") or (wslbox.image_for(self.conf_dir(), s["kind"]) if pk.get("image_from_binary") else pk.get("image"))
            if not image:
                raise EngineSpecError(f"engine {s['name']}: the Safe sandbox has no image for it")
            if pk.get("image_from_binary") and not s.get("image"):
                inner["binary"] = openshell.ENGINE_DIR + "/" + str(pk.get("binary") or s["kind"])   # Setup's image: the Linux build
        elif pk.get("image_from_binary") and not s.get("image"):
            b = s.get("binary") or packages.locate(s["kind"], self.cfg.base_dir)
            if isinstance(b, list) or not b or not os.path.isfile(str(b)):
                raise EngineSpecError(f"engine {s['name']}: its sandbox image is built from its program, which must be one "
                                      f"file on this machine ({b or 'none found'})")
            build_from = os.path.realpath(str(b))
            inner["binary"] = openshell.ENGINE_DIR + "/" + os.path.basename(build_from)
            extras = flashnext_piece(s, self.cfg.base_dir) or []     # a Flash-Next pack: impulse-fx beside it in the image
        # its own key (design decision, 2026-10-20): the same key Nebula calls it with, in a folder of its own that the
        # sandbox sees read-only at /keys; the gateway passes Nebula's Authorization through to it
        kp, key_file = key_path(s, self.cfg), None
        if kp and (wsl or os.getuid() > 0):
            kdir = os.path.join(os.path.dirname(kp), "sandbox-" + s["name"])
            os.makedirs(kdir, mode=0o700, exist_ok=True)
            os.chmod(kdir, 0o700)
            dst = os.path.join(kdir, "key")
            fd = os.open(dst + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(ensure_key(kp))
            os.replace(dst + ".tmp", dst)
            extra.append((kdir, "/keys"))
            key_file = "/keys/key"
        cmd = command_of(inner, None, key_file)
        if wsl:
            env = dict(s.get("env") or {})
            uuids = [str(c.get("uuid") or c.get("index")) for c in cs]
            if uuids:                                          # WSL has one GPU device for all cards: CUDA picks the card
                env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(uuids))
            return wslbox.WslEngine(s, cmd, [(md, "/models")] + extra, image, env=env, gpu=bool(cs) or bool(cards()),
                                    ready_timeout_s=float(s.get("ready_timeout_s") or 900))
        return openshell.SandboxedEngine(s, cmd, md, os.path.join(self.cfg.base_dir or ".", "nebula-sandbox"),
                                         [c.get("index") for c in cs] or None,
                                         ready_timeout_s=float(s.get("ready_timeout_s") or 900), extra_dirs=extra,
                                         keyed=key_file is not None, env=s.get("env") or None, build_from=build_from,
                                         extras=extras)

    def _start_on_machine(self, s: dict, wait: bool) -> dict:
        """An engine on a model machine: its card there, a sandbox there, its model through your network."""
        from nebula import machines, openshell, packages
        name = s["name"]
        if self.network is None or getattr(self.network, "broker", None) is None:
            raise EngineSpecError(f"engine {name}: your network is off here (Add a server turns it on)")
        try:
            machines.row(self.cfg, s["machine"])
        except KeyError:
            raise EngineSpecError(f"engine {name}: no model machine named {s['machine']!r}") from None
        have = [c for c in machines.cards(self.cfg, max_age_s=0) if c.get("machine") == s["machine"]]
        cs = cards_of(s, have)
        gone = next((c for c in cs if c.get("missing")), None)
        if gone:
            raise EngineSpecError(f"engine {name}: {s['machine']} has no card {gone['missing']}")
        if s["kind"] == "command":                           # a container engine: its image and its command as given
            with self.lock:
                p = self.procs.get(name)
                if p is not None and p.state in ("starting", "loading", "ready"):
                    return self.status(name)
                cmd = command_of(s, None, None)
                p = machines.MachineEngine(s, cmd, self.cfg, self.network, os.path.join(self.cfg.base_dir or ".", "nebula-sandbox"),
                                           s.get("folder") or "", [c.get("index") for c in cs] or None, None, s["image"],
                                           ready_timeout_s=float(s.get("ready_timeout_s") or 1800), env=s.get("env") or None)
                self.procs[name] = p
                self.unsandboxed.pop(name, None)
                self.draining.pop(name, None)

            def run_c():
                p.start()
                if p.state == "loading":
                    self._ready(name, p)
            threading.Thread(target=run_c, name=f"engine-{name}", daemon=True).start()
            return self.status(name)
        if cs and s["kind"] == "impulse" and not s.get("image"):   # its build must carry kernels for the cards there
            b = s.get("binary") or packages.locate(s["kind"], self.cfg.base_dir)
            if isinstance(b, str) and os.path.isfile(b):
                gap = machines.arch_gap(b, cs)
                if gap:
                    raise EngineSpecError(f"engine {name}: {gap}")
        with self.lock:
            p = self.procs.get(name)
            if p is not None and p.state in ("starting", "loading", "ready"):
                return self.status(name)
            pk = packages.PACKAGES.get(s["kind"], {})
            # every file it reads there (the model, a projector, an MTP or draft model, a LoRA, absolute paths in its
            # flags) lives under one folder there, mounted read-only at /models
            files_keys = ("mmproj_file", "mtp", "draft_model", "lora", "chat_template_file")
            opts = dict(s.get("options") or {})
            paths = [s["model"]] + [opts[k] for k in files_keys if isinstance(opts.get(k), str) and opts[k].startswith("/")]
            paths += [f for f in s.get("flags") or [] if isinstance(f, str) and f.startswith("/")]
            root = os.path.commonpath([os.path.dirname(x) for x in paths])

            def there(x: str) -> str:
                return "/models" + x[len(root):] if x.startswith(root + "/") else x
            inner = dict(s, host="127.0.0.1", model=there(s["model"]))
            if s["kind"] == "impulse" and _pack_of(s["model"]):      # a Flash-Next pack: its files as the sandbox sees them
                inner["fx_files"] = {k: there(v) for k, v in _pack_of(s["model"]).items()}
                if _piece_profile(self.cfg.base_dir):                # the piece's profile: beside impulse-server in the image
                    from nebula import flashnext
                    inner["fx_files"]["profile"] = openshell.ENGINE_DIR + "/" + flashnext.PIECE_PROFILE
            if opts:
                inner["options"] = {k: (there(v) if k in files_keys and isinstance(v, str) else v) for k, v in opts.items()}
            if s.get("flags"):
                inner["flags"] = [there(f) if isinstance(f, str) else f for f in s["flags"]]
            for k in ("where", "machine"):
                inner.pop(k, None)
            build_from = None
            if pk.get("image_from_binary") and not s.get("image"):
                b = s.get("binary") or packages.locate(s["kind"], self.cfg.base_dir)
                if isinstance(b, list) or not b or not os.path.isfile(str(b)):
                    raise EngineSpecError(f"engine {name}: its sandbox image is built here from its program ({b or 'none found'})")
                build_from = os.path.realpath(str(b))
                inner["binary"] = openshell.ENGINE_DIR + "/" + os.path.basename(build_from)
            elif pk.get("image_binary") and not s.get("binary"):
                inner["binary"] = pk["image_binary"]
            cmd = command_of(inner, None, None)
            p = machines.MachineEngine(s, cmd, self.cfg, self.network, os.path.join(self.cfg.base_dir or ".", "nebula-sandbox"),
                                       root, [c.get("index") for c in cs] or None, build_from, s.get("image") or pk.get("image"),
                                       ready_timeout_s=float(s.get("ready_timeout_s") or 900), env=s.get("env") or None)
            self.procs[name] = p
            self.unsandboxed.pop(name, None)
            self.draining.pop(name, None)

        def run():                                     # the sandbox lives there, not as a child of this process
            p.start()
            if p.state == "loading":
                self._ready(name, p)
        t = threading.Thread(target=run, name=f"engine-{name}", daemon=True)
        t.start()
        if wait:
            t.join(timeout=float(s.get("ready_timeout_s") or 900) + 1800)
        return self.status(name)

    def address(self, name: str) -> str:
        """Where the engine answers: its host and port, or, in a sandbox, the gateway's address for it; for one on a
        model machine, this machine's router (the model travels over your network)."""
        s = self.spec(name)
        if s.get("machine"):
            from nebula import machines
            return machines.router_url(self.cfg)
        return getattr(self.procs.get(name), "address", None) or address_of(s)

    def _ready(self, name: str, p: EngineProcess) -> None:
        if p.wait_ready():
            self._changed()

    def stop(self, name: str) -> dict:
        with self.lock:
            p = self.procs.get(name)
            self.draining.pop(name, None)
        if p is not None:
            p.stop()
        self._changed()
        return self.status(name)

    def drain(self, name: str, timeout_s: float = 600.0, then_stop: bool = True) -> dict:
        """Routing sends the engine nothing new (it is skipped with the reason "draining"); this waits until its
        running calls are done, then stops it. Returns at once; the page follows the state."""
        s = self.spec(name)
        with self.lock:
            self.draining[name] = time.time()

        def run():
            deadline = time.time() + timeout_s
            while time.time() < deadline and name in self.draining:
                if self.in_flight(s) == 0:
                    break
                time.sleep(0.2)
            if name in self.draining and then_stop:
                self.stop(name)
        threading.Thread(target=run, name=f"drain-{name}", daemon=True).start()
        self._changed()
        return self.status(name)

    def restart(self, name: str, wait: bool = False) -> dict:
        self.stop(name)
        return self.start(name, wait=wait)

    def in_flight(self, spec: dict) -> int:
        if self.admission is None:
            return 0
        g = self.admission.gates.get(self.address(spec["name"]))
        return g.status()["held"] if g is not None else 0

    def is_draining(self, address: str | None) -> bool:
        if not address or not self.draining:
            return False
        return any(self.address(s["name"]) == address for s in self.specs() if s["name"] in self.draining)

    def verify(self, name: str) -> dict:
        """It answers, and the model file it reports is the deployment's."""
        s = self.spec(name)
        addr = self.address(name)
        out = {"name": name, "address": addr, "health": health(addr, timeout=3.0), "model_path": None, "expected": None,
               "model_ok": None}
        if s["kind"] != "command":
            out["expected"] = os.path.basename(s["model"])
            try:
                import urllib.request
                kp = key_path(s, self.cfg)
                h = {"Authorization": f"Bearer {ensure_key(kp)}"} if kp else {}
                with urllib.request.urlopen(urllib.request.Request(addr + "/props", headers=h), timeout=3.0) as r:
                    out["model_path"] = json.loads(r.read()).get("model_path")
                out["model_ok"] = bool(out["model_path"]) and os.path.basename(out["model_path"]) == out["expected"]
            except (OSError, ValueError):
                pass
        out["ok"] = out["health"] == 200 and out["model_ok"] is not False
        return out

    def status(self, name: str | None = None):
        if name is None:
            return [self.status(s["name"]) for s in self.specs()]
        s = self.spec(name)
        p = self.procs.get(name)
        st = p.status() if p is not None else {"state": None, "lines": [], "pid": None, "error": None, "seconds": None}
        tiers = [t.name for t in self.cfg.tiers if getattr(t, "managed", None) == name]
        try:
            command, command_error = command_of(s, self.cfg.base_dir, key_path(s, self.cfg)), None
        except EngineSpecError as e:
            command, command_error = None, str(e)
        return {"name": name, "spec": s, "address": self.address(name), "command": command, "command_error": command_error,
                "state": self.state(name), "pid": st.get("pid"), "error": st.get("error"), "seconds": st.get("seconds"),
                "lines": st.get("lines")[-40:], "version": len(self.history(name)), "tiers": tiers,
                "in_flight": self.in_flight(s), "fit": fit(s, self.cfg.base_dir),
                "sandboxed": p is not None and name not in self.unsandboxed,
                "sandbox_warning": self.unsandboxed.get(name) if p is not None else None}

    def adopt_sandboxes(self) -> list[str]:
        """Engines still running in their sandboxes from before Nebula restarted (a sandbox outlives Nebula): taken
        back as they are, at the gateway's address for them, instead of being seen as stopped or "elsewhere"."""
        from nebula import machines, openshell
        took = []
        try:
            local = {x["sandbox"]: x["url"].rstrip("/") for x in openshell._json(["service", "list"]).get("services") or []}
        except Exception:  # noqa: BLE001 - no OpenShell here: nothing to take back
            local = {}
        for s in self.specs():
            name, sb = s["name"], openshell.sandbox_name(s["name"])
            if name in self.procs:
                continue
            try:
                if s.get("machine"):
                    r = machines.row(self.cfg, s["machine"])
                    machines.tunnel(self.cfg, r)
                    there = {x["sandbox"]: x["url"].rstrip("/") for x in
                             (json.loads(openshell.run(["service", "list", "-o", "json"], gateway=r["gateway"]) or "{}").get("services") or [])}
                    if sb not in there or self.network is None:
                        continue
                    # a command engine (the reference engine: an image and a command) has no "model", only its "folder" - reading
                    # s["model"] here skipped it silently and left it "stopped" while its sandbox ran (2026-10-25)
                    root = (s.get("folder") or "") if s.get("kind") == "command" else os.path.dirname(s.get("model") or "")
                    p = machines.MachineEngine(s, [], self.cfg, self.network, os.path.join(self.cfg.base_dir or ".", "nebula-sandbox"),
                                               root, None, None, s.get("image") if s.get("kind") == "command" else None)
                    from nebula import sshinstall
                    import urllib.parse
                    u = urllib.parse.urlsplit(there[sb])
                    p.there = urllib.parse.urlunsplit((u.scheme, f"{u.hostname}:{sshinstall.GATEWAY_PORT}", "", "", ""))
                    p.address, p.state = machines.router_url(self.cfg), "ready"
                elif sb in local:
                    p = openshell.SandboxedEngine(s, [], self.models_dir(), os.path.join(self.cfg.base_dir or ".", "nebula-sandbox"), None)
                    p.address = local[sb]
                    p.state = "ready" if health(p.address, timeout=2.0) == 200 else "loading"
                else:
                    continue
            except Exception:  # noqa: BLE001 - one machine out of reach does not stop the others
                continue
            p._say("[nebula] taken back: its sandbox kept running while Nebula restarted")
            self.procs[name] = p
            took.append(name)
        if took:
            self._changed()
        return took

    def autostart(self) -> None:
        for s in self.specs():
            if s.get("autostart"):
                try:
                    self.start(s["name"])
                except (EngineSpecError, OSError) as e:
                    import logging
                    logging.getLogger("nebula.engines").error("engine %s did not start: %s", s["name"], e)

    def close(self) -> None:
        for p in list(self.procs.values()):
            p.stop()
