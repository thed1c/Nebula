"""nebula/packages.py: the engine packages Nebula can deploy, and every setting each one takes (the Edit form).

A package is a model server: its binary, how to ask its version, and a **flag schema**: each setting's key, its label,
its group on the form, its type, its range, its command-line form and the packages that take it. The Edit form is
drawn from the schema, a deployment is checked against it, and the command line is built from it, so what the page
shows is exactly what runs.

- **llama.cpp**: `llama-server` (github.com/ggml-org/llama.cpp).
- **Impulse** (0.1.1): `impulse-server` (before it, `strata-lite-server`), llama-server's flags plus its own (MTP drafting, the expert cache).
(The Strata package, Niko's upstream server, was removed 2026-10-25 by the owner's word "Yes. Remove it.": Flash-Next
packs run on Impulse (impulse-fx, nebula/flashnext.py). A saved engine of that kind loads as Impulse, see
engines.check_spec.)

Values: `context` is per slot (the server's `-c` is context × slots). `thinking` off sends `--reasoning-budget 0`; on
sends the budget (-1 = unlimited). `mmproj` on with no file picks the projector found beside the model.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

ALL = ("llama.cpp", "impulse")
LLAMA_LIKE = ALL

# How Nebula runs each one (docs/sharing.md): `harden` always (its own chat page off), `harden_private` when no guest
# uses it (its /slots page off), `shared` when guests do (no prompt cache shared across slots), `api_key` (the flag
# that takes the random key only Nebula holds, None when the package has none: loopback is then its only wall), and
# `isolation` (how tenants are kept apart on it: a slot each, or no cache for guests).
PACKAGES = {
    "llama.cpp": {"label": "llama.cpp", "binary": "llama-server", "version_args": ["--version"],
                  "about": "the reference server: any GGUF model, CPU or GPU",
                  "harden": ["--no-webui"], "harden_private": ["--no-slots"],
                  "shared": ["--cache-ram", "0", "--no-cache-idle-slots", "-sps", "0"],
                  "api_key": "--api-key-file", "isolation": "slots", "image": "ghcr.io/ggml-org/llama.cpp:server-cuda",
                  "image_binary": "/app/llama-server"},            # where the image keeps it (its entrypoint)
    "impulse": {"label": "Impulse 0.1.1", "release": "0.1.1", "binary": "impulse-server", "version_args": ["--version"],
                "also": ["strata-lite-server"],             # the name before 0.1.1
                "about": "this suite's own engine: llama.cpp underneath, faster MoE and drafting",
                "harden": ["--no-webui"], "harden_private": [], "shared": ["--cache-ram", "0"],
                "api_key": None, "isolation": "no-cache",
                "image_from_binary": True},                 # no image to pull: one is built here from impulse-server (openshell.engine_image)
}

GROUPS = ["Model", "Context & memory", "GPU", "Speed", "Speculative decoding", "Thinking", "Sampling", "Template", "Network", "Extra"]

# key, label, group, type, cli, packages, extra
SCHEMA: list[dict] = [
    # Model
    {"key": "mmproj", "label": "Pictures (vision projector)", "group": "Model", "type": "bool", "cli": "--mmproj",
     "help": "reads images; on with no file picks the mmproj found beside the model"},
    {"key": "mmproj_file", "label": "Projector file", "group": "Model", "type": "path", "cli": None, "help": "an mmproj-*.gguf"},
    {"key": "mmproj_cpu", "label": "Projector on the CPU", "group": "Model", "type": "bool", "cli": "--no-mmproj-offload",
     "help": "keeps the projector's memory off the card"},
    {"key": "lora", "label": "LoRA adapter", "group": "Model", "type": "path", "cli": "--lora"},
    {"key": "lora_scale", "label": "LoRA scale", "group": "Model", "type": "float", "min": 0, "max": 4, "cli": None},
    {"key": "alias", "label": "Model name it reports", "group": "Model", "type": "str", "cli": "--alias"},
    # Context & memory
    {"key": "cache_type_k", "label": "KV cache, keys", "group": "Context & memory", "type": "choice", "cli": "-ctk",
     "choices": ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"]},
    {"key": "cache_type_v", "label": "KV cache, values", "group": "Context & memory", "type": "choice", "cli": "-ctv",
     "choices": ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"]},
    {"key": "rope_scaling", "label": "RoPE scaling", "group": "Context & memory", "type": "choice", "cli": "--rope-scaling",
     "choices": ["none", "linear", "yarn"]},
    {"key": "rope_scale", "label": "RoPE scale factor", "group": "Context & memory", "type": "float", "min": 0.1, "max": 64, "cli": "--rope-scale"},
    {"key": "yarn_orig_ctx", "label": "YaRN original context", "group": "Context & memory", "type": "int", "min": 0, "cli": "--yarn-orig-ctx"},
    {"key": "mlock", "label": "Keep the model in RAM (mlock)", "group": "Context & memory", "type": "bool", "cli": "--mlock"},
    {"key": "no_mmap", "label": "Load fully, no mmap", "group": "Context & memory", "type": "bool", "cli": "--no-mmap"},
    {"key": "cache_reuse", "label": "Prompt cache reuse (tokens)", "group": "Context & memory", "type": "int", "min": 0, "cli": "--cache-reuse"},
    # GPU
    {"key": "gpu_layers", "label": "Layers on the GPU", "group": "GPU", "type": "int", "min": -1, "max": 999, "cli": "-ngl",
     "help": "999 = all"},
    {"key": "n_cpu_moe", "label": "MoE experts kept on the CPU (layers)", "group": "GPU", "type": "int", "min": 0, "cli": "--n-cpu-moe"},
    {"key": "split_mode", "label": "Split across cards", "group": "GPU", "type": "choice", "choices": ["none", "layer", "row"], "cli": "-sm"},
    {"key": "tensor_split", "label": "Tensor split (per card)", "group": "GPU", "type": "str", "cli": "-ts", "pattern": r"^\d+(\.\d+)?(,\d+(\.\d+)?)*$",
     "help": "e.g. 3,1"},
    {"key": "main_gpu", "label": "Main card", "group": "GPU", "type": "int", "min": 0, "cli": "-mg"},
    # Speed
    {"key": "flash_attn", "label": "Flash attention", "group": "Speed", "type": "choice", "choices": ["auto", "on", "off"], "cli": "--flash-attn"},
    {"key": "batch", "label": "Batch", "group": "Speed", "type": "int", "min": 1, "cli": "-b"},
    {"key": "ubatch", "label": "Micro-batch", "group": "Speed", "type": "int", "min": 1, "cli": "-ub"},
    {"key": "threads", "label": "CPU threads", "group": "Speed", "type": "int", "min": 1, "cli": "-t"},
    {"key": "threads_batch", "label": "CPU threads, prompt", "group": "Speed", "type": "int", "min": 1, "cli": "-tb"},
    # Speculative decoding (design decision 2026-10-20: MTP for every engine that has it, not just Impulse). The flag names
    # differ by engine: Impulse (on llama.cpp b6b003d) and older llama.cpp take --mtp/-md/--draft-max; llama.cpp since
    # its --spec-* rename takes --spec-type/--spec-draft-model/--spec-draft-n-max (spec_args() picks)
    {"key": "spec_type", "label": "Method", "group": "Speculative decoding", "type": "choice", "cli": "spec",
     "choices": ["off", "mtp", "draft", "eagle3", "ngram"], "packages": ("impulse", "llama.cpp"),
     "help": "mtp: the model's multi-token-prediction head (or an MTP drafter file); draft: a small model of the same family; "
             "eagle3: an EAGLE-3 draft head; ngram: repeats from the conversation, no extra model (llama.cpp)"},
    {"key": "mtp", "label": "MTP drafter file", "group": "Speculative decoding", "type": "path", "cli": "spec", "packages": ("impulse", "llama.cpp"),
     "help": "an mtp-*.gguf; empty: the MTP layers inside the model, when it has them (llama.cpp)"},
    {"key": "draft_model", "label": "Draft model file", "group": "Speculative decoding", "type": "path", "cli": "spec", "packages": ("llama.cpp",),
     "help": "for draft or eagle3: a small .gguf that shares the model's vocabulary"},
    {"key": "draft_max", "label": "Draft tokens (max)", "group": "Speculative decoding", "type": "int", "min": 0, "max": 64, "cli": "spec"},
    {"key": "draft_min", "label": "Draft tokens (min)", "group": "Speculative decoding", "type": "int", "min": 0, "max": 64, "cli": "spec",
     "packages": ("llama.cpp",)},
    {"key": "draft_gpu_layers", "label": "Draft model layers on the GPU", "group": "Speculative decoding", "type": "int", "min": -1, "max": 999,
     "cli": "spec", "packages": ("llama.cpp",)},
    {"key": "draft_p_min", "label": "Draft: keep tokens above this probability", "group": "Speculative decoding", "type": "float", "min": 0, "max": 1,
     "cli": "spec", "packages": ("llama.cpp",)},
    {"key": "cont_batching", "label": "Continuous batching", "group": "Speed", "type": "bool", "cli": "--cont-batching"},
    # Thinking
    {"key": "thinking", "label": "Thinking", "group": "Thinking", "type": "bool", "cli": None,
     "help": "off sends a reasoning budget of 0"},
    {"key": "reasoning_budget", "label": "Thinking budget (tokens, -1 = no limit)", "group": "Thinking", "type": "int", "min": -1, "cli": None},
    {"key": "reasoning_format", "label": "Thinking format", "group": "Thinking", "type": "choice", "choices": ["auto", "none", "deepseek"],
     "cli": "--reasoning-format"},
    # Sampling (the server's defaults; a request's own values win)
    {"key": "temp", "label": "Temperature", "group": "Sampling", "type": "float", "min": 0, "max": 5, "cli": "--temp"},
    {"key": "top_p", "label": "Top-p", "group": "Sampling", "type": "float", "min": 0, "max": 1, "cli": "--top-p"},
    {"key": "top_k", "label": "Top-k", "group": "Sampling", "type": "int", "min": 0, "cli": "--top-k"},
    {"key": "min_p", "label": "Min-p", "group": "Sampling", "type": "float", "min": 0, "max": 1, "cli": "--min-p"},
    {"key": "repeat_penalty", "label": "Repeat penalty", "group": "Sampling", "type": "float", "min": 0, "max": 5, "cli": "--repeat-penalty"},
    {"key": "seed", "label": "Seed", "group": "Sampling", "type": "int", "cli": "--seed"},
    # Template
    {"key": "jinja", "label": "Use the model's Jinja template", "group": "Template", "type": "bool", "cli": "--jinja"},
    {"key": "chat_template", "label": "Built-in template", "group": "Template", "type": "str", "cli": "--chat-template",
     "help": "e.g. chatml, llama3, gemma"},
    {"key": "chat_template_file", "label": "Template file", "group": "Template", "type": "path", "cli": "--chat-template-file"},
    # Network
    {"key": "api_key", "label": "Server API key", "group": "Network", "type": "secret", "cli": "--api-key",
     "help": "needed when it listens beyond this machine"},
    {"key": "timeout", "label": "Server timeout (s)", "group": "Network", "type": "int", "min": 1, "cli": "--timeout"},
    {"key": "metrics", "label": "/metrics endpoint", "group": "Network", "type": "bool", "cli": "--metrics"},
]
BY_KEY = {f["key"]: f for f in SCHEMA}


class OptionError(ValueError):
    pass


def supports(package: str, key: str) -> bool:
    f = BY_KEY.get(key)
    return f is not None and package in f.get("packages", ALL)


def schema(package: str) -> list[dict]:
    """The settings one package takes, in form order."""
    return [{k: v for k, v in f.items() if k != "packages"} for f in SCHEMA if package in f.get("packages", ALL)]


def check(package: str, options: dict) -> dict:
    """Options checked against the package's schema: the known keys, their types and ranges."""
    if package not in PACKAGES:
        raise OptionError(f"no package {package!r}: one of {', '.join(PACKAGES)}")
    if not isinstance(options, dict):
        raise OptionError("options is an object of setting → value")
    out = {}
    for k, v in options.items():
        f = BY_KEY.get(k)
        if f is None:
            raise OptionError(f"no setting {k!r} (extra flags go in 'flags')")
        if package not in f.get("packages", ALL):
            raise OptionError(f"{PACKAGES[package]['label']} has no setting {f['label']!r}")
        if v is None or v == "":
            continue
        t = f["type"]
        try:
            if t == "bool":
                if not isinstance(v, bool):
                    raise OptionError(f"{f['label']}: on or off")
            elif t == "int":
                v = int(v)
            elif t == "float":
                v = float(v)
            elif t == "choice":
                if v not in f["choices"]:
                    raise OptionError(f"{f['label']}: one of {', '.join(f['choices'])}")
            else:
                v = str(v)
        except (TypeError, ValueError):
            raise OptionError(f"{f['label']}: a {t} is needed") from None
        if t in ("int", "float"):
            if "min" in f and v < f["min"]:
                raise OptionError(f"{f['label']}: at least {f['min']}")
            if "max" in f and v > f["max"]:
                raise OptionError(f"{f['label']}: at most {f['max']}")
        if f.get("pattern"):
            import re
            if not re.match(f["pattern"], str(v)):
                raise OptionError(f"{f['label']}: {f.get('help') or 'not in the expected form'}")
        out[k] = v
    if out.get("lora_scale") is not None and not out.get("lora"):
        raise OptionError("a LoRA scale needs a LoRA adapter")
    st = out.get("spec_type")
    if st in ("draft", "eagle3") and not out.get("draft_model"):
        raise OptionError(f"{st} needs a draft model file")
    if st == "ngram" and package == "impulse":
        raise OptionError("Impulse drafts with MTP (its drafter file); n-gram drafting is llama.cpp's")
    if st in ("eagle3", "draft") and package == "impulse":
        raise OptionError("Impulse drafts with MTP (its drafter file); a draft model or EAGLE-3 is llama.cpp's")
    if st == "mtp" and package == "impulse" and not out.get("mtp"):
        raise OptionError("Impulse's MTP needs its drafter file (mtp-*.gguf)")
    return out


_STYLE: dict = {}


def spec_style(package: str, binary=None) -> str:
    """"new" (llama.cpp since its --spec-* rename) or "old" (--mtp/-md/--draft-max: Impulse, older llama.cpp).
    A llama.cpp program here is asked (its --help, once); one not here (in a sandbox image) is today's: new."""
    if package != "llama.cpp":
        return "old"
    b = binary[0] if isinstance(binary, list) and binary else binary
    if not b or not (os.path.isabs(str(b)) and os.access(str(b), os.X_OK)) and not shutil.which(str(b)):
        return "new"
    key = str(binary)
    if key not in _STYLE:
        import subprocess
        try:
            p = subprocess.run((list(binary) if isinstance(binary, list) else [str(binary)]) + ["--help"], capture_output=True, text=True, timeout=20)
            text = p.stdout + p.stderr
            _STYLE[key] = "new" if "--spec-type" in text else "old"
        except (OSError, subprocess.TimeoutExpired):
            _STYLE[key] = "new"
    return _STYLE[key]


def spec_args(package: str, o: dict, style: str) -> list[str]:
    """The speculative-decoding words for one engine, in its own flag names."""
    st = o.get("spec_type") or ("mtp" if o.get("mtp") else "draft" if o.get("draft_model") else None)
    if not st or st == "off":
        return []
    out: list[str] = []
    if package == "impulse":                               # Impulse 0.1.1's own flags (impulse-server --help)
        if st == "mtp" and o.get("mtp"):
            out += ["--mtp", o["mtp"]]
        if o.get("draft_max") is not None:
            out += ["--n-draft", str(o["draft_max"])]
        return out
    if style == "new":
        out += ["--spec-type", {"mtp": "draft-mtp", "draft": "draft-simple", "eagle3": "draft-eagle3", "ngram": "ngram-simple"}[st]]
        f = o.get("mtp") if st == "mtp" else o.get("draft_model") if st in ("draft", "eagle3") else None
        if f:
            out += ["--spec-draft-model", f]
        names = {"draft_max": "--spec-draft-n-max", "draft_min": "--spec-draft-n-min", "draft_gpu_layers": "--spec-draft-ngl",
                 "draft_p_min": "--spec-draft-p-min"}
    else:
        if st == "mtp":
            out += (["--mtp", o["mtp"]] if package == "impulse" else ["-md", o["mtp"]] if o.get("mtp") else [])
        elif st in ("draft", "eagle3") and o.get("draft_model"):
            out += ["-md", o["draft_model"]]
        names = {"draft_max": "--draft-max", "draft_min": "--draft-min", "draft_gpu_layers": "-ngld", "draft_p_min": "--draft-p-min"}
    for k, flag in names.items():
        if o.get(k) is not None:
            out += [flag, str(o[k])]
    return out


def find_mmproj(model: str) -> str | None:
    """The projector beside a model file (mmproj*.gguf in the same folder), if there is exactly one best match."""
    d = Path(model).parent
    try:
        cands = sorted(p for p in d.iterdir() if p.name.lower().startswith("mmproj") and p.suffix == ".gguf")
    except OSError:
        return None
    return str(cands[0]) if cands else None


def args(package: str, options: dict, model: str, binary=None) -> list[str]:
    """The command-line words for the options (after the binary's -m/--host/--port/-c/-np)."""
    o = check(package, options)
    out: list[str] = []
    for f in SCHEMA:
        k = f["key"]
        if k not in o or not f.get("cli") or f["cli"] == "spec" or package not in f.get("packages", ALL):
            continue
        v = o[k]
        if k == "mmproj":
            if v:
                proj = o.get("mmproj_file") or find_mmproj(model)
                if not proj:
                    raise OptionError("pictures are on but no projector file was given or found beside the model")
                out += ["--mmproj", proj]
            continue
        if k == "lora":
            out += (["--lora-scaled", v, str(o["lora_scale"])] if o.get("lora_scale") is not None else ["--lora", v])
            continue
        if f["type"] == "bool":
            if v:
                out += [f["cli"]] + ([f["value"]] if f.get("value") else [])
            continue
        out += [f["cli"], str(v)]
    out += spec_args(package, o, spec_style(package, binary))
    if "thinking" in o or "reasoning_budget" in o:
        if o.get("thinking") is False:
            out += ["--reasoning-budget", "0"]
        else:
            out += ["--reasoning-budget", str(o.get("reasoning_budget", -1))]
    return out


def version(binary) -> str | None:
    """The package's own --version line, or None when it cannot be run."""
    import subprocess
    argv = (list(binary) if isinstance(binary, list) else [str(binary)]) + ["--version"]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (p.stdout + "\n" + p.stderr).strip().splitlines()
    return next((ln.strip() for ln in text if "version" in ln.lower()), text[0].strip() if text else None)


def locate(package: str, base_dir: str | None = None) -> str | None:
    """The package's program on this computer: NEBULA_<PACKAGE>_BIN, the one Nebula installed (nebula-impulse/), its name
    on the PATH, an older name."""
    p = PACKAGES.get(package) or {}
    env = os.environ.get(f"NEBULA_{package.upper().replace('.', '_')}_BIN")
    cands = [env] if env else []
    if package == "impulse":
        cands.append(os.path.join(base_dir or os.getcwd(), "nebula-impulse", "impulse-server"))
        data = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
        cands.append(os.path.join(data, "nebula", "impulse", "impulse-server"))     # where the release bundle installs it
        # the one-click installers: impulse/ beside the Python Nebula runs on (python/bin/python3, python\python.exe)
        import sys
        exe = "impulse-server.exe" if os.name == "nt" else "impulse-server"
        for up in Path(sys.executable).resolve().parents[:3]:
            cands.append(str(up / "impulse" / exe))
    cands += [p.get("binary")] + list(p.get("also") or [])
    for c in cands:
        hit = binary_found(c) if c else None
        if hit:
            return hit
    return None


def binary_found(binary) -> str | None:
    import shutil
    b = binary[0] if isinstance(binary, list) and binary else binary
    if not b:
        return None
    return b if os.path.isfile(str(b)) else shutil.which(str(b))


# --- the Raw tab: a command line in, the engine's settings out (and back)
MANAGED = {  # flags Nebula sets itself on every engine it runs: never taken from raw text (docs/sharing.md)
    "--no-webui": "Nebula turns the engine's own chat page off",
    "--webui": "Nebula turns the engine's own chat page off",
    "--no-slots": "Nebula decides /slots (off unless guests share the engine)",
    "--slots": "Nebula decides /slots (off unless guests share the engine)",
    "--cache-ram": "Nebula decides the cross-slot RAM cache (off when guests share the engine)",
    "-cram": "Nebula decides the cross-slot RAM cache (off when guests share the engine)",
    "--no-cache-idle-slots": "Nebula decides the idle-slot cache (off when guests share the engine)",
    "-sps": "Nebula decides slot matching (off when guests share the engine)",
    "--slot-prompt-similarity": "Nebula decides slot matching (off when guests share the engine)",
    "--api-key": "the engine's key is one only Nebula holds",
    "--api-key-file": "the engine's key is one only Nebula holds",
}
TAKES_VALUE_MANAGED = {"--cache-ram", "-cram", "-sps", "--slot-prompt-similarity", "--api-key", "--api-key-file"}
FIELD_FLAGS = {"-m": "model", "--model": "model", "--host": "host", "--port": "port", "-c": "context", "--ctx-size": "context",
               "-np": "slots", "--parallel": "slots"}


SPEC_FLAGS = {"--spec-type": "spec_type", "--mtp": "mtp", "--spec-draft-model": "draft_model", "-md": "draft_model",
              "--model-draft": "draft_model", "--spec-draft-n-max": "draft_max", "--draft-max": "draft_max", "--n-draft": "draft_max", "--draft": "draft_max",
              "--draft-n": "draft_max", "--spec-draft-n-min": "draft_min", "--draft-min": "draft_min", "--draft-n-min": "draft_min",
              "--spec-draft-ngl": "draft_gpu_layers", "-ngld": "draft_gpu_layers", "--gpu-layers-draft": "draft_gpu_layers",
              "--n-gpu-layers-draft": "draft_gpu_layers", "--spec-draft-p-min": "draft_p_min", "--draft-p-min": "draft_p_min"}
SPEC_TYPES = {"draft-mtp": "mtp", "draft-simple": "draft", "draft-eagle3": "eagle3", "ngram-simple": "ngram", "none": "off"}


def parse_command(package: str, text: str) -> dict:
    """A command line (the Raw tab) → {"spec": fields + options + flags, "locked": [...], "warnings": [...], "invalid": [...]}.
    Flags with a setting become that setting; others are kept verbatim in `flags` (never dropped); flags Nebula sets
    itself are left out with the reason (they are put back on every start)."""
    import shlex
    try:
        from nebula import cmdline
        words = cmdline.split(text)
    except ValueError as e:
        return {"spec": {}, "locked": [], "warnings": [], "invalid": [f"the line cannot be read: {e}"]}
    by_cli = {}
    for f in SCHEMA:
        if f.get("cli") and package in f.get("packages", ALL):
            by_cli[f["cli"]] = f
    spec: dict = {"options": {}}
    flags: list[str] = []
    locked, warnings, invalid = [], [], []
    i = 0
    binary = []
    while i < len(words) and not words[i].startswith("-"):
        binary.append(words[i])
        i += 1
    if binary:
        spec["binary"] = binary if len(binary) > 1 else binary[0]
    total_ctx = None
    o = spec["options"]
    while i < len(words):
        w = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else None
        has_val = nxt is not None and not (nxt.startswith("-") and not re.match(r"^-\d", nxt))
        if w in MANAGED:
            locked.append(f"{w}: {MANAGED[w]}")
            i += 2 if (w in TAKES_VALUE_MANAGED and has_val) else 1
            continue
        if w in FIELD_FLAGS and has_val:
            k = FIELD_FLAGS[w]
            if k == "host" and nxt not in ("127.0.0.1", "localhost"):
                locked.append(f"--host {nxt}: engines Nebula runs listen on this machine only (127.0.0.1); guests come through Nebula")
            elif k == "context":
                total_ctx = nxt
            else:
                spec[k] = nxt
            i += 2
            continue
        if w in ("-ctk", "--cache-type-k", "-ctv", "--cache-type-v") and has_val:
            o["cache_type_k" if w in ("-ctk", "--cache-type-k") else "cache_type_v"] = nxt
            i += 2
            continue
        if w == "--mmproj" and has_val:
            o["mmproj"], o["mmproj_file"] = True, nxt
            i += 2
            continue
        if w == "--lora-scaled" and i + 2 < len(words):
            o["lora"], o["lora_scale"] = words[i + 1], words[i + 2]
            i += 3
            continue
        if w == "--reasoning-budget" and has_val:
            if nxt == "0":
                o["thinking"] = False
            else:
                o["thinking"], o["reasoning_budget"] = True, nxt
            i += 2
            continue
        if w in SPEC_FLAGS and has_val:                     # speculative decoding, in either engine's flag names
            k = SPEC_FLAGS[w]
            if k == "spec_type":
                o["spec_type"] = SPEC_TYPES.get(nxt, nxt)
            elif k == "mtp":
                o["spec_type"], o["mtp"] = "mtp", nxt
            else:
                o[k] = nxt
            i += 2
            continue
        f = by_cli.get(w)
        if f is not None and f["type"] == "bool":
            o[f["key"]] = True
            i += 2 if (f.get("value") and has_val and nxt == f.get("value")) else 1
            continue
        if f is not None and has_val:
            o[f["key"]] = nxt
            i += 2
            continue
        if f is not None:
            invalid.append(f"{w} needs a value")
            i += 1
            continue
        flags.append(w)                                     # not a setting here: kept as it is
        if has_val:
            flags.append(nxt)
            i += 2
        else:
            i += 1
        if w.startswith("-"):
            warnings.append(f"{w} is not one of {PACKAGES.get(package, {}).get('label', package)}'s known settings: kept as written")
    if total_ctx is not None:
        try:
            slots = int(spec.get("slots") or 1)
            spec["context"] = max(256, int(total_ctx) // max(1, slots))
        except ValueError:
            invalid.append("-c is a number")
    for k in ("port", "slots"):
        if k in spec:
            try:
                spec[k] = int(spec[k])
            except ValueError:
                invalid.append(f"{k} is a number")
    if o.get("spec_type") == "mtp" and o.get("draft_model") and not o.get("mtp"):   # llama.cpp: the MTP file is its draft model
        o["mtp"] = o.pop("draft_model")
    try:
        spec["options"] = check(package, o)
    except OptionError as e:
        invalid.append(str(e))
    if flags:
        spec["flags"] = flags
    if not spec["options"]:
        spec.pop("options")
    return {"spec": spec, "locked": sorted(set(locked)), "warnings": warnings, "invalid": invalid}


def diff_specs(old: dict, new: dict) -> list[dict]:
    out = []
    keys = sorted(set(old) | set(new))
    for k in keys:
        if k == "options":
            oo, no = old.get("options") or {}, new.get("options") or {}
            for ok in sorted(set(oo) | set(no)):
                if oo.get(ok) != no.get(ok):
                    out.append({"setting": ok, "from": oo.get(ok), "to": no.get(ok)})
        elif old.get(k) != new.get(k):
            out.append({"setting": k, "from": old.get(k), "to": new.get(k)})
    return out
