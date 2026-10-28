"""nebula/flashnext.py: Impulse Flash-Next, the engine for Qwen3.8 Flash-Next packs in Strata's NVFP4 format.

Impulse runs GGUF models on llama.cpp (`impulse`); a Flash-Next *pack* (a folder with `pack/`, a dense `*.gguf`, the PLE
and token-embedding GGUFs and `mtp/`, e.g. the OrcaRouter NVFP4 GPTQ pack the reference engine runs) runs on Impulse Flash-Next
(`impulse/flashnext`: our strata-nvfp4 tree). It runs in a sandbox from an image (`impulse/flashnext/nebula-image`):
the engine definition is a container `command` engine whose command is impulse-fx (Impulse's server for a pack, One
Engine; the reference engine runs it since P3, 2026-10-25): `impulse-fx -m /models/<pack> --host --port --fx-image-python <py>`
plus the run config, either as a file the image carries (`--fx-config <path>`, the reference engine's) or built here and passed
inline (`--fx-config-json <json>`).

`is_pack` tells a pack folder from a GGUF; `run_config` writes the run config the way the reference engine's was written by hand
(tests/test_flashnext.py checks it against the reference engine's live one); `engine_spec` is the Nebula engine definition.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Where a Flash-Next pack set up through "Run a model" runs (0.1.4): "sandbox", in an OpenShell sandbox like every other
# engine (its image carries impulse-fx beside impulse-server), or "local", straight on this computer. The owner's pick
# is asked (2026-10-27, sandbox recommended); the other is this one word.
PLACEMENT = "sandbox"
PACKAGE = "impulse-flashnext"
LABEL = "Impulse Flash-Next"
IMAGE = "nebula-impulse-flashnext:latest"
FX = "/opt/impulse/bin/impulse-fx"                  # Impulse's server for a pack (+ libimpulse_fx beside it)
IMAGE_PYTHON = "/opt/strata-serve/.venv/bin/python"  # a Python with Pillow: picture formats the encoder cannot read
SELFPIN = "/opt/nebula/libnebula-selfpin.so"
EXPERT_PROFILE = "/opt/strata-nvfp4/data/expert-profile.bin"
PIECE_PROFILE = "expert-profile.bin"                # the Impulse Flash-Next piece carries it beside impulse-fx


ROPE_TYPES = ("auto", "none", "linear", "yarn", "dynamic", "ntk")   # impulse-fx's "rope" (fx_server.cpp rope_args)
PARALLEL_MAX = 8                                    # a batch window holds 8 rows (the engine's kVerifyMaxT)
# What one batch slot's sessions take (flashnext/docs/BATCHING.md, int8 K/V, measured): 0.56 GiB at 32K cells, 1.00 at
# 64K, 3.65 at 262K -> ~0.12 GiB fixed (the 36 DeltaNet layers' state, ~113 MB, and the rest) + ~13.75 KiB a cell. With
# --kv-resident the K/V streams: only its 32K window stays in VRAM, the whole of it in pinned RAM.
SLOT_FIXED_MIB = 0.12 * 1024
SLOT_CELL_KIB = 0.44 * 1024 * 1024 / 32768
KV_RESIDENT_WINDOW = 32768
EXPERT_MIB = 2.6                                    # one expert of one layer in the NVFP4 pack (63 GB / 48 x 512)


def slot_cost(cells: int, *, kv_resident: int | None = None, expert_mib: float = EXPERT_MIB) -> dict:
    """An estimate of one batch slot: VRAM and pinned RAM in MiB and the expert-cache slots its VRAM is worth (impulse-fx
    logs the engine's own figure at startup: INFO batch_slot_mib)."""
    in_vram = min(cells, KV_RESIDENT_WINDOW) if kv_resident else cells
    vram = SLOT_FIXED_MIB + in_vram * SLOT_CELL_KIB / 1024
    pinned = cells * SLOT_CELL_KIB / 1024 if kv_resident else 0.0
    return {"vram_mib": round(vram), "pinned_mib": round(pinned), "expert_slots": round(vram / expert_mib)}


def pack_files(folder: str | os.PathLike) -> dict | None:
    """The pack's parts (paths), or None when `folder` is not a Flash-Next pack."""
    p = Path(folder)
    if not p.is_dir() or not (p / "pack").is_dir():
        return None
    dense = sorted(x for x in p.glob("*dense*.gguf"))
    if not dense:
        return None
    out = {"dir": str(p), "pack": str(p / "pack"), "dense": str(dense[0]), "tokenizer": str(p / "pack" / "tokenizer")}
    for key, pattern in (("ple", "ple-*.gguf"), ("embd", "token-embd-*.gguf")):
        hits = sorted(p.glob(pattern))
        if hits:
            out[key] = str(hits[0])
    if (p / "mtp").is_dir():
        out["mtp"] = str(p / "mtp")
    return out


def is_pack(path: str | os.PathLike) -> bool:
    return pack_files(path) is not None


# The reference engine's measured settings on the test server (2026-10-24: two sockets, 5070 + 5080, 376 GB RAM): passed by its own engine
# definition, never defaults. A Flash-Next engine on any other machine gets the engine's own defaults (pcie share tuned
# by the engine, VRAM reserve 700 MiB, all KV on the card, no conversation RAM cache, context from the pack).
REFERENCE_SETTINGS = {"max_context": 262144, "prefill": "auto:32768", "vram_reserve_mib": 768, "pool_workers": 34,
                 "kv_resident": 20480, "conversation_cache_mib": 65536, "conversation_cache_slots": 16, "pcie_frac": 0.04,
                 "lend_pct": 95, "layer_split": "16"}


# Long context for the reference engine (opt-in; its definition keeps 262144, the trained context). YaRN extends the rotation past it:
# dynamic = no scaling while a conversation fits 262144 (answers there identical to today), YaRN with the factor its
# length needs past it (the engine rotates the cached keys when it grows; --rope-dynamic-step), up to `factor`; static yarn =
# one factor for every position (works on today's engine build; slightly changes short answers). The expert cache shrinks
# by itself (the engine sizes it from what the larger session leaves): long_context_cost says by how much.
TRAINED_CTX = 262144
LONG_CONTEXT = {"384k": 393216, "512k": 524288, "768k": 786432, "1m": 1048576}


def long_context(size: str, method: str = "dynamic", step: float = 0.25) -> dict:
    """run_config / engine_spec keyword arguments for a long context: max_context and the "rope" block impulse-fx turns
    into --rope-scaling / --rope-scale / --yarn-orig-ctx (and --rope-dynamic-step)."""
    if size not in LONG_CONTEXT:
        raise ValueError(f"long context: one of {', '.join(LONG_CONTEXT)}, not {size!r}")
    if method not in ("dynamic", "yarn"):
        raise ValueError("long context: method dynamic or yarn")
    ctx = LONG_CONTEXT[size]
    rope = {"type": method, "factor": ctx / TRAINED_CTX, "orig_ctx": TRAINED_CTX}
    if method == "dynamic":
        rope["dynamic_step"] = step
    return {"max_context": ctx, "rope": rope}


def long_context_cost(ctx: int, *, split_qsa=(4, 8), kv_resident: bool = True, expert_mib: float = 2764816 / 2**20) -> dict:
    """What a context past 262144 costs on the reference engine's shape (12 full-attention layers, int8 K/V, layer split 16 on two
    cards: 4 of them on the first, 8 on the second): per card the extra VRAM against 262144 (indexer rows 128 B and page
    table 1 B per layer per cell, the rope table 256 B per card per cell, the prompt path's block scores up to 256 B per card
    per cell) and the expert-cache slots that is; the pinned host RAM the streamed K/V takes (1,056 B per layer per cell,
    the drafter's layer too)."""
    extra = max(0, ctx - TRAINED_CTX)
    per_layer = 128 + 1 + (0 if kv_resident else 1056)
    cards = [extra * (n * per_layer + 256 + 256) / 2**20 for n in split_qsa]
    return {"cards_mib": [round(c) for c in cards], "expert_slots": [round(c / expert_mib) for c in cards],
            "pinned_gib": round(ctx * 1056 * (sum(split_qsa) + 1) / 2**30, 1) if kv_resident else 0.0}


def run_config(files: dict, *, name: str, port: int, gpus: int = 1, layer_split: str | None = None,
               aliases: list[str] | None = None, max_context: int | None = None, prefill: str | None = None,
               pool_workers: int | None = None, vram_reserve_mib: int | None = None, kv_resident: int | None = None,
               conversation_cache_mib: int | None = None, conversation_cache_slots: int | None = None,
               pcie_frac: float | None = None, lend_pct: int | None = None, spec: int = 4, numa_nodes: str | None = None,
               vision: dict | None = None, extra_args: list[str] | None = None, parallel: int = 1,
               batch_max_context: int | None = None, rope: dict | None = None) -> dict:
    """The engine's run config (impulse-fx's --fx-config, Strata's serve.server --config without "exe"): paths as the
    engine sees them (under /models). Every tuning value is optional: None leaves it to the engine (REFERENCE_SETTINGS is
    one machine's measured set).

    `parallel` N > 1: up to N requests decode together in the engine's batch slots (impulse-fx turns it into the engine's
    --batch N; at most 8); 1 = one request at a time (the reference engine). Every slot's sessions come out of the expert cache, so
    size them with `batch_max_context` (the engine's --batch-max-context: the cells a slot holds; a longer request runs
    alone) - `slot_cost` estimates what a slot costs.

    `rope`: {"type": none|linear|yarn|dynamic|ntk, "factor": F, "orig_ctx": N}, impulse-fx turns it into the engine's
    --rope-scaling / --rope-scale / --yarn-orig-ctx. Without it impulse-fx picks by itself only past the trained
    context (262144): dynamic YaRN where the engine runs it, else static YaRN with factor context / 262144."""
    def opt(flag, v):
        return [flag, str(v)] if v is not None else []
    args = ["--pack", files["pack"], "--native", files["dense"], "--expert-profile", EXPERT_PROFILE, "--expert-cache", "auto"]
    args += opt("--prefill", prefill)
    if files.get("mtp") and spec > 0:
        args += ["--spec", str(spec), "--spec-min-p", "0.5", "--mtp", files["mtp"]]
    args += opt("--max-context", max_context) + ["--kv", "int8"]
    if vram_reserve_mib is not None:
        args += ["--vram-reserve-mib", str(vram_reserve_mib), "--vram-reserve-later-mib", str(vram_reserve_mib)]
    args += opt("--pool-workers", pool_workers) + opt("--kv-resident", kv_resident)
    args += opt("--conversation-cache-mib", conversation_cache_mib) + opt("--conversation-cache-slots", conversation_cache_slots)
    args += ["--native-dense-gguf", files["dense"]]
    if files.get("ple"):
        args += ["--ple-gguf", files["ple"]]
    if files.get("embd"):
        args += ["--embd-gguf", files["embd"]]
    if vision:
        args += ["--vision"]
    if pcie_frac is not None:
        args += ["--pcie-frac", f"{pcie_frac:g}"]
    if parallel < 1 or parallel > PARALLEL_MAX:
        raise ValueError(f"parallel must be 1..{PARALLEL_MAX} requests, not {parallel}")
    if batch_max_context is not None and parallel > 1:
        args += ["--batch-max-context", str(batch_max_context)]
    args += list(extra_args or [])
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if lend_pct is not None:
        env["STRATA_PREFILL_LEND_PCT"] = str(lend_pct)
    if numa_nodes:
        env["IMPULSE_NUMA_NODES"] = numa_nodes
    cfg = {"cwd": "/opt/strata-nvfp4", "tokenizer": files["tokenizer"],
           "model_name": name, "aliases": list(aliases or []), "log": "/sandbox/engine.log", "lib_dirs": ["/opt/cuda/lib"],
           "host": "127.0.0.1", "port": port, "gpu": list(range(gpus)), "gpu_order": "as_given"}
    if gpus > 1:
        cfg["layer_split"] = layer_split or "16"
    cfg.update({"parallel": parallel, "open_browser": False, "api_monitor": False, "idle_unload_s": 0, "lazy_load": False,
                "env": env, "args": args})
    if vision:
        cfg["vision"] = dict(vision)
    if rope is not None:
        bad = set(rope) - {"type", "factor", "orig_ctx", "dynamic_step"}
        if bad or rope.get("type", "auto") not in ROPE_TYPES:
            raise ValueError(f"rope: type one of {', '.join(ROPE_TYPES)}, optional factor and orig_ctx (got {rope})")
        cfg["rope"] = dict(rope)
    return cfg


def local_config(files: dict, *, name: str, port: int, gpus: int = 1, parallel: int = 1, profile: str | None = None) -> dict:
    """The run config for impulse-server's own Flash-Next mode (a pack run by the Impulse package, Run a model › a
    Flash-Next pack): run_config with the pack's paths as they are here and `profile` (the expert profile the Impulse
    Flash-Next piece carries; a layer split across cards needs it), without what only the reference image holds (its
    working folder, log and CUDA folder). impulse-server alone passes only --pack and --native, and the engine then
    wants the pack's PLE and embedding files too."""
    cfg = run_config(files, name=name, port=port, gpus=gpus, parallel=parallel, prefill="auto")   # serving needs one (the tray's)
    a = cfg["args"]
    i = a.index("--expert-profile")
    if profile:
        a[i + 1] = profile
    else:
        del a[i:i + 2]
    for k in ("cwd", "log", "lib_dirs"):
        cfg.pop(k, None)
    # no "gpu": impulse-server would set CUDA_VISIBLE_DEVICES to those indexes, over the cards Nebula chose (by UUID)
    cfg.pop("gpu", None)
    ls = cfg.pop("layer_split", None)
    if gpus > 1:
        a += ["--layer-split", ls or "16"]
    return cfg


def inside(files: dict, host_root: str) -> dict:
    """The pack's paths as the sandbox sees them: `host_root` (the engine's model folder) is mounted at /models."""
    root = os.path.realpath(host_root).rstrip("/")

    def there(x: str) -> str:
        r = os.path.realpath(x)
        if not r.startswith(root + "/"):
            raise ValueError(f"{x} is not under the model folder {host_root}")
        return "/models" + r[len(root):]
    return {k: there(v) for k, v in files.items()}


def command(pack_dir: str, port: int, cfg: dict, config_path: str | None = None) -> list[str]:
    """impulse-fx's command line: the pack folder as the sandbox sees it, the run config inline (--fx-config-json) or,
    with `config_path`, the file the image carries (--fx-config; it must hold `cfg`)."""
    conf = ["--fx-config", config_path] if config_path else ["--fx-config-json", json.dumps(cfg, separators=(",", ":"))]
    return [FX, "-m", pack_dir, "--host", "127.0.0.1", "--port", str(port), "--fx-image-python", IMAGE_PYTHON, *conf]


def engine_spec(name: str, *, machine: str | None, folder: str, files: dict, port: int, cards: list[str],
                image: str = IMAGE, numa_nodes: str | None = None, config_path: str | None = None, **cfg_kw) -> dict:
    """The Nebula engine definition: a container `command` engine running impulse-fx from the Impulse Flash-Next image.
    The run config built here goes inline, unless `config_path` names the copy the image carries (the reference engine's)."""
    inner = inside(files, folder)
    cfg = run_config(inner, name=cfg_kw.pop("model_name", name.lower()), port=port, gpus=max(1, len(cards)),
                     numa_nodes=numa_nodes, **cfg_kw)
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "LD_PRELOAD": SELFPIN}
    if cfg["env"].get("STRATA_PREFILL_LEND_PCT"):
        env["STRATA_PREFILL_LEND_PCT"] = cfg["env"]["STRATA_PREFILL_LEND_PCT"]
    if numa_nodes:
        env["IMPULSE_NUMA_NODES"] = numa_nodes
    spec = {"name": name, "kind": "command", "image": image, "folder": folder, "port": port, "card": ",".join(cards),
            "command": command(inner["dir"], port, cfg, config_path), "env": env, "ready_timeout_s": 1800, "host": "127.0.0.1",
            "note": f"{LABEL} ({PACKAGE}): {files.get('dir')}"}
    if machine:
        spec["machine"] = machine
        spec["where"] = "openshell"
    return spec
