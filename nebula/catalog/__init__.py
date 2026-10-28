"""nebula/catalog: the models Nebula offers to deploy (impulse.json: the ones verified with Impulse 0.1.1), and what
this machine can run: its card's memory, its RAM and its free disk, and the one model to recommend."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).with_name("impulse.json")


def load(path: str | None = None) -> dict:
    return json.loads(Path(path or HERE).read_text(encoding="utf-8"))


def machine(models_dir: str | None = None) -> dict:
    """{gpu_mb (the largest card), gpu_name, ram_mb, disk_mb (free where models go)}."""
    from nebula import engines as E
    cards = E.cards()
    best = max(cards, key=lambda c: c.get("memory_mb") or 0) if cards else {}
    ram = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("MemTotal:"):
                    ram = int(ln.split()[1]) // 1024
    except OSError:
        pass
    if os.environ.get("NEBULA_FAKE_RAM_MB"):
        ram = int(os.environ["NEBULA_FAKE_RAM_MB"])
    d = models_dir or os.getcwd()
    while d and not os.path.exists(d):
        d = os.path.dirname(d)
    try:
        disk = shutil.disk_usage(d or "/").free // 2**20
    except OSError:
        disk = 0
    if os.environ.get("NEBULA_FAKE_DISK_MB"):
        disk = int(os.environ["NEBULA_FAKE_DISK_MB"])
    if not cards:                                    # no NVIDIA card: a Radeon (or another card) through Vulkan, if one is here
        best = vulkan_pick(vulkan_cards()) or {}
    return {"gpu_mb": int(best.get("memory_mb") or 0), "gpu_name": best.get("name"), "ram_mb": ram, "disk_mb": int(disk)}


def fits(m: dict, have: dict) -> list[str]:
    """What this machine lacks for the model, in plain words ([] = it fits)."""
    n, out = m.get("needs") or {}, []
    if have["gpu_mb"] < n.get("gpu_mb", 0):
        out.append(f"a graphics card with {round(n['gpu_mb'] / 1024)} GB" + (f" (this one has {round(have['gpu_mb'] / 1024)} GB)" if have["gpu_mb"] else " (none found)"))
    if have["ram_mb"] and have["ram_mb"] < n.get("ram_mb", 0):
        out.append(f"{round(n['ram_mb'] / 1024)} GB of memory (this computer has {round(have['ram_mb'] / 1024)} GB)")
    if have["disk_mb"] and have["disk_mb"] < n.get("disk_mb", 0):
        out.append(f"{round(n['disk_mb'] / 1024)} GB of free disk ({round(have['disk_mb'] / 1024)} GB free)")
    return out


def _lacks(m: dict, have: dict, impulse_runs: bool) -> list[str]:
    """fits(), for this machine's engine: without an NVIDIA card a Flash-Next model has none (Impulse only), and a GGUF
    runs on Impulse portable, where a smaller card is no stop (what does not fit on it stays in RAM)."""
    if impulse_runs:
        return fits(m, have)
    if m.get("ask_repo") or "flashnext" in m["id"]:
        return ["an NVIDIA graphics card (RTX 30, 40 or 50) on Linux: Flash-Next runs on Impulse only"]
    return [x for x in fits(m, have) if not x.startswith("a graphics card")]


def variant_of(m: dict, variant: str | None) -> dict:
    """A model as one of its variants: "standard" (the makers' own release) or "uncensored" (answers anything); the
    variant's fields over the entry's (a None removes one)."""
    v = (m.get("variants") or {}).get(variant or "")
    if v is None:
        return dict(m, variant=None)
    out = {**m, **v}
    return {k: x for k, x in out.items() if x is not None or k in ("revision",)} | {"variant": variant}


def default_variant() -> str:
    return load().get("default_variant") or "standard"


def offer(models_dir: str | None = None, friends: bool = False, variant: str | None = None) -> dict:
    """The catalog for this machine: each model with what it lacks, and the one to recommend (the first of the
    catalog's `recommend` order that fits). On a friend's path, only the models offered to friends."""
    cat, have = load(), machine(models_dir)
    variant = variant or default_variant()
    runs, why = impulse_runs_here()
    rows = [dict(m2, lacks=_lacks(m2, have, runs)) for m2 in (variant_of(m, variant) for m in cat["models"]) if m2.get("friends") or not friends]
    ok = {r["id"] for r in rows if not r["lacks"]}
    rec = next((i for i in cat.get("recommend") or [] if i in ok), next(iter(ok), None))
    return {"engine": cat.get("engine"), "machine": have, "models": rows, "recommended": rec, "impulse": runs, "impulse_why": why}


def get(mid: str, variant: str | None = None) -> dict | None:
    m = next((m for m in load()["models"] if m["id"] == mid), None)
    return variant_of(m, variant or default_variant()) if m is not None else None


def layout(m: dict, have: list[dict] | None = None) -> dict | None:
    """A model's first multi-card layout this machine can take ({"id", "card", "options", "about"}), or None: one card.
    A layout names how many cards it wants and their free memory together (`cards`, `min_total_mb`); the cards with the
    most free memory among the ones Impulse runs on are taken, split in proportion to it (tensor_split)."""
    from nebula import engines as E
    have = E.cards() if have is None else have
    ok = [c for c in have if not c.get("compute_cap") or c["compute_cap"] in IMPULSE_CAPS]
    ok.sort(key=lambda c: (c.get("memory_mb") or 0) - (c.get("used_mb") or 0), reverse=True)
    for lay in m.get("layouts") or []:
        n = int(lay.get("cards") or 1)
        pick = ok[:n]
        free = [max(0, (c.get("memory_mb") or 0) - (c.get("used_mb") or 0)) for c in pick]
        if len(pick) < n or sum(free) < int(lay.get("min_total_mb") or 0):
            continue
        return {"id": lay["id"], "card": ",".join(str(c.get("uuid") or c.get("index")) for c in pick),
                "options": {**(lay.get("options") or {}),
                            **({"tensor_split": ",".join(str(max(1, round(f / 1024))) for f in free)} if n > 1 else {})},
                "about": lay.get("about")}
    return None


_VULKAN: dict = {}
# a card of its own (its own memory), by its name: the rest are graphics inside the processor, whose "memory" is the
# computer's RAM (Intel UHD / Iris, AMD Radeon Graphics / Vega in a Ryzen)
import re as _re
DISCRETE = _re.compile(r"\bRX\b|\bArc\b|GeForce|\bRTX\b|\bGTX\b|Quadro|Tesla|Radeon Pro|FirePro|Instinct|\bW\d{4}\b", _re.I)


def vulkan_cards(base_dir: str | None = None) -> list[dict]:
    """The graphics cards Impulse portable sees through Vulkan, [{"id" (Vulkan0...), "name", "memory_mb", "free_mb",
    "integrated"}] (a Radeon, an Intel card; an NVIDIA card too): read from its --list-devices, at most once a minute;
    [] while Impulse portable is not here, or no card answers through Vulkan (then it runs on the CPU)."""
    import re
    import subprocess
    import time
    fake = os.environ.get("NEBULA_FAKE_VULKAN")
    if fake is not None:
        return json.loads(fake)
    from nebula import halo_engine as HE
    exe = HE.find_portable(base_dir)
    if not exe:
        return []
    seen = _VULKAN.get(exe)
    if seen and time.time() - seen["at"] < 60:
        return seen["cards"]
    try:
        out = subprocess.run([exe, "--list-devices"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    got = [{"id": m.group(1), "name": m.group(2).strip(), "memory_mb": int(m.group(3)), "free_mb": int(m.group(4)),
            "integrated": not DISCRETE.search(m.group(2))}
           for m in re.finditer(r"^\s*(Vulkan\d+): (.+) \((\d+) MiB, (\d+) MiB free\)", out, re.M)]
    _VULKAN[exe] = {"at": time.time(), "cards": got}
    return got


def vulkan_pick(vk: list[dict], name: str | None = None) -> dict | None:
    """The one card a model runs on through Vulkan: the one named; else the card of its own with the most free memory
    (never the graphics inside the processor while a card of its own is here); else the processor's graphics."""
    if name:
        return next((c for c in vk if c["name"] == name), None)
    own = [c for c in vk if not c.get("integrated")]
    return max(own or vk, key=lambda c: c["free_mb"]) if vk else None


IMPULSE_CAPS = ("8.6", "8.9", "12.0")     # the CUDA builds Impulse 0.1.1 ships (sm_86, sm_89 via sm_86, sm_120a)


def compute_caps() -> list[str]:
    import subprocess
    fake = os.environ.get("NEBULA_FAKE_CAPS")
    if fake is not None:
        return [c for c in fake.split(",") if c]
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def impulse_runs_here() -> tuple[bool, str | None]:
    """Whether Impulse 0.1.1 can run on this computer, and if not, why in plain words."""
    import platform
    if platform.system() not in ("Linux", "Windows") or platform.machine().lower() not in ("x86_64", "amd64"):
        return False, "Impulse runs on Linux and Windows PCs (x86-64) with an NVIDIA graphics card; this computer is not one"
    caps = compute_caps()
    if not caps:
        return False, "Impulse needs an NVIDIA graphics card, and none was found"
    if not any(c in IMPULSE_CAPS for c in caps):
        return False, "Impulse runs on NVIDIA RTX 30, 40 and 50 cards; this computer's card is not one of them"
    return True, None
