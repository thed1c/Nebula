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
    rows = [dict(m2, lacks=fits(m2, have)) for m2 in (variant_of(m, variant) for m in cat["models"]) if m2.get("friends") or not friends]
    ok = {r["id"] for r in rows if not r["lacks"]}
    rec = next((i for i in cat.get("recommend") or [] if i in ok), next(iter(ok), None))
    runs, why = impulse_runs_here()
    return {"engine": cat.get("engine"), "machine": have, "models": rows, "recommended": rec, "impulse": runs, "impulse_why": why}


def get(mid: str, variant: str | None = None) -> dict | None:
    m = next((m for m in load()["models"] if m["id"] == mid), None)
    return variant_of(m, variant or default_variant()) if m is not None else None


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
