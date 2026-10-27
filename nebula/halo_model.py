"""nebula/halo_model.py: Halo's own model (nebula/catalog/halo.json): Liquid's LFM2.5-1.2B QAD Q4_0 file, unchanged, plus
Nebula's two LoRA adapters shipped in nebula/assets/halo/ (docs/halo.md).

    verify_assets()                     -> [] or what is wrong with the shipped adapters (missing, size, sha256)
    base_path(models_dir)               -> where the base file lives (hf.Pulls' layout)
    ensure_base(pulls)                  -> None when the base is there and checks out, else a pull id (pinned revision, sha256)
    place(cards, margin_mb, prefer=)    -> "gpu:<index>" or "cpu" (the owner's order: spare GPU > spare VRAM; the CPU by choice
                                           (prefer="cpu", assistant.halo.where) or when no card has room)
    server_args(models_dir, variant, where, lane) -> llama-server's arguments for Halo
    set_variant(url, variant)           -> Standard or Uncensored on a running server, by the Uncensored adapter's scale

Standard is the base + the Halo adapter. Uncensored adds the Heretic adapter at scale 1; it is always loaded (scale 0 in
Standard), so switching is one call to the server's /lora-adapters and needs no reload.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = HERE / "catalog" / "halo.json"
ASSETS = HERE / "assets" / "halo"
VARIANTS = ("standard", "uncensored")


def manifest(path: str | None = None) -> dict:
    return json.loads(Path(path or MANIFEST).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def adapter_path(name: str, m: dict | None = None) -> Path:
    m = m or manifest()
    return ASSETS / m["adapters"][name]["file"]


def verify_assets(m: dict | None = None) -> list[str]:
    """What is wrong with the shipped adapters ([] = both present, the right size and sha256)."""
    m = m or manifest()
    out = []
    for name, a in m["adapters"].items():
        p = ASSETS / a["file"]
        if not p.is_file():
            out.append(f"{name}: {a['file']} is missing")
        elif p.stat().st_size != a["bytes"]:
            out.append(f"{name}: {a['file']} is {p.stat().st_size} bytes, not {a['bytes']}")
        elif _sha256(p) != a["sha256"]:
            out.append(f"{name}: {a['file']} does not match its sha256")
    return out


def base_path(models_dir: str, m: dict | None = None) -> str:
    from nebula.hf import Pulls
    b = (m or manifest())["base"]
    return Pulls(models_dir).target(b["repo"], b["file"])


def base_ok(models_dir: str, m: dict | None = None) -> bool:
    m = m or manifest()
    p = Path(base_path(models_dir, m))
    return p.is_file() and p.stat().st_size == m["base"]["bytes"] and _sha256(p) == m["base"]["sha256"]


def ensure_base(pulls, m: dict | None = None) -> str | None:
    """None when the base file is in place and checks out; else a pull of it at the pinned revision (hf.Pulls checks the
    sha256 before the file is renamed into place)."""
    m = m or manifest()
    if base_ok(pulls.models_dir, m):
        return None
    b = m["base"]
    return pulls.start(b["repo"], [b["file"]], revision=b["revision"], sha256={b["file"]: b["sha256"]}, sizes={b["file"]: b["bytes"]})


def place(cards: list[dict], margin_mb: int = 512, m: dict | None = None, prefer: str = "gpu") -> str:
    """Where Halo runs (design decision 2026-10-21: "I want Halo GPU, CPU by choice"): a spare GPU (no engine on it) >
    spare VRAM on an engine's card; the CPU when the owner picks it (prefer="cpu") or when no card has room.

    cards: [{"index", "free_mb", "engines": [names]}], read AFTER the engines have sized their caches, so what is free
    is what they do not need; Halo takes a card only when its share plus `margin_mb` fits in that."""
    if prefer == "cpu":
        return "cpu"
    need = int((m or manifest())["server"]["gpu_mb"]) + int(margin_mb)
    fit = [c for c in cards if int(c.get("free_mb") or 0) >= need]
    spare = [c for c in fit if not c.get("engines")]
    for group in (spare, fit):
        if group:
            best = max(group, key=lambda c: int(c.get("free_mb") or 0))
            return f"gpu:{best['index']}"
    return "cpu"


def server_args(models_dir: str, variant: str = "standard", where: str = "cpu", lane: str = "worker",
                port: int | None = None, m: dict | None = None, base: str | None = None) -> list[str]:
    """llama-server's arguments for Halo: the base, both adapters (the Uncensored one at the variant's scale), the slots of
    the lane (worker: handoffs and guard looks; router: one slot of its own so a long read never holds a routing call)."""
    m = m or manifest()
    if variant not in VARIANTS:
        raise ValueError(f"variant is one of {', '.join(VARIANTS)}")
    s = m["server"]
    halo, unc = m["adapters"]["halo"], m["adapters"]["uncensored"]
    slots, ctx = (s["router_lane"]["slots"], s["router_lane"]["context"]) if lane == "router" else (s["slots"], s["context"])
    args = ["-m", base or base_path(models_dir, m),
            "--lora-scaled", f"{ASSETS / halo['file']}:{halo['scale']},{ASSETS / unc['file']}:{unc['scale'][variant]}",
            "-c", str(ctx), "-np", str(slots), "--jinja", "--no-webui"]
    if where.startswith("gpu"):
        args += ["-ngl", "99"]
    else:
        args += ["-ngl", "0", "-t", str(s["cpu_threads"])]
    if port:
        args += ["--host", "127.0.0.1", "--port", str(port)]
    return args


def set_variant(url: str, variant: str, m: dict | None = None, timeout: float = 10.0, api_key: str | None = None) -> list[dict]:
    """Standard or Uncensored on a running Halo server: the adapters' scales in load order (Halo, then Uncensored)."""
    m = m or manifest()
    if variant not in VARIANTS:
        raise ValueError(f"variant is one of {', '.join(VARIANTS)}")
    body = [{"id": 0, "scale": float(m["adapters"]["halo"]["scale"])},
            {"id": 1, "scale": float(m["adapters"]["uncensored"]["scale"][variant])}]
    req = urllib.request.Request(url.rstrip("/") + "/lora-adapters", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return body
