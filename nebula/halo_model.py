"""nebula/halo_model.py: Halo's own models (nebula/catalog/halo.json): Liquid's QAD Q4_0 files, unchanged, plus Nebula's two
LoRA adapters for each, shipped in nebula/assets/halo/ (docs/halo.md). Impulse's CUDA build (an NVIDIA card) runs Halo v3.1 on
LFM3-1.7B; Impulse portable (Vulkan / the CPU without an NVIDIA card) and llama-server run Halo v2 on LFM2.5-1.2B, which is
also an NVIDIA machine's model while LFM3's file is not there yet (the owner's 2026-10-30 pick: "Switch on NVIDIA now").

    manifest(model=)                    -> one model's entry (default: the catalog's "default"), the catalog's licence with it
    models()                            -> every model's entry, the default first
    model_for(runtime)                  -> the id of the model a runtime runs (NEBULA_HALO_MODEL forces one)
    verify_assets()                     -> [] or what is wrong with the shipped adapters (missing, size, sha256)
    base_path(models_dir)               -> where the base file lives (hf.Pulls' layout)
    ensure_base(pulls)                  -> None when the base is there and checks out, else a pull id (pinned revision, sha256)
    place(cards, margin_mb, prefer=)    -> "gpu:<index>" or "cpu" (the owner's order: spare GPU > spare VRAM; the CPU by choice
                                           (prefer="cpu", assistant.halo.where) or when no card has room)
    server_args(models_dir, variant, where, lane) -> llama-server's arguments for Halo
    set_variant(url, variant)           -> Standard or Uncensored on a running server, by the Uncensored adapter's scale
    choices(runtime)                    -> the models the owner may pick for this runtime (Halo › Version)
    fix_base(models_dir, m)             -> a model whose base needs a template fix ("fix"): the fixed copy, checked

A model marked "choice" (Halo Omni) runs only when the owner picks it (assistant.halo.model); its adapters may come as a
piece of their own ("piece": nebula/pieces.py) instead of the installer, in <config folder>/nebula-<piece>/ (m["_dir"]).

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


def _catalog(path: str | None = None) -> dict:
    return json.loads(Path(path or MANIFEST).read_text(encoding="utf-8"))


def manifest(path: str | None = None, model: str | None = None) -> dict:
    """One model's entry (its id, base, adapters, server), the catalog's licence with it; model None = the catalog's default.
    A catalog with no "models" list (one model, Nebula 0.1.4's form) is that model."""
    c = _catalog(path)
    if "models" not in c:
        return c
    want = model or c.get("default") or c["models"][0]["id"]
    for m in c["models"]:
        if m["id"] == want:
            return {**m, "license": c.get("license", "")}
    raise KeyError(f"Halo has no model {want!r} (nebula/catalog/halo.json)")


def models(path: str | None = None) -> list[dict]:
    """Every model's entry, the default first."""
    c = _catalog(path)
    if "models" not in c:
        return [c]
    d = c.get("default")
    return [manifest(path, m["id"]) for m in sorted(c["models"], key=lambda m: m["id"] != d)]


def model_for(runtime: str | None, path: str | None = None, choice: str | None = None) -> str:
    """The id of the model `runtime` runs: the owner's choice when this runtime runs it (Halo › Version), else the first
    model (the default first) whose "runtimes" names it, a "choice" model never by the runtime alone; NEBULA_HALO_MODEL
    forces one. Impulse's CUDA build -> LFM3; Impulse portable, llama.cpp, a server of the owner's (halo-server) -> LFM2.5."""
    env = os.environ.get("NEBULA_HALO_MODEL")
    ms = models(path)
    if env:
        if not any(m["id"] == env for m in ms):
            raise KeyError(f"NEBULA_HALO_MODEL={env}: Halo has no such model")
        return env
    if choice and any(m["id"] == choice and runtime in (m.get("runtimes") or []) for m in ms):
        return choice
    for m in ms:
        if runtime in (m.get("runtimes") or []) and not m.get("choice"):
            return m["id"]
    return [m for m in ms if not m.get("choice")][-1]["id"]   # an unknown runtime: the model every llama-server runs


def choices(runtime: str | None, path: str | None = None) -> list[dict]:
    """The models the owner may pick on this runtime: the one it runs by itself, and every "choice" model it can run."""
    first = model_for(runtime, path) if not os.environ.get("NEBULA_HALO_MODEL") else None
    return [m for m in models(path) if m["id"] == first or (m.get("choice") and runtime in (m.get("runtimes") or []))]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assets_dir(m: dict) -> Path:
    """Where a model's adapters are: the installer's assets, or its piece's folder (set as m["_dir"])."""
    return Path(m["_dir"]) if m.get("_dir") else ASSETS


def adapter_path(name: str, m: dict | None = None) -> Path:
    m = m or manifest()
    return assets_dir(m) / m["adapters"][name]["file"]


def verify_assets(m: dict | None = None) -> list[str]:
    """What is wrong with the shipped adapters ([] = both present, the right size and sha256)."""
    m = m or manifest()
    out = []
    for name, a in m["adapters"].items():
        p = assets_dir(m) / a["file"]
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


def fixed_path(models_dir: str, m: dict) -> str:
    """The fixed copy of a base that needs one, beside where the pull puts the base."""
    p = Path(base_path(models_dir, m))
    return str(p.with_name(p.stem + ".halo-fixed" + p.suffix))


def base_ok(models_dir: str, m: dict | None = None) -> bool:
    m = m or manifest()
    if m["base"].get("fix"):
        p = Path(fixed_path(models_dir, m))
        return p.is_file() and p.stat().st_size == m["base"]["bytes"] and _sha256(p) == m["base"]["fix"]["sha256"]
    p = Path(base_path(models_dir, m))
    return p.is_file() and p.stat().st_size == m["base"]["bytes"] and _sha256(p) == m["base"]["sha256"]


def _template_at(path) -> tuple[int, bytes]:
    """The offset and bytes of a GGUF file's tokenizer.chat_template (GGUF v3)."""
    import struct
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

        def skip(ty):
            if ty == 8:
                f.seek(rd("Q"), 1)
            elif ty == 9:
                sub, count = rd("I"), rd("Q")
                for _ in range(count):
                    skip(sub)
            else:
                f.seek(sizes[ty], 1)
        if f.read(4) != b"GGUF" or rd("I") != 3:
            raise ValueError("not a GGUF v3 file")
        rd("Q")
        for _ in range(rd("Q")):
            key, ty = f.read(rd("Q")), rd("I")
            if key == b"tokenizer.chat_template" and ty == 8:
                n = rd("Q")
                return f.tell(), f.read(n)
            skip(ty)
    raise ValueError("the file has no chat template")


def fix_base(models_dir: str, m: dict) -> str:
    """The base's fixed copy (its "fix": equal-length replacements in its chat template, nothing else changed), made
    from the checked base and checked against the fix's sha256; the base itself is then removed (a gigabyte saved).
    Its path."""
    fx, dest = m["base"]["fix"], Path(fixed_path(models_dir, m))
    if base_ok(models_dir, m):
        return str(dest)
    src = Path(base_path(models_dir, m))
    if not (src.is_file() and src.stat().st_size == m["base"]["bytes"] and _sha256(src) == m["base"]["sha256"]):
        raise ValueError(f"{m['name']}'s model file is not here yet, or does not check out")
    offset, tpl = _template_at(src)
    new = tpl
    for old, rep in fx["replace"]:
        if len(old) != len(rep):
            raise ValueError("a template fix keeps every length")
        new = new.replace(old.encode(), rep.encode())
    if new == tpl:
        raise ValueError(f"{m['name']}'s chat template has nothing to fix")
    tmp = dest.with_suffix(".tmp")
    import shutil
    shutil.copyfile(src, tmp)
    with open(tmp, "r+b") as f:
        f.seek(offset)
        f.write(new)
    if _sha256(tmp) != fx["sha256"]:
        tmp.unlink()
        raise ValueError(f"{m['name']}'s fixed copy does not match its sha256: not used")
    os.replace(tmp, dest)
    src.unlink()
    return str(dest)


def ensure_base(pulls, m: dict | None = None) -> str | None:
    """None when the base file is in place and checks out; else a pull of it at the pinned revision (hf.Pulls checks the
    sha256 before the file is renamed into place)."""
    m = m or manifest()
    if base_ok(pulls.models_dir, m):
        return None
    if m["base"].get("fix"):                              # the base is here (just pulled): its fixed copy now
        try:
            fix_base(pulls.models_dir, m)
            return None
        except ValueError:
            pass
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
            "--lora-scaled", f"{assets_dir(m) / halo['file']}:{halo['scale']},{assets_dir(m) / unc['file']}:{unc['scale'][variant]}",
            "-c", str(ctx), "-np", str(slots), "--jinja", "--no-webui", *s.get("flags", [])]   # LFM3: --reasoning off (it thinks by default)
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
