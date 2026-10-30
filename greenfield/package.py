#!/usr/bin/env python3
"""One-click installers for Nebula EH (design decision 2026-10-20: "one click install on Windows and Linux").

    python3 greenfield/package.py --platform linux   --impulse <folder with impulse-server and its libraries> --out dist
    python3 greenfield/package.py --platform windows --impulse <folder with impulse-server.exe and its DLLs>  --out dist

It builds the payload every installer carries, with nothing needed from the user's machine but the NVIDIA driver:
  python/    a standalone CPython 3.12 (python-build-standalone), with Nebula EH and every dependency already installed
             into it (offline: wheels for the target platform, fetched here)
  impulse/   impulse-server and the CUDA runtime it links (cudart, cuBLAS; libgomp on Linux); not Impulse Flash-Next
             (impulse-fx and its engine library): a piece of its own beside the installer, downloaded by the Flash-Next
             guided setup
  Halo       runs on impulse/ with an NVIDIA card; without one on Impulse portable (CPU + Vulkan), which is not inside:
             downloaded on such a machine only (the owner's 2026-10-27 word: "keep the downloads as small as possible";
             nebula/catalog/pieces.json). Its base model (Liquid's QAD Q4_0: LFM3-1.7B, 1.0 GB, with an NVIDIA card; LFM2.5-1.2B, 0.7 GB, without) is not inside
             either: the installer fetches it while it unpacks the rest (design decision 10-21), checked against the
             sha256 in nebula/catalog/halo.json; with no network then, Nebula fetches it when it starts. --halo-base puts
             it inside instead (halo/, an offline installer).
  app/       the launcher (its window, or the browser), the icon, the licences
Then (and, for each platform, nebula-eh-<version>-app-<platform>-x86_64.tar.gz: the small update, Nebula's wheel and
its constraints, for an install whose Impulse and the rest stay as they are):
  linux      nebula-eh-<version>-linux-x86_64.run: one file; run it (or double-click it) and it installs for this user
             (~/.local/share/nebula-eh), adds the "Nebula EH" icon and opens it. No root, no Python, no internet.
  windows    the payload folder and the NSIS script (greenfield/windows/installer.nsi); makensis (on the Windows
             build runner) makes Nebula-EH-Setup.exe from them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PBS = "20261013"
PY = "3.12.15"
PYTHONS = {
    "linux": f"cpython-{PY}+{PBS}-x86_64-unknown-linux-gnu-install_only.tar.gz",
    "windows": f"cpython-{PY}+{PBS}-x86_64-pc-windows-msvc-install_only.tar.gz",
}
PLATFORM_TAGS = {"linux": ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64"], "windows": ["win_amd64"]}
EXTRAS = {"linux": "gravity,monitor,lookup,docling", "windows": "gravity,monitor,lookup,docling,app"}   # docling: always installed
TORCH_CPU = "https://download.pytorch.org/whl/cpu"     # docling's PyTorch, the CPU build (PyPI's pulls in GBs of CUDA)   # Linux: the browser window
PURE_SDISTS = ["proxy_tools"]          # pure Python, published only as source (pywebview's): built into a wheel here first


def say(*a):
    print("[package]", *a, flush=True)


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    say("fetching", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)
    return dest


def version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def site_packages(py: Path, plat: str) -> Path:
    return py / "Lib" / "site-packages" if plat == "windows" else py / "lib" / "python3.12" / "site-packages"


def halo_bases() -> list[dict]:
    """Every Halo model's base (nebula/catalog/halo.json), the default (LFM3, NVIDIA) first."""
    c = json.loads((ROOT / "nebula" / "catalog" / "halo.json").read_text())
    ms = c.get("models") or [c]
    return [m["base"] for m in sorted(ms, key=lambda m: m["id"] != c.get("default"))]


def halo_base() -> dict:
    return halo_bases()[0]


# llama.cpp and ggml share one MIT licence text, "Copyright (c) 2023-2026 The ggml authors"
LLAMA_LICENSE = ROOT / "impulse" / "flashnext" / "third_party" / "ggml" / "LICENSE"   # ggml's MIT notice (One Engine P0 moved it)


def add_halo(pay: Path, base: Path) -> None:
    """halo/: one of Halo's base models, for an offline installer (else the one the machine runs is fetched at install)."""
    digest = hashlib.sha256(base.read_bytes()).hexdigest()
    m = next((b for b in halo_bases() if base.stat().st_size == b["bytes"] and digest == b["sha256"]), None)
    if m is None:
        raise SystemExit(f"{base}: not one of Halo's base models (size or sha256 differs from nebula/catalog/halo.json)")
    h = pay / "halo"
    h.mkdir()
    shutil.copy2(base, h / m["file"])
    say("halo: its base model, inside (offline)")


SANDBOX_FILES = ("sandbox.ps1", "sandbox-setup.sh", "sandbox-undo.ps1")


def add_sandbox(pay: Path, impulse_tar: Path | None) -> None:
    """Windows' Safe sandbox step (greenfield/windows/sandbox.ps1): its scripts and pins beside install.ps1, and in
    online.json the version and, when given, Impulse's Linux build it fetches (its release URL, its sha256 from here)."""
    for name in SANDBOX_FILES:
        shutil.copy2(ROOT / "greenfield" / "windows" / name, pay / name)
    pins = json.loads((ROOT / "nebula" / "catalog" / "wsl.json").read_text())
    shutil.copy2(ROOT / "nebula" / "catalog" / "wsl.json", pay / "wsl.json")
    o = json.loads((pay / "online.json").read_text())
    o["version"] = version()
    if impulse_tar is not None:
        o["sandbox_impulse"] = {"url": pins["impulse"]["url"].replace("{version}", version()),
                                "sha256": hashlib.sha256(impulse_tar.read_bytes()).hexdigest()}
    (pay / "online.json").write_text(json.dumps(o))
    say("sandbox:", ", ".join(SANDBOX_FILES), "+ wsl.json" + (f"; Impulse for Linux {o['sandbox_impulse']['sha256'][:12]}" if impulse_tar else ""))


def add_pair(pay: Path, pair_bin: Path, src: Path | None) -> None:
    """pair/: NVIDIA Personal AI Router's services (Go, static), which Nebula drives for model servers on your network."""
    d = pay / "pair"
    d.mkdir()
    for f in sorted(pair_bin.iterdir()):
        if f.is_file() and (f.name.startswith("nvpair-") or f.name == "inference-dispatcher"):
            shutil.copy2(f, d / f.name)
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        if src is not None and (src / name).is_file():
            shutil.copy2(src / name, d / name)
    say("pair:", ", ".join(sorted(p.name for p in d.iterdir())))


def build_payload(plat: str, impulse: Path | None, work: Path, cache: Path,
                  halo_base: Path | None = None,
                  pair_bin: Path | None = None, pair_src: Path | None = None, online: bool = False,
                  sandbox_impulse: Path | None = None) -> Path:
    pay = work / "payload"
    shutil.rmtree(pay, ignore_errors=True)
    pay.mkdir(parents=True)
    # 1. the standalone Python
    name = PYTHONS[plat]
    tgz = fetch(f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS}/{name}", cache / name)
    with tarfile.open(tgz) as t:
        t.extractall(pay, filter="data")                           # -> payload/python
    py = pay / "python"
    # 2. Nebula EH and its dependencies, installed into it for the target platform (offline at install time)
    wheels = work / "wheels"
    shutil.rmtree(wheels, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "-w", str(wheels), str(ROOT)], check=True)
    whl = next(wheels.glob("nebula_eh-*.whl"))
    subprocess.run([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "-w", str(wheels), *PURE_SDISTS], check=True)
    target = site_packages(py, plat)
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--target", str(target), "--only-binary=:all:",
           "--python-version", "3.12", "--implementation", "cp", "--upgrade", "--find-links", str(wheels)]
    for tag in PLATFORM_TAGS[plat]:
        cmd += ["--platform", tag]
    if "docling" in EXTRAS[plat]:
        cmd += ["--extra-index-url", TORCH_CPU]
    say(f"installing nebula-eh[{EXTRAS[plat]}] for {plat} into the payload")
    subprocess.run(cmd + [f"{whl}[{EXTRAS[plat]}]"], check=True)
    if online:
        # the online installer: not the Python nor the packages, but what installs them at the versions just resolved
        # and tested here (design decision 10-21: "Not Offline. It needs to pip install")
        pins = []
        for meta in sorted(target.glob("*.dist-info/METADATA")):
            head = dict(l.split(": ", 1) for l in meta.read_text(encoding="utf-8", errors="replace").splitlines()[:8] if ": " in l)
            if head.get("Name") and head.get("Version") and head["Name"].lower().replace("_", "-") != "nebula-eh":
                pins.append(f"{head['Name']}=={head['Version']}")
        (pay / "constraints.txt").write_text("\n".join(pins) + "\n")
        (pay / "wheels").mkdir()
        shutil.copy2(whl, pay / "wheels" / whl.name)
        (pay / "online.json").write_text(json.dumps({"python_url": f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS}/{name}",
                                                     "python_sha256": hashlib.sha256(tgz.read_bytes()).hexdigest(),
                                                     "wheel": whl.name, "extras": EXTRAS[plat], "version": version(),
                                                     "extra_index": TORCH_CPU if "docling" in EXTRAS[plat] else ""}))
        shutil.rmtree(py)
        if plat == "windows":
            shutil.copy2(ROOT / "greenfield" / "windows" / "install.ps1", pay / "install.ps1")
            add_sandbox(pay, sandbox_impulse)
        say(f"online: {len(pins)} packages pinned; Python and the packages are fetched at install")
    # 3. Impulse, and Halo's own model
    if impulse:
        # Impulse Flash-Next (impulse-fx and its engine library) is not inside: a piece of its own that the Flash-Next
        # guided setup downloads (nebula/pieces.py; make_flashnext writes it beside the installer)
        sys.path.insert(0, str(Path(__file__).parent))
        from impulse_dist import FX
        skip = list(FX[plat]) + (["libcudart.so*", "libcublas.so*", "libcublasLt.so*", "cudart64_*.dll", "cublas64_*.dll",
                                  "cublasLt64_*.dll"] if online else [])
        shutil.copytree(impulse, pay / "impulse", ignore=shutil.ignore_patterns(*skip))
        if plat == "linux" and (pay / "impulse" / "impulse-server").exists():   # today's wrapper: CUDA beside it, or from pip
            sys.path.insert(0, str(Path(__file__).parent))
            from impulse_dist import WRAPPER
            (pay / "impulse" / "impulse-server").write_text(WRAPPER)
            (pay / "impulse" / "impulse-server").chmod(0o755)

    if halo_base:
        add_halo(pay, halo_base)
    if pair_bin:
        add_pair(pay, pair_bin, pair_src)
    # 4. the launcher and the icon
    app = pay / "app"
    app.mkdir()
    shutil.copy(ROOT / "nebula" / "app" / "icon.svg", app / "nebula.svg")
    shutil.copy(ROOT / "nebula" / "app" / "icon.ico", app / "nebula.ico")
    for f in ("LICENSE", "NOTICE"):
        shutil.copy(ROOT / f, app / f)
    shutil.copy(LLAMA_LICENSE, app / "LICENSE-llama.cpp")          # Impulse links llama.cpp
    if (ROOT / "impulse" / "THIRD_PARTY_NOTICES.md").exists():
        shutil.copy(ROOT / "impulse" / "THIRD_PARTY_NOTICES.md", app / "THIRD_PARTY_NOTICES.md")
    (app / "VERSION").write_text(version() + "\n")
    return pay


RUN_HEADER = r'''#!/bin/sh
# Nebula EH __VERSION__ for Linux x86-64: run this file (or double-click it) to install it for this user.
# It installs from the internet (design decision 10-21: "Not Offline. It needs to pip install"): its own Python (pinned,
# sha256 checked), then Nebula and its packages with pip at the versions it was tested with; the programs nothing else
# provides (Impulse, llama-server, PAIR) are inside. It installs into ~/.local/share/nebula-eh and adds a "Nebula EH"
# icon; nothing needs root.
set -eu
QUIET=""                                               # --quiet: Nebula's own update (nebula/updates.py): no pop-ups, no opening
case "${1:-}" in --quiet) QUIET=1; NEBULA_NO_OPEN=1 ;; esac
DATA=${XDG_DATA_HOME:-$HOME/.local/share}
DEST=$DATA/nebula-eh
CONF=${XDG_CONFIG_HOME:-$HOME/.config}/nebula
say() { printf '%s\n' "$*"; [ -z "$QUIET" ] && command -v notify-send >/dev/null 2>&1 && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && notify-send "Nebula EH" "$*" 2>/dev/null || true; }
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || { say "This installer is for Linux on x86-64."; exit 1; }
say "Installing Nebula EH __VERSION__…"
SKIP=$(awk '/^__PAYLOAD__$/ { print NR + 1; exit }' "$0")
mkdir -p "$DATA" "$CONF"
# Halo's model is fetched while the rest unpacks (its sha256 checked); Nebula fetches it at start if this cannot. Only the one
# this machine runs: LFM3-1.7B (Halo v3.1, Impulse) with an NVIDIA card, LFM2.5-1.2B (Halo v2, Impulse portable) without one
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
  HALO_REL="__HALO_REL__"; HALO_URL="__HALO_URL__"; HALO_SHA="__HALO_SHA__"; HALO_WHAT="__HALO_WHAT__"
else
  HALO_REL="__HALO2_REL__"; HALO_URL="__HALO2_URL__"; HALO_SHA="__HALO2_SHA__"; HALO_WHAT="__HALO2_WHAT__"
fi
HALO_PID=""
if [ -n "$HALO_REL" ] && [ ! -f "$CONF/$HALO_REL" ] && { command -v curl || command -v wget; } >/dev/null 2>&1; then
  say "Fetching Halo's model ($HALO_WHAT) while the rest installs…"
  mkdir -p "$(dirname "$CONF/$HALO_REL")"
  ( P="$CONF/$HALO_REL.part"
    if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 --connect-timeout 15 -C - -o "$P" "$HALO_URL" || exit 1
    else wget -q -c -T 15 -t 3 -O "$P" "$HALO_URL" || exit 1; fi
    [ "$(sha256sum "$P" | cut -d' ' -f1)" = "$HALO_SHA" ] || { rm -f "$P"; echo "sha256 differs"; exit 1; }
    mv "$P" "$CONF/$HALO_REL" ) > "$CONF/halo-download.log" 2>&1 &
  HALO_PID=$!
fi
rm -rf "$DEST.new" && mkdir -p "$DEST.new"
trap 'rm -rf "$DEST.new"' EXIT                         # a failed install leaves nothing half-made (an earlier install stays)
tail -n +"$SKIP" "$0" | tar -xz -C "$DEST.new"
dl() { if command -v curl >/dev/null 2>&1; then curl -fsL --retry 3 --connect-timeout 15 -o "$2" "$1"; else wget -q -T 15 -t 3 -O "$2" "$1"; fi; }
if [ ! -d "$DEST.new/python" ]; then                  # the online installer: Python, then Nebula's packages with pip
  { command -v curl || command -v wget; } >/dev/null 2>&1 || { say "Nebula EH installs from the internet and needs curl or wget: sudo apt install curl"; exit 1; }
  say "Fetching Python and installing Nebula's packages…"
  T="$DEST.new/python.tar.gz"
  dl "__PY_URL__" "$T" || { say "Could not reach the internet to fetch Python: check the connection and run this again."; exit 1; }
  [ "$(sha256sum "$T" | cut -d' ' -f1)" = "__PY_SHA__" ] || { say "Python's download did not check out (sha256): run this again."; exit 1; }
  tar -xzf "$T" -C "$DEST.new" && rm -f "$T"
  "$DEST.new/python/bin/python3" -m pip install -q --no-cache-dir --disable-pip-version-check --no-warn-script-location \
    --constraint "$DEST.new/constraints.txt" __EXTRA_INDEX__ "$DEST.new/wheels/__WHEEL__[__EXTRAS__]" > "$CONF/pip.log" 2>&1 \
    || { say "Installing Nebula's packages failed (see $CONF/pip.log): check the connection and run this again."; exit 1; }
fi
rm -rf "$DEST.old"; [ -d "$DEST" ] && mv "$DEST" "$DEST.old"
mv "$DEST.new" "$DEST" && rm -rf "$DEST.old"
trap - EXIT
PY="$DEST/python/bin/python3"
# what this installer put in place: Nebula's updater offers a small update (code alone) to an install made by its base
printf '{"base": "%s", "version": "%s", "extras": "%s", "extra_index": "%s"}\n' "__VERSION__" "__VERSION__" "__EXTRAS__" "__EXTRA_INDEX_URL__" > "$DEST/app/install.json"
if [ ! -f "$CONF/nebula.json" ]; then
  printf '{\n "port": 8090,\n "ledger": "%s",\n "tiers": []\n}\n' "$CONF/nebula.sqlite3" > "$CONF/nebula.json"
fi
cat > "$DEST/nebula-open" <<LAUNCH
#!/bin/sh
# Opens Nebula EH: starts it when it is not running, then the browser at its page.
CONF="$CONF/nebula.json"
PORT=\$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("port", 8090))' "\$CONF")
if ! "$PY" -c 'import sys,urllib.request; urllib.request.urlopen("http://127.0.0.1:%s/api/signin" % sys.argv[1], timeout=2)' "\$PORT" 2>/dev/null; then
  [ -x "$DEST/openshell/bin/openshell" ] && export NEBULA_OPENSHELL="$DEST/openshell/bin/openshell"
  NEBULA_PAIR_BIN="$DEST/pair" NEBULA_IMPULSE_BIN="$DEST/impulse/impulse-server" nohup "$PY" -m nebula --config "\$CONF" > "$DEST/nebula.log" 2>&1 &
  for i in \$(seq 1 60); do "$PY" -c 'import sys,urllib.request; urllib.request.urlopen("http://127.0.0.1:%s/api/signin" % sys.argv[1], timeout=1)' "\$PORT" 2>/dev/null && break; sleep 0.5; done
fi
xdg-open "http://localhost:\$PORT/" >/dev/null 2>&1 || echo "Nebula EH is at http://localhost:\$PORT/"
LAUNCH
chmod +x "$DEST/nebula-open"
mkdir -p "$DATA/applications"
cat > "$DATA/applications/nebula-eh.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Nebula EH
Comment=Become Infinite: your own AI on this computer, shared with friends
Exec=$DEST/nebula-open
Icon=$DEST/app/nebula.svg
Terminal=false
Categories=Utility;Development;
DESKTOP
update-desktop-database "$DATA/applications" >/dev/null 2>&1 || true
say "Fetching what Nebula needs beside itself (Lookup's store and models, OpenShell, images)…"
"$PY" -m nebula.prereqs --config "$CONF/nebula.json" --dest "$DEST" 2>&1 | sed 's/^\[nebula\] /  /' | tee "$CONF/prereqs.log" || true
if [ -x "$DEST/impulse/impulse-server" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  say "No NVIDIA graphics card here, so Impulse (CUDA) is not used. Nebula still works: Halo and your models run on Impulse portable, a small download the first time (a Radeon or another card through Vulkan, else the CPU)."
elif [ -x "$DEST/impulse/impulse-server" ] && ! WHY=$("$DEST/impulse/impulse-server" --help 2>&1 >/dev/null); then
  case "$WHY" in
    *libcuda.so*) say "No NVIDIA graphics card here, so Impulse (CUDA) is not used. Nebula still works: Halo and your models run on Impulse portable, a small download the first time (a Radeon or another card through Vulkan, else the CPU)." ;;
    *) say "Impulse is included but cannot run on this machine ($(printf '%s' "$WHY" | head -1 | cut -c1-160)). Nebula still works: it can use other engines, or a friend's." ;;
  esac
fi
if [ -n "$HALO_PID" ]; then
  say "Finishing Halo's model download…"
  if wait "$HALO_PID"; then say "Halo's model is in place."; else say "Halo's model could not be fetched now; Nebula fetches it when it starts."; fi
fi
say "Nebula EH is installed. It opens now; next time, open \"Nebula EH\" from your applications."
[ -n "${NEBULA_NO_OPEN:-}" ] || "$DEST/nebula-open" >/dev/null 2>&1 &
exit 0
__PAYLOAD__
'''


def make_flashnext(impulse: Path, plat: str, out: Path) -> Path | None:
    """impulse-flashnext-<version>-<platform>-x86_64: Impulse Flash-Next's files (impulse_dist.FX), the piece the Flash-Next
    guided setup downloads (nebula/catalog/pieces.json): left out of the installers, to keep them small."""
    sys.path.insert(0, str(Path(__file__).parent))
    from impulse_dist import FX
    files = [impulse / n for n in FX[plat] if (impulse / n).is_file()]
    if not files:
        return None
    files.append(Path(__file__).resolve().parents[1] / "impulse" / "flashnext" / "data" / "expert-profile.bin")   # a layer split needs it
    out.mkdir(parents=True, exist_ok=True)
    if plat == "windows":
        import zipfile
        dest = out / f"impulse-flashnext-{version()}-windows-x86_64.zip"
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, f.name)
    else:
        dest = out / f"impulse-flashnext-{version()}-linux-x86_64.tar.gz"
        with tarfile.open(dest, "w:gz", compresslevel=6) as t:
            for f in files:
                t.add(f, arcname=f.name)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    (out / (dest.name + ".sha256")).write_text(f"{h}  {dest.name}\n")
    say("wrote", dest, f"{dest.stat().st_size / 2**20:.0f} MB (Impulse Flash-Next, a piece of its own)")
    return dest


def make_app(pay: Path, plat: str, out: Path) -> Path:
    """nebula-eh-<version>-app-<platform>.tar.gz: the small update (nebula/updates.py), Nebula's wheel and the package
    versions it was tested with, for an install whose Impulse and the rest beside Nebula stay as they are."""
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"nebula-eh-{version()}-app-{plat}-x86_64.tar.gz"
    whl = next((pay / "wheels").glob("nebula_eh-*.whl"))
    with tarfile.open(dest, "w:gz", compresslevel=9) as t:
        t.add(whl, arcname=whl.name)
        t.add(pay / "constraints.txt", arcname="constraints.txt")
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    (out / (dest.name + ".sha256")).write_text(f"{h}  {dest.name}\n")
    say("wrote", dest, f"{dest.stat().st_size / 2**20:.0f} MB (the small update)")
    return dest


def make_run(pay: Path, out: Path) -> Path:
    ver = version()
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"nebula-eh-{ver}-linux-x86_64.run"
    with tempfile.TemporaryDirectory() as d:
        tgz = Path(d) / "payload.tar.gz"
        with tarfile.open(tgz, "w:gz", compresslevel=6) as t:
            for p in sorted(pay.iterdir()):
                t.add(p, arcname=p.name)
        with open(dest, "wb") as f:
            online = json.loads((pay / "online.json").read_text()) if (pay / "online.json").exists() else {}
            head = RUN_HEADER
            for tag, hb, what in (("HALO", halo_bases()[0], "for this NVIDIA card: Halo v3.1"), ("HALO2", halo_bases()[-1], "no NVIDIA card here: Halo v2")):
                fetch = not (pay / "halo" / hb["file"]).exists()                    # fetched at install unless it is inside
                rel = f"models/{hb['repo']}/{hb['file']}" if fetch else ""      # hf.Pulls' layout under the config's folder
                url = f"https://huggingface.co/{hb['repo']}/resolve/{hb['revision']}/{hb['file']}"
                what = f"{hb['file'].split('-QAD')[0]}, {hb['bytes'] / 1e9:.1f} GB, {what}"
                head = (head.replace(f"__{tag}_REL__", rel).replace(f"__{tag}_URL__", url).replace(f"__{tag}_SHA__", hb["sha256"])
                        .replace(f"__{tag}_WHAT__", what))
            f.write(head.replace("__VERSION__", ver).replace("__PY_URL__", online.get("python_url", ""))
                    .replace("__PY_SHA__", online.get("python_sha256", "")).replace("__WHEEL__", online.get("wheel", ""))
                    .replace("__EXTRAS__", online.get("extras", "") or EXTRAS["linux"])
                    .replace("__EXTRA_INDEX_URL__", online.get("extra_index", "") or "")
                    .replace("__EXTRA_INDEX__", f'--extra-index-url "{online["extra_index"]}"' if online.get("extra_index") else "").encode())
            with open(tgz, "rb") as g:
                shutil.copyfileobj(g, f)
    dest.chmod(0o755)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    (out / (dest.name + ".sha256")).write_text(f"{h}  {dest.name}\n")
    say("wrote", dest, f"{dest.stat().st_size / 2**20:.0f} MB")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["linux", "windows"], required=True)
    ap.add_argument("--impulse", type=Path, help="a folder with impulse-server (and its libraries) for that platform")
    ap.add_argument("--halo-base", type=Path, help="one of Halo's base model files (LFM3-1.7B-QAD-Q4_0.gguf or LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf)")
    ap.add_argument("--pair-bin", type=Path, help="PAIR's services (services/build/bin of NVIDIA/Personal-AI-Router)")
    ap.add_argument("--pair-src", type=Path, help="PAIR's source, for its LICENSE and notices")
    ap.add_argument("--sandbox-impulse", type=Path, help="Windows: Impulse's Linux build (impulse-<v>-linux-x86_64.tar.gz) the Safe sandbox fetches from the release")
    ap.add_argument("--offline", action="store_true", help="carry Python, the packages and CUDA's libraries inside (no internet at install)")
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "nebula-package")
    a = ap.parse_args(argv)
    work = a.work or Path(tempfile.mkdtemp(prefix=f"nebula-pkg-{a.platform}-"))
    pay = build_payload(a.platform, a.impulse, work, a.cache, a.halo_base, a.pair_bin, a.pair_src,
                        online=not a.offline, sandbox_impulse=a.sandbox_impulse)
    make_app(pay, a.platform, a.out)
    if a.impulse:
        make_flashnext(a.impulse, a.platform, a.out)
    if a.platform == "linux":
        make_run(pay, a.out)
    else:
        say("payload ready for makensis:", pay)
        (a.out).mkdir(parents=True, exist_ok=True)
        (a.out / "payload-path.txt").write_text(str(pay) + "\n")
    print(json.dumps({"platform": a.platform, "payload": str(pay), "version": version()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
