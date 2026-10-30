"""nebula/prereqs.py: what Nebula needs beside itself, fetched and set up for this user, no administrator needed.

The one-click installer runs it after unpacking (design decision 10-21: "everything it needs should be in the prereq
downloads"); it can be run again any time, and each step is skipped when its piece is already here.

    python -m nebula.prereqs --config <nebula.json> [--dest <the install folder>] [--skip openshell,docker]

Pinned and checked (sha256), each step said in plain words:
  cuda        with an NVIDIA driver: CUDA's runtime and cuBLAS from NVIDIA's own pip packages, the versions Impulse was
              built with (Impulse reads them from this Python's site-packages)
  halo-runtime  without an NVIDIA card: Impulse portable (CPU + Vulkan), Halo's runtime there; not in the installer (to
              keep it small), so downloaded here and said
  docling     Lookup's document reading (PDF, Office, images): docling's layout, table and OCR models, fetched by reading
              one small document as Lookup would
  qdrant      Lookup's own vector store (nebula/rag/vectors.py's pinned release); the one Nebula installed is replaced
              when the pin moves (a Qdrant of your own is left as it is)
  embeddings  Lookup's embedding and reranking models (fastembed, into Lookup's models folder)
  openshell   NVIDIA OpenShell 0.1.4 (the version Nebula is tested with): its CLI and gateway into <dest>/openshell,
              the gateway as this user's systemd service with Nebula's settings, registered with the CLI. Skipped
              when OpenShell is already installed on this machine (a system package).
  docker      with Docker usable by this user: OpenShell's images and the Gravity gateway's image (iroh-relay)

What needs an administrator (Docker itself, NVIDIA's container toolkit, the NVIDIA driver) is never attempted:
`report()` says what is missing and the command to run. Halo's model is fetched by the installer and by Nebula
itself (nebula/halo_engine.py), and here checked against its pinned sha256 (a moved pin fetches the new file); PAIR's
services ship inside the installer.
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
import urllib.request
from pathlib import Path

CUDA_WHEELS = ["nvidia-cuda-runtime==13.2.86", "nvidia-cublas==13.4.1.3"]    # what Impulse was built with (CUDA 13.2.2)
OPENSHELL_VERSION = "0.1.4"
OPENSHELL_FILES = {            # GitHub's own digests of the v0.1.4 release files (x86-64 Linux)
    "cli": ("openshell-x86_64-unknown-linux-musl.tar.gz", "2b65178e62e1ece1d90d482555e9f1f98ff0296e058409b33418104b6333df07"),
    "gateway": ("openshell-gateway-x86_64-unknown-linux-gnu.tar.gz", "43f43554bcaabff2a8147444410d1ad0b7f51bdae7589c2d249982bc8b260b91"),
}
OPENSHELL_IMAGES = [f"ghcr.io/nvidia/openshell/supervisor:{OPENSHELL_VERSION}", f"ghcr.io/nvidia/openshell/sandbox:{OPENSHELL_VERSION}",
                    "ubuntu:24.04"]     # what an Impulse engine's own sandbox image is built on (openshell.BASE_IMAGE)
GATEWAY_PORT = 17670
GATEWAY_TOML = """# OpenShell gateway (written by Nebula's installer): built-in defaults, plus per-sandbox driver config for the Docker
# driver, so Nebula can mount the models folder read-only and pick which card an engine gets. Bind-backed volumes are
# allowed (the models folder is one) but admission stays on: only volumes carrying the approval labels attach.
[openshell]
version = 2

[openshell.drivers.docker]
allow_driver_config = true
enable_bind_mounts = true
"""
UNIT = """[Unit]
Description=OpenShell Gateway (installed by Nebula EH)
Documentation=https://github.com/NVIDIA/OpenShell
After=default.target

[Service]
Type=simple
StateDirectory=openshell/gateway
Environment=OPENSHELL_LOCAL_TLS_DIR=%h/.local/state/openshell/tls
EnvironmentFile=-%E/openshell/gateway.env
ExecStartPre={gw} config preflight
ExecStartPre={gw} generate-certs --output-dir ${{OPENSHELL_LOCAL_TLS_DIR}} --server-san host.openshell.internal
ExecStart={gw}
Restart=on-failure
RestartSec=5s
PrivateTmp=true
UMask=0077

[Install]
WantedBy=default.target
"""


def say(*a) -> None:
    print("[nebula]", *a, flush=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _fetch(url: str, dest: Path, sha256: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if _sha256(part) != sha256:
        part.unlink()
        raise RuntimeError(f"{dest.name}: the download does not match its sha256")
    part.rename(dest)
    return dest


def _home() -> Path:
    return Path(os.path.expanduser("~"))


def _xdg(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or _home() / default)


# --- CUDA's libraries, for Impulse and Halo's llama-server (only with an NVIDIA driver)
WINDOWS = os.name == "nt"
CUDA_LIBS = ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll") if WINDOWS else ("libcudart.so.13", "libcublas.so.13")


def cuda_lib_dir() -> Path:
    import sysconfig
    base = Path(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
    return base / "bin" / "x86_64" if WINDOWS else base / "lib"


def cuda(dest: str | None = None) -> str:
    if not shutil.which("nvidia-smi"):
        return "CUDA's libraries: not needed (no NVIDIA driver here); run this again after installing one"
    if not all((cuda_lib_dir() / f).is_file() for f in CUDA_LIBS):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--disable-pip-version-check", "--no-warn-script-location",
                        *CUDA_WHEELS], check=True, capture_output=True, text=True, timeout=3600)
        msg = "CUDA's libraries: " + ", ".join(CUDA_WHEELS) + " installed"
    else:
        msg = "CUDA's libraries: already here"
    if WINDOWS and dest:
        # Windows finds a program's DLLs beside it first: a copy beside impulse-server.exe and llama-server.exe
        for folder in ("impulse", "halo"):
            d = Path(dest, folder)
            if d.is_dir():
                for f in CUDA_LIBS:
                    if (cuda_lib_dir() / f).is_file() and not (d / f).is_file():
                        shutil.copy2(cuda_lib_dir() / f, d / f)
    return msg


# --- Halo's own model (the Linux installer fetches it while it unpacks; this fetches it where that did not happen). Only the
# model this machine's runtime runs (halo_model.model_for): LFM3-1.7B with an NVIDIA card (Impulse), LFM2.5-1.2B without one
def halo_model(cfg) -> str:
    from nebula import halo_model as hm, halo_engine
    from nebula.hf import Pulls
    import time
    models = (cfg.raw or {}).get("models_dir") or "models"
    models = models if os.path.isabs(models) else os.path.join(cfg.base_dir or os.getcwd(), models)
    runtimes = halo_engine.find_servers(cfg.base_dir)
    m = hm.manifest(model=hm.model_for(runtimes[0][0] if runtimes else None))
    what = (f"{m['base']['file'].split('-QAD')[0]}, {int(m['base']['bytes']) / 1e9:.1f} GB, "
            + ("no NVIDIA card here, so Halo v2 (Impulse portable)" if "Impulse portable" in (m.get("runtimes") or []) else "for this NVIDIA card (Halo v3.1)"))
    found = halo_engine.find_base(models, m)
    if found and hm._sha256(Path(found)) == m["base"]["sha256"]:     # the pinned file, not just one of its size
        return f"Halo's model: already here ({what})"
    if found and Path(found) != Path(hm.base_path(models, m)):
        # the installer's copy, NEBULA_HALO_BASE, or the owner's link to one elsewhere: theirs, never replaced
        return f"Halo's model: {found} is not the version this Nebula pins; left as it is"
    pulls = Pulls(models)
    pid = hm.ensure_base(pulls, m)                                 # a pin that moved: the new file, checked
    while pid is not None:
        st = pulls.status(pid)
        if st["state"] in ("done", "failed", "cancelled"):
            if st["state"] != "done":
                raise RuntimeError(next((f["error"] for f in st["files"] if f.get("error")), st["state"]))
            break
        time.sleep(1)
    return f"Halo's model: in place ({what}, checked against its sha256)"


# --- Lookup
def lookup_dir(cfg) -> str:
    settings = ((cfg.raw or {}).get("lookup") or {})
    return settings.get("data_dir") or os.path.join(cfg.base_dir or os.getcwd(), "nebula-lookup")


def qdrant(cfg) -> str:
    if WINDOWS:
        return "Lookup's vector store: kept inside Nebula on Windows (no separate program needed)"
    from nebula.rag import vectors
    data = lookup_dir(cfg)
    found = vectors.find_binary(((cfg.raw or {}).get("lookup") or {}).get("qdrant_bin"), data)
    own = os.path.join(data, "bin", "qdrant")
    if found and os.path.abspath(found) != os.path.abspath(own):
        return "Lookup's vector store: already here (your own Qdrant, used as it is)"
    if found and vectors.installed_version(found) == vectors.QDRANT_VERSION:
        return "Lookup's vector store: already here"
    vectors.install(data)                                          # none yet, or the one Nebula put here is an older pin
    return f"Lookup's vector store: Qdrant {vectors.QDRANT_VERSION} " + ("updated" if found else "installed")


def embeddings(cfg) -> str:
    for k, v in (("HF_HUB_DISABLE_PROGRESS_BARS", "1"), ("TQDM_DISABLE", "1"), ("HF_HUB_VERBOSITY", "error")):
        os.environ.setdefault(k, v)                                # one line said, not a page of progress bars
    import warnings
    warnings.filterwarnings("ignore", module="huggingface_hub")
    from nebula.rag.embed import DEFAULT_EMBED, DEFAULT_RERANK
    st = (cfg.raw or {}).get("lookup") or {}
    models = st.get("models_dir") or os.environ.get("FASTEMBED_CACHE_PATH") or os.path.join(lookup_dir(cfg), "models")   # as Lookup reads it
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    TextEmbedding(DEFAULT_EMBED, cache_dir=models)                 # downloads once, checked by fastembed
    TextCrossEncoder(DEFAULT_RERANK, cache_dir=models)
    return f"Lookup's models: {DEFAULT_EMBED} and {DEFAULT_RERANK} in place"


def tiny_pdf() -> bytes:
    """A small, valid one-page PDF with a line of text (its cross-reference table computed), for docling's first read."""
    text = b"BT /F1 18 Tf 36 72 Td (Nebula Lookup) Tj ET"
    objs = [b"<</Type/Catalog/Pages 2 0 R>>", b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
            b"<</Length %d>>stream\n" % len(text) + text + b"\nendstream", b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def docling() -> str:
    from nebula.rag import ingest
    if not ingest.docling_available():
        return "Document reading: docling is not installed (pip install \"nebula-eh[docling]\")"
    for k, v in (("HF_HUB_DISABLE_PROGRESS_BARS", "1"), ("TQDM_DISABLE", "1"), ("HF_HUB_VERBOSITY", "error")):
        os.environ.setdefault(k, v)
    with tempfile.TemporaryDirectory() as d:
        pdf = Path(d, "nebula-lookup-check.pdf")
        pdf.write_bytes(tiny_pdf())
        ingest._docling(pdf)                                       # as Lookup reads one: its models come down once
    return "Document reading: docling's models in place (PDF, Word, PowerPoint, Excel, images)"


# --- Halo's runtime without an NVIDIA card
def halo_runtime(cfg) -> str:
    """Impulse portable (CPU + Vulkan) when this machine has no NVIDIA card: Halo's runtime there. Not in the installer, to
    keep it small (the owner's word, 2026-10-27): downloaded here, pinned and sha256 checked, and said."""
    from nebula import halo_engine as HE, pieces
    if HE.has_nvidia():
        return "Halo's runtime: Impulse (this machine has an NVIDIA card)"
    if HE.find_portable(cfg.base_dir):
        return "Halo's runtime: Impulse portable, already here"
    if pieces.pin("impulse-portable") is None:
        return "Halo's runtime: " + pieces.say("impulse-portable") + "; Halo thinks with the Assistant's model"
    pieces.fetch("impulse-portable", cfg.base_dir)
    return "Halo's runtime: " + pieces.say("impulse-portable").replace("downloads", "downloaded", 1)


# --- OpenShell
def openshell_system() -> str | None:
    """An OpenShell already installed on this machine (a system package), which is used as it is."""
    for p in ("/usr/bin/openshell", "/usr/local/bin/openshell"):
        if os.path.isfile(p):
            return p
    return None


def openshell(dest: str) -> str:
    if WINDOWS:
        return "OpenShell: runs on Linux only; on Windows, engines run as plain programs (Nebula says so on the Engines page)"
    if openshell_system():
        return f"OpenShell: already installed on this machine ({openshell_system()}), used as it is"
    bindir = Path(dest, "openshell", "bin")
    cli, gw = bindir / "openshell", bindir / "openshell-gateway"
    if not (cli.is_file() and gw.is_file()):
        with tempfile.TemporaryDirectory() as d:
            for key, target in (("cli", cli), ("gateway", gw)):
                name, sha = OPENSHELL_FILES[key]
                tgz = _fetch(f"https://github.com/NVIDIA/OpenShell/releases/download/v{OPENSHELL_VERSION}/{name}", Path(d, name), sha)
                with tarfile.open(tgz) as t:
                    member = next(m for m in t.getmembers() if m.isfile() and os.path.basename(m.name) == target.name)
                    bindir.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(member) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                target.chmod(0o755)
    conf = _xdg("XDG_CONFIG_HOME", ".config") / "openshell"
    conf.mkdir(parents=True, exist_ok=True)
    toml = conf / "gateway.toml"
    if not toml.exists():
        toml.write_text(GATEWAY_TOML)
    started = _gateway_service(gw)
    registered = _register(cli) if started else False
    return (f"OpenShell {OPENSHELL_VERSION}: installed in {bindir}" +
            ("; its gateway runs as your service" if started else "; its gateway is set up but not started (systemd's user services are not available here)") +
            ("" if registered or not started else "; register it later with: openshell gateway add --local https://127.0.0.1:17670"))


def _gateway_service(gw: Path) -> bool:
    if not shutil.which("systemctl"):
        return False
    units = _xdg("XDG_CONFIG_HOME", ".config") / "systemd" / "user"
    unit = units / "openshell-gateway.service"
    if not unit.exists():
        units.mkdir(parents=True, exist_ok=True)
        unit.write_text(UNIT.format(gw=gw))
    run = lambda *a: subprocess.run(["systemctl", "--user", *a], capture_output=True, text=True, timeout=60)
    if run("daemon-reload").returncode != 0:
        return False
    return run("enable", "--now", "openshell-gateway.service").returncode == 0


def _register(cli: Path) -> bool:
    """The CLI's registration of the local gateway: its client certificate copied where the CLI reads it."""
    import time
    tls = _xdg("XDG_STATE_HOME", ".local/state") / "openshell" / "tls"
    for _ in range(30):                                       # the service writes its certificates as it starts
        if (tls / "client" / "tls.crt").is_file():
            break
        time.sleep(1)
    else:
        return False
    mtls = _xdg("XDG_CONFIG_HOME", ".config") / "openshell" / "gateways" / "openshell" / "mtls"
    mtls.mkdir(parents=True, exist_ok=True)
    for src, dst in ((tls / "ca.crt", "ca.crt"), (tls / "client" / "tls.crt", "tls.crt"), (tls / "client" / "tls.key", "tls.key")):
        if src.is_file():
            shutil.copy2(src, mtls / dst)
            (mtls / dst).chmod(0o600)
    r = subprocess.run([str(cli), "gateway", "add", "--local", "--name", "openshell", f"https://127.0.0.1:{GATEWAY_PORT}"],
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0 or "already" in (r.stdout + r.stderr).lower()


# --- Docker: images, when this user can use it
def docker_usable() -> str | None:
    d = shutil.which("docker")
    if not d:
        return None
    r = subprocess.run([d, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=30)
    return d if r.returncode == 0 else None


def docker_images() -> str:
    if WINDOWS:
        return "Docker images: not used on Windows"
    d = docker_usable()
    if not d:
        return "Docker images: skipped (Docker is not usable by you here)"
    have = set(subprocess.run([d, "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True).stdout.split())
    got = []
    for img in OPENSHELL_IMAGES:
        if img not in have:
            subprocess.run([d, "pull", "-q", img], check=True, capture_output=True, timeout=1800)
            got.append(img.rsplit("/", 1)[-1])
    from nebula import gateway
    if gateway.IMAGE not in have:
        subprocess.run([d, "build", "-q", "-t", gateway.IMAGE, "-f", str(gateway.HERE / "Containerfile"), str(gateway.HERE)],
                       check=True, capture_output=True, timeout=1800)
        got.append(gateway.IMAGE + " (Gravity's gateway)")
    return "Docker images: " + (", ".join(got) + " fetched" if got else "already here")


# --- what needs an administrator
def report() -> list[str]:
    out = []
    if not shutil.which("nvidia-smi"):
        out.append("No NVIDIA driver: Impulse needs an NVIDIA card and its driver (Halo and Lookup run on the CPU without it).")
    if WINDOWS:
        return out
    if not shutil.which("docker"):
        out.append("No Docker: OpenShell sandboxes need it. Install it with: sudo apt install docker.io && sudo usermod -aG docker $USER (then log in again).")
    elif not docker_usable():
        import grp
        try:
            member = os.getuid() == 0 or "docker" in [g.gr_name for g in (grp.getgrgid(x) for x in os.getgroups())]
        except (KeyError, OSError):
            member = False
        out.append("Docker is installed but not running: sudo systemctl enable --now docker" if member else
                   "Docker is installed but you cannot use it: sudo usermod -aG docker $USER (then log in again).")
    elif shutil.which("nvidia-smi") and not (shutil.which("nvidia-ctk") or shutil.which("nvidia-container-runtime")):
        out.append("No NVIDIA container toolkit: sandboxes run without the GPU until it is installed (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).")
    return out


STEPS = ("cuda", "halo", "halo-runtime", "qdrant", "embeddings", "docling", "openshell", "docker")


def run(cfg, dest: str, skip: set[str]) -> int:
    failed = 0
    for step in STEPS:
        if step in skip:
            continue
        try:
            msg = {"cuda": lambda: cuda(dest), "halo": lambda: halo_model(cfg), "halo-runtime": lambda: halo_runtime(cfg), "docling": docling, "qdrant": lambda: qdrant(cfg), "embeddings": lambda: embeddings(cfg), "openshell": lambda: openshell(dest),
                   "docker": docker_images}[step]()
            say(msg)
        except Exception as e:  # noqa: BLE001 - each piece on its own: the rest still installs
            failed += 1
            say(f"{step}: not done now ({str(e)[:200]}); run again later: python -m nebula.prereqs")
    for line in report():
        say(line)
    return failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch and set up what Nebula needs beside itself (no administrator needed)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dest", default=str(Path(sys.executable).resolve().parents[2]), help="the install folder (beside python/)")
    ap.add_argument("--skip", default="", help="steps to leave out, comma separated: " + ",".join(STEPS))
    a = ap.parse_args(argv)
    from nebula.config import Config
    cfg = Config.from_args(["--config", a.config])
    run(cfg, a.dest, {s.strip() for s in a.skip.split(",") if s.strip()})
    return 0                       # the installer goes on whatever happened: each piece said what to do


if __name__ == "__main__":
    sys.exit(main())
