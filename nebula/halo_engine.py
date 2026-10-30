"""nebula/halo_engine.py: Halo's own model, running for Halo alone (docs/halo.md, nebula/halo_model.py).

When the base file and a runtime are on this machine, Nebula starts Halo's server at startup, in an Impulse Engine (design decision 10-21; Impulse portable without an NVIDIA card): the owner's placement (below; on the CPU threads capped and `nice`), the
Standard or Uncensored adapter scale, its log beside the config (halo-engine.log). It is not one of the models: it is
not on the map, not routed to, not shared over Gravity. Halo's own calls carry a view of the config that holds it
alone (nebula/halo.py), so they go through the same chat path (its tools, the ledger) as any other.

    "assistant": {"halo": {"own_model": true, "variant": "standard", "where": "gpu"}}
      own_model false: Halo thinks with the Assistant's model
      where "gpu" (the default, design decision 10-21: "I want Halo GPU, CPU by choice"): a spare GPU, else spare VRAM on an
        engine's card; the CPU only when no card has room (said on Halo's panel); "cpu": the CPU, by the owner's choice

Where things are found (first wins):
  the runtime   NEBULA_HALO_SERVER (a command line); else, with an NVIDIA card, Impulse (packages.locate: the
                installer's impulse/, Nebula's own, PATH; Linux and Windows); else, or when Impulse fails, Impulse
                portable (impulse/portable: CPU + Vulkan, for a Radeon or Intel card or the CPU), which is not in the
                installers: Nebula downloads it (nebula/pieces.py, sha256 checked) on a machine
                without an NVIDIA card, and says so; last, a llama-server of the owner's own on the PATH
  the base      NEBULA_HALO_BASE, the installer's halo/ folder, the models folder (hf.Pulls' layout, as fetched)
  the model     the runtime's (halo_model.model_for, the owner's 2026-10-30 pick "Switch on NVIDIA now"): Impulse (an NVIDIA
                card) runs Halo v3.1 on LFM3-1.7B; Impulse portable and llama-server run Halo v2 on LFM2.5-1.2B. An NVIDIA
                machine with only v2's file (Nebula before 0.1.5) keeps Halo v2 running while LFM3's file downloads and moves
                to it when it is in place; when the download fails, or Impulse cannot run LFM3, Halo v2 stays

The base model is not inside the installer: the installer fetches it while it unpacks the rest (design decision 10-21: "We will
just download while other items are installing"). When it is still missing at start (no network then), Nebula fetches it
here (pinned revision, sha256 checked, resumed from the installer's partial file) and starts Halo's model when it lands,
retrying while the network is away.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import shutil
import socket
import sys
import threading
from pathlib import Path

from nebula import halo_model

LOG = logging.getLogger("nebula.halo")
EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


def _bundle_dirs() -> list[Path]:
    """The installer's halo/ folder: beside its Python (python/bin/python3 or python/python.exe)."""
    here = Path(sys.executable).resolve()
    return [p / "halo" for p in list(here.parents)[:3]]


def find_llama_server() -> list[str] | None:
    for d in _bundle_dirs():
        for name in ("llama-server", EXE):              # the Linux bundle's wrapper sets its library path
            if (d / name).is_file():
                return [str(d / name)]
    w = shutil.which("llama-server")
    return [w] if w else None


# Halo on a Radeon (design decision 10-22: "Halo should also work on the Radeon"): Impulse portable's Vulkan backend (llama.cpp's) reaches AMD and
# Intel cards; Impulse's own build is CUDA only. A discrete non-NVIDIA card with room is the first spare GPU (Impulse's models cannot
# use it, so the NVIDIA card stays theirs); integrated graphics share the system's memory and are left out.
VK_LINE = re.compile(r"^\s*(Vulkan\d+):\s*(.+?)\s*\((\d+) MiB,\s*(\d+) MiB free\)")
INTEGRATED = re.compile(r"UHD|Iris|^AMD Radeon\(TM\) Graphics$|^AMD Radeon Graphics$|llvmpipe|Microsoft Basic", re.I)
_VK_CACHE: dict = {}


def vulkan_gpus(cmd: list[str] | None) -> list[dict]:
    """The non-NVIDIA discrete cards a Vulkan runtime (Impulse portable, llama-server) sees: [{"device": "Vulkan0", "name", "free_mb"}]."""
    if not cmd:
        return []
    key = tuple(cmd)
    if key not in _VK_CACHE:
        out = []
        try:
            r = subprocess.run(list(cmd) + ["--list-devices"], capture_output=True, text=True, timeout=60,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            for line in (r.stdout + "\n" + r.stderr).splitlines():
                m = VK_LINE.match(line)
                if m and "NVIDIA" not in m.group(2).upper() and not INTEGRATED.search(m.group(2).strip()):
                    out.append({"device": m.group(1), "name": m.group(2).strip(), "free_mb": int(m.group(4))})
        except (OSError, subprocess.SubprocessError):
            out = []
        _VK_CACHE[key] = out
    return list(_VK_CACHE[key])


def lora_paths_without_colons(args: list[str], cwd: str | None, nt: bool = os.name == "nt") -> list[str]:
    """llama.cpp's server (Impulse's too) reads --lora-scaled as FILE:SCALE,...: a Windows path's drive colon (C:\\...) breaks it, so on Windows
    each adapter is given relative to the folder the server runs in (same drive: no colon)."""
    if not nt or not cwd or "--lora-scaled" not in args:
        return args
    import ntpath
    out = list(args)
    i = out.index("--lora-scaled") + 1
    parts = []
    for item in out[i].split(","):
        path, _, scale = item.rpartition(":")
        try:
            rel = ntpath.relpath(path, cwd)
        except ValueError:                                # another drive: left as it is
            rel = path
        parts.append(f"{rel}:{scale}")
    out[i] = ",".join(parts)
    return out


def find_portable(base_dir: str | None = None) -> str | None:
    """Impulse portable's program: NEBULA_IMPULSE_PORTABLE_BIN, or the one Nebula downloaded (nebula/pieces.py)."""
    from nebula import pieces
    env = os.environ.get("NEBULA_IMPULSE_PORTABLE_BIN")
    if env and os.path.isfile(env):
        return env
    return pieces.find("impulse-portable", base_dir)


def has_nvidia() -> bool:
    from nebula import engines
    try:
        return bool(engines.cards())
    except Exception:  # noqa: BLE001
        return False


def find_servers(base_dir: str | None = None) -> list[tuple[str, list[str]]]:
    """Halo's runtimes in the order tried (design decision 10-21: Halo in an Impulse Engine): Impulse with an NVIDIA card
    (its CUDA build cannot start without one), then Impulse portable (CPU + Vulkan), then a llama-server of the owner's."""
    env = os.environ.get("NEBULA_HALO_SERVER")
    if env:
        from nebula import cmdline
        return [("halo-server", cmdline.split(env))]
    out = []
    if has_nvidia():
        from nebula import packages
        imp = packages.locate("impulse", base_dir)
        if imp:
            out.append(("Impulse", [imp]))
    port = find_portable(base_dir)
    if port:
        out.append(("Impulse portable", [port]))
    ls = find_llama_server()
    if ls:
        out.append(("llama.cpp", ls))
    return out


def find_base(models_dir: str, m: dict | None = None) -> str | None:
    m = m or halo_model.manifest()
    env = os.environ.get("NEBULA_HALO_BASE")
    if env:
        return env if os.path.isfile(env) else None
    want = int(m["base"]["bytes"])
    for p in [d / m["base"]["file"] for d in _bundle_dirs()]:
        if p.is_file() and p.stat().st_size == want:     # the installer checked its sha256
            return str(p)
    b = m["base"]
    p = Path(models_dir, *b["repo"].split("/"), *b["file"].split("/"))     # where a pull puts it (hf.Pulls' layout)
    if p.is_symlink():
        # the owner's link to a copy elsewhere (another Nebula's, say): read where it points, never refused
        try:
            real = p.resolve(strict=True)
            if real.is_file() and real.stat().st_size == want:
                return str(real)
        except OSError:
            pass
        LOG.warning("Halo's model: %s links to a file that is not his model", p)
        return None
    if p.is_file() and p.stat().st_size == want:         # the pull checked its sha256
        return str(p)
    return None


def _gb(m: dict) -> str:
    return f"{int(m['base']['bytes']) / 1e9:.1f} GB"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class HaloEngine:
    def __init__(self, server):
        self.server, self.cfg = server, server.cfg
        self.proc = None
        self.tier = None                 # the Tier Halo's calls use, once its server answers
        self.where: str | None = None
        self.runtime: str | None = None   # Impulse, Impulse portable or llama.cpp, once one answers
        self.card = ""                    # the card's UUID: an index can name another card than nvidia-smi's on mixed rigs
        self.vk_device: str | None = None   # Halo on a Radeon (or other non-NVIDIA card): its Vulkan device, e.g. Vulkan0
        self.gpu_name: str | None = None
        self.why: str | None = None
        self.state = "off"
        self.progress: dict | None = None
        self.m: dict | None = None        # the model entry it runs (halo_model.manifest), once chosen
        self.upgrade: dict | None = None  # LFM3 downloading while Halo v2 runs: {"model", "progress", "why"}
        self._wake = threading.Event()

    def settings(self) -> dict:
        a = self.cfg.assistant if isinstance(self.cfg.assistant, dict) else {}
        h = a.get("halo") if isinstance(a.get("halo"), dict) else {}
        return {"own_model": h.get("own_model", True) is not False, "variant": h.get("variant") or "standard",
                "where": "cpu" if h.get("where") == "cpu" else "gpu"}

    def models_dir(self) -> str:
        yard = getattr(self.server, "yard", None)
        if yard is not None and hasattr(yard, "models_dir"):
            return yard.models_dir()
        d = (self.cfg.raw or {}).get("models_dir") or "models"         # as nebula/engines.py reads it
        return d if os.path.isabs(d) else os.path.join(self.cfg.base_dir or os.getcwd(), d)

    def place(self) -> str:
        self.vk_device = self.gpu_name = None
        if self.settings()["where"] == "cpu":
            return "cpu"                                  # the owner's choice
        # a spare Radeon (or other non-NVIDIA card) first: Halo there through Impulse portable's Vulkan backend
        need = int((self.m or halo_model.manifest())["server"]["gpu_mb"]) + 512
        ls = next((cmd for rt, cmd in find_servers(self.cfg.base_dir) if rt != "Impulse"), None)
        vk = [d for d in vulkan_gpus(ls) if d["free_mb"] >= need]
        if vk:
            best = max(vk, key=lambda d: d["free_mb"])
            self.vk_device, self.gpu_name, self.card = best["device"], best["name"], ""
            return "vk:" + best["device"][len("Vulkan"):]
        from nebula import engines
        try:
            have = engines.cards()
        except Exception:  # noqa: BLE001 - no cards: the CPU
            have = []
        self.no_cards = not have                          # said on Halo's panel: no GPU here, not "no room"
        specs = (self.cfg.raw or {}).get("engines") or []
        try:                                              # only cards its Impulse build has kernels for (sm_75 is not in every one)
            from nebula import machines
            run = find_servers(self.cfg.base_dir)
            prog = run[0][1][0] if run and run[0][0] == "Impulse" else None
            if prog and os.path.isfile(prog):
                have = [c for c in have if not machines.arch_gap(prog, [c])]
        except Exception:  # noqa: BLE001 - unreadable build: every card, as before
            pass
        cards = [{"index": c["index"], "free_mb": int(c.get("memory_mb", 0)) - int(c.get("used_mb", 0)),
                  "engines": [s.get("name") for s in specs
                              if {c.get("uuid"), str(c.get("index"))} & {x.strip() for x in str(s.get("card") or "").split(",")}]}
                 for c in have]
        where = halo_model.place(cards, m=self.m)
        hit = next((c for c in have if where == f"gpu:{c['index']}"), None)
        self.card = (hit or {}).get("uuid") or (where.split(":", 1)[1] if where.startswith("gpu") else "")
        return where

    def start(self) -> "HaloEngine":
        """Never stops Nebula's own start: whatever goes wrong, Halo says why and thinks with the Assistant's model."""
        try:
            return self._start()
        except Exception as e:  # noqa: BLE001
            self.state, self.why = "unavailable", str(e)[:300]
            LOG.warning("Halo's own model: %s", e)
            return self

    def _start(self) -> "HaloEngine":
        st = self.settings()
        if not st["own_model"]:
            self.state, self.why = "off", "turned off: Halo thinks with the Assistant's model"
            return self
        server = find_servers(self.cfg.base_dir)
        self.m = halo_model.manifest(model=halo_model.model_for(server[0][0] if server else None))
        bad = halo_model.verify_assets(self.m)
        base = find_base(self.models_dir(), self.m)
        if not bad and not has_nvidia() and not any(rt != "Impulse" for rt, _ in server):
            from nebula import pieces                     # no NVIDIA card: Impulse portable, downloaded now, said plainly
            if pieces.pin("impulse-portable") is not None:
                self.state = "downloading"
                self.why = "fetching its runtime: " + pieces.say("impulse-portable")
                threading.Thread(target=self._fetch_runtime_then_start, name="halo-runtime", daemon=True).start()
                return self
        if bad or not server:
            self.state = "unavailable"
            self.why = ("its adapters: " + "; ".join(bad)) if bad else "no runtime for it on this machine (Impulse, or Impulse portable without an NVIDIA card)"
            return self
        if not base:
            if os.environ.get("NEBULA_HALO_BASE"):
                self.state, self.why = "unavailable", "NEBULA_HALO_BASE names no file"
                return self
            fb = self._fallback(server)
            if fb is not None:                            # Halo v2's file is here (an update from 0.1.4): v2 runs now,
                want, (self.m, base) = self.m, fb         # LFM3's file comes in the background and Halo moves to it
                self.upgrade = {"model": want["id"], "progress": None,
                                "why": f"Halo moves to {want['name']} when its model is in place: fetching {_gb(want)} for this NVIDIA card; Halo v2 runs meanwhile"}
                threading.Thread(target=self._fetch_upgrade, args=(want,), name="halo-upgrade", daemon=True).start()
            else:
                self.state, self.why = "downloading", f"fetching its model ({_gb(self.m)}: {self.m['name']})"
                threading.Thread(target=self._fetch_then_run, args=(server, st["variant"]), name="halo-fetch", daemon=True).start()
                return self
        self.state = "starting"
        threading.Thread(target=self._run, args=(server, base, st["variant"]), name="halo-engine", daemon=True).start()
        return self

    RETRY_S = 120.0

    def _fallback(self, server: list[tuple[str, list[str]]]) -> tuple[dict, str] | None:
        """Another model these runtimes run whose file is already here (Halo v2 on an NVIDIA machine): (its entry, its file)."""
        rts = {rt for rt, _ in server}
        for m in halo_model.models():
            if m["id"] != (self.m or {}).get("id") and rts & set(m.get("runtimes") or []) and not halo_model.verify_assets(m):
                b = find_base(self.models_dir(), m)
                if b:
                    return m, b
        return None

    def _fetch_upgrade(self, m: dict) -> None:
        """The chosen model's file (pinned revision, sha256) while the fallback runs; then Halo restarts on it. While offline,
        or when the file does not check out, the fallback stays and the fetch is tried again later."""
        import time
        from nebula.hf import Pulls
        pulls = Pulls(self.models_dir())
        wake = self._wake
        while self.state != "stopped" and self.upgrade is not None:
            try:
                pid = halo_model.ensure_base(pulls, m)
                while pid is not None and self.state != "stopped":
                    st = pulls.status(pid)
                    self.upgrade["progress"] = {"bytes": st.get("bytes"), "total": st.get("total")}
                    if st["state"] in ("done", "failed", "cancelled"):
                        if st["state"] != "done":
                            raise RuntimeError(next((f["error"] for f in st["files"] if f.get("error")), st["state"]))
                        break
                    time.sleep(1)
                if self.state == "stopped":
                    return
                if not find_base(self.models_dir(), m):
                    raise RuntimeError("the file did not check out")
                LOG.info("Halo's model: %s is in place; Halo moves to it", m["name"])
                self.upgrade = None
                self.restart()
                return
            except Exception as e:  # noqa: BLE001 - offline, or the Hub away: Halo v2 keeps running, tried again later
                self.upgrade["why"] = (f"{m['name']} could not be fetched yet ({str(e)[:160]}); Halo v2 runs; trying again in "
                                       f"{int(self.RETRY_S)} s")
                LOG.info("Halo's model: %s", self.upgrade["why"])
                wake.wait(self.RETRY_S)
                if wake is not self._wake or self.state == "stopped":
                    return

    def _fetch_runtime_then_start(self) -> None:
        """Impulse portable (pinned, sha256), then Halo's start as usual; while offline, tried again later."""
        while self.state == "downloading":
            try:
                from nebula import pieces
                pieces.fetch("impulse-portable", self.cfg.base_dir,
                             lambda got, total: setattr(self, "progress", {"bytes": got, "total": total}))
                self.progress = None
                LOG.info("Halo's runtime: Impulse portable is in place")
                self.state = "starting"
                self._start()
                return
            except Exception as e:  # noqa: BLE001 - offline, or the release away: try again later
                self.why = f"its runtime could not be fetched yet ({str(e)[:160]}); trying again in {int(self.RETRY_S)} s"
                LOG.info("Halo's runtime: %s", self.why)
                self._wake.wait(self.RETRY_S)
                self._wake.clear()

    def _fetch_then_run(self, server: list[tuple[str, list[str]]], variant: str) -> None:
        """The base model, fetched (pinned revision, sha256), then Halo's server; while offline, tried again later."""
        import time
        from nebula.hf import Pulls
        pulls = Pulls(self.models_dir())
        while self.state == "downloading":
            try:
                pid = halo_model.ensure_base(pulls, self.m)
                while pid is not None and self.state == "downloading":
                    st = pulls.status(pid)
                    self.progress = {"bytes": st.get("bytes"), "total": st.get("total")}
                    if st["state"] in ("done", "failed", "cancelled"):
                        if st["state"] != "done":
                            raise RuntimeError(next((f["error"] for f in st["files"] if f.get("error")), st["state"]))
                        break
                    time.sleep(1)
                base = find_base(self.models_dir(), self.m)
                if base:
                    self.state = "starting"
                    self._run(server, base, variant)
                    return
                raise RuntimeError("the file did not check out")
            except Exception as e:  # noqa: BLE001 - offline, or the Hub away: try again later
                self.why = f"its model could not be fetched yet ({str(e)[:160]}); trying again in {int(self.RETRY_S)} s"
                LOG.info("Halo's model: %s", self.why)
                self._wake.wait(self.RETRY_S)
                self._wake.clear()

    def _run(self, servers: list[tuple[str, list[str]]], base: str, variant: str) -> None:
        """Halo's server on the first runtime that comes up and runs its model: Impulse, else Impulse portable, else
        llama-server; when none runs it (LFM3 on an Impulse from before it), Halo v2 on them when its file is here."""
        why = None
        for runtime, server in servers:
            if (runtime not in (self.m.get("runtimes") or [runtime]) and halo_model.model_for(runtime) != self.m["id"]
                    and not os.environ.get("NEBULA_HALO_MODEL")):
                continue                                  # Impulse portable / llama.cpp do not run LFM3
            why = self._run_one(runtime, server, base, variant)
            if why is None or self.state == "stopped":
                return
            LOG.warning("Halo's own model (%s) did not start on %s: %s", self.m["id"], runtime, why)
        fb = None if os.environ.get("NEBULA_HALO_MODEL") else self._fallback(servers)
        if fb is not None:
            LOG.warning("Halo's own model: %s did not start (%s); Halo v2 instead", self.m["id"], why)
            self.m, fb_base = fb
            self.upgrade = None
            return self._run(servers, fb_base, variant)
        self.state, self.why = "failed", why or f"no runtime here runs {self.m['name']}"

    def _run_one(self, runtime: str, server: list[str], base: str, variant: str) -> str | None:
        """None once its server has run (and ended); else why it did not come up."""
        from nebula.app.engine import EngineProcess
        from nebula.config import Tier
        try:
            self.where = self.place()
            if self.where.startswith("vk:") and runtime == "Impulse":
                return "Impulse's CUDA build runs on NVIDIA cards only: the Radeon is Impulse portable's (Vulkan)"
            port = _free_port()
            args = halo_model.server_args(self.models_dir(), variant, "gpu" if self.where.startswith("vk:") else self.where,
                                          "worker", port, m=self.m, base=base)
            if self.where.startswith("vk:"):
                args += ["--device", self.vk_device]     # the Radeon, through Vulkan; CUDA hidden from it
            env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": self.card if self.where.startswith("gpu") else ""}
            # Halo's server answers only Nebula: a key of its own each start, in its environment (never on the command
            # line, which goes to the log), so a web page open in the browser cannot use it (llama-server allows any origin)
            import secrets
            self.api_key = secrets.token_urlsafe(24)      # impulse-server (both builds) and llama-server read LLAMA_API_KEY
            env["LLAMA_API_KEY"] = self.api_key
            args = lora_paths_without_colons(args, os.path.dirname(server[0]) or None)
            cmd = server + args
            if self.where == "cpu" and os.name != "nt" and shutil.which("nice"):
                cmd = ["nice", "-n", "10"] + cmd
            url = f"http://127.0.0.1:{port}"
            log = os.path.join(self.cfg.base_dir, "halo-engine.log") if self.cfg.base_dir else None
            self.proc = EngineProcess(cmd, url, cwd=os.path.dirname(server[0]) or None, env=env,
                                      ready_timeout_s=300, stop_timeout_s=10, log_path=log)
            self.proc.start()
            if not self.proc.wait_ready():
                why = self.proc.error or "its server did not come up"
                self.proc.stop()
                return why
            m = self.m
            self.tier = Tier(name="halo-own", engine=url, model=m["id"], context=m["server"]["context"], slots=m["server"]["slots"])
            self.state, self.runtime = "ready", runtime
            self.why = ((("no NVIDIA GPU on this machine: on the CPU" if getattr(self, "no_cards", False) else
                          "no GPU has room for it: on the CPU until one does")) if self.where == "cpu" and self.settings()["where"] == "gpu" else None)
            LOG.info("Halo's own model is ready on the %s in %s (%s)", self.where, runtime, variant)
            # this thread stays with the server for its life: the engine is set to end with the thread that started it
            # (nebula/app/engine.py), so returning here would end it
            if self.proc.proc is not None:
                self.proc.proc.wait()
                if self.state == "ready":
                    self.state, self.why, self.tier = "failed", "its server ended", None
                    LOG.warning("Halo's own model ended; Halo thinks with the Assistant's model until the next start")
            return None
        except Exception as e:  # noqa: BLE001 - the next runtime, else Halo falls back to the Assistant's model
            return str(e)[:300]

    def ready(self) -> bool:
        return self.state == "ready" and self.tier is not None

    def restart(self) -> "HaloEngine":
        """Stop its server and start it again (a new placement: GPU or CPU)."""
        self.close()
        if self.proc is not None and self.proc.proc is not None:
            try:
                self.proc.proc.wait(timeout=15)
            except Exception:  # noqa: BLE001
                pass
        self.proc, self.tier, self.where, self.why, self.state, self.runtime = None, None, None, None, "off", None
        self.upgrade = None
        self._wake = threading.Event()
        return self.start()

    def set_variant(self, variant: str) -> None:
        if self.ready():
            halo_model.set_variant(self.tier.engine, variant, m=self.m, api_key=getattr(self, "api_key", None))

    def close(self) -> None:
        self.state = "stopped"
        self._wake.set()
        if self.proc is not None:
            try:
                self.proc.stop()
            except Exception:  # noqa: BLE001
                pass

    def view(self) -> dict:
        st = self.settings()
        return {"state": self.state, "where": self.where, "runtime": self.runtime, "why": self.why,
                **({"gpu_name": self.gpu_name} if getattr(self, "gpu_name", None) else {}), **({"progress": self.progress} if self.progress else {}),
                **({"model": self.m["id"], "model_name": self.m["name"]} if self.m else {}), **({"upgrade": self.upgrade} if self.upgrade else {}),
                "own_model": st["own_model"], "variant": st["variant"], "runs_on": st["where"]}      # runs_on: the owner's choice

    def wrong_place(self) -> bool:
        """Running somewhere other than chosen: on a GPU though the CPU was chosen, or on the CPU though a GPU was."""
        return self.where is not None and (self.where == "cpu") != (self.settings()["where"] == "cpu")
