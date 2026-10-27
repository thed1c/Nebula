#!/usr/bin/env python3
"""The CUDA toolkit pieces Impulse builds with, from NVIDIA's redistributable archives, merged into one folder.

    python3 greenfield/cuda_redist.py --platform linux-x86_64   --out cuda
    python3 greenfield/cuda_redist.py --platform windows-x86_64 --out cuda

No installer and no root: the release runners use it (greenfield's release workflow), and so can anyone building
Impulse without a system CUDA. Every archive is checked against the sha256 NVIDIA's manifest gives. Then
    cmake -DCMAKE_CUDA_COMPILER=<out>/bin/nvcc -DCUDAToolkit_ROOT=<out> ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "13.2.2"
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = ["cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "libcublas", "cuda_cccl", "cuda_nvrtc",
              "cuda_nvml_dev", "cuda_profiler_api", "cuda_nvtx"]


def fetch(url: str, dest: Path, sha256: str | None = None) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("[cuda] fetching", url, flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.rename(dest)
    if sha256:
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != sha256:
            dest.unlink()
            raise SystemExit(f"[cuda] {dest.name}: sha256 does not match NVIDIA's manifest")
    return dest


def merge(src: Path, out: Path) -> None:
    """Each archive is one folder (<name>-<platform>-<ver>-archive/{bin,include,lib,...}); its contents go into out."""
    top = next(p for p in src.iterdir() if p.is_dir())
    shutil.copytree(top, out, dirs_exist_ok=True, symlinks=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["linux-x86_64", "windows-x86_64"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "cuda-redist")
    a = ap.parse_args(argv)
    man = json.loads(fetch(f"{BASE}redistrib_{VERSION}.json", a.cache / f"redistrib_{VERSION}.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for c in COMPONENTS:
        ent = man[c][a.platform]
        arc = fetch(BASE + ent["relative_path"], a.cache / Path(ent["relative_path"]).name, ent["sha256"])
        with tempfile.TemporaryDirectory() as d:
            if arc.name.endswith(".zip"):
                with zipfile.ZipFile(arc) as z:
                    z.extractall(d)
            else:
                with tarfile.open(arc) as t:
                    t.extractall(d, filter="tar")              # the archives carry .so links
            merge(Path(d), a.out)
        print(f"[cuda] {c} {man[c]['version']}", flush=True)
    lib64 = a.out / "lib64"
    if a.platform.startswith("linux") and not lib64.exists():
        lib64.symlink_to("lib")                                 # nvcc.profile links from lib64, the archives ship lib
    print(json.dumps({"cuda": VERSION, "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
