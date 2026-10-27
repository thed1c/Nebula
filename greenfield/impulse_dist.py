#!/usr/bin/env python3
"""The impulse/ folder an installer carries: impulse-server and every library it needs that a PC may not have.

    python3 greenfield/impulse_dist.py --platform linux   --build build-impulse --cuda cuda --out impulse-dist
    python3 greenfield/impulse_dist.py --platform windows --build build-impulse --cuda cuda --out impulse-dist

Linux:   impulse-server (a wrapper that puts the folder on LD_LIBRARY_PATH) + impulse-server.bin + the CUDA runtime and
         cuBLAS it links + libgomp (ldd says which).
Both:    the Flash-Next engine when the build has it (-DIMPULSE_FLASHNEXT=ON): impulse-fx, which impulse-server execs for a
         Flash-Next pack from its own folder, and the engine library it loads (libimpulse_fx.so + libstrata_kq.so /
         impulse_fx.dll), with whatever CUDA libraries they link. --require-fx makes a build without them an error.
Windows: impulse-server.exe + cudart64/cublas64/cublasLt64 + the MSVC runtime and OpenMP DLLs (app-local, as
         Microsoft allows), so no Visual C++ redistributable is needed.
Only the NVIDIA driver (libcuda / nvcuda.dll) stays the machine's own.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = """#!/bin/sh
# impulse-server with the CUDA runtime it was built with: beside it (an offline bundle), else NVIDIA's pip packages in
# the installer's Python (the online installer)
D=$(dirname "$(readlink -f "$0")")
LD_LIBRARY_PATH="$D:$D/../python/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$D/impulse-server.bin" "$@"
"""
LINUX_BUNDLE = ("libcudart.so", "libcublas.so", "libcublasLt.so", "libgomp.so")
FX = {"linux": ("impulse-fx", "libimpulse_fx.so", "libstrata_kq.so"), "windows": ("impulse-fx.exe", "impulse_fx.dll")}


def fx_files(build: Path, plat: str, require: bool) -> list[Path]:
    """The Flash-Next engine's files in the build's bin/: all of them, or none (a build without -DIMPULSE_FLASHNEXT)."""
    names = FX[plat]
    have = [build / "bin" / n for n in names if (build / "bin" / n).is_file()]
    if have and len(have) != len(names) or require and not have:
        missing = [n for n in names if not (build / "bin" / n).is_file()]
        raise SystemExit(f"the Flash-Next engine is incomplete in {build / 'bin'}: no {', '.join(missing)}"
                         " (build with -DIMPULSE_FLASHNEXT=ON)")
    return have


def bundle_linux(exe: Path, cuda: Path, out: Path) -> None:
    """The libraries of LINUX_BUNDLE that `exe` links (ldd), copied beside it."""
    env = dict(os.environ, LD_LIBRARY_PATH=str(cuda / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    for line in subprocess.run(["ldd", str(exe)], capture_output=True, text=True, env=env, check=True).stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "=>" and parts[0].startswith(LINUX_BUNDLE):
            if parts[2] == "not":
                raise SystemExit(f"{exe.name} needs {parts[0]}, not found")
            if not (out / parts[0]).exists():
                shutil.copy2(os.path.realpath(parts[2]), out / parts[0])


def linux(build: Path, cuda: Path, out: Path, require_fx: bool = False) -> None:
    exe = build / "bin" / "impulse-server"
    shutil.copy2(exe, out / "impulse-server.bin")
    bundle_linux(exe, cuda, out)
    for f in fx_files(build, "linux", require_fx):   # beside impulse-server.bin: it execs impulse-fx from its own folder
        shutil.copy2(f, out / f.name)
        bundle_linux(f, cuda, out)
    (out / "impulse-server").write_text(WRAPPER)
    (out / "impulse-server").chmod(0o755)


def windows(build: Path, cuda: Path, out: Path, require_fx: bool = False) -> None:
    shutil.copy2(build / "bin" / "impulse-server.exe", out / "impulse-server.exe")
    for f in fx_files(build, "windows", require_fx):
        shutil.copy2(f, out / f.name)
    for name in ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll"):
        hits = glob.glob(str(cuda / "bin" / "x64" / name)) or glob.glob(str(cuda / "bin" / name))
        if not hits:
            raise SystemExit(f"no {name} in {cuda}")
        for h in hits:
            shutil.copy2(h, out)
    redist = os.environ.get("VCToolsRedistDir")                     # set by the "x64 Native Tools" environment
    if not redist:
        raise SystemExit("VCToolsRedistDir is not set: run from an MSVC developer environment")
    for sub in ("Microsoft.VC*.CRT", "Microsoft.VC*.OpenMP"):
        for d in glob.glob(os.path.join(redist, "x64", sub)):
            for f in glob.glob(os.path.join(d, "*.dll")):
                shutil.copy2(f, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["linux", "windows"], required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--cuda", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-fx", action="store_true", help="fail unless the build has the Flash-Next engine")
    a = ap.parse_args(argv)
    shutil.rmtree(a.out, ignore_errors=True)
    a.out.mkdir(parents=True)
    (linux if a.platform == "linux" else windows)(a.build, a.cuda, a.out, a.require_fx)
    shutil.copy2(ROOT / "impulse" / "THIRD_PARTY_NOTICES.md", a.out)
    for f in sorted(a.out.iterdir()):
        print(f"[impulse] {f.name}  {f.stat().st_size / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
