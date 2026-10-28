#!/usr/bin/env python3
"""The update feed's key and signature (nebula/updates.py; how to publish: docs/updates.md).

    python3 greenfield/sign_feed.py --keygen
        a new Ed25519 key: the private half to --key (default $NEBULA_UPDATE_KEY, else
        ~/.config/nebula-release/nebula-update-ed25519; 0600, never into the repo), the public half printed for
        nebula/updates.py's PUBLIC_KEY. An existing key is never overwritten.
    python3 greenfield/sign_feed.py --version 0.1.3 --notes "What changed, in plain words" \\
            --file linux-x86_64=dist/nebula-eh-0.1.3-linux-x86_64.run=https://github.com/thed1c/Nebula/releases/download/v0.1.3/nebula-eh-0.1.3-linux-x86_64.run \\
            --file windows-x86_64=dist/Nebula-EH-Setup-0.1.3.exe=https://github.com/.../Nebula-EH-Setup-0.1.3.exe \\
            [--app linux-x86_64=dist/nebula-eh-0.1.4.1-app-linux-x86_64.tar.gz=https://.../nebula-eh-0.1.4.1-app-linux-x86_64.tar.gz=0.1.4] \\
            --out feed/
        latest.json (each file's sha256 read from the file itself) and latest.json.sig beside it, ready to commit to the
        feed repository.

Build side only: it uses `cryptography` when it is here, else Nebula's own Ed25519 (the same signature).
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nebula import updates  # noqa: E402

KEY = Path(os.environ.get("NEBULA_UPDATE_KEY") or Path.home() / ".config" / "nebula-release" / "nebula-update-ed25519")
REPO = Path(__file__).resolve().parents[1]


def keygen(path: Path) -> str:
    if path.exists():
        raise SystemExit(f"{path} is already there: kept (its public key: --public)")
    if REPO in path.resolve().parents:
        raise SystemExit("the private key never goes inside the repository")
    seed = os.urandom(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(seed.hex() + "\n")
    return updates.public_of(seed).hex()


def read_seed(path: Path) -> bytes:
    if path.stat().st_mode & 0o077:
        raise SystemExit(f"{path} is readable by others: chmod 600 it first")
    return bytes.fromhex(path.read_text().strip())


def sign(seed: bytes, body: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        return Ed25519PrivateKey.from_private_bytes(seed).sign(body)
    except ImportError:
        return updates.sign(seed, body)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(version: str, notes: str, files: list[str], page: str | None, apps: list[str] = ()) -> bytes:
    updates.version_tuple(version)
    out = {}
    for spec in files:
        plat, local, url = spec.split("=", 2)
        if not url.startswith("https://"):
            raise SystemExit(f"{plat}: the url is https")
        out[plat] = {"url": url, "sha256": sha256(Path(local))}
    for spec in apps:                          # a small update: PLATFORM=LOCAL=URL=BASE[,BASE...] (installers it fits on)
        plat, local, url, bases = spec.split("=", 3)
        if plat not in out:
            raise SystemExit(f"{plat}: a small update needs the platform's installer (--file) beside it")
        if not url.startswith("https://"):
            raise SystemExit(f"{plat}: the url is https")
        bases = [b.strip().lstrip("v") for b in bases.split(",") if b.strip()]
        for b in bases:
            updates.version_tuple(b)
        if not bases:
            raise SystemExit(f"{plat}: a small update names the installers it fits on")
        out[plat]["app"] = {"url": url, "sha256": sha256(Path(local)), "bytes": Path(local).stat().st_size, "bases": bases}
    feed = {"version": version.lstrip("v"), "notes": notes, "published": datetime.date.today().isoformat(), "files": out,
            "release_page": page or f"https://github.com/thed1c/Nebula/releases/tag/v{version.lstrip('v')}"}
    return (json.dumps(feed, indent=1) + "\n").encode()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Nebula's update feed: its key, and latest.json signed")
    ap.add_argument("--key", type=Path, default=KEY, help="the private key (default $NEBULA_UPDATE_KEY, else %(default)s)")
    ap.add_argument("--keygen", action="store_true")
    ap.add_argument("--public", action="store_true", help="print the key's public half")
    ap.add_argument("--version")
    ap.add_argument("--notes", default="")
    ap.add_argument("--file", action="append", default=[], help="PLATFORM=LOCAL_FILE=URL (linux-x86_64, windows-x86_64)")
    ap.add_argument("--app", action="append", default=[],
                    help="a small update: PLATFORM=LOCAL_FILE=URL=BASES (the installers' versions it fits on, comma separated; "
                         "only when Impulse and the rest beside Nebula did not change since those)")
    ap.add_argument("--release-page")
    ap.add_argument("--out", type=Path, default=Path("."))
    a = ap.parse_args(argv)
    if a.keygen:
        print(keygen(a.key))
        return 0
    seed = read_seed(a.key)
    if a.public:
        print(updates.public_of(seed).hex())
        return 0
    if not a.version:
        ap.error("--version is needed to sign a feed")
    if updates.public_of(seed).hex() != updates.PUBLIC_KEY:
        raise SystemExit("this key is not the one nebula/updates.py trusts (PUBLIC_KEY): refused")
    body = build(a.version, a.notes, a.file, a.release_page, a.app)
    sig = sign(seed, body)
    assert updates.verify(updates.public_of(seed), body, sig)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "latest.json").write_bytes(body)
    (a.out / "latest.json.sig").write_text(base64.b64encode(sig).decode() + "\n")
    print(f"wrote {a.out / 'latest.json'} and its .sig ({a.version}, {len(a.file)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
