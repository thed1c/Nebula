"""nebula/rag/vectors.py: Lookup's vectors, in Nebula's own Qdrant (never another one).

- **Server** (the default when the binary is here): a Qdrant process Nebula starts and stops itself, on loopback at
  a free port, with an API key only Nebula holds (`nebula-keys/qdrant.key`, 0600), its storage in Nebula's data
  folder (`nebula-lookup/qdrant`), telemetry off. The binary is the pinned release Nebula installs (`install()`,
  checked against its sha256), or `lookup.qdrant_bin`. No other Qdrant is ever used: there is no URL to give it.
- **Embedded** (no binary): the same API in this process (qdrant-client's local mode, `nebula-lookup/qdrant-local`),
  for small libraries; the page says which one runs.
- **Memory**: tests.

One Qdrant collection per Lookup collection (`nebula_<id>`), cosine distance, a point per chunk (the catalog's chunk
id).
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import secrets
import socket
import subprocess
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

LOG = logging.getLogger("nebula.lookup")
QDRANT_VERSION = "v1.19.2"
QDRANT_RELEASES = {          # the pinned release's files and their sha256 (GitHub's own digests)
    "x86_64": ("qdrant-x86_64-unknown-linux-musl.tar.gz", "50b253243309ed0ae50a19f678f0d0adcc4f0b567b3bd2c5c0c7e02fa8404bb6"),
    "aarch64": ("qdrant-aarch64-unknown-linux-musl.tar.gz", "6970b93b56fa1203f0cea47d2f330fdb654988fd56617aa77e8478cffd3c0ccb"),
}


class VectorsError(RuntimeError):
    pass


def client_available() -> bool:
    try:
        import qdrant_client  # noqa: F401
        return True
    except ImportError:
        return False


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def find_binary(configured: str | None, data_dir: str) -> str | None:
    for c in (configured, os.path.join(data_dir, "bin", "qdrant")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    import shutil
    return shutil.which("qdrant")


def install(data_dir: str, progress=None) -> str:
    """The pinned Qdrant release, downloaded into `<data>/bin/qdrant` and checked against its sha256."""
    arch = platform.machine().lower().replace("amd64", "x86_64").replace("arm64", "aarch64")
    if platform.system() != "Linux" or arch not in QDRANT_RELEASES:
        raise VectorsError(f"no pinned Qdrant build for {platform.system()} {arch}: install qdrant yourself and set lookup.qdrant_bin")
    name, want = QDRANT_RELEASES[arch]
    url = f"https://github.com/qdrant/qdrant/releases/download/{QDRANT_VERSION}/{name}"
    bindir = Path(data_dir, "bin")
    bindir.mkdir(parents=True, exist_ok=True)
    tmp = bindir / (name + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        for b in iter(lambda: r.read(1 << 20), b""):
            f.write(b)
            h.update(b)
            got += len(b)
            if progress:
                progress(got, total)
    if h.hexdigest() != want:
        tmp.unlink(missing_ok=True)
        raise VectorsError(f"the Qdrant download does not match its pinned sha256 ({h.hexdigest()[:12]}…): refused")
    with tarfile.open(tmp) as t:
        member = t.getmember("qdrant")
        if not member.isfile():
            raise VectorsError("the Qdrant archive holds no binary")
        with t.extractfile(member) as src, open(bindir / "qdrant.new", "wb") as dst:
            dst.write(src.read())
    os.chmod(bindir / "qdrant.new", 0o755)
    os.replace(bindir / "qdrant.new", bindir / "qdrant")
    (bindir / "qdrant.version").write_text(f"{QDRANT_VERSION} {want}\n")     # what is here, for the next pin (nebula/prereqs.py)
    tmp.unlink(missing_ok=True)
    return str(bindir / "qdrant")


def installed_version(binary: str) -> str | None:
    """The version of a Qdrant this module installed: its stamp beside it, else what the binary says it is."""
    stamp = Path(binary).with_name("qdrant.version")
    try:
        if stamp.is_file():
            return stamp.read_text().split()[0]
        import subprocess
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, IndexError, ValueError):
        return None
    import re
    m = re.search(r"(\d+\.\d+\.\d+)", out or "")
    return "v" + m.group(1) if m else None


class QdrantProcess:
    """Nebula's own Qdrant server: loopback, a free port, its own key, its own storage."""

    def __init__(self, binary: str, data_dir: str, key_path: str):
        self.binary, self.data_dir, self.key_path = binary, data_dir, key_path
        self.proc: subprocess.Popen | None = None
        self.port = None
        self.key = None
        self.error = None

    def _key(self) -> str:
        try:
            with open(self.key_path, encoding="utf-8") as f:
                k = f.read().strip()
            if k:
                return k
        except OSError:
            pass
        os.makedirs(os.path.dirname(self.key_path), mode=0o700, exist_ok=True)
        k = secrets.token_urlsafe(32)
        fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(k + "\n")
        return k

    def start(self, timeout: float = 60) -> "QdrantProcess":
        from nebula.app.engine import _die_with_parent
        store = Path(self.data_dir, "qdrant")
        store.mkdir(parents=True, exist_ok=True)
        self.key, self.port = self._key(), _free_port()
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "QDRANT__SERVICE__HOST": "127.0.0.1",
               "QDRANT__SERVICE__HTTP_PORT": str(self.port), "QDRANT__SERVICE__GRPC_PORT": str(_free_port()),
               "QDRANT__SERVICE__API_KEY": self.key, "QDRANT__SERVICE__ENABLE_CORS": "false",
               "QDRANT__STORAGE__STORAGE_PATH": str(store / "storage"), "QDRANT__STORAGE__SNAPSHOTS_PATH": str(store / "snapshots"),
               "QDRANT__TELEMETRY_DISABLED": "true", "QDRANT__LOG_LEVEL": "WARN"}
        with open(store / "qdrant.log", "ab") as log:
            self.proc = subprocess.Popen([self.binary, "--disable-telemetry"], cwd=str(store), env=env, stdout=log, stderr=log,
                                         stdin=subprocess.DEVNULL, preexec_fn=_die_with_parent if os.name == "posix" else None)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.error = f"Qdrant exited ({self.proc.returncode}): see {store / 'qdrant.log'}"
                raise VectorsError(self.error)
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{self.port}/readyz", headers={"api-key": self.key})
                with urllib.request.urlopen(req, timeout=2) as r:
                    if r.status == 200:
                        return self
            except OSError:
                pass
            time.sleep(0.3)
        self.stop()
        raise VectorsError("Qdrant did not become ready in time")

    def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


class Vectors:
    """The vector store: `mode` server | embedded | memory."""

    def __init__(self, mode: str, data_dir: str | None = None, binary: str | None = None, key_path: str | None = None):
        from qdrant_client import QdrantClient
        self.mode, self.proc = mode, None
        self.lock = threading.Lock()
        if mode == "server":
            self.proc = QdrantProcess(binary, data_dir, key_path).start()
            import warnings
            with warnings.catch_warnings():               # loopback only: the key never leaves this machine
                warnings.simplefilter("ignore", UserWarning)
                self.client = QdrantClient(url=f"http://127.0.0.1:{self.proc.port}", api_key=self.proc.key, https=False,
                                           prefer_grpc=False, timeout=60, check_compatibility=False)
        elif mode == "embedded":
            p = Path(data_dir, "qdrant-local")
            p.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(p))
        else:
            self.client = QdrantClient(":memory:")

    @staticmethod
    def name(cid: str) -> str:
        return "nebula_" + cid.replace("-", "_")

    def describe(self) -> dict:
        if self.mode == "server":
            return {"mode": "server", "version": QDRANT_VERSION, "address": f"127.0.0.1:{self.proc.port}", "own": True}
        return {"mode": self.mode, "own": True}

    def ensure(self, cid: str, dims: int) -> None:
        from qdrant_client import models as M
        n = self.name(cid)
        with self.lock:
            if self.client.collection_exists(n):
                info = self.client.get_collection(n)
                have = info.config.params.vectors.size if hasattr(info.config.params.vectors, "size") else None
                if have == dims:
                    return
                self.client.delete_collection(n)               # another embedding model: indexed again
            self.client.create_collection(n, vectors_config=M.VectorParams(size=dims, distance=M.Distance.COSINE))

    def upsert(self, cid: str, ids: list[int], vectors: list[list[float]], doc: int) -> None:
        from qdrant_client import models as M
        if not ids:
            return
        with self.lock:
            self.client.upsert(self.name(cid), points=[M.PointStruct(id=int(i), vector=list(map(float, v)), payload={"doc": doc})
                                                       for i, v in zip(ids, vectors)], wait=True)

    def delete(self, cid: str, ids: list[int]) -> None:
        from qdrant_client import models as M
        if not ids:
            return
        with self.lock:
            if self.client.collection_exists(self.name(cid)):
                self.client.delete(self.name(cid), points_selector=M.PointIdsList(points=[int(i) for i in ids]), wait=True)

    def drop(self, cid: str) -> None:
        with self.lock:
            if self.client.collection_exists(self.name(cid)):
                self.client.delete_collection(self.name(cid))

    def search(self, cids: list[str], vector: list[float], limit: int = 40) -> list[tuple[int, float]]:
        out = []
        with self.lock:
            for cid in cids:
                n = self.name(cid)
                if not self.client.collection_exists(n):
                    continue
                res = self.client.query_points(n, query=list(map(float, vector)), limit=limit, with_payload=False)
                out += [(int(p.id), float(p.score)) for p in res.points]
        out.sort(key=lambda x: -x[1])
        return out[:limit]

    def count(self, cid: str) -> int:
        with self.lock:
            if not self.client.collection_exists(self.name(cid)):
                return 0
            return int(self.client.count(self.name(cid), exact=True).count)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass
        if self.proc is not None:
            self.proc.stop()
