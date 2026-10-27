"""nebula/pairnet.py (internal): model servers on your network, through NVIDIA's Personal AI Router run by Nebula.

Users see "model servers on your network"; internally every machine's Nebula runs PAIR's backend (its broker and
workers, Apache-2.0, built from github.com/NVIDIA/Personal-AI-Router) and drives it over its JSON-RPC, as PAIR's own
terminal client does. PAIR has no background-service mode of its own: a client owns the broker; here Nebula is that
client. Between machines, model traffic goes only over PAIR's mutual TLS (pinned per-machine certificates, set up by
PIN pairing); there is no plain-HTTP model traffic between machines.

    b = Broker(bin_dir, data_dir).start()
    b.call("cluster:get-node-id")
    inv = b.call("cluster:invite-node", {"address": "10.0.0.7"})     # the inviting side: a PIN to carry
    b.call("cluster:respond-to-invite", {"inviteId": ..., "accept": True, "pin": "123456"})   # the invited side
"""
from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import threading
import time

EXE = ".exe" if os.name == "nt" else ""            # PAIR's Windows build (its build.bat) names them .exe
BINARIES = ("nvpair-ui-broker" + EXE, "nvpair-node-scanner" + EXE)


class PairError(RuntimeError):
    pass


def find_bin(bin_dir: str | None = None) -> str | None:
    import sys
    from pathlib import Path
    beside = [str(p / "pair") for p in list(Path(sys.executable).resolve().parents)[:3]]   # an installer's pair/ folder
    for d in [bin_dir, os.environ.get("NEBULA_PAIR_BIN"), *beside]:
        if d and all(os.path.isfile(os.path.join(d, b)) for b in BINARIES):
            return d
    return None


class Broker:
    """PAIR's broker as a child process, with its JSON-RPC (newline-delimited, over stdio) and its notifications."""

    def __init__(self, bin_dir: str, data_dir: str, argv: list[str] | None = None):
        self.bin_dir, self.data_dir = bin_dir, data_dir
        self.argv = argv                     # another way to reach a broker: a model machine's keeper over SSH
        self.proc: subprocess.Popen | None = None
        self.ids = itertools.count(1)
        self.pending: dict[int, "queue.Queue"] = {}
        self.notes: "queue.Queue" = queue.Queue()
        self.lock = threading.Lock()
        self.log: list[str] = []

    def start(self) -> "Broker":
        if self.argv:
            self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         text=True, bufsize=1)
            threading.Thread(target=self._read, daemon=True, name="pair-node").start()
            threading.Thread(target=self._errs, daemon=True, name="pair-node-log").start()
            return self
        os.makedirs(self.data_dir, exist_ok=True)
        env = dict(os.environ, HOME=self.data_dir, XDG_CONFIG_HOME=os.path.join(self.data_dir, ".config"))
        self.proc = subprocess.Popen([os.path.join(self.bin_dir, "nvpair-ui-broker" + EXE),
                                      "--cluster-dir", os.path.join(self.data_dir, "cluster")],
                                     cwd=self.bin_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=env, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True, name="pair-broker").start()
        threading.Thread(target=self._errs, daemon=True, name="pair-broker-log").start()
        return self

    def _errs(self):
        for line in self.proc.stderr:
            self.log = (self.log + [line.rstrip()])[-400:]

    def _read(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "id" in msg and msg.get("id") is not None and ("result" in msg or "error" in msg):
                with self.lock:
                    q = self.pending.pop(msg["id"], None)
                if q is not None:
                    q.put(msg)
            elif msg.get("method"):
                self.notes.put(msg)

    def call(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            raise PairError("the router's backend is not running")
        i = next(self.ids)
        q: "queue.Queue" = queue.Queue()
        with self.lock:
            self.pending[i] = q
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            raise PairError(f"{method}: no answer in {timeout:.0f} s") from None
        if msg.get("error"):
            raise PairError(f"{method}: {msg['error'].get('message') or msg['error']}")
        return msg.get("result") or {}

    def wait_note(self, method: str, timeout: float = 60, where=None) -> dict:
        end = time.time() + timeout
        kept = []
        try:
            while time.time() < end:
                try:
                    n = self.notes.get(timeout=max(0.1, end - time.time()))
                except queue.Empty:
                    break
                if n.get("method") == method and (where is None or where(n.get("params") or {})):
                    return n.get("params") or {}
                kept.append(n)
        finally:
            for n in kept:
                self.notes.put(n)
        raise PairError(f"no {method} within {timeout:.0f} s")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
