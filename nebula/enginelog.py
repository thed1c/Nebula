"""nebula/enginelog.py: the engine's own warnings and errors, into the cockpit's error log (nebula/errors.py).

strata-lite-server (llama.cpp underneath) writes everything to stderr: load steps, allocations, CUDA graph resets, and
also echoes of what it is given (`add_text: ...` with the user's words, chat-template markers, grammar lines that
quote generated tokens). `classify(line)` picks out the lines that report a problem and leaves every echo alone, so
a prompt that says "error" is not copied anywhere:

    error    GGML_ASSERT, CUDA/HIP/SYCL errors, an allocation that failed or ran out of memory, "cannot load/open/
             listen/start/allocate", "error:", "failed to", an abort, a crash, a fatal or a panic
    warning  "warning:", "warn:", a line that starts with "W "

The lines come from two places. An engine the desktop app starts is read by the app already (nebula/app/engine.py
hands each line here). An engine started some other way is followed by the config's "engine_logs": a log file the
engine writes, or a command that prints its log as it grows (a systemd unit's journal):

    {"engine_logs": [{"tier": "fast", "command": ["journalctl", "--user-unit", "my-engine.service", "-f", "-n", "0", "-o", "cat"]},
                     {"tier": "big", "file": "/var/log/strata/big.log"}]}

Each problem line is one entry with the source "engine:<tier>"; the same line again within a minute adds to that
entry's count instead.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time

LOG = logging.getLogger("cockpit.enginelog")

ECHO = re.compile(r"^\s*(add_text:|add_media:|Grammar (still awaiting|triggered)|<\||<channel|image_tokens|batch_f)|<\|[a-z_]+>|<[a-z_]+\|>")
ERROR = re.compile(r"GGML_ASSERT|\b(CUDA|HIP|SYCL|cuBLAS|ROCm) error|\bcuda\w* failed|out of memory|\bOOM\b|terminate called|"
                   r"core dumped|Segmentation fault|\bAborted\b|\bfatal\b|\bpanic\b|^E |\berror:|\bfailed to\b|\bfailed:|"
                   r"\bcannot (load|open|listen|start|allocate|find|create|read|map)\b|^(engine|mtp|vision): ", re.I)
WARNING = re.compile(r"^W |\bwarning:|\bwarn:", re.I)


def classify(line: str) -> str | None:
    """"error", "warning", or None for a line that reports no problem (or only echoes content)."""
    if not line or not line.strip() or ECHO.search(line):
        return None
    if ERROR.search(line):
        return "error"
    if WARNING.search(line):
        return "warning"
    return None


def note(errors, tier: str, line: str) -> None:
    """One engine line into the error log, if it reports a problem."""
    level = classify(line)
    if level and errors is not None:
        errors.add(line.strip(), source=f"engine:{tier}", level=level, collapse_s=60)


class Follower:
    """Follows one engine's log ("file" or "command") on a daemon thread until `close`."""

    def __init__(self, spec: dict, errors, poll_s: float = 0.5):
        if not spec.get("tier") or not (spec.get("file") or spec.get("command")):
            raise ValueError('an engine_logs entry is {"tier", "file"} or {"tier", "command"}')
        self.spec, self.errors, self.poll_s = spec, errors, poll_s
        self.tier = spec["tier"]
        self.stop = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.thread = threading.Thread(target=self._run, name=f"engine-log-{self.tier}", daemon=True)

    def start(self) -> "Follower":
        self.thread.start()
        return self

    def _run(self) -> None:
        try:
            if self.spec.get("command"):
                self._command(self.spec["command"])
            else:
                self._file(self.spec["file"])
        except Exception as e:  # noqa: BLE001 - a log that cannot be followed is itself worth a line
            LOG.warning("engine log for %s: %s", self.tier, e)
            self._say(f"the engine log for {self.tier} cannot be followed: {e}")

    def _say(self, message: str) -> None:
        """A problem with the following itself, in the error log as a warning (a command that is not on this machine,
        a unit that does not exist), so a log that shows nothing is never mistaken for an engine with nothing to say."""
        if self.errors is not None and not self.stop.is_set():
            try:
                self.errors.add(message, source=f"engine log {self.tier}", level="warning")
            except Exception:  # noqa: BLE001
                pass

    def _command(self, command) -> None:
        self.proc = subprocess.Popen(command if isinstance(command, list) else str(command), shell=not isinstance(command, list),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if self.stop.is_set():
                break
            note(self.errors, self.tier, line)
        self.proc.stdout.close()
        code = self.proc.wait()
        self._say(f"the engine log command for {self.tier} ended (exit code {code}): its log is no longer followed")

    def _file(self, path: str) -> None:
        """Like tail -F: from the end of the file as it is now; a rotated-in file from its start; a truncated one again
        from its start. A line is noted once it is whole."""
        f, ino, start, part = None, None, os.SEEK_END, ""
        while not self.stop.is_set():
            if f is None:
                try:
                    f = open(path, encoding="utf-8", errors="replace")
                    f.seek(0, start)
                    ino = os.fstat(f.fileno()).st_ino
                except OSError:
                    self.stop.wait(self.poll_s)
                    continue
            chunk = f.readline()
            if chunk:
                part += chunk
                if part.endswith("\n"):
                    note(self.errors, self.tier, part)
                    part = ""
                continue
            try:
                st = os.stat(path)
                if st.st_ino != ino:                  # rotated: the new file, from its start
                    f.close()
                    f, start, part = None, os.SEEK_SET, ""
                    continue
                if st.st_size < f.tell():             # truncated
                    f.seek(0)
                    part = ""
                    continue
            except OSError:
                pass
            self.stop.wait(self.poll_s)
        if f is not None:
            f.close()

    def close(self) -> None:
        self.stop.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
