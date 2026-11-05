"""nebula/syslogfwd.py: Nebula's warnings and errors, and its security log, to a syslog server; off by default.

    "syslog": {"enabled": false, "host": "", "port": 514, "protocol": "udp", "level": "warning", "security": true,
               "facility": "local0"}

Each line is RFC 5424 (`<PRI>1 TIMESTAMP HOST nebula PID MSGID - MESSAGE`): MSGID `log` for a warning or an error from
Nebula's own loggers (what the Error log on Captain's Log shows), `security` for a security-log event (its kind, who
and what: a sign-in, a refused key, a guest held back; never its detail). Over UDP (one datagram a line), TCP
(octet-counted, RFC 6587) or TLS (RFC 5425, checked against this computer's certificates). UDP and TCP are plain text
on the wire: TLS for a server across a network you do not trust.

Lines go out from one background thread; a server that is down loses lines (counted as failed), never a request.
"""
from __future__ import annotations

import logging
import os
import queue
import socket
import ssl
import threading
import time
from datetime import datetime, timezone

LOG = logging.getLogger("nebula.syslog")
LOGGERS = ("cockpit", "nebula")
PROTOCOLS = ("udp", "tcp", "tls")
LEVELS = {"warning": logging.WARNING, "error": logging.ERROR}   # warnings and errors, or errors only
FACILITIES = {"user": 1, "daemon": 3, "auth": 4, "authpriv": 10, **{f"local{i}": 16 + i for i in range(8)}}
SEVERITY = {logging.CRITICAL: 2, logging.ERROR: 3, logging.WARNING: 4, logging.INFO: 6, logging.DEBUG: 7}
SEC_SEVERITY = {"alert": 1, "warn": 4, "notice": 5, "info": 6}
QUEUE = 2000
LINE_BYTES = 2048


class SyslogError(ValueError):
    pass


def settings(cfg) -> dict:
    s = dict((getattr(cfg, "raw", None) or {}).get("syslog") or {})
    try:
        port = int(s.get("port") or 514)
    except (TypeError, ValueError):
        port = 514                                                # a hand-edited config never stops Nebula starting
    return {"enabled": bool(s.get("enabled")), "host": str(s.get("host") or "").strip(),
            "port": port if 1 <= port <= 65535 else 514, "protocol": s.get("protocol") if s.get("protocol") in PROTOCOLS else "udp",
            "level": s.get("level") if s.get("level") in LEVELS else "warning", "security": s.get("security", True) is not False,
            "facility": s.get("facility") if s.get("facility") in FACILITIES else "local0"}


def check(req: dict, cur: dict) -> dict:
    """The syslog section after the page's change, checked."""
    out = dict(cur)
    if "host" in req:
        host = str(req.get("host") or "").strip()
        if host and (len(host) > 253 or any(c.isspace() for c in host) or "/" in host):
            raise SyslogError("the syslog server is a host name or an address, such as 192.0.2.10 or logs.example.lan")
        out["host"] = host
    if "port" in req:
        try:
            port = int(req["port"])
        except (TypeError, ValueError):
            raise SyslogError("the port is a number from 1 to 65535") from None
        if not 1 <= port <= 65535:
            raise SyslogError("the port is a number from 1 to 65535")
        out["port"] = port
    for k, allowed in (("protocol", PROTOCOLS), ("level", tuple(LEVELS)), ("facility", tuple(FACILITIES))):
        if k in req:
            if req[k] not in allowed:
                raise SyslogError(f"{k} is one of {', '.join(allowed)}")
            out[k] = req[k]
    if "security" in req:
        out["security"] = bool(req["security"])
    if "enabled" in req:
        out["enabled"] = bool(req["enabled"])
    if out.get("enabled") and not out.get("host"):
        raise SyslogError("name the syslog server first")
    return out


class Forwarder(logging.Handler):
    """A logging handler (and a security-log listener) that queues RFC 5424 lines for one sender thread."""

    def __init__(self, cfg, seclog=None):
        super().__init__(logging.WARNING)
        self.cfg, self.seclog = cfg, seclog
        self.q: queue.Queue = queue.Queue(QUEUE)
        self.sent = self.failed = self.dropped = 0
        self.last_error: str | None = None
        self.sock = None
        self.wire = threading.Lock()                              # one sender at a time: the thread, or the page's test
        self.s = settings(cfg)
        self.host = (socket.gethostname() or "-").split(".")[0][:255]
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.attached = False
        if seclog is not None:
            seclog.listeners.append(self._security)
        self.apply()

    # --- the switches
    def apply(self) -> None:
        """Reads the settings again (after the page saved them) and starts, retunes or stops."""
        self.s = settings(self.cfg)
        with self.wire:
            self._close_socket()
        self.setLevel(LEVELS[self.s["level"]])
        on = self.s["enabled"] and bool(self.s["host"])
        if on and not self.attached:
            for name in LOGGERS:
                logging.getLogger(name).addHandler(self)
            self.attached = True
        elif not on and self.attached:
            self._detach()
        if on and (self.thread is None or not self.thread.is_alive()):
            self.stop.clear()
            self.thread = threading.Thread(target=self._run, name="syslog", daemon=True)
            self.thread.start()

    def _detach(self) -> None:
        for name in LOGGERS:
            logging.getLogger(name).removeHandler(self)
        self.attached = False

    def view(self) -> dict:
        return {**self.s, "sent": self.sent, "failed": self.failed, "dropped": self.dropped, "last_error": self.last_error,
                "protocols": list(PROTOCOLS), "levels": list(LEVELS), "facilities": list(FACILITIES)}

    def close(self) -> None:
        self._detach()
        if self.seclog is not None and self._security in self.seclog.listeners:
            self.seclog.listeners.remove(self._security)
        self.stop.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        if self.thread is not None:
            self.thread.join(2)
        with self.wire:
            self._close_socket()
        super().close()

    # --- what goes out
    def line(self, severity: int, msgid: str, message: str, facility: str | None = None) -> bytes:
        pri = FACILITIES[facility or self.s["facility"]] * 8 + severity
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        text = " ".join(str(message).split())                  # one line: no newline inside a syslog message
        out = f"<{pri}>1 {ts} {self.host} nebula {os.getpid()} {msgid} - {text}".encode("utf-8", "replace")
        return out[:LINE_BYTES]

    def _put(self, data: bytes) -> None:
        if not (self.s["enabled"] and self.s["host"]):
            return
        try:
            self.q.put_nowait(data)
        except queue.Full:
            self.dropped += 1                                     # a server that cannot keep up: lines dropped, counted

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("nebula.syslog"):
            return                                                # never its own warnings: they would loop
        try:
            msg = f"{record.name}: {record.getMessage()}"
            if record.exc_info:
                msg += f" ({record.exc_info[0].__name__}: {record.exc_info[1]})"
            self._put(self.line(SEVERITY.get(record.levelno, 6), "log", msg))
        except Exception:  # noqa: BLE001 - a log handler never raises
            pass

    def _security(self, ev: dict) -> None:
        if not self.s["security"] or not ev.get("fresh", True):
            return
        sev = SEC_SEVERITY.get(ev.get("severity") or "info", 6)
        parts = [str(ev.get("kind") or "")] + [f"{k}={ev[k]}" for k in ("who", "what") if ev.get(k)]
        self._put(self.line(sev, "security", "security: " + " ".join(parts)))

    def test(self) -> dict:
        """One line now, sent at once (not queued): what the page's Test button shows."""
        if not self.s["host"]:
            raise SyslogError("name the syslog server first")
        try:
            self._send(self.line(6, "test", "test line from Nebula's Captain's Log"))
            self.sent += 1
            return {"ok": True}
        except OSError as e:
            self.failed += 1
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            raise SyslogError(f"the test line did not reach {self.s['host']}:{self.s['port']} ({e})") from None

    # --- the wire
    def _close_socket(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _send(self, data: bytes) -> None:
        with self.wire:
            self._send_now(data)

    def _send_now(self, data: bytes) -> None:
        host, port, proto = self.s["host"], self.s["port"], self.s["protocol"]
        if proto == "udp":
            if self.sock is None:
                info = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)[0]
                self.sock = socket.socket(info[0], socket.SOCK_DGRAM)
                self.sock.connect(info[4])
            self.sock.send(data)
            return
        if self.sock is None:
            sock = socket.create_connection((host, port), timeout=10)
            if proto == "tls":
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
            self.sock = sock
        try:
            self.sock.sendall(f"{len(data)} ".encode() + data)    # octet counting (RFC 6587 / 5425)
        except OSError:
            self._close_socket()                                  # reconnect on the next line
            raise

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                data = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            if data is None:
                return
            try:
                self._send(data)
                self.sent += 1
            except OSError as e:
                self.failed += 1
                self.last_error = f"{type(e).__name__}: {e}"[:300]
                time.sleep(0.5)
