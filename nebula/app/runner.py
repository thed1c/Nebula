"""nebula/app/runner.py: what the application runs behind its window.

    app = App(cfg, app_cfg)
    app.start()        # the cockpit's server on its port, and the engine starting (when configured) in the background
    ...                # the window shows http://127.0.0.1:<port>/; the page reads app.status() at /api/app
    app.close()        # the server stops taking requests and closes (ledger, MCP servers, schedules), then the engine

`app_cfg` is the config file's "app" section:

    {"engine": {"command": ["/path/strata-lite-server", "-m", "model.gguf", "--port", "8080"],
                "tier": "fast", "cwd": null, "env": {}, "ready_timeout_s": 900, "stop_timeout_s": 30,
                "log": "engine.log"},
     "title": "Cockpit", "width": 1440, "height": 920, "prevent_sleep": false}

Without "engine" the app only attaches to the tiers' engines as the cockpit always does. With it, the engine of the named
tier (the first tier with an engine when none is named) is started from the command unless one already answers there,
and stopped on close if the app started it. One supervisor thread owns the engine's process for the app's whole life,
so the Linux tie (the engine is signalled when the thread that started it ends) holds for a restart asked from the
page too. "prevent_sleep" keeps a Windows PC awake while an attempt is in flight (serve/keepawake.py).
"""
from __future__ import annotations

import logging
import os
import queue
import threading

from nebula import __version__
from nebula.app.engine import EngineProcess
from nebula.enginelog import note
from nebula.server import make_server

LOG = logging.getLogger("cockpit.app")


class App:
    def __init__(self, cfg, app_cfg: dict | None = None):
        self.cfg, self.app_cfg = cfg, dict(app_cfg or {})
        self.engine: EngineProcess | None = None
        e = self.app_cfg.get("engine")
        if isinstance(e, dict) and e.get("command"):
            tier = cfg.tier(e["tier"]) if e.get("tier") else next((t for t in cfg.tiers if t.engine), None)
            if tier is None or not tier.engine:
                raise ValueError('app.engine: no tier with an "engine" URL to start it for')
            base = cfg.base_dir or os.getcwd()
            log = e.get("log")
            self.engine = EngineProcess(e["command"], tier.engine, cwd=e.get("cwd") or base, env=e.get("env"),
                                        ready_timeout_s=float(e.get("ready_timeout_s", 900)),
                                        stop_timeout_s=float(e.get("stop_timeout_s", 30)),
                                        log_path=os.path.join(base, log) if log else None)
            self.engine_tier = tier.name
        self.server = None
        self.jobs: queue.Queue = queue.Queue()
        self.supervisor = threading.Thread(target=self._supervise, name="engine-supervisor", daemon=True)
        self.keepawake = None
        self.closed = threading.Event()

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.cfg.host in ("0.0.0.0", "", "::") else self.cfg.host
        return f"http://{host}:{self.server.server_port if self.server else self.cfg.port}/"

    def start(self) -> None:
        """The server first (the window has its page at once), then the engine. Raises OSError when the port is taken."""
        self.server = make_server(self.cfg)
        self.server.app = self
        if self.engine is not None:                  # its problem lines into the error log
            self.engine.on_line = lambda line: note(self.server.errors, self.engine_tier, line)
        threading.Thread(target=self.server.serve_forever, name="cockpit-server", daemon=True).start()
        self.supervisor.start()
        if self.engine is not None:
            self.jobs.put("start")
        if self.app_cfg.get("prevent_sleep") and self.server.ledger is not None:
            from nebula._vendor.strata_serve.keepawake import KeepAwake
            self.keepawake = KeepAwake.create(True, say=lambda m, **_k: LOG.info(m))
            if self.keepawake is not None:
                threading.Thread(target=self._awake, name="keep-awake", daemon=True).start()

    def _supervise(self) -> None:
        while True:
            job = self.jobs.get()
            if job == "start" and self.engine is not None and self.engine.state in ("idle", "failed", "stopped", "queued"):
                self.engine.start()
                if self.engine.state == "loading":
                    threading.Thread(target=self.engine.wait_ready, name="engine-ready", daemon=True).start()
            elif job == "close":
                if self.engine is not None:
                    self.engine.stop()
                return

    def restart_engine(self) -> bool:
        """Asked from the page after a failed start: queued for the supervisor. False when there is nothing to start."""
        if self.engine is None or self.engine.state not in ("failed", "stopped") or self.closed.is_set():
            return False
        self.engine.state, self.engine.error = "queued", None     # at once: the page (and a caller waiting for the
        self.jobs.put("start")                                      # outcome) never reads the last attempt's "failed"
        return True

    def _awake(self) -> None:
        """Holds the keep-awake while any attempt is running, by the ledger's pulse."""
        ledger, held = self.server.ledger, False
        seen = ledger.activity.version
        while not self.closed.is_set() and not ledger.activity.closing:
            seen = ledger.activity.wait(seen, 5.0)
            try:
                with ledger.lock:
                    n = ledger.db.execute("SELECT count(*) FROM attempts WHERE state='running'").fetchone()[0]
            except Exception:  # noqa: BLE001 - the ledger closed under it
                break
            if n and not held:
                self.keepawake.acquire()
                held = True
            elif not n and held:
                self.keepawake.release()
                held = False
        if held:
            self.keepawake.release()

    def status(self) -> dict:
        return {"app": True, "version": __version__, "url": self.url, "closing": self.closed.is_set(),
                "engine": dict(self.engine.status(), tier=self.engine_tier) if self.engine is not None else None}

    def close(self) -> None:
        """In order: no new requests, the server's own closers (schedules, ledger, telemetry, MCP servers), then the
        engine the app started. Safe to call twice."""
        if self.closed.is_set():
            return
        self.closed.set()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.supervisor.is_alive():
            self.jobs.put("close")
            self.supervisor.join(timeout=(self.engine.stop_timeout_s + 20) if self.engine else 5)
        elif self.engine is not None:
            self.engine.stop()
        if self.keepawake is not None:
            self.keepawake.close()
