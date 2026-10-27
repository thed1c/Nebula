"""python3 -m nebula.app [--config FILE] [--window pywebview|gtk|browser] [--no-window] [--install-shortcut]

The config is the cockpit's own (README.md) with an optional "app" section (nebula/app/runner.py). Without
--config the app reads ~/.config/nebula/nebula.json (on Windows %APPDATA%\\nebula\\nebula.json), or an older
~/.config/strata-cockpit/cockpit.json when only that exists. A
relative "ledger" in the file is taken from the file's folder, since an icon starts the app in the home folder.
Flags the cockpit itself takes (--port, --ledger, --engine, ...) pass through.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path

from nebula import __version__


def default_config() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    new, old = base / "nebula" / "nebula.json", base / "strata-cockpit" / "cockpit.json"
    return old if old.exists() and not new.exists() else new


def _fail(a, title: str, message: str, detail: str = "") -> int:
    print(f"[cockpit] {message}" + (f"\n{detail}" if detail else ""), file=sys.stderr, flush=True)
    if not a.no_window:
        from nebula.app import window
        try:
            window.show(None, title="Cockpit", backend=a.window, html=window.error_page(title, message, detail))
        except Exception as e:  # noqa: BLE001 - no window at all: the message above is what there is
            print(f"[cockpit] no window: {e}", file=sys.stderr)
    return 1


def _wait_for_signal() -> None:
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_a: stop.set())
    while not stop.wait(1.0):
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m nebula.app", description="The cockpit in its own window.")
    p.add_argument("--config", help=f"the cockpit's JSON config (default {default_config()})")
    p.add_argument("--window", choices=("pywebview", "gtk", "browser"), help="the window to use (default: the best here)")
    p.add_argument("--no-window", action="store_true", help="run the server (and the engine) without a window, until Ctrl+C")
    p.add_argument("--engine-command", help="start the engine with this command (overrides the config's app.engine.command)")
    p.add_argument("--install-shortcut", action="store_true", help="make the icon that starts the app, then exit")
    a, rest = p.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="[cockpit] %(message)s")
    logging.getLogger("cockpit").setLevel(logging.WARNING)      # not every request the window makes
    logging.getLogger("cockpit.app").setLevel(logging.INFO)
    config = a.config or (str(default_config()) if default_config().exists() else None)
    if a.install_shortcut:
        from nebula.app.shortcut import install
        print(f"[cockpit] the app's icon: {install(config)}", flush=True)
        return 0
    if config is None:
        return _fail(a, "No config", f"The cockpit needs a config file: put one at {default_config()} or start it "
                     "with --config FILE.", "See README.md for its shape.")
    from nebula.app.runner import App
    from nebula.config import Config
    try:
        raw = json.loads(Path(config).read_text(encoding="utf-8"))
        cfg = Config.from_args(["--config", config, *rest])
        if cfg.ledger and not os.path.isabs(cfg.ledger) and "--ledger" not in rest:
            cfg.ledger = str(Path(cfg.base_dir) / cfg.ledger)
        app_cfg = dict(raw.get("app") or {})
        if a.engine_command:
            app_cfg["engine"] = {**(app_cfg.get("engine") or {}), "command": a.engine_command}
        app = App(cfg, app_cfg)
    except (OSError, ValueError, TypeError) as e:
        return _fail(a, "The config", f"{config} could not be used: {e}")
    try:
        app.start()
    except OSError as e:
        return _fail(a, "The cockpit did not start", f"It cannot listen on {cfg.host}:{cfg.port}: {e}.",
                     "Another cockpit may be running on that port: close it, or give this one another \"port\".")
    print(f"cockpit {__version__}: {app.url} (ledger {cfg.ledger or 'off'})"
          + (f"; starting the engine for {app.engine_tier} unless one answers at {app.engine.address}" if app.engine else ""),
          flush=True)
    try:
        if a.no_window:
            _wait_for_signal()
        else:
            from nebula.app import window
            ac = app.app_cfg
            used = window.show(app.url, title=ac.get("title", "Cockpit"), width=int(ac.get("width", 1440)),
                               height=int(ac.get("height", 920)), backend=a.window)
            print(f"[cockpit] the {used} window closed", flush=True)
    finally:
        print("[cockpit] stopping", flush=True)
        app.close()
        print("[cockpit] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
