"""python -m nebula --engine http://127.0.0.1:8080 --port 8090"""
import sys

from nebula import __version__
from nebula.config import Config
from nebula.server import LedgerInUse, make_server


def local_call(cfg, path: str, body: dict) -> dict:
    """A running Nebula's own API on this machine (the commands below, used by an install over SSH)."""
    import json
    import urllib.request
    from nebula.signin import read_admin_token
    tok = read_admin_token(cfg)
    req = urllib.request.Request(f"http://127.0.0.1:{cfg.port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"X-Nebula-Internal": tok} if tok else {})})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def log_to_file(cfg) -> str | None:
    """Warnings and errors go to NAME.log beside the config (2 MB, two older files kept) and to stderr, as before: the
    desktop window keeps only the last 40 lines of what the server prints, so a traceback was lost with the window."""
    import logging
    root = logging.getLogger()
    if root.handlers:                                  # a caller (the desktop app, a test) set logging up already
        return None
    err = logging.StreamHandler()
    err.setLevel(logging.WARNING)
    err.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root.addHandler(err)
    root.setLevel(logging.WARNING)
    return add_log_file(cfg)


def add_log_file(cfg) -> str | None:
    """NAME.log beside the config: warnings and errors, 2 MB, two older files kept. Its path, or None without a config."""
    import logging
    import logging.handlers
    import os
    path = getattr(cfg, "path", None)
    if not path:
        return None
    log = os.path.splitext(str(path))[0] + ".log"
    root = logging.getLogger()
    if any(getattr(h, "baseFilename", None) == os.path.abspath(log) for h in root.handlers):
        return log
    try:
        h = logging.handlers.RotatingFileHandler(log, maxBytes=2 * 2**20, backupCount=2, encoding="utf-8")
    except OSError:
        return None
    h.setLevel(logging.WARNING)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(h)
    return log


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0) if argv and argv[0] in ("serve", "invite", "accept") else None
    headless = "--headless" in argv
    argv = [a for a in argv if a != "--headless"]
    if cmd in ("invite", "accept"):
        # nebula invite [--kind owner] --config F: a Gravity invite from the running Nebula, printed (one line)
        # nebula accept --pin N --config F: answer the household's invite with its PIN (model servers on your network)
        import argparse
        import json
        p = argparse.ArgumentParser(prog=f"nebula {cmd}")
        p.add_argument("--kind", default="owner")
        p.add_argument("--pin")
        known, rest = p.parse_known_args(argv)
        cfg = Config.from_args(rest)
        if cmd == "invite":
            print(local_call(cfg, "/api/yard/node/invite", {"kind": known.kind})["invite"])
        else:
            print(json.dumps(local_call(cfg, "/api/yard/network/accept", {"pin": known.pin})))
        return 0
    cfg = Config.from_args(argv)
    if headless:
        # a model server (docs/network.md): Gravity on (managed from your main Nebula over its admin link), the household
        # network on as a server, nothing but loopback listening for HTTP
        cfg.host = "127.0.0.1"
        cfg.raw.setdefault("gravity", {})["enabled"] = True
        net = cfg.raw.setdefault("network", {})
        net.setdefault("enabled", True)
        net["role"] = "server"
    log_to_file(cfg)
    import signal

    held = {}

    def _stop(*_a):                              # the window closing (SIGTERM): the same orderly close as Ctrl+C, so
        if held.get("srv") is not None:          # running team work is marked stopped while the ledger is still open;
            held["srv"].keep_engines = True      # the engines it started keep running (a window close, a restart or
        raise KeyboardInterrupt                  # an update must not take a model down on another machine)  
    try:
        signal.signal(signal.SIGTERM, _stop)
    except (ValueError, OSError):                # not the main thread (a test), or no such signal here
        pass
    try:
        srv = make_server(cfg)
    except LedgerInUse as e:
        print(f"nebula: {e}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:                    # closed while it was still starting
        return 0
    held["srv"] = srv
    if cfg.tiers:
        target = f"engine {cfg.engine}"
    elif not cfg.watch:
        target = "no models yet (add one from the page, or this machine is a model server waiting for its first)"
    else:
        target = f"watching the Switchyard at {cfg.watch.get('url')}, read only" if srv.watch is not None else "no models yet"
    print(f"nebula {__version__}: http://{cfg.host}:{srv.server_port} -> {target}; "
          f"ledger {cfg.ledger or 'off'}", flush=True)
    try:
        if not cfg.tiers:                        # only watching: no probes of this machine's engines
            srv.serve_forever()
            return 0
        from nebula.lan import probe_clis, probe_local_engines
        engines = [e["url"] for e in probe_local_engines()]
        clis = [c["name"] for c in probe_clis()]
        print(f"around: engines {engines or 'none'}; tools {clis or 'none'}", flush=True)
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
