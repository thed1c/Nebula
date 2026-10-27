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
    try:
        srv = make_server(cfg)
    except LedgerInUse as e:
        print(f"nebula: {e}", file=sys.stderr, flush=True)
        return 1
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
