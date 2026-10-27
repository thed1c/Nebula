"""nebula/nodekeeper.py: the one file a model machine gets from the main Nebula (copied over SSH; no Nebula there).

A model machine on your network runs only engines (in OpenShell sandboxes, on rootless Podman) and the Personal AI
Router's node programs (internal: PAIR, Apache-2.0), which carry the main Nebula's requests to it over mutual TLS.
This keeper is the small piece that makes those work without a Nebula on the machine:

- **Holds PAIR's broker.** PAIR's broker serves its JSON-RPC on stdio and stops all of PAIR when its input closes, so
  the keeper owns it for as long as the machine runs (a user service), and relays the main Nebula's calls to it:
  `nodekeeper.py connect` (run over SSH) is a JSON-RPC client of the keeper's socket, ids remapped, PAIR's
  notifications sent to every connected client and the recent ones kept (`keeper:notes`).
- **The machine's one engine address for PAIR** (127.0.0.1:8081, llama.cpp's router face: `/props` says
  `"role": "router"`, `/health`, `/models`). PAIR delivers every request for this machine there; the keeper passes it
  to the engine serving the request's `model` (the map the main Nebula sets with `keeper:set-engines`), streamed,
  unchanged. It decides nothing else: which engines run, on which cards, is the main Nebula's.

    python3 nodekeeper.py serve --pair-bin DIR --state DIR [--facade-port 8081]
    python3 nodekeeper.py connect --state DIR            # stdio <-> the keeper (the main Nebula runs this over SSH)

Standard library only (Python 3.10+), so the machine needs nothing but python3.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PASS_POST = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses", "/v1/embeddings",
             "/completion", "/tokenize", "/detokenize", "/chat/completions", "/completions", "/embeddings")
KEEP_NOTES = 50


def sock_path(state: str) -> str:
    return os.path.join(state, "keeper.sock")


class Keeper:
    def __init__(self, pair_bin: str | None, state: str, facade_port: int = 8081, facade_host: str = "127.0.0.1"):
        self.pair_bin, self.state = pair_bin, state
        self.facade_port, self.facade_host = facade_port, facade_host
        self.proc: subprocess.Popen | None = None
        self.ids = itertools.count(1)
        self.pending: dict[int, tuple] = {}             # keeper id -> (client, the client's own id)
        self.clients: set = set()
        self.notes: list[dict] = []
        self.seq = itertools.count(1)
        self.lock = threading.Lock()
        self.wlock = threading.Lock()
        self.engines_file = os.path.join(state, "engines.json")
        self.engines: dict[str, str] = {}
        self.log: list[str] = []
        self.http: ThreadingHTTPServer | None = None
        self.stopping = False

    # --- PAIR's broker
    def start_broker(self):
        if not self.pair_bin:
            return
        env = dict(os.environ, HOME=self.state, XDG_CONFIG_HOME=os.path.join(self.state, ".config"))
        self.proc = subprocess.Popen([os.path.join(self.pair_bin, "nvpair-ui-broker"), "--cluster-dir",
                                      os.path.join(self.state, "cluster")], cwd=self.pair_bin, env=env, text=True, bufsize=1,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._broker_out, daemon=True).start()
        threading.Thread(target=self._broker_err, daemon=True).start()

    def _broker_err(self):
        for line in self.proc.stderr:
            self.log = (self.log + [line.rstrip()])[-400:]

    def _broker_out(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("id") is not None and ("result" in msg or "error" in msg):
                with self.lock:
                    who = self.pending.pop(msg["id"], None)
                if who is not None:
                    client, cid = who
                    if cid is None:                      # the keeper's own call
                        client.put(msg)
                    else:
                        client.send(dict(msg, id=cid))
            elif msg.get("method"):
                self._note(msg)
        if not self.stopping:                            # PAIR ended: the service manager starts the keeper again
            os._exit(3)

    def _note(self, msg: dict):
        with self.lock:
            self.notes = (self.notes + [dict(msg, seq=next(self.seq), at=time.time())])[-KEEP_NOTES:]
            clients = list(self.clients)
        for c in clients:
            c.send(msg)

    def to_broker(self, msg: dict, client, cid):
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("the network's node programs are not running here")
        i = next(self.ids)
        with self.lock:
            self.pending[i] = (client, cid)
        with self.wlock:
            self.proc.stdin.write(json.dumps(dict(msg, id=i)) + "\n")
            self.proc.stdin.flush()

    def call(self, method: str, params: dict | None = None, timeout: float = 60) -> dict:
        q: "queue.Queue" = queue.Queue()
        self.to_broker({"jsonrpc": "2.0", "method": method, "params": params or {}}, q, None)
        msg = q.get(timeout=timeout)
        if msg.get("error"):
            raise RuntimeError(f"{method}: {msg['error'].get('message') or msg['error']}")
        return msg.get("result") or {}

    # --- the engines map
    def load_engines(self):
        try:
            with open(self.engines_file, encoding="utf-8") as f:
                self.engines = {str(k): str(v) for k, v in (json.load(f) or {}).items()}
        except (OSError, ValueError):
            self.engines = {}

    def set_engines(self, engines: dict) -> dict:
        clean = {}
        for model, url in (engines or {}).items():
            u = urllib.parse.urlsplit(str(url))
            if u.scheme != "http" or not _loopback(u.hostname or ""):
                raise ValueError(f"{model}: an engine on this machine (http on loopback or *.localhost), not {url}")
            clean[str(model)] = str(url).rstrip("/")
        tmp = self.engines_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=1)
        os.replace(tmp, self.engines_file)
        self.engines = clean
        return {"engines": clean}

    # --- keeper's own methods, then PAIR's
    def handle(self, msg: dict, client):
        m, cid = msg.get("method") or "", msg.get("id")
        try:
            if m == "keeper:notes":
                after = int((msg.get("params") or {}).get("after") or 0)
                with self.lock:
                    res = {"notes": [n for n in self.notes if n["seq"] > after]}
            elif m == "keeper:set-engines":
                res = self.set_engines((msg.get("params") or {}).get("engines") or {})
            elif m == "keeper:set-engine":                  # one model: its engine's address there, or none (stopped)
                pr = msg.get("params") or {}
                eng = dict(self.engines)
                if pr.get("url"):
                    eng[str(pr["model"])] = str(pr["url"])
                else:
                    eng.pop(str(pr.get("model")), None)
                res = self.set_engines(eng)
            elif m == "keeper:health":                      # an engine's /health, asked from this machine
                res = {"status": health((msg.get("params") or {}).get("url") or "")}
            elif m == "keeper:status":
                res = {"pair": self.proc is not None and self.proc.poll() is None, "engines": self.engines,
                       "facade": f"{self.facade_host}:{self.facade_port}", "log": self.log[-20:]}
            elif m.startswith("keeper:"):
                raise ValueError(f"no method {m}")
            else:
                self.to_broker(msg, client, cid)
                return
        except Exception as e:                           # an answer, never a dead connection
            if cid is not None:
                client.send({"jsonrpc": "2.0", "id": cid, "error": {"code": -32000, "message": str(e)}})
            return
        if cid is not None:
            client.send({"jsonrpc": "2.0", "id": cid, "result": res})

    # --- the socket the main Nebula reaches over SSH
    def serve_socket(self):
        path = sock_path(self.state)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            s.bind(path)
        finally:
            os.umask(old)
        s.listen(8)
        threading.Thread(target=self._accept, args=(s,), daemon=True).start()

    def _accept(self, s):
        while True:
            conn, _ = s.accept()
            threading.Thread(target=self._client, args=(Client(conn),), daemon=True).start()

    def _client(self, c):
        with self.lock:
            self.clients.add(c)
        try:
            for line in c.lines():
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if isinstance(msg, dict):
                    self.handle(msg, c)
        finally:
            with self.lock:
                self.clients.discard(c)
                for k in [k for k, (cl, _) in self.pending.items() if cl is c]:
                    self.pending[k] = (_Drop(), None)
            c.close()

    # --- the engine address PAIR uses
    def serve_facade(self):
        keeper = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                p = self.path.split("?")[0]
                if p == "/health":
                    return self._json(200, {"status": "ok"})
                if p == "/props":
                    return self._json(200, {"role": "router", "build_info": "nebula-node-keeper"})
                if p in ("/models", "/v1/models"):
                    now = int(time.time())
                    return self._json(200, {"object": "list", "data": [
                        {"id": m, "object": "model", "created": now, "owned_by": "nebula", "status": {"value": "loaded"}}
                        for m in sorted(keeper.engines)]})
                return self._json(404, {"error": {"message": f"no {p} here"}})

            def do_POST(self):
                p = self.path.split("?")[0]
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                if p not in PASS_POST:
                    return self._json(404, {"error": {"message": f"no {p} here"}})
                try:
                    model = (json.loads(body or b"{}") or {}).get("model")
                except ValueError:
                    return self._json(400, {"error": {"message": "the body is not JSON"}})
                url = keeper.engines.get(str(model)) if model else None
                if url is None and len(keeper.engines) == 1 and not model:
                    url = next(iter(keeper.engines.values()))
                if url is None:
                    return self._json(404, {"error": {"message": f"no model {model!r} on this machine",
                                                      "type": "model_not_found"}})
                keeper.forward(self, url, p, body)

        self.http = ThreadingHTTPServer((self.facade_host, self.facade_port), H)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def forward(self, h, url: str, path: str, body: bytes):
        u = urllib.parse.urlsplit(url)
        host = u.hostname or "127.0.0.1"
        conn = HTTPConnection("127.0.0.1" if host.endswith(".localhost") else host, u.port or 80, timeout=3600)
        headers = {k: v for k, v in h.headers.items() if k.lower() not in ("host", "content-length", "connection",
                                                                              "transfer-encoding", "keep-alive")}
        headers["Host"] = u.netloc
        headers["Content-Length"] = str(len(body))
        try:
            conn.request("POST", (u.path.rstrip("/") or "") + path, body=body, headers=headers)
            r = conn.getresponse()
        except OSError as e:
            return h._json(502, {"error": {"message": f"the engine for this model does not answer: {e}"}})
        h.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("content-length", "transfer-encoding", "connection", "keep-alive"):
                h.send_header(k, v)
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        try:
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk:
                    break
                h.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                h.wfile.flush()
            h.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass
        finally:
            conn.close()

    def run(self):
        os.makedirs(self.state, mode=0o700, exist_ok=True)
        self.load_engines()
        self.serve_facade()
        self.start_broker()
        self.serve_socket()
        if self.proc is not None:                        # PAIR adopts the face above as this machine's engine
            threading.Thread(target=self._adopt, daemon=True).start()
        done = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: done.set())
        done.wait()
        self.stop()

    def _adopt(self):
        for _ in range(30):
            try:
                self.call("engine:start", {"engine": "llamacpp"}, timeout=60)
                return
            except Exception as e:                       # PAIR still starting its workers
                self.log = (self.log + [f"[keeper] engine:start: {e}"])[-400:]
                time.sleep(2)

    def stop(self):
        self.stopping = True
        if self.http is not None:
            self.http.shutdown()
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def health(url: str, timeout: float = 3.0) -> int | None:
    """GET <url>/health on this machine (a gateway's *.localhost name goes to loopback with its Host kept)."""
    u = urllib.parse.urlsplit(url)
    if u.scheme != "http" or not _loopback(u.hostname or ""):
        return None
    host = u.hostname or "127.0.0.1"
    conn = HTTPConnection("127.0.0.1" if host.endswith(".localhost") else host, u.port or 80, timeout=timeout)
    try:
        conn.request("GET", (u.path.rstrip("/") or "") + "/health", headers={"Host": u.netloc})
        r = conn.getresponse()
        r.read()
        return r.status
    except OSError:
        return None
    finally:
        conn.close()


def _loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1") or host.endswith(".localhost")


class Client:
    def __init__(self, conn: socket.socket):
        self.conn, self.lock = conn, threading.Lock()

    def lines(self):
        buf = b""
        while True:
            try:
                b = self.conn.recv(65536)
            except OSError:
                return
            if not b:
                return
            buf += b
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode(errors="replace")

    def send(self, msg: dict):
        data = (json.dumps(msg) + "\n").encode()
        with self.lock:
            try:
                self.conn.sendall(data)
            except OSError:
                pass

    def close(self):
        try:
            self.conn.close()
        except OSError:
            pass


class _Drop:
    def send(self, msg):
        pass

    def put(self, msg):
        pass


def connect(state: str) -> int:
    """stdio <-> the keeper's socket, for the main Nebula over SSH."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(sock_path(state))
    except OSError as e:
        sys.stderr.write(f"the node keeper is not running here ({e})\n")
        return 2

    def up():
        for line in sys.stdin.buffer:
            s.sendall(line)
        try:
            s.shutdown(socket.SHUT_WR)
        except OSError:
            pass
    threading.Thread(target=up, daemon=True).start()
    out = sys.stdout.buffer
    while True:
        b = s.recv(65536)
        if not b:
            return 0
        out.write(b)
        out.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nodekeeper.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve")
    sv.add_argument("--pair-bin")
    sv.add_argument("--state", required=True)
    sv.add_argument("--facade-port", type=int, default=8081)
    cn = sub.add_parser("connect")
    cn.add_argument("--state", required=True)
    a = ap.parse_args(argv)
    state = os.path.expanduser(a.state)
    if a.cmd == "connect":
        return connect(state)
    Keeper(os.path.expanduser(a.pair_bin) if a.pair_bin else None, state, a.facade_port).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
