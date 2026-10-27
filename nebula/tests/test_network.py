"""Model servers on your network (nebula/network.py, nebula/pairface.py): no plain HTTP between machines; a server's
Nebula answers as its router's engine; the tenant travels signed, and a guest stays a guest there. Real PAIR between
machines is greenfield/household.sh."""
import json
import os
import tempfile
import threading
import unittest
import urllib.request
import uuid
from pathlib import Path

from nebula.config import Config
from nebula.engine import CALL_TENANT, Engine, EngineError, plain_remote
from nebula.pairface import Face, sign
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call


def serve(d):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "nebula.json")
    Path(path).write_text(json.dumps({"port": 0, "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False,
                                      "lan": {"decided": True, "discoverable": False, "uuid": uuid.uuid4().hex, "name": "n"}, **d}))
    srv = make_server(Config.from_args(["--config", path]))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


class PlainHttpTest(unittest.TestCase):
    def test_never_to_another_machine(self):
        self.assertTrue(plain_remote("http://10.9.9.9:8080"))
        self.assertTrue(plain_remote("http://second-box:8080"))
        for ok in ("http://127.0.0.1:8080", "http://localhost:1", "https://api.example.com"):
            self.assertFalse(plain_remote(ok), ok)
        with self.assertRaises(EngineError) as cm:
            Engine("http://10.9.9.9:8080", timeout_s=2).chat({"messages": []})
        self.assertIn("never sends model traffic unencrypted", cm.exception.message)
        srv, base = serve({"tiers": [{"name": "local", "engine": "http://127.0.0.1:9"}]})
        try:
            st, err = call(base, "/api/models", {"name": "box", "kind": "engine", "url": "http://10.9.9.9:8080"})
            self.assertEqual(st, 400, err)
            self.assertIn("model server on your network", err["error"]["message"])
        finally:
            srv.shutdown(); srv.server_close(); srv.tools.close()


class FaceTest(unittest.TestCase):
    """A server's Nebula, as its router's engine: router props, its models, requests into Nebula, the tenant signed."""

    def setUp(self):
        self.engine = FakeEngine("tiny")
        self.box, self.boxbase = serve({"tiers": [{"name": "tiny", "engine": self.engine.url, "shared": True},
                                                  {"name": "private", "engine": self.engine.url}]})
        self.secret = "household-secret"
        self.face = Face(self.box, 0, secret=self.secret)
        self.url = f"http://127.0.0.1:{self.face.port}"

    def tearDown(self):
        self.face.close()
        self.box.shutdown(); self.box.server_close(); self.box.tools.close()
        self.engine.shutdown()

    def last_tenant(self):
        row = self.box.ledger.db.execute("SELECT inputs FROM decisions WHERE advisory = 0 ORDER BY rowid DESC LIMIT 1").fetchone()
        return json.loads(row[0]).get("tenant")

    def test_router_shape(self):
        self.assertEqual(json.loads(urllib.request.urlopen(self.url + "/props").read())["role"], "router")
        ids = [m["id"] for m in json.loads(urllib.request.urlopen(self.url + "/models").read())["data"]]
        self.assertEqual(ids, ["tiny", "private"])

    def test_the_tenant_travels_signed(self):
        main = Engine(self.url, timeout_s=20, sign_secret=self.secret)
        body = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}
        tok = CALL_TENANT.set("caller:friend-app")
        try:
            main.chat(dict(body))
        finally:
            CALL_TENANT.reset(tok)
        self.assertEqual(self.last_tenant(), "peer:via:caller:friend-app")              # a guest there too
        tok = CALL_TENANT.set("caller:friend-app")
        try:
            with self.assertRaises(EngineError) as cm:
                main.chat({"model": "private", "messages": [{"role": "user", "content": "hi"}]})
            self.assertIn("not shared with guests", cm.exception.message)                  # and kept to shared models
        finally:
            CALL_TENANT.reset(tok)
        Engine(self.url, timeout_s=20, sign_secret=self.secret).chat(dict(body))      # the owner's own: owner
        self.assertEqual(self.last_tenant(), "owner")
        forged = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                        headers={"Content-Type": "application/json", "X-Nebula-Tenant": "caller:x",
                                                 "X-Nebula-Tenant-Sig": sign("wrong", "caller:x", json.dumps(body).encode())})
        urllib.request.urlopen(forged, timeout=20).read()
        self.assertEqual(self.last_tenant(), "owner")                                     # a bad signature is not a tenant


if __name__ == "__main__":
    unittest.main()
