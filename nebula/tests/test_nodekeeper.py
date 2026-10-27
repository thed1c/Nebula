"""The node keeper (nebula/nodekeeper.py), a model machine's one file: PAIR's engine address (router props, its models,
each request to the engine serving its model, streamed), engines only on that machine, and its JSON-RPC socket with
the keeper's own methods; PAIR's methods go to PAIR's broker (absent here: an answer, not a dead link)."""
import json
import os
import socket
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from nebula import nodekeeper as K
from nebula.tests import linux_only
from nebula.tests.fake_engine import FakeEngine


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@linux_only
class KeeperTest(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.mkdtemp(prefix="nk", dir="/tmp")                # a socket path stays short
        self.k = K.Keeper(None, self.state, free_port())
        os.makedirs(self.state, exist_ok=True)
        self.k.serve_facade()
        self.k.serve_socket()
        self.url = f"http://127.0.0.1:{self.k.facade_port}"
        self.a, self.b = FakeEngine("alpha"), FakeEngine("beta")

    def tearDown(self):
        self.k.stop()
        self.a.stop(); self.b.stop()

    def rpc(self, method, params=None):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(K.sock_path(self.state))
        s.sendall((json.dumps({"jsonrpc": "2.0", "id": 5, "method": method, "params": params or {}}) + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            buf += s.recv(65536)
        s.close()
        return json.loads(buf)

    def get(self, path):
        return json.loads(urllib.request.urlopen(self.url + path, timeout=10).read())

    def post(self, body):
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=20).read())

    def test_router_face_and_requests_by_model(self):
        self.assertEqual(self.get("/props")["role"], "router")
        self.assertEqual(self.get("/models")["data"], [])
        got = self.rpc("keeper:set-engines", {"engines": {"m-alpha": self.a.url, "m-beta": self.b.url}})
        self.assertEqual(sorted(got["result"]["engines"]), ["m-alpha", "m-beta"])
        self.assertEqual([m["id"] for m in self.get("/v1/models")["data"]], ["m-alpha", "m-beta"])
        msgs = [{"role": "user", "content": "hi"}]
        self.post({"model": "m-beta", "messages": msgs})
        self.post({"model": "m-alpha", "messages": msgs})
        self.post({"model": "m-beta", "messages": msgs})
        self.assertEqual((len(self.a.requests), len(self.b.requests)), (1, 2))
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post({"model": "nope", "messages": msgs})
        self.assertEqual(cm.exception.code, 404)
        with open(os.path.join(self.state, "engines.json")) as f:              # kept for the next start
            self.assertEqual(sorted(json.load(f)), ["m-alpha", "m-beta"])

    def test_only_engines_on_this_machine(self):
        got = self.rpc("keeper:set-engines", {"engines": {"x": "http://192.0.2.5:8080"}})
        self.assertIn("on this machine", got["error"]["message"])
        got = self.rpc("keeper:set-engines", {"engines": {"x": "http://default--nebeng-x.openshell.localhost:17670"}})
        self.assertIn("x", got["result"]["engines"])

    def test_pair_methods_without_pair_are_answered(self):
        got = self.rpc("cluster:get-node-id")
        self.assertIn("node programs are not running", got["error"]["message"])
        self.assertEqual(self.rpc("keeper:notes")["result"], {"notes": []})


if __name__ == "__main__":
    unittest.main()
