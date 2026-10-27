"""Only this page drives this Nebula (nebula/origin.py): a Host that is not one of its names is refused (DNS
rebinding), and so is a browser request from another site's page (cross-site, another port on this machine too);
apps and scripts, which send no Origin, are not affected; refusals are in the security log; "allowed_hosts" adds a
name; the same holds on an entry point."""
import http.client
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from nebula import origin
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call
from nebula.tests.test_entries import free_port


class Headers(dict):
    def get(self, k, d=None):
        return next((v for kk, v in self.items() if kk.lower() == k.lower()), d)


class UnitTest(unittest.TestCase):
    def test_host_names(self):
        class C:
            raw = {"allowed_hosts": ["Nebula.Home.Arpa."]}
        for h in ("127.0.0.1:8300", "[::1]:8300", "192.168.1.20", "localhost:8300", "app.localhost", "nebula.home.arpa:1",
                  None, f"{origin.lan.host_name()}:8300"):
            self.assertTrue(origin.host_ok(C, h), h)
        for h in ("evil.example:8300", "evil.example", "127.0.0.1.evil.example", "localhost.evil.example"):
            self.assertFalse(origin.host_ok(C, h), h)

    def test_cross_site(self):
        same = Headers({"Host": "127.0.0.1:8300", "Origin": "http://127.0.0.1:8300", "Sec-Fetch-Site": "same-origin"})
        self.assertFalse(origin.cross_site(same))
        self.assertFalse(origin.cross_site(Headers({"Host": "127.0.0.1:8300"})))                       # an app, a script
        self.assertFalse(origin.cross_site(Headers({"Host": "x", "Sec-Fetch-Site": "none"})))          # typed in the bar
        self.assertTrue(origin.cross_site(Headers({"Host": "127.0.0.1:8300", "Origin": "https://evil.example"})))
        self.assertTrue(origin.cross_site(Headers({"Host": "127.0.0.1:8300", "Origin": "http://127.0.0.1:3000"})))
        self.assertTrue(origin.cross_site(Headers({"Host": "127.0.0.1:8300", "Origin": "null"})))
        self.assertTrue(origin.cross_site(Headers({"Host": "127.0.0.1:8300", "Sec-Fetch-Site": "cross-site"})))

    def test_what_is_guarded(self):
        self.assertTrue(origin.guarded("POST", "/hooks/x"))
        self.assertTrue(origin.guarded("GET", "/api/config"))
        self.assertTrue(origin.guarded("GET", "/v1/models"))
        self.assertFalse(origin.guarded("GET", "/"))
        self.assertFalse(origin.guarded("GET", "/health"))


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.small = FakeEngine("small")
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
            "tiers": [{"name": "small", "engine": self.small.url}], "allowed_hosts": ["nebula.home.arpa"],
            "assistant": {"halo": {"on": False}},
            "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.port = self.srv.server_port
        self.base = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        self.small.stop()

    def req(self, method, path, headers=None, body=None, port=None):
        c = http.client.HTTPConnection("127.0.0.1", port or self.port, timeout=30)
        h = {"Host": f"127.0.0.1:{port or self.port}", **(headers or {})}
        data = None if body is None else json.dumps(body).encode()
        if data is not None:
            h.setdefault("Content-Type", "application/json")
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        out = (r.status, r.read())
        c.close()
        return out

    def events(self, kind):
        return [e for e in call(self.base, "/api/security")[1]["events"] if e["kind"] == kind]

    def test_rebinding_host_refused(self):
        st, body = self.req("GET", "/api/config", {"Host": f"evil.example:{self.port}"})
        self.assertEqual(st, 403)
        self.assertIn("allowed_hosts", json.loads(body)["error"]["message"])
        self.assertEqual(self.req("GET", "/", {"Host": f"evil.example:{self.port}"})[0], 403)
        self.assertEqual(self.req("POST", "/v1/chat/completions", {"Host": "evil.example"},
                                  {"model": "small", "messages": [{"role": "user", "content": "hi"}]})[0], 403)
        self.assertTrue(self.events("auth.bad_host"))
        # its own names, and a listed one, are fine
        self.assertEqual(self.req("GET", "/api/config", {"Host": f"localhost:{self.port}"})[0], 200)
        self.assertEqual(self.req("GET", "/api/config", {"Host": f"nebula.home.arpa:{self.port}"})[0], 200)

    def test_cross_site_page_refused_own_page_and_apps_fine(self):
        evil = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
        self.assertEqual(self.req("POST", "/api/look", evil, {"theme": "dark"})[0], 403)
        self.assertEqual(self.req("POST", "/v1/chat/completions", evil,
                                  {"model": "small", "messages": [{"role": "user", "content": "hi"}]})[0], 403)
        self.assertEqual(self.req("GET", "/api/config", evil)[0], 403)
        other_port = {"Origin": "http://127.0.0.1:3000", "Sec-Fetch-Site": "same-site"}
        self.assertEqual(self.req("POST", "/api/look", other_port, {"theme": "dark"})[0], 403)
        self.assertTrue(self.events("auth.cross_site"))
        # a link from anywhere still opens the page
        self.assertEqual(self.req("GET", "/", {"Sec-Fetch-Site": "cross-site"})[0], 200)
        # the page itself
        own = {"Origin": self.base, "Sec-Fetch-Site": "same-origin"}
        self.assertEqual(self.req("POST", "/v1/chat/completions", own,
                                  {"model": "small", "messages": [{"role": "user", "content": "hi"}]})[0], 200)
        # an app or a script: no Origin
        self.assertEqual(self.req("POST", "/v1/chat/completions", None,
                                  {"model": "small", "messages": [{"role": "user", "content": "hi"}]})[0], 200)

    def test_entry_point_guarded_too(self):
        st, e = call(self.base, "/api/api-servers", {"name": "Kids", "scope": "loopback", "port": free_port(),
                                                     "formats": ["openai", "ollama"], "models": ["small"], "require_key": False})
        self.assertEqual(st, 200, e)
        p = e["port"]
        self.assertEqual(self.req("GET", "/api/tags", {"Host": f"evil.example:{p}"}, port=p)[0], 403)
        self.assertEqual(self.req("POST", "/api/chat", {"Origin": "https://evil.example"},
                                  {"model": "small", "messages": [{"role": "user", "content": "hi"}], "stream": False}, port=p)[0], 403)
        self.assertEqual(self.req("GET", "/api/tags", port=p)[0], 200)


if __name__ == "__main__":
    unittest.main()
