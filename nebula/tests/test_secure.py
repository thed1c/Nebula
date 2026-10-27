"""Secure by default (design decision, 2026-10-20): this machine's API needs a key unless the owner turns it off; the page
needs a signed-in browser once a PIN or a passkey is set (nebula/signin.py): PINs are hashed and throttled, passkeys
are checked (signature, challenge, origin, user verified, counter), Nebula's own callers carry its internal token."""
import base64
import hashlib
import json
import os
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from nebula import signin, stars
from nebula.tests import WINDOWS, linux_only  # noqa: F401
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine

VEC = {"x": "0x9a0e7bcf651ec121a397a2c1e034db8dd580ccb3b2db6155766652535fc11337", "y": "0x67c6f00666ae37ec21700b8c526bd26d80a0792687ed5f7e9887d8c30ee88b39", "sigs": [["6e6562756c61207369676e2d696e", "304502200f4be01ef92e73658e94e5a232aac6661d858aad9204709b1554c386e8e1c1c8022100f64d1ac1547e10ace0a1b7169bc4a2836049e3205f1fe141c533520da04cea43"], ["", "30450220487b36be4e0aca17a46a763cb0fc4f2eaed9c642a817b4c5d1d5d14641fed904022100ba754f3ea53db896364f3ff71e623e0a937c7ac6459f167195eb7cc4b095e9e6"], ["000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7", "3045022100c3d3c107a6fb6ffed9a213188c641dcabc5d3cb0e4c9787fee8ef0f1d72e0be702203467774328abce530a312884dcb844a7e6767c87cdabd8ee3ddef5692efed334"]]}


def req(base, path, body=None, headers=None, method=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    r = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read() or b"{}"), resp.headers.get("Set-Cookie")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), e.headers.get("Set-Cookie")


class Es256Test(unittest.TestCase):
    def test_fixed_vectors(self):
        x, y = int(VEC["x"], 16), int(VEC["y"], 16)
        for m, s in VEC["sigs"]:
            self.assertTrue(signin.es256_verify(x, y, bytes.fromhex(m), bytes.fromhex(s)))
            self.assertFalse(signin.es256_verify(x, y, bytes.fromhex(m) + b"!", bytes.fromhex(s)))
            bad = bytearray(bytes.fromhex(s)); bad[-1] ^= 1
            self.assertFalse(signin.es256_verify(x, y, bytes.fromhex(m), bytes(bad)))
        self.assertFalse(signin.es256_verify(x, y + 1, b"x", bytes.fromhex(VEC["sigs"][0][1])))   # a point off the curve

    def test_cbor(self):
        v, _ = signin.cbor(bytes.fromhex("a3616101616282020361636d6865"))   # {"a": 1, "b": [2, 3], "c": "he"}
        self.assertEqual(v, {"a": 1, "b": [2, 3], "c": "he"})
        self.assertEqual(signin.cbor(bytes.fromhex("3818"))[0], -25)


class Rig:
    def __init__(self, extra=None):
        self.eng = FakeEngine("small")
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({"port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
                                               "tiers": [{"name": "small", "engine": self.eng.url}], "assistant": {"halo": {"on": False}},
                                               "lan": {"decided": True, "discoverable": False}, **(extra or {})}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.port = self.srv.server_port
        self.base = f"http://127.0.0.1:{self.port}"
        self.local = f"http://localhost:{self.port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close(); self.eng.stop()


class KeyDefaultTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()

    def tearDown(self):
        self.rig.stop()

    def chat(self, headers=None):
        return req(self.rig.base, "/v1/chat/completions", {"model": "small", "messages": [{"role": "user", "content": "hi"}]}, headers)[0]

    def test_apps_on_this_pc_need_no_key_unless_turned_on(self):
        self.assertEqual(self.chat(), 200)                                                        # design decision
        self.assertFalse(req(self.rig.base, "/api/stars")[1]["api"]["require_key"])
        st, k, _ = req(self.rig.base, "/api/stars/api/keys", {"label": "app"})
        req(self.rig.base, "/api/stars/api", {"require_key": True})                               # the owner's choice
        self.assertEqual(self.chat(), 401)
        self.assertEqual(self.chat({"Authorization": "Bearer " + k["key"]}), 200)
        self.assertEqual(self.chat({"X-Nebula-Internal": self.rig.srv.internal_token}), 200)      # Nebula itself
        self.assertEqual(self.chat({"X-Nebula-Internal": "guess"}), 401)

    def test_first_run_makes_no_key_when_none_is_needed(self):
        from nebula.tests.test_wizards import finish
        st, job, _ = req(self.rig.base, "/api/wizards/first-run/apply", {"answers": {"source": "other", "other": "running", "url": self.rig.eng.url}})
        self.assertEqual(st, 200, job)
        v = finish(self.rig.base, job)
        self.assertEqual(v["state"], "done", v)
        self.assertNotIn("api_key", v["result"])
        self.assertEqual(self.chat(), 200)

    def test_admin_token_file_is_private(self):
        p = os.path.join(self.rig.tmp, "nebula-keys", "admin.token")
        if not os.path.exists(p):
            p = next(str(x) for x in Path(self.rig.tmp).rglob("admin.token"))
        if not WINDOWS:
            self.assertEqual(os.stat(p).st_mode & 0o777, 0o600)
        self.assertEqual(Path(p).read_text(), self.rig.srv.internal_token)


class Authenticator:
    """A software passkey (what a phone or a laptop does), for the test."""

    def __init__(self, rp):
        from cryptography.hazmat.primitives.asymmetric import ec
        self.ec, self.rp, self.k, self.cid, self.count = ec, rp, ec.generate_private_key(ec.SECP256R1()), os.urandom(16), 0

    def _ad(self, flags, attested=False):
        self.count += 1
        ad = hashlib.sha256(self.rp.encode()).digest() + bytes([flags]) + struct.pack(">I", self.count)
        if attested:
            n = self.k.public_key().public_numbers()
            cose = bytes.fromhex("a5010203262001215820") + n.x.to_bytes(32, "big") + bytes.fromhex("225820") + n.y.to_bytes(32, "big")
            ad += b"\0" * 16 + struct.pack(">H", len(self.cid)) + self.cid + cose
        return ad

    def create(self, opts, origin, flags=0x45):
        cd = json.dumps({"type": "webauthn.create", "challenge": opts["challenge"], "origin": origin}).encode()
        ad = self._ad(flags, attested=True)
        att = bytes.fromhex("a363666d74646e6f6e656761747453746d74a068617574684461746159") + struct.pack(">H", len(ad)) + ad
        return {"clientDataJSON": signin.b64u(cd), "attestationObject": signin.b64u(att), "name": "test key"}

    def get(self, opts, origin, flags=0x05, count=None):
        from cryptography.hazmat.primitives import hashes
        cd = json.dumps({"type": "webauthn.get", "challenge": opts["challenge"], "origin": origin}).encode()
        ad = self._ad(flags)
        if count is not None:
            ad = ad[:33] + struct.pack(">I", count) + ad[37:]
        sig = self.k.sign(ad + hashlib.sha256(cd).digest(), self.ec.ECDSA(hashes.SHA256()))
        return {"id": signin.b64u(self.cid), "clientDataJSON": signin.b64u(cd), "authenticatorData": signin.b64u(ad), "signature": signin.b64u(sig)}


class SignInTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()

    def tearDown(self):
        self.rig.stop()

    def test_open_until_set_then_pin_signs_in(self):
        b = self.rig.base
        self.assertFalse(req(b, "/api/signin")[1]["configured"])
        self.assertEqual(req(b, "/api/stars")[0], 200)                                     # nothing set yet: open here
        self.assertEqual(req(b, "/api/signin/pin/set", {"pin": "1234"})[0], 400)           # a run
        st, out, cookie = req(b, "/api/signin/pin/set", {"pin": "4826"})
        self.assertEqual(st, 200, out)
        self.assertIn("HttpOnly", cookie); self.assertIn("SameSite=Strict", cookie)       # the first one signs this browser in
        sess = {"Cookie": cookie.split(";")[0]}
        self.assertEqual(req(b, "/api/stars")[0], 401)                                     # anyone else: sign in
        self.assertEqual(req(b, "/api/stars", headers=sess)[0], 200)
        self.assertEqual(req(b, "/api/stars", headers={"X-Nebula-Internal": self.rig.srv.internal_token})[0], 200)
        self.assertEqual(req(b, "/api/signin/pin/set", {"pin": "9999"})[0], 401)           # changing it: signed in only
        pin = json.loads(Path(self.rig.path).read_text())["signin"]["pin"]               # only its hash is kept
        self.assertEqual(set(pin), {"salt", "scrypt", "set_at"})
        self.assertNotIn("4826", [str(v) for v in pin.values()])
        st, _, c2 = req(b, "/api/signin/pin", {"pin": "4826"})
        self.assertEqual(st, 200)
        s2 = {"Cookie": c2.split(";")[0]}
        self.assertEqual(req(b, "/api/stars", headers=s2)[0], 200)
        req(b, "/api/signin/out", {}, s2)
        self.assertEqual(req(b, "/api/stars", headers=s2)[0], 401)
        kinds = [e["kind"] for e in req(b, "/api/security", headers=sess)[1]["events"]]
        self.assertIn("signin.ok", kinds); self.assertIn("signin.changed", kinds)

    def test_wrong_pins_are_throttled(self):
        b = self.rig.base
        req(b, "/api/signin/pin/set", {"pin": "4826"})
        codes = [req(b, "/api/signin/pin", {"pin": "0000"})[0] for _ in range(signin.FREE_TRIES)]
        self.assertEqual(codes, [401] * signin.FREE_TRIES)
        self.assertEqual(req(b, "/api/signin/pin", {"pin": "4826"})[0], 429)               # even the right one waits

    def test_passkey_made_and_used(self):
        try:
            import cryptography  # noqa: F401
        except ImportError:
            self.skipTest("the software passkey needs the cryptography package")
        L = self.rig.local
        self.assertEqual(req(self.rig.base, "/api/signin/passkey/create")[0], 400)          # an address: no passkeys
        st, opts, _ = req(L, "/api/signin/passkey/create")
        self.assertEqual((st, opts["rp"]["id"]), (200, "localhost"), opts)
        a = Authenticator("localhost")
        self.assertEqual(req(L, "/api/signin/passkey/create", a.create(opts, "http://evil.example"))[0], 400)   # another page
        st, opts, _ = req(L, "/api/signin/passkey/create")
        st, out, cookie = req(L, "/api/signin/passkey/create", a.create(opts, L))
        self.assertEqual(st, 200, out)
        self.assertEqual(req(L, "/api/stars")[0], 401)
        st, g, _ = req(L, "/api/signin/passkey")
        st, out, cookie = req(L, "/api/signin/passkey", a.get(g, L))
        self.assertEqual(st, 200, out)
        self.assertEqual(req(L, "/api/stars", headers={"Cookie": cookie.split(";")[0]})[0], 200)
        st, g, _ = req(L, "/api/signin/passkey")
        replay = a.get(g, L)
        self.assertEqual(req(L, "/api/signin/passkey", replay)[0], 200)
        self.assertEqual(req(L, "/api/signin/passkey", replay)[0], 400)                    # its challenge is used up
        st, g, _ = req(L, "/api/signin/passkey")
        self.assertEqual(req(L, "/api/signin/passkey", a.get(g, L, flags=0x01))[0], 401)   # the person not verified
        st, g, _ = req(L, "/api/signin/passkey")
        self.assertEqual(req(L, "/api/signin/passkey", a.get(g, L, count=1))[0], 401)      # its counter went backwards
        st, g, _ = req(L, "/api/signin/passkey")
        other = Authenticator("localhost"); other.cid = a.cid                              # the same id, another key
        self.assertEqual(req(L, "/api/signin/passkey", other.get(g, L))[0], 401)


class SandboxDefaultTest(unittest.TestCase):
    """Engines go into OpenShell sandboxes by default; without one they run and warn, and a
    model shared from such an engine carries the warning to the other side."""

    @linux_only

    def test_the_plan(self):
        from nebula.engines import sandbox_plan
        from nebula import openshell
        llama = {"name": "e", "kind": "llama.cpp", "model": "m.gguf", "port": 9}
        self.assertEqual(sandbox_plan({"name": "c", "kind": "command", "command": ["x"]})[0], "local")
        self.assertIn("you chose", sandbox_plan(dict(llama, where="local"))[1])
        self.assertIn("no sandbox image", sandbox_plan(dict(llama, kind="nonesuch"))[1])
        self.assertIn("not installed", sandbox_plan(dict(llama, kind="impulse"), exe="/nonexistent/openshell")[1])   # its image is built here
        self.assertIn("not installed", sandbox_plan(llama, exe="/nonexistent/openshell")[1])
        self.assertEqual(sandbox_plan(dict(llama, where="openshell"), exe="/nonexistent/openshell"), ("openshell", None))
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, "openshell")
            Path(fake).write_text("#!/bin/sh\necho '  Status: Connected'\n"); os.chmod(fake, 0o755)
            self.assertEqual(sandbox_plan(llama, exe=fake), ("openshell", None))
            down = os.path.join(d, "down"); Path(down).write_text("#!/bin/sh\necho refused >&2; exit 1\n"); os.chmod(down, 0o755)
            self.assertIn("gateway is not running", sandbox_plan(llama, exe=down)[1])
            openshell._READY.clear()

    def test_policy_is_locked_down(self):
        from nebula.openshell import DENY_ALL
        self.assertIn("network_policies: {}", DENY_ALL)
        self.assertIn('run_as_user: "65534"', DENY_ALL)
        self.assertIn("/models", DENY_ALL.split("read_write")[0])          # the models: read-only

    def test_a_shared_model_from_an_unsandboxed_engine_warns(self):
        from nebula import gravity

        class T:
            managed, risky, mesh, name = "e", False, None, "m"

        class E:
            unsandboxed = {"e": "OpenShell is not installed on this machine"}
        self.assertIn("without a sandbox", gravity.needs_care(T(), E()))
        self.assertIsNone(gravity.needs_care(T(), type("E2", (), {"unsandboxed": {}})()))


class LocalNamesTest(unittest.TestCase):
    def test_localhost_names_are_this_machine(self):
        from nebula.engine import plain_remote
        self.assertFalse(plain_remote("http://default--nebula-engine-x.openshell.localhost:17670"))   # a sandbox's engine
        self.assertFalse(plain_remote("http://127.0.0.1:8080"))
        self.assertTrue(plain_remote("http://192.168.1.20:8080"))
        self.assertTrue(plain_remote("http://localhost.evil.example:8080"))


if __name__ == "__main__":
    unittest.main()
