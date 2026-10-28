"""The update check (nebula/updates.py): only a feed signed with Nebula's key is believed; versions compare as numbers;
"off" never touches the network; a Nebula that cannot update itself opens the release page; a download that does not
match its sha256 is refused. And the prereqs that follow a moved pin (Qdrant, Halo's model)."""
import base64
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nebula import updates as U

SEED = bytes(range(32))                         # a test key: its public half stands in for PUBLIC_KEY
PUB = U.public_of(SEED).hex()


def feed(version="9.9.9", **extra) -> bytes:
    body = {"version": version, "notes": "Faster replies.", "published": "2026-10-25",
            "files": {U.platform_key() or "linux-x86_64": {"url": "https://example.invalid/nebula.run", "sha256": "ab" * 32}},
            "release_page": f"https://github.com/thed1c/Nebula/releases/tag/v{version}", **extra}
    return json.dumps(body).encode()


def signed(body: bytes, seed: bytes = SEED) -> bytes:
    return base64.b64encode(U.sign(seed, body))


class Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def cfg_in(tmp, extra=None):
    path = os.path.join(tmp, "nebula.json")
    raw = {"port": 0, **(extra or {})}
    Path(path).write_text(json.dumps(raw))
    saved = []
    return SimpleNamespace(raw=raw, path=path, base_dir=tmp, save=lambda: saved.append(1) or path)


class SignatureTest(unittest.TestCase):
    def test_rfc8032_vector(self):
        pub = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
        sig = bytes.fromhex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")
        self.assertTrue(U.verify(pub, b"", sig))
        self.assertEqual(U.sign(bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"), b""), sig)

    def test_a_good_signature_is_believed(self):
        body = feed()
        f = U.read_feed(body, signed(body), PUB)
        self.assertEqual(f["version"], "9.9.9")

    def test_a_bad_signature_a_changed_byte_or_another_key_is_ignored(self):
        body = feed()
        self.assertIsNone(U.read_feed(body, signed(body, bytes(32)), PUB))                 # another key
        self.assertIsNone(U.read_feed(body.replace(b"Faster", b"Slower"), signed(body), PUB))   # one word changed
        self.assertIsNone(U.read_feed(body, b"", PUB))                                     # unsigned
        self.assertIsNone(U.read_feed(body, b"not base64!", PUB))

    def test_a_signed_but_malformed_feed_is_ignored(self):
        body = feed(release_page="http://example.invalid/")
        self.assertIsNone(U.read_feed(body, signed(body), PUB))

    def test_the_real_key_is_a_point(self):
        self.assertEqual(len(bytes.fromhex(U.PUBLIC_KEY)), 32)
        self.assertIsNotNone(U._point(bytes.fromhex(U.PUBLIC_KEY)))

    def test_cryptography_agrees_when_it_is_here(self):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError:
            self.skipTest("no cryptography here")
        k = Ed25519PrivateKey.from_private_bytes(SEED)
        self.assertEqual(k.sign(b"latest.json"), U.sign(SEED, b"latest.json"))


class VersionTest(unittest.TestCase):
    def test_older_equal_newer(self):
        self.assertTrue(U.is_newer("0.1.3", "0.1.2"))
        self.assertFalse(U.is_newer("0.1.2", "0.1.2"))
        self.assertFalse(U.is_newer("0.1.1", "0.1.2"))
        self.assertTrue(U.is_newer("0.1.10", "0.1.9"))        # numbers, not text
        self.assertFalse(U.is_newer("v0.2", "0.2.0"))
        self.assertTrue(U.is_newer("0.1.3.1", "0.1.3"))        # a fix release: four numbers
        self.assertFalse(U.is_newer("0.1.3", "0.1.3.1"))
        self.assertTrue(U.is_newer("0.1.4", "0.1.3.1"))
        with self.assertRaises(ValueError):
            U.version_tuple("0.1.3-beta")


class CheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.key = mock.patch.object(U, "PUBLIC_KEY", PUB)
        self.key.start()
        self.env = mock.patch.dict(os.environ, {"NEBULA_UPDATES_FEED": ""})
        self.env.start()

    def tearDown(self):
        self.key.stop()
        self.env.stop()

    def updates(self, extra=None):
        cfg = cfg_in(self.tmp, extra)
        return U.Updates(SimpleNamespace(cfg=cfg, ledger=None, admission=None))

    def serve(self, body):
        sig = signed(body)
        return lambda req, timeout=0: Resp(sig if req.full_url.endswith(".sig") else body)

    def test_mode_off_makes_no_request(self):
        u = self.updates({"updates": {"mode": "off"}})
        with mock.patch.object(U.urllib.request, "urlopen", side_effect=AssertionError("no network when off")) as op:
            v = u.check()
        op.assert_not_called()
        self.assertEqual(v["mode"], "off")
        self.assertFalse(v["newer"])

    def test_the_feed_url_is_the_configs(self):
        u = self.updates({"updates": {"feed": "https://example.invalid/mine.json"}})
        seen = []
        with mock.patch.object(U.urllib.request, "urlopen", side_effect=lambda req, timeout=0: seen.append(req.full_url) or self.serve(feed())(req)):
            u.check()
        self.assertEqual(seen[:2], ["https://example.invalid/mine.json", "https://example.invalid/mine.json.sig"])

    def test_notify_while_the_release_is_private(self):
        u = self.updates()
        with mock.patch.object(U.urllib.request, "urlopen", side_effect=self.serve(feed())), \
             mock.patch.object(U, "install_dir", return_value=Path(self.tmp)), \
             mock.patch.object(U, "downloadable", return_value=False):
            v = u.check()
        self.assertTrue(v["newer"])
        self.assertFalse(v["can_install"])
        self.assertEqual(v["latest"], "9.9.9")
        self.assertIn("releases/tag/v9.9.9", v["release_page"])
        r = u.install()
        self.assertEqual(r["action"], "open")
        self.assertEqual(r["open"], v["release_page"])
        self.assertTrue(Path(self.tmp, "nebula.updates.json").is_file())          # the last answer, kept
        self.assertEqual(self.updates().view()["latest"], "9.9.9")

    def test_install_once_public_and_put_in_place_by_the_installer(self):
        u = self.updates()
        with mock.patch.object(U.urllib.request, "urlopen", side_effect=self.serve(feed())), \
             mock.patch.object(U, "install_dir", return_value=Path(self.tmp)), \
             mock.patch.object(U, "downloadable", return_value=True):
            v = u.check()
        self.assertTrue(v["can_install"])
        started = []
        with mock.patch.object(U, "install_dir", return_value=Path(self.tmp)), \
             mock.patch.object(U.threading, "Thread", side_effect=lambda target, args, **k: SimpleNamespace(start=lambda: started.append(args))):
            r = u.install()
        self.assertEqual(r["action"], "installing")
        self.assertEqual(started[0][0]["sha256"], "ab" * 32)

    def test_a_checkout_or_pip_install_is_told_only(self):
        u = self.updates()
        with mock.patch.object(U.urllib.request, "urlopen", side_effect=self.serve(feed())), \
             mock.patch.object(U, "install_dir", return_value=None), \
             mock.patch.object(U, "downloadable", side_effect=AssertionError("not asked")):
            v = u.check()
        self.assertTrue(v["newer"])
        self.assertFalse(v["can_install"])

    def test_the_same_or_an_older_version_is_not_offered(self):
        for ver in (U.__version__, "0.0.1"):
            u = self.updates()
            with mock.patch.object(U.urllib.request, "urlopen", side_effect=self.serve(feed(ver))):
                v = u.check()
            self.assertFalse(v["newer"], ver)
            self.assertIsNone(v["release_page"])
            self.assertEqual(u.install()["action"], "none")

    def test_a_badly_signed_feed_changes_nothing(self):
        u = self.updates()
        body = feed()
        with mock.patch.object(U.urllib.request, "urlopen", side_effect=lambda req, timeout=0: Resp(signed(body, bytes(32)) if req.full_url.endswith(".sig") else body)):
            v = u.check()
        self.assertFalse(v["newer"])
        self.assertEqual(v["error"], "not signed")

    def test_a_download_that_does_not_match_its_sha256_is_refused(self):
        dest = Path(self.tmp, "nebula.run")
        with mock.patch.object(U.urllib.request, "urlopen", return_value=Resp(b"tampered")):
            with self.assertRaises(RuntimeError):
                U.fetch_checked("https://example.invalid/nebula.run", dest, "0" * 64)
        self.assertFalse(dest.exists())
        self.assertFalse(dest.with_suffix(".run.part").exists())
        body = b"#!/bin/sh\necho installed\n"
        with mock.patch.object(U.urllib.request, "urlopen", return_value=Resp(body)):
            self.assertEqual(U.fetch_checked("https://example.invalid/nebula.run", dest, hashlib.sha256(body).hexdigest()), dest)

    def test_a_refused_download_never_runs_the_installer(self):
        u = self.updates()
        with mock.patch.object(U, "fetch_checked", side_effect=RuntimeError("the download does not match its sha256")), \
             mock.patch.object(U, "run_installer", side_effect=AssertionError("never run")):
            u._install({"url": "https://example.invalid/nebula.run", "sha256": "0" * 64})
        self.assertTrue(u.installing.startswith("failed"))

    def test_it_waits_for_a_turn_in_flight(self):
        u = self.updates()
        calls = iter([True, True, False])
        ran = []
        with mock.patch.object(U, "fetch_checked", side_effect=lambda url, dest, sha: dest), \
             mock.patch.object(U, "busy", side_effect=lambda s: next(calls)), \
             mock.patch.object(u.closing, "wait", return_value=False), \
             mock.patch.object(U, "install_dir", return_value=Path(self.tmp)), \
             mock.patch.object(U, "run_installer", side_effect=lambda *a: ran.append(a)), \
             mock.patch.object(U.os, "name", "posix"):
            u._install({"url": "https://example.invalid/nebula.run", "sha256": "0" * 64})
        self.assertEqual(len(ran), 1)
        self.assertEqual(u.installing, "installing")

    def test_the_mode_is_saved(self):
        cfg = cfg_in(self.tmp)
        self.assertEqual(U.set_mode(cfg, "off")["mode"], "off")
        with self.assertRaises(ValueError):
            U.set_mode(cfg, "always")

    def test_demand_is_the_default_and_the_old_ask_reads_as_demand(self):
        for raw in ({}, {"updates": {"mode": "ask"}}, {"updates": {"mode": "nonsense"}}):
            self.assertEqual(U.settings(SimpleNamespace(raw=raw))["mode"], "demand", raw)
        self.assertEqual(U.settings(SimpleNamespace(raw={"updates": {"mode": "launch"}}))["mode"], "launch")

    def test_demand_and_off_start_no_thread_and_make_no_request(self):
        for mode in ("demand", "off"):
            u = self.updates({"updates": {"mode": mode}})
            with mock.patch.object(U.urllib.request, "urlopen", side_effect=AssertionError("no request")), \
                 mock.patch.object(U.threading, "Thread", side_effect=AssertionError("no thread")):
                u.start()
            self.assertIsNone(u.thread, mode)

    def test_launch_looks_once_and_never_again(self):
        cfg = cfg_in(self.tmp, {"updates": {"mode": "launch"}})
        u = U.Updates(SimpleNamespace(cfg=cfg, ledger=None, admission=None), first_s=0)
        seen = []
        with mock.patch.object(U.urllib.request, "urlopen", side_effect=lambda req, timeout=0: seen.append(req.full_url) or self.serve(feed())(req)):
            u.start()
            u.thread.join(5)
        self.assertFalse(u.thread.is_alive())                   # it ended: no timer left behind
        self.assertEqual(len(seen), 2)                          # the feed and its signature, once
        self.assertTrue(u.view()["newer"])

    def test_launch_switched_off_before_its_minute_does_not_look(self):
        cfg = cfg_in(self.tmp, {"updates": {"mode": "launch"}})
        u = U.Updates(SimpleNamespace(cfg=cfg, ledger=None, admission=None), first_s=0.2)
        with mock.patch.object(U.urllib.request, "urlopen", side_effect=AssertionError("no request")):
            u.start()
            U.set_mode(cfg, "off")
            u.thread.join(5)
        self.assertIsNone(u.view()["last_checked"])

    def test_look_now_works_on_demand(self):
        u = self.updates()                                      # no mode: demand
        with mock.patch.object(U.urllib.request, "urlopen", side_effect=self.serve(feed())):
            self.assertTrue(u.check()["newer"])


class ApiTest(unittest.TestCase):
    """GET /api/updates, POST /api/updates (the Ask/Off switch), /check, /install, on a real test server."""

    def test_the_page_api(self):
        import threading
        import urllib.request
        from nebula.config import Config
        from nebula.server import make_server
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "nebula.json")
        Path(path).write_text(json.dumps({"port": 0, "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False, "tiers": [],
                                          "assistant": {"halo": {"on": False}},
                                          "lan": {"decided": True, "discoverable": False, "uuid": "u-1", "name": "n"}}))
        srv = make_server(Config.from_args(["--config", path]))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_port}"

        def ask(p, body=None):
            req = urllib.request.Request(base + p, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        try:
            v = ask("/api/updates")
            self.assertEqual((v["mode"], v["current"], v["newer"]), ("demand", U.__version__, False))
            self.assertEqual(ask("/api/updates/check", {})["error"], "not reached")    # the tests' feed: nobody there
            self.assertEqual(ask("/api/updates", {"mode": "off"})["mode"], "off")
            self.assertEqual(json.loads(Path(path).read_text())["updates"]["mode"], "off")
            self.assertEqual(ask("/api/updates/install", {})["action"], "none")
        finally:
            srv.shutdown(); srv.server_close(); srv.tools.close()


class InstallDirTest(unittest.TestCase):
    def test_only_the_installers_own_python(self):
        tmp = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": tmp, "LOCALAPPDATA": tmp}), \
             mock.patch.object(U.os, "name", "posix"):
            with mock.patch.object(U.sys, "executable", os.path.join(tmp, "nebula-eh", "python", "bin", "python3")):
                self.assertEqual(U.install_dir(), Path(tmp, "nebula-eh"))
            with mock.patch.object(U.sys, "executable", "/usr/bin/python3"):
                self.assertIsNone(U.install_dir())


class PinnedPrereqsTest(unittest.TestCase):
    """A pin that moves reaches an install that already has the older piece."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_qdrant_from_an_older_pin_is_replaced_and_your_own_is_kept(self):
        from nebula import prereqs
        from nebula.rag import vectors
        data = os.path.join(self.tmp, "nebula-lookup")
        own = Path(data, "bin", "qdrant")
        own.parent.mkdir(parents=True)
        own.write_text("#!/bin/sh\n")
        own.chmod(0o755)
        Path(data, "bin", "qdrant.version").write_text("v1.0.0 x\n")
        cfg = SimpleNamespace(raw={"lookup": {"data_dir": data}}, base_dir=self.tmp)
        with mock.patch.object(prereqs, "WINDOWS", False), mock.patch.object(vectors, "install") as inst:
            self.assertIn("updated", prereqs.qdrant(cfg))
        inst.assert_called_once()
        Path(data, "bin", "qdrant.version").write_text(f"{vectors.QDRANT_VERSION} x\n")
        with mock.patch.object(prereqs, "WINDOWS", False), mock.patch.object(vectors, "install") as inst:
            self.assertIn("already here", prereqs.qdrant(cfg))
        inst.assert_not_called()
        mine = Path(self.tmp, "my-qdrant")
        mine.write_text("#!/bin/sh\n")
        mine.chmod(0o755)
        cfg.raw["lookup"]["qdrant_bin"] = str(mine)
        with mock.patch.object(prereqs, "WINDOWS", False), mock.patch.object(vectors, "install") as inst:
            self.assertIn("your own", prereqs.qdrant(cfg))
        inst.assert_not_called()

    def test_halos_model_of_the_right_size_but_an_older_pin_is_fetched_again(self):
        from nebula import halo_model as hm, prereqs
        models = os.path.join(self.tmp, "models")
        m = json.loads(json.dumps(hm.manifest()))
        old = b"an older pin's file"
        m["base"].update(bytes=len(old), sha256=hashlib.sha256(b"the new pin's file!").hexdigest())
        p = Path(hm.base_path(models, m))
        p.parent.mkdir(parents=True)
        p.write_bytes(old)                                      # same size as the pin: presence alone would keep it
        cfg = SimpleNamespace(raw={"models_dir": models}, base_dir=self.tmp)
        with mock.patch.dict(os.environ, {"NEBULA_HALO_BASE": ""}), mock.patch.object(hm, "manifest", return_value=m), \
             mock.patch.object(hm, "ensure_base", return_value=None) as ens:
            prereqs.halo_model(cfg)
        ens.assert_called_once()
        m["base"]["sha256"] = hashlib.sha256(old).hexdigest()
        with mock.patch.dict(os.environ, {"NEBULA_HALO_BASE": ""}), mock.patch.object(hm, "manifest", return_value=m), \
             mock.patch.object(hm, "ensure_base", side_effect=AssertionError("not fetched")):
            self.assertIn("already here", prereqs.halo_model(cfg))


if __name__ == "__main__":
    unittest.main()


def app_feed(bases=("0.1.4",), app_url="https://example.invalid/nebula-app.tar.gz") -> bytes:
    plat = U.platform_key() or "linux-x86_64"
    body = json.loads(feed())
    body["files"][plat]["app"] = {"url": app_url, "sha256": "cd" * 32, "bytes": 23 * 2**20, "bases": list(bases)}
    return json.dumps(body).encode()


class SmallUpdateTest(unittest.TestCase):
    """The owner's "small updates" (2026-10-27): Nebula's code alone when nothing beside it changed, only for an install
    made by one of the update's base installers; any other gets the whole installer."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "app"))
        self.key = mock.patch.object(U, "PUBLIC_KEY", PUB)
        self.key.start()
        self.env = mock.patch.dict(os.environ, {"NEBULA_UPDATES_FEED": ""})
        self.env.start()

    def tearDown(self):
        self.key.stop()
        self.env.stop()

    def installed_by(self, base):
        Path(self.tmp, "app", "install.json").write_text(json.dumps({"base": base, "version": base, "extras": "lookup"}))

    def look(self, body):
        u = U.Updates(SimpleNamespace(cfg=cfg_in(self.tmp), ledger=None, admission=None))
        sig = signed(body)
        with mock.patch.object(U.urllib.request, "urlopen", side_effect=lambda req, timeout=0: Resp(sig if req.full_url.endswith(".sig") else body)), \
             mock.patch.object(U, "install_dir", return_value=Path(self.tmp)), \
             mock.patch.object(U, "downloadable", return_value=True):
            return u, u.check()

    def test_an_install_from_a_base_installer_gets_the_small_update(self):
        self.installed_by("0.1.4")
        u, v = self.look(app_feed())
        self.assertEqual((v["can_install"], v["kind"], v["size_mb"]), (True, "small", 23))
        started = []
        with mock.patch.object(U, "install_dir", return_value=Path(self.tmp)), \
             mock.patch.object(U.threading, "Thread", side_effect=lambda target, args, **k: SimpleNamespace(start=lambda: started.append(args))):
            u.install()
        f, app = started[0]
        self.assertEqual((f["sha256"], app["sha256"]), ("ab" * 32, "cd" * 32))         # the installer kept as the fallback

    def test_any_other_install_gets_the_whole_installer(self):
        for base in ("0.1.3.1", None):
            if base:
                self.installed_by(base)
            else:
                os.remove(os.path.join(self.tmp, "app", "install.json"))           # an install from before 0.1.4: none
            u, v = self.look(app_feed())
            self.assertEqual((v["can_install"], v["kind"], v["size_mb"]), (True, "full", None), base)

    def test_the_small_update_runs_the_app_helper_and_the_installer_does_not(self):
        self.installed_by("0.1.4")
        u = U.Updates(SimpleNamespace(cfg=cfg_in(self.tmp), ledger=None, admission=None))
        with mock.patch.object(U, "fetch_checked", side_effect=lambda url, dest, sha: dest), \
             mock.patch.object(U, "busy", return_value=False), \
             mock.patch.object(U, "install_dir", return_value=Path(self.tmp)), \
             mock.patch.object(U, "run_installer", side_effect=AssertionError("not the installer")), \
             mock.patch.object(U, "run_app_update") as app, mock.patch.object(U.threading, "Timer"):
            u._install({"url": "https://example.invalid/nebula.run", "sha256": "ab" * 32},
                       {"url": "https://example.invalid/nebula-app.tar.gz", "sha256": "cd" * 32})
        self.assertEqual(app.call_args[0][0].name, "nebula-app.tar.gz")
        self.assertEqual(u.installing, "installing")

    def test_a_small_update_without_its_bases_or_https_is_ignored(self):
        for body in (app_feed(bases="0.1.4"), app_feed(app_url="http://example.invalid/a.tar.gz")):
            self.assertIsNone(U.read_feed(body, signed(body), PUB))
        bad = json.loads(app_feed()); bad["files"][U.platform_key() or "linux-x86_64"]["app"]["bases"] = ["latest"]
        body = json.dumps(bad).encode()
        self.assertIsNone(U.read_feed(body, signed(body), PUB))

    @unittest.skipIf(os.name == "nt", "the venv layout below is Linux's")
    def test_the_apply_step_puts_the_new_code_in_place_and_records_it(self):
        """APPLY_APP for real, in a venv of its own: the wheel installed, install.json and VERSION moved on."""
        import subprocess
        import sys
        import tarfile
        import zipfile
        home = Path(self.tmp)
        if subprocess.run([sys.executable, "-m", "venv", str(home / "python")], capture_output=True).returncode != 0:
            self.skipTest("no venv with pip here")
        py = home / "python" / "bin" / "python3"
        self.installed_by("0.1.4")
        work = Path(tempfile.mkdtemp())
        whl = work / "nebula_eh-0.1.4.1-py3-none-any.whl"                              # a stand-in Nebula
        with zipfile.ZipFile(whl, "w") as z:
            z.writestr("nebula_stand_in/__init__.py", "VERSION = '0.1.4.1'\n")
            z.writestr("nebula_eh-0.1.4.1.dist-info/METADATA", "Metadata-Version: 2.1\nName: nebula-eh\nVersion: 0.1.4.1\nProvides-Extra: lookup\n")
            z.writestr("nebula_eh-0.1.4.1.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            z.writestr("nebula_eh-0.1.4.1.dist-info/RECORD", "")
        (work / "constraints.txt").write_text("")
        arc = work / "nebula-eh-0.1.4.1-app-linux-x86_64.tar.gz"
        with tarfile.open(arc, "w:gz") as t:
            t.add(whl, arcname=whl.name)
            t.add(work / "constraints.txt", arcname="constraints.txt")
        (work / "apply.py").write_text(U.APPLY_APP)
        r = subprocess.run([str(py), str(work / "apply.py"), str(arc), str(home)], capture_output=True, text=True, timeout=300,
                           env={**os.environ, "PIP_NO_INDEX": "1"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(json.loads((home / "app" / "install.json").read_text())["version"], "0.1.4.1")
        self.assertEqual(json.loads((home / "app" / "install.json").read_text())["base"], "0.1.4")       # its base stays
        self.assertEqual((home / "app" / "VERSION").read_text(), "0.1.4.1\n")
        out = subprocess.run([str(py), "-c", "import nebula_stand_in as n; print(n.VERSION)"], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "0.1.4.1")


class SignFeedAppTest(unittest.TestCase):
    def test_the_signer_writes_the_small_update_with_its_bases(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "greenfield"))
        import sign_feed
        tmp = Path(tempfile.mkdtemp())
        (tmp / "n.run").write_bytes(b"run")
        (tmp / "a.tar.gz").write_bytes(b"app")
        body = json.loads(sign_feed.build("0.1.4.1", "fixes", ["linux-x86_64=%s=https://x.invalid/n.run" % (tmp / "n.run")], None,
                                          ["linux-x86_64=%s=https://x.invalid/a.tar.gz=0.1.4, v0.1.4" % (tmp / "a.tar.gz")]))
        app = body["files"]["linux-x86_64"]["app"]
        self.assertEqual((app["bases"], app["bytes"], app["sha256"]), (["0.1.4", "0.1.4"], 3, hashlib.sha256(b"app").hexdigest()))
        with self.assertRaises(SystemExit):
            sign_feed.build("0.1.4.1", "", [], None, ["linux-x86_64=%s=https://x.invalid/a.tar.gz=0.1.4" % (tmp / "a.tar.gz")])
