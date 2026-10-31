"""Why a download could not happen, in plain words (nebula/netwhy.py), and a resumed download that keeps moving is not
given up on (nebula/hf.py)."""
import http.client
import os
import socket
import ssl
import tempfile
import unittest
import urllib.error
from unittest import mock

from nebula import hf, netwhy


class CauseTest(unittest.TestCase):
    def test_each_kind_of_failure_is_named(self):
        cases = [
            (urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed")), "does not resolve"),
            (urllib.error.URLError(ConnectionRefusedError(10061, "No connection could be made because the target machine actively refused it")), "refused"),
            (urllib.error.URLError(ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate")), "certificate is not trusted"),
            (urllib.error.URLError(TimeoutError("timed out")), "did not answer in time"),
            (ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host"), "cut partway"),
            (http.client.IncompleteRead(b"x", 10), "cut partway"),
            (urllib.error.HTTPError("https://huggingface.co/x", 403, "Forbidden", {}, None), "refused it (403"),
            (urllib.error.HTTPError("https://huggingface.co/x", 503, "busy", {}, None), "trouble (503)"),
        ]
        for e, want in cases:
            self.assertIn(want, netwhy.cause(e), repr(e))

    def test_a_cause_wrapped_by_another_library_is_found(self):
        class ConnectError(Exception):
            pass
        try:
            try:
                raise socket.gaierror(-2, "Name or service not known")
            except socket.gaierror as inner:
                raise ConnectError("[Errno -2] Name or service not known") from inner
        except ConnectError as e:
            self.assertIn("does not resolve", netwhy.cause(e))

    def test_a_probe_says_why_not(self):
        with mock.patch.object(netwhy.urllib.request, "urlopen", side_effect=urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed"))):
            self.assertIn("does not resolve", netwhy.probe("https://huggingface.co/"))

    def test_only_windows_and_macos_use_the_systems_certificates(self):
        with mock.patch.object(netwhy.sys, "platform", "linux"):
            self.assertIsNone(netwhy.tls_context())


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pulls = hf.Pulls(self.tmp)
        self.sleep = mock.patch.object(hf.time, "sleep")
        self.sleep.start()

    def tearDown(self):
        self.sleep.stop()

    def _pull(self):
        p = {"repo": "o/r", "revision": "main", "cancel": False}
        f = {"path": "m.gguf", "target": os.path.join(self.tmp, "o", "r", "m.gguf"), "bytes": 0, "total": 0, "sha256": None,
             "state": "waiting", "error": None}
        return p, f

    def test_never_one_byte_says_the_hub_cannot_be_reached_and_why(self):
        p, f = self._pull()
        err = urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed"))
        with mock.patch.object(hf.Pulls, "_fetch", side_effect=err) as fetch:
            with self.assertRaises(hf.HfError) as c:
                self.pulls._one(p, f, None)
        self.assertEqual(fetch.call_count, hf.RETRIES + 1)
        self.assertIn("Hugging Face cannot be reached from here: the server's name does not resolve", str(c.exception))

    def test_a_link_that_keeps_moving_is_not_given_up_on(self):
        p, f = self._pull()
        calls = {"n": 0}

        def fetch(pulls, p_, f_, url, part, token):
            calls["n"] += 1
            with open(part, "ab") as out:
                out.write(b"x")                           # a little more each time, then the link drops
            if calls["n"] < hf.RETRIES * 3:
                raise ConnectionResetError(10054, "forcibly closed")
            return False
        with mock.patch.object(hf.Pulls, "_fetch", fetch):
            self.pulls._one(p, f, None)
        self.assertEqual(calls["n"], hf.RETRIES * 3)
        self.assertEqual(f["state"], "done")
        self.assertEqual(open(f["target"], "rb").read(), b"x" * (hf.RETRIES * 3))


class WindowIconTest(unittest.TestCase):
    def test_windows_gets_the_ico(self):
        from nebula.app import window
        self.assertTrue(os.path.isfile(window.ICON_ICO))
        with open(window.ICON_ICO, "rb") as f:
            self.assertEqual(f.read(4), b"\x00\x00\x01\x00")    # an .ico, which System.Drawing.Icon reads
        fake = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"webview": fake}), mock.patch.object(window.os, "name", "nt"), \
             mock.patch.object(window.sys, "platform", "win32"), mock.patch.object(window.signal, "signal"):
            window._pywebview("http://127.0.0.1:1/", None, "Nebula", 800, 600)
        self.assertEqual(fake.start.call_args.kwargs["icon"], window.ICON_ICO)
