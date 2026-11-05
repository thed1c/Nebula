"""Syslog (nebula/syslogfwd.py): off by default; when on, Nebula's warnings and errors and its security log go to the
syslog server as RFC 5424 lines, over UDP or TCP (octet-counted); the page turns it on, checks it and sends a test."""
import json
import logging
import re
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

from nebula import syslogfwd
from nebula.tests.test_escalation import Rig

LINE = re.compile(r"^<(\d+)>1 \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z \S+ nebula \d+ (\w+) - (.*)$")


class Udp:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(("127.0.0.1", 0))
        self.s.settimeout(5)
        self.port = self.s.getsockname()[1]

    def get(self) -> str:
        return self.s.recv(4096).decode()


class Tcp:
    def __init__(self):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.buf, self.lines = b"", []
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        c, _ = self.srv.accept()
        while True:
            d = c.recv(4096)
            if not d:
                return
            self.buf += d
            while b" " in self.buf:
                n, rest = self.buf.split(b" ", 1)
                if len(rest) < int(n):
                    break
                self.lines.append(rest[:int(n)].decode())
                self.buf = rest[int(n):]

    def get(self) -> str:
        for _ in range(100):
            if self.lines:
                return self.lines.pop(0)
            time.sleep(0.05)
        raise AssertionError("no line came")


class ForwarderTest(unittest.TestCase):
    def fw(self, **s):
        cfg = SimpleNamespace(raw={"syslog": {"enabled": True, "host": "127.0.0.1", **s}})
        seclog = SimpleNamespace(listeners=[])
        f = syslogfwd.Forwarder(cfg, seclog)
        self.addCleanup(f.close)
        return f, seclog

    def test_off_by_default_sends_nothing(self):
        f = syslogfwd.Forwarder(SimpleNamespace(raw={}), None)
        self.addCleanup(f.close)
        self.assertFalse(f.view()["enabled"])
        self.assertFalse(f.attached)

    def test_warnings_errors_and_security_over_udp(self):
        srv = Udp()
        f, seclog = self.fw(port=srv.port, protocol="udp", facility="local3")
        logging.getLogger("nebula.teamwork").warning("a work item failed: %s", "boom")
        m = LINE.match(srv.get())
        self.assertEqual((int(m.group(1)), m.group(2), m.group(3)), (19 * 8 + 4, "log", "nebula.teamwork: a work item failed: boom"))
        logging.getLogger("cockpit").info("a request")                  # below the level: not sent
        logging.getLogger("nebula.syslog").warning("its own")            # never its own: no loop
        for listener in seclog.listeners:
            listener({"kind": "signin.refused", "severity": "warn", "who": "192.0.2.7", "what": "pin", "fresh": True})
        m = LINE.match(srv.get())
        self.assertEqual((int(m.group(1)), m.group(2), m.group(3)), (19 * 8 + 4, "security", "security: signin.refused who=192.0.2.7 what=pin"))

    def test_errors_only_and_a_multi_line_message_is_one_line(self):
        srv = Udp()
        f, _ = self.fw(port=srv.port, level="error", security=False)
        logging.getLogger("nebula.halo").warning("not this")
        logging.getLogger("nebula.halo").error("line one\nline two")
        m = LINE.match(srv.get())
        self.assertEqual(m.group(3), "nebula.halo: line one line two")

    def test_tcp_is_octet_counted_and_the_test_line(self):
        srv = Tcp()
        f, _ = self.fw(port=srv.port, protocol="tcp")
        self.assertEqual(f.test(), {"ok": True})
        self.assertEqual(LINE.match(srv.get()).group(2), "test")

    def test_tls_checks_the_servers_certificate(self):
        import shutil
        import ssl
        import subprocess
        import tempfile
        if not shutil.which("openssl"):
            self.skipTest("no openssl to make a certificate")
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=localhost", "-days", "1",
                        "-keyout", f"{d}/k.pem", "-out", f"{d}/c.pem"], check=True, capture_output=True)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(f"{d}/c.pem", f"{d}/k.pem")
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        self.addCleanup(srv.close)

        def serve():
            try:
                c, _ = srv.accept()
                ctx.wrap_socket(c, server_side=True).recv(10)
            except (OSError, ssl.SSLError):
                pass
        threading.Thread(target=serve, daemon=True).start()
        f, _ = self.fw(port=srv.getsockname()[1], protocol="tls")
        with self.assertRaises(syslogfwd.SyslogError):            # a self-signed server: refused, nothing sent
            f.test()
        self.assertEqual((f.sent, f.failed), (0, 1))
        self.assertIn("CERTIFICATE_VERIFY_FAILED", f.last_error)

    def test_the_settings_are_checked(self):
        cur = syslogfwd.settings(SimpleNamespace(raw={}))
        for bad in ({"enabled": True}, {"port": 0}, {"port": "x"}, {"protocol": "http"}, {"host": "a b"}, {"level": "debug"}):
            with self.assertRaises(syslogfwd.SyslogError, msg=bad):
                syslogfwd.check(bad, cur)
        self.assertEqual(syslogfwd.check({"enabled": True, "host": "logs.lan", "port": 6514, "protocol": "tls"}, cur)["protocol"], "tls")
        self.assertEqual(syslogfwd.settings(SimpleNamespace(raw={"syslog": {"port": "x", "protocol": "?"}}))["port"], 514)


class PageTest(unittest.TestCase):
    def test_the_page_turns_it_on_and_sends_a_test(self):
        rig = Rig()
        self.addCleanup(rig.stop)
        srv = Udp()

        def post(path, body):
            r = urllib.request.Request(rig.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=10) as resp:
                return json.loads(resp.read())
        with urllib.request.urlopen(rig.base + "/api/syslog", timeout=10) as r:
            self.assertFalse(json.loads(r.read())["enabled"])
        v = post("/api/syslog", {"enabled": True, "host": "127.0.0.1", "port": srv.port, "protocol": "udp"})
        self.assertTrue(v["enabled"])
        self.assertEqual([e["what"] for e in rig.srv.seclog.read(kinds=["syslog.settings"])], [None])   # who changed it
        m = LINE.match(srv.get())                                   # its own change goes out first, as a security line
        self.assertEqual((m.group(2), m.group(3)), ("security", "security: syslog.settings who=owner"))
        post("/api/syslog/test", {})
        self.assertEqual(LINE.match(srv.get()).group(2), "test")
        try:
            post("/api/syslog", {"enabled": True, "host": ""})
            self.fail("an empty server was accepted")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)


if __name__ == "__main__":
    unittest.main()
