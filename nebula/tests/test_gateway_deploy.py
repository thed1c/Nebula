"""The gateway (nebula/gateway.py): the relay config Nebula writes (members only, TLS with a hostname, plain HTTP
without), and deploying it here through Docker (a fake docker records what it was asked). The real relay across two
networks is greenfield/gatewaynet.sh."""
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from nebula import gateway as G
from nebula.tests import WINDOWS, linux_only  # noqa: F401


class ConfigTest(unittest.TestCase):
    def test_members_only_and_tls(self):
        c = G.config(["aa", "bb", "aa"])
        self.assertIn('access.allowlist = ["aa", "bb"]', c)
        self.assertIn('http_bind_addr = "0.0.0.0:3340"', c)
        self.assertNotIn("[tls]", c)
        c = G.config(["aa"], hostname="gw.example.org", email="me@example.org")
        self.assertIn('cert_mode = "LetsEncrypt"', c)
        self.assertIn('hostname = ["gw.example.org"]', c)
        self.assertLess(c.index("access.allowlist"), c.index("[tls]"))        # a top-level key, before the table
        self.assertEqual(G.url_of("gw.example.org"), "https://gw.example.org")


class DeployTest(unittest.TestCase):
    @linux_only
    def test_deploy_here_with_docker(self):
        tmp = tempfile.mkdtemp()
        log = os.path.join(tmp, "docker.log")
        fake = os.path.join(tmp, "docker")
        Path(fake).write_text(f"#!{sys.executable}\nimport json, sys\nopen({log!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
                              "sys.exit(1 if sys.argv[1:3] == ['image', 'inspect'] else 0)\n")
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
        os.environ["NEBULA_DOCKER"] = fake
        try:
            out = G.deploy(os.path.join(tmp, "conf"), ["k1", "k2"])
        finally:
            del os.environ["NEBULA_DOCKER"]
        calls = [json.loads(x) for x in open(log).read().splitlines()]
        self.assertEqual([c[0] for c in calls], ["image", "build", "rm", "run"])
        run = calls[-1]
        self.assertIn("-p", run)
        self.assertIn("3340:3340", run)
        self.assertEqual(out["url"], "http://127.0.0.1:3340")
        self.assertIn('"k1", "k2"', open(os.path.join(tmp, "conf", "relay.toml")).read())


if __name__ == "__main__":
    unittest.main()
