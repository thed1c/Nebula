"""Plan step F, external destinations, and the nebula as a pool: a CLI on this machine is a tier that runs a
command; pairing goes both ways, each machine offering its own models; a paired machine escalates through this
one's CLI; what a machine adds later reaches its peers."""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from nebula.commands import CommandEngine, build_command, parse_claude_json, prompt_from_messages
from nebula.tests import WINDOWS, linux_only  # noqa: F401
from nebula.tests import WINDOWS, linux_only  # noqa: F401
from nebula.config import Tier
from nebula.engine import EngineError
from nebula.replay import verify
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_escalation import Rig
from nebula.tests.test_gateway import call
from nebula.tests.test_ui import req as ui_req

FAKE = f"{sys.executable} {Path(__file__).with_name('fake_cli.py')}"


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return json.loads(r.read())


class FormsTest(unittest.TestCase):
    def test_the_three_clis_and_an_own_command(self):
        argv, stdin, how = build_command("claude", None, "fable", "Be terse.", "/p", "/o")
        self.assertEqual((argv[:4], stdin, how), (["claude", "-p", "--output-format", "json"], True, "json"))
        self.assertIn("--system-prompt", argv)
        argv, stdin, how = build_command("codex", None, None, "sys", "/p", "/o")
        self.assertEqual((argv[:2], argv[-3:], stdin, how), (["codex", "exec"], ["-o", "/o", "-"], True, "file"))
        argv, stdin, how = build_command("grok", None, "grok-4", None, "/p", "/o")
        self.assertEqual((argv[:3], stdin, how), (["grok", "--prompt-file", "/p"], False, "stdout"))
        self.assertEqual(argv[argv.index("-m") + 1], "grok-4")
        argv, stdin, how = build_command(None, "mytool --in {prompt_file} --model {model}", "m1", None, "/p", "/o")
        self.assertEqual((argv, stdin, how), (["mytool", "--in", "/p", "--model", "m1"], False, "stdout"))
        self.assertEqual(build_command(None, ["cat"], None, None, "/p", "/o"), (["cat"], True, "stdout"))
        with self.assertRaises(ValueError):
            build_command("gemini", None, None, None, "/p", "/o")

    def test_prompt_and_claude_output(self):
        self.assertEqual(prompt_from_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "one"}]), ("S", "one"))
        system, prompt = prompt_from_messages([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}])
        self.assertIsNone(system)
        self.assertTrue(prompt.endswith("### user\nc\n\n### assistant"))
        text, usage = parse_claude_json(json.dumps({"type": "result", "result": "hi", "usage": {"input_tokens": 10, "output_tokens": 2}, "total_cost_usd": 0.01}))
        self.assertEqual((text, usage["prompt_tokens"], usage["completion_tokens"], usage["cost_usd"]), ("hi", 10, 2, 0.01))
        self.assertEqual(parse_claude_json("plain words\n"), ("plain words", None))


class CommandEngineTest(unittest.TestCase):
    def test_runs_the_command_and_reports_like_an_engine(self):
        with tempfile.TemporaryDirectory() as d:
            eng = CommandEngine(Tier("tool", command=FAKE + " --prompt-file {prompt_file} -o {output_file}"), timeout_s=20, base_dir=d)
            self.assertEqual(eng.get("/health")[0], 200)
            self.assertEqual((eng.props()["total_slots"], eng.n_ctx(), eng.busy(), eng.tokenize("x")), (2, None, 0, None))
            out = eng.chat({"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "hello"}], "tools": [{"type": "function"}]})
            self.assertEqual(out["choices"][0]["message"]["content"], "cli says: hello")
            self.assertTrue(out["usage"]["estimated"])
            chunks = list(eng.chat_stream({"messages": [{"role": "user", "content": "via stdin"}]}))
            self.assertEqual((len(chunks), chunks[0]["choices"][0]["delta"]["content"]), (1, "cli says: via stdin"))
            self.assertTrue(os.path.isdir(os.path.join(d, "nebula-work")))
            with self.assertRaises(EngineError) as cm:
                CommandEngine(Tier("f", command=FAKE + " --fail"), timeout_s=20, base_dir=d).chat({"messages": [{"role": "user", "content": "x"}]})
            self.assertEqual(cm.exception.status, 502)
            self.assertIn("exited 3", cm.exception.message)
            with self.assertRaises(EngineError) as cm:
                CommandEngine(Tier("s", command=FAKE + " --slow 5"), timeout_s=0.5, base_dir=d).chat({"messages": [{"role": "user", "content": "x"}]})
            self.assertEqual(cm.exception.status, 504)
            missing = CommandEngine(Tier("m", command="no-such-binary-xyz"), timeout_s=1, base_dir=d)
            self.assertEqual(missing.get("/health")[0], 503)
            self.assertEqual(Tier("g", cli="grok").address, "cli:grok")


class GatewayCommandTest(unittest.TestCase):
    def test_a_command_tier_is_routed_to_like_any_model(self):
        rig = Rig(extra_tiers=[Tier("tool", command=FAKE)])
        try:
            rig.srv.cfg.base_dir = rig.tmp.name
            node = [n for n in get(rig.base, "/api/registry") if n["name"] == "tool"][0]
            self.assertEqual((node["kind"], node["up"], node["slots"], node["address"]), ("command", True, 2, "cli:" + sys.executable))
            status, _c, body = call(rig.base, "/v1/chat/completions", {"model": "cockpit/tool", "messages": [{"role": "user", "content": "run it"}]}, timeout=30)
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "cli says: run it")
            self.assertEqual(rig.attempts()[-1]["tier"], "tool")
            self.assertEqual(len(rig.a.requests) + len(rig.b.requests), 0)
            # added from the page: a command of one's own; a CLI without a form, or not installed, is refused
            status, _h, raw = ui_req(rig.base, "/api/models", {"name": "own", "kind": "command", "command": FAKE + " --prompt-file {prompt_file}"})
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["node"]["kind"], "command")
            self.assertEqual(ui_req(rig.base, "/api/models", {"name": "x", "kind": "command", "cli": "gemini"})[0], 400)
            self.assertEqual(ui_req(rig.base, "/api/models", {"name": "y", "kind": "command", "command": "no-such-binary-xyz"})[0], 400)
            status, _c, body = call(rig.base, "/v1/chat/completions", {"model": "cockpit/own", "messages": [{"role": "user", "content": "own one"}]}, timeout=30)
            self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "cli says: own one")
            ok, lines = verify(rig.path)
            self.assertTrue(ok, "\n".join(lines))
        finally:
            rig.stop()


FAKE_OPENSHELL = str(Path(__file__).with_name("fake_openshell.py"))


@linux_only
class SandboxedCommandTest(unittest.TestCase):
    """Step O1 on the fake openshell: the same call inside a sandbox, created once and kept; down with the reason
    when openshell is missing; step O3's token opens the model API only."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["FAKE_OPENSHELL_HOME"] = os.path.join(self.tmp.name, "openshell")

    def tearDown(self):
        os.environ.pop("FAKE_OPENSHELL_HOME", None)
        self.tmp.cleanup()

    def calls(self):
        with open(os.path.join(os.environ["FAKE_OPENSHELL_HOME"], "calls.log"), encoding="utf-8") as f:
            return [json.loads(ln) for ln in f]

    def test_runs_inside_the_sandbox(self):
        tier = Tier("tool", command=FAKE + " --prompt-file {prompt_file} -o {output_file}", sandbox={"image": "img:1", "openshell": FAKE_OPENSHELL, "env": {"A": "1"}})
        eng = CommandEngine(tier, timeout_s=30, base_dir=self.tmp.name)
        self.assertEqual(eng.get("/health")[0], 200)                    # openshell is there: the sandbox is made on the first call
        self.assertEqual(eng.sandbox.status()["exists"], False)
        out = eng.chat({"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "hello"}]})
        self.assertEqual(out["choices"][0]["message"]["content"], "cli says: hello")
        creates = [c for c in self.calls() if c[:2] == ["sandbox", "create"]]
        self.assertEqual(len(creates), 1)
        self.assertEqual(creates[0][2:9], ["--name", "nebula-tool", "--from", "img:1", "--detach", "--env", "A=1"])
        self.assertEqual(creates[0][-3:], ["--", "sleep", "infinity"])
        plain = CommandEngine(Tier("plain", command=FAKE, sandbox={"image": "img:1", "openshell": FAKE_OPENSHELL}), timeout_s=30, base_dir=self.tmp.name)
        out = plain.chat({"messages": [{"role": "user", "content": "via stdin"}]})
        self.assertEqual(out["choices"][0]["message"]["content"], "cli says: via stdin")
        self.assertEqual(eng.chat({"messages": [{"role": "user", "content": "again"}]})["choices"][0]["message"]["content"], "cli says: again")
        self.assertEqual(len([c for c in self.calls() if c[:2] == ["sandbox", "create"]]), 2)      # one per tier, kept
        self.assertEqual(eng.get("/health")[0], 200)
        self.assertEqual(eng.props()["sandbox"], {"name": "nebula-tool", "image": "img:1"})
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "cockpit-work")))           # nothing ran on the host
        with self.assertRaises(EngineError) as cm:
            CommandEngine(Tier("f", command=FAKE + " --fail", sandbox={"image": "img:1", "openshell": FAKE_OPENSHELL}), timeout_s=30, base_dir=self.tmp.name).chat({"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(cm.exception.status, 502)
        self.assertIn("in nebula-f) exited 3", cm.exception.message)
        # no openshell: down with the reason, never run bare
        missing = CommandEngine(Tier("m", command=FAKE, sandbox={"image": "img:1", "openshell": "no-such-openshell"}), timeout_s=5, base_dir=self.tmp.name)
        self.assertEqual((missing.get("/health")[0], json.loads(missing.get("/health")[2])["status"]), (503, "openshell is not installed on this machine"))
        with self.assertRaises(EngineError) as cm:
            missing.chat({"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(cm.exception.status, 503)

    def test_gateway_and_the_sandbox_token(self):
        rig = Rig(extra_tiers=[Tier("boxed", command=FAKE, sandbox={"image": "img:1", "openshell": FAKE_OPENSHELL, "model_via_nebula": True})])
        try:
            rig.srv.cfg.base_dir = self.tmp.name
            from nebula.server import sandbox_tokens
            sandbox_tokens(rig.srv)
            env = rig.srv.engines["boxed"].sandbox.nebula_env
            self.assertTrue(env["OPENAI_BASE_URL"].endswith(f":{rig.srv.server_port}/v1"))
            self.assertNotIn("127.0.0.1", env["OPENAI_BASE_URL"])            # the LAN address, else OpenShell's host name
            token = env["OPENAI_API_KEY"]
            self.assertEqual(open(os.path.join(self.tmp.name, "nebula-keys", "sandbox-boxed.key")).read().strip(), token)
            status, _c, body = call(rig.base, "/v1/chat/completions", {"model": "cockpit/boxed", "messages": [{"role": "user", "content": "in a box"}]}, timeout=30)
            self.assertEqual((status, json.loads(body)["choices"][0]["message"]["content"]), (200, "cli says: in a box"))
            node = [n for n in get(rig.base, "/api/registry") if n["name"] == "boxed"][0]
            self.assertEqual((node["kind"], node["up"], node["sandbox"]["name"]), ("command", True, "nebula-boxed"))
            # the token opens the model API from beyond the machine, and nothing else
            rig.srv.cfg.lan_auth_loopback = True
            try:
                def with_token(path, body=None):
                    req = urllib.request.Request(rig.base + path, data=None if body is None else json.dumps(body).encode(),
                                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(req, timeout=10) as r:
                            return r.status
                    except urllib.error.HTTPError as e:
                        return e.code
                self.assertEqual(with_token("/v1/models"), 200)
                self.assertEqual(with_token("/v1/chat/completions", {"model": "cockpit/fast", "messages": [{"role": "user", "content": "hi"}]}), 200)
                self.assertEqual(rig.attempts()[-1]["tier"], "fast")
                self.assertEqual(with_token("/api/lan"), 403)
                self.assertEqual(with_token("/api/registry"), 403)
            finally:
                rig.srv.cfg.lan_auth_loopback = False
        finally:
            rig.stop()

    def test_with_no_lan_address_the_sandbox_reaches_nebula_by_openshells_host_name(self):
        from unittest import mock
        from nebula import lan
        from nebula.server import SANDBOX_HOST, sandbox_tokens
        rig = Rig(extra_tiers=[Tier("boxed", command=FAKE, sandbox={"image": "img:1", "openshell": FAKE_OPENSHELL, "model_via_nebula": True})])
        try:
            rig.srv.cfg.base_dir = self.tmp.name
            with mock.patch.object(lan, "own_addresses", return_value=["127.0.1.1"]):
                sandbox_tokens(rig.srv)
            sb = rig.srv.engines["boxed"].sandbox
            self.assertEqual(sb.nebula_env["OPENAI_BASE_URL"], f"http://{SANDBOX_HOST}:{rig.srv.server_port}/v1")
            self.assertEqual(sb.nebula_endpoint, (SANDBOX_HOST, rig.srv.server_port))   # the one host its policy allows
        finally:
            rig.stop()


@linux_only
class EgressPolicyTest(unittest.TestCase):
    """Step O2 on the fake openshell: the policy written per tier, the hosts the owner allows, the asks read from the
    sandbox's denials, approve and reject."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["FAKE_OPENSHELL_HOME"] = os.path.join(self.tmp.name, "openshell")

    def tearDown(self):
        os.environ.pop("FAKE_OPENSHELL_HOME", None)
        self.tmp.cleanup()

    def calls(self):
        with open(os.path.join(os.environ["FAKE_OPENSHELL_HOME"], "calls.log"), encoding="utf-8") as f:
            return [json.loads(ln) for ln in f]

    def test_policy_text(self):
        from nebula.sandbox import policy_yaml, parse_host
        text = policy_yaml("grok", ["pypi.org", "github.com:443:read-write"], ["/usr/bin/grok"], ("192.0.2.9", 8090))
        self.assertTrue(text.startswith("# written by Nebula"))
        for needle in ("version: 1", "  vendor:", "host: api.x.ai", "  nebula:", "host: 192.0.2.9", "port: 8090", "  owner:",
                       "host: pypi.org", "access: read-only", "host: github.com", "access: read-write", "- path: /usr/bin/grok"):
            self.assertIn(needle, text)
        self.assertNotIn("api.anthropic.com", text)
        empty = policy_yaml(None, [], ["/usr/bin/bash"], None)
        self.assertIn("filesystem_policy:", empty)                          # the defaults, written out: see sandbox.py
        self.assertNotIn("network_policies", empty)
        self.assertEqual(parse_host("docs.example:8443:read-write"), {"host": "docs.example", "port": 8443, "access": "read-write", "path": None})
        self.assertIsNone(parse_host(""))

    def test_hosts_and_asks_through_the_api(self):
        rig = Rig()
        try:
            rig.srv.cfg.path = os.path.join(self.tmp.name, "nebula.json")
            rig.srv.cfg.base_dir = self.tmp.name
            status, _h, raw = ui_req(rig.base, "/api/models", {"name": "boxed", "kind": "command", "command": FAKE,
                                                                "sandbox": {"openshell": FAKE_OPENSHELL, "image": "img:1", "model_via_nebula": True}, "hosts": "pypi.org, github.com:443:read-write"})
            self.assertEqual(status, 200, raw)
            self.assertEqual(ui_req(rig.base, "/api/models", {"name": "bad", "kind": "command", "command": FAKE, "sandbox": {"openshell": FAKE_OPENSHELL}, "hosts": [":"]})[0], 400)
            # the first call creates the sandbox with the written policy
            status, _c, body = call(rig.base, "/v1/chat/completions", {"model": "cockpit/boxed", "messages": [{"role": "user", "content": "go"}]}, timeout=30)
            self.assertEqual(status, 200, body)
            create = [c for c in self.calls() if c[:2] == ["sandbox", "create"]][0]
            policy = create[create.index("--policy") + 1]
            self.assertTrue(policy.endswith("cockpit-sandbox/nebula-boxed.policy.yaml"))
            text = open(policy, encoding="utf-8").read()
            self.assertIn("host: pypi.org", text)
            self.assertIn("host: github.com", text)
            self.assertIn("  nebula:", text)                                   # this Nebula, for the CLI's own model calls
            view = get(rig.base, "/api/sandboxes")
            self.assertEqual((view[0]["tier"], view[0]["running"], [h["host"] for h in view[0]["hosts"]], view[0]["asks"]), ("boxed", True, ["pypi.org", "github.com"], []))
            # denials in the sandbox's log become asks; approving one allows the host and sets the policy live
            with open(os.path.join(os.environ["FAKE_OPENSHELL_HOME"], "nebula-boxed", ".log"), "w", encoding="utf-8") as f:
                f.write("[1792163323.447] [sandbox] [OCSF ] [ocsf] NET:OPEN [MED] DENIED /usr/bin/bash(0) -> files.pythonhosted.org:443 [reason:transparent_tcp_policy_denied]\n"
                        "[1792163324.000] [sandbox] [OCSF ] [ocsf] NET:OPEN [MED] DENIED /usr/bin/bash(0) -> files.pythonhosted.org:443 [reason:transparent_tcp_policy_denied]\n"
                        "[1792163325.000] [sandbox] [OCSF ] [ocsf] NET:OPEN [MED] DENIED /usr/bin/curl(12) -> evil.example:8443 [reason:transparent_tcp_policy_denied]\n"
                        "[1792163326.000] [sandbox] [OCSF ] [ocsf] NET:OPEN [MED] DENIED /usr/bin/bash(0) -> pypi.org:443 [reason:x]\n")
            asks = get(rig.base, "/api/sandboxes")[0]["asks"]
            self.assertEqual([(a["host"], a["port"], a["binary"], a["count"]) for a in asks],
                             [("files.pythonhosted.org", 443, "/usr/bin/bash", 2), ("evil.example", 8443, "/usr/bin/curl", 1)])   # pypi.org is allowed already
            status, _h, raw = ui_req(rig.base, f"/api/sandboxes/boxed/asks", {"action": "approve", "host": "files.pythonhosted.org", "port": 443})
            self.assertEqual(status, 200, raw)
            js = json.loads(raw)
            self.assertEqual((js["allowed"], js["applied"]), ("files.pythonhosted.org:443:read-only", True))
            self.assertIn(["policy", "set", "nebula-boxed", "--policy", policy, "--wait"], self.calls())
            self.assertIn("host: files.pythonhosted.org", open(policy, encoding="utf-8").read())
            saved = json.load(open(rig.srv.cfg.path, encoding="utf-8"))
            boxed = [t for t in saved["tiers"] if t["name"] == "boxed"][0]
            self.assertEqual(boxed["sandbox"]["hosts"], ["pypi.org", "github.com:443:read-write", "files.pythonhosted.org:443:read-only"])
            status, _h, raw = ui_req(rig.base, f"/api/sandboxes/boxed/asks", {"action": "reject", "host": "evil.example", "port": 8443})
            self.assertEqual(status, 200, raw)
            view = get(rig.base, "/api/sandboxes")[0]
            self.assertEqual((view["asks"], view["rejected"]), ([], ["evil.example:8443"]))
            # a host removed by hand, the policy set again; the sandbox stopped and started
            self.assertEqual(json.loads(ui_req(rig.base, "/api/sandboxes/boxed/hosts", {"remove": "github.com:443"})[2])["applied"], True)
            self.assertNotIn("host: github.com", open(policy, encoding="utf-8").read())
            self.assertEqual(json.loads(ui_req(rig.base, "/api/sandboxes/boxed/stop", {})[2])["running"], False)
            self.assertEqual(json.loads(ui_req(rig.base, "/api/sandboxes/boxed/start", {})[2])["running"], True)
            self.assertEqual(ui_req(rig.base, "/api/sandboxes/fast/policy", {})[0], 404)
        finally:
            rig.stop()


@unittest.skipUnless(shutil.which("openshell") and os.environ.get("NEBULA_OPENSHELL_LIVE"), "needs openshell, Docker and NEBULA_OPENSHELL_LIVE=1")
class LiveOpenShellTest(unittest.TestCase):
    """On a real OpenShell gateway with Docker (as run here 2026-10-16, openshell 0.1.4): the sandbox is kept between
    calls, the file form works, egress is blocked, and (step O3) the sandbox reaches Nebula on the LAN address through
    a policy with its own token, and not without it."""

    def test_cat_inside_a_real_sandbox(self):
        with tempfile.TemporaryDirectory() as d:
            eng = CommandEngine(Tier("live-cat", command="cat", sandbox={"image": "ubuntu:24.04"}), timeout_s=300, base_dir=d)
            try:
                out = eng.chat({"messages": [{"role": "user", "content": "echoed through openshell"}]})
                self.assertEqual(out["choices"][0]["message"]["content"], "echoed through openshell")
                st = eng.sandbox.status()
                self.assertEqual((st["exists"], st["running"]), (True, True), st)
                out = eng.chat({"messages": [{"role": "user", "content": "second call, same sandbox"}]})      # kept: not created again
                self.assertEqual(out["choices"][0]["message"]["content"], "second call, same sandbox")
                filer = CommandEngine(Tier("live-file", command="sh -c 'tr a-z A-Z < {prompt_file} > {output_file}'", sandbox={"image": "ubuntu:24.04"}), timeout_s=300, base_dir=d)
                try:
                    self.assertEqual(filer.chat({"messages": [{"role": "user", "content": "shout"}]})["choices"][0]["message"]["content"], "SHOUT")
                finally:
                    filer.sandbox.delete()
                _rc, o, e = eng.sandbox.exec(["bash", "-c", "exec 3<>/dev/tcp/1.1.1.1/443 && echo CONNECTED || echo BLOCKED"], timeout=60)
                self.assertIn("BLOCKED", (o + e).decode())
                # step O2 live: a refused connection by name is an ask; approving it allows the host and sets the policy
                _rc, o, e = eng.sandbox.exec(["bash", "-c", "exec 3<>/dev/tcp/pypi.org/443 && echo CONNECTED || echo BLOCKED"], timeout=60)
                self.assertIn("BLOCKED", (o + e).decode())
                asks = []
                for _ in range(40):                                              # the denial reaches the log a moment later
                    asks = eng.sandbox.asks()
                    if ("pypi.org", 443) in [(a["host"], a["port"]) for a in asks]:
                        break
                    time.sleep(0.25)
                self.assertIn(("pypi.org", 443), [(a["host"], a["port"]) for a in asks], asks)
                eng.sandbox.spec.setdefault("hosts", []).append("pypi.org:443:read-only")
                applied = eng.sandbox.apply_policy()
                self.assertTrue(applied["applied"], applied)
                self.assertNotIn(("pypi.org", 443), [(a["host"], a["port"]) for a in eng.sandbox.asks()])   # the 1.1.1.1 ask stays
                _rc, o, e = eng.sandbox.exec(["bash", "-c", "exec 3<>/dev/tcp/pypi.org/443 && echo CONNECTED || echo BLOCKED"], timeout=60)
                self.assertIn("CONNECTED", (o + e).decode(), (o + e).decode()[-300:])
            finally:
                self.assertTrue(eng.sandbox.delete())

    def test_the_sandbox_calls_home_with_its_token(self):
        import threading
        from nebula import lan
        from nebula.config import Config
        from nebula.server import make_server
        port = 18199
        ip = lan.own_addresses()[0]
        fake = FakeEngine("fastfake")
        with tempfile.TemporaryDirectory() as d:
            policy = os.path.join(d, "nebula-policy.yaml")
            with open(policy, "w", encoding="utf-8") as f:
                f.write(f"version: 1\nnetwork_policies:\n  nebula:\n    endpoints:\n      - host: {ip}\n        port: {port}\n"
                        "        protocol: rest\n        access: read-write\n        enforcement: enforce\n    binaries:\n      - path: /usr/bin/bash\n")
            home = ("bash -c 'h=${NEBULA_URL#http://}; host=${h%%:*}; port=${h##*:}; exec 3<>/dev/tcp/$host/$port; "
                    "printf \"GET /v1/models HTTP/1.0\\r\\nAuthorization: Bearer $OPENAI_API_KEY\\r\\nHost: $h\\r\\n\\r\\n\" >&3; cat <&3'")
            cfg = Config(host="0.0.0.0", port=port, ledger=os.path.join(d, "l.sqlite3"), monitor=False,
                         tiers=[Tier("fast", fake.url, thinking=False),
                                Tier("boxed", command=home, sandbox={"image": "ubuntu:24.04", "model_via_nebula": True, "policy": policy})])
            cfg.base_dir = d
            srv = make_server(cfg)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                status, _c, body = call(f"http://127.0.0.1:{port}", "/v1/chat/completions", {"model": "cockpit/boxed", "messages": [{"role": "user", "content": "call home"}]}, timeout=300)
                self.assertEqual(status, 200, body)
                text = json.loads(body)["choices"][0]["message"]["content"]
                self.assertTrue(text.startswith("HTTP/1.0 200"), text[:200])
                self.assertIn('"id": "fast"', text)
                sb = srv.engines["boxed"].sandbox
                bare = home[len("bash -c '"):-1].replace("Authorization: Bearer $OPENAI_API_KEY\\r\\n", "")
                _rc, o, e = sb.exec(["bash", "-c", bare], env={"NEBULA_URL": sb.nebula_env["NEBULA_URL"]}, timeout=60)
                self.assertTrue((o + e).decode().startswith("HTTP/1.0 403"), (o + e).decode()[:200])
                self.assertTrue(sb.delete())
            finally:
                srv.shutdown()
                srv.server_close()
                fake.stop()


if __name__ == "__main__":
    unittest.main()
