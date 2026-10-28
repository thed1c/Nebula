"""OpenShell from Nebula (nebula/openshell.py), on the fake `openshell`: the panel (read only), the provider profile that
reaches only this Nebula, and an engine deployed into a sandbox (its card, the models read-only, nothing out, the
gateway's address routed to, stopped by deleting the sandbox)."""
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from nebula import openshell as O
from nebula.tests import WINDOWS, linux_only  # noqa: F401
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.test_callers import call
from nebula.tests.test_engines import CARDS, FAKE, ROOT, free_port, make_gguf, wait_for

FAKE_OPENSHELL = str(Path(__file__).with_name("fake_openshell.py"))


@linux_only
class OpenShellTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.models = os.path.join(self.tmp, "models")
        os.makedirs(os.path.join(self.models, "someone", "repo"))
        make_gguf(os.path.join(self.models, "someone", "repo", "m.gguf"))
        self.env = {"NEBULA_OPENSHELL": FAKE_OPENSHELL, "FAKE_OPENSHELL_HOME": os.path.join(self.tmp, "osh"),
                    "NEBULA_DOCKER": str(Path(__file__).with_name("fake_docker.py")),
                    "FAKE_DOCKER_VOLUMES": os.path.join(self.tmp, "volumes.json"), "FAKE_DOCKER_IMAGES": os.path.join(self.tmp, "images.json"), "FAKE_OPENSHELL_MODELS": self.models, "FAKE_OPENSHELL_CWD": ROOT,
                    "NEBULA_FAKE_CARDS": json.dumps(CARDS)}
        self.old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        self.port = free_port()
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
            "engines": [{"name": "boxed", "kind": "llama.cpp", "binary": FAKE, "model": "models/someone/repo/m.gguf",
                         "port": self.port, "context": 2048, "slots": 1, "card": "GPU-bbbb", "where": "openshell",
                         "ready_timeout_s": 30}],
            "lan": {"decided": True, "discoverable": False}}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        try:
            call(self.base, "/api/yard/engines/boxed/stop", {})
        finally:
            self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
            for k, v in self.old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_an_engine_in_a_sandbox(self):
        st, out = call(self.base, "/api/yard/engines/boxed/start", {"wait": True})
        self.assertEqual((st, out["state"]), (200, "ready"), out)
        spec = json.load(open(os.path.join(self.env["FAKE_OPENSHELL_HOME"], O.sandbox_name("boxed"), ".spec")))
        driver = json.loads(spec["driver-config-json"])
        self.assertEqual(driver["docker"]["cdi_devices"], ["nvidia.com/gpu=1"])            # its card, by index
        self.assertEqual(driver["docker"]["mounts"][0], {"type": "volume", "source": O.volume_name(os.path.join(self.tmp, "models")), "target": "/models", "read_only": True})
        self.assertTrue(spec["gpu"] and spec["detach"])
        self.assertEqual(open(spec["policy"]).read(), O.policy_text(os.getuid()))         # nothing out; its key: the owner's user reads it
        self.assertEqual(spec["from"], "ghcr.io/ggml-org/llama.cpp:server-cuda")
        call(self.base, "/api/yard/engines/boxed/route", {})
        self.assertEqual(self.srv.cfg.tier("boxed").engine, f"http://127.0.0.1:{self.port}")   # the gateway's address
        st, body = call(self.base, "/v1/chat/completions", {"model": "boxed", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200, body)
        # its own key (design decision, 2026-10-20): the engine in the sandbox takes the key only Nebula holds, and the
        # gateway passes Nebula's Authorization through to it; anything else on this machine is refused
        calls = [json.loads(ln) for ln in open(os.path.join(self.env["FAKE_OPENSHELL_HOME"], "calls.log"))]
        create = next(c for c in calls if c[:2] == ["sandbox", "create"])
        self.assertIn("--api-key-file /keys/key", " ".join(create))
        self.assertIn(["--authorization-mode", "bearer-passthrough"], [c[4:6] for c in calls if c[:2] == ["service", "expose"]])
        st, _ = call(f"http://127.0.0.1:{self.port}", "/v1/chat/completions", {"model": "boxed", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 401)
        panel = call(self.base, "/api/yard/openshell")[1]
        self.assertEqual(([s["name"] for s in panel["sandboxes"]], panel["gateways"][0]["name"]), ([O.sandbox_name("boxed")], "local"))
        pol = call(self.base, "/api/yard/openshell/sandboxes/" + O.sandbox_name("boxed") + "/policy")[1]
        self.assertFalse(pol["editable"])
        self.assertEqual(call(self.base, "/api/yard/engines/boxed/stop", {})[1]["state"], "stopped")
        self.assertFalse(os.path.isdir(os.path.join(self.env["FAKE_OPENSHELL_HOME"], O.sandbox_name("boxed"))))

    def test_an_impulse_engine_runs_from_an_image_built_from_its_program(self):
        # Impulse has no image to pull: Nebula builds one from impulse-server (here a one-file stand-in), and the engine
        # gets its two cards and its own settings (GGML_CUDA_P2P for a split model) inside the sandbox
        prog = os.path.join(self.tmp, "bin", "impulse-server")
        os.makedirs(os.path.dirname(prog))
        Path(prog).write_text(f"#!/bin/sh\nexec {sys.executable} -m nebula.tests.fake_impulse \"$@\"\n")
        os.chmod(prog, 0o755)
        seen = os.path.join(self.tmp, "args.txt")
        spec = dict(self.srv.yard.spec("boxed"), kind="impulse", binary=prog, card="GPU-aaaa,GPU-bbbb", cpus="0",
                    env={"GGML_CUDA_P2P": "1", "FAKE_IMPULSE_ARGS": seen})
        self.assertEqual(call(self.base, "/api/yard/engines", spec)[0], 200)
        st, out = call(self.base, "/api/yard/engines/boxed/start", {"wait": True})
        self.assertEqual((st, out["state"]), (200, "ready"), out)
        tag = O.engine_image_tag(prog)
        self.assertTrue(tag.startswith("nebula-engine-impulse-server:"), tag)
        box = json.load(open(os.path.join(self.env["FAKE_OPENSHELL_HOME"], O.sandbox_name("boxed"), ".spec")))
        self.assertEqual(box["from"], tag)
        self.assertEqual(json.loads(box["driver-config-json"])["docker"]["cdi_devices"], ["nvidia.com/gpu=0", "nvidia.com/gpu=1"])
        self.assertIn("GGML_CUDA_P2P=1", box["env"])
        self.assertEqual(box["gpu"], 2)                                                     # OpenShell: the count matches
        built = json.load(open(self.env["FAKE_DOCKER_IMAGES"]))[tag]
        dockerfile = open(os.path.join(built, "Dockerfile")).read()
        self.assertTrue(dockerfile.startswith(f"FROM {O.BASE_IMAGE}\nCOPY engine/ /opt/engine/\n"), dockerfile)
        create = next(c for c in (json.loads(ln) for ln in open(os.path.join(self.env["FAKE_OPENSHELL_HOME"], "calls.log")))
                      if c[:2] == ["sandbox", "create"])
        self.assertEqual(create[create.index("--") + 1:create.index("--") + 5], ["taskset", "-c", "0", "/opt/engine/impulse-server"])
        self.assertTrue(os.path.exists(seen))                                             # its settings reached it
        call(self.base, "/api/yard/engines/boxed/route", {})
        st, body = call(self.base, "/v1/chat/completions", {"model": "boxed", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200, body)
        self.assertEqual(call(self.base, "/api/yard/engines/boxed/stop", {})[1]["state"], "stopped")
        # a second start uses the image already built
        builds = len(json.load(open(self.env["FAKE_DOCKER_IMAGES"])))
        call(self.base, "/api/yard/engines/boxed/start", {"wait": True})
        self.assertEqual(len(json.load(open(self.env["FAKE_DOCKER_IMAGES"]))), builds)

    def test_a_flashnext_pack_runs_in_a_sandbox_with_impulse_fx_beside_impulse_server(self):
        """0.1.4: Impulse Flash-Next is a piece of its own (nebula/pieces.py). A pack on Impulse in a sandbox gets an image
        with impulse-fx and its engine library beside impulse-server; without the piece it is refused, saying what to get."""
        prog = os.path.join(self.tmp, "bin", "impulse-server")
        os.makedirs(os.path.dirname(prog))
        Path(prog).write_text(f"#!/bin/sh\nexec {sys.executable} -m nebula.tests.fake_impulse \"$@\"\n")
        os.chmod(prog, 0o755)
        pack = os.path.join(self.models, "fn-pack")
        os.makedirs(os.path.join(pack, "pack"))
        Path(pack, "pack", "index.txt").write_text("")
        make_gguf(os.path.join(pack, "fn-dense.gguf"))
        spec = dict(self.srv.yard.spec("boxed"), kind="impulse", binary=prog, model="models/fn-pack", card="GPU-aaaa")
        self.assertEqual(call(self.base, "/api/yard/engines", spec)[0], 200)
        st, out = call(self.base, "/api/yard/engines/boxed/start", {"wait": True})
        self.assertNotEqual(out.get("state"), "ready")
        self.assertIn("Impulse Flash-Next, which is not here yet", json.dumps(out))
        fx = os.path.join(self.tmp, "nebula-impulse-flashnext")                           # the piece, as the setup leaves it
        os.makedirs(fx)
        for f in ("impulse-fx", "libimpulse_fx.so", "libstrata_kq.so"):
            Path(fx, f).write_text(f)
        st, out = call(self.base, "/api/yard/engines/boxed/start", {"wait": True})
        self.assertEqual((st, out["state"]), (200, "ready"), out)
        extras = sorted(os.path.join(fx, f) for f in os.listdir(fx))
        tag = O.engine_image_tag(prog, extras)
        self.assertNotEqual(tag, O.engine_image_tag(prog))                                 # its own image
        box = json.load(open(os.path.join(self.env["FAKE_OPENSHELL_HOME"], O.sandbox_name("boxed"), ".spec")))
        self.assertEqual(box["from"], tag)
        built = json.load(open(self.env["FAKE_DOCKER_IMAGES"]))[tag]
        self.assertEqual(sorted(os.listdir(os.path.join(built, "engine"))),
                         ["impulse-fx", "impulse-server", "lib", "libimpulse_fx.so", "libstrata_kq.so"])

    def test_sandbox_names_fit_openshells_limit(self):
        self.assertEqual(O.sandbox_name("boxed"), "nebeng-boxed")
        long1, long2 = O.sandbox_name("storyteller-big-31b"), O.sandbox_name("storyteller-big-26b")
        self.assertTrue(len(long1) <= 19 and len(long2) <= 19 and long1 != long2, (long1, long2))
        self.assertTrue(O.is_engine_sandbox(long1) and O.is_engine_sandbox("nebula-engine-old"))
        self.assertFalse(O.is_engine_sandbox("nebula-claude"))
        from nebula import sandbox
        self.assertLessEqual(len(sandbox.sandbox_name("a-very-long-tier-name-indeed")), 19)

    def test_an_engines_image_carries_its_libraries_but_never_the_c_library_or_the_driver(self):
        libs = [n for n, _p in O.engine_libs(sys.executable)]
        self.assertFalse([n for n in libs if n.startswith(("libc.so", "ld-linux", "libm.so"))], libs)
        self.assertTrue(O._SKIP_LIB.match("libcuda.so.1") and O._SKIP_LIB.match("libnvidia-ml.so.1"))
        self.assertFalse(O._SKIP_LIB.match("libcublas.so.13") or O._SKIP_LIB.match("libcudart.so.13") or O._SKIP_LIB.match("libgomp.so.1"))
        script = os.path.join(self.tmp, "s.sh")
        Path(script).write_text("#!/bin/sh\n")
        self.assertEqual(O.engine_libs(script), [])

    def test_the_model_must_be_in_the_models_folder_and_others_sandboxes_are_left_alone(self):
        outside = os.path.join(self.tmp, "elsewhere.gguf")
        make_gguf(outside)
        spec = dict(self.srv.yard.spec("boxed"), model=outside)
        call(self.base, "/api/yard/engines", spec)
        st, err = call(self.base, "/api/yard/engines/boxed/start", {})
        self.assertEqual(st, 400)
        self.assertIn("models folder", err["error"]["message"])
        self.assertEqual(call(self.base, "/api/yard/openshell/sandboxes/someone-elses/delete", {})[0], 403)
        self.assertEqual(call(self.base, "/api/yard/engines", dict(spec, kind="command", command=["x"], model=None))[0], 400)

    def test_the_provider_profile_reaches_only_this_nebula(self):
        p = call(self.base, "/api/yard/openshell/profile")[1]
        self.assertIn(f"host: {O.HOST}", p["profile"])
        self.assertIn(f"port: {self.srv.server_port}", p["profile"])
        self.assertIn("env_vars: [NEBULA_API_KEY]", p["profile"])
        self.assertEqual(p["profile"].count("- host:"), 1)                                  # one endpoint: this Nebula
        self.assertTrue(any("provider profile import" in s for s in p["steps"]))

    def test_without_openshell(self):
        os.environ["NEBULA_OPENSHELL"] = "/nonexistent/openshell"
        self.assertEqual(call(self.base, "/api/yard/openshell")[1], {"available": False, "why": "openshell is not installed on this machine"})


if __name__ == "__main__":
    unittest.main()


class NumaLayoutTest(unittest.TestCase):
    """The machine's sockets, stated to an engine whose CPUs span two of them (IMPULSE_NUMA_NODES): the test server's layout, an
    engine held to one socket, a one-socket machine, the engine's own setting, and "numa": "off"."""
    TWO_SOCKETS = ("/sys/devices/system/node/node0/cpulist 0-17,36-53\n"
            "/sys/devices/system/node/node1/cpulist 18-35,54-71\n")

    def test_list_answers_in_any_wrapping_become_rows(self):
        rows = [{"name": "a", "phase": "Ready"}]
        self.assertEqual(O._rows(rows, "sandboxes"), rows)
        self.assertEqual(O._rows({"sandboxes": rows}, "sandboxes"), rows)
        self.assertEqual(O._rows({"items": rows, "count": 1}, "sandboxes"), rows)
        self.assertEqual(O._rows({"a": {"phase": "Ready"}}, "sandboxes"), rows)
        self.assertEqual(O._rows({"name": "a", "phase": "Ready"}, "sandboxes"), rows)
        self.assertEqual(O._rows({}, "sandboxes"), [])
        self.assertEqual(O._rows("nope", "sandboxes"), [])

    def test_cpulists(self):
        self.assertEqual(O.parse_cpulist("0-3,8"), [0, 1, 2, 3, 8])
        self.assertEqual(O.parse_cpulist("3-1"), [])
        self.assertEqual(O.parse_cpulist("a"), [])
        self.assertEqual(O.cpulist_text([8, 0, 1, 2, 3, 3]), "0-3,8")
        self.assertEqual(O.cpulist_text(O.parse_cpulist("18-35,54-71")), "18-35,54-71")

    def test_node_cpus(self):
        n = O.node_cpus(self.TWO_SOCKETS + "garbage line\n/sys/devices/system/node/node2/cpulist \n")
        self.assertEqual(sorted(n), [0, 1])                                     # a memory-only node (no CPUs) is left out
        self.assertEqual(len(n[0]), 36)
        self.assertEqual(O.node_cpus(""), {})

    def test_numa_env(self):
        nodes = O.node_cpus(self.TWO_SOCKETS)
        spec = {"name": "bb", "kind": "command"}
        cmd = ["/opt/strata-serve/.venv/bin/python", "-m", "serve.server"]
        self.assertEqual(O.numa_env(spec, cmd, nodes), {"IMPULSE_NUMA_NODES": "0=0-17,36-53;1=18-35,54-71"})
        # held to socket 1 by its own taskset (the reference engine today): nothing to split, nothing stated
        self.assertEqual(O.numa_env(spec, ["/usr/bin/taskset", "-c", "18-35,54-71"] + cmd, nodes), {})
        # a taskset on both sockets: the layout of those CPUs only
        self.assertEqual(O.numa_env(spec, ["taskset", "-c", "1-35"] + cmd, nodes),
                         {"IMPULSE_NUMA_NODES": "0=1-17;1=18-35"})
        self.assertEqual(O.numa_env(spec, cmd, {0: list(range(16))}), {})        # one socket: nothing
        self.assertEqual(O.numa_env(spec, cmd, {}), {})                          # layout unknown: nothing
        self.assertEqual(O.numa_env(dict(spec, env={"IMPULSE_NUMA_NODES": "0=0-3;1=4-7"}), cmd, nodes), {})
        self.assertEqual(O.numa_env(dict(spec, numa="off"), cmd, nodes), {"STRATA_NUMA_POOL": "0"})

    def test_deploy_argv_states_the_layout(self):
        nodes = O.node_cpus(self.TWO_SOCKETS)
        spec = {"name": "bb", "kind": "command", "image": "img:1"}
        argv = O.deploy_argv(spec, ["/bin/engine"], [2, 3], "vol", "/p.yaml", env={"A": "1"}, cpus=72, nodes=nodes)
        envs = [argv[i + 1] for i, a in enumerate(argv) if a == "--env"]
        self.assertEqual(envs, ["A=1", "IMPULSE_NUMA_NODES=0=0-17,36-53;1=18-35,54-71"])
        # the engine's own env wins over what Nebula adds
        argv = O.deploy_argv(dict(spec, numa="off"), ["/bin/engine"], None, "vol", "/p.yaml",
                             env={"STRATA_NUMA_POOL": "1"}, cpus=72, nodes=nodes)
        self.assertIn("STRATA_NUMA_POOL=1", argv)
        # a one-socket machine: the argv is what it was
        before = O.deploy_argv(spec, ["/bin/engine"], None, "vol", "/p.yaml", env={"A": "1"}, cpus=8, nodes={0: list(range(8))})
        self.assertEqual([a for i, a in enumerate(before) if i and before[i - 1] == "--env"], ["A=1"])
