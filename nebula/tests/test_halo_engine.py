"""Halo's own model (nebula/halo_engine.py): started for Halo when it is on this machine, used by Halo's calls only, never
one of the models (not routed to, not listed)."""
import json
import os
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from nebula.config import Config
from nebula.server import make_server
from nebula.tests import free_port
from nebula.tests.fake_engine import FakeEngine

FAKE = textwrap.dedent('''
    import json, sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import os
    args = sys.argv[1:]
    if "--list-devices" in args:                         # llama-server's own listing, as the test sets it
        print("Available devices:")
        for d in json.loads(os.environ.get("FAKE_DEVICES") or "[]"):
            print(f"  {d[0]}: {d[1]} ({d[2]} MiB, {d[3]} MiB free)")
        sys.exit(0)
    port = int(args[args.index("--port") + 1])
    log = open(sys.argv[0] + ".log", "a")
    log.write(json.dumps({"args": args}) + "\\n"); log.flush()
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            self._send({"status": "ok"} if self.path == "/health" else {"model_path": "base.gguf"})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            key = __import__("os").environ.get("LLAMA_API_KEY")
            log.write(json.dumps({"keyed_path": self.path, "keyed": bool(key)}) + "\\n"); log.flush()
            if key and self.headers.get("Authorization") != "Bearer " + key:
                self.send_response(401); self.end_headers(); return
            log.write(json.dumps({"path": self.path, "body": body}) + "\\n"); log.flush()
            if self.path == "/lora-adapters":
                return self._send([])
            if body.get("stream"):                       # Nebula streams from its engines: SSE
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                for d in ({"role": "assistant", "content": "all calm, "}, {"content": "from Halo's own model"}):
                    ch = {"id": "x", "object": "chat.completion.chunk", "model": "halo", "created": 0, "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
                    self.wfile.write(("data: " + json.dumps(ch) + "\\n\\n").encode())
                end = {"id": "x", "object": "chat.completion.chunk", "model": "halo", "created": 0, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}}
                self.wfile.write(("data: " + json.dumps(end) + "\\n\\ndata: [DONE]\\n\\n").encode())
                return
            self._send({"id": "x", "object": "chat.completion", "model": "halo", "created": 0,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "all calm, from Halo's own model"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}})
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')


class HaloEngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fake = os.path.join(self.tmp, "fake_llama_server.py")
        Path(self.fake).write_text(FAKE)
        base = os.path.join(self.tmp, "base.gguf")
        Path(base).write_bytes(b"GGUF")
        self.engine = FakeEngine("small")
        path = os.path.join(self.tmp, "nebula.json")
        json.dump({"port": free_port(), "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
                   "tiers": [{"name": "small", "engine": self.engine.url}]}, open(path, "w"))
        env = {"NEBULA_HALO_SERVER": f'"{sys.executable}" "{self.fake}"', "NEBULA_HALO_BASE": base, "NEBULA_FAKE_CARDS": "[]"}
        self.env = env
        self.envp = mock.patch.dict(os.environ, env)      # for the whole test: Halo places itself on its own thread
        self.envp.start()
        self.srv = make_server(Config.from_args(["--config", path]))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        end = time.time() + 30
        while not self.srv.halo_engine.ready() and self.srv.halo_engine.state in ("starting",) and time.time() < end:
            time.sleep(0.2)

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close(); self.engine.stop(); self.envp.stop()

    def lines(self):
        return [json.loads(x) for x in Path(self.fake + ".log").read_text().splitlines()]

    def test_it_starts_on_the_cpu_with_both_adapters_and_halo_thinks_with_it(self):
        self.assertTrue(self.srv.halo_engine.ready(), self.srv.halo_engine.view())
        args = self.lines()[0]["args"]
        self.assertEqual(args[args.index("-m") + 1], os.path.join(self.tmp, "base.gguf"))
        loras = args[args.index("--lora-scaled") + 1]
        self.assertIn("halo-guard-v2-lora-f16.gguf:1.0", loras)                       # Halo's adapter, always on
        self.assertIn("halo-uncensored-heretic-t198-lora-f16.gguf:0.0", loras)        # Standard: the Uncensored one at 0
        self.assertEqual(args[args.index("-ngl") + 1], "0")                            # no card here: the CPU
        self.assertEqual(self.srv.halo_engine.view()["where"], "cpu")
        self.assertEqual(self.srv.halo._ask_model("anything new?"), "all calm, from Halo's own model")
        self.assertEqual(len(self.engine.requests), 0)                                 # not the Assistant's model
        self.assertEqual([t.name for t in self.srv.cfg.tiers], ["small"])              # not one of the models
        with urllib.request.urlopen(f"http://127.0.0.1:{self.srv.server_port}/v1/models") as r:
            self.assertNotIn("halo-own", r.read().decode())
        self.assertEqual(json.loads(self.srv.halo_engine.view() and json.dumps(self.srv.halo.view()["own_model"]))["state"], "ready")

    def test_standard_or_uncensored_is_the_adapters_scale_on_the_running_server(self):
        from nebula.tests.test_ui import req
        st, _h, raw = req(f"http://127.0.0.1:{self.srv.server_port}", "/api/halo", {"variant": "uncensored"})
        self.assertEqual(st, 200, raw)
        self.assertEqual(json.loads(raw)["own_model"]["variant"], "uncensored")
        sent = [x["body"] for x in self.lines() if x.get("path") == "/lora-adapters"]
        self.assertEqual(sent[-1], [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 1.0}])
        self.assertEqual(req(f"http://127.0.0.1:{self.srv.server_port}", "/api/halo", {"variant": "spicy"})[0], 400)

    def test_halos_server_answers_only_nebula(self):
        """A key of its own each start, in its environment: a web page (llama-server allows any origin) gets a 401."""
        key = self.srv.halo_engine.api_key
        self.assertTrue(key and len(key) > 20)
        self.assertNotIn(key, " ".join([l["args"] for l in self.lines() if "args" in l][0]))      # never on the command line
        self.assertEqual(self.srv.halo._ask_model("anything new?"), "all calm, from Halo's own model")
        req = urllib.request.Request(self.srv.halo_engine.tier.engine + "/v1/chat/completions", b'{"messages": []}',
                                     {"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(e.exception.code, 401)

    def test_the_routers_calls_carry_its_key(self):
        """The smart router asks Halo's own model too (nebula/stage.py): with the key of this start, as Halo's own calls
        do, so a server that enforces the key (impulse-server since One Engine) still routes."""
        from nebula import stage
        self.assertTrue(self.srv.halo_engine.ready(), self.srv.halo_engine.view())
        got = stage.ask_halo({"messages": [{"role": "user", "content": "What is the capital of Portugal?"}]}, ["everyday", "larger"])
        self.assertIsNotNone(got)                                                       # a 401 would be None
        asked = [x for x in self.lines() if x.get("path") == "/v1/chat/completions"
                 and "You are Halo, the router" in json.dumps((x.get("body") or {}).get("messages"))]
        self.assertEqual(len(asked), 1)                                                 # through the key check

    def test_an_owner_route_wins_over_its_own_model(self):
        self.srv.cfg.assistant = {**(self.srv.cfg.assistant or {}), "halo": {"route": {"tier": "small"}}}
        self.srv.halo._ask_model("anything new?")
        self.assertEqual(len(self.engine.requests), 1)


class HaloEngineRuntimeTest(HaloEngineTest):
    """Impulse first (design decision 10-21), llama-server when Impulse does not come up."""
    def test_it_starts_on_the_cpu_with_both_adapters_and_halo_thinks_with_it(self):
        pass                                                                           # the parent's

    def test_impulse_first_then_llama_server(self):
        from nebula import halo_engine as HE
        with mock.patch.dict(os.environ, {"NEBULA_HALO_SERVER": ""}), \
                mock.patch("nebula.packages.locate", return_value="/opt/impulse/impulse-server"), \
                mock.patch.object(HE, "find_llama_server", return_value=["/opt/halo/llama-server"]):
            self.assertEqual(HE.find_servers(), [("Impulse", ["/opt/impulse/impulse-server"]), ("llama.cpp", ["/opt/halo/llama-server"])])
        broken = [sys.executable, "-c", "import sys; sys.exit(3)"]                    # an Impulse that cannot run it
        fake = [sys.executable, self.fake]
        with mock.patch.dict(os.environ, self.env), \
                mock.patch.object(HE, "find_servers", return_value=[("Impulse", broken), ("llama.cpp", fake)]):
            eng = self.srv.halo_engine.restart()
            end = time.time() + 30
            while not eng.ready() and eng.state != "failed" and time.time() < end:
                time.sleep(0.2)
        self.assertTrue(eng.ready(), eng.view())
        self.assertEqual(eng.view()["runtime"], "llama.cpp")
        self.assertEqual(self.srv.halo._ask_model("anything new?"), "all calm, from Halo's own model")

    def test_halo_in_impulse_has_a_key_too(self):
        """impulse-server reads LLAMA_API_KEY as llama-server does: Halo in Impulse answers only Nebula as well."""
        from nebula import halo_engine as HE
        with mock.patch.dict(os.environ, self.env), \
                mock.patch.object(HE, "find_servers", return_value=[("Impulse", [sys.executable, self.fake])]):
            eng = self.srv.halo_engine.restart()
            end = time.time() + 30
            while not eng.ready() and eng.state != "failed" and time.time() < end:
                time.sleep(0.2)
        self.assertTrue(eng.ready(), eng.view())
        self.assertEqual(eng.view()["runtime"], "Impulse")
        self.assertTrue(eng.api_key and len(eng.api_key) > 20)
        self.assertEqual(self.srv.halo._ask_model("anything new?"), "all calm, from Halo's own model")
        req = urllib.request.Request(eng.tier.engine + "/v1/chat/completions", b'{"messages": []}', {"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(e.exception.code, 401)


class HaloModelLinkTest(unittest.TestCase):
    """His model file linked from elsewhere (another Nebula's copy) is used, and nothing about it stops Nebula's start."""
    def test_a_linked_model_is_used(self):
        from nebula import halo_engine as HE, halo_model as HM
        m = HM.manifest()
        tmp = tempfile.mkdtemp()
        elsewhere = os.path.join(tmp, "other-disk", m["base"]["file"])
        os.makedirs(os.path.dirname(elsewhere))
        with open(elsewhere, "wb") as f:
            f.truncate(int(m["base"]["bytes"]))
        models = os.path.join(tmp, "models")
        link = Path(models, *m["base"]["repo"].split("/"), m["base"]["file"])
        link.parent.mkdir(parents=True)
        link.symlink_to(elsewhere)
        with mock.patch.dict(os.environ, {"NEBULA_HALO_BASE": ""}):
            self.assertEqual(HE.find_base(models, m), os.path.realpath(elsewhere))
            os.remove(elsewhere)                                                       # a broken link: not his model
            self.assertIsNone(HE.find_base(models, m))

    def test_nebula_starts_and_halo_runs_from_the_link(self):
        from nebula import halo_model as HM
        m = HM.manifest()
        tmp = tempfile.mkdtemp()
        fake = os.path.join(tmp, "fake_llama_server.py")
        Path(fake).write_text(FAKE)
        elsewhere = os.path.join(tmp, "other-disk", m["base"]["file"])
        os.makedirs(os.path.dirname(elsewhere))
        with open(elsewhere, "wb") as f:
            f.truncate(int(m["base"]["bytes"]))
        models = os.path.join(tmp, "models")
        link = Path(models, *m["base"]["repo"].split("/"), m["base"]["file"])
        link.parent.mkdir(parents=True)
        link.symlink_to(elsewhere)
        engine = FakeEngine("small")
        path = os.path.join(tmp, "nebula.json")
        with open(path, "w") as f:
            json.dump({"port": free_port(), "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False, "models_dir": models,
                       "tiers": [{"name": "small", "engine": engine.url}]}, f)
        env = {"NEBULA_HALO_SERVER": f'"{sys.executable}" "{fake}"', "NEBULA_HALO_BASE": "", "NEBULA_FAKE_CARDS": "[]"}
        with mock.patch.dict(os.environ, env):
            srv = make_server(Config.from_args(["--config", path]))
        try:
            end = time.time() + 30
            while srv.halo_engine.state == "starting" and time.time() < end:
                time.sleep(0.2)
            self.assertTrue(srv.halo_engine.ready(), srv.halo_engine.view())
            args = [json.loads(x) for x in Path(fake + ".log").read_text().splitlines()][0]["args"]
            self.assertEqual(args[args.index("-m") + 1], os.path.realpath(elsewhere))
        finally:
            srv.server_close(); srv.tools.close(); engine.stop()

    def test_nothing_stops_nebulas_start(self):
        from nebula import halo_engine as HE
        srv = mock.Mock()
        srv.cfg.assistant = {}
        eng = HE.HaloEngine(srv)
        with mock.patch.object(HE, "find_servers", side_effect=RuntimeError("a file outside the models folder")):
            self.assertIs(eng.start(), eng)
        self.assertEqual(eng.state, "unavailable")
        self.assertIn("outside the models folder", eng.why)


class HaloEngineAbsentTest(unittest.TestCase):
    def test_without_its_server_halo_says_why_and_uses_the_assistants_model(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "nebula.json")
        json.dump({"port": free_port(), "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False, "tiers": []}, open(path, "w"))
        with mock.patch.dict(os.environ, {"NEBULA_HALO_SERVER": "", "PATH": tmp}):
            srv = make_server(Config.from_args(["--config", path]))
        try:
            self.assertEqual(srv.halo_engine.state, "unavailable")
            self.assertTrue(srv.halo_engine.why)
        finally:
            srv.server_close(); srv.tools.close()


class HaloRadeonTest(unittest.TestCase):
    """Halo also works on a Radeon (llama-server's Vulkan backend; Impulse is CUDA only)."""

    def start(self, devices, cards):
        from nebula import halo_engine
        halo_engine._VK_CACHE.clear()
        self.tmp = tempfile.mkdtemp()
        self.fake = os.path.join(self.tmp, "fake_llama_server.py")
        Path(self.fake).write_text(FAKE)
        base = os.path.join(self.tmp, "base.gguf")
        Path(base).write_bytes(b"GGUF")
        path = os.path.join(self.tmp, "nebula.json")
        json.dump({"port": free_port(), "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False, "tiers": []}, open(path, "w"))
        self.envp = mock.patch.dict(os.environ, {"NEBULA_HALO_SERVER": f'"{sys.executable}" "{self.fake}"', "NEBULA_HALO_BASE": base,
                                                 "NEBULA_FAKE_CARDS": json.dumps(cards), "FAKE_DEVICES": json.dumps(devices)})
        self.envp.start()
        self.srv = make_server(Config.from_args(["--config", path]))
        end = time.time() + 30
        while not self.srv.halo_engine.ready() and time.time() < end:
            time.sleep(0.2)

    def tearDown(self):
        self.srv.server_close(); self.srv.tools.close(); self.envp.stop()

    def args(self):
        return [json.loads(x) for x in Path(self.fake + ".log").read_text().splitlines() if '"args"' in x][-1]["args"]

    NVIDIA = [{"uuid": "GPU-n", "index": 0, "name": "GTX 1660 SUPER", "memory_mb": 6144, "used_mb": 0}]

    def test_a_spare_radeon_takes_halo_and_leaves_the_nvidia_card_to_the_engines(self):
        self.start([["Vulkan0", "AMD Radeon RX 5500M", 4080, 3900], ["Vulkan1", "Intel(R) UHD Graphics", 8000, 8000],
                    ["Vulkan2", "NVIDIA GeForce GTX 1660 SUPER", 6144, 6000]], self.NVIDIA)
        v = self.srv.halo_engine.view()
        self.assertEqual((v["state"], v["where"], v["gpu_name"]), ("ready", "vk:0", "AMD Radeon RX 5500M"))
        a = self.args()
        self.assertEqual((a[a.index("--device") + 1], a[a.index("-ngl") + 1]), ("Vulkan0", "99"))

    def test_a_radeon_without_room_leaves_halo_on_the_nvidia_card(self):
        self.start([["Vulkan0", "AMD Radeon RX 5500M", 4080, 900]], self.NVIDIA)
        self.assertEqual(self.srv.halo_engine.view()["where"], "gpu:0")
        self.assertNotIn("--device", self.args())

    def test_integrated_graphics_alone_is_not_a_gpu_for_halo(self):
        self.start([["Vulkan0", "Intel(R) UHD Graphics", 8000, 8000]], [])
        self.assertEqual(self.srv.halo_engine.view()["where"], "cpu")


class LoraPathsOnWindowsTest(unittest.TestCase):
    def test_adapters_are_given_relative_so_no_drive_colon_breaks_file_scale(self):
        from nebula.halo_engine import lora_paths_without_colons
        args = ["-m", "C:\\x\\base.gguf", "--lora-scaled",
                "C:\\P\\Nebula EH\\python\\Lib\\site-packages\\nebula\\assets\\halo\\a.gguf:1.0,C:\\P\\Nebula EH\\python\\Lib\\site-packages\\nebula\\assets\\halo\\b.gguf:0.0"]
        out = lora_paths_without_colons(args, "C:\\P\\Nebula EH\\halo", nt=True)
        items = out[out.index("--lora-scaled") + 1].split(",")
        self.assertEqual(items, ["..\\python\\Lib\\site-packages\\nebula\\assets\\halo\\a.gguf:1.0",
                                 "..\\python\\Lib\\site-packages\\nebula\\assets\\halo\\b.gguf:0.0"])
        self.assertTrue(all(i.count(":") == 1 for i in items))
        self.assertEqual(lora_paths_without_colons(args, "/x", nt=False), args)            # Linux: as it was


if __name__ == "__main__":
    unittest.main()


class HaloFetchTest(unittest.TestCase):
    """No base model yet (the installer had no network): Nebula fetches it (pinned revision, sha256) and then starts
    Halo's model."""

    def test_the_base_is_fetched_then_halo_starts(self):
        import hashlib
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from nebula import halo_model
        tmp = tempfile.mkdtemp()
        blob = b"GGUF" + b"x" * 5000
        m = halo_model.manifest()
        m["base"] = {"repo": "t/halo-base", "revision": "r1", "file": "b.gguf", "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        man = os.path.join(tmp, "halo.json")
        Path(man).write_text(json.dumps(m))
        asked = []

        class Hub(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                asked.append(self.path)
                if self.path != "/t/halo-base/resolve/r1/b.gguf":
                    self.send_response(404); self.end_headers(); return
                self.send_response(200); self.send_header("Content-Length", str(len(blob))); self.end_headers(); self.wfile.write(blob)
        hub = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
        threading.Thread(target=hub.serve_forever, daemon=True).start()
        fake = os.path.join(tmp, "fake_llama_server.py")
        Path(fake).write_text(FAKE)
        engine = FakeEngine("small")
        path = os.path.join(tmp, "nebula.json")
        json.dump({"port": free_port(), "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False,
                   "tiers": [{"name": "small", "engine": engine.url}]}, open(path, "w"))
        env = {"NEBULA_HALO_SERVER": f'"{sys.executable}" "{fake}"', "NEBULA_FAKE_CARDS": "[]",
               "HF_ENDPOINT": f"http://127.0.0.1:{hub.server_port}"}
        with mock.patch.dict(os.environ, env), mock.patch.object(halo_model, "MANIFEST", Path(man)):
            os.environ.pop("NEBULA_HALO_BASE", None)          # inside: patch.dict puts it back after (the suite's default)
            srv = make_server(Config.from_args(["--config", path]))
            try:
                end = time.time() + 30
                while not srv.halo_engine.ready() and time.time() < end:
                    time.sleep(0.2)
                self.assertTrue(srv.halo_engine.ready(), srv.halo_engine.view())
                got = os.path.join(tmp, "models", "t", "halo-base", "b.gguf")
                self.assertEqual(Path(got).read_bytes(), blob)                      # in the models folder, checked
                args = [json.loads(x) for x in Path(fake + ".log").read_text().splitlines()][0]["args"]
                self.assertEqual(args[args.index("-m") + 1], got)
            finally:
                srv.server_close(); srv.tools.close(); engine.stop(); hub.shutdown()


class HaloWhereTest(unittest.TestCase):
    """Halo on a GPU by default, the CPU by choice."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fake = os.path.join(self.tmp, "fake_llama_server.py")
        Path(self.fake).write_text(FAKE)
        base = os.path.join(self.tmp, "base.gguf")
        Path(base).write_bytes(b"GGUF")
        path = os.path.join(self.tmp, "nebula.json")
        json.dump({"port": free_port(), "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False, "tiers": []}, open(path, "w"))
        cards = json.dumps([{"uuid": "GPU-a", "index": 0, "name": "RTX 4060 Ti", "memory_mb": 16380, "used_mb": 16}])
        self.env = mock.patch.dict(os.environ, {"NEBULA_HALO_SERVER": f'"{sys.executable}" "{self.fake}"', "NEBULA_HALO_BASE": base,
                                                "NEBULA_FAKE_CARDS": cards})
        self.env.start()
        self.srv = make_server(Config.from_args(["--config", path]))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.wait()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close(); self.env.stop()

    def wait(self):
        end = time.time() + 30
        while not self.srv.halo_engine.ready() and time.time() < end:
            time.sleep(0.2)

    def last_args(self):
        return [json.loads(x) for x in Path(self.fake + ".log").read_text().splitlines() if '"args"' in x][-1]["args"]

    def test_a_gpu_by_default_the_cpu_by_choice(self):
        from nebula.tests.test_ui import req
        v = self.srv.halo_engine.view()
        self.assertEqual((v["state"], v["where"], v["runs_on"]), ("ready", "gpu:0", "gpu"))
        self.assertEqual(self.last_args()[self.last_args().index("-ngl") + 1], "99")
        st, _h, raw = req(f"http://127.0.0.1:{self.srv.server_port}", "/api/halo", {"where": "cpu"})
        self.assertEqual(st, 200, raw)
        self.wait()
        v = self.srv.halo_engine.view()
        self.assertEqual((v["state"], v["where"], v["runs_on"], v["why"]), ("ready", "cpu", "cpu", None))
        self.assertEqual(self.last_args()[self.last_args().index("-ngl") + 1], "0")
        self.assertEqual(self.srv.cfg.assistant["halo"]["where"], "cpu")                 # kept in the config
        self.assertEqual(req(f"http://127.0.0.1:{self.srv.server_port}", "/api/halo", {"where": "moon"})[0], 400)

    def test_gpu_chosen_but_no_card_has_room_runs_on_the_cpu_and_says_so(self):
        with mock.patch.dict(os.environ, {"NEBULA_FAKE_CARDS": json.dumps([{"uuid": "GPU-a", "index": 0, "name": "x", "memory_mb": 8192, "used_mb": 7800}])}):
            self.srv.halo_engine.restart()
            self.wait()
        v = self.srv.halo_engine.view()
        self.assertEqual((v["where"], v["runs_on"]), ("cpu", "gpu"))
        self.assertIn("no GPU has room", v["why"])

    def test_no_gpu_at_all_says_so(self):
        with mock.patch.dict(os.environ, {"NEBULA_FAKE_CARDS": "[]"}):
            self.srv.halo_engine.restart()
            self.wait()
        self.assertEqual(self.srv.halo_engine.view()["why"], "no NVIDIA GPU on this machine: on the CPU")
