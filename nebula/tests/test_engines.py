"""The engines Nebula runs itself (nebula/engines.py, nebula/yard.py): deployments, the fit estimate, cards, and the
lifecycle (start, route, verify, drain, restart, undo) against the pretend Impulse binary."""
import json
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from nebula import engines as E
from nebula.tests import WINDOWS, linux_only  # noqa: F401
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine

ROOT = str(Path(__file__).resolve().parents[2])
FAKE = [sys.executable, "-m", "nebula.tests.fake_impulse"]
CARDS = [{"uuid": "GPU-aaaa", "index": 0, "name": "Pretend 16G", "memory_mb": 16384, "used_mb": 1024},
         {"uuid": "GPU-bbbb", "index": 1, "name": "Pretend 8G", "memory_mb": 8192, "used_mb": 512}]


def make_gguf(path, layers=32, kv_heads=8, head=128, arch="llama", pad_mb=1):
    """A GGUF v3 file with the metadata the fit estimate reads (and one array it must skip), padded to `pad_mb`."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [(f"general.architecture", 8, s(arch)),
          ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + s("a") + s("b")),
          ("general.alignment", 4, struct.pack("<I", 32)),
          (f"{arch}.block_count", 4, struct.pack("<I", layers)),
          (f"{arch}.attention.head_count", 4, struct.pack("<I", kv_heads * 4)),
          (f"{arch}.attention.head_count_kv", 4, struct.pack("<I", kv_heads)),
          (f"{arch}.attention.key_length", 4, struct.pack("<I", head)),
          (f"{arch}.embedding_length", 4, struct.pack("<I", kv_heads * 4 * head))]
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for k, t, v in kv:
        body += s(k) + struct.pack("<I", t) + v
    with open(path, "wb") as f:
        f.write(body)
        f.write(b"\0" * (pad_mb * 2**20))


from nebula.tests import free_port  # noqa: E402,F401  (outside the system's outgoing ports)


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_for(fn, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.1)
    return False


class SpecTest(unittest.TestCase):
    def test_checks(self):
        with self.assertRaises(E.EngineSpecError):
            E.check_spec({"name": "a", "binary": "x", "model": "m.gguf", "port": 1, "bogus": 1})
        with self.assertRaises(E.EngineSpecError):
            E.check_spec({"name": "a", "kind": "vllm", "port": 1})
        with self.assertRaises(E.EngineSpecError):
            E.check_spec({"name": "a", "binary": "x", "model": "m.gguf"})            # no port
        with self.assertRaises(E.EngineSpecError):
            E.check_spec({"name": "a", "binary": "x", "model": "m.gguf", "port": 1, "cache_type": "q3"})
        s = E.check_spec({"name": "a", "command": ["srv", "--port", "9"], "port": 9})
        self.assertEqual((s["kind"], s["host"]), ("command", "127.0.0.1"))

    def test_command(self):
        s = E.check_spec({"name": "a", "kind": "impulse", "binary": "/opt/impulse/strata-lite-server", "model": "m.gguf",
                          "port": 8080, "context": 32768, "slots": 2, "cache_type": "q8_0", "flags": ["--flash-attn", "on"]})
        self.assertEqual(E.command_of(s, "/cfg"), ["/opt/impulse/strata-lite-server", "-m", str(Path("/cfg", "m.gguf")), "--host", "127.0.0.1",
                                                   "--port", "8080", "-c", "65536", "-np", "2", "-ctk", "q8_0", "-ctv", "q8_0",
                                                   "--no-webui", "--flash-attn", "on"])
        s["binary"] = ["python3", "srv.py"]
        self.assertEqual(E.command_of(s)[:3], ["python3", "srv.py", "-m"])


class FitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.model = os.path.join(self.tmp, "small.gguf")
        make_gguf(self.model, layers=32, kv_heads=8, head=128, pad_mb=4)

    def test_header_and_estimate(self):
        m = E.gguf_meta(self.model)
        self.assertEqual((m["block_count"], m["attention.head_count_kv"], m["attention.key_length"], m["general.architecture"]),
                         (32, 8, 128, "llama"))
        s = {"name": "a", "kind": "impulse", "binary": "x", "model": self.model, "port": 1, "context": 8192, "slots": 2,
             "card": "GPU-bbbb"}
        f = E.fit(s, have=CARDS)
        # 32 layers × 8 heads × (128 + 128) × 2 bytes (f16) × 16384 tokens = 2048 MiB
        self.assertEqual((f["kv_mb"], f["card"], f["free_mb"], f["fits"]), (2048, "GPU-bbbb", 7680, True))
        s["context"] = 65536                                               # 16 GiB of cache: not on 8 GB
        f = E.fit(s, have=CARDS)
        self.assertFalse(f["fits"])
        self.assertIn("MB free", f["why"])
        s["cache_type"] = "q4_0"
        self.assertLess(E.fit(s, have=CARDS)["kv_mb"], 16384 // 3)
        s["card"] = "GPU-zzzz"
        self.assertEqual(E.fit(s, have=CARDS)["why"], "this machine has no card GPU-zzzz")
        s["card"] = "GPU-aaaa, GPU-bbbb"                                   # a model split across two cards: both count
        f = E.fit(s, have=CARDS)
        self.assertEqual((f["card"], f["free_mb"]), ("GPU-aaaa,GPU-bbbb", 15360 + 7680))
        s["cpus"] = "0-3, 8"
        self.assertEqual(E.command_of(E.check_spec(dict(s, name="p", port=9)))[:3], ["taskset", "-c", "0-3,8"])
        with self.assertRaises(E.EngineSpecError):
            E.check_spec(dict(s, name="p", port=9, cpus="node0"))
        s.pop("cpus")
        s["card"] = "GPU-aaaa,GPU-zzzz"
        self.assertEqual(E.fit(s, have=CARDS)["why"], "this machine has no card GPU-zzzz")

    def test_fake_cards(self):
        os.environ["NEBULA_FAKE_CARDS"] = json.dumps(CARDS)
        try:
            self.assertEqual([c["uuid"] for c in E.cards()], ["GPU-aaaa", "GPU-bbbb"])
        finally:
            del os.environ["NEBULA_FAKE_CARDS"]


def make_flint_gguf(path, experts_mb=12, other_mb=2):
    """A small `flint` GGUF: its hparams, and tensors whose offsets give the routed experts' and the other weights' bytes."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    u = lambda k, v: (f"flint.{k}", 4, struct.pack("<I", v))
    kv = [("general.architecture", 8, s("flint")), ("general.alignment", 4, struct.pack("<I", 32)),
          u("block_count", 4), u("attention.head_count", 32), u("attention.head_count_kv", 4), u("attention.key_length", 128),
          u("attention.value_length", 128), u("embedding_length", 2048), u("expert_count", 8),
          ("flint.attention.sliding_window_pattern", 9, struct.pack("<IQ", 7, 4) + bytes([0, 0, 1, 1]))]
    MB = 2**20
    tensors = [("token_embd.weight", other_mb // 2 * MB), ("blk.2.ffn_up_exps.weight", experts_mb // 2 * MB),
               ("blk.3.ffn_up_exps.weight", experts_mb // 2 * MB), ("output.weight", other_mb // 2 * MB)]
    body = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        body += s(k) + struct.pack("<I", t) + v
    off = 0
    for name, size in tensors:
        body += s(name) + struct.pack("<I", 1) + struct.pack("<Q", size) + struct.pack("<I", 0) + struct.pack("<Q", off)
        off += size
    body += b"\0" * ((-len(body)) % 32)
    with open(path, "wb") as f:
        f.write(body)
        f.write(b"\0" * off)


class FlintFitTest(unittest.TestCase):
    """Flint on Impulse: the card holds what is not a routed expert, the KV cache and the smallest expert cache; RAM holds
    the experts. A start is refused (fits False, a plain reason) when either is short, before anything loads."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.m = os.path.join(self.tmp, "flint.gguf")
        make_flint_gguf(self.m)
        self.card = {"uuid": "a", "index": 0, "memory_mb": 8192, "used_mb": 0}

    def tearDown(self):
        os.environ.pop("NEBULA_FAKE_RAM_FREE_MB", None)

    def spec(self, **kw):
        return {"name": "f", "kind": "impulse", "model": self.m, "context": 8192, "slots": 1, **kw}

    def test_the_file_splits_into_experts_and_the_rest(self):
        tb = E.gguf_tensor_bytes(self.m)
        self.assertEqual((tb["experts"], tb["other"], tb["expert_layers"]), (12 * 2**20, 2 * 2**20, 2))

    def test_one_card_and_the_experts_in_ram(self):
        os.environ["NEBULA_FAKE_RAM_FREE_MB"] = "64000"
        f = E.fit(self.spec(), None, [self.card])
        kv = round(4 * 4 * 128 * 2 * E.CACHE_BYTES["q4_0"] * 8192 / 2**20)          # every layer, q4_0 (Impulse's default)
        self.assertEqual((f["fits"], f["kv_mb"], f["weights_mb"], f["ram_need_mb"]), (True, kv, 2, 12 + E.RAM_OVERHEAD_MB + 2 * kv))
        self.assertEqual(f["card"], "a")                                                # no card chosen: the one with the most free
        self.assertGreater(f["expert_cache_mb"], 7000)
        self.assertEqual(f["expert_cache_share"], 1.0)

    def test_short_ram_is_refused_with_a_plain_reason(self):
        os.environ["NEBULA_FAKE_RAM_FREE_MB"] = "2048"
        f = E.fit(self.spec(), None, [self.card])
        self.assertFalse(f["fits"])
        self.assertIn("GB of free RAM for its experts", f["why"])

    def test_short_ram_under_wsl_says_how_to_raise_it(self):
        os.environ["NEBULA_FAKE_RAM_FREE_MB"] = "2048"
        os.environ["NEBULA_FAKE_WSL"] = "1"
        try:
            f = E.fit(self.spec(), None, [self.card])
        finally:
            os.environ.pop("NEBULA_FAKE_WSL")
        self.assertFalse(f["fits"])
        for w in ("free RAM for its experts", "WSL", "memory=", ".wslconfig", "wsl --shutdown"):
            self.assertIn(w, f["why"])
        os.environ["NEBULA_FAKE_WSL"] = "0"
        try:
            self.assertNotIn("wslconfig", E.fit(self.spec(), None, [self.card])["why"])
        finally:
            os.environ.pop("NEBULA_FAKE_WSL")

    def test_a_small_card_or_a_long_context_is_refused(self):
        os.environ["NEBULA_FAKE_RAM_FREE_MB"] = "64000"
        f = E.fit(self.spec(context=262144), None, [dict(self.card, memory_mb=1024)])
        self.assertFalse(f["fits"])
        self.assertIn("KV cache", f["why"])

    def test_all_on_two_cards_needs_no_ram_for_experts(self):
        os.environ["NEBULA_FAKE_RAM_FREE_MB"] = "100"
        f = E.fit(self.spec(card="a,b", options={"n_cpu_moe": 0, "tensor_split": "16,12"}), None,
                  [self.card, {"uuid": "b", "index": 1, "memory_mb": 12288, "used_mb": 0}])
        self.assertEqual((f["fits"], f["ram_need_mb"], f["weights_mb"]), (True, 2 * f["kv_mb"], 14))


class LifecycleTest(unittest.TestCase):
    """A Nebula with two managed engines (the pretend Impulse binary) and a config file."""

    def setUp(self):
        os.environ["NEBULA_FAKE_CARDS"] = json.dumps(CARDS)
        self.tmp = tempfile.mkdtemp()
        make_gguf(os.path.join(self.tmp, "small.gguf"))
        make_gguf(os.path.join(self.tmp, "other.gguf"))
        self.p1, self.p2 = free_port(), free_port()
        eng = lambda name, port, model: {"name": name, "kind": "impulse", "binary": FAKE, "model": model, "port": port,
                                         "context": 4096, "slots": 2, "card": "GPU-aaaa", "cwd": ROOT,
                                         "ready_timeout_s": 30, "stop_timeout_s": 5}
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({"port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
                                               "engines": [eng("one", self.p1, "small.gguf"), eng("two", self.p2, "other.gguf")],
                                               "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.srv.tools.close()
        del os.environ["NEBULA_FAKE_CARDS"]

    def state(self, name):
        return call(self.base, f"/api/yard/engines/{name}")[1]["state"]

    def test_start_route_verify_chat_stop(self):
        st, v = call(self.base, "/api/yard/engines")
        self.assertEqual(([e["name"] for e in v["engines"]], [c["uuid"] for c in v["cards"]]), (["one", "two"], ["GPU-aaaa", "GPU-bbbb"]))
        self.assertEqual(v["engines"][0]["state"], "stopped")
        self.assertTrue(v["engines"][0]["fit"]["fits"])
        st, out = call(self.base, "/api/yard/engines/one/start", {"wait": True})
        self.assertEqual((st, out["state"]), (200, "ready"), out)
        self.assertTrue(any("loading small.gguf" in ln for ln in out["lines"]))
        st, r = call(self.base, "/api/yard/engines/one/route", {})
        self.assertEqual((st, r["tier"], r["created"]), (200, "one", True))
        self.assertEqual(json.loads(Path(self.path).read_text())["tiers"][-1]["managed"], "one")   # saved
        ok = call(self.base, "/api/yard/engines/one/verify", {})[1]
        self.assertEqual((ok["health"], ok["model_ok"], ok["ok"]), (200, True, True))
        st, body = call(self.base, "/v1/chat/completions", {"model": "one", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200, body)
        pid = call(self.base, "/api/yard/engines/one")[1]["pid"]
        self.assertEqual(call(self.base, "/api/yard/engines/one/stop", {})[1]["state"], "stopped")
        self.assertTrue(wait_for(lambda: not os.path.exists(f"/proc/{pid}") or open(f"/proc/{pid}/stat").read().split()[2] == "Z"))

    def test_a_running_engine_elsewhere_is_attached_not_started_twice(self):
        other = FakeEngine("someone-elses", port=self.p2)
        try:
            out = call(self.base, "/api/yard/engines/two/start", {"wait": True})[1]
            self.assertEqual(out["state"], "attached")
            self.assertIsNone(out["pid"])
        finally:
            other.shutdown(); other.server_close()

    def test_a_missing_card_or_a_failed_load_is_refused_with_the_reason(self):
        spec = dict(self.srv.yard.spec("two"), card="GPU-zzzz")
        call(self.base, "/api/yard/engines", spec)
        st, err = call(self.base, "/api/yard/engines/two/start", {})
        self.assertEqual((st, "no card GPU-zzzz" in err["error"]["message"]), (400, True), err)
        spec = dict(spec, card="GPU-bbbb", env={"FAKE_IMPULSE_FAIL": "1"})
        call(self.base, "/api/yard/engines", spec)
        out = call(self.base, "/api/yard/engines/two/start", {"wait": True})[1]
        self.assertEqual(out["state"], "failed")
        self.assertIn("exit code 3", out["error"])

    def test_deployments_and_undo(self):
        spec = self.srv.yard.spec("two")
        st, out = call(self.base, "/api/yard/engines", dict(spec, context=8192))
        self.assertEqual((st, out["version"], out["changed"]), (200, 1, True))
        out = call(self.base, "/api/yard/engines", dict(spec, context=16384))[1]
        self.assertEqual(out["version"], 2)
        self.assertEqual(call(self.base, "/api/yard/engines", dict(spec, context=16384))[1]["changed"], False)
        out = call(self.base, "/api/yard/engines/two/undo", {})[1]
        self.assertEqual(self.srv.yard.spec("two")["context"], 8192)
        self.assertEqual(len(call(self.base, "/api/yard/engines/two")[1]["history"]), 3)
        self.assertEqual(json.loads(Path(self.path).read_text())["engines"][1]["context"], 8192)   # saved
        self.assertTrue(Path(self.path).with_suffix(".deployments.json").exists())
        clash = dict(spec, name="three")                                       # the same port as "two"
        self.assertEqual(call(self.base, "/api/yard/engines", clash)[0], 400)

    def test_drain_skips_the_engine_then_stops_it(self):
        for n in ("one", "two"):
            self.assertEqual(call(self.base, f"/api/yard/engines/{n}/start", {"wait": True})[1]["state"], "ready")
            call(self.base, f"/api/yard/engines/{n}/route", {})
        self.srv.cfg.default_models = ["one", "two"]
        gate = self.srv.admission.gates[f"http://127.0.0.1:{self.p1}"]
        gate.held += 1                                         # a call still running on "one"
        out = call(self.base, "/api/yard/engines/one/drain", {"timeout_s": 30})[1]
        self.assertEqual(out["state"], "draining")
        st, body = call(self.base, "/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200, body)
        d = self.srv.ledger.db.execute("SELECT tier, inputs FROM decisions ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        self.assertEqual(d[0], "two")
        self.assertEqual(self.state("one"), "draining")        # still running its call
        gate.held -= 1
        self.assertTrue(wait_for(lambda: self.state("one") == "stopped"))
        self.assertEqual(self.state("two"), "ready")

    def test_autostart(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        d = json.loads(Path(self.path).read_text())
        d["engines"][0]["autostart"] = True
        Path(self.path).write_text(json.dumps(d))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        self.assertTrue(wait_for(lambda: self.state("one") == "ready"))
        self.assertEqual(self.state("two"), "stopped")


class SpeculativeTest(unittest.TestCase):
    """Speculative decoding for every engine that has it (design decision 2026-10-20: MTP not just for Impulse), each in its
    own flag names: Impulse's (--mtp, -md, --draft-max) and llama.cpp's since its rename (--spec-type, --spec-draft-*)."""

    def test_flags_per_engine(self):
        from nebula import packages as P
        self.assertEqual(P.args("impulse", {"mtp": "/m/mtp.gguf", "draft_max": 4}, "/m/x.gguf"), ["--mtp", "/m/mtp.gguf", "--n-draft", "4"])   # impulse-server --help
        self.assertEqual(P.args("llama.cpp", {"spec_type": "mtp", "draft_max": 6}, "/m/x.gguf", "/app/llama-server"),
                         ["--spec-type", "draft-mtp", "--spec-draft-n-max", "6"])
        self.assertEqual(P.args("llama.cpp", {"spec_type": "draft", "draft_model": "/m/d.gguf", "draft_gpu_layers": 99}, "/m/x.gguf", "/app/llama-server"),
                         ["--spec-type", "draft-simple", "--spec-draft-model", "/m/d.gguf", "--spec-draft-ngl", "99"])
        self.assertEqual(P.args("llama.cpp", {"spec_type": "ngram"}, "/m/x.gguf", "/app/llama-server"), ["--spec-type", "ngram-simple"])
        self.assertEqual(P.args("llama.cpp", {"spec_type": "off", "draft_max": 3}, "/m/x.gguf", "/app/llama-server"), [])
        for bad in ({"spec_type": "ngram"}, {"spec_type": "eagle3"}, {"spec_type": "draft"}, {"spec_type": "mtp"}, {"draft_model": "d"}):
            with self.assertRaises(P.OptionError):
                P.check("impulse", bad)
        with self.assertRaises(P.OptionError):
            P.check("llama.cpp", {"spec_type": "draft"})                 # a draft model is needed
        self.assertIn("spec_type", [f["key"] for f in P.schema("llama.cpp")])
        self.assertIn("Speculative decoding", P.GROUPS)

    @linux_only

    def test_an_older_llama_cpp_gets_the_older_names(self):
        import tempfile
        from nebula import packages as P
        d = tempfile.mkdtemp(); old = os.path.join(d, "llama-server")
        Path(old).write_text("#!/bin/sh\necho '-md, --model-draft FNAME'\necho '--draft-max N'\n"); os.chmod(old, 0o755)
        self.assertEqual(P.args("llama.cpp", {"spec_type": "draft", "draft_model": "/m/d.gguf", "draft_max": 5}, "/m/x.gguf", old),
                         ["-md", "/m/d.gguf", "--draft-max", "5"])

    def test_raw_lines_read_back(self):
        from nebula import packages as P
        o = P.parse_command("llama.cpp", "llama-server -m x.gguf --spec-type draft-mtp --spec-draft-model mtp.gguf --spec-draft-n-max 3")["spec"]["options"]
        self.assertEqual((o["spec_type"], o["mtp"], o["draft_max"]), ("mtp", "mtp.gguf", 3))
        o = P.parse_command("impulse", "impulse-server -m x.gguf --mtp d.gguf --n-draft 4")["spec"]["options"]
        self.assertEqual((o["spec_type"], o["mtp"], o["draft_max"]), ("mtp", "d.gguf", 4))


if __name__ == "__main__":
    unittest.main()
