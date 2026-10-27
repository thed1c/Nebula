"""Impulse Flash-Next (nebula/flashnext.py): a pack folder is told from a GGUF, and the run config Nebula writes for
The reference engine's pack is the one the reference engine runs on the test server (data/reference-run-config-fx-2026-10-25.json: its image
nebula-flashnext:impulse-36f62a1-r768's /opt/flashnext/run-config-fx.json, the live run config minus "exe", read on the test server
2026-10-25), started as the reference engine's engine definition starts impulse-fx since One Engine P3."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from nebula import flashnext as F

LIVE = json.loads((Path(__file__).parent / "data" / "reference-run-config-fx-2026-10-25.json").read_text())
# The reference engine's live engine definition (a two-GPU command engine): command and env
LIVE_COMMAND = ["/opt/impulse/bin/impulse-fx", "-m", "/models/orca-flashnext-nvfp4-gptq-strata-2692e6a", "--host", "127.0.0.1",
                "--port", "18123", "--fx-image-python", "/opt/strata-serve/.venv/bin/python",
                "--fx-config", "/opt/flashnext/run-config-fx.json"]
LIVE_ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "STRATA_PREFILL_LEND_PCT": "95", "LD_PRELOAD": "/opt/nebula/libnebula-selfpin.so",
            "IMPULSE_NUMA_NODES": "0=0-17,36-53;1=18-35,54-71"}
PACK = "orca-flashnext-nvfp4-gptq-strata-2692e6a"
NUMA = "0=0-17,36-53;1=18-35,54-71"


def make_pack(root: str, name: str = PACK) -> str:
    d = Path(root) / name
    (d / "pack" / "tokenizer").mkdir(parents=True)
    (d / "mtp").mkdir()
    for f in ("orca-nvfp4-dense.gguf", "ple-fp8.gguf", "token-embd-bf16.gguf"):
        (d / f).write_bytes(b"GGUF")
    return str(d)


class TestFlashNext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.pack = make_pack(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pack_or_not(self):
        self.assertTrue(F.is_pack(self.pack))
        gguf = Path(self.root) / "model.gguf"
        gguf.write_bytes(b"GGUF")
        self.assertFalse(F.is_pack(gguf))
        self.assertFalse(F.is_pack(self.root))                     # a folder of packs is not a pack
        files = F.pack_files(self.pack)
        self.assertEqual(os.path.basename(files["dense"]), "orca-nvfp4-dense.gguf")
        self.assertEqual(os.path.basename(files["ple"]), "ple-fp8.gguf")
        self.assertEqual(os.path.basename(files["embd"]), "token-embd-bf16.gguf")

    def reference_spec(self, **kw):
        return F.engine_spec("FlashNext", machine="Server", folder=self.root, files=F.pack_files(self.pack), port=18123,
                             cards=["GPU-5070", "GPU-5080"], image="nebula-impulse-flashnext:test", numa_nodes=NUMA,
                             model_name="flash-next", aliases=LIVE["aliases"], vision=LIVE["vision"], **F.REFERENCE_SETTINGS, **kw)

    def inline_config(self, s):
        c = s["command"]
        self.assertEqual(c.count("--fx-config-json"), 1)
        self.assertNotIn("--fx-config", c)
        return json.loads(c[c.index("--fx-config-json") + 1])

    def test_reference_run_config_is_the_live_one(self):
        cfg = self.inline_config(self.reference_spec())
        self.assertEqual(cfg["args"], LIVE["args"])
        self.assertEqual(cfg, LIVE)

    def test_reference_definition_is_the_live_one(self):
        """With the config the image carries, the command and env are the reference engine's live ones (P3's spec-impulse.json)."""
        s = self.reference_spec(config_path="/opt/flashnext/run-config-fx.json")
        self.assertEqual(s["command"], LIVE_COMMAND)
        self.assertEqual(s["env"], LIVE_ENV)

    def test_engine_definition(self):
        s = self.reference_spec()
        self.assertEqual(s["kind"], "command")
        c = s["command"]
        self.assertEqual(c[:9], LIVE_COMMAND[:9])                    # impulse-fx, the pack, host, port, image python
        self.assertEqual(c[c.index("--fx-config-json") + 1:], [json.dumps(LIVE, separators=(",", ":"))])
        self.assertNotIn("IMPULSE_FLASHNEXT_CONFIG", s["env"])
        self.assertEqual(s["env"]["LD_PRELOAD"], F.SELFPIN)
        self.assertEqual(s["env"]["IMPULSE_NUMA_NODES"], NUMA)
        self.assertEqual(s["card"], "GPU-5070,GPU-5080")
        self.assertEqual((s["machine"], s["where"], s["port"]), ("Server", "openshell", 18123))

    def test_nebula_accepts_it(self):
        from nebula import engines as E
        s = E.check_spec(self.reference_spec())
        self.assertEqual((s["kind"], s["where"]), ("command", "openshell"))
        self.assertEqual(E.sandbox_plan(s), ("openshell", None))

    def test_one_card_no_split_no_numa(self):
        cfg = F.run_config(F.inside(F.pack_files(self.pack), self.root), name="flash", port=18200)
        self.assertNotIn("layer_split", cfg)
        self.assertNotIn("IMPULSE_NUMA_NODES", cfg["env"])
        self.assertNotIn("--pool-workers", cfg["args"])

    def test_no_reference_tuning_by_default(self):
        """the reference engine's settings never reach another machine's engine unless passed."""
        cfg = F.run_config(F.inside(F.pack_files(self.pack), self.root), name="flash", port=18200)
        for flag in ("--pcie-frac", "--vram-reserve-mib", "--vram-reserve-later-mib", "--kv-resident", "--conversation-cache-mib",
                     "--conversation-cache-slots", "--max-context", "--prefill", "--pool-workers"):
            self.assertNotIn(flag, cfg["args"])
        self.assertNotIn("STRATA_PREFILL_LEND_PCT", cfg["env"])
        spec = F.engine_spec("flash", machine=None, folder=self.root, files=F.pack_files(self.pack), port=18200, cards=["GPU-x"])
        self.assertNotIn("STRATA_PREFILL_LEND_PCT", spec["env"])
        inline = self.inline_config(spec)
        self.assertEqual(inline["args"], cfg["args"])
        self.assertEqual(spec["command"][:3], [F.FX, "-m", "/models/" + PACK])
        self.assertTrue(cfg["tokenizer"].startswith("/models/" + PACK + "/"))

    def test_outside_the_folder_is_refused(self):
        other = tempfile.TemporaryDirectory()
        try:
            with self.assertRaises(ValueError):
                F.inside(F.pack_files(self.pack), other.name)
        finally:
            other.cleanup()

    def test_parallel_default_is_one(self):
        cfg = F.run_config(F.inside(F.pack_files(self.pack), self.root), name="flash", port=18200)
        self.assertEqual(cfg["parallel"], 1)
        self.assertNotIn("--batch", cfg["args"])
        self.assertNotIn("--batch-max-context", cfg["args"])
        self.assertEqual(self.inline_config(self.reference_spec())["parallel"], 1)   # the reference engine stays one at a time

    def test_parallel_n(self):
        cfg = F.run_config(F.inside(F.pack_files(self.pack), self.root), name="flash", port=18200, parallel=2,
                           batch_max_context=32768)
        self.assertEqual(cfg["parallel"], 2)                     # impulse-fx adds --batch 2 itself
        self.assertNotIn("--batch", cfg["args"])
        i = cfg["args"].index("--batch-max-context")
        self.assertEqual(cfg["args"][i + 1], "32768")
        spec = F.engine_spec("flash", machine=None, folder=self.root, files=F.pack_files(self.pack), port=18200, cards=["GPU-x"],
                             parallel=4)
        self.assertEqual(self.inline_config(spec)["parallel"], 4)
        for bad in (0, 9):
            with self.assertRaises(ValueError):
                F.run_config(F.inside(F.pack_files(self.pack), self.root), name="flash", port=18200, parallel=bad)

    def test_slot_cost(self):
        """BATCHING.md's measured slots (int8 K/V): 0.56 GiB at 32K, 1.00 at 64K, 3.65 at 262K."""
        for cells, gib in ((32768, 0.56), (65536, 1.00), (262144, 3.65)):
            self.assertAlmostEqual(F.slot_cost(cells)["vram_mib"] / 1024, gib, delta=0.03)
        r = F.slot_cost(131072, kv_resident=20480)               # streamed K/V: the 32K window in VRAM, the rest pinned
        self.assertEqual(r["vram_mib"], F.slot_cost(32768)["vram_mib"])
        self.assertAlmostEqual(r["pinned_mib"] / 1024, 1.76, delta=0.05)
        self.assertGreater(F.slot_cost(32768)["expert_slots"], 200)

    def test_rope(self):
        files = F.inside(F.pack_files(self.pack), self.root)
        self.assertNotIn("rope", F.run_config(files, name="flash", port=18200))      # none unless asked (the reference engine)
        self.assertNotIn("rope", self.inline_config(self.reference_spec()))
        cfg = F.run_config(files, name="flash", port=18200, max_context=524288, rope={"type": "ntk", "factor": 2})
        self.assertEqual(cfg["rope"], {"type": "ntk", "factor": 2})
        for bad in ({"type": "longrope"}, {"kind": "yarn"}):
            with self.assertRaises(ValueError):
                F.run_config(files, name="flash", port=18200, rope=bad)

    def test_long_context(self):
        """the reference engine keeps 262144; the opt-in long context sets the context and the rope block, inline (not the image's file)."""
        self.assertEqual(F.REFERENCE_SETTINGS["max_context"], 262144)
        spec = F.engine_spec("FlashNext", machine="Server", folder=self.root, files=F.pack_files(self.pack), port=18123,
                             cards=["GPU-5070", "GPU-5080"], numa_nodes=NUMA, model_name="flash-next",
                             **{**F.REFERENCE_SETTINGS, **F.long_context("512k")})
        cfg = self.inline_config(spec)
        i = cfg["args"].index("--max-context")
        self.assertEqual(cfg["args"][i + 1], "524288")
        self.assertEqual(cfg["rope"], {"type": "dynamic", "factor": 2.0, "orig_ctx": 262144, "dynamic_step": 0.25})
        self.assertEqual(F.long_context("1m", "yarn")["rope"], {"type": "yarn", "factor": 4.0, "orig_ctx": 262144})
        for bad in (("2m", "dynamic"), ("512k", "ntk")):
            with self.assertRaises(ValueError):
                F.long_context(*bad)
        c = F.long_context_cost(1048576)
        self.assertEqual(len(c["cards_mib"]), 2)
        self.assertTrue(500 < c["cards_mib"][0] < 1000 and 900 < c["cards_mib"][1] < 1500, c)
        self.assertAlmostEqual(c["pinned_gib"], 12.9, delta=0.5)
        self.assertEqual(F.long_context_cost(262144)["cards_mib"], [0, 0])


if __name__ == "__main__":
    unittest.main()
