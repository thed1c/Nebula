"""The wizards (nebula/wizards.py): each one's happy path, Cancel (everything up to the summary) changes nothing, Undo
takes back what Apply did, and a step that fails takes back the ones before it."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nebula import gravity
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call
from nebula.tests.test_deploy import PretendHub, gguf_bytes
from nebula.tests.test_engines import FAKE, ROOT


def serve(d: dict, tmp=None):
    tmp = tmp or tempfile.mkdtemp()
    path = os.path.join(tmp, "nebula.json")
    Path(path).write_text(json.dumps({"port": 0, "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False, "tiers": [],
                                      "assistant": {"halo": {"on": False}},
                                      "lan": {"decided": True, "discoverable": False, "uuid": os.urandom(8).hex(), "name": "n"}, **d}))
    srv = make_server(Config.from_args(["--config", path]))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}", path


def stop(srv):
    srv.shutdown(); srv.server_close(); srv.tools.close()
    if srv.gravity is not None:
        srv.gravity.close()


def finish(base, job, s=60):
    end = time.time() + s
    while time.time() < end:
        v = call(base, f"/api/wizards/jobs/{job['id']}")[1]
        if v["state"] != "running":
            return v
        time.sleep(0.2)
    raise AssertionError(f"the job did not finish: {v}")


class Base(unittest.TestCase):
    def snapshot(self):
        return Path(self.path).read_bytes(), sorted(os.listdir(os.path.dirname(self.path)))

    def plan(self, name, answers):
        return call(self.base, f"/api/wizards/{name}/plan", {"answers": answers})

    def apply(self, name, answers):
        st, job = call(self.base, f"/api/wizards/{name}/apply", {"answers": answers})
        self.assertEqual(st, 200, job)
        return finish(self.base, job)


class ListTest(Base):
    def test_listed_with_their_steps_and_defaults(self):
        srv, base, _ = serve({})
        try:
            names = [w["name"] for w in call(base, "/api/wizards")[1]["wizards"]]
            self.assertEqual(names, ["first-run", "deploy", "routing", "join", "add-server", "entry", "share", "gateway", "collection"])
            w = call(base, "/api/wizards/first-run")[1]
            self.assertEqual(w["defaults"]["source"], "here")
            self.assertEqual([s["id"] for s in w["steps"]][0], "how")
            self.assertEqual(call(base, "/api/wizards/nope")[0], 404)
        finally:
            stop(srv)


class ModelWizardsTest(Base):
    """The catalog-first paths, against a pretend Hub that holds the catalog's own repositories (with tiny files)."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        g = lambda n: gguf_bytes(self.tmp, n)
        self.hub = PretendHub({"LiquidAI/LFM2.5-24B-A2B-GGUF": {"LFM2.5-24B-A2B-QAD-Q4_0.gguf": g("a")},
                               "HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP": {
                                   "Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf": g("b"),
                                   "mtp-gemma-4-26B-A4B-it.gguf": g("c"),
                                   "mmproj-Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf": g("d")},
                               "someone/tiny-GGUF": {"tiny-Q4_K_M.gguf": g("e")},
                               "unsloth/flint-GGUF": {"flint-UD-Q4_K_XL.gguf": g("f"), "flint-UD-Q6_K_XL.gguf": g("h")},
                               "dima6312/Flint-40B-A2.5B-abliterated-GGUF": {"Flint-40B-A2.5B-abliterated-Q4_K_M.gguf": g("i")}})
        self.env = {"HF_ENDPOINT": self.hub.url, "NEBULA_FAKE_CAPS": "8.9", "NEBULA_FAKE_RAM_MB": "65536", "NEBULA_FAKE_DISK_MB": "200000",
                    "NEBULA_FAKE_CARDS": json.dumps([{"uuid": "g", "index": 0, "name": "RTX test", "memory_mb": 12282, "used_mb": 0}])}
        os.environ.update(self.env)
        self.engine = FakeEngine("there")
        cfgdir = os.path.join(self.tmp, "cfg")
        os.makedirs(cfgdir)
        self.srv, self.base, self.path = serve({"models_dir": os.path.join(self.tmp, "models")}, cfgdir)

    def tearDown(self):
        stop(self.srv)
        self.hub.close()
        self.engine.stop()
        for k in self.env:
            os.environ.pop(k, None)
        os.environ.pop("NEBULA_IMPULSE_BIN", None)

    def test_the_catalog_recommends_one_that_fits(self):
        opts = call(self.base, "/api/wizards/first-run/options", {"field": "catalog", "answers": {}})[1]["options"]
        self.assertEqual([o["value"] for o in opts][0], "gemma4-26b")                   # 12 GB card, 64 GB RAM
        self.assertTrue(opts[0]["recommended"])
        qwen = next(o for o in opts if o["value"] == "qwen36-35b")
        self.assertTrue(qwen["lacks"])                                                    # wants a 16 GB card
        friends = call(self.base, "/api/wizards/join/options", {"field": "friend_catalog", "answers": {}})[1]["options"]
        self.assertEqual(sorted(o["value"] for o in friends), ["flint-40b", "flint-40b-q6", "gemma4-26b", "lfm25-24b", "qwen36-35b"])   # all offered to friends

    def test_run_here_one_click_cancel_apply_undo(self):
        answers = {"source": "here", "model": "gemma4-26b", "variant": "uncensored", "binary": FAKE, "cwd": ROOT}
        before = self.snapshot()
        st, p = self.plan("first-run", answers)
        self.assertEqual(st, 200, p)
        self.assertIn("Impulse 0.1.1", p["lines"][1])
        self.assertIn("uncensored", " ".join(p["warnings"]))
        self.assertEqual(self.snapshot(), before)                                        # Cancel: nothing changed
        v = self.apply("first-run", answers)
        self.assertEqual(v["state"], "done", v)
        self.assertTrue(v["result"]["test"]["ok"], v)                                    # it answered the hello
        spec = self.srv.cfg.engines[0]
        self.assertEqual(spec["kind"], "impulse")
        self.assertTrue(spec["options"]["mtp"].endswith("mtp-gemma-4-26B-A4B-it.gguf"))   # its drafter and its projector
        self.assertTrue(spec["options"]["mmproj_file"].endswith("BF16.gguf"))
        st, v = call(self.base, f"/api/wizards/jobs/{v['id']}/undo", {})
        self.assertEqual(st, 200, v)
        self.assertEqual((self.srv.cfg.engines, [t.name for t in self.srv.cfg.tiers]), ([], []))

    def test_flint_layouts(self):
        """Flint: one card holds the expert cache in front of the CPU pool; two cards that hold the whole model take it all."""
        from nebula import catalog as C
        m = C.get("flint-40b")
        c16 = {"uuid": "a", "index": 0, "memory_mb": 16376, "used_mb": 0, "compute_cap": "8.9"}
        c12 = {"uuid": "b", "index": 1, "memory_mb": 12288, "used_mb": 0, "compute_cap": "8.6"}
        t4 = {"uuid": "t", "index": 2, "memory_mb": 15360, "used_mb": 0, "compute_cap": "7.5"}
        lay = C.layout(m, [c12, c16])
        self.assertEqual((lay["id"], lay["card"]), ("all-on-cards", "a,b"))               # the most free memory first
        self.assertEqual(lay["options"], {"n_cpu_moe": 0, "tensor_split": "16,12"})
        self.assertEqual(C.layout(m, [c16, dict(c12, used_mb=6000)]), {"id": "one-card", "card": "a", "options": {},
                         "about": m["layouts"][1]["about"]})                             # one card: the expert cache, the experts in RAM
        self.assertEqual(C.layout(m, [c16, t4])["id"], "one-card")                        # a card Impulse does not run on
        self.assertIsNone(C.layout(m, [dict(c12, memory_mb=4096)]))                       # under 8 GB: no layout
        self.assertIsNone(C.layout(C.get("gemma4-26b"), [c16, c12]))                      # a model with no layouts
        self.assertEqual((m["licence"], m["download_gb"], m["sampling"]["temp"], m["sampling"]["top_k"]), ("Apache-2.0", 25.1, 1.0, 20))
        q6 = C.get("flint-40b-q6")
        self.assertEqual((q6["file"], q6["needs"]["ram_free_mb"], q6["context"]), ("flint-UD-Q6_K_XL.gguf", 37500, 32768))   # 32K by default      # Q6: more RAM
        self.assertEqual(C.layout(q6, [c16, c12])["id"], "one-card")                     # 36 GB: not all on 28 GB of cards
        for mid in ("flint-40b", "flint-40b-q6"):                                         # the uncensored build: dima6312's
            u = C.get(mid, "uncensored")
            self.assertEqual((u["repo"], u["content"], len(u["sha256"])), ("dima6312/Flint-40B-A2.5B-abliterated-GGUF", "uncensored build", 64))
            self.assertIsNone(C.get(mid, "standard").get("content"))
        self.assertEqual(m["needs"]["gpu_mb"], 7900)                                      # listed for 8 GB cards and up

    def test_flint_on_two_cards_with_the_card_sampling(self):
        from unittest import mock
        from nebula import catalog as C
        import hashlib
        cat = C.load()
        f = next(x for x in cat["models"] if x["id"] == "flint-40b")
        hub_file = self.hub.repos["unsloth/flint-GGUF"]["flint-UD-Q4_K_XL.gguf"]
        os.environ["NEBULA_FAKE_CARDS"] = json.dumps([{"uuid": "a", "index": 0, "name": "RTX 16", "memory_mb": 16376, "used_mb": 0},
                                                      {"uuid": "b", "index": 1, "name": "RTX 12", "memory_mb": 12288, "used_mb": 0}])
        answers = {"source": "here", "model": "flint-40b", "binary": FAKE, "cwd": ROOT}
        v = self.apply("first-run", answers)                                              # the real pin: the Hub's file is another
        self.assertEqual(v["state"], "failed", v)
        self.assertIn("not the file the catalog names", json.dumps(v))
        self.assertEqual(self.srv.cfg.engines, [])
        f["sha256"] = hashlib.sha256(hub_file).hexdigest()                              # the Hub's file
        with mock.patch.object(C, "load", lambda path=None: cat):
            st, p = self.plan("first-run", answers)
            self.assertEqual(st, 200, p)
            self.assertIn("2 cards together", " ".join(p["lines"]))
            self.assertNotIn("experts in RAM", " ".join(p["warnings"]))                  # all on the cards
            os.environ["NEBULA_FAKE_CARDS"] = json.dumps([{"uuid": "a", "index": 0, "name": "RTX 8", "memory_mb": 8192, "used_mb": 0}])
            os.environ["NEBULA_FAKE_RAM_FREE_MB"] = "8000"
            st, p1 = self.plan("first-run", answers)
            self.assertIn("needs about 26 GB of free RAM", " ".join(p1["warnings"]))
            self.assertIn("will not start", " ".join(p1["warnings"]))
            os.environ.pop("NEBULA_FAKE_RAM_FREE_MB")
            os.environ["NEBULA_FAKE_CARDS"] = json.dumps([{"uuid": "a", "index": 0, "name": "RTX 16", "memory_mb": 16376, "used_mb": 0},
                                                          {"uuid": "b", "index": 1, "name": "RTX 12", "memory_mb": 12288, "used_mb": 0}])
            v = self.apply("first-run", answers)
        self.assertEqual(v["state"], "done", v)
        spec = self.srv.cfg.engines[0]
        self.assertEqual((spec["kind"], spec["card"], spec["context"]), ("impulse", "a,b", 32768))
        o = spec["options"]
        self.assertEqual((o["n_cpu_moe"], o["tensor_split"], o["temp"], o["top_p"], o["top_k"], o["min_p"]), (0, "16,12", 1.0, 0.95, 20, 0.0))

    def test_a_split_model_pins_each_shard(self):
        """A catalog sha256 given per file name (a split model): each shard is checked against its own pin."""
        from unittest import mock
        from nebula import catalog as C
        import hashlib
        b = self.hub.repos["someone/tiny-GGUF"]["tiny-Q4_K_M.gguf"]
        cat = C.load()
        cat["models"].append({"id": "split", "name": "Split", "maker": "x", "about": "x", "repo": "someone/tiny-GGUF", "file": "tiny-Q4_K_M.gguf",
                              "sha256": {"tiny-Q4_K_M.gguf": hashlib.sha256(b).hexdigest()}, "download_gb": 0.001,
                              "needs": {"gpu_mb": 0, "ram_mb": 0, "disk_mb": 0}, "context": 2048, "variants": {"standard": {}}})
        with mock.patch.object(C, "load", lambda path=None: cat):
            v = self.apply("first-run", {"source": "here", "model": "split", "binary": FAKE, "cwd": ROOT})
            self.assertEqual(v["state"], "done", v)
            call(self.base, f"/api/wizards/jobs/{v['id']}/undo", {})
            cat["models"][-1]["sha256"] = {"tiny-Q4_K_M.gguf": "0" * 64}
            v = self.apply("first-run", {"source": "here", "model": "split", "binary": FAKE, "cwd": ROOT})
        self.assertEqual(v["state"], "failed", v)
        self.assertIn("not the file the catalog names", json.dumps(v))

    def test_flint_uncensored_says_so(self):
        st, p = self.plan("first-run", {"source": "here", "model": "flint-40b", "variant": "uncensored", "binary": FAKE})
        self.assertEqual(st, 200, p)
        self.assertIn("Flint 40B (Q4) is an uncensored build.", p["warnings"])
        unc = {o["value"]: o for o in call(self.base, "/api/wizards/first-run/options", {"field": "catalog", "answers": {"variant": "uncensored"}})[1]["options"]}
        self.assertEqual((unc["flint-40b"]["note"], unc["flint-40b-q6"]["note"]), ("uncensored build", "uncensored build"))

    def test_standard_or_uncensored(self):
        """The makers' own release (Standard, the default) or the uncensored build, per model."""
        d = call(self.base, "/api/wizards/first-run")[1]["defaults"]
        self.assertEqual(d["variant"], "standard")
        std = {o["value"]: o for o in call(self.base, "/api/wizards/first-run/options", {"field": "catalog", "answers": {}})[1]["options"]}
        unc = {o["value"]: o for o in call(self.base, "/api/wizards/first-run/options", {"field": "catalog", "answers": {"variant": "uncensored"}})[1]["options"]}
        self.assertIsNone(std["gemma4-26b"]["note"])
        self.assertEqual(unc["gemma4-26b"]["note"], "uncensored build")
        from nebula import catalog as C
        self.assertEqual(C.get("gemma4-26b", "standard")["repo"], "google/gemma-4-26B-A4B-it-qat-q4_0-gguf")
        self.assertEqual(C.get("lfm25-24b", "uncensored")["repo"], "AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF")
        st, p = self.plan("first-run", {"source": "here", "model": "gemma4-26b", "variant": "standard", "binary": FAKE})
        self.assertEqual(st, 200, p)
        self.assertNotIn("uncensored", " ".join(p["warnings"]))

    def test_no_impulse_here_is_said_plainly(self):
        os.environ["NEBULA_FAKE_CAPS"] = "8.9"
        st, e = self.plan("first-run", {"source": "here", "model": "lfm25-24b"})
        self.assertEqual(st, 409, e)
        self.assertIn("Impulse isn't on this computer yet", e["error"]["message"])
        os.environ["NEBULA_FAKE_CAPS"] = "6.1"                                           # an old card: Impulse cannot run
        from unittest import mock
        with mock.patch("nebula.pieces.pin", return_value=None):                          # and no Impulse portable to fetch
            st, e = self.plan("first-run", {"source": "here", "model": "lfm25-24b"})
            self.assertEqual(st, 409, e)
            self.assertIn("RTX 30, 40 and 50", e["error"]["message"])
            st, p = self.plan("first-run", {"source": "here", "model": "lfm25-24b", "binary": FAKE})   # llama.cpp stands in
            self.assertEqual(st, 200, p)
            self.assertIn("llama.cpp", p["lines"][1])

    def test_a_server_already_running_and_a_failed_start_leaves_nothing(self):
        v = self.apply("first-run", {"source": "other", "other": "running", "url": self.engine.url, "name": "first"})
        self.assertEqual(v["state"], "done", v)
        call(self.base, f"/api/wizards/jobs/{v['id']}/undo", {})
        self.assertIsNone(self.srv.cfg.tier("first"))
        v = self.apply("first-run", {"source": "here", "model": "lfm25-24b", "binary": "/nonexistent/impulse-server"})
        self.assertEqual(v["state"], "failed", v)
        self.assertEqual((self.srv.cfg.engines, [t.name for t in self.srv.cfg.tiers]), ([], []))   # taken back

    def test_deploy_advanced_from_a_link(self):
        answers = {"advanced": True, "link": "https://huggingface.co/someone/tiny-GGUF", "quant": "tiny-Q4_K_M.gguf", "package": "impulse",
                   "context": 2048, "binary": FAKE, "cwd": ROOT, "name": "tiny"}
        self.assertEqual(self.plan("deploy", answers)[0], 200)
        v = self.apply("deploy", answers)
        self.assertEqual(v["state"], "done", v)
        self.assertTrue(v["result"]["test"]["ok"], v)
        call(self.base, f"/api/wizards/jobs/{v['id']}/undo", {})
        self.assertEqual(self.srv.cfg.engines, [])

    def test_bad_answers_are_said_plainly(self):
        st, e = self.plan("first-run", {"source": "other", "other": "provider", "base_url": "http://example.com", "model": "m"})
        self.assertEqual(st, 400)
        self.assertIn("https://", e["error"]["message"])
        self.assertEqual(self.plan("deploy", {"model": "nope"})[0], 400)
        self.assertEqual(self.plan("join", {"invite": "hello"})[0], 400)


@unittest.skipUnless(gravity.available()[0], "Gravity needs iroh")
class ShareTest(Base):
    def setUp(self):
        self.engine = FakeEngine("big")
        self.srv, self.base, self.path = serve({"tiers": [{"name": "big", "engine": self.engine.url},
                                                          {"name": "cloud", "provider": "openai-compatible", "base_url": "https://api.example.com/v1", "model": "m"}]})
        self.friend, self.fbase, _ = serve({})

    def tearDown(self):
        stop(self.srv); stop(self.friend)
        self.engine.stop()

    def test_invite_with_models_confirm_join_undo(self):
        opts = {o["value"]: o for o in call(self.base, "/api/wizards/share/options", {"field": "shareable", "answers": {}})[1]["options"]}
        self.assertTrue(opts["cloud"]["disabled"])                                 # Anti-Gravity
        self.assertEqual(self.plan("share", {"friend": "Sam", "models": ["cloud"]})[0], 400)
        before = self.snapshot()
        answers = {"friend": "Sam", "models": ["big"]}
        st, p = self.plan("share", answers)
        self.assertEqual(st, 200, p)
        self.assertEqual(p["confirm"], ["big"])                                   # Nebula does not run its engine: care
        self.assertEqual(self.snapshot(), before)                                 # Cancel: nothing
        self.assertEqual(call(self.base, "/api/wizards/share/apply", {"answers": answers})[0], 409)   # not confirmed
        v = self.apply("share", {**answers, "confirm": ["big"]})
        self.assertEqual(v["state"], "done", v)
        self.assertTrue(self.srv.cfg.tier("big").shared)
        call(self.fbase, "/api/yard/node/on", {})
        st, j = call(self.fbase, "/api/yard/node/join", {"invite": v["result"]["invite"]})
        self.assertEqual(st, 200, j)
        self.assertEqual(self.srv.gravity.peers()[0]["grants"], ["big"])          # given on joining
        st, v2 = call(self.base, f"/api/wizards/jobs/{v['id']}/undo", {})
        self.assertEqual(st, 200, v2)
        self.assertFalse(self.srv.cfg.tier("big").shared)
        self.assertIsNone(self.srv.gravity)                                       # it turned Gravity on; undo turned it off

    def test_the_friend_joins_with_the_join_wizard(self):
        v = self.apply("share", {"friend": "Sam", "models": ["big"], "confirm": ["big"]})
        invite = v["result"]["invite"]
        st, p = call(self.fbase, "/api/wizards/join/plan", {"answers": {"invite": invite}})
        self.assertEqual(st, 200, p)
        self.assertIn("Turn on Gravity", p["lines"][0])
        st, job = call(self.fbase, "/api/wizards/join/apply", {"answers": {"invite": invite}})
        job = finish(self.fbase, job)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(len(job["result"]["models"]), 1)
        self.assertTrue(job["result"]["test"]["ok"], job)                               # the shared model answered
        st, v2 = call(self.fbase, f"/api/wizards/jobs/{job['id']}/undo", {})
        self.assertEqual(st, 200, v2)
        self.assertEqual([t.name for t in self.friend.cfg.tiers], [])                    # its models gone, the peer left

    def test_an_unused_invite_is_cancelled_by_undo(self):
        v = self.apply("share", {"friend": "Sam", "models": ["big"], "confirm": ["big"]})
        self.assertEqual(len(self.srv.cfg.raw["gravity"]["invites"]), 1)
        call(self.base, f"/api/wizards/jobs/{v['id']}/undo", {})
        self.assertEqual(self.srv.cfg.raw["gravity"]["invites"], [])             # the invite no longer works



class RadeonTest(Base):
    """No NVIDIA card (the owner's 2026-10-27 "Let's add Vulkan support"): a GGUF runs on Impulse portable, on a Radeon
    through Vulkan as far as it fits and the rest in RAM, said plainly; Flash-Next is refused, saying why."""
    RADEON = [{"id": "Vulkan0", "name": "Intel(R) UHD Graphics", "memory_mb": 8072, "free_mb": 7413, "integrated": True},
              {"id": "Vulkan1", "name": "AMD Radeon RX 5500M", "memory_mb": 4080, "free_mb": 3900, "integrated": False}]

    def setUp(self):
        ModelWizardsTest.setUp(self)                        # the same pretend Hub and Nebula, without an NVIDIA card
        self.more = {"NEBULA_FAKE_CAPS": "", "NEBULA_FAKE_CARDS": "[]", "NEBULA_FAKE_VULKAN": json.dumps(self.RADEON)}
        os.environ.update(self.more)
        self.env.update(self.more)

    def tearDown(self):
        ModelWizardsTest.tearDown(self)
        os.environ.pop("NEBULA_IMPULSE_PORTABLE_BIN", None)

    def portable(self) -> str:
        """A stand-in Impulse portable: one program (the fake engine), as the piece's impulse-server is."""
        exe = os.path.join(self.tmp, "impulse-server")
        Path(exe).write_text(f'#!/bin/sh\ncd "{ROOT}" && exec "{FAKE[0]}" {" ".join(FAKE[1:])} "$@"\n')
        os.chmod(exe, 0o755)
        os.environ["NEBULA_IMPULSE_PORTABLE_BIN"] = exe
        return exe

    def test_a_catalog_model_runs_on_impulse_portable_through_vulkan(self):
        st, p = self.plan("first-run", {"source": "here", "model": "lfm25-24b"})
        self.assertEqual(st, 200, p)
        self.assertTrue(p["lines"][1].startswith("Downloads Impulse portable ("), p)          # said first: what, how big, why
        self.assertIn("to keep the download small", p["lines"][1])
        self.assertIn("Impulse portable, directly on this computer (not in a sandbox", p["lines"][2])
        w = " ".join(p["warnings"])
        self.assertIn("on the AMD Radeon RX 5500M through Vulkan", w)
        self.assertIn("About 2.8 GB of it goes on the AMD Radeon RX 5500M and the rest, about 10.8 GB, stays in RAM", w)
        self.assertNotIn("a graphics card with", w)                                       # a smaller card is no stop

    def test_it_deploys_on_the_portable_directly_and_answers(self):
        exe = self.portable()
        st, p = self.plan("first-run", {"source": "here", "model": "lfm25-24b"})
        self.assertEqual(st, 200, p)
        self.assertNotIn("Downloads Impulse portable", " ".join(p["lines"]))                # already here
        v = self.apply("first-run", {"source": "here", "model": "lfm25-24b", "cwd": ROOT})
        self.assertEqual(v["state"], "done", v)
        spec = self.srv.cfg.engines[0]
        self.assertEqual((spec["kind"], spec["binary"], spec["where"]), ("llama.cpp", exe, "local"))
        self.assertEqual(spec["vulkan_card"], "AMD Radeon RX 5500M")             # one card of its own, by its name
        self.assertTrue(v["result"]["test"]["ok"], v)                                    # it answered the hello
        self.assertNotIn("card", spec)

    def test_a_model_that_fits_on_the_card_says_so(self):
        from nebula import wizards as W
        self.assertEqual(W._vulkan_fit(self.srv, 1500), ["It fits on the AMD Radeon RX 5500M (3.8 GB free)."])
        from unittest import mock
        with mock.patch("nebula.engines.ram_free_mb", return_value=4096):
            self.assertIn("so it may not start", W._vulkan_fit(self.srv, 20000)[0])

    def test_no_vulkan_card_runs_on_the_cpu_and_says_so(self):
        os.environ["NEBULA_FAKE_VULKAN"] = "[]"
        self.portable()
        from nebula import wizards as W
        self.assertIn("it runs on the CPU", W._vulkan_fit(self.srv, 1500)[0])
        self.assertIn("on the CPU (no graphics card answers through Vulkan here)", W._vulkan_where(self.srv))

    def test_flash_next_is_refused_saying_why(self):
        st, e = self.plan("deploy", {"model": "qwen38-flashnext", "advanced": False, "fn_link": "x"})
        self.assertEqual(st, 409, e)
        self.assertIn("NVIDIA graphics card (RTX 30, 40 or 50)", e["error"]["message"])
        pack = os.path.join(self.tmp, "pack-x")
        os.makedirs(os.path.join(pack, "pack"))
        Path(pack, "dense.gguf").write_bytes(b"GGUF")
        st, e = self.plan("deploy", {"advanced": True, "link": pack})
        self.assertEqual(st, 409, e)
        self.assertIn("A Flash-Next pack runs on Impulse with an NVIDIA graphics card", e["error"]["message"])
        opts = call(self.base, "/api/wizards/first-run/options", {"field": "catalog", "answers": {}})[1]["options"]
        lacks = {o["value"]: o.get("lacks") for o in opts}
        self.assertFalse(lacks["lfm25-24b"])                                             # the card's size is no stop

    def test_vulkan_cards_are_read_from_impulse_portable(self):
        from nebula import catalog as C
        fake = os.path.join(self.tmp, "impulse-server")
        Path(fake).write_text("#!/bin/sh\necho 'Available devices:'\n"
                              "echo '  Vulkan0: AMD Radeon RX 5500M (AMD proprietary driver) (4080 MiB, 3900 MiB free)'\n")
        os.chmod(fake, 0o755)
        os.environ.pop("NEBULA_FAKE_VULKAN")
        os.environ["NEBULA_IMPULSE_PORTABLE_BIN"] = fake
        self.assertEqual(C.vulkan_cards(), [{"id": "Vulkan0", "name": "AMD Radeon RX 5500M (AMD proprietary driver)", "memory_mb": 4080,
                                             "free_mb": 3900, "integrated": False}])
        self.assertEqual(C.machine()["gpu_name"], "AMD Radeon RX 5500M (AMD proprietary driver)")

    def test_one_card_of_its_own_never_the_processors_graphics_and_found_by_name_at_start(self):
        # a real run (2026-10-27): with a Radeon, a GTX 1660 and Intel UHD, llama.cpp spread a model over all of them
        from nebula import catalog as C, engines as E
        three = [{"id": "Vulkan0", "name": "NVIDIA GeForce GTX 1660 SUPER", "free_mb": 5411, "integrated": False},
                 {"id": "Vulkan1", "name": "AMD Radeon RX 5500M Series", "free_mb": 3446, "integrated": False},
                 {"id": "Vulkan2", "name": "Intel(R) UHD Graphics", "free_mb": 7413, "integrated": True}]
        self.assertEqual(C.vulkan_pick(three)["name"], "NVIDIA GeForce GTX 1660 SUPER")   # the most free of its own
        self.assertEqual(C.vulkan_pick(three, "AMD Radeon RX 5500M Series")["id"], "Vulkan1")
        self.assertEqual(C.vulkan_pick(three[2:])["name"], "Intel(R) UHD Graphics")       # only the processor's: that
        spec = {"name": "m", "kind": "llama.cpp", "binary": "impulse-server", "model": "/m.gguf", "host": "127.0.0.1",
                "port": 1, "vulkan_card": "AMD Radeon RX 5500M Series"}
        os.environ["NEBULA_FAKE_VULKAN"] = json.dumps([dict(c, id=f"Vulkan{i}") for i, c in enumerate(three[1:])])   # the 1660 gone: renumbered
        argv = E.command_of(spec)
        self.assertEqual(argv[argv.index("--device") + 1], "Vulkan0")
        os.environ["NEBULA_FAKE_VULKAN"] = json.dumps(three[2:])
        with self.assertRaises(E.EngineSpecError) as e:
            E.command_of(spec)
        self.assertIn("does not answer through Vulkan now", str(e.exception))

class GatewayElsewhereTest(Base):
    @unittest.skipUnless(gravity.available()[0], "Gravity needs iroh")
    def test_commands_for_a_server_elsewhere(self):
        self.srv, self.base, self.path = serve({})
        try:
            before = self.snapshot()
            st, p = self.plan("gateway", {"where": "elsewhere", "port": 3340})
            self.assertEqual(st, 200, p)
            self.assertEqual(self.snapshot(), before)
            v = self.apply("gateway", {"where": "elsewhere", "port": 3340})
            self.assertEqual(v["state"], "done", v)
            self.assertIn("relay.toml", v["result"]["then"])
        finally:
            stop(self.srv)


class AddServerTest(Base):
    def test_a_failed_install_takes_back_the_trust(self):
        from nebula import sshinstall
        pair = tempfile.mkdtemp()                       # a pretend PAIR here: its broker just reads its input
        for b in ("nvpair-ui-broker", "nvpair-node-scanner"):
            Path(pair, b).write_text("#!/bin/sh\ncat > /dev/null\n")
            os.chmod(os.path.join(pair, b), 0o755)
        self.srv, self.base, self.path = serve({"network": {"enabled": False, "pair_bin": pair}})
        real = (sshinstall.scan, sshinstall.install)
        sshinstall.scan = lambda host, port=22: [{"line": f"{host} ssh-ed25519 AAAAfake", "fingerprint": "SHA256:abc", "type": "ssh-ed25519"}]

        def boom(*a, **k):
            raise sshinstall.SshError("box: connection refused")
        sshinstall.install = boom
        try:
            fp = call(self.base, "/api/wizards/add-server/options", {"field": "fingerprints", "answers": {"host": "box"}})[1]["options"]
            self.assertEqual(fp[0]["value"], "SHA256:abc")
            answers = {"host": "box", "user": "me", "fingerprint": "SHA256:abc", "name": "box"}
            before = self.snapshot()
            self.assertEqual(self.plan("add-server", answers)[0], 200)
            self.assertEqual(self.snapshot(), before)
            v = self.apply("add-server", dict(answers, sudo_password="hunter2"))
            self.assertEqual(v["state"], "failed", v)
            job = self.srv.wizards.job(v["id"])
            self.assertNotIn("sudo_password", job.answers)                         # forgotten once the job has run
            self.assertIn("connection refused", v["error"])
            from nebula import yard
            kh = yard._known_hosts(self.srv)
            self.assertNotIn("box ssh-ed25519", Path(kh).read_text() if os.path.exists(kh) else "")   # trust taken back
            self.assertEqual(self.srv.cfg.raw.get("servers") or [], [])
        finally:
            sshinstall.scan, sshinstall.install = real
            if getattr(self.srv, "network", None) is not None:
                self.srv.network.close()
            stop(self.srv)


class RoutingWizardTest(Base):
    """Set up routing: what matters most, which models are capable, which goes first; saved to the
    Routes tab's own settings, and undone back to what was there."""
    def setUp(self):
        self.a, self.b = FakeEngine("small"), FakeEngine("big")
        self.srv, self.base, self.path = serve({"tiers": [{"name": "small", "engine": self.a.url}, {"name": "big", "engine": self.b.url}]})

    def tearDown(self):
        stop(self.srv); self.a.stop(); self.b.stop()

    def test_set_up_and_undo(self):
        st, p = self.plan("routing", {"goal": "balanced", "capable": ["big"], "first": "small", "busy": "live"})
        self.assertEqual(st, 200, p)
        self.assertIn("Larger: big", p["lines"])
        self.assertEqual(p["warnings"], [])
        st, p = self.plan("routing", {"goal": "best", "capable": [], "first": "small"})
        self.assertIn("needs both kinds", p["warnings"][0])
        before = call(self.base, "/api/plan")[1]
        v = self.apply("routing", {"goal": "best", "capable": ["big"], "first": "small", "busy": "off"})
        self.assertEqual(v["state"], "done", v)
        got = call(self.base, "/api/plan")[1]
        self.assertEqual((got["policy"]["big"]["stage"], got["policy"]["small"]["stage"]), ("larger", "everyday"))
        self.assertTrue(got["policy"]["small"].get("prefer"))
        self.assertEqual((got["rules"]["stage_router"]["mode"], got["rules"]["stage_router"]["picker"], got["rules"]["capacity_mode"]),
                         ("live", "capable_first", "off"))
        d = call(self.base, "/api/wizards/routing")[1]["defaults"]
        self.assertEqual((d["goal"], d["capable"], d["first"], d["busy"]), ("best", ["big"], "small", "off"))   # read back
        st, u = call(self.base, f"/api/wizards/jobs/{v['id']}/undo", {})
        self.assertEqual(st, 200, u)
        after = call(self.base, "/api/plan")[1]
        self.assertEqual((after["rules"]["stage_router"]["mode"], after["rules"]["capacity_mode"]),
                         (before["rules"]["stage_router"]["mode"], before["rules"]["capacity_mode"]))
        self.assertNotIn("stage", after["policy"].get("big") or {})


if __name__ == "__main__":
    unittest.main()


class FlashNextSetupTest(unittest.TestCase):
    """0.1.4: Qwen3.8 Flash-Next as a guided setup. The catalog card pulls every file of the split GGUF (pinned) and runs
    the first on Impulse; a Flash-Next pack says first that it downloads Impulse Flash-Next, how big and why."""
    def test_a_setup_gives_its_model_a_card(self):
        from unittest import mock
        from nebula import wizards as W
        cards = [{"uuid": "GPU-busy", "index": 0, "memory_mb": 16000, "used_mb": 15000},
                 {"uuid": "GPU-free", "index": 1, "memory_mb": 12000, "used_mb": 500}]
        with mock.patch.dict(os.environ, {"NEBULA_FAKE_CARDS": json.dumps(cards)}):
            self.assertEqual(W._cards_for({}), "GPU-free")                                 # the one with room: a sandbox gets a GPU
            self.assertEqual(W._cards_for({"cards": ["GPU-a", "GPU-b"]}), "GPU-a,GPU-b")       # or the ones asked for
        with mock.patch.dict(os.environ, {"NEBULA_FAKE_CARDS": "[]"}):
            self.assertIsNone(W._cards_for({}))

    Q4 = {"id": "UD-Q4_K_XL/q4.gguf", "quant": "UD-Q4_K_XL", "files": ["UD-Q4_K_XL/q4-00001-of-00004.gguf"], "size": 111 * 10**9}
    Q2 = {"id": "UD-Q2_K_XL/q2.gguf", "quant": "UD-Q2_K_XL", "files": ["UD-Q2_K_XL/q2-00001-of-00002.gguf"], "size": 60 * 10**9}

    def test_the_card_asks_for_a_repository_and_no_pin(self):
        from nebula import catalog as C
        m = C.get("qwen38-flashnext")
        self.assertTrue(m["ask_repo"])
        for k in ("repo", "file", "shards", "revision", "sha256", "download_gb"):
            self.assertNotIn(k, m)                                                          # the owner's choice, not ours

    def test_a_repository_with_several_sizes_asks_which(self):
        from unittest import mock
        from nebula import catalog as C, hf, wizards as W
        from types import SimpleNamespace
        r = {"repo": "someone/x", "revision": "main", "choices": [self.Q2, self.Q4], "chosen": None}
        with mock.patch.object(W, "_call", return_value=r), mock.patch.object(hf, "remote_architecture", return_value="qwen4exp"), \
             mock.patch("nebula.yard.pulls", return_value=SimpleNamespace(token=lambda: None)):
            with self.assertRaises(W.WizardError) as e:
                W._repo_choice(None, C.get("qwen38-flashnext"), {"fn_link": "https://huggingface.co/someone/x"})
        self.assertIn("This repository has 2 sizes. Pick one: UD-Q2_K_XL (56 GB), UD-Q4_K_XL (103 GB).", str(e.exception))

    def test_it_checks_the_file_is_flashnext_before_any_download(self):
        from unittest import mock
        from nebula import catalog as C, hf, wizards as W
        from types import SimpleNamespace
        r = {"repo": "LiquidAI/x", "revision": "main", "choices": [self.Q2, self.Q4], "chosen": None}   # sizes: asked after
        with mock.patch.object(W, "_call", return_value=r), mock.patch.object(hf, "remote_architecture", return_value="lfm2"), \
             mock.patch("nebula.yard.pulls", return_value=SimpleNamespace(token=lambda: None)):
            with self.assertRaises(W.WizardError) as e:
                W._repo_choice(None, C.get("qwen38-flashnext"), {"fn_link": "x"})
        self.assertEqual(e.exception.status, 409)
        self.assertIn("This file is not Qwen3.8 Flash-Next: it says it is lfm2", str(e.exception))

    def test_too_big_names_the_smaller_sizes_that_fit(self):
        from unittest import mock
        from nebula import catalog as C, hf, wizards as W
        from types import SimpleNamespace
        r = {"repo": "someone/x", "revision": "main", "choices": [self.Q2, self.Q4], "chosen": None}
        with mock.patch.object(W, "_call", return_value=r), mock.patch.object(hf, "remote_architecture", return_value="qwen4exp"), \
             mock.patch("nebula.yard.pulls", return_value=SimpleNamespace(token=lambda: None)), \
             mock.patch("nebula.yard.models_dir", return_value=tempfile.mkdtemp()), \
             mock.patch.dict(os.environ, {"NEBULA_FAKE_CARDS": json.dumps([{"uuid": "G", "index": 0, "memory_mb": 16000, "used_mb": 0}]),
                                          "NEBULA_FAKE_RAM_MB": str(96 * 1024), "NEBULA_FAKE_DISK_MB": str(500 * 1024)}):
            with self.assertRaises(W.WizardError) as e:
                W._repo_choice(None, C.get("qwen38-flashnext"), {"fn_link": "x", "fn_quant": "UD-Q4_K_XL"})
            self.assertIn("UD-Q4_K_XL (103 GB) will not fit this computer", str(e.exception))
            self.assertIn("Smaller sizes in this repository that fit: UD-Q2_K_XL (56 GB).", str(e.exception))
            with mock.patch("nebula.engines.sandbox_plan", return_value=("openshell", None)):
                lines, _w = W._repo_lines(None, C.get("qwen38-flashnext"), {"fn_link": "x", "fn_quant": "UD-Q2_K_XL"})
        self.assertIn("Download UD-Q2_K_XL from someone/x (56 GB)", lines[0])
        self.assertIn("checked against the checksum Hugging Face publishes", lines[0])
        self.assertEqual(lines[1], "It is Qwen3.8 Flash-Next: its file says qwen4exp")
        self.assertIn("in a sandbox", lines[3])

    def test_flash_next_says_plainly_when_it_runs_without_a_sandbox(self):
        # a real run, 2026-10-27: no OpenShell gateway, and the preview still said "in a sandbox"
        from unittest import mock
        from nebula import wizards as W
        with mock.patch("nebula.engines.sandbox_plan", return_value=("local", "the OpenShell gateway is not running")):
            where, warn = W._fn_placement(None)
        self.assertEqual(where, "directly on this computer, not in a sandbox (the OpenShell gateway is not running)")
        self.assertEqual(warn, ["it runs outside a sandbox: it can read what your user can"])
        with mock.patch("nebula.engines.sandbox_plan", return_value=("openshell", None)):
            self.assertEqual(W._fn_placement(None), ("in a sandbox, like your other models", []))

    def test_run_a_model_previews_it_and_the_first_run_never_offers_it(self):
        """Through the wizard itself (the preview went to the wrong one once)."""
        from types import SimpleNamespace
        from unittest import mock
        from nebula import hf, wizards as W
        r = {"repo": "someone/x", "revision": "main", "choices": [self.Q2], "chosen": None}
        srv = SimpleNamespace(cfg=SimpleNamespace(raw={}, base_dir=tempfile.mkdtemp(), path=None))
        with mock.patch.object(W, "_call", return_value=r), mock.patch.object(hf, "remote_architecture", return_value="qwen4exp"), \
             mock.patch("nebula.yard.pulls", return_value=SimpleNamespace(token=lambda: None)), \
             mock.patch("nebula.yard.models_dir", return_value=srv.cfg.base_dir), \
             mock.patch.dict(os.environ, {"NEBULA_FAKE_CARDS": json.dumps([{"uuid": "G", "index": 0, "memory_mb": 16000, "used_mb": 0}]),
                                          "NEBULA_FAKE_RAM_MB": str(256 * 1024), "NEBULA_FAKE_DISK_MB": str(500 * 1024)}):
            for variant in ("standard", "uncensored"):
                p = W.DeployOwn().plan(srv, {"model": "qwen38-flashnext", "fn_link": "x", "advanced": False, "variant": variant})
                self.assertTrue(p["lines"][0].startswith("Download UD-Q2_K_XL from someone/x (56 GB)"), p)

    def test_a_drafter_file_is_not_offered_as_a_model(self):
        from unittest import mock
        from nebula import hf
        tree = [{"type": "file", "path": p, "size": 10, "lfs": {"oid": "ab" * 32, "size": n}} for p, n in
                (("MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf", 4 * 10**9), ("mtp-Qwen3.8-Flash-Next-Q8_0.gguf", 4 * 10**9),
                 ("UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00002.gguf", 30 * 10**9),
                 ("UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00002.gguf", 30 * 10**9))]
        with mock.patch.object(hf, "_get_json", return_value=tree):
            r = hf.resolve("https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF")
        self.assertEqual([(c["id"], len(c["files"])) for c in r["choices"]], [("UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL.gguf", 2)])

    def test_a_gated_repository_says_a_token_is_needed(self):
        from unittest import mock
        from nebula import catalog as C, wizards as W
        gated = W.WizardError(401, "this repository is gated or private: add a Hugging Face token (it is kept in Nebula's key file only) and accept the model's terms on its page")
        with mock.patch.object(W, "_call", side_effect=gated):
            with self.assertRaises(W.WizardError) as e:
                W._repo_choice(None, C.get("qwen38-flashnext"), {"fn_link": "https://huggingface.co/gated/x"})
        self.assertIn("add a Hugging Face token", str(e.exception))

    def test_the_header_names_the_architecture(self):
        import struct
        from nebula import hf
        def kv_str(k, v):
            return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v
        head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 3)
        head += struct.pack("<Q", 7) + b"general" + struct.pack("<I", 4) + struct.pack("<I", 1)            # a u32 first
        head += struct.pack("<Q", 4) + b"list" + struct.pack("<I", 9) + struct.pack("<IQ", 8, 2) + struct.pack("<Q", 1) + b"a" + struct.pack("<Q", 1) + b"b"
        head += kv_str(b"general.architecture", b"qwen4exp")
        self.assertEqual(hf.gguf_architecture(head), "qwen4exp")
        self.assertIsNone(hf.gguf_architecture(head[:40]))                                    # cut short: not guessed
        self.assertIsNone(hf.gguf_architecture(b"NOTGGUF..."))

    def test_a_pack_says_it_downloads_impulse_flashnext_first(self):
        from unittest import mock
        from nebula import pieces
        tmp = tempfile.mkdtemp()
        pack = os.path.join(tmp, "models", "fn-pack")
        os.makedirs(os.path.join(pack, "pack"))
        Path(pack, "pack", "index.txt").write_text("")
        Path(pack, "fn-dense.gguf").write_bytes(b"GGUF")
        srv, base, _path = serve({}, tmp)
        try:
            pin = {"url": "https://x.invalid/f.tar.gz", "sha256": "ab" * 32, "bytes": 109 * 2**20, "ext": "tar.gz", "version": "0.1.4",
                   "label": "Impulse Flash-Next", "why": "the engine for Qwen3.8 Flash-Next models"}
            with mock.patch.object(pieces, "pin", return_value=pin):
                st, p = call(base, "/api/wizards/deploy/plan", {"answers": {"advanced": True, "link": pack}})
            self.assertEqual(st, 200, p)
            self.assertIn("Downloads Impulse Flash-Next (109 MB)", p["lines"][1])
            self.assertIn("not in the installer, to keep the download small", p["lines"][1])
            self.assertIn("in a sandbox", p["lines"][2])
            with mock.patch.object(pieces, "pin", return_value=None):                        # not published: said, not guessed
                st, p = call(base, "/api/wizards/deploy/plan", {"answers": {"advanced": True, "link": pack}})
            self.assertEqual(st, 409)
            self.assertIn("not published", json.dumps(p))
        finally:
            stop(srv)


class FlashNextCardPlacesTest(Base):
    """The Flash-Next card (it asks for its file) is in Run a model, not in the one-click first run."""
    def setUp(self):
        self.srv, self.base, self.path = serve({})

    def tearDown(self):
        stop(self.srv)

    def test_where_the_card_shows(self):
        first = [o["value"] for o in call(self.base, "/api/wizards/first-run/options", {"field": "catalog", "answers": {}})[1]["options"]]
        deploy = call(self.base, "/api/wizards/deploy/options", {"field": "catalog", "answers": {}})[1]["options"]
        self.assertNotIn("qwen38-flashnext", first)
        card = next(o for o in deploy if o["value"] == "qwen38-flashnext")
        self.assertEqual(card["size"], "you choose the file and its size")
