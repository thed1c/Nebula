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
                               "someone/tiny-GGUF": {"tiny-Q4_K_M.gguf": g("e")}})
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
        self.assertEqual(sorted(o["value"] for o in friends), ["gemma4-26b", "lfm25-24b", "qwen36-35b"])   # all offered to friends

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
