"""The Hugging Face cache (nebula/hfcache.py): GGUF models already on this computer are offered in
Run a model and Get started, with no download; a catalog model already in it is not downloaded again; an engine runs
the file where it is, and a sandboxed one sees the cache read-only at /hf."""
import json
import os
import shutil
import tempfile
import unittest

from nebula import hfcache
from nebula.tests import WINDOWS, linux_only  # noqa: F401
from nebula.tests.test_callers import call
from nebula.tests.test_engines import FAKE, ROOT, make_gguf
from nebula.tests import test_wizards as TW


def blob(root, repo, rev, rel, sha, real=True, size_from=None):
    """A file as huggingface_hub keeps it: snapshots/<rev>/<rel> -> ../../blobs/<sha>."""
    d = os.path.join(root, "models--" + repo.replace("/", "--"))
    os.makedirs(os.path.join(d, "blobs"), exist_ok=True)
    b = os.path.join(d, "blobs", sha + ("" if real else ".incomplete"))
    if size_from:
        shutil.copy(size_from, b)
    elif real:
        make_gguf(b)
    else:
        open(b, "wb").write(b"part")
    s = os.path.join(d, "snapshots", rev, rel)
    os.makedirs(os.path.dirname(s), exist_ok=True)
    os.symlink(os.path.relpath(b, os.path.dirname(s)), s)
    return s


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.environ["NEBULA_HF_CACHE"] = self.root
        hfcache._CACHE.update(at=0.0, key=None)

    def tearDown(self):
        os.environ["NEBULA_HF_CACHE"] = ""

    def test_what_is_offered(self):
        blob(self.root, "o/vision-GGUF", "r1", "m-Q4_K_M.gguf", "a1")
        blob(self.root, "o/vision-GGUF", "r1", "mmproj-F16.gguf", "a2")
        for i in (1, 2):
            blob(self.root, "o/big-GGUF", "r2", f"Q8_0/big-Q8_0-0000{i}-of-00002.gguf", f"b{i}")
        blob(self.root, "o/half-GGUF", "r3", "half-Q4_0-00001-of-00002.gguf", "c1")          # its second shard missing
        blob(self.root, "o/busy-GGUF", "r4", "busy-Q4_0.gguf", "d1", real=False)             # still downloading
        os.makedirs(os.path.join(self.root, "models--o--not-gguf", "snapshots", "r5"))
        got = {m["repo"]: m for m in hfcache.models()}
        self.assertEqual(sorted(got), ["o/big-GGUF", "o/vision-GGUF"])
        v = got["o/vision-GGUF"]
        self.assertEqual((v["quant"], v["revision"], v["shards"]), ("Q4_K_M", "r1", 1))
        self.assertTrue(v["mmproj"].endswith("mmproj-F16.gguf"))
        self.assertIn("reads pictures", hfcache.label(v))
        self.assertEqual((got["o/big-GGUF"]["shards"], got["o/big-GGUF"]["quant"]), (2, "Q8_0"))
        self.assertEqual(hfcache.find(None, "o/vision-GGUF", "r1", "m-Q4_K_M.gguf"), v["path"])
        self.assertIsNone(hfcache.find(None, "o/vision-GGUF", "r9", "m-Q4_K_M.gguf"))       # another revision: not it
        self.assertEqual(hfcache.root_of(None, v["path"]), os.path.realpath(self.root))


class WizardCacheTest(TW.ModelWizardsTest):
    locals().update({n: None for n in dir(TW.ModelWizardsTest) if n.startswith("test_")})   # its own tests only
    def setUp(self):
        super().setUp()
        self.cache = os.path.join(self.tmp, "hfcache")
        os.environ["NEBULA_HF_CACHE"] = self.cache
        hfcache._CACHE.update(at=0.0, key=None)

    def tearDown(self):
        os.environ["NEBULA_HF_CACHE"] = ""
        super().tearDown()

    def test_a_cached_model_runs_without_a_download(self):
        p = blob(self.cache, "someone/cached-GGUF", "rev1", "cached-Q4_K_M.gguf", "e1")
        opts = call(self.base, "/api/wizards/deploy/options", {"field": "hfcache", "answers": {}})[1]["options"]
        pick = next(o for o in opts if o["value"].startswith("someone/cached-GGUF"))
        self.assertIn("Q4_K_M", pick["label"])
        answers = {"cached": pick["value"], "binary": FAKE, "cwd": ROOT}
        st, plan = self.plan("deploy", answers)
        self.assertEqual(st, 200, plan)
        self.assertIn("no download", plan["lines"][0])
        self.hub.close()                                                                     # nothing may be fetched
        v = self.apply("deploy", answers)
        self.assertEqual(v["state"], "done", v)
        self.assertTrue(v["result"]["test"]["ok"], v)
        self.assertEqual(os.path.realpath(self.srv.cfg.engines[0]["model"]), os.path.realpath(p))   # used where it is
        self.assertTrue(os.path.islink(p))                                                   # the cache untouched

    def test_a_catalog_model_in_the_cache_is_not_downloaded(self):
        from nebula import catalog as C
        m = C.get("lfm25-24b")
        blob(self.cache, m["repo"], m["revision"], m["file"], "f1")
        opts = call(self.base, "/api/wizards/first-run/options", {"field": "catalog", "answers": {}})[1]["options"]
        self.assertIn("no download", next(o for o in opts if o["value"] == "lfm25-24b")["size"])
        self.hub.close()
        v = self.apply("first-run", {"source": "here", "model": "lfm25-24b", "binary": FAKE, "cwd": ROOT})
        self.assertEqual(v["state"], "done", v)
        self.assertIn("hfcache", self.srv.cfg.engines[0]["model"])

    @linux_only

    def test_a_sandboxed_engine_reads_the_cache_read_only(self):
        p = blob(self.cache, "someone/cached-GGUF", "rev1", "cached-Q4_K_M.gguf", "e1")
        mm = blob(self.cache, "someone/cached-GGUF", "rev1", "mmproj-F16.gguf", "e2")
        sp = {"name": "boxed", "kind": "llama.cpp", "model": p, "port": 9, "options": {"mmproj": True, "mmproj_file": mm}}
        e = self.srv.yard._sandboxed(sp, None)
        self.assertEqual(e.extra_dirs[0], (os.path.realpath(self.cache), "/hf"))
        cmd = " ".join(e.inner_cmd)
        self.assertIn("-m /hf/models--someone--cached-GGUF/snapshots/rev1/cached-Q4_K_M.gguf", cmd)
        self.assertIn("/hf/models--someone--cached-GGUF/snapshots/rev1/mmproj-F16.gguf", cmd)
        from nebula import openshell
        argv = openshell.deploy_argv(sp, e.inner_cmd, None, "v", "pol", [("vh", "/hf")])
        mounts = json.loads(argv[argv.index("--driver-config-json") + 1])["docker"]["mounts"]
        self.assertEqual(mounts[1], {"type": "volume", "source": "vh", "target": "/hf", "read_only": True})


if __name__ == "__main__":
    unittest.main()
