"""The model library (nebula/library.py): the Models page's cards are every model file Nebula can see (its own folder,
the folders you added, the Hugging Face cache), once each, solid while an engine runs it; downloads go to the folder
picked for them; a folder is added or removed without touching its files; a file is deleted only on request, never one an engine names,
never from the Hugging Face cache."""
import os
import tempfile
import unittest

from nebula import hfcache, library


class Cfg:
    def __init__(self, base):
        self.base_dir, self.raw = base, {}


def gguf(path, size=64):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GGUF" + b"\0" * (size - 4))
    return path


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.hf = tempfile.mkdtemp()
        os.environ["NEBULA_HF_CACHE"] = self.hf           # an empty cache: only the folders here
        hfcache._CACHE.update(at=0.0, key=None)
        self.cfg = Cfg(self.base)

    def tearDown(self):
        os.environ.pop("NEBULA_HF_CACHE", None)
        hfcache._CACHE.update(at=0.0, key=None)

    def test_a_file_an_engine_here_names_is_listed_wherever_it_is(self):
        far = tempfile.mkdtemp()
        p = gguf(os.path.join(far, "big-Q4_K_M.gguf"), 80)
        engines = [{"name": "big", "state": "ready", "tiers": ["larger"], "spec": {"model": p}},
                   {"name": "far", "state": "ready", "spec": {"model": "/elsewhere/x.gguf", "machine": "box"}}]   # another machine's: not here
        v = library.view(self.cfg, engines)
        mine = [f for f in v["files"] if f["source"] == "engine"]
        self.assertEqual([(f["name"], f["loaded"]) for f in mine], [("big-Q4_K_M", True)])
        self.assertIn({"path": far, "kind": "engine", "exists": True, "free": None, "downloads": False}, v["folders"])

    def test_own_folder_shards_projectors_and_partial_downloads(self):
        own = os.path.join(self.base, "models")
        gguf(os.path.join(own, "acme", "small-GGUF", "small-Q4_K_M.gguf"))
        gguf(os.path.join(own, "acme", "small-GGUF", "mmproj-small-f16.gguf"))
        gguf(os.path.join(own, "acme", "big-GGUF", "big-Q8_0-00001-of-00002.gguf"), 100)
        gguf(os.path.join(own, "acme", "big-GGUF", "big-Q8_0-00002-of-00002.gguf"), 50)
        gguf(os.path.join(own, "acme", "half-GGUF", "half-Q4_0-00001-of-00002.gguf"))     # its second shard missing
        with open(os.path.join(own, "acme", "small-GGUF", "next.gguf.part"), "wb") as f:
            f.write(b"x")
        v = library.view(self.cfg)
        names = sorted(f["name"] for f in v["files"])
        self.assertEqual(names, ["big-Q8_0", "small-Q4_K_M"])
        big = next(f for f in v["files"] if f["name"] == "big-Q8_0")
        self.assertEqual((big["size"], big["shards"], big["quant"], big["repo"]), (150, 2, "Q8_0", "acme/big-GGUF"))
        small = next(f for f in v["files"] if f["name"] == "small-Q4_K_M")
        self.assertTrue(small["mmproj"].endswith("mmproj-small-f16.gguf"))
        self.assertEqual(v["download_dir"], os.path.realpath(own))

    def test_loaded_follows_the_engine_that_runs_the_file(self):
        own = os.path.join(self.base, "models")
        p = gguf(os.path.join(own, "a.gguf"))
        gguf(os.path.join(own, "b.gguf"))
        engines = [{"name": "eng-a", "state": "ready", "spec": {"model": "models/a.gguf"}, "tiers": ["everyday"]},
                   {"name": "eng-b", "state": "stopped", "spec": {"model": os.path.join(own, "b.gguf")}, "tiers": []}]
        files = {f["name"]: f for f in library.view(self.cfg, engines)["files"]}
        self.assertTrue(files["a"]["loaded"])
        self.assertEqual(files["a"]["engines"][0]["models"], ["everyday"])
        self.assertFalse(files["b"]["loaded"])                 # an engine names it, but it is stopped
        self.assertEqual(files["b"]["engines"][0]["engine"], "eng-b")
        self.assertEqual(library.view(self.cfg, engines)["files"][0]["path"], p)   # the loaded ones first

    def test_a_folder_added_is_scanned_and_downloads_can_go_there(self):
        ext = tempfile.mkdtemp()
        gguf(os.path.join(ext, "deep", "x-IQ4_XS.gguf"))
        self.assertEqual(library.view(self.cfg)["files"], [])
        with self.assertRaises(library.LibraryError):
            library.set_download_dir(self.cfg, ext)            # not one of the library's folders yet
        library.add_folder(self.cfg, ext)
        library.add_folder(self.cfg, ext)                      # once
        self.assertEqual(self.cfg.raw["model_dirs"], [os.path.realpath(ext)])
        v = library.view(self.cfg)
        self.assertEqual([f["name"] for f in v["files"]], ["x-IQ4_XS"])
        self.assertEqual(v["files"][0]["source"], "added")
        self.assertEqual(library.set_download_dir(self.cfg, ext), os.path.realpath(ext))
        self.assertTrue(next(f for f in library.folders(self.cfg) if f["path"] == os.path.realpath(ext))["downloads"])
        library.remove_folder(self.cfg, ext)                   # its files stay; downloads go home again
        self.assertTrue(os.path.exists(os.path.join(ext, "deep", "x-IQ4_XS.gguf")))
        self.assertNotIn("model_dirs", self.cfg.raw)
        self.assertNotIn("download_dir", self.cfg.raw)
        self.assertEqual(library.download_dir(self.cfg), os.path.realpath(os.path.join(self.base, "models")))

    def test_refuses_a_missing_folder_or_a_whole_home(self):
        with self.assertRaises(library.LibraryError):
            library.add_folder(self.cfg, os.path.join(self.base, "nope"))
        with self.assertRaises(library.LibraryError):
            library.add_folder(self.cfg, os.path.expanduser("~"))
        with self.assertRaises(library.LibraryError):
            library.add_folder(self.cfg, "")

    def test_delete_takes_every_shard_and_refuses_what_it_must(self):
        own = os.path.join(self.base, "models")
        a = gguf(os.path.join(own, "acme", "big-GGUF", "big-Q8_0-00001-of-00002.gguf"), 100)
        b = gguf(os.path.join(own, "acme", "big-GGUF", "big-Q8_0-00002-of-00002.gguf"), 50)
        used = gguf(os.path.join(own, "used.gguf"))
        proj = gguf(os.path.join(own, "acme", "big-GGUF", "mmproj-big-f16.gguf"))
        engines = [{"name": "eng", "state": "stopped", "spec": {"model": used}, "tiers": []}]
        with self.assertRaises(library.LibraryError) as e:
            library.delete_file(self.cfg, used, engines)              # an engine names it, even stopped
        self.assertEqual(e.exception.status, 409)
        with self.assertRaises(library.LibraryError):
            library.delete_file(self.cfg, os.path.join(self.base, "nebula.json"), engines)   # not a model file
        r = library.delete_file(self.cfg, a, engines)
        self.assertEqual((r["deleted"], r["files"], r["freed"]), ("big-Q8_0", 2, 150))
        self.assertFalse(os.path.exists(a) or os.path.exists(b))
        self.assertTrue(os.path.exists(proj) and os.path.exists(used))     # the projector and the other file stay
        os.remove(proj)
        small = gguf(os.path.join(own, "acme", "small-GGUF", "small-Q4_K_M.gguf"))
        library.delete_file(self.cfg, small, engines)
        self.assertFalse(os.path.exists(os.path.dirname(small)))             # its own folder goes once empty
        self.assertTrue(os.path.isdir(own))                                  # the library folder never does

    def test_delete_leaves_the_hugging_face_cache_alone(self):
        snap = os.path.join(self.hf, "models--acme--x-GGUF", "snapshots", "abc")
        p = gguf(os.path.join(snap, "x-Q4_K_M.gguf"))
        hfcache._CACHE.update(at=0.0, key=None)
        if not any(f["source"] == "hf" for f in library.view(self.cfg)["files"]):
            self.skipTest("this cache layout is not read here")
        with self.assertRaises(library.LibraryError):
            library.delete_file(self.cfg, p)
        self.assertTrue(os.path.exists(p))


if __name__ == "__main__":
    unittest.main()
