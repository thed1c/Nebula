"""Halo's model (nebula/halo_model.py, nebula/catalog/halo.json): the shipped adapters match their hashes, the base is pinned
and fetched through hf.Pulls with its sha256, placement follows the owner's order (spare GPU > spare VRAM > RAM), the
server's arguments load both adapters, and Standard / Uncensored is the Uncensored adapter's scale set on a running server."""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from nebula import halo_model as HM


class FakePulls:
    def __init__(self, d):
        self.models_dir, self.started = d, []

    def target(self, repo, path):
        return f"{self.models_dir}/{repo}/{path}"

    def start(self, repo, paths, revision="main", sha256=None, sizes=None):
        self.started.append((repo, paths, revision, sha256, sizes))
        return "pull1"


class HaloModelTest(unittest.TestCase):
    def test_manifest_and_assets(self):
        m = HM.manifest()
        self.assertEqual(HM.verify_assets(m), [])                       # the shipped adapters are the ones described
        b = m["base"]
        self.assertEqual(len(b["revision"]), 40)                        # a pinned commit, never "main"
        self.assertEqual(len(b["sha256"]), 64)
        self.assertTrue((HM.ASSETS / "LICENSE-LFM-1.0").is_file())

    def test_ensure_base_pulls_pinned_with_sha(self):
        d = tempfile.mkdtemp()
        p = FakePulls(d)
        import nebula.hf as hf
        orig = hf.Pulls
        hf.Pulls = lambda models_dir: FakePulls(models_dir)
        try:
            self.assertEqual(HM.ensure_base(p), "pull1")
        finally:
            hf.Pulls = orig
        repo, paths, rev, sha, sizes = p.started[0]
        m = HM.manifest()["base"]
        self.assertEqual((repo, paths, rev, sha[m["file"]]), (m["repo"], [m["file"]], m["revision"], m["sha256"]))

    def test_place(self):
        self.assertEqual(HM.place([{"index": 0, "free_mb": 9000, "engines": ["gemma"]}, {"index": 1, "free_mb": 3000, "engines": []}]), "gpu:1")
        self.assertEqual(HM.place([{"index": 0, "free_mb": 3000, "engines": ["gemma"]}]), "gpu:0")       # spare VRAM
        self.assertEqual(HM.place([{"index": 0, "free_mb": 900, "engines": ["gemma"]}]), "cpu")          # an engine's card it fills
        self.assertEqual(HM.place([]), "cpu")
        spare = [{"index": 1, "free_mb": 3000, "engines": []}]
        self.assertEqual(HM.place(spare, prefer="cpu"), "cpu")                                       # the owner's choice
        self.assertEqual(HM.place(spare, prefer="gpu"), "gpu:1")

    def test_server_args(self):
        a = HM.server_args("/models", "uncensored", "gpu:0", port=18500)
        lora = a[a.index("--lora-scaled") + 1]
        self.assertIn("halo-guard-v2-lora-f16.gguf:1.0,", lora)
        self.assertTrue(lora.endswith("halo-uncensored-heretic-t198-lora-f16.gguf:1.0"))
        self.assertIn("-ngl", a); self.assertEqual(a[a.index("-ngl") + 1], "99")
        s = HM.server_args("/models", "standard", "cpu", lane="router")
        self.assertTrue(s[s.index("--lora-scaled") + 1].endswith(":0.0"))
        self.assertEqual(s[s.index("-np") + 1], "1")                    # the router lane: one slot of its own
        self.assertEqual(s[s.index("-ngl") + 1], "0")
        with self.assertRaises(ValueError):
            HM.server_args("/models", "spicy")

    def test_set_variant(self):
        got = []

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                got.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_response(200); self.end_headers(); self.wfile.write(b"{}")

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            HM.set_variant(f"http://127.0.0.1:{srv.server_port}", "uncensored")
            HM.set_variant(f"http://127.0.0.1:{srv.server_port}", "standard")
        finally:
            srv.shutdown(); srv.server_close()
        self.assertEqual(got[0], ("/lora-adapters", [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 1.0}]))
        self.assertEqual(got[1][1][1], {"id": 1, "scale": 0.0})


if __name__ == "__main__":
    unittest.main()
