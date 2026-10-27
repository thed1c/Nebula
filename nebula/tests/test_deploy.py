"""Deploy your own (nebula/packages.py, nebula/hf.py, nebula/yard.py): the packages' settings and the command line they
make, a Hugging Face link resolved and pulled (resume, checksum, gated) from a pretend Hub, and the one-call deploy."""
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nebula import engines as E
from nebula.tests import WINDOWS, linux_only  # noqa: F401
from nebula import hf
from nebula import packages as P
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.test_engines import FAKE, ROOT, call, free_port, make_gguf, wait_for


class PretendHub:
    """The Hub's tree API and resolve downloads (with Range), a gated repository, and a slow mode for pausing."""

    def __init__(self, repos: dict):
        self.repos, self.slow, self.seen = repos, False, []          # repo -> {path: bytes}
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.seen.append((self.path, self.headers.get("Range"), self.headers.get("Authorization")))
                parts = self.path.split("?")[0].strip("/").split("/")
                if parts[:2] == ["api", "models"]:
                    repo = "/".join(parts[2:4])
                    files = outer.files(repo, self.headers)
                    if files is None:
                        return self._code(401 if repo in outer.repos else 404)
                    body = json.dumps([{"type": "file", "path": p, "size": len(b),
                                        "lfs": {"oid": hashlib.sha256(b).hexdigest(), "size": len(b)}} for p, b in files.items()]
                                      + [{"type": "directory", "path": "sub"}, {"type": "file", "path": "README.md", "size": 3}]).encode()
                    return self._send(200, body)
                repo = "/".join(parts[0:2])
                files = outer.files(repo, self.headers)
                if files is None or parts[2] != "resolve":
                    return self._code(401 if files is None else 404)
                path = "/".join(parts[4:])
                data = files.get(path)
                if data is None:
                    return self._code(404)
                start = 0
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(data):
                        return self._code(416)
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                for i in range(start, len(data), 65536):
                    self.wfile.write(data[i:i + 65536])
                    if outer.slow:
                        time.sleep(0.05)

            def _send(self, code, body):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _code(self, code):
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}"

    def files(self, repo, headers):
        if repo not in self.repos:
            return None
        if repo.startswith("gated/") and headers.get("Authorization") != "Bearer hf_good":
            return None
        return self.repos[repo]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def gguf_bytes(tmp, name, **kw):
    path = os.path.join(tmp, name)
    make_gguf(path, **kw)
    return Path(path).read_bytes()


class PackagesTest(unittest.TestCase):
    def test_settings_make_the_command_line(self):
        tmp = tempfile.mkdtemp()
        model = os.path.join(tmp, "m.gguf")
        Path(model).write_bytes(b"x")
        Path(tmp, "mmproj-F16.gguf").write_bytes(b"x")
        a = P.args("llama.cpp", {"mmproj": True, "thinking": False, "flash_attn": "on", "gpu_layers": 999, "temp": 0.6,
                                 "jinja": True, "lora": "a.gguf", "lora_scale": 0.5}, model)
        self.assertIn("--mmproj", a)
        self.assertEqual(a[a.index("--mmproj") + 1], os.path.join(tmp, "mmproj-F16.gguf"))   # found beside the model
        self.assertEqual(a[a.index("--reasoning-budget") + 1], "0")
        self.assertEqual(a[a.index("--lora-scaled") + 1:a.index("--lora-scaled") + 3], ["a.gguf", "0.5"])
        for w in ("--flash-attn", "on", "-ngl", "999", "--temp", "0.6", "--jinja"):
            self.assertIn(w, a)
        a = P.args("impulse", {"thinking": True, "reasoning_budget": 4096, "mtp": "mtp.gguf"}, model)
        self.assertEqual(a[-2:], ["--reasoning-budget", "4096"])
        self.assertIn("--mtp", a)

    def test_each_package_its_own_settings_and_ranges(self):
        self.assertEqual(P.ALL, ("llama.cpp", "impulse"))            # Strata's package removed (the owner, 2026-10-25)
        with self.assertRaises(P.OptionError):
            P.check("llama.cpp", {"vram_reserve_mib": 2048})          # was Strata's only; gone with it
        self.assertEqual(P.check("llama.cpp", {"mtp": "x.gguf"}), {"mtp": "x.gguf"})   # design decision 10-20: MTP for llama.cpp too
        with self.assertRaises(P.OptionError):
            P.check("llama.cpp", {"top_p": 2})
        with self.assertRaises(P.OptionError):
            P.check("llama.cpp", {"flash_attn": "maybe"})
        with self.assertRaises(P.OptionError):
            P.check("llama.cpp", {"tensor_split": "3;1"})
        with self.assertRaises(P.OptionError):
            P.check("llama.cpp", {"nonsense": 1})
        with self.assertRaises(P.OptionError):
            P.check("impulse", {"adapt_async": True})                # was Strata's only
        with self.assertRaises(P.OptionError):
            P.args("llama.cpp", {"mmproj": True}, os.path.join(tempfile.mkdtemp(), "m.gguf"))   # none found
        keys = {f["key"] for f in P.schema("llama.cpp")}
        self.assertTrue({"mmproj", "thinking", "gpu_layers", "flash_attn", "temp", "chat_template", "lora"} <= keys)
        self.assertTrue({"spec_type", "mtp", "draft_model", "draft_max"} <= keys)          # MTP for llama.cpp too

    def test_a_saved_strata_engine_loads_as_impulse(self):
        s = E.check_spec({"name": "old", "kind": "strata", "binary": "/x/strata-server", "model": "m.gguf", "port": 8080,
                          "options": {"vram_reserve_mib": 512, "adapt_async": True, "thinking": False}, "note": "big"})
        self.assertEqual(s["kind"], "impulse")
        self.assertNotIn("binary", s)
        self.assertEqual(s.get("options"), {"thinking": False})
        self.assertIn("moved from the removed strata package", s["note"])

    def test_an_engine_with_options(self):
        s = E.check_spec({"name": "a", "kind": "llama.cpp", "model": "m.gguf", "port": 8080, "context": 8192, "slots": 2,
                          "options": {"cache_type_k": "q8_0", "gpu_layers": 20, "thinking": False}, "flags": ["--verbose"]})
        argv = E.command_of(s, "/cfg")
        self.assertEqual(argv[:2], ["llama-server", "-m"])            # the package's own binary when none is given
        self.assertEqual(argv[argv.index("-ctk") + 1], "q8_0")
        self.assertEqual(argv[-1], "--verbose")
        with self.assertRaises(E.EngineSpecError):
            E.check_spec({"name": "a", "kind": "llama.cpp", "model": "m.gguf", "port": 8080, "options": {"top_k": "many"}})


class HubTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        g = lambda n: gguf_bytes(self.tmp, n)
        self.repos = {"someone/tiny-GGUF": {"tiny-Q4_K_M.gguf": g("a"), "tiny-Q8_0.gguf": g("b") + b"\1" * 1000,
                                             "mmproj-tiny-F16.gguf": g("c"),
                                             "big/tiny-Q2_K-00001-of-00002.gguf": g("d"), "big/tiny-Q2_K-00002-of-00002.gguf": g("e")},
                      "gated/secret": {"secret-Q4_0.gguf": g("f")}}
        self.hub = PretendHub(self.repos)
        os.environ["HF_ENDPOINT"] = self.hub.url

    def tearDown(self):
        self.hub.close()
        del os.environ["HF_ENDPOINT"]

    def test_links(self):
        self.assertEqual(hf.parse("https://huggingface.co/someone/tiny-GGUF"), {"repo": "someone/tiny-GGUF", "revision": "main", "path": None})
        self.assertEqual(hf.parse("https://huggingface.co/someone/tiny-GGUF/blob/main/tiny-Q8_0.gguf")["path"], "tiny-Q8_0.gguf")
        self.assertEqual(hf.parse("https://huggingface.co/someone/tiny-GGUF/resolve/abc123/x/y.gguf"),
                         {"repo": "someone/tiny-GGUF", "revision": "abc123", "path": "x/y.gguf"})
        self.assertEqual(hf.parse("someone/tiny-GGUF")["repo"], "someone/tiny-GGUF")
        with self.assertRaises(hf.HfError):
            hf.parse("https://huggingface.co/onlyowner")
        self.assertEqual([hf.quant_of(x) for x in ("a-Q4_K_M.gguf", "a.IQ2_XS.gguf", "a-BF16.gguf", "mmproj-a-f16.gguf", "a.gguf")],
                         ["Q4_K_M", "IQ2_XS", "BF16", "F16", None])

    def test_resolve_groups_quants_shards_and_finds_the_projector(self):
        r = hf.resolve("https://huggingface.co/someone/tiny-GGUF")
        ids = {c["id"]: c for c in r["choices"]}
        self.assertEqual(set(ids), {"tiny-Q4_K_M.gguf", "tiny-Q8_0.gguf", "big/tiny-Q2_K.gguf"})
        self.assertEqual(ids["big/tiny-Q2_K.gguf"]["files"], ["big/tiny-Q2_K-00001-of-00002.gguf", "big/tiny-Q2_K-00002-of-00002.gguf"])
        self.assertEqual((r["mmproj"], r["chosen"]), ("mmproj-tiny-F16.gguf", None))
        r = hf.resolve("https://huggingface.co/someone/tiny-GGUF/blob/main/tiny-Q8_0.gguf")
        self.assertEqual(r["chosen"], "tiny-Q8_0.gguf")
        with self.assertRaises(hf.HfError) as cm:
            hf.resolve("https://huggingface.co/someone/tiny-GGUF/blob/main/nope.gguf")
        self.assertEqual(cm.exception.status, 404)

    def test_gated_needs_the_token_from_nebulas_key_file(self):
        with self.assertRaises(hf.HfError) as cm:
            hf.resolve("https://huggingface.co/gated/secret")
        self.assertEqual(cm.exception.status, 401)
        self.assertIn("token", str(cm.exception))
        pulls = hf.Pulls(os.path.join(self.tmp, "models"), os.path.join(self.tmp, "keys", "huggingface.token"))
        pulls.set_token("hf_good")
        if not WINDOWS:                                       # Windows: the user's profile is the wall, not a mode
            self.assertEqual(oct(os.stat(pulls.token_path).st_mode & 0o777), "0o600")
        self.assertEqual(hf.resolve("https://huggingface.co/gated/secret", pulls.token())["chosen"], "secret-Q4_0.gguf")

    def test_pull_pause_resume_and_checksum(self):
        pulls = hf.Pulls(os.path.join(self.tmp, "models"))
        data = self.repos["someone/tiny-GGUF"]["tiny-Q8_0.gguf"]
        sha = hashlib.sha256(data).hexdigest()
        self.hub.slow = True
        pid = pulls.start("someone/tiny-GGUF", ["tiny-Q8_0.gguf"], sha256={"tiny-Q8_0.gguf": sha}, sizes={"tiny-Q8_0.gguf": len(data)})
        self.assertTrue(wait_for(lambda: pulls.status(pid)["bytes"] > 0))
        pulls.cancel(pid)
        self.assertTrue(wait_for(lambda: pulls.status(pid)["state"] == "cancelled"))
        part = pulls.target("someone/tiny-GGUF", "tiny-Q8_0.gguf") + ".part"
        self.assertTrue(0 < os.path.getsize(part) < len(data))
        self.hub.slow = False
        pid = pulls.start("someone/tiny-GGUF", ["tiny-Q8_0.gguf"], sha256={"tiny-Q8_0.gguf": sha})
        self.assertTrue(wait_for(lambda: pulls.status(pid)["state"] == "done"), pulls.status(pid))
        self.assertTrue(any(rng and rng.startswith("bytes=") for _p, rng, _a in self.hub.seen))   # resumed by Range
        self.assertEqual(Path(pulls.target("someone/tiny-GGUF", "tiny-Q8_0.gguf")).read_bytes(), data)
        pid = pulls.start("someone/tiny-GGUF", ["tiny-Q4_K_M.gguf"], sha256={"tiny-Q4_K_M.gguf": "0" * 64})
        self.assertTrue(wait_for(lambda: pulls.status(pid)["state"] == "failed"))
        self.assertIn("checksum", pulls.status(pid)["files"][0]["error"])
        self.assertFalse(os.path.exists(pulls.target("someone/tiny-GGUF", "tiny-Q4_K_M.gguf")))
        with self.assertRaises(hf.HfError):
            pulls.target("someone/tiny-GGUF", "../../escape.gguf")


class DeployYourOwnTest(unittest.TestCase):
    """The wizard's calls in order: packages, a link resolved, the file pulled, a preview, deploy; then a chat."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.hub = PretendHub({"someone/tiny-GGUF": {"tiny-Q4_K_M.gguf": gguf_bytes(self.tmp, "t.gguf")}})
        os.environ["HF_ENDPOINT"] = self.hub.url
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({"port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
                                               "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        self.srv = make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close(); self.hub.close()
        del os.environ["HF_ENDPOINT"]

    def test_link_to_lane(self):
        st, v = call(self.base, "/api/yard/packages")
        self.assertEqual([p["name"] for p in v["packages"]], ["llama.cpp", "impulse"])
        self.assertIn("Thinking", v["groups"])
        st, r = call(self.base, "/api/yard/hf/resolve", {"url": "https://huggingface.co/someone/tiny-GGUF"})
        self.assertEqual((st, r["chosen"]), (200, "tiny-Q4_K_M.gguf"), r)
        self.assertTrue(r["choices"][0]["fits_disk"])
        f = r["files"][0]
        st, pull = call(self.base, "/api/yard/hf/pull", {"repo": r["repo"], "files": [f["path"]], "meta": [f]})
        self.assertTrue(wait_for(lambda: call(self.base, "/api/yard/hf/pulls")[1]["pulls"][0]["state"] == "done"))
        model = pull["files"][0]["target"]
        self.assertTrue(os.path.realpath(model).startswith(os.path.realpath(os.path.join(self.tmp, "models"))))
        engine = {"name": "tiny", "kind": "impulse", "binary": FAKE, "model": model, "context": 2048, "slots": 1, "cwd": ROOT,
                  "options": {"thinking": False, "temp": 0.7}}
        st, pv = call(self.base, "/api/yard/preview", {"engine": engine})
        self.assertEqual(st, 200, pv)
        self.assertIn("--reasoning-budget", pv["command"])
        st, out = call(self.base, "/api/yard/deploy", {"engine": engine, "wait": True})
        self.assertEqual((st, out["engine"]["state"], out["route"]["tier"]), (200, "ready", "tiny"), out)
        self.assertGreaterEqual(out["engine"]["spec"]["port"], 18300)
        st, body = call(self.base, "/v1/chat/completions", {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200, body)
        saved = json.loads(Path(self.path).read_text())
        self.assertEqual((saved["engines"][0]["name"], saved["tiers"][-1]["managed"]), ("tiny", "tiny"))
        st, _ = call(self.base, "/api/yard/hf/token", {"token": "hf_secret"})
        self.assertNotIn("hf_secret", Path(self.path).read_text())         # the token never goes into the config
        self.assertTrue(call(self.base, "/api/yard/hf/pulls")[1]["token"])


if __name__ == "__main__":
    unittest.main()
