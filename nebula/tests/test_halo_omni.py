"""Halo Omni (nebula/catalog/halo.json "halo-omni-2b"): a model the owner picks (assistant.halo.model), never by the
runtime alone nor as a fallback; its base runs as a copy with its chat template fixed (equal-length, checked by sha256);
its adapters are a piece of their own (any platform, a data file)."""
import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nebula import halo_engine as HE, halo_model as HM, pieces


def gguf(path: str, template: str, pad: int = 64) -> None:
    """A small GGUF v3 file: two keys (the chat template one of them), no tensors, some bytes after."""
    def s(b: bytes) -> bytes:
        return struct.pack("<Q", len(b)) + b
    kv = s(b"general.name") + struct.pack("<I", 8) + s(b"tiny") + s(b"tokenizer.chat_template") + struct.pack("<I", 8) + s(template.encode())
    Path(path).write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2) + kv + b"\0" * pad)


class ChoiceTest(unittest.TestCase):
    def test_omni_runs_only_when_picked_and_where_its_runtime_is(self):
        self.assertEqual(HM.model_for("Impulse"), "halo-lfm3-1.7b")
        self.assertEqual(HM.model_for("Impulse", choice="halo-omni-2b"), "halo-omni-2b")
        self.assertEqual(HM.model_for("Impulse portable", choice="halo-omni-2b"), "halo-lfm25-1.2b")   # not there: the runtime's own
        self.assertEqual(HM.model_for(None), "halo-lfm25-1.2b")
        self.assertEqual([m["id"] for m in HM.choices("Impulse")], ["halo-lfm3-1.7b", "halo-omni-2b"])
        self.assertEqual([m["id"] for m in HM.choices("llama.cpp")], ["halo-lfm25-1.2b"])
        m = HM.manifest(model="halo-omni-2b")
        self.assertEqual((m["piece"], m["base"]["fix"]["sha256"][:8], m["adapters"]["uncensored"]["scale"]), ("halo-omni", "c436d96c", {"standard": 0.0, "uncensored": 1.0}))

    def test_the_engine_reads_its_adapters_from_the_piece_and_never_falls_back_to_it(self):
        d = tempfile.mkdtemp()
        cfg = SimpleNamespace(assistant={"halo": {"model": "halo-omni-2b"}}, base_dir=d, raw={})
        e = HE.HaloEngine.__new__(HE.HaloEngine)
        e.cfg, e.server, e.m = cfg, None, None
        m = e.chosen("Impulse")
        self.assertEqual((m["id"], m["_dir"]), ("halo-omni-2b", str(pieces.folder("halo-omni", d))))
        self.assertTrue(HM.verify_assets(m))                         # not downloaded: missing
        self.assertEqual(Path(HM.server_args("/m", m=m, base="/b.gguf")[3].split(":")[0]).parent, pieces.folder("halo-omni", d))
        e.m = HM.manifest()                                          # LFM3 runs; its fallback is never the owner's pick
        with mock.patch.object(HE, "find_base", return_value="/m/x.gguf"):
            fb = e._fallback([("Impulse", ["/bin/impulse-server"])])
        self.assertNotEqual((fb or ({},))[0].get("id"), "halo-omni-2b")


class FixTest(unittest.TestCase):
    def test_the_fixed_copy_changes_only_the_template_and_must_check_out(self):
        d = tempfile.mkdtemp()
        tpl = "{%- set carry = namespace(items=[]) -%}{{ carry.items | length }}{%- for p in carry.items %}x{% endfor %}"
        fixed = tpl.replace("namespace(items=[])", "namespace(parts=[])").replace("carry.items", "carry.parts")
        m = {"id": "t", "name": "Tiny", "base": {"repo": "o/r", "file": "t.gguf", "fix": {
            "replace": [["namespace(items=[])", "namespace(parts=[])"], ["carry.items", "carry.parts"]]}}}
        src = HM.base_path(d, m)
        os.makedirs(os.path.dirname(src))
        gguf(src, tpl)
        want = Path(src).read_bytes().replace(tpl.encode(), fixed.encode())
        m["base"].update(bytes=len(want), sha256=hashlib.sha256(Path(src).read_bytes()).hexdigest())
        m["base"]["fix"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):                          # a copy that does not check out is not used
            HM.fix_base(d, m)
        self.assertFalse(os.path.exists(HM.fixed_path(d, m)))
        self.assertTrue(os.path.isfile(src))
        m["base"]["fix"]["sha256"] = hashlib.sha256(want).hexdigest()
        out = HM.fix_base(d, m)
        self.assertEqual(Path(out).read_bytes(), want)
        self.assertFalse(os.path.exists(src))                        # the unfixed one is not kept
        self.assertTrue(HM.base_ok(d, m))
        env = {k: v for k, v in os.environ.items() if k != "NEBULA_HALO_BASE"}   # the tests' own "no base" set aside
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(HE.find_base(d, m), out)
        self.assertEqual(HM._template_at(out)[1].decode(), fixed)
        pulls = SimpleNamespace(models_dir=d, start=lambda *a, **k: self.fail("nothing to pull"))
        self.assertIsNone(HM.ensure_base(pulls, m))


class PageTest(unittest.TestCase):
    def test_a_model_this_machine_cannot_run_is_refused(self):
        import urllib.error
        import urllib.request
        from nebula.tests.test_escalation import Rig
        rig = Rig()
        self.addCleanup(rig.stop)

        def post(body):
            r = urllib.request.Request(rig.base + "/api/halo", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=10) as resp:
                return json.loads(resp.read())
        with mock.patch.object(type(rig.srv.halo_engine), "choices_view", return_value=[
                {"id": "halo-lfm3-1.7b", "choice": False}, {"id": "halo-omni-2b", "choice": True}]), \
                mock.patch.object(type(rig.srv.halo_engine), "restart") as restart, \
                mock.patch("nebula.ui.find_runtime", return_value="Impulse"):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                post({"model": "halo-nope"})
            self.assertEqual(cm.exception.code, 400)
            post({"model": "halo-omni-2b"})
            self.assertEqual(rig.srv.cfg.assistant["halo"]["model"], "halo-omni-2b")
            post({"model": "halo-lfm3-1.7b"})                            # the machine's own: no pick kept
            self.assertNotIn("model", rig.srv.cfg.assistant["halo"])
            self.assertTrue(restart.called)


class PieceTest(unittest.TestCase):
    def test_a_data_piece_for_every_platform(self):
        cat = {"version": "0.2.0", "url": "https://example.org/v{version}/{piece}-{version}-{platform}.{ext}",
               "pieces": {"halo-omni": {"label": "Halo Omni's adapters", "why": "w", "program": "a.gguf", "data": True,
                                        "files": {"any": {"ext": "tar.gz", "sha256": "ab" * 32, "bytes": 3 * 2**20}}}}}
        d = tempfile.mkdtemp()
        p = os.path.join(d, "pieces.json")
        Path(p).write_text(json.dumps(cat))
        with mock.patch.object(pieces, "CATALOG", Path(p)), mock.patch.object(pieces.os, "name", "nt"):
            self.assertEqual(pieces.program("halo-omni"), "a.gguf")                  # a data file: no .exe
            self.assertEqual(pieces.pin("halo-omni")["url"], "https://example.org/v0.2.0/halo-omni-0.2.0-any.tar.gz")
        self.assertIn("halo-omni", json.loads(pieces.CATALOG.read_text())["pieces"])   # the real catalog: published with 0.2.0
        self.assertTrue(pieces.pin("halo-omni")["url"].endswith("/releases/download/v0.2.0/halo-omni-0.2.0-any.tar.gz"))


if __name__ == "__main__":
    unittest.main()
