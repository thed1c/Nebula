"""What the installer fetches beside Nebula (nebula/prereqs.py): each step on its own, skipped when already here, never
anything that needs an administrator."""
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nebula import prereqs
from nebula.config import Config


def cfg_in(tmp, extra=None):
    path = os.path.join(tmp, "nebula.json")
    json.dump({"port": 0, "ledger": os.path.join(tmp, "l.sqlite3"), "tiers": [], **(extra or {})}, open(path, "w"))
    return Config.from_args(["--config", path])


def tgz(name: str, body: bytes) -> bytes:
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        info = tarfile.TarInfo(f"pkg/{name}")
        info.size = len(body)
        t.addfile(info, io.BytesIO(body))
    return b.getvalue()


class PrereqsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = mock.patch.dict(os.environ, {"HOME": self.tmp, "XDG_CONFIG_HOME": os.path.join(self.tmp, "cfg"),
                                                "XDG_STATE_HOME": os.path.join(self.tmp, "state")})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_a_system_openshell_is_used_as_it_is(self):
        with mock.patch.object(prereqs, "openshell_system", return_value="/usr/bin/openshell"):
            self.assertIn("already installed", prereqs.openshell(self.tmp))
        self.assertFalse(Path(self.tmp, "openshell").exists())

    def test_openshell_is_fetched_checked_and_set_up_for_this_user(self):
        files = {"cli": tgz("openshell", b"#!/bin/sh\necho cli\n"), "gateway": tgz("openshell-gateway", b"#!/bin/sh\necho gw\n")}
        pinned = {k: (prereqs.OPENSHELL_FILES[k][0], __import__("hashlib").sha256(v).hexdigest()) for k, v in files.items()}
        by_name = {pinned[k][0]: v for k, v in files.items()}

        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        with mock.patch.object(prereqs, "openshell_system", return_value=None), \
             mock.patch.object(prereqs, "OPENSHELL_FILES", pinned), \
             mock.patch.object(prereqs.urllib.request, "urlopen", side_effect=lambda url, timeout=0: R(by_name[url.rsplit("/", 1)[1]])), \
             mock.patch.object(prereqs, "_gateway_service", return_value=False):
            msg = prereqs.openshell(self.tmp)
        self.assertIn("installed", msg)
        self.assertIn("not started", msg)                               # no user services here: said, not failed
        cli = Path(self.tmp, "openshell", "bin", "openshell")
        self.assertTrue(os.access(cli, os.X_OK))
        self.assertIn("enable_bind_mounts = true", Path(self.tmp, "cfg", "openshell", "gateway.toml").read_text())

    def test_a_download_that_does_not_match_its_sha256_is_refused(self):
        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        with mock.patch.object(prereqs.urllib.request, "urlopen", return_value=R(b"tampered")):
            with self.assertRaises(RuntimeError):
                prereqs._fetch("https://example.invalid/x.tgz", Path(self.tmp, "x.tgz"), "0" * 64)
        self.assertFalse(Path(self.tmp, "x.tgz").exists())
        self.assertFalse(Path(self.tmp, "x.tgz.part").exists())

    def test_one_step_failing_leaves_the_others_and_says_how_to_retry(self):
        cfg = cfg_in(self.tmp)
        said = []
        with mock.patch.object(prereqs, "qdrant", side_effect=RuntimeError("no network")), \
             mock.patch.object(prereqs, "hub_probe", return_value=None), \
             mock.patch.object(prereqs, "embeddings", return_value="Lookup's models: in place"), \
             mock.patch.object(prereqs, "report", return_value=["No Docker: ..."]), \
             mock.patch.object(prereqs, "say", side_effect=lambda *a: said.append(" ".join(a))):
            failed = prereqs.run(cfg, self.tmp, {"cuda", "halo", "halo-runtime", "docling", "openshell", "docker"})
        self.assertEqual(failed, 1)
        self.assertTrue(any("qdrant: not fetched now (no network)" in s and "python -m nebula.prereqs" in s for s in said), said)
        self.assertIn("Lookup's models: in place", said)
        self.assertIn("No Docker: ...", said)

    def test_a_hub_out_of_reach_is_said_once_with_its_reason_and_nebula_fetches_later(self):
        cfg = cfg_in(self.tmp)
        said = []
        why = "the server's name does not resolve (no internet, or a DNS filter or this network blocks it)"
        with mock.patch.object(prereqs, "hub_probe", return_value=why), \
             mock.patch("nebula.halo_engine.find_base", return_value=None), \
             mock.patch("nebula.hf.Pulls.start", side_effect=AssertionError("no download tried")), \
             mock.patch.object(prereqs, "report", return_value=[]), \
             mock.patch.object(prereqs, "say", side_effect=lambda *a: said.append(" ".join(a))):
            failed = prereqs.run(cfg, self.tmp, {"cuda", "halo-runtime", "qdrant", "embeddings", "docling", "openshell", "docker"})
        self.assertEqual(failed, 1)
        self.assertIn("cannot be reached from here: " + why, said[0])
        self.assertTrue(said[1].startswith("Halo's model: not fetched now (the server's name does not resolve"), said)
        self.assertIn("Nebula fetches it by itself when it opens", said[1])

    def test_the_log_keeps_every_line_and_each_failure_in_full(self):
        conf = os.path.join(self.tmp, "conf")
        os.makedirs(conf)
        cfg_path = os.path.join(conf, "nebula.json")
        json.dump({"port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "tiers": []}, open(cfg_path, "w"))
        with mock.patch.object(prereqs, "qdrant", side_effect=RuntimeError("no network at all")), \
             mock.patch.object(prereqs, "hub_probe", return_value=None), \
             mock.patch.object(prereqs, "report", return_value=[]), \
             mock.patch("builtins.print"):
            prereqs.main(["--config", cfg_path, "--dest", self.tmp,
                          "--skip", "cuda,halo,halo-runtime,embeddings,docling,openshell,docker"])
        log = Path(conf, "prereqs.log").read_text(encoding="utf-8")
        self.assertIn("Traceback", log)
        self.assertIn("RuntimeError: no network at all", log)
        self.assertIn("qdrant: not fetched now (no network at all)", log)
        prereqs.LOG = None

    def test_lookups_folder_is_the_one_lookup_uses(self):
        self.assertEqual(prereqs.lookup_dir(cfg_in(self.tmp)), os.path.join(self.tmp, "nebula-lookup"))
        self.assertEqual(prereqs.lookup_dir(cfg_in(self.tmp, {"lookup": {"data_dir": "/x/y"}})), "/x/y")



class TinyPdfTest(unittest.TestCase):
    def test_its_cross_reference_table_points_at_its_objects(self):
        b = prereqs.tiny_pdf()
        self.assertTrue(b.startswith(b"%PDF-1.4") and b.rstrip().endswith(b"%%EOF"))
        xref = int(b.rsplit(b"startxref", 1)[1].split()[0])
        self.assertTrue(b[xref:].startswith(b"xref"))
        rows = b[xref:].split(b"\n")[3:8]
        for i, row in enumerate(rows, 1):
            self.assertTrue(b[int(row[:10]):].startswith(b"%d 0 obj" % i), (i, row))


class CudaStepTest(unittest.TestCase):
    def test_no_driver_no_cuda_download(self):
        with mock.patch.object(prereqs.shutil, "which", return_value=None), mock.patch.object(prereqs.subprocess, "run") as run:
            self.assertIn("not needed", prereqs.cuda())
        run.assert_not_called()

    def test_with_a_driver_the_pinned_wheels_are_installed_once(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(prereqs.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
             mock.patch.object(prereqs, "cuda_lib_dir", return_value=Path(d)), mock.patch.object(prereqs.subprocess, "run") as run:
            self.assertIn("installed", prereqs.cuda())
            self.assertEqual(run.call_args[0][0][-2:], prereqs.CUDA_WHEELS)
            for f in ("libcublas.so.13", "libcudart.so.13"):
                Path(d, f).write_bytes(b"")
            self.assertIn("already here", prereqs.cuda())



class PairBesidePythonTest(unittest.TestCase):
    def test_an_installers_pair_folder_beside_python_is_found(self):
        from nebula import pairnet
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d, "python", "bin", "python3")
            exe.parent.mkdir(parents=True)
            exe.write_text("")
            pair = Path(d, "pair")
            pair.mkdir()
            for b in pairnet.BINARIES:
                Path(pair, b).write_text("")
            with mock.patch.object(pairnet.sys if hasattr(pairnet, "sys") else __import__("sys"), "executable", str(exe)), \
                 mock.patch.dict(os.environ, {"NEBULA_PAIR_BIN": ""}):
                self.assertEqual(os.path.realpath(pairnet.find_bin(None)), os.path.realpath(str(pair)))


if __name__ == "__main__":
    unittest.main()
