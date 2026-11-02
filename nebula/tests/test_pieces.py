"""Pieces Nebula downloads only when they are needed (nebula/pieces.py): pinned, sha256 checked, said plainly, and a
half or wrong download leaves nothing behind."""
import hashlib
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nebula import pieces


class PiecesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        prog = self.tmp / "impulse-fx"
        prog.write_text("#!/bin/sh\n")
        self.arc = self.tmp / "impulse-flashnext-9.9-linux-x86_64.tar.gz"
        with tarfile.open(self.arc, "w:gz") as t:
            t.add(prog, arcname="impulse-fx")

    def catalog(self, sha, ext="tar.gz"):
        p = self.tmp / "pieces.json"
        p.write_text(json.dumps({"version": "9.9", "url": "file://" + str(self.tmp) + "/{piece}-{version}-{platform}.{ext}",
                                 "pieces": {"impulse-flashnext": {"label": "Impulse Flash-Next", "why": "the engine for Flash-Next models",
                                                                  "program": "impulse-fx",
                                                                  "files": {"linux-x86_64": {"ext": ext, "sha256": sha, "bytes": 3 * 2**20}}}}}))
        return mock.patch.multiple(pieces, CATALOG=p, platform_key=lambda: "linux-x86_64")

    def test_the_real_catalog_names_every_piece_for_both_platforms(self):
        m = json.loads(pieces.CATALOG.read_text())
        for name in ("impulse-portable", "impulse-flashnext", "impulse-legacy"):
            self.assertEqual(set(m["pieces"][name]["files"]), {"linux-x86_64", "windows-x86_64"})
        leg = m["pieces"]["impulse-legacy"]
        self.assertEqual((leg["label"], leg["program"]), ("Impulse (L)", "impulse-server"))
        self.assertEqual({k: f["ext"] for k, f in leg["files"].items()}, {"linux-x86_64": "tar.xz", "windows-x86_64": "zip"})
        self.assertIn("older NVIDIA cards", leg["why"])

    def test_a_tar_xz_and_a_zip_unpack_too(self):
        prog = self.tmp / "impulse-fx"
        for ext in ("tar.xz", "zip"):
            arc = self.tmp / f"impulse-flashnext-9.9-linux-x86_64.{ext}"
            if ext == "zip":
                import zipfile
                with zipfile.ZipFile(arc, "w") as z:
                    z.write(prog, "impulse-fx")
            else:
                with tarfile.open(arc, "w:xz") as t:
                    t.add(prog, arcname="impulse-fx")
            base = self.tmp / ("conf-" + ext)
            with self.catalog(hashlib.sha256(arc.read_bytes()).hexdigest(), ext):
                self.assertEqual(pieces.fetch("impulse-flashnext", str(base)), str(base / "nebula-impulse-flashnext" / "impulse-fx"))
            self.assertEqual((base / "nebula-impulse-flashnext" / "impulse-fx").read_text(), "#!/bin/sh\n")

    def test_it_says_what_how_big_and_why_then_fetches_and_checks(self):
        base = self.tmp / "conf"
        with self.catalog(hashlib.sha256(self.arc.read_bytes()).hexdigest()):
            self.assertEqual(pieces.say("impulse-flashnext"), "downloads Impulse Flash-Next (3 MB): the engine for Flash-Next models; "
                                                               "it is not in the installer, to keep the download small")
            self.assertIsNone(pieces.find("impulse-flashnext", str(base)))
            got = pieces.fetch("impulse-flashnext", str(base))
            self.assertEqual(got, str(base / "nebula-impulse-flashnext" / "impulse-fx"))
            self.assertEqual(pieces.find("impulse-flashnext", str(base)), got)

    def test_a_wrong_download_is_refused_and_leaves_nothing(self):
        base = self.tmp / "conf"
        with self.catalog("0" * 64):
            with self.assertRaises(RuntimeError) as e:
                pieces.fetch("impulse-flashnext", str(base))
            self.assertIn("does not match its sha256", str(e.exception))
            self.assertFalse((base / "nebula-impulse-flashnext").exists())
            self.assertEqual([p.name for p in base.iterdir()], [])                      # no temporary left either

    def test_a_piece_published_with_a_later_release_names_its_own_version(self):
        with self.catalog("ab" * 32):
            self.assertTrue(pieces.pin("impulse-flashnext")["url"].endswith("/impulse-flashnext-9.9-linux-x86_64.tar.gz"))
            m = json.loads(pieces.CATALOG.read_text())
            m["pieces"]["impulse-flashnext"]["version"] = "10.0"
            pieces.CATALOG.write_text(json.dumps(m))
            p = pieces.pin("impulse-flashnext")
            self.assertEqual((p["version"], p["url"].rsplit("/", 1)[1]), ("10.0", "impulse-flashnext-10.0-linux-x86_64.tar.gz"))
        m = json.loads(pieces.CATALOG.read_text())                       # the real catalog: Impulse (L) comes with 0.1.7
        self.assertEqual(m["pieces"]["impulse-legacy"]["version"], "0.1.7")

    def test_an_unpublished_piece_says_so(self):
        with self.catalog(""):
            self.assertIsNone(pieces.pin("impulse-flashnext"))
            self.assertIn("is not published for linux-x86_64 yet", pieces.say("impulse-flashnext"))
            with self.assertRaises(RuntimeError):
                pieces.fetch("impulse-flashnext", str(self.tmp / "conf"))


if __name__ == "__main__":
    unittest.main()
