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

    def catalog(self, sha):
        p = self.tmp / "pieces.json"
        p.write_text(json.dumps({"version": "9.9", "url": "file://" + str(self.tmp) + "/{piece}-{version}-{platform}.{ext}",
                                 "pieces": {"impulse-flashnext": {"label": "Impulse Flash-Next", "why": "the engine for Flash-Next models",
                                                                  "program": "impulse-fx",
                                                                  "files": {"linux-x86_64": {"ext": "tar.gz", "sha256": sha, "bytes": 3 * 2**20}}}}}))
        return mock.patch.multiple(pieces, CATALOG=p, platform_key=lambda: "linux-x86_64")

    def test_the_real_catalog_names_both_pieces_for_both_platforms(self):
        m = json.loads(pieces.CATALOG.read_text())
        for name in ("impulse-portable", "impulse-flashnext"):
            self.assertEqual(set(m["pieces"][name]["files"]), {"linux-x86_64", "windows-x86_64"})

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

    def test_an_unpublished_piece_says_so(self):
        with self.catalog(""):
            self.assertIsNone(pieces.pin("impulse-flashnext"))
            self.assertIn("is not published for linux-x86_64 yet", pieces.say("impulse-flashnext"))
            with self.assertRaises(RuntimeError):
                pieces.fetch("impulse-flashnext", str(self.tmp / "conf"))


if __name__ == "__main__":
    unittest.main()
