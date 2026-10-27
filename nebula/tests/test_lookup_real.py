"""Lookup for real (greenfield/ragtest.sh sets NEBULA_RAG_REAL=1): Nebula starts its own Qdrant server (loopback, its
own key), embeds with fastembed's bge-small and reranks with the MiniLM cross-encoder on the CPU, reads a PDF with
Docling (pages, headings), and answers a search with the right page. No network."""
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from nebula import server as S
from nebula.config import Config
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call

REAL = os.environ.get("NEBULA_RAG_REAL") == "1" and shutil.which("qdrant")


def make_pdf(path: Path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=A4)
    pages = [("Lighthouse keeping", ["The keeper trims the wick every evening before dusk.",
                                     "Fresnel lenses focus the beam into a narrow band."]),
             ("Tide tables", ["Spring tides follow the full and new moon.",
                              "Neap tides come at the quarter moons, with the smallest range."]),
             ("Fog signals", ["In fog the horn sounds two blasts every minute.",
                              "Mariners count the seconds between blasts to name the station."])]
    for title, lines in pages:
        c.setFont("Helvetica-Bold", 20)
        c.drawString(72, 760, title)
        c.setFont("Helvetica", 12)
        for i, line in enumerate(lines):
            c.drawString(72, 720 - 20 * i, line)
        c.showPage()
    c.save()


@unittest.skipUnless(REAL, "the real run: greenfield/ragtest.sh")
class RealLookupTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine("tiny")
        self.tmp = tempfile.mkdtemp()
        docs = Path(self.tmp, "docs")
        docs.mkdir()
        make_pdf(docs / "harbour.pdf")
        (docs / "notes.md").write_text("# Knots\n## Bowline\nThe bowline makes a fixed loop that does not slip.\n")
        self.docs = docs
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
            "tiers": [{"name": "tiny", "engine": self.engine.url}],
            "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        self.srv = S.make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        if self.srv.library is not None:
            self.srv.library.close()
        self.engine.stop()

    def test_end_to_end(self):
        lib = self.srv.library
        self.assertEqual(lib.state, "idle")                                        # no collection: no Qdrant, no model
        self.assertIsNone(lib.vectors)
        t0 = time.time()
        st, c = call(self.base, "/api/lookup/collections", {"name": "Harbour", "folders": [str(self.docs)]})
        self.assertEqual(st, 200, c)
        self.assertTrue(lib.wait_idle(600))
        docs = {Path(d["path"]).name: d for d in call(self.base, f"/api/lookup/collections/{c['id']}/documents")[1]["documents"]}
        self.assertEqual((docs["harbour.pdf"]["state"], docs["harbour.pdf"]["parser"], docs["harbour.pdf"]["pages"]),
                         ("indexed", "docling", 3), docs["harbour.pdf"])
        v = call(self.base, "/api/lookup")[1]
        self.assertEqual(lib.state, "ready", lib.why)
        self.assertEqual(v["vectors"]["mode"], "server")
        addr = v["vectors"]["address"]
        self.assertTrue(addr.startswith("127.0.0.1:"))
        self.assertEqual(v["embedder"], "fastembed:BAAI/bge-small-en-v1.5")
        self.assertTrue(v["docling"])
        with self.assertRaises(urllib.error.HTTPError) as cm:                      # its own key: nothing else gets in
            urllib.request.urlopen(f"http://{addr}/collections", timeout=5)
        self.assertEqual(cm.exception.code, 401)
        key = Path(self.tmp, "nebula-keys", "qdrant.key")
        self.assertEqual(oct(key.stat().st_mode & 0o777), "0o600")
        print(f"\n  indexed 2 files (a 3-page PDF by Docling) in {time.time() - t0:.1f} s", flush=True)

        t0 = time.time()
        r = call(self.base, "/api/lookup/search", {"query": "when are the tides smallest?"})[1]["results"]
        print(f"  search with rerank: {time.time() - t0:.2f} s; top: {r[0]['title']} p.{r[0]['page']} {r[0]['heading']!r}", flush=True)
        self.assertTrue(r[0]["path"].endswith("harbour.pdf"))
        self.assertEqual(r[0]["page"], 2)
        self.assertIn("Neap", r[0]["text"])
        r = call(self.base, "/api/lookup/search", {"query": "a loop that will not slip"})[1]["results"]
        self.assertEqual(r[0]["heading"], "Knots › Bowline")
        pid = lib.vectors.proc.proc.pid
        lib.close()
        self.srv.library = None
        time.sleep(0.5)
        self.assertFalse(os.path.exists(f"/proc/{pid}") and open(f"/proc/{pid}/stat").read().split()[2] != "Z")   # stopped


if __name__ == "__main__":
    unittest.main()
