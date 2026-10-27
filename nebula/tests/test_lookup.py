"""Lookup's library (nebula/rag): collections indexed into Nebula's own vectors and SQL, hybrid search with
citations, folder watching, tenancy (a guest sees only what is shared with it, nothing by default), the tools in an
agent's turn and over MCP, the security tool kept to the owner, answers that cite.

These run where Lookup's parts are installed (greenfield/ragtest.sh); a small hashed embedder stands in for the
model here, and greenfield/ragtest.sh's real run uses the real embedder, reranker, Qdrant server and Docling."""
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nebula import server as S
from nebula.config import Config
from nebula.rag import ingest
from nebula.rag.library import available
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call

OK, WHY = available()


class HashEmbedder:
    """Words hashed into 256 dimensions: close texts share words. Deterministic, no model."""
    name, dims_ = "test:hash256", 256

    def _v(self, text):
        v = [0.0] * self.dims_
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dims_] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)

    def dims(self):
        return self.dims_


def memory_hook(lib):
    from nebula.rag.vectors import Vectors
    lib.embedder, lib.reranker, lib.vectors = HashEmbedder(), None, Vectors("memory")
    lib.settings["rerank"] = False


class ChunkTest(unittest.TestCase):
    def test_headings_and_sizes(self):
        secs = ingest._markdown("# Guide\nintro\n## Install\n" + "pip install it. " * 300 + "\n## Use\nrun it\n```\n# not a heading\n```\n")
        self.assertEqual([s["heading"] for s in secs], ["Guide", "Guide › Install", "Guide › Use"])
        chunks = ingest.chunk(secs, size=500, overlap=50)
        self.assertTrue(all(len(c["text"]) <= 560 for c in chunks))
        self.assertTrue(any(c["heading"] == "Guide › Install" for c in chunks))
        self.assertIn("# not a heading", chunks[-1]["text"])

    def test_html_and_skips(self):
        secs = ingest._html("<html><title>T</title><script>x()</script><h1>Top</h1><p>Hello &amp; bye</p></html>")
        self.assertEqual(secs[0]["heading"], "Top")
        self.assertIn("Hello & bye", secs[0]["text"])
        p = Path(tempfile.mkdtemp(), "a.pdf")
        p.write_bytes(b"%PDF-1.4")
        with self.assertRaises(ingest.Skip):
            ingest.read(p, use_docling=False)


@unittest.skipUnless(OK, WHY or "")
class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine("tiny")
        self.tmp = tempfile.mkdtemp()
        self.docs = Path(self.tmp, "docs")
        (self.docs / "sub").mkdir(parents=True)
        (self.docs / "boats.md").write_text("# Boats\n## Sails\nA sloop has one mast and a mainsail.\n## Hulls\nA catamaran has two hulls.\n")
        (self.docs / "sub" / "kitchen.txt").write_text("Sourdough needs a starter, flour, water and salt. Bake at 230 C.")
        (self.docs / "page.html").write_text("<h1>Comets</h1><p>A comet's tail points away from the sun.</p>")
        (self.docs / ".hidden.md").write_text("secret stuff")
        self.path = os.path.join(self.tmp, "nebula.json")
        Path(self.path).write_text(json.dumps({
            "port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
            "tiers": [{"name": "tiny", "engine": self.engine.url}],
            "callers": [{"id": "friend-app", "name": "A friend's app", "guest": True}],
            "lookup": {"watch_seconds": 0.5},
            "lan": {"decided": True, "discoverable": False, "uuid": "u", "name": "n"}}))
        S.LIBRARY_HOOK = memory_hook
        self.srv = S.make_server(Config.from_args(["--config", self.path]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        S.LIBRARY_HOOK = None
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close()
        if self.srv.library is not None:
            self.srv.library.close()
        self.engine.stop()

    def collection(self, name="Notes"):
        st, c = call(self.base, "/api/lookup/collections", {"name": name, "folders": [str(self.docs)]})
        self.assertEqual(st, 200, c)
        self.assertTrue(self.srv.library.wait_idle(60))
        return c

    def test_index_search_cite(self):
        c = self.collection()
        v = call(self.base, "/api/lookup")[1]
        row = v["collections"][0]
        self.assertEqual((row["documents"], row["indexed"], row["lit"]), (3, 3, True))
        self.assertEqual(v["vectors"]["mode"], "memory")
        st, r = call(self.base, "/api/lookup/search", {"query": "how many hulls does a catamaran have"})
        self.assertEqual(st, 200, r)
        top = r["results"][0]
        self.assertTrue(top["path"].endswith("boats.md"))
        self.assertEqual((top["heading"], top["n"]), ("Boats › Hulls", 1))
        self.assertEqual(r["results"][0]["collection"], c["id"])
        docs = call(self.base, f"/api/lookup/collections/{c['id']}/documents")[1]["documents"]
        self.assertNotIn(".hidden.md", " ".join(d["path"] for d in docs))           # hidden files are not read

    def test_watching_a_folder(self):
        c = self.collection()
        (self.docs / "new.md").write_text("# Volcanoes\nMagma becomes lava when it reaches the surface.\n")
        (self.docs / "sub" / "kitchen.txt").unlink()
        end = time.time() + 20
        while time.time() < end:
            paths = [d["path"] for d in call(self.base, f"/api/lookup/collections/{c['id']}/documents")[1]["documents"]]
            if any(p.endswith("new.md") for p in paths) and not any(p.endswith("kitchen.txt") for p in paths):
                break
            time.sleep(0.3)
        self.assertTrue(any(p.endswith("new.md") for p in paths), paths)
        self.assertFalse(any(p.endswith("kitchen.txt") for p in paths), paths)
        self.srv.library.wait_idle(30)
        hits = call(self.base, "/api/lookup/search", {"query": "sourdough starter flour"})[1]["results"]
        self.assertFalse(any(h["path"].endswith("kitchen.txt") for h in hits))           # its vectors went with it

    def test_guests_see_only_what_is_shared(self):
        c = self.collection()
        other = self.collection("Private")
        st, made = call(self.base, "/api/yard/callers/friend-app/keys", {"label": "theirs"})
        key = made["key"]

        def mcp(method, params=None, k=key):
            return call(self.base, "/mcp", {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
                        {"Authorization": f"Bearer {k}"} if k else None)

        self.assertEqual([t["name"] for t in mcp("tools/list")[1]["result"]["tools"]], ["search", "read_passage", "collections"])
        out = mcp("tools/call", {"name": "search", "arguments": {"query": "catamaran hulls"}})[1]["result"]
        self.assertIn("no passages found", out["content"][0]["text"])                    # nothing shared by default
        st, r = call(self.base, f"/api/lookup/collections/{c['id']}/share", {"tenants": ["caller:friend-app"]})
        self.assertEqual(st, 200, r)
        out = mcp("tools/call", {"name": "search", "arguments": {"query": "catamaran hulls"}})[1]["result"]
        self.assertIn("[1] Boats", out["content"][0]["text"])
        listed = mcp("tools/call", {"name": "collections"})[1]["result"]["content"][0]["text"]
        self.assertIn(c["id"], listed)
        self.assertNotIn(other["id"], listed)                                             # the other one stays private
        chunk = int(re.search(r"chunk (\d+)", out["content"][0]["text"]).group(1))
        hidden = self.srv.library.catalog.chunk_ids(other["id"])[0]
        self.assertFalse(mcp("tools/call", {"name": "read_passage", "arguments": {"chunk": chunk}})[1]["result"]["isError"])
        self.assertTrue(mcp("tools/call", {"name": "read_passage", "arguments": {"chunk": hidden}})[1]["result"]["isError"])
        self.assertEqual(mcp("tools/list", k="nbc_wrong")[0], 200)                         # no key needed here by default…
        call(self.base, "/api/stars/api/keys", {"label": "mine"})
        call(self.base, "/api/stars/api", {"require_key": True})
        self.assertEqual(mcp("tools/list", k="nbc_wrong")[0], 401)                        # …and refused once one is
        self.assertIn("share.changed", [e["kind"] for e in self.srv.seclog.read()])

    def test_the_assistant_searches_and_the_security_log_is_the_owners(self):
        self.collection()
        import urllib.request
        req = urllib.request.Request(self.base + "/api/chat", data=json.dumps({"text": "look up catamaran hulls"}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        rows = self.srv.ledger.db.execute("SELECT payload FROM events WHERE kind = 'tool_round' ORDER BY seq DESC LIMIT 1").fetchone()
        self.assertIsNotNone(rows)
        self.assertIn("[1] Boats", rows[0])
        from nebula.engine import CALL_TENANT
        sec = self.srv.tools.local["security"]
        tok = CALL_TENANT.set("caller:friend-app")
        try:
            self.assertTrue(self.srv.tools.call("security__read_log", {}).startswith("error:"))
        finally:
            CALL_TENANT.reset(tok)
        self.assertTrue(json.loads(sec.call("read_log", {})) is not None)

    def test_ask_answers_with_citations(self):
        self.collection()
        st, r = call(self.base, "/api/lookup/ask", {"question": "what does a sloop have"})
        self.assertEqual(st, 200, r)
        self.assertTrue(r["answer"])
        self.assertTrue(r["citations"])
        self.assertIn("title", r["citations"][0])
        row = self.srv.ledger.db.execute("SELECT inputs FROM decisions ORDER BY rowid DESC LIMIT 1").fetchone()
        self.assertEqual(json.loads(row[0]).get("label"), "lookup/ask")                 # through the Switchyard

    def test_the_collection_wizard(self):
        answers = {"folder": str(self.docs), "name": "Notes", "watch": True, "shared": []}
        st, p = call(self.base, "/api/wizards/collection/plan", {"answers": answers})
        self.assertEqual(st, 200, p)
        self.assertIn("3 files", p["lines"][0])
        self.assertEqual(self.srv.library.catalog.collections(), [])                     # Cancel: nothing
        st, job = call(self.base, "/api/wizards/collection/apply", {"answers": answers, "wait": True})
        self.assertEqual((st, job["state"]), (200, "done"), job)
        self.assertTrue(self.srv.library.wait_idle(60))
        self.assertEqual(call(self.base, "/api/lookup")[1]["collections"][0]["indexed"], 3)
        st, v = call(self.base, f"/api/wizards/jobs/{job['id']}/undo", {})
        self.assertEqual(st, 200, v)
        self.assertEqual(self.srv.library.catalog.collections(), [])

    def test_bad_requests(self):
        self.assertEqual(call(self.base, "/api/lookup/collections", {"name": "x", "folders": ["/nope/nowhere"]})[0], 400)
        self.assertEqual(call(self.base, "/api/lookup/collections", {"name": "", "folders": [str(self.docs)]})[0], 400)
        c = self.collection()
        self.assertEqual(call(self.base, f"/api/lookup/collections/{c['id']}/share", {"tenants": ["everyone"]})[0], 400)
        self.assertEqual(call(self.base, "/api/lookup/collections/none-here/reindex", {})[0], 404)
        self.assertEqual(call(self.base, f"/api/lookup/collections/{c['id']}/remove", {})[0], 200)
        self.assertEqual(call(self.base, "/api/lookup")[1]["collections"], [])


@unittest.skipUnless(OK, WHY or "")
class InstallTest(unittest.TestCase):
    def test_a_download_that_does_not_match_is_refused(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from nebula.rag import vectors as V

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200); self.send_header("Content-Length", "5"); self.end_headers(); self.wfile.write(b"bogus")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        real = V.urllib.request.urlopen
        V.urllib.request.urlopen = lambda url, timeout=60: real(f"http://127.0.0.1:{srv.server_port}/x", timeout=timeout)
        try:
            d = tempfile.mkdtemp()
            with self.assertRaises(V.VectorsError) as cm:
                V.install(d)
            self.assertIn("sha256", str(cm.exception))
            self.assertFalse(os.path.exists(os.path.join(d, "bin", "qdrant")))
        finally:
            V.urllib.request.urlopen = real
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
