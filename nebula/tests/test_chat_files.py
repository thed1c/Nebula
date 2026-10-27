"""The chat's attachments: photos go to the model as pictures, text files as
text, documents read into text here; wrong kinds and sizes are refused with the reason. And Gravity's public relay
option, on the owner's word only."""
import base64
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from nebula import ui
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from nebula.tests.test_callers import call

PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()


class ContentTest(unittest.TestCase):
    def test_text_files_and_photos(self):
        self.assertEqual(ui.chat_content("hi", []), "hi")
        c = ui.chat_content("look", [{"kind": "text", "name": "notes.txt", "text": "the plan"}, {"kind": "image", "name": "a.png", "data_url": PNG}])
        self.assertEqual(c[0], {"type": "text", "text": "look\n\n[notes.txt]\nthe plan"})
        self.assertEqual(c[1], {"type": "image_url", "image_url": {"url": PNG}})
        md = base64.b64encode(b"# Title\n\nsome words").decode()
        self.assertIn("some words", ui.chat_content("", [{"kind": "file", "name": "doc.md", "data": md}]))   # read here

    def test_refusals(self):
        with self.assertRaises(ui.UiError) as e:
            ui.chat_content("x", [{"kind": "image", "name": "a.bmp", "data_url": "data:image/bmp;base64,AAAA"}])
        self.assertEqual(e.exception.status, 400)
        big = "data:image/png;base64," + "A" * (ui.MAX_IMAGE * 4 // 3 + 100)
        with self.assertRaises(ui.UiError) as e:
            ui.chat_content("x", [{"kind": "image", "name": "huge.png", "data_url": big}])
        self.assertEqual(e.exception.status, 413)
        with mock.patch("nebula.rag.ingest.docling_available", return_value=False):
            with self.assertRaises(ui.UiError) as e:
                ui.chat_content("x", [{"kind": "file", "name": "a.pdf", "data": base64.b64encode(b"%PDF-1.4").decode()}])
        self.assertEqual(e.exception.status, 415)
        self.assertIn("Docling", str(e.exception))

    def test_long_text_is_cut(self):
        c = ui.chat_content("", [{"kind": "text", "name": "a.txt", "text": "x" * (ui.MAX_TEXT + 10)}, {"kind": "text", "name": "b.txt", "text": "y"}])
        self.assertIn("the rest was cut", c)
        self.assertIn("[b.txt: left out", c)


class ChatTest(unittest.TestCase):
    def setUp(self):
        self.eng = FakeEngine("small")
        self.tmp = tempfile.mkdtemp()
        p = os.path.join(self.tmp, "nebula.json")
        Path(p).write_text(json.dumps({"port": 0, "ledger": os.path.join(self.tmp, "l.sqlite3"), "monitor": False,
                                       "tiers": [{"name": "small", "engine": self.eng.url}], "assistant": {"halo": {"on": False}},
                                       "lan": {"decided": True, "discoverable": False}}))
        self.srv = make_server(Config.from_args(["--config", p]))
        self.base = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.srv.tools.close(); self.eng.stop()

    def test_a_photo_reaches_the_model(self):
        import urllib.request
        body = json.dumps({"text": "what is in it", "attachments": [{"kind": "image", "name": "a.png", "data_url": PNG}]}).encode()
        with urllib.request.urlopen(urllib.request.Request(self.base + "/api/chat", data=body, headers={"Content-Type": "application/json"}), timeout=30) as r:
            r.read()
        last = self.eng.requests[-1]["messages"][-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual([p["type"] for p in last["content"]], ["text", "image_url"])


class PublicRelayTest(unittest.TestCase):
    def test_on_and_off_by_the_owner(self):
        from nebula import yard

        class Node:
            started = 0

            def __init__(self, server):
                self.server, self.sec = server, server.cfg.raw.setdefault("gravity", {"relays": []})

            def section(self):
                return self.sec

            def start(self):
                Node.started += 1
                return self

            def close(self):
                pass

            def view(self):
                return {"public_relay": bool(self.sec.get("public_relay"))}
        srv = mock.Mock()
        srv.cfg = Config.from_dict({"tiers": []})
        srv.gravity = Node(srv)
        with mock.patch("nebula.gravity.GravityNode", Node), mock.patch("nebula.yard._save"):
            self.assertEqual(yard.post(srv, "/api/yard/node/public-relay", {"on": True}), {"public_relay": True})
            self.assertTrue(srv.cfg.raw["gravity"]["public_relay"])
            self.assertEqual(yard.post(srv, "/api/yard/node/public-relay", {"on": False}), {"public_relay": False})
        self.assertEqual(Node.started, 2)                               # a new endpoint each time


if __name__ == "__main__":
    unittest.main()
