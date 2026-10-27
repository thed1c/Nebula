"""The page's API, through the gateway with the two fake engines."""
import json
import unittest
import urllib.error
import urllib.request

from nebula.replay import verify
from nebula.tests.test_escalation import Rig


def req(base, path, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def frames(raw: bytes):
    out = []
    for frame in raw.decode().split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data:"):
                p = line[5:].strip()
                out.append(p if p == "[DONE]" else json.loads(p))
    return out


class UiTest(unittest.TestCase):
    def setUp(self):
        self.rig = Rig()
        self.base = self.rig.base

    def tearDown(self):
        self.rig.stop()

    def chat(self, body):
        status, headers, raw = req(self.base, "/api/chat", body)
        self.assertEqual(status, 200, raw)
        return headers.get("X-Cockpit-Session-Id"), frames(raw)

    def view(self, sid):
        status, _h, raw = req(self.base, "/api/sessions/" + sid)
        self.assertEqual(status, 200, raw)
        return json.loads(raw)

    def test_the_page_has_the_apps_icon(self):
        self.assertIn(b'rel="icon" type="image/svg+xml" href="/favicon.svg"', req(self.base, "/")[2])
        status, headers, raw = req(self.base, "/favicon.svg")
        self.assertEqual((status, headers["Content-Type"]), (200, "image/svg+xml"))
        self.assertTrue(raw.startswith(b"<svg") and b"Halo" in raw)

    def test_page_and_config(self):
        status, headers, raw = req(self.base, "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"cockpit", raw)
        self.assertIn(b"/api/chat", raw)
        cfg = json.loads(req(self.base, "/api/config")[2])
        self.assertEqual([t["name"] for t in cfg["tiers"]], ["fast", "think", "big"])
        self.assertTrue(cfg["ledger"])

    def test_a_turn_makes_a_session_the_list_and_the_view_show(self):
        sid, fr = self.chat({"text": "hi"})
        self.assertTrue(sid.startswith("ses_"))
        self.assertEqual(fr[-1], "[DONE]")
        self.assertEqual("".join(f["choices"][0]["delta"].get("content", "") for f in fr if f != "[DONE]" and f.get("choices")), "Hello world")
        lst = json.loads(req(self.base, "/api/sessions")[2])
        self.assertEqual([(s["session_id"], s["title"], s["turns"], s["current_tier"], s["lifecycle"]) for s in lst],
                         [(sid, "hi", 1, "fast", "active")])
        v = self.view(sid)
        self.assertEqual([(i["kind"], i.get("role")) for i in v["items"]], [("message", "user"), ("reply", "assistant")])
        reply = v["items"][1]
        self.assertEqual((reply["attempt"]["tier"], reply["attempt"]["model_id"], reply["attempt"]["thinking"]), ("fast", "fastfake", 0))
        self.assertEqual(reply["decision"]["reason"], "floor")
        self.assertEqual(reply["decision"]["inputs"]["floor"], "fast")
        self.assertEqual(reply["cost"]["prompt_n"], 24)
        self.assertEqual(v["context"]["n_ctx"], 262144)
        sid2, _ = self.chat({"session_id": sid, "text": "more"})
        self.assertEqual(sid2, sid)
        self.assertEqual([i["content"] for i in self.view(sid)["items"] if i["role"] == "user"], ["hi", "more"])
        self.assertEqual(json.loads(req(self.base, "/api/sessions")[2])[0]["turns"], 2)

    def test_escalate_button_regenerates_one_tier_up_and_shows_the_move(self):
        sid, _ = self.chat({"text": "hi"})
        seq = self.view(sid)["items"][1]["seq"]
        _, fr = self.chat({"session_id": sid, "regenerate_from_seq": seq, "escalate": "big"})
        self.assertEqual((len(self.rig.a.requests), len(self.rig.b.requests)), (1, 1))
        v = self.view(sid)
        kinds = [(i["kind"], i.get("role")) for i in v["items"]]
        self.assertEqual(kinds, [("message", "user"), ("handoff", None), ("reply", "assistant")])
        self.assertEqual(v["items"][1]["to_tier"], "big")
        self.assertIn("pressed escalate", v["items"][1]["text"])
        reply = v["items"][2]
        self.assertEqual((reply["attempt"]["tier"], reply["attempt"]["model_id"]), ("big", "bigfake"))
        self.assertIn("escalate:big", reply["decision"]["reason"])
        self.assertIn("regenerations:1", reply["decision"]["reason"])
        ok, lines = verify(self.rig.path)
        self.assertTrue(ok, "\n".join(lines))

    def test_tier_pick_and_plain_regeneration(self):
        sid, _ = self.chat({"text": "hi", "tier": "think"})
        self.assertEqual(self.view(sid)["items"][1]["attempt"]["tier"], "think")
        self.assertEqual(self.rig.a.requests[-1]["chat_template_kwargs"], {"enable_thinking": True})
        seq = self.view(sid)["items"][1]["seq"]
        self.chat({"session_id": sid, "regenerate_from_seq": seq})     # from a think floor, one step up is big
        v = self.view(sid)
        self.assertEqual([i["kind"] for i in v["items"]], ["message", "handoff", "reply"])
        self.assertEqual(v["items"][-1]["attempt"]["tier"], "big")
        self.assertEqual(v["items"][-1]["decision"]["reason"], "regenerations:1")
        self.assertIn("asked for this turn 2 times", v["items"][1]["text"])

    def test_close_and_reopen(self):
        sid, _ = self.chat({"text": "hi"})
        status, _h, raw = req(self.base, f"/api/sessions/{sid}/close", {})
        self.assertEqual((status, json.loads(raw)["lifecycle"]), (200, "closed"))
        status, _h, raw = req(self.base, "/api/chat", {"session_id": sid, "text": "more"})
        self.assertEqual(status, 409)
        self.assertIn("closed", json.loads(raw)["error"]["message"])
        self.assertEqual(json.loads(req(self.base, "/api/sessions")[2])[0]["lifecycle"], "closed")
        status, _h, raw = req(self.base, f"/api/sessions/{sid}/reopen", {})
        self.assertEqual(json.loads(raw)["lifecycle"], "active")
        self.chat({"session_id": sid, "text": "more"})
        kinds = [i["kind"] for i in self.view(sid)["items"]]
        self.assertEqual(kinds, ["message", "reply", "lifecycle", "lifecycle", "message", "reply"])
        ok, lines = verify(self.rig.path)
        self.assertTrue(ok, "\n".join(lines))

    def test_an_api_client_reopens_a_closed_session_by_writing(self):
        hdr = {"Content-Type": "application/json", "X-Cockpit-Session": "k1"}
        r = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode(), headers=hdr)
        urllib.request.urlopen(r, timeout=5).read()
        sid = json.loads(req(self.base, "/api/sessions")[2])[0]["session_id"]
        req(self.base, f"/api/sessions/{sid}/close", {})
        urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello world"},
                                        {"role": "user", "content": "more"}]}).encode(), headers=hdr), timeout=5).read()
        v = self.view(sid)
        self.assertEqual(v["session"]["lifecycle"], "active")
        self.assertEqual([(i["kind"], i.get("to"), i.get("why")) for i in v["items"] if i["kind"] == "lifecycle"],
                         [("lifecycle", "closed", "user"), ("lifecycle", "active", "request")])

    def test_recent_feed(self):
        sid, _ = self.chat({"text": "hi"})
        seq = self.view(sid)["items"][1]["seq"]
        self.chat({"session_id": sid, "regenerate_from_seq": seq, "escalate": "big"})
        r = json.loads(req(self.base, "/api/recent?limit=5")[2])
        self.assertEqual([a["tier"] for a in r["attempts"]], ["big", "fast", "fast"])
        self.assertEqual(r["attempts"][0]["title"], "hi")
        self.assertEqual(r["attempts"][0]["predicted_n"], 3)
        self.assertEqual([d["reason"] for d in r["decisions"]][0], "regenerations:1,escalate:big")
        self.assertEqual(r["decisions"][0]["inputs"]["floor"], "fast")
        self.assertEqual((r["handoffs"][0]["to_tier"], r["handoffs"][0]["reason"]["source"]), ("big", "regenerations:1,escalate:big"))
        self.assertNotIn("packet", r["handoffs"][0])

    def test_monitor_and_errors(self):
        m = json.loads(req(self.base, "/api/monitor")[2])
        self.assertEqual([(e["tier"], e["up"], e["busy"], e["n_ctx"], e["model"]) for e in m["engines"]],
                         [("fast", True, 0, 262144, "fastfake"), ("think", True, 0, 262144, "fastfake"), ("big", True, 0, 262144, "bigfake")])
        self.assertEqual(m["sessions"], {"sessions": 0, "attempts": 0, "handoffs": 0})
        self.assertEqual(req(self.base, "/api/sessions/nope")[0], 404)
        self.assertEqual(req(self.base, "/api/sessions/nope/close", {})[0], 404)
        self.assertEqual(req(self.base, "/api/chat", {"text": "   "})[0], 400)
        self.assertEqual(req(self.base, "/api/nothing")[0], 404)


if __name__ == "__main__":
    unittest.main()
