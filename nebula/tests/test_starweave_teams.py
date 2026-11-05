"""Starweave 2b (nebula/profiles.py): Halo asks before it reads a model's card, builds a profile whose every line
says where it came from, ranks models per route and suggests teams you accept, edit or dismiss; an accepted team
leads the team sheet; each answer carries its team line and the work items it gave."""
import json
import os
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

from nebula import ledger as L
from nebula import profiles, team
from nebula.config import Tier
from nebula.tests.test_escalation import Rig
from nebula.tests.test_starweave_work import lead, teammate_requests, tool_names

CARD = {"license": "apache-2.0", "base_model": "Org/Base-27B", "language": ["en", "fr"],
        "model-index": [{"name": "Coder-27B", "results": [
            {"task": {"type": "text-generation"}, "dataset": {"name": "LiveCodeBench v6"}, "metrics": [{"type": "pass@1", "value": 0.612}]},
            {"task": {"type": "text-generation"}, "dataset": {"name": "MMLU-Pro"}, "metrics": [{"type": "acc", "value": 71.5}]}]}]}
README = "# Coder\n\nCoder-27B is tuned for agentic code and tool calls across long repositories.\n"


class Hub(ThreadingHTTPServer):
    """A Hugging Face Hub with two repositories: a coder with a full card, and an uncensored one with none."""

    def __init__(self):
        hub = self
        self.hits = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                hub.hits.append(self.path)
                if self.path == "/api/models/org/coder-27b":
                    return self.send(200, {"sha": "abc123", "cardData": CARD, "tags": ["code", "function-calling"],
                                           "pipeline_tag": "text-generation"})
                if self.path == "/api/models/org/wild-uncensored":
                    return self.send(200, {"sha": "def456", "cardData": {}, "tags": [], "pipeline_tag": "text-generation"})
                if self.path == "/org/coder-27b/resolve/abc123/config.json":
                    return self.send(200, {"max_position_embeddings": 131072})
                if self.path == "/org/coder-27b/resolve/abc123/README.md":
                    return self.send(200, README.encode())
                return self.send(404, {"error": "no"})

            def send(self, code, body):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        super().__init__(("127.0.0.1", 0), H)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}"


class Policy:
    def __init__(self, stages):
        self.stages = stages

    def stage(self, name):
        return self.stages.get(name)


def cfg_of(tmp, names=("fast", "think", "big"), engines=None):
    tiers = [Tier(n, f"http://127.0.0.1:1/{n}", managed=(engines or {}).get(n) and n) for n in names]
    return SimpleNamespace(base_dir=tmp, tiers=tiers, rules={}, engines=[{"name": n, "model": m} for n, m in (engines or {}).items()])


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.hub = Hub()
        self.addCleanup(lambda: (self.hub.shutdown(), self.hub.server_close()))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {"HF_ENDPOINT": self.hub.url})
        env.start()
        self.addCleanup(env.stop)

    def test_where_a_model_came_from(self):
        cfg = cfg_of(self.tmp.name, engines={"big": "/x/hub/models--org--coder-27b/snapshots/abc/coder-Q4_K_M.gguf",
                                             "fast": os.path.join(self.tmp.name, "models", "org", "small", "s.gguf")})
        self.assertEqual(profiles.repo_of(cfg, cfg.tiers[2]), "org/coder-27b")
        self.assertEqual(profiles.repo_of(cfg, cfg.tiers[0], os.path.join(self.tmp.name, "models")), "org/small")
        self.assertIsNone(profiles.repo_of(cfg, cfg.tiers[1]))

    def test_halo_asks_first_and_reads_nothing_until_told(self):
        cfg = cfg_of(self.tmp.name)
        self.assertEqual([q["model"] for q in profiles.asks(cfg)], ["fast", "think", "big"])
        self.assertIn("Can I read the model card and README for big", profiles.asks(cfg)[2]["message"])
        self.assertIn("This contacts Hugging Face", profiles.asks(cfg)[2]["message"])
        self.assertEqual(self.hub.hits, [])
        profiles.skip(cfg, "fast")
        self.assertEqual([q["model"] for q in profiles.asks(cfg)], ["think", "big"])

    def test_a_profile_says_where_each_line_came_from(self):
        cfg = cfg_of(self.tmp.name)
        prof = profiles.research(cfg, cfg.tiers[2], "org/coder-27b")
        self.assertEqual((prof["repo"], prof["revision"], prof["license"], prof["trained_context"]), ("org/coder-27b", "abc123", "apache-2.0", 131072))
        good = {g["text"]: g["source"] for g in prof["good_at"]}
        self.assertEqual(good["larger work: reports LiveCodeBench v6 61.2"], "card: model-index")    # a fraction read as percent
        self.assertEqual(good["everyday work: reports MMLU-Pro 71.5"], "card: model-index")
        self.assertEqual(good["tool calls"], "card: tag function-calling")
        self.assertEqual(prof["cautions"], [])
        self.assertEqual([q["model"] for q in profiles.asks(cfg)], ["fast", "think"])               # asked and answered
        wild = profiles.research(cfg, cfg.tiers[0], "org/wild-uncensored")
        self.assertEqual([c["text"] for c in wild["cautions"]][0], "uncensored finetune")
        self.assertIn("not said to be trained for tool calls", wild["cautions"][1]["text"])
        profiles.forget(cfg, "fast")
        self.assertNotIn("fast", profiles.load(cfg)["profiles"])

    def test_research_without_a_repository_says_so(self):
        cfg = cfg_of(self.tmp.name)
        with self.assertRaises(profiles.ProfileError) as e:
            profiles.research(cfg, cfg.tiers[1], None)
        self.assertIn("cannot tell which Hugging Face repository think came from", str(e.exception))
        with self.assertRaises(profiles.ProfileError) as e:
            profiles.research(cfg, cfg.tiers[1], "org/missing")
        self.assertEqual(e.exception.status, 404)

    def test_suggested_teams_rank_by_routing_card_and_live_results(self):
        cfg = cfg_of(self.tmp.name, names=("fast", "think", "big", "check"))
        profiles.research(cfg, cfg.tiers[2], "org/coder-27b")
        pol = Policy({"fast": "everyday", "think": "everyday", "big": "larger", "check": "critique"})
        card = profiles.suggest(cfg, pol, None)
        self.assertEqual(sorted(card["routes"]), ["critique", "everyday", "larger"])
        larger = card["routes"]["larger"]
        self.assertEqual(larger["lead"]["model"], "big")
        self.assertIn("you route larger to it", larger["lead"]["why"])
        self.assertIn("card reports LiveCodeBench v6 61.2 (reported)", larger["lead"]["why"])
        self.assertEqual(larger["reviewer"]["model"], "check")
        self.assertNotIn("check", [h["model"] for h in larger["helpers"]])
        everyday = card["routes"]["everyday"]                                   # your routing outweighs a card's claims
        self.assertEqual(everyday["lead"]["model"], "fast")
        self.assertEqual([h["model"] for h in everyday["helpers"]], ["think", "big"])
        self.assertIn("card reports MMLU-Pro 71.5 (reported)", everyday["helpers"][1]["why"])
        self.assertEqual(profiles.load(cfg)["suggestion"], card)                    # kept for the card; never applied
        self.assertEqual(cfg.rules, {})

    def test_accepting_writes_the_teams_and_good_at_lines(self):
        cfg = cfg_of(self.tmp.name)
        profiles.research(cfg, cfg.tiers[2], "org/coder-27b")
        profiles.suggest(cfg, Policy({"fast": "everyday", "think": "everyday", "big": "larger"}), None)
        sw = team.check_settings({"mode": "on", **profiles.accept(cfg)})
        self.assertEqual(sw["teams"]["larger"]["lead"], "big")
        self.assertEqual(sw["good_at"]["big"], "everyday work: reports MMLU-Pro 71.5")
        self.assertIsNone(profiles.load(cfg)["suggestion"])
        edited = profiles.accept(cfg, {"everyday": {"lead": "fast", "helpers": ["think"], "reviewer": None}})
        self.assertEqual(edited["teams"], {"everyday": {"lead": "fast", "helpers": ["think"], "reviewer": None}})
        with self.assertRaises(profiles.ProfileError):
            profiles.accept(cfg, {"everyday": {"lead": "nobody"}})
        with self.assertRaises(profiles.ProfileError):
            profiles.accept(cfg, {"sideways": {"lead": "fast"}})

    def test_live_evidence_comes_from_teamwork(self):
        led = L.Ledger(os.path.join(self.tmp.name, "l.sqlite3"))
        try:
            for state in ("done", "done", "failed"):
                i = led.add_work_item(task_id="t", session_id="s", model="think", mode="execute", task_text=state + str(id(state)))
                led.update_work_item(i, state=state)
            o = led.open_request(dialect="chat", session_key="s", key_source="header", messages=[{"role": "user", "content": "x"}],
                                 tools=None, requested_model="m", thinking=None, tier="think", engine=None)
            led.team_action(o.session_id, o.attempt_id, {"action": "hand_up", "from": "think", "to": "big", "ruling": "allowed", "mode": "on"})
            self.assertEqual(profiles.live(led)["think"], {"done": 2, "failed": 1, "handed_up_from": 1})
            sc, why = profiles.score("think", "everyday", None, profiles.live(led)["think"], None)
            self.assertGreater(sc, 0)
            self.assertEqual(why, ["2 of 3 team items done here (live)"])
        finally:
            led.close()


class TeamLineTest(unittest.TestCase):
    def test_team_line(self):
        acts = [{"action": "hand_up", "from": "fast", "to": "big", "ruling": "allowed", "mode": "on"},
                {"action": "delegate", "items": [{"id": "a", "to": "think"}, {"id": "b", "to": "think"}, {"id": "c", "to": "check"}],
                 "ruling": "allowed", "mode": "on"},
                {"action": "ask", "to": "think", "ruling": "allowed", "mode": "on"},
                {"action": "note", "ruling": "allowed", "mode": "on"},
                {"action": "delegate", "items": [], "ruling": "refused", "mode": "on"}]
        self.assertEqual(L.team_line("big", acts),
                         "Lead: big · took over from fast · delegated to think ×2, check · asked think · 1 note · Halo refused 1")
        self.assertEqual(L.team_line("fast", [{"action": "delegate", "items": [{"to": "big"}], "ruling": "allowed", "mode": "practice"}]),
                         "Lead: fast · practice: 1 call recorded, not run")


class TeamsThroughTheApp(unittest.TestCase):
    def rig(self, **rules):
        r = Rig(rules={"starweave": {"mode": "on", **rules}})
        self.addCleanup(r.stop)
        return r

    def api(self, rig, path, body=None):
        req = urllib.request.Request(rig.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def test_the_page_asks_researches_suggests_and_accepts(self):
        hub = Hub()
        self.addCleanup(lambda: (hub.shutdown(), hub.server_close()))
        rig = self.rig()
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": hub.url}):
            v = self.api(rig, "/api/starweave")
            self.assertEqual([q["model"] for q in v["asks"]], ["fast", "think", "big"])
            v = self.api(rig, "/api/starweave/research", {"model": "big", "repo": "org/coder-27b"})
            self.assertIn("big", v["profiles"])
            v = self.api(rig, "/api/starweave/skip", {"model": "fast"})
            self.assertEqual([q["model"] for q in v["asks"]], ["think"])
            v = self.api(rig, "/api/starweave/suggest", {})
            self.assertIn("everyday", v["suggestion"]["routes"])
            v = self.api(rig, "/api/starweave/accept", {"teams": {"everyday": {"lead": "fast", "helpers": ["big"], "reviewer": None}}})
        self.assertEqual(v["teams"], {"everyday": {"lead": "fast", "helpers": ["big"], "reviewer": None}})
        self.assertIsNone(v["suggestion"])
        self.assertEqual(rig.srv.cfg.rules["starweave"]["teams"]["everyday"]["helpers"], ["big"])
        rig.chat([{"role": "user", "content": "hi"}])                      # the team sheet now leads with the route's helper
        spec = {t["function"]["name"]: t["function"] for t in rig.a.requests[-1]["tools"]}
        self.assertEqual(spec[team.DELEGATE]["parameters"]["properties"]["items"]["items"]["properties"]["to"]["enum"], ["big", "think"])
        self.assertIn("- big: everyday; on this route's team; everyday work: reports MMLU-Pro 71.5", spec[team.DELEGATE]["description"])

    def test_an_answer_carries_its_team_line_and_work_items(self):
        rig = self.rig()
        rig.a.script = lead("split", (team.DELEGATE, {"items": [{"id": "sum", "to": "big", "task": "Summarise."}]}))
        rig.chat([{"role": "user", "content": "split it"}])
        sid = [s for s in rig.srv.ledger.summaries()][0]["session_id"]
        reply = [i for i in rig.srv.ledger.view(sid)["items"] if i["kind"] == "reply"][-1]
        self.assertEqual(reply["team"], "Lead: fast · delegated to big")
        self.assertEqual([(w["label"], w["model"], w["state"]) for w in reply["work_items"]], [("sum", "big", "done")])
        self.assertEqual(len(teammate_requests(rig.b)), 1)



class CommandTeammateTest(unittest.TestCase):
    """A command tier (Codex, Claude Code) is a teammate only when the owner named it on the team."""

    def turn(self, starweave):
        from types import SimpleNamespace
        from nebula.config import Tier
        big, codex, other = Tier(name="big", engine="http://x"), Tier(name="codex", cli="codex"), Tier(name="claude", cli="claude")
        return SimpleNamespace(tier=big, team_ctx=None, rules=SimpleNamespace(starweave=starweave), policy=None,
                               team_pool=[big, codex, other])

    def test_unnamed_command_tiers_stay_out(self):
        self.assertEqual([t.name for t in team.teammates(self.turn({"mode": "on"}))], [])

    def test_an_accepted_helper_joins(self):
        t = self.turn({"mode": "on", "teams": {"everyday": {"lead": "big", "helpers": ["codex"]}}})
        self.assertEqual([m.name for m in team.teammates(t)], ["codex"])

    def test_the_reviewer_setting_joins(self):
        t = self.turn({"mode": "on", "reviewer": "claude"})
        self.assertEqual([m.name for m in team.teammates(t)], ["claude"])


class CommandTeammateRunTest(unittest.TestCase):
    def test_the_lead_delegates_to_an_accepted_command_tier(self):
        import sys
        from nebula.config import Tier
        from nebula.tests.test_escalation import Rig, rows
        from nebula.tests.test_starweave_work import lead
        cwd = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(cwd, ignore_errors=True))
        prog = "import sys; t = sys.stdin.read(); open('done.txt', 'w').write('x'); print('codex wrote done.txt')"
        codex = Tier("codex", command=[sys.executable, "-c", prog], cwd=cwd, slots=1)
        rig = Rig(rules={"starweave": {"mode": "on", "teams": {"everyday": {"lead": "fast", "helpers": ["codex"]}}}},
                  extra_tiers=[codex])
        self.addCleanup(rig.stop)
        rig.a.script = lead("build", (team.DELEGATE, {"items": [{"id": "s", "to": "codex", "task": "Write done.txt."}]}))
        reply = rig.chat([{"role": "user", "content": "build it"}])
        self.assertIn("codex wrote done.txt", reply["content"])
        self.assertTrue(os.path.exists(os.path.join(cwd, "done.txt")))
        (item,) = rows(rig.path, "SELECT * FROM work_items")
        self.assertEqual((item["model"], item["state"]), ("codex", "done"))

    def test_an_explicit_only_command_tier_named_on_the_team_still_joins(self):
        """The owner's Codex is kept off the ladder ("used only when a request names it"), so ordinary chats never land
        on it; on a team that names it, the lead may still give it work."""
        import sys
        from nebula.config import Tier
        from nebula.tests.test_escalation import Rig, rows
        from nebula.tests.test_starweave_work import lead
        cwd = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(cwd, ignore_errors=True))
        prog = "import sys; t = sys.stdin.read(); open('done.txt', 'w').write('x'); print('codex wrote done.txt')"
        codex = Tier("codex", command=[sys.executable, "-c", prog], cwd=cwd, slots=1)
        rig = Rig(rules={"starweave": {"mode": "on", "teams": {"everyday": {"lead": "fast", "helpers": ["codex"]}}}},
                  extra_tiers=[codex])
        self.addCleanup(rig.stop)
        rig.srv.policy.put({"models": {"codex": {"explicit": True}}, "advertise": {}})
        rig.a.script = lead("build", (team.DELEGATE, {"items": [{"id": "s", "to": "codex", "task": "Write done.txt."}]}))
        reply = rig.chat([{"role": "user", "content": "build it"}])
        self.assertIn("codex wrote done.txt", reply["content"])
        (item,) = rows(rig.path, "SELECT * FROM work_items")
        self.assertEqual((item["model"], item["state"]), ("codex", "done"))
        self.assertNotIn("codex", [d["tier"] for d in rig.decisions() if d["reason"] != "team"])   # never the lead

    def test_an_explicit_only_model_not_named_stays_out(self):
        from types import SimpleNamespace
        from nebula.config import Tier
        big, think = Tier(name="big", engine="http://x"), Tier(name="think", engine="http://y")
        policy = SimpleNamespace(explicit=lambda n: n == "think")
        t = SimpleNamespace(tier=big, team_ctx=None, rules=SimpleNamespace(starweave={"mode": "on"}), policy=policy,
                            team_pool=[think])
        self.assertEqual(team.teammates(t), [])
        t.rules = SimpleNamespace(starweave={"mode": "on", "teams": {"everyday": {"lead": "big", "helpers": ["think"]}}})
        self.assertEqual([m.name for m in team.teammates(t)], ["think"])

if __name__ == "__main__":
    unittest.main()


class ModelTagsTest(unittest.TestCase):
    """A model's tags (set on the Models tab, shown read only on Star Weaving's crew): checked, kept tidy, read back."""
    def test_tags_are_checked_and_read_back(self):
        out = team.check_settings({"mode": "on", "tags": {"fast": ["Code ", "code", "Quick"], "big": []}})
        self.assertEqual(out["tags"], {"fast": ["code", "quick"]})
        self.assertEqual(team.settings(SimpleNamespace(starweave=out))["tags"], {"fast": ["code", "quick"]})
        for bad in ({"tags": ["code"]}, {"tags": {"fast": "code"}}, {"tags": {"fast": ["x" * (team.TAG + 1)]}},
                    {"tags": {"fast": [f"t{i}" for i in range(team.TAGS + 1)]}}):
            with self.assertRaises(team.TeamError):
                team.check_settings(bad)

    def test_the_models_tab_sets_a_models_tags(self):
        from nebula import ui
        with tempfile.TemporaryDirectory() as d:
            cfg = cfg_of(d)
            cfg.raw, cfg.path = {}, None
            cfg.tier = lambda n: next((t for t in cfg.tiers if t.name == n), None)
            srv = SimpleNamespace(cfg=cfg)
            v = ui.starweave_post(srv, "/api/starweave/tags", {"model": "fast", "tags": ["Code", "quick"]})
            self.assertEqual(v["tags"], {"fast": ["code", "quick"]})
            v = ui.starweave_post(srv, "/api/starweave/tags", {"model": "big", "tags": ["long"]})
            self.assertEqual(v["tags"], {"fast": ["code", "quick"], "big": ["long"]})
            v = ui.starweave_post(srv, "/api/starweave/tags", {"model": "fast", "tags": []})
            self.assertEqual(v["tags"], {"big": ["long"]})
            with self.assertRaises(ui.UiError):
                ui.starweave_post(srv, "/api/starweave/tags", {"model": "nobody", "tags": ["x"]})
