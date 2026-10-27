"""Three Nebulas on one machine, joined by real Gravity invites through their pages, recorded as frames for two GIFs
(greenfield/constellation_demo.sh): show a friend being invited and joining a constellation, then a
third, with models shared differently: Alice has four, Bob and Carol one each; Carol has Alice's qwen-coder too, so she
keeps hers to herself; everyone sees every model, the ones not shared with them chained.
Run inside the UI container with the tree at /work/Nebula; frames and captions go to /tmp/demo."""
import json, os, sys, tempfile, threading, uuid
sys.path.insert(0, "/work/Nebula")
from nebula import gravity
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from playwright.sync_api import sync_playwright

gravity.GravityNode.LOOK_S = 2.0                       # the demo's members look at each other often
OUT = "/tmp/demo"; os.makedirs(OUT, exist_ok=True)
eng = FakeEngine("any")


def nebula(name, models):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "nebula.json")
    json.dump({"port": 0, "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False,
               "tiers": [{"name": m, "engine": eng.url, "shared": True} for m in models],
               "gravity": {"enabled": True, "name": name},
               "lan": {"decided": True, "discoverable": False, "uuid": uuid.uuid4().hex, "name": name}}, open(path, "w"))
    srv = make_server(Config.from_args(["--config", path]))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/"


people = {"Alice": nebula("Alice", ["gemma-4b", "qwen-coder", "llama-8b", "phi-mini"]),
          "Bob": nebula("Bob", ["mistral-7b"]),
          "Carol": nebula("Carol", ["qwen-coder"])}
frames, errors = [], []

with sync_playwright() as p:
    br = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
    pages = {}
    for who, (srv, base) in people.items():
        pg = br.new_page(viewport={"width": 1500, "height": 900}, device_scale_factor=1.5)
        pg.on("pageerror", lambda e, who=who: errors.append(f"{who}: {e}"))
        pg.goto(base); pg.wait_for_timeout(1200)
        pg.fill("#g-pin", "4826"); pg.fill("#g-pin2", "4826"); pg.click("#g-set"); pg.wait_for_timeout(1200)
        pages[who] = pg

    def tab(pg, name):
        g = pg.locator(f'#nav .ngroup:has([data-tab="{name}"]) .ng')
        if g.count():
            g.click(); pg.wait_for_timeout(150)
        pg.click(f'#nav [data-tab="{name}"]'); pg.wait_for_timeout(700)

    def shot(who, caption, gif, n=1, gap=260, what="map"):
        pg = pages[who]
        for i in range(n):
            if what == "map":
                box = pg.locator("#stage").bounding_box()
            else:
                box = pg.locator("#gv-body").bounding_box(); box = {**box, "height": min(box["height"], 520)}
            f = f"{OUT}/{len(frames):03d}.png"
            pg.screenshot(path=f, clip=box)
            frames.append({"file": os.path.basename(f), "caption": caption, "who": who, "gif": gif, "ms": gap if i < n - 1 else 1800})
            if i < n - 1:
                pg.wait_for_timeout(gap)

    def place(who, where):                             # the friends' systems: in view, to the right of the hole
        srv, base = people[who]
        import urllib.request
        tok = {"X-Nebula-Internal": srv.internal_token, "Content-Type": "application/json"}
        look = json.loads(urllib.request.urlopen(urllib.request.Request(base + "api/look", headers=tok)).read())
        look.setdefault("systems", {}).update(where)
        urllib.request.urlopen(urllib.request.Request(base + "api/look", data=json.dumps(look).encode(), headers=tok, method="POST")).read()
        pages[who].reload(); pages[who].wait_for_timeout(2500)

    def node(who):
        return people[who][0].gravity

    def invite(frm, to, gif):
        tab(pages[frm], "gravity")
        if pages[frm].inner_text("#gv-on").strip() == "Turn on":
            pages[frm].click("#gv-on"); pages[frm].wait_for_timeout(1500)
        pages[frm].click("#gv-inv"); pages[frm].wait_for_timeout(600)
        text = pages[frm].inner_text("#gv-out").split("\n")[0].strip()
        shot(frm, f"{frm} clicks “Invite a friend” and sends the invite to {to}", gif, what="tab")
        tab(pages[to], "gravity")
        if pages[to].inner_text("#gv-on").strip() == "Turn on":
            pages[to].click("#gv-on"); pages[to].wait_for_timeout(1500)
        pages[to].fill("#gv-join", text)
        shot(to, f"{to} pastes it on their own Nebula's Gravity tab and presses Join", gif, what="tab")
        pages[to].click("#gv-joinb"); pages[to].wait_for_timeout(2500)

    def share(frm, to, models):
        pid = next(x["id"] for x in node(frm).peers() if x["name"] == to)
        node(frm).set_grants(pid, models, confirm=models)

    # --- GIF 1: Alice invites Bob; they are a constellation; each shares what they choose
    for who in people:
        tab(pages[who], "home")
    pages["Alice"].wait_for_timeout(1500)
    shot("Alice", "Alice's Nebula: her own four models, round her black hole", 1)
    invite("Alice", "Bob", 1)
    bob = next(x["id"] for x in node("Alice").peers() if x["name"] == "Bob")
    place("Alice", {bob: [1.1, 0.4]})
    tab(pages["Alice"], "home"); pages["Alice"].click("text=Fit"); pages["Alice"].wait_for_timeout(1500)
    shot("Alice", "Bob joined: a wormhole to his Nebula. Alice and Bob are a constellation", 1, n=6)
    share("Bob", "Alice", ["mistral-7b"])
    share("Alice", "Bob", ["gemma-4b", "llama-8b"])
    pages["Alice"].wait_for_timeout(6500)
    shot("Alice", "Bob shares his mistral-7b with Alice: lit, a route through Gravity", 1, n=4)
    tab(pages["Bob"], "home")
    alice_at_bob = next(x["id"] for x in node("Bob").peers() if x["name"] == "Alice")
    place("Bob", {alice_at_bob: [1.1, 0.4]})
    tab(pages["Bob"], "home"); pages["Bob"].click("text=Fit"); pages["Bob"].wait_for_timeout(2500)
    shot("Bob", "Bob's view: Alice shares 2 of her 4. The other two are chained on his map", 1, n=3)

    # --- GIF 2: a third, Carol; models shared differently
    invite("Alice", "Carol", 2)
    carol = next(x["id"] for x in node("Alice").peers() if x["name"] == "Carol")
    share("Alice", "Carol", ["gemma-4b", "llama-8b", "phi-mini"])  # Carol has qwen-coder herself, and keeps hers
    place("Alice", {bob: [1.1, 0.22], carol: [1.1, 0.8]})
    tab(pages["Alice"], "home"); pages["Alice"].click("text=Fit"); pages["Alice"].wait_for_timeout(6500)
    shot("Alice", "Carol joins. She has qwen-coder too and keeps hers to herself: chained on Alice's map", 2, n=5)
    alice_at_carol = next(x["id"] for x in node("Carol").peers() if x["name"] == "Alice")
    place("Carol", {alice_at_carol: [1.1, 0.4]})
    tab(pages["Carol"], "home"); pages["Carol"].click("text=Fit"); pages["Carol"].wait_for_timeout(2500)
    shot("Carol", "Carol's view: Alice's qwen-coder is chained for her, the other three free (shared with her)", 2, n=3)
    tab(pages["Alice"], "gravity"); pages["Alice"].wait_for_timeout(1500)
    shot("Alice", "Alice's Gravity tab: what she shares with each friend, ticked per person", 2, what="tab")
    br.close()

json.dump({"frames": frames, "errors": errors}, open(f"{OUT}/frames.json", "w"), indent=1)
print("frames", len(frames), "errors", errors)
