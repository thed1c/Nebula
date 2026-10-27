"""The page, loaded in Chromium (greenfield/guitest.sh): every tab and the main panels opened, every console error and
page error collected; it fails on any. Run inside the UI container with the tree at /work/Nebula."""
import json, os, sys, tempfile, threading, time
sys.path.insert(0, "/work/Nebula")
from nebula.config import Config
from nebula.server import make_server
from nebula.tests.fake_engine import FakeEngine
from playwright.sync_api import sync_playwright

os.environ.update({"NEBULA_FAKE_CAPS": "8.9", "NEBULA_FAKE_RAM_MB": "65536",
                   "NEBULA_FAKE_CARDS": json.dumps([{"uuid": "g", "index": 0, "name": "RTX 4070", "memory_mb": 12282, "used_mb": 0}])})
eng = FakeEngine("small")
# a pretend Hugging Face cache with one model in it
from nebula.tests.test_hfcache import blob as _blob
_hf = tempfile.mkdtemp(); _blob(_hf, "someone/cached-GGUF", "rev1", "cached-Q4_K_M.gguf", "e1"); os.environ["NEBULA_HF_CACHE"] = _hf
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "nebula.json")
import socket
_s = socket.socket(); _s.bind(("127.0.0.1", 0)); kids_port = _s.getsockname()[1]; _s.close()
json.dump({"port": 0, "ledger": os.path.join(tmp, "l.sqlite3"), "monitor": False,
           "tiers": [{"name": "small", "engine": eng.url, "shared": True}, {"name": "big", "engine": eng.url},
                     {"name": "cloud-mini", "provider": "openai-compatible", "base_url": "https://api.example.invalid/v1", "model": "m"}],
           "api": {"entries": [{"id": "kids", "name": "Kids", "scope": "loopback", "port": kids_port, "formats": ["openai", "ollama"],
                                "caller": "entry-kids", "require_key": True}]},
           "gravity": {"enabled": True, "peers": [{"id": "5" * 64, "name": "Sam's PC", "addr": {"id": "5" * 64}, "grants": ["small"],
                                                   "role": "friend", "joined_at": 0}]},
           "engines": [{"name": "boxed", "kind": "llama.cpp", "model": "m.gguf", "port": 18399, "context": 2048}],
           "callers": [{"id": "app", "name": "An app", "tags": ["app/*"], "models": ["small"]},
                       {"id": "entry-kids", "name": "Kids (entry point)", "models": ["small"]}],
           "hosts": [{"name": "far", "wake": {"kind": "wol", "mac": "aa:bb:cc:dd:ee:ff"}}],
           "lan": {"decided": True, "discoverable": False}}, open(path, "w"))
srv = make_server(Config.from_args(["--config", path]))
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}/"
errors = []
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
    page = b.new_page(viewport={"width": 1400, "height": 900})
    page.on("console", lambda m: errors.append(f"console {m.type}: {m.text} {m.location.get('url', '')}") if m.type == "error" else None)
    page.on("requestfailed", lambda r: errors.append(f"request failed: {r.url}"))
    page.on("response", lambda r: errors.append(f"{r.status}: {r.url}") if r.status >= 400 else None)
    page.on("pageerror", lambda e: errors.append(f"page error: {e}"))
    page.goto(base)
    page.wait_for_timeout(1500)
    # first run: the page asks how to sign in before anything else (design decision: a passkey or a PIN)
    assert page.is_visible("#gate") and "protect this nebula" in page.inner_text("#gatebox").lower(), ("the sign-in setup comes first", errors, page.inner_html("#gatebox")[:300])
    page.screenshot(path="/tmp/signin-setup.png")
    page.fill("#g-pin", "4826"); page.fill("#g-pin2", "4826"); page.click("#g-set"); page.wait_for_timeout(1500)
    assert not page.is_visible("#gate"), "a PIN set: the page opens"
    import urllib.request as _u
    assert _u.urlopen(_u.Request(base + "api/stars", headers={"X-Nebula-Internal": srv.internal_token})).status == 200
    def tab_click(tab):
        g = page.locator(f'#nav .ngroup:has([data-tab="{tab}"]) .ng')
        if g.count():
            g.click(); page.wait_for_timeout(150)
        page.click(f'#nav [data-tab="{tab}"]')
    for tab in ("plan", "engines", "callers", "gravity", "ledger", "system", "home"):
        tab_click(tab)
        page.wait_for_timeout(900)
        print("tab", tab, "ok", flush=True)
    import urllib.request, urllib.error
    for _ in range(3):                                                         # bad keys at the Kids Quasar: they bounce
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{kids_port}/v1/models", headers={"Authorization": "Bearer nope"}), timeout=5)
        except urllib.error.HTTPError:
            pass
    page.wait_for_timeout(2500)
    page.screenshot(path="/tmp/map.png")
    assert page.locator('.body.quasar:not(.leaving)').count() == 2, f"two Quasars (This PC, Kids): {page.locator('.body.quasar').count()}"
    assert page.locator('.body.mcp[data-key^="t:"]').count() == 0, "tool stars are not drawn round the core"
    assert page.locator('.body[data-key="c:lookup"] .swarm .col').count() == 6, "Lookup's swarm: six empty collectors with no collection"
    assert "HALO" in page.inner_text('.body[data-key="c:assistant"]').upper(), "the Assistant is called Halo"
    sat = page.eval_on_selector('.body[data-key="c:lookup"] .swarm .col', "e => getComputedStyle(e).borderTopColor")
    assert sat not in ("rgba(0, 0, 0, 0)", ""), sat   # silver satellites, visible
    p1 = page.locator('.body[data-key="c:lookup"] .swarm .col').first.bounding_box(); page.wait_for_timeout(700)
    p2 = page.locator('.body[data-key="c:lookup"] .swarm .col').first.bounding_box()
    assert (p1["x"], p1["y"]) != (p2["x"], p2["y"]), "the swarm does not move"
    # Gravity's peers as star systems joined by wormholes (design decision 10-21); the cloud stars chained beyond the line
    assert page.locator('.hole.remote[data-sys]').count() == 1, "Sam's PC: a star system through Gravity"
    assert "SAM" in page.inner_text('.hole.remote').upper(), page.inner_text('.hole.remote')
    page.click("text=Fit"); page.wait_for_timeout(1200); page.screenshot(path="/tmp/wormhole.png")
    page.click("text=Home"); page.wait_for_timeout(2000)                         # the camera settles before the clicks below
    cb = page.locator('.body.cloud').first.bounding_box()                      # a cloud star in its chains
    page.screenshot(path="/tmp/chains.png", clip={"x": cb["x"] - 70, "y": cb["y"] - 60, "width": cb["width"] + 220, "height": cb["height"] + 120})
    page.screenshot(path="/tmp/map-swarm.png", clip={"x": 160, "y": 470, "width": 260, "height": 220})
    # callers call the API (design decision 10-21): a caller's stream runs into the Quasar it calls through, the call's light
    # goes on through the Quasar into the router, and the answer comes back to the caller as small bursts
    eng.on_request = lambda _b: time.sleep(1.5)                                 # long enough to see it in flight
    def _call():
        rq = urllib.request.Request(base + "v1/chat/completions", data=json.dumps({"model": "small", "messages": [{"role": "user", "content": "hello"}]}).encode(),
                                    headers={"Content-Type": "application/json", "X-Nebula-Caller": "writer"})
        urllib.request.urlopen(rq, timeout=20).read()
    threading.Thread(target=_call, daemon=True).start()
    for i, ms in enumerate((900, 700, 900, 700, 600, 600, 600, 600)):
        page.wait_for_timeout(ms); page.screenshot(path=f"/tmp/exchange-{i + 1}.png")
    page.wait_for_timeout(2500); page.screenshot(path="/tmp/exchange-rest.png")
    eng.on_request = None
    assert page.locator("#antigrav").count() == 0, "no Anti-Gravity box: the cloud stars are chained instead"
    print("map: quasars", page.locator('.body.quasar').count(), "worm", page.locator('.body.worm').count(), flush=True)
    q = page.locator('.body.quasar[data-key="q:kids"]').bounding_box()           # tap the Kids Quasar: its panel
    page.mouse.click(q["x"] + q["width"] / 2, q["y"] + q["height"] / 2); page.wait_for_timeout(700)
    panel = page.inner_text("#starpanel")
    assert "Kids" in panel and "bounced" in panel, panel[:300]
    page.screenshot(path="/tmp/quasar.png")
    page.click("#starpanel [data-close]")
    ab0 = page.locator('.body[data-key="c:assistant"]').bounding_box()                # Halo's panel: his own model, GPU/CPU
    page.mouse.click(ab0["x"] + ab0["width"] / 2, ab0["y"] + ab0["height"] / 2); page.wait_for_timeout(900)
    if page.locator("#starpanel [data-halo-where]").count():
        page.locator("#starpanel").screenshot(path="/tmp/halo-panel.png")
    page.click("#starpanel [data-close]")
    # the mouse: right-click = configure (the black hole: what it shows, Gravity, add a model); shift+right-click = its look
    hb = page.locator("#hole").bounding_box(); hx, hy = hb["x"] + hb["width"] / 2, hb["y"] + hb["height"] / 2
    for _try in range(4):                         # a ring star can pass over the hole's middle: try again a moment later
        page.mouse.click(hx, hy, button="right"); page.wait_for_timeout(300)
        if page.is_visible("#holemenu"):
            break
        page.keyboard.press("Escape"); page.wait_for_timeout(700)
    assert page.is_visible("#holemenu"), ("right-click on the black hole: its menu",
                                          page.evaluate(f"document.elementFromPoint({hx}, {hy})?.outerHTML.slice(0, 160)"))
    assert page.locator("#hm-models label").count() >= 2, "the models on the map, in the black hole's menu"
    assert "become infinite" in page.inner_text(".brand").lower() and "event horizon" in page.inner_text(".brand").lower(), "Nebula: Event Horizon, Become Infinite"
    assert page.locator("#nav .ng", has_text="Navigation").count() == 1 and page.locator("#nav .ng", has_text="Engineering").count() == 1
    assert page.locator('#nav [data-tab="ledger"]').inner_text() == "Captain's Log"
    page.uncheck('#holemenu [data-show="c:lookup"]'); page.wait_for_timeout(3500)
    assert page.locator('.body[data-key="c:lookup"]').count() == 0, "Show Lookup off: Lookup leaves the map"
    page.check('#holemenu [data-show="c:lookup"]'); page.wait_for_timeout(1500)
    assert page.locator('.body[data-key="c:lookup"]').count() == 1, "Show Lookup on: it is back"
    page.screenshot(path="/tmp/holemenu.png")
    page.click("#hm-done")
    page.keyboard.down("Shift"); page.mouse.click(hx, hy, button="right"); page.keyboard.up("Shift"); page.wait_for_timeout(300)
    assert page.is_visible("#nodeedit") and page.is_visible("#ne-fx"), "shift+right-click on the black hole: its look and effects"
    page.uncheck('#ne-fx [data-fx="glow"]'); page.wait_for_timeout(200)
    assert "fx-noglow" in page.get_attribute("#stage", "class"), "glow off"
    page.check('#ne-fx [data-fx="glow"]'); page.click("#ne-done")
    lb = page.locator('.body[data-key="c:lookup"]').bounding_box()
    page.keyboard.down("Shift"); page.mouse.click(lb["x"] + lb["width"] / 2, lb["y"] + lb["height"] / 2, button="right"); page.keyboard.up("Shift"); page.wait_for_timeout(300)
    assert page.is_visible("#ne-swarmw"), "Lookup's look: its satellites' colour"
    page.click('#ne-swarm button[data-c="#4cc9f0"]'); page.wait_for_timeout(200)
    assert "#4cc9f0" in page.get_attribute("#stage", "style"), "the satellites' colour"
    page.screenshot(path="/tmp/swarm-colour.png", clip={"x": lb["x"] - 80, "y": lb["y"] - 80, "width": 260, "height": 220})
    page.click("#ne-done")
    # the colour wheel: a star's colour picked on the wheel, here a red at the wheel's right edge
    mb0 = page.locator('.body[data-key="m:small"]').bounding_box()
    page.keyboard.down("Shift"); page.mouse.click(mb0["x"] + mb0["width"] / 2, mb0["y"] + mb0["height"] / 2, button="right"); page.keyboard.up("Shift"); page.wait_for_timeout(300)
    wb = page.locator("#ne-wheel canvas").bounding_box()
    assert wb and wb["width"] > 100, "the colour wheel"
    page.mouse.click(wb["x"] + wb["width"] - 4, wb["y"] + wb["height"] / 2); page.wait_for_timeout(300)
    hexv = page.input_value("#ne-wheel input[type=text]")
    assert hexv[1:3] == "ff" and int(hexv[3:5], 16) < 0x30 and int(hexv[5:7], 16) < 0x30, hexv   # a red
    c = page.eval_on_selector('.body[data-key="m:small"]', "e => e.style.getPropertyValue('--c')")
    assert c.strip() == hexv, (c, hexv)
    page.locator("#nodeedit").screenshot(path="/tmp/wheel.png")
    page.click("#ne-reset"); page.wait_for_timeout(200)
    # right-click on Lookup: its quick menu; Add a folder… opens the folder picker, then the collection wizard with it
    page.mouse.click(lb["x"] + lb["width"] / 2, lb["y"] + lb["height"] / 2, button="right"); page.wait_for_timeout(600)
    assert page.is_visible("#coremenu") and page.is_visible("#cm-add"), "right-click on Lookup: its quick menu"
    page.screenshot(path="/tmp/lookup-menu.png")
    page.click("#cm-add"); page.wait_for_timeout(800)
    assert page.is_visible("#picker") and page.locator("#pk-places button").count() >= 2, "Browse: the folder picker"
    page.click("#pk-places button >> nth=0"); page.wait_for_timeout(500)
    page.screenshot(path="/tmp/picker.png")
    chosen = page.inner_text("#pk-path")
    page.click("#pk-use"); page.wait_for_timeout(1200)
    assert page.is_visible("#wizard") and page.input_value('#wzbody input[name=folder]') == chosen, "the collection wizard, with the folder"
    assert page.is_visible("#wzbody [data-browse]"), "the folder field has its own Browse…"
    page.click("#wzbody [data-wz-close]"); page.wait_for_timeout(300)
    # the Assistant: three gyroscope rings flipping round it; right-click: its quick menu; a tap: its panel
    ab = page.locator('.body[data-key="c:assistant"]').bounding_box()
    assert page.locator('.body[data-key="c:assistant"] .gyro i').count() == 3, "the gyroscope's rings"
    g1 = page.eval_on_selector('.body[data-key="c:assistant"] .gyro i', "e => getComputedStyle(e).transform"); page.wait_for_timeout(400)
    g2 = page.eval_on_selector('.body[data-key="c:assistant"] .gyro i', "e => getComputedStyle(e).transform")
    assert g1 != g2, "the rings do not turn"
    page.screenshot(path="/tmp/gyro.png", clip={"x": ab["x"] - 60, "y": ab["y"] - 60, "width": ab["width"] + 120, "height": ab["height"] + 120})
    page.mouse.click(ab["x"] + ab["width"] / 2, ab["y"] + ab["height"] / 2, button="right"); page.wait_for_timeout(600)
    assert page.is_visible("#coremenu") and page.is_visible("#cm-halo") and page.is_visible("#cm-new"), "right-click on the Assistant: its quick menu"
    page.screenshot(path="/tmp/assistant-menu.png")
    page.click("#cm-all"); page.wait_for_timeout(700)
    assert page.is_visible("#starpanel"), "All settings…: its panel"
    page.click("#starpanel [data-close]")
    # the routing wizard, from the black hole's menu
    page.mouse.click(hx, hy, button="right"); page.wait_for_timeout(300)
    page.click("#hm-routing"); page.wait_for_timeout(1000)
    assert page.is_visible("#wizard") and "routing" in page.inner_text("#wzbody").lower(), "Set up routing"
    page.screenshot(path="/tmp/routing-wizard.png")
    page.click("#wzbody [data-wz-close]"); page.wait_for_timeout(300)
    # a thrown star comes back into view
    mb = page.locator('.body[data-key^="m:"]').first.bounding_box(); sb = page.locator("#stage").bounding_box()
    page.mouse.move(mb["x"] + mb["width"] / 2, mb["y"] + mb["height"] / 2); page.mouse.down()
    for k in range(1, 6): page.mouse.move(mb["x"] + 60 * k, mb["y"] + 10 * k); page.wait_for_timeout(16)
    page.mouse.up(); page.wait_for_timeout(2500)
    mb2 = page.locator('.body[data-key^="m:"]').first.bounding_box()
    assert sb["x"] <= mb2["x"] + mb2["width"] / 2 <= sb["x"] + sb["width"], f"a thrown star stays in view: {mb2} {sb}"
    print("mouse: menus, looks, swarm colour, keep in view ok", flush=True)
    tab_click("system"); page.wait_for_timeout(800)          # signing out, and in again with the PIN
    assert "A PIN is set" in page.inner_text("#signin"), page.inner_text("#signin")
    page.click("#si-out"); page.wait_for_timeout(1500)
    assert page.is_visible("#gate") and "sign in" in page.inner_text("#gatebox").lower(), "signed out: the sign-in screen"
    page.screenshot(path="/tmp/signin.png")
    page.fill("#g-pin", "4826"); page.click("#g-go"); page.wait_for_timeout(2500)
    assert not page.is_visible("#gate"), "signed in again"
    page.click('#nav [data-tab="home"]'); page.wait_for_timeout(800)
    print("sign-in: set, out, in ok", flush=True)
    tab_click("engines"); page.wait_for_timeout(800)
    page.click("#eng-new"); page.wait_for_timeout(800)                          # an engine's settings: speculative decoding for both
    page.fill('#eng-form input[name=name]', "spec"); page.fill('#eng-form input[name=model]', "models/x.gguf"); page.fill('#eng-form input[name=port]', "8099")
    page.select_option('#eng-form select[name=kind]', "llama.cpp"); page.wait_for_timeout(400)
    assert page.locator('#eng-form summary', has_text="Speculative decoding").count() == 1, "llama.cpp has speculative decoding"
    page.locator('#eng-form summary', has_text="Speculative decoding").click(); page.wait_for_timeout(200)
    assert "mtp" in page.eval_on_selector_all('#eng-form select[data-opt=spec_type] option', "os => os.map(o => o.textContent)"), "MTP for llama.cpp"
    page.locator('#eng-form details', has_text="Speculative decoding").screenshot(path="/tmp/engine-spec.png")
    page.click("#eng-cancel"); page.wait_for_timeout(300)
    page.click("#eng-deploy"); page.wait_for_timeout(800)                       # Deploy your own (a wizard)
    assert page.is_visible("#wizard"), "the wizard did not open"
    assert page.locator("#wzbody .mcard").count() >= 1, "Deploy shows the model cards"
    page.check("#wzbody input[name=advanced]")
    page.click("#wzbody [data-wz-next]"); page.wait_for_timeout(300)             # Advanced: a link, an engine
    assert page.is_visible("#wzbody input[name=link]") and page.is_visible("#wzbody [data-wz-back]"), "no Advanced step"
    page.click("#wzbody [data-wz-close]")                                       # Cancel: nothing applied
    page.click('#eng-list [data-eng="boxed"] [data-a="edit"]'); page.wait_for_timeout(800)   # Edit, its settings and Raw
    assert page.is_visible("#eng-form"), "the edit form did not open"
    page.click("#eng-rawd summary"); page.wait_for_timeout(300)
    page.click("#eng-rawchk"); page.wait_for_timeout(800)
    raw = page.inner_text("#eng-rawout")
    page.screenshot(path="/tmp/engines.png", full_page=True)
    print("raw check:", raw[:120].replace("\n", " | "), flush=True)
    page.click("#pmenub"); page.wait_for_timeout(500)                           # the + menu: every wizard
    menu = page.inner_text("#pmenul")
    assert "Get started" in menu and "Join a friend" in menu and "Share with a friend" in menu, menu
    page.click('#pmenul [data-wz="first-run"]'); page.wait_for_timeout(800)
    assert page.is_visible("#wizard") and "Run AI on this computer" in page.inner_text("#wzbody"), "first-run did not open"
    page.click("#wzbody [data-wz-next]"); page.wait_for_timeout(1200)           # the catalog: one recommended, the others tucked away
    page.screenshot(path="/tmp/wizard.png")
    page.click("#wzbody [data-wz-close]")
    print("wizards: + menu and first-run ok", flush=True)
    tab_click("plan"); page.wait_for_timeout(1200)            # the Plan tab: models, waiting, smart routing
    assert page.locator("#plan-models tr:not(.pmore)").count() >= 2, "the Plan's models"
    page.click('#plan-models button[data-more="small"]'); page.wait_for_timeout(200)
    assert page.is_visible('#plan-models tr.pmore[data-more="small"]'), "a model's More"
    page.click('#plan-smart [data-preset="watch"]'); page.wait_for_timeout(200)  # smart routing's presets
    assert "on" in page.get_attribute('#plan-smart [data-preset="watch"]', "class"), "the Just watch preset"
    page.select_option('#plan-models .schemab[data-schema="small"] select[data-s="tpl"]', "score")   # typed answers, no JSON
    page.wait_for_timeout(200)
    assert page.locator('#plan-models .schemab[data-schema="small"] .frow').count() == 2, "the score template's fields"
    page.locator('#plan-models .schemab[data-schema="small"]').screenshot(path="/tmp/schema.png")
    page.locator("#plan-smart").screenshot(path="/tmp/smart.png")
    page.select_option('#plan-models select[data-pm="small"][data-path="stage"]', "capable")
    page.once("dialog", lambda d: d.accept())
    page.click("#plan-apply"); page.wait_for_timeout(1200)
    got = json.loads(urllib.request.urlopen(urllib.request.Request(base + "api/plan", headers={"X-Nebula-Internal": srv.internal_token})).read())
    assert got["rules"]["stage_router"]["mode"] == "shadow" and got["policy"]["small"].get("stage") == "capable", (got["rules"], got["policy"])
    sc = got["policy"]["small"].get("schema") or {}
    assert sc.get("properties", {}).get("score", {}).get("maximum") == 10 and sc.get("required") == ["score", "reason"], sc
    page.screenshot(path="/tmp/plan.png", full_page=True)
    page.click('#nav [data-tab="home"]'); page.wait_for_timeout(800)
    page.click("#planmap"); page.wait_for_timeout(1500)                           # the plan on the map
    page.screenshot(path="/tmp/map-plan.png")
    page.click("#planmap"); page.wait_for_timeout(300)
    print("plan: tab and map ok", flush=True)
    # Routes › Rules: a rule made on the page, saved by Apply
    tab_click("plan"); page.wait_for_timeout(1200)
    page.click("#rule-add"); page.wait_for_timeout(200)
    page.fill('#plan-rules .rule [data-r=name]', "no secrets"); page.dispatch_event('#plan-rules .rule [data-r=name]', "change")
    page.select_option('#plan-rules .rule [data-c=what]', "words"); page.wait_for_timeout(200)
    page.fill('#plan-rules .rule [data-c=value]', "password, secret"); page.dispatch_event('#plan-rules .rule [data-c=value]', "change")
    page.select_option('#plan-rules .rule [data-t=kind]', "refuse"); page.wait_for_timeout(200)
    page.fill('#plan-rules .rule input[data-t=value]', "not that here"); page.dispatch_event('#plan-rules .rule input[data-t=value]', "change")
    page.locator("#plan-rules").screenshot(path="/tmp/rules.png")
    page.once("dialog", lambda d: d.accept()); page.click("#plan-apply"); page.wait_for_timeout(1200)
    got = json.loads(urllib.request.urlopen(urllib.request.Request(base + "api/plan", headers={"X-Nebula-Internal": srv.internal_token})).read())
    rr = got["rules"]["route_rules"]
    assert rr and rr[0]["name"] == "no secrets" and rr[0]["if"][0] == {"what": "words", "is": True, "value": "password, secret"} \
        and rr[0]["then"] == {"refuse": "not that here"}, rr
    print("rules: made and applied", flush=True)
    # the chat: Halo, not "assistant"; + Files or Photo
    page.click('#nav [data-tab="home"]'); page.wait_for_timeout(800)
    opts = page.eval_on_selector_all("#agent option", "os => os.map(o => o.textContent)")
    assert "assistant" not in opts and "Halo" in " ".join(opts), opts
    import base64 as _b64
    _png = os.path.join(tmp, "photo.png"); open(_png, "wb").write(_b64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="))
    _txt = os.path.join(tmp, "notes.txt"); open(_txt, "w").write("the plan for today")
    page.set_input_files("#cphoto", _png); page.set_input_files("#cfile", _txt); page.wait_for_timeout(400)
    assert page.locator("#cchips .cchip").count() == 2, "two attachments shown"
    page.locator("#form").screenshot(path="/tmp/attach.png")
    n0 = len(eng.requests)
    page.fill("#text", "what is in the photo?"); page.click("#send"); page.wait_for_timeout(2500)
    last = eng.requests[-1]["messages"][-1]
    assert len(eng.requests) > n0 and isinstance(last["content"], list) and [x["type"] for x in last["content"]] == ["text", "image_url"], last
    assert "the plan for today" in last["content"][0]["text"], last["content"][0]
    assert page.locator("#cchips .cchip").count() == 0, "the attachments are cleared once sent"
    print("chat: Halo, a photo and a file sent", flush=True)
    tab_click("callers"); page.wait_for_timeout(600)
    page.click("#cl-new"); page.wait_for_timeout(300)
    assert page.is_visible("#cl-form"), "the caller form did not open"
    page.click('#nav [data-tab="gravity"]'); page.wait_for_timeout(600)
    print("gravity:", page.inner_text("#gv-body")[:80].replace("\n", " "), flush=True)
    assert page.evaluate("document.fonts.check('16px Figtree')"), "the bundled web font did not load"
    page.screenshot(path="/tmp/gravity.png", full_page=True)
    # a brand-new Nebula (no model): the Welcome card's Start opens the first-run wizard
    tmp2 = tempfile.mkdtemp(); path2 = os.path.join(tmp2, "nebula.json")
    json.dump({"port": 0, "ledger": os.path.join(tmp2, "l.sqlite3"), "monitor": False, "tiers": [],
               "lan": {"decided": True, "discoverable": False}}, open(path2, "w"))
    srv2 = make_server(Config.from_args(["--config", path2])); threading.Thread(target=srv2.serve_forever, daemon=True).start()
    page.goto(f"http://127.0.0.1:{srv2.server_port}/"); page.wait_for_timeout(1500)
    assert page.is_visible("#gate"), "a brand-new Nebula asks how to sign in first"
    page.fill("#g-pin", "7351"); page.fill("#g-pin2", "7351"); page.click("#g-set"); page.wait_for_timeout(2000)
    if page.is_visible("#launch"): page.keyboard.press("Escape"); page.wait_for_timeout(300)
    assert page.is_visible("#welcome [data-open-wz]"), "no Welcome card on an empty Nebula"
    page.click("#welcome [data-open-wz]"); page.wait_for_timeout(1200)
    page.screenshot(path="/tmp/welcome.png")
    assert page.is_visible("#wizard"), "the Welcome card's Start did not open the wizard"
    page.click("#wzbody [data-wz-next]"); page.wait_for_timeout(1500)
    page.screenshot(path="/tmp/firstrun.png")
    labels = page.eval_on_selector_all('#wzbody select[name=cached] option', "os => os.map(o => o.textContent)")
    assert any("cached-GGUF" in x and "Q4_K_M" in x for x in labels), ("the Hugging Face cache's model is offered", labels)
    page.select_option("#wzbody select[name=cached]", index=1); page.wait_for_timeout(200)
    page.locator("#wzbody select[name=cached]").screenshot(path="/tmp/hfcache.png")
    assert page.locator("#wzbody .mcard").count() >= 1, "no model cards"
    # Standard or Uncensored: the cards follow the choice
    assert page.is_checked('#wzbody input[name=variant][value=standard]'), "Standard by default"
    assert page.text_content("#wzbody").lower().count("uncensored build") == 0, "Standard: no uncensored build"
    page.check('#wzbody input[name=variant][value=uncensored]'); page.wait_for_timeout(1200)
    assert page.text_content("#wzbody").lower().count("uncensored build") >= 2, "Uncensored: the uncensored builds"
    page.screenshot(path="/tmp/variants.png")
    page.check('#wzbody input[name=variant][value=standard]'); page.wait_for_timeout(800)
    page.click("#wzbody [data-wz-close]"); page.wait_for_timeout(300)
    page.goto(f"http://127.0.0.1:{srv2.server_port}/#join=" + "nebula-invite%3AeyJ2IjoxfQ"); page.wait_for_timeout(2500)   # a friend opens the link
    assert page.is_visible("#wizard") and "Your friend sent you an invite" in page.inner_text("#wzbody"), "the invite link did not open the join wizard"
    assert "nebula-invite:" in page.input_value("#wzbody textarea[name=invite]"), "the invite was not filled in"
    page.screenshot(path="/tmp/join.png")
    print("first run: Start opens the wizard", flush=True)
    srv2.shutdown()
    b.close()
srv.shutdown(); eng.shutdown()
print("\n".join(errors) if errors else "no errors", flush=True)
sys.exit(1 if errors else 0)
