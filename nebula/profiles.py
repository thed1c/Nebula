"""nebula/profiles.py: what each model is good at, and the teams Halo suggests from it (Starweave 2b).

Halo researches a model only with your OK. For a model it has no profile of, it asks first ("Can I read the model card
and README for <model> to assess what it's good at? This contacts Hugging Face."); you choose Read card or Skip, and
the same choice stays on the model as "Research capabilities". Nothing is fetched before that.

**Read.** The model's Hugging Face repository (from where its file came: `models/<owner>/<repo>/...` as a pull puts it,
or the Hub cache's `models--<owner>--<repo>`, or the repository you give) is read through the Hub's API, under the
same token rules as a download (nebula/hf.py): the card's data (its `model-index` benchmark results, license, base
model, tags, languages), config.json (the context it was trained for) and the README. Parsing is code, not a model.
The profile's "good at" lines and cautions each say where they came from (a benchmark, a tag, a quoted line).

**Evidence**, always labelled, and weighed when Halo ranks models for a route:

- reported (from the card): low weight; self-reported, usually at full precision, in prompt formats that differ;
- live (from teamwork on this Nebula): work items done against failed, and how often work is handed up away from the
  model; its weight grows with use, and is full after about 50 items;
- your routing: a model you put on a route (its stage in the routing policy) leads there first.

- measured (a probe on your own hardware, which you start, its GPU time shown first): high weight. Eleven short calls
  at background class through the router: one for speed (tokens a second, time to read the prompt), five asking for
  a JSON answer to a schema, five offering a tool to call. Kept with the machine and the date it ran on.

**Suggest.** Halo proposes a team per route (lead, helpers, reviewer), each with a one-line reason and its sources,
as a card you accept, edit or dismiss. It never applies one on its own. An accepted team goes into the config
(`rules.starweave.teams`, and each member's first "good at" line into `rules.starweave.good_at` unless you wrote one),
and puts that route's helpers and reviewer first on the lead's team sheet.

The profiles live in `nebula-profiles.json` beside the config. "Forget research" deletes one.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
from pathlib import Path

from nebula import hf

ROUTES = ("everyday", "larger", "writing", "critique")
BENCHMARKS = {                      # route -> the card benchmarks that count for it (lowercase, matched in the name)
    "everyday": ("mmlu-pro", "mmlu pro", "ifeval"),
    "larger": ("livecodebench", "swe-bench", "swe bench", "gpqa", "aime"),
    "writing": ("eq-bench", "creative writing"),
    "critique": (),
}
WEIGHT_REPORTED, WEIGHT_LIVE, WEIGHT_MEASURED, BONUS_ROUTE = 0.2, 0.6, 0.5, 0.3
PROBE_JSON, PROBE_TOOLS = 5, 5
PROBE_CALLS = 1 + PROBE_JSON + PROBE_TOOLS
LIVE_FULL = 50                      # live items for live evidence to count in full
TAG_WORDS = {                       # a card tag -> a "good at" line
    "function-calling": "tool calls", "tool-use": "tool calls", "tools": "tool calls", "code": "code",
    "coding": "code", "math": "math", "reasoning": "reasoning", "roleplay": "role-play and fiction",
    "creative-writing": "creative writing", "vision": "pictures", "image-text-to-text": "pictures",
    "multilingual": "many languages",
}
CAUTION_WORDS = {"uncensored": "uncensored finetune", "abliterated": "uncensored finetune (abliterated)",
                 "nsfw": "trained on adult content"}
_LOCK = threading.Lock()


class ProfileError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def path_of(cfg) -> str:
    return os.path.join(getattr(cfg, "base_dir", None) or os.getcwd(), "nebula-profiles.json")


def load(cfg) -> dict:
    """{"profiles": {model: profile}, "skipped": [model], "suggestion": card or None}"""
    try:
        d = json.loads(Path(path_of(cfg)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    return {"profiles": dict(d.get("profiles") or {}), "skipped": list(d.get("skipped") or []),
            "suggestion": d.get("suggestion")}


def save(cfg, d: dict) -> None:
    p = path_of(cfg)
    tmp = p + ".tmp"
    Path(tmp).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


# --- which repository a model came from
REPO_IN_PATH = (re.compile(r"models--([\w.-]+)--([\w.-]+)"),)


def real_name(cfg, tier, models_dir: str | None = None) -> str | None:
    """The model's own name, not the name it was given here (its alias): its Hugging Face repository's name, else its
    file's or folder's name (an engine's model path, or the -m of its command), quantisation and .gguf left off; None
    when nothing tells (a cloud model, a router)."""
    repo = repo_of(cfg, tier, models_dir)
    if repo:
        return repo.rsplit("/", 1)[-1]
    model = None
    if getattr(tier, "managed", None):
        e = next((e for e in getattr(cfg, "engines", None) or [] if e.get("name") == tier.managed), None) or {}
        model = e.get("model")
        cmd = e.get("command") or []
        if not model and "-m" in cmd and cmd.index("-m") + 1 < len(cmd):
            model = cmd[cmd.index("-m") + 1]
    if not model:
        return None
    name = os.path.basename(str(model).rstrip("/"))
    name = re.sub(r"\.gguf$", "", name, flags=re.I)
    name = re.sub(r"-\d{5}-of-\d{5}$", "", name)                                   # a split file's part number
    name = re.sub(r"[.-](i1-)?(IQ\d\w*|Q\d(_[A-Z0-9]+)*|BF16|F16|F32)$", "", name, flags=re.I)   # its quantisation
    return name or None


def repo_of(cfg, tier, models_dir: str | None = None) -> str | None:
    """The Hugging Face repository a model's file came from: the name you gave it (hf_repo), else what can be told
    from where its file is."""
    if getattr(tier, "hf_repo", None):
        return tier.hf_repo
    model = None
    if getattr(tier, "managed", None):
        e = next((e for e in getattr(cfg, "engines", None) or [] if e.get("name") == tier.managed), None)
        model = (e or {}).get("model")
    if not model:
        return None
    for rx in REPO_IN_PATH:
        m = rx.search(model)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
    if models_dir:
        try:
            rel = os.path.relpath(os.path.abspath(model), os.path.abspath(models_dir))
        except ValueError:
            rel = ""
        parts = Path(rel).parts
        if len(parts) >= 3 and parts[0] not in ("..", "."):
            return f"{parts[0]}/{parts[1]}"
    return None


# --- reading a card
def _get(url: str, token: str | None, raw: bool = False):
    import urllib.request
    req = urllib.request.Request(url, headers=hf._headers(token))
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read(1 << 20)
    except Exception:  # noqa: BLE001 - a missing README or config is no profile error
        return None
    if raw:
        return data.decode("utf-8", "replace")
    try:
        return json.loads(data)
    except ValueError:
        return None


def read_card(repo: str, token: str | None = None) -> dict:
    """The Hub's facts about a repository: card data, config, README (the parts a profile needs)."""
    if not re.match(r"^[\w.-]+/[\w.-]+$", repo or ""):
        raise ProfileError(400, "a repository is owner/name")
    base = hf.endpoint()
    info = hf._get_json(f"{base}/api/models/{repo}", token)        # raises HfError: gated, missing, unreachable
    rev = urllib.parse.quote(info.get("sha") or "main", safe="")
    config = _get(f"{base}/{repo}/resolve/{rev}/config.json", token) or {}
    readme = _get(f"{base}/{repo}/resolve/{rev}/README.md", token, raw=True) or ""
    return {"repo": repo, "revision": info.get("sha"), "card": info.get("cardData") or {}, "tags": info.get("tags") or [],
            "pipeline_tag": info.get("pipeline_tag"), "config": config if isinstance(config, dict) else {}, "readme": readme}


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x * 100 if 0 <= x <= 1 else x


def benchmarks(card: dict) -> list[dict]:
    """The card's model-index results: [{"name", "metric", "value"}], value in percent where it was a fraction."""
    out = []
    for entry in card.get("model-index") or []:
        for r in (entry or {}).get("results") or []:
            ds = (r.get("dataset") or {})
            name = str(ds.get("name") or ds.get("type") or "").strip()
            for m in r.get("metrics") or []:
                v = _num(m.get("value"))
                if name and v is not None:
                    out.append({"name": name, "metric": str(m.get("type") or m.get("name") or ""), "value": round(v, 1)})
    return out


def _quote(readme: str, word: str) -> str | None:
    """The README's first line that mentions `word`, for a source."""
    for line in readme.splitlines():
        s = line.strip().strip("#*-> ").strip()
        if word in s.lower() and 12 <= len(s) <= 300:
            return s
    return None


def build(facts: dict) -> dict:
    """A profile from the Hub's facts. Code, not a model: each line carries its source."""
    card, tags, readme = facts.get("card") or {}, [str(t).lower() for t in facts.get("tags") or []], facts.get("readme") or ""
    lower = readme.lower()
    good, cautions = [], []
    bench = benchmarks(card)
    for route, names in BENCHMARKS.items():
        hits = [b for b in bench if any(n in b["name"].lower() for n in names)]
        if hits:
            best = max(hits, key=lambda b: b["value"])
            good.append({"text": f"{route} work: reports {best['name']} {best['value']:g}", "source": "card: model-index"})
    seen = set()
    for t in tags + ([str(facts.get("pipeline_tag")).lower()] if facts.get("pipeline_tag") else []):
        w = TAG_WORDS.get(t)
        if w and w not in seen:
            seen.add(w)
            good.append({"text": w, "source": f"card: tag {t}"})
    for word, line in (("tool", "tool calls"), ("function call", "tool calls"), ("code", "code")):
        if line not in seen and word in lower:
            q = _quote(readme, word)
            if q:
                seen.add(line)
                good.append({"text": line, "source": f"README: “{q[:160]}”"})
    name = (facts.get("repo") or "").lower()
    for word, text in CAUTION_WORDS.items():
        if word in name or word in tags or word in lower[:4000]:
            if text not in [c["text"] for c in cautions]:
                cautions.append({"text": text, "source": "repository name" if word in name else "card"})
    if "tool calls" not in seen and not any(t in tags for t in ("function-calling", "tool-use", "tools")):
        cautions.append({"text": "not said to be trained for tool calls (team tools may need the JSON block)", "source": "card: no tool tag"})
    cfg = facts.get("config") or {}
    ctx = cfg.get("max_position_embeddings") or (cfg.get("text_config") or {}).get("max_position_embeddings")
    langs = card.get("language")
    return {"repo": facts.get("repo"), "revision": facts.get("revision"), "read_at": time.time(),
            "license": card.get("license"), "base_model": card.get("base_model"),
            "languages": langs if isinstance(langs, list) else ([langs] if langs else []),
            "trained_context": ctx if isinstance(ctx, int) else None,
            "good_at": good[:6], "cautions": cautions[:4], "reported": bench[:40]}


# --- research, with the owner's OK
def research(cfg, tier, repo: str | None, token: str | None = None, models_dir: str | None = None) -> dict:
    repo = (repo or "").strip() or repo_of(cfg, tier, models_dir)
    if not repo:
        raise ProfileError(400, f"Nebula cannot tell which Hugging Face repository {tier.name} came from: give it (owner/name)")
    try:
        facts = read_card(repo, token)
    except hf.HfError as e:
        raise ProfileError(e.status, str(e)) from None
    prof = build(facts)
    with _LOCK:
        d = load(cfg)
        d["profiles"][tier.name] = prof
        d["skipped"] = [m for m in d["skipped"] if m != tier.name]
        save(cfg, d)
    return prof


def skip(cfg, model: str) -> None:
    with _LOCK:
        d = load(cfg)
        if model not in d["skipped"]:
            d["skipped"].append(model)
        save(cfg, d)


def forget(cfg, model: str) -> None:
    with _LOCK:
        d = load(cfg)
        d["profiles"].pop(model, None)
        save(cfg, d)


def asks(cfg) -> list[dict]:
    """Halo's asks: a model it has no profile of, not skipped, not a command or a friend's."""
    d = load(cfg)
    out = []
    for t in cfg.tiers:
        if t.name in d["profiles"] or t.name in d["skipped"] or t.peer or t.is_command:
            continue
        out.append({"model": t.name, "message": f"Can I read the model card and README for {t.name} to assess what "
                                                "it's good at? This contacts Hugging Face."})
    return out


# --- live evidence and ranking
def live(ledger) -> dict:
    """Per model: {"done", "failed", "handed_up_from", "answers"} from the ledger's teamwork."""
    out: dict = {}
    if ledger is None:
        return out
    try:
        for r in ledger.db.execute("SELECT model, state, COUNT(*) AS n FROM work_items GROUP BY model, state"):
            m = out.setdefault(r["model"], {"done": 0, "failed": 0, "handed_up_from": 0})
            if r["state"] == "done":
                m["done"] += r["n"]
            elif r["state"] == "failed":
                m["failed"] += r["n"]
        for r in ledger.db.execute("SELECT json_extract(payload, '$.from') AS m, COUNT(*) AS n FROM events WHERE "
                                   "kind='team_action' AND json_extract(payload, '$.action')='hand_up' AND "
                                   "json_extract(payload, '$.ruling')='allowed' GROUP BY m"):
            if r["m"]:
                out.setdefault(r["m"], {"done": 0, "failed": 0, "handed_up_from": 0})["handed_up_from"] += r["n"]
    except Exception:  # noqa: BLE001 - a ledger before Starweave's table
        return {}
    return out


def score(model: str, route: str, prof: dict | None, lv: dict | None, stage: str | None) -> tuple[float, list[str]]:
    """(the score of `model` for `route`, the reasons with their sources)."""
    s, why = 0.0, []
    if stage == route:
        s += BONUS_ROUTE
        why.append(f"you route {route} to it")
    m = (prof or {}).get("measured")
    if m:
        speed = min((m.get("tok_s") or 0) / (40 if route == "writing" else 60), 1.0)
        js, tl = m.get("json_ok") or 0.0, m.get("tools_ok") or 0.0
        part = {"everyday": 0.5 * speed + 0.5 * js, "larger": 0.6 * tl + 0.4 * js, "writing": speed, "critique": js}.get(route, 0.0)
        s += WEIGHT_MEASURED * part
        why.append(f"measured on {m.get('machine') or 'this Nebula'} {time.strftime('%Y-%m-%d', time.localtime(m.get('at') or 0))}: "
                   f"{m.get('tok_s') or 0:g} tokens/s, JSON {round(js * PROBE_JSON)}/{PROBE_JSON}, tools {round(tl * PROBE_TOOLS)}/{PROBE_TOOLS}")
    names = BENCHMARKS.get(route) or ()
    hits = [b for b in (prof or {}).get("reported") or [] if any(n in b["name"].lower() for n in names)]
    if hits:
        avg = sum(min(b["value"], 100) for b in hits) / len(hits) / 100
        s += WEIGHT_REPORTED * avg
        why.append("card reports " + ", ".join(f"{b['name']} {b['value']:g}" for b in hits[:3]) + " (reported)")
    if lv:
        n = lv["done"] + lv["failed"]
        if n:
            ok = lv["done"] / n
            away = lv["handed_up_from"] / max(n, 1)
            w = WEIGHT_LIVE * min(n / LIVE_FULL, 1.0)
            s += w * max(0.0, ok - 0.5 * away)
            why.append(f"{lv['done']} of {n} team items done here (live)")
    return round(s, 4), why


def suggest(cfg, policy, ledger) -> dict:
    """Halo's suggested team per route: {"made_at", "routes": {route: {"lead", "helpers", "reviewer"}}}, each member
    {"model", "why"}. Only routes some model is on; never applied here."""
    d = load(cfg)
    lv = live(ledger)
    tiers = [t for t in cfg.tiers if not t.peer and not t.is_command]
    stages = {t.name: (policy.stage(t.name) if policy is not None else "everyday") or "everyday" for t in tiers}
    routes = {}
    for route in ROUTES:
        if route not in stages.values():
            continue
        ranked = sorted(tiers, key=lambda t: -score(t.name, route, d["profiles"].get(t.name), lv.get(t.name), stages[t.name])[0])
        def member(t):
            sc, why = score(t.name, route, d["profiles"].get(t.name), lv.get(t.name), stages[t.name])
            prof = d["profiles"].get(t.name) or {}
            good = [g["text"] for g in prof.get("good_at") or []][:2]
            return {"model": t.name, "why": "; ".join(why + ([f"good at {', '.join(good)} (card)"] if good else []))
                    or "on this Nebula, nothing known yet", "score": sc}
        lead = ranked[0]
        critics = [t for t in tiers if stages[t.name] == "critique" and t.name != lead.name]
        reviewer = critics[0] if critics else None
        helpers = [t for t in ranked[1:] if reviewer is None or t.name != reviewer.name][:2]
        routes[route] = {"lead": member(lead), "helpers": [member(t) for t in helpers],
                         "reviewer": member(reviewer) if reviewer is not None else None}
    card = {"made_at": time.time(), "routes": routes}
    with _LOCK:
        d = load(cfg)
        d["suggestion"] = card
        save(cfg, d)
    return card


def accept(cfg, teams: dict | None = None) -> dict:
    """The suggestion (or the owner's edit of it) as the config's teams: {route: {"lead", "helpers", "reviewer"}} of
    model names, plus each member's first "good at" line where the owner wrote none. Returns the starweave section."""
    d = load(cfg)
    if teams is None:
        card = d.get("suggestion") or {}
        teams = {r: {"lead": (v.get("lead") or {}).get("model"), "helpers": [h["model"] for h in v.get("helpers") or []],
                     "reviewer": (v.get("reviewer") or {}).get("model")} for r, v in (card.get("routes") or {}).items()}
    names = {t.name for t in cfg.tiers}
    clean = {}
    for r, v in (teams or {}).items():
        if r not in ROUTES or not isinstance(v, dict):
            raise ProfileError(400, f"a team is for a route: {', '.join(ROUTES)}")
        lead, helpers, rev = v.get("lead"), [h for h in v.get("helpers") or [] if h], v.get("reviewer")
        for m in [lead, rev, *helpers]:
            if m and m not in names:
                raise ProfileError(400, f"no model {m!r}")
        clean[r] = {"lead": lead, "helpers": helpers[:4], "reviewer": rev}
    sw = dict((cfg.rules or {}).get("starweave") or {})
    from nebula import contracts
    old = sw.get("teams") or {}
    # named teams (contracts) keep everything else: the route's first team takes the members Halo suggested
    sw["teams"] = clean if not old or contracts.all_legacy(old) else contracts.merge_route_teams(old, clean)
    good = dict(sw.get("good_at") or {})
    for m, prof in d["profiles"].items():
        if m not in good and prof.get("good_at"):
            good[m] = prof["good_at"][0]["text"][:200]
    if good:
        sw["good_at"] = good
    with _LOCK:
        d["suggestion"] = None
        save(cfg, d)
    return sw


def dismiss(cfg) -> None:
    with _LOCK:
        d = load(cfg)
        d["suggestion"] = None
        save(cfg, d)


# --- the probe: measured evidence, on the owner's word
def probe_estimate(admission, tier) -> float:
    """Seconds of GPU time the probe takes on this model, from the gate's average call."""
    avg = admission.average(tier) if admission is not None and tier.address else 5.0
    return round(PROBE_CALLS * avg, 1)


def start_probe(server, tier) -> None:
    with _LOCK:
        d = load(server.cfg)
        prof = d["profiles"].setdefault(tier.name, {"repo": None, "good_at": [], "cautions": [], "reported": []})
        if (prof.get("probe") or {}).get("state") == "running":
            raise ProfileError(409, f"{tier.name} is being measured already")
        prof["probe"] = {"state": "running", "since": time.time()}
        save(server.cfg, d)
    threading.Thread(target=_probe, args=(server, tier), name=f"probe-{tier.name}", daemon=True).start()


PROBE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer", "confidence"],
                "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}}}
PROBE_TOOL = {"type": "function", "function": {"name": "get_time", "description": "The local time in a city.", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
PROBE_QUESTIONS = ("What is the capital of Australia?", "How many legs does a spider have?", "What gas do plants take in?",
                   "Which planet is largest?", "What is 17 times 3?")
PROBE_CITIES = ("Paris", "Tokyo", "Lima", "Oslo", "Cairo")


def _probe(server, tier) -> None:
    from nebula.dialects import Context
    from nebula.events import assemble_reply
    from nebula.turn import Turn
    t0 = time.time()

    def call(req):
        ctx = Context(server.cfg, server.ledger, {}, server.engines, server.tools, input="probe", priority="background",
                      registry=getattr(server, "registry", None), admission=getattr(server, "admission", None),
                      lot=getattr(server, "lot", None), policy=getattr(server, "policy", None), yard=getattr(server, "yard", None))
        req = {"model": tier.name, **req}
        turn = Turn(ctx, "chat", req)
        body = {k: v for k, v in req.items() if k != "model"}
        engine, body = turn.prepare(body, messages=req["messages"], tools=req.get("tools"), requested_model=tier.name,
                                    thinking_requested=None)
        events = list(turn.run(engine, body))
        return assemble_reply(events) or {}, turn.done

    try:
        _reply, done = call({"messages": [{"role": "user", "content": "In about 150 words, explain why the sky is blue."}],
                             "max_tokens": 300})
        tm = done.timings or {}
        tok_s = round(float(tm.get("predicted_per_second") or 0), 1)
        prompt_ms = round(float(tm.get("prompt_ms") or 0), 1)
        js = 0
        for q in PROBE_QUESTIONS[:PROBE_JSON]:
            r, _d = call({"messages": [{"role": "user", "content": q + " Answer as JSON."}], "max_tokens": 200,
                          "response_format": {"type": "json_schema", "json_schema": {"name": "probe", "strict": True, "schema": PROBE_SCHEMA}}})
            try:
                v = json.loads(str(r.get("content") or ""))
                js += isinstance(v, dict) and isinstance(v.get("answer"), str) and isinstance(v.get("confidence"), (int, float))
            except ValueError:
                pass
        tl = 0
        for c in PROBE_CITIES[:PROBE_TOOLS]:
            r, _d = call({"messages": [{"role": "user", "content": f"What time is it in {c}? Use the tool."}], "max_tokens": 200,
                          "tools": [PROBE_TOOL]})
            for tc in r.get("tool_calls") or []:
                f = tc.get("function") or {}
                try:
                    args = json.loads(f.get("arguments") or "{}") if isinstance(f.get("arguments"), str) else (f.get("arguments") or {})
                except ValueError:
                    args = {}
                if f.get("name") == "get_time" and str(args.get("city") or "").lower() == c.lower():
                    tl += 1
                    break
        measured = {"at": time.time(), "machine": tier.host or "this Nebula", "tok_s": tok_s, "prompt_ms": prompt_ms,
                    "json_ok": round(js / PROBE_JSON, 2), "tools_ok": round(tl / PROBE_TOOLS, 2), "calls": PROBE_CALLS,
                    "seconds": round(time.time() - t0, 1)}
        probe = {"state": "done", "at": time.time()}
    except Exception as e:  # noqa: BLE001 - the probe's failure is the owner's to read
        measured, probe = None, {"state": "failed", "error": str(getattr(e, "message", None) or e)[:300], "at": time.time()}
    with _LOCK:
        d = load(server.cfg)
        prof = d["profiles"].setdefault(tier.name, {"repo": None, "good_at": [], "cautions": [], "reported": []})
        if measured is not None:
            prof["measured"] = measured
        prof["probe"] = probe
        save(server.cfg, d)


def owner_asks(ledger) -> list[dict]:
    """The team's questions waiting on the owner, newest first."""
    if ledger is None:
        return []
    try:
        rows = ledger.db.execute("SELECT item_id FROM work_items WHERE mode='ask_owner' AND state='awaiting_owner' "
                                 "ORDER BY created_at DESC LIMIT 50").fetchall()
    except Exception:  # noqa: BLE001 - a ledger before Starweave's table
        return []
    out = []
    for r in rows:
        i = ledger.work_item(r["item_id"])
        out.append({"item_id": i["item_id"], "session_id": i["session_id"], "from": i["from_model"], "question": i["task_text"],
                    "options": (i.get("verdict") or {}).get("options") or [], "at": i["created_at"]})
    return out


def route_view(cfg) -> dict:
    """The teams as the page's per-route cards read them: {route: {lead, helpers, reviewer}} of each route's first team."""
    from nebula import contracts
    out = {}
    for k, c in contracts.teams(cfg.rules or {}).items():
        for r in c["routes"]:
            out.setdefault(r, {"lead": c["lead"], "helpers": list(c["helpers"]), "reviewer": c["reviewer"]})
    return out


def contracts_view(cfg) -> dict:
    """Every team's contract for the Star Weaving page: {id: {"contract" (every part), "version", "plain" (its
    sentences)}}, in the config's order."""
    from nebula import contracts, team as T
    from nebula.router import Rules
    base = T.settings(Rules.from_dict(cfg.rules or {}))
    return {k: {"contract": c, "version": contracts.version(c), "plain": contracts.plain(c, base)}
            for k, c in contracts.teams(cfg.rules or {}).items()}


def view(cfg, policy=None, ledger=None, admission=None) -> dict:
    d = load(cfg)
    sw = (cfg.rules or {}).get("starweave") or {}
    from nebula import contracts
    return {"profiles": d["profiles"], "skipped": d["skipped"], "asks": asks(cfg), "suggestion": d.get("suggestion"),
            "teams": route_view(cfg), "contracts": contracts_view(cfg), "templates": sorted(contracts.TEMPLATES),
            "good_at": sw.get("good_at") or {}, "tags": sw.get("tags") or {}, "reviewer": sw.get("reviewer"),
            "live": live(ledger), "owner_asks": owner_asks(ledger),
            "probe_estimate": {t.name: probe_estimate(admission, t) for t in cfg.tiers if not t.peer and not t.is_command}}
