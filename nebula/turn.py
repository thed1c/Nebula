"""nebula/turn.py: one request's turn, from the dialect's body to the engine that answers it.

    turn = Turn(ctx, "chat", req)
    engine, body = turn.prepare(body, messages=..., tools=..., requested_model=..., thinking_requested=...)
    for ev in turn.run(engine, body): ...   # the reply's events; the attempt is recorded and closed by run
    turn.done                                # the counts, once the events ended

`prepare` picks the agent, records the request in the ledger (the rows before anything is sent), reads the
session's state for that agent, builds the ladder from the ticks (the input's, else the agent's, else the default
list; the whole tier list without any) less the engines the registry says are down, asks the router for the tier, and, on a crossing, builds
the packet, escalates in the ledger and renders the destination's prompt from the packet. On a tier the session
already crossed to, the prompt is rendered the same way. Then the agent's system prompt, tools and output shape go
into the body, and the tier's thinking setting. Without a ledger the router sees only the request's own signals.

`run` takes a slot at the engine's gate first (served by class then arrival; the wait is recorded on the attempt),
then streams the engine and, when the engine calls tools the agent owns (the MCP hub's), runs them, records the
round, and asks the engine again, up to the agent's rounds; a round that also calls the client's tools ends the
turn with those. It closes the attempt, acknowledges the handoff the reply answered, and runs a compaction when the
reply's prompt passed the configured share of the engine's context.
"""
from __future__ import annotations

import itertools
import json

import logging
import time

from nebula import names, stars
from nebula.engine import CALL_INPUT
from nebula.admission import RANK, QueueTimeout, priority_of
from nebula.agents import apply_agent, select_agent, system_text
from nebula import route_settings, team, toolbox, skill_tools
from nebula.parking import ParkedTooBig
from nebula.sizing import estimate_tokens, holds, needed
from nebula.engine import EngineError
from nebula.events import Done, Stream, assemble_reply
from nebula.handoff import (build_packet, compact, compaction_due, full_read_due, idempotency_key, packet_sha256, render_messages,
                             transcript_messages)
from nebula.ledger import SessionState, text_of
from nebula.router import Rules, Signals, decide, same_engine, tier_named

LOG = logging.getLogger("cockpit.turn")


def session_key_of(req: dict, headers, dialect: str) -> tuple[str | None, str]:
    """(the request's session key or None, where it came from)."""
    run = names.header(headers, "Run")                 # a run (the Switchyard's sticky unit): all its calls are one session
    if run:
        return "run:" + str(run), "run"
    header = names.header(headers, "Session")
    if header:
        return str(header), "header"
    meta = req.get("metadata") if isinstance(req.get("metadata"), dict) else {}
    if dialect == "messages" and meta.get("user_id"):
        return str(meta["user_id"]), "metadata.user_id"
    if meta.get("session_id"):
        return str(meta["session_id"]), "metadata.session_id"
    if isinstance(req.get("user"), str) and req["user"]:
        return req["user"], "user"
    return None, "opening"


def apply_thinking(body: dict, thinking: bool | None) -> None:
    if thinking is None:
        return
    kw = dict(body.get("chat_template_kwargs") or {})
    kw["enable_thinking"] = bool(thinking)
    if thinking and kw.get("reasoning_effort") in ("none", "off"):
        kw.pop("reasoning_effort")
    body["chat_template_kwargs"] = kw


NEED_WORDS = {"vision": "pictures", "tools": "tools", "structured": "structured answers", "long_context": "long context"}


EXPLICIT_ONLY = "used only when a request names it"   # the policy's explicit-only models: off the ladder, never a fallback


class Turn:
    def __init__(self, ctx, dialect: str, req: dict):
        self.ctx = ctx                                      # kept for Starweave's work items (child turns, nebula/team.py)
        self.team_ctx = getattr(ctx, "team", None)          # set when this turn IS a work item
        self.cfg, self.ledger = ctx.cfg, getattr(ctx, "ledger", None)
        self.engines = getattr(ctx, "engines", None) or {}
        self.tools = getattr(ctx, "tools", None)
        self.headers = getattr(ctx, "headers", None) or {}
        self.dialect, self.req = getattr(ctx, "input", None) or dialect, req
        self.rules = Rules.from_dict(getattr(self.cfg, "rules", None))
        self.agent = None
        self.team_tools: list[str] = []                 # Starweave's team tools offered this turn (team.offer)
        self.skill_tools: list[str] = []                # skill tools offered this turn
        self.skill_catalog = None
        self.attached_skills: list[str] = []
        self.team_pool: list = []                       # the models this turn's team may give work to
        self.team_task: str | None = None               # the task its work items belong to (team.task_of)
        self.opened = None
        self.attempt_id: str | None = None
        self.handoff_id: str | None = None
        self.acknowledged = False
        self.engine = None
        self.tier = None
        self.decision = None
        self.registry = getattr(ctx, "registry", None)
        self.policy = getattr(ctx, "policy", None)          # this Nebula's routing policy (nebula/policy.py)
        self.yard = getattr(ctx, "yard", None)              # the engines this Nebula runs (nebula/engines.py)
        self.seclog = getattr(ctx, "seclog", None)          # the security log (nebula/seclog.py)
        self.reply_headers = getattr(ctx, "reply_headers", None)
        self.guard = getattr(ctx, "guard", None)            # holds on guests (nebula/guard.py)
        self.otel = getattr(ctx, "otel", None)              # a span per turn when OpenTelemetry is on (nebula/otel.py)
        self.t0, self.exported = time.time(), False
        self.caller = getattr(ctx, "caller", None)          # {"id", "label"}: who asks (nebula/callers.py)
        self.passthrough = getattr(ctx, "mode", None) == "passthrough"
        self.power = getattr(ctx, "power", None)            # sleeping machines (nebula/power.py)
        self.woke, self.asleep_why = None, None
        self.retries, self.tried = 0, set()
        from nebula import tenancy
        self.tenant, self.guest = tenancy.tenant_of(self.caller if (self.caller or {}).get("id") else None,
                                                    getattr(ctx, "peer", None))
        self.slots = getattr(ctx, "slots", None)            # the slot each tenant has on a shared engine
        self.isolation = None
        self.route = None                                   # the route it is on (nebula/route_settings.py)
        self.pinned = False                                 # the request named its model: never moved in silence
        self.peer = getattr(ctx, "peer", None)              # the paired Nebula asking, when one is
        self.blocked, self.not_offered = {}, set()
        self.admission = getattr(ctx, "admission", None)
        self.lot = getattr(ctx, "lot", None)
        self.parked: dict | None = None
        self.size: dict | None = None
        self.max_tokens = None
        self.input_priority = getattr(ctx, "priority", None)
        self.priority = "interactive"
        self.queued_s = 0.0
        self.input_ticks = getattr(ctx, "ticks", None) or None
        self.ladder = list(self.cfg.tiers)
        self.events = []
        self.done = Done()
        self.client_calls = 0                # tool calls handed to the client this turn
        self.rounds = 0                      # tool rounds the cockpit ran this turn
        self.finished = False
        self.star = stars.star_of(self.dialect)       # the API Server or the Assistant, when it came in by one
        self.model_override = None                    # a star's dedicated model: the name its tier is asked for
        CALL_INPUT.set(stars.call_input(self.dialect))
        from nebula.engine import CALL_TENANT
        CALL_TENANT.set(self.tenant)

    # --- before the engine is called
    def prepare(self, body: dict, *, messages: list[dict], tools, requested_model: str,
                thinking_requested: bool | None):
        tiers = self.cfg.tiers
        self._client_tools = list(body.get("tools") or [])     # the request's own tools, before any are added
        self.agent = select_agent(self.cfg, self.req, self.headers)
        agent = self.agent
        if agent is not None and agent.thinking is not None:
            thinking_requested = agent.thinking
        # the ticks: the input's own, else the agent's, else the default list; the ladder the router climbs is the
        # ticked tiers in that order, and a model named outside them is not used
        ticks, ladder, usable_names = self._resolve_ladder(agent)
        if self.power is not None and ladder and not usable_names:
            # every model it may use is asleep: an interactive call wakes one machine and waits for it; others never do
            interactive = priority_of(self.dialect, self.input_priority, agent.priority if agent else None) == "interactive"
            self.woke, self.asleep_why = self.power.wake_for(ladder, interactive)
            if self.woke:
                ticks, ladder, usable_names = self._resolve_ladder(agent)
        outside = [f"woke:{self.woke}"] if self.woke else []
        named = tier_named(self.req.get("model"), tiers)
        st_tier, st_model = stars.route_of(self.cfg, self.star)
        if st_tier:                                     # the star's dedicated model wins over the name a client sent
            named, self.model_override = st_tier, (st_tier, st_model) if st_model else None
        refusal = None                                  # the policy's word, given once the request is in the ledger
        if named is not None and named in self.not_offered:
            refusal = (403, f"{named} is not offered to you", "policy:not_offered")
        elif named is not None and named in self.blocked:
            refusal = (503, f"{named} is not to be used now: {self.blocked[named]}", "policy:" + self.blocked[named])
        elif not ladder:
            why = "; ".join(f"{n}: {w}" for n, w in self.blocked.items()) or "nothing is offered"
            refusal = (503, f"no model may be used now ({why})", "policy:none_allowed")
        if named is not None and named not in [t.name for t in ladder]:
            outside.append(f"outside_ticks:{named}")
            named = None
        escalate = names.header(self.headers, "Escalate") or None
        if escalate and escalate != "1" and escalate in [t.name for t in tiers] and escalate not in [t.name for t in ladder]:
            outside.append(f"outside_ticks:{escalate}")
            escalate = "1"
        # a model the request named itself is pinned: it is tried even when the registry saw it
        # down, and fails with the engine's own answer rather than being routed elsewhere in silence
        if self.passthrough:                            # the named model as it is (else the caller's first tick)
            pick = named or (ladder[0].name if ladder else None)
            ladder = [t for t in ladder if t.name == pick] or ladder[:1]
            named = pick
        self.pinned = named is not None
        self.team_pool = [t for t in ladder if t.name in usable_names and not t.peer]   # Starweave's teammates
        # a model used only when a request names it (the owner's Codex, Claude Code) is off the ladder, yet still a
        # teammate where the owner named it on the team: kept beside the pool; team.teammates admits it only when named
        self.team_pool += [t for t in (self.cfg.tier(n) for n, why in self.blocked.items() if why == EXPLICIT_ONLY)
                           if t is not None and not t.peer and (not ticks or t.name in ticks)]
        self.ladder = ladder if named is not None else ([t for t in ladder if t.name in usable_names] or ladder)
        signals = Signals(named_tier=named, escalate=escalate, thinking_requested=thinking_requested)
        fresh = not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages)   # no answer before it
        if self.ledger is None:
            state = SessionState("")
        else:
            key, source = session_key_of(self.req, self.headers, self.dialect)
            if self.team_ctx:                           # a work item: its own child session, out of the lists
                key, source = f"team:{self.team_ctx['item_id']}", "team"
            elif self.dialect == "halo":                # Halo's own looks: out of the chat lists unless one alerts (nebula/halo.py)
                import uuid
                key, source = f"halo:{names.header(self.headers, 'Halo-Look') or uuid.uuid4().hex}", "halo"
            if self.tenant != "owner":                  # sessions are per tenant: the same opening never joins another's
                key = f"{self.tenant}|{key or ''}"
            # where the attempt opens: the session's floor when it has one (open_request keeps that); on a fresh
            # conversation's first turn the model the request named (a team item's teammate), else the agent's own
            # floor, else the first model it may use. Never simply the first tier in the config, which may not be used
            on_ladder = {t.name: t for t in self.ladder}
            first = ((on_ladder.get(named) if named is not None and fresh else None)
                     or (on_ladder.get(agent.tier) if agent is not None and agent.tier and not ticks else None)
                     or (self.ladder[0] if self.ladder else tiers[0]))
            self.opened = self.ledger.open_request(dialect=self.dialect, session_key=key, key_source=source,
                                                   messages=messages, tools=tools, requested_model=requested_model,
                                                   thinking=thinking_requested, tier=first.name,
                                                   engine=first.address, agent=agent.name if agent else None)
            self.attempt_id = self.opened.attempt_id
            state = self.ledger.state(self.opened.session_id, self.attempt_id, self.rules.stuck_phrases,
                                      agent=agent.name if agent else None, ignore_outages=self.rules.escalation_ignore_outages)
            self._unstick(state, ladder)
        if refusal is not None:
            self._refuse(*refusal, ladder or list(tiers), thinking_requested, agent)
        if state.current_tier is None and named is not None and fresh:
            state.current_tier = named          # a fresh first turn that names its model starts there: nothing to hand off
        elif state.current_tier is None and agent is not None and agent.tier and not ticks:
            state.current_tier = agent.tier     # the agent's own floor, before it has answered in this session
        # the size, and which ticked models hold it; none = the parking lot (step C)
        sizing = list(messages) + ([{"role": "system", "content": system_text(agent, self.cfg.base_dir)}] if agent else [])
        if route_settings.longest(self.rules):          # its route is not known yet: the longest route instructions count
            sizing.append({"role": "system", "content": route_settings.longest(self.rules)})
        est, how = estimate_tokens(self._engine(self.ladder[0]) if self.ladder else None, sizing, tools, self.rules.estimate_margin)
        self.max_tokens = self.req.get("max_tokens") or self.req.get("max_output_tokens")
        if self.rules.clamp_output:                     # the answer's budget is cut to the slot later: fit the prompt first
            self.max_tokens = None
        self.size = {"estimate": est, "how": how, "need": needed(est, self.max_tokens, self.rules.output_reserve_tokens)}
        fits = self._fitting(self.ladder, est)
        if not fits and self.ladder:
            # nothing ticked holds it: the lot, then the ladder as it stands on release
            self._park(agent, named, est, thinking_requested)
            ticks, ladder, usable_names = self._resolve_ladder(agent)
            self.ladder = self._fitting([t for t in ladder if t.name in usable_names] or ladder, est)
            named = None                          # released onto whatever holds it now
            signals = Signals(named_tier=None, escalate=escalate, thinking_requested=thinking_requested)
        if ticks and not usable_names and self.parked is None:
            # none of the ticked models answers: the request waits for one (step B queues it; here it is refused).
            # Without ticks the whole ladder stands and the engine's own refusal (a 502) says what is wrong.
            msg = f"waiting for a ticked model: none of {[t.name for t in ladder]} answers" + (f" ({self.asleep_why})" if self.asleep_why else "")
            if self.ledger is not None:
                self.ledger.decide(self.attempt_id, tier=ladder[0].name, thinking=thinking_requested,
                                   reason="waiting:no_usable_ticked_model",
                                   inputs={"ticks": ticks, "usable": [], "agent": agent.name if agent else None})
                self.ledger.close_request(self.attempt_id, reply=None, done=None, error=msg)
            raise EngineError(503, {"error": {"message": msg, "type": "server_error"}})
        from nebula import route_rules, stage
        # Custom routing's own rules first (nebula/route_rules.py): the first that holds sends it, or refuses it
        self.routed = route_rules.apply(self, est, state, named)
        if self.routed is not None:
            th = self.routed["then"]
            if "refuse" in th:
                self._refuse(403, th["refuse"], f"rule:{self.routed['rule']}", self.ladder, thinking_requested, agent)
            if self.routed.get("missing"):
                self._refuse(503, f"the rule “{self.routed['rule']}” sends this to {self.routed['missing']} only, which is not "
                                  "offered to it now", f"rule:{self.routed['rule']}", self.ladder, thinking_requested, agent)
        # the stage router: the everyday tier or the larger one (not when a rule has placed it)
        self.stage = None if (self.routed and self.routed["moved"]) else stage.pick(self, est, state)
        floor_state = state
        if self.stage and self.stage.get("mode") == "live" and self.stage.get("how") in ("halo", "jev") and self.ladder:
            import copy                                   # Halo's route is this turn's floor (the session may move between routes)
            floor_state = copy.copy(state)
            floor_state.current_tier = self.ladder[0].name
        down = self._team_down(state)                   # Starweave: the previous answer handed the next turns down
        if down:
            import copy
            floor_state = copy.copy(floor_state)
            floor_state.current_tier = down
        decision = decide(self.ladder, self.rules, floor_state, signals, ceiling=agent.ceiling if (agent and not ticks) else None)
        if down:
            decision.reasons.append(f"team:hand_down:{down}")
        if self.routed is not None:
            decision.reasons.append(f"rule:{self.routed['rule']}")
        if self.stage and self.stage.get("how") in ("halo", "jev"):
            who = "Halo" if self.stage["how"] == "halo" else "Jev"
            decision.reasons.append(("" if self.stage["mode"] == "live" else "practice:") + f"route:{self.stage['picked']}"
                                    + (f" ({who}: {self.stage['why']})" if self.stage.get("why") else f" ({who})"))
        elif self.stage and self.stage.get("score") is not None:
            decision.reasons.append(("" if self.stage["mode"] == "live" else "shadow:") + f"tier:{self.stage['picked']}"
                                    + (" (judge)" if self.stage.get("judge") else ""))
        decision.reasons += outside + ([self.unstuck] if getattr(self, "unstuck", None) else [])
        # the decided tier must hold the request when it is on the floor's engine (a crossing carries a packet, which
        # compaction can shrink below); a pinned model that cannot is parked, never trimmed; any other is left for
        # the first ticked tier that holds it
        if not decision.crossing and not holds(self._context_of(decision.tier), est, self.max_tokens, self.rules.output_reserve_tokens):
            if named is not None:
                self._park(agent, named, est, thinking_requested)
                ticks, ladder, usable_names = self._resolve_ladder(agent)
                self.ladder = self._fitting([t for t in ladder if t.name in usable_names] or ladder, est)
                signals = Signals(named_tier=None, escalate=escalate, thinking_requested=thinking_requested)
                decision = decide(self.ladder, self.rules, state, signals, ceiling=None)
            else:
                fitting = [j for j, t in enumerate(self.ladder) if holds(self._context_of(t), est, self.max_tokens, self.rules.output_reserve_tokens)]
                j = next((k for k in fitting if k >= decision.index), fitting[0] if fitting else decision.index)
                if j != decision.index:
                    decision.reasons.append(f"too_big_for:{decision.tier.name}")
                    decision.tier, decision.index = self.ladder[j], j
                    decision.crossing = not same_engine(decision.tier, self.ladder[decision.floor])
                    decision.thinking = decision.tier.thinking if decision.tier.thinking is not None else thinking_requested
        # admission: the class at the gates, and the overflow to a later ticked engine when this one is busy
        self.priority = priority_of(self.dialect, self.input_priority, agent.priority if agent else None)
        rank = RANK.get(self.priority, 1)
        if self.admission is not None:
            mode = self.rules.capacity_mode if self.rules.capacity_mode in ("live", "shadow", "off") else "live"
            j, why = (decision.index, None) if (self.passthrough or mode == "off") else \
                self.admission.choose(self.ladder, decision.index, rank, sticky=state.current_tier is not None,
                                      wait_s=(lambda t: self.policy.wait_s(t.name)) if self.policy is not None else None)
            if why and mode == "shadow":                # the Switchyard's shadow mode: say what it would pick, change nothing
                decision.inputs["capacity_would_pick"] = self.ladder[j].name
                decision.reasons.append(f"shadow:{why}")
                j, why = decision.index, None
            if why:
                decision.tier, decision.index = self.ladder[j], j
                decision.crossing = not same_engine(decision.tier, self.ladder[decision.floor])
                decision.thinking = decision.tier.thinking if decision.tier.thinking is not None else thinking_requested
                decision.reasons.append(why)
            decision.inputs["priority"] = self.priority
            decision.inputs["expected_wait_s"] = round(self.admission.expected_wait(decision.tier, rank), 2)
        # the route it is on, and that route's own instructions (nebula/route_settings.py)
        self.route = route_settings.route_of(self, decision.tier.name)
        route_text = route_settings.instructions(self.rules, self.route)
        if route_text:
            decision.reasons.append(f"instructions:{self.route}")
            decision.inputs["route_instructions"] = {"route": self.route, "chars": len(route_text)}
        floor_tier = self.ladder[decision.floor]
        engine = self._engine(decision.tier)
        packet = None
        inputs = dict(decision.inputs, agent=agent.name if agent else None, ticks=ticks, usable=usable_names,
                      size=self.size, parked=self.parked)
        inputs["tenant"] = self.tenant
        if self.caller:
            inputs["caller"], inputs["label"] = self.caller.get("id"), self.caller.get("label")
        if self.peer and not str(self.peer).startswith("sandbox:"):
            inputs["peer"] = self.peer                  # a friend's call over Gravity: the map shows it come in
        self.est = est
        inputs["stages"] = [
            {"stage": "ticks", "models": list(ticks) if ticks else "all", "mode": "passthrough" if self.passthrough else "composite"},
            {"stage": "allowed", "off": dict(self.blocked)},
            {"stage": "answers", "models": list(usable_names)},
            {"stage": "fits", "models": [t.name for t in self.ladder if holds(self._context_of(t), est, self.max_tokens,
                                                                                  self.rules.output_reserve_tokens)],
             "need": (self.size or {}).get("need")},
            *([{"stage": "tier", **self.stage}] if getattr(self, "stage", None) else []),
            {"stage": "choice", "model": decision.tier.name, "why": list(decision.reasons) or [decision.reason]},
        ]
        if self.policy is not None and self.policy.thinking(decision.tier.name) is not None:
            decision.thinking = self.policy.thinking(decision.tier.name)
        if route_settings.thinking(self.rules, self.route) is not None:     # the route's kind says (nebula/route_settings.py)
            decision.thinking = route_settings.thinking(self.rules, self.route)
            decision.reasons.append(f"thinking:{self.route}")
        if self.policy is not None:                     # the model's own output cap, then the budget the slot leaves
            cap = self.policy.max_output(decision.tier.name)
            if cap:
                body["max_tokens"] = min(int(body.get("max_tokens") or cap), cap)
        self._budget(body, decision, inputs)
        if self.ledger is not None:
            if decision.crossing:
                reason = {"source": decision.reason, "text": self._reason_text(decision, state)}
                packet = build_packet(self.ledger, self.opened.session_id, reason=reason,
                                      destination_context=engine.n_ctx())
                if not packet["budget"]["fits"]:
                    # make room first: a briefing of what came before this request, then the packet again (the
                    # request's own new messages follow the briefing in it)
                    prior = [seq for seq, _p in self.ledger.transcript(self.opened.session_id) if seq < self.opened.first_seq]
                    if prior and self._compact(floor_tier, prior[-1], until_seq=self.opened.first_seq):
                        packet = build_packet(self.ledger, self.opened.session_id, reason=reason,
                                              destination_context=engine.n_ctx())
                if not packet["budget"]["fits"]:
                    decision.reasons.append("blocked:does_not_fit")
                    decision.blocked = "does_not_fit"
                    decision.crossing = False
                    decision.tier, decision.index = floor_tier, decision.floor
                    decision.thinking = floor_tier.thinking if floor_tier.thinking is not None else thinking_requested
                    engine = self._engine(floor_tier)
                    inputs["packet_estimated_tokens"] = packet["budget"]["estimated_tokens"]
                    packet = None
                else:
                    inputs["move_cost_tokens"] = packet["budget"]["estimated_tokens"] + 3000 * engine.busy()
                    inputs["move_over_ceiling"] = inputs["move_cost_tokens"] > self.rules.move_ceiling_tokens
            self.ledger.decide(self.attempt_id, tier=decision.tier.name, thinking=decision.thinking,
                               reason=decision.reason, inputs=inputs)
            if decision.crossing:
                sha = packet_sha256(packet)
                key = idempotency_key(sha, self.opened.request_sha256)
                self.handoff_id, self.attempt_id = self.ledger.escalate(
                    self.opened.attempt_id, to_tier=decision.tier.name, to_engine=decision.tier.address, packet=packet,
                    packet_sha256=sha, idempotency_key=key, thinking=decision.thinking, reason=decision.reason)
                if not same_engine(decision.tier, tiers[0]):
                    body["messages"] = render_messages(packet)
                # back on the origin engine, the raw history continues what its cache holds
            else:
                # the attempt opened on the floor; the decision's tier (and its thinking) is what it runs as
                self.ledger.retier(self.attempt_id, decision.tier.name, decision.tier.address, decision.thinking)
                if not same_engine(decision.tier, tiers[0]):
                    # a tier the session crossed to earlier: the same rendering, so its cache holds
                    last = self.ledger.latest_handoff(self.opened.session_id)
                    reason = last["packet"]["escalation_reason"] if last else {"source": "", "text": ""}
                    body["messages"] = render_messages(build_packet(self.ledger, self.opened.session_id, reason=reason,
                                                                    destination_context=engine.n_ctx()))
        if agent is not None:
            spec = self.tools.openai_tools(agent.tools) if (self.tools is not None and agent.tools) else []
            apply_agent(body, agent, system_text(agent, self.cfg.base_dir), spec)
        if route_text:                                  # the route's own instructions lead, before the agent's and the app's
            route_settings.apply(body, route_text)
        apply_thinking(body, decision.thinking)
        if self.policy is not None:
            self.policy.note_call(decision.tier.name)
        self.engine, self.tier, self.decision = engine, decision.tier, decision
        self._owner_answers(body)                       # Starweave: what the owner answered since the last turn
        self.team_tools = team.offer(self, body)        # Starweave's hand moves, when it is on or practising
        self.team_tools += toolbox.offer(self, body)    # the workspace toolbox: one folder, commands in OpenShell
        self.skill_tools = skill_tools.offer(self, body)
        if self.model_override and self.tier is not None and self.tier.name == self.model_override[0]:
            body["model"] = self.model_override[1]      # the star's dedicated model, by its name on that tier
        self._isolate(body)
        return engine, body

    # --- Starweave (nebula/team.py)
    def _owner_answers(self, body: dict) -> None:
        """The owner's answers to the team's questions, read by the lead at the start of its next turn."""
        if self.ledger is None or self.opened is None or self.team_ctx or team.settings(self.rules)["mode"] == "off":
            return
        got = self.ledger.owner_answers(self.opened.session_id, take=True)
        if got:
            route_settings.apply(body, "\n".join(f"The owner answered your team's question “{i['task_text'][:300]}”: {i['result']}"
                                                 for i in got))

    def _team_down(self, state) -> str | None:
        """The model an allowed, live hand-down in the previous turn named, when it is on this request's ladder."""
        if self.ledger is None or self.opened is None or self.pinned or team.settings(self.rules)["mode"] != "on":
            return None
        to = self.ledger.pending_hand_down(self.opened.session_id, self.attempt_id)
        return to if to and to in [t.name for t in self.ladder] else None

    def _team_call(self, ev, blocked: str | None):
        """A team call the model made: Halo's ruling, the ledger row, the text the model reads back, and the move to
        make now (a live hand-up: (tier, its packet or None, why)) or None."""
        name, args = ev.call.name, team.arguments_of(ev.call)
        if toolbox.is_box(name):                        # the workspace toolbox: its own checks, its own ledger row
            return toolbox.call(self, name, args), None
        mode = team.settings(self.rules)["mode"]
        if name in team.WORK:                           # phase 2: the team's work, run by nebula/teamwork.py
            return self._team_work(name, args, blocked if blocked != "one hand move per round" else None, mode), None
        if blocked:
            allowed, why, tier = False, blocked, None
        else:
            allowed, why, tier = team.check(self, name, args)
        packet = None
        if allowed and mode == "on" and name == team.HAND_UP and not same_engine(tier, self.tier):
            partial = text_of((assemble_reply(self.events) or {}).get("content")) if self.events else ""
            reason = {"source": f"team:hand_up:{self.tier.name}", "text": team.reason_text(self.tier.name, args.get("why"), partial)}
            packet = build_packet(self.ledger, self.opened.session_id, reason=reason,
                                  destination_context=self._engine(tier).n_ctx())
            if not packet["budget"]["fits"]:
                allowed, why, tier, packet = False, f"the conversation does not fit {tier.name}; carry on yourself", None, None
        team.record(self, name, args, allowed, why)
        text = team.answer(name, args, allowed, why, mode)
        if not allowed or mode != "on":
            return text, None
        if name == team.HAND_DOWN:
            self.hand_down_to = tier.name
            return text, None
        return text, (tier, packet, str(args.get("why") or ""))

    def _team_work(self, name: str, args: dict, blocked: str | None, mode: str) -> str:
        """A work call (delegate, collect, ask, note): Halo's ruling, the ledger row, then the work when it is live."""
        from nebula import teamwork
        try:
            allowed, why = (False, blocked) if blocked else team.check_work(self, name, args)
        except Exception as e:  # noqa: BLE001 - a malformed call is the model's to fix
            allowed, why = False, f"the call could not be read: {e}"
        team.record(self, name, args, allowed, why)
        if not allowed:
            return f"Not done: {why}."
        if mode != "on":
            return team.practice_text(name)
        try:
            return teamwork.call(self, name, args) + team.budget_note(self)
        except Exception as e:  # noqa: BLE001
            LOG.warning("starweave: %s failed: %s", name, e)
            return f"Not done: {e}."

    def _hand_up(self, move, body: dict, assistant: dict, results: list[dict]):
        """A live hand-up mid-turn: the larger model continues this answer. On the same engine (another thinking
        level) the attempt is re-tiered and the conversation goes on; on another engine the attempt crosses with the
        packet, as the router's crossings do, and the body is rendered from it. Returns the engine."""
        tier, packet, why = move
        frm = self.tier
        wanted = (self.decision.inputs or {}).get("thinking_requested") if self.decision is not None else None
        thinking = tier.thinking if tier.thinking is not None else wanted
        if self.policy is not None and self.policy.thinking(tier.name) is not None:
            thinking = self.policy.thinking(tier.name)
        reason = f"team:hand_up:{frm.name}->{tier.name}"
        inputs = {"from": frm.name, "to": tier.name, "why": why[:500], "round": self.rounds}
        engine = self._engine(tier)
        if packet is None:                              # the same engine: the conversation goes on there
            self.ledger.decide(self.attempt_id, tier=tier.name, thinking=thinking, reason=reason, inputs=inputs)
            self.ledger.retier(self.attempt_id, tier.name, tier.address, thinking)
            body["messages"] = list(body.get("messages") or []) + [assistant] + results
            self.tier, self.engine = tier, engine
            apply_thinking(body, thinking)
        else:
            self.ledger.decide(self.attempt_id, tier=tier.name, thinking=thinking, reason=reason, inputs=inputs)
            sha = packet_sha256(packet)
            key = idempotency_key(sha, f"{self.opened.request_sha256}:{reason}:{self.rounds}")
            self.handoff_id, self.attempt_id = self.ledger.escalate(
                self.attempt_id, to_tier=tier.name, to_engine=tier.address, packet=packet, packet_sha256=sha,
                idempotency_key=key, thinking=thinking, reason=reason)
            self.acknowledged = False
            keep = {k: v for k, v in body.items() if k not in ("messages", "tools", "id_slot", "cache_prompt")}
            body.clear()
            body.update(keep, messages=render_messages(packet))
            if self._client_tools:
                body["tools"] = list(self._client_tools)
            self.tier, self.engine = tier, engine
            if self.agent is not None:
                spec = self.tools.openai_tools(self.agent.tools) if (self.tools is not None and self.agent.tools) else []
                apply_agent(body, self.agent, system_text(self.agent, self.cfg.base_dir), spec)
            self.route = route_settings.route_of(self, tier.name)
            route_text = route_settings.instructions(self.rules, self.route)
            if route_text:
                route_settings.apply(body, route_text)
            apply_thinking(body, thinking)
        if self.policy is not None:
            cap = self.policy.max_output(tier.name)
            if cap:
                body["max_tokens"] = min(int(body.get("max_tokens") or cap), cap)
            self.policy.note_call(tier.name)
        tools = [t for t in body.get("tools") or [] if not team.is_team((t.get("function") or {}).get("name"))
                 and not toolbox.is_box((t.get("function") or {}).get("name"))
                 and (t.get("function") or {}).get("name") not in (skill_tools.LOAD, skill_tools.READ,
                                                                     skill_tools.RUN, skill_tools.SUGGEST)]
        if tools:
            body["tools"] = tools
        else:
            body.pop("tools", None)
        self.team_tools = team.offer(self, body)        # the new model's own moves: up and down from where it stands
        self.team_tools += toolbox.offer(self, body)
        self.skill_tools = skill_tools.offer(self, body)
        self._isolate(body)
        LOG.info("starweave: %s handed this turn up to %s", frm.name, tier.name)
        return engine

    def _budget(self, body: dict, decision, inputs: dict | None = None) -> None:
        """The Switchyard's output budget: what the answer may use is cut to what the slot holds after the prompt, the
        thinking budget giving way before the answer; the caller is told (X-Nebula-Budget, and on the decision)."""
        if not self.rules.clamp_output:
            return
        ctx = self._context_of(decision.tier)
        est = int((self.size or {}).get("estimate") or getattr(self, "est", 0) or 0)
        if not ctx or not est:
            return
        room = max(0, int(ctx) - est)
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        think = decision.tier.reasoning_budget_tokens if decision.thinking else None
        info = {"room": room, "asked": asked}
        if think and think > 0 and (asked or 0) + think > room:
            keep = max(0, room - (asked or self.rules.output_reserve_tokens))
            if keep < think:                            # thinking gives way first
                info.update(reasoning_was=think, reasoning_budget=keep)
                kw = dict(body.get("chat_template_kwargs") or {})
                kw["thinking_budget"] = keep
                body["chat_template_kwargs"] = kw
                body["reasoning_budget_tokens"] = keep
        if asked and int(asked) > room:
            body["max_tokens"] = max(1, room)
            info.update(max_tokens=body["max_tokens"], clamped=True, why=f"the slot holds {ctx} and the prompt is about {est}")
        else:
            info.update(max_tokens=asked or room, clamped=False)
        self.budget = info
        h = f"{info['max_tokens']} of {room}" + (f" (clamped: {info['why']})" if info.get("clamped") else "")
        if "reasoning_was" in info:
            h += f"; thinking {info['reasoning_budget']} (was {info['reasoning_was']})"
        rh = getattr(self, "reply_headers", None)
        if rh is not None:
            rh["X-Nebula-Budget"] = h
        decision.inputs["budget"] = info
        if inputs is not None:
            inputs["budget"] = info

    def _unstick(self, state, ladder) -> None:
        """The Switchyard's stickiness, bounded: a session idle longer than `sticky_seconds` may move again; one that an
        escalation lifted goes back to its floor after `escalation_hold_seconds` without a new hard turn."""
        from datetime import datetime
        if state.current_tier is None or not state.last_answer_at:
            return
        try:
            idle = time.time() - datetime.fromisoformat(state.last_answer_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return
        r = self.rules
        if r.sticky_seconds and idle > r.sticky_seconds:
            self.unstuck = f"sticky_expired:{state.current_tier}"
            state.current_tier = None
        elif r.escalation_hold_seconds and ladder and state.current_tier != ladder[0].name and idle > r.escalation_hold_seconds \
                and not state.tool_error_streak and not state.regenerations:
            self.unstuck = f"hold_expired:{state.current_tier}"
            state.current_tier = None

    def needs(self, agent=None) -> list[str]:
        """What this request needs of a model (the Switchyard's capabilities): pictures, tools, a structured answer, a
        long context."""
        if getattr(self, "_needs", None) is not None:
            return self._needs
        r, out = self.req if isinstance(self.req, dict) else {}, []
        msgs = r.get("messages") or r.get("input") or []
        text = 0
        for m in msgs if isinstance(msgs, list) else []:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, list):
                if any(isinstance(p, dict) and (p.get("type") in ("image_url", "image", "input_image")) for p in c):
                    if "vision" not in out:
                        out.append("vision")
                text += sum(len(str(p.get("text") or "")) for p in c if isinstance(p, dict))
            elif isinstance(c, str):
                text += len(c)
        if r.get("tools") or (agent is not None and agent.tools):
            out.append("tools")
        fmt = r.get("response_format") or {}
        if isinstance(fmt, dict) and fmt.get("type") in ("json_schema", "json_object"):
            out.append("structured")
        if text / 3.5 > self.rules.long_context_tokens:
            out.append("long_context")
        self._needs = out
        return out

    def _eligible(self, t, need: str) -> bool:
        if need == "vision":                            # what the engine reports, overlaid by the owner's checkbox
            reported = True
            if self.registry is not None:
                node = self.registry.node(t.name) if self.registry.up(t.name) else None
                reported = bool(node and node.get("vision")) if node is not None else True
            return self.policy.eligible(t.name, "vision", reported)
        return self.policy.eligible(t.name, need, True)

    def _resolve_ladder(self, agent):
        """(ticks, the ticked tiers in order, the names of those whose engine answers), as the config and the
        registry stand now; a parked request asks this again."""
        tiers = self.cfg.tiers
        ticks = self.input_ticks or (list(agent.models) if agent is not None and agent.models else None) \
            or (list(self.cfg.default_models) or None)
        ladder = [self.cfg.tier(n) for n in ticks if self.cfg.tier(n) is not None] if ticks else list(tiers)
        if self.peer:                                   # a paired Nebula: only what this one advertises to it,
            own = [t.name for t in tiers if not t.peer and not t.is_command]   # and a CLI never (the owner's account)
            offered = set(self.policy.advertised(self.peer, own) if self.policy is not None else own)
            self.not_offered = {t.name for t in tiers} - offered
            ladder = [t for t in ladder if t.name in offered]
        if self.policy is not None:                     # off, out of its hours, over its count: off the ladder
            ladder = self.policy.kept(ladder) if ticks else self.policy.order(ladder)   # a chosen list keeps its order
            self.blocked = {}
            for t in ladder:
                ok, why = self.policy.allowed(t.name)
                if not ok:
                    self.blocked[t.name] = why
            for t in tiers:
                if t.name not in self.blocked:
                    ok, why = self.policy.allowed(t.name)
                    if not ok:
                        self.blocked[t.name] = why
            ladder = [t for t in ladder if t.name not in self.blocked]
        if self.guest:                                  # a guest: only the models shared with guests
            for t in tiers:
                if not getattr(t, "shared", False):
                    self.blocked.setdefault(t.name, "not shared with guests")
            ladder = [t for t in ladder if t.name not in self.blocked]
            named = names.bare_model(self.req.get("model")) if isinstance(self.req.get("model"), str) else None
            if named and self.blocked.get(named) == "not shared with guests" and self.seclog is not None:
                self.seclog.add("guest.refused", self.tenant, f"model:{named}")
        guard = getattr(self, "guard", None)
        for m in (guard.denied(self.tenant) if guard is not None and self.guest else []):   # a model Halo held back
            if self.cfg.tier(m) is not None:
                self.blocked.setdefault(m, "held back from this guest (Halo)")
        ladder = [t for t in ladder if t.name not in self.blocked]
        if self.yard is not None and self.yard.draining:  # an engine being drained takes nothing new
            for t in tiers:
                if self.yard.is_draining(t.address):
                    self.blocked[t.name] = "draining"
            ladder = [t for t in ladder if t.name not in self.blocked]
        if self.policy is not None:                     # the Switchyard's route policy: eligibility and explicit-only
            named = names.bare_model(self.req.get("model")) if isinstance(self.req.get("model"), str) else None
            for t in tiers:
                if t.name in self.blocked:
                    continue
                if self.policy.explicit(t.name) and t.name != named:
                    self.blocked[t.name] = EXPLICIT_ONLY
                    continue
                miss = [n for n in self.needs(agent) if not self._eligible(t, n)]
                if miss:
                    self.blocked[t.name] = "not eligible for " + " and ".join(NEED_WORDS[n] for n in miss)
            ladder = [t for t in ladder if t.name not in self.blocked]
        usable = self.registry.usable([t.name for t in ladder]) if self.registry is not None else [t.name for t in ladder]
        return ticks, ladder, usable

    def _refuse(self, status: int, msg: str, reason: str, ladder, thinking_requested, agent):
        """A request the policy will not route: its decision and its refusal recorded, then the error."""
        if self.ledger is not None and getattr(self, "attempt_id", None):
            self.ledger.decide(self.attempt_id, tier=(ladder[0].name if ladder else self.cfg.tiers[0].name),
                               thinking=thinking_requested, reason=reason,
                               inputs={"blocked": self.blocked, "peer": self.peer, "agent": agent.name if agent else None})
            self.ledger.close_request(self.attempt_id, reply=None, done=None, error=msg)
        raise EngineError(status, {"error": {"message": msg, "type": "policy"}})

    def _context_of(self, tier) -> int | None:
        if self.registry is not None:
            node = self.registry.node(tier.name)
            if node is not None:
                return node.get("context")
        return tier.context

    def _fitting(self, ladder, estimate: int):
        return [t for t in ladder if holds(self._context_of(t), estimate, self.max_tokens, self.rules.output_reserve_tokens)]

    def _park(self, agent, named, estimate: int, thinking_requested) -> None:
        """Wait in the lot until a ticked model holds the request, or refuse it after the wait."""
        need = self.size["need"] if self.size else estimate
        contexts = {t.name: self._context_of(t) for t in self.ladder}
        biggest = max((c for c in contexts.values() if c), default=None)
        reason = (f"needs {need} tokens; " + (f"the pinned model {named} holds {contexts.get(named)}" if named else
                  f"the ticked models hold at most {biggest}")) if biggest else f"needs {need} tokens; no ticked model holds it"
        info = {"attempt_id": self.attempt_id, "session_id": self.opened.session_id if self.opened else None,
                "input": self.dialect, "agent": agent.name if agent else None, "need": need, "size": self.size,
                "ticks": [t.name for t in self.ladder], "contexts": contexts, "reason": reason}
        if self.ledger is not None:
            self.ledger.decide(self.attempt_id, tier=self.ladder[0].name, thinking=thinking_requested,
                               reason="parked:too_big", inputs={**info, "agent": agent.name if agent else None})
        if self.lot is None:
            raise EngineError(503, {"error": {"message": "parked_too_big: " + reason, "type": "parked_too_big"}})
        if len(self.lot.parked()) >= max(0, int(self.rules.parking_max_calls)):
            msg = f"parking_full: {self.rules.parking_max_calls} request(s) already wait for a model big enough; " + reason
            if self.ledger is not None:
                self.ledger.close_request(self.attempt_id, reply=None, done=None, error=msg)
            raise EngineError(503, {"error": {"message": msg, "type": "parking_full"}})

        def fits_now():
            if named is not None:                 # pinned: only that model can take it, once it holds it
                t = self.cfg.tier(named)
                return named if t is not None and self._fitting([t], estimate) else None
            _ticks, lad, usable = self._resolve_ladder(agent)
            f = self._fitting([t for t in lad if t.name in usable], estimate)
            return f[0].name if f else None
        try:
            item, tier = self.lot.park(info, fits_now, self.rules.parking_wait_seconds)
        except ParkedTooBig as e:
            if self.ledger is not None:
                self.ledger.close_request(self.attempt_id, reply=None, done=None, error=str(e))
            raise EngineError(503, {"error": {"message": str(e), "type": "parked_too_big"}}) from None
        self.parked = {"waited_s": item.get("waited_s"), "released_to": tier, "reason": reason}

    def _engine(self, tier):
        table = self.registry.engines if self.registry is not None else self.engines
        eng = table.get(tier.name)
        if eng is None:
            raise ValueError(f"tier {tier.name!r} has no engine")
        return eng

    @staticmethod
    def _reason_text(decision, state: SessionState) -> str:
        bits = []
        if state.regenerations:
            bits.append(f"the user asked for this turn {state.regenerations + 1} times")
        if state.tool_error_streak:
            bits.append(f"{state.tool_error_streak} tool results in a row reported errors")
        if decision.inputs.get("stuck_phrase"):
            bits.append(f"the previous model said: {state.last_reply_text[:200]!r}")
        if decision.inputs.get("escalate"):
            bits.append("the user pressed escalate")
        if decision.inputs.get("named_tier"):
            bits.append(f"the request named the {decision.inputs['named_tier']} tier")
        return "; ".join(bits) or decision.reason

    def _compact(self, tier, covers_through_seq: int, until_seq: int | None = None) -> bool:
        """A briefing by the secretary agent when one is configured (its tier, its system prompt), else by the
        tier's engine as it is."""
        secretary = self.cfg.secretary_agent()
        if secretary is not None and secretary.tier and self.cfg.tier(secretary.tier) is not None:
            tier = self.cfg.tier(secretary.tier)
        engine = self._engine(tier)
        full = full_read_due(self.ledger.briefing_count(self.opened.session_id), self.cfg.compaction_full_every)
        messages = transcript_messages(self.ledger, self.opened.session_id, until_seq, full=full)
        if secretary is not None:
            system = system_text(secretary, self.cfg.base_dir)
            if system:
                messages = [{"role": "system", "content": system}] + messages
        key = "compaction:" + self.opened.session_id
        self.ledger.activity.begin(key, what="compaction", agent=secretary.name if secretary else None, tier=tier.name,
                                   session_id=self.opened.session_id)
        try:
            b = compact(engine, messages)
        except Exception as e:  # noqa: BLE001 - a failed compaction must not fail the user's turn
            LOG.warning("compaction on %s failed: %s", tier.name, e)
            return False
        finally:
            self.ledger.activity.end(key)
        if b is None:
            return False
        self.ledger.briefing(self.opened.session_id, self.attempt_id,
                             {**b, "covers_through_seq": covers_through_seq, "tier": tier.name,
                              "agent": secretary.name if secretary else None})
        return True

    # --- the reply
    def add(self, ev) -> None:
        if ev.kind == "tool_call":
            self.client_calls += 1
        if self.opened is None:
            return
        self.events.append(ev)
        self.ledger.activity.token(self.attempt_id)   # the page's pulse; the ledger keeps the counts at the close
        if self.handoff_id and not self.acknowledged:
            self.acknowledged = True
            self.ledger.acknowledge(self.handoff_id)

    def run(self, engine, body: dict):
        """The reply's events, across the agent's tool rounds. Records and closes the attempt."""
        body = dict(body)
        body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}   # the counts, every round
        own_tools = self.agent is not None and bool(self.agent.tools) and self.tools is not None
        team_names = set(getattr(self, "team_tools", None) or [])
        skill_names = set(getattr(self, "skill_tools", None) or [])
        gate = self.admission.gate(self.tier) if (self.admission is not None and self.tier is not None) else None
        held = False
        started = None
        info = None
        try:
            if gate is not None:
                info = self._gate_info()
                try:
                    self.queued_s = gate.acquire(RANK.get(self.priority, 1), info, timeout=self._queue_check(gate))
                except QueueTimeout as e:
                    raise EngineError(503, {"error": {"message": f"the engine for {self.tier.name} stayed busy: {e}",
                                                      "type": "server_error"}}) from None
                held = True
                started = time.monotonic()
            first_round = True
            while True:
                chunks = engine.chat_stream(body)
                if first_round:                      # nothing answered yet: a failure here may move to the next model
                    first_round = False
                    while True:
                        try:
                            head = next(chunks, None)
                            break
                        except EngineError as e:
                            moved = self.reroute(e, body)
                            if moved is None:
                                raise
                            if held:
                                gate.release(info)
                                held = False
                            engine, gate = moved
                            if gate is not None:
                                info = self._gate_info()
                                self.queued_s += gate.acquire(RANK.get(self.priority, 1), info, timeout=self.rules.queue_timeout_seconds)
                                held, started = True, time.monotonic()
                            chunks = engine.chat_stream(body)
                    chunks = itertools.chain([head], chunks) if head is not None else iter(())
                stream = Stream(chunks)
                own, round_events = [], []
                for ev in stream:
                    round_events.append(ev)
                    if ev.kind == "tool_call" and ((own_tools and self.tools.owns(ev.call.name)) or
                                                    ev.call.name in team_names or ev.call.name in skill_names):
                        own.append(ev)
                        continue
                    self.add(ev)
                    yield ev
                self._accumulate(stream.done)
                if not own:
                    break
                self.rounds += 1
                assistant = assemble_reply(round_events)
                results = []
                client_calls_this_round = [ev for ev in round_events if ev.kind == "tool_call" and ev not in own]
                hand = None
                waits = held and not client_calls_this_round and any(ev.call.name in team.WAITS for ev in own) \
                    and team.settings(self.rules)["mode"] == "on"
                if waits:                    # the lead waits on its team: its engine slot is given back meanwhile
                    gate.release(info)
                    held = False
                    if started is not None and self.admission is not None:
                        self.admission.learn(self.tier, time.monotonic() - started)
                for ev in own:
                    if ev.call.name in skill_names:
                        try:
                            args = json.loads(ev.call.arguments or "{}") if isinstance(ev.call.arguments, str) else ev.call.arguments
                            result = skill_tools.call(self, ev.call.name, args)
                        except Exception as e:  # noqa: BLE001 - malformed model call returns to the model
                            result = f"Not done: {e}."
                        results.append({"role": "tool", "tool_call_id": ev.call.id, "content": result})
                        continue
                    if ev.call.name in team_names:      # Starweave: Halo rules on the move first
                        blocked = ("the turn also called the client's tools; answer those first" if client_calls_this_round
                                   else "one hand move per round" if hand is not None else None)
                        text, move = self._team_call(ev, blocked)
                        hand = move or hand
                        results.append({"role": "tool", "tool_call_id": ev.call.id, "content": text})
                        continue
                    try:
                        text = self.tools.call(ev.call.name, ev.call.arguments)
                    except Exception as e:  # noqa: BLE001 - the tool's failure is the model's to read
                        LOG.warning("agent %s: the tool %s failed: %s", self.agent.name, ev.call.name, e)
                        text = f"error: {e}"
                    results.append({"role": "tool", "tool_call_id": ev.call.id, "content": text})
                if self.ledger is not None and self.opened is not None:
                    self.ledger.tool_round(self.opened.session_id, self.attempt_id, assistant, results)
                max_rounds = self.agent.max_rounds if self.agent is not None else team.MAX_ROUNDS
                if any(toolbox.is_box(n) for n in team_names):     # work in a folder takes many small steps
                    max_rounds = max(max_rounds, toolbox.settings(self.rules)["max_rounds"])
                if client_calls_this_round:
                    break                    # the client's calls end the turn
                if hand is not None:         # a live hand-up: the larger model continues this answer
                    if held:
                        gate.release(info)
                        held = False
                        if started is not None and self.admission is not None:
                            self.admission.learn(self.tier, time.monotonic() - started)
                    engine = self._hand_up(hand, body, assistant, results)
                    team_names = set(getattr(self, "team_tools", None) or [])
                    skill_names = set(getattr(self, "skill_tools", None) or [])
                    self.rounds = 0
                    gate = self.admission.gate(self.tier) if self.admission is not None else None
                    if gate is not None:
                        info = self._gate_info()
                        self.queued_s += gate.acquire(RANK.get(self.priority, 1), info, timeout=self.rules.queue_timeout_seconds)
                        held, started = True, time.monotonic()
                    continue
                if self.rounds >= max_rounds:
                    break                    # so does the rounds limit
                budget = (getattr(self, "team_ctx", None) or {}).get("token_budget")
                if budget:                   # a priced teammate's piece: its rounds stop at its share of the cloud cap
                    from nebula import teamwork
                    used = (self.done.prompt_tokens or 0) + (self.done.completion_tokens or 0)
                    room = budget - used - teamwork.prompt_tokens(list(body.get("messages") or []) + [assistant] + results)
                    if room < 64:
                        break
                    body["max_tokens"] = min(body.get("max_tokens") or room, room)
                if waits and gate is not None:   # the lead goes on: its slot again, at its own class
                    info = self._gate_info()
                    self.queued_s += gate.acquire(RANK.get(self.priority, 1), info, timeout=self.rules.queue_timeout_seconds)
                    held, started = True, time.monotonic()
                body["messages"] = list(body.get("messages") or []) + [assistant] + results
            self.finish(self.done)
        except EngineError as e:
            if e.status == 502 and self.registry is not None and self.tier is not None:
                self.registry.mark_down(self.tier.name)
            self.finish(error=e.message)
            raise
        finally:
            if held:
                gate.release(info)
                if started is not None and self.admission is not None:
                    self.admission.learn(self.tier, time.monotonic() - started)
            self.finish(self.done, completeness="interrupted")

    def reroute(self, error: EngineError, body: dict | None = None):
        """The engine failed before it answered anything (unreachable, overloaded, timed out): the next eligible model,
        once per `retries_before_first_byte`, recorded as a decision of its own. None when there is none, the turn is a
        passthrough, the request named its model itself (it fails with that engine's own answer), or the failure was the
        request's own (a policy refusal, a request too big). Returns (engine, gate)."""
        if self.passthrough or self.pinned or self.tier is None or error.status not in (502, 503, 504) \
                or self.retries >= self.rules.retries_before_first_byte:
            return None
        kind = ((error.body or {}).get("error") or {}).get("type") if isinstance(error.body, dict) else None
        if kind in ("policy", "parked_too_big"):
            return None
        self.tried.add(self.tier.name)
        if error.status == 502 and self.registry is not None:
            self.registry.mark_down(self.tier.name)
        est = getattr(self, "est", 0)
        nxt = None
        for t in self.ladder:
            if t.name in self.tried or (self.registry is not None and not self.registry.up(t.name)):
                continue
            if self.yard is not None and self.yard.is_draining(t.address):
                continue
            if not holds(self._context_of(t), est, self.max_tokens, self.rules.output_reserve_tokens):
                continue
            nxt = t
            break
        if nxt is None:
            return None
        self.retries += 1
        was = self.tier.name
        self.tier, self.engine = nxt, self._engine(nxt)
        if self.ledger is not None and self.attempt_id:
            self.ledger.decide(self.attempt_id, tier=nxt.name, thinking=nxt.thinking, reason=f"retry_before_first_byte:{was}",
                               inputs={"from": was, "error": str(error.message)[:200], "status": error.status,
                                       "retry": self.retries, "caller": (self.caller or {}).get("id")})
            self.ledger.retier(self.attempt_id, nxt.name, nxt.address, nxt.thinking)
        LOG.warning("%s failed before answering (%s): retrying on %s", was, error.status, nxt.name)
        gate = self.admission.gate(nxt) if self.admission is not None else None
        if body is not None:
            self._isolate(body)                         # the new model keeps the tenant to its own cache too
        return self.engine, gate

    def _isolate(self, body: dict) -> None:
        """On a shared model, this request kept to its tenant's own cache (nebula/tenancy.py); recorded on the decision."""
        from nebula import tenancy
        body.pop("id_slot", None)
        body.pop("cache_prompt", None) if self.isolation else None
        self.isolation = tenancy.isolate(body, self.tier, self.engine, self.tenant, self.guest, self.slots)
        if self.isolation and self.ledger is not None and self.attempt_id:
            self.ledger.decide(self.attempt_id, tier=self.tier.name, thinking=None, reason="isolated",
                               inputs={"tenant": self.tenant, "isolation": self.isolation}, advisory=True)

    def _gate_info(self) -> dict:
        """What a gate shows of the call that waits at it or holds its slot (the Lanes)."""
        return {"attempt_id": self.attempt_id, "tier": self.tier.name if self.tier else None,
                "agent": self.agent.name if self.agent else None, "input": self.dialect,
                "session_id": self.opened.session_id if self.opened else None,
                "caller": (self.caller or {}).get("id"), "label": (self.caller or {}).get("label"), "class": self.priority,
                "tenant": self.tenant}

    def _queue_check(self, gate) -> float:
        """The model's own queue (the Switchyard's): refused at once when as many of this class already wait as it allows
        (503 queue_full); returns how long this call may wait at its gate."""
        q = self.policy.queue(self.tier.name) if (self.policy is not None and self.tier is not None) else {}
        rank = RANK.get(self.priority, 1)
        cap = (q.get("max_queued") or {}).get(self.priority)
        if cap is not None and not gate.free(rank) and gate.queued(rank) >= cap:
            raise EngineError(503, {"error": {"message": f"queue_full: {self.tier.name} already has {cap} {self.priority} "
                                                         "call(s) waiting; try again soon", "type": "queue_full"}})
        return float(q.get("timeout_s") or self.rules.queue_timeout_seconds)

    def slot(self, gate):
        """The gate taken around an engine call the dialect makes itself (the chat passthrough)."""
        turn = self

        class _Slot:
            def __enter__(self_):
                self_.held, self_.started = False, None
                if gate is None:
                    return self_
                info = self_.info = turn._gate_info()
                try:
                    turn.queued_s = gate.acquire(RANK.get(turn.priority, 1), info, timeout=turn._queue_check(gate))
                except QueueTimeout as e:
                    raise EngineError(503, {"error": {"message": f"the engine for {turn.tier.name} stayed busy: {e}",
                                                      "type": "server_error"}}) from None
                self_.held, self_.started = True, time.monotonic()
                return self_

            def __exit__(self_, *exc):
                if self_.held:
                    gate.release(getattr(self_, "info", None))
                    if turn.admission is not None:
                        turn.admission.learn(turn.tier, time.monotonic() - self_.started)
                return False
        return _Slot()

    def _accumulate(self, d: Done) -> None:
        """One round's counts into the turn's: tokens and timings add up; finish, model and id are the last round's."""
        t = self.done
        t.prompt_tokens += d.prompt_tokens
        t.completion_tokens += d.completion_tokens
        t.cached_tokens += d.cached_tokens
        t.finish, t.model, t.id = d.finish, d.model or t.model, d.id or t.id
        timings = dict(t.timings)
        for k, v in d.timings.items():
            if isinstance(v, (int, float)) and k in ("prompt_n", "cache_n", "predicted_n", "draft_n", "draft_n_accepted",
                                                     "prompt_ms", "predicted_ms"):
                timings[k] = timings.get(k, 0) + v
            else:
                timings[k] = v
        t.timings = timings
        t.usage = {"prompt_tokens": t.prompt_tokens, "completion_tokens": t.completion_tokens,
                   "total_tokens": t.prompt_tokens + t.completion_tokens,
                   "prompt_tokens_details": {"cached_tokens": t.cached_tokens}}

    def finish(self, done: Done | None = None, completeness: str = "complete", error: str | None = None) -> None:
        if not self.exported:                               # one span per turn, ledger or not (nebula/otel.py)
            self.exported = True
            from nebula import otel
            otel.turn_span(self.otel, self, done, completeness, error)
        if self.opened is None or self.finished:
            return
        self.finished = True
        reply = assemble_reply(self.events) if self.events else None
        self.ledger.close_request(self.attempt_id, reply=reply, done=done, completeness=completeness, error=error,
                                  queued_s=self.queued_s if self.admission is not None else None)
        if done is None or error or completeness != "complete":
            return
        latest = self.ledger.latest_briefing(self.opened.session_id)
        conv = self.ledger.transcript(self.opened.session_id)
        last_seq = conv[-1][0] if conv else 0
        if compaction_due(done.prompt_tokens + done.completion_tokens, self.engine.n_ctx(),
                          self.cfg.compaction_fraction, latest[0] if latest else None, last_seq) \
                or getattr(self, "hand_down_to", None):  # Starweave: the smaller model starts from a briefing
            self._compact(self.tier, last_seq)
