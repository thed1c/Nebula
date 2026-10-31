"""nebula/router.py: which tier answers this turn (docs/DESIGN.md, "The router").

The tiers are a ladder, in the config's order. A session's floor is the tier that answered its last completed
turn (the first tier before any): the router does not drop back on its own, since a lower tier would read the
whole prompt again. Signals raise the target above the floor; a tier named in the request (its model field, or
the X-Nebula-Escalate header) sets it outright, up or down. The lowest tier that satisfies every signal wins.

A crossing is a target whose engine differs from the floor's. It needs the transcript at a turn boundary: a
trailing call without its result (a pending tool) blocks it, and the highest tier on the same engine answers
instead, with the reason recorded. v1 rules are thresholds, no learning: every decision is a row the evaluation
in VALIDATION.md reads later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from nebula import names
from nebula.config import Tier
from nebula.ledger import STUCK_DEFAULT, SessionState


@dataclass
class Rules:
    regenerations_to_rise: int = 1            # the same turn asked again: one tier up
    regenerations_to_cross: int = 2           # ... and again: the next engine
    tool_errors_to_rise: int = 2              # tool results reporting errors, in a row
    tool_errors_to_cross: int = 4
    stuck_phrases: tuple[str, ...] = STUCK_DEFAULT
    move_ceiling_tokens: int = 65536          # a crossing that re-reads more than this is flagged (the UI asks)
    tolerance_seconds: dict = field(default_factory=lambda: {"interactive": 60.0, "agent": 15.0, "background": 5.0})
    queue_timeout_seconds: float = 600.0      # a turn that waits longer than this at a gate is refused 503
    parking_wait_seconds: float = 600.0       # a parked request is refused after this
    parking_recheck_seconds: float = 3.0      # how often a parked request asks again
    output_reserve_tokens: int = 1024         # the output allowed for when a request names no max_tokens
    estimate_margin: float = 1.1              # on the character estimate, when the engine has no tokenizer
    retries_before_first_byte: int = 1        # an engine that fails before it answers anything: the next eligible model
    capacity_mode: str = "live"               # live: the free slot / shortest wait picks; shadow: only logged; off: as decided
    sticky_seconds: float = 0.0               # a run or session stays on its model; >0: it may move after this long idle
    parking_max_calls: int = 50               # how many may wait in the parking lot at once (more: refused, parking_full)
    long_context_tokens: int = 32768          # a request this long needs a model eligible for long context
    clamp_output: bool = True                 # the answer's budget is cut to fit the slot (thinking gives way first)
    escalation_hold_seconds: float = 0.0      # an escalated session goes back to its floor after this long calm (0: stays)
    escalation_ignore_outages: bool = True    # a tool error that is an outage (a 50x, a refused connection) is no strike
    stage_router: dict = field(default_factory=dict)   # nebula/stage.py: {mode, picker, threshold, judge}
    route_rules: list = field(default_factory=list)    # nebula/route_rules.py: if … (is / is not) … then …, in order
    routes: dict = field(default_factory=dict)         # nebula/route_settings.py: {route: {"instructions": …}}

    @staticmethod
    def from_dict(d: dict | None) -> "Rules":
        r = Rules()
        for k, v in (d or {}).items():
            if k == "stuck_phrases":
                r.stuck_phrases = tuple(str(x).lower() for x in v)
            elif k == "tolerance_seconds" and isinstance(v, dict):
                r.tolerance_seconds = {**r.tolerance_seconds, **{str(a): float(b) for a, b in v.items()}}
            elif k == "stage_router" and isinstance(v, dict):
                r.stage_router = dict(v)
            elif k == "route_rules" and isinstance(v, list):
                r.route_rules = list(v)
            elif k == "routes" and isinstance(v, dict):
                r.routes = {str(a): dict(b) for a, b in v.items() if isinstance(b, dict)}
            elif k in ("clamp_output", "escalation_ignore_outages"):
                r.__dict__[k] = bool(v)
            elif hasattr(r, k):
                setattr(r, k, type(getattr(r, k))(v))
        return r


@dataclass
class Signals:
    """What this request itself says."""
    named_tier: str | None = None             # the model field names a tier ("big", "nebula/big")
    escalate: str | None = None               # the X-Nebula-Escalate header: "1" for one tier up, or a tier's name
    thinking_requested: bool | None = None    # the request's own thinking setting


@dataclass
class Decision:
    tier: Tier
    index: int
    floor: int
    thinking: bool | None
    reasons: list[str] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    crossing: bool = False
    blocked: str | None = None                # a crossing that was wanted and not made, and why

    @property
    def reason(self) -> str:
        return ",".join(self.reasons) if self.reasons else "floor"


def same_engine(a: Tier, b: Tier) -> bool:
    return (a.address or "") == (b.address or "")


def decide(tiers: list[Tier], rules: Rules, state: SessionState, signals: Signals,
           ceiling: str | None = None) -> Decision:
    names = [t.name for t in tiers]
    floor = names.index(state.current_tier) if state.current_tier in names else 0
    top = names.index(ceiling) if ceiling in names else len(tiers) - 1   # an agent's ceiling caps the ladder
    floor = min(floor, top)

    def first_crossing_above(i: int) -> int:
        for j in range(i + 1, len(tiers)):
            if not same_engine(tiers[j], tiers[i]):
                return j
        return top

    target, reasons = floor, []
    want = []
    if state.regenerations >= rules.regenerations_to_cross:
        want.append((first_crossing_above(floor), f"regenerations:{state.regenerations}"))
    elif state.regenerations >= rules.regenerations_to_rise:
        want.append((floor + 1, f"regenerations:{state.regenerations}"))
    if state.tool_error_streak >= rules.tool_errors_to_cross:
        want.append((first_crossing_above(floor), f"tool_errors:{state.tool_error_streak}"))
    elif state.tool_error_streak >= rules.tool_errors_to_rise:
        want.append((floor + 1, f"tool_errors:{state.tool_error_streak}"))
    low = state.last_reply_text.lower()
    hit = next((p for p in rules.stuck_phrases if p in low), None)
    if hit and not state.regenerations:       # a regeneration already counts the turn; the claim is logged once
        want.append((floor + 1, "model_claim:stuck"))
    if signals.escalate:
        if signals.escalate in names:
            want.append((names.index(signals.escalate), f"escalate:{signals.escalate}"))
        else:
            want.append((floor + 1, "escalate:1"))
    for idx, why in want:
        if idx > target:
            target = idx
        reasons.append(why)
    if signals.named_tier in names:
        target = names.index(signals.named_tier)
        reasons.append(f"named:{signals.named_tier}")
    if target > top:
        target = top
        reasons.append(f"ceiling:{names[top]}")
    blocked = None
    if not same_engine(tiers[target], tiers[floor]) and state.pending_tool:
        blocked = "pending_tool"                 # not at a turn boundary: the floor answers, the reason is kept
        target = floor
        reasons.append("blocked:pending_tool")
    tier = tiers[target]
    thinking = tier.thinking if tier.thinking is not None else signals.thinking_requested
    inputs = {"floor": tiers[floor].name, "regenerations": state.regenerations,
              "tool_error_streak": state.tool_error_streak, "stuck_phrase": hit, "pending_tool": state.pending_tool,
              "named_tier": signals.named_tier, "escalate": signals.escalate,
              "thinking_requested": signals.thinking_requested, "transcript_len": state.transcript_len,
              "last_prompt_tokens": state.last_prompt_tokens}
    return Decision(tier, target, floor, thinking, reasons, inputs,
                    crossing=not same_engine(tier, tiers[floor]), blocked=blocked)


def tier_named(model: str | None, tiers: list[Tier]) -> str | None:
    """A model field that names a tier: "big" or "nebula/big" (or the older "cockpit/big")."""
    if not isinstance(model, str):
        return None
    name = names.bare_model(model)
    return name if any(t.name == name for t in tiers) else None
