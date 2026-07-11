"""Greatness Monitor core (HIGH_CONVICTION_FEATURES_PLAN.md sec 1, 15, 16.2).

Pure confirmation engine for promising D1 candidates: the slow Master/D1 scan
defines an ordered ConfirmationPlan (what greatness would look like); this
engine consumes completed intraday bars and moves the candidate through typed
stages as it clears steps. A five-minute wick is evidence (TESTING_LEVEL /
WICK_THROUGH), never confirmation - READY requires the plan's own conditions
on completed closes. Failed attempts re-arm under an explicit policy instead
of consuming the day's trigger.

Side symmetry via side_sign (+1 long / -1 short): identical logic both ways,
labels only differ. State survives restarts through to_dict/from_dict - a UI
refresh or rescan can never reset the progression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from market_state import M5Bar

ENGINE_VERSION = "greatness_v1"


class Stage(str, Enum):
    DISCOVERED = "DISCOVERED"
    DEVELOPING = "DEVELOPING"
    NEAR_TRIGGER = "NEAR_TRIGGER"
    TESTING_LEVEL = "TESTING_LEVEL"
    CONFIRMING = "CONFIRMING"
    READY = "READY"
    FAILED = "FAILED"
    REARMED = "REARMED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


TERMINAL_STAGES = {Stage.INVALIDATED, Stage.EXPIRED}


class Condition(str, Enum):
    TOUCH = "TOUCH"            # reaching the level is enough (soft steps)
    CLOSE = "CLOSE"            # completed close through the level
    ACCEPT = "ACCEPT"          # N consecutive completed closes beyond
    RETEST_HOLD = "RETEST_HOLD"  # close through, pull back to level, hold


class EventType(str, Enum):
    LEVEL_TOUCHED = "LEVEL_TOUCHED"
    WICK_THROUGH = "WICK_THROUGH"
    CLOSED_THROUGH = "CLOSED_THROUGH"
    ACCEPTED = "ACCEPTED"
    RETEST_HELD = "RETEST_HELD"
    STEP_CLEARED = "STEP_CLEARED"
    FAILED_ATTEMPT = "FAILED_ATTEMPT"
    REARMED = "REARMED"
    READY = "READY"
    INVALIDATED = "INVALIDATED"


@dataclass
class TransitionEvent:
    symbol: str
    event: EventType
    step_label: str
    ts: datetime
    price: float
    attempt: int
    stage: Stage

    @property
    def dedupe_key(self) -> str:
        return f"{self.symbol}|{self.event.value}|{self.step_label}|{self.attempt}"


@dataclass
class ConfirmationStep:
    label: str
    level: float
    condition: Condition = Condition.CLOSE
    required_bars: int = 2  # ACCEPT only
    mandatory: bool = True
    cleared: bool = False
    cleared_at: str = ""
    # runtime progress (persisted so restarts keep partial acceptance)
    accept_progress: int = 0
    closed_through: bool = False  # RETEST_HOLD phase 1 done


@dataclass
class RearmPolicy:
    max_attempts: int = 2
    min_reset_bars: int = 3


@dataclass
class ConfirmationPlan:
    side_sign: int
    steps: list[ConfirmationStep] = field(default_factory=list)
    invalidation: float | None = None
    obstacle: float | None = None
    target: float | None = None
    rearm: RearmPolicy = field(default_factory=RearmPolicy)
    version: str = ENGINE_VERSION

    def mandatory_steps(self) -> list[ConfirmationStep]:
        return [s for s in self.steps if s.mandatory]

    def current_step(self) -> ConfirmationStep | None:
        for step in self.steps:
            if step.mandatory and not step.cleared:
                return step
        return None


@dataclass
class DevelopmentCandidate:
    symbol: str
    side: str
    setup_family: str
    session_date: str
    plan: ConfirmationPlan
    stage: Stage = Stage.DISCOVERED
    attempts: int = 0
    bars_since_fail: int = 0
    stage_entered_at: str = ""
    history: list[dict] = field(default_factory=list)

    @property
    def side_sign(self) -> int:
        return self.plan.side_sign

    @property
    def readiness(self) -> float:
        mandatory = self.plan.mandatory_steps()
        if not mandatory:
            return 0.0
        return sum(1 for s in mandatory if s.cleared) / len(mandatory)

    # ------------------------------------------------------------------
    # persistence: restart must never reset the progression (sec 1.2)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": ENGINE_VERSION,
            "symbol": self.symbol,
            "side": self.side,
            "setup_family": self.setup_family,
            "session_date": self.session_date,
            "stage": self.stage.value,
            "attempts": self.attempts,
            "bars_since_fail": self.bars_since_fail,
            "stage_entered_at": self.stage_entered_at,
            "history": list(self.history),
            "plan": {
                "side_sign": self.plan.side_sign,
                "invalidation": self.plan.invalidation,
                "obstacle": self.plan.obstacle,
                "target": self.plan.target,
                "version": self.plan.version,
                "rearm": {
                    "max_attempts": self.plan.rearm.max_attempts,
                    "min_reset_bars": self.plan.rearm.min_reset_bars,
                },
                "steps": [
                    {
                        "label": s.label,
                        "level": s.level,
                        "condition": s.condition.value,
                        "required_bars": s.required_bars,
                        "mandatory": s.mandatory,
                        "cleared": s.cleared,
                        "cleared_at": s.cleared_at,
                        "accept_progress": s.accept_progress,
                        "closed_through": s.closed_through,
                    }
                    for s in self.plan.steps
                ],
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DevelopmentCandidate":
        plan_raw = payload.get("plan") or {}
        plan = ConfirmationPlan(
            side_sign=int(plan_raw.get("side_sign", 1)),
            invalidation=plan_raw.get("invalidation"),
            obstacle=plan_raw.get("obstacle"),
            target=plan_raw.get("target"),
            version=str(plan_raw.get("version") or ENGINE_VERSION),
            rearm=RearmPolicy(
                max_attempts=int((plan_raw.get("rearm") or {}).get("max_attempts", 2)),
                min_reset_bars=int((plan_raw.get("rearm") or {}).get("min_reset_bars", 3)),
            ),
            steps=[
                ConfirmationStep(
                    label=str(s.get("label") or ""),
                    level=float(s.get("level")),
                    condition=Condition(s.get("condition", "CLOSE")),
                    required_bars=int(s.get("required_bars", 2)),
                    mandatory=bool(s.get("mandatory", True)),
                    cleared=bool(s.get("cleared")),
                    cleared_at=str(s.get("cleared_at") or ""),
                    accept_progress=int(s.get("accept_progress", 0)),
                    closed_through=bool(s.get("closed_through")),
                )
                for s in plan_raw.get("steps", [])
            ],
        )
        return cls(
            symbol=str(payload.get("symbol") or ""),
            side=str(payload.get("side") or "LONG"),
            setup_family=str(payload.get("setup_family") or ""),
            session_date=str(payload.get("session_date") or ""),
            plan=plan,
            stage=Stage(payload.get("stage", "DISCOVERED")),
            attempts=int(payload.get("attempts", 0)),
            bars_since_fail=int(payload.get("bars_since_fail", 0)),
            stage_entered_at=str(payload.get("stage_entered_at") or ""),
            history=list(payload.get("history") or []),
        )


@dataclass(frozen=True)
class GreatnessConfig:
    near_trigger_pct: float = 0.6   # % distance to current level -> NEAR_TRIGGER
    touch_tolerance_pct: float = 0.05
    version: str = ENGINE_VERSION


class GreatnessEngine:
    """Feed completed bars per candidate; emits typed transition events."""

    def __init__(self, config: GreatnessConfig | None = None) -> None:
        self.config = config or GreatnessConfig()

    # ------------------------------------------------------------------
    def on_bar(self, candidate: DevelopmentCandidate, bar: M5Bar) -> list[TransitionEvent]:
        if not bar.complete or candidate.stage in TERMINAL_STAGES:
            return []
        events: list[TransitionEvent] = []
        sign = candidate.side_sign

        # 1) hard invalidation first: aligned close beyond the kill level.
        invalidation = candidate.plan.invalidation
        if invalidation is not None and sign * (bar.close - invalidation) < 0:
            self._set_stage(candidate, Stage.INVALIDATED, bar)
            events.append(self._event(candidate, EventType.INVALIDATED, "", bar))
            return events

        # 2) failed candidates count reset bars toward a re-arm.
        if candidate.stage == Stage.FAILED:
            candidate.bars_since_fail += 1
            if (
                candidate.attempts < candidate.plan.rearm.max_attempts
                and candidate.bars_since_fail >= candidate.plan.rearm.min_reset_bars
            ):
                self._set_stage(candidate, Stage.REARMED, bar)
                events.append(self._event(candidate, EventType.REARMED, "", bar))
            else:
                return events

        step = candidate.plan.current_step()
        if step is None:
            return events  # already READY (or plan is empty)

        favorable_extreme = bar.high if sign > 0 else bar.low
        touched = sign * (favorable_extreme - step.level) >= -(
            self.config.touch_tolerance_pct / 100.0
        ) * step.level
        closed_beyond = sign * (bar.close - step.level) > 0
        wicked_beyond = sign * (favorable_extreme - step.level) > 0 and not closed_beyond

        # 3) proximity stages while the level has not been reached.
        if not touched:
            distance_pct = abs(step.level - bar.close) / step.level * 100.0 if step.level else 0.0
            near = distance_pct <= self.config.near_trigger_pct
            target_stage = Stage.NEAR_TRIGGER if near else Stage.DEVELOPING
            if candidate.stage in (Stage.DISCOVERED, Stage.DEVELOPING, Stage.NEAR_TRIGGER, Stage.REARMED):
                if candidate.stage != target_stage:
                    self._set_stage(candidate, target_stage, bar)
            return events

        # 4) the level is in play.
        if candidate.stage not in (Stage.TESTING_LEVEL, Stage.CONFIRMING):
            self._set_stage(candidate, Stage.TESTING_LEVEL, bar)
            events.append(self._event(candidate, EventType.LEVEL_TOUCHED, step.label, bar))

        if step.condition == Condition.TOUCH:
            events.extend(self._clear_step(candidate, step, bar))
            return events

        if step.condition == Condition.CLOSE:
            if closed_beyond:
                events.append(self._event(candidate, EventType.CLOSED_THROUGH, step.label, bar))
                events.extend(self._clear_step(candidate, step, bar))
            elif wicked_beyond:
                events.extend(self._fail_attempt(candidate, step, bar))
            return events

        if step.condition == Condition.ACCEPT:
            if closed_beyond:
                step.accept_progress += 1
                if candidate.stage != Stage.CONFIRMING:
                    self._set_stage(candidate, Stage.CONFIRMING, bar)
                    events.append(self._event(candidate, EventType.CLOSED_THROUGH, step.label, bar))
                if step.accept_progress >= max(1, step.required_bars):
                    events.append(self._event(candidate, EventType.ACCEPTED, step.label, bar))
                    events.extend(self._clear_step(candidate, step, bar))
            else:
                if step.accept_progress > 0:
                    events.extend(self._fail_attempt(candidate, step, bar))
                step.accept_progress = 0
                if wicked_beyond and step.accept_progress == 0 and candidate.stage == Stage.TESTING_LEVEL:
                    events.extend(self._fail_attempt(candidate, step, bar))
            return events

        if step.condition == Condition.RETEST_HOLD:
            if not step.closed_through:
                if closed_beyond:
                    step.closed_through = True
                    self._set_stage(candidate, Stage.CONFIRMING, bar)
                    events.append(self._event(candidate, EventType.CLOSED_THROUGH, step.label, bar))
                elif wicked_beyond:
                    events.extend(self._fail_attempt(candidate, step, bar))
                return events
            # phase 2: holding the retest
            adverse_extreme = bar.low if sign > 0 else bar.high
            revisited = sign * (adverse_extreme - step.level) <= 0
            if not closed_beyond:
                step.closed_through = False
                events.extend(self._fail_attempt(candidate, step, bar))
                return events
            if revisited and closed_beyond:
                events.append(self._event(candidate, EventType.RETEST_HELD, step.label, bar))
                events.extend(self._clear_step(candidate, step, bar))
            return events

        return events

    # ------------------------------------------------------------------
    def _clear_step(self, candidate: DevelopmentCandidate, step: ConfirmationStep, bar: M5Bar) -> list[TransitionEvent]:
        step.cleared = True
        step.cleared_at = bar.ts.isoformat(timespec="seconds")
        events = [self._event(candidate, EventType.STEP_CLEARED, step.label, bar)]
        if candidate.plan.current_step() is None:
            self._set_stage(candidate, Stage.READY, bar)
            events.append(self._event(candidate, EventType.READY, step.label, bar))
        else:
            self._set_stage(candidate, Stage.CONFIRMING, bar)
        return events

    def _fail_attempt(self, candidate: DevelopmentCandidate, step: ConfirmationStep, bar: M5Bar) -> list[TransitionEvent]:
        candidate.attempts += 1
        candidate.bars_since_fail = 0
        step.accept_progress = 0
        step.closed_through = False
        events = [self._event(candidate, EventType.WICK_THROUGH, step.label, bar)]
        self._set_stage(candidate, Stage.FAILED, bar)
        events.append(self._event(candidate, EventType.FAILED_ATTEMPT, step.label, bar))
        return events

    def _set_stage(self, candidate: DevelopmentCandidate, stage: Stage, bar: M5Bar) -> None:
        if candidate.stage == stage:
            return
        candidate.stage = stage
        candidate.stage_entered_at = bar.ts.isoformat(timespec="seconds")
        candidate.history.append({"stage": stage.value, "ts": candidate.stage_entered_at})

    def _event(self, candidate: DevelopmentCandidate, event: EventType, step_label: str, bar: M5Bar) -> TransitionEvent:
        return TransitionEvent(
            symbol=candidate.symbol,
            event=event,
            step_label=step_label,
            ts=bar.ts,
            price=bar.close,
            attempt=candidate.attempts,
            stage=candidate.stage,
        )


# ---------------------------------------------------------------------------
# compatibility adapter (sec 16.1): existing D1 armed-level rows -> plan
# ---------------------------------------------------------------------------

def candidate_from_d1_trigger_levels(
    symbol: str,
    side: str,
    armed_levels: list[dict],
    *,
    session_date: str,
    setup_family: str = "",
    invalidation: float | None = None,
    target: float | None = None,
    confirmation: Condition = Condition.CLOSE,
) -> DevelopmentCandidate | None:
    """Translate the existing `_build_d1_watchlist_trigger_levels` rows into
    an ordered ConfirmationPlan. Levels are ordered by aligned distance so the
    nearest requirement is confirmed first; duplicates collapse."""
    side_norm = str(side or "LONG").strip().upper()
    sign = -1 if side_norm == "SHORT" else 1
    steps: list[ConfirmationStep] = []
    seen: set[float] = set()
    rows = [r for r in armed_levels or [] if r.get("level") is not None]
    for row in sorted(rows, key=lambda r: sign * float(r["level"])):
        level = round(float(row["level"]), 4)
        if level in seen:
            continue
        seen.add(level)
        steps.append(
            ConfirmationStep(
                label=str(row.get("label") or row.get("alert_label") or f"L{len(steps) + 1}"),
                level=level,
                condition=confirmation,
            )
        )
    if not steps:
        return None
    return DevelopmentCandidate(
        symbol=str(symbol).strip().upper(),
        side=side_norm,
        setup_family=setup_family or str(rows[0].get("setup_family") or ""),
        session_date=session_date,
        plan=ConfirmationPlan(
            side_sign=sign,
            steps=steps,
            invalidation=invalidation,
            target=target,
        ),
        stage=Stage.DISCOVERED,
    )
