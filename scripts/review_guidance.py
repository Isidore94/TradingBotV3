"""Per-alert guidance for the Alert Center: rank, annotate, hint. Never hide.

Phase 2 of the review-learning loop. Combines two documents:

- ``review_preference_state.json`` (review_learning.py) - the automatic
  scoreboard: P(take|shown) per segment, taken-vs-passed outcomes, blind
  spots / leaks.
- ``review_policy.json`` (review_policy.py) - the AI reviewer's curated
  directives: priority deltas, annotations, watch presets.

For each alert the panel asks ``ReviewGuide.guidance_for(alert)`` and gets an
``AlertGuidance``: an ordering score for the review queue, the take
probability behind it, callout notes ("Blind spot: ..."), and an optional
watch hint. The chart shows the notes; whether the queue may USE the score is
decided by the ordering mode below. Nothing here can suppress an alert - the
ceiling of this module's power is choosing what the trader sees FIRST, in
keeping with the house rule that muted means CAUTION, not silence.

Ordering gate (docs/archive/GUI_TRADE_DISCOVERY_LEARNING_PLAN.md sec 4.6 / 14, Phase 0
task 6): the scoreboard folds episodes by (trade_date, symbol), so a Swing
and an M5 thesis for the same ticker - or a long and a short - currently
collapse into one segment sample. Until the Phase 3 identity/parity gate
lands, preference-derived ordering runs in ANNOTATION-ONLY mode: the score is
still computed, shown on the chart, and stamped onto every impression, but
the active queue stays FIFO. ``ordering_mode="preference"`` restores the
champion-preference ordering without a code revert (also settable per process
via ``TRADINGBOT_REVIEW_QUEUE_ORDERING=preference``).

Cold start: with no state and no policy every alert scores 0 and the queue
stays FIFO - the desk behaves exactly as it did before this module existed.

Import-light on purpose (no Qt, no pandas): the panel calls it on every
enqueue, and tests drive it headless.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_paths import (  # noqa: E402
    REVIEW_POLICY_FILE,
    REVIEW_PREFERENCE_STATE_FILE,
)
from review_events import alert_context_fields  # noqa: E402
from review_learning import DIMENSIONS, Episode, _as_float  # noqa: E402
from review_policy import PolicyRule, load_review_policy  # noqa: E402

# A segment thinner than this in the scoreboard contributes nothing: four
# sightings is not a preference, it is an anecdote.
MIN_SEGMENT_SHOWN = 4
# Outcome averages need at least this many graded episodes to color a score.
MIN_OUTCOME_SAMPLES = 4
# Score composition weights: take-probability dominates, measured edge
# seasons it, and the AI's policy deltas can outrank either (a +/-5 delta
# swings 50 points, i.e. half the whole take-probability axis).
TAKE_PROB_WEIGHT = 100.0
EDGE_R_WEIGHT = 20.0
POLICY_DELTA_WEIGHT = 10.0

# Queue-ordering modes. Annotation-only is the Phase 0 champion-compatible
# default: guidance still annotates and stamps impressions, but contributes
# zero to queue position, so the active order is exactly the pre-guidance
# FIFO. "preference" is the pre-gate behavior, retained behind the switch.
ORDERING_ANNOTATION_ONLY = "annotation_only"
ORDERING_PREFERENCE = "preference"
DEFAULT_ORDERING_MODE = ORDERING_ANNOTATION_ONLY
ORDERING_MODE_ENV_VAR = "TRADINGBOT_REVIEW_QUEUE_ORDERING"


def resolve_ordering_mode(mode: str | None = None) -> str:
    """Explicit argument, then the env override, then the gated default.

    Anything unrecognized falls back to annotation-only: a typo in the switch
    must never silently re-enable ungated preference ordering.
    """
    for candidate in (mode, os.environ.get(ORDERING_MODE_ENV_VAR)):
        text = str(candidate or "").strip().lower()
        if text == ORDERING_PREFERENCE:
            return ORDERING_PREFERENCE
        if text == ORDERING_ANNOTATION_ONLY:
            return ORDERING_ANNOTATION_ONLY
    return DEFAULT_ORDERING_MODE


@dataclass
class AlertGuidance:
    """What the desk knows about an alert before the trader decides."""

    score: float = 0.0
    take_prob: float | None = None
    edge_r: float | None = None
    policy_delta: int = 0
    notes: list[str] = dataclass_field(default_factory=list)
    watch_kind: str = ""
    fill_source: str = ""
    segments: list[tuple[str, str]] = dataclass_field(default_factory=list)

    def summary_text(self) -> str:
        """One compact line for the review pane; empty when nothing to say."""
        parts = []
        if self.take_prob is not None:
            parts.append(f"take-prob {self.take_prob * 100:.0f}%")
        if self.edge_r is not None:
            parts.append(f"segment avg {self.edge_r:+.2f}R")
        if self.policy_delta:
            parts.append(f"AI priority {self.policy_delta:+d}")
        if self.watch_kind:
            hint = self.watch_kind
            if self.fill_source:
                hint += f" @ {self.fill_source}"
            parts.append(f"your usual arm: {hint}")
        line = " · ".join(parts)
        for note in self.notes:
            line = f"{line}\n{note}" if line else note
        return line


def alert_segments(fields: dict[str, Any], *, now: datetime | None = None) -> list[tuple[str, str]]:
    """The same (dimension, segment) keys the scoreboard aggregates on.

    Reuses review_learning's DIMENSIONS against a synthetic Episode so the
    lookup can never drift from the aggregation.
    """
    moment = now or datetime.now()
    episode = Episode(
        trade_date=moment.date().isoformat(),
        symbol=str(fields.get("symbol") or ""),
        side=str(fields.get("side") or ""),
        shown_ts=moment.isoformat(timespec="seconds"),
        tier=str(fields.get("tier") or ""),
        tag=str(fields.get("tag") or ""),
        timeframe=str(fields.get("timeframe") or ""),
        is_d1=bool(fields.get("is_d1")),
        bounce_types=str(fields.get("bounce_types") or ""),
        market_environment=str(fields.get("market_environment") or ""),
        session_rvol=_as_float(fields.get("session_rvol")),
        rrs_spy=_as_float(fields.get("rrs_spy")),
    )
    segments = []
    for dimension, key_fn in DIMENSIONS.items():
        try:
            for segment in key_fn(episode):
                segments.append((dimension, segment))
        except Exception:
            continue
    return segments


def _segment_stats(state: dict[str, Any], dimension: str, segment: str) -> dict | None:
    table = (state.get("dimensions") or {}).get(dimension) or {}
    stats = table.get(segment)
    return stats if isinstance(stats, dict) else None


def _callout_note(entry: dict[str, Any], *, blind: bool) -> str:
    label = "Blind spot" if blind else "Leak"
    prefix = "passed" if blind else "taken"
    if f"{prefix}_r_avg" in entry:
        measured = f"{prefix} avg {entry[f'{prefix}_r_avg']:+.2f}R (n={entry[f'{prefix}_r_n']})"
    else:
        measured = (
            f"{prefix} avg {entry.get(f'{prefix}_fwd_avg_pct', 0):+.1f}% "
            f"(n={entry.get(f'{prefix}_fwd_n', 0)})"
        )
    return (
        f"{label}: you take {entry.get('take_rate', 0) * 100:.0f}% of "
        f"{entry.get('segment')}; {measured}."
    )


def build_guidance(
    fields: dict[str, Any],
    state: dict[str, Any] | None,
    rules: list[PolicyRule] | None,
    *,
    now: datetime | None = None,
) -> AlertGuidance:
    guidance = AlertGuidance()
    guidance.segments = alert_segments(fields, now=now)
    matched = set(guidance.segments)

    if state:
        take_rates, edges = [], []
        for dimension, segment in guidance.segments:
            stats = _segment_stats(state, dimension, segment)
            if not stats or stats.get("shown", 0) < MIN_SEGMENT_SHOWN:
                continue
            rate = _as_float(stats.get("take_rate_shrunk"))
            if rate is not None:
                take_rates.append(rate)
            for bucket in ("taken", "passed"):
                outcome = stats.get(bucket) or {}
                if (outcome.get("r_n") or 0) >= MIN_OUTCOME_SAMPLES:
                    edge = _as_float(outcome.get("r_avg"))
                    if edge is not None:
                        edges.append(edge)
                    break  # taken preferred; passed only as fallback
        if take_rates:
            guidance.take_prob = round(sum(take_rates) / len(take_rates), 3)
        if edges:
            guidance.edge_r = round(sum(edges) / len(edges), 3)
        for entry in state.get("blind_spots") or []:
            if (str(entry.get("dimension")), str(entry.get("segment"))) in matched:
                guidance.notes.append(_callout_note(entry, blind=True))
        for entry in state.get("leaks") or []:
            if (str(entry.get("dimension")), str(entry.get("segment"))) in matched:
                guidance.notes.append(_callout_note(entry, blind=False))

    for rule in rules or []:
        if rule.key() not in matched:
            continue
        guidance.policy_delta += rule.priority_delta
        if rule.annotation:
            guidance.notes.append(rule.annotation)
        if rule.watch_kind and not guidance.watch_kind:
            guidance.watch_kind = rule.watch_kind
            guidance.fill_source = rule.fill_source

    guidance.score = round(
        (guidance.take_prob or 0.0) * TAKE_PROB_WEIGHT
        + (guidance.edge_r or 0.0) * EDGE_R_WEIGHT
        + guidance.policy_delta * POLICY_DELTA_WEIGHT,
        2,
    )
    return guidance


class ReviewGuide:
    """Cached loader + scorer for the panel. Paths of None disable it.

    Both documents are re-read when their mtime moves, so an overnight
    scoreboard rebuild or a freshly promoted policy takes effect on the next
    alert without a restart.
    """

    def __init__(
        self,
        state_path: Path | None = REVIEW_PREFERENCE_STATE_FILE,
        policy_path: Path | None = REVIEW_POLICY_FILE,
        *,
        ordering_mode: str | None = None,
    ) -> None:
        self._state_path = Path(state_path) if state_path is not None else None
        self._policy_path = Path(policy_path) if policy_path is not None else None
        self._state: dict[str, Any] | None = None
        self._state_mtime: float | None = None
        self._rules: list[PolicyRule] = []
        self._rules_mtime: float | None = None
        self.ordering_mode = resolve_ordering_mode(ordering_mode)

    @property
    def enabled(self) -> bool:
        return self._state_path is not None or self._policy_path is not None

    @property
    def orders_queue(self) -> bool:
        """True only when preference is allowed to move the active queue."""
        return self.ordering_mode == ORDERING_PREFERENCE

    def queue_score(self, guidance: AlertGuidance) -> float:
        """The score the review queue may sort on - 0.0 while gated.

        Callers must route every queue-ordering decision through here rather
        than reading ``guidance.score`` directly, so the gate cannot be
        bypassed by a new call site.
        """
        return float(guidance.score) if self.orders_queue else 0.0

    def _mtime(self, path: Path | None) -> float | None:
        try:
            return path.stat().st_mtime if path is not None and path.exists() else None
        except OSError:
            return None

    def _refresh(self) -> None:
        state_mtime = self._mtime(self._state_path)
        if state_mtime != self._state_mtime:
            self._state_mtime = state_mtime
            self._state = None
            if self._state_path is not None and state_mtime is not None:
                try:
                    payload = json.loads(self._state_path.read_text(encoding="utf-8"))
                    self._state = payload if isinstance(payload, dict) else None
                except (OSError, json.JSONDecodeError):
                    self._state = None
        rules_mtime = self._mtime(self._policy_path)
        if rules_mtime != self._rules_mtime:
            self._rules_mtime = rules_mtime
            self._rules = (
                load_review_policy(self._policy_path)
                if self._policy_path is not None and rules_mtime is not None
                else []
            )

    def guidance_for(self, alert, *, now: datetime | None = None) -> AlertGuidance:
        if not self.enabled:
            return AlertGuidance()
        try:
            self._refresh()
            fields = alert_context_fields(alert)
            # alert_context_fields carries the payload context; symbol and
            # side live directly on the alert.
            fields["symbol"] = getattr(alert, "symbol", "")
            fields["side"] = getattr(alert, "side", "")
            return build_guidance(fields, self._state, self._rules, now=now)
        except Exception:
            # Guidance is advisory; a corrupt document must never cost the
            # trader an alert or an exception dialog.
            return AlertGuidance()
