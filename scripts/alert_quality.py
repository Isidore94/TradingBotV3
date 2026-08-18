"""Alert-quality measurement: what the desk DELIVERED, not what the trader did.

Phase 0 of ``docs/ALERT_CENTER_QUALITY_PACKET.md``.

``review_learning.py`` measures the trader - P(take | shown), taken-vs-passed,
blind spots. Nothing measures the alerting itself. The desk decides on every
alert whether to make noise (``alert_is_loud`` / ``alert_should_sound`` gate the
beep in ``AlertCenterPanel.add_alert``) and that decision is never recorded, so
"how many loud alerts did I get, and how many were the same name shouting
twice?" has no answer.

This module is the honest audit of that gap. It reads the existing review-event
store and reports, per metric in ``GUI_TRADE_DISCOVERY_LEARNING_PLAN.md``
sec 17, whether the data on disk can support it - computing the ones that can
be computed and printing ``Unknown`` (never ``0``) for the ones that cannot.

The distinction this module refuses to blur: a metric reading zero because the
desk was quiet, versus a metric reading zero because nothing was ever recorded.
The second is not a measurement, and reporting it as one would make the whole
scoreboard untrustworthy.

Read-only. It owns no file, writes no state, and cannot influence a detector, a
score, a ranking, an alert gate, or a sound decision. Import-light on purpose
(no Qt, no pandas), matching ``review_events`` / ``review_guidance``: the
offline job and the tests both drive it headless.

Usage::

    .venv/Scripts/python.exe scripts/alert_quality.py
    .venv/Scripts/python.exe scripts/alert_quality.py --days 30
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from alert_delivery_events import (
    DELIVERED,
    WATCH_DELIVERED,
    load_delivery_events,
    watch_identity,
)
from review_events import load_review_events

#: Below this many independent samples a rate is reported as its floor, not as
#: a number. Matches the n>=8 gate review_learning.py uses for blind spots and
#: leaks, so the two scoreboards agree on when a segment has earned an opinion.
MIN_SAMPLES = 8

#: The impression: an alert became the active visual review. Today this is the
#: only denominator the store can offer, and it covers the REVIEW QUEUE only -
#: not the feed. An alert that scrolled past in the feed and was never reviewed
#: leaves no row at all, which is precisely the blind spot Phase 1 closes.
IMPRESSION_ACTION = "shown"

#: Written by packet Phase 1 into the machine-local delivery store. Their
#: absence in a real store means capture is not running - which is a finding,
#: not a quiet desk, and the report says so in those words.
DELIVERY_ACTION = DELIVERED
WATCH_DELIVERY_ACTION = WATCH_DELIVERED
DELIVERY_ACTIONS = frozenset({DELIVERY_ACTION, WATCH_DELIVERY_ACTION})

#: Tier ordering for the escalation rule. Higher wins; an unrecognised or
#: absent tier ranks below every real one so a missing tier can never be read
#: as a promotion.
TIER_RANK = {"D": 1, "C": 2, "B": 3, "A": 4, "S": 5}

#: Declared latency bound for "visibly delivered" (sec 17 user-armed hit
#: delivery). The trader armed this exact condition and is waiting on it, so
#: the bar is tight. Stated as a constant rather than buried in a comparison
#: because the metric is meaningless without the bound quoted beside it.
LATENCY_BOUND_MS = 2000

#: Acting on an alert. "Take" deliberately includes arming a watch: committing
#: to wait for a condition is a decision about the alert, not a pass.
TAKE_ACTIONS = frozenset(
    {
        "add_focus",
        "arm_watch",
        "arm_level",
        "favorite",
        "toggle_d1_focus",
        "toggle_m5_focus",
    }
)

#: Explicitly declining. Distinct from "never resolved", which is neither.
PASS_ACTIONS = frozenset({"skip", "remove_today", "dislike"})

STATUS_COMPUTABLE = "COMPUTABLE"
STATUS_PARTIAL = "PARTIAL"
STATUS_BLOCKED = "BLOCKED"
STATUS_DEFERRED = "DEFERRED"

#: Why a blocked metric is blocked, stated once so the report and the packet
#: cannot drift apart.
BLOCKER_NO_DELIVERY = (
    "no delivery row is ever written; needs packet Phase 1 (delivered / "
    "watch_delivered) plus a typed alert_event_id"
)
BLOCKER_NEEDS_CAPTURE = (
    "requires delivery capture (packet Phase 1) to be running; reports Unknown "
    "rather than zero when the delivery store is empty"
)
BLOCKER_NO_READY = (
    "depends on the canonical Ready lifecycle and versioned target/stop, which "
    "this packet does not build (GUI learning plan Phase 3+)"
)


@dataclass(frozen=True)
class MetricSpec:
    """One sec 17 alert-quality metric and what it would take to trust it.

    ``outcome_definition_id`` exists because the sec 17 preamble forbids
    comparing numbers produced under different horizons or fill assumptions as
    though they were the same metric. Freezing the id here is what lets a later
    promotion argument cite an exact definition instead of a metric name.
    """

    key: str
    title: str
    definition: str
    outcome_definition_id: str
    status: str
    blocker: str = ""
    #: "rate" renders as a percentage; "per_session" and "ms" are absolute
    #: numbers. A per-session count rendered as a percentage would be nonsense,
    #: and silently wrong rather than obviously wrong.
    unit: str = "rate"


#: The sec 17 "Alert quality" list, audited against what the store holds.
METRIC_REGISTRY: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="alert_to_action",
        title="Alert-to-action conversion by action type",
        definition=(
            "Resolved impressions by action type divided by all impressions, "
            "keyed on (trade_date, symbol, side, surface)"
        ),
        outcome_definition_id="alert_to_action_v1",
        status=STATUS_PARTIAL,
        blocker=(
            "covers reviewed alerts only; an alert that never reached the "
            "review queue leaves no impression to divide by"
        ),
    ),
    MetricSpec(
        key="watch_conversion",
        title="User-armed watch conversion",
        definition="Armed watches that fired divided by all armed watches",
        outcome_definition_id="watch_conversion_v1",
        status=STATUS_COMPUTABLE,
    ),
    MetricSpec(
        key="loud_per_session",
        title="Loud alerts per session",
        definition="Loud deliveries divided by sessions",
        outcome_definition_id="loud_per_session_v1",
        status=STATUS_COMPUTABLE,
        blocker=BLOCKER_NEEDS_CAPTURE,
        unit="per_session",
    ),
    MetricSpec(
        key="duplicate_loud_rate",
        title="Duplicate loud rate",
        definition=(
            "Loud deliveries after the first for the same typed alert_event_id "
            "without a genuine escalation, divided by all loud deliveries"
        ),
        outcome_definition_id="duplicate_loud_rate_v1",
        status=STATUS_COMPUTABLE,
        blocker=BLOCKER_NEEDS_CAPTURE,
    ),
    MetricSpec(
        key="armed_hit_delivery",
        title="User-armed hit delivery rate and latency",
        definition=(
            "Distinct fired watch_id events visibly delivered within the "
            "declared latency bound divided by all distinct fired watches"
        ),
        outcome_definition_id="armed_hit_delivery_v1",
        status=STATUS_COMPUTABLE,
        blocker=BLOCKER_NEEDS_CAPTURE,
    ),
    MetricSpec(
        key="missed_winner_rate",
        title="Missed-winner rate among quiet/queued items",
        definition=(
            "Eligible matured opportunities not shown before their action "
            "window that later succeeded, divided by eligible not-shown"
        ),
        outcome_definition_id="missed_winner_rate_v1",
        status=STATUS_BLOCKED,
        blocker=(
            "the quiet cohort is never logged, so the denominator does not "
            "exist; needs packet Phase 1"
        ),
    ),
    MetricSpec(
        key="ready_precision",
        title="Ready precision, precision@1, precision@3",
        definition=(
            "Among independent matured attempts that entered canonical Ready, "
            "the fraction reaching the success condition before invalidation"
        ),
        outcome_definition_id="ready_precision_v1",
        status=STATUS_DEFERRED,
        blocker=BLOCKER_NO_READY,
    ),
    MetricSpec(
        key="remaining_expected_r",
        title="Remaining Expected R at alert",
        definition=(
            "Side-adjusted distance from alert price to versioned target "
            "divided by distance to versioned invalidation"
        ),
        outcome_definition_id="remaining_expected_r_v1",
        status=STATUS_DEFERRED,
        blocker=BLOCKER_NO_READY,
    ),
)


@dataclass
class CaptureCoverage:
    """What the event store actually holds, before any metric is attempted."""

    rows: int = 0
    sessions: int = 0
    first_trade_date: str = ""
    last_trade_date: str = ""
    action_counts: dict[str, int] = field(default_factory=dict)
    installations: int = 0
    machines: tuple[str, ...] = ()
    schemas: tuple[str, ...] = ()
    #: Counts from the SEPARATE machine-local delivery store, not from the
    #: review rows above - the two live in different storage classes.
    delivery_rows: int = 0
    delivery_sessions: int = 0
    has_delivery_capture: bool = False

    @property
    def impressions(self) -> int:
        return self.action_counts.get(IMPRESSION_ACTION, 0)


def _trade_date_of(row: dict[str, Any]) -> str:
    return str(row.get("trade_date") or "").strip()


def _action_of(row: dict[str, Any]) -> str:
    return str(row.get("action") or "").strip().lower()


def _episode_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Identity for one reviewed alert.

    Side and surface are part of the key on purpose. The existing scoreboard
    folds episodes by ``(trade_date, symbol)`` alone, which collapses a long
    and a short - or a Swing and an M5 thesis - for the same ticker into one
    sample; ``review_guidance`` names that fold as the reason queue ordering is
    still gated to FIFO. An audit that inherited the same fold could not see
    the problem it is meant to report, so this key keeps them apart and the
    report states the divergence rather than hiding it.
    """

    return (
        _trade_date_of(row),
        str(row.get("symbol") or "").strip().upper(),
        str(row.get("side") or "").strip().upper(),
        str(row.get("surface") or "").strip().lower(),
    )


def _within_days(row: dict[str, Any], cutoff: date | None) -> bool:
    if cutoff is None:
        return True
    text = _trade_date_of(row)
    if not text:
        return False
    try:
        return date.fromisoformat(text) >= cutoff
    except ValueError:
        return False


def filter_recent(
    rows: Iterable[dict[str, Any]],
    days: int | None,
    *,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Rows within ``days`` trailing sessions. ``None`` keeps everything.

    Rows with an unparseable trade_date are dropped when a window is active
    rather than being swept into it: a row that cannot prove it belongs in the
    window is not evidence for the window.
    """

    if days is None or days <= 0:
        return [row for row in rows if isinstance(row, dict)]
    anchor = today or date.today()
    cutoff = anchor - timedelta(days=days)
    return [row for row in rows if isinstance(row, dict) and _within_days(row, cutoff)]


def audit_capture(
    rows: Sequence[dict[str, Any]],
    delivery_rows: Sequence[dict[str, Any]] = (),
) -> CaptureCoverage:
    """Describe both stores: volume, span, writers, and what is missing."""

    coverage = CaptureCoverage()
    coverage.rows = len(rows)
    coverage.delivery_rows = len(delivery_rows)
    delivery_sessions, _, _ = _span(list(delivery_rows))
    coverage.delivery_sessions = delivery_sessions
    coverage.has_delivery_capture = bool(delivery_rows)
    if not rows:
        return coverage

    actions: Counter[str] = Counter()
    trade_dates: set[str] = set()
    installations: set[str] = set()
    machines: set[str] = set()
    schemas: set[str] = set()
    for row in rows:
        action = _action_of(row)
        if action:
            actions[action] += 1
        trade_date = _trade_date_of(row)
        if trade_date:
            trade_dates.add(trade_date)
        installation = str(row.get("installation_id") or "").strip()
        if installation:
            installations.add(installation)
        machine = str(row.get("machine") or "").strip()
        if machine:
            machines.add(machine)
        schema = str(row.get("schema") or "").strip()
        if schema:
            schemas.add(schema)

    coverage.action_counts = dict(sorted(actions.items()))
    coverage.sessions = len(trade_dates)
    coverage.installations = len(installations)
    coverage.machines = tuple(sorted(machines))
    coverage.schemas = tuple(sorted(schemas))
    if trade_dates:
        ordered = sorted(trade_dates)
        coverage.first_trade_date = ordered[0]
        coverage.last_trade_date = ordered[-1]
    return coverage


@dataclass
class MetricResult:
    """A computed metric, or an honest statement of why there is no number.

    ``value is None`` is not zero and must never be rendered as zero: it means
    the store cannot answer, either because the capture is missing or because
    the sample has not cleared the floor.
    """

    spec: MetricSpec
    value: float | None = None
    numerator: int = 0
    denominator: int = 0
    sessions: int = 0
    first_trade_date: str = ""
    last_trade_date: str = ""
    note: str = ""
    breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.value is not None

    def display_value(self) -> str:
        if self.value is None:
            return "Unknown"
        if self.spec.unit == "per_session":
            return f"{self.value:.1f} per session"
        if self.spec.unit == "ms":
            return f"{self.value:.0f} ms"
        return f"{self.value * 100:.1f}%"


def _span(rows: Sequence[dict[str, Any]]) -> tuple[int, str, str]:
    trade_dates = {d for d in (_trade_date_of(row) for row in rows) if d}
    if not trade_dates:
        return 0, "", ""
    ordered = sorted(trade_dates)
    return len(ordered), ordered[0], ordered[-1]


def compute_alert_to_action(rows: Sequence[dict[str, Any]]) -> MetricResult:
    """Resolved impressions by action type, over reviewed alerts only.

    An impression with no subsequent action is neither a take nor a pass: it is
    unresolved, and it is counted and reported as such. Folding unresolved
    impressions into "passed" would flatter every take rate on the desk.
    """

    spec = next(s for s in METRIC_REGISTRY if s.key == "alert_to_action")
    result = MetricResult(spec=spec)

    impressions: set[tuple[str, str, str, str]] = set()
    actions_by_episode: dict[tuple[str, str, str, str], set[str]] = {}
    relevant: list[dict[str, Any]] = []
    for row in rows:
        action = _action_of(row)
        if not action:
            continue
        key = _episode_key(row)
        if action == IMPRESSION_ACTION:
            impressions.add(key)
            relevant.append(row)
        elif action in TAKE_ACTIONS or action in PASS_ACTIONS:
            actions_by_episode.setdefault(key, set()).add(action)
            relevant.append(row)

    result.denominator = len(impressions)
    result.sessions, result.first_trade_date, result.last_trade_date = _span(relevant)

    breakdown: Counter[str] = Counter()
    taken = 0
    passed = 0
    for key in impressions:
        acted = actions_by_episode.get(key, set())
        if not acted:
            breakdown["unresolved"] += 1
            continue
        for action in acted:
            breakdown[action] += 1
        if acted & TAKE_ACTIONS:
            taken += 1
        elif acted & PASS_ACTIONS:
            passed += 1
    result.breakdown = dict(sorted(breakdown.items()))
    result.numerator = taken

    if result.denominator < MIN_SAMPLES:
        result.note = (
            f"{result.denominator} impressions; needs {MIN_SAMPLES} before a "
            "rate is reported"
        )
        return result

    result.value = taken / result.denominator
    result.note = (
        f"{taken} taken, {passed} explicitly passed, "
        f"{breakdown.get('unresolved', 0)} unresolved"
    )
    return result


def compute_watch_conversion(rows: Sequence[dict[str, Any]]) -> MetricResult:
    """Armed watches that fired, divided by armed watches that resolved.

    Watches still armed at read time are excluded from the denominator: an
    unresolved watch is not a failed one, and counting it as such would drive
    the conversion rate down every time the report is run mid-session.
    """

    spec = next(s for s in METRIC_REGISTRY if s.key == "watch_conversion")
    result = MetricResult(spec=spec)

    armed: set[tuple[str, str, str, str, str]] = set()
    fired: set[tuple[str, str, str, str, str]] = set()
    expired: set[tuple[str, str, str, str, str]] = set()
    relevant: list[dict[str, Any]] = []
    for row in rows:
        action = _action_of(row)
        if action not in {"arm_watch", "watch_fired", "watch_expired"}:
            continue
        detail = row.get("detail")
        kind = ""
        if isinstance(detail, dict):
            kind = str(detail.get("kind") or "").strip().lower()
        key = (*_episode_key(row), kind)
        relevant.append(row)
        if action == "arm_watch":
            armed.add(key)
        elif action == "watch_fired":
            fired.add(key)
        else:
            expired.add(key)

    resolved = armed & (fired | expired)
    result.denominator = len(resolved)
    result.numerator = len(armed & fired)
    result.sessions, result.first_trade_date, result.last_trade_date = _span(relevant)
    result.breakdown = {
        "armed": len(armed),
        "fired": len(armed & fired),
        "expired": len(armed & expired),
        "still_armed": len(armed - fired - expired),
    }

    if result.denominator < MIN_SAMPLES:
        result.note = (
            f"{result.denominator} resolved watches; needs {MIN_SAMPLES} "
            "before a rate is reported"
        )
        return result

    result.value = result.numerator / result.denominator
    result.note = f"{len(armed - fired - expired)} still armed, excluded"
    return result


# --- delivery-backed metrics (packet Phase 1 data) --------------------------


def _spec(key: str) -> MetricSpec:
    return next(spec for spec in METRIC_REGISTRY if spec.key == key)


def _is_loud(row: dict[str, Any]) -> bool:
    return bool(row.get("loud"))


def _tier_rank(row: dict[str, Any]) -> int:
    return TIER_RANK.get(str(row.get("tier") or "").strip().upper(), 0)


#: The three ways an armed condition fires. All are conditions the trader set
#: by hand and is waiting on, so all three belong in the armed-hit denominator;
#: counting only ``watch_fired`` would silently exclude every price level and
#: D1 event the trader armed.
FIRED_ACTIONS = frozenset({"watch_fired", "level_fired", "d1_event_fired"})


def _fired_kind(detail: dict[str, Any]) -> str:
    """The identity component for a fired armed condition.

    Chart and D1-event watches carry a ``kind``; a price level is identified by
    its direction and price instead. Must mirror what the panel builds when it
    records the delivery, or the two stores cannot be joined.
    """

    kind = str(detail.get("kind") or "").strip()
    if kind:
        return kind
    direction = str(detail.get("direction") or "").strip()
    level = detail.get("level")
    if direction or level is not None:
        return f"{direction}@{level}"
    return ""


def _delivery_rows(rows: Sequence[dict[str, Any]], action: str) -> list[dict[str, Any]]:
    picked = [row for row in rows if _action_of(row) == action]
    picked.sort(key=lambda item: str(item.get("ts") or ""))
    return picked


def is_escalation(row: dict[str, Any], state: dict[str, Any]) -> bool:
    """Did this repeat delivery earn its noise? (trader-confirmed rule)

    A repeat is an escalation, not a duplicate, when any of three things is
    true: the tier rose, a previously quiet alert became loud, or an armed
    condition fired on a name that was only queued before. The rule lives here
    rather than at the write site precisely so it can be revised against
    already-captured data.
    """

    if row.get("is_armed_fire"):
        return True
    if _tier_rank(row) > int(state.get("max_tier", 0)):
        return True
    if _is_loud(row) and not state.get("was_loud", False):
        return True
    return False


def compute_duplicate_loud_rate(delivery_rows: Sequence[dict[str, Any]]) -> MetricResult:
    """Loud repeats of one typed alert that did not escalate.

    Every delivery is walked, not only the loud ones: the quiet-to-loud arm of
    the escalation rule needs to know that a quiet delivery came first, and a
    loud-only pass would be blind to it and would misreport those as duplicates.
    """

    result = MetricResult(spec=_spec("duplicate_loud_rate"))
    rows = _delivery_rows(delivery_rows, DELIVERY_ACTION)
    result.sessions, result.first_trade_date, result.last_trade_date = _span(rows)

    state_by_id: dict[str, dict[str, Any]] = {}
    loud_total = 0
    duplicates = 0
    escalations = 0
    for row in rows:
        event_id = str(row.get("alert_event_id") or "").strip()
        if not event_id:
            continue
        state = state_by_id.get(event_id)
        loud = _is_loud(row)
        if loud:
            loud_total += 1
        if state is None:
            # First delivery of this thesis is never a duplicate.
            state_by_id[event_id] = {
                "max_tier": _tier_rank(row),
                "was_loud": loud,
            }
            continue
        if loud:
            if is_escalation(row, state):
                escalations += 1
            else:
                duplicates += 1
        state["max_tier"] = max(int(state["max_tier"]), _tier_rank(row))
        state["was_loud"] = state["was_loud"] or loud

    result.denominator = loud_total
    result.numerator = duplicates
    result.breakdown = {
        "loud_deliveries": loud_total,
        "duplicates": duplicates,
        "escalations": escalations,
        "distinct_alerts": len(state_by_id),
    }
    if loud_total < MIN_SAMPLES:
        result.note = (
            f"{loud_total} loud deliveries; needs {MIN_SAMPLES} before a rate "
            "is reported"
        )
        return result
    result.value = duplicates / loud_total
    result.note = f"{escalations} repeats counted as escalation, not duplicate"
    return result


def compute_loud_per_session(delivery_rows: Sequence[dict[str, Any]]) -> MetricResult:
    """How much noise a session actually makes.

    The floor is on sessions, not on alerts: one very loud day is not evidence
    about the desk's normal volume, however many rows it contributes.
    """

    result = MetricResult(spec=_spec("loud_per_session"))
    rows = _delivery_rows(delivery_rows, DELIVERY_ACTION)
    result.sessions, result.first_trade_date, result.last_trade_date = _span(rows)

    loud = [row for row in rows if _is_loud(row)]
    result.numerator = len(loud)
    result.denominator = result.sessions
    result.breakdown = {
        "loud": len(loud),
        "quiet": len(rows) - len(loud),
        "sounded": sum(1 for row in loud if row.get("sounded")),
    }
    if result.sessions < MIN_SAMPLES:
        result.note = (
            f"{result.sessions} sessions; needs {MIN_SAMPLES} before a "
            "per-session number is reported"
        )
        return result
    result.value = len(loud) / result.sessions
    muted = len(loud) - result.breakdown["sounded"]
    result.note = f"{muted} loud alerts made no sound (feed muted)"
    return result


def compute_armed_hit_delivery(
    delivery_rows: Sequence[dict[str, Any]],
    review_rows: Sequence[dict[str, Any]] = (),
) -> MetricResult:
    """Fired watches the trader could actually see, within the declared bound.

    The denominator comes from the review store's ``watch_fired`` rows and the
    numerator from the delivery store, so a watch that fired and was never
    delivered counts against the rate instead of vanishing from both sides. A
    delivery with no recorded latency is not assumed to be fast: it cannot
    prove it met the bound, so it does not count as a hit.
    """

    result = MetricResult(spec=_spec("armed_hit_delivery"))
    delivered = _delivery_rows(delivery_rows, WATCH_DELIVERY_ACTION)
    fired_ids: set[str] = set()
    for row in review_rows:
        if _action_of(row) not in FIRED_ACTIONS:
            continue
        detail = row.get("detail")
        detail = detail if isinstance(detail, dict) else {}
        explicit = str(detail.get("watch_id") or "").strip()
        fired_ids.add(
            explicit
            or watch_identity(
                _trade_date_of(row),
                str(row.get("symbol") or ""),
                str(row.get("side") or ""),
                _fired_kind(detail),
            )
        )
    fired_ids.discard("")

    delivered_by_id: dict[str, int | None] = {}
    for row in delivered:
        watch_id = str(row.get("watch_id") or "").strip()
        if not watch_id:
            continue
        latency = row.get("fired_to_delivered_ms")
        latency = int(latency) if isinstance(latency, (int, float)) else None
        previous = delivered_by_id.get(watch_id, "missing")
        if previous == "missing" or (
            latency is not None and (previous is None or latency < previous)
        ):
            delivered_by_id[watch_id] = latency

    # A watch delivered but never seen firing in the review store still counts:
    # the delivery is proof it fired, and dropping it would flatter the rate.
    universe = fired_ids | set(delivered_by_id)
    within = [
        watch_id
        for watch_id in universe
        if delivered_by_id.get(watch_id) is not None
        and delivered_by_id[watch_id] <= LATENCY_BOUND_MS
    ]
    latencies = [
        value for value in delivered_by_id.values() if isinstance(value, int)
    ]

    result.sessions, result.first_trade_date, result.last_trade_date = _span(delivered)
    result.denominator = len(universe)
    result.numerator = len(within)
    result.breakdown = {
        "fired": len(universe),
        "delivered": len(delivered_by_id),
        "within_bound": len(within),
        "no_latency_recorded": sum(
            1 for value in delivered_by_id.values() if value is None
        ),
        "median_ms": int(sorted(latencies)[len(latencies) // 2]) if latencies else 0,
    }
    if result.denominator < MIN_SAMPLES:
        result.note = (
            f"{result.denominator} fired watches; needs {MIN_SAMPLES} before a "
            "rate is reported"
        )
        return result
    result.value = len(within) / result.denominator
    result.note = f"bound {LATENCY_BOUND_MS} ms; {len(universe) - len(delivered_by_id)} never delivered"
    return result


def compute_metrics(
    rows: Sequence[dict[str, Any]],
    delivery_rows: Sequence[dict[str, Any]] = (),
) -> list[MetricResult]:
    """Every registry metric: computed where possible, Unknown where honest."""

    computed = {
        "alert_to_action": compute_alert_to_action(rows),
        "watch_conversion": compute_watch_conversion(rows),
        "loud_per_session": compute_loud_per_session(delivery_rows),
        "duplicate_loud_rate": compute_duplicate_loud_rate(delivery_rows),
        "armed_hit_delivery": compute_armed_hit_delivery(delivery_rows, rows),
    }
    results: list[MetricResult] = []
    for spec in METRIC_REGISTRY:
        if spec.key in computed:
            results.append(computed[spec.key])
        else:
            results.append(MetricResult(spec=spec, note=spec.blocker))
    return results


def build_report(
    coverage: CaptureCoverage,
    results: Sequence[MetricResult],
    *,
    days: int | None = None,
    now: datetime | None = None,
) -> str:
    """The human-facing audit. Every claim carries its sample and its span."""

    stamp = (now or datetime.now()).isoformat(timespec="seconds")
    window = f"last {days} days" if days else "all recorded sessions"
    lines: list[str] = [
        "ALERT QUALITY - CAPTURE AUDIT AND SCOREBOARD",
        f"generated {stamp}  |  window: {window}",
        "",
        "CAPTURE COVERAGE",
    ]

    if not coverage.rows:
        lines += [
            "  No review events in range.",
            "  This is an empty store, NOT a quiet desk - no metric below can",
            "  be computed, and none is reported as zero.",
        ]
    else:
        span = (
            f"{coverage.first_trade_date} .. {coverage.last_trade_date}"
            if coverage.first_trade_date
            else "unknown"
        )
        lines += [
            f"  rows            : {coverage.rows}",
            f"  sessions        : {coverage.sessions}",
            f"  date range      : {span}",
            f"  installations   : {coverage.installations}",
            f"  machines        : {', '.join(coverage.machines) or 'unknown'}",
            f"  schemas         : {', '.join(coverage.schemas) or 'unknown'}",
            f"  impressions     : {coverage.impressions}",
        ]
        if coverage.installations > 1:
            lines.append(
                "  WARNING: more than one installation wrote this store; "
                "episode counts are only trustworthy per writer."
            )
        lines.append("  actions:")
        for action, count in coverage.action_counts.items():
            lines.append(f"    {action:<18} {count}")

    lines += ["", "DELIVERY CAPTURE (machine-local store)"]
    if coverage.has_delivery_capture:
        lines += [
            "  present - Phase 1 capture is running.",
            f"  delivery rows   : {coverage.delivery_rows}",
            f"  sessions        : {coverage.delivery_sessions}",
        ]
    else:
        lines += [
            "  ABSENT. No 'delivered' or 'watch_delivered' row exists.",
            "  The desk's own alerting is therefore unmeasured: loud volume,",
            "  duplicate rate, armed-hit latency, and missed winners all have",
            "  no denominator. This is the finding the packet exists to fix,",
            "  not a transient gap.",
        ]

    lines += ["", "METRICS (GUI_TRADE_DISCOVERY_LEARNING_PLAN.md sec 17)"]
    for result in results:
        spec = result.spec
        lines.append(f"  [{spec.status:<10}] {spec.title}")
        lines.append(f"    definition   : {spec.definition}")
        lines.append(f"    outcome id   : {spec.outcome_definition_id}")
        if spec.status in {STATUS_BLOCKED, STATUS_DEFERRED}:
            lines.append("    value        : Unknown")
            lines.append(f"    blocked by   : {spec.blocker}")
        else:
            lines.append(f"    value        : {result.display_value()}")
            lines.append(
                f"    sample       : {result.numerator}/{result.denominator} "
                f"over {result.sessions} sessions"
            )
            if result.first_trade_date:
                lines.append(
                    f"    date range   : {result.first_trade_date} .. "
                    f"{result.last_trade_date}"
                )
            if result.breakdown:
                detail = ", ".join(
                    f"{name}={count}" for name, count in result.breakdown.items()
                )
                lines.append(f"    breakdown    : {detail}")
            if spec.blocker:
                lines.append(f"    caveat       : {spec.blocker}")
            if result.note:
                lines.append(f"    note         : {result.note}")
        lines.append("")

    lines += [
        "EPISODE IDENTITY",
        "  This audit keys episodes on (trade_date, symbol, side, surface).",
        "  review_learning.py folds on (trade_date, symbol) alone, so a long",
        "  and a short - or a Swing and an M5 thesis - collapse into one",
        "  sample there. Take rates here will not match that scoreboard, by",
        "  design; the fold is the reason queue ordering is still gated to",
        "  FIFO (review_guidance.py).",
    ]
    return "\n".join(lines)


def run_audit(
    *,
    days: int | None = None,
    path: Path | None = None,
    shards_dir: Path | None = None,
    delivery_dir: Path | None = None,
    rows: Sequence[dict[str, Any]] | None = None,
    delivery_rows: Sequence[dict[str, Any]] | None = None,
    today: date | None = None,
    now: datetime | None = None,
) -> tuple[CaptureCoverage, list[MetricResult], str]:
    """Load both stores, window, audit, compute, render.

    ``rows`` / ``delivery_rows`` bypass their stores for tests and for the
    offline job. The two are loaded independently because they are different
    storage classes: review decisions are Drive-synced, deliveries are
    machine-local.
    """

    if rows is None:
        kwargs: dict[str, Any] = {}
        if path is not None:
            kwargs["path"] = path
        if shards_dir is not None:
            kwargs["shards_dir"] = shards_dir
        loaded = load_review_events(**kwargs)
    else:
        loaded = list(rows)

    if delivery_rows is None:
        loaded_deliveries = load_delivery_events(delivery_dir)
    else:
        loaded_deliveries = list(delivery_rows)

    windowed = filter_recent(loaded, days, today=today)
    windowed_deliveries = filter_recent(loaded_deliveries, days, today=today)
    coverage = audit_capture(windowed, windowed_deliveries)
    results = compute_metrics(windowed, windowed_deliveries)
    report = build_report(coverage, results, days=days, now=now)
    return coverage, results, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit what the Alert Center's own alerting can be measured on."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only consider the trailing N days (default: every recorded session).",
    )
    args = parser.parse_args(argv)

    _, _, report = run_audit(days=args.days)
    print(report)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
