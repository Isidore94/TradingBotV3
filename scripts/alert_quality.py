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

#: Written only once packet Phase 1 lands. Their absence is the finding.
DELIVERY_ACTION = "delivered"
WATCH_DELIVERY_ACTION = "watch_delivered"
DELIVERY_ACTIONS = frozenset({DELIVERY_ACTION, WATCH_DELIVERY_ACTION})

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
        status=STATUS_BLOCKED,
        blocker=BLOCKER_NO_DELIVERY,
    ),
    MetricSpec(
        key="duplicate_loud_rate",
        title="Duplicate loud rate",
        definition=(
            "Loud deliveries after the first for the same typed alert_event_id "
            "without a genuine escalation, divided by all loud deliveries"
        ),
        outcome_definition_id="duplicate_loud_rate_v1",
        status=STATUS_BLOCKED,
        blocker=BLOCKER_NO_DELIVERY,
    ),
    MetricSpec(
        key="armed_hit_delivery",
        title="User-armed hit delivery rate and latency",
        definition=(
            "Distinct fired watch_id events visibly delivered within the "
            "declared latency bound divided by all distinct fired watches"
        ),
        outcome_definition_id="armed_hit_delivery_v1",
        status=STATUS_BLOCKED,
        blocker=(
            "watch_fired records the fire, not the visible delivery or its "
            "timestamp; needs packet Phase 1 (watch_delivered)"
        ),
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


def audit_capture(rows: Sequence[dict[str, Any]]) -> CaptureCoverage:
    """Describe the store itself: volume, span, writers, and what is missing."""

    coverage = CaptureCoverage()
    coverage.rows = len(rows)
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
    coverage.has_delivery_capture = any(
        actions.get(action, 0) > 0 for action in DELIVERY_ACTIONS
    )
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


def compute_metrics(rows: Sequence[dict[str, Any]]) -> list[MetricResult]:
    """Every registry metric: computed where possible, blocked where honest."""

    computed = {
        "alert_to_action": compute_alert_to_action(rows),
        "watch_conversion": compute_watch_conversion(rows),
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
        "ALERT QUALITY - CAPTURE AUDIT (packet Phase 0)",
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

    lines += ["", "DELIVERY CAPTURE"]
    if coverage.has_delivery_capture:
        lines.append("  present - Phase 1 capture is running.")
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
    rows: Sequence[dict[str, Any]] | None = None,
    today: date | None = None,
    now: datetime | None = None,
) -> tuple[CaptureCoverage, list[MetricResult], str]:
    """Load, window, audit, compute, render. ``rows`` bypasses the store."""

    if rows is None:
        kwargs: dict[str, Any] = {}
        if path is not None:
            kwargs["path"] = path
        if shards_dir is not None:
            kwargs["shards_dir"] = shards_dir
        loaded = load_review_events(**kwargs)
    else:
        loaded = list(rows)

    windowed = filter_recent(loaded, days, today=today)
    coverage = audit_capture(windowed)
    results = compute_metrics(windowed)
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
