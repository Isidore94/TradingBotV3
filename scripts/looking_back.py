"""Looking back (WISHLIST P2-9): pick equity curves and the hold-out window.

Display only. Nothing here is read by a detector, a score, an alert, a grade or
a ranking. Two populations, never pooled:

* **Swing picks** - one row per Setup Tracker episode (the tracker's own
  `selection_policy` default picks the episode), dated by its scan date, with the
  representative closed R the tracker's family rows average. Open picks count as
  pending and add nothing to the curve.
* **M5 alerts** - one row per alert, `setup_grades.bracket_results`: +1R when
  +1R came before -1R, -1R when the stop came first. Undecided and open alerts
  count apart and add nothing.

The curve is the running sum of those per-pick R values by session, with n.
The hold-out helpers split the same evidence by date: the last 20 sessions and
the window just before them.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Callable, Iterable, Mapping

SCHEMA = "looking_back_v1"

SWING = "swing"
M5 = "m5"
POPULATIONS = ((SWING, "Swing picks by day"), (M5, "M5 alerts by day"))

#: The M5 curve covers the recent window plus the prior one: two windows of
#: `evidence_stats.LATELY_SESSIONS`.
M5_CURVE_WINDOWS = 2

#: A swing pick whose tracker record has no representative scenario.
UNMEASURABLE = "unmeasurable"


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _day(value: Any) -> date | None:
    text = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


def split_windows(end: Any = None, *, sessions: int | None = None) -> dict[str, tuple[str, str]]:
    """`{"recent": (first, last), "prior": (first, last)}` on the exchange calendar.

    `recent` is `evidence_stats.lately_window(end)`; `prior` is the same number
    of sessions ending the session before `recent` starts.
    """
    import evidence_stats
    import market_calendar

    count = int(sessions or evidence_stats.LATELY_SESSIONS)
    recent = evidence_stats.lately_window(end, sessions=count)
    before = market_calendar.previous_session(date.fromisoformat(recent[0]))
    prior = evidence_stats.lately_window(before, sessions=count)
    return {"recent": tuple(recent), "prior": tuple(prior)}


def in_window(stamp: Any, window: tuple[str, str]) -> bool:
    text = str(stamp or "").strip()[:10]
    return bool(text) and window[0] <= text <= window[1]


def tracker_prior_reference(recent_reference: date, lookback_days: int) -> date:
    """The tracker's family rows cover ages 0..`lookback_days` calendar days.

    The prior window is the same span ending the day before the recent one
    starts, so the two never share a scan date.
    """
    return recent_reference - timedelta(days=int(lookback_days) + 1)


# ---------------------------------------------------------------------------
# per-pick results
# ---------------------------------------------------------------------------


def _default_context(setup: Mapping[str, Any]) -> tuple[str, str, str]:
    from master_avwap_lib import legacy

    context = legacy._tracker_setup_context(dict(setup))
    return (
        str(context.get("side") or ""),
        str(context.get("priority_bucket") or "").strip(),
        str(context.get("setup_family") or "general"),
    )


def swing_pick_results(
    setups: Mapping[str, Mapping[str, Any]] | None,
    *,
    reference: date | None = None,
    context: Callable[[Mapping[str, Any]], tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """One row per tracker episode: `{session, r, side, family, symbol}`.

    `setups` is the tracker's compact scoring snapshot (`setups` mapping), whose
    `_scoring_outcome_summary` is the record the family rows read. `r` is the
    representative closed R, or None while the representative is still open.
    """
    from master_avwap_lib import selection_policy

    context_of = context or _default_context
    rows: list[dict[str, Any]] = []
    for setup in (setups or {}).values():
        if not isinstance(setup, Mapping):
            continue
        scan_day = _day(setup.get("scan_date"))
        if scan_day is None or (reference is not None and scan_day > reference):
            continue
        summary = setup.get("_scoring_outcome_summary")
        if not isinstance(summary, Mapping):
            continue
        if int(summary.get("tradeable_scenario_count") or 0) <= 0:
            continue
        side, bucket, family = context_of(setup)
        if not side or not bucket:
            continue
        status = str(summary.get("representative_status") or "")
        closed_r = _float(summary.get("representative_closed_r"))
        rows.append(
            {
                "symbol": str(setup.get("symbol") or "").strip().upper(),
                "scan_date": scan_day.isoformat(),
                "anchor_date": str(setup.get("anchor_date") or "").strip(),
                "side": side,
                "priority_bucket": bucket,
                "setup_family": family,
                "closed_setups": 1 if int(summary.get("closed_tradeable_scenario_count") or 0) > 0 else 0,
                "representative_exit_date": str(summary.get("representative_exit_date") or ""),
                "_r": closed_r if status == "closed" else None,
                # No representative scenario: this pick can never be graded.
                "_status": status or UNMEASURABLE,
            }
        )
    chosen = selection_policy.select_episode_rows(rows)
    return [
        {
            "session": row["scan_date"],
            "r": row["_r"],
            "status": row["_status"],
            "side": row["side"],
            "family": row["setup_family"],
            "symbol": row["symbol"],
        }
        for row in chosen
    ]


def m5_alert_results(outcome_rows: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """One row per M5 alert: `{session, r, side, family}` with r = +1, -1 or None."""
    import setup_grades

    out = []
    for result in setup_grades.bracket_results(outcome_rows or ()):
        outcome = result.get("result")
        r = 1.0 if outcome == setup_grades.WIN else (-1.0 if outcome == setup_grades.LOSS else None)
        out.append(
            {
                "session": str(result.get("trade_date") or ""),
                "r": r,
                "side": str(result.get("side") or ""),
                "family": str(result.get("bounce_type") or ""),
            }
        )
    return out


# ---------------------------------------------------------------------------
# the curve
# ---------------------------------------------------------------------------


def equity_curve(results: Iterable[Mapping[str, Any]], *, population: str) -> dict[str, Any]:
    """Cumulative R by session, with n.

    A None R is "not graded yet", unless its status is `unmeasurable` (it never
    can be graded). Both are counted apart and add nothing to the curve.
    """
    by_day: dict[str, list[float]] = defaultdict(list)
    not_graded = 0
    unmeasurable = 0
    for row in results or ():
        session = str(row.get("session") or "").strip()[:10]
        r = _float(row.get("r"))
        if not session or r is None:
            if str(row.get("status") or "") == UNMEASURABLE:
                unmeasurable += 1
            else:
                not_graded += 1
            continue
        by_day[session].append(r)
    points = []
    cum_r = 0.0
    cum_n = 0
    for session in sorted(by_day):
        values = by_day[session]
        day_r = sum(values)
        cum_r += day_r
        cum_n += len(values)
        points.append(
            {
                "session": session,
                "day_r": round(day_r, 6),
                "day_n": len(values),
                "cum_r": round(cum_r, 6),
                "cum_n": cum_n,
            }
        )
    return {
        "population": population,
        "points": points,
        "n": cum_n,
        "total_r": round(cum_r, 6),
        "avg_r": round(cum_r / cum_n, 6) if cum_n else None,
        "first_session": points[0]["session"] if points else "",
        "last_session": points[-1]["session"] if points else "",
        "not_graded": not_graded,
        "unmeasurable": unmeasurable,
    }


def curve_line(curve: Mapping[str, Any] | None) -> str:
    """One sentence: total R, n, and since when."""
    if not curve or not curve.get("n"):
        return "no graded picks yet"
    avg = curve.get("avg_r")
    return (
        f"{float(curve['total_r']):+.2f}R over n={int(curve['n'])} "
        f"(avg {float(avg):+.2f}R), {curve['first_session']} to {curve['last_session']}; "
        f"{int(curve.get('not_graded') or 0)} not graded yet"
        + (
            f"; {int(curve['unmeasurable'])} unmeasurable"
            if int(curve.get("unmeasurable") or 0)
            else ""
        )
    )


def build_payload(
    *,
    swing_results: Iterable[Mapping[str, Any]] | None,
    m5_results: Iterable[Mapping[str, Any]] | None,
    holdout: Mapping[str, Any] | None = None,
    as_of: str = "",
) -> dict[str, Any]:
    """The one looking-back reading the Results page renders."""
    return {
        "schema": SCHEMA,
        "as_of": str(as_of or ""),
        "curves": {
            SWING: equity_curve(swing_results or (), population=SWING),
            M5: equity_curve(m5_results or (), population=M5),
        },
        "holdout": dict(holdout or {}),
    }
