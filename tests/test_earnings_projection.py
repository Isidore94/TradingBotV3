"""Projecting the next earnings candle from a cache that holds no future dates.

The premise, measured on the desk's own cache (2026-08-20): 1,885 symbols and
not one future date. So "when does this name report next" cannot be looked up -
it is projected from the cadence the symbol kept, and everything downstream has
to present it as an estimate.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import earnings_projection as ep  # noqa: E402


def _quarterly(count: int, *, start=date(2025, 1, 30), step: int = 91) -> list[str]:
    from datetime import timedelta

    return [(start + timedelta(days=step * i)).isoformat() for i in range(count)]


# ---------------------------------------------------------------- parsing
def test_parse_drops_junk_instead_of_raising():
    """It reads a cache written by a fetcher; one bad row must not cost the
    whole symbol."""
    parsed = ep.parse_dates(
        ["2026-01-30", "", None, "not-a-date", "2026-04-30T00:00:00", "2026-01-30"]
    )
    assert parsed == [date(2026, 1, 30), date(2026, 4, 30)]


# ---------------------------------------------------------------- cadence
def test_cadence_is_the_median_quarter():
    assert ep.cadence_days(_quarterly(6)) == 91


def test_one_wild_gap_does_not_drag_the_cadence():
    """The reason it is a median and not a mean."""
    dates = ["2025-01-30", "2025-05-01", "2025-07-31", "2025-10-30", "2027-06-01"]
    assert ep.cadence_days(dates) == 91


def test_too_few_reports_is_unmeasurable_not_a_guess():
    assert ep.cadence_days(["2026-01-30", "2026-04-30"]) is None
    assert ep.cadence_days([]) is None


def test_gaps_outside_the_band_are_not_cadence():
    """Duplicated rows and multi-year holes are not a reporting rhythm."""
    dates = ["2026-01-30", "2026-01-31", "2026-02-01", "2026-02-02", "2026-02-03"]
    assert ep.cadence_days(dates) is None


# ------------------------------------------------------------- projection
def test_it_projects_one_cadence_past_the_last_report():
    # _quarterly(6) runs 2025-01-30 .. 2026-04-30 on a 91-day step.
    result = ep.project_next_earnings(_quarterly(6), today=date(2026, 5, 15))
    assert result is not None
    assert result["date"] == date(2026, 7, 30)
    assert result["cadence_days"] == 91
    assert result["overdue"] is False
    assert result["sessions_ahead"] > 0


def test_a_projection_that_just_passed_is_kept_and_flagged():
    """The NVDA case that made this rule.

    On 2026-08-20 NVDA's last known report was 2026-05-20 and the cadence 91
    days, projecting 08/19 - one day ago. Rolling that forward reported
    November for a report landing that week, throwing away the single most
    useful thing the chart could have said.
    """
    result = ep.project_next_earnings(
        ["2025-08-27", "2025-11-19", "2026-02-25", "2026-05-20"],
        today=date(2026, 8, 20),
    )
    assert result is not None
    # Gaps here are 84/98/84, so the cadence is 84 and the projection is 08/12
    # - eight days behind the reference and inside the grace window.
    assert result["date"] == date(2026, 8, 12)
    assert result["overdue"] is True
    assert result["date"] < date(2026, 9, 1), "must not skip a whole quarter"


def test_a_stale_projection_rolls_forward():
    """Past the grace window it IS a missed cycle, not an imminent one."""
    result = ep.project_next_earnings(_quarterly(6), today=date(2026, 8, 20))
    assert result is not None
    assert result["date"] > date(2026, 8, 20) - __import__("datetime").timedelta(
        days=ep.OVERDUE_GRACE_DAYS
    )


def test_an_unmeasurable_cadence_projects_nothing():
    assert ep.project_next_earnings(["2026-01-30"], today=date(2026, 2, 1)) is None
    assert ep.project_next_earnings([], today=date(2026, 2, 1)) is None


def test_a_semiannual_reporter_still_gets_a_projection():
    """There is no "too far out" cap, deliberately.

    One was written and removed as dead: a projection lands at most one cadence
    past the last report, and MAX_CADENCE_DAYS already bounds that at 200 days,
    so any cap at or above 200 could never fire. The estimate label carries the
    honesty instead.
    """
    dates = ["2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01"]
    result = ep.project_next_earnings(dates, today=date(2026, 7, 2))
    assert result is not None
    assert result["cadence_days"] == 181
    assert result["date"] == date(2026, 12, 29)
    assert not hasattr(ep, "MAX_PROJECTION_DAYS")


# ----------------------------------------------------------------- marks
def test_marks_align_to_the_bars_actually_drawn():
    from datetime import timedelta

    base = date(2026, 6, 1)
    bars = [{"dt": base + timedelta(days=i)} for i in range(10)]
    marks = ep.earnings_marks(
        [(base + timedelta(days=3)).isoformat(), (base + timedelta(days=7)).isoformat()],
        bars,
        today=date(2026, 6, 5),
    )
    assert marks["indexes"] == [3, 7]


def test_a_report_on_a_day_the_chart_does_not_hold_gets_no_marker():
    """Never invented onto a neighbouring candle - that would put an E on a
    candle that is not the one that moved."""
    bars = [{"dt": date(2026, 6, 1)}, {"dt": date(2026, 6, 2)}]
    marks = ep.earnings_marks(["2026-05-15"], bars, today=date(2026, 6, 3))
    assert marks["indexes"] == []
    assert "2026-05-15" in marks["dates"], "still known, just not drawable"


def test_marks_survive_bars_with_no_timestamps():
    marks = ep.earnings_marks(["2026-06-01"], [{"close": 1.0}, {}], today=date(2026, 6, 2))
    assert marks["indexes"] == []


def test_marks_with_no_bars_still_project():
    marks = ep.earnings_marks(_quarterly(6), None, today=date(2026, 3, 1))
    assert marks["indexes"] == []
    assert marks["projected"] is not None


# --------------------------------------------------------------- sessions
def test_sessions_between_counts_weekdays_only():
    # Fri 2026-06-05 -> Mon 2026-06-08 is one weekday.
    assert ep.sessions_between(date(2026, 6, 5), date(2026, 6, 8)) == 1
    assert ep.sessions_between(date(2026, 6, 8), date(2026, 6, 5)) == 0
    assert ep.sessions_between(date(2026, 6, 5), date(2026, 6, 5)) == 0
