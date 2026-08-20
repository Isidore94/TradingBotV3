"""Where the next earnings candle is likely to land.

The earnings cache is HISTORY ONLY. Measured on the desk's own cache
(2026-08-20): 1,885 symbols, and **not one** carries a date in the future. So
"when is the next report" cannot be looked up - it has to be projected from the
cadence the symbol has actually kept, and it has to be labelled as a
projection wherever it is shown.

The cadence is the MEDIAN gap between consecutive reports, not the mean and not
the last gap. A company that reports quarterly still moves a report by a couple
of weeks now and then, and one 40-day or 140-day gap would drag a mean around;
the median ignores it. Measured across the desk's cache the median cadence is
91 days, which is the quarter it should be.

Gaps outside :data:`MIN_CADENCE_DAYS`..:data:`MAX_CADENCE_DAYS` are dropped
before the median is taken. Those are not cadence - they are a duplicated row,
a restatement, or a hole where the cache missed a report - and the point of
this module is to be honest about a number the trader will plan around.

Nothing here fetches, and nothing here is used by a detector, a score or an
alert. It answers a question for the chart.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping

#: Gaps outside this band are not a reporting cadence; see the module docstring.
MIN_CADENCE_DAYS = 40
MAX_CADENCE_DAYS = 200

#: Two gaps is a coincidence. Three is a cadence.
MIN_GAPS_FOR_CADENCE = 3

#: How far a projection may sit in the PAST and still be the answer. Earnings
#: dates drift by a week or two either way, so a projection that has only just
#: passed means "it is happening about now, and the cache has not caught the
#: row yet" - not "skip a quarter". Without this, NVDA on 2026-08-20 projected
#: 08/19, was rolled forward, and the chart reported November for a report
#: landing that week: the single most useful thing it could have said, lost.
OVERDUE_GRACE_DAYS = 10

# There is deliberately NO separate "too far out" cap. One was written and then
# removed as dead: a projection can only ever land at most one cadence past the
# last report, and MAX_CADENCE_DAYS already bounds a cadence at 200 days, so any
# such cap at or above that value can never fire. A cap that looks like a guard
# and never runs is worse than no cap. The honesty comes from the label instead
# - every consumer shows the date as an estimate.


def parse_dates(values: Iterable[Any]) -> list[date]:
    """ISO-ish strings (and dates) to a sorted, de-duplicated date list.

    Unparseable entries are dropped rather than raising: this reads a cache
    written by a fetcher, and one bad row must not cost the whole symbol.
    """
    parsed: set[date] = set()
    for value in values or ():
        if isinstance(value, datetime):
            parsed.add(value.date())
            continue
        if isinstance(value, date):
            parsed.add(value)
            continue
        text = str(value or "").strip()[:10]
        if not text:
            continue
        try:
            parsed.add(date.fromisoformat(text))
        except ValueError:
            continue
    return sorted(parsed)


def cadence_days(dates: Iterable[Any]) -> int | None:
    """Median days between consecutive reports, or None if unmeasurable."""
    parsed = parse_dates(dates)
    if len(parsed) < MIN_GAPS_FOR_CADENCE + 1:
        return None
    gaps = [
        (parsed[index + 1] - parsed[index]).days for index in range(len(parsed) - 1)
    ]
    gaps = [gap for gap in gaps if MIN_CADENCE_DAYS <= gap <= MAX_CADENCE_DAYS]
    if len(gaps) < MIN_GAPS_FOR_CADENCE:
        return None
    return int(round(statistics.median(gaps)))


def sessions_between(start: date, end: date) -> int:
    """Weekdays strictly after ``start`` up to and including ``end``.

    Weekdays, not exchange sessions: there is no exchange calendar in this
    repo (see ``chart_snapshot.latest_completed_session_date``, which carries
    the same caveat), so a holiday makes this over-count by one. That is a day
    of slack on a number already carrying weeks of cadence drift, and it is
    never presented as anything but an estimate.
    """
    if end <= start:
        return 0
    days = 0
    cursor = start
    while cursor < end:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            days += 1
    return days


def project_next_earnings(
    dates: Iterable[Any], *, today: date | None = None
) -> dict[str, Any] | None:
    """Project the next report from the symbol's own cadence.

    Returns ``{"date", "cadence_days", "sessions_ahead", "overdue"}``, or None
    when there is no measurable cadence. A projection is never withheld for
    being far out - it cannot land more than one cadence ahead, and the label
    carries the estimate caveat.

    ``overdue`` is True when the projection has already passed - the last
    known report is more than a full cadence old. That is NOT an error and is
    deliberately still returned: it means either the cache missed a report or
    one is imminent, and both are things the trader wants to see rather than
    have silently blanked. Measured on the desk's cache, 203 of 1,636 symbols
    with a usable cadence were in that state.
    """
    parsed = parse_dates(dates)
    if not parsed:
        return None
    cadence = cadence_days(parsed)
    if cadence is None:
        return None
    reference = today or date.today()
    projected = parsed[-1] + timedelta(days=cadence)
    # Roll forward only past a projection that is STALE - more than the grace
    # window behind us. One that just passed is kept and flagged overdue: see
    # OVERDUE_GRACE_DAYS for why rolling it would throw away the answer.
    floor = reference - timedelta(days=OVERDUE_GRACE_DAYS)
    while projected < floor:
        projected += timedelta(days=cadence)
    overdue = projected <= reference
    return {
        "date": projected,
        "cadence_days": cadence,
        "sessions_ahead": sessions_between(reference, projected),
        "overdue": overdue,
    }


def earnings_marks(
    dates: Iterable[Any],
    bars: Iterable[Mapping[str, Any]] | None = None,
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """The chart's earnings payload: which drawn bars are reports, and when next.

    ``indexes`` are positions into ``bars``, resolved HERE rather than on the
    paint path, so the chart never has to search its own bars to draw a marker.
    A report that falls on a day the chart does not hold (a weekend stamp, a
    gap in the store) simply has no index - it is not invented onto a
    neighbouring candle, which would put an E on a candle that is not the one
    that moved.
    """
    parsed = parse_dates(dates)
    by_day = {value: position for position, value in enumerate(_bar_dates(bars))}
    indexes = [by_day[value] for value in parsed if value in by_day]
    return {
        "indexes": indexes,
        "dates": [value.isoformat() for value in parsed],
        "projected": project_next_earnings(parsed, today=today),
    }


def _bar_dates(bars: Iterable[Mapping[str, Any]] | None) -> list[date | None]:
    resolved: list[date | None] = []
    for bar in bars or ():
        stamp = bar.get("dt") if hasattr(bar, "get") else None
        if isinstance(stamp, datetime):
            resolved.append(stamp.date())
        elif isinstance(stamp, date):
            resolved.append(stamp)
        else:
            resolved.append(None)
    return resolved
