"""TJ-12 item 4 - the same six lines re-cut over a list of sessions.

*"A `week(sessions) -> ReportCard` re-cut of the same lines over a list of
sessions, for TJ-5's week / four-week / month strip: same functions, longer
window, `n` everywhere, a window under its floor named and not ranked."*

And the packet's own test note: *"the week re-cut over five sessions equals the
sum of its days where a sum is meaningful and **re-computes the Wilson from
pooled counts, never averages rates**."*

The fixture is two sessions of UNEQUAL length on purpose - four clicks and six -
because that is the only shape where pooling and averaging disagree:

    pooled    (3 + 1) right of (4 + 6) = 4/10 = 0.400
    averaged  (0.750 + 0.1667) / 2            = 0.458
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tj12_support as fx  # noqa: E402


@pytest.fixture()
def week(tmp_path):
    import day_report_card

    fx.one_session_of_clicks(tmp_path, session=fx.SESSION, plan=fx.CLICK_PLAN)
    fx.one_session_of_clicks(
        tmp_path, session=fx.PRIOR_SESSION, plan=fx.PRIOR_CLICK_PLAN
    )
    days = [
        fx.day_inputs(tmp_path, session=fx.PRIOR_SESSION),
        fx.day_inputs(tmp_path, session=fx.SESSION),
    ]
    return day_report_card.week(days)


def _line(card, key):
    for line in card.lines:
        if line["key"] == key:
            return line
    raise AssertionError(f"no {key!r} line: {[l['key'] for l in card.lines]}")


def test_the_week_is_the_same_six_lines_in_the_same_order(week):
    import day_report_card

    assert tuple(line["key"] for line in week.lines) == tuple(day_report_card.LINE_KEYS)


def test_every_week_line_still_carries_its_n(week):
    for line in week.lines:
        assert isinstance(line["n"], int), line["key"]
        assert isinstance(line["measured"], int), line["key"]
        assert line["measured"] <= line["n"], line["key"]


def test_the_week_names_the_sessions_it_pooled(week):
    assert tuple(week.sessions) == (fx.PRIOR_SESSION, fx.SESSION)
    assert str(len(week.sessions)) in _line(week, "your_reads")["text"]


def test_the_week_pools_the_counts_it_can_sum(week, tmp_path):
    """`n` is the sum of the days; a day's own counts are never re-measured."""
    line = _line(week, "your_reads")
    assert line["n"] == fx.WEEK_READS_N
    assert line["right"] == fx.WEEK_READS_RIGHT

    did_well = _line(week, "did_well")
    assert did_well["n"] == 2 * 4, "four likes on each of the two days"
    assert did_well["measured"] == 2 * 3

    process = _line(week, "process")
    assert process["n"] == 2 * 4
    assert process["planned"] == 2 * 2


def test_the_week_recomputes_the_wilson_from_the_pooled_counts(week):
    """The ONE Wilson (z 1.96), within a named horizon only.

    The top-level line is coverage only.  This fixture's ten reads are all
    rest-of-day calls, so its named cell carries the pooled rate and bound.
    """
    from swing_headline import wilson_lower_bound

    line = _line(week, "your_reads")
    cell = line["horizons"]["rest_of_day"]

    assert "rate" not in line
    assert cell["rate"] == pytest.approx(0.4), "0.458 is the averaged rate"
    assert cell["rate_lb"] == pytest.approx(
        wilson_lower_bound(fx.WEEK_READS_RIGHT, fx.WEEK_READS_N)
    )


def test_the_day_line_carries_no_wilson_of_its_own(tmp_path):
    import day_report_card

    fx.one_session_of_clicks(tmp_path)
    day = day_report_card.build(fx.day_inputs(tmp_path))
    assert "rate_lb" not in _line(day, "your_reads")


def test_a_window_under_the_floor_is_named_and_ranks_nothing(week):
    """Ten same-horizon graded reads is under the floor; the cell says so."""
    import evidence_stats

    assert fx.WEEK_READS_N < evidence_stats.MIN_REPORTABLE_N
    line = _line(week, "your_reads")
    cell = line["horizons"]["rest_of_day"]
    assert cell["meets_floor"] is False
    assert "rest of day" in line["text"].lower(), line["text"]


def test_the_week_names_no_family_under_the_floor(week):
    for line in week.lines:
        assert fx.UNDER_FLOOR_FAMILY not in line["text"], line["key"]


def test_an_empty_week_says_so_rather_than_printing_zero():
    import day_report_card

    card = day_report_card.week([])
    assert tuple(line["key"] for line in card.lines) == tuple(day_report_card.LINE_KEYS)
    for line in card.lines:
        assert line["n"] == 0
        assert line["text"].strip()
        assert "%" not in line["text"], line["key"]
