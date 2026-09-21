r"""TJ-5 change 3 - the report card re-cut by week and month. RED.

`plan.md` §12.4 TJ-5 change 3 (**AMENDED 2026-09-19, trader**): *"Under the five
cards, one deterministic strip of TJ-12's report-card lines re-cut by exchange
week for the last four weeks and for the calendar month to date (same functions,
longer window, `n` on every cell, a week under its floor named and not ranked).
No new page, no model."*

**NO MODEL IS EVER CALLED HERE, AND NO Qt WIDGET IS BUILT.** These are the
numbers, on their own.

The contract these tests pin (the builder may ADD keys, never remove one)
------------------------------------------------------------------------

``scripts/ui/services/weekend_prep_service.py``::

    WEEK_STRIP_WEEKS = 4

    week_strip(*, friday, root=None, ledger_path=None) -> {
        "weeks": (              # NEWEST FIRST, exactly WEEK_STRIP_WEEKS of them
            {"week_id", "sessions", "sessions_with_facts", "lines"}, ...),
        "month_to_date": {"month_id", "sessions", "sessions_with_facts", "lines"},
    }

Every ``lines`` value is a list of `day_report_card` line dicts keyed by
`day_report_card.LINE_KEYS`, each carrying ``n``, ``measured`` and - where the
line has a rate - ``rate``, ``rate_lb`` and ``meets_floor``, exactly as
`day_report_card._rate_keys` writes them. The strip computes no new statistic:
``rate_lb`` is the ONE Wilson (`swing_headline.wilson_lower_bound`, z 1.96).

THE ADVISORY THIS FILE CARRIES (TJ-12's review, 2026-09-20)
-----------------------------------------------------------
`day_report_card.how_fresh` filters the ledger tail on ``freshness["session"]``
and its guard is ``if session and ...``: called with an EMPTY session it pools
EVERY night the tail holds. A week strip that built its day inputs without a
session per night would multiply its own `n` by the number of days in the
window. `test_the_strip_counts_one_night_per_session` is that advisory, as a
number.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tj5_support as fx  # noqa: E402


def _lines(block):
    return {line["key"]: line for line in block["lines"]}


@pytest.fixture
def strip_root(tmp_path, monkeypatch):
    """A scratch `DAY_REVIEW_DIR` with the three packs the live store reaches."""
    import project_paths

    assert "TradingBotData" not in str(project_paths.DATA_DIR), project_paths.DATA_DIR
    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    fx.write_week(root)
    return root


# ---------------------------------------------------------------------------
# the fixture's own ground, checked against the calendar rather than trusted
# ---------------------------------------------------------------------------
def test_the_week_is_five_real_exchange_sessions():
    """2026-09-14 .. 2026-09-18, Monday to Friday, no holiday. GREEN GUARD:
    this passes today and exists so a later calendar change cannot move the
    fixture under the tests that count from it."""
    import evidence_stats
    import market_calendar

    assert evidence_stats.WEEK_SESSIONS == 5
    assert len(fx.WEEK) == 5
    for day in fx.WEEK:
        assert market_calendar.is_session(date.fromisoformat(day)), day
    assert fx.WEEK_ID == "2026-W38"


# ---------------------------------------------------------------------------
# the live gate: the week's pooled n is the sum of the day cards' n
# ---------------------------------------------------------------------------
def test_the_weeks_pooled_counts_are_the_sum_of_the_day_cards(strip_root):
    """Live gate carried from TJ-12's review.

    Hand-counted from `tj5_support.CARD_COUNTS`, over the THREE sessions that
    have packs::

        did_well   5+1+3 = 9 considered, 4+1+3 = 8 measured, 3+0+2 = 5 runs
        missed     4+7+2 = 13 rejected,  4+6+1 = 11 measured, 2+3+1 = 6 runs

    Each addend is also read back out of `day_report_card.build` below, so a
    fixture that drifted would fail here rather than quietly agree with itself.
    """
    import day_report_card

    from ui.services import weekend_prep_service as prep

    cards = [fx.card_for(session) for session in fx.PACKED_SESSIONS]
    by_day = [{line["key"]: line for line in card.lines} for card in cards]
    assert sum(day["did_well"]["n"] for day in by_day) == 9
    assert sum(day["did_well"]["measured"] for day in by_day) == 8
    assert sum(day["did_well"]["runs"] for day in by_day) == 5
    assert sum(day["missed"]["n"] for day in by_day) == 13
    assert sum(day["missed"]["measured"] for day in by_day) == 11
    assert sum(day["missed"]["runs"] for day in by_day) == 6

    strip = prep.week_strip(friday=fx.FRIDAY, root=strip_root)
    this_week = strip["weeks"][0]
    assert this_week["week_id"] == fx.WEEK_ID
    assert set(_lines(this_week)) == set(day_report_card.LINE_KEYS)

    did_well = _lines(this_week)["did_well"]
    assert (did_well["n"], did_well["measured"], did_well["runs"]) == (9, 8, 5)
    missed = _lines(this_week)["missed"]
    assert (missed["n"], missed["measured"], missed["runs"]) == (13, 11, 6)


def test_a_session_with_no_pack_is_named_and_is_never_a_zero_day(strip_root):
    """Two of this week's five sessions were never packed.

    Padding them with zeroes would make `did_well` 9 of 8 measured over FIVE
    days, which is a claim about two days nobody measured. The strip names them
    and counts three.
    """
    from ui.services import weekend_prep_service as prep

    this_week = prep.week_strip(friday=fx.FRIDAY, root=strip_root)["weeks"][0]
    assert tuple(this_week["sessions"]) == fx.WEEK
    assert tuple(this_week["sessions_with_facts"]) == fx.PACKED_SESSIONS
    # the honest headline: K of 5
    assert len(this_week["sessions_with_facts"]) == 3
    # and NOT the five-session totals (15 / 13 / 8 - see tj5_support's table)
    assert _lines(this_week)["did_well"]["n"] != 15


def test_the_strip_counts_one_night_per_session(tmp_path, monkeypatch):
    """The TJ-12 advisory, as a number.

    Ledger written by hand - distinct JOBS per session, which is what
    `day_report_card._slot_verdicts` counts::

        2026-09-16   (no rows)                                     0 jobs
        2026-09-17   day_review_narration ok, ticker_briefs failed  2 jobs
        2026-09-18   day_review_narration ok                        1 job
        -----------------------------------------------------------------
        the week                                                    3

    With an EMPTY `freshness["session"]` every card pools the WHOLE tail - two
    distinct jobs each - and the week would read 6. The `skipped` rows below
    are the ones the live ledger writes every half hour and they decide
    nothing.
    """
    import project_paths

    from ui.services import weekend_prep_service as prep
    from ai_jobs import ledger

    root = tmp_path / "day_review"
    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", root, raising=False)
    led = tmp_path / "ai_jobs.jsonl"
    fx.write_ledger(
        led,
        [
            fx.ledger_row("day_review_narration", "2026-09-17", ledger.STATUS_OK),
            fx.ledger_row("ticker_briefs", "2026-09-17", ledger.STATUS_FAILED),
            fx.ledger_row("ticker_briefs", "2026-09-17", ledger.STATUS_SKIPPED),
            fx.ledger_row("day_review_narration", "2026-09-18", ledger.STATUS_OK),
            fx.ledger_row("day_review_narration", "2026-09-18", ledger.STATUS_SKIPPED),
        ],
    )
    fx.write_week(root, ledger_path=led)

    this_week = prep.week_strip(friday=fx.FRIDAY, root=root, ledger_path=led)["weeks"][0]
    fresh = _lines(this_week)["how_fresh"]
    assert fresh["n"] == 3, fresh["text"]
    assert "ticker_briefs" in tuple(fresh.get("failed_slots") or ())


# ---------------------------------------------------------------------------
# the statistics contract
# ---------------------------------------------------------------------------
def test_a_week_under_the_floor_is_named_and_never_printed_as_a_rate(strip_root):
    """8 measured `did_well` names is under `MIN_REPORTABLE_N` (30).

    Decision 0016's clause: nothing under the floor is NAMED as a rate. The cell
    still carries its integers - `n` beside every number - and says it is too
    few to call.
    """
    import evidence_stats

    from ui.services import weekend_prep_service as prep

    did_well = _lines(prep.week_strip(friday=fx.FRIDAY, root=strip_root)["weeks"][0])["did_well"]
    assert did_well["measured"] == 8 < evidence_stats.MIN_REPORTABLE_N
    assert did_well["meets_floor"] is False
    assert "too few to call" in did_well["text"].lower()


def test_the_strips_only_wilson_is_the_one_wilson(strip_root):
    """`rate_lb` is `swing_headline.wilson_lower_bound` on the POOLED pair, and
    the strip computes no second statistic. A week that averaged two days' rates
    is a different number on days of unequal length."""
    import swing_headline

    from ui.services import weekend_prep_service as prep

    did_well = _lines(prep.week_strip(friday=fx.FRIDAY, root=strip_root)["weeks"][0])["did_well"]
    assert did_well["rate"] == pytest.approx(5 / 8)
    assert did_well["rate_lb"] == pytest.approx(swing_headline.wilson_lower_bound(5, 8))


# ---------------------------------------------------------------------------
# four weeks, and the month to date
# ---------------------------------------------------------------------------
def test_the_strip_names_four_weeks_newest_first_even_the_empty_ones(strip_root):
    """"the last four weeks" is FOUR entries, not "however many had facts".

    A week that simply vanished because nothing was packed would read as a week
    that never happened. Three of the four here hold nothing at all, and each
    is still named with its own `week_id` and its zero `sessions_with_facts`.
    """
    from ui.services import weekend_prep_service as prep

    strip = prep.week_strip(friday=fx.FRIDAY, root=strip_root)
    weeks = list(strip["weeks"])
    assert len(weeks) == prep.WEEK_STRIP_WEEKS == 4
    ids = [block["week_id"] for block in weeks]
    assert ids[0] == fx.WEEK_ID
    assert ids == sorted(ids, reverse=True), ids
    assert len(set(ids)) == 4
    for block in weeks[1:]:
        assert tuple(block["sessions_with_facts"]) == ()


def test_the_month_to_date_stops_at_the_friday_and_reaches_no_further_back(strip_root):
    """September 2026 to date: 2026-09-01 .. 2026-09-18 inclusive, and nothing
    in August. The three packed sessions are all inside it, so the month's
    `did_well` counts are the week's - 9 considered, 8 measured, 5 runs."""
    from ui.services import weekend_prep_service as prep

    month = prep.week_strip(friday=fx.FRIDAY, root=strip_root)["month_to_date"]
    assert month["month_id"] == "2026-09"
    sessions = tuple(month["sessions"])
    assert sessions[0].startswith("2026-09")
    assert sessions[-1] == fx.FRIDAY
    assert all(day <= fx.FRIDAY for day in sessions)
    assert not any(day.startswith("2026-08") for day in sessions)
    assert tuple(month["sessions_with_facts"]) == fx.PACKED_SESSIONS
    did_well = _lines(month)["did_well"]
    assert (did_well["n"], did_well["measured"], did_well["runs"]) == (9, 8, 5)
