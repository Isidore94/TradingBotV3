"""V3 items 1 and 3 - win rate leads, and "lately" is one number.

Decision 0016 answer 3: *"the trader gives swings room; losses run about 1.5x the
best wins, so **win rate is the number that matters**, not average R."*

Answer 6: *"'this market regime' needs no definition. 'Lately' is a rolling window
(about 20 sessions). No regime label."*

Two rules with one shape: a number the trader reads has to mean the same thing on
every screen, and it has to say what it rests on.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Item 3 - one constant, counted in trading sessions
# ---------------------------------------------------------------------------


def test_lately_is_twenty_and_lives_in_one_place():
    import evidence_stats

    assert evidence_stats.LATELY_SESSIONS == 20


def test_the_window_is_trading_sessions_never_calendar_days():
    """Twenty calendar days is fourteen sessions, and twelve across a holiday."""
    import evidence_stats

    first, last = evidence_stats.lately_window(date(2026, 9, 2))
    assert last == "2026-09-02"
    # 2026-08-06 .. 2026-09-02 inclusive is twenty NYSE sessions.
    assert first == "2026-08-06"

    from market_calendar import is_session, previous_session

    counted = 0
    cursor = date(2026, 9, 2)
    while cursor.isoformat() >= first:
        if is_session(cursor):
            counted += 1
        cursor = previous_session(cursor)
    assert counted == evidence_stats.LATELY_SESSIONS


def test_a_window_asked_for_on_a_weekend_ends_on_the_last_session():
    import evidence_stats

    first, _last = evidence_stats.lately_window(date(2026, 9, 6))  # a Sunday
    assert first == "2026-08-10"


def test_a_window_that_cannot_be_computed_errs_long_rather_than_short():
    """It can include an extra session; it must never silently drop one."""
    import evidence_stats

    stamp = evidence_stats.lately_start("not-a-date")
    assert isinstance(stamp, date)


def test_the_held_run_score_reads_the_one_constant_and_writes_no_second_literal():
    import evidence_stats
    import held_run_score

    assert held_run_score.ROLLING_SESSIONS is evidence_stats.LATELY_SESSIONS

    source = (ROOT / "scripts" / "held_run_score.py").read_text(encoding="utf-8")
    code = chr(10).join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    body = code.split('"""', 2)[-1]
    assert "= 20" not in body, "a second 'lately' literal is how two screens disagree"


# ---------------------------------------------------------------------------
# Item 1 - the headline, and what it rests on
# ---------------------------------------------------------------------------


def test_win_rate_is_the_first_column_and_mean_stays_beside_it():
    """Never replaced: the two answer different questions."""
    import swing_headline

    assert swing_headline.HEADLINE_COLUMNS[0] == "win_rate"
    assert "avg_r" in swing_headline.HEADLINE_COLUMNS
    assert "n" in swing_headline.HEADLINE_COLUMNS


def test_a_deep_family_outranks_a_lucky_thin_one():
    """A 100% on three and a 62% on ninety look the same to a skimming reader."""
    import swing_headline

    thin = swing_headline.headline_from_counts("thin", wins=3, losses=0)
    deep = swing_headline.headline_from_counts("deep", wins=56, losses=34)

    assert thin.win_rate == 1.0 and deep.win_rate is not None
    assert thin.win_rate > deep.win_rate, "the raw rate says the wrong thing"
    assert deep.win_rate_lb > thin.win_rate_lb, "the lower bound says the right one"
    assert [item.name for item in swing_headline.rank([thin, deep])] == ["deep", "thin"]


def test_the_lower_bound_says_something_at_zero_and_one():
    """The normal approximation returns exactly p there, which is no interval."""
    import swing_headline

    assert swing_headline.wilson_lower_bound(3, 3) < 1.0
    # Floating point, not a bug: the algebra cancels to zero and leaves ~5e-17.
    # Asserted as "at zero" rather than "== 0.0" so the test is about the bound
    # and not about IEEE 754.
    assert swing_headline.wilson_lower_bound(0, 3) < 1e-9
    assert swing_headline.wilson_lower_bound(0, 0) is None
    # And it widens as n shrinks.
    assert swing_headline.wilson_lower_bound(5, 10) < swing_headline.wilson_lower_bound(50, 100)


def test_the_floor_is_the_desks_one_contract_not_a_second_threshold():
    import evidence_stats
    import swing_headline

    at_floor = swing_headline.headline_from_counts(
        "x", wins=evidence_stats.MIN_REPORTABLE_N, losses=0
    )
    under = swing_headline.headline_from_counts(
        "y", wins=evidence_stats.MIN_REPORTABLE_N - 1, losses=0
    )
    assert at_floor.meets_floor is True
    assert under.meets_floor is False


def test_an_unreadable_row_is_counted_in_neither():
    """Unmeasured is not a loss; folding it in drifts every rate downward."""
    import swing_headline

    headline = swing_headline.headline_from_outcomes(
        "x",
        [{"close_r": 1.0}, {"close_r": None}, {"close_r": "n/a"}, {"close_r": -0.5}],
    )
    assert headline.n == 2
    assert headline.win_rate == 0.5


def test_the_tracker_verdict_is_read_and_never_re_derived():
    """Two definitions of a win is how two screens disagree."""
    import swing_headline

    headline = swing_headline.headline_from_tracker_rows(
        "family",
        [
            {"win": "1", "side_return_pct": "2.4"},
            {"win": "0", "side_return_pct": "-1.1"},
            {"win": "", "side_return_pct": "9.9"},
        ],
    )
    assert headline.n == 2, "the ungraded row is counted in neither"
    assert headline.win_rate == 0.5


def test_the_average_says_what_unit_it_is_in():
    """A column headed "Avg R" showing a percent is a number that lies."""
    import swing_headline

    tracker = swing_headline.headline_from_tracker_rows(
        "family", [{"win": "1", "side_return_pct": "2.4"}]
    )
    grid = swing_headline.headline_from_outcomes("grid", [{"close_r": 1.0}])

    assert tracker.avg_unit == "%"
    assert grid.avg_unit == "R"
    assert "%" in tracker.sentence() and "R" in grid.sentence()


def test_the_spelling_is_the_same_everywhere():
    import swing_headline

    row = swing_headline.headline_from_counts("x", wins=56, losses=34).as_row()
    assert swing_headline.format_win_rate(row) == "62% (>=52%, n=90)"
    assert swing_headline.format_win_rate({"win_rate": None}) == "-"


# ---------------------------------------------------------------------------
# The setup docs sentence
# ---------------------------------------------------------------------------


def test_a_setup_doc_states_its_record_and_never_hardcodes_it():
    """A number typed into a docstring quietly ages into a claim."""
    import setup_docs

    sentence = setup_docs.family_record_sentence(
        "avwap_breakout",
        tracker_rows=[
            {"win": "1", "side_return_pct": "2.4"},
            {"win": "0", "side_return_pct": "-1.1"},
            {"win": "1", "side_return_pct": "0.8"},
        ],
    )
    assert "67% win rate" in sentence
    assert "n=3" in sentence
    assert "discovery" in sentence, "it must say when it is under the floor"

    source = (ROOT / "scripts" / "setup_docs.py").read_text(encoding="utf-8")
    assert "win rate" not in source.split("def family_record_sentence")[0].lower(), (
        "a win rate written into the doc table is a number that cannot be refreshed"
    )


def test_a_family_with_nothing_graded_says_so_rather_than_zero():
    import setup_docs

    sentence = setup_docs.family_record_sentence("avwap_breakout", tracker_rows=[])
    assert "no graded swings" in sentence
    assert "0%" not in sentence


def test_the_doc_reader_never_raises_at_a_reader():
    """Reference material the trader opens mid-session."""
    import setup_docs

    assert setup_docs._read_family_outcomes("a_family_that_does_not_exist") == []
