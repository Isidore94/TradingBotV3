"""R10.C - ground rule 10, implemented once.

Each test below pins one clause of the rule, and most of them exist because a
surface in this repo previously answered that clause differently or not at all:
the cohort rollup published a bare mean, a win rate and a profit factor with no
robust companion and no interval; the scoreboard published quantiles but no
concentration; nothing published an interval at all.

The properties that carry the most weight are the refusals - the places this
module declines to produce a number rather than producing a misleading one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evidence_stats as stats  # noqa: E402


def _sessions(n, count=4):
    return [f"2026-08-{10 + (index % count):02d}" for index in range(n)]


# ==========================================================================
# raw and robust, side by side
# ==========================================================================
def test_the_bare_mean_is_printed_but_never_alone():
    """It is the statistic that produced `regime_pause_rw`'s -1.82R. Hiding it
    would be its own dishonesty; publishing it alone is the defect."""
    values = [0.1] * 20 + [-40.0]
    summary = stats.summarize(values, sessions=_sessions(21))

    assert summary["raw"]["mean"] < -1.0
    assert summary["raw"]["trimmed_mean"] > 0
    assert summary["raw"]["median"] == pytest.approx(0.1)
    assert summary["raw"]["p10"] is not None and summary["raw"]["p90"] is not None


def test_uncapped_and_clipped_are_reported_together():
    """The 4R clip is what the ranking views already use. Showing both makes
    its effect visible instead of baked in."""
    values = [0.5, 1.0, 40.0]
    summary = stats.summarize(values, sessions=_sessions(3))

    assert summary["clip"] == 4.0
    assert summary["raw"]["mean"] > summary["clipped"]["mean"]
    assert summary["clipped"]["mean"] == pytest.approx((0.5 + 1.0 + 4.0) / 3, abs=1e-4)


def test_a_caller_can_turn_the_clip_off_and_the_summary_says_so():
    summary = stats.summarize([1.0, 2.0], sessions=_sessions(2), clip=None)
    assert summary["clip"] is None and summary["clipped"] is None


# ==========================================================================
# the refusals
# ==========================================================================
def test_a_cohort_with_no_losers_reports_no_profit_factor():
    """A PF with a zero denominator is a claim about a division nobody
    performed. A large finite number there is a lie."""
    summary = stats.summarize([0.4, 1.2, 0.9], sessions=_sessions(3))
    pf = summary["profit_factor"]

    assert pf["value"] is None
    assert pf["all_wins"] is True
    assert "never as a large finite number" in pf["convention"]


def test_a_cohort_with_no_winners_reports_zero_and_says_so():
    summary = stats.summarize([-0.4, -1.2], sessions=_sessions(2))
    pf = summary["profit_factor"]
    assert pf["value"] == 0.0 and pf["all_losses"] is True


def test_the_convention_travels_with_every_profit_factor():
    """Never bare: the reader has to be able to see which convention produced
    the number without opening this file."""
    assert stats.summarize([1.0, -1.0], sessions=_sessions(2))["profit_factor"]["convention"]


def test_one_session_gets_no_interval_and_names_the_reason():
    """An interval over one block describes one day as though it were a range."""
    boot = stats.summarize([0.2, 0.4, -0.1], sessions=["2026-08-21"] * 3)["bootstrap"]

    assert boot["measured"] is False
    assert "one day" in boot["reason"]
    assert boot["sessions"] == 1


def test_rows_without_session_identity_get_no_interval():
    boot = stats.summarize([0.2, 0.4], sessions=None)["bootstrap"]
    assert boot["measured"] is False
    assert "no session identity" in boot["reason"]


def test_the_interval_resamples_whole_sessions():
    """Trades inside one session share the tape, so they are not independent
    draws. Resampling them individually would report a precision the data does
    not have."""
    values = [1.0] * 10 + [-1.0] * 10
    sessions = ["2026-08-20"] * 10 + ["2026-08-21"] * 10
    boot = stats.summarize(values, sessions=sessions)["bootstrap"]

    assert boot["measured"] is True
    assert boot["sessions"] == 2
    # Whole blocks: every resample is all-of-one-day, all-of-the-other, or a
    # mixture, so the interval must reach both extremes.
    assert boot["low"] == pytest.approx(-1.0, abs=1e-6)
    assert boot["high"] == pytest.approx(1.0, abs=1e-6)


def test_two_runs_over_identical_inputs_agree():
    """A report that changes between runs cannot be checked by anyone. The
    bootstrap seeds from the data, not from a system RNG."""
    values = [0.3, -0.2, 1.1, -0.7, 0.4, 0.9]
    sessions = _sessions(6)
    first = stats.summarize(values, sessions=sessions)["bootstrap"]
    second = stats.summarize(values, sessions=sessions)["bootstrap"]
    assert first == second


def test_different_data_seeds_differently():
    """A hard-coded seed would make every cell resample in the same order
    regardless of content."""
    a = stats.summarize([0.3, -0.2, 1.1, 0.4], sessions=_sessions(4))["bootstrap"]
    b = stats.summarize([0.9, -0.8, 0.1, 0.2], sessions=_sessions(4))["bootstrap"]
    assert (a["low"], a["high"]) != (b["low"], b["high"])


# ==========================================================================
# counts, concentration, exclusions
# ==========================================================================
def test_concentration_shows_when_a_big_n_is_one_name():
    """n=200 from one symbol on one session has a sample size of roughly one,
    and only concentration can say so."""
    values = [0.1] * 20
    symbols = ["AAPL"] * 18 + ["MSFT", "NVDA"]
    summary = stats.summarize(values, symbols=symbols, sessions=_sessions(20))

    by_symbol = summary["concentration"]["by_symbol"]
    assert by_symbol["distinct"] == 3
    assert by_symbol["top"] == "AAPL"
    assert by_symbol["top_share"] == pytest.approx(0.9)


def test_excluded_and_unresolved_ride_beside_n_by_reason():
    """A number that quietly dropped 40% of its rows is not the number the
    reader thinks it is."""
    summary = stats.summarize(
        [0.1, 0.2],
        sessions=_sessions(2),
        excluded={"risk_below_floor": 7, "fabricated_zero": 3, "nothing": 0},
        unresolved={"no_eod_close": 5},
    )

    assert summary["excluded_total"] == 10
    assert summary["excluded_by_reason"] == {"risk_below_floor": 7, "fabricated_zero": 3}
    assert summary["unresolved_total"] == 5
    assert "risk_below_floor 7" in stats.format_note(summary)


def test_counts_separate_events_symbols_and_sessions():
    summary = stats.summarize(
        [0.1, 0.2, 0.3],
        symbols=["AAPL", "AAPL", "MSFT"],
        sessions=["2026-08-20", "2026-08-20", "2026-08-21"],
    )
    assert summary["counts"] == {"events": 3, "symbols": 2, "sessions": 2}


def test_unreadable_values_are_dropped_not_zeroed():
    """A blank is not a zero return. Coercing it would invent a flat trade."""
    summary = stats.summarize([0.5, "", None, "abc", float("nan"), 1.5])
    assert summary["n"] == 2
    assert summary["raw"]["mean"] == pytest.approx(1.0)


# ==========================================================================
# the label, and the floor
# ==========================================================================
def test_everything_post_hoc_is_discovery():
    summary = stats.summarize([0.1] * 500, sessions=_sessions(500, count=60))
    assert summary["evidence_label"] == stats.LABEL_DISCOVERY
    assert summary["meets_n_floor"] is True


def test_confirmation_requires_a_named_window_and_is_never_inferred_from_n():
    """A large post-hoc sample is a large discovery. Only a window declared in
    advance can make it a confirmation."""
    big = stats.summarize([0.1] * 5000, sessions=_sessions(5000, count=90))
    assert big["evidence_label"] == stats.LABEL_DISCOVERY

    declared = stats.summarize(
        [0.1] * 40,
        sessions=_sessions(40, count=40),
        confirmation_window="R9.3 40-session window declared 2026-08-21",
    )
    assert declared["evidence_label"] == stats.LABEL_CONFIRMATION
    assert "40-session" in declared["confirmation_window"]


def test_the_n_floor_is_necessary_never_sufficient():
    """Nothing here returns a 'reportable' flag off a count alone."""
    summary = stats.summarize([0.1] * 31, sessions=_sessions(31))
    assert summary["meets_n_floor"] is True
    assert "reportable" not in summary
    assert "NECESSARY, not sufficient" in summary["n_floor_note"]


def test_a_small_cell_says_it_is_below_the_floor():
    note = stats.format_note(stats.summarize([0.1, 0.2], sessions=_sessions(2)))
    assert "below the n>=30 floor" in note


def test_an_empty_sample_produces_nulls_not_zeros():
    summary = stats.summarize([])
    assert summary["n"] == 0
    assert summary["raw"]["mean"] is None
    assert summary["profit_factor"]["value"] is None
    assert summary["bootstrap"]["measured"] is False


def test_the_schema_is_named_never_numbered():
    assert stats.SUMMARY_SCHEMA == "evidence_summary_v1"
    assert stats.summarize([1.0])["schema"] == stats.SUMMARY_SCHEMA
