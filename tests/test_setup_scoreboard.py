"""R9.3: the setup scoreboard rebuilt from the stores that carry outcomes.

Every test here is on the pure functions and on synthetic frames. The real
inputs are a 200 MB CSV and a 30 MB one; the point of the module is that it
never needs either of them loaded whole, and the point of these tests is that
the three exclusion rules the plan names are enforced *before* anything is
ranked, not described in prose beside a number that ignored them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_scoreboard as sb  # noqa: E402


# ---------------------------------------------------------------------------
# identity parsing
# ---------------------------------------------------------------------------
def test_the_bounce_type_is_the_tail_of_the_event_id():
    assert (
        sb.bounce_type_from_event_id("AAPL_long_20260724_06_30_00_h1_blue_after_red")
        == "h1_blue_after_red"
    )


def test_a_multi_type_event_keeps_its_whole_dash_joined_key():
    assert (
        sb.bounce_type_from_event_id("NVDA_short_20260812_10_05_00_eod_vwap-impulse_retest_vwap_eod-vwap")
        == "eod_vwap-impulse_retest_vwap_eod-vwap"
    )


def test_a_class_share_symbol_does_not_shift_the_split():
    """Symbols use `-` for class shares, never `_`, so the six fixed parts hold."""
    assert (
        sb.bounce_type_from_event_id("BRK-B_long_20260801_07_00_00_regime_pause_rs")
        == "regime_pause_rs"
    )


@pytest.mark.parametrize("bad", ["", None, "NOPE", "A_b_c"])
def test_an_unparseable_identity_yields_no_type_rather_than_a_guess(bad):
    assert sb.bounce_type_from_event_id(bad) == ""


# ---------------------------------------------------------------------------
# the three exclusion rules
# ---------------------------------------------------------------------------
def test_the_risk_floor_catches_a_penny_stop():
    """The review measured regime_pause_rw at -1.82R all-time against -0.28R
    trimmed. Stops of a cent on a $100 name are what produced the difference."""
    frame = pd.DataFrame(
        {
            "entry_price": [100.0, 100.0, 100.0, 100.0],
            "risk_per_share": [0.01, 0.09, 0.10, 2.50],
        }
    )
    assert sb.risk_floor_mask(frame).tolist() == [True, True, False, False]


def test_an_unmeasurable_risk_is_excluded_not_assumed_fine():
    """Missing data is uncertainty, never confirmation (plan.md sec 5)."""
    frame = pd.DataFrame({"entry_price": [100.0, 0.0, None], "risk_per_share": [None, 1.0, 1.0]})
    assert sb.risk_floor_mask(frame).tolist() == [True, True, True]


def test_a_zero_close_r_with_a_defaulted_eod_close_is_excluded():
    """The 16.9% mass: `close_r == 0` only because `eod_close` was defaulted to
    the entry. Not a scratch - an outcome that was never obtained."""
    frame = pd.DataFrame(
        {
            "close_r": [0.0, 0.0, 0.0, 1.4],
            "eod_close": [50.0, 51.0, 50.0, 50.0],
            "entry_price": [50.0, 50.0, 50.0, 50.0],
            "bars_elapsed": [0, 4, 6, 9],
        }
    )
    assert sb.unsettled_close_mask(frame).tolist() == [True, False, True, False]
    # A genuine flat close (eod != entry but R rounds to 0) is NOT excluded.
    assert not bool(sb.unsettled_close_mask(frame).iloc[1])


def test_never_measured_is_the_subset_that_never_advanced_a_bar():
    frame = pd.DataFrame(
        {
            "close_r": [0.0, 0.0],
            "eod_close": [50.0, 50.0],
            "entry_price": [50.0, 50.0],
            "bars_elapsed": [0, 7],
        }
    )
    assert sb.never_measured_mask(frame).tolist() == [True, False]


# ---------------------------------------------------------------------------
# the statistics
# ---------------------------------------------------------------------------
def test_the_trimmed_mean_survives_one_absurd_row():
    """A single 45R row is the whole reason the plain mean is not quoted alone."""
    ordinary = pd.Series([0.1] * 20)
    with_outlier = pd.Series([0.1] * 19 + [45.0])
    assert sb.trimmed_mean(ordinary) == pytest.approx(0.1)
    assert with_outlier.mean() > 2.0
    assert sb.trimmed_mean(with_outlier) == pytest.approx(0.1)


def test_a_tiny_sample_still_returns_a_number_rather_than_crashing():
    assert sb.trimmed_mean(pd.Series([1.0])) == pytest.approx(1.0)
    assert sb.trimmed_mean(pd.Series([], dtype=float)) is None


def test_a_thin_cell_is_reported_but_never_ranked():
    frame = pd.DataFrame(
        {
            "bounce_type": ["fat"] * 40 + ["thin"] * 5,
            "close_r": [0.5] * 40 + [9.0] * 5,
            "stop_hit": [False] * 45,
        }
    )
    out = sb.summarise(frame, "bounce_type")
    by_cell = {row["cell"]: row for _, row in out.iterrows()}
    assert by_cell["fat"]["reportable"] is True
    assert by_cell["thin"]["reportable"] is False
    # The thin cell has by far the best R and still must not sort above the fat one.
    assert list(out["cell"]) == ["fat", "thin"]


def test_every_r_is_reported_three_ways_with_its_stop_out_rate():
    frame = pd.DataFrame(
        {
            "bounce_type": ["x"] * 40,
            "close_r": [1.0] * 20 + [-1.0] * 20,
            "stop_hit": [False] * 20 + [True] * 20,
        }
    )
    row = sb.summarise(frame, "bounce_type").iloc[0]
    for column in ("mean_r", "trimmed_mean_r", "median_r", "stop_out_rate", "p10_r", "p90_r"):
        assert column in row.index
    assert row["stop_out_rate"] == pytest.approx(50.0)
    assert row["n"] == 40


def test_the_baseline_control_is_excluded_from_the_ranking_it_scores():
    frame = pd.DataFrame(
        {
            "family": ["alpha"] * 40 + [sb.BASELINE_FAMILY] * 40,
            "side": ["long"] * 80,
            "net_r": [0.6] * 40 + [0.2] * 40,
            "entry_price": [100.0] * 80,
            "risk_per_share": [2.0] * 80,
        }
    )
    frame["below_risk_floor"] = sb.risk_floor_mask(frame)
    lift = sb.baseline_lift(frame)
    assert list(lift["cell"]) == ["alpha / long"], "a pipe would split the markdown column"
    assert lift.iloc[0]["baseline_trimmed_r"] == pytest.approx(0.2)
    assert lift.iloc[0]["lift_vs_baseline"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# the promise the report makes
# ---------------------------------------------------------------------------
def test_the_declared_window_is_the_only_forward_looking_output():
    window = sb.declared_window("2026-08-22")
    assert window["length_sessions"] >= 40
    assert set(window["must_span"]) == {"bullish", "bearish", "chop"}
    assert window["control"] == sb.BASELINE_FAMILY
    # The exclusions are fixed in advance, which is what makes the next window
    # gate-2 eligible rather than another post-hoc read.
    assert len(window["exclusions_fixed_in_advance"]) == 2
    assert "no promotion or demotion" in window["decision_rule"]


def test_the_module_promotes_nothing():
    """A scoreboard that could move a rung would need a frozen window first."""
    source = (SCRIPTS_DIR / "setup_scoreboard.py").read_text(encoding="utf-8")
    for forbidden in ("PROMOTED", "review_policy", "focus_service", "record_review_event"):
        assert forbidden not in source, forbidden


def test_the_module_never_reads_a_whole_csv():
    """Both inputs are large enough that a bare read_csv is a defect."""
    source = (SCRIPTS_DIR / "setup_scoreboard.py").read_text(encoding="utf-8")
    for call in source.split("pd.read_csv(")[1:]:
        head = call[:200]
        assert "chunksize" in head, head
        assert "usecols" in head, head
