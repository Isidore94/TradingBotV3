"""P1-4 / 4f - wider facets, one fixture each. Missing data is unknown, never a default."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutation_context as spc  # noqa: E402
import setup_permutations as sp  # noqa: E402

NAN = float("nan")


def _get(name, ctx=None, **row):
    base = {"side": "LONG", "setup_family": "avwap_band_bounce"}
    base.update(row)
    return sp.facets_for_row(base, ctx).get(name)


@pytest.mark.parametrize(("value", "expected"), [
    (0.4, "gap_below_1atr"), (-1.5, "gap_1_2atr"), (3.0, "gap_2_4atr"), (6.2, "gap_4atr_plus"), (NAN, "unknown"),
])
def test_earnings_gap_size(value, expected):
    assert _get("earnings_gap_size", post_earnings_gap_atr_multiple=value) == expected


@pytest.mark.parametrize(("value", "expected"), [
    (2.0, "off_52w_high_0_5pct"), (9.0, "off_52w_high_5_15pct"), (20.0, "off_52w_high_15_30pct"),
    (45.0, "off_52w_high_30pct_plus"), (None, "unknown"), (-1.0, "unknown"),
])
def test_pullback_from_the_52_week_high(value, expected):
    assert _get("pullback_52w", top_pattern_weekly_pullback_from_52w_high_pct=value) == expected


def test_htf_retest_age_needs_a_confirmed_retest():
    assert _get("htf_retest_age", htf_retest_confirmed="True", htf_retest_age_bars=1) == "htf_retest_0_2bars"
    assert _get("htf_retest_age", htf_retest_confirmed="True", htf_retest_age_bars=12.0) == "htf_retest_8_19bars"
    assert _get("htf_retest_age", htf_retest_confirmed="False", htf_retest_age_bars=1) == "unknown"
    assert _get("htf_retest_age", htf_retest_confirmed="True", htf_retest_age_bars="") == "unknown"


def test_industry_rs_consistency_is_unknown_when_it_was_not_computed():
    computed = {"industry_etf": "XLK", "rs_vs_industry": 1.2}
    assert _get("industry_rs_consistent", industry_rs_consistent="True", **computed) == "industry_rs_consistent"
    assert _get("industry_rs_consistent", industry_rs_consistent="False", **computed) == "industry_rs_mixed"
    # The writer stores False with no industry ETF or no rs_vs_industry: that is not "mixed".
    assert _get("industry_rs_consistent", industry_rs_consistent="False", industry_etf="",
                rs_vs_industry=1.2) == "unknown"
    assert _get("industry_rs_consistent", industry_rs_consistent="False", industry_etf="XLK",
                rs_vs_industry=NAN) == "unknown"


def test_entry_trigger_time_from_the_first_watch_fired_of_the_session():
    events = [
        {"action": "watch_fired", "trade_date": "2026-09-24", "ts": "2026-09-24T15:20:00-04:00",
         "symbol": "ABC", "side": "LONG", "detail": {"kind": "pullback", "trigger": "h1_ema15_bounce"}},
        {"action": "watch_fired", "trade_date": "2026-09-24", "ts": "2026-09-24T10:05:00-04:00",
         "symbol": "ABC", "side": "LONG", "detail": {"kind": "band_bounce"}},
        {"action": "watch_fired", "trade_date": "2026-09-24", "ts": "2026-09-24T12:30:00-04:00",
         "symbol": "XYZ", "side": "SHORT", "detail": {"kind": "hod_avwap"}},
        {"action": "watch_fired", "trade_date": "2026-09-24", "ts": "not a time", "symbol": "BAD", "side": "LONG"},
    ]
    times = spc.entry_trigger_checkpoints(events, "2026-09-24")
    assert times == {("ABC", "LONG"): "open", ("XYZ", "SHORT"): "midday"}
    context = spc.SessionContext(triggers=spc.entry_triggers(events, "2026-09-24"), trigger_times=times)
    assert _get("entry_trigger_time", context.ctx_for("ABC", "LONG")) == "trigger_open"
    assert _get("entry_trigger_time", context.ctx_for("XYZ", "SHORT")) == "trigger_midday"
    assert _get("entry_trigger_time", context.ctx_for("NONE", "LONG")) == "unknown"
    assert _get("entry_trigger_time", {"entry_trigger_checkpoint": "lunch"}) == "unknown"


def test_every_wider_facet_is_one_registered_function_off_the_short_label():
    for name in ("earnings_gap_size", "pullback_52w", "htf_retest_age", "industry_rs_consistent",
                 "entry_trigger_time"):
        assert name in sp.FACETS
        assert sp.FACETS[name].in_label is False
