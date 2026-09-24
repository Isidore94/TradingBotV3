"""P1-4 / 4a - `scripts/setup_permutations.py`, the pure permutation key.

Fixture rows use the real `d1_features_history.csv` column names and the value
shapes the live file holds (True/False and old 1.0/0.0 booleans, blanks as NaN).
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_permutations as sp  # noqa: E402

NAN = float("nan")


def _row(**overrides):
    row = {
        "run_date": "2026-09-24",
        "side": "LONG",
        "setup_family": "avwap_band_bounce",
        "last_close": 100.0,
        "atr20": 2.0,
        "current_anchor_date": "2026-08-06",
        "previous_anchor_nearest_level": "UPPER_1",
        "previous_anchor_nearest_distance_atr": 0.4,
        "current_band_zone": "VWAP to UPPER_1",
        "distance_from_current_vwap": 1.0,
        "distance_from_current_upper_1": -3.0,
        "distance_from_current_lower_1": 5.0,
        "trend_ma_alignment": "True",
        "ema21": 99.0,
        "top_pattern_weekly_ema15_hold": "True",
        "top_pattern_weekly_above_sma100": "False",
        "top_pattern_weekly_sma50_retest_recent": "0.0",
        "hv_level_nearby_count": 1,
        "hv_level_blocking_count": 0,
        "hv_level_nearest_bucket": "green",
        "hv_level_nearest_distance_atr": -0.3,
        "cloud_level_nearby_count": 0,
        "cloud_level_nearest_distance_atr": NAN,
        "previous_day_range_break": "False",
        "compression_flag": "False",
        "compression_break_today": "False",
        "compression_break_direction": NAN,
        "compression_break_recent": 0.0,
        "daily_relative_strength_score": 1.4,
        "rs_vs_industry_5d": 2.49,
        "industry_13w_return_pct": 22.9,
        "relvol": 1.7,
        "post_earnings_active": "False",
        "mid_earnings_watch": "True",
        "latest_release_sessions_since_gap": 33.0,
        "days_to_next_earnings": 41.0,
        "htf_trend_1h": "UP",
        "htf_trend_4h": "NEUTRAL",
        "htf_trend_aligned": "True",
        "htf_retest_confirmed": "True",
        "htf_retest_sma": "4h:SMA_20;1h:SMA_100",
        "htf_retest_timeframes": "1h;4h",
        "market_regime_label": "mixed",
        "spy_above_sma20": "True",
        "spy_above_sma50": "False",
        "spy_five_day_return_pct": 0.612,
        "side_aligned_day": NAN,
    }
    row.update(overrides)
    return row


def _all_mas(**dists):
    base = {f"dist_{ma}_atr": 3.0 for ma in sp.SUPPORT_MAS}
    base.update({f"dist_{ma}_atr": value for ma, value in dists.items()})
    return base


def test_full_row_gives_the_expected_facets():
    key = sp.facets_for_row(_row())
    got = key.as_dict()
    assert key.family == "avwap_band_bounce"
    assert key.side == "LONG"
    assert got["anchor_age"] == "anchor_31_60d"
    assert got["previous_anchor_level"] == "prev_anchor_upper_1_near"
    assert got["band_zone"] == "vwap_upper1"
    assert got["vwap_distance"] == "vwap_above_0to1atr"
    assert got["upper1_distance"] == "upper1_below_1to2atr"
    assert got["lower1_distance"] == "lower1_above_2atr"
    assert got["trend_ma_alignment"] == "ema15_sma20_aligned"
    assert got["price_vs_ema21"] == "above_ema21"
    assert got["weekly_ema15_hold"] == "weekly_ema15_hold"
    assert got["weekly_above_sma100"] == "weekly_below_sma100"
    assert got["weekly_sma50_retest"] == "no_weekly_sma50_retest"
    assert got["hv_level"] == "hv_level_green_below"
    assert got["cloud_level"] == "no_cloud_level"
    assert got["prev_day_range_break"] == "no_pdr_break"
    assert got["compression"] == "not_compressed"
    assert got["rs_vs_spy"] == "rs_spy_strong"
    assert got["rs_vs_industry_5d"] == "beats_industry_5d"
    assert got["industry_13w"] == "industry_13w_above_10"
    assert got["relvol"] == "relvol_1_5_3"
    assert got["earnings_phase"] == "mid_earnings"
    assert got["earnings_gap_age"] == "gap_21_40s"
    assert got["next_earnings"] == "earnings_31d_plus"
    assert got["htf_trend"] == "h1_up_h4_neutral"
    assert got["htf_aligned"] == "htf_aligned"
    assert got["htf_retest"] == "htf_retest_1h_sma100+4h_sma20"
    assert got["market_regime"] == "regime_mixed"
    assert got["spy_trend"] == "spy_above_sma20_below_sma50"
    assert got["spy_5d"] == "spy_5d_0_2"
    assert got["weekday"] == "thu"


def test_the_version_string_rides_on_every_key():
    key = sp.facets_for_row(_row())
    assert sp.PERMUTATION_RULE_VERSION == "setup_permutations.v1"
    assert key.permutation_rule_version == sp.PERMUTATION_RULE_VERSION
    assert key.key.startswith("setup_permutations.v1|avwap_band_bounce|LONG|")


def test_every_registered_facet_is_on_the_key_in_order():
    key = sp.facets_for_row(_row())
    assert [name for name, _ in key.facets] == list(sp.FACETS)


def test_empty_row_is_unknown_everywhere_never_a_default():
    key = sp.facets_for_row({})
    assert key.family == sp.UNKNOWN
    assert key.side == sp.UNKNOWN
    assert set(key.as_dict().values()) == {sp.UNKNOWN}
    assert key.label == ""


@pytest.mark.parametrize("blank", [None, NAN, "", "nan", "  "])
def test_blank_and_nan_inputs_are_unknown(blank):
    row = {column: blank for column in _row()}
    key = sp.facets_for_row(row, {"d1_environment": blank, "discovery_slot": blank,
                                  "entry_trigger": blank, "m5_bounce_type": blank})
    assert set(key.as_dict().values()) == {sp.UNKNOWN}


@pytest.mark.parametrize(
    ("column", "facet_name"),
    [
        ("current_band_zone", "band_zone"),
        ("trend_ma_alignment", "trend_ma_alignment"),
        ("top_pattern_weekly_ema15_hold", "weekly_ema15_hold"),
        ("hv_level_nearby_count", "hv_level"),
        ("previous_day_range_break", "prev_day_range_break"),
        ("compression_flag", "compression"),
        ("relvol", "relvol"),
        ("industry_13w_return_pct", "industry_13w"),
        ("days_to_next_earnings", "next_earnings"),
        ("htf_trend_4h", "htf_trend"),
        ("htf_retest_confirmed", "htf_retest"),
        ("spy_above_sma50", "spy_trend"),
        ("spy_five_day_return_pct", "spy_5d"),
        ("run_date", "weekday"),
        ("atr20", "vwap_distance"),
    ],
)
def test_one_missing_input_makes_only_its_facet_unknown(column, facet_name):
    full = sp.facets_for_row(_row()).as_dict()
    missing = sp.facets_for_row(_row(**{column: NAN})).as_dict()
    assert full[facet_name] != sp.UNKNOWN
    assert missing[facet_name] == sp.UNKNOWN


def test_band_zone_is_side_independent():
    long_key = sp.facets_for_row(_row(side="LONG", current_band_zone="LOWER_2 to LOWER_1"))
    short_key = sp.facets_for_row(_row(side="SHORT", current_band_zone="LOWER_1 to LOWER_2"))
    assert long_key.get("band_zone") == short_key.get("band_zone") == "lower2_lower1"
    assert sp.facets_for_row(_row(current_band_zone="UPPER_3")).get("band_zone") == "at_upper3"
    assert sp.facets_for_row(_row(current_band_zone="SOMEWHERE")).get("band_zone") == sp.UNKNOWN


def test_old_one_zero_booleans_parse_like_true_false():
    assert sp.facets_for_row(_row(trend_ma_alignment=1.0)).get("trend_ma_alignment") == "ema15_sma20_aligned"
    assert sp.facets_for_row(_row(trend_ma_alignment="0.0")).get("trend_ma_alignment") == "ema15_sma20_not_aligned"
    assert sp.facets_for_row(_row(trend_ma_alignment="maybe")).get("trend_ma_alignment") == sp.UNKNOWN


def test_ma_support_long_takes_the_ma_below_price():
    # dist = (close - ma) / ATR: +0.5 is below price (long support), -0.5 above.
    row = _row(side="LONG", **_all_mas(sma100=0.5, sma50=-0.5))
    assert sp.facets_for_row(row).get("ma_support") == "sma100_support"


def test_ma_support_short_takes_the_ma_above_price():
    row = _row(side="SHORT", **_all_mas(sma100=0.5, sma50=-0.5))
    assert sp.facets_for_row(row).get("ma_support") == "sma50_support"


def test_ma_support_none_and_multiple():
    assert sp.facets_for_row(_row(**_all_mas())).get("ma_support") == "no_ma_support"
    both = _row(side="LONG", **_all_mas(sma20=0.2, ema8=0.9))
    assert sp.facets_for_row(both).get("ma_support") == "multiple_ma_support"
    beyond = _row(side="LONG", **_all_mas(sma200=1.01))
    assert sp.facets_for_row(beyond).get("ma_support") == "no_ma_support"


def test_ma_support_is_unknown_when_any_ma_is_missing():
    # Today's scan row has no dist_<ma>_atr columns: unknown, never "none".
    assert sp.facets_for_row(_row()).get("ma_support") == sp.UNKNOWN
    partial = _all_mas(sma100=0.5)
    partial["dist_sma200_atr"] = NAN
    assert sp.facets_for_row(_row(**partial)).get("ma_support") == sp.UNKNOWN


def test_ma_support_derives_ema21_from_the_row():
    mas = _all_mas(sma20=3.0)
    del mas["dist_ema21_atr"]
    # ema21 = 99, close 100, ATR 2 -> +0.5 ATR, below price: long support.
    assert sp.facets_for_row(_row(side="LONG", **mas)).get("ma_support") == "ema21_support"
    assert sp.facets_for_row(_row(side="SHORT", **mas)).get("ma_support") == "no_ma_support"


def test_ma_order_needs_sma50_and_sma200():
    assert sp.facets_for_row(_row()).get("ma_order") == sp.UNKNOWN
    row = _row(dist_sma50_atr=2.0, dist_sma200_atr=-1.0)
    # ema21 +0.5 (below price), sma50 +2 (further below), sma200 -1 (above price).
    assert sp.facets_for_row(row).get("ma_order") == "sma200>price>ema21>sma50"


def test_levels_sign_and_side():
    above = sp.facets_for_row(_row(hv_level_nearest_bucket="red", hv_level_nearest_distance_atr=0.2))
    assert above.get("hv_level") == "hv_level_red_above"
    cloud = sp.facets_for_row(_row(cloud_level_nearby_count=1, cloud_level_nearest_distance_atr=-0.1))
    assert cloud.get("cloud_level") == "cloud_level_below"
    none = sp.facets_for_row(_row(hv_level_nearby_count=0, hv_level_blocking_count=0))
    assert none.get("hv_level") == "no_hv_level"


def test_compression_states():
    assert sp.facets_for_row(_row(compression_break_today="True", compression_break_direction="down")).get(
        "compression") == "compression_break_down"
    assert sp.facets_for_row(_row(compression_break_today="True")).get("compression") == sp.UNKNOWN
    assert sp.facets_for_row(_row(compression_break_recent=1.0)).get("compression") == "compression_break_recent"
    assert sp.facets_for_row(_row(compression_flag="True")).get("compression") == "compressed"
    assert sp.facets_for_row(_row(compression_break_recent=NAN)).get("compression") == sp.UNKNOWN


def test_earnings_phase_needs_both_flags_to_say_none():
    assert sp.facets_for_row(_row(post_earnings_active="True")).get("earnings_phase") == "post_earnings"
    neither = _row(mid_earnings_watch="False")
    assert sp.facets_for_row(neither).get("earnings_phase") == "no_earnings_phase"
    neither["post_earnings_active"] = NAN
    assert sp.facets_for_row(neither).get("earnings_phase") == sp.UNKNOWN


def test_columns_never_written_live_stay_unknown():
    # previous_anchor_*, daily_relative_strength_score and side_aligned_day are blank in every live row.
    row = _row(previous_anchor_nearest_level=NAN, previous_anchor_nearest_distance_atr=NAN,
               daily_relative_strength_score=NAN)
    got = sp.facets_for_row(row).as_dict()
    assert got["previous_anchor_level"] == sp.UNKNOWN
    assert got["rs_vs_spy"] == sp.UNKNOWN
    assert got["side_aligned_day"] == sp.UNKNOWN
    assert got["weekly_ema8_streak"] == sp.UNKNOWN


def test_other_store_facets_are_unknown_without_ctx_and_read_ctx_when_given():
    bare = sp.facets_for_row(_row()).as_dict()
    for name in ("discovery_slot", "entry_trigger", "m5_confirmation", "d1_environment"):
        assert bare[name] == sp.UNKNOWN
    ctx = {"discovery_slot": "10:00", "entry_trigger": "h1_ema_bounce", "m5_bounce_type": "none",
           "d1_environment": "Trend_Up"}
    got = sp.facets_for_row(_row(), ctx).as_dict()
    assert got["discovery_slot"] == "slot_1000"
    assert got["entry_trigger"] == "h1_ema_bounce"
    assert got["m5_confirmation"] == "no_m5_confirmation"
    assert got["d1_environment"] == "env_trend_up"
    assert sp.facets_for_row(_row(), {"m5_bounce_type": "VWAP_Reclaim"}).get("m5_confirmation") == "m5_vwap_reclaim"
    assert sp.facets_for_row(_row(), {"discovery_slot": "09:00"}).get("discovery_slot") == sp.UNKNOWN


def test_weekly_ema8_streak_reads_the_scan_value_when_present():
    assert sp.facets_for_row(_row(weekly_ema8_hold_weeks=4)).get("weekly_ema8_streak") == "weekly_ema8_hold_3_5w"


def test_short_label_skips_unknown_and_quiet_values():
    row = _row(**_all_mas(sma100=0.5))
    label = sp.facets_for_row(row).label
    parts = label.split("|")
    assert "sma100_support" in parts
    assert "weekly_ema15_hold" in parts
    assert sp.UNKNOWN not in parts
    assert "no_weekly_sma50_retest" not in parts
    assert "vwap_above_0to1atr" not in parts


def test_adding_a_facet_is_one_registered_function(monkeypatch):
    monkeypatch.setattr(sp, "FACETS", dict(sp.FACETS))

    @sp.facet("gap_up_today", "test", quiet=("no_gap_up",))
    def _gap_up(row, ctx, side):
        change = sp._num(row.get("current_day_change_pct"))
        if change is None:
            return sp.UNKNOWN
        return "gap_up" if change > 2 else "no_gap_up"

    key = sp.facets_for_row(_row(current_day_change_pct=3.1))
    assert key.facets[-1] == ("gap_up_today", "gap_up")
    assert key.label.endswith("|gap_up")
    assert sp.facets_for_row(_row()).get("gap_up_today") == sp.UNKNOWN
    with pytest.raises(ValueError):
        sp.facet("gap_up_today", "test")(_gap_up)


def test_a_facet_that_raises_is_unknown_not_a_crash(monkeypatch):
    monkeypatch.setattr(sp, "FACETS", dict(sp.FACETS))

    @sp.facet("broken", "test")
    def _broken(row, ctx, side):
        return str(1 / 0)

    assert sp.facets_for_row(_row()).get("broken") == sp.UNKNOWN


def test_module_is_pure_no_io_no_qt():
    tree = ast.parse((SCRIPTS_DIR / "setup_permutations.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "math", "dataclasses", "datetime", "typing"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "open" not in names


def test_nan_is_never_a_value():
    for value in sp.facets_for_row(_row(relvol=math.nan)).as_dict().values():
        assert "nan" not in value.split("_")
