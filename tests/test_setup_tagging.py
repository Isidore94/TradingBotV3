from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_retest_tags_separate_family_trigger_and_confirmations():
    from master_avwap_lib.setup_tagging import derive_setup_tag_payload

    payload = derive_setup_tag_payload(
        {
            "symbol": "ADM",
            "side": "LONG",
            "setup_family": "avwap_retest_followthrough",
            "setup_tags": ["BOUNCE_VWAP", "CROSS_UP_VWAP"],
            "favorite_signals": ["BOUNCE_VWAP", "CROSS_UP_VWAP"],
            "top_pattern_entry": True,
            "previous_day_range_break": True,
            "breakout_5d": True,
        }
    )

    assert payload["setup_tags"] == [
        "AVWAP_RETEST",
        "VWAP_BOUNCE",
        "VWAP_RECLAIM",
        "TOP_PATTERN_ENTRY",
        "RANGE_BREAK_CONFIRMED",
        "FIVE_DAY_BREAKOUT",
    ]
    assert payload["setup_signal_tags"] == ["BOUNCE_VWAP", "CROSS_UP_VWAP"]
    assert payload["setup_tag_roles"]["AVWAP_RETEST"] == "primary"
    assert payload["setup_tag_roles"]["VWAP_RECLAIM"] == "trigger"
    assert "favorite_signals=CROSS_UP_VWAP" in payload["setup_tag_evidence"]["VWAP_RECLAIM"]


def test_blank_legacy_tags_still_get_family_and_side_aware_evidence():
    from master_avwap_lib.setup_tagging import derive_setup_tag_payload

    payload = derive_setup_tag_payload(
        {
            "side": "SHORT",
            "setup_family": "favorite_zone_watch",
            "setup_tags": [],
            "previous_day_range_break": True,
            "breakout_5d": True,
            "trend_20d": "DOWN",
            "trend_ma_alignment": True,
            "daily_relative_strength_score": -5.0,
            "industry_relative_strength_score": -2.0,
        }
    )

    assert payload["setup_tags"] == [
        "FAVORITE_ZONE_WATCH",
        "RANGE_BREAK_CONFIRMED",
        "FIVE_DAY_BREAKOUT",
        "D1_TREND_ALIGNED",
        "MA_TREND_ALIGNED",
        "D1_RW",
    ]
    assert "D1_RS" not in payload["setup_tags"]


def test_directionally_inconsistent_signal_is_not_displayed_as_confirmation():
    from master_avwap_lib.setup_tagging import derive_setup_tag_payload

    payload = derive_setup_tag_payload(
        {
            "side": "LONG",
            "setup_family": "avwap_breakout",
            "favorite_signals": ["CROSS_DOWN_VWAP"],
            "setup_tags": ["CROSS_DOWN_VWAP"],
        }
    )

    assert payload["setup_tags"] == ["AVWAP_BREAKOUT"]
    assert payload["setup_signal_tags"] == ["CROSS_DOWN_VWAP"]
    assert payload["setup_tag_warnings"] == ["CROSS_DOWN_VWAP conflicts with side=LONG"]


def test_v2_derivation_is_stable_and_preserves_custom_study_tag():
    from master_avwap_lib.setup_tagging import apply_setup_tag_payload, derive_setup_tag_payload

    row = {
        "side": "LONG",
        "setup_family": "weekly_ema8_hold_retest",
        "setup_tags": ["WEEKLY_EMA8_HOLD"],
    }
    first = apply_setup_tag_payload(row)
    second = derive_setup_tag_payload(row)

    assert first == second
    assert row["setup_tags"] == ["WEEKLY_EMA8_HOLD_RETEST", "WEEKLY_EMA8_HOLD"]


def test_canonicalization_updates_scanner_state_and_feature_row():
    from master_avwap_lib.setup_tagging import canonicalize_priority_setup_tags

    rows = [
        {
            "symbol": "CELH",
            "side": "SHORT",
            "setup_family": "avwap_breakout",
            "favorite_signals": ["CROSS_DOWN_VWAP"],
            "setup_tags": ["CROSS_DOWN_VWAP"],
        }
    ]
    ai_state = {"symbols": {"CELH": {}}}
    features = {"CELH": {}}

    assert canonicalize_priority_setup_tags(rows, ai_state, features) == 1
    assert rows[0]["setup_tags"] == ["VWAP_BREAKDOWN"]
    assert ai_state["symbols"]["CELH"]["setup_tags"] == ["VWAP_BREAKDOWN"]
    assert features["CELH"]["setup_tags"] == "VWAP_BREAKDOWN"
    assert features["CELH"]["setup_signal_tags"] == "CROSS_DOWN_VWAP"
