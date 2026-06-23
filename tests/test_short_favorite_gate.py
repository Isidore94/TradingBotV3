import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_apply_short_directional_flags_marks_weakness():
    from master_avwap_lib import legacy

    row = {"side": "SHORT"}
    symbol_entry = {"last_close": 100.0, "previous_day_low": 101.0}
    legacy._apply_short_directional_flags(row, symbol_entry, {"SMA_20": 105.0, "SMA_50": 110.0})
    # close 100 < SMA20 105 < SMA50 110 -> bearish stack; close 100 < prior low 101 -> breakdown
    assert row["priority_short_trend_aligned"] is True
    assert row["priority_closed_below_prev_day_low"] is True


def test_apply_short_directional_flags_marks_elevated_short():
    from master_avwap_lib import legacy

    row = {"side": "SHORT"}
    symbol_entry = {"last_close": 108.0, "previous_day_low": 101.0}
    legacy._apply_short_directional_flags(row, symbol_entry, {"SMA_20": 105.0, "SMA_50": 110.0})
    assert row["priority_short_trend_aligned"] is False  # close above SMA20
    assert row["priority_closed_below_prev_day_low"] is False  # close above prior-day low


def test_short_gate_demotes_elevated_short():
    from master_avwap_lib import legacy

    row = {"side": "SHORT", "priority_short_trend_aligned": False, "priority_closed_below_prev_day_low": False}
    reason = legacy._short_favorite_demotion_reason(row)
    assert "SMA" in reason
    assert "prior-day low" in reason


def test_short_gate_allows_confirmed_short():
    from master_avwap_lib import legacy

    row = {"side": "SHORT", "priority_short_trend_aligned": True, "priority_closed_below_prev_day_low": True}
    assert legacy._short_favorite_demotion_reason(row) == ""


def test_short_gate_fails_open_on_missing_data():
    from master_avwap_lib import legacy

    row = {"side": "SHORT", "priority_short_trend_aligned": None, "priority_closed_below_prev_day_low": None}
    assert legacy._short_favorite_demotion_reason(row) == ""


def test_short_gate_does_not_touch_longs():
    from master_avwap_lib import legacy

    row = {"side": "LONG", "priority_short_trend_aligned": False, "priority_closed_below_prev_day_low": False}
    assert legacy._short_favorite_demotion_reason(row) == ""
