"""Packet 1: exit-independent, timestamp-true forward movement.

These tests deliberately drive the new pure publication seam.  They pin the
facts that an entry-quality row must carry; they do not change the legacy
outcome-path/exit simulation contract.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


ET = ZoneInfo("America/New_York")


def _moment(day: str, clock: str) -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}").replace(tzinfo=ET)


def _bar(day: str, clock: str, high: float, low: float, close: float) -> dict:
    """A completed five-minute bar whose timestamp is its completed end."""
    return {"time": _moment(day, clock).isoformat(), "high": high, "low": low, "close": close}


def _entry(**overrides) -> dict:
    row = {
        "opportunity_id": "opp|M5|ACME|2026-03-09T10:00:00-04:00",
        "attempt_id": "attempt|opp|M5|ACME|base_entry_v1",
        "symbol": "ACME",
        "side": "LONG",
        "entry_rule": "base_entry",
        "entry_rule_version": "base_entry_v1",
        "trigger_knowledge_time": "2026-03-09T10:00:00-04:00",
        "feasible_entry_time": "2026-03-09T10:00:00-04:00",
        "feasible_entry_price": 100.0,
        "source_knowledge_basis": "observed",
        "anchor_knowledge_basis": "observed",
        "state": "complete",
        "risk_price": 99.0,
        "entry_atr": 2.0,
        "favorable_threshold_pct": 2.0,
        "adverse_threshold_pct": 1.0,
    }
    row.update(overrides)
    return row


def _m5(result: dict, minutes: int) -> dict:
    return result["windows"][f"{minutes}m"]


def _close(result: dict) -> dict:
    return result["windows"]["session_close"]


def test_m5_windows_use_completed_timestamps_and_never_spill_a_session():
    """A late entry has a valid close but cannot invent an after-close 180m row."""
    import entry_quality

    bars = [
        _bar("2026-03-09", "15:35:00", 101, 99.8, 100.5),
        _bar("2026-03-09", "15:40:00", 102, 100, 101),
        _bar("2026-03-09", "15:45:00", 103, 100, 102),
        _bar("2026-03-09", "15:50:00", 104, 101, 103),
        _bar("2026-03-09", "15:55:00", 105, 102, 104),
        _bar("2026-03-09", "16:00:00", 106, 103, 105),
        # A very good next-day bar must never rescue the same session's 180m window.
        _bar("2026-03-10", "09:35:00", 150, 149, 150),
    ]
    result = entry_quality.measure_m5_forward(
        _entry(feasible_entry_time="2026-03-09T15:30:00-04:00"), bars, as_of=_moment("2026-03-10", "12:00:00")
    )

    assert tuple(result["window_minutes"]) == (5, 15, 30, 60, 120, 180)
    assert _m5(result, 5)["state"] == "complete"
    assert _m5(result, 30)["state"] == "complete"
    assert _m5(result, 60)["state"] == "unavailable"
    assert _m5(result, 180)["state"] == "unavailable"
    assert _m5(result, 180)["reason"] == "window_ends_after_session_close"
    assert _close(result)["state"] == "complete"
    assert _close(result)["mfe_pct"] == pytest.approx(6.0)


def test_early_close_holiday_and_dst_use_exchange_session_boundaries():
    """NYSE time, not a fixed 78-bar/16:00 or calendar-day approximation."""
    import entry_quality

    # Black Friday is an early close.  A 13:00 ET close bar is the endpoint.
    half_day = entry_quality.measure_m5_forward(
        _entry(
            feasible_entry_time="2026-11-27T12:45:00-05:00",
            trigger_knowledge_time="2026-11-27T12:45:00-05:00",
        ),
        [
            _bar("2026-11-27", "12:50:00", 101, 99.8, 100.4),
            _bar("2026-11-27", "12:55:00", 102, 100, 101),
            _bar("2026-11-27", "13:00:00", 103, 100, 102),
            _bar("2026-11-27", "13:05:00", 199, 198, 199),
        ],
        as_of=_moment("2026-11-30", "10:00:00"),
    )
    assert _close(half_day)["state"] == "complete"
    assert _close(half_day)["endpoint_time"] == "2026-11-27T13:00:00-05:00"
    assert _close(half_day)["mfe_pct"] == pytest.approx(3.0)

    # Good Friday is skipped, and Monday after the March DST change is EDT.
    assert entry_quality.exchange_session_endpoints(date(2026, 4, 2), (1,)) == {1: date(2026, 4, 6)}
    assert entry_quality.exchange_session_endpoints(date(2026, 3, 6), (1,)) == {1: date(2026, 3, 9)}
    assert entry_quality.session_endpoint(date(2026, 3, 9)).isoformat().endswith("-04:00")


def test_missing_middle_bars_are_partial_and_cannot_stretch_a_60_minute_window():
    """Two bars by 11:00 are not twelve bars, and 11:30 is outside the hour."""
    import entry_quality

    result = entry_quality.measure_m5_forward(
        _entry(),
        [
            _bar("2026-03-09", "10:05:00", 101, 99.5, 100.5),
            _bar("2026-03-09", "11:00:00", 105, 100, 104),
            _bar("2026-03-09", "11:30:00", 150, 149, 150),
        ],
        as_of=_moment("2026-03-09", "16:30:00"),
    )

    sixty = _m5(result, 60)
    assert sixty["state"] == "partial"
    assert sixty["endpoint_time"] == "2026-03-09T11:00:00-04:00"
    assert sixty["coverage"] == {"expected_bars": 12, "observed_bars": 2, "missing_bars": 10}
    assert sixty["mfe_pct"] == pytest.approx(5.0)
    assert _m5(result, 120)["state"] == "missing_data"
    assert _m5(result, 120)["reason"] == "no_bar_at_or_before_window_endpoint"


def test_exit_independent_mfe_keeps_post_stop_rally_and_leaves_legacy_path_byte_identical():
    """A stopped hypothetical trade can still describe the opportunity price offered."""
    import outcome_path

    legacy_bars = [
        {"high": 100.5, "low": 98.0, "close": 99.0},
        {"high": 104.0, "low": 99.0, "close": 103.0},
    ]
    before = json.dumps(
        outcome_path.capture_path(entry_price=100, stop_price=99, side="long", bars=legacy_bars),
        sort_keys=True,
    )

    import entry_quality

    result = entry_quality.measure_m5_forward(
        _entry(),
        [
            _bar("2026-03-09", "10:05:00", 100.5, 98.0, 99.0),
            _bar("2026-03-09", "10:10:00", 104.0, 99.0, 103.0),
            _bar("2026-03-09", "10:15:00", 104.0, 102.0, 103.0),
        ],
        as_of=_moment("2026-03-09", "16:30:00"),
    )
    fifteen = _m5(result, 15)

    assert fifteen["mfe_pct"] == pytest.approx(4.0)
    assert fifteen["first_touch_order"] == "adverse_first"
    assert fifteen["first_adverse_time"] == "2026-03-09T10:05:00-04:00"
    assert json.dumps(
        outcome_path.capture_path(entry_price=100, stop_price=99, side="long", bars=legacy_bars),
        sort_keys=True,
    ) == before


def test_strictly_after_knowledge_time_excludes_confirmation_candle_and_mirrors_short():
    """The candle that confirms an entry cannot donate its earlier wick."""
    import entry_quality

    long_result = entry_quality.measure_m5_forward(
        _entry(),
        [
            _bar("2026-03-09", "10:00:00", 109, 99, 108),  # confirmation candle: excluded
            _bar("2026-03-09", "10:05:00", 102, 99, 101),
            _bar("2026-03-09", "10:10:00", 102, 100, 101),
            _bar("2026-03-09", "10:15:00", 102, 100, 101),
        ],
        as_of=_moment("2026-03-09", "16:30:00"),
    )
    short_result = entry_quality.measure_m5_forward(
        _entry(side="SHORT", risk_price=101.0),
        [
            _bar("2026-03-09", "10:00:00", 101, 91, 92),  # excluded
            _bar("2026-03-09", "10:05:00", 101, 98, 99),
            _bar("2026-03-09", "10:10:00", 100, 98, 99),
            _bar("2026-03-09", "10:15:00", 100, 98, 99),
        ],
        as_of=_moment("2026-03-09", "16:30:00"),
    )

    assert _m5(long_result, 15)["mfe_pct"] == pytest.approx(2.0)
    assert _m5(short_result, 15)["mfe_pct"] == pytest.approx(2.0)
    assert _m5(long_result, 15)["bars_excluded_before_knowledge"] == 1
    assert _m5(short_result, 15)["bars_excluded_before_knowledge"] == 1


def test_daily_endpoints_are_exchange_sessions_and_same_bar_touches_are_ambiguous():
    """Daily OHLC retains the order ambiguity instead of pretending it knows a wick sequence."""
    import entry_quality

    daily = [
        {"date": "2026-01-05", "high": 105, "low": 95, "close": 101},
        {"date": "2026-01-06", "high": 106, "low": 99, "close": 104},
        {"date": "2026-01-07", "high": 107, "low": 100, "close": 105},
        {"date": "2026-01-08", "high": 108, "low": 100, "close": 106},
        {"date": "2026-01-09", "high": 109, "low": 101, "close": 107},
    ]
    result = entry_quality.measure_swing_forward(
        _entry(
            opportunity_id="opp|D1|ACME|2026-01-02",
            attempt_id="attempt|opp|D1|ACME|base_entry_v1",
            trigger_knowledge_time="2026-01-02T16:00:00-05:00",
            feasible_entry_time="2026-01-02T16:00:00-05:00",
            favorable_threshold_pct=4.0,
            adverse_threshold_pct=3.0,
        ),
        daily,
        last_completed_session=date(2026, 1, 9),
    )

    assert tuple(result["window_sessions"]) == (1, 2, 3, 5, 10)
    assert result["entry_session_convention"] == "next_completed_exchange_session_close_v1"
    assert result["windows"]["1_session"]["endpoint_session"] == "2026-01-05"
    assert result["windows"]["5_session"]["endpoint_session"] == "2026-01-09"
    assert result["windows"]["10_session"]["state"] == "pending"
    assert result["windows"]["1_session"]["first_touch_order"] == "ambiguous_same_daily_bar"


def test_percentage_atr_and_r_are_separate_and_r_requires_contemporaneous_valid_risk():
    """Changing a later risk model may not invent an entry-quality edge."""
    import entry_quality

    bars = [
        _bar("2026-03-09", "10:05:00", 104, 98, 102),
        _bar("2026-03-09", "10:10:00", 104, 99, 102),
        _bar("2026-03-09", "10:15:00", 104, 100, 102),
    ]
    result = entry_quality.measure_m5_forward(_entry(), bars, as_of=_moment("2026-03-09", "16:30:00"))
    row = _m5(result, 15)
    assert row["mfe_pct"] == pytest.approx(4.0)
    assert row["mfe_atr"] == pytest.approx(2.0)
    assert row["mfe_r"] == pytest.approx(4.0)

    no_risk = entry_quality.measure_m5_forward(
        _entry(risk_price=100.0), bars, as_of=_moment("2026-03-09", "16:30:00")
    )
    assert _m5(no_risk, 15)["mfe_r"] is None
    assert _m5(no_risk, 15)["risk_reason"] == "invalid_entry_risk_reference"


def test_stable_identity_states_reasons_and_coverage_are_never_silently_dropped():
    """Every base attempt says what happened, including all unmeasured cases."""
    import entry_quality

    one_bar = [_bar("2026-03-09", "10:05:00", 101, 99.5, 100.5)]
    cases = {
        "complete": (_entry(), one_bar),
        "pending": (_entry(), one_bar),
        "partial": (_entry(), one_bar),
        "no_trigger": (_entry(state="no_trigger"), one_bar),
        "invalid_entry": (_entry(feasible_entry_price=None), one_bar),
        "missing_data": (_entry(), []),
    }
    complete = entry_quality.measure_m5_forward(cases["complete"][0], cases["complete"][1], as_of=_moment("2026-03-09", "10:06:00"))
    pending = entry_quality.measure_m5_forward(cases["pending"][0], cases["pending"][1], as_of=_moment("2026-03-09", "10:06:00"))
    partial = entry_quality.measure_m5_forward(cases["partial"][0], cases["partial"][1], as_of=_moment("2026-03-09", "16:30:00"))
    no_trigger = entry_quality.measure_m5_forward(cases["no_trigger"][0], cases["no_trigger"][1], as_of=_moment("2026-03-09", "16:30:00"))
    invalid = entry_quality.measure_m5_forward(cases["invalid_entry"][0], cases["invalid_entry"][1], as_of=_moment("2026-03-09", "16:30:00"))
    missing = entry_quality.measure_m5_forward(cases["missing_data"][0], cases["missing_data"][1], as_of=_moment("2026-03-09", "16:30:00"))

    assert complete["opportunity_id"] == cases["complete"][0]["opportunity_id"]
    assert complete["attempt_id"] == cases["complete"][0]["attempt_id"]
    assert _m5(complete, 5)["state"] == "complete"
    assert _m5(pending, 15)["state"] == "pending"
    assert _m5(partial, 15)["state"] == "partial"
    assert _m5(no_trigger, 5)["state"] == "no_trigger"
    assert _m5(invalid, 5)["state"] == "invalid_entry"
    assert _m5(missing, 5)["state"] == "missing_data"
    for result in (complete, pending, partial, no_trigger, invalid, missing):
        row = _m5(result, 5)
        assert row["reason"]
        assert set(row["coverage"]) == {"expected_bars", "observed_bars", "missing_bars"}


def test_reconstructed_knowledge_is_labelled_and_cannot_confirm_a_prospective_claim():
    """A repaired anchor can be useful context but never retrospective proof."""
    import entry_quality

    result = entry_quality.measure_m5_forward(
        _entry(anchor_knowledge_basis="reconstructed", source_knowledge_basis="reconstructed"),
        [
            _bar("2026-03-09", "10:05:00", 103, 99, 102),
            _bar("2026-03-09", "10:10:00", 104, 100, 103),
            _bar("2026-03-09", "10:15:00", 104, 100, 103),
        ],
        as_of=_moment("2026-03-09", "16:30:00"),
    )

    assert result["knowledge_label"] == "reconstructed"
    assert result["prospective_eligible"] is False
    assert _m5(result, 15)["confirmation_valid"] is False
    assert _m5(result, 15)["confirmation_reason"] == "reconstructed_knowledge_cannot_confirm_prospective_claim"
