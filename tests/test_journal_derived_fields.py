"""Derived per-trade fields exposed by journal_analytics for grouping."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _trade(opened_at, closed_at="", status="CLOSED", net_pnl=0.0):
    return {"opened_at": opened_at, "closed_at": closed_at, "status": status, "net_pnl": net_pnl}


def test_a_day_trade_reads_its_clock_weekday_and_hold():
    from journal_analytics import derived_trade_fields

    # 07:40 PT on a Thursday = 10:40 ET, 70 minutes into the session.
    fields = derived_trade_fields(_trade("2026-09-10T07:40:00-07:00", "2026-09-10T08:10:00-07:00"))

    assert fields == {
        "time_of_day": "late_morning",
        "weekday": "Thu",
        "hold_minutes": 30.0,
        "hold_bucket": "day_trade",
        "horizon": "day",
    }


def test_a_swing_and_an_open_trade():
    from journal_analytics import trade_hold_bucket, trade_horizon

    swing = _trade("2026-09-08T10:00:00-04:00", "2026-09-11T10:00:00-04:00")
    assert trade_hold_bucket(swing) == "swing"
    assert trade_horizon(swing) == "swing"
    still_on = _trade("2026-09-08T10:00:00-04:00", status="OPEN")
    assert trade_horizon(still_on) == "open"
    partly = _trade("2026-09-08T10:00:00-04:00", status="CLOSED_PARTIAL")
    assert trade_horizon(partly) == "open"


def test_a_date_only_fill_is_unknown_not_premarket():
    from journal_analytics import derived_trade_fields

    fields = derived_trade_fields(_trade("2026-09-10T00:00:00-04:00", "2026-09-10T00:00:00-04:00"))

    assert fields["time_of_day"] == "unknown"
    assert fields["hold_minutes"] is None
    assert fields["weekday"] == "Thu"  # the date itself is known
    assert fields["horizon"] == "day"


def test_garbage_is_unknown():
    from journal_analytics import derived_trade_fields

    fields = derived_trade_fields(_trade("not a time", "also not"))
    assert fields == {
        "time_of_day": "unknown",
        "weekday": "unknown",
        "hold_minutes": None,
        "hold_bucket": "unknown",
        "horizon": "unknown",
    }


def test_group_summary_buckets_by_a_derived_field():
    from journal_analytics import DERIVED_GROUPS, derived_group_summary

    trades = [
        _trade("2026-09-10T10:40:00-04:00", "2026-09-10T11:00:00-04:00", net_pnl=50.0),
        _trade("2026-09-10T10:45:00-04:00", "2026-09-10T11:30:00-04:00", net_pnl=-20.0),
        _trade("2026-09-08T10:00:00-04:00", "2026-09-11T10:00:00-04:00", net_pnl=100.0),
    ]
    assert set(DERIVED_GROUPS) == {"time of day", "weekday", "hold", "day vs swing"}

    rows = derived_group_summary(trades, "day vs swing")

    by_label = {row["label"]: row for row in rows}
    assert by_label["day"]["closed"] == 2 and by_label["day"]["net_pnl"] == 30.0
    assert by_label["swing"]["closed"] == 1 and by_label["swing"]["wins"] == 1
    assert rows[0]["label"] == "day"
