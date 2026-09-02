"""Armed alerts expire on the TRADING-day clock - Phase 0.12 A2.

Trader, 2026-09-01: an armed watch that never fires used to sit in the Armed
inventory forever, so the surface that is supposed to say "these are the
conditions I am waiting on" gradually stopped meaning anything.

Three things this file pins, because each is a way the rule could go wrong:

* the clock is SESSIONS, never weekdays - a Friday-armed 5-day watch is due
  the Friday after, and Thanksgiving is not a day;
* an unanswerable calendar does not expire anything - uncertainty never
  deletes;
* nothing is silently lost - every expiry writes a row naming symbol, kind,
  when it was armed and when it came due.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_five_day_watches_get_five_sessions_and_everything_else_gets_ten():
    import armed_alert_expiry as expiry

    assert expiry.expiry_trading_days("new_5d_high") == 5
    assert expiry.expiry_trading_days("new_5d_low") == 5
    assert expiry.expiry_trading_days("new_20d_high") == 10
    assert expiry.expiry_trading_days("new_20d_low") == 10
    assert expiry.expiry_trading_days("d1_level_above") == 10
    assert expiry.expiry_trading_days("any_bounce") == 10
    assert expiry.expiry_trading_days("price_alert") == 10
    assert expiry.expiry_trading_days("") == expiry.DEFAULT_EXPIRY_TRADING_DAYS


def test_the_clock_counts_sessions_not_weekdays():
    import armed_alert_expiry as expiry

    armed = datetime(2026, 8, 3, 9, 40)  # Monday
    # Five sessions later is Monday the 10th - the weekend is not two days.
    assert expiry.is_expired(armed, "new_5d_high", today=date(2026, 8, 7)) is False
    assert expiry.is_expired(armed, "new_5d_high", today=date(2026, 8, 10)) is True
    # The 10-session default is still open on the same date.
    assert expiry.is_expired(armed, "d1_level_above", today=date(2026, 8, 10)) is False
    assert expiry.is_expired(armed, "d1_level_above", today=date(2026, 8, 17)) is True


def test_an_unanswerable_calendar_never_expires():
    """`market_calendar` raises past its validated horizon. A watch armed on a
    date it cannot reason about stays armed - the whole point of failing
    closed here is that the alternative is deleting the trader's watch on a
    guess."""
    import armed_alert_expiry as expiry

    assert expiry.is_expired(datetime(1990, 1, 2), "new_5d_high", today=date(2026, 8, 10)) is None


def test_partition_keeps_what_is_not_due_and_names_what_is():
    import armed_alert_expiry as expiry
    from chart_watch import D1EventWatch

    fresh = D1EventWatch("AAPL", "new_5d_high", datetime(2026, 8, 7, 10, 0))
    stale = D1EventWatch("MSFT", "new_5d_high", datetime(2026, 7, 20, 10, 0))
    unanswerable = D1EventWatch("IBM", "new_5d_high", datetime(1990, 1, 2, 10, 0))

    kept, expired = expiry.partition(
        [fresh, stale, unanswerable],
        store="d1_event_watches",
        today=date(2026, 8, 10),
    )
    assert [w.symbol for w in kept] == ["AAPL", "IBM"]
    assert [row["symbol"] for row in expired] == ["MSFT"]
    row = expired[0]
    assert row["kind"] == "new_5d_high"
    assert row["store"] == "d1_event_watches"
    assert row["armed_at"].startswith("2026-07-20")
    assert row["expired_at"] == "2026-08-10"
    assert row["trading_days"] == 5
    assert row["schema"] == expiry.SCHEMA_ARMED_ALERT_EXPIRY


def test_the_expiry_row_is_appended_and_a_failed_append_costs_only_the_row(tmp_path):
    import armed_alert_expiry as expiry
    from evidence_ledger import EvidenceLedger

    ledger = EvidenceLedger(
        stream=expiry.STREAM, schema=expiry.SCHEMA_ARMED_ALERT_EXPIRY, directory=tmp_path
    )
    rows = [
        expiry.expiry_row(
            store="price_alerts",
            symbol="SPY",
            kind="price_alert",
            armed_at=datetime(2026, 7, 20, 16, 0),
            expired_at=date(2026, 8, 10),
            trading_days=10,
        )
    ]
    assert expiry.record_expiries(rows, ledger=ledger) == 1
    segments = list(tmp_path.glob("armed_alert_expiry-*.jsonl"))
    assert len(segments) == 1
    written = [json.loads(line) for line in segments[0].read_text().splitlines() if line.strip()]
    assert written[0]["symbol"] == "SPY"
    assert written[0]["store"] == "price_alerts"

    class _Broken:
        def append_many(self, rows, **kwargs):
            raise OSError("disk gone")

    # The store never costs the thing it records: a dead ledger returns 0.
    assert expiry.record_expiries(rows, ledger=_Broken()) == 0


# --------------------------------------------------------------------------
# Manual price alerts. These DISARM rather than delete: the entry is a name
# the trader typed with a level beside it, and "user-entered names are never
# automatically removed" (plan.md sec 5) still holds. Disarming takes it off
# the Armed surface, which is what the trader asked for, and leaves the level,
# the note and the trigger history exactly where they were.
# --------------------------------------------------------------------------
def test_a_price_alert_without_a_stamp_gets_today_never_an_older_guess():
    import price_alerts

    entry = price_alerts.normalize_price_alert({"symbol": "SPY", "above": 700.0})
    assert entry["armed_at"] == date.today().isoformat()


def test_a_price_alert_keeps_the_stamp_it_already_has():
    import price_alerts

    entry = price_alerts.normalize_price_alert(
        {"symbol": "SPY", "above": 700.0, "armed_at": "2026-07-20"}
    )
    assert entry["armed_at"] == "2026-07-20"


def test_a_stale_price_alert_is_disarmed_not_deleted():
    import price_alerts

    entries = [
        price_alerts.normalize_price_alert(
            {"symbol": "SPY", "above": 700.0, "below": 600.0, "armed_at": "2026-07-20"}
        ),
        price_alerts.normalize_price_alert(
            {"symbol": "QQQ", "above": 500.0, "armed_at": "2026-08-07"}
        ),
    ]
    updated, rows = price_alerts.expire_stale_alerts(entries, today=date(2026, 8, 10))

    spy = next(entry for entry in updated if entry["symbol"] == "SPY")
    assert spy["armed_above"] is False and spy["armed_below"] is False
    assert spy["above"] == 700.0 and spy["below"] == 600.0  # nothing is lost
    qqq = next(entry for entry in updated if entry["symbol"] == "QQQ")
    assert qqq["armed_above"] is True

    assert [row["symbol"] for row in rows] == ["SPY"]
    assert rows[0]["kind"] == "price_alert"
    assert rows[0]["store"] == "price_alerts"


def test_an_already_disarmed_price_alert_writes_no_expiry_row():
    import price_alerts

    entries = [
        price_alerts.normalize_price_alert(
            {"symbol": "SPY", "above": 700.0, "armed_above": False, "armed_at": "2026-07-20"}
        )
    ]
    updated, rows = price_alerts.expire_stale_alerts(entries, today=date(2026, 8, 10))
    assert rows == []
    assert updated == entries


def test_a_price_alert_the_calendar_cannot_date_stays_armed():
    import price_alerts

    entries = [
        price_alerts.normalize_price_alert(
            {"symbol": "SPY", "above": 700.0, "armed_at": "1990-01-02"}
        )
    ]
    updated, rows = price_alerts.expire_stale_alerts(entries, today=date(2026, 8, 10))
    assert rows == []
    assert updated[0]["armed_above"] is True
