"""Price-level alert watchlist: trigger rules, one-shot arming, store I/O."""

import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import price_alerts  # noqa: E402


def _entry(**overrides):
    base = {"symbol": "SPY", "above": 560.0, "below": 550.0, "note": "core"}
    base.update(overrides)
    return base


def test_normalize_drops_garbage_and_disarms_missing_levels():
    assert price_alerts.normalize_price_alert({"symbol": "  "}) is None
    entry = price_alerts.normalize_price_alert({"symbol": "spy", "above": "x", "below": -5})
    assert entry["symbol"] == "SPY"
    assert entry["above"] is None and entry["below"] is None
    assert entry["armed_above"] is False and entry["armed_below"] is False


def test_cross_above_fires_once_then_disarms():
    entries = [_entry()]
    updated, triggers = price_alerts.evaluate_price_alerts(
        entries, {"SPY": {"last": 561.25}}, datetime(2026, 8, 3, 5, 30)
    )
    assert len(triggers) == 1
    trigger = triggers[0]
    assert trigger["side"] == "above" and trigger["level"] == 560.0 and trigger["last"] == 561.25
    assert updated[0]["armed_above"] is False
    assert updated[0]["armed_below"] is True  # the other side stays armed

    # Same quote again: nothing fires - the wake-up channel never spams.
    again, triggers2 = price_alerts.evaluate_price_alerts(
        updated, {"SPY": {"last": 562.0}}, datetime(2026, 8, 3, 5, 31)
    )
    assert triggers2 == []
    assert again[0]["armed_above"] is False


def test_cross_below_fires_and_history_records():
    updated, triggers = price_alerts.evaluate_price_alerts(
        [_entry()], {"SPY": {"last": 549.9}}, datetime(2026, 8, 3, 5, 30)
    )
    assert triggers[0]["side"] == "below"
    assert updated[0]["history"][-1]["side"] == "below"


def test_no_quote_or_bad_quote_changes_nothing():
    entries = [_entry()]
    updated, triggers = price_alerts.evaluate_price_alerts(entries, {}, datetime.now())
    assert triggers == [] and updated[0]["armed_above"] is True
    updated, triggers = price_alerts.evaluate_price_alerts(
        entries, {"SPY": {"last": "nan-ish"}}, datetime.now()
    )
    assert triggers == [] and updated[0]["armed_above"] is True


def test_gap_through_both_levels_fires_only_the_crossed_side():
    # A collapse below the lower level must not also fire the "above" side.
    updated, triggers = price_alerts.evaluate_price_alerts(
        [_entry()], {"SPY": {"last": 500.0}}, datetime.now()
    )
    assert [t["side"] for t in triggers] == ["below"]


def test_store_round_trip_and_armed_symbols(tmp_path):
    path = tmp_path / "price_alerts.json"
    assert price_alerts.save_price_alerts(
        [_entry(), _entry(symbol="NVDA", below=None), {"symbol": ""}], path
    )
    loaded = price_alerts.load_price_alerts(path)
    assert [entry["symbol"] for entry in loaded] == ["SPY", "NVDA"]
    assert loaded[1]["armed_below"] is False
    assert price_alerts.armed_symbols(loaded) == ["NVDA", "SPY"]
    # Disarm everything -> no symbols to poll.
    for entry in loaded:
        entry["armed_above"] = entry["armed_below"] = False
    assert price_alerts.armed_symbols(loaded) == []


def test_trigger_log_round_trip(tmp_path):
    path = tmp_path / "triggers.csv"
    moment = datetime(2026, 8, 3, 5, 12)
    _updated, triggers = price_alerts.evaluate_price_alerts(
        [_entry()], {"SPY": {"last": 549.0}}, moment
    )
    price_alerts.append_trigger_log(triggers, path)
    price_alerts.append_trigger_log(triggers, path)  # header written once
    rows = price_alerts.todays_triggers(moment, path)
    assert len(rows) == 2
    assert rows[0]["symbol"] == "SPY" and rows[0]["side"] == "below"
    assert price_alerts.todays_triggers(datetime(2026, 8, 4, 5, 12), path) == []


def test_trigger_message_format():
    message = price_alerts.format_trigger_message(
        {"symbol": "SPY", "side": "below", "level": 550.0, "last": 549.12, "note": "core"}
    )
    assert message == "SPY 549.12 crossed BELOW your 550.00 alert level (core)"
