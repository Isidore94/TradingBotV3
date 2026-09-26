"""P8 P1: backfill an empty journal plan from the desk's own pre-entry alert or D1 plan.

The downstream proof is the point: after one `--apply`, the trade's R, MFE/MAE
in R and entry grade stop being unknown. The refusals matter as much: a plan
the trader set is never touched, and only rows stamped before the fill count.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

NY = ZoneInfo("America/New_York")
OUTCOME_FIELDS = ["schema_version", "event_id", "event_type", "logged_at", "trade_date", "symbol",
                  "direction", "entry_time", "entry_price", "stop_price", "risk_per_share"]
DAILY_FIELDS = ["setup_id", "scan_date", "symbol", "side", "trade_date", "close", "current_avwape",
                "current_upper_1", "current_lower_1", "current_upper_2", "current_lower_2",
                "current_upper_3", "current_lower_3", "atr20"]


def _store(tmp_path):
    from journal_store import JournalStore

    return JournalStore(tmp_path / "journal.sqlite3")


def _seed_trade(store, trade_id="T1", *, symbol="AAA", direction="LONG", security_type="STK",
                opened="2026-08-20T10:00:00-04:00", closed="2026-08-20T11:00:00-04:00",
                entry=50.0, quantity=100.0):
    with store.connection() as conn:
        conn.execute(
            """
            INSERT INTO trades(
                trade_id, broker, account_number, symbol, security_type, currency, direction,
                status, opened_at, closed_at, trade_date, quantity_opened, quantity_closed,
                average_entry_price, average_exit_price, net_pnl, net_pnl_cad, updated_at
            ) VALUES(?, 'IBKR', 'U1', ?, ?, 'USD', ?, 'CLOSED', ?, ?, ?, ?, ?, ?, 52.0, 200.0, 270.0, ?)
            """,
            (trade_id, symbol, security_type, direction, opened, closed, opened[:10],
             quantity, quantity, entry, "2026-08-20T00:00:00"),
        )
    return trade_id


def _alerts(tmp_path, rows):
    path = tmp_path / "intraday_bounce_outcomes.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTCOME_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"schema_version": 1, "event_id": f"E{index}", "event_type": "registered",
                             "trade_date": row["logged_at"][:10], "risk_per_share": "", **row})
    return path


def _daily(tmp_path, rows=()):
    path = tmp_path / "master_avwap_setup_daily.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAILY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _alert(logged_at, *, symbol="AAA", direction="long", entry=49.9, stop=49.5):
    return {"logged_at": logged_at, "symbol": symbol, "direction": direction,
            "entry_time": logged_at[:19], "entry_price": entry, "stop_price": stop}


def _run(tmp_path, store, outcomes, daily, *extra):
    import journal_stop_backfill as backfill

    return backfill.main(["--db", str(store.db_path), "--outcomes", str(outcomes),
                          "--setup-daily", str(daily), *extra])


def _trade(store, trade_id="T1"):
    return next(row for row in store.list_trades() if row["trade_id"] == trade_id)


def _m5_bars(start, count=12):
    return [{"dt": start + timedelta(minutes=5 * index), "open": 50.0,
             "high": 50.0 + 0.1 * index, "low": 49.8, "close": 50.0} for index in range(count)]


def test_apply_fills_the_plan_and_r_mfe_and_entry_grade_become_known(tmp_path, capsys):
    from entry_plan import entry_grade
    from journal_analytics import trade_r_multiple
    from journal_excursion import excursion

    store = _store(tmp_path)
    _seed_trade(store)
    outcomes = _alerts(tmp_path, [
        _alert("2026-08-20T06:30:00-07:00", entry=49.0, stop=48.0),   # earlier, same session
        _alert("2026-08-20T06:50:00-07:00"),                         # 09:50 ET: the closest before
        _alert("2026-08-20T07:10:00-07:00", entry=51.0, stop=50.5),  # after the fill
    ])
    daily = _daily(tmp_path)
    now = datetime(2026, 8, 21, 12, 0, tzinfo=NY)
    bars = _m5_bars(datetime(2026, 8, 20, 10, 0, tzinfo=NY))

    before = _trade(store)
    assert trade_r_multiple(before) is None
    assert entry_grade(before["average_entry_price"], before["planned_entry"], before["planned_stop"]) is None
    assert excursion(before, bars, now=now)["mfe_r"] is None

    assert _run(tmp_path, store, outcomes, daily, "--apply") == 0

    after = _trade(store)
    assert after["planned_stop"] == 49.5
    assert after["planned_entry"] == 49.9
    assert after["planned_risk"] == (50.0 - 49.5) * 100
    assert after["risk_source"] == "backfill_m5_alert"
    assert trade_r_multiple(after) == 270.0 / 50.0
    assert abs(entry_grade(after["average_entry_price"], after["planned_entry"], after["planned_stop"]) - 0.25) < 1e-9
    measured = excursion(after, bars, now=now)
    assert measured["state"] == "measured"
    assert measured["mfe_r"] is not None and measured["mae_r"] is not None
    out = capsys.readouterr().out
    assert "Filled: 1 by backfill_m5_alert, 0 by backfill_d1_plan; skipped 0; no source found 0" in out


def test_dry_run_prints_the_line_and_writes_nothing(tmp_path, capsys):
    store = _store(tmp_path)
    _seed_trade(store)
    outcomes = _alerts(tmp_path, [_alert("2026-08-20T06:50:00-07:00")])

    assert _run(tmp_path, store, outcomes, _daily(tmp_path)) == 0

    out = capsys.readouterr().out
    assert "T1 2026-08-20 AAA LONG: backfill_m5_alert stop 49.5 entry 49.9 risk 50.00 USD" in out
    assert "Would fill: 1 by backfill_m5_alert" in out
    assert "Dry run: nothing written" in out
    assert _trade(store)["planned_stop"] is None
    assert _trade(store)["risk_source"] == ""


def test_a_plan_the_trader_set_is_never_touched(tmp_path, capsys):
    store = _store(tmp_path)
    _seed_trade(store)
    store.save_risk_fields("T1", planned_entry=50.0, planned_stop=None, planned_risk=None, risk_source="manual")
    outcomes = _alerts(tmp_path, [_alert("2026-08-20T06:50:00-07:00")])

    _run(tmp_path, store, outcomes, _daily(tmp_path), "--apply")

    trade = _trade(store)
    assert trade["planned_stop"] is None
    assert trade["planned_entry"] == 50.0
    assert trade["risk_source"] == "manual"
    assert "skipped 1" in capsys.readouterr().out


def test_store_guard_refuses_a_row_with_any_plan_field(tmp_path):
    store = _store(tmp_path)
    _seed_trade(store)
    store.save_risk_fields("T1", planned_stop=48.0, risk_source="")

    wrote = store.backfill_risk_fields("T1", planned_entry=49.9, planned_stop=49.5,
                                       planned_risk=50.0, risk_source="backfill_m5_alert")

    assert wrote is False
    assert _trade(store)["planned_stop"] == 48.0


def test_store_guard_refuses_when_only_risk_source_is_set(tmp_path):
    store = _store(tmp_path)
    _seed_trade(store)
    store.save_risk_fields("T1", risk_source="manual")

    wrote = store.backfill_risk_fields("T1", planned_entry=49.9, planned_stop=49.5,
                                       planned_risk=50.0, risk_source="backfill_m5_alert")

    assert wrote is False
    trade = _trade(store)
    assert trade["planned_stop"] is None and trade["risk_source"] == "manual"


def test_store_guard_refuses_when_only_planned_risk_is_set(tmp_path):
    store = _store(tmp_path)
    _seed_trade(store)
    store.save_risk_fields("T1", planned_risk=75.0, risk_source="")

    wrote = store.backfill_risk_fields("T1", planned_entry=49.9, planned_stop=49.5,
                                       planned_risk=50.0, risk_source="backfill_m5_alert")

    assert wrote is False
    trade = _trade(store)
    assert trade["planned_stop"] is None and trade["planned_risk"] == 75.0


def test_an_alert_more_than_1r_from_the_fill_is_refused(tmp_path, capsys):
    """ZETA 2026-08-11: alert entry 27.83, stop 27 (1R = 0.83), filled at 30.26."""
    store = _store(tmp_path)
    _seed_trade(store, symbol="ZETA", opened="2026-08-11T10:00:00-04:00",
                closed="2026-08-11T11:00:00-04:00", entry=30.26)
    outcomes = _alerts(tmp_path, [_alert("2026-08-11T06:30:00-07:00", symbol="ZETA", entry=27.83, stop=27.0)])

    _run(tmp_path, store, outcomes, _daily(tmp_path), "--apply")

    assert _trade(store)["planned_stop"] is None
    out = capsys.readouterr().out
    assert "alert too far from fill" in out
    assert "no source found 1" in out


def test_only_same_session_same_side_alerts_before_the_fill_count(tmp_path, capsys):
    store = _store(tmp_path)
    _seed_trade(store)
    outcomes = _alerts(tmp_path, [
        _alert("2026-08-20T07:10:00-07:00"),                    # after the fill
        _alert("2026-08-20T06:50:00-07:00", direction="short", stop=50.5),  # other side
        _alert("2026-08-19T12:00:00-07:00"),                    # prior session
        _alert("2026-08-20T06:50:00-07:00", symbol="BBB"),      # other symbol
        {**_alert("2026-08-20T06:50:00"), "logged_at": "2026-08-20T06:50:00"},  # naive stamp
    ])

    _run(tmp_path, store, outcomes, _daily(tmp_path), "--apply")

    assert _trade(store)["planned_stop"] is None
    assert "no source found 1" in capsys.readouterr().out


def test_a_stop_on_the_wrong_side_of_the_fill_is_refused(tmp_path, capsys):
    store = _store(tmp_path)
    _seed_trade(store)
    outcomes = _alerts(tmp_path, [_alert("2026-08-20T06:50:00-07:00", entry=51.0, stop=50.5)])

    _run(tmp_path, store, outcomes, _daily(tmp_path), "--apply")

    assert _trade(store)["planned_stop"] is None
    assert "wrong side" in capsys.readouterr().out


def test_d1_plan_from_the_latest_scan_read_on_the_session_before_the_entry(tmp_path):
    from entry_plan import plan_for_row

    store = _store(tmp_path)
    _seed_trade(store, entry=51.2)
    bands = {"current_upper_1": 52, "current_lower_1": 48, "current_upper_2": 54,
             "current_lower_2": 46, "current_upper_3": 56, "current_lower_3": 44}
    base = {"symbol": "AAA", "side": "LONG", "close": 51.0, "current_avwape": 50.0, "atr20": 1.0, **bands}
    setup = "2026-08-12:AAA:LONG:2026-07-30:near_favorite_zone"
    older = "2026-08-05:AAA:LONG:2026-07-01:near_favorite_zone"
    daily = _daily(tmp_path, [
        # An older scan of the same name reads other bands: the latest scan wins.
        {**base, "setup_id": older, "scan_date": "2026-08-05", "trade_date": "2026-08-19", "current_lower_1": 47.0},
        {**base, "setup_id": setup, "scan_date": "2026-08-12", "trade_date": "2026-08-18", "current_lower_1": 47.5},
        {**base, "setup_id": setup, "scan_date": "2026-08-12", "trade_date": "2026-08-19"},
        # The entry day's own session carries that day's later close: not point in time.
        {**base, "setup_id": setup, "scan_date": "2026-08-12", "trade_date": "2026-08-20", "current_lower_1": 49.0},
        # A scan made after the entry never counts.
        {**base, "setup_id": "2026-08-21:AAA:LONG:2026-07-30:near_favorite_zone", "scan_date": "2026-08-21",
         "trade_date": "2026-08-19", "current_lower_1": 49.5},
    ])
    expected = plan_for_row(
        symbol="AAA", side="LONG", setup_family="near_favorite_zone", last_close=51.0,
        levels_by_symbol={"AAA": {"bands": {"UPPER_1": 52, "LOWER_1": 48, "UPPER_2": 54, "LOWER_2": 46,
                                            "UPPER_3": 56, "LOWER_3": 44}, "vwap": 50.0, "atr20": 1.0}},
    )
    assert expected["stop"] == 48.0

    _run(tmp_path, store, _alerts(tmp_path, []), daily, "--apply")

    trade = _trade(store)
    assert trade["risk_source"] == "backfill_d1_plan"
    assert trade["planned_stop"] == expected["stop"]
    assert trade["planned_entry"] == expected["entry"]
    assert abs(trade["planned_risk"] - (51.2 - 48.0) * 100) < 1e-9


def test_an_option_is_skipped_even_when_its_underlying_alert_matches(tmp_path, capsys):
    store = _store(tmp_path)
    _seed_trade(store, symbol="AAA260918C00050000", security_type="OPT", entry=1.2, quantity=2)
    outcomes = _alerts(tmp_path, [_alert("2026-08-20T06:50:00-07:00")])

    _run(tmp_path, store, outcomes, _daily(tmp_path), "--apply")

    assert _trade(store)["planned_stop"] is None
    assert "option" in capsys.readouterr().out


def test_since_and_trade_id_filter_the_pass(tmp_path, capsys):
    store = _store(tmp_path)
    _seed_trade(store, "T1")
    _seed_trade(store, "T2", opened="2026-08-10T10:00:00-04:00", closed="2026-08-10T11:00:00-04:00")
    _seed_trade(store, "T3")
    outcomes = _alerts(tmp_path, [_alert("2026-08-20T06:50:00-07:00")])

    _run(tmp_path, store, outcomes, _daily(tmp_path), "--apply", "--since", "2026-08-15", "--trade-id", "T3")

    assert _trade(store, "T3")["risk_source"] == "backfill_m5_alert"
    assert _trade(store, "T1")["risk_source"] == ""
    assert _trade(store, "T2")["risk_source"] == ""
    assert "(of 1 trades)" in capsys.readouterr().out


def test_an_unreadable_source_is_named_not_counted_as_no_source(tmp_path, capsys):
    store = _store(tmp_path)
    _seed_trade(store)

    _run(tmp_path, store, tmp_path / "missing.csv", _daily(tmp_path))

    assert "Unreadable source" in capsys.readouterr().out
