"""P1-4 4a - the setup key stamped on M5 alerts, in a sidecar keyed by event_id.

Shadow only. The outcome CSV bytes are identical with the hook on and off, and
the live stamp gives the same facets as the backfill for the same input.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import m5_setup_key_stamp as stamp  # noqa: E402
import market_calendar  # noqa: E402
import project_paths  # noqa: E402
import setup_permutation_backfill as bf  # noqa: E402
import setup_permutation_context as spc  # noqa: E402
import setup_permutations as sp  # noqa: E402


def _sessions(count: int) -> list[date]:
    day = date(2026, 8, 3)
    out = []
    while len(out) < count:
        if market_calendar.is_session(day):
            out.append(day)
        day += timedelta(days=1)
    return out


SESSIONS = _sessions(4)
PREV, TODAY = SESSIONS[2], SESSIONS[3]
SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]


def _write_csv(path: Path, rows: list[dict]) -> Path:
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _scan(session: date, run: str, symbol: str, side: str, **extra) -> dict:
    row = {
        "run_id": f"run-{run}",
        "run_timestamp": run,
        "run_date": session.isoformat(),
        "last_trade_date": session.isoformat(),
        "symbol": symbol,
        "side": side,
        "setup_family": "avwap_band_bounce",
        "last_close": 100.0,
        "atr20": 2.0,
        "relvol": 2.0,
        "current_band_zone": "VWAP to UPPER_1",
    }
    row.update(extra)
    return row


def _history_rows() -> list[dict]:
    rows = []
    for session in SESSIONS[:2]:
        for symbol in SYMBOLS:
            rows.append(_scan(session, f"{session.isoformat()}T13:05:00", symbol, "LONG", relvol=0.4))
    for symbol in SYMBOLS:
        rows.append(_scan(PREV, f"{PREV.isoformat()}T13:05:00", symbol, "LONG", relvol=0.5))
        rows.append(_scan(PREV, f"{PREV.isoformat()}T13:05:00", symbol, "SHORT", relvol=0.5))
    # The next morning's pre-market scan still dates PREV: the later row wins.
    rows.append(_scan(PREV, f"{TODAY.isoformat()}T07:30:00", "AAA", "LONG", relvol=3.0))
    # Same-day intraday scan dated TODAY: never the D1 picture of today's alert.
    for symbol in SYMBOLS:
        rows.append(_scan(TODAY, f"{TODAY.isoformat()}T10:00:00", symbol, "LONG", relvol=9.0,
                          current_band_zone="UPPER_2 to UPPER_3"))
    return rows


def _m5_rows(day: date) -> list[dict]:
    rows = []
    for n, symbol in enumerate(SYMBOLS[:3]):
        direction = "short" if n == 2 else "long"
        event_id = f"{symbol}_{direction}_{day.strftime('%Y%m%d')}_07_00_00_ema_15"
        common = {"event_id": event_id, "trade_date": day.isoformat(), "symbol": symbol, "direction": direction,
                  "entry_time": f"{day.isoformat()}T07:00:00", "context_json": "{}"}
        rows.append({**common, "event_type": "registered", "bars_elapsed": 0, "minutes_elapsed": "",
                     "mfe_r": "", "stop_hit": "False", "logged_at": f"{day.isoformat()}T07:01:00-07:00"})
        rows.append({**common, "event_type": "update", "bars_elapsed": 8, "minutes_elapsed": 40,
                     "mfe_r": 1.5 + n, "stop_hit": "False", "logged_at": f"{day.isoformat()}T07:40:00-07:00"})
    return rows


@pytest.fixture()
def files(tmp_path):
    m5_prev = _m5_rows(PREV)
    m5_today = _m5_rows(TODAY)
    return {
        "history": _write_csv(tmp_path / "d1_features_history.csv", _history_rows()),
        "m5": _write_csv(tmp_path / "intraday_bounce_outcomes.csv", [*m5_prev, *m5_today]),
        "m5_today": m5_today,
    }


def _lookup(files) -> stamp.ScanKeyLookup:
    return stamp.ScanKeyLookup(
        files["history"],
        context_loader=lambda session: spc.SessionContext.load(session, m5_outcomes_path=files["m5"]),
    )


@pytest.fixture()
def sidecar(tmp_path, monkeypatch):
    path = tmp_path / "m5_setup_key_stamps.jsonl"
    monkeypatch.setattr(project_paths, "M5_SETUP_KEY_STAMPS_FILE", path)
    monkeypatch.setattr(stamp, "_market_today", lambda: TODAY.isoformat())
    monkeypatch.delenv(stamp.ENABLED_ENV, raising=False)
    yield path
    stamp.reset_for_tests()


# ---------------------------------------------------------------------------
# the rule: the previous session's LAST scan row, the backfill's rule
# ---------------------------------------------------------------------------
def test_the_stamp_takes_the_previous_session_s_last_scan_row(files):
    got = _lookup(files).stamp("AAA", "LONG", TODAY.isoformat())
    assert got["status"] == stamp.STATUS_STAMPED
    assert got["d1_session"] == PREV.isoformat()
    assert got["facets"]["relvol"] == sp.FACETS["relvol"].fn({"relvol": 3.0}, {}, "LONG")
    assert got["scan_row_id"] == f"AAA:{PREV.isoformat()}:run-{TODAY.isoformat()}T07:30:00"
    assert got["permutation_rule_version"] == sp.PERMUTATION_RULE_VERSION
    assert got["permutation_key"].startswith(f"{sp.PERMUTATION_RULE_VERSION}|avwap_band_bounce|LONG|")
    # the same-day intraday scan (UPPER_2 to UPPER_3) never reaches today's alert
    assert "upper2" not in got["permutation_key"]


def test_the_live_stamp_and_the_backfill_give_the_same_key(files):
    """Same history, same M5 log: the live stamp's facets equal the backfill's f_ columns."""
    stores = bf.ContextStores(m5_outcomes=files["m5"])
    result = bf.build_permutation_outcomes(
        files["history"], horizons=_write_csv(files["history"].with_name("h.csv"), [{"scan_row_id": "x"}]),
        m5_outcomes=files["m5"], stores=stores, last_completed=TODAY,
    )
    m5 = [row for row in result.rows if row["population"] == bf.POPULATION_M5 and row["session"] == TODAY.isoformat()]
    assert len(m5) == 3
    lookup = _lookup(files)
    for row in m5:
        live = lookup.stamp(row["symbol"], row["side"], row["session"])
        backfill = {name: row[bf.facet_column(name)] for name in sp.FACETS}
        assert live["facets"] == backfill, row["episode_id"]
    # the M5 ctx facet is read (not unknown), so the comparison covers a ctx source too
    assert {row["f_m5_confirmation"] for row in m5} != {sp.UNKNOWN}


def test_no_scan_row_is_a_blank_unknown_stamp(files):
    got = _lookup(files).stamp("ZZZ", "LONG", TODAY.isoformat())
    assert got["status"] == stamp.STATUS_NO_SCAN_ROW
    assert all(got[name] == "" for name in stamp.STAMP_FIELDS)
    assert "facets" not in got


def test_the_lookup_is_cached_until_the_history_file_changes(files):
    lookup = _lookup(files)
    lookup.stamp("AAA", "LONG", TODAY.isoformat())
    lookup.stamp("BBB", "LONG", TODAY.isoformat())
    assert lookup.loads == 1
    with files["history"].open("a", encoding="utf-8", newline="") as handle:
        handle.write("\n")
    time.sleep(0.01)
    lookup.stamp("BBB", "LONG", TODAY.isoformat())
    assert lookup.loads == 2


def test_a_failed_lookup_is_a_blank_stamp_with_one_logged_reason(files, caplog, monkeypatch):
    stamp.reset_for_tests()
    lookup = _lookup(files)

    def boom(session):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(lookup, "keys_for_session", boom)
    with caplog.at_level(logging.WARNING):
        first = lookup.stamp("AAA", "LONG", TODAY.isoformat())
        second = lookup.stamp("BBB", "LONG", TODAY.isoformat())
    assert first["status"] == second["status"] == stamp.STATUS_FAILED
    assert "disk on fire" in first["reason"]
    assert all(first[name] == "" for name in stamp.STAMP_FIELDS)
    assert sum("disk on fire" in record.getMessage() for record in caplog.records) == 1


def test_the_history_tail_is_capped(files):
    lookup = stamp.ScanKeyLookup(files["history"], context_loader=lambda s: spc.SessionContext(),
                                 max_tail_bytes=10)
    got = lookup.stamp("AAA", "LONG", TODAY.isoformat())
    assert got["status"] == stamp.STATUS_FAILED
    assert "TailReadCapped" in got["reason"]


# ---------------------------------------------------------------------------
# the hook: never blocks, never raises, one record per event
# ---------------------------------------------------------------------------
def _outcome_row(symbol="AAA", direction="long", day=None, suffix="ema_15"):
    day = day or TODAY
    return {
        "event_id": f"{symbol}_{direction}_{day.strftime('%Y%m%d')}_07_00_00_{suffix}",
        "event_type": "registered",
        "trade_date": day.isoformat(),
        "symbol": symbol,
        "direction": direction,
    }


def test_submit_writes_one_record_per_event_to_the_sidecar(files, sidecar):
    stamp.reset_for_tests(_lookup(files))
    row = _outcome_row()
    before = dict(row)
    assert stamp.submit(row) is True
    assert stamp.submit({**row, "event_type": "update"}) is False  # same event: stamped once
    assert stamp.submit(_outcome_row(day=PREV)) is False  # an old session's row: left to the backfill
    assert stamp.drain()
    assert row == before
    records = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["event_id"] == row["event_id"]
    assert records[0]["schema"] == stamp.SCHEMA
    assert records[0]["status"] == stamp.STATUS_STAMPED
    assert records[0]["side"] == "LONG"
    assert records[0]["stamped_at"].endswith("+00:00")


def test_submit_never_waits_for_a_slow_lookup(files, sidecar):
    release = threading.Event()

    class Slow(stamp.ScanKeyLookup):
        def keys_for_session(self, session):
            release.wait(5.0)
            return {}

    stamp.reset_for_tests(Slow(files["history"]))
    started = time.perf_counter()
    for n in range(50):
        stamp.submit(_outcome_row(symbol=f"S{n:02d}"))
    elapsed = time.perf_counter() - started
    release.set()
    assert stamp.drain()
    assert elapsed < 0.25, f"50 submits took {elapsed:.3f}s while the lookup was stuck"


def test_a_full_queue_drops_the_stamp_not_the_caller(files, sidecar, monkeypatch):
    import queue

    stamp.reset_for_tests(_lookup(files))
    tiny: queue.Queue = queue.Queue(maxsize=1)
    monkeypatch.setattr(stamp, "_queue", tiny)
    monkeypatch.setattr(stamp, "_ensure_worker", lambda: None)  # nobody drains it
    assert stamp.submit(_outcome_row(symbol="AAA")) is True
    assert stamp.submit(_outcome_row(symbol="BBB")) is False


def test_the_hook_can_be_switched_off(files, sidecar, monkeypatch):
    stamp.reset_for_tests(_lookup(files))
    monkeypatch.setenv(stamp.ENABLED_ENV, "0")
    assert stamp.submit(_outcome_row()) is False
    assert not sidecar.exists()


# ---------------------------------------------------------------------------
# golden: the M5 writer's CSV bytes are identical with the hook on and off
# ---------------------------------------------------------------------------
class _Writer:
    """A minimal host for the one function that writes an outcome row (real CSV writer)."""


def _host():
    from bounce_bot_lib.legacy import BounceBot

    host = _Writer.__new__(_Writer)
    host.pending_bounce_outcomes = {}
    for name in ("_append_learning_row", "_learning_csv_header", "_parse_bar_time", "_json_for_learning",
                 "_context_with_finalization", "_exit_facts", "_completed_session_rows",
                 "_rows_after_bounce_entry_for_session", "_append_bounce_outcome_row"):
        setattr(host, name, getattr(BounceBot, name).__get__(host, _Writer))
    host._mirror_outcome_row_to_ledger = lambda row, state: None
    host._naive_market_local = BounceBot._naive_market_local
    host.OUTCOME_BAR_MINUTES = BounceBot.OUTCOME_BAR_MINUTES
    return host


def _state(symbol: str, direction: str) -> dict:
    return {
        "event_id": f"{symbol}_{direction}_{TODAY.strftime('%Y%m%d')}_07_00_00_ema_15",
        "symbol": symbol,
        "direction": direction,
        "trade_date": TODAY.isoformat(),
        "entry_time": f"{TODAY.isoformat()}T07:00:00",
        "entry_price": 100.0,
        "stop_price": 99.0 if direction == "long" else 101.0,
        "risk_per_share": 1.0,
        "target_1r": 101.0 if direction == "long" else 99.0,
        "target_2r": 102.0 if direction == "long" else 98.0,
        "milestones_logged": [],
        "outcome_mode": "eod_hold",
        "context": {"tier": "B"},
    }


def _bars() -> pd.DataFrame:
    return pd.DataFrame([
        {"datetime": pd.Timestamp(f"{TODAY.isoformat()} 07:{5 * (i + 1):02d}:00"),
         "open": 100.0, "high": 100.5 + i, "low": 99.5, "close": 100.2 + i, "volume": 1000.0}
        for i in range(3)
    ])


def _write_session(tmp_path: Path, name: str, monkeypatch, *, seed_header: list[str] | None) -> tuple[bytes, list]:
    import bounce_bot_lib.legacy as legacy

    target = tmp_path / name
    if seed_header is not None:
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=seed_header)
            writer.writeheader()
            writer.writerow({column: f"old-{column}" for column in seed_header})
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOMES_CSV", target)
    host = _host()
    statuses = []
    for symbol, direction in (("AAA", "long"), ("BBB", "long"), ("CCC", "short")):
        state = _state(symbol, direction)
        statuses.append(host._append_bounce_outcome_row(state, "registered", 0, None, pd.DataFrame()))
        statuses.append(host._append_bounce_outcome_row(state, "milestone", 3, 3, _bars()))
        statuses.append(host._append_bounce_outcome_row(state, "final", 3, None, _bars(), finalize_eod=True))
        statuses.append(json.dumps(state, sort_keys=True, default=str))  # the bot's state after the writes
    stamp.drain()
    return target.read_bytes(), statuses


@pytest.mark.parametrize("seed", ["new_file", "current_header", "old_header_needs_migration"])
def test_golden_outcome_csv_bytes_identical_with_the_hook_on_and_off(files, sidecar, tmp_path, monkeypatch, seed):
    import bounce_bot_lib.legacy as legacy

    frozen = pd.Timestamp(f"{TODAY.isoformat()} 13:00:00").to_pydatetime()
    monkeypatch.setattr(legacy, "get_market_local_now", lambda: frozen)
    header = {
        "new_file": None,
        "current_header": list(legacy.BOUNCE_OUTCOME_COLUMNS),
        "old_header_needs_migration": [c for c in legacy.BOUNCE_OUTCOME_COLUMNS if c not in {"mfe_pct", "mae_pct"}],
    }[seed]

    stamp.reset_for_tests(_lookup(files))
    monkeypatch.setenv(stamp.ENABLED_ENV, "0")
    off_bytes, off_status = _write_session(tmp_path, "off.csv", monkeypatch, seed_header=header)
    assert not sidecar.exists()

    stamp.reset_for_tests(_lookup(files))
    monkeypatch.setenv(stamp.ENABLED_ENV, "1")
    on_bytes, on_status = _write_session(tmp_path, "on.csv", monkeypatch, seed_header=header)

    assert on_bytes == off_bytes
    assert on_status == off_status
    first_line = on_bytes.split(b"\r\n", 1)[0].decode()
    assert "permutation" not in first_line  # the outcome CSV header never grows
    records = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["event_id"] for r in records) == sorted(_state(s, d)["event_id"]
                                                          for s, d in (("AAA", "long"), ("BBB", "long"),
                                                                       ("CCC", "short")))
    assert {r["status"] for r in records} == {stamp.STATUS_STAMPED}


def test_a_hook_that_raises_never_costs_the_row(files, sidecar, tmp_path, monkeypatch):
    import bounce_bot_lib.legacy as legacy

    def boom(row):
        raise RuntimeError("stamp exploded")

    monkeypatch.setattr(legacy.m5_setup_key_stamp, "submit", boom)
    target = tmp_path / "rows.csv"
    monkeypatch.setattr(legacy, "INTRADAY_BOUNCE_OUTCOMES_CSV", target)
    status = _host()._append_bounce_outcome_row(_state("AAA", "long"), "registered", 0, None, pd.DataFrame())
    assert status == "open"
    assert len(list(csv.DictReader(target.open(encoding="utf-8")))) == 1


def test_the_outcome_csv_columns_are_unchanged():
    """The stamp lives in the sidecar; readers of the outcome CSV see the same columns."""
    import bounce_bot_lib.legacy as legacy

    assert not any("permutation" in column for column in legacy.BOUNCE_OUTCOME_COLUMNS)
    assert len(legacy.BOUNCE_OUTCOME_COLUMNS) == 29


# ---------------------------------------------------------------------------
# the backfill joins the sidecar on event_id
# ---------------------------------------------------------------------------
def test_the_backfill_joins_the_live_stamp_on_event_id(files, tmp_path):
    lookup = _lookup(files)
    sidecar = tmp_path / "stamps.jsonl"
    for row in files["m5_today"]:
        if row["event_type"] == "registered":
            stamp.append_record(stamp.record_for(row, lookup), sidecar)
    horizons = _write_csv(tmp_path / "h.csv", [{"scan_row_id": "x"}])
    result = bf.build_permutation_outcomes(
        files["history"], horizons=horizons, m5_outcomes=files["m5"],
        stores=bf.ContextStores(m5_outcomes=files["m5"]), last_completed=TODAY, m5_stamps=sidecar,
    )
    assert result.counts["m5_live_stamped"] == 3
    assert result.counts["m5_live_stamp_disagreed"] == 0


def test_the_live_stamp_wins_and_an_event_without_one_is_the_recompute(files, tmp_path):
    sidecar = tmp_path / "stamps.jsonl"
    event = files["m5_today"][0]["event_id"]
    forced = {name: sp.UNKNOWN for name in sp.FACETS}
    forced["relvol"] = "relvol_live_marker"
    stamp.append_record({"schema": stamp.SCHEMA, "event_id": event, "status": stamp.STATUS_STAMPED,
                         "permutation_rule_version": sp.PERMUTATION_RULE_VERSION, "facets": forced}, sidecar)
    stamp.append_record({"schema": stamp.SCHEMA, "event_id": event, "status": stamp.STATUS_STAMPED,
                         "permutation_rule_version": sp.PERMUTATION_RULE_VERSION, "facets": {}}, sidecar)
    horizons = _write_csv(tmp_path / "h.csv", [{"scan_row_id": "x"}])
    result = bf.build_permutation_outcomes(files["history"], horizons=horizons, m5_outcomes=files["m5"],
                                           last_completed=TODAY, m5_stamps=sidecar)
    by_id = {row["episode_id"]: row for row in result.rows if row["population"] == bf.POPULATION_M5}
    assert by_id[event]["f_relvol"] == "relvol_live_marker"  # the first record per event wins
    other = files["m5_today"][2]["event_id"]
    assert by_id[other]["f_relvol"] != "relvol_live_marker"
    assert result.counts["m5_live_stamped"] == 1


def test_old_outcome_rows_without_a_stamp_or_a_scan_row_are_unknown(files, tmp_path):
    history = _write_csv(tmp_path / "empty_history.csv", [_scan(SESSIONS[0], "x", "QQQ", "LONG")])
    horizons = _write_csv(tmp_path / "h.csv", [{"scan_row_id": "x"}])
    result = bf.build_permutation_outcomes(history, horizons=horizons, m5_outcomes=files["m5"],
                                           last_completed=TODAY, m5_stamps=tmp_path / "missing.jsonl")
    m5 = [row for row in result.rows if row["population"] == bf.POPULATION_M5]
    assert m5 and all(row[bf.facet_column(name)] == sp.UNKNOWN for row in m5 for name in sp.FACETS)
    assert result.counts["m5_live_stamped"] == 0


def test_the_tail_reader_returns_the_session_s_rows_in_file_order(files):
    rows = spc._tail_rows_for_session(files["history"], PREV.isoformat(),
                                      date_columns=("last_trade_date", "run_date"))
    assert {row["last_trade_date"] for row in rows} == {PREV.isoformat()}
    assert len(rows) == 9
    assert rows[-1]["run_timestamp"] == f"{TODAY.isoformat()}T07:30:00"
