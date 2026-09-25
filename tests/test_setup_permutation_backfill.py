"""P1-4 / 4b - the scratch-only permutation backfill, on a 200-row fixture."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_calendar  # noqa: E402
import setup_permutation_backfill as bf  # noqa: E402
import setup_permutations as sp  # noqa: E402

SYMBOLS = [f"S{n:02d}" for n in range(18)]


def _sessions(count: int) -> list[date]:
    day = date(2026, 8, 3)
    out = []
    while len(out) < count:
        if market_calendar.is_session(day):
            out.append(day)
        day += timedelta(days=1)
    return out


SESSIONS = _sessions(10)


def _write_csv(path: Path, rows: list[dict]) -> Path:
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _feature_rows() -> list[dict]:
    rows = []
    for s_index, session in enumerate(SESSIONS):
        for n, symbol in enumerate(SYMBOLS):
            base = {
                "run_id": f"run-{session.isoformat()}-close",
                "run_timestamp": f"{session.isoformat()}T13:05:00",
                "run_date": session.isoformat(),
                "last_trade_date": session.isoformat(),
                "symbol": symbol,
                "side": "LONG" if n % 3 else "SHORT",
                "setup_family": "avwap_band_bounce" if n % 2 else "post_earnings_candle_break",
                "last_close": 100.0 + n,
                "atr20": 2.0,
                "relvol": 2.0,
                "current_band_zone": "VWAP to UPPER_1",
                "priority_bucket": "favorite_setup",
            }
            rows.append(base)
    # A second, EARLIER scan for two sessions: the collapse must keep the later one.
    for symbol in SYMBOLS[:10]:
        for session in SESSIONS[:2]:
            rows.insert(0, {
                "run_id": f"run-{session.isoformat()}-open",
                "run_timestamp": f"{session.isoformat()}T07:35:00",
                "run_date": session.isoformat(),
                "last_trade_date": session.isoformat(),
                "symbol": symbol,
                "side": "LONG" if int(symbol[1:]) % 3 else "SHORT",
                "setup_family": "avwap_band_bounce",
                "last_close": 90.0,
                "atr20": 2.0,
                "relvol": 0.5,  # the earlier scan's relvol; must not reach the table
                "current_band_zone": "VWAP to UPPER_1",
                "priority_bucket": "favorite_setup",
            })
    return rows


def _horizon_rows(features: list[dict]) -> list[dict]:
    out = []
    latest = {}
    for row in features:
        latest[(row["symbol"], row["side"], row["last_trade_date"])] = row
    for row in latest.values():
        for horizon in bf.SWING_HORIZONS:
            measured = horizon != 10
            out.append({
                "scan_row_id": f"{row['symbol']}:{row['last_trade_date']}:{row['run_id']}",
                "horizon_sessions": horizon,
                "measured": measured,
                "favorable": (int(row["symbol"][1:]) + horizon) % 2 == 0 if measured else "",
                "side_return_pct": 1.0 if measured else "",
                "entry_close": row["last_close"],
            })
    return out


def _m5_rows() -> list[dict]:
    rows = []
    day = SESSIONS[5]
    for n, symbol in enumerate(SYMBOLS[:4]):
        direction = "long" if n % 3 else "short"
        event_id = f"{symbol}_{direction}_{day.strftime('%Y%m%d')}_07_00_00_ema_15"
        common = {"event_id": event_id, "trade_date": day.isoformat(), "symbol": symbol, "direction": direction,
                  "entry_time": f"{day.isoformat()}T07:00:00", "context_json": "{}"}
        rows.append({**common, "event_type": "registered", "bars_elapsed": 0, "minutes_elapsed": "",
                     "mfe_r": "", "stop_hit": "False", "logged_at": f"{day.isoformat()}T07:01:00-07:00"})
        rows.append({**common, "event_type": "update", "bars_elapsed": 2 if n == 0 else 8,
                     "minutes_elapsed": 10 if n == 0 else 40,
                     "mfe_r": 1.5 + n, "stop_hit": "True" if n == 0 else "False",
                     "logged_at": f"{day.isoformat()}T07:40:00-07:00"})
    return rows


@pytest.fixture()
def fixture_files(tmp_path):
    features = _feature_rows()
    assert len(features) == 200
    return {
        "features": _write_csv(tmp_path / "d1_features_history.csv", features),
        "horizons": _write_csv(tmp_path / "session_horizons.csv", _horizon_rows(features)),
        "m5": _write_csv(tmp_path / "intraday_bounce_outcomes.csv", _m5_rows()),
        "raw": features,
    }


def test_refuses_a_live_input_or_output(tmp_path):
    with pytest.raises(bf.LiveStoreRefused):
        bf.build_permutation_outcomes(Path(r"C:\TradingBotData\data\runtime\d1_features_history.csv"),
                                      horizons=tmp_path / "h.csv")
    with pytest.raises(bf.LiveStoreRefused):
        bf.write_parquet([], Path(r"C:\TradingBotData\permutation_outcomes.parquet"))
    code = bf.main(["--scratch", str(tmp_path), "--features", str(tmp_path / "f.csv"),
                    "--horizons", r"C:\TradingBotData\data\runtime\master_avwap_session_horizon_outcomes.csv",
                    "--out", str(tmp_path / "o.parquet")])
    assert code == 2
    assert not (tmp_path / "o.parquet").exists()


def test_the_backfill_keys_every_episode_horizon_as_of_the_scan_date(fixture_files, tmp_path):
    import pyarrow.parquet as pq

    result = bf.build_permutation_outcomes(fixture_files["features"], horizons=fixture_files["horizons"],
                                           m5_outcomes=fixture_files["m5"], last_completed=SESSIONS[-1])
    swing = [row for row in result.rows if row["population"] == bf.POPULATION_SWING]
    assert result.counts["scan_rows_keyed"] == len(SYMBOLS) * len(SESSIONS)
    # 3 measured horizons per representative; horizon 10 is unmeasured and absent.
    assert len(swing) == len(SYMBOLS) * len(SESSIONS) * 3
    assert {row["horizon"] for row in swing} == {1, 3, 5}
    # The session's LAST scan row speaks for it: relvol 2.0, never the 07:35 row's 0.5.
    assert {row["f_relvol"] for row in swing} == {"relvol_1_5_3"}
    assert all(row["f_band_zone"] == "vwap_upper1" for row in swing)
    assert all(row["permutation_rule_version"] == sp.PERMUTATION_RULE_VERSION for row in swing)
    one = next(row for row in swing if row["symbol"] == "S01" and row["horizon"] == 1)
    assert one["r"] == pytest.approx(1.0 / (2.0 / 101.0 * 100.0))
    assert one["win"] is ((1 + 1) % 2 == 0)

    out = bf.write_parquet(result.rows, tmp_path / "permutation_outcomes.parquet")
    table = pq.read_table(out)
    assert table.num_rows == len(result.rows)
    assert table.column_names == bf.output_columns()
    assert all(bf.facet_column(name) in table.column_names for name in sp.FACETS)


def test_m5_rows_take_the_previous_session_scan_never_the_same_day(fixture_files):
    result = bf.build_permutation_outcomes(fixture_files["features"], horizons=fixture_files["horizons"],
                                           m5_outcomes=fixture_files["m5"], last_completed=SESSIONS[-1])
    m5 = [row for row in result.rows if row["population"] == bf.POPULATION_M5]
    assert len(m5) == 4
    assert {row["family"] for row in m5} == {"ema_15"}
    assert {row["session"] for row in m5} == {SESSIONS[5].isoformat()}
    held = {row["symbol"]: row["win"] for row in m5}
    assert held == {"S00": False, "S01": True, "S02": True, "S03": True}
    assert all(row["horizon"] == bf.M5_HORIZON and row["r_unit"] == "mfe_r" for row in m5)
    assert all(row["f_weekday"] == SESSIONS[4].strftime("%a").lower() for row in m5)


def test_daily_bars_rebuild_matches_the_session_horizon_file(fixture_files, tmp_path):
    bars = tmp_path / "daily_bars"
    bars.mkdir()
    for n, symbol in enumerate(SYMBOLS):
        rows = []
        for index, day in enumerate(_sessions(30)):
            rows.append({"datetime": day.isoformat(), "open": 1, "high": 1, "low": 1,
                         "close": 100.0 + n + (0.5 if index % 2 else -0.5)})
        _write_csv(bars / f"{symbol}.csv", rows)
    result = bf.build_permutation_outcomes(fixture_files["features"], daily_bars=bars,
                                           last_completed=_sessions(30)[-1])
    swing = result.rows
    assert len(swing) == len(SYMBOLS) * len(SESSIONS) * len(bf.SWING_HORIZONS)
    for row in swing:
        assert row["win"] in (True, False)
        assert row["r"] is not None


def test_the_cli_writes_the_parquet_in_a_child_with_scratch_roots(fixture_files, tmp_path):
    out = tmp_path / "out" / "permutation_outcomes.parquet"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "setup_permutation_backfill.py"), "--scratch", str(tmp_path / "scratch"),
         "--features", str(fixture_files["features"]), "--horizons", str(fixture_files["horizons"]),
         "--m5-outcomes", str(fixture_files["m5"]), "--last-completed", SESSIONS[-1].isoformat(),
         "--out", str(out)],
        capture_output=True, text=True, timeout=600, env=environment, cwd=str(SCRIPTS_DIR),
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    assert summary["swing_rows"] == len(SYMBOLS) * len(SESSIONS) * 3
    assert summary["m5_rows"] == 4
    assert out.is_file()


def test_ctx_facets_come_from_the_copied_stores_for_the_row_s_own_session(fixture_files, tmp_path):
    events = tmp_path / "review_events.jsonl"
    events.write_text(json.dumps({"action": "watch_fired", "trade_date": SESSIONS[5].isoformat(),
                                  "ts": f"{SESSIONS[5].isoformat()}T08:00:00", "symbol": "S01", "side": "LONG",
                                  "detail": {"kind": "pullback", "trigger": "sma_retest"}}) + "\n",
                      encoding="utf-8")
    stores = bf.ContextStores(review_events=events, m5_outcomes=fixture_files["m5"])
    result = bf.build_permutation_outcomes(fixture_files["features"], horizons=fixture_files["horizons"],
                                           stores=stores, last_completed=SESSIONS[-1])
    rows = {(row["symbol"], row["session"]): row for row in result.rows if row["horizon"] == 1}
    same_day = rows[("S01", SESSIONS[5].isoformat())]
    assert same_day["f_entry_trigger"] == "pullback_sma_retest"
    assert same_day["f_m5_confirmation"] == "m5_ema_15"
    other_day = rows[("S01", SESSIONS[6].isoformat())]
    assert other_day["f_entry_trigger"] == "no_trigger"
    assert other_day["f_m5_confirmation"] == "no_m5_confirmation"
    assert other_day["f_discovery_slot"] == sp.UNKNOWN  # no scan reports copied: unknown
