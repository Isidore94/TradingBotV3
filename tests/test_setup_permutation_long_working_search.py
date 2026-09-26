"""S15 item 4 - the conditional search: longs inside a "working" market, judged RAW.

The trader 2026-09-26: longs need the market on their side; judge longs raw inside the regime.
The backfill carries each swing row's raw win (side return > 0, not vs SPY) and its entry day's
`long_regime_working` verdict (the scan's stamp, else recomputed: the trader's regime first, then
SPY above a rising 20-day). The search reports LONG rows whose verdict is "yes" as their own
population, win = raw, facets as today, each horizon named and the deciding rules counted.
"""

from __future__ import annotations

import csv
import importlib.util
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_calendar  # noqa: E402
import setup_permutation_backfill as bf  # noqa: E402
import setup_permutation_search as search  # noqa: E402
import setup_permutations as sp  # noqa: E402

POP = search.POPULATION_LONG_WORKING_RAW


def _backfill_fixture():
    path = Path(__file__).with_name("test_setup_permutation_backfill.py")
    spec = importlib.util.spec_from_file_location("_s15_4_backfill_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sessions_between(first: date, last: date) -> list[date]:
    out, day = [], first
    while day <= last:
        if market_calendar.is_session(day):
            out.append(day)
        day += timedelta(days=1)
    return out


def _write_csv(path, rows):
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture()
def files(tmp_path):
    fx = _backfill_fixture()
    features = fx._feature_rows()
    # The newest session's rows carry the live stamp (S15 item 2): "no" by the trader's rule.
    stamped_day = fx.SESSIONS[-1].isoformat()
    for row in features:
        if row["last_trade_date"] == stamped_day:
            row["perm_regime_working"] = "no"
            row["perm_regime_working_rule"] = sp.WORKING_RULE_TRADER
    horizons = fx._horizon_rows(features)
    for row in horizons:
        if row["side_return_pct"] != "":
            # A falling raw return on odd symbols: raw loses there even though SPY is flat.
            row["side_return_pct"] = -0.5 if int(row["scan_row_id"][1:3]) % 2 else 1.0
    spy_days = _sessions_between(date(2026, 6, 1), fx.SESSIONS[-1])
    spy = [{"datetime": day.isoformat(), "close": 400.0 + index} for index, day in enumerate(spy_days)]
    return {
        "fx": fx,
        "features": _write_csv(tmp_path / "d1_features_history.csv", features),
        "horizons": _write_csv(tmp_path / "session_horizons.csv", horizons),
        "spy": _write_csv(tmp_path / "SPY.csv", spy),
    }


def _journal(tmp_path, *segments):
    from journal_store import JournalStore

    path = tmp_path / "journal_copy.sqlite3"
    store = JournalStore(path)
    stamp = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    for start, regime in segments:
        store.append_structural_regime(start_date=start, regime=regime, entered_at=stamp)
    return path


# ---------------------------------------------------------------------------
# backfill: raw win and the entry day's regime
# ---------------------------------------------------------------------------
def test_swing_rows_carry_the_raw_win_and_the_spy_rule_without_a_trader_regime(files):
    fx = files["fx"]
    result = bf.build_permutation_outcomes(files["features"], horizons=files["horizons"], spy_bars=files["spy"])
    swing = [row for row in result.rows if row["population"] == bf.POPULATION_SWING]
    assert swing
    for row in swing:
        assert row["raw_win"] is (int(row["symbol"][1:]) % 2 == 0)
        if row["session"] == fx.SESSIONS[-1].isoformat():
            assert (row["regime_working"], row["regime_working_rule"]) == ("no", sp.WORKING_RULE_TRADER)
        else:
            # A rising SPY and no trader label: SPY above a rising 20-day.
            assert (row["regime_working"], row["regime_working_rule"]) == ("yes", sp.WORKING_RULE_SPY)
    assert result.counts["regime_rule_trader"] > 0 and result.counts[f"regime_rule_{sp.WORKING_RULE_SPY}"] > 0


def test_the_trader_regime_decides_before_spy(files, tmp_path):
    fx = files["fx"]
    middle = fx.SESSIONS[5].isoformat()
    journal = _journal(tmp_path, ("2026-07-01", "bull_run"), (middle, "bear_channel_lower_highs"))
    result = bf.build_permutation_outcomes(files["features"], horizons=files["horizons"], spy_bars=files["spy"],
                                           structural_regime=journal)
    for row in (row for row in result.rows if row["population"] == bf.POPULATION_SWING):
        if row["session"] == fx.SESSIONS[-1].isoformat():
            assert row["regime_working"] == "no"  # the live stamp stands
        elif row["session"] < middle:
            assert (row["regime_working"], row["regime_working_rule"]) == ("yes", sp.WORKING_RULE_TRADER)
        else:
            assert (row["regime_working"], row["regime_working_rule"]) == ("no", sp.WORKING_RULE_TRADER)


def test_without_spy_history_the_regime_is_unknown(files):
    result = bf.build_permutation_outcomes(files["features"], horizons=files["horizons"],
                                           spy_closes={files["fx"].SESSIONS[0].isoformat(): 500.0})
    # With one SPY close the tape is unknown too, so no swing row is written; the keyed rows say unknown.
    assert result.counts.get(f"regime_rule_{sp.UNKNOWN}", 0) > 0


def test_the_journal_copy_must_not_be_live(files):
    with pytest.raises(bf.LiveStoreRefused):
        bf.build_permutation_outcomes(files["features"], horizons=files["horizons"], spy_bars=files["spy"],
                                      structural_regime=Path(r"C:\TradingBotData\data\runtime\trade_journal.sqlite3"))


def test_the_parquet_keeps_raw_win_as_a_nullable_bool(files, tmp_path):
    import pyarrow.parquet as pq

    result = bf.build_permutation_outcomes(files["features"], horizons=files["horizons"], spy_bars=files["spy"])
    table = pq.read_table(bf.write_parquet(result.rows, tmp_path / "o.parquet"))
    assert str(table.schema.field("raw_win").type) == "bool"
    assert {"raw_win", "regime_working", "regime_working_rule"} <= set(table.column_names)


# ---------------------------------------------------------------------------
# search: its own population, raw win, horizon named, rules counted
# ---------------------------------------------------------------------------
SESSIONS = [(date(2026, 1, 5) + timedelta(days=i)).isoformat() for i in range(60)]


def _rows(seed=3, per_session=30):
    rng = random.Random(seed)
    rows = []
    for day in SESSIONS:
        for n in range(per_session):
            side = "LONG" if n % 5 else "SHORT"
            working = "yes" if n % 3 else "no"
            key_on = rng.random() < 0.3
            raw = rng.random() < (0.9 if key_on else 0.35)
            rows.append({
                "population": "swing", "family": "avwap_band_bounce", "side": side, "horizon": 5,
                "horizon_name": "5_sessions", "session": day, "episode_id": f"{day}:{n}",
                # The tape win disagrees with the raw one on purpose.
                "win": not raw, "r": 1.0 if raw else -1.0, "raw_win": raw,
                "regime_working": working,
                "regime_working_rule": sp.WORKING_RULE_TRADER if n % 2 else sp.WORKING_RULE_SPY,
                "f_ma_support": "sma100_support" if key_on else "no_ma_support",
                "f_noise": rng.choice(["a", "b"]),
            })
    return rows


def test_the_conditional_population_is_working_longs_on_the_raw_win():
    rows = _rows()
    rows.append({**rows[1], "raw_win": None, "episode_id": "no-raw"})
    chosen = search.long_working_raw_rows(rows)
    assert chosen and all(row["population"] == POP for row in chosen)
    source = [row for row in rows if row["side"] == "LONG" and row["regime_working"] == "yes"
              and row["raw_win"] is not None]
    assert len(chosen) == len(source)
    assert all(row["win"] is row["raw_win"] for row in chosen)
    assert all(row["side"] == "LONG" and row["regime_working"] == "yes" for row in chosen)


def test_the_report_has_its_own_population_with_the_horizon_named(tmp_path):
    report = search.build_report(_rows(), ledger_root=tmp_path)
    assert set(report["populations"]) == {"swing", POP}
    block = report["populations"][POP]
    assert block["definition"] == search.LONG_WORKING_RAW_DEFINITION
    assert "raw" in block["definition"] and "not vs SPY" in block["definition"]
    horizon = block["horizons"]["5"]
    assert horizon["horizon_name"] == "5_sessions"
    assert set(horizon["families"]) == {"avwap_band_bounce LONG"}
    family = horizon["families"]["avwap_band_bounce LONG"]
    assert family["horizon_name"] == "5_sessions"
    # The planted raw key is found inside the working population.
    assert family["verdict"] == search.VERDICT_KEY
    assert family["keys"][0]["facets"] == {"ma_support": "sma100_support"}
    # Which rule decided "working", counted per horizon.
    rules = horizon["working_rules"]
    assert set(rules) == {sp.WORKING_RULE_TRADER, sp.WORKING_RULE_SPY}
    assert sum(rules.values()) == sum(1 for row in search.long_working_raw_rows(_rows()))
    # The ordinary swing population is untouched: tape win, both sides.
    assert set(report["populations"]["swing"]["horizons"]["5"]["families"]) == {
        "avwap_band_bounce LONG", "avwap_band_bounce SHORT"}


def test_rows_without_the_regime_columns_add_no_population(tmp_path):
    rows = [{key: value for key, value in row.items() if key not in ("raw_win", "regime_working")}
            for row in _rows()]
    report = search.build_report(rows, ledger_root=tmp_path)
    assert set(report["populations"]) == {"swing"}
