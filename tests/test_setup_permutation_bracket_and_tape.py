"""TODO S3 (bracket_1r horizon in the M5 permutation search) and S4 (tape-relative swing search)."""

from __future__ import annotations

import csv
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for entry in (str(ROOT_DIR), str(SCRIPTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import market_calendar  # noqa: E402
import setup_permutation_backfill as bf  # noqa: E402
import setup_permutation_search as search  # noqa: E402
import setup_permutations as sp  # noqa: E402
from ai_jobs import setup_keys_narration as narration  # noqa: E402
from research_warehouse import trial_ledger  # noqa: E402


def _sessions(count: int, start: date = date(2026, 8, 3)) -> list[date]:
    day, out = start, []
    while len(out) < count:
        if market_calendar.is_session(day):
            out.append(day)
        day += timedelta(days=1)
    return out


SESSIONS = _sessions(8)
ALERT_DAY = SESSIONS[5]


def _write_csv(path: Path, rows: list[dict]) -> Path:
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _features() -> list[dict]:
    rows = []
    for session in SESSIONS:
        for symbol, side in (("S01", "LONG"), ("S03", "SHORT"), ("S04", "LONG"), ("S06", "SHORT")):
            rows.append({
                "run_id": f"run-{session.isoformat()}", "run_timestamp": f"{session.isoformat()}T13:05:00",
                "run_date": session.isoformat(), "last_trade_date": session.isoformat(), "symbol": symbol,
                "side": side, "setup_family": "avwap_band_bounce", "last_close": 100.0, "atr20": 2.0,
                "relvol": 2.0, "current_band_zone": "VWAP to UPPER_1", "priority_bucket": "favorite_setup",
            })
    return rows


def _event_id(symbol: str, direction: str) -> str:
    return f"{symbol}_{direction}_{ALERT_DAY.strftime('%Y%m%d')}_07_00_00_ema_15"


def _outcome(symbol, direction, event_type, bars, *, stop=False, t1=False, close_r="", eod=""):
    day = ALERT_DAY.isoformat()
    return {
        "event_id": _event_id(symbol, direction), "event_type": event_type, "trade_date": day, "symbol": symbol,
        "direction": direction, "entry_time": f"{day}T07:00:00", "entry_price": 100.0, "risk_per_share": 1.0,
        "bars_elapsed": bars, "minutes_elapsed": bars * 5, "mfe_r": 0.5, "close_r": close_r, "eod_close": eod,
        "stop_hit": str(stop), "target_1r_hit": str(t1), "target_2r_hit": "False",
        "logged_at": f"{day}T07:0{min(bars, 9)}:00-07:00", "context_json": "{}",
    }


def _six_events() -> tuple[list[dict], list[dict]]:
    """E1 wins, E2 loses (no D1 row), E3 ties = loss, E4 undecided, E5 not confirmed, E6 open but won."""
    outcomes = [
        _outcome("S01", "long", "registered", 0),
        _outcome("S01", "long", "update", 3, t1=True),
        _outcome("S01", "long", "final", 78, t1=True, close_r=0.8, eod=100.8),
        _outcome("S02", "long", "registered", 0),
        _outcome("S02", "long", "update", 2, stop=True),
        _outcome("S02", "long", "final", 78, stop=True, close_r=-1.5, eod=98.5),
        _outcome("S03", "short", "registered", 0),
        _outcome("S03", "short", "update", 4, stop=True, t1=True),
        _outcome("S03", "short", "final", 78, stop=True, t1=True, close_r=0.3, eod=99.7),
        _outcome("S04", "long", "registered", 0),
        _outcome("S04", "long", "update", 5),
        _outcome("S04", "long", "final", 78, close_r=0.2, eod=100.2),
        _outcome("S05", "long", "registered", 0),
        _outcome("S05", "long", "update", 3, t1=True),
        _outcome("S05", "long", "final", 78, t1=True, close_r=1.2, eod=101.2),
        _outcome("S06", "short", "registered", 0),
        _outcome("S06", "short", "update", 6, t1=True),
    ]
    candidates = [
        {"event_id": _event_id(symbol, direction), "event_type": kind, "symbol": symbol, "direction": direction}
        for symbol, direction, kind in (
            ("S01", "long", "detected"), ("S01", "long", "confirmed"), ("S02", "long", "confirmed"),
            ("S03", "short", "confirmed"), ("S04", "long", "confirmed"), ("S05", "long", "near_miss"),
            ("S06", "short", "confirmed"),
        )
    ]
    return outcomes, candidates


def _horizons(features: list[dict], side_return: float = 1.0) -> list[dict]:
    out = []
    for row in features:
        for horizon in (1, 5):
            target = date.fromisoformat(row["last_trade_date"])
            for _ in range(horizon):
                target = market_calendar.next_session(target)
            out.append({
                "scan_row_id": f"{row['symbol']}:{row['last_trade_date']}:{row['run_id']}",
                "side": row["side"], "scan_date": row["last_trade_date"], "target_session": target.isoformat(),
                "horizon_sessions": horizon, "measured": "True", "maturity": "mature",
                "favorable": "True", "side_return_pct": side_return, "entry_close": 100.0,
            })
    return out


@pytest.fixture()
def files(tmp_path):
    features = _features()
    outcomes, candidates = _six_events()
    return {
        "features": _write_csv(tmp_path / "d1_features_history.csv", features),
        "horizons": _write_csv(tmp_path / "horizons.csv", _horizons(features)),
        "m5": _write_csv(tmp_path / "intraday_bounce_outcomes.csv", outcomes),
        "candidates": _write_csv(tmp_path / "intraday_bounce_candidates.csv", candidates),
        "features_rows": features,
    }


# --- S3: the bracket horizon


def test_bracket_rows_come_from_every_decided_confirmed_alert(files, tmp_path):
    import pyarrow.parquet as pq

    result = bf.build_permutation_outcomes(
        files["features"], horizons=files["horizons"], m5_outcomes=files["m5"], m5_candidates=files["candidates"],
        last_completed=SESSIONS[-1], spy_closes={},
    )
    bracket = {row["symbol"]: row for row in result.rows if row.get("horizon_name") == bf.HORIZON_BRACKET_1R}
    # E4 is undecided, E5 was never confirmed: both are left out. E2 has no D1 scan row and stays.
    assert set(bracket) == {"S01", "S02", "S03", "S06"}
    assert {s: (row["win"], row["r"]) for s, row in bracket.items()} == {
        "S01": (True, 0.8), "S02": (False, -1.5), "S03": (False, 0.3), "S06": (True, None),
    }
    assert result.counts["m5_bracket_rows"] == 4
    for row in bracket.values():
        assert row["population"] == bf.POPULATION_M5 and row["horizon"] == bf.M5_HORIZON
        assert row["family"] == "ema_15" and row["outcome_kind"] == bf.BRACKET_OUTCOME_KIND
    # D1 facets are the PREVIOUS session's scan; the unkeyed alert keeps unknown facets.
    assert bracket["S01"]["f_weekday"] == SESSIONS[4].strftime("%a").lower()
    assert all(bracket["S02"][bf.facet_column(name)] == sp.UNKNOWN for name in sp.FACETS)
    held = [row for row in result.rows if row["population"] == bf.POPULATION_M5
            and row.get("horizon_name") != bf.HORIZON_BRACKET_1R]
    assert held and all(row["horizon_name"] == bf.HORIZON_HELD30 for row in held)

    table = pq.read_table(bf.write_parquet(result.rows, tmp_path / "out.parquet"))
    assert "horizon_name" in table.column_names
    assert set(table.column("horizon_name").to_pylist()) >= {bf.HORIZON_HELD30, bf.HORIZON_BRACKET_1R}


def test_bracket_input_refuses_a_live_candidates_path(files, tmp_path):
    live = r"C:\TradingBotData\data\runtime\intraday_bounce_candidates.csv"
    with pytest.raises(bf.LiveStoreRefused):
        bf.build_permutation_outcomes(files["features"], horizons=files["horizons"], m5_outcomes=files["m5"],
                                      m5_candidates=Path(live), spy_closes={})
    code = bf.main(["--scratch", str(tmp_path / "s"), "--features", str(files["features"]),
                    "--horizons", str(files["horizons"]), "--m5-outcomes", str(files["m5"]),
                    "--m5-candidates", live, "--out", str(tmp_path / "o.parquet")])
    assert code == 2
    assert not (tmp_path / "o.parquet").exists()


def _m5_population(name: str, seed: int, sessions: list[str]) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for day in sessions:
        for n in range(20):
            key_on = rng.random() < 0.3
            win = rng.random() < (0.85 if key_on else 0.4)
            rows.append({
                "population": "m5", "family": "ema_15", "side": "LONG", "horizon": 0, "horizon_name": name,
                "session": day, "episode_id": f"{name}:{day}:{n}", "win": win, "r": 1.0 if win else -1.0,
                "f_m5_time_bucket": "open" if key_on else "midday", "f_noise": rng.choice(["a", "b"]),
            })
    return rows


def test_the_search_runs_both_m5_horizons_and_names_them_everywhere(tmp_path):
    days = [(date(2026, 1, 5) + timedelta(days=i)).isoformat() for i in range(60)]
    rows = _m5_population("held30", 3, days) + _m5_population("bracket_1r", 4, days)
    report = search.build_report(rows, ledger_root=tmp_path)
    horizons = report["populations"]["m5"]["horizons"]
    assert set(horizons) == {"0", "bracket_1r"}  # held30 keeps its old key for the verdict history
    for key, name in (("0", "held30"), ("bracket_1r", "bracket_1r")):
        block = horizons[key]
        assert block["horizon_name"] == name
        family = block["families"]["ema_15 LONG"]
        assert family["horizon_name"] == name
        assert family["keys"], "the planted key must be found on both horizons"
        assert all(k["horizon_name"] == name for k in family["keys"])
    trial_ids = [str(row.get("trial_id")) for row in trial_ledger.load(tmp_path)]
    assert any(":hbracket_1r:" in tid for tid in trial_ids) and any(":h0:" in tid for tid in trial_ids)

    inputs = narration.build_inputs(report)
    assert {f["horizon_name"] for f in inputs["families"]} == {"held30", "bracket_1r"}
    for family in inputs["families"]:
        assert all(family["horizon_name"] in fact["text"] for fact in family["facts"])
    named = narration._named([{"text": "The open bucket held up.", "cites": []}], "bracket_1r")
    assert named[0]["text"].startswith("bracket_1r")


def test_the_setup_keys_panel_lists_a_named_horizon():
    pytest.importorskip("PySide6")
    from ui.panels import setup_keys_panel

    report = {"populations": {"m5": {"horizons": {"bracket_1r": {}, "0": {}}}}}
    assert setup_keys_panel.horizons_in(report, "m5") == ["0", "bracket_1r"]


# --- S4: tape-relative swing win and the 20-session selection floor


def test_swing_win_is_the_side_return_beating_spy(files, tmp_path):
    features = files["features_rows"]
    horizons = _write_csv(tmp_path / "h.csv", _horizons(features, side_return=1.0))
    rising = {day.isoformat(): 100.0 * (1.02 ** index) for index, day in enumerate(_sessions(20))}
    rising.pop(SESSIONS[-1].isoformat())  # the newest target has no SPY close: unknown, left out
    result = bf.build_permutation_outcomes(files["features"], horizons=horizons, spy_closes=rising,
                                           last_completed=SESSIONS[-1])
    swing = result.rows
    assert swing
    # SPY rises 2% a session: +1% is a LOSS for a long, and a WIN for a short (SPY's short return is negative).
    assert all(row["win"] is (row["side"] == "SHORT") for row in swing)
    assert all(row["outcome_kind"] == bf.SWING_OUTCOME_KIND for row in swing)
    assert result.counts["swing_tape_unknown"] > 0
    assert not [r for r in swing if r["horizon"] == 1 and r["session"] == SESSIONS[-2].isoformat()]
    # With no SPY at all nothing is written: unknown is never a loss.
    empty = bf.build_permutation_outcomes(files["features"], horizons=horizons, spy_closes={},
                                          last_completed=SESSIONS[-1])
    assert [row for row in empty.rows if row["population"] == bf.POPULATION_SWING] == []


def _swing_population(sessions: list[str]) -> list[dict]:
    rng = random.Random(9)
    return [
        {"population": "swing", "family": "avwap_band_bounce", "side": "SHORT", "horizon": 5,
         "session": day, "episode_id": f"{day}:{n}", "win": rng.random() < 0.5, "r": 0.1,
         "f_ma_support": rng.choice(["x", "y"])}
        for day in sessions for n in range(20)
    ]


def test_a_selection_window_under_20_sessions_is_refused_with_its_reason(tmp_path):
    days = [(date(2026, 8, 3) + timedelta(days=i)).isoformat() for i in range(30)]
    report = search.build_report(_swing_population(days), ledger_root=tmp_path)
    block = report["populations"]["swing"]["horizons"]["5"]
    assert block["refused"] is True
    assert block["selection_sessions"] < search.MIN_SELECTION_SESSIONS
    assert "under 20" in block["refused_reason"] and "5_sessions" in block["refused_reason"]
    family = block["families"]["avwap_band_bounce SHORT"]
    assert family["verdict"] == search.VERDICT_THIN and family["keys"] == []
    assert "win_rate" not in family["baseline"]  # nothing measured is published
    assert trial_ledger.load(tmp_path) == []  # no grid registered, no outcome read

    long_days = [(date(2026, 3, 2) + timedelta(days=i)).isoformat() for i in range(50)]
    published = search.build_report(_swing_population(long_days), ledger_root=tmp_path / "b")
    assert "refused" not in published["populations"]["swing"]["horizons"]["5"]
