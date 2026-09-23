"""wall_gate_replay: a read-only count of what the wall gate would have hidden."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import wall_gate_replay  # noqa: E402

DAY1 = date(2026, 9, 21)
DAY2 = date(2026, 9, 22)


def _bars(segments, end: date, tail=()):
    """Daily bars ending the day before ``end``, then ``tail`` closes from ``end`` on."""
    closes = []
    for count, first, last in segments:
        for i in range(count):
            closes.append(first + (0.0 if count == 1 else (last - first) * i / (count - 1)))
    start = datetime.combine(end, datetime.min.time()) - timedelta(days=len(closes))
    out = [
        {"dt": start + timedelta(days=i), "open": c, "high": c + 0.5, "low": c - 0.5, "close": c}
        for i, c in enumerate(closes)
    ]
    for i, c in enumerate(tail):
        out.append({"dt": datetime.combine(end, datetime.min.time()) + timedelta(days=i),
                    "open": c, "high": c + 0.5, "low": c - 0.5, "close": c})
    return out


# SMA100 = 21 over a long at 20.8 (0.2 ATR): a wall. SMA200 = 19.5, SMA50 = 20.
WALL = [(150, 18.0, 18.0), (50, 22.0, 22.0), (50, 20.0, 20.0)]
# Only support below a long at 20.8.
CLEAR = [(150, 18.0, 18.0), (100, 20.0, 20.0)]


def _shown(day, symbol, side="LONG", px=20.8, ts="10:00:00", **extra):
    row = {
        "action": "shown",
        "trade_date": day.isoformat(),
        "symbol": symbol,
        "side": side,
        "is_d1": True,
        "timeframe": "D1",
        "trigger": f"{symbol} ({side.lower()}) zone1 bounce [@1; px={px}; bar=12:55]",
        "ts": f"{day.isoformat()}T{ts}",
    }
    row.update(extra)
    return row


def test_it_counts_hidden_walls_per_day_and_joins_outcomes():
    bars = {"WAL": _bars(WALL, DAY1, tail=[20.8, 20.9]), "CLR": _bars(CLEAR, DAY1)}
    shown = [_shown(DAY1, "WAL"), _shown(DAY1, "CLR"), _shown(DAY2, "WAL")]
    outcomes = {("WAL", "LONG", DAY1.isoformat()): {1: -1.5, 3: -2.0, 5: 0.5},
                ("CLR", "LONG", DAY1.isoformat()): {1: 2.0}}
    report = wall_gate_replay.replay(shown, lambda s: bars.get(s, []), outcomes)
    day1, day2 = report["days"]
    assert (day1["alerts"], day1["hidden"], day1["peak_armed"]) == (2, 1, 1)
    assert day1["walls"] == {"SMA100": 1}
    # Day 2: WAL is still under its SMA100 and still armed (no close through it).
    assert (day2["alerts"], day2["hidden"], day2["peak_armed"]) == (1, 1, 1)
    hidden = report["outcomes"]["hidden"]
    assert hidden["count"] == 2 and hidden["h1_n"] == 1 and hidden["h1_mean_pct"] == -1.5
    assert report["outcomes"]["shown"]["h1_mean_pct"] == 2.0


def test_point_in_time_the_trade_dates_own_bar_is_not_used():
    # The day's own bar (a 50.0 close) must not move the SMAs or the ATR.
    bars = {"WAL": _bars(WALL, DAY1, tail=[50.0])}
    report = wall_gate_replay.replay([_shown(DAY1, "WAL")], lambda s: bars[s])
    assert report["rows"][0]["wall"] == "SMA100"


def test_the_cap_shows_the_overflow_and_a_close_through_frees_a_place():
    bars = {f"S{i}": _bars(WALL, DAY1, tail=[20.8, 20.8]) for i in range(3)}
    bars["S0"] = _bars(WALL, DAY1, tail=[21.5, 21.5])  # closes through its SMA100 on DAY1
    shown = [_shown(DAY1, f"S{i}", ts=f"10:0{i}:00") for i in range(3)]
    shown.append(_shown(DAY2, "S2", ts="10:00:00"))
    report = wall_gate_replay.replay(shown, lambda s: bars[s], cap=2)
    day1, day2 = report["days"]
    assert (day1["hidden"], day1["shown_at_wall_cap"], day1["peak_armed"]) == (2, 1, 2)
    # S0's arm fired on DAY1's close; S2 now fits.
    assert (day2["hidden"], day2["shown_at_wall_cap"]) == (1, 0)


def test_missing_bars_are_unknown_and_shown():
    report = wall_gate_replay.replay([_shown(DAY1, "NONE")], lambda s: [])
    assert report["days"][0]["unknown"] == 1 and report["days"][0]["hidden"] == 0


def test_load_shown_keeps_one_row_per_chart(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = [
        _shown(DAY1, "AAA", ts="11:00:00"),
        _shown(DAY1, "AAA", ts="10:00:00"),
        {"action": "skip", "trade_date": DAY1.isoformat(), "symbol": "AAA", "side": "LONG"},
        _shown(DAY1, "BBB", side="WATCH"),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    shown = wall_gate_replay.load_shown(path)
    assert [(row["symbol"], row["ts"][-8:]) for row in shown] == [("AAA", "10:00:00")]


def test_price_prefers_the_trigger_px():
    assert wall_gate_replay.price_for({"trigger": "x [px=12.5; bar]"}, []) == (12.5, "trigger_px")
    completed = [{"close": 9.0}]
    assert wall_gate_replay.price_for({"trigger": "none"}, completed) == (9.0, "prior_close")


def _run_cli(tmp_path, *extra, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "wall_gate_replay.py"), *extra],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=120,
    )


def test_the_cli_refuses_live_inputs(tmp_path):
    env = dict(os.environ)
    result = _run_cli(
        tmp_path,
        "--scratch", str(tmp_path / "scratch"),
        "--events", r"C:\TradingBotData\alert_review_events",
        "--daily-bars", str(tmp_path / "bars"),
        env=env,
    )
    assert result.returncode == 2
    assert "REFUSED" in result.stderr


def test_the_cli_refuses_a_scratch_inside_the_real_localappdata(tmp_path):
    env = dict(os.environ)
    env["LOCALAPPDATA"] = str(tmp_path / "real_local")
    result = _run_cli(
        tmp_path,
        "--scratch", str(tmp_path / "real_local" / "TradingBotV3" / "x"),
        "--events", str(tmp_path / "e.jsonl"),
        "--daily-bars", str(tmp_path / "bars"),
        env=env,
    )
    assert result.returncode == 2


def test_the_cli_runs_on_copies_and_writes_nothing_but_stdout(tmp_path):
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    rows = [
        {"datetime": bar["dt"].isoformat(), "open": bar["open"], "high": bar["high"],
         "low": bar["low"], "close": bar["close"], "volume": 1}
        for bar in _bars(WALL, DAY1)
    ]
    (tmp_path / "bars.json").write_text(json.dumps(rows), encoding="utf-8")
    # The parquet is written in a child process: pandas/pyarrow in the test
    # worker itself can collide with other tests' module reloads.
    writer = (
        "import json, sys, pandas as pd; "
        "f = pd.DataFrame(json.load(open(sys.argv[1]))); "
        "f['datetime'] = pd.to_datetime(f['datetime']); "
        "f.to_parquet(sys.argv[2])"
    )
    subprocess.run(
        [sys.executable, "-c", writer, str(tmp_path / "bars.json"), str(bars_dir / "WAL.parquet")],
        check=True,
        timeout=120,
    )
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps(_shown(DAY1, "WAL")) + "\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    before = sorted(p.name for p in tmp_path.iterdir())
    env = dict(os.environ)
    result = _run_cli(
        tmp_path,
        "--scratch", str(scratch),
        "--events", str(events),
        "--daily-bars", str(bars_dir),
        "--json",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["days"][0]["hidden"] == 1
    after = sorted(p.name for p in tmp_path.iterdir())
    assert set(after) - set(before) == {"scratch"}
