"""Movers M30 / Daily pick outcomes: log once, resolve once, excess by side, summaries."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import movers_timeframe_outcomes as mto  # noqa: E402

NY = ZoneInfo("America/New_York")


def _days(count, start=date(2026, 9, 14)):
    out, cursor = [], start
    while len(out) < count:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


DAYS = _days(8)


def _daily(closes):
    return [{"dt": datetime(d.year, d.month, d.day, tzinfo=NY), "close": c}
            for d, c in zip(DAYS, closes, strict=False)]


def _board(tf="d1", session=DAYS[0]):
    return {
        "tf": tf, "session": session.isoformat(), "as_of": f"{session.isoformat()}T00:00:00-04:00",
        "scanned_at": "x", "state": {"spy_last": 400.0},
        "pop": {"long": [{"symbol": "UP", "last": 100.0, "pop_score": 2.0}],
                "short": [{"symbol": "DN", "last": 50.0, "pop_score": -1.5}]},
        "swing": {"long": [{"symbol": "UP", "last": 100.0, "dip_score": 0.8}],
                  "short": [{"symbol": "DN", "last": 50.0, "dip_score": -0.4}]},
    }


def test_board_picks_log_each_box_row_once_per_session():
    picks = mto.board_picks(_board())
    assert [(p["box"], p["side"], p["symbol"], p["rank"]) for p in picks] == [
        ("pop", "long", "UP", 1), ("pop", "short", "DN", 1),
        ("dip_strong", "long", "UP", 1), ("dip_weak", "short", "DN", 1)]
    assert picks[0]["entry_close"] == 100.0 and picks[0]["spy_entry"] == 400.0
    assert picks[0]["score"] == 2.0 and picks[2]["score"] == 0.8
    assert mto.new_picks(_board(), picks) == []  # a re-scan of the same session adds nothing
    assert len(mto.new_picks(_board(session=DAYS[1]), picks)) == 4


def test_resolve_once_with_the_excess_sign_by_side():
    picks = mto.board_picks(_board())
    daily = {"UP": _daily([100.0, 102.0, 103.0, 104.0]),
             "DN": _daily([50.0, 49.0, 48.0, 47.0]),
             "SPY": _daily([400.0, 404.0, 404.0, 404.0])}
    rows = mto.resolve(picks, daily, resolved_at="now")
    # +1 and +3 are closed; +5 is not yet.
    assert sorted({r["horizon"] for r in rows}) == [1, 3]
    up1 = next(r for r in rows if r["symbol"] == "UP" and r["box"] == "pop" and r["horizon"] == 1)
    assert up1["ret_pct"] == pytest.approx(2.0) and up1["spy_ret_pct"] == pytest.approx(1.0)
    assert up1["excess_pct"] == pytest.approx(1.0) and up1["beat"] is True
    assert up1["target_session"] == DAYS[1].isoformat()
    dn3 = next(r for r in rows if r["symbol"] == "DN" and r["box"] == "dip_weak"
               and r["horizon"] == 3)
    # A short that fell 6% while SPY rose 1%: +6% side-adjusted, +7% excess.
    assert dn3["ret_pct"] == pytest.approx(6.0) and dn3["excess_pct"] == pytest.approx(7.0)
    # Never twice: the same data resolves nothing new; more data adds only +5.
    assert mto.resolve(picks + rows, daily) == []
    daily = {k: _daily([b["close"] for b in v] + [v[-1]["close"]] * 2) for k, v in daily.items()}
    later = mto.resolve(picks + rows, daily)
    assert {r["horizon"] for r in later} == {5} and len(later) == 4


def test_resolve_skips_a_pick_whose_session_or_spy_is_missing():
    picks = mto.board_picks(_board(session=date(2026, 8, 3)))
    daily = {"UP": _daily([100.0, 101.0]), "SPY": _daily([400.0, 401.0])}
    assert mto.resolve(picks, daily) == []
    picks = mto.board_picks(_board())
    assert mto.resolve(picks, {"UP": _daily([100.0, 101.0])}) == []  # no SPY


def test_summarize_reads_the_last_sessions_and_the_line_reads_3d_first(tmp_path):
    records = []
    for index, day in enumerate(DAYS[:3]):
        records += mto.board_picks(_board(session=day))
        for horizon, excess in ((1, 1.0), (3, 2.0 if index else -1.0)):
            records.append({"kind": "outcome", "tf": "d1", "box": "pop", "side": "long",
                            "symbol": "UP", "session": day.isoformat(), "horizon": horizon,
                            "excess_pct": excess})
    summary = mto.summarize(records, "d1", "pop", "long", lookback_sessions=2)
    assert summary["sessions"] == 2 and summary["picks"] == 2
    assert summary["n"] == {"1": 2, "3": 2, "5": 0}
    assert summary["mean_excess_pct"]["3"] == pytest.approx(2.0)
    assert summary["beat_pct"]["3"] == pytest.approx(100.0)
    assert mto.summary_line(summary) == "Last 2 sessions: +2.0% vs SPY at 3d, 100% beat, n=2"
    full = mto.summarize(records, "d1", "pop", "long")
    assert full["mean_excess_pct"]["3"] == pytest.approx(1.0)
    assert mto.summary_line(mto.summarize(records, "m30", "pop", "long")) == "no results yet"
    only_1d = mto.summarize([r for r in records if r.get("horizon") != 3], "d1", "pop", "long")
    assert "at 1d" in mto.summary_line(only_1d)
    assert set(mto.summaries(records, "d1")) == {"pop", "dip_strong", "dip_weak"}


def test_cli_prints_both_timeframes(tmp_path, capsys):
    path = tmp_path / "picks.jsonl"
    assert mto.append_records(path, mto.board_picks(_board()))
    assert mto.main(["--summary", "--path", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["d1"]["pop"]["long"]["picks"] == 1
    assert out["m30"]["dip_weak"]["short"]["line"] == "no results yet"
    assert mto.main([]) == 2
