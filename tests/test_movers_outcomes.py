"""Movers Dip-strong outcome log (`scripts/movers_outcomes.py`)."""

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

import movers_outcomes as mo  # noqa: E402

NY = ZoneInfo("America/New_York")
OPEN = datetime(2026, 9, 22, 9, 30, tzinfo=NY)


def _bars(closes, *, start=OPEN, poke=None):
    out, prev = [], closes[0]
    for i, close in enumerate(closes):
        out.append({"dt": start + timedelta(minutes=5 * i), "open": prev,
                    "high": max(prev, close) + (0.6 if i == poke else 0.1),
                    "low": min(prev, close) - 0.1, "close": close, "volume": 1000.0})
        prev = close
    return out


SPY = _bars([400.0, 400.0] + [405.0] * 8 + [404.5, 403.8, 403.2, 403.0, 403.2, 403.5,
                                           403.6, 403.8, 404.0], poke=9)
HOLD = _bars([100.0] * 10 + [99.8, 99.5, 99.8, 100.0, 100.3, 100.5, 100.6, 100.8, 101.0])
START = SPY[9]["dt"]


def _board(on=True):
    state = {"state": "up_day", "pullback": on, "bounce": False,
             "start_dt": START.isoformat() if on else "", "extreme_price": SPY[9]["high"],
             "spy_from_extreme_pct": -0.4}
    return {"state": state,
            "dip": {"long": [{"symbol": "HOLD", "dip_score": 1.5, "since_start_pct": 0.2,
                              "rvol": 1.3}] if on else [], "short": []}}


def _tick(tracker, upto, on=True):
    now = SPY[upto]["dt"] + timedelta(minutes=5, seconds=20)
    return tracker.observe(_board(on), {"HOLD": HOLD[: upto + 1]}, SPY[: upto + 1], now=now)


def test_flag_once_then_outcome_after_six_bars():
    tracker = mo.DipOutcomeTracker()
    first = _tick(tracker, 12)
    assert [r["kind"] for r in first] == ["flag"]
    flag = first[0]
    assert flag["symbol"] == "HOLD" and flag["rank"] == 1 and flag["episode"] == START.isoformat()
    assert flag["recorded_at"].endswith("-04:00")  # timezone on every stamp
    for upto in range(13, 18):
        assert _tick(tracker, upto) == []  # no duplicate flag, not ended yet
    last = _tick(tracker, 18)
    assert [r["kind"] for r in last] == ["outcome"]
    out = last[0]
    assert out["end_reason"] == "six_bars"
    assert out["anchor_bar"] == HOLD[11]["dt"].isoformat()
    low = HOLD[11]["low"]
    assert out["ret3_pct"] == pytest.approx((100.3 / low - 1) * 100)
    assert out["ret6_pct"] == pytest.approx((100.8 / low - 1) * 100)
    assert out["spy_ret3_pct"] == pytest.approx((SPY[14]["close"] / SPY[11]["close"] - 1) * 100)
    assert out["excess6_pct"] == pytest.approx(out["ret6_pct"] - out["spy_ret6_pct"])
    assert tracker.episodes == {}


def test_episode_ends_when_spy_reclaims_the_high():
    tracker = mo.DipOutcomeTracker()
    _tick(tracker, 12)
    spy = SPY[:13] + _bars([406.5], start=SPY[12]["dt"] + timedelta(minutes=5))
    now = spy[-1]["dt"] + timedelta(minutes=5, seconds=20)
    rows = tracker.observe(_board(), {"HOLD": HOLD[:14]}, spy, now=now)
    assert rows == []  # ended, but +6 bars after HOLD's low are not in yet
    episode = next(iter(tracker.episodes.values()))
    assert episode["end_reason"] == "spy_reclaimed_high"


def test_session_close_records_unknown_for_missing_bars():
    late = OPEN.replace(hour=15, minute=0)
    spy = _bars([400.0, 400.0] + [405.0] * 8 + [404.5, 403.8], start=late, poke=9)
    hold = _bars([100.0] * 10 + [99.8, 99.5], start=late)
    tracker = mo.DipOutcomeTracker()
    board = _board()
    board["state"]["start_dt"] = spy[9]["dt"].isoformat()
    now = spy[-1]["dt"] + timedelta(minutes=5, seconds=20)
    rows = tracker.observe(board, {"HOLD": hold}, spy, now=now)
    kinds = [r["kind"] for r in rows]
    assert kinds == ["flag", "outcome"]
    assert rows[1]["end_reason"] == "session_close"
    assert rows[1]["ret3_pct"] is None and rows[1]["excess6_pct"] is None


def test_failed_write_loses_the_rows_but_never_raises(tmp_path, caplog):
    assert mo.append_records(tmp_path, [{"kind": "flag"}]) is False  # a directory
    assert "write failed" in caplog.text
    target = tmp_path / "log.jsonl"
    assert mo.append_records(target, [{"kind": "flag"}, {"kind": "outcome"}]) is True
    assert len(mo.load_records(target)) == 2


def test_summary_and_cli(tmp_path, capsys):
    rows = [
        {"kind": "outcome", "session": "2026-09-21", "episode": "a", "side": "long",
         "excess3_pct": 0.2, "excess6_pct": 0.5},
        {"kind": "outcome", "session": "2026-09-22", "episode": "b", "side": "long",
         "excess3_pct": -0.1, "excess6_pct": -0.3},
        {"kind": "outcome", "session": "2026-09-22", "episode": "c", "side": "short",
         "excess3_pct": 0.4, "excess6_pct": None},
        {"kind": "flag", "session": "2026-09-22"},
    ]
    summary = mo.summarize(rows)
    assert summary["all"]["outcomes"] == 3 and summary["all"]["graded"] == 3
    assert summary["all"]["hit_rate"] == pytest.approx(2 / 3)
    assert summary["long"]["avg_excess6_pct"] == pytest.approx(0.1)
    ranged = mo.summarize(rows, start=date(2026, 9, 22))
    assert ranged["all"]["outcomes"] == 2
    path = tmp_path / "log.jsonl"
    mo.append_records(path, rows)
    assert mo.main(["--summary", "--path", str(path), "--start", "2026-09-22"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["all"]["outcomes"] == 2


def test_the_log_path_is_a_project_paths_constant():
    import project_paths

    assert project_paths.MOVERS_DIP_OUTCOMES_FILE.name == "movers_dip_outcomes.jsonl"
