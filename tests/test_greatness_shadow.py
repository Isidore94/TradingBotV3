"""Greatness shadow board: persistence, dedupe, fail-safety (plan 16.2)."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from greatness_monitor import Stage  # noqa: E402
from greatness_shadow import GreatnessBoard, record_d1_shadow  # noqa: E402

START = datetime(2026, 7, 13, 9, 35)
LEVELS = [{"label": "UPPER_1", "level": 101.0, "setup_family": "test"}]


def frame(rows):
    return pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(START + timedelta(minutes=5 * i)),
                "open": o, "high": h, "low": l, "close": c, "volume": 1000,
            }
            for i, (o, h, l, c) in enumerate(rows)
        ]
    )


def _board(tmp_path):
    return GreatnessBoard(
        store_path=tmp_path / "cands.json", events_path=tmp_path / "events.jsonl"
    )


def _bars(board, df, now):
    from greatness_shadow import _bars_from_frame

    return _bars_from_frame(df, now=now)


def test_board_progresses_persists_and_never_double_counts(tmp_path):
    board = _board(tmp_path)
    df = frame([(100.0, 100.6, 99.8, 100.4), (100.4, 101.7, 100.2, 101.4)])
    now = START + timedelta(minutes=11)

    events = board.update("NVDA", "LONG", LEVELS, _bars(board, df, now), session_date="2026-07-13")
    assert any(e.event.value == "READY" for e in events)

    # same frame again: no duplicate transitions
    again = board.update("NVDA", "LONG", LEVELS, _bars(board, df, now), session_date="2026-07-13")
    assert again == []

    # restart: stage survives via the store
    reborn = _board(tmp_path)
    cand = reborn.candidates["NVDA|2026-07-13"]
    assert cand.stage == Stage.READY

    logged = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["event"] == "READY" for row in logged)


def test_forming_last_bar_does_not_confirm(tmp_path):
    board = _board(tmp_path)
    df = frame([(100.0, 100.6, 99.8, 100.4), (100.4, 101.7, 100.2, 101.4)])
    mid_bar = START + timedelta(minutes=7)  # second bar still forming
    events = board.update("NVDA", "LONG", LEVELS, _bars(board, df, mid_bar), session_date="2026-07-13")
    assert not any(e.event.value == "READY" for e in events)


def test_record_d1_shadow_reads_bot_maps_and_never_raises(tmp_path, monkeypatch):
    import greatness_shadow as gs

    monkeypatch.setattr(gs, "_board", None)
    monkeypatch.setattr(gs, "_diag_dir", lambda: tmp_path)
    bot = SimpleNamespace(
        master_avwap_d1_upgrade_alerts={},
        master_avwap_d1_watchlist={
            "NVDA": {"side": "LONG", "trigger_levels": LEVELS, "active_current_scan": True}
        },
    )
    df = frame([(100.0, 100.6, 99.8, 100.4), (100.4, 101.7, 100.2, 101.4)])
    events = record_d1_shadow(bot, "NVDA", df, now=START + timedelta(minutes=11))
    assert any(e.event.value == "READY" for e in events)

    # garbage inputs are silently absorbed
    assert record_d1_shadow(bot, "NONE", None) == []
    assert record_d1_shadow(SimpleNamespace(), "NVDA", df) == []
