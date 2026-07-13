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
    assert len(reborn.candidates) == 1
    candidate_id, cand = next(iter(reborn.candidates.items()))
    assert candidate_id.startswith("NVDA|LONG|test|2026-07-13|greatness_v1|")
    assert cand.stage == Stage.READY

    logged = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["event"] == "READY" for row in logged)
    assert all(row["schema"] == "greatness_shadow_v2" for row in logged)
    assert all(row["candidate_id"] == candidate_id for row in logged)
    assert all(row["evaluated_at"] and row["bar_ts"] for row in logged)
    assert all(datetime.fromisoformat(row["evaluated_at"]).utcoffset() is not None for row in logged)
    assert all(datetime.fromisoformat(row["bar_ts"]).utcoffset() is not None for row in logged)
    assert all(row["timezone"] for row in logged)
    assert all(row["engine_version"] == "greatness_v1" and row["config_hash"] for row in logged)
    stored = json.loads((tmp_path / "cands.json").read_text(encoding="utf-8"))
    assert stored["schema"] == "greatness_store_v2"
    assert stored["coverage"]["evaluations"] == 2
    assert stored["coverage"]["bars_consumed"] == 2
    assert stored["coverage"]["bars_skipped_duplicate"] == 2


def test_forming_last_bar_does_not_confirm(tmp_path):
    board = _board(tmp_path)
    df = frame([(100.0, 100.6, 99.8, 100.4), (100.4, 101.7, 100.2, 101.4)])
    mid_bar = START + timedelta(minutes=7)  # second bar still forming
    events = board.update("NVDA", "LONG", LEVELS, _bars(board, df, mid_bar), session_date="2026-07-13")
    assert not any(e.event.value == "READY" for e in events)


def test_aware_now_keeps_forming_last_bar_incomplete(tmp_path, monkeypatch):
    from zoneinfo import ZoneInfo

    monkeypatch.setenv("TRADINGBOT_MARKET_TIMEZONE", "America/Vancouver")
    board = _board(tmp_path)
    df = frame([(100.0, 100.6, 99.8, 100.4), (100.4, 101.7, 100.2, 101.4)])
    aware_mid_bar = (START + timedelta(minutes=7)).replace(
        tzinfo=ZoneInfo("America/Vancouver")
    )

    events = board.update(
        "NVDA",
        "LONG",
        LEVELS,
        _bars(board, df, aware_mid_bar),
        session_date="2026-07-13",
        evaluated_at=aware_mid_bar,
    )

    assert not any(event.event.value == "READY" for event in events)


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


def test_default_shadow_paths_honor_diagnostics_override(tmp_path, monkeypatch):
    import greatness_shadow as gs
    import market_state_bridge as bridge

    monkeypatch.setenv("TRADINGBOT_DIAGNOSTICS_DIR", str(tmp_path))
    assert gs._diag_dir() == tmp_path
    assert bridge.shadow_log_path() == tmp_path / "spy_state_shadow.jsonl"


def test_legacy_event_log_is_archived_before_v2_rows_are_written(tmp_path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text('{"ts":"2026-05-06T09:40:00","symbol":"AAPL","event":"READY"}\n', encoding="utf-8")
    board = GreatnessBoard(store_path=tmp_path / "cands.json", events_path=events_path)
    df = frame([(100.0, 100.6, 99.8, 100.4), (100.4, 101.7, 100.2, 101.4)])

    board.update(
        "NVDA",
        "LONG",
        LEVELS,
        _bars(board, df, START + timedelta(minutes=11)),
        session_date="2026-07-13",
        evaluated_at=START + timedelta(minutes=11),
    )

    archives = list(tmp_path.glob("events.legacy-*.jsonl"))
    assert len(archives) == 1
    assert '"symbol":"AAPL"' in archives[0].read_text(encoding="utf-8")
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert rows and all(row["schema"] == "greatness_shadow_v2" for row in rows)


def test_new_session_prunes_old_candidates_instead_of_replaying_test_fixtures(tmp_path):
    board = _board(tmp_path)
    bars = _bars(board, frame([(100.0, 101.2, 99.8, 101.1)]), START + timedelta(minutes=6))
    board.update("AAPL", "LONG", LEVELS, bars, session_date="2026-07-11")
    assert any(candidate.symbol == "AAPL" for candidate in board.candidates.values())

    board.update("NVDA", "LONG", LEVELS, bars, session_date="2026-07-13")
    assert {candidate.symbol for candidate in board.candidates.values()} == {"NVDA"}
    assert all("AAPL" not in key for key in board.last_bar_ts)
