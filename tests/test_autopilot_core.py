"""Tests for the Auto Pilot (mini PC mode) core logic."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autopilot_core as core  # noqa: E402

PACIFIC = "America/Los_Angeles"
# 2026-07-02 is a Thursday; regular session 06:30-13:00 Pacific.
REF = datetime(2026, 7, 2, 8, 0)


def test_swing_slots_start_open_plus_hour_then_hourly_from_first_full_hour():
    slots = core.get_autopilot_swing_slots(REF, local_timezone_name=PACIFIC)
    assert slots == ["07:30", "09:00", "10:00", "11:00", "12:00", "13:00"]


def test_hourly_away_report_slots_start_at_0700_and_run_once_per_hour():
    before = datetime(2026, 7, 2, 6, 59)
    at_seven = datetime(2026, 7, 2, 7, 0)
    catch_up = datetime(2026, 7, 2, 8, 37)

    assert core.hourly_away_report_slot_due(before) is None
    seven_slot = core.hourly_away_report_slot_due(at_seven)
    assert seven_slot == "2026-07-02|07:00"
    assert core.hourly_away_report_slot_due(at_seven, last_completed_slot=seven_slot) is None
    assert core.hourly_away_report_slot_due(catch_up, last_completed_slot=seven_slot) == "2026-07-02|08:00"
    assert core.hourly_away_report_slot_due(datetime(2026, 7, 2, 14, 0)) is None
    assert core.hourly_away_report_slot_due(datetime(2026, 7, 4, 9, 0)) is None


def test_tracker_writes_only_in_the_final_hour_slots():
    assert core.slot_writes_setup_tracker("12:00", REF, local_timezone_name=PACIFIC)
    assert core.slot_writes_setup_tracker("13:00", REF, local_timezone_name=PACIFIC)
    assert not core.slot_writes_setup_tracker("07:30", REF, local_timezone_name=PACIFIC)
    assert not core.slot_writes_setup_tracker("11:00", REF, local_timezone_name=PACIFIC)


def test_summarize_open_move():
    bars = [
        {"open": 102.0, "high": 103.0, "low": 101.5, "close": 102.5},
        {"open": 102.5, "high": 104.0, "low": 102.4, "close": 103.8},
    ]
    summary = core.summarize_open_move(100.0, bars)
    assert round(summary["gap_pct"], 2) == 2.0
    assert round(summary["early_move_pct"], 2) == round((103.8 - 102.0) / 102.0 * 100, 2)
    assert core.summarize_open_move(100.0, []) is None


def test_build_watchlists_gap_and_rs_selection():
    moves = {
        "GAPU": {"gap_pct": 4.0, "early_move_pct": 0.5},   # gap-up long
        "RSST": {"gap_pct": 0.2, "early_move_pct": 2.4},   # RS long (SPY +0.4)
        "GAPD": {"gap_pct": -3.0, "early_move_pct": -0.2},  # gap-down short
        "RWWK": {"gap_pct": -0.1, "early_move_pct": -1.8},  # RW short
        "FLAT": {"gap_pct": 0.3, "early_move_pct": 0.5},    # nothing
        "SPY": {"gap_pct": 0.0, "early_move_pct": 0.4},
    }
    built = core.build_watchlists_from_moves(moves, moves["SPY"])
    assert set(built["longs"]) == {"GAPU", "RSST"}
    assert set(built["shorts"]) == {"GAPD", "RWWK"}
    assert "FLAT" not in built["longs"] + built["shorts"]
    assert "SPY" not in built["longs"] + built["shorts"]
    # Gap magnitude + excess move ranks GAPU first.
    assert built["longs"][0] == "GAPU"


def test_build_watchlists_caps_and_resolves_conflicts():
    moves = {f"L{i:02d}": {"gap_pct": 2.0 + i * 0.1, "early_move_pct": 1.0} for i in range(30)}
    # A name that gapped up but is dumping vs SPY: short side wins on score.
    moves["BOTH"] = {"gap_pct": 2.5, "early_move_pct": -9.0}
    built = core.build_watchlists_from_moves(moves, {"early_move_pct": 0.0})
    assert len(built["longs"]) == core.AUTOPILOT_WATCHLIST_CAP
    assert "BOTH" in built["shorts"] and "BOTH" not in built["longs"]


def test_near_extreme_candidates_both_sides():
    snapshot = {
        "NHOD": {"last": 99.5, "day_high": 100.0, "day_low": 95.0},   # 0.5% off HOD
        "FADE": {"last": 96.0, "day_high": 100.0, "day_low": 95.0},   # 4% off HOD
        "NLOD": {"last": 95.3, "day_high": 100.0, "day_low": 95.0},   # near LOD
    }
    assert core.near_extreme_candidates(snapshot, "long") == ["NHOD"]
    assert core.near_extreme_candidates(snapshot, "short") == ["NLOD"]


def test_append_watchlist_symbols(tmp_path):
    target = tmp_path / "longs.txt"
    core.write_watchlist_file(target, ["AAPL", "MSFT"])
    added = core.append_watchlist_symbols(target, ["msft", "NVDA", "AMD"])
    assert added == ["NVDA", "AMD"]
    assert target.read_text(encoding="utf-8").split() == ["AAPL", "MSFT", "NVDA", "AMD"]


def test_watchlist_atomic_replace_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "longs.txt"
    target.write_text("AAPL\n", encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("simulated shared-drive replace failure")

    monkeypatch.setattr(core.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated shared-drive"):
        core.write_watchlist_file(target, ["NVDA"])

    assert target.read_text(encoding="utf-8") == "AAPL\n"
    assert list(tmp_path.glob(".longs.txt.*.tmp")) == []


def test_render_away_report_is_phone_digestible():
    text = core.render_away_report(
        {
            "generated_at": "2026-07-02 10:15:00",
            "enabled": True,
            "ib_status": "connected",
            "regime": "bullish_weak",
            "longs": ["AAPL", "NVDA"],
            "shorts": ["XYZ"],
            "swing_picks": [{"symbol": "AAPL", "side": "LONG", "bucket": "Favorite", "expected_r": 1.25}],
            "alerts": ["[S-TIER] ORB BREAKOUT AAPL (long): ..."],
            "slots_done": ["07:30", "09:00"],
            "next_slot": "10:00",
            "log_lines": ["[10:00:01] Swing scan started"],
        }
    )
    assert "AAPL, NVDA" in text
    assert "AAPL (LONG) | Favorite | 1.25R" in text
    assert "Next swing slot: 10:00" in text
    assert "bullish_weak" in text
    assert "ORB BREAKOUT" in text


def test_industry_snapshot_identity_is_shared_and_mismatch_is_loud():
    board = {"status": "ok", "snapshot_id": "board-123"}
    intraday = {
        "snapshot_id": "m5-456",
        "source_board_snapshot_id": "board-123",
        "qualified_industry_count": 12,
        "industry_count": 20,
    }
    line = core.format_industry_snapshot_line(board, intraday)
    assert "snapshot board-123" in line
    assert "M5 advisory m5-456" in line
    assert "12/20 qualified" in line
    assert "SOURCE MISMATCH" not in line

    mismatch = core.format_industry_snapshot_line(
        board,
        {**intraday, "source_board_snapshot_id": "old-board"},
    )
    assert "SOURCE MISMATCH" in mismatch

    report = core.render_away_report(
        {
            "generated_at": "2026-07-14 10:00:00",
            "industry_line": line,
            "swing_data_current": True,
        }
    )
    assert line in report
    assert report.index(line) < report.index("== SWING OPPORTUNITIES ==")


def _fake_frame(rows, start):
    import pandas as pd

    index = [start + timedelta(minutes=5 * i) for i in range(len(rows))]
    return pd.DataFrame(
        {
            "Open": [row[0] for row in rows],
            "High": [row[1] for row in rows],
            "Low": [row[2] for row in rows],
            "Close": [row[3] for row in rows],
        },
        index=pd.DatetimeIndex(index),
    )


def test_fetch_open_scan_moves_with_injected_downloader():
    prev_day = datetime(2026, 7, 1, 6, 30)
    today = datetime(2026, 7, 2, 6, 30)

    def downloader(symbols, *, period, interval):
        assert period == "2d" and interval == "5m"
        import pandas as pd

        frames = {}
        for symbol in symbols:
            if symbol == "SPY":
                prev = _fake_frame([(500.0, 500.5, 499.5, 500.0)] * 3, prev_day)
                cur = _fake_frame([(501.0, 501.5, 500.5, 501.0), (501.0, 502.0, 500.9, 502.0)], today)
            else:
                prev = _fake_frame([(100.0, 100.5, 99.5, 100.0)] * 3, prev_day)
                cur = _fake_frame([(103.0, 103.5, 102.5, 103.0), (103.0, 105.5, 102.9, 105.0)], today)
            frames[symbol] = pd.concat([prev, cur])
        return frames

    moves = core.fetch_open_scan_moves(["GAPR"], downloader=downloader)
    assert set(moves) == {"SPY", "GAPR"}
    assert round(moves["GAPR"]["gap_pct"], 2) == 3.0  # 103 open vs 100 prev close
    assert round(moves["GAPR"]["early_move_pct"], 2) == round((105.0 - 103.0) / 103.0 * 100, 2)
    assert round(moves["SPY"]["gap_pct"], 2) == 0.2
    # The session date rides along for the holiday/stale-feed guard.
    assert moves["SPY"]["session_date"] == today.date()


def test_last_completed_session_close_walks_back_correctly():
    # Mid-session Thursday: the last finished session is Wednesday's.
    assert core.last_completed_session_close(
        datetime(2026, 7, 2, 8, 0), PACIFIC
    ) == datetime(2026, 7, 1, 13, 0)
    # After Thursday's close, Thursday's own close counts.
    assert core.last_completed_session_close(
        datetime(2026, 7, 2, 14, 0), PACIFIC
    ) == datetime(2026, 7, 2, 13, 0)
    # Saturday: walk back to Friday's close.
    assert core.last_completed_session_close(
        datetime(2026, 7, 4, 10, 0), PACIFIC
    ) == datetime(2026, 7, 3, 13, 0)


def test_universe_staleness_rule():
    built_wed_afternoon = datetime(2026, 7, 1, 16, 0)
    # Fresh all through Thursday's session...
    assert not core.universe_is_stale(datetime(2026, 7, 2, 8, 0), built_wed_afternoon, PACIFIC)
    # ...stale the moment Thursday's close passes (wrap-up rebuild time).
    assert core.universe_is_stale(datetime(2026, 7, 2, 14, 0), built_wed_afternoon, PACIFIC)
    # Missing files are always stale.
    assert core.universe_is_stale(datetime(2026, 7, 2, 8, 0), None, PACIFIC)


def test_autopilot_auto_arm_due_daily_hands_off_rules():
    wednesday_early = datetime(2026, 7, 8, 6, 45)
    wednesday_late = datetime(2026, 7, 8, 7, 0)
    saturday = datetime(2026, 7, 11, 9, 0)

    # Before 07:00 -> not yet; at/after 07:00 on a weekday -> arm.
    assert not core.autopilot_auto_arm_due(wednesday_early, enabled=False, armed_date=None)
    assert core.autopilot_auto_arm_due(wednesday_late, enabled=False, armed_date=None)
    # Launching at 10:30 arms immediately.
    assert core.autopilot_auto_arm_due(datetime(2026, 7, 8, 10, 30), enabled=False, armed_date=None)
    # Already ON, already armed today (manual OFF sticks), weekends, or the
    # setting disabled -> never arm.
    assert not core.autopilot_auto_arm_due(wednesday_late, enabled=True, armed_date=None)
    assert not core.autopilot_auto_arm_due(wednesday_late, enabled=False, armed_date="2026-07-08")
    assert not core.autopilot_auto_arm_due(saturday, enabled=False, armed_date=None)
    assert not core.autopilot_auto_arm_due(
        wednesday_late, enabled=False, armed_date=None, auto_arm_enabled=False
    )
    # Yesterday's arm mark does not block today.
    assert core.autopilot_auto_arm_due(wednesday_late, enabled=False, armed_date="2026-07-07")


def test_after_close_wrapup_due_needs_all_slots_done():
    slots = core.get_autopilot_swing_slots(REF, local_timezone_name=PACIFIC)
    after_close = datetime(2026, 7, 2, 13, 20)
    assert core.after_close_wrapup_due(after_close, slots, False, False, PACIFIC)
    assert not core.after_close_wrapup_due(after_close, slots[:-1], False, False, PACIFIC)
    assert not core.after_close_wrapup_due(after_close, slots, True, False, PACIFIC)  # already done
    assert not core.after_close_wrapup_due(after_close, slots, False, True, PACIFIC)  # scan running
    assert not core.after_close_wrapup_due(datetime(2026, 7, 4, 13, 20), slots, False, False, PACIFIC)  # weekend


def test_merge_autopilot_watchlist_keeps_manual_names_only():
    result = core.merge_autopilot_watchlist(
        ["NVDA", "AAPL"],
        # File currently holds: yesterday's auto picks + the trader's dump.
        ["OLDPICK", "MYPICK", "AAPL"],
        ["OLDPICK", "AAPL"],  # what Auto Pilot wrote last time
    )
    assert result["symbols"] == ["NVDA", "AAPL", "MYPICK"]
    assert result["manual_kept"] == ["MYPICK"]

    # No prior state: everything currently in the file is treated as manual.
    first_run = core.merge_autopilot_watchlist(["NVDA"], ["MYPICK"], [])
    assert first_run["symbols"] == ["NVDA", "MYPICK"]


def test_score_autopilot_picks_joins_candidates_and_outcomes():
    picks = [
        {"date": "2026-07-02", "symbol": "AAPL", "side": "long"},
        {"date": "2026-07-02", "symbol": "XYZ", "side": "short"},
    ]
    candidates = [
        {"event_id": "e1", "event_type": "confirmed", "trade_date": "2026-07-02", "symbol": "AAPL", "direction": "long"},
        {"event_id": "e2", "event_type": "confirmed", "trade_date": "2026-07-02", "symbol": "OTHER", "direction": "long"},
        {"event_id": "e3", "event_type": "candidate", "trade_date": "2026-07-02", "symbol": "XYZ", "direction": "short"},
    ]
    outcomes = [{"event_id": "e1", "close_r": "0.5", "mfe_r": "2.0"}]
    scorecard = core.score_autopilot_picks(picks, candidates, outcomes)
    assert scorecard["picks"] == 2 and scorecard["longs"] == 1 and scorecard["shorts"] == 1
    assert scorecard["alerted"] == 1 and scorecard["alerted_symbols"] == ["AAPL"]
    assert scorecard["avg_close_r"] == 0.5
    assert scorecard["avg_mfe_r"] == 2.0
    line = core.format_scorecard_line(scorecard)
    assert "1 longs + 1 shorts" in line and "1 alerted" in line and "+0.50R" in line


def test_rebuild_universe_if_stale_outcomes():
    now = datetime(2026, 7, 2, 8, 0)
    fresh_built = datetime(2026, 7, 1, 16, 0)
    calls = []

    def fake_builder():
        calls.append(1)
        return {"all": ["A", "B"], "longs": ["A"], "shorts": ["B"]}

    # Fresh -> no rebuild, builder untouched.
    assert core.rebuild_universe_if_stale(now, builder=fake_builder, built_at=fresh_built) == "fresh"
    assert not calls

    # Stale -> rebuilds via the injected builder.
    assert core.rebuild_universe_if_stale(now, builder=fake_builder, built_at=None) == "rebuilt"
    assert calls == [1]

    # Someone else is rebuilding -> busy, no double work.
    assert core._UNIVERSE_REBUILD_LOCK.acquire(blocking=False)
    try:
        assert core.rebuild_universe_if_stale(now, builder=fake_builder, built_at=None) == "busy"
    finally:
        core._UNIVERSE_REBUILD_LOCK.release()
    assert calls == [1]

    # A crashing builder reports failure instead of raising.
    def broken_builder():
        raise RuntimeError("boom")

    assert core.rebuild_universe_if_stale(now, builder=broken_builder, built_at=None) == "failed"


def test_pick_grouping_and_suggestion_message():
    groups = core.group_picks_by_source(
        [
            {"symbol": "A", "source": "open_scan"},
            {"symbol": "B", "source": "hod_add"},
            {"symbol": "C", "source": "manual"},
            {"symbol": "D", "source": "suggestion"},
        ]
    )
    assert [pick["symbol"] for pick in groups["auto"]] == ["A", "B"]
    assert [pick["symbol"] for pick in groups["manual"]] == ["C"]
    assert [pick["symbol"] for pick in groups["suggested"]] == ["D"]

    message = core.format_suggestion_message(
        {
            "longs": ["GAPU", "RSST"],
            "shorts": ["GAPD"],
            "long_reasons": {"GAPU": "gap +4.0%"},
            "short_reasons": {},
        }
    )
    assert message.startswith("AUTO PILOT SUGGESTS")
    assert "GAPU (gap +4.0%)" in message and "RSST" in message and "shorts: GAPD" in message
    assert core.format_suggestion_message({"longs": [], "shorts": []}) == ""

    labeled = core.format_scorecard_line(
        {"picks": 3, "longs": 2, "shorts": 1, "alerted": 1, "avg_close_r": 0.9, "avg_mfe_r": None},
        label="Your picks",
    )
    assert labeled.startswith("Your picks:") and "+0.90R" in labeled


def test_away_report_has_tv_paste_and_status_lines():
    text = core.render_away_report(
        {
            "generated_at": "2026-07-02 13:30:00",
            "enabled": True,
            "ib_status": "connected",
            "regime": "bullish_strong",
            "longs": ["AAPL", "NVDA"],
            "shorts": [],
            "auto_longs": ["ABVX", "ATAI"],
            "auto_shorts": ["ACMR"],
            "universe_line": "Universe: fresh (built 2026-07-02 13:10)",
            "scorecard_line": "Auto picks today: 2 longs + 1 shorts -> 1 alerted.",
        }
    )
    assert "TV paste: AAPL,NVDA" in text
    assert "BOT PICKS - LONGS" in text and "ABVX, ATAI" in text
    assert "BOT PICKS - SHORTS" in text and "ACMR" in text
    assert "Universe: fresh" in text
    assert "Auto picks today:" in text
