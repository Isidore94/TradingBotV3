"""Tests for the Auto Pilot (mini PC mode) core logic."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

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
