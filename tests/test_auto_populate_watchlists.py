"""Tests for the universe -> longs.txt/shorts.txt auto-populate engine."""

import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from watchlist_utils import read_watchlist_symbols  # noqa: E402


AGGRESSIVE_FIXTURE = Path(__file__).parent / "fixtures" / "aggressive_watchlist_candidates_v1.json"


def test_auto_populate_caps_are_a_flat_ceiling():
    """2026-07-16 trader directive: the quality bar governs the count; the
    cap is only a symmetric ceiling (up to 100 per side), never a target."""
    from autopilot_core import AUTO_POPULATE_MAX_PER_SIDE, auto_populate_caps

    for env in ("bullish_strong", "bullish_weak", "bearish_strong", "bearish_weak", "neutral_chop", ""):
        assert auto_populate_caps(env) == (AUTO_POPULATE_MAX_PER_SIDE, AUTO_POPULATE_MAX_PER_SIDE)
    assert AUTO_POPULATE_MAX_PER_SIDE == 100


def _ctx(prev_high, prev_low, prev_close, adr):
    return {"prev_high": prev_high, "prev_low": prev_low, "prev_close": prev_close, "adr": adr}


def _profile(last, *, at_high=0.0, at_low=0.0):
    return {
        "last": last,
        "day_high": last,
        "day_low": last,
        "time_at_high_frac": at_high,
        "time_at_low_frac": at_low,
    }


def test_adr_breakout_candidates_require_break_and_adr_move():
    from autopilot_core import build_adr_breakout_candidates

    profiles = {
        # PDH break + 1.5 ADR move + lots of time at HOD -> strong long.
        "AAA": _profile(103.0, at_high=0.8),
        # PDH break but a tiny move vs ADR -> rejected.
        "BBB": _profile(100.6, at_high=0.9),
        # Big move but still under PDH -> rejected.
        "CCC": _profile(99.9, at_high=0.5),
        # PDL break with -1.0 ADR -> short.
        "DDD": _profile(96.0, at_low=0.6),
        "SPY": _profile(500.0, at_high=1.0),
    }
    context = {
        "AAA": _ctx(100.5, 98.0, 100.0, 2.0),
        "BBB": _ctx(100.5, 98.0, 100.0, 2.0),
        "CCC": _ctx(100.5, 98.0, 100.0, 2.0),
        "DDD": _ctx(100.5, 98.0, 100.0, 4.0),
        "SPY": _ctx(499.0, 490.0, 495.0, 5.0),
    }

    result = build_adr_breakout_candidates(profiles, context)
    long_symbols = [row["symbol"] for row in result["longs"]]
    short_symbols = [row["symbol"] for row in result["shorts"]]

    assert long_symbols == ["AAA"]  # SPY is excluded even when qualifying
    assert short_symbols == ["DDD"]
    assert "PDH break" in result["longs"][0]["reason"]
    assert "+1.5 ADR" in result["longs"][0]["reason"]
    assert "80% of day at HOD" in result["longs"][0]["reason"]
    assert "PDL break" in result["shorts"][0]["reason"]


def test_adr_candidates_rank_by_move_plus_extreme_time():
    import pytest

    from autopilot_core import AUTO_POPULATE_MIN_SCORE, build_adr_breakout_candidates

    profiles = {
        "FAST": _profile(104.0, at_high=0.1),  # 2.0 ADR + 0.1 -> ~2.1
        "GRIND": _profile(102.4, at_high=0.9),  # 1.2 ADR + 0.9 -> ~2.1 (grinder ties the spike)
        "MEH": _profile(101.6, at_high=0.2),  # 0.8 ADR + 0.2 -> 1.0: under the quality bar
    }
    context = {sym: _ctx(100.5, 98.0, 100.0, 2.0) for sym in profiles}

    result = build_adr_breakout_candidates(profiles, context)
    scores = {row["symbol"]: row["score"] for row in result["longs"]}
    assert scores["FAST"] == pytest.approx(2.1)
    assert scores["GRIND"] == pytest.approx(2.1)
    # Quality over quantity: a qualifying-but-marginal breaker stays out
    # entirely instead of padding the list toward the cap.
    assert "MEH" not in scores
    assert 1.0 < AUTO_POPULATE_MIN_SCORE
    # The bar is a builder default, not baked in: an explicit lower bar
    # still admits the marginal name (used by research/backtests).
    loose = build_adr_breakout_candidates(profiles, context, min_score=0.0)
    assert {row["symbol"] for row in loose["longs"]} == {"FAST", "GRIND", "MEH"}


def test_aggressive_regime_candidate_golden_fixture():
    from autopilot_core import (
        AGGRESSIVE_EXTREME_WINDOW_BARS,
        AGGRESSIVE_MAX_DATA_AGE_MINUTES,
        AGGRESSIVE_MIN_EXTREME_BREAK_PCT,
        AGGRESSIVE_MIN_NEW_EXTREMES,
        AGGRESSIVE_NEAR_EXTREME_PCT,
        AGGRESSIVE_SPY_PULLBACK_MIN_MOVE_PCT,
        build_aggressive_regime_candidates,
    )

    fixture = json.loads(AGGRESSIVE_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["schema"] == "aggressive_watchlist_candidates_v1"
    profile_payload = json.dumps(fixture["profiles"], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(profile_payload.encode()).hexdigest() == fixture["raw_input_sha256"]
    assert fixture["configuration"] == {
        "recent_window_bars": AGGRESSIVE_EXTREME_WINDOW_BARS,
        "minimum_new_extremes": AGGRESSIVE_MIN_NEW_EXTREMES,
        "minimum_break_pct": AGGRESSIVE_MIN_EXTREME_BREAK_PCT,
        "near_extreme_pct": AGGRESSIVE_NEAR_EXTREME_PCT,
        "maximum_data_age_minutes": AGGRESSIVE_MAX_DATA_AGE_MINUTES,
        "minimum_spy_pullback_move_pct": AGGRESSIVE_SPY_PULLBACK_MIN_MOVE_PCT,
    }
    for case in fixture["cases"]:
        result = build_aggressive_regime_candidates(
            fixture["profiles"],
            case["environment"],
            spy_pullback_active=case["spy_pullback_active"],
        )
        stable = {
            side: [[row["symbol"], row["source_rule"]] for row in result[side]]
            for side in ("longs", "shorts")
        }
        assert stable == case["expected"], case["name"]


def test_intraday_extreme_metrics_exclude_forming_bar():
    from autopilot_core import _intraday_extreme_metrics

    tz = ZoneInfo("America/New_York")
    start = datetime(2026, 7, 14, 9, 30, tzinfo=tz)
    highs = [100.0, 100.06, 100.12, 100.18, 100.18, 100.18, 105.0]
    rows = [
        {
            "dt": start + timedelta(minutes=5 * index),
            "open": high - 0.1,
            "high": high,
            "low": high - 0.2,
            "close": high - 0.02,
        }
        for index, high in enumerate(highs)
    ]

    metrics = _intraday_extreme_metrics(
        rows,
        now=start + timedelta(minutes=34, seconds=59),
    )

    assert metrics["completed_bar_count"] == 6
    assert metrics["completed_day_high"] == 100.18
    assert metrics["recent_new_highs"] == 3
    assert metrics["as_of"] == (start + timedelta(minutes=30)).isoformat(timespec="seconds")
    assert metrics["data_health"] == "ok"

    stale = _intraday_extreme_metrics(
        rows,
        now=start + timedelta(minutes=51),
    )
    assert stale["data_health"] == "stale"


def test_new_legacy_spy_pullback_bypasses_auto_populate_cooldown(monkeypatch):
    import autopilot_core as core
    import bounce_bot_lib.legacy as legacy

    class ImmediateThread:
        def __init__(self, *, target, **_kwargs):
            self._target = target

        def start(self):
            self._target()

    calls = []
    monkeypatch.setattr(core, "minutes_since_open", lambda _now: 90)
    monkeypatch.setattr(legacy.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(legacy.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        core,
        "refresh_auto_populated_watchlists",
        lambda env, **kwargs: calls.append(
            (env, kwargs["spy_pullback_active"], kwargs["preserve_existing_auto"])
        )
        or {},
    )

    class Stub:
        _auto_populate_last_ts = 990.0  # well inside the ordinary 30-minute cooldown
        _auto_populate_running = False
        _auto_populate_spy_pullback_key = ""
        gui_callback = None

        @staticmethod
        def get_market_environment():
            return "bearish_strong"

        @staticmethod
        def _spy_session_bars(*, cached_only=False):
            assert cached_only
            return [object()], 100.0

        @staticmethod
        def _detect_spy_pause_start(_bars, side):
            assert side == "short"
            return datetime(2026, 7, 14, 11, 0)

        @staticmethod
        def _window_change_pct(_bars, _pause_start):
            return 0.08, _bars

    too_small = Stub()
    too_small._window_change_pct = lambda bars, _pause_start: (0.01, bars)
    legacy.BounceBot._maybe_refresh_auto_populated_watchlists(too_small)
    assert calls == []

    stub = Stub()
    legacy.BounceBot._maybe_refresh_auto_populated_watchlists(stub)
    legacy.BounceBot._maybe_refresh_auto_populated_watchlists(stub)

    assert calls == [("bearish_strong", True, True)]


def test_apply_auto_populate_rotates_only_owned_names(tmp_path, monkeypatch):
    import autopilot_core as core
    from candidate_registry import CandidateRegistry

    apply_auto_populated_watchlists = core.apply_auto_populated_watchlists
    record_auto_watchlist_cut = core.record_auto_watchlist_cut

    longs_path = tmp_path / "longs.txt"
    shorts_path = tmp_path / "shorts.txt"
    membership_path = tmp_path / "auto_membership.json"
    registry_path = tmp_path / "candidate_registry.json"
    monkeypatch.setattr(core, "candidate_registry_path", lambda: registry_path)
    longs_path.write_text("TRADER1\nTRADER2\n", encoding="utf-8")
    shorts_path.write_text("", encoding="utf-8")

    def candidates(long_syms, short_syms=()):
        return {
            "longs": [{"symbol": s, "score": 10 - i, "reason": "PDH break"} for i, s in enumerate(long_syms)],
            "shorts": [{"symbol": s, "score": 10 - i, "reason": "PDL break"} for i, s in enumerate(short_syms)],
        }

    first = apply_auto_populated_watchlists(
        candidates(["AAA", "BBB", "TRADER1"], ["ZZZ"]),
        "neutral_chop",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
    )
    # Trader names stay first; TRADER1 is not double-added as an auto name.
    assert read_watchlist_symbols(longs_path) == ["TRADER1", "TRADER2", "AAA", "BBB"]
    assert read_watchlist_symbols(shorts_path) == ["ZZZ"]
    assert first["long"]["added"] == ["AAA", "BBB"]
    assert first["long"]["trader_names"] == 2
    registry = CandidateRegistry.load(registry_path)
    assert registry.get("AAA", "LONG").memberships["auto_populate"].lease_expires_at
    assert "auto_populate" in registry.get("ZZZ", "SHORT").memberships
    assert registry.get("TRADER1", "LONG") is None

    # Next refresh: BBB drops out, CCC arrives; trader names untouched.
    second = apply_auto_populated_watchlists(
        candidates(["AAA", "CCC"], ["ZZZ"]),
        "neutral_chop",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
    )
    assert read_watchlist_symbols(longs_path) == ["TRADER1", "TRADER2", "AAA", "CCC"]
    assert second["long"]["rotated_out"] == ["BBB"]
    assert second["long"]["added"] == ["CCC"]
    registry = CandidateRegistry.load(registry_path)
    assert not registry.get("BBB", "LONG").active
    assert "auto_populate" in registry.get("CCC", "LONG").memberships

    # A VWAP-cut name is blacklisted for the day and not re-added.
    record_auto_watchlist_cut("AAA", "long", membership_path=membership_path)
    registry = CandidateRegistry.load(registry_path)
    assert not registry.get("AAA", "LONG").active
    third = apply_auto_populated_watchlists(
        candidates(["AAA", "CCC"], ["ZZZ"]),
        "neutral_chop",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
    )
    assert "AAA" not in read_watchlist_symbols(longs_path)
    assert third["long"]["total_auto"] == 1  # just CCC


def test_apply_auto_populate_respects_ceiling_and_side_exclusivity(tmp_path):
    from autopilot_core import AUTO_POPULATE_MAX_PER_SIDE, apply_auto_populated_watchlists

    longs_path = tmp_path / "longs.txt"
    shorts_path = tmp_path / "shorts.txt"
    membership_path = tmp_path / "auto_membership.json"

    long_rows = [{"symbol": f"L{i:03d}", "score": 500 - i, "reason": "PDH"} for i in range(200)]
    short_rows = [{"symbol": f"S{i:03d}", "score": 500 - i, "reason": "PDL"} for i in range(200)]
    # A symbol appearing on both sides only keeps its first (long) placement.
    short_rows.insert(0, {"symbol": "L000", "score": 999, "reason": "PDL"})

    summary = apply_auto_populated_watchlists(
        {"longs": long_rows, "shorts": short_rows},
        "bullish_strong",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
    )
    assert summary["caps"] == (AUTO_POPULATE_MAX_PER_SIDE, AUTO_POPULATE_MAX_PER_SIDE)
    assert len(read_watchlist_symbols(longs_path)) == AUTO_POPULATE_MAX_PER_SIDE
    shorts = read_watchlist_symbols(shorts_path)
    assert len(shorts) == AUTO_POPULATE_MAX_PER_SIDE
    assert "L000" not in shorts

    # A short candidate list keeps its natural size: the ceiling is not a
    # target (quality bar upstream decides how many qualify).
    sparse = apply_auto_populated_watchlists(
        {"longs": long_rows[:7], "shorts": short_rows[:5]},
        "bearish_weak",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
    )
    assert sparse["long"]["total_auto"] == 7
    assert len(read_watchlist_symbols(longs_path)) == 7


def test_pullback_refresh_appends_without_early_auto_rotation(tmp_path, monkeypatch):
    import autopilot_core as core

    longs_path = tmp_path / "longs.txt"
    shorts_path = tmp_path / "shorts.txt"
    membership_path = tmp_path / "auto_membership.json"
    monkeypatch.setattr(core, "candidate_registry_path", lambda: tmp_path / "registry.json")

    first = {
        "longs": [
            {"symbol": "OLD1", "score": 2.0, "reason": "scheduled"},
            {"symbol": "OLD2", "score": 1.0, "reason": "scheduled"},
        ],
        "shorts": [],
    }
    core.apply_auto_populated_watchlists(
        first,
        "bullish_strong",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
    )

    event = core.apply_auto_populated_watchlists(
        {
            "longs": [{"symbol": "NEW", "score": 3.0, "reason": "SPY pullback HOD"}],
            "shorts": [],
        },
        "bullish_strong",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
        preserve_existing_auto=True,
    )

    assert read_watchlist_symbols(longs_path) == ["OLD1", "OLD2", "NEW"]
    assert event["long"]["rotated_out"] == []
    assert event["long"]["added"] == ["NEW"]
    assert event["preserved_existing_auto"] is True
