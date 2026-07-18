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


def _trend(prev_close, ema15, sma50, sma200):
    return {"prev_close": prev_close, "ema15": ema15, "sma50": sma50, "sma200": sma200}


def test_daily_trend_gate_long_and_short_rules():
    from autopilot_core import passes_daily_trend_gate

    # Long: needs close >= 15EMA and >= 200SMA.
    assert passes_daily_trend_gate("long", _trend(110, 105, 102, 100)) is True
    assert passes_daily_trend_gate("long", _trend(103, 102, 101, 108)) is False  # below 200SMA
    assert passes_daily_trend_gate("long", _trend(104, 106, 100, 100)) is False  # below 15EMA
    # Short: needs close <= 15EMA and <= 50SMA.
    assert passes_daily_trend_gate("short", _trend(90, 95, 100, 80)) is True
    assert passes_daily_trend_gate("short", _trend(97, 98, 95, 120)) is False  # above 50SMA
    assert passes_daily_trend_gate("short", _trend(99, 96, 100, 80)) is False  # above 15EMA
    # Missing the required average -> fails (cannot verify the structure).
    assert passes_daily_trend_gate("long", _trend(110, 105, 102, None)) is False
    assert passes_daily_trend_gate("short", _trend(90, 95, None, 80)) is False
    assert passes_daily_trend_gate("long", None) is False


def test_build_watchlists_from_moves_applies_daily_trend_gate():
    from autopilot_core import build_watchlists_from_moves

    moves = {
        "SPY": {"early_move_pct": 0.0},
        "GOOD": {"early_move_pct": 3.0, "gap_pct": 3.0},   # above 15EMA + 200SMA
        "TRASH": {"early_move_pct": 3.0, "gap_pct": 3.0},  # 1-day pop, below 200SMA
        "WEAKGOOD": {"early_move_pct": -3.0, "gap_pct": -3.0},   # below 15EMA + 50SMA
        "WEAKTRASH": {"early_move_pct": -3.0, "gap_pct": -3.0},  # above 50SMA
    }
    trend = {
        "GOOD": _trend(110, 105, 102, 100),
        "TRASH": _trend(103, 102, 101, 108),
        "WEAKGOOD": _trend(90, 95, 100, 80),
        "WEAKTRASH": _trend(97, 98, 95, 120),
    }

    built = build_watchlists_from_moves(
        moves, {"early_move_pct": 0.0}, gap_min_pct=1.0, trend_context=trend
    )
    assert built["longs"] == ["GOOD"]
    assert built["shorts"] == ["WEAKGOOD"]

    # Fails open with no daily store: every mover is kept.
    ungated = build_watchlists_from_moves(moves, {"early_move_pct": 0.0}, gap_min_pct=1.0)
    assert set(ungated["longs"]) == {"GOOD", "TRASH"}
    assert set(ungated["shorts"]) == {"WEAKGOOD", "WEAKTRASH"}
    empty_store = build_watchlists_from_moves(
        moves, {"early_move_pct": 0.0}, gap_min_pct=1.0, trend_context={}
    )
    assert set(empty_store["longs"]) == {"GOOD", "TRASH"}


def test_filter_candidates_by_daily_trend_drops_trash_and_fails_open():
    from autopilot_core import filter_candidates_by_daily_trend

    candidates = {
        "longs": [{"symbol": "GOOD", "reason": "x"}, {"symbol": "TRASH", "reason": "y"}],
        "shorts": [{"symbol": "WEAKGOOD", "reason": "z"}, {"symbol": "WEAKTRASH", "reason": "w"}],
    }
    trend = {
        "GOOD": _trend(110, 105, 102, 100),
        "TRASH": _trend(103, 102, 101, 108),
        "WEAKGOOD": _trend(90, 95, 100, 80),
        "WEAKTRASH": _trend(97, 98, 95, 120),
    }

    gated = filter_candidates_by_daily_trend(candidates, trend)
    assert [row["symbol"] for row in gated["longs"]] == ["GOOD"]
    assert [row["symbol"] for row in gated["shorts"]] == ["WEAKGOOD"]

    # Empty context fails open (no daily store available) -> unchanged.
    passthrough = filter_candidates_by_daily_trend(candidates, {})
    assert [row["symbol"] for row in passthrough["longs"]] == ["GOOD", "TRASH"]


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


# ---------------------------------------------------------------------------
# Relative weakness/strength discovery (2026-07-17: SNPS/AA/SBLK/PSKY study)
# ---------------------------------------------------------------------------
def _rw_profile(
    move_pct,
    *,
    low_distance_pct=5.0,
    high_distance_pct=5.0,
    new_lows=0,
    new_highs=0,
    at_low=0.0,
    at_high=0.0,
):
    """A healthy completed-bars profile positioned move_pct off a 100.0 open."""
    last = 100.0 * (1.0 + move_pct / 100.0)
    day_high = last * (1.0 + high_distance_pct / 100.0)
    day_low = last * (1.0 - low_distance_pct / 100.0)
    return {
        "last": last,
        "last_complete": last,
        "completed_day_high": max(day_high, last),
        "completed_day_low": min(day_low, last),
        "completed_session_open": 100.0,
        "completed_move_pct": move_pct,
        "completed_time_at_high_frac": at_high,
        "completed_time_at_low_frac": at_low,
        "recent_new_highs": new_highs,
        "recent_new_lows": new_lows,
        "recent_window_bars": 6,
        "completed_bar_count": 12,
        "as_of": "2026-07-17T10:00:00",
        "data_age_minutes": 3.0,
        "data_health": "ok",
    }


def test_resolve_discovery_env_prefers_current_then_opening():
    from autopilot_core import resolve_discovery_env

    assert resolve_discovery_env("bearish_weak", "bearish_strong") == "bearish_weak"
    assert resolve_discovery_env("neutral_chop", "bearish_strong") == "bearish_strong"
    assert resolve_discovery_env("neutral_chop", "") == "neutral_chop"
    assert resolve_discovery_env("", None) == ""
    # A genuine reversal wins over the opening anchor.
    assert resolve_discovery_env("bullish_strong", "bearish_strong") == "bullish_strong"


def test_opening_environment_first_directional_read_sticks(tmp_path):
    from datetime import datetime as dt

    from autopilot_core import load_opening_environment, record_opening_environment

    path = tmp_path / "opening_env.json"
    morning = dt(2026, 7, 17, 7, 5)

    assert record_opening_environment("neutral_chop", path=path, now=morning) == ""
    assert load_opening_environment(path=path, now=morning) == ""
    assert record_opening_environment("bearish_strong", path=path, now=morning) == "bearish_strong"
    # Decay to neutral cannot erase the opening read...
    later = dt(2026, 7, 17, 11, 30)
    assert record_opening_environment("neutral_chop", path=path, now=later) == "bearish_strong"
    # ...nor can a later directional flip rewrite it (first write wins today).
    assert record_opening_environment("bullish_weak", path=path, now=later) == "bearish_strong"
    assert load_opening_environment(path=path, now=later) == "bearish_strong"
    # A new day starts clean.
    next_day = dt(2026, 7, 20, 7, 5)
    assert load_opening_environment(path=path, now=next_day) == ""
    assert record_opening_environment("bullish_strong", path=path, now=next_day) == "bullish_strong"


def test_relative_weakness_pre_candidates_bearish_anchor():
    from autopilot_core import build_relative_weakness_candidates

    profiles = {
        "SPY": _rw_profile(-0.5, low_distance_pct=0.4),
        # Way weaker than SPY, pressing its lows: the SNPS signature.
        "WEAK": _rw_profile(-3.5, low_distance_pct=0.2, new_lows=2, at_low=0.4),
        # Weaker than SPY but lifted off the lows with no fresh ones: skip.
        "LIFTED": _rw_profile(-3.5, low_distance_pct=2.5, new_lows=0),
        # Down, but only in line with SPY (excess under the bar): skip.
        "INLINE": _rw_profile(-1.8, low_distance_pct=0.1, new_lows=1),
        # Grinding new lows mid-day even while off the LOD by >1%: the AA case.
        "GRIND": _rw_profile(-2.9, low_distance_pct=1.4, new_lows=1, at_low=0.2),
        # Sick data never qualifies.
        "STALE": {**_rw_profile(-5.0, low_distance_pct=0.1, new_lows=3), "data_health": "stale"},
    }

    result = build_relative_weakness_candidates(profiles, "bearish_strong")

    assert result["longs"] == []
    short_symbols = [row["symbol"] for row in result["shorts"]]
    assert short_symbols == ["WEAK", "GRIND"]
    top = result["shorts"][0]
    assert top["source_rule"] == "relative_weakness"
    assert "RW vs SPY: -3.0% excess" in top["reason"]
    assert "rvol" not in top["reason"]  # pre-candidates carry no rvol claim


def test_relative_weakness_rvol_gate_and_scoring():
    from autopilot_core import build_relative_weakness_candidates

    profiles = {
        "SPY": _rw_profile(-0.5, low_distance_pct=0.4),
        "WEAK": _rw_profile(-3.5, low_distance_pct=0.2, new_lows=2, at_low=0.4),
        "THIN": _rw_profile(-4.5, low_distance_pct=0.2, new_lows=2),
    }

    gated = build_relative_weakness_candidates(
        profiles,
        "bearish_weak",
        rvol_by_symbol={"WEAK": 1.4, "THIN": 0.6},
    )
    symbols = [row["symbol"] for row in gated["shorts"]]
    assert symbols == ["WEAK"]  # THIN fails the >=1.00 rvol bar
    assert "rvol 1.40" in gated["shorts"][0]["reason"]

    # A name with no computable reading does not qualify either.
    missing = build_relative_weakness_candidates(
        profiles,
        "bearish_weak",
        rvol_by_symbol={"THIN": 0.6},
    )
    assert missing["shorts"] == []

    # Higher rvol sweetens the score (capped), never the other way round.
    hotter = build_relative_weakness_candidates(
        profiles, "bearish_weak", rvol_by_symbol={"WEAK": 4.4}
    )
    cooler = build_relative_weakness_candidates(
        profiles, "bearish_weak", rvol_by_symbol={"WEAK": 1.1}
    )
    assert hotter["shorts"][0]["score"] > cooler["shorts"][0]["score"]
    assert hotter["shorts"][0]["score"] - cooler["shorts"][0]["score"] <= 1.5


def test_relative_strength_mirrors_on_bullish_anchor():
    from autopilot_core import build_relative_weakness_candidates

    profiles = {
        "SPY": _rw_profile(0.4, high_distance_pct=0.3),
        "STRONG": _rw_profile(3.1, high_distance_pct=0.2, new_highs=2, at_high=0.5),
        "WEAKISH": _rw_profile(-2.8, low_distance_pct=0.2, new_lows=2),
    }

    result = build_relative_weakness_candidates(profiles, "bullish_strong")
    assert [row["symbol"] for row in result["longs"]] == ["STRONG"]
    assert result["shorts"] == []  # one-sided by anchor: no RW shorts on bullish days
    assert result["longs"][0]["source_rule"] == "relative_strength"
    assert "RS vs SPY: +2.7% excess" in result["longs"][0]["reason"]


def test_relative_weakness_needs_directional_anchor_and_healthy_spy():
    from autopilot_core import build_relative_weakness_candidates

    weak_only = {"WEAK": _rw_profile(-3.5, low_distance_pct=0.2, new_lows=2)}
    with_spy = {"SPY": _rw_profile(-0.5, low_distance_pct=0.4), **weak_only}

    assert build_relative_weakness_candidates(with_spy, "neutral_chop") == {"longs": [], "shorts": []}
    assert build_relative_weakness_candidates(weak_only, "bearish_strong") == {"longs": [], "shorts": []}
    sick_spy = {**with_spy, "SPY": {**with_spy["SPY"], "data_health": "stale"}}
    assert build_relative_weakness_candidates(sick_spy, "bearish_strong") == {"longs": [], "shorts": []}


def test_fetch_session_rvol_uses_same_slot_baseline():
    import pandas as pd

    from autopilot_core import fetch_session_rvol

    stamps = []
    volumes = []
    for day in range(1, 8):  # 6 prior sessions + today
        for bar in range(3):
            stamps.append(pd.Timestamp(2026, 7, 6 + day, 9, 30) + pd.Timedelta(minutes=5 * bar))
            volumes.append(100.0 if day < 7 else 250.0)
    frame = pd.DataFrame(
        {
            "Open": 10.0,
            "High": 10.1,
            "Low": 9.9,
            "Close": 10.0,
            "Volume": volumes,
        },
        index=pd.DatetimeIndex(stamps),
    )

    def downloader(symbols, *, period, interval):
        assert period == "1mo" and interval == "5m"
        assert symbols == ["WEAK"]
        return frame

    readings = fetch_session_rvol(["WEAK", "weak", ""], downloader=downloader)
    assert readings == {"WEAK": 2.5}


def test_refresh_pipeline_folds_rw_candidates_and_anchor(monkeypatch):
    import autopilot_core as core

    profiles = {
        "SPY": _rw_profile(-0.5, low_distance_pct=0.4),
        "WEAK": _rw_profile(-3.5, low_distance_pct=0.2, new_lows=2, at_low=0.4),
    }
    captured = {}

    def fake_profiles(pool, **kwargs):
        captured["profile_pool"] = list(pool)
        return profiles

    def fake_rvol(symbols, **kwargs):
        captured["rvol_symbols"] = list(symbols)
        return {"WEAK": 1.4}

    def fake_apply(candidates, env_key, **kwargs):
        captured["candidates"] = candidates
        captured["env_key"] = env_key
        return {}

    monkeypatch.setattr(core, "load_universe_pool", lambda: ["WEAK"])
    monkeypatch.setattr(core, "load_daily_context", lambda pool, **kwargs: {})
    monkeypatch.setattr(core, "fetch_intraday_profiles", fake_profiles)
    monkeypatch.setattr(core, "fetch_session_rvol", fake_rvol)
    monkeypatch.setattr(core, "apply_auto_populated_watchlists", fake_apply)

    summary = core.refresh_auto_populated_watchlists(
        "neutral_chop",
        opening_env_key="bearish_strong",
    )

    assert captured["profile_pool"][0] == "SPY"
    assert captured["rvol_symbols"] == ["WEAK"]
    assert [row["symbol"] for row in captured["candidates"]["shorts"]] == ["WEAK"]
    assert "rvol 1.40" in captured["candidates"]["shorts"][0]["reason"]
    # File rotation still runs under the LIVE env; the anchor drives discovery.
    assert captured["env_key"] == "neutral_chop"
    assert summary["discovery_env"] == "bearish_strong"
    assert summary["relative_candidates"]["shorts"] == 1
    assert summary["relative_candidates"]["feature_version"] == "relative_weakness_v1"
