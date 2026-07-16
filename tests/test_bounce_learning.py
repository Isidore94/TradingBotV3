"""Tests for the BounceBot learning loop (pure logic + compaction)."""

import csv
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bounce_bot_lib import learning  # noqa: E402


def _perf_row(dimension, direction, segment, n, avg_r, **extra):
    return {
        "dimension": dimension,
        "direction": direction,
        "segment": segment,
        "sample_count": n,
        "avg_close_r": avg_r,
        "stop_rate": extra.get("stop_rate", 0.5),
        "target_1r_rate": extra.get("target_1r_rate", 0.6),
        "avg_mfe_r": extra.get("avg_mfe_r"),
        "median_close_r": extra.get("median_close_r"),
        "session_count": extra.get("session_count"),
    }


def _sample_state():
    # Default rates (stop 0.5, 1R 0.6) give ambiguity 0.1 and harvest +0.2,
    # so entry_r = 0.9*avg_close_r + 0.02 for every row below.
    rows = [
        _perf_row("bounce_type", "short", "dynamic_vwap_lower_band", 18, 1.42),
        _perf_row("bounce_type", "long", "eod_vwap_upper_band", 40, -0.82, session_count=15),
        _perf_row("bounce_type", "long", "small_n_loser", 25, -0.82, session_count=15),
        _perf_row("bounce_type", "long", "young_loser", 40, -0.82, session_count=3),
        _perf_row("bounce_type", "long", "vwap", 61, 0.74),
        _perf_row("time_bucket", "short", "midday", 21, -0.35),
        _perf_row("time_bucket", "short", "opening_drive", 49, 1.44),
        _perf_row("market_environment", "long", "bearish_weak", 39, -0.23),
        _perf_row("master_avwap_priority_bucket", "long", "favorite_setup", 12, 1.41),
        _perf_row("bounce_type", "long", "thin_type", 4, -3.0),  # below min samples
    ]
    return learning.build_learning_state(rows)


def test_entry_quality_r_ambiguity_blend():
    # No rates -> close-R verbatim.
    assert learning.entry_quality_r(-0.55) == (-0.55, 0.0)
    # Clean measurements (stop + 1R rates below 1) -> close-R verbatim.
    entry_r, ambiguity = learning.entry_quality_r(0.48, 0.42, 0.25)
    assert ambiguity == 0.0 and entry_r == 0.48
    # Heavy both-touch segment: the 1R-harvest expectancy takes over.
    entry_r, ambiguity = learning.entry_quality_r(-0.55, 0.7759, 0.8103)
    assert 0.58 < ambiguity < 0.59
    assert 0.09 < entry_r < 0.11  # negative close-R, but 78% of entries reached +1R


def test_build_learning_state_thresholds_and_mutes():
    state = _sample_state()
    segments = state["segments"]
    assert "thin_type" not in str(segments["bounce_type"])  # min-samples filter
    losers = segments["bounce_type"]["long|eod_vwap_upper_band"]
    assert losers["muted"] is True
    assert losers["score_delta"] == -14  # entry_r -0.718 * 20
    # Mutes demand real evidence: sample floor and session spread.
    assert segments["bounce_type"]["long|small_n_loser"]["muted"] is False
    assert segments["bounce_type"]["long|young_loser"]["muted"] is False
    winner = segments["bounce_type"]["short|dynamic_vwap_lower_band"]
    assert winner["muted"] is False
    assert winner["score_delta"] == 26  # entry_r 1.298 * 20
    # Context can drag the tier but never mute.
    fav = segments["master_avwap_priority_bucket"]["long|favorite_setup"]
    assert fav["muted"] is False
    env = segments["market_environment"]["long|bearish_weak"]
    assert env["muted"] is False
    bucket = segments["time_bucket"]["short|midday"]
    assert bucket["muted"] is False


def test_evaluate_quality_mutes_proven_losers():
    state = _sample_state()
    verdict = learning.evaluate_bounce_quality(
        state,
        direction="long",
        bounce_types=["eod_vwap_upper_band"],
        time_bucket="opening_drive",
        market_environment="bullish_strong",
    )
    assert verdict["muted"] is True
    assert verdict["tier"] == "D"
    assert any("eod_vwap_upper_band" in reason for reason in verdict["mute_reasons"])

    verdict = learning.evaluate_bounce_quality(
        state,
        direction="short",
        bounce_types=["dynamic_vwap_lower_band"],
        time_bucket="opening_drive",
        market_environment="bullish_strong",
    )
    assert verdict["muted"] is False
    assert verdict["tier"] == "S"
    assert verdict["composite_r"] > 0.9

    # A bad midday drags the tier but no longer mutes a good bounce type —
    # and without the context veto the segment's PROVEN floor applies again.
    verdict = learning.evaluate_bounce_quality(
        state,
        direction="short",
        bounce_types=["dynamic_vwap_lower_band"],
        time_bucket="midday",
        market_environment="",
    )
    assert verdict["muted"] is False
    assert verdict["proven"] is True
    assert verdict["tier"] == "S"  # proven avg 1.42R clears the S bar


def test_evaluate_quality_without_state_is_neutral():
    verdict = learning.evaluate_bounce_quality(
        None, direction="long", bounce_types=["vwap"], time_bucket="midday"
    )
    assert verdict["tier"] == "B"
    assert verdict["muted"] is False


def test_evaluate_quality_favorite_context_boosts_tier():
    state = _sample_state()
    base = learning.evaluate_bounce_quality(
        state, direction="long", bounce_types=["vwap"], time_bucket="", market_environment=""
    )
    boosted = learning.evaluate_bounce_quality(
        state,
        direction="long",
        bounce_types=["vwap"],
        time_bucket="",
        market_environment="",
        priority_bucket="favorite_setup",
    )
    assert boosted["composite_r"] > base["composite_r"]


def test_proven_segments_marked_and_flagged_live():
    rows = [
        _perf_row("master_avwap_swing_trait", "long", "trendline_break_recent", 31, 1.93, median_close_r=1.72),
        _perf_row("bounce_combo", "short", "eod_vwap+impulse_retest_vwap_eod", 17, 0.53, median_close_r=1.64),
        _perf_row("master_avwap_setup_family", "long", "avwap_band_bounce", 53, 0.51, median_close_r=0.42),
        _perf_row("bounce_type", "long", "thin_winner", 11, 2.00, median_close_r=1.0),  # n below the proven bar
        _perf_row("bounce_type", "long", "skewed_winner", 40, 0.80, median_close_r=-0.10),  # outlier-carried avg
        _perf_row("time_bucket", "long", "late_morning", 200, 0.90, median_close_r=0.5),  # dim not proven-eligible
    ]
    state = learning.build_learning_state(rows)
    segments = state["segments"]
    assert segments["master_avwap_swing_trait"]["long|trendline_break_recent"]["proven"]
    assert segments["bounce_combo"]["short|eod_vwap+impulse_retest_vwap_eod"]["proven"]
    assert not segments["bounce_type"]["long|thin_winner"]["proven"]
    assert not segments["bounce_type"]["long|skewed_winner"]["proven"]
    assert not segments["time_bucket"]["long|late_morning"]["proven"]

    # A live bounce matching a proven swing trait gets flagged + S floor (avg >= S bar).
    verdict = learning.evaluate_bounce_quality(
        state,
        direction="long",
        bounce_types=["10_candle"],
        swing_traits=["trendline_break_recent"],
    )
    assert verdict["proven"] and verdict["tier"] == "S"
    assert any("trendline_break_recent" in reason for reason in verdict["proven_reasons"])

    # A proven combo below the S bar floors at A.
    combo_verdict = learning.evaluate_bounce_quality(
        state,
        direction="short",
        bounce_combo="eod_vwap+impulse_retest_vwap_eod",
    )
    assert combo_verdict["proven"] and combo_verdict["tier"] == "A"

    # No proven match -> unchanged behavior.
    plain = learning.evaluate_bounce_quality(state, direction="long", bounce_types=["vwap"])
    assert not plain["proven"] and plain["proven_reasons"] == []


def test_mute_overrides_proven():
    rows = [
        _perf_row("master_avwap_swing_trait", "long", "trendline_break_recent", 31, 1.93, median_close_r=1.72),
        _perf_row("bounce_type", "long", "orb_breakout", 40, -0.90, session_count=15),
    ]
    state = learning.build_learning_state(rows)
    verdict = learning.evaluate_bounce_quality(
        state,
        direction="long",
        bounce_types=["orb_breakout"],
        swing_traits=["trendline_break_recent"],
    )
    assert verdict["muted"] and not verdict["proven"] and verdict["tier"] == "D"


def test_golden_fixture_entry_quality_scoring():
    """Golden results (plan.md Milestone 3 discipline): real measured segments
    from the 2026-07-15 learning state, frozen through the entry-quality
    scoring path. Any drift in entry_r, mutes, or scenario tiers must be an
    intentional, fixture-updating change."""
    import json

    fixture_path = ROOT_DIR / "tests" / "fixtures" / "bounce_entry_quality_v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    state = learning.build_learning_state(fixture["perf_rows"])
    for dimension, expected_entries in fixture["expected_segments"].items():
        for key, expected in expected_entries.items():
            entry = state["segments"][dimension][key]
            assert entry["entry_r"] == expected["entry_r"], (dimension, key)
            assert entry["ambiguity"] == expected["ambiguity"], (dimension, key)
            assert entry["muted"] == expected["muted"], (dimension, key)
            assert entry["proven"] == expected["proven"], (dimension, key)

    for scenario in fixture["scenarios"]:
        verdict = learning.evaluate_bounce_quality(state, **scenario["kwargs"])
        expected = scenario["expected"]
        assert verdict["tier"] == expected["tier"], scenario["name"]
        assert verdict["muted"] == expected["muted"], scenario["name"]
        assert verdict["proven"] == expected["proven"], scenario["name"]
        assert verdict["composite_r"] == expected["composite_r"], scenario["name"]


def test_stale_v1_state_files_follow_the_new_mute_policy():
    """A pre-v2 learning state on disk (no entry_r, baked-in context mutes)
    must stop muting immediately: the mute verdict is recomputed live from
    the stored stats, and context dimensions are no longer mutable."""
    v1_state = {
        "schema_version": 1,
        "segments": {
            "time_bucket": {
                "long|opening_drive": {
                    "avg_close_r": -0.34,
                    "sample_count": 443,
                    "muted": True,  # baked in by the old builder
                }
            },
            "bounce_type": {
                "long|vwap": {"avg_close_r": 0.74, "sample_count": 61, "muted": False},
                "long|bad_type": {"avg_close_r": -0.9, "sample_count": 45, "muted": True},
            },
        },
    }
    verdict = learning.evaluate_bounce_quality(
        v1_state, direction="long", bounce_types=["vwap"], time_bucket="opening_drive"
    )
    assert verdict["muted"] is False  # the old context mute is ignored
    # A genuinely bad bounce_type (close-R fallback) still mutes.
    verdict = learning.evaluate_bounce_quality(
        v1_state, direction="long", bounce_types=["bad_type"]
    )
    assert verdict["muted"] is True


def test_format_bounce_alert_message_carries_proven_stamp():
    from bounce_bot_lib.legacy import _format_bounce_alert_message

    quality = {
        "tier": "S",
        "proven": True,
        "proven_reasons": ["trendline_break_recent: +1.93R (n=31)"],
        "reasons": ["dynamic_vwap_upper_band long +0.88R (n=59)"],
    }
    message = _format_bounce_alert_message("NVDA", "long", "dynamic_vwap_upper_band", {}, quality)
    assert message.startswith("[S-TIER] PROVEN NVDA:")
    assert "proven: trendline_break_recent: +1.93R (n=31)" in message

    unproven = _format_bounce_alert_message("NVDA", "long", "vwap", {}, {"tier": "B"})
    assert unproven.startswith("[B-TIER] NVDA:")
    assert "PROVEN" not in unproven


def test_time_bucket_for_matches_session_windows():
    eastern = ZoneInfo("America/New_York")
    assert learning.time_bucket_for(datetime(2026, 7, 2, 9, 45, tzinfo=eastern)) == "opening_drive"
    assert learning.time_bucket_for(datetime(2026, 7, 2, 11, 0, tzinfo=eastern)) == "late_morning"
    assert learning.time_bucket_for(datetime(2026, 7, 2, 13, 0, tzinfo=eastern)) == "midday"
    assert learning.time_bucket_for(datetime(2026, 7, 2, 14, 30, tzinfo=eastern)) == "afternoon"
    assert learning.time_bucket_for(datetime(2026, 7, 2, 15, 45, tzinfo=eastern)) == "closing_window"
    assert learning.time_bucket_for(None) == "unknown"


def test_time_bucket_for_is_correct_on_a_pacific_machine():
    pacific = ZoneInfo("America/Los_Angeles")
    assert learning.time_bucket_for(datetime(2026, 7, 13, 6, 45, tzinfo=pacific)) == "opening_drive"
    assert learning.time_bucket_for(datetime(2026, 7, 13, 9, 42, tzinfo=pacific)) == "midday"
    assert learning.time_bucket_for(datetime(2026, 7, 13, 12, 45, tzinfo=pacific)) == "closing_window"


def test_quality_time_prefers_signal_bar_over_notification_time():
    from bounce_bot_lib.legacy import _bounce_quality_time, _bounce_time_bucket

    row = {
        "signal_time": "2026-07-13T09:42:00-07:00",
        "logged_at": "2026-07-13T12:45:00-07:00",
    }
    signal_time = _bounce_quality_time(row)
    assert signal_time is not None
    assert learning.time_bucket_for(signal_time) == "midday"
    assert _bounce_time_bucket(row["signal_time"]) == "midday"


def test_compact_candidates_csv_event_type_aware_retention(tmp_path):
    from datetime import timedelta

    path = tmp_path / "candidates.csv"
    today = datetime.now().date()
    stale_near_miss = (today - timedelta(days=45)).isoformat()
    fresh_near_miss = (today - timedelta(days=5)).isoformat()
    old_confirmed = (today - timedelta(days=45)).isoformat()
    ancient_confirmed = (today - timedelta(days=400)).isoformat()
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "event_type", "trade_date", "symbol"])
        writer.writerow(["nm_old", "near_miss", stale_near_miss, "AAA"])
        writer.writerow(["nm_new", "near_miss", fresh_near_miss, "BBB"])
        writer.writerow(["conf_old", "confirmed", old_confirmed, "CCC"])  # kept: 365d window
        writer.writerow(["conf_ancient", "confirmed", ancient_confirmed, "DDD"])
        writer.writerow(["det_new", "detected", fresh_near_miss, "EEE"])

    result = learning.compact_bounce_candidates_csv(path, min_bytes_to_bother=1)
    assert result["compacted"] is True
    assert result["kept"] == 3
    assert result["dropped"] == 2
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["event_id"] for row in rows] == ["nm_new", "conf_old", "det_new"]


def test_compact_candidates_csv_noops_below_threshold(tmp_path):
    path = tmp_path / "candidates.csv"
    path.write_text("event_id,trade_date\nx,2020-01-01\n", encoding="utf-8")
    result = learning.compact_bounce_candidates_csv(path, min_bytes_to_bother=10_000_000)
    assert result["compacted"] is False


def test_priority_watchlist_emphasis_cycle_logic():
    from types import SimpleNamespace

    from bounce_bot_lib.legacy import BounceBot

    stub = SimpleNamespace(
        longs=["aapl", " nvda "],
        shorts=["tsla"],
        auto_longs=["gapr"],
        auto_shorts=[],
        _human_focus_symbols=lambda: {"HOOD"},
        _scan_cycle_index=0,
        latest_bars={},
    )
    stub._auto_watch_symbols = lambda side=None: BounceBot._auto_watch_symbols(stub, side)
    priority = BounceBot.get_priority_scan_symbols(stub)
    assert priority == {"AAPL", "NVDA", "TSLA", "HOOD", "GAPR"}

    # Cycle 0 refreshes background; the next two defer it; cycle 3 refreshes.
    refresh_pattern = []
    for cycle in range(4):
        stub._scan_cycle_index = cycle
        refresh_pattern.append(BounceBot._is_background_refresh_cycle(stub))
    assert refresh_pattern == [True, False, False, True]


def test_human_focus_fast_lane_runs_before_broad_scan_and_reuses_bars():
    from types import SimpleNamespace

    from bounce_bot_lib.legacy import BounceBot

    calls = []
    stub = SimpleNamespace(
        atr_cache={"AAPL": 2.0, "TSLA": 3.0},
        _human_focus_symbols=lambda: {"AAPL", "TSLA"},
        get_scan_symbol_set=lambda: {"AAPL", "TSLA", "NVDA"},
        is_scanning_enabled=lambda: True,
        request_and_detect_bounce=lambda symbol, allowed_bounce_types=None: calls.append(
            ("bounce", symbol, frozenset(allowed_bounce_types or ()))
        ),
        check_orb_break_setups=lambda symbols=None: calls.append(("orb", frozenset(symbols or ()))),
        check_ema8_grind_setups=lambda symbols=None: calls.append(("ema8", frozenset(symbols or ()))),
    )

    processed = BounceBot._scan_human_focus_fast_lane(stub, {"vwap", "ema_8"})

    assert processed == {"AAPL", "TSLA"}
    assert calls[:2] == [
        ("bounce", "AAPL", frozenset({"vwap", "ema_8"})),
        ("bounce", "TSLA", frozenset({"vwap", "ema_8"})),
    ]
    assert calls[2:] == [
        ("orb", frozenset({"AAPL", "TSLA"})),
        ("ema8", frozenset({"AAPL", "TSLA"})),
    ]


def test_prune_latest_bars_keeps_only_background_on_off_cycles():
    from types import SimpleNamespace

    from bounce_bot_lib.legacy import BounceBot

    bars = {
        "AAPL|5 D|5 mins": ["p"],  # priority -> refetch
        "SPY|5 D|5 mins": ["b"],  # benchmark -> refetch
        "XLK|5 D|5 mins": ["e"],  # sector ETF -> refetch
        "ZETA|5 D|5 mins": ["bg"],  # background -> keep
        "ZETA": ["bg"],  # plain-symbol alias key -> keep
    }
    stub = SimpleNamespace(latest_bars=dict(bars))
    BounceBot._prune_latest_bars_for_cycle(stub, False, {"ZETA"})
    assert set(stub.latest_bars) == {"ZETA|5 D|5 mins", "ZETA"}

    stub = SimpleNamespace(latest_bars=dict(bars))
    BounceBot._prune_latest_bars_for_cycle(stub, True, {"ZETA"})
    assert stub.latest_bars == {}


def _make_bar(dt, open_, high, low, close):
    from bounce_bot_lib.legacy import IbBar

    return IbBar(dt=dt, open=open_, high=high, low=low, close=close)


def _regime_stub(spy_bars, sym_bars_map, *, env, longs=(), shorts=()):
    import threading

    from bounce_bot_lib.legacy import BounceBot

    class Stub:
        pass

    for name in (
        "_spy_session_bars",
        "update_auto_market_environment",
        "_detect_spy_pause_start",
        "check_regime_pause_setups",
        "_sweep_regime_pause_bangers",
        "_regime_pause_day_alerted",
        "_regime_pause_observation_store",
        "_record_regime_pause_observation",
        "get_market_environment",
        "set_market_environment",
        "clear_market_environment_override",
    ):
        setattr(Stub, name, getattr(BounceBot, name))
    Stub._window_change_pct = BounceBot.__dict__["_window_change_pct"]

    stub = Stub()
    stub.market_environment = env
    stub.market_environment_lock = threading.Lock()
    stub.market_environment_user_override = False
    stub._regime_pause_state = None
    stub._regime_pause_observations = None
    stub.longs = list(longs)
    stub.shorts = list(shorts)
    stub.emitted = []
    stub.summaries = []
    stub.get_cached_5m_bars = lambda symbol, _spy=spy_bars, _m=sym_bars_map: (
        _spy if symbol == "SPY" else _m.get(symbol, [])
    )
    stub._record_regime_pause_banger = lambda hit: stub.emitted.append(hit)
    stub._emit_regime_pause_summary = lambda side, spy_window, hits, state: stub.summaries.append(
        (side, [hit["symbol"] for hit in hits])
    )
    stub._save_regime_pause_observations = lambda: None
    stub._refresh_rrs_gui = lambda **kwargs: None
    return stub


def _downtrend_session(base, *, candles, step, start=None, green_last=False):
    from datetime import datetime, timedelta

    start = start or datetime(2026, 7, 2, 9, 30)
    bars = []
    price = base
    for index in range(candles):
        open_ = price
        close = price + step  # step negative = falling
        if green_last and index == candles - 1:
            close = open_ + abs(step)  # one green candle
        high = max(open_, close) + 0.05
        low = min(open_, close) - 0.05
        bars.append(_make_bar(start + timedelta(minutes=5 * index), open_, high, low, close))
        price = close
    return bars


def test_auto_market_regime_tracks_spy_and_respects_override():
    from datetime import datetime, timedelta

    # Yesterday's close 100; today SPY trades down ~0.9% -> bearish_strong.
    prev = [_make_bar(datetime(2026, 7, 1, 15, 55), 100.0, 100.1, 99.9, 100.0)]
    today = _downtrend_session(100.0, candles=10, step=-0.09, start=datetime(2026, 7, 2, 9, 30))
    stub = _regime_stub(prev + today, {}, env="bullish_strong")

    env = stub.update_auto_market_environment()
    assert env == "bearish_strong"
    assert stub.get_market_environment() == "bearish_strong"
    assert stub.market_environment_user_override is False  # auto never pins

    # A manual selection pins the regime; auto then stands down.
    stub.set_market_environment("bullish_weak", source="user")
    assert stub.market_environment_user_override is True
    assert stub.update_auto_market_environment() is None
    assert stub.get_market_environment() == "bullish_weak"
    stub.clear_market_environment_override()
    assert stub.update_auto_market_environment() == "bearish_strong"


def test_regime_pause_flags_nonparticipating_weak_name():
    from datetime import datetime

    start = datetime(2026, 7, 2, 9, 30)
    # SPY: down all day, final candle green (the pause).
    spy = _downtrend_session(100.0, candles=12, step=-0.08, start=start, green_last=True)
    # AAOI-style: down 10x harder all day and STILL makes a new low on the pause.
    weak = _downtrend_session(140.0, candles=12, step=-1.6, start=start)
    # A name that bounces WITH SPY on the pause candle: not a banger.
    bouncer = _downtrend_session(50.0, candles=12, step=-0.3, start=start, green_last=True)

    stub = _regime_stub(spy, {"AAOI": weak, "BNCR": bouncer}, env="bearish_strong", shorts=["AAOI", "BNCR"])
    flagged = stub.check_regime_pause_setups()

    symbols = [hit["symbol"] for hit in flagged]
    assert symbols == ["AAOI"]
    hit = flagged[0]
    assert hit["side"] == "short"
    assert hit["day_excess"] > 5  # dramatically weaker than SPY on the day
    assert stub.emitted and stub.emitted[0]["symbol"] == "AAOI"
    # The feed surface is ONE summary line per sweep batch, not per symbol.
    assert stub.summaries == [("short", ["AAOI"])]
    # Pause defiance is recorded as swing-scan evidence.
    observations = stub._regime_pause_observations["sides"]["short"]
    assert observations["AAOI"]["pause_count"] == 1

    # Same pause: no duplicate alert for AAOI.
    assert stub.check_regime_pause_setups() == []

    # SPY resumes making new lows -> pause state resets.
    resumed = spy + [_make_bar(datetime(2026, 7, 2, 10, 35), spy[-1].close, spy[-1].close, spy[-1].close - 1.5, spy[-1].close - 1.4)]
    stub.get_cached_5m_bars = lambda symbol: resumed if symbol == "SPY" else weak
    assert stub.check_regime_pause_setups() == []
    assert stub._regime_pause_state is None

    # SPY pauses AGAIN later the same day; AAOI still looks like a banger but
    # already alerted today -> no re-spam, yet the defiance count still grows
    # so the swing scan sees "held through 2 pauses".
    spy_close = resumed[-1].close
    paused_again = resumed + [
        _make_bar(datetime(2026, 7, 2, 10, 40), spy_close, spy_close + 0.3, spy_close - 0.05, spy_close + 0.25)
    ]
    weak_again = weak + _downtrend_session(
        weak[-1].close, candles=3, step=-1.6, start=datetime(2026, 7, 2, 10, 30)
    )
    stub.get_cached_5m_bars = lambda symbol: (
        paused_again if symbol == "SPY" else (weak_again if symbol == "AAOI" else [])
    )
    assert stub.check_regime_pause_setups() == []
    assert stub._regime_pause_state is not None  # new pause tracked, alert suppressed
    assert stub.summaries == [("short", ["AAOI"])]  # still just the first summary
    assert stub._regime_pause_observations["sides"]["short"]["AAOI"]["pause_count"] == 2


def test_regime_pause_inverts_for_bullish_tape():
    from datetime import datetime

    start = datetime(2026, 7, 2, 9, 30)
    # Bullish tape: SPY up all day, final candle red (the pause).
    spy = [
        _make_bar(b.dt, 200 - (b.open - 100), 200 - (b.low - 100), 200 - (b.high - 100), 200 - (b.close - 100))
        for b in _downtrend_session(100.0, candles=12, step=-0.08, start=start, green_last=True)
    ]
    # Strong name: up far more than SPY and still making highs through the pause.
    strong = [
        _make_bar(b.dt, 280 - (b.open - 140), 280 - (b.low - 140), 280 - (b.high - 140), 280 - (b.close - 140))
        for b in _downtrend_session(140.0, candles=12, step=-1.6, start=start)
    ]
    stub = _regime_stub(spy, {"MSTR": strong}, env="bullish_strong", longs=["MSTR"])
    flagged = stub.check_regime_pause_setups()
    assert [hit["symbol"] for hit in flagged] == ["MSTR"]
    assert flagged[0]["side"] == "long"


def _h1_trend_bars(start_close, step, count, start=None):
    """Hourly bars with a steady per-bar close drift (for H1 color signals)."""
    from datetime import timedelta

    start = start or datetime(2026, 7, 2, 9, 30)
    bars = []
    close = start_close
    for index in range(count):
        open_ = close
        close = open_ + step
        high = max(open_, close) + 0.1
        low = min(open_, close) - 0.1
        bars.append(_make_bar(start + timedelta(hours=index), open_, high, low, close))
    return bars


def test_classify_h1_candle_color_covers_all_regimes():
    from bounce_bot_lib.legacy import classify_h1_candle_color

    # Uptrend structure: EMA15 above SMA20.
    assert classify_h1_candle_color(101.0, 100.0, 99.0) == "green"
    assert classify_h1_candle_color(99.5, 100.0, 99.0) == "orange"
    assert classify_h1_candle_color(98.5, 100.0, 99.0) == "yellow"
    # Downtrend structure: EMA15 below SMA20.
    assert classify_h1_candle_color(98.0, 99.0, 100.0) == "red"
    assert classify_h1_candle_color(101.0, 99.0, 100.0) == "blue"
    # Recovery still below the SMA20 has no color.
    assert classify_h1_candle_color(99.5, 99.0, 100.0) is None
    assert classify_h1_candle_color(None, 99.0, 100.0) is None


def test_h1_blue_after_red_flags_reclaim_long():
    from datetime import timedelta

    from bounce_bot_lib.legacy import H1_BLUE_AFTER_RED_TYPE, detect_h1_color_signals

    bars = _h1_trend_bars(100.0, -0.5, 30)  # red downtrend
    last = bars[-1]
    reclaim = _make_bar(last.dt + timedelta(hours=1), last.close, 91.2, last.close - 0.2, 91.0)
    hits = detect_h1_color_signals(bars + [reclaim], "long")

    assert [hit["type"] for hit in hits] == [H1_BLUE_AFTER_RED_TYPE]
    assert hits[0]["prev_color"] == "red"
    assert hits[0]["color"] == "blue"

    # The same tape is not a short signal.
    assert detect_h1_color_signals(bars + [reclaim], "short") == []


def test_h1_green_to_yellow_flags_breakdown_short():
    from datetime import timedelta

    from bounce_bot_lib.legacy import H1_GREEN_TO_YELLOW_TYPE, detect_h1_color_signals

    bars = _h1_trend_bars(100.0, 0.5, 30)  # green uptrend, close ~115
    last = bars[-1]
    dump = _make_bar(last.dt + timedelta(hours=1), last.close, last.close + 0.2, 108.8, 109.0)
    hits = detect_h1_color_signals(bars + [dump], "short")

    assert [hit["type"] for hit in hits] == [H1_GREEN_TO_YELLOW_TYPE]
    assert hits[0]["prev_color"] == "green"
    assert hits[0]["color"] == "yellow"

    # An orange pullback (holding the 20-SMA) is NOT the short signal.
    mild = _make_bar(last.dt + timedelta(hours=1), last.close, last.close + 0.2, 110.9, 111.1)
    assert detect_h1_color_signals(bars + [mild], "short") == []


def test_h1_ema10_bounce_detects_pierce_and_recover():
    from datetime import timedelta

    from bounce_bot_lib.legacy import H1_EMA10_BOUNCE_TYPE, detect_h1_color_signals

    bars = _h1_trend_bars(100.0, 0.5, 30)  # green uptrend; H1 10-EMA ~112.8
    last = bars[-1]
    # Green candle opens above the 10-EMA, pierces it intracandle, closes back above.
    bounce = _make_bar(last.dt + timedelta(hours=1), 115.4, 115.7, 112.9, 115.6)
    hits = detect_h1_color_signals(bars + [bounce], "long")
    assert [hit["type"] for hit in hits] == [H1_EMA10_BOUNCE_TYPE]

    # Same green candle without the 10-EMA tag: no bounce signal.
    no_touch = _make_bar(last.dt + timedelta(hours=1), 115.4, 115.9, 115.0, 115.6)
    assert detect_h1_color_signals(bars + [no_touch], "long") == []


def test_closed_h1_bars_drops_forming_bucket():
    from datetime import timedelta

    from bounce_bot_lib.legacy import _closed_h1_bars
    from market_session import get_market_session_open_naive

    session_open = get_market_session_open_naive(reference=datetime(2026, 7, 2, 12, 0))

    def _bars_5m(count):
        return [
            _make_bar(session_open + timedelta(minutes=5 * index), 100.0, 100.2, 99.8, 100.1)
            for index in range(count)
        ]

    # 12 bars = first hour complete; 3 extra bars form a partial second hour.
    assert len(_closed_h1_bars(_bars_5m(12))) == 1
    assert len(_closed_h1_bars(_bars_5m(15))) == 1
    assert len(_closed_h1_bars(_bars_5m(24))) == 2
    # A full 6.5h session: the short final bucket closes at the bell and counts.
    assert len(_closed_h1_bars(_bars_5m(78))) == 7


def test_h1_live_sweep_rejects_a_same_day_but_stale_symbol_bucket(monkeypatch):
    from datetime import timedelta

    import bounce_bot_lib.legacy as legacy
    from bounce_bot_lib.legacy import BounceBot

    class Stub:
        _evaluate_h1_color_signals = BounceBot._evaluate_h1_color_signals

    current = _make_bar(datetime(2026, 7, 13, 9, 30), 100.0, 101.0, 99.0, 100.5)
    stale = _make_bar(current.dt - timedelta(hours=1), 100.0, 101.0, 99.0, 100.5)
    stub = Stub()
    cache = {"SPY": [current], "AWK": [stale]}
    stub.get_cached_5m_bars = lambda symbol: cache.get(symbol, [])
    monkeypatch.setattr(legacy, "_closed_h1_bars", lambda bars: list(bars))
    monkeypatch.setattr(
        legacy,
        "detect_h1_color_signals",
        lambda bars, side: [{"type": "test", "signal_bar": bars[-1]}],
    )

    assert stub._evaluate_h1_color_signals("AWK", "long", current.dt.date()) == []

    cache["AWK"] = [current]
    assert stub._evaluate_h1_color_signals("AWK", "long", current.dt.date())


def test_h1_color_sweep_dedupes_per_candle():
    from datetime import timedelta

    from bounce_bot_lib.legacy import BounceBot, H1_EMA10_BOUNCE_TYPE

    class Stub:
        pass

    for name in ("check_h1_color_setups", "_watchlist_day_sweep_symbols"):
        setattr(Stub, name, getattr(BounceBot, name))

    today_bar = _make_bar(datetime(2026, 7, 2, 10, 30), 100.0, 100.5, 99.5, 100.2)
    signal_bar = _make_bar(datetime(2026, 7, 2, 10, 30), 115.4, 115.7, 112.9, 115.6)

    stub = Stub()
    stub._spy_session_bars = lambda: ([today_bar], 100.0)
    stub._h1_color_state = None
    stub.longs = ["MSTR"]
    stub.shorts = []
    stub.emitted = []
    stub._evaluate_h1_color_signals = lambda symbol, side, today: [
        {"type": H1_EMA10_BOUNCE_TYPE, "signal_bar": signal_bar, "symbol": symbol, "side": side}
    ]
    stub._emit_h1_color_alert = lambda hit: stub.emitted.append(hit)

    assert len(stub.check_h1_color_setups()) == 1
    # Same candle again: deduped.
    assert stub.check_h1_color_setups() == []
    assert len(stub.emitted) == 1

    # The next hourly candle is a fresh signal.
    next_bar = _make_bar(signal_bar.dt + timedelta(hours=1), 116.0, 116.4, 114.0, 116.2)
    stub._evaluate_h1_color_signals = lambda symbol, side, today: [
        {"type": H1_EMA10_BOUNCE_TYPE, "signal_bar": next_bar, "symbol": symbol, "side": side}
    ]
    assert len(stub.check_h1_color_setups()) == 1
    assert len(stub.emitted) == 2


def _daytrade_sweep_stub(spy_bars, sym_bars_map, *, longs=(), shorts=()):
    from bounce_bot_lib.legacy import BounceBot

    class Stub:
        pass

    for name in (
        "_spy_session_bars",
        "_watchlist_day_sweep_symbols",
        "_symbol_session_bars",
        "check_orb_break_setups",
        "_evaluate_orb_break",
        "check_ema8_grind_setups",
        "_evaluate_ema8_grind",
    ):
        setattr(Stub, name, getattr(BounceBot, name))

    stub = Stub()
    stub._orb_break_state = None
    stub._ema8_grind_state = None
    stub.longs = list(longs)
    stub.shorts = list(shorts)
    stub.orb_emitted = []
    stub.ema8_emitted = []
    stub.get_cached_5m_bars = lambda symbol, _spy=spy_bars, _m=sym_bars_map: (
        _spy if symbol == "SPY" else _m.get(symbol, [])
    )
    stub._emit_orb_break_alert = lambda hit: stub.orb_emitted.append(hit)
    stub._emit_ema8_grind_alert = lambda hit: stub.ema8_emitted.append(hit)
    return stub


def _session_bars(start, ohlc_rows):
    from datetime import timedelta

    return [
        _make_bar(start + timedelta(minutes=5 * index), o, h, l, c)
        for index, (o, h, l, c) in enumerate(ohlc_rows)
    ]


def _flat_session(start, candles, price=100.0, half_range=0.5):
    return _session_bars(
        start, [(price, price + half_range, price - half_range, price)] * candles
    )


def test_orb_break_fires_only_after_30_minutes():
    from datetime import date

    from market_session import get_market_session_open_naive

    # Machine-local session open (9:30 ET expressed in this box's timezone) -
    # the sweep must work wherever the bot runs, not only on Eastern clocks.
    start = get_market_session_open_naive(reference=date(2026, 7, 2))
    spy = _flat_session(start, 8)

    # BRKR: OR high 101; a +15m wick through survives; first CLOSE through at +30m.
    brkr = _session_bars(
        start,
        [
            (100.0, 101.0, 99.0, 100.5),  # 9:30 opening range
            (100.5, 100.9, 100.2, 100.6),
            (100.6, 100.8, 100.1, 100.4),
            (100.4, 101.3, 100.3, 100.8),  # +15m wick above the OR high, closes inside
            (100.8, 100.9, 100.4, 100.7),
            (100.7, 101.0, 100.5, 100.9),
            (100.9, 101.6, 100.8, 101.4),  # +30m first close above 101 -> alert
        ],
    )
    # ERLY: closes through the OR high at +10m -> the delayed break is dead all day.
    erly = _session_bars(
        start,
        [
            (50.0, 50.5, 49.5, 50.2),
            (50.2, 50.6, 50.0, 50.4),
            (50.4, 51.0, 50.3, 50.9),  # +10m early close through 50.5
            (50.9, 50.9, 50.2, 50.3),
            (50.3, 50.4, 50.0, 50.1),
            (50.1, 50.3, 49.9, 50.0),
            (50.0, 51.2, 49.9, 51.1),  # +30m breaks again - stays dead
        ],
    )
    stub = _daytrade_sweep_stub(spy, {"BRKR": brkr, "ERLY": erly}, longs=["BRKR", "ERLY"])
    hits = stub.check_orb_break_setups()
    assert [hit["symbol"] for hit in hits] == ["BRKR"]
    hit = hits[0]
    assert hit["side"] == "long"
    assert hit["level"] == 101.0
    assert hit["minutes_after_open"] == 30
    assert stub.orb_emitted and stub.orb_emitted[0]["symbol"] == "BRKR"
    assert "ERLY|long" in stub._orb_break_state["dead"]

    # Same session state: no duplicate alert on the next sweep.
    assert stub.check_orb_break_setups() == []


def test_orb_breakdown_shorts_and_stale_breaks_stay_quiet():
    from datetime import date

    from market_session import get_market_session_open_naive

    start = get_market_session_open_naive(reference=date(2026, 7, 2))
    spy = _flat_session(start, 13)

    # SHRT: OR low 49.5 holds on closes (one wick through) until +35m -> breakdown.
    shrt = _session_bars(
        start,
        [
            (50.0, 50.5, 49.5, 49.8),  # opening range
            (49.8, 50.0, 49.6, 49.7),
            (49.7, 49.9, 49.4, 49.6),  # wick below holds on the close
            (49.6, 49.8, 49.55, 49.7),
            (49.7, 49.9, 49.6, 49.8),
            (49.8, 49.9, 49.5, 49.6),
            (49.6, 49.7, 49.5, 49.55),
            (49.55, 49.6, 49.1, 49.2),  # +35m first close below 49.5
        ],
    )
    # STAL: legit +30m breakout, but first seen six candles later (restart) ->
    # stale news, marked dead instead of alerted.
    stal_rows = [
        (100.0, 101.0, 99.0, 100.5),
        (100.5, 100.9, 100.2, 100.6),
        (100.6, 100.8, 100.1, 100.4),
        (100.4, 100.9, 100.3, 100.8),
        (100.8, 100.9, 100.4, 100.7),
        (100.7, 101.0, 100.5, 100.9),
        (100.9, 101.6, 100.8, 101.4),  # +30m close through
    ] + [(101.4, 101.7, 101.2, 101.5)] * 6
    stal = _session_bars(start, stal_rows)

    stub = _daytrade_sweep_stub(spy, {"SHRT": shrt, "STAL": stal}, longs=["STAL"], shorts=["SHRT"])
    hits = stub.check_orb_break_setups()
    assert [hit["symbol"] for hit in hits] == ["SHRT"]
    assert hits[0]["side"] == "short"
    assert hits[0]["level"] == 49.5
    assert hits[0]["minutes_after_open"] == 35
    assert "STAL|long" in stub._orb_break_state["dead"]


def test_orb_break_window_expires_after_max_delay():
    """Evidence cap (2026-07-16): breaks 60-270m after the open measured as
    the worst bucket under every stop model - a first break past
    ORB_MAX_DELAY_MINUTES is stale drift, not the delayed-ORB setup."""
    from datetime import date

    from market_session import get_market_session_open_naive

    start = get_market_session_open_naive(reference=date(2026, 7, 2))
    spy = _flat_session(start, 20)

    def rows_with_break_at(minutes):
        rows = [(100.0, 101.0, 99.0, 100.5)]  # opening range, high 101
        rows += [(100.5, 100.9, 100.2, 100.6)] * (minutes // 5 - 1)
        rows.append((100.6, 101.6, 100.5, 101.4))  # first close above 101
        return rows

    # LATE: first break at +90m -> the window expired at +60m, no alert.
    late = _session_bars(start, rows_with_break_at(90))
    # EDGE: break exactly at +60m is still inside the window.
    edge = _session_bars(start, rows_with_break_at(60))
    stub = _daytrade_sweep_stub(spy, {"LATE": late, "EDGE": edge}, longs=["LATE", "EDGE"])
    hits = stub.check_orb_break_setups()
    assert [hit["symbol"] for hit in hits] == ["EDGE"]
    assert hits[0]["minutes_after_open"] == 60
    assert "LATE|long" in stub._orb_break_state["dead"]


def test_bounce_trade_plan_stop_override_and_atr_floor():
    """The trade plan (alert text + outcome measurement) honors a
    setup-defined stop and widens noise stops to the representative
    0.15*ATR floor so recorded R is comparable across setups."""
    from types import SimpleNamespace

    from bounce_bot_lib.legacy import BOUNCE_STOP_ATR_FLOOR, BounceBot

    stub = SimpleNamespace(atr_cache={"NOIS": 4.0, "WIDE": 4.0})
    stub._to_float_or_blank = lambda value: BounceBot._to_float_or_blank(stub, value)
    plan_for = lambda *args, **kwargs: BounceBot._build_bounce_trade_plan(stub, *args, **kwargs)

    # Tiny signal bar (risk 0.10 on a 100.00 stock, ATR 4.00): the floor
    # widens the stop to 0.15 * ATR = 0.60.
    from pytest import approx

    tiny_bar = {"time": "20260702  10:00:00", "open": 100.0, "high": 100.05, "low": 99.9, "close": 100.0}
    plan = plan_for("long", {"vwap": 99.9}, tiny_bar, symbol="NOIS")
    assert plan["risk_per_share"] == approx(BOUNCE_STOP_ATR_FLOOR * 4.0)
    assert plan["stop_price"] == approx(100.0 - 0.6)
    assert plan["target_1r"] == approx(100.6)

    # A wide signal bar keeps its own stop (floor only ever widens).
    wide_bar = {"time": "20260702  10:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
    plan = plan_for("long", {"vwap": 99.0}, wide_bar, symbol="WIDE")
    assert plan["stop_price"] == approx(99.0) and plan["risk_per_share"] == approx(1.0)

    # Setup-defined stop (ORB: OR opposite edge) replaces the bar extreme;
    # shorts mirror. No ATR known -> no floor, override still honored.
    stub2 = SimpleNamespace(atr_cache={})
    stub2._to_float_or_blank = lambda value: BounceBot._to_float_or_blank(stub2, value)
    short_bar = {"time": "20260702  10:00:00", "open": 50.0, "high": 50.1, "low": 49.8, "close": 49.9}
    plan = BounceBot._build_bounce_trade_plan(
        stub2, "short", {"orb_breakdown": 50.0}, short_bar, symbol="SHRT", stop_override=51.2
    )
    assert plan["stop_price"] == approx(51.2)
    assert plan["risk_per_share"] == approx(51.2 - 49.9)
    assert plan["target_1r"] == approx(49.9 - (51.2 - 49.9))


def test_orb_break_accepts_non_eastern_local_clocks(monkeypatch):
    # Regression (2026-07-13): bars are stamped in the machine's local clock,
    # so on a Pacific box the opening 5m candle is 06:30. The old hard-coded
    # (9, 30) guard rejected it and the ORB sweep never fired at all.
    from datetime import datetime

    import bounce_bot_lib.legacy as legacy

    pacific_open = datetime(2026, 7, 2, 6, 30)
    monkeypatch.setattr(
        legacy, "get_market_session_open_naive", lambda reference=None, **_: pacific_open
    )
    spy = _flat_session(pacific_open, 8)
    brkr = _session_bars(
        pacific_open,
        [
            (100.0, 101.0, 99.0, 100.5),  # opening range
            (100.5, 100.9, 100.2, 100.6),
            (100.6, 100.8, 100.1, 100.4),
            (100.4, 100.9, 100.3, 100.8),
            (100.8, 100.9, 100.4, 100.7),
            (100.7, 101.0, 100.5, 100.9),
            (100.9, 101.6, 100.8, 101.4),  # +30m first close above 101 -> alert
        ],
    )
    stub = _daytrade_sweep_stub(spy, {"BRKR": brkr}, longs=["BRKR"])
    hits = stub.check_orb_break_setups()
    assert [hit["symbol"] for hit in hits] == ["BRKR"]
    assert hits[0]["minutes_after_open"] == 30
    assert stub.orb_emitted and stub.orb_emitted[0]["symbol"] == "BRKR"


def _ema8_grind_session_rows():
    rows = [
        (100.0, 102.1, 99.9, 102.0),  # 9:30 rip
        (102.0, 102.9, 101.9, 102.8),
        (102.8, 103.1, 102.6, 103.0),  # HOD 103.1
        (103.0, 103.0, 102.1, 102.2),  # pullback
        (102.2, 102.3, 101.8, 101.9),
    ]
    for close in (101.9, 101.95, 101.9, 101.95, 101.9, 101.95):  # the grind
        rows.append((rows[-1][3], close + 0.15, close - 0.15, close))
    rows.append((101.95, 103.5, 101.9, 103.4))  # push into a new HOD
    return rows


def test_ema8_grind_squeeze_fires_on_new_hod_push():
    from datetime import datetime

    prev_day = _flat_session(datetime(2026, 7, 1, 12, 0), 20)
    start = datetime(2026, 7, 2, 9, 30)
    spy = _flat_session(start, 12)

    grind = prev_day + _session_bars(start, _ema8_grind_session_rows())
    # A one-way rip: every bar a new HOD, no pullback -> not this setup.
    rip_rows = [(100.0 + i, 101.2 + i, 99.9 + i, 101.0 + i) for i in range(12)]
    rip = prev_day + _session_bars(start, rip_rows)

    stub = _daytrade_sweep_stub(spy, {"GRND": grind, "RIPP": rip}, longs=["GRND", "RIPP"])
    hits = stub.check_ema8_grind_setups()
    assert [hit["symbol"] for hit in hits] == ["GRND"]
    hit = hits[0]
    assert hit["side"] == "long"
    assert hit["new_extreme"] == 103.5
    assert hit["day_pct"] > 3
    assert hit["squeeze_gap_atr"] <= 0.35
    assert stub.ema8_emitted and stub.ema8_emitted[0]["symbol"] == "GRND"

    # No duplicate on the next sweep.
    assert stub.check_ema8_grind_setups() == []


def test_ema8_grind_rejects_broken_grind_and_inverts_for_shorts():
    from datetime import datetime

    prev_day = _flat_session(datetime(2026, 7, 1, 12, 0), 20)
    start = datetime(2026, 7, 2, 9, 30)
    spy = _flat_session(start, 12)

    rows = _ema8_grind_session_rows()
    broken = list(rows)
    o, h, _l, _c = broken[8]
    broken[8] = (o, h, 100.8, 101.0)  # one 5m close below the 8-EMA kills the grind
    broken_bars = prev_day + _session_bars(start, broken)

    # Mirror of the good long session (price' = 200 - price): grind below the
    # 8-EMA into a new LOD. The flat-100 prior day is its own mirror.
    mirrored = [(200 - o, 200 - l, 200 - h, 200 - c) for (o, h, l, c) in rows]
    short_bars = prev_day + _session_bars(start, mirrored)

    stub = _daytrade_sweep_stub(
        spy, {"BRKN": broken_bars, "SHGR": short_bars}, longs=["BRKN"], shorts=["SHGR"]
    )
    hits = stub.check_ema8_grind_setups()
    assert [hit["symbol"] for hit in hits] == ["SHGR"]
    assert hits[0]["side"] == "short"
    assert hits[0]["new_extreme"] == 96.5  # 200 - 103.5


def _structure_df(rows):
    import pandas as pd

    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c} for (o, h, l, c) in rows]
    )


def test_session_structure_gate_wants_trend_plus_simple_retest():
    from bounce_bot_lib.legacy import _session_structure_report

    # Advance off the open, ONE pullback retesting lower, still up on the day.
    clean = _structure_df(
        [
            (100.0, 101.0, 99.9, 100.9),
            (100.9, 102.0, 100.8, 101.9),
            (101.9, 103.0, 101.8, 102.9),
            (102.9, 103.2, 102.7, 103.0),
            (103.0, 103.0, 102.0, 102.2),  # the pullback
            (102.2, 102.3, 101.6, 101.8),
            (101.8, 102.4, 101.7, 102.3),  # bounce candle
        ]
    )
    ok, reason = _session_structure_report(clean, "long", 0.5)
    assert ok, reason
    assert "pullbacks 1" in reason

    # Mirrored tape: same structure reads clean for a short.
    mirrored = _structure_df(
        [(200 - o, 200 - l, 200 - h, 200 - c) for (o, h, l, c) in clean[["open", "high", "low", "close"]].itertuples(index=False)]
    )
    ok_short, reason_short = _session_structure_report(mirrored, "short", 0.5)
    assert ok_short, reason_short


def test_session_structure_gate_rejects_compression_chop_and_giveback():
    from bounce_bot_lib.legacy import _session_structure_report

    # Compressed: no directional leg off the open.
    compressed = _structure_df(
        [
            (100.0, 100.4, 99.7, 100.1),
            (100.1, 100.5, 99.8, 100.0),
            (100.0, 100.3, 99.7, 100.2),
            (100.2, 100.6, 99.8, 100.4),
            (100.4, 100.5, 99.9, 100.3),
        ]
    )
    ok, reason = _session_structure_report(compressed, "long", 0.5)
    assert not ok and "compressed" in reason

    # Choppy: real advance but three deep pullbacks.
    choppy = _structure_df(
        [
            (100.0, 102.0, 99.9, 101.9),
            (101.9, 104.0, 101.8, 103.9),
            (103.9, 104.0, 102.2, 102.4),  # pullback 1
            (102.4, 104.2, 102.3, 104.0),
            (104.0, 104.2, 102.5, 102.7),  # pullback 2
            (102.7, 104.5, 102.6, 104.3),
            (104.3, 104.5, 102.8, 103.0),  # pullback 3
        ]
    )
    ok, reason = _session_structure_report(choppy, "long", 0.5)
    assert not ok and "choppy" in reason

    # Advance fully given back: not "moving higher" any more.
    giveback = _structure_df(
        [
            (100.0, 101.5, 99.9, 101.4),
            (101.4, 103.0, 101.3, 102.9),
            (102.9, 103.2, 101.0, 101.2),
            (101.2, 101.3, 99.5, 99.7),
        ]
    )
    ok, reason = _session_structure_report(giveback, "long", 0.5)
    assert not ok and "given back" in reason

    # Too few bars to judge: pass through, per-level checks still apply.
    young = _structure_df([(100.0, 100.2, 99.9, 100.1)] * 3)
    ok, _reason = _session_structure_report(young, "long", 0.5)
    assert ok


def _vwap_bar(dt, close, *, spread=0.1, volume=1000.0):
    from bounce_bot_lib.legacy import IbBar

    return IbBar(dt=dt, open=close - 0.05, high=close + spread, low=close - spread, close=close, volume=volume)


def test_vwap_regime_classification():
    from datetime import datetime, timedelta

    from bounce_bot_lib.legacy import _classify_spy_vwap_regime

    start = datetime(2026, 7, 2, 9, 30)

    # Steady trend day: closes ride above VWAP+1stdev for most of the session.
    strong = [_vwap_bar(start + timedelta(minutes=5 * i), 100.0 + i) for i in range(12)]
    assert _classify_spy_vwap_regime(strong, prev_close=99.5) == "bullish_strong"

    # Green but band-less chop: above VWAP, up on the day, never band-extended.
    weak_closes = [99.9] * 6 + [100.3] * 6
    weak = [_vwap_bar(start + timedelta(minutes=5 * i), c) for i, c in enumerate(weak_closes)]
    assert _classify_spy_vwap_regime(weak, prev_close=100.0) == "bullish_weak"

    # Mirrored: closes pinned under VWAP-1stdev -> bearish_strong.
    down = [_vwap_bar(start + timedelta(minutes=5 * i), 100.0 - i) for i in range(12)]
    assert _classify_spy_vwap_regime(down, prev_close=100.5) == "bearish_strong"

    # Too young / no volume -> None (caller falls back to the day% rule).
    assert _classify_spy_vwap_regime(strong[:6], prev_close=99.5) is None
    no_volume = [_vwap_bar(start + timedelta(minutes=5 * i), 100.0 + i, volume=0.0) for i in range(12)]
    assert _classify_spy_vwap_regime(no_volume, prev_close=99.5) is None


def test_auto_regime_reading_is_read_only_and_reports_possibilities():
    from datetime import datetime, timedelta

    from bounce_bot_lib.legacy import BounceBot, IbBar

    start = datetime(2026, 7, 8, 9, 30)
    prev_day = datetime(2026, 7, 7, 15, 55)
    bars = [IbBar(dt=prev_day, open=99.4, high=99.6, low=99.3, close=99.5, volume=1000.0)]
    bars += [
        _vwap_bar(start + timedelta(minutes=5 * i), 100.0 + i)  # trend day above the band
        for i in range(12)
    ]

    bot = BounceBot.__new__(BounceBot)
    bot.latest_bars = {"SPY|5 D|5 mins": bars}
    bot.market_environment_user_override = True  # override must NOT hide the auto read
    bot.market_environment = "bearish_weak"
    import threading

    bot.market_environment_lock = threading.Lock()

    reading = bot.get_auto_regime_reading()
    assert reading["env_key"] == "bullish_strong"
    assert reading["source"] == "vwap"
    assert reading["override_active"] is True
    assert reading["active_env_key"] == "bearish_weak"
    assert reading["above_band_frac"] > reading["band_fraction_needed"]
    assert reading["day_pct"] > 0
    # Read-only: the applied environment is untouched.
    assert bot.market_environment == "bearish_weak"

    # No cached bars -> no reading (never triggers an IB fetch).
    bot.latest_bars = {}
    assert bot.get_auto_regime_reading() is None


def _entry_stub_bot(env, spy_closes, symbol_closes, *, start=None, longs=(), shorts=()):
    """Minimal BounceBot stub with cached 5m bars for entry-assist tests."""
    import threading
    from datetime import datetime, timedelta

    from bounce_bot_lib.legacy import BounceBot

    start = start or datetime(2026, 7, 8, 9, 30)

    def bars_for(closes):
        return [_vwap_bar(start + timedelta(minutes=5 * i), c) for i, c in enumerate(closes)]

    bot = BounceBot.__new__(BounceBot)
    bot.latest_bars = {"SPY|5 D|5 mins": bars_for(spy_closes)}
    for symbol, closes in symbol_closes.items():
        bot.latest_bars[f"{symbol}|5 D|5 mins"] = bars_for(closes)
    bot.longs = list(longs)
    bot.shorts = list(shorts)
    bot.market_environment = env
    bot.market_environment_lock = threading.Lock()
    bot.market_environment_user_override = False
    bot.alerts = []
    bot.gui_callback = lambda message, tag: bot.alerts.append((str(message), str(tag)))
    return bot


def _extend_bars(bot, key, closes):
    from datetime import timedelta

    bars = bot.latest_bars[f"{key}|5 D|5 mins"]
    last_dt = bars[-1].dt
    bars.extend(
        _vwap_bar(last_dt + timedelta(minutes=5 * (i + 1)), c) for i, c in enumerate(closes)
    )


def test_entry_assist_mode_mapping():
    from bounce_bot_lib.legacy import entry_assist_mode_for_env

    assert entry_assist_mode_for_env("bullish_strong") == {"mode": "window", "sides": ("long",)}
    assert entry_assist_mode_for_env("bearish_strong") == {"mode": "window", "sides": ("short",)}
    assert entry_assist_mode_for_env("bullish_weak") == {"mode": "movers", "sides": ("long",)}
    assert entry_assist_mode_for_env("bearish_weak") == {"mode": "movers", "sides": ("short",)}
    assert entry_assist_mode_for_env("neutral_chop") == {"mode": "movers", "sides": ("long", "short")}


def test_entry_window_ranks_holders_through_spy_pullback():
    # Uptrend, then SPY pulls back; AAA holds flat (RS), BBB follows SPY down.
    bot = _entry_stub_bot(
        "bullish_strong",
        spy_closes=[100.0, 100.5, 101.0, 101.5, 102.0],
        symbol_closes={
            "AAA": [50.0, 50.2, 50.4, 50.6, 50.8],
            "BBB": [80.0, 80.4, 80.8, 81.2, 81.6],
        },
        longs=("AAA", "BBB"),
    )

    opened = bot.entry_assist_action()  # click 1: pullback started
    assert opened["ok"] and bot.entry_assist_state()["window_active"]
    assert any("ENTRY WINDOW OPEN" in message for message, _tag in bot.alerts)

    # During the pullback SPY drops 1%; AAA holds, BBB drops with it.
    _extend_bars(bot, "SPY", [101.5, 101.0])
    _extend_bars(bot, "AAA", [50.8, 50.8])
    _extend_bars(bot, "BBB", [80.8, 80.0])

    closed = bot.entry_assist_action()  # click 2: pullback over
    assert closed["ok"] and not bot.entry_assist_state()["window_active"]
    ranked = closed["results"]["long"]
    assert [row["symbol"] for row in ranked][0] == "AAA"
    summary = next(message for message, _tag in bot.alerts if message.startswith("ENTRY WINDOW (long)"))
    assert "AAA" in summary and "held strongest" in summary


def test_entry_assist_weak_and_chop_emit_trailing_movers():
    bot = _entry_stub_bot(
        "bullish_weak",
        spy_closes=[100.0] * 8,
        symbol_closes={
            "AAA": [50.0] * 4 + [50.0, 50.5, 51.0, 51.5],  # strongest 30m
            "BBB": [80.0] * 8,
        },
        longs=("AAA", "BBB"),
        shorts=("CCC",),
    )
    bot.latest_bars["CCC|5 D|5 mins"] = list(bot.latest_bars["BBB|5 D|5 mins"])

    result = bot.entry_assist_action()
    assert result["ok"]
    assert result["results"]["long"][0]["symbol"] == "AAA"
    assert any(message.startswith("STRONGEST 30M (long)") for message, _tag in bot.alerts)

    # Neutral/chop emits BOTH lists in one click.
    bot.market_environment = "neutral_chop"
    bot.alerts.clear()
    both = bot.entry_assist_action()
    assert set(both["results"]) == {"long", "short"}
    assert any(message.startswith("STRONGEST 30M (long)") for message, _tag in bot.alerts)
    assert any(message.startswith("WEAKEST 30M (short)") for message, _tag in bot.alerts)


def test_entry_assist_command_windows_toggle_and_switch_sides():
    bot = _entry_stub_bot(
        "neutral_chop",  # commands are regime-independent
        spy_closes=[100.0, 100.5, 101.0, 101.5, 102.0],
        symbol_closes={
            "AAA": [50.0, 50.2, 50.4, 50.6, 50.8],
            "CCC": [30.0, 29.8, 29.6, 29.4, 29.2],
        },
        longs=("AAA",),
        shorts=("CCC",),
    )

    opened = bot.entry_assist_command("pullback_window")
    assert opened["ok"]
    state = bot.entry_assist_state()
    assert state["window_active"] and state["window_sides"] == ["long"]

    # Clicking the OTHER window button closes the long window (emitting its
    # list) and opens a short one.
    switched = bot.entry_assist_command("bounce_window")
    assert switched["ok"]
    state = bot.entry_assist_state()
    assert state["window_active"] and state["window_sides"] == ["short"]

    # Same button again ends the window.
    _extend_bars(bot, "SPY", [102.5, 103.0])
    _extend_bars(bot, "AAA", [50.8, 50.8])
    _extend_bars(bot, "CCC", [29.0, 28.8])
    closed = bot.entry_assist_command("bounce_window")
    assert closed["ok"] and not bot.entry_assist_state()["window_active"]
    assert any(message.startswith("ENTRY WINDOW (short)") for message, _tag in bot.alerts)


def test_entry_assist_command_movers_and_unknown():
    bot = _entry_stub_bot(
        "bullish_strong",  # movers commands work even in a window regime
        spy_closes=[100.0] * 8,
        symbol_closes={
            "AAA": [50.0] * 4 + [50.0, 50.5, 51.0, 51.5],
            "CCC": [80.0] * 4 + [80.0, 79.5, 79.0, 78.5],
        },
        longs=("AAA",),
        shorts=("CCC",),
    )
    strongest = bot.entry_assist_command("strongest_30m")
    assert strongest["ok"] and set(strongest["results"]) == {"long"}
    weakest = bot.entry_assist_command("weakest_30m")
    assert weakest["ok"] and set(weakest["results"]) == {"short"}
    both = bot.entry_assist_command("movers_30m")
    assert both["ok"] and set(both["results"]) == {"long", "short"}
    assert any(message.startswith("STRONGEST 30M (long)") for message, _tag in bot.alerts)
    assert any(message.startswith("WEAKEST 30M (short)") for message, _tag in bot.alerts)

    unknown = bot.entry_assist_command("nope")
    assert not unknown["ok"] and "Unknown entry-assist command" in unknown["note"]


def test_entry_assist_board_snapshot_covers_every_option():
    # Trending SPY that stalls (no new highs for 3 candles) -> pause detected.
    bot = _entry_stub_bot(
        "bullish_strong",
        spy_closes=[100.0, 102.0, 101.0, 101.2, 101.1],
        symbol_closes={
            "AAA": [50.0, 50.5, 50.6, 50.7, 50.8],  # holds through the stall
            "CCC": [30.0, 29.8, 29.6, 29.4, 29.2],
        },
        longs=("AAA",),
        shorts=("CCC",),
    )
    board = bot.entry_assist_board_snapshot()
    assert board["env_key"] == "bullish_strong" and board["env_label"]
    assert board["pause"]["detected"] and board["pause"]["trend_side"] == "long"
    # No window open: the board previews what a pause window would say.
    assert not board["window"]["active"]
    preview = board["pause_preview"]
    assert preview["side"] == "long"
    assert any(row["symbol"] == "AAA" for row in preview["rows"])
    # Both-side trailing movers are always on the board, regardless of regime.
    assert {"long", "short"} <= set(board["movers"])
    assert any(row["symbol"] == "CCC" for row in board["movers"]["short"])

    # Open a window: the board now carries its LIVE ranking (pre-close).
    opened = bot.entry_assist_command("pullback_window")
    assert opened["ok"]
    board = bot.entry_assist_board_snapshot()
    assert board["window"]["active"] and board["window"]["sides"] == ["long"]
    assert "long" in board["window"]["rankings"]

    # No cached SPY bars -> empty board, never an IB fetch.
    bot.latest_bars = {}
    assert bot.entry_assist_board_snapshot() == {}


def test_rank_window_movers_uses_explicit_from_to_window():
    from datetime import datetime

    bot = _entry_stub_bot(
        "neutral_chop",
        # SPY: flat first 4 bars, then -1% over bars 4..7, then recovers.
        spy_closes=[100.0, 100.0, 100.0, 100.0, 99.5, 99.0, 99.0, 100.0],
        symbol_closes={
            "AAA": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],  # holds through the dip = RS
            "BBB": [80.0, 80.0, 80.0, 80.0, 79.0, 78.4, 78.4, 80.0],  # follows SPY down
            "CCC": [30.0, 30.0, 30.0, 30.0, 29.4, 29.0, 29.0, 30.0],  # weak = RW for shorts
        },
        longs=("AAA", "BBB"),
        shorts=("CCC",),
    )
    start = datetime(2026, 7, 8, 9, 50)  # bar index 4
    end = datetime(2026, 7, 8, 10, 0)  # bar index 6 (excludes the recovery bar)
    result = bot.rank_window_movers(start, end)
    assert result["ok"]
    assert result["spy_pct"] < 0
    by_symbol = {row["symbol"]: row for row in result["rows"]}
    assert by_symbol["AAA"]["excess"] > by_symbol["BBB"]["excess"]  # AAA held = most RS
    assert by_symbol["CCC"]["side"] == "SHORT" and by_symbol["CCC"]["excess"] > 0  # weak short = RW
    # The recovery bar after `end` must NOT be measured: BBB bounces back to
    # 80.0 on bar 7, so including it would flip its window move positive.
    assert by_symbol["BBB"]["window_pct"] < 0

    # A window with no SPY coverage reports cleanly instead of ranking garbage.
    empty = bot.rank_window_movers(datetime(2026, 7, 9, 9, 30), datetime(2026, 7, 9, 10, 0))
    assert not empty["ok"] and empty["rows"] == []


def test_rank_window_movers_completed_only_excludes_forming_m5_and_exports_same_cache():
    from datetime import datetime, timedelta

    start = datetime(2026, 7, 8, 9, 30)
    bot = _entry_stub_bot(
        "neutral_chop",
        spy_closes=[100.0, 100.0, 110.0],
        symbol_closes={"AAA": [50.0, 50.5, 60.0]},
        start=start,
        longs=("AAA",),
    )
    requested_end = start + timedelta(minutes=10)
    now = requested_end + timedelta(minutes=2)  # 09:40 bar is still forming.

    preview = bot.rank_window_movers(start, requested_end)
    completed = bot.rank_window_movers(
        start,
        requested_end,
        completed_only=True,
        now=now,
    )
    assert preview["spy_pct"] != completed["spy_pct"]
    assert completed["completed_only"] is True
    assert completed["data_complete_through"] == (start + timedelta(minutes=5)).isoformat(
        timespec="minutes"
    )

    cache = bot.cached_m5_window_bars(
        start,
        requested_end,
        completed_only=True,
        now=now,
    )
    assert set(cache) == {"SPY", "AAA"}
    assert all(len(rows) == 2 for rows in cache.values())
    assert all(rows[-1]["dt"] == start + timedelta(minutes=5) for rows in cache.values())


def test_completed_window_ranking_requires_exact_spy_endpoints_and_reports_coverage():
    from datetime import datetime, timedelta

    start = datetime(2026, 7, 8, 9, 30)
    bot = _entry_stub_bot(
        "neutral_chop",
        spy_closes=[100, 101, 102, 103, 104],
        symbol_closes={
            "AAA": [50, 51, 52, 53, 54],
            "BBB": [80, 81, 82, 83, 84],
        },
        start=start,
        longs=("AAA", "BBB"),
    )
    # AAA lacks SPY's exact first endpoint. The old preview remains permissive,
    # while the completed-bar advisory view excludes the misaligned name.
    bot.latest_bars["AAA|5 D|5 mins"] = bot.latest_bars["AAA|5 D|5 mins"][1:]
    end = start + timedelta(minutes=20)
    preview = bot.rank_window_movers(start, end)
    completed = bot.rank_window_movers(
        start,
        end,
        completed_only=True,
        now=end + timedelta(minutes=5),
    )

    assert {row["symbol"] for row in preview["rows"]} == {"AAA", "BBB"}
    assert {row["symbol"] for row in completed["rows"]} == {"BBB"}
    assert completed["candidate_coverage"] == 0.5
    assert completed["candidates_considered"] == 2
    assert completed["rows"][0]["timestamp_coverage"] == 1.0


def test_spy_m5_chart_bars_cached_only():
    bot = _entry_stub_bot(
        "neutral_chop",
        spy_closes=[100.0, 101.0, 102.0],
        symbol_closes={},
    )
    bars = bot.spy_m5_chart_bars()
    assert len(bars) == 3
    assert set(bars[0]) == {"dt", "open", "high", "low", "close", "volume"}
    assert bars[-1]["close"] == 102.0
    bot.latest_bars = {}
    assert bot.spy_m5_chart_bars() == []


def test_entry_assist_auto_tick_opens_and_closes_pullback_window():
    # Trending SPY that stops making new highs for 3 candles -> pause -> auto
    # window. (_vwap_bar candles are always green, so the no-new-high branch
    # of _detect_spy_pause_start is the trigger here.)
    bot = _entry_stub_bot(
        "bullish_strong",
        spy_closes=[100.0, 102.0, 101.0, 101.2, 101.1],  # highs stall under 102.1
        symbol_closes={"AAA": [50.0, 50.1, 50.2, 50.3, 50.3]},
        longs=("AAA",),
    )
    bot.entry_assist_auto_tick()
    state = bot.entry_assist_state()
    assert state["window_active"] and state["window_source"] == "auto"

    # Tape resumes with new session highs -> auto close + emit.
    _extend_bars(bot, "SPY", [102.5, 103.0])
    _extend_bars(bot, "AAA", [50.5, 50.7])
    bot.entry_assist_auto_tick()
    assert not bot.entry_assist_state()["window_active"]
    assert any(message.startswith("ENTRY WINDOW (long)") for message, _tag in bot.alerts)


def test_auto_regime_reading_classifies_mixed_mature_tape_as_chop():
    from datetime import datetime, timedelta

    from bounce_bot_lib.legacy import _spy_vwap_regime_stats

    start = datetime(2026, 7, 8, 9, 30)
    # Green on the day (prev 99.0) but chopping around VWAP and finishing under
    # it - mature session, no band hold, VWAP position disagrees with day color.
    closes = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 99.9]
    bars = [_vwap_bar(start + timedelta(minutes=5 * i), c) for i, c in enumerate(closes)]
    stats = _spy_vwap_regime_stats(bars, prev_close=99.0)
    assert stats is not None and stats["classification"] is None  # the mixed shape

    bot = _entry_stub_bot("bullish_weak", spy_closes=closes, symbol_closes={})
    from datetime import datetime as _dt

    prev_day = _vwap_bar(_dt(2026, 7, 7, 15, 55), 99.0)
    bot.latest_bars["SPY|5 D|5 mins"] = [prev_day] + bot.latest_bars["SPY|5 D|5 mins"]
    reading = bot.get_auto_regime_reading()
    assert reading["env_key"] == "neutral_chop"
    assert reading["label"] == "Neutral / Chop"


def test_learning_state_keeps_mfe_and_exit_note_renders():
    rows = [
        _perf_row("bounce_type", "long", "ema_8", 23, 0.5, avg_mfe_r=2.1, median_close_r=0.3),
        _perf_row("bounce_type", "long", "vwap", 40, 0.2, avg_mfe_r=0.8),
        _perf_row("bounce_type", "short", "ema_8", 15, 0.4),  # no MFE recorded
    ]
    state = learning.build_learning_state(rows)
    segment = state["segments"]["bounce_type"]["long|ema_8"]
    assert segment["avg_mfe_r"] == 2.1
    assert segment["median_close_r"] == 0.3

    # Big MFE vs small close -> the harvest advice appears.
    note = learning.measured_exit_note(state, direction="long", bounce_types=["ema_8"])
    assert "avg MFE 2.1R" in note and "n=23" in note and "harvest" in note

    # Modest MFE: stats only, no advice.
    vwap_note = learning.measured_exit_note(state, direction="long", bounce_types=["vwap"])
    assert "avg MFE 0.8R" in vwap_note and "harvest" not in vwap_note

    # Best-sampled matching segment wins when several types triggered.
    both = learning.measured_exit_note(state, direction="long", bounce_types=["ema_8", "vwap"])
    assert "vwap" in both  # n=40 beats n=23

    # No evidence -> empty note, alert stays clean.
    assert learning.measured_exit_note(state, direction="short", bounce_types=["ema_8"]) == ""
    assert learning.measured_exit_note(None, direction="long", bounce_types=["ema_8"]) == ""


def test_measured_exit_suffix_never_raises():
    from bounce_bot_lib.legacy import BounceBot

    class Stub:
        pass

    Stub._measured_exit_suffix = BounceBot._measured_exit_suffix
    stub = Stub()
    # Whatever the learning state on disk looks like, the suffix is a string.
    assert isinstance(stub._measured_exit_suffix("long", {"ema_8": 101.5}), str)
    assert isinstance(stub._measured_exit_suffix("short", None), str)


def test_auto_watchlists_get_same_treatment_as_trader_lists():
    from bounce_bot_lib.legacy import BounceBot

    class Stub:
        pass

    for name in (
        "get_symbol_direction",
        "get_scan_symbol_set",
        "get_priority_scan_symbols",
        "_auto_watch_symbols",
    ):
        setattr(Stub, name, getattr(BounceBot, name))

    stub = Stub()
    stub.longs = ["AAPL"]
    stub.shorts = ["XYZ"]
    stub.auto_longs = ["NVDA", "XYZ"]  # XYZ conflicts with the trader's short
    stub.auto_shorts = ["AMD"]
    stub._human_focus_side_for_symbol = lambda symbol, direction=None: ""
    stub._human_focus_symbols = lambda: set()
    stub.master_avwap_d1_watchlist = {}
    stub.master_avwap_d1_upgrade_alerts = {}
    stub.master_avwap_focus_map = {}
    stub.get_master_avwap_d1_watch_symbols = lambda: []

    # Bot picks get directions like watchlist names...
    assert stub.get_symbol_direction("NVDA") == "long"
    assert stub.get_symbol_direction("AMD") == "short"
    # ...but the trader's call always wins a conflict.
    assert stub.get_symbol_direction("XYZ") == "short"

    # And they are scanned with full (priority) treatment.
    assert {"NVDA", "AMD", "AAPL", "XYZ"} <= stub.get_scan_symbol_set()
    assert {"NVDA", "AMD"} <= stub.get_priority_scan_symbols()


def test_pacing_backoff_registers_and_escalates():
    import threading

    from bounce_bot_lib.legacy import (
        IB_PACING_BACKOFF_INITIAL_SECONDS,
        IB_PACING_BACKOFF_MAX_SECONDS,
        BounceBot,
    )

    class Stub:
        pass

    for name in ("_register_pacing_violation", "pacing_delay_remaining"):
        setattr(Stub, name, getattr(BounceBot, name))

    stub = Stub()
    stub.pacing_lock = threading.Lock()
    stub.pacing_backoff_seconds = 0.0
    stub.pacing_backoff_until = 0.0
    stub.last_pacing_violation_at = 0.0
    stub.gui_callback = None

    assert stub.pacing_delay_remaining() == 0.0
    stub._register_pacing_violation(162)
    assert stub.pacing_backoff_seconds == IB_PACING_BACKOFF_INITIAL_SECONDS
    assert stub.pacing_delay_remaining() > 0

    # A repeat violation inside the reset window doubles the cooldown, capped.
    stub._register_pacing_violation(100)
    assert stub.pacing_backoff_seconds == IB_PACING_BACKOFF_INITIAL_SECONDS * 2
    for _ in range(10):
        stub._register_pacing_violation(100)
    assert stub.pacing_backoff_seconds == IB_PACING_BACKOFF_MAX_SECONDS


def test_d1_flag_gate_requires_actionable_bucket_and_evidence(monkeypatch):
    from types import SimpleNamespace

    import bounce_bot_lib.learning as learning
    from bounce_bot_lib.legacy import BounceBot

    # Hermetic: the gate consults the LIVE learning state and wall clock; on
    # a machine whose state mutes the current time bucket (e.g. opening_drive
    # proven negative) the unpatched test fails at some hours of the day.
    monkeypatch.setattr(learning, "load_bounce_learning_state", lambda: {})

    stub = SimpleNamespace(
        _human_focus_side_for_symbol=lambda symbol: "",
        master_avwap_focus_map={},
        _describe_master_avwap_focus=lambda entry: "",
        get_market_environment=lambda: "bullish_strong",
    )

    emit, reason = BounceBot._should_emit_d1_flag(
        stub, {"symbol": "AAA", "priority_bucket": "near_favorite_zone", "side": "LONG"}
    )
    assert emit is False and "not an actionable bucket" in reason

    emit, reason = BounceBot._should_emit_d1_flag(
        stub, {"symbol": "AAA", "priority_bucket": "favorite_setup", "side": "LONG", "expected_r": -0.4}
    )
    assert emit is False and "expected R" in reason

    emit, _reason = BounceBot._should_emit_d1_flag(
        stub, {"symbol": "AAA", "priority_bucket": "favorite_setup", "side": "LONG", "expected_r": 0.25}
    )
    assert emit is True

    # human focus picks always alert, regardless of bucket
    focus_stub = SimpleNamespace(
        _human_focus_side_for_symbol=lambda symbol: "LONG",
        master_avwap_focus_map={},
        _describe_master_avwap_focus=lambda entry: "",
        get_market_environment=lambda: "bullish_strong",
    )
    emit, reason = BounceBot._should_emit_d1_flag(
        focus_stub, {"symbol": "AAA", "priority_bucket": "near_favorite_zone", "side": "LONG"}
    )
    assert emit is True and reason == "human focus pick"


def test_migrate_csv_header_widens_old_files(tmp_path):
    from bounce_bot_lib.legacy import _migrate_csv_header

    path = tmp_path / "bounces.csv"
    path.write_text(
        "time_local,trade_date,symbol,direction,bounce_types\n"
        "10:00:00,2026-07-01,AAPL,long,vwap\n",
        encoding="utf-8",
    )
    fieldnames = ["time_local", "trade_date", "symbol", "direction", "bounce_types", "tier", "composite_r"]
    _migrate_csv_header(path, fieldnames)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == fieldnames
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["tier"] == ""
    # idempotent on the new layout
    _migrate_csv_header(path, fieldnames)
    with open(path, newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    # unknown layouts are left untouched
    other = tmp_path / "other.csv"
    other.write_text("a,b\n1,2\n", encoding="utf-8")
    _migrate_csv_header(other, fieldnames)
    assert other.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_format_bounce_alert_message_includes_tier_plan_and_reasons():
    from bounce_bot_lib.legacy import _format_bounce_alert_message

    message = _format_bounce_alert_message(
        "AAOI",
        "long",
        ["vwap"],
        {"entry_price": 12.34, "stop_price": 12.10, "risk_per_share": 0.24, "target_1r": 12.58},
        {"tier": "A", "reasons": ["vwap long +0.74R (n=61)"]},
    )
    assert message.startswith("[A-TIER] AAOI: Bounce confirmed (long)")
    assert "entry 12.34, stop 12.10 (risk 0.24)" in message
    assert "take 50% at +1R 12.58" in message
    assert "why: vwap long +0.74R" in message
