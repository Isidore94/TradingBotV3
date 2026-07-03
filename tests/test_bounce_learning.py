"""Tests for the BounceBot learning loop (pure logic + compaction)."""

import csv
import sys
from datetime import datetime
from pathlib import Path

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
    }


def _sample_state():
    rows = [
        _perf_row("bounce_type", "short", "dynamic_vwap_lower_band", 18, 1.42),
        _perf_row("bounce_type", "long", "eod_vwap_upper_band", 25, -0.82),
        _perf_row("bounce_type", "long", "vwap", 61, 0.74),
        _perf_row("time_bucket", "short", "midday", 21, -0.35),
        _perf_row("time_bucket", "short", "opening_drive", 49, 1.44),
        _perf_row("market_environment", "long", "bearish_weak", 39, -0.23),
        _perf_row("master_avwap_priority_bucket", "long", "favorite_setup", 12, 1.41),
        _perf_row("bounce_type", "long", "thin_type", 4, -3.0),  # below min samples
    ]
    return learning.build_learning_state(rows)


def test_build_learning_state_thresholds_and_mutes():
    state = _sample_state()
    segments = state["segments"]
    assert "thin_type" not in str(segments["bounce_type"])  # min-samples filter
    losers = segments["bounce_type"]["long|eod_vwap_upper_band"]
    assert losers["muted"] is True
    assert losers["score_delta"] == -16
    winner = segments["bounce_type"]["short|dynamic_vwap_lower_band"]
    assert winner["muted"] is False
    assert winner["score_delta"] == 28
    # master context can boost but never mute
    fav = segments["master_avwap_priority_bucket"]["long|favorite_setup"]
    assert fav["muted"] is False
    env = segments["market_environment"]["long|bearish_weak"]
    assert env["muted"] is True


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
    assert verdict["composite_r"] > 1.0

    # Midday short mutes even when the bounce type is good
    verdict = learning.evaluate_bounce_quality(
        state,
        direction="short",
        bounce_types=["dynamic_vwap_lower_band"],
        time_bucket="midday",
        market_environment="",
    )
    assert verdict["muted"] is True


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


def test_time_bucket_for_matches_session_windows():
    assert learning.time_bucket_for(datetime(2026, 7, 2, 9, 45)) == "opening_drive"
    assert learning.time_bucket_for(datetime(2026, 7, 2, 11, 0)) == "late_morning"
    assert learning.time_bucket_for(datetime(2026, 7, 2, 13, 0)) == "midday"
    assert learning.time_bucket_for(datetime(2026, 7, 2, 14, 30)) == "afternoon"
    assert learning.time_bucket_for(datetime(2026, 7, 2, 15, 45)) == "closing_window"
    assert learning.time_bucket_for(None) == "unknown"


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
        _human_focus_symbols=lambda: {"HOOD"},
        _scan_cycle_index=0,
        latest_bars={},
    )
    priority = BounceBot.get_priority_scan_symbols(stub)
    assert priority == {"AAPL", "NVDA", "TSLA", "HOOD"}

    # Cycle 0 refreshes background; the next two defer it; cycle 3 refreshes.
    refresh_pattern = []
    for cycle in range(4):
        stub._scan_cycle_index = cycle
        refresh_pattern.append(BounceBot._is_background_refresh_cycle(stub))
    assert refresh_pattern == [True, False, False, True]


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
    stub.longs = list(longs)
    stub.shorts = list(shorts)
    stub.emitted = []
    stub.get_cached_5m_bars = lambda symbol, _spy=spy_bars, _m=sym_bars_map: (
        _spy if symbol == "SPY" else _m.get(symbol, [])
    )
    stub._emit_regime_pause_banger = lambda hit: stub.emitted.append(hit)
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

    # Same pause: no duplicate alert for AAOI.
    assert stub.check_regime_pause_setups() == []

    # SPY resumes making new lows -> pause state resets.
    resumed = spy + [_make_bar(datetime(2026, 7, 2, 10, 35), spy[-1].close, spy[-1].close, spy[-1].close - 1.5, spy[-1].close - 1.4)]
    stub.get_cached_5m_bars = lambda symbol: resumed if symbol == "SPY" else weak
    assert stub.check_regime_pause_setups() == []
    assert stub._regime_pause_state is None


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


def test_d1_flag_gate_requires_actionable_bucket_and_evidence():
    from types import SimpleNamespace

    from bounce_bot_lib.legacy import BounceBot

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
