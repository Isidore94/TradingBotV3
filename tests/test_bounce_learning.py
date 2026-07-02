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
