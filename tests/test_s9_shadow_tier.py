"""S9: the shadow tier - information-weighted, beside the live tier, never live."""

import csv
import sys
from pathlib import Path

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bounce_bot_lib import learning  # noqa: E402


def _row(dimension, direction, segment, n, avg_r, std=1.0, **extra):
    row = {
        "dimension": dimension,
        "direction": direction,
        "segment": segment,
        "sample_count": n,
        "avg_close_r": avg_r,
        "std_close_r": std,
        "session_count": 20,
    }
    row.update(extra)
    return row


def test_state_keeps_the_close_r_spread():
    state = learning.build_learning_state([_row("bounce_type", "long", "vwap", 40, 0.2, std=1.3)])
    assert state["segments"]["bounce_type"]["long|vwap"]["std_close_r"] == 1.3


def test_information_weight_is_mean_over_se_capped_at_one():
    # SE = 1.0 / sqrt(100) = 0.1: a +0.03R mean carries 0.3, a +0.5R mean carries 1.
    thin = {"avg_close_r": 0.03, "sample_count": 100, "std_close_r": 1.0}
    strong = {"avg_close_r": 0.5, "sample_count": 100, "std_close_r": 1.0}
    assert abs(learning.segment_information(thin) - 0.3) < 1e-9
    assert learning.segment_information(strong) == 1.0
    # No spread stored: the fallback spread is used.
    fallback = {"avg_close_r": 0.03, "sample_count": 100}
    expected = 0.03 / (learning.SHADOW_S9_FALLBACK_STD_R / 10.0)
    assert abs(learning.segment_information(fallback) - expected) < 1e-9


def test_a_huge_near_zero_segment_no_longer_drowns_a_strong_combo():
    state = learning.build_learning_state(
        [
            _row("bounce_type", "long", "h1_ema10_bounce", 2961, -0.03),
            _row("bounce_combo", "long", "h1_ema10_bounce+regime_pause_rs", 300, 0.46),
        ]
    )
    shadow = learning.evaluate_shadow_tier(
        state,
        direction="long",
        bounce_types=["h1_ema10_bounce"],
        bounce_combo="h1_ema10_bounce+regime_pause_rs",
    )
    live = learning.evaluate_bounce_quality(
        state, direction="long", bounce_types=["h1_ema10_bounce"]
    )
    assert live["tier"] == "D"  # the live composite never blends the combo
    # Both segments are fully informative here, so the combo gets an equal say.
    assert shadow == {"tier": "A", "composite_r": 0.215}
    # A mean that sits inside its own noise gets little say however big n is.
    flat = learning.build_learning_state(
        [
            _row("bounce_type", "long", "h1_ema10_bounce", 2961, -0.002, std=1.5),
            _row("bounce_combo", "long", "h1_ema10_bounce+regime_pause_rs", 300, 0.46),
        ]
    )
    weight = learning.segment_information(flat["segments"]["bounce_type"]["long|h1_ema10_bounce"])
    assert weight < 0.1
    assert learning.evaluate_shadow_tier(
        flat,
        direction="long",
        bounce_types=["h1_ema10_bounce"],
        bounce_combo="h1_ema10_bounce+regime_pause_rs",
    )["tier"] == "S"


def test_time_and_environment_carry_a_quarter_weight():
    assert dict(learning.SHADOW_S9_DIMENSIONS)["time_bucket"] == 0.25
    assert dict(learning.SHADOW_S9_DIMENSIONS)["market_environment"] == 0.25
    state = learning.build_learning_state(
        [
            _row("bounce_type", "short", "vwap", 400, 0.20),
            _row("time_bucket", "short", "midday", 400, -0.60),
        ]
    )
    shadow = learning.evaluate_shadow_tier(
        state, direction="short", bounce_types=["vwap"], time_bucket="midday"
    )
    # Both fully informative: (1.0 * 0.20 + 0.25 * -0.60) / 1.25 = +0.04.
    assert shadow["composite_r"] == 0.04


def test_structural_regime_is_a_dimension_and_unknown_adds_nothing():
    state = learning.build_learning_state(
        [
            _row("bounce_type", "short", "vwap", 400, 0.10),
            _row("structural_regime", "short", "bear_channel_lower_highs", 400, 0.40),
        ]
    )
    known = learning.evaluate_shadow_tier(
        state, direction="short", bounce_types=["vwap"], structural_regime="bear_channel_lower_highs"
    )
    unknown = learning.evaluate_shadow_tier(state, direction="short", bounce_types=["vwap"])
    assert known["composite_r"] == 0.186  # (1.0 * 0.10 + 0.4 * 0.40) / 1.4
    assert unknown["composite_r"] == 0.1


def test_shadow_keeps_the_mute_and_drops_the_proven_floor():
    state = learning.build_learning_state(
        [
            _row("bounce_type", "long", "loser", 60, -0.8),
            _row("bounce_type", "long", "flat", 400, 0.0),
            _row("master_avwap_setup_family", "long", "fam", 20, 1.2, median_close_r=0.5),
        ]
    )
    muted = learning.evaluate_shadow_tier(state, direction="long", bounce_types=["loser"])
    assert muted["tier"] == "D"
    live = learning.evaluate_bounce_quality(
        state, direction="long", bounce_types=["flat"], setup_family="fam"
    )
    assert live["proven"] is True and live["tier"] == "S"
    assert learning.evaluate_shadow_tier(state, direction="long", bounce_types=["flat"])["tier"] == "C"
    assert learning.evaluate_shadow_tier(state, direction="long", bounce_types=["nothing"]) == {
        "tier": "B",
        "composite_r": None,
    }


def test_live_tier_on_the_golden_is_untouched_and_shadow_computes():
    golden = load_fixture_contract("s9_p14_tier_golden_v1")
    state = golden["raw"]["learning_state"]
    keys = ("direction", "bounce_types", "time_bucket", "market_environment", "priority_bucket",
            "focus_label", "bounce_combo", "setup_family")
    tiers = set()
    for case in golden["raw"]["cases"]:
        before = learning.evaluate_bounce_quality(state, **case["inputs"])
        shadow = learning.evaluate_shadow_tier(state, **{k: case["inputs"][k] for k in keys})
        assert learning.evaluate_bounce_quality(state, **case["inputs"]) == before
        assert before == golden["expected"][case["id"]]["quality"]
        tiers.add(shadow["tier"])
    assert tiers == {"S", "A", "B", "C", "D"}


def test_alert_quality_carries_the_shadow_beside_the_unchanged_live_verdict(monkeypatch):
    from bounce_bot_lib import legacy

    state = learning.build_learning_state([_row("bounce_type", "long", "vwap", 400, 0.3)])
    monkeypatch.setattr(learning, "load_bounce_learning_state", lambda path=None: state)
    monkeypatch.setattr(learning, "structural_regime_on", lambda day: "")
    row = {"market_environment": "", "signal_time": "2026-09-25 10:15:00"}
    quality = legacy.BounceBot._evaluate_bounce_alert_quality(
        legacy.BounceBot, "long", {"vwap": 1.0}, row
    )
    shadow = quality.pop("shadow_s9")
    assert quality == learning.evaluate_bounce_quality(
        state,
        direction="long",
        bounce_types=["vwap"],
        time_bucket=learning.time_bucket_for(legacy._bounce_quality_time(row)),
        bounce_combo="vwap",
    )
    assert shadow == {"tier": "S", "composite_r": 0.3}


def test_performance_rows_carry_the_spread_and_the_structural_regime(tmp_path):
    from bounce_bot_lib.legacy import build_intraday_bounce_performance_rows

    candidates = tmp_path / "candidates.csv"
    outcomes = tmp_path / "outcomes.csv"
    with candidates.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "event_type", "logged_at", "trade_date", "symbol", "direction", "bounce_types"])
        for index, day in enumerate(["2026-08-04", "2026-08-05", "2026-07-06"]):
            writer.writerow([f"e{index}", "confirmed", f"{day} 10:00:00", day, "AAA", "short", "vwap"])
    with outcomes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "event_type", "logged_at", "entry_time", "bars_elapsed", "close_r", "status"])
        for index, close_r in enumerate([1.0, -1.0, 0.5]):
            writer.writerow(
                [f"e{index}", "final", "2026-08-06T16:00:00", "2026-08-06T10:00:00", 12, close_r, "eod_complete"]
            )
    rows = build_intraday_bounce_performance_rows(
        candidates_path=candidates,
        outcomes_path=outcomes,
        min_samples=1,
        regime_segments=[{"start_date": "2026-08-01", "regime": "bear_channel_lower_highs"}],
    )
    by_key = {(row["dimension"], row["segment"]): row for row in rows}
    bounce = by_key[("bounce_type", "vwap")]
    assert bounce["sample_count"] == 3
    assert abs(bounce["std_close_r"] - 1.0408329997330665) < 1e-9
    regime = by_key[("structural_regime", "bear_channel_lower_highs")]
    assert regime["sample_count"] == 2  # the July row is before the first segment: unknown
    assert not any(row["dimension"] == "structural_regime" and row["segment"] == "unknown" for row in rows)


def test_bounce_log_records_the_shadow_beside_the_live_tier(tmp_path, monkeypatch):
    from bounce_bot_lib import legacy

    monkeypatch.setattr(legacy, "INTRADAY_BOUNCES_CSV", tmp_path / "intraday_bounces.csv")
    monkeypatch.setattr(legacy, "BOUNCE_LOG_FILENAME", tmp_path / "bounces.txt")
    monkeypatch.setattr(legacy, "DATA_DIR", tmp_path)
    quality = {"tier": "C", "composite_r": 0.01, "shadow_s9": {"tier": "A", "composite_r": 0.18}}
    legacy.BounceBot.log_bounce_to_file(
        legacy.BounceBot, "AAA", "long", {"vwap": 1.0}, None, {"time": "20260925  10:15:00"}, 0.1,
        quality=quality,
    )
    with (tmp_path / "intraday_bounces.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["tier"] == "C"
    assert rows[0]["shadow_s9_tier"] == "A"
    assert rows[0]["shadow_s9_composite_r"] == "0.18"


def test_tracker_reads_recent_shadow_rows_newest_first(tmp_path):
    from ui.panels.daytrade_tracker_panel import read_shadow_s9_rows

    path = tmp_path / "intraday_bounces.csv"
    path.write_text(
        "time_local,trade_date,symbol,direction,bounce_types,tier,composite_r,shadow_s9_tier,shadow_s9_composite_r\n"
        "09:40:00,2026-09-25,OLD,long,vwap,B,0.06,,\n"
        "09:45:00,2026-09-25,AAA,long,vwap,C,0.01,A,0.18\n"
        "09:50:00,2026-09-25,BBB,short,ema_15,A,0.16,D,-0.05\n",
        encoding="utf-8",
    )
    rows = read_shadow_s9_rows(path)
    assert [row["symbol"] for row in rows] == ["BBB", "AAA"]
    assert rows[1]["tier"] == "C" and rows[1]["shadow_s9_tier"] == "A"
    assert rows[1]["shadow_s9_composite_r"] == 0.18
    assert read_shadow_s9_rows(tmp_path / "missing.csv") == []
