import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_coarse_regime_bucket_maps_labels():
    from master_avwap_lib import legacy

    assert legacy._coarse_regime_bucket("bullish") == "bull"
    assert legacy._coarse_regime_bucket("bullish_strong") == "bull"
    assert legacy._coarse_regime_bucket("bearish") == "bear"
    assert legacy._coarse_regime_bucket("mixed") == "neutral"
    assert legacy._coarse_regime_bucket("") == ""


def test_setup_regime_label_reads_feature_row():
    from master_avwap_lib import legacy

    assert legacy._setup_regime_label({"feature_row": {"market_regime_label": "bullish"}}) == "bullish"
    assert legacy._setup_regime_label({"market_regime_label": "bearish"}) == "bearish"
    assert legacy._setup_regime_label({}) == ""


def test_regime_conditioning_downweights_cross_regime_history(monkeypatch):
    from master_avwap_lib import legacy

    def fake_outcome(setup):
        r = setup["_r"]
        return {
            "tradeable_scenario_count": 1,
            "closed_tradeable_scenario_count": 1,
            "avg_total_r": r,
            "avg_closed_r": r,
            "representative_total_r": r,
            "representative_closed_r": r,
            "any_target_hit": r > 0,
            "any_stopped": r < 0,
        }

    monkeypatch.setattr(legacy, "_summarize_tracker_setup_outcome", fake_outcome)

    today = legacy.datetime.now().date()

    def make(symbol, r, regime):
        return {
            "symbol": symbol,
            "side": "SHORT",
            "priority_bucket": "favorite_setup",
            "scan_date": today.isoformat(),
            "anchor_date": "2026-01-02",
            "setup_status": "CLOSED",
            "feature_row": {"market_regime_label": regime},
            "_r": r,
        }

    # a same-regime winner and a cross-regime loser, same family group
    setups = {"win": make("AAA", 1.0, "bullish"), "loss": make("BBB", -1.0, "mixed")}

    pooled = legacy.build_recent_tracker_setup_family_rows(setups, reference_date=today)
    conditioned = legacy.build_recent_tracker_setup_family_rows(
        setups, reference_date=today, current_regime_label="bullish"
    )

    assert pooled and conditioned
    # pooled ~ mean(+1, -1) = 0; conditioning halves the cross-regime loser -> shifts positive
    assert conditioned[0]["avg_closed_r"] > pooled[0]["avg_closed_r"]
    assert conditioned[0]["avg_closed_r"] > 0
