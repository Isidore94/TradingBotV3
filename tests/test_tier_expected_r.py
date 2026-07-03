import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _labels(tiers):
    return {tier["label"]: [row["symbol"] for row in tier["rows"]] for tier in tiers}


def test_tier_partition_drops_negative_expected_r_to_b_not_a():
    from master_avwap_lib import legacy

    good = {"symbol": "AAA", "side": "SHORT", "score": 200, "expected_r": 0.10}
    # High static score but clearly-negative Expected-R: 2026-07-02 policy says
    # it earns NEITHER S nor A - it belongs on the B stalk list only.
    bad = {"symbol": "BBB", "side": "SHORT", "score": 250, "expected_r": -0.20}

    tiers = legacy._priority_partition_tier_rows(
        actionable_rows=[],
        report_rows=[bad],
        high_conviction_rows=[good, bad],
        best_swing_rows=[],
    )
    labels = _labels(tiers)
    assert "AAA" in labels["S Tier"]
    assert "BBB" not in labels["S Tier"]
    assert "BBB" not in labels["A Tier"]
    assert "BBB" in labels["B Tier"]


def test_tier_partition_keeps_missing_or_ok_expected_r_in_s():
    from master_avwap_lib import legacy

    no_expected = {"symbol": "CCC", "score": 100}  # fail-open: unknown ExpR stays S
    slightly_neg = {"symbol": "DDD", "score": 100, "expected_r": -0.01}  # above threshold

    tiers = legacy._priority_partition_tier_rows(
        actionable_rows=[],
        report_rows=[],
        high_conviction_rows=[no_expected, slightly_neg],
        best_swing_rows=[],
    )
    labels = _labels(tiers)
    assert "CCC" in labels["S Tier"]
    assert "DDD" in labels["S Tier"]
