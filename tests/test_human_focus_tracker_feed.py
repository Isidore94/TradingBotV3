import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_human_focus_comparison_rows_weight_bot_sa_baseline():
    from ui.services.human_focus_tracker_feed import build_human_focus_comparison_rows

    human_rows = [
        {
            "cohort": "human_focus_pick",
            "side": "ALL",
            "horizon_sessions": "5",
            "sample_count": "4",
            "win_rate": "0.7500",
            "avg_side_return": "0.0400",
            "profit_factor": "3.0",
        },
        {
            "cohort": "human_focus_pick",
            "side": "LONG",
            "horizon_sessions": "10",
            "sample_count": "2",
            "win_rate": "0.5000",
            "avg_side_return": "0.0250",
            "profit_factor": "1.5",
        },
    ]
    tier_rows = [
        {
            "tier": "S",
            "side": "LONG",
            "horizon_sessions": "5",
            "observation_count": "2",
            "win_rate": "0.5000",
            "avg_side_return_pct": "1.0",
        },
        {
            "tier": "A",
            "side": "SHORT",
            "horizon_sessions": "5",
            "observation_count": "6",
            "win_rate": "0.6667",
            "avg_side_return_pct": "3.0",
        },
        {
            "tier": "B",
            "side": "LONG",
            "horizon_sessions": "5",
            "observation_count": "99",
            "win_rate": "1.0000",
            "avg_side_return_pct": "99.0",
        },
        {
            "tier": "A",
            "side": "LONG",
            "horizon_sessions": "10",
            "observation_count": "2",
            "win_rate": "0.2500",
            "avg_side_return_pct": "1.0",
        },
    ]

    rows = build_human_focus_comparison_rows(human_rows, tier_rows)

    all_h5 = rows[0]
    long_h10 = rows[1]
    assert all_h5["side"] == "ALL"
    assert all_h5["horizon_sessions"] == "5"
    assert all_h5["avg_side_return_pct"] == "4.0000"
    assert all_h5["bot_sa_sample_count"] == "8"
    assert all_h5["bot_sa_avg_side_return_pct"] == "2.5000"
    assert all_h5["avg_side_return_delta_pct"] == "1.5000"
    assert long_h10["side"] == "LONG"
    assert long_h10["bot_sa_avg_side_return_pct"] == "1.0000"

