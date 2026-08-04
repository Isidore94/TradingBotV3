from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from conftest import load_fixture_contract


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap  # noqa: E402


def _project_outputs(contract) -> dict:
    tier_rows = master_avwap.build_bot_tier_pick_rows(
        pd.DataFrame(contract["tier_history"]),
        leaderboard_rows=[],
    )
    guardrail_rows = [dict(row) for row in contract["guardrail_rows"]]
    ai_state = {"symbols": {row["symbol"]: {} for row in guardrail_rows}}
    features = {
        row["symbol"]: {"priority_score": row["score"]}
        for row in guardrail_rows
    }
    master_avwap.apply_tracker_scoring_guardrails(
        guardrail_rows,
        ai_state,
        features,
    )
    return {
        "tiers": [
            [row["symbol"], row["tier"], row["priority_score"]]
            for row in tier_rows
        ],
        "guardrails": [
            [row["symbol"], row["score"], bool(row.get("watch_only"))]
            for row in guardrail_rows
        ],
    }


def test_setup_scoring_flagging_promoted_snapshot():
    contract = load_fixture_contract("setup_scoring_flagging_v1")
    contract.assert_matches(
        _project_outputs(contract),
        contract["promoted_expected"],
        "setup scoring/flagging promoted behavior",
    )
