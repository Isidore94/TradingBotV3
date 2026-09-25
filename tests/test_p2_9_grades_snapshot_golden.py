"""P2-9 golden: the looking-back views add columns and change no computed value.

`setup_grades.build_payload` and `working_lately.build_snapshot` are pinned
byte-for-byte on one fixed fixture. The expected block was written from
`origin/main` (a793f661) BEFORE the P2-9 read-only views were added, so any
change to a grade, a cell or a verdict fails here. The fixture is
contract-bearing: its inputs are hashed and carried in the file.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from conftest import load_fixture_contract  # noqa: E402

FIXTURE = "p2_9_grades_snapshot_golden_v1"
AS_OF = date(2026, 9, 18)
SESSIONS = (
    "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14",
    "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
)


def _recent_rows() -> list[dict]:
    """The generator the fixture's `recent_rows` were frozen from."""
    rows = []
    for index, (family, wins, losses, flats, rep) in enumerate(
        (
            ("avwape_to_1stdev", 90, 8, 2, "0.38"),
            ("avwap_breakout", 20, 15, 1, "0.12"),
            ("general", 5, 9, 0, "-0.3"),
            ("hv_level_break", 40, 10, 0, "0.5"),
            # Either side of the n=30 floor, so a floor change fails here.
            ("floor_exactly_30", 25, 5, 0, "0.2"),
            ("floor_just_under_30", 24, 5, 0, "0.2"),
        )
    ):
        rows.append(
            {
                "namespace": "study" if family == "hv_level_break" else "live",
                "side": "LONG" if index % 2 == 0 else "SHORT",
                "priority_bucket": "favorite_setup",
                "setup_family": family,
                "closed_setups": str(wins + losses + flats),
                "tracked_setups": str(wins + losses + flats + 3),
                "avg_closed_r": "0.21",
                "representative_closed_r": rep,
                "n_wins": str(wins),
                "n_losses": str(losses),
                "n_flats": str(flats),
                "n_unmeasured": "1",
                "n_pending": "2",
                "n_symbols": str(10 + index),
                "n_entry_sessions": str(12 + index),
                "outcome_kind": "representative_closed_r",
                "outcome_version": "recent_types_v2",
                "knowledge_basis": "entry_scan_row_close_to_representative_exit",
                "horizon_basis": "30d lookback, representative exit",
                "latest_measured_session": SESSIONS[-1 - index],
            }
        )
    return rows


def _outcome_rows() -> list[dict]:
    """The generator the fixture's `outcome_rows` were frozen from."""
    rows = []
    for day_index, session in enumerate(SESSIONS):
        stamp = session.replace("-", "")
        for n in range(6):
            bounce = ("vwap", "ema_15", "eod_vwap-vwap")[n % 3]
            direction = "long" if n % 2 == 0 else "short"
            event_id = f"S{n}_{direction}_{stamp}_10_{n:02d}_00_{bounce}"
            won = (n + day_index) % 3 != 0
            for bars in (1, 2, 3):
                rows.append(
                    {
                        "event_id": event_id,
                        "event_type": "final" if bars == 3 else "update",
                        "trade_date": session,
                        "symbol": f"S{n}",
                        "direction": direction,
                        "entry_time": f"{session}T10:{n:02d}:00",
                        "bars_elapsed": str(bars),
                        "minutes_elapsed": str(bars * 5 + 30),
                        "target_1r_hit": "True" if (won and bars >= 2) else "False",
                        "stop_hit": "True" if (not won and bars >= 2) else "False",
                        "mfe_r": "2.0" if won else "0.3",
                        "context_json": json.dumps({"market_environment": "bullish_strong"}),
                    }
                )
    return rows


def _favorable_rows() -> list[dict]:
    """The generator the fixture's `favorable_rows` were frozen from."""
    rows = []
    for day_index, session in enumerate(SESSIONS[:6]):
        for n in range(8):
            move = 1.5 if (n + day_index) % 3 else -0.8
            rows.append(
                {
                    "observation_id": f"F{n}:{session}:5",
                    "scan_date": session,
                    "future_scan_date": SESSIONS[min(day_index + 3, len(SESSIONS) - 1)],
                    "horizon_sessions": "5",
                    "tier": "S",
                    "symbol": f"F{n}",
                    "side": "LONG" if n % 2 else "SHORT",
                    "priority_bucket": "favorite_setup",
                    "setup_family": ("avwap_breakout", "general")[n % 2],
                    "side_return_pct": str(move),
                    "win": "True" if move > 0 else "False",
                    "stale_horizon": "",
                }
            )
    return rows


def build_outputs(inputs: dict, setups_path: Path) -> dict:
    import held_run_score
    import setup_grades
    import swing_evidence
    import working_lately

    as_of = date.fromisoformat(inputs["as_of"])
    sessions = inputs["sessions"]
    recent = inputs["recent_rows"]
    outcome = inputs["outcome_rows"]
    grades = setup_grades.build_payload(
        recent_rows=recent, outcome_rows=outcome, as_of=as_of.isoformat()
    )
    favorable = swing_evidence.read_eligible_rows(
        inputs["favorable_rows"],
        swing_evidence.POLICY_SCANROW_V1,
        window=(sessions[0], sessions[-1]),
    )
    episodes = held_run_score.load_episodes(
        rows=outcome, as_of=as_of.isoformat(), setups_path=setups_path
    )
    summaries = held_run_score.dimension_summaries(episodes, as_of=as_of.isoformat(), min_n=3)
    snapshot = working_lately.build_snapshot(
        recent_rows=recent,
        favorable_read=favorable,
        held_run_summaries=summaries,
        last_completed_session=as_of,
        previous_verdicts={},
    )
    payload = snapshot.to_payload()
    payload.pop("built_at", None)
    return {"setup_grades": grades, "working_lately": payload}


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, indent=1, default=str)


def test_the_fixture_inputs_are_the_generators_output():
    contract = load_fixture_contract(FIXTURE)
    inputs = contract["inputs"]
    assert inputs["recent_rows"] == _recent_rows()
    assert inputs["outcome_rows"] == _outcome_rows()
    assert inputs["favorable_rows"] == _favorable_rows()


def test_grades_and_snapshot_are_byte_identical_to_main(tmp_path):
    contract = load_fixture_contract(FIXTURE)
    # `load_episodes` joins the D1 snapshot; an absent one keeps it off live stores.
    actual = build_outputs(contract["inputs"], tmp_path / "absent_scoring_snapshot.json")
    assert _canonical(actual) == _canonical(contract["expected"])
