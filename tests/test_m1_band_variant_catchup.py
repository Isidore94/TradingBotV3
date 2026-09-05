"""Packet M1.1 - the tracker catch-up path hands the challenger block on.

Phase 0.10 built the AVWAP band challenger as a shadow beside the champion and
Phase 0.19 found it had measured nothing in the ten days since: every row of
`master_avwap_band_variant_stats.csv` read `n_variant = 0`, and every tracker
record's `current_anchor_variant` read the placeholder
`"no band-variant block on the scan entry"` while `master_avwap_ai_state.json`
carried a full block for all 423 symbols of the same scan.

The reason is two builders for one symbol entry. `runner.py`'s live scan sets
`current_anchor_variant` / `previous_anchor_variant`; the tracker staleness
catch-up (`backfill_setup_tracker_from_recent_sessions` ->
`_evaluate_priority_snapshot_for_date`) builds its own ~100-key entry and never
did. The catch-up is what writes the persisted tracker on a normal day, so the
placeholder was the only value a record ever saw.

These tests fail on `e7b12ebe` (the block is absent from the catch-up entry and
the record reads the placeholder) and pass once ONE function serves both paths.
Shadow only: nothing here reaches a detector, score, rank, tier, alert, zone
arm, Focus list or the review queue, and `calc_anchored_vwap_bands` is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from indicators.avwap_band_variants import FEATURE_VERSION  # noqa: E402
from master_avwap_lib import legacy, runner  # noqa: E402

#: The placeholder a record gets when nobody handed it a block. After this
#: packet it must be unreachable from the catch-up path.
PLACEHOLDER = "no band-variant block on the scan entry"

#: What `build_anchor_band_variant_meta` says when the 20-close window is short.
SHORT_WINDOW_REASON = "fewer than the lookback's closes before this bar"


def _daily_frame(sessions: int) -> pd.DataFrame:
    """A deterministic completed-bar daily frame in the scanner's own shape."""
    dates = pd.bdate_range("2026-05-01", periods=sessions)
    closes = [100.0 + index * 0.5 + (2.0 if index % 3 == 0 else -1.0) for index in range(sessions)]
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(dates),
            "open": [close - 0.4 for close in closes],
            "high": [close + 1.2 for close in closes],
            "low": [close - 1.3 for close in closes],
            "close": closes,
            "volume": [1_000_000.0 + 1_000 * index for index in range(sessions)],
        }
    )


def _snapshot(frame: pd.DataFrame, anchor_position: int) -> dict:
    anchor_iso = frame["datetime"].iloc[anchor_position].date().isoformat()
    previous_iso = frame["datetime"].iloc[max(anchor_position - 5, 0)].date().isoformat()
    snapshot = legacy._evaluate_priority_snapshot_for_date(
        symbol="TEST",
        side="LONG",
        df_full=frame,
        evaluation_date=frame["datetime"].iloc[-1].date(),
        current_anchor_iso=anchor_iso,
        previous_anchor_iso=previous_iso,
        recent_earnings_dates=[],
        latest_release_info=None,
        history_state={},
    )
    assert snapshot is not None
    return snapshot


def _record(snapshot: dict, frame: pd.DataFrame) -> dict:
    return legacy.build_tracker_setup_record(
        dict(snapshot["priority_row"]),
        snapshot["symbol_entry"],
        snapshot["feature_row"],
        "2026-09-05T00:00:00",
        None,
        scan_date=frame["datetime"].iloc[-1].date().isoformat(),
    )


@pytest.fixture(scope="module")
def long_frame() -> pd.DataFrame:
    """60 sessions: the 20-close sigma window is fully covered."""
    return _daily_frame(60)


@pytest.fixture(scope="module")
def short_frame() -> pd.DataFrame:
    """12 sessions: fewer closes than the challenger's lookback."""
    return _daily_frame(12)


def test_one_function_serves_both_call_paths():
    """The live scan and the catch-up must compute the SAME challenger block.

    Two copies of a formula is two formulas: the whole point of the shadow is
    that the number in the tracker is the number the scan computed.
    """
    assert runner.build_anchor_band_variant_meta is legacy.build_anchor_band_variant_meta


def test_the_catch_up_snapshot_carries_the_challenger_block(long_frame):
    entry = _snapshot(long_frame, 30)["symbol_entry"]

    for key in ("current_anchor_variant", "previous_anchor_variant"):
        block = entry.get(key)
        assert isinstance(block, dict), f"{key} missing from the catch-up symbol entry"
        assert block["formula_version"] == FEATURE_VERSION
        assert isinstance(block["stdev"], float)
        assert block["stdev"] > 0.0
        assert isinstance(block["vwap"], float)
        assert set(block["bands"]) == {
            "UPPER_1",
            "UPPER_2",
            "UPPER_3",
            "LOWER_1",
            "LOWER_2",
            "LOWER_3",
        }
        assert block["reason"] == ""


def test_the_challenger_block_is_anchored_where_the_champion_is(long_frame):
    """Same frame, same anchor date - only the formula differs."""
    entry = _snapshot(long_frame, 30)["symbol_entry"]

    assert entry["current_anchor_variant"]["date"] == entry["current_anchor"]["date"]
    assert entry["previous_anchor_variant"]["date"] == entry["previous_anchor"]["date"]
    # The challenger is a different width, never the champion's re-labelled.
    assert entry["current_anchor_variant"]["stdev"] != entry["current_anchor"]["stdev"]


def test_a_record_built_through_the_catch_up_path_measures_the_challenger(long_frame):
    record = _record(_snapshot(long_frame, 30), long_frame)

    block = record["current_anchor_variant"]
    assert block["reason"] != PLACEHOLDER
    assert isinstance(block["stdev"], float)
    assert block["formula_version"] == FEATURE_VERSION


def test_the_record_gains_the_challenger_stop_on_the_four_baseline_templates(long_frame):
    snapshot = _snapshot(long_frame, 30)
    record = _record(snapshot, long_frame)

    entry_for_stops = dict(snapshot["symbol_entry"])
    entry_for_stops["entry_feature_snapshot"] = record["entry_feature_snapshot"]
    candidates = legacy._find_tracker_stop_candidates(
        dict(snapshot["priority_row"]), entry_for_stops
    )
    variant_candidates = [
        candidate for candidate in candidates if legacy._is_band_variant_stop(candidate)
    ]
    assert len(variant_candidates) == 1
    assert variant_candidates[0]["label"] == "VARIANT_LOWER_1"
    assert variant_candidates[0]["level"] is not None

    scenarios = [
        scenario
        for scenario in (record["scenarios"] or {}).values()
        if legacy._is_band_variant_scenario(scenario)
    ]
    baseline_templates = {
        str(template["id"])
        for template in legacy.SETUP_EXIT_TEMPLATES
        if not bool(template.get("experimental"))
    }
    assert len(scenarios) == len(baseline_templates)
    assert {str(scenario["exit_template_id"]) for scenario in scenarios} == baseline_templates


def test_a_short_frame_says_it_could_not_measure_rather_than_the_placeholder(short_frame):
    """Missing data is uncertainty with a stated reason, never a zero band."""
    snapshot = _snapshot(short_frame, 2)
    entry = snapshot["symbol_entry"]

    block = entry["current_anchor_variant"]
    assert block["stdev"] is None
    assert block["bands"] == {}
    assert block["reason"] == SHORT_WINDOW_REASON

    record = _record(snapshot, short_frame)
    assert record["current_anchor_variant"]["reason"] == SHORT_WINDOW_REASON
    assert not [
        scenario
        for scenario in (record["scenarios"] or {}).values()
        if legacy._is_band_variant_scenario(scenario)
    ]


@pytest.mark.parametrize("sessions,anchor_position", [(60, 30), (12, 2)])
def test_the_placeholder_is_unreachable_from_the_catch_up_path(sessions, anchor_position):
    """No live-shaped catch-up entry may produce "no block on the scan entry".

    That string is for a caller that predates the shadow - a replay, an old
    payload. Once the catch-up sets the block it can never be the reason a
    record gives, and a table full of it is the defect this packet fixes.
    """
    frame = _daily_frame(sessions)
    snapshot = _snapshot(frame, anchor_position)
    record = _record(snapshot, frame)

    for key in ("current_anchor_variant", "previous_anchor_variant"):
        assert snapshot["symbol_entry"][key]["reason"] != PLACEHOLDER
        assert record[key]["reason"] != PLACEHOLDER
