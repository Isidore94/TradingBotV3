"""The saved report must receive measured Points facts from its own scan."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_real_report_header_round_trips_bounded_stable_scan_facts(tmp_path, monkeypatch):
    import project_paths
    from ui.models.setup_table_model import SetupTableModel
    from ui.services import data_feed
    from ui.services.scan_service import ScanService

    report = tmp_path / "priority.txt"
    report.write_text(
        "Master AVWAP priority setups\nGenerated at 2026-09-18 12:55:44\n\n"
        "Ranked by Expected-R (blended)\n------------------------------\n"
        "NVDA LONG ExpR=+0.50R score=88 family=earnings gap bucket=favorite_setup\n",
        encoding="utf-8",
    )
    sidecar = tmp_path / "points-projection.json"
    monkeypatch.setattr(project_paths, "MASTER_AVWAP_PRIORITY_SETUPS_FILE", report)
    monkeypatch.setattr(project_paths, "SETUP_POINTS_SCAN_PROJECTION_FILE", sidecar)
    monkeypatch.setattr(data_feed, "MASTER_AVWAP_PRIORITY_SETUPS_FILE", report)
    monkeypatch.setattr(data_feed, "SETUP_POINTS_SCAN_PROJECTION_FILE", sidecar)
    monkeypatch.setattr(data_feed, "_POINTS_PROJECTION_CACHE", {"rows": (), "valid": False})
    ScanService._write_points_scan_projection({
        "run_id": "2026-09-18-125544", "run_date": "2026-09-18",
        "run_timestamp": "2026-09-18T12:55:44-07:00",
        "stable_priority_rows": [{
            "symbol": "NVDA", "side": "LONG", "bar_status": "COMPLETED", "view_mode": "STABLE",
            "setup_family": "earnings gap", "last_close": 100.0, "atr20": 2.0,
            "hv_level_blocking_count": 1, "hv_level_nearby_count": 0,
            "hv_level_nearest_distance_atr": 1.0, "cloud_level_nearby_count": 0,
            "ema21": 90.0, "sma_breakout_sma_level": 90.0,
            "priority_trendline_note": "", "daily_relative_strength_score": 1.0,
        }],
        "tracked_rows": [{"symbol": "NVDA", "side": "SHORT", "bar_status": "FORMING", "view_mode": "PREVIEW", "hv_level_blocking_count": 9}],
    })
    assert sidecar.exists()
    assert data_feed.warm_points_projection(path=report)
    rows = data_feed.load_setup_rows_from_priority_report(report)
    data_feed.enrich_report_rows_with_cached_scan(rows, data_feed.cached_points_projection(rows))
    assert len(rows) == 1
    assert rows[0].raw["scan_stamp"] == "2026-09-18-125544"
    assert rows[0].raw["hv_level_blocking_count"] == 1
    assert rows[0].raw["previous_close"] == 100.0
    assert SetupTableModel(rows).points_for(rows[0]).sr == 6.0
    report.write_text(report.read_text(encoding="utf-8").replace("score=88", "score=89"), encoding="utf-8")
    revised = data_feed.load_setup_rows_from_priority_report(report)
    assert data_feed.cached_points_projection(revised) == []


def test_equal_value_groups_are_kept_whole_at_a_half_boundary():
    import setup_points_evidence as evidence

    logged, outcomes = [], []
    for index in range(120):
        symbol = f"S{index:03d}"
        day = f"2026-09-{8 + index % 5:02d}"
        logged.append({"scan_date": day, "symbol": symbol, "side": "LONG", "total": 120 - index,
                       "setup": 0, "sr": 0, "rs": 0, "bounce": 15 if index < 40 else 0})
        outcomes.append({"scan_date": day, "symbol": symbol, "side": "LONG", "win": index < 40})
    grade = evidence.grade(logged, outcomes, horizon_sessions=5)
    upper, lower = grade.part_halves["bounce"]
    assert (upper.n, lower.n) == (40, 80)
    assert grade.part_lift["bounce"] == 1.0


def test_scan_and_observation_stamps_compare_as_instants():
    import setup_points_evidence as evidence

    row = {"points_version": "points_v2", "scan_date": "2026-09-18", "symbol": "NVDA", "side": "LONG",
           "observed_at": "2026-09-18T09:30:00-07:00", "scan_timestamp": "2026-09-18T16:00:00+00:00",
           "logged_at": "2026-09-18T16:31:00+00:00", "total": 5.0}
    chosen = evidence._declared_pre_close_observations([row])
    assert len(chosen) == 1


def test_claim_overlay_never_borrows_opposite_side_or_unidentified_analysis():
    from ui.services.claimed_setup_rows import merge_claims

    claim = {"symbol": "NVDA", "side": "SHORT", "claimed_setup_id": "alpha"}
    rows = merge_claims([], [claim], analysis_by_symbol={"NVDA": {"side": "LONG", "expected_r": 3.0}})
    assert "expected_r" not in rows[0].raw
    assert rows[0].raw["current_analysis"] == {}
    rows = merge_claims([], [claim], analysis_by_symbol={"NVDA": {"expected_r": 3.0}})
    assert "expected_r" not in rows[0].raw


def test_stable_projection_does_not_add_rows_to_the_live_run_result():
    from ui.services.data_feed import rows_from_run_result

    payload = {
        "tracked_rows": [{"symbol": "AAA", "side": "LONG", "priority_bucket": "favorite_setup"}],
        "stable_priority_rows": [{"symbol": "BBB", "side": "LONG", "priority_bucket": "favorite_setup"}],
    }
    assert [row.symbol for row in rows_from_run_result(payload)] == ["AAA"]
