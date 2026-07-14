from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import master_avwap as m  # noqa: E402


def _setup(symbol: str, scan_date: str, family: str, zone: str, result_r: float) -> dict:
    status = "TARGET_HIT" if result_r > 0 else "STOPPED"
    return {
        "setup_id": f"{scan_date}:{symbol}",
        "symbol": symbol,
        "side": "LONG",
        "scan_date": scan_date,
        "anchor_date": scan_date,
        "setup_status": "CLOSED",
        "priority_score": 120.0,
        "priority_bucket": "favorite_setup",
        "setup_family": family,
        "favorite_zone": zone,
        "retest_followthrough": False,
        "compression_flag": False,
        "market_regime_label": "bull",
        "entry_attributes": {"trend.trend_20d": "UP", "unused.large.field": "x" * 5000},
        "daily_marks": [{"trade_date": scan_date, "feature_snapshot": {"blob": "x" * 5000}}],
        "scenarios": {
            "primary": {
                "stop_reference_label": "AVWAPE",
                "tradeable": True,
                "experimental": False,
                "status": status,
                "total_r": result_r,
                "days_held": 4,
            }
        },
    }


def _tracker() -> dict:
    setups = {}
    specs = [
        ("AAPL", "2026-07-12", "hot_family", "AVWAPE to UPPER_1", 1.5),
        ("MSFT", "2026-07-10", "hot_family", "AVWAPE to UPPER_1", 1.2),
        ("NVDA", "2026-07-08", "hot_family", "AVWAPE to UPPER_1", 1.0),
        ("META", "2026-07-06", "hot_family", "AVWAPE to UPPER_1", 0.8),
        ("CRM", "2026-07-11", "cold_family", "None", -0.8),
        ("ORCL", "2026-07-09", "cold_family", "None", -0.6),
        ("ADBE", "2026-07-07", "cold_family", "None", -0.4),
        ("NOW", "2026-07-05", "cold_family", "None", -0.2),
    ]
    for spec in specs:
        setup = _setup(*spec)
        setups[setup["setup_id"]] = setup
    return {"schema_version": 2, "updated_at": "2026-07-13T13:00:00", "setups": setups}


def test_compact_scoring_snapshot_preserves_family_and_setup_type_results():
    tracker = _tracker()
    snapshot = m.build_setup_tracker_scoring_payload(tracker)

    full_recent = m.build_recent_tracker_setup_family_rows(
        tracker["setups"], reference_date=date(2026, 7, 13), current_regime_label="bull"
    )
    compact_recent = m.build_recent_tracker_setup_family_rows(
        snapshot["setups"], reference_date=date(2026, 7, 13), current_regime_label="bull"
    )
    full_types = m.build_tracker_setup_type_rows(tracker["setups"])
    compact_types = m.build_tracker_setup_type_rows(snapshot["setups"])

    assert compact_recent == full_recent
    assert compact_types == full_types
    assert len(json.dumps(snapshot)) < len(json.dumps(tracker)) / 4


def test_valid_scoring_snapshot_load_does_not_touch_full_tracker(tmp_path):
    source = tmp_path / "tracker.json"
    snapshot_path = tmp_path / "scoring.json"
    source.write_text("{}", encoding="utf-8")
    snapshot = m.save_setup_tracker_scoring_payload(_tracker(), snapshot_path)
    source_time = snapshot_path.stat().st_mtime - 10
    os.utime(source, (source_time, source_time))

    with mock.patch.object(m, "load_setup_tracker_payload", side_effect=AssertionError("full tracker read")):
        loaded = m.load_setup_tracker_scoring_payload(snapshot_path, tracker_path=source)

    assert loaded == snapshot
    assert loaded["source_record_count"] == 8
