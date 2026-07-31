from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


NY = ZoneInfo("America/New_York")
DAY = "2026-07-30"


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _base(event_type, event_id):
    return {
        "code_version": "regime_infrastructure_phase1_v1",
        "event_type": event_type,
        "event_id": event_id,
        "session_date": DAY,
        "as_of": f"{DAY}T10:30:00-04:00",
        "written_at": f"{DAY}T10:30:01-04:00",
    }


def test_healthy_after_close_session_passes_acceptance_audit(tmp_path):
    from regime_collection_audit import audit_regime_collection

    technical = tmp_path / "technical.jsonl"
    breadth = tmp_path / "vold.jsonl"
    rows = [
        {
            **_base("level_resolved", "r1"),
            "followup_tracking_version": "regime_infrastructure_phase1_v1",
        },
        {
            **_base("post_resolution_tracking_started", "r1|followup"),
            "source_resolution_id": "r1",
        },
    ]
    for horizon in (30, 60, 90):
        rows.append(
            {
                **_base("post_resolution_followup", f"r1|followup|{horizon}"),
                "source_resolution_id": "r1",
                "horizon_minutes": horizon,
                "truncated": horizon > 30,
                "data_gap": False,
            }
        )
    rows.extend(
        [
            {
                **_base("frozen_intraday_snapshot", "s1030"),
                "snapshot_key": f"{DAY}|10:30",
                "target_market_time": "10:30",
            },
            {
                **_base("frozen_intraday_snapshot", "s1200"),
                "snapshot_key": f"{DAY}|12:00",
                "target_market_time": "12:00",
            },
            {
                **_base("opening_range_baseline", "opening"),
                "snapshot_key": f"{DAY}|opening_range",
                "data_gap": False,
            },
        ]
    )
    _write(technical, rows)
    contract = {
        "symbol": "TICK-NYSE",
        "proxy_kind": "nyse_tick_proxy",
        "is_exact_vold": False,
    }
    breadth_rows = [
        {
            **_base("contract_verified", "contract"),
            "contract": contract,
        }
    ]
    open_time = datetime(2026, 7, 30, 9, 30, tzinfo=NY)
    for index in range(78):
        end = open_time + timedelta(minutes=5 * (index + 1))
        breadth_rows.append(
            {
                **_base("breadth_bar", f"b{index}"),
                "contract": contract,
                "bar_end": end.isoformat(timespec="seconds"),
                "as_of": end.isoformat(timespec="seconds"),
            }
        )
    _write(breadth, breadth_rows)

    report = audit_regime_collection(
        session_date=DAY,
        technical_events_path=technical,
        breadth_events_path=breadth,
        now=datetime(2026, 7, 31, 10, 0, tzinfo=NY),
    )

    assert report["status"] == "HEALTHY"
    assert report["promotion_status"] == "EXPLORATORY / NON-PROMOTABLE"
    assert report["technical_followups"]["horizon_counts"] == {
        "30": 1,
        "60": 1,
        "90": 1,
    }
    assert report["breadth_recorder"]["semantic_status"] == "PROXY:nyse_tick_proxy"


def test_missing_snapshot_duplicate_bar_and_incomplete_chain_are_blockers(tmp_path):
    from regime_collection_audit import audit_regime_collection

    technical = tmp_path / "technical.jsonl"
    breadth = tmp_path / "vold.jsonl"
    _write(
        technical,
        [
            {
                **_base("level_resolved", "r1"),
                "followup_tracking_version": "regime_infrastructure_phase1_v1",
            },
            {
                **_base("post_resolution_tracking_started", "r1|followup"),
                "source_resolution_id": "r1",
            },
            {
                **_base("post_resolution_followup", "f30"),
                "source_resolution_id": "r1",
                "horizon_minutes": 30,
            },
        ],
    )
    duplicate = {
        **_base("breadth_bar", "b1"),
        "bar_end": f"{DAY}T09:35:00-04:00",
    }
    _write(breadth, [duplicate, duplicate])

    report = audit_regime_collection(
        session_date=DAY,
        technical_events_path=technical,
        breadth_events_path=breadth,
        now=datetime(2026, 7, 30, 16, 5, tzinfo=NY),
    )

    assert report["status"] == "UNHEALTHY"
    assert report["frozen_snapshots"]["missing_labels"] == ["10:30", "12:00"]
    assert report["technical_followups"]["incomplete_chains"]["r1"] == [60, 90]
    assert report["breadth_recorder"]["duplicate_bar_ends"] == [
        f"{DAY}T09:35:00-04:00"
    ]
    assert "breadth ledger is partial without enough explicit data-gap coverage" in report[
        "blockers"
    ]
    assert "breadth bars have no contract-verification event" in report["blockers"]
