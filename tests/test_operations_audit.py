from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _healthy_fixture(tmp_path: Path) -> tuple[Path, Path, datetime]:
    diagnostics = tmp_path / "diagnostics"
    registry = tmp_path / "candidate_registry.json"
    now = datetime.fromisoformat("2026-07-13T12:30:00-07:00")
    _write(
        diagnostics / "heartbeat.json",
        {"schema": "heartbeat_v1", "pid": 123, "ts": "2026-07-13T12:29:00-07:00", "next_job": "13:00"},
    )
    events = [
        {
            "schema": "job_ledger_v1",
            "event": "scheduled",
            "key": "2026-07-13|swing_scan|12:00|shared-v1",
            "market_date": "2026-07-13",
            "job_type": "swing_scan",
            "slot": "12:00",
            "config_hash": "shared-v1",
            "ts": "2026-07-13T12:00:00-07:00",
        },
        {"schema": "job_ledger_v1", "event": "started", "key": "2026-07-13|swing_scan|12:00|shared-v1", "run_id": "run-1", "ts": "2026-07-13T12:00:01-07:00"},
        {"schema": "job_ledger_v1", "event": "completed", "key": "2026-07-13|swing_scan|12:00|shared-v1", "run_id": "run-1", "ts": "2026-07-13T12:18:00-07:00"},
    ]
    (diagnostics / "job_ledger.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (diagnostics / "job_ledger.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8")
    _write(
        diagnostics / "run_manifests" / "run-1.json",
        {"schema": "run_manifest_v1", "run_id": "run-1", "job_type": "master_scan", "started_at": "2026-07-13T19:00:00+00:00", "ended_at": "2026-07-13T19:18:00+00:00", "status": "ok", "total_seconds": 1080},
    )
    _write(
        diagnostics / "spy_state_shadow_status.json",
        {"session_date": "2026-07-13", "evaluations": 4, "usable_evaluations": 4, "errors": 0, "last_evaluation_at": "2026-07-13T12:28:00-07:00"},
    )
    _write(
        diagnostics / "greatness_candidates.json",
        {"session_date": "2026-07-13", "updated_at": "2026-07-13T12:25:00-07:00", "coverage": {"session_date": "2026-07-13", "evaluations": 20, "bars_consumed": 50, "errors": 0, "last_evaluation_at": "2026-07-13T12:25:00-07:00"}, "candidates": [{"symbol": "AAPL"}]},
    )
    _write(
        registry,
        {"schema": "candidate_registry_v1", "generation": 4, "candidates": [{"symbol": "AAPL", "side": "LONG", "stage": "DEVELOPING", "memberships": {"open_scan": {}}}]},
    )
    timestamp = now.timestamp()
    os.utime(registry, (timestamp, timestamp))
    return diagnostics, registry, now


def test_healthy_runtime_audit_composes_all_sol3_surfaces(tmp_path):
    from operations_audit import build_operations_audit

    diagnostics, registry, now = _healthy_fixture(tmp_path)
    payload = build_operations_audit(now=now, diagnostics_dir=diagnostics, candidate_registry_path=registry)

    assert payload["schema"] == "operations_audit_v1"
    assert payload["status"] == "healthy"
    assert payload["summary"] == {"healthy": 6, "degraded": 0, "unhealthy": 0, "total": 6}
    assert {item["id"] for item in payload["checks"]} == {
        "heartbeat", "job_ledger", "run_manifest", "spy_shadow", "greatness_shadow", "candidate_registry"
    }
    assert payload["latest_manifest"]["run_id"] == "run-1"
    assert payload["excluded"] == ["large setup-tracker payload"]


def test_stale_heartbeat_and_shadow_make_audit_unhealthy(tmp_path):
    from operations_audit import build_operations_audit

    diagnostics, registry, now = _healthy_fixture(tmp_path)
    heartbeat = json.loads((diagnostics / "heartbeat.json").read_text(encoding="utf-8"))
    heartbeat["ts"] = "2026-07-13T11:00:00-07:00"
    _write(diagnostics / "heartbeat.json", heartbeat)
    greatness = json.loads((diagnostics / "greatness_candidates.json").read_text(encoding="utf-8"))
    greatness["coverage"]["last_evaluation_at"] = "2026-07-13T11:00:00-07:00"
    _write(diagnostics / "greatness_candidates.json", greatness)

    payload = build_operations_audit(now=now, diagnostics_dir=diagnostics, candidate_registry_path=registry)
    statuses = {item["id"]: item["status"] for item in payload["checks"]}

    assert payload["status"] == "unhealthy"
    assert statuses["heartbeat"] == "unhealthy"
    assert statuses["greatness_shadow"] == "unhealthy"


def test_audit_write_is_atomic_and_large_tracker_is_never_read(tmp_path):
    from operations_audit import refresh_operations_audit

    diagnostics, registry, now = _healthy_fixture(tmp_path)
    payload = refresh_operations_audit(now=now, diagnostics_dir=diagnostics, candidate_registry_path=registry)
    written = json.loads((diagnostics / "operations_audit.json").read_text(encoding="utf-8"))

    assert written == payload
    assert not list(diagnostics.glob("*.tmp"))
