from __future__ import annotations

import json
import os
import hashlib
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import writer_role  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict], *, terminated: bool = True) -> None:
    """Write a JSONL artifact; ``terminated=False`` leaves the crash signature."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row) for row in rows)
    path.write_text(text + ("\n" if terminated else ""), encoding="utf-8")


def _spy_row(ts: str, state: str = "BULL_IMPULSE", **overrides) -> dict:
    row = {
        "schema": "spy_state_shadow_v4",
        "ts": ts,
        "evaluated_at": ts,
        "bar_ts": ts,
        "session_date": ts[:10],
        "timezone": "Pacific Daylight Time",
        "machine": "test-machine",
        "engine_version": "spy_state_v1",
        "config_hash": "spy-config-1",
        "state": state,
        "usable": True,
        "incomplete_bar": False,
        "complete_bar_ts": ts,
        "stale": False,
    }
    row.update(overrides)
    return row


def _greatness_row(ts: str, event: str = "LEVEL_TOUCHED", **overrides) -> dict:
    row = {
        "schema": "greatness_shadow_v4",
        "engine_version": "greatness_v1",
        "config_hash": "greatness-config-1",
        "machine": "test-machine",
        "timezone": "Pacific Daylight Time",
        "session_date": ts[:10],
        "candidate_id": "AAPL|LONG|family|2026-07-13|greatness_v1|abc",
        "symbol": "AAPL",
        "side": "LONG",
        "evaluated_at": ts,
        "ts": ts,
        "bar_ts": ts,
        "bar": {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0, "complete": True},
        "event": event,
        "step": "AVWAPE",
        "stage": "TESTING_LEVEL",
    }
    row.update(overrides)
    return row


def _spy_log(diagnostics: Path) -> Path:
    return diagnostics / "spy_state_shadow.jsonl"


def _greatness_log(diagnostics: Path) -> Path:
    return diagnostics / "greatness_shadow.jsonl"


def _rows_for(rows: list[dict], session: str, kind: str) -> int:
    """Count rows the way the validator does, using the validator's own profile."""
    from diagnostics.shadow_log_audit import SPY_PROFILE

    return sum(
        1
        for row in rows
        if row.get("session_date") == session
        and SPY_PROFILE.kind(str(row.get("schema") or "")) == kind
    )


def _set_spy_log(diagnostics: Path, rows: list[dict] | None, **kwargs) -> None:
    """Replace the SPY log and keep the sidecar's row claims consistent.

    Tests that target one defect must not trip the *reconciliation* finding as a
    side effect: a sidecar left claiming the rows of a log that was just
    rewritten is a different failure, and it has its own test.
    """
    path = _spy_log(diagnostics)
    if rows is None:
        path.unlink(missing_ok=True)
        rows = []
    else:
        _write_jsonl(path, rows, **kwargs)
    status_path = diagnostics / "spy_state_shadow_status.json"
    sidecar = json.loads(status_path.read_text(encoding="utf-8"))
    session = str(sidecar.get("session_date") or "")
    sidecar["rows_written"] = _rows_for(rows, session, "primary")
    sidecar["episode_rows_written"] = _rows_for(rows, session, "episode")
    _write(status_path, sidecar)


def _set_greatness_log(diagnostics: Path, rows: list[dict], **kwargs) -> None:
    _write_jsonl(_greatness_log(diagnostics), rows, **kwargs)
    store_path = diagnostics / "greatness_candidates.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    session = str(store["coverage"].get("session_date") or "")
    store["coverage"]["events_emitted"] = sum(
        1 for row in rows if row.get("session_date") == session
    )
    _write(store_path, store)


def _touch(path: Path, moment: datetime) -> None:
    stamp = moment.timestamp()
    os.utime(path, (stamp, stamp))


#: Owned process/thread counts are per-process, so the audit reads them from
#: ``ui.services.scan_service`` only when it is running *inside* the owning
#: process. Tests inject the snapshot explicitly: whether that Qt module happens
#: to be in ``sys.modules`` because another test imported it must never decide
#: what these assertions see.
_IDLE_PROCESS_SNAPSHOT = {
    "process_pid": 4242,
    "owned_child_count": 0,
    "registered_child_count": 0,
    "exited_children_pending_cleanup": 0,
    "lingering_child_pids": [],
    "children": [],
    "active_scan_label": "",
    "scan_owner_claimed": False,
    "scan_worker_threads": 0,
    "python_thread_count": 3,
    "non_daemon_thread_count": 1,
    "thread_names": ["MainThread"],
}


def _measured_fixture(tmp_path: Path) -> tuple[Path, Path, datetime]:
    """Every dimension this repo actually implements, measured and in bounds.

    Deliberately NOT called "healthy": ``provider_counters`` has no capture
    point anywhere in the codebase, so it is emitted as UNKNOWN and the best
    this fixture can honestly reach is UNKNOWN.
    """
    diagnostics = tmp_path / "diagnostics"
    registry = tmp_path / "candidate_registry.json"
    now = datetime.fromisoformat("2026-07-13T12:30:00-07:00")
    # R10.A: yesterday's evidence snapshot, so the backup dimension is MEASURED
    # like every other one here rather than adding a second unknown. The check
    # reads only this manifest.
    _write(
        tmp_path / "machine_cache" / "evidence_snapshots" / "2026-07-13" / "manifest.json",
        {
            "schema": "evidence_snapshot_manifest_v1",
            "snapshot_date": "2026-07-13",
            "finished_at": "2026-07-13T02:14:00-07:00",
            "files": 41,
            "source_bytes": 3_500_000_000,
            "stored_bytes": 900_000_000,
            "skipped": 0,
            "skipped_by_reason": {},
            "entries": [],
        },
    )
    _write(
        diagnostics / "writer_health.json",
        {
            "schema": "writer_health_v1",
            "written_at": "2026-07-13T12:29:30-07:00",
            "designated_writer": "test-machine",
            "machine": "test-machine",
            "role": "designated_writer",
            "read_only": False,
            "read_only_reason": "",
            "pid": 123,
            "instance_id": "instance-1",
            "lease_path": str(diagnostics / "away.lease"),
            "lease_holder": "test-machine",
            "lease_instance_id": "instance-1",
            "lease_acquired_at": "2026-07-13T12:00:00-07:00",
            "lease_expires_at": "2026-07-13T13:00:00-07:00",
            "last_renewal_at": "2026-07-13T12:29:00-07:00",
            "fencing_generation": 7,
            "emergency_override": {"active": False, "expires_at": "", "reason": ""},
            "last_verified_publication": {
                "at": "2026-07-13T12:29:00-07:00",
                "holder": "test-machine",
                "generation": 7,
            },
            "status": "published",
            "healthy": True,
        },
    )
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
        {
            "schema": "spy_state_shadow_status_v3",
            "session_date": "2026-07-13",
            "evaluations": 4,
            "usable_evaluations": 4,
            "errors": 0,
            "rows_written": 2,
            "episode_rows_written": 1,
            "last_evaluation_at": "2026-07-13T12:28:00-07:00",
        },
    )
    _write(
        diagnostics / "greatness_candidates.json",
        {"session_date": "2026-07-13", "updated_at": "2026-07-13T12:25:00-07:00", "coverage": {"session_date": "2026-07-13", "evaluations": 20, "bars_consumed": 50, "errors": 0, "events_emitted": 3, "last_evaluation_at": "2026-07-13T12:25:00-07:00"}, "candidates": [{"symbol": "AAPL"}]},
    )
    # The RAW shadow logs. The fixture used to write only the sidecars, which is
    # precisely the blind spot this packet closes: with the logs absent the
    # audit had nothing to check the writers' self-reports against, and both
    # engines still rendered green.
    _write_jsonl(
        _spy_log(diagnostics),
        [
            _spy_row("2026-07-13T12:10:00-07:00", "BULL_IMPULSE"),
            _spy_row("2026-07-13T12:20:00-07:00", "COUNTERMOVE_ARMED"),
            # Episode rows carry their own schema in the SAME file.
            {
                "schema": "spy_episode_shadow_v1",
                "ts": "2026-07-13T12:20:00-07:00",
                "evaluated_at": "2026-07-13T12:20:00-07:00",
                "session_date": "2026-07-13",
                "machine": "test-machine",
                "engine_version": "spy_state_v1",
                "config_hash": "spy-config-1",
                "episode_id": "ep-1",
                "outcome": "OPEN",
                "derived_from_completed_bars": True,
            },
        ],
    )
    _write_jsonl(
        _greatness_log(diagnostics),
        [
            _greatness_row("2026-07-13T12:15:00-07:00", "LEVEL_TOUCHED"),
            _greatness_row("2026-07-13T12:20:00-07:00", "CLOSED_THROUGH"),
            _greatness_row("2026-07-13T12:25:00-07:00", "STEP_CLEARED"),
        ],
    )
    _write(
        registry,
        {"schema": "candidate_registry_v1", "generation": 4, "candidates": [{"symbol": "AAPL", "side": "LONG", "stage": "DEVELOPING", "memberships": {"open_scan": {}}}]},
    )
    timestamp = now.timestamp()
    os.utime(registry, (timestamp, timestamp))
    report = diagnostics / "autopilot_today.txt"
    report.write_text("verified away report", encoding="utf-8")
    _write(
        diagnostics / "autopilot_today.txt.meta.json",
        {
            "schema": "away_report_publish_v1",
            "verified_at": "2026-07-13T12:29:00-07:00",
            "sha256": hashlib.sha256(b"verified away report").hexdigest(),
            "holder": "test-machine",
        },
    )
    _write(diagnostics / "autopilot_state.json", {"enabled": True, "profile": "AWAY"})
    _write(
        diagnostics / "industry_board_snapshot.json",
        {
            "schema": "industry_board_snapshot_v1",
            "status": "ok",
            "last_attempt_at": "2026-07-13T12:20:00-07:00",
            "last_success_at": "2026-07-13T12:20:00-07:00",
            "snapshot_id": "snapshot-1",
            "sector_count": 11,
            "industry_count": 55,
            "symbol_count": 800,
            "error": "",
        },
    )
    # Universe + market-data freshness: a universe built yesterday and a daily
    # bar store written today.
    for name, symbols in (
        ("universe_all.txt", ("AAPL", "MSFT", "NVDA")),
        ("universe_longs.txt", ("AAPL", "NVDA")),
        ("universe_shorts.txt", ("MSFT",)),
    ):
        path = diagnostics / name
        path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
        _touch(path, datetime.fromisoformat("2026-07-12T16:00:00-07:00"))
    probe = diagnostics / "daily_bars" / "SPY.parquet"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_bytes(b"daily-bars")
    _touch(probe, now)
    return diagnostics, registry, now


def _build(diagnostics: Path, registry: Path, now: datetime, **kwargs) -> dict:
    from operations_audit import build_operations_audit

    kwargs.setdefault("process_snapshot", dict(_IDLE_PROCESS_SNAPSHOT))
    return build_operations_audit(
        now=now, diagnostics_dir=diagnostics, candidate_registry_path=registry, **kwargs
    )


def test_measured_runtime_audit_composes_all_sol3_surfaces(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    payload = _build(diagnostics, registry, now)

    assert payload["schema"] == "operations_audit_v2"
    # REPLACES the old assertion `payload["status"] == "healthy"`: the page can
    # only be green when every required dimension is measured, and provider
    # counters have no capture point anywhere in the codebase (plan.md sec 6.3).
    assert payload["status"] == "unknown"
    assert payload["summary"]["unhealthy"] == 0
    operational = {
        item["id"]
        for item in payload["checks"]
        if not item["id"].startswith(("review_", "setup_scoring"))
    }
    assert operational == {
        "runtime_profile", "writer_lease", "heartbeat", "job_ledger", "run_manifest",
        "away_report", "industry_board", "spy_shadow", "greatness_shadow",
        "candidate_registry", "owned_process_counts", "provider_counters",
        "universe_and_market_data_freshness", "disk_storage_warnings",
        # R6(a): the overnight AI batch layer. It reports HEALTHY here because
        # no ai_store_dir is configured in the test environment - a measured
        # "deliberately off", not an unmeasured unknown, which is why the
        # unknown_operational assertion below still holds at exactly one entry.
        "ai_jobs",
        # R10.A: the nightly dated backup of the hot evidence the cold push
        # excludes on purpose (data/runtime, the home-root files, diagnostics).
        "evidence_snapshot",
    }
    # Every dimension H2 implemented reports a measured status, and the ONLY
    # remaining UNKNOWN is the one nothing captures.
    implemented = {
        item["id"]: item["status"]
        for item in payload["checks"]
        if item["id"] in operational and item["id"] != "provider_counters"
    }
    assert set(implemented.values()) == {"healthy"}
    unknown_operational = {
        item["id"] for item in payload["checks"] if item["id"] in operational and item["status"] == "unknown"
    }
    assert unknown_operational == {"provider_counters"}
    # Cold-start learning artifacts are degraded, not broken: runtime health
    # must stay honest about the scheduler, not be dragged down by a ledger
    # that no reviewed alert has created yet.
    assert payload["capture_readiness"]["status"] == "degraded"
    assert payload["capture_readiness"]["evidence_label"] == "Exploratory / Non-Promotable"
    assert payload["latest_manifest"]["run_id"] == "run-1"
    assert payload["excluded"] == ["large setup-tracker payload"]


def test_status_precedence_is_deterministic():
    from operations_audit import (
        STATUS_DEGRADED,
        STATUS_HEALTHY,
        STATUS_UNHEALTHY,
        STATUS_UNKNOWN,
        STATUS_VALUES,
        worst_status,
    )

    assert STATUS_VALUES == (STATUS_HEALTHY, STATUS_UNKNOWN, STATUS_DEGRADED, STATUS_UNHEALTHY)
    assert worst_status([STATUS_HEALTHY, STATUS_UNKNOWN]) == STATUS_UNKNOWN
    assert worst_status([STATUS_UNKNOWN, STATUS_DEGRADED]) == STATUS_DEGRADED
    assert worst_status([STATUS_UNKNOWN, STATUS_DEGRADED, STATUS_UNHEALTHY]) == STATUS_UNHEALTHY
    assert worst_status([STATUS_HEALTHY, STATUS_HEALTHY]) == STATUS_HEALTHY
    # Nothing to roll up is itself unknown, never a cheerful default.
    assert worst_status([]) == STATUS_UNKNOWN


def test_every_plan_6_3_dimension_is_declared_and_emitted(tmp_path):
    """The inventory is data, and unimplemented dimensions still emit a row."""
    from operations_audit import REQUIRED_CHECK_INVENTORY

    requirements = [entry.requirement for entry in REQUIRED_CHECK_INVENTORY]
    assert requirements == [
        "runtime profile and machine identity",
        "heartbeat age",
        "current and next job",
        "last attempt and last verified success per job/export",
        "job-ledger failures and exhausted retries",
        "owned process/thread counts",
        "writer-lease holder and expiry",
        "report freshness and verification state",
        "provider request, cache-hit, throttling, and failure counts",
        "universe and market-data freshness",
        "most recent scan manifest and phase timings",
        "SPY and Greatness shadow engine versions, last evaluations, coverage, and errors",
        "disk/storage warnings",
    ]

    diagnostics, registry, now = _measured_fixture(tmp_path)
    payload = _build(diagnostics, registry, now)
    rows = {row["id"]: row for row in payload["required_checks"]}
    assert set(rows) == {entry.id for entry in REQUIRED_CHECK_INVENTORY}

    emitted = {item["id"] for item in payload["checks"]}
    unimplemented = [entry for entry in REQUIRED_CHECK_INVENTORY if not entry.covered_by]
    # Every sec 6.3 dimension now has a measuring check: provider counters -
    # the last gap - gained a capture point (diagnostics.provider_counters via
    # the run manifest) and a real check that reports UNKNOWN until the first
    # instrumented scan writes counters. The rule that outlives the gaps:
    # every declared dimension is emitted, never silently absent from max().
    assert unimplemented == []
    for entry in REQUIRED_CHECK_INVENTORY:
        assert set(entry.covered_by) & emitted, (
            f"declared dimension {entry.id!r} has no emitted check covering it"
        )
        assert rows[entry.id]["implemented"] is True, entry.id

    # Honest cold start: the provider dimension is implemented but its check
    # still reports UNKNOWN until the first instrumented scan writes
    # provider.* counters into a manifest ("not measured" != "measured zero").
    provider = next(item for item in payload["checks"] if item["id"] == "provider_counters")
    assert provider["status"] == "unknown"
    assert provider["details"]["captured"] is False


def test_a_partial_payload_can_never_roll_up_to_healthy(tmp_path):
    """Even with every implemented row green, an UNKNOWN blocks HEALTHY."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    payload = _build(diagnostics, registry, now, review_capture=False)

    assert payload["status"] != "healthy"
    assert payload["status"] == "unknown"
    unknown_ids = {item["id"] for item in payload["checks"] if item["status"] == "unknown"}
    assert unknown_ids
    assert payload["summary"]["unknown"] == len(unknown_ids)


def test_capture_readiness_checks_reach_the_audit_and_never_read_the_shared_home(tmp_path):
    """Phase 0 task 4: the learning surfaces are visible in System Health."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    (diagnostics / "alert_review_events.jsonl").write_text(
        json.dumps(
            {
                "schema": "review_events_v1",
                "ts": "2026-07-13T10:15:00",
                "trade_date": "2026-07-13",
                "machine": "desk",
                "action": "shown",
                "symbol": "NVDA",
                "side": "LONG",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = _build(diagnostics, registry, now)
    checks = {item["id"]: item for item in payload["checks"]}

    assert checks["review_event_log"]["status"] == "healthy"
    assert checks["review_event_log"]["details"]["rows"] == 1
    assert checks["review_policy_gate"]["details"]["orders_active_queue"] is False
    assert checks["review_evidence_label"]["details"]["promotion_clock_started"] is False
    # The sandbox is self-contained: no check resolved back to the shared home.
    # Inventory-gap rows cite the plan bullet they came from, not a file.
    for check in payload["checks"]:
        assert (
            str(tmp_path) in check["source"]
            or check["source"].startswith("plan.md")
            or check["source"].endswith(
                ("review_capture_audit.py", "operations_audit.py", "scan_service.py")
            )
        )


def test_a_capture_gate_violation_makes_the_whole_audit_unhealthy(tmp_path, monkeypatch):
    """The gate that stopped holding is the one capture failure that must shout."""
    from review_guidance import ORDERING_MODE_ENV_VAR

    diagnostics, registry, now = _measured_fixture(tmp_path)
    monkeypatch.setenv(ORDERING_MODE_ENV_VAR, "preference")
    payload = _build(diagnostics, registry, now)
    checks = {item["id"]: item for item in payload["checks"]}

    assert payload["status"] == "unhealthy"
    assert checks["review_policy_gate"]["status"] == "unhealthy"
    assert checks["review_evidence_label"]["status"] == "unhealthy"


def test_stale_heartbeat_and_shadow_make_audit_unhealthy(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    heartbeat = json.loads((diagnostics / "heartbeat.json").read_text(encoding="utf-8"))
    heartbeat["ts"] = "2026-07-13T11:00:00-07:00"
    _write(diagnostics / "heartbeat.json", heartbeat)
    greatness = json.loads((diagnostics / "greatness_candidates.json").read_text(encoding="utf-8"))
    greatness["coverage"]["last_evaluation_at"] = "2026-07-13T11:00:00-07:00"
    _write(diagnostics / "greatness_candidates.json", greatness)

    payload = _build(diagnostics, registry, now)
    statuses = {item["id"]: item["status"] for item in payload["checks"]}

    assert payload["status"] == "unhealthy"
    assert statuses["heartbeat"] == "unhealthy"
    assert statuses["greatness_shadow"] == "unhealthy"


def test_audit_write_is_atomic_and_large_tracker_is_never_read(tmp_path):
    from operations_audit import refresh_operations_audit

    diagnostics, registry, now = _measured_fixture(tmp_path)
    payload = refresh_operations_audit(
        now=now,
        diagnostics_dir=diagnostics,
        candidate_registry_path=registry,
        process_snapshot=dict(_IDLE_PROCESS_SNAPSHOT),
    )
    written = json.loads((diagnostics / "operations_audit.json").read_text(encoding="utf-8"))

    assert written == payload
    assert not list(diagnostics.glob("*.tmp"))


def test_requested_tracker_write_that_was_skipped_degrades_manifest_check(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    manifest_path = diagnostics / "run_manifests" / "run-1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counters"] = {
        "update_setup_tracker": True,
        "setup_tracker_allowed": False,
        "setup_tracker_updated": False,
    }
    manifest["outputs"] = {
        "setup_tracker_skip_reason": "Tracked setup used non-IBKR daily data."
    }
    _write(manifest_path, manifest)

    payload = _build(diagnostics, registry, now)
    check = next(item for item in payload["checks"] if item["id"] == "run_manifest")

    assert check["status"] == "degraded"
    assert "tracker write skipped" in check["summary"].lower()
    assert check["details"]["setup_tracker_skip_reason"] == "Tracked setup used non-IBKR daily data."


def test_enabled_away_report_that_is_stale_or_tampered_is_unhealthy(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    metadata_path = diagnostics / "autopilot_today.txt.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["verified_at"] = "2026-07-13T10:00:00-07:00"
    _write(metadata_path, metadata)

    stale_payload = _build(diagnostics, registry, now)
    stale = next(item for item in stale_payload["checks"] if item["id"] == "away_report")
    assert stale["status"] == "unhealthy"
    assert "old" in stale["summary"]

    (diagnostics / "autopilot_today.txt").write_text("tampered", encoding="utf-8")
    tampered_payload = _build(diagnostics, registry, now)
    tampered = next(item for item in tampered_payload["checks"] if item["id"] == "away_report")
    assert tampered["status"] == "unhealthy"
    assert "hash" in tampered["summary"].lower()


def test_away_report_without_verification_metadata_is_never_green(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    (diagnostics / "autopilot_today.txt.meta.json").unlink()

    payload = _build(diagnostics, registry, now)
    check = next(item for item in payload["checks"] if item["id"] == "away_report")

    assert check["status"] == "degraded"
    assert "metadata is missing" in check["summary"]


def test_away_report_freshness_matches_hourly_schedule(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    metadata_path = diagnostics / "autopilot_today.txt.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected = [
        ("2026-07-13T11:30:00-07:00", "healthy"),
        ("2026-07-13T11:10:00-07:00", "degraded"),
        ("2026-07-13T10:29:00-07:00", "unhealthy"),
    ]
    for verified_at, expected_status in expected:
        metadata["verified_at"] = verified_at
        _write(metadata_path, metadata)
        payload = _build(diagnostics, registry, now)
        check = next(item for item in payload["checks"] if item["id"] == "away_report")
        assert check["status"] == expected_status


def _check_status(payload: dict, check_id: str) -> str:
    return next(item["status"] for item in payload["checks"] if item["id"] == check_id)


def _audit(diagnostics: Path, registry: Path, now: datetime) -> dict:
    return _build(diagnostics, registry, now, review_capture=False)


def test_missing_evidence_and_corrupt_evidence_are_different_statuses(tmp_path):
    """A file that was never written is UNKNOWN; one we cannot parse is UNHEALTHY.

    ``_read_json`` used to collapse OSError and JSONDecodeError into a single
    ``None``, so both rendered identically even though one means "wait for the
    writer" and the other means "go repair the artifact".
    """
    diagnostics, registry, now = _measured_fixture(tmp_path)

    (diagnostics / "heartbeat.json").unlink()
    registry.unlink()
    (diagnostics / "writer_health.json").unlink()
    missing = _audit(diagnostics, registry, now)
    assert _check_status(missing, "heartbeat") == "unknown"
    assert _check_status(missing, "candidate_registry") == "unknown"
    assert _check_status(missing, "runtime_profile") == "unknown"
    assert _check_status(missing, "writer_lease") == "unknown"
    assert missing["status"] == "unknown"

    (diagnostics / "heartbeat.json").write_text('{"pid": 12', encoding="utf-8")
    registry.write_text("not json at all", encoding="utf-8")
    (diagnostics / "writer_health.json").write_text('{"schema": "writer_health_v1"', encoding="utf-8")
    corrupt = _audit(diagnostics, registry, now)
    assert _check_status(corrupt, "heartbeat") == "unhealthy"
    assert _check_status(corrupt, "candidate_registry") == "unhealthy"
    assert _check_status(corrupt, "runtime_profile") == "unhealthy"
    assert _check_status(corrupt, "writer_lease") == "unhealthy"
    # Precedence: a proven failure still outranks the unmeasured dimensions.
    assert corrupt["status"] == "unhealthy"


def test_no_scan_manifest_for_today_is_unhealthy_not_a_stale_yellow_row(tmp_path):
    """"No scan ran today" must not fall back to grading last week's manifest."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    manifest_path = diagnostics / "run_manifests" / "run-1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["started_at"] = "2026-07-08T19:00:00+00:00"
    manifest["ended_at"] = "2026-07-08T19:18:00+00:00"
    _write(manifest_path, manifest)

    payload = _audit(diagnostics, registry, now)
    check = next(item for item in payload["checks"] if item["id"] == "run_manifest")

    assert check["status"] == "unhealthy"
    assert "No scan has run today." in check["summary"]
    assert check["details"]["manifest_for_market_date"] is False
    assert check["details"]["newest_manifest_date"] == "2026-07-08"

    # Before the first scheduled slot of the day this is absent evidence, not
    # a failure.
    pre_market = datetime.fromisoformat("2026-07-14T05:00:00-07:00")
    early = _audit(diagnostics, registry, pre_market)
    assert _check_status(early, "run_manifest") == "unknown"


def test_candidate_registry_outside_regular_hours_is_graded_by_age(tmp_path):
    """One active candidate no longer makes the registry green at any age."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    stale = datetime.fromisoformat("2026-07-13T08:00:00-07:00").timestamp()
    os.utime(registry, (stale, stale))

    post_market = datetime.fromisoformat("2026-07-13T20:00:00-07:00")
    payload = _audit(diagnostics, registry, post_market)
    check = next(item for item in payload["checks"] if item["id"] == "candidate_registry")
    assert check["status"] == "unhealthy"
    # Post-market freshness is measured to the close, not to 20:00.
    assert check["details"]["age_minutes_to_session_reference"] == 300.0

    current = datetime.fromisoformat("2026-07-13T12:00:00-07:00").timestamp()
    os.utime(registry, (current, current))
    fresh = _audit(diagnostics, registry, post_market)
    assert _check_status(fresh, "candidate_registry") == "healthy"

    # A registry nobody has refreshed for days is not green in pre-market either.
    ancient = datetime.fromisoformat("2026-07-08T12:00:00-07:00").timestamp()
    os.utime(registry, (ancient, ancient))
    pre_market = datetime.fromisoformat("2026-07-14T05:00:00-07:00")
    assert _check_status(_audit(diagnostics, registry, pre_market), "candidate_registry") == "unhealthy"


def test_shadow_store_needs_freshness_not_only_a_session_date_match(tmp_path):
    """Session-date match plus one evaluation used to be green with no age bound."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    greatness = json.loads((diagnostics / "greatness_candidates.json").read_text(encoding="utf-8"))
    greatness["coverage"]["last_evaluation_at"] = "2026-07-13T09:00:00-07:00"
    greatness["updated_at"] = "2026-07-13T09:00:00-07:00"
    _write(diagnostics / "greatness_candidates.json", greatness)
    spy = json.loads((diagnostics / "spy_state_shadow_status.json").read_text(encoding="utf-8"))
    spy["last_evaluation_at"] = "2026-07-13T12:55:00-07:00"
    _write(diagnostics / "spy_state_shadow_status.json", spy)

    post_market = datetime.fromisoformat("2026-07-13T20:00:00-07:00")
    payload = _audit(diagnostics, registry, post_market)
    assert _check_status(payload, "greatness_shadow") == "unhealthy"
    # A shadow that evaluated up to the bell is still green in the evening:
    # off-hours freshness is measured to the close, not to "now".
    assert _check_status(payload, "spy_shadow") == "healthy"

    # No evaluations for the current market date is absent evidence: UNKNOWN,
    # not the old cheerful/degraded pair.
    next_day = datetime.fromisoformat("2026-07-14T05:00:00-07:00")
    tomorrow = _audit(diagnostics, registry, next_day)
    assert _check_status(tomorrow, "greatness_shadow") == "unknown"
    assert _check_status(tomorrow, "spy_shadow") == "unknown"


def test_provider_counters_unknown_until_measured_then_graded(tmp_path):
    """Sec 6.3 bullet 9, v2 contract.

    Pre-instrumentation manifests stay UNKNOWN; an empty run is HEALTHY only
    when the declared boundary coverage is complete; partial coverage, capture
    errors, orphans, throttles and per-provider failure ratios each degrade or
    fail the row - and ratios only ever use matching (family, provider)
    attempt denominators.
    """
    from diagnostics import provider_counters as pc

    diagnostics, registry, now = _measured_fixture(tmp_path)

    # The fixture's manifest predates instrumentation: UNKNOWN, and emitted as
    # ONE check (no duplicate inventory-gap row for the same dimension). An
    # old v1 provider.captured stamp alone must NOT count as measured.
    payload = _audit(diagnostics, registry, now)
    provider_rows = [c for c in payload["checks"] if c["id"] == "provider_counters"]
    assert len(provider_rows) == 1
    assert provider_rows[0]["status"] == "unknown"
    assert provider_rows[0]["details"]["captured"] is False

    full = ",".join(pc.FAMILIES_EXPECTED)

    def _with(counters, *, instrumented=full, expected=full):
        manifest = {
            "schema": "run_manifest_v1",
            "run_id": "run-2",
            "job_type": "master_scan",
            "started_at": "2026-07-13T19:30:00+00:00",
            "ended_at": "2026-07-13T19:48:00+00:00",
            "status": "ok",
            "total_seconds": 1080,
            "counters": {
                "provider.schema_version": 2,
                "provider.capture_errors": 0,
                "provider.orphan_events": 0,
                **counters,
            },
            "outputs": {
                "provider_families_expected": expected,
                "provider_families_instrumented": instrumented,
            },
        }
        _write(diagnostics / "run_manifests" / "run-2.json", manifest)
        result = _audit(diagnostics, registry, now)
        return next(c for c in result["checks"] if c["id"] == "provider_counters")

    # v1 stamp alone (no schema_version) is still UNKNOWN.
    legacy_v1_manifest = {
        "schema": "run_manifest_v1",
        "run_id": "run-2",
        "job_type": "master_scan",
        "started_at": "2026-07-13T19:30:00+00:00",
        "status": "ok",
        "counters": {"provider.captured": 1},
    }
    _write(diagnostics / "run_manifests" / "run-2.json", legacy_v1_manifest)
    v1_only = next(
        c
        for c in _audit(diagnostics, registry, now)["checks"]
        if c["id"] == "provider_counters"
    )
    assert v1_only["status"] == "unknown"

    # Measured true zero with COMPLETE declared coverage: healthy, and says so.
    zero = _with({})
    assert zero["status"] == "healthy"
    assert "zero provider lookups occurred" in zero["summary"]

    # The same empty run with a missing instrumentation family: never healthy.
    partial = _with({}, instrumented="daily_bars,intraday_bars")
    assert partial["status"] == "degraded"
    assert "PARTIAL coverage" in partial["summary"]
    assert "theta_options" in partial["summary"]

    # Capture errors: the accounting itself failed - not healthy.
    broken_capture = _with({"provider.capture_errors": 2})
    assert broken_capture["status"] == "degraded"
    assert "capture error" in broken_capture["summary"]

    # A healthy measured run with real per-provider numbers.
    clean = _with(
        {
            "provider.daily_bars.lookup": 200,
            "provider.daily_bars.cache_hit": 160,
            "provider.daily_bars.attempt.ibkr": 40,
            "provider.daily_bars.success.ibkr": 40,
        }
    )
    assert clean["status"] == "healthy"
    assert clean["details"]["totals"]["lookup"] == 200
    assert clean["details"]["cache_hit_ratio"] == 0.8

    # Throttle without any ordinary failure: degraded, named as pacing.
    throttled = _with(
        {
            "provider.daily_bars.attempt.ibkr": 40,
            "provider.daily_bars.success.ibkr": 40,
            "provider.daily_bars.throttle.ibkr": 2,
        }
    )
    assert throttled["status"] == "degraded"
    assert "throttle" in throttled["summary"].lower()

    # Per-provider ratio uses the MATCHING denominator: 20 ibkr failures over
    # 40 ibkr attempts is unhealthy even though 200 logical lookups happened.
    failing = _with(
        {
            "provider.daily_bars.lookup": 200,
            "provider.daily_bars.attempt.ibkr": 40,
            "provider.daily_bars.failure.ibkr": 20,
        }
    )
    assert failing["status"] == "unhealthy"
    assert "daily_bars/ibkr" in failing["summary"]

    # Failures with no recorded attempts is an accounting anomaly, reported as
    # such - never divided by an unrelated total, never silently green.
    orphan_failures = _with({"provider.daily_bars.failure.yahoo": 3})
    assert orphan_failures["status"] == "degraded"
    assert "no recorded attempts" in orphan_failures["summary"]

    # Malformed counter values are tolerated and flagged, never a crash.
    malformed = _with({"provider.daily_bars.attempt.ibkr": "garbage"})
    assert malformed["status"] == "degraded"
    assert malformed["details"]["malformed_counter_values"] == 1


def test_spy_rollover_failure_is_unhealthy_and_non_promotable(tmp_path):
    """A persisted rollover failure is 'recording is STUCK', never mistaken for
    'no shadow event occurred' - and it blocks promotability outright."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    spy = json.loads((diagnostics / "spy_state_shadow_status.json").read_text(encoding="utf-8"))
    spy["rollover_failure"] = {
        "schema": "spy_rollover_failure_v1",
        "failed_at": "2026-07-13T06:31:00-07:00",
        "from_scope": {"session_date": "2026-07-12", "config_hash": "abc"},
        "to_scope": {"session_date": "2026-07-13", "config_hash": "abc"},
        "error_type": "TypeError",
        "error": "can't compare offset-naive and offset-aware datetimes",
    }
    _write(diagnostics / "spy_state_shadow_status.json", spy)

    payload = _audit(diagnostics, registry, now)
    check = next(item for item in payload["checks"] if item["id"] == "spy_shadow")
    assert check["status"] == "unhealthy"
    assert "rollover FAILED" in check["summary"]
    assert "NOT 'no shadow event occurred'" in check["summary"]
    assert check["details"]["promotable"] is False
    assert any(
        "rollover failed" in reason.lower()
        for reason in check["details"]["non_promotable_reasons"]
    )


def test_away_report_with_autopilot_disabled_is_not_green_at_any_age(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    _write(diagnostics / "autopilot_state.json", {"enabled": False, "profile": "DESK"})
    metadata_path = diagnostics / "autopilot_today.txt.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata["verified_at"] = "2026-07-13T11:45:00-07:00"
    _write(metadata_path, metadata)
    assert _check_status(_audit(diagnostics, registry, now), "away_report") == "healthy"

    metadata["verified_at"] = "2026-07-09T11:45:00-07:00"
    _write(metadata_path, metadata)
    aged = _audit(diagnostics, registry, now)
    assert _check_status(aged, "away_report") == "degraded"

    # Nothing published at all, and nothing scheduled to: unknown, not green.
    (diagnostics / "autopilot_today.txt").unlink()
    assert _check_status(_audit(diagnostics, registry, now), "away_report") == "unknown"


def test_writer_identity_and_lease_rows_consume_writer_health(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    payload = _audit(diagnostics, registry, now)
    profile = next(item for item in payload["checks"] if item["id"] == "runtime_profile")
    lease = next(item for item in payload["checks"] if item["id"] == "writer_lease")

    assert profile["status"] == "healthy"
    assert profile["details"]["machine"] == "test-machine"
    assert profile["details"]["role"] == "designated_writer"
    assert lease["status"] == "healthy"
    assert lease["details"]["fencing_generation"] == 7
    assert "test-machine" in lease["summary"]

    health_path = diagnostics / "writer_health.json"
    state = json.loads(health_path.read_text(encoding="utf-8"))
    state["lease_expires_at"] = "2026-07-13T12:00:00-07:00"
    _write(health_path, state)
    expired = _audit(diagnostics, registry, now)
    assert _check_status(expired, "writer_lease") == "degraded"

    state["lease_holder"] = ""
    state["lease_expires_at"] = ""
    _write(health_path, state)
    unowned = _audit(diagnostics, registry, now)
    assert _check_status(unowned, "writer_lease") == "unknown"


def test_job_ledger_absence_is_unknown_before_the_first_slot(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    (diagnostics / "job_ledger.jsonl").unlink()

    pre_market = datetime.fromisoformat("2026-07-13T05:00:00-07:00")
    assert _check_status(_audit(diagnostics, registry, pre_market), "job_ledger") == "unknown"
    assert _check_status(_audit(diagnostics, registry, now), "job_ledger") == "unhealthy"


# ---------------------------------------------------------------------------
# H2: the dimensions that used to be emitted as UNKNOWN
# ---------------------------------------------------------------------------
def _fail_job(diagnostics: Path, attempts: int, error_class: str) -> None:
    """Append a failed job that has burned ``attempts`` attempts."""
    key = "2026-07-13|swing_scan|13:00|shared-v1"
    events = [
        {
            "event": "scheduled",
            "key": key,
            "market_date": "2026-07-13",
            "job_type": "swing_scan",
            "slot": "13:00",
            "config_hash": "shared-v1",
            "ts": "2026-07-13T13:00:00-07:00",
        }
    ]
    for attempt in range(attempts):
        events.append({"event": "started", "key": key, "run_id": f"retry-{attempt}", "ts": "2026-07-13T13:00:01-07:00"})
        events.append(
            {
                "event": "failed",
                "key": key,
                "error_class": error_class,
                "error": "provider said no",
                "ts": "2026-07-13T13:05:00-07:00",
            }
        )
    with (diagnostics / "job_ledger.jsonl").open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps({"schema": "job_ledger_v1", **event}) + "\n")


def test_a_job_that_burned_its_whole_retry_budget_is_reported_as_exhausted(tmp_path):
    """Bullet 5's second half: failures AND exhausted retries.

    ``JobLedger`` has tracked attempts and a per-error-class budget since Phase
    2.5, but the audit never read ``should_retry``, so a job that will be tried
    again looked exactly like one nothing will ever re-run today.
    """
    diagnostics, registry, now = _measured_fixture(tmp_path)

    # bad_local_state has a budget of 1: one attempt still has a retry left.
    _fail_job(diagnostics, attempts=1, error_class="bad_local_state")
    retrying = next(
        item for item in _audit(diagnostics, registry, now)["checks"] if item["id"] == "job_ledger"
    )
    assert retrying["status"] == "unhealthy"
    assert retrying["details"]["retry_exhausted"] == []
    assert retrying["details"]["retry_available"][0]["attempt"] == 1
    assert "retry available" in retrying["summary"] or "retry available" in retrying["details"]["problems"][0]

    # Two more attempts put it past the budget: nothing will re-run it today.
    _fail_job(diagnostics, attempts=2, error_class="bad_local_state")
    payload = _audit(diagnostics, registry, now)
    exhausted = next(item for item in payload["checks"] if item["id"] == "job_ledger")
    assert exhausted["details"]["retry_exhausted_count"] == 1
    assert exhausted["details"]["retry_available"] == []
    assert exhausted["details"]["retry_exhausted"][0]["attempt"] == 3
    assert exhausted["details"]["retry_exhausted"][0]["retry_budget"] == 1
    assert "out of retries" in exhausted["summary"]
    assert "RETRIES EXHAUSTED" in exhausted["details"]["problems"][0]
    # The per-job detail the panel renders is published alongside it.
    assert any(job["slot"] == "13:00" and job["attempt"] == 3 for job in payload["jobs"])
    assert exhausted["details"]["last_verified_success_at"] == "2026-07-13T12:18:00-07:00"


def test_manifest_phase_timings_reach_the_payload_instead_of_one_aggregate_total(tmp_path):
    """Bullet 11: per-phase timings were recorded and then dropped."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    manifest_path = diagnostics / "run_manifests" / "run-1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["phases"] = [
        {"label": "universe", "seconds": 120.0},
        {"label": "tracker update", "seconds": 600.0},
        {"label": "TOTAL", "seconds": 1080.0},
    ]
    _write(manifest_path, manifest)

    payload = _audit(diagnostics, registry, now)
    check = next(item for item in payload["checks"] if item["id"] == "run_manifest")

    assert check["details"]["slowest_phase"]["label"] == "tracker update"
    assert [row["label"] for row in check["details"]["phases"]] == ["tracker update", "universe", "TOTAL"]
    # The aggregate TOTAL is not double-counted in the share.
    assert check["details"]["phases"][0]["share_pct"] == 83.3
    assert check["details"]["phases"][-1]["share_pct"] is None
    assert "Slowest phase tracker update 10.0m." in check["summary"]
    assert payload["latest_manifest"]["phases"][0]["label"] == "universe"


def test_owned_process_counts_are_measured_in_process_and_unknown_outside_it(tmp_path):
    """Bullet 6 + sec 6.1 after-session items 2-3."""
    diagnostics, registry, now = _measured_fixture(tmp_path)

    idle = next(
        item for item in _audit(diagnostics, registry, now)["checks"] if item["id"] == "owned_process_counts"
    )
    assert idle["status"] == "healthy"
    assert idle["details"]["owned_child_count"] == 0
    assert "No owned scan children" in idle["summary"]

    working = dict(_IDLE_PROCESS_SNAPSHOT)
    working.update(
        {
            "owned_child_count": 1,
            "lingering_child_pids": [9001],
            "active_scan_label": "Running shared-watchlist Master AVWAP scan...",
            "scan_worker_threads": 1,
        }
    )
    running = next(
        item
        for item in _build(diagnostics, registry, now, review_capture=False, process_snapshot=working)["checks"]
        if item["id"] == "owned_process_counts"
    )
    assert running["status"] == "healthy"

    orphaned = dict(working)
    orphaned["active_scan_label"] = ""
    lingering = next(
        item
        for item in _build(diagnostics, registry, now, review_capture=False, process_snapshot=orphaned)["checks"]
        if item["id"] == "owned_process_counts"
    )
    assert lingering["status"] == "degraded"
    assert "9001" in lingering["summary"]

    overdue_bounce = dict(_IDLE_PROCESS_SNAPSHOT)
    overdue_bounce.update(
        {
            "bounce_service_count": 1,
            "bounce_service_running_count": 0,
            "bounce_service_connected_count": 0,
            "bounce_unretired_worker_count": 1,
            "bounce_unretired_workers": ["startup worker (qt-bouncebot-start)"],
        }
    )
    overdue = next(
        item
        for item in _build(
            diagnostics,
            registry,
            now,
            review_capture=False,
            process_snapshot=overdue_bounce,
        )["checks"]
        if item["id"] == "owned_process_counts"
    )
    assert overdue["status"] == "degraded"
    assert "exceeded the shutdown budget" in overdue["summary"]
    assert overdue["details"]["bounce_unretired_worker_count"] == 1

    # Out of process the count is not zero - it is unmeasurable, and reporting
    # zero from a CLI that owns nothing would be a fabricated green.
    import operations_audit

    original = operations_audit._runtime_process_snapshot
    operations_audit._runtime_process_snapshot = lambda: None
    try:
        outside = next(
            item
            for item in _build(diagnostics, registry, now, review_capture=False, process_snapshot=None)["checks"]
            if item["id"] == "owned_process_counts"
        )
    finally:
        operations_audit._runtime_process_snapshot = original
    assert outside["status"] == "unknown"
    assert "owns no scanners" in outside["summary"]


def test_the_process_snapshot_hook_reports_the_real_scan_service_registry():
    """The audit consumes the existing ownership machinery, not a copy of it."""
    from ui.services import scan_service

    with scan_service._owned_processes_lock:
        parked = list(scan_service._owned_processes)
        scan_service._owned_processes.clear()
    try:
        snapshot = scan_service.owned_scan_process_snapshot()
        assert snapshot["owned_child_count"] == 0
        assert snapshot["process_pid"] == os.getpid()
        assert snapshot["python_thread_count"] >= 1

        class _FakeChild:
            pid = 4321

            def poll(self):
                return None

        child = _FakeChild()
        with scan_service._owned_processes_lock:
            scan_service._owned_processes.append(child)
        busy = scan_service.owned_scan_process_snapshot()
        assert busy["owned_child_count"] == 1
        assert busy["lingering_child_pids"] == [4321]
        # Strictly observational: the registry is not pruned or reaped by a read.
        with scan_service._owned_processes_lock:
            assert scan_service._owned_processes == [child]
    finally:
        with scan_service._owned_processes_lock:
            scan_service._owned_processes.clear()
            scan_service._owned_processes.extend(parked)


def test_disk_check_measures_free_space_writability_and_footprint(tmp_path):
    """Bullet 13 + sec 6.1 pre-session item 7 (the directory is writable)."""
    import operations_audit

    diagnostics, registry, now = _measured_fixture(tmp_path)
    check = next(
        item for item in _audit(diagnostics, registry, now)["checks"] if item["id"] == "disk_storage_warnings"
    )

    assert check["status"] == "healthy"
    assert check["details"]["diagnostics_writable"] is True
    assert check["details"]["free_gb"] > 0
    assert check["details"]["diagnostics_file_count"] > 0
    assert check["details"]["largest_artifacts"]
    # The probe leaves nothing behind.
    assert not list(diagnostics.glob("health-write-probe-*"))

    class _Usage:
        total = 200 * 1024**3
        used = total - (2 * 1024**3)
        free = 2 * 1024**3

    original = operations_audit.shutil.disk_usage
    operations_audit.shutil.disk_usage = lambda _path: _Usage
    try:
        low = next(
            item
            for item in _audit(diagnostics, registry, now)["checks"]
            if item["id"] == "disk_storage_warnings"
        )
        _Usage.free = 512 * 1024**2
        critical = next(
            item
            for item in _audit(diagnostics, registry, now)["checks"]
            if item["id"] == "disk_storage_warnings"
        )
    finally:
        operations_audit.shutil.disk_usage = original

    assert low["status"] == "degraded"
    assert critical["status"] == "unhealthy"


def test_an_unpruned_artifact_is_a_named_storage_warning(tmp_path, monkeypatch):
    """The shadow logs have no retention: the footprint must be visible."""
    import operations_audit

    diagnostics, registry, now = _measured_fixture(tmp_path)
    monkeypatch.setattr(operations_audit, "SINGLE_ARTIFACT_DEGRADED_MB", 0.05)
    (diagnostics / "greatness_shadow.jsonl").write_bytes(b"x" * 200_000)

    check = next(
        item for item in _audit(diagnostics, registry, now)["checks"] if item["id"] == "disk_storage_warnings"
    )

    assert check["status"] == "degraded"
    assert "greatness_shadow.jsonl" in check["summary"]
    assert check["details"]["largest_artifacts"][0]["artifact"] == "greatness_shadow.jsonl"


def test_an_unwritable_diagnostics_directory_is_unhealthy(tmp_path, monkeypatch):
    import operations_audit

    diagnostics, registry, now = _measured_fixture(tmp_path)
    monkeypatch.setattr(
        operations_audit,
        "_writability_probe",
        lambda directory: (False, "PermissionError: [Errno 13] denied", ""),
    )

    check = next(
        item for item in _audit(diagnostics, registry, now)["checks"] if item["id"] == "disk_storage_warnings"
    )

    assert check["status"] == "unhealthy"
    assert "not writable" in check["summary"]


def test_universe_and_market_data_freshness_is_graded_not_assumed(tmp_path):
    """Bullet 10: nothing schedules a universe rebuild, so age must be visible."""
    diagnostics, registry, now = _measured_fixture(tmp_path)

    fresh = next(
        item
        for item in _audit(diagnostics, registry, now)["checks"]
        if item["id"] == "universe_and_market_data_freshness"
    )
    assert fresh["status"] == "healthy"
    assert fresh["details"]["universe_symbol_total"] == 6
    assert fresh["details"]["market_data"]["calendar_days_behind"] == 0

    for name in ("universe_all.txt", "universe_longs.txt", "universe_shorts.txt"):
        _touch(diagnostics / name, datetime.fromisoformat("2026-07-01T16:00:00-07:00"))
    stale = next(
        item
        for item in _audit(diagnostics, registry, now)["checks"]
        if item["id"] == "universe_and_market_data_freshness"
    )
    assert stale["status"] == "degraded"
    assert "rebuilds are manual only" in stale["summary"]

    for name in ("universe_all.txt", "universe_longs.txt", "universe_shorts.txt"):
        _touch(diagnostics / name, datetime.fromisoformat("2026-05-01T16:00:00-07:00"))
    ancient = next(
        item
        for item in _audit(diagnostics, registry, now)["checks"]
        if item["id"] == "universe_and_market_data_freshness"
    )
    assert ancient["status"] == "unhealthy"

    # No universe at all is absent evidence, not a pass.
    for name in ("universe_all.txt", "universe_longs.txt", "universe_shorts.txt"):
        (diagnostics / name).unlink()
    missing = next(
        item
        for item in _audit(diagnostics, registry, now)["checks"]
        if item["id"] == "universe_and_market_data_freshness"
    )
    assert missing["status"] == "unknown"


def test_a_stale_daily_bar_store_degrades_the_freshness_row(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    probe = diagnostics / "daily_bars" / "SPY.parquet"

    # A normal weekend is not staleness.
    _touch(probe, datetime.fromisoformat("2026-07-10T13:00:00-07:00"))
    weekend = next(
        item
        for item in _audit(diagnostics, registry, now)["checks"]
        if item["id"] == "universe_and_market_data_freshness"
    )
    assert weekend["status"] == "healthy"

    _touch(probe, datetime.fromisoformat("2026-07-06T13:00:00-07:00"))
    stale = next(
        item
        for item in _audit(diagnostics, registry, now)["checks"]
        if item["id"] == "universe_and_market_data_freshness"
    )
    assert stale["status"] == "degraded"
    assert stale["details"]["market_data_status"] == "degraded"

    probe.unlink()
    unknown = next(
        item
        for item in _audit(diagnostics, registry, now)["checks"]
        if item["id"] == "universe_and_market_data_freshness"
    )
    assert unknown["status"] == "unknown"
    assert "market-data age is unknown" in unknown["summary"]


def test_writer_rows_report_live_coordination_state_not_the_last_publish(tmp_path):
    """Bullet 7 + the user's full Layer 5 field list."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    health_path = diagnostics / "writer_health.json"
    state = json.loads(health_path.read_text(encoding="utf-8"))
    state.update(
        {
            "config_source": "local_settings:designated_writer",
            "holder_identity": "test-machine:123:instance-1",
            "local_lock": {
                "key": "away",
                "name": "Global\\TradingBotV3-away",
                "held": True,
                "mutex": "held",
                "file_lock": "held",
                "abandoned_by_previous_owner": False,
            },
            "last_failure": {"at": "2026-07-13T11:00:00-07:00", "kind": "ownership", "message": "lease held elsewhere"},
            "last_blocked_at": "2026-07-13T11:00:00-07:00",
        }
    )
    _write(health_path, state)

    payload = _audit(diagnostics, registry, now)
    profile = next(item for item in payload["checks"] if item["id"] == "runtime_profile")
    lease = next(item for item in payload["checks"] if item["id"] == "writer_lease")

    # Identity, configuration and machine-local exclusion are all present.
    assert profile["details"]["config_source"] == "local_settings:designated_writer"
    assert profile["details"]["holder_identity"] == "test-machine:123:instance-1"
    assert profile["details"]["local_mutex"] == "held"
    assert profile["details"]["local_lock_held"] is True
    assert profile["details"]["last_failure"]["kind"] == "ownership"
    assert "PID 123" in profile["summary"]

    # The LIVE lease, plus the last verified publication as history beside it -
    # the audit used to report the last publish's holder/expiry as if it were
    # the current one.
    assert lease["details"]["lease_acquired_at"] == "2026-07-13T12:00:00-07:00"
    assert lease["details"]["lease_expires_at"] == "2026-07-13T13:00:00-07:00"
    assert lease["details"]["last_renewal_at"] == "2026-07-13T12:29:00-07:00"
    assert lease["details"]["last_verified_publication_generation"] == 7
    assert lease["details"]["emergency_override_active"] is False
    assert "Last verified publication by test-machine" in lease["summary"]


def test_an_unconfigured_or_read_only_machine_is_never_silently_green(
    tmp_path, monkeypatch
):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    health_path = diagnostics / "writer_health.json"
    state = json.loads(health_path.read_text(encoding="utf-8"))

    unconfigured = dict(state)
    unconfigured.update(
        {
            "role": "unconfigured",
            "designated_writer": "",
            "read_only": True,
            "read_only_reason": "no designated writer is configured on this machine",
            "lease_holder": "",
            "lease_expires_at": "",
        }
    )
    _write(health_path, unconfigured)
    # The LIVE role governs this row, so genuinely unconfigure the machine
    # rather than only writing an unconfigured artifact. conftest names this
    # host the designated writer for the suite; clear that here.
    for key in writer_role.ENV_WRITER_KEYS + writer_role.ENV_ROLE_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(writer_role, "_local_setting", lambda key: None)

    profile = next(
        item for item in _audit(diagnostics, registry, now)["checks"] if item["id"] == "runtime_profile"
    )
    assert profile["status"] == "degraded"
    assert "no usable writer configuration" in profile["summary"]
    assert "no designated writer is configured" in profile["summary"]


def test_stale_writer_telemetry_cannot_manufacture_a_live_role(tmp_path):
    """A leftover artifact must not be reported as the machine's current role.

    The smoke check publishes as a self-named designated writer; before this was
    fixed it left writer_health.json in the real diagnostics directory, and the
    audit reported that as the live role -- a confident green row on a machine
    that was actually unconfigured and would refuse to publish.
    """
    diagnostics, registry, now = _measured_fixture(tmp_path)
    health_path = diagnostics / "writer_health.json"
    state = json.loads(health_path.read_text(encoding="utf-8"))

    # conftest makes the live role designated_writer for this host; the stale
    # artifact claims something else entirely.
    stale = dict(state)
    stale.update({"role": "secondary", "designated_writer": "some-other-box"})
    _write(health_path, stale)

    profile = next(
        item for item in _audit(diagnostics, registry, now)["checks"] if item["id"] == "runtime_profile"
    )
    assert profile["status"] == "degraded"
    assert profile["details"]["telemetry_role_disagrees_with_live"] is True
    assert profile["details"]["live_role"] == writer_role.ROLE_DESIGNATED
    assert "stale" in profile["summary"].lower()


def test_a_correctly_configured_read_only_secondary_is_a_working_state(
    tmp_path, monkeypatch
):
    """Secondary is not a fault -- but the row says so, and claims no lease.

    The lease row must stay UNKNOWN rather than reporting the *other* machine's
    lease as evidence about this one.
    """
    diagnostics, registry, now = _measured_fixture(tmp_path)
    health_path = diagnostics / "writer_health.json"
    state = json.loads(health_path.read_text(encoding="utf-8"))

    # Live role must agree with the artifact, otherwise this exercises the
    # stale-telemetry path instead of the secondary path.
    for key in writer_role.ENV_WRITER_KEYS:
        monkeypatch.setenv(key, "test-machine")
    for key in writer_role.ENV_ROLE_KEYS:
        monkeypatch.setenv(key, writer_role.ROLE_SECONDARY)

    secondary = dict(state)
    secondary.update(
        {
            "machine": "mini-pc",
            "role": "secondary",
            "designated_writer": "test-machine",
            "read_only": True,
            "read_only_reason": "this machine is a read-only secondary",
            "lease_holder": "",
            "lease_expires_at": "",
        }
    )
    _write(health_path, secondary)
    payload = _audit(diagnostics, registry, now)
    profile = next(item for item in payload["checks"] if item["id"] == "runtime_profile")
    lease = next(item for item in payload["checks"] if item["id"] == "writer_lease")
    assert profile["status"] == "healthy"
    assert "read-only secondary" in profile["summary"]
    assert "test-machine publishes" in profile["summary"]
    assert lease["status"] == "unknown"
    assert "not observable from here" in lease["summary"]


def test_a_designated_writer_that_cannot_publish_is_a_real_problem(tmp_path):
    """Configured to publish but refusing is the case worth waking up for."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    health_path = diagnostics / "writer_health.json"
    state = json.loads(health_path.read_text(encoding="utf-8"))

    blocked = dict(state)
    blocked.update({"read_only": True, "read_only_reason": "lease held by mini-pc"})
    _write(health_path, blocked)
    stuck = next(
        item for item in _audit(diagnostics, registry, now)["checks"] if item["id"] == "runtime_profile"
    )
    assert stuck["status"] == "degraded"
    assert "currently read-only" in stuck["summary"]


# ---------------------------------------------------------------------------
# PACKET H3 - the raw shadow logs, not the writers' self-reports
#
# Every assertion below fails against the pre-H3 audit, which read only
# spy_state_shadow_status.json and greatness_candidates.json. That made the
# writing process the sole witness to its own output: a truncated tail, a
# half-written interior row, a drifted schema and a sidecar claiming rows the
# log does not contain were all invisible AND green.
# ---------------------------------------------------------------------------
def _shadow(payload: dict, check_id: str) -> dict:
    return next(item for item in payload["checks"] if item["id"] == check_id)


def _scan_of(payload: dict, check_id: str) -> dict:
    return _shadow(payload, check_id)["details"]["log_scan"]


def test_the_validator_accepts_exactly_the_schemas_the_writers_emit():
    """Multiple row schemas share one file; the validator must know all of them."""
    import greatness_shadow
    import market_state_bridge
    from diagnostics import shadow_log_audit

    # Pinned so a writer bump cannot silently become "unknown schema" noise.
    assert market_state_bridge.SHADOW_SCHEMA == "spy_state_shadow_v4"
    assert market_state_bridge.STATUS_SCHEMA == "spy_state_shadow_status_v3"
    assert market_state_bridge.EPISODE_SCHEMA == "spy_episode_shadow_v1"

    assert shadow_log_audit.SPY_PROFILE.accepted_schemas == frozenset(
        market_state_bridge.COMPATIBLE_SHADOW_SCHEMAS
    )
    # Episode rows live in the SAME log and are counted apart from state rows,
    # because the sidecar counts them apart too.
    assert shadow_log_audit.SPY_PROFILE.episode_schemas == frozenset(
        {market_state_bridge.EPISODE_SCHEMA}
    )
    assert market_state_bridge.EPISODE_SCHEMA not in shadow_log_audit.SPY_PROFILE.primary_schemas
    assert shadow_log_audit.GREATNESS_PROFILE.accepted_schemas == frozenset(
        greatness_shadow.COMPATIBLE_SHADOW_SCHEMAS
    )


def test_both_shadow_logs_are_actually_opened_and_reported(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    payload = _build(diagnostics, registry, now)

    spy = _scan_of(payload, "spy_shadow")
    assert spy["path"] == str(_spy_log(diagnostics))
    assert spy["valid_rows"] == 3
    assert spy["primary_rows"] == 2 and spy["episode_rows"] == 1
    assert spy["schemas"] == {"spy_state_shadow_v4": 2, "spy_episode_shadow_v1": 1}
    assert spy["engine_versions"] == {"spy_state_v1": 3}
    assert spy["config_hashes"] == {"spy-config-1": 3}
    assert spy["session_dates"] == {"2026-07-13": 3}
    assert spy["completed_bar_rows"] == 3 and spy["incomplete_bar_rows"] == 0
    # The latest valid record is reported, and it is a SUMMARY - a Greatness row
    # carries a provenance block and an event list that must never be inlined
    # into a health payload rendered every 15 seconds.
    assert spy["latest_valid_record"]["line"] == 3
    assert spy["latest_valid_record"]["episode_id"] == "ep-1"
    assert "events" not in spy["latest_valid_record"]

    great = _scan_of(payload, "greatness_shadow")
    assert great["valid_rows"] == 3
    assert great["schemas"] == {"greatness_shadow_v4": 3}
    assert great["completed_bar_rows_for_market_date"] == 3
    assert great["latest_valid_record"]["event"] == "STEP_CLEARED"

    for check_id in ("spy_shadow", "greatness_shadow"):
        details = _shadow(payload, check_id)["details"]
        assert details["promotable"] is True
        assert details["non_promotable_reasons"] == []
        assert details["session_progress"]["summary_count"] == 0
        assert details["session_progress"]["affects_promotion"] is False
    assert payload["shadow_evidence"]["promotable"] is True


def test_finalized_session_floor_progress_is_counted_but_never_promotes(tmp_path):
    from diagnostics.shadow_session_rollup import SPY_ENGINE, finalize_session

    diagnostics, registry, now = _measured_fixture(tmp_path)
    historical = diagnostics / "spy_history_staging.jsonl"
    rows = [
        _spy_row("2026-07-12T12:10:00-07:00"),
        {
            **_spy_row("2026-07-12T12:20:00-07:00", schema="spy_episode_shadow_v1"),
            "episode_uid": "SPY|2026-07-12|ep-old",
            "episode_id": "ep-old",
            "outcome": "RESUMED",
            "direction": "BULLISH",
            "derived_from_completed_bars": True,
        },
    ]
    historical.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    finalize_session(
        engine=SPY_ENGINE,
        log_path=historical,
        coverage={
            "session_date": "2026-07-12",
            "config_hash": "spy-config-1",
            "evaluations": 3,
            "usable_evaluations": 3,
            "errors": 0,
        },
        finalized_at=now,
        reason="session_rollover",
        engine_version="spy_state_v1",
        machine="test-machine",
        timezone="America/Vancouver",
        configuration="spy-config-1",
    )

    payload = _build(diagnostics, registry, now)
    progress = _shadow(payload, "spy_shadow")["details"]["session_progress"]

    assert progress["eligible_sessions"] == 1
    assert progress["complete_chains"] == 1
    assert progress["section_7_floor_progress"]["eligible_sessions"] == {
        "count": 1,
        "floor": 10,
    }
    assert progress["manual_reviewed_chains"] == 0
    assert progress["promotion_decision"] == "NONE"
    # Counter visibility does not alter the existing raw-log promotion verdict.
    assert payload["shadow_evidence"]["promotable"] is True
    assert (
        payload["shadow_evidence"]["engines"]["spy_shadow"]["session_progress"][
            "affects_promotion"
        ]
        is False
    )


def test_a_truncated_final_line_is_unhealthy_and_not_promotable(tmp_path):
    """The crash signature: a writer died between the record and its newline."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    log = _greatness_log(diagnostics)
    rows = [
        _greatness_row("2026-07-13T12:15:00-07:00"),
        _greatness_row("2026-07-13T12:20:00-07:00"),
        _greatness_row("2026-07-13T12:25:00-07:00"),
    ]
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    # A half-flushed record with no terminator - exactly what a killed process
    # leaves behind. Every runtime reader in this repo skips it in silence.
    half = json.dumps(_greatness_row("2026-07-13T12:30:00-07:00"))[:120]
    log.write_text(text + half, encoding="utf-8")

    payload = _build(diagnostics, registry, now)
    check = _shadow(payload, "greatness_shadow")
    scan = check["details"]["log_scan"]

    assert scan["truncated_final_line"] is True
    assert "no line terminator" in scan["truncated_final_line_detail"]
    # A truncated TAIL is not the same failure as a corrupt interior line and is
    # never counted as one.
    assert scan["malformed_lines"] == 0
    assert scan["valid_rows"] == 3
    assert check["status"] == "unhealthy"
    assert "TRUNCATED FINAL LINE" in check["summary"]
    assert "NOT PROMOTABLE" in check["summary"]
    assert check["details"]["promotable"] is False
    assert any("truncated" in reason for reason in check["details"]["non_promotable_reasons"])
    # One damaged log blocks the whole shadow-evidence claim.
    assert payload["shadow_evidence"]["promotable"] is False
    assert payload["shadow_evidence"]["engines"]["greatness_shadow"]["truncated_final_line"] is True
    assert payload["status"] == "unhealthy"


def test_a_malformed_interior_line_is_a_different_finding_from_a_truncated_tail(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    log = _spy_log(diagnostics)
    lines = [
        json.dumps(_spy_row("2026-07-13T12:10:00-07:00")),
        '{"schema": "spy_state_shadow_v4", "ts": "2026-07-13T12:15',  # interleaved append
        "[1, 2, 3]",  # valid JSON, not a record
        json.dumps(_spy_row("2026-07-13T12:20:00-07:00")),
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = _build(diagnostics, registry, now)
    check = _shadow(payload, "spy_shadow")
    scan = check["details"]["log_scan"]

    assert scan["truncated_final_line"] is False  # the file ends cleanly
    assert scan["malformed_lines"] == 2
    assert scan["non_object_rows"] == 1
    assert [item["line"] for item in scan["malformed_examples"]] == [2, 3]
    assert scan["valid_rows"] == 2
    assert check["status"] == "unhealthy"
    assert check["details"]["promotable"] is False


def test_a_sidecar_claiming_more_rows_than_the_log_holds_is_caught(tmp_path):
    """The self-report failure this packet exists to catch."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    status_path = diagnostics / "spy_state_shadow_status.json"
    sidecar = json.loads(status_path.read_text(encoding="utf-8"))
    sidecar["rows_written"] = 9  # the log holds 2 state rows for this session
    _write(status_path, sidecar)

    payload = _build(diagnostics, registry, now)
    check = _shadow(payload, "spy_shadow")
    row = next(
        item
        for item in check["details"]["sidecar_reconciliation"]
        if item["claim"] == "rows_written"
    )
    assert row["state"] == "over_claimed"
    assert row["claimed"] == 9 and row["observed"] == 2
    assert check["status"] == "unhealthy"
    assert check["details"]["promotable"] is False
    assert any(
        "not supported by its own output" in text
        for text in check["details"]["non_promotable_reasons"]
    )


def test_greatness_events_emitted_is_a_floor_not_an_equality(tmp_path):
    """The board appends audit rows that never bump ``events_emitted``."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    store_path = diagnostics / "greatness_candidates.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    store["coverage"]["events_emitted"] = 2  # log holds 3 rows: 1 is an audit row
    _write(store_path, store)

    check = _shadow(_build(diagnostics, registry, now), "greatness_shadow")
    row = check["details"]["sidecar_reconciliation"][0]
    assert row["relation"] == "floor"
    assert row["state"] == "reconciled"
    assert check["status"] == "healthy"
    assert check["details"]["promotable"] is True

    # But claiming MORE events than rows exist is still a contradiction.
    store["coverage"]["events_emitted"] = 40
    _write(store_path, store)
    broken = _shadow(_build(diagnostics, registry, now), "greatness_shadow")
    assert broken["status"] == "unhealthy"
    assert broken["details"]["promotable"] is False


def test_an_absent_shadow_log_is_unknown_and_never_green(tmp_path):
    """Absent required evidence is uncertainty, not confirmation (sec 5)."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    _set_spy_log(diagnostics, None)

    check = _shadow(_build(diagnostics, registry, now), "spy_shadow")
    assert check["status"] == "unknown"
    assert check["details"]["log_status"] == "unknown"
    assert check["details"]["promotable"] is False
    assert any("does not exist" in text for text in check["details"]["non_promotable_reasons"])


def test_a_corrupt_sidecar_does_not_stop_the_log_from_being_audited(tmp_path):
    """A broken self-report is exactly when the raw evidence matters most."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    (diagnostics / "spy_state_shadow_status.json").write_text("{not json", encoding="utf-8")

    check = _shadow(_build(diagnostics, registry, now), "spy_shadow")
    assert check["status"] == "unhealthy"  # corrupt sidecar
    # The log was still streamed, and its findings are reported alongside.
    assert check["details"]["log_scan"]["valid_rows"] == 3
    assert check["details"]["log_status"] == "healthy"
    # ...but nothing is promotable while the self-report cannot be reconciled.
    assert check["details"]["promotable"] is False
    assert any("cannot be reconciled" in text for text in check["details"]["non_promotable_reasons"])


def test_schema_drift_in_the_log_is_degraded_and_not_promotable(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    rows = [
        _spy_row("2026-07-13T12:10:00-07:00"),
        _spy_row("2026-07-13T12:20:00-07:00", schema="spy_state_shadow_v9"),
    ]
    _set_spy_log(diagnostics, rows)

    check = _shadow(_build(diagnostics, registry, now), "spy_shadow")
    scan = check["details"]["log_scan"]
    assert scan["unknown_schema_rows"] == 1
    assert scan["unknown_schemas"] == {"spy_state_shadow_v9": 1}
    assert scan["malformed_lines"] == 0  # it parses fine; it is just not ours
    assert check["status"] == "degraded"
    assert check["details"]["promotable"] is False


def test_out_of_order_and_future_timestamps_are_findings(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    _set_spy_log(
        diagnostics,
        [
            _spy_row("2026-07-13T12:20:00-07:00"),
            _spy_row("2026-07-13T12:05:00-07:00"),  # appended after a later row
        ],
    )
    check = _shadow(_build(diagnostics, registry, now), "spy_shadow")
    scan = check["details"]["log_scan"]
    assert scan["out_of_order_rows"] == 1
    assert scan["out_of_order_examples"][0]["line"] == 2
    assert check["status"] == "degraded"
    assert check["details"]["promotable"] is False

    # A row stamped after the moment the audit runs is a clock/timezone fault.
    _set_spy_log(
        diagnostics,
        [
            _spy_row("2026-07-13T12:20:00-07:00"),
            _spy_row("2026-07-13T18:00:00-07:00"),  # `now` is 12:30
        ],
    )
    future = _shadow(_build(diagnostics, registry, now), "spy_shadow")
    assert future["details"]["log_scan"]["future_rows"] == 1
    assert future["status"] == "degraded"
    assert future["details"]["promotable"] is False


def test_a_row_with_an_unusable_timestamp_is_reported_not_dropped(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    _set_spy_log(
        diagnostics,
        [
            _spy_row("2026-07-13T12:10:00-07:00"),
            {**_spy_row("2026-07-13T12:15:00-07:00"), "ts": "not-a-date", "evaluated_at": ""},
            {**_spy_row("2026-07-13T12:20:00-07:00"), "ts": "", "evaluated_at": ""},
        ],
    )
    scan = _scan_of(_build(diagnostics, registry, now), "spy_shadow")
    assert scan["rows_with_unparsable_timestamp"] == 1
    assert scan["rows_missing_timestamp"] == 1
    assert scan["valid_rows"] == 3  # the rows themselves are still counted


def test_completed_bar_evidence_distinguishes_forming_bars(tmp_path):
    """plan.md sec 5: completed bars only; a forming bar is preview."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    forming = {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0, "complete": False}
    _set_greatness_log(
        diagnostics,
        [
            _greatness_row("2026-07-13T12:15:00-07:00"),
            _greatness_row("2026-07-13T12:20:00-07:00", bar=forming),
            _greatness_row("2026-07-13T12:25:00-07:00", bar={}),
        ],
    )
    scan = _scan_of(_build(diagnostics, registry, now), "greatness_shadow")
    assert scan["completed_bar_rows"] == 1
    assert scan["incomplete_bar_rows"] == 1
    assert scan["rows_without_bar_evidence"] == 1
    assert scan["completed_bar_rows_for_market_date"] == 1

    # A day whose rows carry NO completed-bar evidence at all cannot support a
    # promotion claim, even though every line parses.
    _set_greatness_log(
        diagnostics,
        [_greatness_row("2026-07-13T12:20:00-07:00", bar=forming)],
    )
    check = _shadow(_build(diagnostics, registry, now), "greatness_shadow")
    assert check["status"] == "degraded"
    assert check["details"]["promotable"] is False
    assert any("completed-bar evidence" in text for text in check["details"]["non_promotable_reasons"])


def test_two_machines_appending_to_one_shadow_log_is_a_finding(tmp_path):
    diagnostics, registry, now = _measured_fixture(tmp_path)
    _set_spy_log(
        diagnostics,
        [
            _spy_row("2026-07-13T12:10:00-07:00"),
            _spy_row("2026-07-13T12:20:00-07:00", machine="mini-pc"),
        ],
    )
    check = _shadow(_build(diagnostics, registry, now), "spy_shadow")
    assert set(check["details"]["log_scan"]["machines"]) == {"test-machine", "mini-pc"}
    assert check["status"] == "degraded"
    assert check["details"]["promotable"] is False


def test_no_rows_yet_today_is_a_note_not_a_promotability_block(tmp_path):
    """Every pre-market run has no rows yet; that must not cry wolf."""
    diagnostics, registry, now = _measured_fixture(tmp_path)
    _set_spy_log(diagnostics, [_spy_row("2026-07-10T12:10:00-07:00")])

    check = _shadow(_build(diagnostics, registry, now), "spy_shadow")
    assert check["details"]["log_status"] == "healthy"
    assert check["details"]["non_promotable_reasons"] == []
    assert any("contributes nothing" in text for text in check["details"]["log_notes"])


def test_the_log_is_streamed_and_never_read_whole(tmp_path, monkeypatch):
    """greatness_shadow.jsonl is ~14.5 MB with no retention policy."""
    from pathlib import Path as _Path

    diagnostics, registry, now = _measured_fixture(tmp_path)
    rows = [_greatness_row("2026-07-13T12:15:00-07:00") for _ in range(500)]
    _set_greatness_log(diagnostics, rows)

    logs = ("greatness_shadow.jsonl", "spy_state_shadow.jsonl")
    original_read_text = _Path.read_text
    original_read_bytes = _Path.read_bytes

    def _guard_text(self, *args, **kwargs):
        if self.name.endswith(logs):
            raise AssertionError(f"{self.name} was slurped instead of streamed")
        return original_read_text(self, *args, **kwargs)

    def _guard_bytes(self, *args, **kwargs):
        if self.name.endswith(logs):
            raise AssertionError(f"{self.name} was slurped instead of streamed")
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "read_text", _guard_text)
    monkeypatch.setattr(_Path, "read_bytes", _guard_bytes)

    scan = _scan_of(_build(diagnostics, registry, now), "greatness_shadow")
    assert scan["valid_rows"] == 500


def test_distinct_value_reporting_is_bounded(tmp_path):
    """A damaged log must not turn the audit into the memory problem."""
    from diagnostics import shadow_log_audit

    diagnostics, registry, now = _measured_fixture(tmp_path)
    rows = [
        _spy_row("2026-07-13T12:10:00-07:00", config_hash=f"hash-{index}")
        for index in range(shadow_log_audit.MAX_DISTINCT_VALUES + 20)
    ]
    _set_spy_log(diagnostics, rows)
    scan = _scan_of(_build(diagnostics, registry, now), "spy_shadow")
    assert len(scan["config_hashes"]) == shadow_log_audit.MAX_DISTINCT_VALUES + 1
    assert scan["config_hashes"]["(other)"] == 20
    assert len(scan["malformed_examples"]) <= shadow_log_audit.MAX_EXAMPLES
