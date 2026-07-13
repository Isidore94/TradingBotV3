"""Small, deterministic health audit for the unattended Sol3 runtime.

Only machine-local diagnostics and the compact candidate registry are read.
The large setup-tracker payload is intentionally outside this audit.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from diagnostics.run_manifest import load_recent_manifests
from job_ledger import JobLedger
from market_session import get_market_local_timezone, get_market_session_window, normalize_market_local_datetime
from project_paths import CACHE_DIR, get_diagnostics_dir


AUDIT_SCHEMA = "operations_audit_v1"
_STATUS_ORDER = {"healthy": 0, "degraded": 1, "unhealthy": 2}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_timestamp(value: Any, local_tz) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_tz)
    return parsed.astimezone(local_tz)


def _age_minutes(value: Any, now: datetime, local_tz) -> float | None:
    parsed = _parse_timestamp(value, local_tz)
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds() / 60.0)


def _phase(now: datetime) -> tuple[str, Any]:
    session = get_market_session_window(now)
    if now.weekday() >= 5:
        return "closed", session
    if now < session.open_local:
        return "pre_market", session
    if now <= session.close_local:
        return "regular", session
    return "post_market", session


def _check(
    check_id: str,
    label: str,
    status: str,
    summary: str,
    *,
    source: Path,
    updated_at: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "label": label,
        "status": status,
        "summary": summary,
        "updated_at": str(updated_at or ""),
        "source": str(source),
        "details": details or {},
    }


def _freshness_status(age: float | None, healthy_minutes: float, unhealthy_minutes: float) -> str:
    if age is None or age > unhealthy_minutes:
        return "unhealthy"
    if age > healthy_minutes:
        return "degraded"
    return "healthy"


def _heartbeat_check(path: Path, now: datetime, local_tz) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return _check("heartbeat", "Runtime heartbeat", "unhealthy", "Heartbeat is missing or unreadable.", source=path)
    age = _age_minutes(payload.get("ts"), now, local_tz)
    status = _freshness_status(age, 2.5, 10.0)
    age_text = "unknown age" if age is None else f"{age:.1f}m old"
    current = str(payload.get("current_job") or "idle")
    summary = f"PID {payload.get('pid') or '?'}; {current}; {age_text}."
    return _check(
        "heartbeat",
        "Runtime heartbeat",
        status,
        summary,
        source=path,
        updated_at=str(payload.get("ts") or ""),
        details={
            "machine": payload.get("machine") or "",
            "pid": payload.get("pid"),
            "current_job": payload.get("current_job") or "",
            "next_job": payload.get("next_job") or "",
            "last_success": payload.get("last_success") or "",
            "age_minutes": round(age, 2) if age is not None else None,
        },
    )


def _ledger_check(path: Path, market_date: str, now: datetime, local_tz, market_phase: str) -> tuple[dict[str, Any], list[dict]]:
    if not path.exists():
        status = "degraded" if market_phase == "pre_market" else "unhealthy"
        return _check("job_ledger", "Scheduled jobs", status, "Job ledger is missing.", source=path), []
    ledger = JobLedger(path)
    jobs = ledger.jobs_for_date(market_date)
    state_counts = Counter(job.state for job in jobs)
    problems: list[str] = []
    for job in jobs:
        if job.state in {"FAILED", "STALE"}:
            problems.append(f"{job.slot} {job.state.lower()}: {job.error or job.error_class or 'unknown error'}")
        if job.state == "RUNNING":
            running_age = _age_minutes(job.started_at, now, local_tz)
            if running_age is None or running_age > 35.0:
                problems.append(f"{job.slot} running too long")

    completed = int(state_counts.get("COMPLETED", 0))
    active = int(state_counts.get("RUNNING", 0) + state_counts.get("QUEUED", 0))
    if problems:
        status = "unhealthy"
    elif completed or active:
        status = "healthy"
    elif market_phase == "pre_market":
        status = "degraded"
    else:
        status = "unhealthy"
        problems.append("No jobs recorded for the current market date")
    summary = f"{completed} completed, {active} active, {len(problems)} problem(s)."
    serialized = [asdict(job) for job in jobs]
    return (
        _check(
            "job_ledger",
            "Scheduled jobs",
            status,
            summary,
            source=path,
            updated_at=max((job.ended_at or job.started_at or job.scheduled_at for job in jobs), default=""),
            details={"state_counts": dict(state_counts), "problems": problems, "job_count": len(jobs)},
        ),
        serialized,
    )


def _manifest_check(path: Path, market_date: str, now: datetime, local_tz, running_job: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    manifests = load_recent_manifests(path, limit=30)
    current = [item for item in manifests if str(item.get("started_at") or "")[:10] == market_date]
    latest = current[0] if current else (manifests[0] if manifests else {})
    if not latest:
        return _check("run_manifest", "Latest scan manifest", "unhealthy", "No scan manifest is available.", source=path), {}
    ended = latest.get("ended_at") or latest.get("started_at") or ""
    age = _age_minutes(ended, now, local_tz)
    latest_status = str(latest.get("status") or "").strip().lower()
    if latest_status not in {"ok", "success", "completed"}:
        status = "unhealthy"
    elif age is None:
        status = "degraded"
    elif age > (240.0 if running_job else 180.0):
        status = "degraded"
    else:
        status = "healthy"
    summary = (
        f"{latest.get('job_type') or 'scan'} {latest_status or 'unknown'}; "
        f"{float(latest.get('total_seconds') or 0.0) / 60.0:.1f}m; "
        f"{('unknown age' if age is None else f'{age:.1f}m old')}."
    )
    return (
        _check(
            "run_manifest",
            "Latest scan manifest",
            status,
            summary,
            source=path,
            updated_at=str(ended),
            details={
                "run_id": latest.get("run_id") or "",
                "trigger": latest.get("trigger") or "",
                "status": latest_status,
                "error": latest.get("error") or "",
                "total_seconds": latest.get("total_seconds"),
                "counters": latest.get("counters") or {},
                "age_minutes": round(age, 2) if age is not None else None,
            },
        ),
        latest,
    )


def _shadow_check(
    *,
    check_id: str,
    label: str,
    path: Path,
    now: datetime,
    local_tz,
    market_date: str,
    market_phase: str,
    coverage_key: str | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return _check(check_id, label, "unhealthy", "Diagnostic store is missing or unreadable.", source=path)
    coverage = payload.get(coverage_key) if coverage_key else payload
    coverage = coverage if isinstance(coverage, dict) else {}
    session_date = str(coverage.get("session_date") or payload.get("session_date") or "")
    evaluations = int(coverage.get("evaluations", 0) or 0)
    errors = int(coverage.get("errors", 0) or 0)
    last_eval = coverage.get("last_evaluation_at") or payload.get("updated_at") or ""
    age = _age_minutes(last_eval, now, local_tz)
    if errors:
        status = "unhealthy"
    elif market_phase == "regular":
        status = _freshness_status(age, 20.0, 45.0)
        if session_date != market_date or evaluations <= 0:
            status = "unhealthy"
    elif session_date == market_date and evaluations > 0:
        status = "healthy"
    else:
        status = "degraded"
    bars = int(coverage.get("bars_consumed", coverage.get("usable_evaluations", 0)) or 0)
    summary = f"{evaluations} evaluations, {bars} usable bars/evaluations, {errors} errors."
    details = dict(coverage)
    details.update(
        {
            "session_date": session_date,
            "engine_version": payload.get("engine_version") or "",
            "config_hash": payload.get("config_hash") or "",
            "timezone": payload.get("timezone") or "",
            "candidate_count": len(payload.get("candidates") or []),
            "age_minutes": round(age, 2) if age is not None else None,
        }
    )
    return _check(check_id, label, status, summary, source=path, updated_at=str(last_eval), details=details)


def _registry_check(path: Path, now: datetime, local_tz, market_phase: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return _check("candidate_registry", "Candidate registry", "unhealthy", "Registry is missing or unreadable.", source=path)
    candidates = [item for item in payload.get("candidates", []) if isinstance(item, dict)]
    active = [item for item in candidates if item.get("stage") not in {"INVALID", "EXPIRED"} and item.get("memberships")]
    sources: Counter[str] = Counter()
    for candidate in active:
        sources.update(str(source) for source in (candidate.get("memberships") or {}))
    try:
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=local_tz).isoformat(timespec="seconds")
    except OSError:
        updated_at = ""
    age = _age_minutes(updated_at, now, local_tz)
    if not active:
        status = "unhealthy"
    elif market_phase == "regular":
        status = _freshness_status(age, 90.0, 180.0)
    else:
        status = "healthy"
    summary = f"Generation {int(payload.get('generation', 0) or 0)}; {len(active)} active candidates."
    return _check(
        "candidate_registry",
        "Candidate registry",
        status,
        summary,
        source=path,
        updated_at=updated_at,
        details={
            "generation": int(payload.get("generation", 0) or 0),
            "candidate_count": len(candidates),
            "active_count": len(active),
            "source_counts": dict(sources),
            "age_minutes": round(age, 2) if age is not None else None,
        },
    )


def build_operations_audit(
    *,
    now: datetime | None = None,
    diagnostics_dir: Path | str | None = None,
    candidate_registry_path: Path | str | None = None,
) -> dict[str, Any]:
    local_tz, timezone_name = get_market_local_timezone()
    moment = normalize_market_local_datetime(now, local_timezone=local_tz)
    market_phase, session = _phase(moment)
    diagnostics = Path(diagnostics_dir) if diagnostics_dir is not None else get_diagnostics_dir()
    registry_path = Path(candidate_registry_path) if candidate_registry_path is not None else CACHE_DIR.parent / "candidate_registry.json"
    market_date = session.market_date.isoformat()

    heartbeat = _heartbeat_check(diagnostics / "heartbeat.json", moment, local_tz)
    ledger, jobs = _ledger_check(diagnostics / "job_ledger.jsonl", market_date, moment, local_tz, market_phase)
    running_job = any(job.get("state") == "RUNNING" for job in jobs)
    manifest, latest_manifest = _manifest_check(
        diagnostics / "run_manifests", market_date, moment, local_tz, running_job
    )
    checks = [
        heartbeat,
        ledger,
        manifest,
        _shadow_check(
            check_id="spy_shadow",
            label="SPY state shadow",
            path=diagnostics / "spy_state_shadow_status.json",
            now=moment,
            local_tz=local_tz,
            market_date=market_date,
            market_phase=market_phase,
        ),
        _shadow_check(
            check_id="greatness_shadow",
            label="Greatness shadow",
            path=diagnostics / "greatness_candidates.json",
            now=moment,
            local_tz=local_tz,
            market_date=market_date,
            market_phase=market_phase,
            coverage_key="coverage",
        ),
        _registry_check(registry_path, moment, local_tz, market_phase),
    ]
    overall = max((check["status"] for check in checks), key=lambda item: _STATUS_ORDER[item])
    counts = Counter(check["status"] for check in checks)
    return {
        "schema": AUDIT_SCHEMA,
        "generated_at": moment.isoformat(timespec="seconds"),
        "timezone": timezone_name,
        "market_date": market_date,
        "market_phase": market_phase,
        "market_session": session.session_label,
        "status": overall,
        "summary": {
            "healthy": int(counts.get("healthy", 0)),
            "degraded": int(counts.get("degraded", 0)),
            "unhealthy": int(counts.get("unhealthy", 0)),
            "total": len(checks),
        },
        "checks": checks,
        "jobs": jobs,
        "latest_manifest": latest_manifest,
        "excluded": ["large setup-tracker payload"],
    }


def write_operations_audit(payload: dict[str, Any], path: Path | str | None = None) -> Path:
    target = Path(path) if path is not None else get_diagnostics_dir() / "operations_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
    return target


def refresh_operations_audit(**kwargs) -> dict[str, Any]:
    payload = build_operations_audit(**kwargs)
    diagnostics_dir = kwargs.get("diagnostics_dir")
    target = Path(diagnostics_dir) / "operations_audit.json" if diagnostics_dir is not None else None
    write_operations_audit(payload, target)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit TradingBotV3 unattended runtime health.")
    parser.add_argument("--json", action="store_true", help="Print the complete JSON payload.")
    parser.add_argument("--no-write", action="store_true", help="Do not persist operations_audit.json.")
    args = parser.parse_args(argv)
    payload = build_operations_audit()
    if not args.no_write:
        write_operations_audit(payload)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        summary = payload["summary"]
        print(
            f"{payload['status'].upper()}: {summary['healthy']} healthy, "
            f"{summary['degraded']} degraded, {summary['unhealthy']} unhealthy"
        )
        for check in payload["checks"]:
            print(f"{check['status'].upper():9} {check['label']}: {check['summary']}")
    return 0 if payload["status"] == "healthy" else 1


if __name__ == "__main__":
    raise SystemExit(main())
