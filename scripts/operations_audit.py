"""Small, deterministic health audit for the unattended Sol3 runtime.

Only machine-local diagnostics and the compact candidate registry are read.
The large setup-tracker payload is intentionally outside this audit.

Four statuses, not three (plan.md sec 6.3: "The page must show ``UNKNOWN`` when
evidence is absent. It must not convert missing telemetry into a green state."):

``UNKNOWN``
    Required telemetry is absent or has not been measured yet.
``UNHEALTHY``
    Evidence proves failure, OR expected evidence is unreadable/corrupt.
``DEGRADED``
    Measured, but stale or marginal.
``HEALTHY``
    Measured and within bounds.

Precedence is ``UNHEALTHY > DEGRADED > UNKNOWN > HEALTHY``, so a single
required dimension nobody measures keeps the page out of green while a real
measured failure still outranks it.

The other half of that rule is :data:`REQUIRED_CHECK_INVENTORY`: every sec 6.3
dimension is declared as data, and any dimension no implementation collects yet
is *emitted* as an UNKNOWN check rather than silently left out of the roll-up.
Omitting them is what let a machine with unknown disk, unknown provider health,
an unknown owned-process count and an unknown universe age render HEALTHY.

Disk/storage, owned process and thread counts, universe and market-data
freshness, runtime profile, writer coordination and provider request /
cache-hit / throttle / failure counters are now *measured* here.  The provider
counters come from :mod:`diagnostics.provider_counters` via the newest run
manifest's ``provider.*`` counters; until the first scan on an instrumented
build has written a manifest carrying ``provider.captured``, the dimension
honestly reports UNKNOWN rather than inventing a number.

The shadow rows are graded on the RAW LOGS, not on the writers' sidecars. This
module used to read ``spy_state_shadow_status.json`` and
``greatness_candidates.json`` only - each written by the very process it
described - so a truncated last line after a crash, a half-written row, or a
schema-drifted record passed green by construction.
:mod:`diagnostics.shadow_log_audit` streams the actual JSONL evidence, the
sidecar claims are reconciled against the rows that exist, and every shadow
check now carries an explicit ``promotable`` verdict so plan.md sec 7's
evidence floors cannot be claimed over a damaged log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import writer_health
import writer_role
from diagnostics import shadow_log_audit
from diagnostics.run_manifest import load_recent_manifests
from job_ledger import DEFAULT_RETRY_BUDGET, JobLedger
from market_session import get_market_local_timezone, get_market_session_window, normalize_market_local_datetime
from project_paths import (
    AUTOPILOT_REPORT_FILE,
    AUTOPILOT_STATE_FILE,
    CACHE_DIR,
    INDUSTRY_BOARD_STATE_FILE,
    JOURNAL_DB_FILE,
    MASTER_AVWAP_DAILY_BARS_DIR,
    UNIVERSE_ALL_FILE,
    UNIVERSE_LONGS_FILE,
    UNIVERSE_SHORTS_FILE,
    get_diagnostics_dir,
    get_local_setting,
)


AUDIT_SCHEMA = "operations_audit_v2"
AWAY_REPORT_DEGRADED_AFTER_MINUTES = 75.0
AWAY_REPORT_UNHEALTHY_AFTER_MINUTES = 120.0
#: With Auto Pilot switched off nothing refreshes the report, so a retained
#: report is evidence of the last publish - but not forever.
AWAY_REPORT_RETAINED_DEGRADED_AFTER_MINUTES = 24 * 60.0

#: Free space on the volume that carries the diagnostics root. Below the
#: degraded floor an unattended day is at risk; below the unhealthy floor the
#: next scan's artifacts may simply fail to land.
DISK_FREE_DEGRADED_GB = 5.0
DISK_FREE_UNHEALTHY_GB = 1.0
#: Nothing prunes the shadow logs yet, so the footprint is reported and bounded
#: rather than silently growing. The audit MEASURES the live files every run;
#: no current size is written down here, because the two that used to be - one
#: in this comment, one in the R6 packet - were both true when written and
#: stale by growth within weeks. Dated, for scale only:
#: ``technical_integrity_events.jsonl`` measured
#: 370 MB / 318,040 rows / 25 sessions on 2026-08-17, and rotation was DECLINED
#: that day (plan.md item 6(b)) - the ledger is a warehouse bronze source whose
#: retention unlocks only after verified ingest. What this module owes is a
#: read-only measurement, which is what `jsonl_ledgers` below reports.
DIAGNOSTICS_FOOTPRINT_DEGRADED_MB = 1024.0
SINGLE_ARTIFACT_DEGRADED_MB = 100.0

#: The self-built universe is folded into every master scan and *nothing*
#: schedules a rebuild, so it is graded in days: a silently months-old universe
#: is the failure mode this check exists to make visible.
UNIVERSE_DEGRADED_AFTER_DAYS = 7.0
UNIVERSE_UNHEALTHY_AFTER_DAYS = 30.0
#: Daily-bar store freshness, in *calendar* days against the market date, so a
#: normal weekend or a long holiday weekend is not reported as staleness.
MARKET_DATA_DEGRADED_AFTER_DAYS = 4
MARKET_DATA_UNHEALTHY_AFTER_DAYS = 10

STATUS_HEALTHY = "healthy"
STATUS_UNKNOWN = "unknown"
STATUS_DEGRADED = "degraded"
STATUS_UNHEALTHY = "unhealthy"

#: Deterministic precedence: UNHEALTHY > DEGRADED > UNKNOWN > HEALTHY. UNKNOWN
#: sits above HEALTHY (absent evidence can never be green) and below DEGRADED
#: (a measured problem is more actionable than an unmeasured dimension).
_STATUS_ORDER = {STATUS_HEALTHY: 0, STATUS_UNKNOWN: 1, STATUS_DEGRADED: 2, STATUS_UNHEALTHY: 3}
STATUS_VALUES = tuple(sorted(_STATUS_ORDER, key=lambda name: _STATUS_ORDER[name]))

_ARTIFACT_OK = "ok"
_ARTIFACT_MISSING = "missing"
_ARTIFACT_UNREADABLE = "unreadable"
_ARTIFACT_CORRUPT = "corrupt"
#: Absent evidence is uncertainty; evidence we hold but cannot parse is a
#: failure. They demand different operator actions, so they are never merged.
_ARTIFACT_STATUS = {
    _ARTIFACT_MISSING: STATUS_UNKNOWN,
    _ARTIFACT_UNREADABLE: STATUS_UNHEALTHY,
    _ARTIFACT_CORRUPT: STATUS_UNHEALTHY,
}


def worst_status(statuses: Iterable[str], default: str = STATUS_UNKNOWN) -> str:
    """Roll statuses up by the documented precedence."""
    known = [str(status) for status in statuses if str(status) in _STATUS_ORDER]
    if not known:
        return default
    return max(known, key=lambda item: _STATUS_ORDER[item])


@dataclass(frozen=True)
class RequiredCheck:
    """One dimension the Health page must expose (plan.md sec 6.3)."""

    id: str
    label: str
    requirement: str
    covered_by: tuple[str, ...] = ()


#: The complete sec 6.3 bullet list, in plan order. ``covered_by`` names the
#: emitted check(s) that measure the dimension; an empty tuple means nothing
#: measures it yet, and the audit emits an explicit UNKNOWN row for it.
REQUIRED_CHECK_INVENTORY: tuple[RequiredCheck, ...] = (
    RequiredCheck(
        "runtime_profile",
        "Runtime profile and machine identity",
        "runtime profile and machine identity",
        ("runtime_profile",),
    ),
    RequiredCheck("heartbeat_age", "Heartbeat age", "heartbeat age", ("heartbeat",)),
    RequiredCheck("current_next_job", "Current and next job", "current and next job", ("heartbeat",)),
    RequiredCheck(
        "job_attempts_and_verified_success",
        "Last attempt and last verified success per job/export",
        "last attempt and last verified success per job/export",
        ("job_ledger", "away_report"),
    ),
    RequiredCheck(
        "job_failures_and_retries",
        "Job-ledger failures and exhausted retries",
        "job-ledger failures and exhausted retries",
        ("job_ledger",),
    ),
    RequiredCheck(
        "owned_process_counts",
        "Owned process/thread counts",
        "owned process/thread counts",
        ("owned_process_counts",),
    ),
    RequiredCheck(
        "writer_lease",
        "Writer-lease holder and expiry",
        "writer-lease holder and expiry",
        ("writer_lease",),
    ),
    RequiredCheck(
        "report_freshness",
        "Report freshness and verification state",
        "report freshness and verification state",
        ("away_report",),
    ),
    RequiredCheck(
        "provider_counters",
        "Provider request, cache-hit, throttling, and failure counts",
        "provider request, cache-hit, throttling, and failure counts",
        ("provider_counters",),
    ),
    RequiredCheck(
        "universe_and_market_data_freshness",
        "Universe and market-data freshness",
        "universe and market-data freshness",
        ("universe_and_market_data_freshness",),
    ),
    RequiredCheck(
        "scan_manifest_and_phase_timings",
        "Most recent scan manifest and phase timings",
        "most recent scan manifest and phase timings",
        ("run_manifest",),
    ),
    RequiredCheck(
        "shadow_engine_coverage",
        "SPY and Greatness shadow engines (versions, last evaluations, coverage, errors)",
        "SPY and Greatness shadow engine versions, last evaluations, coverage, and errors",
        ("spy_shadow", "greatness_shadow"),
    ),
    RequiredCheck(
        "disk_storage_warnings",
        "Disk/storage warnings",
        "disk/storage warnings",
        ("disk_storage_warnings",),
    ),
)

_PLAN_SOURCE = Path("plan.md#sec-6.3")


def _read_json_artifact(path: Path) -> tuple[str, dict[str, Any] | None, str]:
    """Read a JSON object, keeping ABSENT and CORRUPT distinguishable.

    Returns ``(state, payload, detail)``. This module used to collapse
    ``OSError`` and ``JSONDecodeError`` into a single ``None``, which made "no
    evidence has been written yet" (UNKNOWN, wait or start the writer) look
    exactly like "the evidence on disk is broken" (UNHEALTHY, go repair it).
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _ARTIFACT_MISSING, None, f"{path.name} has not been written."
    except OSError as exc:
        return _ARTIFACT_UNREADABLE, None, f"{path.name} could not be read: {exc}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _ARTIFACT_CORRUPT, None, f"{path.name} is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return _ARTIFACT_CORRUPT, None, f"{path.name} is not a JSON object."
    return _ARTIFACT_OK, payload, ""


#: One ISO-8601 parser for the audit and the shadow-log validator, so "this
#: timestamp is unusable" can never mean two different things in one payload.
_parse_timestamp = shadow_log_audit.parse_timestamp


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
    """Grade an age. An age that cannot be established is UNKNOWN, not red."""
    if age is None:
        return STATUS_UNKNOWN
    if age > unhealthy_minutes:
        return STATUS_UNHEALTHY
    if age > healthy_minutes:
        return STATUS_DEGRADED
    return STATUS_HEALTHY


def _artifact_gap_check(
    check_id: str,
    label: str,
    state: str,
    detail: str,
    path: Path,
    *,
    missing_summary: str = "",
) -> dict[str, Any]:
    """The UNKNOWN/UNHEALTHY row for an artifact that could not be read."""
    status = _ARTIFACT_STATUS.get(state, STATUS_UNKNOWN)
    summary = missing_summary if (state == _ARTIFACT_MISSING and missing_summary) else detail
    return _check(
        check_id,
        label,
        status,
        summary,
        source=path,
        details={"artifact_state": state, "detail": detail},
    )


def _reference_moment(now: datetime, session, market_phase: str) -> datetime:
    """The moment freshness is measured against.

    After the close nothing is evaluating any more, so an artifact written at
    12:55 is not getting staler at 20:00 - it is exactly as fresh (or stale) as
    it was when work stopped. Measuring post-market artifacts against ``now``
    would paint every evening yellow; measuring them against nothing at all
    (the old ``status = "healthy"`` branch) painted every evening green.
    """
    if market_phase == "post_market":
        return min(now, session.close_local)
    return now


def _heartbeat_check(path: Path, now: datetime, local_tz) -> dict[str, Any]:
    state, payload, detail = _read_json_artifact(path)
    if payload is None:
        return _artifact_gap_check(
            "heartbeat",
            "Runtime heartbeat",
            state,
            detail,
            path,
            missing_summary="No heartbeat has been written; this runtime's liveness is unknown.",
        )
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


#: The AI batch layer's store root, resolved WITHOUT importing `ai_jobs`.
#:
#: `ai_jobs` is in PACKAGES_NOT_IN_THE_BUNDLE: the frozen desk genuinely does
#: not contain it, because its only entry point is a scheduled CLI run from the
#: repo checkout. Importing it here would give a row that works in the checkout
#: and dies in the exe - and this audit renders inside System Health, which is
#: frozen. So the two-source rule (env override, then the local setting) is
#: mirrored here by path alone, and
#: test_ai_jobs_store_resolution_matches_the_batch_layer keeps the two in step.
AI_STORE_DIR_ENV = "TRADINGBOTV3_AI_STORE_DIR"
AI_STORE_DIR_SETTING = "ai_store_dir"
AI_JOB_LEDGER_NAME = "ai_job_ledger.jsonl"


def _ai_store_dir() -> Path | None:
    """The configured AI store root, or None when the batch layer is off."""
    raw = str(os.environ.get(AI_STORE_DIR_ENV) or "").strip()
    if not raw:
        value = get_local_setting(AI_STORE_DIR_SETTING)
        raw = value.strip() if isinstance(value, str) else ""
    if not raw:
        return None
    try:
        return Path(raw).expanduser()
    except (OSError, ValueError):
        return None


def _questrade_chain_check(now: datetime, db_path: Path) -> dict[str, Any]:
    """Is the Questrade credential chain alive? (AI-P4)

    A broken chain is invisible everywhere else on the desk: the nightly slot
    records `failed`, the coverage grid fills with red for a broker the trader
    is not looking at, and the journal simply stops gaining Questrade trades.
    On 2026-08-24 that had been true since 2026-08-19 - 0 of 142 days covered,
    one whole broker including a TFSA missing - and it was found by opening the
    SQLite database by hand.

    `not_configured` is HEALTHY on purpose, and the distinction from `unknown`
    is the same one `_ai_jobs_check` makes: a machine with no Questrade token
    has been measured, and the answer is that this broker was never asked for.
    A database that could not be read has NOT been measured.
    """
    try:
        import journal_health
    except Exception:  # noqa: BLE001
        return _check(
            "questrade_chain",
            "Questrade credential chain",
            STATUS_UNKNOWN,
            "The Questrade chain check could not be loaded, so the chain's "
            "health is unmeasured - which is not the same as fine.",
            source=Path(__file__),
        )

    # The path is INJECTED, never resolved here. Every other check in this
    # audit takes its store as a parameter for the same reason: a test that
    # hands this function a sandbox must not have it read the trader's real
    # journal instead (test_capture_readiness_checks_reach_the_audit_and_
    # never_read_the_shared_home, which caught exactly that here).
    verdict = journal_health.questrade_chain_health(now=now, db_path=db_path)
    status = {
        journal_health.STATE_OK: STATUS_HEALTHY,
        journal_health.STATE_NOT_CONFIGURED: STATUS_HEALTHY,
        journal_health.STATE_STALE: STATUS_DEGRADED,
        journal_health.STATE_DEAD: STATUS_UNHEALTHY,
    }.get(verdict["state"], STATUS_UNKNOWN)
    summary = verdict["headline"]
    if verdict["action"]:
        summary = f"{summary} {verdict['action']}"
    return _check(
        "questrade_chain",
        "Questrade credential chain",
        status,
        summary,
        source=db_path,
        updated_at=str(verdict.get("last_refresh_at") or ""),
        details=dict(verdict),
    )


def _ai_jobs_check(now: datetime, local_tz) -> dict[str, Any]:
    """Read-only visibility for the overnight AI batch layer.

    The layer runs entirely outside the GUI (a scheduled task against the repo
    checkout), which is exactly why it needed a row: until now a night where it
    never ran and a night where it had nothing to do looked identical from the
    desk - the failure mode `run_ai_jobs.ps1` was written to end, reproduced one
    level up.

    An UNSET store is reported HEALTHY, not UNKNOWN, and the distinction is the
    point: UNKNOWN means "could not measure". A machine with no `ai_store_dir`
    has been measured, and the answer is that the layer is deliberately off.
    """
    store = _ai_store_dir()
    if store is None:
        return _check(
            "ai_jobs",
            "AI batch layer",
            STATUS_HEALTHY,
            "Not configured on this machine (no ai_store_dir) - the batch layer is off by choice.",
            # There is no artifact to point at, and an empty path would claim
            # there was. The source of this answer is the configuration lookup
            # in this module, so that is what it names.
            source=Path(__file__),
            details={"configured": False},
        )

    path = store / "logs" / AI_JOB_LEDGER_NAME
    if not path.exists():
        return _check(
            "ai_jobs",
            "AI batch layer",
            STATUS_UNKNOWN,
            "AI store is configured but no job ledger has been written yet.",
            source=path,
            details={"configured": True},
        )

    from diagnostics.artifact_io import read_jsonl

    # skip_bad is the default and is wanted here: a partially flushed tail line
    # written while the scheduled task was mid-run must not make the whole
    # ledger unreadable to a read-only viewer.
    rows = read_jsonl(path)
    if not rows:
        return _check(
            "ai_jobs",
            "AI batch layer",
            STATUS_UNKNOWN,
            "The AI job ledger exists but holds no readable rows.",
            source=path,
            details={"configured": True, "row_count": 0},
        )

    last = rows[-1]
    updated_at = str(last.get("ts") or last.get("timestamp") or "")
    age = _age_minutes(updated_at, now, local_tz)
    statuses = Counter(str(row.get("status") or "") for row in rows)
    failed = statuses.get("failed", 0)
    degraded = statuses.get("degraded", 0)

    # Freshness is deliberately generous: the layer is nightly, so a run that
    # is hours old is normal and only a run that is more than a day-and-a-half
    # old suggests the schedule itself stopped firing.
    if failed:
        status = STATUS_UNHEALTHY
        summary = f"{failed} AI job(s) failed in the ledger; last row {updated_at or 'undated'}."
    elif degraded:
        status = STATUS_DEGRADED
        summary = f"{degraded} AI job(s) degraded in the ledger; last row {updated_at or 'undated'}."
    else:
        status = _freshness_status(age, healthy_minutes=36 * 60, unhealthy_minutes=72 * 60)
        summary = (
            f"Last AI job row {updated_at or 'undated'}"
            + (f" ({age / 60:.1f}h ago)." if age is not None else " (age unknown).")
        )

    return _check(
        "ai_jobs",
        "AI batch layer",
        status,
        summary,
        source=path,
        updated_at=updated_at,
        details={
            "configured": True,
            "row_count": len(rows),
            "status_counts": dict(statuses),
            "last_job": str(last.get("job") or ""),
            "last_status": str(last.get("status") or ""),
            "age_hours": round(age / 60, 2) if age is not None else None,
        },
    )


def _ledger_check(path: Path, market_date: str, now: datetime, local_tz, market_phase: str) -> tuple[dict[str, Any], list[dict]]:
    if not path.exists():
        # No ledger before the first scheduled slot of the day (or on a closed
        # day) is absent evidence, not a failure; during the session it is one.
        status = STATUS_UNKNOWN if market_phase in {"pre_market", "closed"} else STATUS_UNHEALTHY
        summary = (
            "No job ledger has been written yet today."
            if status == STATUS_UNKNOWN
            else "Job ledger is missing during a session in which jobs should have run."
        )
        return _check("job_ledger", "Scheduled jobs", status, summary, source=path), []
    ledger = JobLedger(path)
    jobs = ledger.jobs_for_date(market_date)
    state_counts = Counter(job.state for job in jobs)
    problems: list[str] = []
    # plan.md sec 6.3 asks for failures AND exhausted retries. The ledger has
    # tracked both since Phase 2.5 (attempt + a per-error-class budget behind
    # should_retry), but this row never read them, so a job that burned its
    # whole budget - nothing will retry it today - looked exactly like one that
    # failed once and is about to be tried again.
    exhausted: list[dict[str, Any]] = []
    retrying: list[dict[str, Any]] = []
    for job in jobs:
        if job.state in {"FAILED", "STALE"}:
            error_class = job.error_class or "unexpected"
            budget = int(
                DEFAULT_RETRY_BUDGET.get(error_class, DEFAULT_RETRY_BUDGET.get("unexpected", 0))
            )
            record = {
                "slot": job.slot,
                "job_type": job.job_type,
                "state": job.state,
                "attempt": int(job.attempt),
                "retry_budget": budget,
                "error_class": error_class,
                "error": job.error or "",
                "ended_at": job.ended_at or "",
            }
            if ledger.should_retry(job.key):
                retrying.append(record)
                retry_text = f"attempt {job.attempt} of {budget + 1}, retry available"
            else:
                exhausted.append(record)
                retry_text = f"attempt {job.attempt} of {budget + 1}, RETRIES EXHAUSTED"
            problems.append(
                f"{job.slot} {job.state.lower()}: "
                f"{job.error or error_class or 'unknown error'} ({retry_text})"
            )
        if job.state == "RUNNING":
            running_age = _age_minutes(job.started_at, now, local_tz)
            if running_age is None or running_age > 35.0:
                problems.append(f"{job.slot} running too long")

    completed = int(state_counts.get("COMPLETED", 0))
    active = int(state_counts.get("RUNNING", 0) + state_counts.get("QUEUED", 0))
    if problems:
        status = STATUS_UNHEALTHY
    elif completed or active:
        status = STATUS_HEALTHY
    elif market_phase in {"pre_market", "closed"}:
        status = STATUS_UNKNOWN
    else:
        status = STATUS_UNHEALTHY
        problems.append("No jobs recorded for the current market date")
    summary = f"{completed} completed, {active} active, {len(problems)} problem(s)."
    if exhausted:
        slots = ", ".join(str(item["slot"]) for item in exhausted)
        summary += f" {len(exhausted)} job(s) out of retries ({slots}): nothing will re-run them today."
    serialized = [asdict(job) for job in jobs]
    last_success = max(
        (job.ended_at for job in jobs if job.state == "COMPLETED" and job.ended_at),
        default="",
    )
    return (
        _check(
            "job_ledger",
            "Scheduled jobs",
            status,
            summary,
            source=path,
            updated_at=max((job.ended_at or job.started_at or job.scheduled_at for job in jobs), default=""),
            details={
                "state_counts": dict(state_counts),
                "problems": problems,
                "job_count": len(jobs),
                "last_verified_success_at": last_success,
                "retry_exhausted": exhausted,
                "retry_available": retrying,
                "retry_exhausted_count": len(exhausted),
                "retry_budget": dict(DEFAULT_RETRY_BUDGET),
            },
        ),
        serialized,
    )


def _manifest_check(
    path: Path,
    market_date: str,
    now: datetime,
    local_tz,
    running_job: bool,
    market_phase: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifests = load_recent_manifests(path, limit=30)
    current = [item for item in manifests if str(item.get("started_at") or "")[:10] == market_date]
    latest = current[0] if current else (manifests[0] if manifests else {})
    # "No scan ran today" is not a stale scan. Grading the newest manifest of
    # any date on age alone capped this row at degraded, so a day on which the
    # scheduler never fired rendered yellow instead of red.
    before_first_scan = market_phase in {"pre_market", "closed"}
    if not current:
        newest_date = str(latest.get("started_at") or "")[:10] if latest else ""
        status = STATUS_UNKNOWN if before_first_scan else STATUS_UNHEALTHY
        if not latest:
            summary = f"No scan manifest has ever been written under {path.name}."
        else:
            summary = (
                f"No scan manifest for {market_date}; the newest manifest is from "
                f"{newest_date or 'an unknown date'}."
            )
            if not before_first_scan:
                summary += " No scan has run today."
        return (
            _check(
                "run_manifest",
                "Latest scan manifest",
                status,
                summary,
                source=path,
                updated_at=str(latest.get("ended_at") or latest.get("started_at") or "") if latest else "",
                details={
                    "market_date": market_date,
                    "manifest_for_market_date": False,
                    "newest_manifest_date": newest_date,
                    "newest_run_id": latest.get("run_id") or "" if latest else "",
                    "manifest_count": len(manifests),
                },
            ),
            latest,
        )
    ended = latest.get("ended_at") or latest.get("started_at") or ""
    age = _age_minutes(ended, now, local_tz)
    latest_status = str(latest.get("status") or "").strip().lower()
    counters = latest.get("counters") if isinstance(latest.get("counters"), dict) else {}
    outputs = latest.get("outputs") if isinstance(latest.get("outputs"), dict) else {}
    tracker_write_requested = counters.get("update_setup_tracker") is True
    tracker_updated = counters.get("setup_tracker_updated") is True
    tracker_skip_reason = str(outputs.get("setup_tracker_skip_reason") or "").strip()
    if latest_status not in {"ok", "success", "completed"}:
        status = STATUS_UNHEALTHY
    elif age is None:
        status = STATUS_UNKNOWN
    elif age > (240.0 if running_job else 180.0):
        status = STATUS_DEGRADED
    else:
        status = STATUS_HEALTHY
    if status == STATUS_HEALTHY and tracker_write_requested and not tracker_updated:
        status = STATUS_DEGRADED
    summary = (
        f"{latest.get('job_type') or 'scan'} {latest_status or 'unknown'}; "
        f"{float(latest.get('total_seconds') or 0.0) / 60.0:.1f}m; "
        f"{('unknown age' if age is None else f'{age:.1f}m old')}."
    )
    if tracker_write_requested and not tracker_updated:
        summary += " Requested setup-tracker write skipped."
    # Per-phase timings are recorded by ManifestRecorder.record_phase on every
    # run and were then dropped here: only one aggregate total ever reached a
    # reader, so "which phase got slower" (sec 6.3 bullet 11) was unanswerable
    # without opening the manifest file by hand.
    phases = _phase_rows(latest)
    slowest = next((row for row in phases if not row["aggregate"]), {})
    if slowest:
        summary += f" Slowest phase {slowest['label']} {slowest['seconds'] / 60.0:.1f}m."
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
                "manifest_for_market_date": True,
                "error": latest.get("error") or "",
                "total_seconds": latest.get("total_seconds"),
                "counters": counters,
                "setup_tracker_skip_reason": tracker_skip_reason or "No skip reason recorded.",
                "age_minutes": round(age, 2) if age is not None else None,
                "phase_count": len(phases),
                "slowest_phase": slowest,
                "phases": phases,
            },
        ),
        latest,
    )


def _phase_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-phase timings from a manifest, slowest first, with their share.

    The scanner records both disjoint phases and one aggregate ``TOTAL`` phase
    (see :meth:`ManifestRecorder.to_dict`), so the aggregate is excluded from
    the share denominator rather than double-counted.
    """
    raw = manifest.get("phases") if isinstance(manifest, dict) else None
    rows: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return rows
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "")
        try:
            seconds = float(entry.get("seconds") or 0.0)
        except (TypeError, ValueError):
            seconds = 0.0
        rows.append(
            {"label": label, "seconds": round(seconds, 3), "aggregate": label.upper().startswith("TOTAL")}
        )
    denominator = sum(row["seconds"] for row in rows if not row["aggregate"])
    for row in rows:
        row["share_pct"] = (
            round(100.0 * row["seconds"] / denominator, 1) if denominator > 0 and not row["aggregate"] else None
        )
    rows.sort(key=lambda row: (row["aggregate"], -row["seconds"]))
    return rows


def _away_report_check(
    report_path: Path,
    state_path: Path,
    now: datetime,
    local_tz,
    market_phase: str,
) -> dict[str, Any]:
    state_artifact, state_payload, state_detail = _read_json_artifact(state_path)
    if state_artifact in {_ARTIFACT_UNREADABLE, _ARTIFACT_CORRUPT}:
        # Without the Auto Pilot state we cannot say whether a report is even
        # expected, and the evidence we hold is broken rather than absent.
        return _check(
            "away_report",
            "Away report",
            STATUS_UNHEALTHY,
            f"Auto Pilot state is unreadable, so Away expectations cannot be established: {state_detail}",
            source=state_path,
            details={"artifact_state": state_artifact, "detail": state_detail},
        )
    state = state_payload or {}
    enabled = bool(state.get("enabled"))
    if not report_path.exists():
        status = STATUS_UNHEALTHY if enabled else STATUS_UNKNOWN
        return _check(
            "away_report",
            "Away report",
            status,
            "Report is missing while Auto Pilot is enabled."
            if enabled
            else "No report has been published and Auto Pilot is disabled; freshness is unknown.",
            source=report_path,
            details={"enabled": enabled, "autopilot_state": state_artifact},
        )

    metadata_path = report_path.with_suffix(report_path.suffix + ".meta.json")
    metadata_artifact, metadata, metadata_detail = _read_json_artifact(metadata_path)
    if metadata_artifact in {_ARTIFACT_UNREADABLE, _ARTIFACT_CORRUPT}:
        return _check(
            "away_report",
            "Away report",
            STATUS_UNHEALTHY,
            f"Verified-publication metadata is unreadable: {metadata_detail}",
            source=metadata_path,
            details={"enabled": enabled, "artifact_state": metadata_artifact, "detail": metadata_detail},
        )
    verified_at = metadata.get("verified_at") if isinstance(metadata, dict) else None
    if verified_at is None:
        try:
            verified_at = datetime.fromtimestamp(report_path.stat().st_mtime).isoformat()
        except OSError:
            verified_at = ""
    age = _age_minutes(verified_at, now, local_tz)
    hash_ok = None
    if isinstance(metadata, dict) and metadata.get("sha256"):
        try:
            actual = hashlib.sha256(report_path.read_bytes()).hexdigest()
            hash_ok = actual == str(metadata.get("sha256"))
        except OSError:
            hash_ok = False

    if hash_ok is False:
        status = STATUS_UNHEALTHY
        summary = "Verified-publication hash no longer matches the report."
    elif not isinstance(metadata, dict):
        status = STATUS_DEGRADED
        summary = "Report exists, but verified-publication metadata is missing."
    elif age is None:
        status = STATUS_UNKNOWN
        summary = "Report exists, but its verified freshness cannot be established."
    elif not enabled:
        # Auto Pilot off means nothing is refreshing this report, so a recent
        # one is still evidence - but it ages out. It used to read healthy at
        # any age at all, which is exactly the green-on-unknown the plan bans.
        prefix = (
            "No Auto Pilot state recorded"
            if state_artifact == _ARTIFACT_MISSING
            else "Auto Pilot disabled"
        )
        if age > AWAY_REPORT_RETAINED_DEGRADED_AFTER_MINUTES:
            status = STATUS_DEGRADED
            summary = (
                f"{prefix}; the retained verified report is {age / 60.0:.1f}h old "
                "and is no longer current evidence."
            )
        else:
            status = STATUS_HEALTHY
            summary = f"{prefix}; last verified Away report is {age:.1f}m old."
    elif market_phase == "regular" and age > AWAY_REPORT_UNHEALTHY_AFTER_MINUTES:
        status = STATUS_UNHEALTHY
        summary = f"Report is {age:.1f}m old during the market session."
    elif market_phase == "regular" and age > AWAY_REPORT_DEGRADED_AFTER_MINUTES:
        status = STATUS_DEGRADED
        summary = f"Report is {age:.1f}m old during the market session."
    elif age > AWAY_REPORT_RETAINED_DEGRADED_AFTER_MINUTES:
        status = STATUS_DEGRADED
        summary = f"Last verified report is {age / 60.0:.1f}h old."
    else:
        status = STATUS_HEALTHY
        summary = f"Verified report is {age:.1f}m old."
    return _check(
        "away_report",
        "Away report",
        status,
        summary,
        source=report_path,
        updated_at=str(verified_at or ""),
        details={
            "enabled": enabled,
            "autopilot_state": state_artifact,
            "profile": state.get("profile") or "",
            "metadata_path": str(metadata_path),
            "hash_verified": hash_ok,
            "age_minutes": round(age, 2) if age is not None else None,
            "holder": metadata.get("holder") if isinstance(metadata, dict) else "",
            "lease_expires_at": metadata.get("lease_expires_at") if isinstance(metadata, dict) else "",
        },
    )


def _industry_board_check(path: Path, now: datetime, local_tz, market_phase: str) -> dict[str, Any]:
    state, payload, detail = _read_json_artifact(path)
    if payload is None:
        return _artifact_gap_check(
            "industry_board",
            "Industry Board",
            state,
            detail,
            path,
            missing_summary="No verified Industry Board refresh state has been written yet.",
        )
    last_success = payload.get("last_success_at") or ""
    age = _age_minutes(last_success, now, local_tz)
    failed = str(payload.get("status") or "").lower() == "failed"
    if not last_success and failed:
        status = STATUS_UNHEALTHY
        summary = "Industry Board refresh failed and no good snapshot has ever been produced."
    elif not last_success:
        status = STATUS_UNKNOWN
        summary = "Industry Board has never recorded a successful refresh."
    elif age is None:
        status = STATUS_UNKNOWN
        summary = "Industry Board reports a success, but its age cannot be established."
    elif failed:
        status = STATUS_DEGRADED
        summary = (
            "Latest Industry Board refresh failed; the last good snapshot remains active."
        )
    elif market_phase == "regular" and age > 120.0:
        status = STATUS_UNHEALTHY
        summary = "Industry Board is stale during the market session."
    elif age > 65.0:
        status = STATUS_DEGRADED
        summary = "Industry Board is older than the hourly freshness target."
    else:
        status = STATUS_HEALTHY
        summary = (
            f"Snapshot {payload.get('snapshot_id') or '?'}; "
            f"{payload.get('sector_count', 0)} sectors / "
            f"{payload.get('industry_count', 0)} industries; {age:.1f}m old."
        )
    return _check(
        "industry_board",
        "Industry Board",
        status,
        summary,
        source=path,
        updated_at=str(last_success),
        details={
            "last_attempt_at": payload.get("last_attempt_at") or "",
            "last_success_at": last_success,
            "snapshot_id": payload.get("snapshot_id") or "",
            "sector_count": int(payload.get("sector_count", 0) or 0),
            "industry_count": int(payload.get("industry_count", 0) or 0),
            "symbol_count": int(payload.get("symbol_count", 0) or 0),
            "last_error": payload.get("error") or "",
            "age_minutes": round(age, 2) if age is not None else None,
        },
    )


def _shadow_check(
    *,
    check_id: str,
    label: str,
    path: Path,
    log_path: Path,
    log_profile: shadow_log_audit.ShadowLogProfile,
    claims_builder,
    now: datetime,
    local_tz,
    market_date: str,
    market_phase: str,
    session,
    coverage_key: str | None = None,
) -> dict[str, Any]:
    """Grade a shadow engine on its RAW LOG first, then on its own sidecar.

    ``path`` is the writer-maintained sidecar; ``log_path`` is the append-only
    JSONL the engine actually produces. Reading only the sidecar was the defect
    this packet exists to fix: the writing process was the sole witness to its
    own output, so a truncated tail, an interleaved half-record or a drifted
    schema could never turn the tile any colour but green.

    The log is therefore ALWAYS streamed - even when the sidecar is missing or
    corrupt, because a broken sidecar is exactly when the raw evidence matters
    most - and the two verdicts are rolled up with :func:`worst_status`.
    """
    state, payload, detail = _read_json_artifact(path)
    coverage_source = payload or {}
    coverage = coverage_source.get(coverage_key) if coverage_key else coverage_source
    coverage = coverage if isinstance(coverage, dict) else {}
    session_date = str(coverage.get("session_date") or coverage_source.get("session_date") or "")

    log = shadow_log_audit.audit_shadow_log(
        log_path,
        log_profile,
        now=now,
        local_tz=local_tz,
        market_date=market_date,
        # Reconcile against the session the SIDECAR is describing: its counters
        # reset per session, so comparing them to today's rows on a stale
        # sidecar would invent a discrepancy that is really just staleness.
        reconcile_session_date=session_date or market_date,
        claims=claims_builder(coverage),
    )
    session_progress = shadow_log_audit.audit_shadow_session_progress(
        log_path,
        log_profile,
    )
    log_evidence = {
        "log_path": str(log_path),
        "log_status": log["status"],
        "promotable": log["promotable"],
        "non_promotable_reasons": list(log["non_promotable_reasons"]),
        "log_notes": list(log["notes"]),
        "promotion_note": log["promotion_note"],
        "log_scan": log["scan"],
        "sidecar_reconciliation": log["reconciliation"],
        "sidecar_schema": str(coverage_source.get("schema") or ""),
        "sidecar_state": state,
        "session_progress": session_progress,
    }

    if payload is None:
        gap = _artifact_gap_check(
            check_id,
            label,
            state,
            detail,
            path,
            missing_summary="No shadow-coverage store has been written; coverage is unknown.",
        )
        gap["status"] = worst_status([gap["status"], log["status"]])
        gap["summary"] = f"{gap['summary']} {log['summary']}"
        gap["details"].update(log_evidence)
        # A sidecar that cannot be read can never be reconciled against the log,
        # so the evidence is not promotable regardless of the log's own state.
        gap["details"]["promotable"] = False
        gap["details"]["non_promotable_reasons"] = [
            f"The coverage sidecar {path.name} is {state}: the log cannot be reconciled "
            "against any self-report.",
            *log["non_promotable_reasons"],
        ]
        return gap

    evaluations = int(coverage.get("evaluations", 0) or 0)
    errors = int(coverage.get("errors", 0) or 0)
    last_eval = coverage.get("last_evaluation_at") or coverage_source.get("updated_at") or ""
    age = _age_minutes(last_eval, now, local_tz)
    # Off-hours freshness is measured to the close, not to "now" (see
    # _reference_moment): after the bell the shadow is not supposed to advance.
    effective_age = _age_minutes(last_eval, _reference_moment(now, session, market_phase), local_tz)
    if errors:
        status = STATUS_UNHEALTHY
    elif market_phase == "regular":
        status = _freshness_status(age, 20.0, 45.0)
        if session_date != market_date or evaluations <= 0:
            status = STATUS_UNHEALTHY
    elif session_date != market_date or evaluations <= 0:
        # No evaluations recorded for this market date is absent evidence, not
        # a verdict either way - and it must never be the "healthy" branch.
        status = STATUS_UNKNOWN
    elif market_phase == "post_market":
        status = _freshness_status(effective_age, 20.0, 45.0)
    else:
        status = _freshness_status(effective_age, 60.0, 240.0)
    bars = int(coverage.get("bars_consumed", coverage.get("usable_evaluations", 0)) or 0)
    summary = f"{evaluations} evaluations, {bars} usable bars/evaluations, {errors} errors."
    if status == STATUS_UNKNOWN and (session_date != market_date or evaluations <= 0):
        summary = (
            f"No evaluations recorded for {market_date} "
            f"(last session on record: {session_date or 'none'}); coverage is unknown."
        )
    elif age is None:
        summary += " Last evaluation time is unknown."
    else:
        summary += f" Last evaluation {age:.0f}m ago."
    summary = f"{summary} {log['summary']}"
    if not log["promotable"]:
        summary += " NOT PROMOTABLE."

    status = worst_status([status, log["status"]])
    details = dict(coverage)
    details.update(
        {
            "session_date": session_date,
            "engine_version": coverage_source.get("engine_version") or "",
            "config_hash": coverage_source.get("config_hash") or "",
            "timezone": coverage_source.get("timezone") or "",
            "candidate_count": len(coverage_source.get("candidates") or []),
            "age_minutes": round(age, 2) if age is not None else None,
            "age_minutes_to_session_reference": round(effective_age, 2) if effective_age is not None else None,
        }
    )
    details.update(log_evidence)

    # A persisted rollover failure means evidence recording is STUCK - the
    # writer is alive but every call is refusing to record until the failed
    # session rollover succeeds.  That is a different fact from "no shadow
    # event occurred" and must never be mistaken for it (or for health).
    rollover_failure = coverage_source.get("rollover_failure")
    if isinstance(rollover_failure, dict) and rollover_failure:
        status = STATUS_UNHEALTHY
        failed_at = str(rollover_failure.get("failed_at") or "unknown time")
        error = str(rollover_failure.get("error") or "unknown error")
        summary = (
            f"Shadow session rollover FAILED at {failed_at} "
            f"({rollover_failure.get('error_type') or 'Exception'}: {error}). "
            "Evidence recording is paused and retries each call; this is NOT "
            f"'no shadow event occurred'. {summary}"
        )
        details["rollover_failure"] = dict(rollover_failure)
        details["promotable"] = False
        details["non_promotable_reasons"] = [
            "A session rollover failed and shadow recording is paused; the "
            "evidence stream has a gap of unknown length.",
            *details.get("non_promotable_reasons", []),
        ]
    return _check(check_id, label, status, summary, source=path, updated_at=str(last_eval), details=details)


def _registry_check(path: Path, now: datetime, local_tz, market_phase: str, session) -> dict[str, Any]:
    state, payload, detail = _read_json_artifact(path)
    if payload is None:
        return _artifact_gap_check(
            "candidate_registry",
            "Candidate registry",
            state,
            detail,
            path,
            missing_summary="No candidate registry has been written; the watchlist state is unknown.",
        )
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
    effective_age = _age_minutes(updated_at, _reference_moment(now, session, market_phase), local_tz)
    if not updated_at:
        status = STATUS_UNKNOWN
    elif not active:
        status = STATUS_UNHEALTHY
    elif market_phase == "regular":
        status = _freshness_status(age, 90.0, 180.0)
    elif market_phase == "post_market":
        # Measured against the close, so an evening check reports how current
        # the registry was when work stopped.
        status = _freshness_status(effective_age, 90.0, 180.0)
    else:
        # Pre-market and closed days: the registry carries over from the last
        # session, so it is graded in sessions, not minutes - but it is still
        # graded. It used to be healthy at literally any age.
        status = _freshness_status(age, 24 * 60.0, 96 * 60.0)
    summary = f"Generation {int(payload.get('generation', 0) or 0)}; {len(active)} active candidates."
    if status == STATUS_UNKNOWN:
        summary += " Registry age cannot be established."
    elif age is not None and market_phase != "regular":
        summary += f" Last written {age / 60.0:.1f}h ago."
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
            "age_minutes_to_session_reference": round(effective_age, 2) if effective_age is not None else None,
        },
    )


def _writer_health_checks(path: Path, now: datetime, local_tz) -> list[dict[str, Any]]:
    """Runtime profile/identity and writer-lease rows, from ``writer_health``.

    The writer-coordination telemetry already carries every Layer 5 field, so
    this consumes :mod:`writer_health` rather than growing a second, partial
    sidecar. Only the clock is this module's: ``read_writer_health`` grades
    staleness against wall-clock now, while the audit grades everything against
    its own (injectable) moment.
    """
    state = writer_health.read_writer_health(path=path)
    reader_status = str(state.get("status") or "")
    labels = (
        ("runtime_profile", "Runtime profile and machine identity"),
        ("writer_lease", "Writer-lease holder and expiry"),
    )
    if reader_status == "missing":
        return [
            _check(
                check_id,
                label,
                STATUS_UNKNOWN,
                "No writer-health telemetry has been written on this machine yet.",
                source=path,
                details={"reader_status": reader_status, "error": state.get("error") or ""},
            )
            for check_id, label in labels
        ]
    if reader_status.startswith(("unreadable", "corrupt")):
        return [
            _check(
                check_id,
                label,
                STATUS_UNHEALTHY,
                f"Writer-health telemetry is {reader_status}: {state.get('error') or 'unusable'}",
                source=path,
                details={"reader_status": reader_status, "error": state.get("error") or ""},
            )
            for check_id, label in labels
        ]

    written_at = str(state.get("written_at") or "")
    age = _age_minutes(written_at, now, local_tz)
    stale = age is not None and age > writer_health.MAX_TELEMETRY_AGE_MINUTES
    machine = str(state.get("machine") or "")
    role = str(state.get("role") or "")
    designated = str(state.get("designated_writer") or "")
    read_only = bool(state.get("read_only"))
    read_only_reason = str(state.get("read_only_reason") or "")
    local_lock = state.get("local_lock") if isinstance(state.get("local_lock"), dict) else {}
    last_failure = state.get("last_failure") if isinstance(state.get("last_failure"), dict) else {}
    common = {
        "reader_status": reader_status,
        "machine": machine,
        "role": role,
        "designated_writer": designated,
        "read_only": read_only,
        "read_only_reason": read_only_reason,
        "config_source": state.get("config_source") or "",
        "pid": state.get("pid"),
        "instance_id": state.get("instance_id") or "",
        "holder_identity": state.get("holder_identity") or "",
        "local_lock": local_lock,
        "local_lock_held": bool(local_lock.get("held")),
        "local_mutex": local_lock.get("mutex") or "unknown",
        "local_file_lock": local_lock.get("file_lock") or "unknown",
        "abandoned_by_previous_owner": bool(local_lock.get("abandoned_by_previous_owner")),
        "last_failure": last_failure,
        "written_at": written_at,
        "age_minutes": round(age, 2) if age is not None else None,
        "telemetry_limit_minutes": writer_health.MAX_TELEMETRY_AGE_MINUTES,
    }

    # An unconfigured or misconfigured machine publishes nothing and says so in
    # the artifact; the audit must repeat that instead of rendering "role:
    # unconfigured" as a green row. A correctly configured read-only secondary
    # is a *working* state, so it stays green - but the summary always names it,
    # because "why is nothing being published here?" is the question this row
    # exists to answer.
    # The artifact is a RECORD OF A PAST PUBLISH, not the current configuration.
    # "Can this machine publish right now?" is a live question, so resolve it
    # live and let that govern. Without this, any stale artifact -- one written
    # by an earlier differently-configured run, by a drill, or by the smoke
    # check -- is reported as the machine's present role. That is worse than
    # UNKNOWN: it is a confident green row on a machine that will refuse to
    # publish, which is exactly what plan.md sec 6.3 forbids.
    try:
        live = writer_role.resolve_writer_role()
        live_role = live.role
        live_designated = live.designated_writer
        live_may_publish = bool(live.may_publish)
        live_reason = live.reason
        live_resolved = True
    except Exception as exc:  # pragma: no cover - resolver failure is pathological
        live_role = live_designated = ""
        live_reason = f"writer role could not be resolved: {exc}"
        live_may_publish = False
        live_resolved = False

    # A disagreement means the telemetry describes a machine state that is no
    # longer true. Surface it rather than silently preferring either side.
    role_disagrees = bool(live_resolved and role and live_role and live_role != role)
    common.update(
        {
            "live_role": live_role,
            "live_designated_writer": live_designated,
            "live_may_publish": live_may_publish,
            "live_role_resolved": live_resolved,
            "telemetry_role_disagrees_with_live": role_disagrees,
        }
    )

    misconfigured = (
        live_role in {"unconfigured", "misconfigured"}
        or not live_designated
        if live_resolved
        else (role in {"unconfigured", "misconfigured"} or not designated)
    )
    self_is_designated = bool(machine) and machine.strip().lower() == designated.strip().lower()
    if not live_resolved:
        profile_status = STATUS_UNKNOWN
        profile_summary = live_reason
    elif role_disagrees:
        profile_status = STATUS_DEGRADED
        profile_summary = (
            f"Writer telemetry says {role!r} but this machine currently resolves to "
            f"{live_role!r} (designated writer {live_designated or 'unset'}). The telemetry "
            f"is stale or was written by another run; the live role governs. "
            f"{'' if live_may_publish else live_reason}"
        ).strip()
    elif age is None:
        profile_status = STATUS_UNKNOWN
        profile_summary = "Writer telemetry has no usable timestamp, so its age cannot be established."
    elif not machine or not role:
        profile_status = STATUS_UNKNOWN
        profile_summary = "Writer telemetry does not identify this machine's role."
    elif misconfigured:
        profile_status = STATUS_DEGRADED
        profile_summary = (
            f"{machine} has no usable writer configuration (role {role!r}, designated writer "
            f"{designated or 'unset'}); shared publishing is refused. "
            f"{read_only_reason or 'Run scripts/writer_role.py on this machine.'}"
        )
    elif read_only and self_is_designated:
        profile_status = STATUS_DEGRADED
        profile_summary = (
            f"{machine} is the configured designated writer but is currently read-only: "
            f"{read_only_reason or 'no reason recorded'}."
        )
    elif stale:
        profile_status = STATUS_DEGRADED
        profile_summary = (
            f"{role} on {machine}; telemetry is {age:.0f}m old "
            f"(limit {writer_health.MAX_TELEMETRY_AGE_MINUTES}m)."
        )
    elif read_only:
        profile_status = STATUS_HEALTHY
        profile_summary = (
            f"{machine} is a read-only {role}; {designated} publishes. PID "
            f"{state.get('pid') or '?'}, instance {state.get('instance_id') or '?'}; {age:.0f}m old."
        )
    else:
        profile_status = STATUS_HEALTHY
        profile_summary = (
            f"{role} on {machine}; designated writer {designated or 'unset'}; PID "
            f"{state.get('pid') or '?'}, instance {state.get('instance_id') or '?'}; {age:.0f}m old."
        )
    if profile_status != STATUS_UNKNOWN and local_lock.get("abandoned_by_previous_owner"):
        profile_status = worst_status([profile_status, STATUS_DEGRADED])
        profile_summary += " The previous owner died holding the machine-local lock."

    # The live lease state, not the last publish's metadata. The Away-report row
    # reports holder/expiry as they were at the last successful publish, which
    # is the stale winner: on a machine that has since lost (or never held) the
    # lease it names a holder that is no longer current.
    holder = str(state.get("lease_holder") or "")
    expires_at = str(state.get("lease_expires_at") or "")
    expires_age = _age_minutes(expires_at, now, local_tz)
    override = state.get("emergency_override") if isinstance(state.get("emergency_override"), dict) else {}
    publication = (
        state.get("last_verified_publication")
        if isinstance(state.get("last_verified_publication"), dict)
        else {}
    )
    lease_details = dict(common)
    lease_details.update(
        {
            "lease_path": state.get("lease_path") or "",
            "lease_holder": holder,
            "lease_instance_id": state.get("lease_instance_id") or "",
            "lease_acquired_at": state.get("lease_acquired_at") or "",
            "lease_expires_at": expires_at,
            "last_renewal_at": state.get("last_renewal_at") or "",
            "last_blocked_at": state.get("last_blocked_at") or "",
            "fencing_generation": state.get("fencing_generation"),
            "emergency_override": override,
            "emergency_override_active": bool(override.get("active")),
            "emergency_override_expires_at": override.get("expires_at") or "",
            "last_verified_publication": publication,
            "last_verified_publication_holder": publication.get("holder") or "",
            "last_verified_publication_generation": publication.get("generation"),
        }
    )
    if not holder:
        lease_status = STATUS_UNKNOWN
        if read_only and designated and not self_is_designated:
            lease_summary = (
                f"This machine holds no lease (read-only secondary). {designated} is the configured "
                "writer, and its lease is not observable from here."
            )
        else:
            lease_summary = "No lease holder is recorded; which machine may publish is unknown."
    elif not expires_at or expires_age is None:
        lease_status = STATUS_UNKNOWN
        lease_summary = f"Lease holder {holder} is recorded, but its expiry is unknown."
    elif bool(override.get("active")):
        lease_status = STATUS_DEGRADED
        lease_summary = (
            f"Emergency override is active until {override.get('expires_at') or 'an unstated time'}; "
            f"lease holder {holder}."
        )
    elif expires_age > 0:
        lease_status = STATUS_DEGRADED
        lease_summary = f"Lease from {holder} expired {expires_age:.0f}m ago; no machine currently holds it."
    elif stale:
        lease_status = STATUS_DEGRADED
        lease_summary = f"Lease holder {holder} until {expires_at}, but the telemetry is {age:.0f}m old."
    else:
        lease_status = STATUS_HEALTHY
        lease_summary = (
            f"Lease held by {holder} until {expires_at} ({abs(expires_age):.0f}m remaining); "
            f"fencing generation {state.get('fencing_generation')}."
        )
    if publication.get("holder"):
        lease_summary += (
            f" Last verified publication by {publication.get('holder')} "
            f"(generation {publication.get('generation')})."
        )

    return [
        _check("runtime_profile", labels[0][1], profile_status, profile_summary, source=path, updated_at=written_at, details=common),
        _check("writer_lease", labels[1][1], lease_status, lease_summary, source=path, updated_at=written_at, details=lease_details),
    ]


#: The module that owns the scan children. Process ownership is per-process, so
#: the count is only meaningful when the audit runs *inside* the owning process.
_PROCESS_OWNER_MODULE = "ui.services.scan_service"
_BOUNCE_OWNER_MODULE = "ui.services.bounce_service"


def _runtime_process_snapshot() -> dict[str, Any] | None:
    """This process's owned-child/thread accounting, or ``None`` if not its process.

    Deliberately ``sys.modules`` rather than an import: importing the Qt scan
    service from the CLI would create a *fresh* registry that owns nothing and
    then report "0 children, all clear" about a GUI running in another process
    entirely. That is a fabricated green, so out-of-process the answer is
    UNKNOWN.
    """
    module = sys.modules.get(_PROCESS_OWNER_MODULE)
    getter = getattr(module, "owned_scan_process_snapshot", None) if module is not None else None
    if getter is None:
        # This row primarily answers scan-process ownership.  Bounce-only
        # telemetry must not turn an unmeasured scan count into a green zero.
        return None
    snapshot = dict(getter())
    bounce_module = sys.modules.get(_BOUNCE_OWNER_MODULE)
    bounce_getter = (
        getattr(bounce_module, "owned_bounce_thread_snapshot", None)
        if bounce_module is not None
        else None
    )
    if bounce_getter is not None:
        snapshot.update(dict(bounce_getter()))
    return snapshot


def _process_check(
    now: datetime,
    market_phase: str,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Owned child-process and worker-thread counts (plan.md sec 6.3 bullet 6).

    sec 6.1's after-session checklist asks two questions this answers: does the
    owned child-process count return to zero, and is any scanner or worker left
    orphaned. Strictly observational - it reads
    :func:`ui.services.scan_service.owned_scan_process_snapshot` and changes no
    ownership or reaping behavior.
    """
    source = Path(f"{_PROCESS_OWNER_MODULE.replace('.', '/')}.py")
    if snapshot is None:
        try:
            snapshot = _runtime_process_snapshot()
        except Exception as exc:
            return _check(
                "owned_process_counts",
                "Owned process/thread counts",
                STATUS_UNHEALTHY,
                f"Owned-process accounting failed: {exc}",
                source=source,
                details={"error": str(exc), "audit_pid": os.getpid()},
            )
    if snapshot is None:
        return _check(
            "owned_process_counts",
            "Owned process/thread counts",
            STATUS_UNKNOWN,
            (
                f"Not measured from here: this audit is running in PID {os.getpid()}, which owns no "
                "scanners. Owned child-process and worker-thread counts exist only inside the GUI "
                "process that spawned them - the Health page measures them; this CLI cannot."
            ),
            source=source,
            details={
                "audit_pid": os.getpid(),
                "reason": "audit is not running inside the scan-owning process",
                "owner_module": _PROCESS_OWNER_MODULE,
            },
        )

    live = int(snapshot.get("owned_child_count", 0) or 0)
    lingering = list(snapshot.get("lingering_child_pids") or [])
    active_label = str(snapshot.get("active_scan_label") or "")
    worker_threads = int(snapshot.get("scan_worker_threads", 0) or 0)
    bounce_overdue = int(snapshot.get("bounce_unretired_worker_count", 0) or 0)
    bounce_workers = list(snapshot.get("bounce_unretired_workers") or [])
    threads = int(snapshot.get("python_thread_count", 0) or 0)
    details = dict(snapshot)
    details.update({"audit_pid": os.getpid(), "market_phase": market_phase})

    if bounce_overdue:
        status = STATUS_DEGRADED
        summary = (
            f"{bounce_overdue} BounceBot worker(s) exceeded the shutdown budget "
            f"({', '.join(bounce_workers)}); {live} owned scan child(ren), "
            f"{worker_threads} scan worker thread(s), {threads} thread(s) in "
            f"PID {snapshot.get('process_pid')}."
        )
    elif live == 0:
        status = STATUS_HEALTHY
        summary = (
            f"No owned scan children; {worker_threads} scan worker thread(s), {threads} thread(s) "
            f"in PID {snapshot.get('process_pid')}."
        )
    elif active_label:
        status = STATUS_HEALTHY
        summary = (
            f"{live} owned scan child(ren) for the active scan ({active_label}); "
            f"{worker_threads} scan worker thread(s)."
        )
    else:
        # A child alive with no scan claiming it is the orphan sec 6.1 after-session
        # items 2-3 ask about (the deferred theta tail is the benign version of it,
        # which is exactly why it is reported rather than assumed either way).
        status = STATUS_DEGRADED
        summary = (
            f"{live} owned scan child(ren) still alive with no active scan "
            f"(PIDs {', '.join(str(pid) for pid in lingering) or 'unknown'}); "
            "either a deferred tail or an orphan."
        )
    return _check(
        "owned_process_counts",
        "Owned process/thread counts",
        status,
        summary,
        source=source,
        updated_at=now.isoformat(timespec="seconds"),
        details=details,
    )


def _bytes_mb(value: float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 2)


def _bytes_gb(value: float) -> float:
    return round(float(value) / (1024.0 * 1024.0 * 1024.0), 2)


def _writability_probe(directory: Path) -> tuple[bool, str, str]:
    """sec 6.1 pre-session item 7: prove the diagnostics directory is writable.

    A real write-and-delete, because "the directory exists" is not evidence that
    an artifact can land in it: a full volume, a revoked ACL or a sync client
    holding the folder all pass an existence check and then lose the day's
    evidence silently. The probe file is removed on every path.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=str(directory), prefix="health-write-probe-", suffix=".tmp")
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}", ""
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
    except OSError as exc:
        os.close(fd)
        try:
            os.remove(name)
        except OSError:
            pass
        return False, f"{type(exc).__name__}: {exc}", str(name)
    try:
        with handle:
            handle.write("operations_audit write probe\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}", str(name)
    finally:
        try:
            os.remove(name)
        except OSError:
            pass
    return True, "", str(name)


#: Ledgers below this size are not worth estimating rows for; the estimate
#: costs a small read and the answer would be noise.
JSONL_LEDGER_MIN_MB = 1.0
#: How much of a ledger is sampled to estimate its row count. Reading 370 MB
#: to count newlines on every audit would make System Health the thing that
#: stalls the desk; a sampled mean line length answers "how many rows, give
#: or take" for a fraction of the cost, and the field says `estimated`.
JSONL_LEDGER_SAMPLE_BYTES = 256 * 1024


def _jsonl_ledger_rows(diagnostics: Path, largest: list[tuple[int, str]]) -> list[dict]:
    """Read-only measurement of the append-only JSONL evidence ledgers.

    R6(b), 2026-08-17: rotation of these files is DECLINED for now, so what
    the desk needs is not a pruner but an honest number - size, estimated
    rows and last write, per ledger, measured rather than remembered. It
    reuses the footprint walk's own results: no second traversal, no writes,
    and nothing here can delete, truncate or compact anything.
    """
    rows: list[dict] = []
    for size, name in largest:
        if not name.lower().endswith(".jsonl"):
            continue
        megabytes = _bytes_mb(size)
        if megabytes < JSONL_LEDGER_MIN_MB:
            continue
        path = diagnostics / name
        estimated_rows = None
        sampled_lines = 0
        try:
            with path.open("rb") as handle:
                chunk = handle.read(JSONL_LEDGER_SAMPLE_BYTES)
            sampled_lines = chunk.count(b"\n")
            if sampled_lines:
                # Measure COMPLETE lines only. The read almost always ends
                # mid-row, and counting that fragment's bytes against the
                # rows that did finish biases every estimate the same way.
                complete = chunk.rfind(b"\n") + 1
                mean_line = complete / float(sampled_lines)
                estimated_rows = int(round(size / mean_line))
            elif chunk:
                # One enormous line, or a file still on its first row.
                estimated_rows = 1
        except OSError:
            # Unreadable is UNKNOWN, never zero: a ledger the audit cannot
            # open has not been shown to be empty.
            estimated_rows = None
        modified_at = ""
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                timespec="seconds"
            )
        except OSError:
            modified_at = ""
        rows.append(
            {
                "artifact": name,
                "megabytes": megabytes,
                "estimated_rows": estimated_rows,
                "sampled_lines": sampled_lines,
                "modified_at": modified_at,
                "retention": "declined 2026-08-17 (plan.md item 6(b)); warehouse-owned",
            }
        )
    return rows

def _disk_check(diagnostics: Path, now: datetime) -> dict[str, Any]:
    """Free space, writability and artifact footprint (sec 6.3 bullet 13).

    Nothing in ``scripts/`` probed free space at all before this, and the
    diagnostics artifacts have no retention policy: the footprint is reported
    with its largest contributors so "the shadow logs quietly ate the disk" is
    visible before it becomes "the scan could not write its manifest".
    """
    source = diagnostics
    writable, write_error, probe_path = _writability_probe(diagnostics)

    usage_target = diagnostics
    while not usage_target.exists() and usage_target.parent != usage_target:
        usage_target = usage_target.parent
    try:
        usage = shutil.disk_usage(str(usage_target))
    except OSError as exc:
        return _check(
            "disk_storage_warnings",
            "Disk/storage warnings",
            STATUS_UNKNOWN,
            f"Free space on {usage_target} could not be measured: {exc}",
            source=source,
            updated_at=now.isoformat(timespec="seconds"),
            details={
                "volume": str(usage_target),
                "error": str(exc),
                "diagnostics_writable": writable,
                "write_probe_error": write_error,
            },
        )

    total_bytes = 0
    file_count = 0
    largest: list[tuple[int, str]] = []
    walk_error = ""
    try:
        for path in diagnostics.rglob("*"):
            try:
                if not path.is_file():
                    continue
                size = path.stat().st_size
            except OSError:
                continue
            total_bytes += size
            file_count += 1
            largest.append((size, str(path.relative_to(diagnostics))))
    except OSError as exc:
        walk_error = f"{type(exc).__name__}: {exc}"
    largest.sort(reverse=True)
    largest_rows = [{"artifact": name, "megabytes": _bytes_mb(size)} for size, name in largest[:5]]
    ledger_rows = _jsonl_ledger_rows(diagnostics, largest)

    free_gb = _bytes_gb(usage.free)
    footprint_mb = _bytes_mb(total_bytes)
    oversized = [row for row in largest_rows if row["megabytes"] >= SINGLE_ARTIFACT_DEGRADED_MB]

    warnings: list[str] = []
    if not writable:
        status = STATUS_UNHEALTHY
        warnings.append(f"diagnostics directory is not writable ({write_error})")
    elif free_gb < DISK_FREE_UNHEALTHY_GB:
        status = STATUS_UNHEALTHY
        warnings.append(f"only {free_gb:.2f} GB free on {usage_target.anchor or usage_target}")
    elif free_gb < DISK_FREE_DEGRADED_GB:
        status = STATUS_DEGRADED
        warnings.append(f"{free_gb:.2f} GB free is below the {DISK_FREE_DEGRADED_GB:.0f} GB floor")
    elif footprint_mb > DIAGNOSTICS_FOOTPRINT_DEGRADED_MB or oversized:
        status = STATUS_DEGRADED
        if footprint_mb > DIAGNOSTICS_FOOTPRINT_DEGRADED_MB:
            warnings.append(f"diagnostics footprint is {footprint_mb:.0f} MB with no retention policy")
        for row in oversized:
            warnings.append(f"{row['artifact']} is {row['megabytes']:.1f} MB and is never pruned")
    else:
        status = STATUS_HEALTHY
    if walk_error:
        warnings.append(f"footprint is incomplete: {walk_error}")

    summary = (
        f"{free_gb:.1f} GB free of {_bytes_gb(usage.total):.1f} GB; diagnostics {footprint_mb:.0f} MB "
        f"across {file_count} file(s); write probe {'ok' if writable else 'FAILED'}."
    )
    if warnings:
        # Deliberately not sentence-cased: these strings carry file names, and
        # capitalizing "greatness_shadow.jsonl" would rename the evidence.
        summary += " Warnings: " + "; ".join(warnings) + "."
    return _check(
        "disk_storage_warnings",
        "Disk/storage warnings",
        status,
        summary,
        source=source,
        updated_at=now.isoformat(timespec="seconds"),
        details={
            "volume": str(usage_target),
            "free_gb": free_gb,
            "total_gb": _bytes_gb(usage.total),
            "used_gb": _bytes_gb(usage.used),
            "free_pct": round(100.0 * usage.free / usage.total, 2) if usage.total else None,
            "diagnostics_dir": str(diagnostics),
            "diagnostics_footprint_mb": footprint_mb,
            "diagnostics_file_count": file_count,
            "largest_artifacts": largest_rows,
            "jsonl_ledgers": ledger_rows,
            "diagnostics_writable": writable,
            "write_probe_error": write_error,
            "write_probe_path": probe_path,
            "warnings": warnings,
            "thresholds": {
                "free_degraded_gb": DISK_FREE_DEGRADED_GB,
                "free_unhealthy_gb": DISK_FREE_UNHEALTHY_GB,
                "footprint_degraded_mb": DIAGNOSTICS_FOOTPRINT_DEGRADED_MB,
                "single_artifact_degraded_mb": SINGLE_ARTIFACT_DEGRADED_MB,
            },
        },
    )


#: Above this failure share of a provider's OWN attempts (never a mixed or
#: unrelated denominator), that (family, provider) pair is failing outright.
PROVIDER_FAILURE_UNHEALTHY_RATIO = 0.25


def _provider_check(latest_manifest: dict[str, Any], manifests_dir: Path) -> dict[str, Any]:
    """Sec 6.3 bullet 9, measured from the newest run manifest (schema v2).

    Honesty rules, in grading order:

    * no ``provider.schema_version`` -> UNKNOWN (pre-instrumentation build or
      no scan; an old ``provider.captured`` stamp alone proves nothing).
    * instrumented families != declared expected families -> PARTIAL coverage,
      capped at DEGRADED.  An empty run may be HEALTHY only when the contract
      proves every required boundary was instrumented and zero lookups
      occurred.
    * ``capture_errors`` or ``orphan_events`` -> not healthy: the accounting
      itself was damaged or contaminated-adjacent, and saying so beats a
      confident number.
    * failure ratios are computed per (family, provider) against that
      provider's own attempt denominator only.
    * malformed counter values are tolerated, counted, and reported - never a
      crash, never silently zero.
    """
    counters = (
        latest_manifest.get("counters")
        if isinstance(latest_manifest.get("counters"), dict)
        else {}
    )
    outputs = (
        latest_manifest.get("outputs")
        if isinstance(latest_manifest.get("outputs"), dict)
        else {}
    )
    check_id, label = "provider_counters", "Provider request/cache/throttle/failure counts"
    run_id = str(latest_manifest.get("run_id") or "") if latest_manifest else ""

    def _int(value) -> tuple[int, bool]:
        try:
            return int(value), True
        except (TypeError, ValueError):
            return 0, False

    schema_version, schema_ok = _int(counters.get("provider.schema_version"))
    if not schema_ok or schema_version < 2:
        return _check(
            check_id,
            label,
            STATUS_UNKNOWN,
            "Not measured yet: the newest scan manifest carries no v2 provider "
            "telemetry (pre-instrumentation build, or no scan has run). The "
            "first master scan on this build will record it.",
            source=manifests_dir,
            details={"manifest_run_id": run_id, "captured": False},
        )

    expected = [f for f in str(outputs.get("provider_families_expected") or "").split(",") if f]
    instrumented = [
        f for f in str(outputs.get("provider_families_instrumented") or "").split(",") if f
    ]
    missing_families = sorted(set(expected) - set(instrumented))

    malformed_values = 0
    per_family: dict[str, dict[str, int]] = {}
    provider_attempts: dict[tuple[str, str], int] = {}
    provider_failures: dict[tuple[str, str], int] = {}
    totals = {
        "lookup": 0,
        "cache_hit": 0,
        "attempt": 0,
        "success": 0,
        "failure": 0,
        "throttle": 0,
        "fallback_used": 0,
        "refresh_unusable": 0,
    }
    for key, raw in counters.items():
        text = str(key)
        if not text.startswith("provider.") or text in (
            "provider.schema_version",
            "provider.capture_errors",
            "provider.orphan_events",
            "provider.captured",
        ):
            continue
        parts = text.split(".")
        value, ok = _int(raw)
        if not ok:
            malformed_values += 1
            continue
        if len(parts) == 3:
            _, family, outcome = parts
            provider = ""
        elif len(parts) == 4:
            _, family, outcome, provider = parts
        else:
            malformed_values += 1
            continue
        per_family.setdefault(family, {})[".".join(parts[2:])] = value
        if outcome in totals:
            totals[outcome] += value
        if provider:
            if outcome == "attempt":
                provider_attempts[(family, provider)] = (
                    provider_attempts.get((family, provider), 0) + value
                )
            elif outcome == "failure":
                provider_failures[(family, provider)] = (
                    provider_failures.get((family, provider), 0) + value
                )

    capture_errors, _ = _int(counters.get("provider.capture_errors"))
    orphan_events, _ = _int(counters.get("provider.orphan_events"))

    # Matching denominators only: a (family, provider) failure count is judged
    # against that pair's own attempts, never a mixed total.
    failing_pairs = []
    for pair, failures in provider_failures.items():
        attempts = provider_attempts.get(pair, 0)
        if failures and not attempts:
            # Failures with no recorded attempts is itself an accounting
            # anomaly - report it rather than dividing by something unrelated.
            failing_pairs.append((pair, failures, 0, None))
        elif attempts and (failures / attempts) > PROVIDER_FAILURE_UNHEALTHY_RATIO:
            failing_pairs.append((pair, failures, attempts, failures / attempts))

    problems: list[str] = []
    if missing_families:
        problems.append(
            f"PARTIAL coverage: uninstrumented boundary families: {', '.join(missing_families)}."
        )
    if capture_errors:
        problems.append(f"{capture_errors} capture error(s): the accounting itself failed.")
    # Out-of-run events are NOT a problem: the GUI legitimately hits provider
    # boundaries between scans (e.g. the D1 chart's stale-tail backfill). The
    # bucket exists so they can never contaminate a run's counts; the count is
    # reported for attribution honesty, not graded.
    if malformed_values:
        problems.append(f"{malformed_values} malformed counter value(s).")
    if totals["throttle"]:
        problems.append(f"{totals['throttle']} pacing-class throttle event(s).")
    for pair, failures, attempts, ratio in failing_pairs:
        family, provider = pair
        if ratio is None:
            problems.append(
                f"{family}/{provider}: {failures} failure(s) with no recorded attempts."
            )
        else:
            problems.append(
                f"{family}/{provider}: {failures}/{attempts} attempts failed ({ratio:.0%})."
            )

    if any(ratio is not None for _, _, _, ratio in failing_pairs):
        status = STATUS_UNHEALTHY
    elif problems:
        status = STATUS_DEGRADED
    else:
        status = STATUS_HEALTHY

    lookups = totals["lookup"]
    cache_ratio = (totals["cache_hit"] / lookups) if lookups else None
    summary = (
        f"Last scan: {lookups} lookup(s), {totals['cache_hit']} cache hit(s)"
        + (f" ({cache_ratio:.0%})" if cache_ratio is not None else "")
        + f", {totals['attempt']} outbound attempt(s), {totals['success']} success(es), "
        f"{totals['failure']} failure(s), {totals['throttle']} throttle(s), "
        f"{totals['fallback_used']} fallback(s)."
    )
    if not lookups and not problems:
        summary = (
            "Instrumentation active across all declared boundaries; zero provider "
            f"lookups occurred this run. {summary}"
        )
    if problems:
        summary += " " + " ".join(problems)
    if orphan_events:
        summary += (
            f" {orphan_events} out-of-run event(s) (GUI/backfill activity between "
            "scans; excluded from this run's counts)."
        )

    return _check(
        check_id,
        label,
        status,
        summary,
        source=manifests_dir,
        details={
            "manifest_run_id": run_id,
            "captured": True,
            "schema_version": schema_version,
            "families_expected": expected,
            "families_instrumented": instrumented,
            "families_missing": missing_families,
            "totals": totals,
            "cache_hit_ratio": round(cache_ratio, 4) if cache_ratio is not None else None,
            "per_family": per_family,
            "capture_errors": capture_errors,
            "orphan_events": orphan_events,
            "malformed_counter_values": malformed_values,
            "failure_unhealthy_ratio": PROVIDER_FAILURE_UNHEALTHY_RATIO,
        },
    )


def _symbol_count(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def _universe_check(
    universe_paths: tuple[Path, ...],
    market_data_path: Path,
    market_date: str,
    now: datetime,
    local_tz,
) -> dict[str, Any]:
    """Universe and market-data freshness (plan.md sec 6.3 bullet 10).

    Read-only over artifacts that already exist; it builds nothing. The universe
    build age lived only on the Universe page, which fails sec 6.3's "without
    opening files manually" framing - and since *nothing schedules a rebuild*, a
    months-old universe rides every master scan with no surface saying so.
    """
    source = universe_paths[0] if universe_paths else market_data_path
    files: list[dict[str, Any]] = []
    newest_age_days: float | None = None
    for path in universe_paths:
        row: dict[str, Any] = {"path": str(path), "name": path.name, "exists": path.exists()}
        if row["exists"]:
            try:
                built_at = datetime.fromtimestamp(path.stat().st_mtime, tz=local_tz)
            except OSError as exc:
                row["error"] = str(exc)
                built_at = None
            if built_at is not None:
                age_days = max(0.0, (now - built_at).total_seconds() / 86400.0)
                row["built_at"] = built_at.isoformat(timespec="seconds")
                row["age_days"] = round(age_days, 2)
                row["symbols"] = _symbol_count(path)
                newest_age_days = age_days if newest_age_days is None else min(newest_age_days, age_days)
        files.append(row)

    present = [row for row in files if row.get("exists")]
    symbol_total = sum(int(row.get("symbols") or 0) for row in present)
    built_at_text = max((str(row.get("built_at") or "") for row in present), default="")

    market_row: dict[str, Any] = {"path": str(market_data_path), "exists": market_data_path.exists()}
    market_status = STATUS_UNKNOWN
    if market_row["exists"]:
        try:
            written = datetime.fromtimestamp(market_data_path.stat().st_mtime, tz=local_tz)
        except OSError as exc:
            market_row["error"] = str(exc)
            written = None
        if written is not None:
            market_row["updated_at"] = written.isoformat(timespec="seconds")
            try:
                stale_days = (datetime.fromisoformat(market_date).date() - written.date()).days
            except ValueError:
                stale_days = None
            market_row["calendar_days_behind"] = stale_days
            if stale_days is None:
                market_status = STATUS_UNKNOWN
            elif stale_days > MARKET_DATA_UNHEALTHY_AFTER_DAYS:
                market_status = STATUS_UNHEALTHY
            elif stale_days > MARKET_DATA_DEGRADED_AFTER_DAYS:
                market_status = STATUS_DEGRADED
            else:
                market_status = STATUS_HEALTHY

    if not present:
        universe_status = STATUS_UNKNOWN
        universe_text = "No self-built universe exists on this machine, so its freshness is unknown."
    elif symbol_total <= 0:
        universe_status = STATUS_UNHEALTHY
        universe_text = "The self-built universe exists but is empty; every scan is running without it."
    elif newest_age_days is None:
        universe_status = STATUS_UNKNOWN
        universe_text = "The universe exists, but its build time cannot be established."
    elif newest_age_days > UNIVERSE_UNHEALTHY_AFTER_DAYS:
        universe_status = STATUS_UNHEALTHY
        universe_text = (
            f"Universe is {newest_age_days:.0f} days old ({symbol_total} symbols) and nothing "
            "schedules a rebuild; every scan is running on it."
        )
    elif newest_age_days > UNIVERSE_DEGRADED_AFTER_DAYS:
        universe_status = STATUS_DEGRADED
        universe_text = (
            f"Universe is {newest_age_days:.1f} days old ({symbol_total} symbols); rebuilds are "
            "manual only."
        )
    else:
        universe_status = STATUS_HEALTHY
        universe_text = f"Universe {symbol_total} symbols, built {newest_age_days:.1f} days ago."

    if not market_row["exists"]:
        market_text = f"No daily-bar probe artifact at {market_data_path.name}; market-data age is unknown."
    elif market_status == STATUS_UNKNOWN:
        market_text = f"{market_data_path.name} exists, but its age cannot be established."
    else:
        market_text = (
            f"Daily bars ({market_data_path.name}) last written "
            f"{market_row.get('calendar_days_behind')} calendar day(s) before {market_date}."
        )

    status = worst_status([universe_status, market_status])
    return _check(
        "universe_and_market_data_freshness",
        "Universe and market-data freshness",
        status,
        f"{universe_text} {market_text}",
        source=source,
        updated_at=built_at_text or str(market_row.get("updated_at") or ""),
        details={
            "universe_status": universe_status,
            "universe_files": files,
            "universe_symbol_total": symbol_total,
            "universe_age_days": round(newest_age_days, 2) if newest_age_days is not None else None,
            "market_data_status": market_status,
            "market_data": market_row,
            "market_date": market_date,
            "rebuild_is_scheduled": False,
            "thresholds": {
                "universe_degraded_after_days": UNIVERSE_DEGRADED_AFTER_DAYS,
                "universe_unhealthy_after_days": UNIVERSE_UNHEALTHY_AFTER_DAYS,
                "market_data_degraded_after_days": MARKET_DATA_DEGRADED_AFTER_DAYS,
                "market_data_unhealthy_after_days": MARKET_DATA_UNHEALTHY_AFTER_DAYS,
            },
        },
    )


def _inventory_gap_checks(emitted_ids: set[str]) -> list[dict[str, Any]]:
    """An explicit UNKNOWN row for every sec 6.3 dimension nothing measures.

    This is the point of the packet: leaving an unimplemented dimension out of
    the check list entirely is how ``max()`` over the implemented checks kept
    returning HEALTHY on a machine whose disk, provider counters, owned-process
    count and universe age were never measured at all.
    """
    gaps: list[dict[str, Any]] = []
    for entry in REQUIRED_CHECK_INVENTORY:
        if entry.covered_by and all(check_id in emitted_ids for check_id in entry.covered_by):
            continue
        if entry.covered_by:
            missing = [check_id for check_id in entry.covered_by if check_id not in emitted_ids]
            summary = (
                f"Not measured: {entry.requirement}. Expected check(s) "
                f"{', '.join(missing)} were not emitted."
            )
            reason = "expected check did not run"
        else:
            summary = (
                f"Not measured: {entry.requirement}. Required by plan.md sec 6.3; "
                "no telemetry collects it yet."
            )
            reason = "no implementation collects this evidence yet"
        gaps.append(
            _check(
                entry.id,
                entry.label,
                STATUS_UNKNOWN,
                summary,
                source=_PLAN_SOURCE,
                details={
                    "requirement": entry.requirement,
                    "reason": reason,
                    "expected_checks": list(entry.covered_by),
                    "plan_reference": "plan.md sec 6.3",
                },
            )
        )
    return gaps


def _required_inventory_view(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The sec 6.3 inventory with the status each dimension actually resolved to."""
    by_id = {str(check.get("id")): check for check in checks}
    rows: list[dict[str, Any]] = []
    for entry in REQUIRED_CHECK_INVENTORY:
        covering = [by_id[check_id] for check_id in entry.covered_by if check_id in by_id]
        if covering:
            status = worst_status(check["status"] for check in covering)
            check_ids = [str(check["id"]) for check in covering]
        else:
            fallback = by_id.get(entry.id, {})
            status = str(fallback.get("status") or STATUS_UNKNOWN)
            check_ids = [entry.id] if fallback else []
        rows.append(
            {
                "id": entry.id,
                "label": entry.label,
                "requirement": entry.requirement,
                "status": status,
                # Whether an implementation attempted this dimension at all -
                # not whether it came back with evidence.
                "implemented": bool(covering),
                "check_ids": check_ids,
            }
        )
    return rows


def _outcome_sweep_check(now: datetime, local_tz, diagnostics: Path | None = None) -> dict[str, Any]:
    """Did the after-close sweep run, and is the pending backlog draining? (R10.A / D3)

    Reads the coverage file the sweep writes; it never sweeps anything itself.
    **No file is `unknown`, not healthy** - a sweep that has never reported is
    indistinguishable from a sweep that never ran, and that indistinguishability
    is exactly how 576 pending outcomes accumulated unnoticed over two months.
    """
    root = Path(diagnostics) if diagnostics is not None else Path(get_diagnostics_dir())
    path = root / "outcome_sweep_coverage.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _check(
            "outcome_sweep", "Outcome finalization sweep", STATUS_UNKNOWN,
            "The after-close outcome sweep has not reported yet. It runs once per "
            "weekday after close+grace; until it does, whether pending trades are "
            "finalizing is unmeasured - which is not the same as fine.",
            source=path,
        )

    pending_after = int(payload.get("pending_after") or 0)
    finalized = int(payload.get("finalized") or 0)
    expired = int(payload.get("expired") or 0)
    unparseable = int(payload.get("unparseable") or 0)
    swept_at = str(payload.get("swept_at") or "")
    details = {
        "pending_before": payload.get("pending_before"),
        "pending_after": pending_after,
        "finalized": finalized,
        "expired": expired,
        "unparseable": unparseable,
        "by_reason": payload.get("by_reason") or {},
        "swept_at": swept_at,
    }
    age_days = None
    try:
        stamp = datetime.fromisoformat(swept_at)
        reference = now.replace(tzinfo=None) if stamp.tzinfo is None else now
        age_days = (reference - stamp).total_seconds() / 86400.0
    except ValueError:
        pass

    tail = (
        f" Last sweep {swept_at}: {finalized} finalized"
        + (f", {expired} expired" if expired else "")
        + (f", {unparseable} unparseable" if unparseable else "")
        + "."
    )
    # Four calendar days covers a long weekend without letting a silent week pass.
    if age_days is not None and age_days > 4:
        return _check(
            "outcome_sweep", "Outcome finalization sweep", STATUS_DEGRADED,
            f"The last sweep was {age_days:.1f} days ago, so pending trades may be "
            "accumulating again." + tail,
            source=path, details=details,
        )
    if pending_after > 200:
        return _check(
            "outcome_sweep", "Outcome finalization sweep", STATUS_DEGRADED,
            f"{pending_after} outcomes are still pending after the last sweep. The "
            "backlog D3 measured was 576; a number in that range means trades are "
            "being registered faster than they are finalized." + tail,
            source=path, details=details,
        )
    return _check(
        "outcome_sweep", "Outcome finalization sweep", STATUS_HEALTHY,
        f"{pending_after} outcome(s) pending after the last sweep." + tail,
        source=path, details=details,
    )


def _daily_bar_units_check(now: datetime, local_tz, diagnostics: Path | None = None) -> dict[str, Any]:
    """Is the durable daily store still single-unit? (R10.V step 6)

    Read from the file the nightly snapshot job writes, never measured here: the
    measurement takes ~7 s over 1,958 files, and a tile a human waits on is a
    tile nobody opens.

    The gate is the **unit**, not the cliff. A >20x volume step is a real thing
    that happens to real stocks - after the 2026-08-23 backfill, 19 all-`yahoo`
    files still show one (DJT at its listing, OKLO's de-SPAC) - so a cliff in a
    single-source file means market event, not defect. It is reported and it
    never sets the status.
    """
    from ops.daily_bar_cliff import HEALTH_FILENAME

    root = Path(diagnostics) if diagnostics is not None else Path(get_diagnostics_dir())
    path = root / HEALTH_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _check(
            "daily_bar_units", "Daily bar units", STATUS_UNKNOWN,
            "No daily-bar unit measurement has been filed yet. The nightly evidence "
            "snapshot writes it; until it runs, the store's unit mix is unmeasured - "
            "which is not the same as clean.",
            source=path,
        )

    units = payload.get("rows_by_volume_unit") or {}
    rows = int(payload.get("rows") or 0)
    shares = int(units.get("shares") or 0)
    other = max(0, rows - shares)
    cliffed = int((payload.get("cliff") or {}).get("cliffed") or 0)
    measured_at = str(payload.get("measured_at") or "")
    age_hours = None
    try:
        stamp = datetime.fromisoformat(measured_at)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        age_hours = (now.astimezone(timezone.utc) - stamp).total_seconds() / 3600.0
    except ValueError:
        pass

    details = {
        "rows": rows,
        "rows_shares": shares,
        "rows_other": other,
        "rows_shares_pct": payload.get("rows_shares_pct"),
        "files_without_unit_column": payload.get("files_without_unit_column"),
        "rows_by_volume_unit": units,
        "files_by_schema": payload.get("files_by_schema") or {},
        "files_not_all_shares": payload.get("files_not_all_shares"),
        "cliffed_files": cliffed,
        "measured_at": measured_at,
    }
    share_pct = (shares / rows * 100.0) if rows else 0.0
    cliff_note = (
        f" {cliffed} file(s) still step >20x; in an all-shares file that is a market "
        "event, not a unit mix."
        if cliffed
        else " No file steps >20x."
    )

    # A measurement older than two nights cannot answer today's question.
    if age_hours is not None and age_hours > 48:
        return _check(
            "daily_bar_units", "Daily bar units", STATUS_DEGRADED,
            f"The unit measurement is {age_hours / 24:.1f} days old ({measured_at}), so it "
            "cannot say what the store holds now. Check that the evidence snapshot task "
            "is running.",
            source=path, details=details,
        )
    # Two different states, and only one of them is actionable.
    #
    # `lots_rth` is a REGRESSION: the write seam refuses IB volume, so a single
    # such row means something got past it, and that is the splice starting
    # again. Degraded.
    #
    # `unknown` is the known residue - 188 rows Yahoo has no data for, in files
    # named in the backfill manifest. Nobody can clear it without a different
    # vendor, and a permanent alarm is an alarm people learn to ignore. It is
    # reported in full and it does not set the status.
    lots = int(units.get("lots_rth") or 0)
    unknown = int(units.get("unknown") or 0)
    no_column = int(units.get("no_column") or 0)
    files_without_column = int(payload.get("files_without_unit_column") or 0)
    parts = []
    if unknown:
        parts.append(
            f"{unknown:,} row(s) remain unmeasured - Yahoo has no data for them and the "
            "backfill manifest names the files"
        )
    if no_column:
        parts.append(
            f"{no_column:,} row(s) in {files_without_column} file(s) predate the unit "
            "column and were not rewritten (Yahoo covered too little of them, or would "
            "have made them worse)"
        )
    residue = (" " + "; ".join(parts) + ".") if parts else ""
    if lots:
        return _check(
            "daily_bar_units", "Daily bar units", STATUS_DEGRADED,
            f"{lots:,} of {rows:,} rows carry IB round-lot volume. The write seam refuses "
            "it, so this means something got past it - a volume-weighted AVWAP computed "
            "across those rows is the R10.V splice starting again."
            + residue + cliff_note,
            source=path, details=details,
        )
    return _check(
        "daily_bar_units", "Daily bar units", STATUS_HEALTHY,
        f"{shares:,} of {rows:,} rows are share-denominated ({share_pct:.2f}%); no row "
        "carries IB round-lot volume." + residue + cliff_note,
        source=path, details=details,
    )


def _daily_bar_history_note(manifest_dir: Path | None) -> tuple[str, dict[str, Any]]:
    """What the run manifests say the store already contains (R10.0b).

    Read-only context, never a status: the pin governs what happens NEXT, and a
    history that cannot be changed must not raise a permanent alarm. The verdict
    per session comes from `evidence_rules.daily_volume_mixed_v1`, which reads
    the manifests' own provider counters; sessions outside the 90-run manifest
    window read `unknown`, not clean.
    """
    if manifest_dir is None:
        return "", {}
    try:
        import evidence_rules

        verdicts = evidence_rules.daily_volume_session_verdicts(manifest_dir)
    except Exception:  # pragma: no cover - health must never take the audit down
        logging.exception("daily-bar unit history unavailable")
        return "", {}
    if not verdicts:
        return "", {"history": "no readable run manifests"}
    mixed = sum(1 for verdict in verdicts.values() if verdict == evidence_rules.VERDICT_MIXED)
    note = (
        f" History: {mixed} of {len(verdicts)} manifest-covered sessions carry IB "
        f"round-lot volume (`{evidence_rules.RULE_DAILY_VOLUME_MIXED}`); earlier "
        "sessions are unmeasured, not clean. R10.V repairs the store."
    )
    return note, {
        "sessions_covered": len(verdicts),
        "sessions_mixed": mixed,
        "rule": evidence_rules.RULE_DAILY_VOLUME_MIXED,
    }


def _daily_bar_source_check(
    now: datetime, local_tz, manifest_dir: Path | None = None
) -> dict[str, Any]:
    """Which source the durable daily-bar store is taking volume from (R10.0b).

    The store is mixed: IB returns regular-session volume in round lots while
    Yahoo returns the consolidated session in shares, and the observed ratio is
    symbol-dependent (SPY 1.0x, TSLA 56x, AAPL 81x, A 162x, NVDA 188x), so no
    constant converts one into the other. Until the cliff packet lands, a pin to
    Yahoo is the only thing keeping the store from getting more mixed - which
    makes "am I pinned?" a question System Health should answer without anyone
    opening a settings file.
    """
    from master_avwap_lib.legacy import DAILY_BARS_SOURCE_SETTING, daily_bars_source_pin
    from project_paths import LOCAL_SETTINGS_FILE

    try:
        pin = daily_bars_source_pin()
    except Exception:  # pragma: no cover - health must never take the audit down
        logging.exception("daily-bar source pin unavailable")
        return _check(
            "daily_bar_source", "Daily bar source", STATUS_UNKNOWN,
            "The daily-bar source pin could not be read.", source=Path(__file__),
        )
    history_note, history_details = _daily_bar_history_note(manifest_dir)
    details = {"pin": pin, "setting": DAILY_BARS_SOURCE_SETTING,
               "settings_file": str(LOCAL_SETTINGS_FILE), **history_details}
    if pin == "yahoo":
        return _check(
            "daily_bar_source", "Daily bar source", STATUS_HEALTHY,
            "Pinned to yahoo: the durable store takes share-denominated volume from one "
            "source and spends no IB budget on daily bars." + history_note,
            source=Path(__file__),
            details=details,
        )
    return _check(
        "daily_bar_source", "Daily bar source", STATUS_UNKNOWN,
        "Not pinned (auto). Whether a given scan wrote IB round-lot volume or Yahoo "
        "share volume into the durable store depends on IB availability that run, so "
        "this cannot be answered from the setting alone - the run manifests carry it. "
        f'Set {DAILY_BARS_SOURCE_SETTING}="yahoo" to make it answerable until the '
        "cliff packet lands." + history_note,
        source=Path(__file__),
        details=details,
    )


def _evidence_snapshot_check(now: datetime, local_tz, staging: Path | None = None) -> dict[str, Any]:
    r"""Is the evidence the cold push excludes actually being backed up? (R10.A)

    ``data\runtime`` is 3.5 GB of hot state - the setup tracker, the outcome
    CSV, the journal SQLite, every cohort and Focus store - and the cold push
    excludes all of it by design because it is rewritten constantly. The nightly
    dated snapshot is the only thing standing between that and a single disk.

    A backup nobody has restored is a hypothesis, so the restore-test date is
    reported beside the snapshot date rather than assumed.
    """
    from datetime import date as _date

    from ops import evidence_snapshot

    RESEARCH_DAS_ROOT = "\\\\MINI-PC\\Trading Bot Data"
    staging = Path(staging) if staging is not None else Path(CACHE_DIR) / "evidence_snapshots"
    try:
        info = evidence_snapshot.health(staging, das_root=Path(RESEARCH_DAS_ROOT))
    except Exception:  # pragma: no cover - health must never take the audit down
        logging.exception("evidence snapshot health unavailable")
        return _check(
            "evidence_snapshot", "Evidence snapshot", STATUS_UNKNOWN,
            "Snapshot health could not be read.", source=staging,
        )

    stamp = str(info.get("last_snapshot_date") or "")
    if not stamp:
        # Absent evidence is UNKNOWN, never unhealthy - the repo's own rule, and
        # a machine that has not been scheduled yet is not a machine in trouble.
        # The summary still says plainly what is unprotected.
        return _check(
            "evidence_snapshot", "Evidence snapshot", STATUS_UNKNOWN,
            r"No evidence snapshot on record; data\runtime, the home-root evidence "
            "files and the diagnostics tree exist on one disk only until "
            "snapshot_to_das.ps1 is scheduled.",
            source=staging,
        )
    age_days = None
    try:
        age_days = (now.date() - _date.fromisoformat(stamp)).days
    except (TypeError, ValueError):
        pass
    if age_days is None:
        status = STATUS_UNKNOWN
    elif age_days <= 1:
        status = STATUS_HEALTHY
    elif age_days <= 3:
        status = STATUS_DEGRADED
    else:
        status = STATUS_UNHEALTHY
    skipped = int(info.get("skipped") or 0)
    if skipped and status == STATUS_HEALTHY:
        # Counted, never hidden: a snapshot that quietly omitted the 960 MB
        # tracker looks identical to one that captured it.
        status = STATUS_DEGRADED
    summary = (
        f"Last snapshot {stamp}"
        + (f" ({age_days}d ago)" if age_days is not None else "")
        + f": {info.get('files', 0)} files, "
        f"{int(info.get('stored_bytes') or 0) / 1e6:.0f} MB stored"
        + (f", {skipped} skipped" if skipped else "")
        + (". DAS reachable." if info.get("das_reachable") else ". DAS unreachable - staged locally.")
        + (f" Last restore test {info['last_restore_test'][:10]}."
           if info.get("last_restore_test") else " No restore test on record.")
    )
    return _check(
        "evidence_snapshot", "Evidence snapshot", status, summary,
        source=staging, updated_at=str(info.get("last_snapshot_at") or ""),
        details=dict(info),
    )


def build_operations_audit(
    *,
    now: datetime | None = None,
    diagnostics_dir: Path | str | None = None,
    candidate_registry_path: Path | str | None = None,
    away_report_path: Path | str | None = None,
    autopilot_state_path: Path | str | None = None,
    industry_state_path: Path | str | None = None,
    journal_db_path: Path | str | None = None,
    writer_health_path: Path | str | None = None,
    universe_paths: Iterable[Path | str] | None = None,
    market_data_probe_path: Path | str | None = None,
    process_snapshot: dict[str, Any] | None = None,
    review_capture: bool = True,
    **review_capture_paths: Path | str | None,
) -> dict[str, Any]:
    local_tz, timezone_name = get_market_local_timezone()
    moment = normalize_market_local_datetime(now, local_timezone=local_tz)
    market_phase, session = _phase(moment)
    diagnostics = Path(diagnostics_dir) if diagnostics_dir is not None else get_diagnostics_dir()
    registry_path = Path(candidate_registry_path) if candidate_registry_path is not None else CACHE_DIR.parent / "candidate_registry.json"
    report_path = Path(away_report_path) if away_report_path is not None else (
        diagnostics / "autopilot_today.txt" if diagnostics_dir is not None else Path(AUTOPILOT_REPORT_FILE)
    )
    auto_state_path = Path(autopilot_state_path) if autopilot_state_path is not None else (
        diagnostics / "autopilot_state.json" if diagnostics_dir is not None else Path(AUTOPILOT_STATE_FILE)
    )
    industry_path = Path(industry_state_path) if industry_state_path is not None else (
        diagnostics / "industry_board_snapshot.json"
        if diagnostics_dir is not None
        else Path(INDUSTRY_BOARD_STATE_FILE)
    )
    journal_path = Path(journal_db_path) if journal_db_path is not None else (
        diagnostics / "trade_journal.sqlite3"
        if diagnostics_dir is not None
        else Path(JOURNAL_DB_FILE)
    )
    health_path = (
        Path(writer_health_path)
        if writer_health_path is not None
        else diagnostics / writer_health.HEALTH_FILENAME
    )
    if universe_paths is not None:
        universe_files = tuple(Path(item) for item in universe_paths)
    elif diagnostics_dir is not None:
        universe_files = (
            diagnostics / "universe_all.txt",
            diagnostics / "universe_longs.txt",
            diagnostics / "universe_shorts.txt",
        )
    else:
        universe_files = (UNIVERSE_ALL_FILE, UNIVERSE_LONGS_FILE, UNIVERSE_SHORTS_FILE)
    if market_data_probe_path is not None:
        market_probe = Path(market_data_probe_path)
    elif diagnostics_dir is not None:
        market_probe = diagnostics / "daily_bars" / "SPY.parquet"
    else:
        # One stat on the benchmark the champion D1 alerts run on, deliberately
        # not a walk of the ~1,900-file shared bar store: the Health page
        # refreshes every 15s and must not stat a Drive directory each time.
        market_probe = Path(MASTER_AVWAP_DAILY_BARS_DIR) / "SPY.parquet"
    market_date = session.market_date.isoformat()
    if diagnostics_dir is not None:
        # Same convention the report/state/industry paths use above: an
        # explicit diagnostics directory means "audit this sandbox", so the
        # learning artifacts resolve inside it instead of the shared home.
        for keyword, filename in (
            ("review_events_path", "alert_review_events.jsonl"),
            ("preference_state_path", "review_preference_state.json"),
            ("policy_path", "review_policy.json"),
            ("policy_draft_path", "review_policy_draft.json"),
            ("scoring_config_path", "master_avwap_scoring_config.json"),
        ):
            review_capture_paths.setdefault(keyword, diagnostics / filename)

    heartbeat = _heartbeat_check(diagnostics / "heartbeat.json", moment, local_tz)
    ledger, jobs = _ledger_check(diagnostics / "job_ledger.jsonl", market_date, moment, local_tz, market_phase)
    running_job = any(job.get("state") == "RUNNING" for job in jobs)
    manifest, latest_manifest = _manifest_check(
        diagnostics / "run_manifests", market_date, moment, local_tz, running_job, market_phase
    )
    checks = [
        *_writer_health_checks(health_path, moment, local_tz),
        heartbeat,
        ledger,
        _ai_jobs_check(moment, local_tz),
        _questrade_chain_check(moment, journal_path),
        manifest,
        _away_report_check(report_path, auto_state_path, moment, local_tz, market_phase),
        _industry_board_check(industry_path, moment, local_tz, market_phase),
        _shadow_check(
            check_id="spy_shadow",
            label="SPY state shadow",
            path=diagnostics / "spy_state_shadow_status.json",
            log_path=diagnostics / "spy_state_shadow.jsonl",
            log_profile=shadow_log_audit.SPY_PROFILE,
            claims_builder=shadow_log_audit.spy_claims,
            now=moment,
            local_tz=local_tz,
            market_date=market_date,
            market_phase=market_phase,
            session=session,
        ),
        _shadow_check(
            check_id="greatness_shadow",
            label="Greatness shadow",
            path=diagnostics / "greatness_candidates.json",
            log_path=diagnostics / "greatness_shadow.jsonl",
            log_profile=shadow_log_audit.GREATNESS_PROFILE,
            claims_builder=shadow_log_audit.greatness_claims,
            now=moment,
            local_tz=local_tz,
            market_date=market_date,
            market_phase=market_phase,
            session=session,
            coverage_key="coverage",
        ),
        _registry_check(registry_path, moment, local_tz, market_phase, session),
        _process_check(moment, market_phase, process_snapshot),
        _universe_check(universe_files, market_probe, market_date, moment, local_tz),
        _disk_check(diagnostics, moment),
        _daily_bar_source_check(moment, local_tz, diagnostics / "run_manifests"),
        _daily_bar_units_check(moment, local_tz, diagnostics),
        _outcome_sweep_check(moment, local_tz, diagnostics),
        _evidence_snapshot_check(
            moment, local_tz, staging=diagnostics.parent / "machine_cache" / "evidence_snapshots"
        ),
        _provider_check(latest_manifest, diagnostics / "run_manifests"),
    ]
    # Every sec 6.3 dimension nothing measures is emitted as UNKNOWN, so the
    # roll-up below can see the gaps instead of averaging over what happens to
    # be implemented.
    checks.extend(_inventory_gap_checks({str(check.get("id")) for check in checks}))
    # Runtime health is what the unattended scheduler is judged on; capture
    # readiness is a separate dimension that will read "degraded" every day
    # until the trader has reviewed a first alert. Both are shown, but a
    # cold-start learning ledger must not make the runtime look broken - only
    # a capture check that is outright unhealthy (a held gate that stopped
    # holding) raises the operational verdict.
    overall = worst_status(check["status"] for check in checks)
    capture_checks: list[dict[str, Any]] = []
    evidence_label = ""
    if review_capture:
        # Imported lazily so a broken learning artifact can never take down
        # the operational audit the unattended runtime depends on.
        try:
            from review_capture_audit import build_review_capture_checks

            capture_checks = build_review_capture_checks(now=moment, **review_capture_paths)
        except Exception as exc:
            capture_checks = [
                _check(
                    "review_capture_audit",
                    "Learning capture readiness",
                    STATUS_UNHEALTHY,
                    f"Capture-readiness audit failed: {exc}",
                    source=Path("review_capture_audit.py"),
                )
            ]
        for check in capture_checks:
            if check["id"] == "review_evidence_label":
                evidence_label = str((check.get("details") or {}).get("label") or "")
        if any(check["status"] == STATUS_UNHEALTHY for check in capture_checks):
            overall = STATUS_UNHEALTHY
        checks.extend(capture_checks)

    counts = Counter(check["status"] for check in checks)
    capture_counts = Counter(check["status"] for check in capture_checks)
    return {
        "schema": AUDIT_SCHEMA,
        "generated_at": moment.isoformat(timespec="seconds"),
        "evidence_label": evidence_label,
        "capture_readiness": {
            "status": worst_status(
                (check["status"] for check in capture_checks), default=STATUS_UNKNOWN
            ),
            "evidence_label": evidence_label,
            "summary": _status_summary(capture_counts, len(capture_checks)),
        },
        "timezone": timezone_name,
        "market_date": market_date,
        "market_phase": market_phase,
        "market_session": session.session_label,
        "status": overall,
        "status_precedence": list(STATUS_VALUES),
        # Promotability is deliberately a TOP-LEVEL verdict: plan.md sec 7's
        # evidence floors are counted in the shadow logs, and "is this evidence
        # claimable?" must be answerable without unfolding a check's details.
        "shadow_evidence": _shadow_evidence_view(checks),
        "summary": _status_summary(counts, len(checks)),
        "checks": checks,
        "required_checks": _required_inventory_view(checks),
        "jobs": jobs,
        "latest_manifest": latest_manifest,
        "excluded": ["large setup-tracker payload"],
    }


#: The shadow checks whose raw logs carry plan.md sec 7 promotion evidence.
_SHADOW_CHECK_IDS = ("spy_shadow", "greatness_shadow")


def _shadow_evidence_view(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-engine promotability, rolled up so one damaged log blocks the claim."""
    by_id = {str(check.get("id")): check for check in checks}
    engines: dict[str, Any] = {}
    for check_id in _SHADOW_CHECK_IDS:
        check = by_id.get(check_id)
        details = (check or {}).get("details") or {}
        scan = details.get("log_scan") if isinstance(details.get("log_scan"), dict) else {}
        engines[check_id] = {
            "label": (check or {}).get("label") or check_id,
            "status": (check or {}).get("status") or STATUS_UNKNOWN,
            "promotable": bool(details.get("promotable")),
            "non_promotable_reasons": list(details.get("non_promotable_reasons") or []),
            "log_path": details.get("log_path") or "",
            "log_status": details.get("log_status") or STATUS_UNKNOWN,
            "valid_rows": scan.get("valid_rows"),
            "malformed_lines": scan.get("malformed_lines"),
            "truncated_final_line": scan.get("truncated_final_line"),
            "schemas": scan.get("schemas") or {},
            "engine_versions": scan.get("engine_versions") or {},
            "config_hashes": scan.get("config_hashes") or {},
            "latest_valid_record": scan.get("latest_valid_record") or {},
            "sidecar_reconciliation": details.get("sidecar_reconciliation") or [],
            "session_progress": details.get("session_progress") or {},
        }
    return {
        "promotable": all(engine["promotable"] for engine in engines.values()),
        "engines": engines,
        "note": (
            "Promotability is a property of the RAW logs, not of the writers' sidecars. "
            "plan.md sec 7 evidence floors may not be claimed over a damaged, drifted, or "
            "self-contradicting log; neither shadow engine is promoted by this audit."
        ),
    }


def _status_summary(counts: Counter, total: int) -> dict[str, int]:
    summary = {status: int(counts.get(status, 0)) for status in STATUS_VALUES}
    summary["total"] = int(total)
    return summary


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
            f"{summary['degraded']} degraded, {summary['unhealthy']} unhealthy, "
            f"{summary['unknown']} unknown"
        )
        for check in payload["checks"]:
            print(f"{check['status'].upper():9} {check['label']}: {check['summary']}")
    return 0 if payload["status"] == STATUS_HEALTHY else 1


if __name__ == "__main__":
    raise SystemExit(main())
