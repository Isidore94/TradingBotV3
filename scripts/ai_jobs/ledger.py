"""Append-only job ledger (plan sec 3.3 / 6.3).

Every job writes a row whether it succeeds, fails, or is skipped -- a job that
silently did nothing is indistinguishable from a healthy one otherwise, and
this layer runs while nobody is watching. The ledger is also the idempotency
authority: the runner asks it "did this job already complete for this session
date?" rather than keeping a second piece of state that could disagree.

Rows are append-only and never rewritten, matching every other evidence ledger
in the system.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

LEDGER_SCHEMA = "ai_job_ledger_v1"
LEDGER_NAME = "ai_job_ledger.jsonl"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
#: The job ran, published a real document, and that document deliberately
#: carries no narrative -- because there was nothing usable to narrate, or
#: because the model twice cited evidence that does not exist. Distinct from
#: ``ok`` (a trustworthy brief) and from ``failed`` (nothing was published), and
#: deliberately NOT counted as completed, so the next firing in the window
#: retries it (checkpoint review 2026-08-08 second review).
STATUS_DEGRADED = "degraded_no_narrative"
#: A deliberate operator/manual run. It publishes real artifacts, but it is
#: NEVER canonical evidence that a session was covered -- a Saturday-afternoon
#: test run is not that session's nightly brief, and three such rows were
#: sitting in the ledger looking like coverage (Sol 5.6 verification review).
STATUS_MANUAL = "manual_test"
#: A correction. The ledger is append-only, so a row that turns out to have
#: been mis-attributed is annotated by appending one of these, never by
#: rewriting the original.
STATUS_CORRECTION = "correction"

#: Statuses a job may report and the runner will honour. Anything else is
#: recorded as STATUS_FAILED: "I do not know what happened" must never be
#: filed as success.
RECOGNISED_JOB_STATUSES = frozenset(
    {STATUS_OK, STATUS_DEGRADED, STATUS_MANUAL, STATUS_FAILED, STATUS_SKIPPED}
)

#: What counts as "this session is covered". ``manual_test`` is deliberately
#: absent, so a manual run never satisfies the canonical-completion check and
#: the scheduled run still happens.
CANONICAL_COMPLETION_STATUSES = frozenset({STATUS_OK})


def ledger_path(*, create: bool = True) -> Path:
    """Where the ledger lives. ``create=False`` for read-only callers."""
    from ai_jobs.store import store_logs_dir

    return store_logs_dir(create=create) / LEDGER_NAME


def _read_rows(path: Path) -> list[dict[str, Any]]:
    from diagnostics.artifact_io import read_jsonl

    try:
        return read_jsonl(path)
    except OSError:
        return []


def append_row(row: Mapping[str, Any], *, path: Path | None = None) -> Path:
    from diagnostics.artifact_io import append_jsonl_rows

    target = Path(path) if path is not None else ledger_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl_rows(target, (dict(row),), fsync=True)
    return target


def record(
    *,
    job: str,
    status: str,
    session_date: str,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    model: str = "",
    reason: str = "",
    outputs: Iterable[str] = (),
    tokens: Mapping[str, Any] | None = None,
    error: str = "",
    path: Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one ledger row and return it.

    ``extra`` adds fields to the row. Fields are only ever added -- the schema
    is append-only, so nothing here renames or drops an existing key.
    """
    started = started_at or datetime.now().astimezone()
    finished = finished_at or datetime.now().astimezone()
    row = {
        "schema": LEDGER_SCHEMA,
        "job": str(job),
        "status": str(status),
        "session_date": str(session_date),
        "model": str(model or ""),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "reason": str(reason or ""),
        "outputs": [str(value) for value in outputs],
        "tokens": dict(tokens or {}),
        "error": str(error or ""),
    }
    for key, value in dict(extra or {}).items():
        row.setdefault(str(key), value)
    append_row(row, path=path)
    return row


def completed_jobs(session_date: str, *, path: Path | None = None) -> set[str]:
    """Jobs already finished successfully for ``session_date``.

    This is what makes the runner safe to fire repeatedly through the window:
    a second launch sees the completed job and skips it instead of redoing it.
    """
    target = Path(path) if path is not None else ledger_path(create=False)
    completed: set[str] = set()
    for row in _read_rows(target):
        if str(row.get("session_date") or "") != str(session_date):
            continue
        job = str(row.get("job") or "")
        if not job:
            continue
        status = str(row.get("status") or "")
        if status in CANONICAL_COMPLETION_STATUSES:
            completed.add(job)
        elif status == STATUS_CORRECTION and row.get("noncanonical"):
            # Rows are replayed in write order, so a correction retracts the
            # coverage claimed before it and a genuine run after it restores
            # the claim. This is how the append-only ledger annotates a
            # mis-attributed row without rewriting history.
            completed.discard(job)
    return completed


def mark_noncanonical(
    *,
    job: str,
    session_date: str,
    reason: str,
    corrects: Iterable[str] = (),
    path: Path | None = None,
) -> dict[str, Any]:
    """Append a correction retracting a session's coverage claim for ``job``.

    Used when rows were keyed to a date that was never a session, or were
    written by a manual run before ``manual_test`` existed. ``corrects``
    carries the ``finished_at`` stamps of the rows being annotated, so the
    original rows stay exactly as written and the correction says which ones
    it speaks about.
    """
    return record(
        job=job,
        status=STATUS_CORRECTION,
        session_date=session_date,
        reason=reason,
        path=path,
        extra={"noncanonical": True, "corrects": [str(value) for value in corrects]},
    )


def recent_rows(limit: int = 50, *, path: Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path is not None else ledger_path()
    rows = _read_rows(target)
    return rows[-max(1, int(limit)) :]
