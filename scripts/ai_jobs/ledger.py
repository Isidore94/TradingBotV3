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


def ledger_path() -> Path:
    from ai_jobs.store import store_logs_dir

    return store_logs_dir() / LEDGER_NAME


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
) -> dict[str, Any]:
    """Write one ledger row and return it."""
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
    append_row(row, path=path)
    return row


def completed_jobs(session_date: str, *, path: Path | None = None) -> set[str]:
    """Jobs already finished successfully for ``session_date``.

    This is what makes the runner safe to fire repeatedly through the window:
    a second launch sees the completed job and skips it instead of redoing it.
    """
    target = Path(path) if path is not None else ledger_path()
    return {
        str(row.get("job") or "")
        for row in _read_rows(target)
        if str(row.get("session_date") or "") == str(session_date)
        and str(row.get("status") or "") == STATUS_OK
        and str(row.get("job") or "")
    }


def recent_rows(limit: int = 50, *, path: Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path is not None else ledger_path()
    rows = _read_rows(target)
    return rows[-max(1, int(limit)) :]
