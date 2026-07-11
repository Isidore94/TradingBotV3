"""Durable job ledger for unattended scheduling (plan.md Phase 2.3-2.5).

Replaces "slots done" mutable JSON as the record of scheduled work: an
append-only JSONL event stream that is replayed into per-job state on load.
A restart can therefore distinguish queued, started, completed, failed,
skipped, and stale work instead of guessing from a set of slot labels.

Jobs are idempotent by (market_date, job_type, slot, config_hash); attempts
are explicit; retry eligibility is a pure function of the last attempt's
state and error class.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LEDGER_SCHEMA = "job_ledger_v1"

# terminal = no further attempts wanted for this key today
_TERMINAL_STATES = {"COMPLETED", "SKIPPED"}
_ACTIVE_STATES = {"QUEUED", "RUNNING"}

# bounded retry budget by error class (plan.md Phase 2.5)
DEFAULT_RETRY_BUDGET = {
    "drive_unavailable": 5,
    "ib_disconnected": 5,
    "provider_rate_limit": 3,
    "no_market_session": 0,
    "bad_local_state": 1,
    "unexpected": 2,
}


def job_key(market_date: str, job_type: str, slot: str, config_hash: str = "") -> str:
    return "|".join((str(market_date), str(job_type), str(slot), str(config_hash)))


@dataclass
class JobState:
    key: str
    market_date: str
    job_type: str
    slot: str
    config_hash: str = ""
    state: str = "QUEUED"
    attempt: int = 0
    run_id: str = ""
    worker_pid: int | None = None
    scheduled_at: str = ""
    started_at: str = ""
    ended_at: str = ""
    error_class: str = ""
    error: str = ""
    history: list[dict] = field(default_factory=list)


class JobLedger:
    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        if self._path is not None and self._path.exists():
            self._replay()

    # ------------------------------------------------------------------
    # event api
    # ------------------------------------------------------------------
    def schedule(self, market_date: str, job_type: str, slot: str, *, config_hash: str = "", now: datetime | None = None) -> JobState:
        return self._apply(
            {
                "event": "scheduled",
                "key": job_key(market_date, job_type, slot, config_hash),
                "market_date": market_date,
                "job_type": job_type,
                "slot": slot,
                "config_hash": config_hash,
                "ts": _ts(now),
            }
        )

    def start(self, key: str, *, run_id: str = "", worker_pid: int | None = None, now: datetime | None = None) -> JobState:
        return self._apply(
            {"event": "started", "key": key, "run_id": run_id, "worker_pid": worker_pid, "ts": _ts(now)}
        )

    def complete(self, key: str, *, run_id: str = "", now: datetime | None = None) -> JobState:
        return self._apply({"event": "completed", "key": key, "run_id": run_id, "ts": _ts(now)})

    def fail(self, key: str, *, error_class: str = "unexpected", error: str = "", now: datetime | None = None) -> JobState:
        return self._apply(
            {"event": "failed", "key": key, "error_class": error_class, "error": str(error)[:500], "ts": _ts(now)}
        )

    def skip(self, key: str, *, reason: str = "", now: datetime | None = None) -> JobState:
        return self._apply({"event": "skipped", "key": key, "error": reason, "ts": _ts(now)})

    def mark_stale_running(self, *, now: datetime | None = None) -> list[JobState]:
        """On startup: anything still RUNNING did not survive the restart."""
        stale = []
        for job in list(self._jobs.values()):
            if job.state == "RUNNING":
                stale.append(
                    self._apply(
                        {"event": "stale", "key": job.key, "error": "runner did not survive restart", "ts": _ts(now)}
                    )
                )
        return stale

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def get(self, key: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(key)

    def is_done(self, key: str) -> bool:
        job = self.get(key)
        return job is not None and job.state in _TERMINAL_STATES

    def is_active(self, key: str) -> bool:
        job = self.get(key)
        return job is not None and job.state in _ACTIVE_STATES

    def should_retry(self, key: str, budget: dict[str, int] | None = None) -> bool:
        """Bounded retry by error class; terminal and active jobs never retry."""
        job = self.get(key)
        if job is None:
            return False
        if job.state in _TERMINAL_STATES or job.state in _ACTIVE_STATES:
            return False
        limits = budget or DEFAULT_RETRY_BUDGET
        allowed = limits.get(job.error_class or "unexpected", limits.get("unexpected", 0))
        return job.attempt <= allowed

    def jobs_for_date(self, market_date: str) -> list[JobState]:
        with self._lock:
            return sorted(
                (j for j in self._jobs.values() if j.market_date == market_date),
                key=lambda j: (j.job_type, j.slot),
            )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _apply(self, event: dict) -> JobState:
        with self._lock:
            job = self._reduce(event)
            self._append(event)
            return job

    def _reduce(self, event: dict) -> JobState:
        key = event["key"]
        job = self._jobs.get(key)
        kind = event["event"]
        if job is None:
            market_date, job_type, slot, config_hash = (key.split("|") + ["", "", "", ""])[:4]
            job = JobState(
                key=key,
                market_date=event.get("market_date", market_date),
                job_type=event.get("job_type", job_type),
                slot=event.get("slot", slot),
                config_hash=event.get("config_hash", config_hash),
            )
            self._jobs[key] = job
        if kind == "scheduled":
            job.scheduled_at = job.scheduled_at or event["ts"]
            if job.state not in _TERMINAL_STATES and job.state != "RUNNING":
                job.state = "QUEUED"
        elif kind == "started":
            job.state = "RUNNING"
            job.attempt += 1
            job.run_id = event.get("run_id", "")
            job.worker_pid = event.get("worker_pid")
            job.started_at = event["ts"]
            job.error_class = ""
            job.error = ""
        elif kind == "completed":
            job.state = "COMPLETED"
            job.run_id = event.get("run_id", job.run_id)
            job.ended_at = event["ts"]
        elif kind == "failed":
            job.state = "FAILED"
            job.error_class = event.get("error_class", "unexpected")
            job.error = event.get("error", "")
            job.ended_at = event["ts"]
        elif kind == "skipped":
            job.state = "SKIPPED"
            job.error = event.get("error", "")
            job.ended_at = event["ts"]
        elif kind == "stale":
            job.state = "STALE"
            job.error = event.get("error", "")
            job.ended_at = event["ts"]
        job.history.append({k: v for k, v in event.items() if k != "key"})
        return job

    def _append(self, event: dict) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"schema": LEDGER_SCHEMA, **event}) + "\n")
        except OSError:
            import logging

            logging.exception("Job ledger append failed (in-memory state still valid).")

    def _replay(self) -> None:
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "key" in event and "event" in event:
                    self._reduce(event)
        except OSError:
            pass


def _ts(now: datetime | None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def default_ledger_path() -> Path:
    try:
        from project_paths import CACHE_DIR

        return Path(CACHE_DIR).parent / "diagnostics" / "job_ledger.jsonl"
    except Exception:
        return Path.home() / ".tradingbotv3" / "job_ledger.jsonl"
