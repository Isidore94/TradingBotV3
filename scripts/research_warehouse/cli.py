"""The warehouse build job (plan sec 8.4, Phase 8).

No new process, no daemon, no service: one CLI invoked post-scan or at EOD,
registered in the existing job ledger, holding a **single-flight lock** so a
manual run during a scheduled build refuses with a clear message instead of
two writers racing.

    python -m scripts.research_warehouse.cli build
    python -m scripts.research_warehouse.cli status
    python -m scripts.research_warehouse.cli restore-check --target <dir>

Every command is a no-op with a clear message when ``research_store_dir`` is
unset. The build job is resumable by construction: each step is idempotent, so
an interrupted run (sleep, wake, TWS restart, power loss) simply repeats the
steps that did not finish and rewrites nothing that did.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

try:  # package import
    from . import backup as backup_mod, config, queries
    from .aggregate import build_derived_bars, build_trading_sessions, build_weekly_bars
    from .ingest_existing import run_bronze_ingest, run_daily_snapshots
    from .manifest import utc_now
    from .spool import seal_spool
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import backup as backup_mod  # type: ignore
    import config  # type: ignore
    import queries  # type: ignore
    from aggregate import build_derived_bars, build_trading_sessions, build_weekly_bars  # type: ignore
    from ingest_existing import run_bronze_ingest, run_daily_snapshots  # type: ignore
    from manifest import utc_now  # type: ignore
    from spool import seal_spool  # type: ignore
    from store import ResearchStore  # type: ignore

LOCK_NAME = "research_build.lock"
JOB_TYPE = "research_warehouse_build"


class SingleFlightError(RuntimeError):
    """Another build already holds the lock."""


def _lock_path() -> Path:
    return Path(config.research_spool_dir()) / LOCK_NAME


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


@contextmanager
def single_flight(lock_path: Path | None = None):
    """One build at a time. A dead holder's lock is reclaimed, not obeyed."""
    path = Path(lock_path) if lock_path is not None else _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = {}
        try:
            holder = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            holder = {}
        pid = int(holder.get("pid") or 0)
        # A live holder refuses even when it is this same process: a build
        # must never nest, or two passes write the lake concurrently.
        if pid and _process_alive(pid):
            raise SingleFlightError(
                f"a research warehouse build is already running (pid {pid}, started "
                f"{holder.get('started_at', 'unknown')}). Wait for it, or stop it first."
            )
        # The holder is gone (crash, power loss): reclaim rather than wedge.
        path.unlink(missing_ok=True)
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(handle, json.dumps({"pid": os.getpid(), "started_at": utc_now().isoformat()}).encode("utf-8"))
        os.close(handle)
        yield path
    finally:
        path.unlink(missing_ok=True)


def _record_job(state: str, detail: dict | None = None) -> None:
    """Register the run in the existing job ledger; never fatal."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from job_ledger import JobLedger  # type: ignore

        ledger = JobLedger()
        recorder = getattr(ledger, "record_event", None) or getattr(ledger, "append", None)
        if callable(recorder):
            recorder({"job_type": JOB_TYPE, "state": state, **(detail or {})})
    except Exception:
        pass  # telemetry must never break a build


@dataclass
class BuildReport:
    status: str = "OK"  # OK | DISABLED | REFUSED
    steps: dict = field(default_factory=dict)
    message: str = ""


def run_build(
    store: ResearchStore | None = None,
    *,
    session_date: date | None = None,
    now: datetime | None = None,
    run_id: str = "",
    lock_path: Path | None = None,
) -> BuildReport:
    """Seal the spool, wrap bronze, snapshot, then derive. Idempotent throughout."""
    report = BuildReport()
    target = store if store is not None else ResearchStore.open()
    if target is None:
        report.status = "DISABLED"
        report.message = "research_store_dir is not configured; the warehouse is a no-op."
        return report
    stamp = now or utc_now()
    day = session_date or stamp.date()
    try:
        with single_flight(lock_path):
            _record_job("RUNNING", {"run_id": run_id})
            report.steps["reconcile"] = vars(target.reconcile(job_id=run_id or "build"))
            report.steps["spool"] = vars(seal_spool(target))
            report.steps["bronze"] = [vars(item) for item in run_bronze_ingest(target, run_id=run_id, now=stamp)]
            report.steps["snapshots"] = [
                vars(item) for item in run_daily_snapshots(target, session_date=day, run_id=run_id, now=stamp)
            ]
            report.steps["sessions"] = vars(build_trading_sessions(target, day, day, now=stamp, run_id=run_id))
            report.steps["derived"] = vars(build_derived_bars(target, [day], as_of=stamp, now=stamp, run_id=run_id))
            report.steps["weekly"] = vars(build_weekly_bars(target, [day], as_of=stamp, now=stamp, run_id=run_id))
            report.steps["retired"] = vars(target.collect_retired(now=stamp))
            _record_job("COMPLETED", {"run_id": run_id})
    except SingleFlightError as exc:
        report.status = "REFUSED"
        report.message = str(exc)
        _record_job("SKIPPED", {"reason": "single_flight"})
    return report


def run_status(store: ResearchStore | None = None) -> dict:
    """What the lake holds, straight from the ledger. Reads no bar data."""
    target = store if store is not None else ResearchStore.open()
    if target is None:
        return {"enabled": False, "message": "research_store_dir is not configured."}
    return {
        "enabled": True,
        "root": str(target.root),
        "health": target.health_counts(),
        "datasets": queries.dataset_inventory(target),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="research_warehouse", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="seal the spool, wrap bronze, snapshot, derive")
    build.add_argument("--session-date", default="")
    build.add_argument("--run-id", default="")
    sub.add_parser("status", help="lake inventory and health counters")
    restore = sub.add_parser("restore-check", help="restore one partition to a new root and verify it")
    restore.add_argument("--target", required=True)
    restore.add_argument("--dataset", default="bar_m5")
    restore.add_argument("--partition", default="")

    args = parser.parse_args(argv)
    store = ResearchStore.open()
    if args.command == "status":
        print(json.dumps(run_status(store), indent=2, default=str))
        return 0
    if args.command == "build":
        day = date.fromisoformat(args.session_date) if args.session_date else None
        report = run_build(store, session_date=day, run_id=args.run_id)
        print(json.dumps({"status": report.status, "message": report.message, "steps": report.steps}, indent=2, default=str))
        return 0 if report.status in {"OK", "DISABLED"} else 1
    report = backup_mod.restore_check(
        store, args.target, dataset=args.dataset, partition=args.partition or None
    )
    print(json.dumps(vars(report), indent=2, default=str))
    return 0 if report.passed or report.status == "DISABLED" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
