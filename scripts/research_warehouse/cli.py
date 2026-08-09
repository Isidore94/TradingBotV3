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
from datetime import date, datetime, timedelta
from pathlib import Path

try:  # package import
    from . import backup as backup_mod, config, features, occurrences, outcomes, queries, schemas
    from .aggregate import build_derived_bars, build_trading_sessions, build_weekly_bars
    from .ingest_existing import ingest_daily_bars, run_bronze_ingest, run_daily_snapshots
    from .manifest import utc_now
    from .spool import seal_spool
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import backup as backup_mod  # type: ignore
    import config  # type: ignore
    import features  # type: ignore
    import occurrences  # type: ignore
    import outcomes  # type: ignore
    import queries  # type: ignore
    import schemas  # type: ignore
    from aggregate import build_derived_bars, build_trading_sessions, build_weekly_bars  # type: ignore
    from ingest_existing import ingest_daily_bars, run_bronze_ingest, run_daily_snapshots  # type: ignore
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
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) on Windows is TerminateProcess, not a liveness probe:
        # probing the lock holder would kill the running build. Query instead.
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        ERROR_ACCESS_DENIED = 5
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Access denied means the pid exists but is not ours: treat as alive.
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
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


#: Bronze payloads that are JSON *text*, whatever the source file was. A CSV
#: row is wrapped as ``CSV_ROW`` but its payload is a JSON object of the
#: header-to-value mapping, so it decodes the same way.
_JSON_PAYLOAD_FORMATS = {
    schemas.BRONZE_FORMAT_JSON,
    schemas.BRONZE_FORMAT_JSONL,
    schemas.BRONZE_FORMAT_CSV_ROW,
}


def _bronze_payloads(store: ResearchStore, dataset: str) -> list[dict]:
    """Decode a bronze wrap's verbatim payloads back into dicts."""
    rows = []
    try:
        table = store.read_table(dataset, columns=["payload", "payload_format"])
    except Exception:
        return rows
    for payload, fmt in zip(
        table.column("payload").to_pylist(), table.column("payload_format").to_pylist()
    ):
        if str(fmt or "").upper() not in _JSON_PAYLOAD_FORMATS:
            continue
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, dict):
            rows.append(decoded)
    return rows


def anchors_from_bronze(store: ResearchStore) -> list[dict]:
    """Earnings anchors for ``anchor_instance``, from the bronze CSV wrap.

    The trader's ``earnings_avwap_anchors.csv`` carries one current anchor per
    ticker (`ticker`, `anchor_date`, ...), and bronze keeps every version of
    that file it has ever seen. LD-09 scopes the slice to *current and
    previous* earnings, so per ticker the newest distinct ``anchor_date``
    becomes ``EARNINGS_CURRENT`` and the one before it ``EARNINGS_PREVIOUS``;
    older ones are history, not slice anchors. Nothing is invented - a ticker
    bronze has only ever seen once simply has no previous anchor.
    """
    by_symbol: dict[str, set] = {}
    for payload in _bronze_payloads(store, "bronze_earnings_avwap_anchors"):
        symbol = str(payload.get("ticker") or payload.get("symbol") or "").strip().upper()
        raw = str(payload.get("anchor_date") or "").strip()
        if not symbol or not raw:
            continue
        try:
            day = date.fromisoformat(raw[:10])
        except ValueError:
            continue
        by_symbol.setdefault(symbol, set()).add(day)

    anchors = []
    for symbol, days in sorted(by_symbol.items()):
        ordered = sorted(days, reverse=True)
        for anchor_type, day in zip(
            (features.ANCHOR_TYPE_CURRENT, features.ANCHOR_TYPE_PREVIOUS), ordered
        ):
            anchors.append(
                {
                    "symbol": symbol,
                    "anchor_type": anchor_type,
                    "anchor_bar_date": day,
                    "source": "earnings_avwap_anchors.csv",
                }
            )
    return anchors


def cohort_for(store: ResearchStore, day: date) -> list[str]:
    """The session's captured cohort, from its own membership snapshot.

    Read from ``universe_membership_daily`` rather than from today's watchlist
    files: LD-05 makes that snapshot the point-in-time truth about who was a
    member, and a rebuild months later must see the same cohort.
    """
    symbols = set()
    for row in store.read_table(
        "universe_membership_daily", f"year={day.year}", columns=["session_date", "symbol"]
    ).to_pylist():
        session_day = row.get("session_date")
        if isinstance(session_day, datetime):
            session_day = session_day.date()
        if session_day == day:
            value = str(row.get("symbol") or "").strip().upper()
            if value:
                symbols.add(value)
    return sorted(symbols)


def anchor_dates_by_symbol(store: ResearchStore, day: date) -> dict:
    """Current-earnings anchor bar date per symbol, for the daily AVWAP block."""
    dates: dict[str, date] = {}
    for year in (day.year, day.year - 1):
        for row in store.read_table("anchor_instance", f"year={year}").to_pylist():
            if str(row.get("anchor_type") or "") != features.ANCHOR_TYPE_CURRENT:
                continue
            bar_date = row.get("anchor_bar_date")
            if isinstance(bar_date, datetime):
                bar_date = bar_date.date()
            if bar_date is None or bar_date > day:
                continue  # an anchor that had not happened yet is not knowable
            symbol = str(row.get("symbol") or "")
            if symbol and (symbol not in dates or bar_date > dates[symbol]):
                dates[symbol] = bar_date
    return dates


def _run_backups(store: ResearchStore, stamp: datetime) -> dict:
    """Class A/B copies, but only to destinations the trader configured."""
    steps: dict = {}
    class_a = config.backup_class_a_dirs()
    if class_a:
        steps["class_a"] = vars(backup_mod.backup_class_a(store, class_a, now=stamp))
    else:
        steps["class_a"] = {
            "status": "NO_TARGET",
            "message": (
                f"no Class-A backup target: set {config.BACKUP_CLASS_A_SETTING} in "
                f"local_settings.json (or {config.BACKUP_CLASS_A_ENV})."
            ),
        }
    class_b = config.backup_class_b_dir()
    if class_b is not None:
        steps["class_b"] = vars(backup_mod.backup_class_b(store, class_b, now=stamp))
    else:
        steps["class_b"] = {
            "status": "NO_TARGET",
            "message": (
                f"no Class-B backup target: set {config.BACKUP_CLASS_B_SETTING} in "
                f"local_settings.json (or {config.BACKUP_CLASS_B_ENV}). It must be a "
                "second physical disk, never the Drive folder."
            ),
        }
    return steps


def _m5_partitions_for(known: dict, day: date) -> list[str]:
    """Month partitions holding the M5 bars these occurrences can be simulated on.

    ``known`` spans two years of occurrences and BD-53 re-simulates every
    non-terminal one on every build, but the M5 read was the *build day's*
    month alone. An intraday occurrence triggered in any earlier month was
    therefore re-simulated against an empty archive every night, and drew
    conclusions from that absence rather than from its own session (BD-69).
    The trigger's own month is read, plus the following one, because a winter
    session's ETH tail lives there (BD-66).
    """
    months = {f"month={day:%Y-%m}"}
    for row in known.values():
        trigger = row.get("trigger_at")
        if not isinstance(trigger, datetime):
            continue
        entry = trigger.date()
        months.add(f"month={entry:%Y-%m}")
        months.add(f"month={entry + timedelta(days=1):%Y-%m}")
    return sorted(months)


def _run_outcomes(store: ResearchStore, day: date, stamp: datetime, run_id: str) -> dict:
    """Simulate outcomes for occurrences already in the lake.

    Occurrence *ingestion* is still blocked on the BD-44 detector adapter, so
    this step is a clean no-op until something writes ``setup_occurrence``.
    That is a declared gap, not a silent one.
    """
    known = {}
    for year in (day.year, day.year - 1):
        known.update(occurrences.latest_occurrences(store, year))
    if not known:
        return {
            "status": "NO_OCCURRENCES",
            "message": "no setup_occurrence rows yet; the detector adapter is BD-44.",
        }

    symbols = {str(row.get("symbol") or "") for row in known.values()}
    _partitions, d1_by_symbol = features.daily_history_window(store, day)
    d1_by_symbol = {symbol: rows for symbol, rows in d1_by_symbol.items() if symbol in symbols}

    m5_by_symbol: dict[str, list] = {}
    for partition in _m5_partitions_for(known, day):
        for row in store.read_table("bar_m5", partition).to_pylist():
            symbol = str(row.get("symbol") or "")
            if symbol in symbols:
                m5_by_symbol.setdefault(symbol, []).append(row)

    return vars(
        outcomes.build_outcomes(
            store,
            list(known.values()),
            d1_by_symbol=d1_by_symbol,
            m5_by_symbol=m5_by_symbol,
            bands_by_occurrence=_bands_by_occurrence(store, known),
            as_of=stamp,
            now=stamp,
            run_id=run_id,
        )
    )


def _bands_by_occurrence(store: ResearchStore, known: dict) -> dict:
    """AVWAP bands pinned to each occurrence's own trigger session.

    The review's point-in-time note: bands computed later than the trigger
    would be look-ahead, so they are read from the ``feature_snapshot_daily``
    row for the trigger session, never from today's.
    """
    wanted = {}
    for identity, row in known.items():
        trigger = row.get("trigger_at")
        if isinstance(trigger, datetime):
            wanted[identity] = (str(row.get("symbol") or ""), trigger.date())

    if not wanted:
        return {}
    snapshots: dict[tuple[str, date], dict] = {}
    for year in {day.year for _symbol, day in wanted.values()}:
        for row in store.read_table("feature_snapshot_daily", f"year={year}").to_pylist():
            session_day = row.get("session_date")
            if isinstance(session_day, datetime):
                session_day = session_day.date()
            snapshots[(str(row.get("symbol") or ""), session_day)] = row

    bands = {}
    for identity, key in wanted.items():
        snapshot = snapshots.get(key)
        if snapshot is None:
            continue
        resolved = {
            band.upper(): snapshot.get(f"avwape_{band}")
            for band in ("upper_1", "upper_2", "upper_3", "lower_1", "lower_2", "lower_3")
        }
        if any(value is not None for value in resolved.values()):
            bands[identity] = resolved
    return bands


def run_build(
    store: ResearchStore | None = None,
    *,
    session_date: date | None = None,
    now: datetime | None = None,
    run_id: str = "",
    lock_path: Path | None = None,
) -> BuildReport:
    """Run the full EOD step list. Every step is idempotent; a re-run is a no-op.

    The order is a dependency order, not a preference (BD-61): reconcile and
    seal first so the lake is consistent and the session's spooled M5 bars are
    in it; bronze next because the D1 wrap, the universe snapshots and the
    anchors all read wrapped artifacts; then ``bar_d1``, because sessions,
    aggregates and every feature snapshot read it; then sessions and the
    derived/weekly frames the intraday features join to; then anchors, because
    a daily snapshot's AVWAP block needs its ``anchor_instance``; then the
    feature snapshots; then outcomes; then backups, which should copy the
    night's work rather than yesterday's; then retirement last, so nothing is
    swept before it has been backed up.
    """
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
            cohort = cohort_for(target, day)
            report.steps["bar_d1"] = vars(
                ingest_daily_bars(target, cohort, as_of=day, run_id=run_id, now=stamp)
            )
            report.steps["sessions"] = vars(build_trading_sessions(target, day, day, now=stamp, run_id=run_id))
            report.steps["derived"] = vars(build_derived_bars(target, [day], as_of=stamp, now=stamp, run_id=run_id))
            report.steps["weekly"] = vars(build_weekly_bars(target, [day], as_of=stamp, now=stamp, run_id=run_id))
            report.steps["anchors"] = vars(
                features.build_anchor_instances(
                    target, anchors_from_bronze(target), now=stamp, run_id=run_id
                )
            )
            report.steps["features_daily"] = vars(
                features.build_daily_snapshots(
                    target,
                    day,
                    anchors_by_symbol=anchor_dates_by_symbol(target, day),
                    now=stamp,
                    run_id=run_id,
                )
            )
            report.steps["features_intraday"] = vars(
                features.build_intraday_snapshots(target, day, now=stamp, run_id=run_id)
            )
            report.steps["outcomes"] = _run_outcomes(target, day, stamp, run_id)
            report.steps["backups"] = _run_backups(target, stamp)
            report.steps["retired"] = vars(target.collect_retired(now=stamp))
            _record_job("COMPLETED", {"run_id": run_id})
    except SingleFlightError as exc:
        report.status = "REFUSED"
        report.message = str(exc)
        _record_job("SKIPPED", {"reason": "single_flight"})
    return report


# ---------------------------------------------------------------------------
# The backfill jobs (plan sec 5.1-5.2, LD-02/LD-03)
# ---------------------------------------------------------------------------
#: Overnight window: 20:00-02:00 ET minus the ~23:45 TWS restart is ~5.5 usable
#: hours (sec 5.1). The default budget is the part of it a single invocation may
#: spend *waiting* on the pacer; the job stops cleanly when it runs out and the
#: next night resumes from the gaps it recorded.
NIGHTLY_TIME_BUDGET_SECONDS = 4 * 3600
#: The Saturday full-universe sweep is ~4.3 h at the published floor (sec 5.2).
WEEKLY_TIME_BUDGET_SECONDS = 5 * 3600

JOB_NIGHTLY = "nightly"
JOB_WEEKLY = "weekly"
JOB_SEED = "seed"


def _capture_fetcher(pacer=None):
    """The real IB capture fetcher, or ``None`` with a reason.

    Built here rather than inside :mod:`backfill` so the job logic stays
    provider-agnostic and offline-testable (BD-15): the socket lives in exactly
    one adapter. This is the BD-25 path - it has no offline coverage by
    construction, so a failure to build it is reported, never raised.
    """
    try:
        from . import ib_capture
    except ImportError:  # pragma: no cover - scripts/ on sys.path
        import ib_capture  # type: ignore
    spec = ib_capture.backfill_connection_spec()
    transport = ib_capture.build_ib_transport(spec)
    return ib_capture.IbCaptureFetcher(transport, spec=spec, pacer=pacer)


def run_backfill_job(
    store: ResearchStore | None = None,
    *,
    job: str = JOB_NIGHTLY,
    session_date: date | None = None,
    cohort=None,
    fetcher=None,
    time_budget_seconds: float | None = None,
    max_requests: int | None = None,
    now: datetime | None = None,
    run_id: str = "",
    lock_path: Path | None = None,
) -> dict:
    """Run one nightly / weekly / seed backfill pass.

    Holds the **same** single-flight lock as the EOD build: both write the lake,
    and LD-01 allows exactly one writer. A backfill that lands while a build is
    running refuses with a clear message instead of racing it.

    The cohort is the session's own ``universe_membership_daily`` snapshot for
    the nightly job (LD-05 point-in-time membership, the same source the D1 wrap
    uses) and the full captured universe for the weekly sweep.
    """
    target = store if store is not None else ResearchStore.open()
    if target is None:
        return {
            "status": "DISABLED",
            "message": "research_store_dir is not configured; the warehouse is a no-op.",
        }
    stamp = now or utc_now()
    day = session_date or (stamp.date() - timedelta(days=1))

    try:
        with single_flight(lock_path):
            _record_job("RUNNING", {"run_id": run_id, "job": f"backfill_{job}"})
            if job == JOB_SEED:
                report = _run_seed(target, cohort, fetcher=fetcher, now=stamp, run_id=run_id)
            else:
                report = _run_ib_backfill(
                    target,
                    job=job,
                    day=day,
                    cohort=cohort,
                    fetcher=fetcher,
                    time_budget_seconds=time_budget_seconds,
                    max_requests=max_requests,
                    stamp=stamp,
                    run_id=run_id,
                )
            _record_job("COMPLETED", {"run_id": run_id, "job": f"backfill_{job}"})
            return report
    except SingleFlightError as exc:
        _record_job("SKIPPED", {"reason": "single_flight", "job": f"backfill_{job}"})
        return {"status": "REFUSED", "message": str(exc)}


def _run_ib_backfill(
    store: ResearchStore,
    *,
    job: str,
    day: date,
    cohort,
    fetcher,
    time_budget_seconds: float | None,
    max_requests: int | None,
    stamp: datetime,
    run_id: str,
) -> dict:
    try:
        from . import backfill as backfill_mod
    except ImportError:  # pragma: no cover - scripts/ on sys.path
        import backfill as backfill_mod  # type: ignore

    symbols = list(cohort) if cohort is not None else _backfill_cohort(store, job, day)
    if not symbols:
        return {
            "status": "NO_COHORT",
            "message": (
                "no cohort for this session: universe_membership_daily has no rows for "
                f"{day.isoformat()}. Run the EOD build for that session first."
            ),
        }

    if fetcher is None:
        try:
            fetcher = _capture_fetcher()
        except Exception as exc:  # no TWS, no ibapi, bad client id
            return {
                "status": "NO_PROVIDER",
                "message": f"capture transport unavailable: {exc}",
            }

    if job == JOB_WEEKLY:
        report = backfill_mod.run_weekly_universe_sweep(
            store,
            symbols,
            fetcher=fetcher,
            week_ending=day,
            time_budget_seconds=(
                WEEKLY_TIME_BUDGET_SECONDS if time_budget_seconds is None else time_budget_seconds
            ),
            max_requests=max_requests,
            now=stamp,
            run_id=run_id or "weekly_universe_sweep",
        )
    else:
        report = backfill_mod.run_nightly_backfill(
            store,
            symbols,
            fetcher=fetcher,
            session_date=day,
            time_budget_seconds=(
                NIGHTLY_TIME_BUDGET_SECONDS if time_budget_seconds is None else time_budget_seconds
            ),
            max_requests=max_requests,
            now=stamp,
            run_id=run_id or "nightly_backfill",
        )
    payload = vars(report)
    payload["cohort"] = len(symbols)
    payload["session_date"] = day.isoformat()
    return payload


def _run_seed(store: ResearchStore, cohort, *, fetcher, now: datetime, run_id: str) -> dict:
    try:
        from . import backfill as backfill_mod
    except ImportError:  # pragma: no cover - scripts/ on sys.path
        import backfill as backfill_mod  # type: ignore

    symbols = list(cohort) if cohort is not None else _captured_universe(store)
    if not symbols:
        return {"status": "NO_COHORT", "message": "no universe to seed."}
    if fetcher is None:
        return {
            "status": "NO_PROVIDER",
            "message": (
                "the yfinance seed needs an explicit fetcher; it is a one-time "
                "trickle and is never started implicitly (R11)."
            ),
        }
    report = backfill_mod.run_yahoo_seed(
        store,
        symbols,
        fetcher=fetcher,
        spool_dir=config.research_spool_dir(),
        now=now,
        run_id=run_id or "yahoo_m5_seed",
    )
    return vars(report)


def _backfill_cohort(store: ResearchStore, job: str, day: date) -> list[str]:
    """Who this job covers: the session's cohort, or everything already captured."""
    if job == JOB_WEEKLY:
        return _captured_universe(store)
    return cohort_for(store, day)


def _captured_universe(store: ResearchStore) -> list[str]:
    """Every symbol the lake has ever recorded membership for (sec 5.2)."""
    symbols = set()
    for row in store.read_table("universe_membership_daily", columns=["symbol"]).to_pylist():
        value = str(row.get("symbol") or "").strip().upper()
        if value:
            symbols.add(value)
    return sorted(symbols)


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
    backfill_cmd = sub.add_parser(
        "backfill", help="net-new provider capture: nightly ETH, weekly sweep, or the one-time seed"
    )
    backfill_cmd.add_argument("--job", choices=(JOB_NIGHTLY, JOB_WEEKLY, JOB_SEED), default=JOB_NIGHTLY)
    backfill_cmd.add_argument("--session-date", default="", help="defaults to yesterday")
    backfill_cmd.add_argument("--run-id", default="")
    backfill_cmd.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="wall clock this run may spend waiting on the pacer (0 never waits)",
    )
    backfill_cmd.add_argument("--max-requests", type=int, default=None)
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
    if args.command == "backfill":
        day = date.fromisoformat(args.session_date) if args.session_date else None
        report = run_backfill_job(
            store,
            job=args.job,
            session_date=day,
            time_budget_seconds=args.time_budget_seconds,
            max_requests=args.max_requests,
            run_id=args.run_id,
        )
        print(json.dumps(report, indent=2, default=str))
        # A missing cohort or provider is a condition to report, not a crash;
        # only a refused (racing) run is a non-zero exit.
        return 1 if report.get("status") == "REFUSED" else 0
    report = backup_mod.restore_check(
        store, args.target, dataset=args.dataset, partition=args.partition or None
    )
    print(json.dumps(vars(report), indent=2, default=str))
    return 0 if report.passed or report.status == "DISABLED" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
