"""Named-slot job runner (plan sec 3.4 / 6.3).

The scheduling shape is the one ``master_avwap_mini_pc.py`` established: named
slots, per-slot status, and **skip-don't-pile-up** on overrun. A missed slot is
skipped, never replayed late.

Idempotency is the design choice that makes this robust. The runner is safe to
launch repeatedly through the window -- Task Scheduler fires it every 30
minutes -- because each job asks the ledger whether it already completed for
this session date. Combined with the launch window that means an outage at
01:00 self-heals at 01:30 rather than losing the night, which is the same
lesson the durability packet learned about the trading desk.

Failure philosophy matches the report writers: a failed job means "no digest
tonight" and leaves prior artifacts untouched. It never leaves a partial one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from ai_jobs import ledger, store, window


@dataclass(frozen=True)
class JobSlot:
    """One named unit of overnight work."""

    name: str
    run: Callable[..., dict[str, Any]]
    #: Minutes this job should reserve; the runner refuses to launch it when
    #: less than this remains in the window, rather than running into the open.
    reserve_minutes: float = 15.0
    description: str = ""
    #: False keeps a slot registered but dormant (staged, not yet trusted).
    enabled: bool = True


@dataclass
class RunReport:
    session_date: str
    started_at: datetime
    results: list[dict[str, Any]] = field(default_factory=list)
    store_ok: bool = True
    store_reason: str = ""

    @property
    def ran(self) -> int:
        return sum(1 for row in self.results if row.get("status") == ledger.STATUS_OK)

    @property
    def failed(self) -> int:
        return sum(1 for row in self.results if row.get("status") == ledger.STATUS_FAILED)

    @property
    def skipped(self) -> int:
        return sum(1 for row in self.results if row.get("status") == ledger.STATUS_SKIPPED)

    @property
    def degraded(self) -> int:
        return sum(1 for row in self.results if row.get("status") == ledger.STATUS_DEGRADED)

    def summary(self) -> str:
        if not self.store_ok:
            return f"AI jobs did not run: {self.store_reason}"
        return (
            f"AI jobs for {self.session_date}: "
            f"{self.ran} ok, {self.degraded} degraded, {self.failed} failed, "
            f"{self.skipped} skipped"
        )


def session_date_for(now: datetime | None = None) -> str:
    """The session this overnight run belongs to.

    A run at 01:00 ET Wednesday is processing *Tuesday's* session, so the
    ledger and every artifact are keyed to the day whose evidence is being
    read, not to the wall-clock date of the run.
    """
    moment = window.market_now(now)
    hour_cutoff = 17  # anything before the evening belongs to the previous day
    day = moment.date()
    if moment.hour < hour_cutoff:
        from datetime import timedelta

        day = day - timedelta(days=1)
    return day.isoformat()


def run_slots(
    slots: list[JobSlot],
    *,
    now: datetime | None = None,
    force: bool = False,
    only: str = "",
    ledger_path=None,
) -> RunReport:
    """Run every due slot once. Never raises: a crash here is a lost night."""
    moment = window.market_now(now)
    session_date = session_date_for(moment)
    report = RunReport(session_date=session_date, started_at=moment)

    store_ok, store_reason = store.store_available()
    report.store_ok = store_ok
    report.store_reason = store_reason
    if not store_ok:
        # No ledger either -- it lives in the store. Log and leave cleanly.
        logging.error("AI job runner: %s", store_reason)
        return report

    already = set() if force else ledger.completed_jobs(session_date, path=ledger_path)

    for slot in slots:
        if only and slot.name != only:
            continue
        if not slot.enabled:
            continue
        if slot.name in already:
            logging.info("AI job %s already completed for %s; skipping.", slot.name, session_date)
            continue

        # --force is an operator convenience for the *window* -- "run it now,
        # I know it is 09:00 ET on a Sunday" -- and nothing more. It never
        # reaches the market-session block, which is a plan sec 2 hard rule:
        # during the session the desk runs the full trading complement and a
        # 14GB model load competes with it. A flag that could switch a hard
        # rule off is not a hard rule (checkpoint review 2026-08-08 second
        # review, which found --force bypassing it here and at the post-job
        # break below).
        session_block = window.market_session_block(moment)
        if session_block:
            allowed, reason = False, session_block
        elif force:
            allowed, reason = True, "forced (window checks skipped; session block still enforced)"
        else:
            allowed, reason = window.launch_allowed(
                moment, reserve_minutes=slot.reserve_minutes
            )
        if not allowed:
            row = ledger.record(
                job=slot.name,
                status=ledger.STATUS_SKIPPED,
                session_date=session_date,
                reason=reason,
                path=ledger_path,
            )
            report.results.append(row)
            logging.info("AI job %s skipped: %s", slot.name, reason)
            continue

        started = datetime.now().astimezone()
        clock = time.perf_counter()
        try:
            outcome = slot.run(session_date=session_date, now=moment) or {}
            # A job may report that it published an honestly degraded document
            # rather than a trustworthy one. That is not "ok", and because
            # completed_jobs counts only STATUS_OK, the next firing retries it.
            status = str(outcome.get("status") or ledger.STATUS_OK)
            if status not in {ledger.STATUS_OK, ledger.STATUS_DEGRADED}:
                status = ledger.STATUS_OK
            row = ledger.record(
                job=slot.name,
                status=status,
                session_date=session_date,
                started_at=started,
                model=str(outcome.get("model") or ""),
                reason=str(outcome.get("reason") or ""),
                outputs=outcome.get("outputs") or (),
                tokens=outcome.get("tokens") or {},
                path=ledger_path,
            )
            logging.info(
                "AI job %s finished in %.1fs: %s",
                slot.name,
                time.perf_counter() - clock,
                row["reason"] or "ok",
            )
        except Exception as exc:
            row = ledger.record(
                job=slot.name,
                status=ledger.STATUS_FAILED,
                session_date=session_date,
                started_at=started,
                error=f"{type(exc).__name__}: {exc}",
                path=ledger_path,
            )
            logging.exception("AI job %s failed; prior artifacts are untouched.", slot.name)
        report.results.append(row)

        # Re-read the clock: a long job may have crossed the window end, and
        # sec 6.1 says finish the current call then stop gracefully. --force
        # does not exempt a run from this either: the open arriving mid-run is
        # exactly when stopping matters most.
        moment = window.market_now()
        if window.market_session_block(moment):
            logging.warning("Market session reached; stopping the remaining AI jobs.")
            break

    return report


def default_slots() -> list[JobSlot]:
    """The Phase 1 slate. Later phases append; they never reorder these."""
    from ai_jobs import briefs

    return [
        JobSlot(
            name="ai_summary",
            run=briefs.run_daily_summary,
            reserve_minutes=20.0,
            description="Advisory evidence summary over the day's artifacts",
        ),
    ]
