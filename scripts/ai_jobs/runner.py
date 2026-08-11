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
    #: Attempts this job may spend on one session before the runner declares it
    #: finished for the night. 0 means unlimited, which is the historical
    #: behaviour and still correct for a slot that costs seconds to retry.
    max_attempts: int = 0


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

    @property
    def manual(self) -> int:
        return sum(1 for row in self.results if row.get("status") == ledger.STATUS_MANUAL)

    def summary(self) -> str:
        if not self.store_ok:
            return f"AI jobs did not run: {self.store_reason}"
        return (
            f"AI jobs for session {self.session_date}: "
            f"{self.ran} ok, {self.manual} manual, {self.degraded} degraded, "
            f"{self.failed} failed, {self.skipped} skipped"
        )


def session_date_for(now: datetime | None = None) -> str:
    """The NYSE session this overnight run belongs to.

    The most recent session whose close is at or before the run time. A run at
    01:00 ET Wednesday is processing *Tuesday*; a run at 21:00 ET Saturday is
    still processing *Friday*, because Saturday was never a session.

    This used to be weekday arithmetic -- subtract a day before 17:00, and
    otherwise take today -- with no calendar involved at all. On a Saturday it
    therefore returned Saturday, and three ledger rows claimed `ok` coverage of
    2026-08-08, a date on which the exchange never opened (Sol 5.6
    verification review, item 2).

    Raises :class:`market_calendar.SessionCalendarError` when the calendar
    cannot answer. Callers must fail closed: keying an artifact or an `ok` row
    to a guessed date is exactly the defect being repaired.
    """
    from market_calendar import last_completed_session

    return last_completed_session(window.market_now(now)).isoformat()


def is_session_day(now: datetime | None = None) -> bool:
    """Is the run's own ET date a trading session? Raises if unanswerable."""
    from market_calendar import is_session

    return is_session(window.market_now(now).date())


def market_calendar_describe(now: datetime | None = None) -> str:
    from market_calendar import describe

    return describe(window.market_now(now).date())


def _already_recorded_no_session(job: str, session_date: str, *, path=None) -> bool:
    """Has this job already logged a no-session skip for this session?"""
    target = path if path is not None else ledger.ledger_path(create=False)
    try:
        rows = ledger._read_rows(target)
    except (OSError, ValueError):
        return False
    return any(
        str(row.get("job") or "") == job
        and str(row.get("session_date") or "") == session_date
        and row.get("no_session")
        for row in rows
    )


def run_slots(
    slots: list[JobSlot],
    *,
    now: datetime | None = None,
    force: bool = False,
    only: str = "",
    ledger_path=None,
) -> RunReport:
    """Run every due slot once. Never raises: a crash here is a lost night."""
    from market_calendar import SessionCalendarError

    moment = window.market_now(now)
    # Session identity comes first and fails closed. Without it there is no
    # honest key for an artifact or a ledger row, and writing one anyway is
    # how three `ok` rows came to claim coverage of a Saturday.
    try:
        session_date = session_date_for(moment)
        session_today = is_session_day(moment)
    except SessionCalendarError as exc:
        report = RunReport(session_date="", started_at=moment)
        report.store_ok = False
        report.store_reason = f"session calendar cannot answer: {exc}"
        logging.error(
            "AI job runner: %s. Refusing to run rather than key artifacts to a "
            "guessed session date.",
            report.store_reason,
        )
        return report
    report = RunReport(session_date=session_date, started_at=moment)

    store_ok, store_reason = store.store_available()
    report.store_ok = store_ok
    report.store_reason = store_reason
    if not store_ok:
        # No ledger either -- it lives in the store. Log and leave cleanly.
        logging.error("AI job runner: %s", store_reason)
        return report

    # A manual or forced run publishes real artifacts but never claims the
    # session is covered, so it cannot stand in for the scheduled run -- and,
    # being deliberate, it runs even when the session is already covered.
    manual = bool(force)
    already = set() if force else ledger.completed_jobs(session_date, path=ledger_path)

    for slot in slots:
        if only and slot.name != only:
            continue
        if not slot.enabled:
            continue
        if slot.name in already:
            if session_today:
                logging.info(
                    "AI job %s already completed for %s; skipping.", slot.name, session_date
                )
                continue
            # A weekend or holiday firing whose last completed session is
            # already covered has nothing to do. It gets one ledger row saying
            # so -- once, not once per 30-minute repeat, which would bury the
            # ledger under ~27 rows a night.
            reason = (
                f"no session: {market_calendar_describe(moment)}; "
                f"{session_date} is already covered"
            )
            if _already_recorded_no_session(
                slot.name, session_date, path=ledger_path
            ):
                logging.debug("AI job %s: %s (already recorded).", slot.name, reason)
                continue
            row = ledger.record(
                job=slot.name,
                status=ledger.STATUS_SKIPPED,
                session_date=session_date,
                reason=reason,
                path=ledger_path,
                extra={"no_session": True},
            )
            report.results.append(row)
            logging.info("AI job %s skipped: %s", slot.name, reason)
            continue

        # An exhausted session is finished for this job, and says so once. The
        # marker is what makes every later firing cost about a second, the way
        # a no-session firing already does; --force still overrides it, because
        # an operator asking for a run by hand is the one case where the cap is
        # not protecting anybody.
        if slot.max_attempts and not force:
            if ledger.has_terminal_marker(slot.name, session_date, path=ledger_path):
                logging.debug(
                    "AI job %s: already finished for %s; skipping.", slot.name, session_date
                )
                continue
            cap_reason = ledger.attempt_cap_reason(
                slot.name,
                session_date,
                max_attempts=slot.max_attempts,
                path=ledger_path,
            )
            if cap_reason:
                row = ledger.mark_terminal(
                    job=slot.name,
                    session_date=session_date,
                    reason=cap_reason,
                    path=ledger_path,
                )
                report.results.append(row)
                logging.warning("AI job %s stopped for the session: %s", slot.name, cap_reason)
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
            #
            # An unrecognised status fails CLOSED. It used to coerce to
            # STATUS_OK, so a job reporting a status this runner did not
            # understand -- a typo, a status added by a later phase, a
            # half-written return value -- was recorded as a trustworthy
            # completion and never retried (Sol 5.6 verification review, item
            # 7). "I do not know what happened" is the one thing that must
            # never be filed as success.
            status = str(outcome.get("status") or ledger.STATUS_OK)
            if status not in ledger.RECOGNISED_JOB_STATUSES:
                logging.error(
                    "AI job %s reported an unrecognised status %r; recording it as "
                    "failed rather than assuming success.",
                    slot.name,
                    status,
                )
                outcome = {
                    **outcome,
                    "reason": f"unrecognised job status {status!r}: {outcome.get('reason') or ''}".strip(),
                }
                status = ledger.STATUS_FAILED
            elif manual and status == ledger.STATUS_OK:
                # A deliberate operator run produced real artifacts, but it is
                # not the session's nightly brief and must not be counted as
                # coverage. Degraded and failed keep their own meaning.
                status = ledger.STATUS_MANUAL
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
        JobSlot(
            name="ticker_briefs",
            run=briefs.run_ticker_briefs,
            reserve_minutes=120.0,
            description="Medium-tier advisory briefs for Focus/watchlist tickers",
            max_attempts=briefs.TICKER_BRIEFS_MAX_ATTEMPTS,
        ),
    ]
