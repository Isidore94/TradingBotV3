"""Named-slot job runner (plan sec 3.4 / 6.3).

The scheduling shape is the one the retired ``master_avwap_mini_pc.py``
established (removed 2026-08-24, P1.5): named
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
from typing import Any, Callable, Mapping

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


#: One AI-jobs runner per machine. See `run_slots` for why this arrived only
#: once the summary started taking hours.
RUNNER_LOCK_KEY = "ai_jobs_runner"
#: The phrase `local_writer_lock` uses when the box has no exclusion primitive
#: at all, as opposed to another process holding one.
NO_PRIMITIVE_MARKER = "no machine-local exclusion primitive is available"


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

    # ONE runner at a time on this machine (2026-08-28). The scheduled task
    # fires every 30 minutes for eight hours, which was harmless while every
    # slot finished in minutes. It stopped being harmless when the summary
    # started reading the evidence in slices: that job runs for hours, the
    # ledger only records a row when a job FINISHES, so the 22:30 firing would
    # find no completion for a job still running at 22:00 and start a second
    # copy of it. Two copies against a server with OLLAMA_NUM_PARALLEL=1 do not
    # go twice as fast; they queue, and both take longer than one would have.
    #
    # A held lock means a run is already in progress, which is a normal state
    # and not a failure - the caller exits cleanly and the next firing tries
    # again. The lock is released by the kernel if the holder is killed, so a
    # crashed run never wedges the night.
    from local_writer_lock import LocalLockUnavailable, local_writer_lock

    try:
        with local_writer_lock(RUNNER_LOCK_KEY, timeout_seconds=0.0):
            return _run_slots_locked(
                slots, now=now, force=force, only=only, ledger_path=ledger_path
            )
    except LocalLockUnavailable as exc:
        # The lock reports both "someone else holds it" and "this box has no
        # exclusion primitive" as the same exception, and the two want opposite
        # answers, so they are told apart by the sentence the module itself
        # writes. `test_a_second_runner_stands_down_while_one_is_working` and
        # `test_no_primitive_runs_unguarded_rather_than_skipping_the_night` pin
        # both branches, so a reworded message breaks a test rather than the
        # behaviour.
        if NO_PRIMITIVE_MARKER in str(exc):
            # Run anyway. The unguarded behaviour was correct for eight months
            # of short jobs and is better than skipping the night outright --
            # but say so, because it is the condition under which two runners
            # can overlap.
            logging.warning(
                "AI jobs: no cross-process lock available (%s); running unguarded.", exc
            )
            return _run_slots_locked(
                slots, now=now, force=force, only=only, ledger_path=ledger_path
            )
        logging.info(
            "AI jobs: another run is already in progress on this machine; leaving it "
            "to finish rather than starting a second copy."
        )
        return RunReport(session_date="", started_at=window.market_now(now))


def _run_slots_locked(
    slots: list[JobSlot],
    *,
    now: datetime | None = None,
    force: bool = False,
    only: str = "",
    ledger_path=None,
) -> RunReport:
    """The body of :func:`run_slots`, always under the machine-local lock."""
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
            # Job implementations historically used display-oriented uppercase
            # values (``OK``/``FAILED``), while the durable ledger vocabulary is
            # lowercase.  Normalize at the seam; validation below still fails
            # closed for genuinely unknown values.
            status = str(outcome.get("status") or ledger.STATUS_OK).strip().lower()
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
                reason=_failure_reason(slot.name, status, outcome),
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


def _failure_reason(job: str, status: str, outcome: Mapping[str, Any]) -> str:
    """The reason recorded on a ledger row, with a floor under failures (R10.0).

    A failing job that records `reason=""` is indistinguishable after the fact
    from one that never ran: `journal_import` failed on 20 nightly runs with a
    blank `error` AND a blank `reason`, and all that survived was "something
    went wrong".

    The explanation was never actually missing. `run_nightly_journal_import`
    returns it in ``messages`` - "journal database requires trader-present
    preparation in the GUI", every night - and this seam read only ``reason``,
    so the diagnostic was produced and then dropped. So: prefer ``reason``, fall
    back to ``messages``, and if a job fails with nothing to say at all, say
    THAT rather than leaving the field empty. Silence and "it declined to
    explain itself" look identical in a file and are completely different to
    debug.

    Successful rows are untouched - this is a floor under failures, not a
    manufactured narrative for every row.
    """
    reason = str(outcome.get("reason") or "").strip()
    if reason:
        return reason
    if status not in {ledger.STATUS_FAILED, ledger.STATUS_DEGRADED}:
        return ""
    messages = outcome.get("messages") or ()
    if isinstance(messages, str):
        messages = [messages]
    text = "; ".join(str(m).strip() for m in messages if str(m).strip())
    if text:
        return text[:500]
    return (
        f"{job} reported {status!r} with no reason and no messages; "
        "the job itself is the only place that knows why"
    )


def default_slots(*, summary_scopes: tuple[str, ...] | None = None) -> list[JobSlot]:
    """The Phase 1 slate, plus R7's journal pull at the front.

    ``journal_import`` is deliberately **first**, and ``journal_auto_tag``
    deliberately **second**. Those two are the only sanctioned exceptions to
    "later phases append; they never reorder these"
    (``docs/LOCAL_AI_AUTOMATION_PLAN.md`` §6.4c, promoted into R7 §6). The
    summary and the ticker briefs read the journal; running them before the
    night's trades are in it means they read yesterday's. It also costs seconds
    rather than the briefs' hours, so putting it first spends nothing.

    ``journal_auto_tag`` (V2, decision 0016 answer 10) is the second exception
    and for the same reason read from the other end: the import is what puts the
    night's trades in the journal, so tagging ahead of it would tag yesterday's,
    and every cohort slot below reads the journal, so tagging after them would
    hand them a journal one night stale. It is deterministic and costs seconds.

    **Every other slot still appends.** Two positions are argued for and written
    down here; a third would mean this list has an ordering nobody can state.
    """
    from ai_jobs import (
        briefs,
        cohorts,
        digest,
        enrichment,
        evidence_report,
        journal_auto_tag,
        note_vocabulary_audit,
        policy_draft,
        setup_research,
    )
    from journal_runner import run_nightly_journal_import
    from preference_trade_outcomes import run_preference_trade_outcomes

    return [
        JobSlot(
            name="journal_import",
            run=lambda **kwargs: run_nightly_journal_import(trigger="nightly"),
            reserve_minutes=5.0,
            description="Broker journal pull, gap self-heal, FX booking and reconciliation",
            max_attempts=3,
        ),
        # V2 item 1, APPENDED RIGHT AFTER `journal_import` and before everything
        # else - which is a POSITION, not a reorder of the slots above it.
        # `journal_import` is what puts the night's trades in the journal, so
        # tagging ahead of it would tag yesterday's; and every cohort slot below
        # reads the journal, so tagging after them would hand them a journal one
        # night stale. Same reasoning that put `journal_import` first.
        #
        # Deterministic and cheap: no model, seconds of work, and its own ledger
        # row so a tagging failure reads as a tagging failure rather than as a
        # journal one.
        JobSlot(
            name="journal_auto_tag",
            run=journal_auto_tag.run_journal_auto_tag,
            reserve_minutes=5.0,
            description=(
                "Provisional setup tags on the night's closed trades "
                "(deterministic, no model; never overwrites a confirmed tag)"
            ),
            max_attempts=3,
        ),
        JobSlot(
            name="ai_summary",
            # ``summary_scopes`` is an OPERATOR override for a manual run, not
            # a configuration knob: the nightly path passes nothing and gets
            # briefs.DEFAULT_SCOPES, so an opt-in scope stays opt-in and
            # cannot leak into the unattended slate by being set once.
            run=(
                briefs.run_daily_summary
                if summary_scopes is None
                else (
                    lambda scopes=tuple(summary_scopes), **kwargs: briefs.run_daily_summary(
                        scopes=scopes, **kwargs
                    )
                )
            ),
            # Resolved at slot-build time from the mode the summary is in:
            # a chunked run is a ~170-minute job, not a 20-minute one.
            reserve_minutes=briefs.summary_reserve_minutes(),
            description="Advisory evidence summary over the day's artifacts",
        ),
        JobSlot(
            name="ticker_briefs",
            run=briefs.run_ticker_briefs,
            reserve_minutes=120.0,
            description="Medium-tier advisory briefs for Focus/watchlist tickers",
            max_attempts=briefs.TICKER_BRIEFS_MAX_ATTEMPTS,
        ),
        # APPENDED, per this function's own rule: "later phases append; they
        # never reorder these". A fourth slot rather than a step bolted onto
        # journal_import, because the slot IS the unit the runner already
        # gives every job - its own ledger row, its own retry budget, its own
        # reserve check, and its own failure isolation. Folding grading into
        # journal_import would make a grading failure read as a journal
        # failure in the ledger, and the two have nothing to do with each
        # other.
        #
        # Last, not first: it costs seconds, nothing downstream reads it, and
        # the briefs must not lose window time to it. Deterministic - no model
        # is called - so it is cheap to retry, hence journal_import's
        # attempt budget rather than the briefs'.
        JobSlot(
            name="veto_cohort_grading",
            run=cohorts.run_veto_cohort_grading,
            reserve_minutes=5.0,
            description="Forward-grade the trader's veto cohort (deterministic, no model)",
            max_attempts=3,
        ),
        # R10.F, APPENDED after the veto slot. Later phases append; they never
        # reorder these. The two cohorts are the two halves of one decision -
        # what the trader rejected and what they endorsed - and audit C1 found
        # only the first half had ever been graded.
        JobSlot(
            name="like_cohort_grading",
            run=cohorts.run_like_cohort_grading,
            reserve_minutes=5.0,
            description="Forward-grade the trader's LIKE cohort (deterministic, no model)",
            max_attempts=3,
        ),
        # P9, APPENDED before the pass slot - which is where it has to be,
        # because it FEEDS it: a capture sidecar ends at the click, so the
        # entry bar the intraday grade asks for is never in it, and this puts
        # the rest of the session on disk before the grade looks. The same
        # night completes and grades; the alternative is the morning after.
        #
        # Later phases append; they never reorder. Deterministic, no model.
        JobSlot(
            name="sidecar_completion",
            run=cohorts.run_sidecar_completion,
            reserve_minutes=5.0,
            description="Finish yesterday's capture sidecars to the close (deterministic, no model)",
            max_attempts=3,
        ),
        # P5, APPENDED after the like slot. Later phases append; they never
        # reorder these. With these two every verdict the trader can record now
        # has a forward record: veto, like, pass, not-today and dislike.
        JobSlot(
            name="pass_cohort_grading",
            run=cohorts.run_pass_cohort_grading,
            reserve_minutes=5.0,
            description="Forward-grade day-trade PASSES (deterministic, no model)",
            max_attempts=3,
        ),
        JobSlot(
            name="rejection_cohort_grading",
            run=cohorts.run_rejection_cohort_grading,
            reserve_minutes=5.0,
            description="Forward-grade NOT-TODAY and DISLIKE (deterministic, no model)",
            max_attempts=3,
        ),
        # P10 A4, APPENDED after the cohort slots. It reads the annotation log
        # and nothing the cohorts produce, so its position among them carries no
        # dependency - but it belongs with them because it is the same KIND of
        # job: deterministic, cheap, no model, its own ledger row and its own
        # failure isolation. Later phases append; they never reorder.
        JobSlot(
            name="note_vocabulary_audit",
            run=note_vocabulary_audit.run_note_vocabulary_audit,
            reserve_minutes=5.0,
            description=(
                "What the trader wrote that no code says - listed, never coded "
                "(deterministic, no model)"
            ),
            max_attempts=3,
        ),
        # P6, APPENDED after the cohort slots and BEFORE the evidence report,
        # which is where it belongs: it READS the cohort outcome files for the
        # paper half of each row. Later phases append; they never reorder.
        JobSlot(
            name="preference_trade_outcomes",
            run=run_preference_trade_outcomes,
            reserve_minutes=5.0,
            description=(
                "What I said, what I did, what happened - statements joined to "
                "trades with a stated match confidence (deterministic, no model)"
            ),
            max_attempts=3,
        ),
        # R10.I, APPENDED last. Later phases append; they never reorder. It runs
        # after both cohorts because it READS what they produced - a report
        # ahead of its inputs would describe last night's evidence.
        #
        # Built ahead of its two-week collection window under the trader's
        # recorded sequencing override (decision record §4). The override covers
        # SEQUENCING only: until the window is met every report states in words
        # that it is scaffolding rather than a finding.
        JobSlot(
            name="evidence_report",
            run=evidence_report.run_evidence_report,
            reserve_minutes=5.0,
            description="Deterministic nightly evidence report (no model)",
            max_attempts=3,
        ),
        # LOCAL-AI Phase 2, APPENDED last. Later phases append; they never
        # reorder these.
        #
        # Last because the fact pack reads what the night produced - the job
        # ledger rows above it included - and a digest written ahead of its own
        # inputs would describe a night that had not happened yet. The ledger
        # row for THIS slot is written after it returns, so a pack never
        # contains its own outcome; that is a known and accepted one-row lag,
        # stated rather than hidden.
        #
        # Its two artifacts fail independently: the fact pack is deterministic
        # and is written even when the model is down, and a failed narration
        # returns `degraded_no_narrative`, which the runner does not count as
        # coverage - so the next firing retries the narration without rewriting
        # the facts (a superseding sibling is written instead; a pack is never
        # edited).
        JobSlot(
            name="daily_digest",
            run=digest.run_daily_digest,
            reserve_minutes=10.0,
            description="Deterministic daily fact pack, plus medium-tier narration",
            max_attempts=3,
        ),
        # LOCAL-AI Phase 3, APPENDED. Later phases append; they never reorder.
        #
        # It runs after the digest because its GATE is the digest's counter -
        # ten clean fact packs - and below that gate it calls no model and
        # writes nothing, so an ungated night costs a ledger row and a second.
        # Advisory fields only: R7's I7 keeps tags, notes and planned risk with
        # the trader, and this pass writes its own table instead.
        JobSlot(
            name="journal_enrichment",
            run=enrichment.run_journal_enrichment,
            reserve_minutes=20.0,
            description="Advisory summaries and setup tags for the night's journal rows (gated)",
            max_attempts=3,
        ),
        # LOCAL-AI Phase 4, APPENDED last.
        #
        # Unlike the pass above, this one RUNS while its gate is unmet - the
        # gate IS two weeks of drafts compared side by side, so a writer that
        # refused would make the window unreachable. It writes
        # `review_policy_draft.json` and archives one copy per session; the live
        # `review_policy.json` is the trader's to save, and no code path here
        # can resolve it.
        JobSlot(
            name="review_policy_draft",
            run=policy_draft.run_review_policy_draft,
            reserve_minutes=10.0,
            description="Draft review policy (ranks and annotates only; never the live file)",
            max_attempts=3,
        ),
        # SETUP DATABASE Phase 6.1, APPENDED last.  The deterministic layer
        # computes every number.  The medium local model only narrates after
        # the evidence floor is met and can never write a live policy.
        JobSlot(
            name="setup_research",
            run=setup_research.run_setup_research,
            reserve_minutes=20.0,
            description="Stop/target recipe research with five-timeframe market context",
            max_attempts=3,
        ),
    ]


def optional_slots() -> list[JobSlot]:
    """Slots that are registered but NEVER nightly.

    The precedent is `--scopes`: an opt-in thing must not be able to become
    unattended by being set once, so this list is constructed per call and
    `default_slots()` never reaches it.

    `weekly_synthesis` (LOCAL-AI §7.3, built 2026-08-24) is the first entry. Its
    cadence is weekly on the weekend surface and its gate is two weeks of graded
    cohort rows; below that gate it writes deterministic scaffolding and asks no
    model anything. Reached by ``run_ai_jobs.py --weekly-synthesis``, which on a
    Saturday morning also wants ``--force`` - the window checks exist for the
    unattended slate, and the market-session block is never skipped by either.
    """
    from ai_jobs import synthesis

    return [
        JobSlot(
            name="weekly_synthesis",
            run=synthesis.run_weekly_synthesis,
            reserve_minutes=15.0,
            description="Weekly rollup over both graded cohorts (gated; medium tier only)",
            max_attempts=3,
        ),
    ]
