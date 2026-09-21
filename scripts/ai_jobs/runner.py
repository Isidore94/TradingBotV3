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
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

from ai_jobs import ledger, store, window
from ai_jobs.window import market_now


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
    #: Whether running this slot can start LOCAL INFERENCE (TJ-13A item 1).
    #:
    #: It is what ``--force`` may not buy. *"I always want the bot to run
    #: overnight never during the day so I can restart it or use it for market
    #: prep"* (trader, 2026-09-19; decision 0021 answer 19): a 14 GB model load
    #: by day is the thing the rule is about, so a forced daytime run of a slot
    #: marked here records SKIPPED instead. A deterministic slot costs seconds,
    #: calls no model, and stays forceable by day - which is the repair
    #: ``--force`` exists for.
    #:
    #: Declared per slot rather than inferred from a name, so a later packet
    #: that adds a model to a slot says so here in the same edit.
    uses_model: bool = False
    #: Keyword arguments that make this slot's OWN ``run`` model-free, or None
    #: when there is no such thing (TJ-13A fix round, point 3).
    #:
    #: Only ``daily_digest`` has one today: its fact pack is deterministic and
    #: its narration is a second artifact, and ``run_daily_digest`` already took
    #: ``narrate=False`` before this packet existed. Marking the slot
    #: ``uses_model`` took the fact pack away from a forced daytime run, which
    #: is seconds of work and calls nothing; this gives it back without
    #: reopening the door the flag closed.
    #:
    #: It is KEYWORDS rather than a second callable on purpose. A test that
    #: swaps ``run`` for a spy - ``dataclasses.replace(slot, run=...)`` is the
    #: house pattern - would not swap a second callable, and would reach the
    #: real job through it. There is one callable per slot.
    #:
    #: A slot with no model-free half declares None and keeps being skipped:
    #: there is no summary without a model.
    model_free_kwargs: Mapping[str, Any] | None = None


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
    session_override: str = "",
) -> RunReport:
    """Run every due slot once. Never raises: a crash here is a lost night.

    ``session_override`` (TJ-4 change 4) is the ONE narrow door for "narrate
    THAT day": it applies only to the slot named by ``only`` and reaches
    nothing else. `session_date_for`, `night_kind` and every other slot's
    already-done check are untouched, so a redo cannot re-key the night.
    """

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
                slots,
                now=now,
                force=force,
                only=only,
                ledger_path=ledger_path,
                session_override=session_override,
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
                slots,
                now=now,
                force=force,
                only=only,
                ledger_path=ledger_path,
                session_override=session_override,
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
    session_override: str = "",
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
        # TJ-4 change 4: the Redo button's one named day. It reaches ONLY the
        # slot the operator typed, and from here down that slot's whole run -
        # its already-done check, its attempt cap and every ledger row it
        # writes, the window refusal included - is keyed to the session it
        # WORKED ON. A row claiming tonight's session for work done on last
        # Tuesday's, and an old-day redo skipped because TONIGHT is already
        # covered, are the two dishonest halves of this (reviewer, 2026-09-20).
        # With the keyword at its default `run_session` IS `session_date` and
        # nothing here moves.
        overridden = bool(session_override) and bool(only) and slot.name == only
        run_session = session_override if overridden else session_date
        slot_already = (
            (set() if force else ledger.completed_jobs(run_session, path=ledger_path))
            if overridden
            else already
        )
        if slot.name in slot_already:
            if session_today:
                logging.info(
                    "AI job %s already completed for %s; skipping.", slot.name, run_session
                )
                continue
            # A weekend or holiday firing whose last completed session is
            # already covered has nothing to do. It gets one ledger row saying
            # so -- once, not once per 30-minute repeat, which would bury the
            # ledger under ~27 rows a night.
            reason = (
                f"no session: {market_calendar_describe(moment)}; "
                f"{run_session} is already covered"
            )
            if _already_recorded_no_session(
                slot.name, run_session, path=ledger_path
            ):
                logging.debug("AI job %s: %s (already recorded).", slot.name, reason)
                continue
            row = ledger.record(
                job=slot.name,
                status=ledger.STATUS_SKIPPED,
                session_date=run_session,
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
            if ledger.has_terminal_marker(slot.name, run_session, path=ledger_path):
                logging.debug(
                    "AI job %s: already finished for %s; skipping.", slot.name, run_session
                )
                continue
            cap_reason = ledger.attempt_cap_reason(
                slot.name,
                run_session,
                max_attempts=slot.max_attempts,
                path=ledger_path,
            )
            if cap_reason:
                row = ledger.mark_terminal(
                    job=slot.name,
                    session_date=run_session,
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
        #
        # TJ-13A item 1 narrows what it buys. It still skips the window for a
        # DETERMINISTIC slot - seconds of work, no model, and that is the
        # repair the flag exists for - but it no longer skips the clock for a
        # slot that starts local inference. "Run it now, I know it is 14:00 on
        # a Saturday" is how a 14 GB model load lands in front of the trader's
        # own market prep on the one day they are at the desk all afternoon.
        # A forced model slot still gets --force's other two meanings: the
        # attempt caps and the already-completed check above.
        session_block = window.market_session_block(moment)
        #: Set when this run is the slot's DETERMINISTIC half only. It changes
        #: what the slot is called with and what its ledger row says, never
        #: whether the row counts as coverage.
        model_free = False
        if session_block:
            allowed, reason = False, session_block
        elif force and not slot.uses_model:
            allowed, reason = True, "forced (window checks skipped; session block still enforced)"
        else:
            allowed, reason = window.launch_allowed(
                moment, reserve_minutes=slot.reserve_minutes
            )
            if not allowed and force and slot.uses_model and slot.model_free_kwargs:
                # TJ-13A fix round, point 3. The clock refused this slot because
                # it can call a model - but this one has a deterministic half
                # its own `run` already knows how to produce, so a forced
                # daytime run gets THAT rather than nothing. The model stays
                # unreachable by day; only the part that was never the problem
                # runs.
                allowed = True
                model_free = True
                reason = (
                    "forced by day: deterministic half only "
                    f"({', '.join(f'{k}={v!r}' for k, v in slot.model_free_kwargs.items())}); "
                    "no local inference outside the night window"
                )
        if not allowed:
            row = ledger.record(
                job=slot.name,
                status=ledger.STATUS_SKIPPED,
                session_date=run_session,
                reason=reason,
                path=ledger_path,
            )
            report.results.append(row)
            logging.info("AI job %s skipped: %s", slot.name, reason)
            continue

        started = datetime.now().astimezone()
        clock = time.perf_counter()
        try:
            extra_kwargs = dict(slot.model_free_kwargs or {}) if model_free else {}
            if overridden:
                # The operator asked for ONE day, so a slot that also sweeps a
                # queue of its own (TJ-4's day story) does what it was asked for
                # and nothing else. Only the overridden slot can ever be handed
                # this, so no other job sees a keyword it does not know.
                extra_kwargs["only_this_session"] = True
            outcome = slot.run(session_date=run_session, now=moment, **extra_kwargs) or {}
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
            row_reason = _failure_reason(slot.name, status, outcome)
            if model_free:
                # A reader of this row must never take it for the night's full
                # digest. It says what ran and what was left out, in that order.
                row_reason = (
                    f"{row_reason} [forced daytime run: deterministic facts only, "
                    "narration left out - local inference is night-only]"
                ).strip()
            row = ledger.record(
                job=slot.name,
                status=status,
                session_date=run_session,
                started_at=started,
                model=str(outcome.get("model") or ""),
                reason=row_reason,
                outputs=outcome.get("outputs") or (),
                tokens=outcome.get("tokens") or {},
                # WS-AI1: a slot may add fields of its own to its ledger row -
                # the daily summary adds `completion`. `ledger.record` only ever
                # ADDS (setdefault), so a slot cannot overwrite a ledger field.
                extra=outcome.get("extra") or None,
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
                session_date=run_session,
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
    """The nightly slate, in THREE STAGES (decision 0018, 2026-09-04).

    ``docs/decisions/0018-deterministic-stage-before-narration.md``. The order
    used to be "later phases append; they never reorder", and under that rule
    the narration slots - ``ai_summary``, ``ticker_briefs`` and the bounded
    market-story narration, up to three
    and a half hours of reserve between them - sat ahead of every deterministic
    slot for no reason other than having been written first. On 2026-09-01 the
    run took six hours, and a reservation that cannot fit inside what is left
    of the window records SKIPPED. That put the cheap, deterministic, no-model
    work - the cohort grades, the vocabulary audit, the preference join, the
    evidence report, the fact pack - behind the two jobs most likely to spend
    the night. **No deterministic slot reads either narration slot's output**,
    so the dependency that would have justified the old position does not
    exist.

    So the slate is:

    1. **the deterministic stage** - ``journal_import``, ``journal_auto_tag``,
       the cohort grades, ``note_vocabulary_audit``,
       ``preference_trade_outcomes``, ``evidence_report``, ``daily_digest``.
       Their RELATIVE order is unchanged; it is the appended-only order this
       docstring used to describe and the comments below still argue for each
       position;
    2. **narration** - ``ai_summary`` then ``ticker_briefs``, moved as a UNIT
       to after ``daily_digest``. The digest's own narration is unaffected: it
       is that slot's second artifact and it reads only the fact pack;
    3. **the model-gated slots** - ``journal_enrichment``,
       ``review_policy_draft``, ``setup_research``, unchanged and still last.

    ``journal_import`` is still **first** and ``journal_auto_tag`` still
    **second** (``docs/LOCAL_AI_AUTOMATION_PLAN.md`` §6.4c, promoted into R7
    §6; V2 / decision 0016 answer 10): the import is what puts the night's
    trades in the journal, so anything that reads the journal - which is every
    slot below - would otherwise read yesterday's.

    **A later phase appends inside its stage and never reorders across
    stages.** Every slot's reserve and retry budget is exactly what it was.
    """
    from ai_jobs import (
        briefs,
        cohorts,
        day_review_narration,
        digest,
        enrichment,
        evidence_report,
        exit_note_fields,
        improvement_ideas,
        journal_auto_tag,
        market_story_narration,
        measured_report_publish,
        miss_contrast,
        note_vocabulary_audit,
        observation_tags,
        policy_draft,
        prediction_contrast,
        read_grades_mature,
        setup_research,
        theta_grading,
        week_review_narration,
    )
    from journal_runner import run_nightly_journal_import
    from market_story_rollups import run_market_story_rollups
    from preference_trade_outcomes import run_preference_trade_outcomes

    return [
        # ------------------------------------------------------------------
        # STAGE 1: the deterministic slots. No model, minutes not hours, and
        # every one of them writes evidence some later night depends on.
        # ------------------------------------------------------------------
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
            # Its FACTS are deterministic, but its second artifact is narrated
            # by the medium local model, so this slot CAN start inference and
            # --force must not buy it the daytime clock (TJ-13A item 1).
            uses_model=True,
            # ...and it is the one slot with a real deterministic half:
            # `run_daily_digest` already took `narrate=False` before this
            # packet, and the fact pack is written even when the model is down.
            # So a forced daytime run writes the facts and says so in its
            # ledger row (TJ-13A fix round, point 3).
            model_free_kwargs={"narrate": False},
        ),
        # Packet WS-TH (2026-09-12), APPENDED at the END of the deterministic
        # stage. A later phase appends INSIDE its stage and never reorders
        # across stages, so it goes after `daily_digest` - which closes the
        # block - and stays ahead of `ai_summary`.
        #
        # Last inside the stage rather than beside the cohort grades because
        # nothing here feeds them: it reads `theta_picks.jsonl` and the durable
        # daily bars, and writes one CSV nothing else in the night opens. It
        # deliberately sits AFTER the digest, so a theta grade can never delay
        # the fact pack. Deterministic, no model, seconds of work - hence
        # `journal_import`'s attempt budget rather than the briefs'.
        JobSlot(
            name="theta_pick_grading",
            run=theta_grading.run_theta_pick_grading,
            reserve_minutes=5.0,
            description="Grade the recorded theta picks at 5/10/20 sessions and at expiry (deterministic, no model)",
            max_attempts=3,
        ),
        # TJ-15 (2026-09-19), APPENDED INSIDE stage 1: after the four cohort
        # graders because it reads what a decision turned OUT to be, and
        # deliberately BEFORE `market_story_rollups` + `measured_report` - the pair
        # that CLOSES the stage (WS-10D and WS-RP each pin `measured_report` directly
        # after `market_story_rollups`; the lead moved this slot above the pair at
        # integration, 2026-09-19, when the full suite showed those two pins red).
        #
        # That position is load-bearing. `_STAGE_ONE_LAST_SLOT` is
        # `measured_report` and `_deterministic_stage` walks the slate up to
        # and INCLUDING it, so a slot appended after that name is not in stage 1
        # by that function's reckoning and silently leaves the SUNDAY slate,
        # however deterministic it is. Nothing here reads `measured_report`'s
        # output and nothing there reads this pack, so the two are free to sit
        # in either order - and only one of the two orders runs on a Sunday.
        #
        # Deterministic: no model, seconds of work, one JSON pack beside the
        # digest, and a missing input is a recorded reason on an `ok` row.
        # Hence `journal_import`'s attempt budget rather than the briefs'.
        # TJ-10 (2026-09-20), APPENDED INSIDE stage 1, after the cohort graders
        # and BEFORE `miss_contrast` / `market_story_rollups` / `measured_report`.
        #
        # It closes the market reads whose horizon has matured - a five-session
        # call made on Friday cannot be graded until the fifth session closes,
        # and this is what comes back for it. Beside the cohort graders because
        # it is the same kind of work (a decision, measured after the fact) and
        # ahead of the three closers for the same reason TJ-15 sits there:
        # `_STAGE_ONE_LAST_SLOT` is `measured_report` and `_deterministic_stage`
        # walks up to and INCLUDING it, so a slot appended after that name
        # silently leaves the Sunday slate however deterministic it is. Nothing
        # below reads the ledger and the ledger reads nothing above it, so only
        # the position is a choice.
        #
        # Deterministic: no model, seconds of work, append-only rows in the Day
        # Review read ledger, and an unreachable store is a recorded reason on
        # an `ok` row - never an exception into the runner, and never a verdict.
        # A night that cannot measure writes NOTHING: an `unmeasured` result may
        # not supersede a `pending` row (plan.md sec 5).
        JobSlot(
            name="read_grades_mature",
            run=read_grades_mature.run_read_grades_mature,
            reserve_minutes=2.0,
            description=(
                "Close the market reads whose horizon has matured - re-measure "
                "every open read and append the new verdict (deterministic, no model)"
            ),
            max_attempts=3,
        ),
        JobSlot(
            name="miss_contrast",
            run=miss_contrast.run_miss_contrast,
            reserve_minutes=5.0,
            description=(
                "What the misses had in common - a point-in-time feature "
                "contrast per veto reason and for likes (deterministic, no model)"
            ),
            max_attempts=3,
        ),
        # TJ-16 (2026-09-20), APPENDED INSIDE stage 1, DIRECTLY AFTER
        # `miss_contrast` and still above the `market_story_rollups` /
        # `measured_report` pair that CLOSES the stage.
        #
        # Beside TJ-15 because it is the same question asked of the other half
        # of the record: that pack asks what the MISSES had in common, this one
        # asks what the trader's RIGHT calls had in common, and both answer it
        # through the one `evidence_contrast.contrast`. It reads the Day Review
        # read ledger, which `read_grades_mature` above has already closed for
        # the night, so its position after that slot is a real dependency and
        # not a preference.
        #
        # The position above the closing pair is load-bearing for the same
        # reason it is for TJ-15: `_STAGE_ONE_LAST_SLOT` is `measured_report`
        # and `_deterministic_stage` walks the slate up to and INCLUDING it, so
        # a slot appended after that name silently leaves the SUNDAY slate
        # however deterministic it is.
        #
        # Deterministic: no model, seconds of work, one JSON pack beside the
        # digest, and a missing input is a recorded reason on an `ok` row.
        # Hence `journal_import`'s attempt budget rather than the briefs'.
        JobSlot(
            name="prediction_contrast",
            run=prediction_contrast.run_prediction_contrast,
            reserve_minutes=5.0,
            description=(
                "What leads to a good call - the trader's right reads against "
                "their wrong ones, per point-in-time context field "
                "(deterministic, no model)"
            ),
            max_attempts=3,
        ),
        # Packet WS-10D (2026-09-12), APPENDED at the END of the deterministic
        # stage, after `theta_pick_grading`, and it CLOSES the block. It reads
        # the Market Journal's own entries and the exchange calendar, writes
        # the weekly/monthly/quarterly story packs, and feeds nothing above it
        # - so it sits last inside the stage and still ahead of `ai_summary`.
        #
        # Deterministic: no model is called. The local-AI narration of these
        # packs is a later packet and joins the narration stage, which is
        # exactly why the two are not one slot. Hence `journal_import`'s
        # attempt budget rather than the briefs'.
        JobSlot(
            name="market_story_rollups",
            run=run_market_story_rollups,
            reserve_minutes=5.0,
            description=(
                "Weekly, monthly and quarterly Market Journal story packs "
                "(deterministic, no model; rebuilt only when an input changed)"
            ),
            max_attempts=3,
        ),
        # Packet WS-RP (2026-09-13), APPENDED after `market_story_rollups` and
        # it now CLOSES the deterministic stage. It reads the day's own
        # evidence stores - the journal money, the intraday outcomes, the
        # session-horizon file and the warehouse - and publishes ONE measured
        # report plus its markdown sibling. Everything it reads is written by a
        # slot above it, so it belongs last inside the stage; it calls no model
        # and nothing below it reads its output, so it stays ahead of
        # `ai_summary` rather than joining the narration stage.
        #
        # Deterministic, seconds of work, and a failure never fails the night -
        # hence `journal_import`'s attempt budget rather than the briefs'.
        JobSlot(
            name="measured_report",
            run=measured_report_publish.run_measured_report,
            reserve_minutes=5.0,
            description=(
                "One measured report for the session - the five WISHLIST 10K "
                "answers with their populations (deterministic, no model)"
            ),
            max_attempts=3,
        ),
        # ------------------------------------------------------------------
        # STAGE 2: narration (decision 0018, 2026-09-04)
        #
        # These two were the FIRST two AI slots ever written and sat at the
        # front for that reason alone. They hold up to two and a half hours of
        # reserve between them; on 2026-09-01 the night ran six hours, and a
        # slot whose reserve no longer fits the window records SKIPPED. Nothing
        # deterministic reads either one's OUTPUT files, so nothing downstream
        # of them lost anything by their moving - and everything deterministic
        # gained a night that finishes.
        #
        # They move as a UNIT and keep their order: `ticker_briefs` reads what
        # the summary already established about the day, and the briefs' own
        # resumable design is untouched.
        # ------------------------------------------------------------------
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
            uses_model=True,
            # Unbounded until TJ-13A's review round. `max_attempts=0` was
            # written when a failing summary was EXPENSIVE: it spent hours, so
            # the 30-minute firings could not repeat it inside one night and
            # the missing cap never showed. Item 3 made a dead endpoint cheap -
            # it now gives up on the first call - and cheap plus unbounded is a
            # loop: driven over a weekend night's ~16 firings, a degrading
            # summary RAN TEN TIMES, writing ten `degraded_no_narrative` rows
            # and ten export sets for one session, and Sunday would repeat it.
            #
            # Three, like every other capped slot, and for the same reason: a
            # fault that survives three attempts is deterministic, not
            # transient. It is now a weekly slot, so three attempts is three
            # attempts at ONE session a week. plan.md §12.3: set max_attempts,
            # never 0.
            max_attempts=3,
        ),
        # TJ-4, APPENDED INSIDE stage 2 and deliberately AHEAD of the briefs.
        # Gate #158 reads the ledger for a day story finished before 23:30
        # Pacific and `ticker_briefs` reserves 120 minutes in front of it, so
        # the story goes first. It cannot move further forward: two existing
        # pins say `ai_summary` sits directly after `measured_report`
        # (`test_ws_10d_market_story.py`, `test_ws_rp_shared_report.py`), and
        # decision 0018's stage boundaries do not move for a new slot. It reads
        # only the deterministic day pack, and a failure preserves the last
        # verified story.
        JobSlot(
            name="day_review_narration",
            run=day_review_narration.run_day_review_narration,
            reserve_minutes=10.0,
            description=(
                "Grounded overnight story of one session, plus the rolling D1 "
                "view of what the trader believes lately"
            ),
            max_attempts=3,
            uses_model=True,
        ),
        # TJ-16 item 4 (2026-09-20), APPENDED INSIDE stage 2, AFTER `ai_summary`
        # and BEFORE `ticker_briefs`.
        #
        # After `ai_summary` because WS-10D pins `measured_report` DIRECTLY
        # before it and that pair must stay adjacent; before `ticker_briefs`
        # because the briefs hold two hours of reserve and this is seconds of
        # work per note, so queueing behind them would cost the tags a whole
        # night for nothing.
        #
        # It loads a local MEDIUM model, so `uses_model` is declared honestly
        # and --force may not buy it the daytime clock (TJ-13A item 1). It has
        # no deterministic half - the whole job is the model reading words -
        # so no `model_free_kwargs`. A rejected reply publishes nothing and the
        # last verified file stands.
        JobSlot(
            name="observation_tags",
            run=observation_tags.run_observation_tags,
            reserve_minutes=15.0,
            description=(
                "Grounded codes for the trader's own words - a closed "
                "vocabulary, an exact span per code, and no verdict in the "
                "prompt"
            ),
            max_attempts=3,
            uses_model=True,
        ),
        # TJ-5 (2026-09-20), APPENDED INSIDE stage 2, DIRECTLY after
        # `observation_tags` and DIRECTLY before `ticker_briefs`. That is the
        # position `plan.md` §12.4 TJ-13 item 9 lists for it.
        #
        # Ahead of the briefs because they reserve 120 minutes and the week
        # story is the thing the trader OPENS on a Saturday; it cannot move
        # further forward either, because `measured_report` sits directly before
        # `ai_summary` and two other files pin that pair.
        #
        # It is on the SATURDAY slate only: `WEEKEND_ONLY_SLOTS` is the seam
        # that takes a slot off the weeknight slate, and a weeknight that loaded
        # the largest model the desk owns would have a session behind it and
        # another in front. Sunday picks it up through `_owed_slot_names` when
        # Saturday attempted it and did not finish.
        #
        # `uses_model=True` and NO `model_free_kwargs`: there is no half of a
        # week STORY that runs without a model. The deterministic week and month
        # strip is TJ-5 change 3, it lives on the page and it calls nothing.
        JobSlot(
            name="week_review_narration",
            run=week_review_narration.run_week_review_narration,
            reserve_minutes=week_review_narration.reserve_minutes(),
            description=(
                "The week the trader reads on a Saturday - five day cards "
                "narrated as one grounded story, on the large local model"
            ),
            max_attempts=3,
            uses_model=True,
        ),
        # TJ-9E (2026-09-21), APPENDED INSIDE stage 2, AFTER
        # `week_review_narration` and DIRECTLY BEFORE `ticker_briefs`.
        #
        # That position is the lead's, and it is a choice between two working
        # ones. The packet asked for "directly after `observation_tags`", which
        # would split TJ-5's immediate-adjacency pin
        # (`test_tj5_week_slot_and_slate.py`: the week story sits DIRECTLY after
        # the word tagger). Nothing in the week story reads an exit field in
        # this packet, so there is no dependency to buy by splitting that pair -
        # and this slot still lands where it has to: ahead of `ticker_briefs`,
        # whose two hours of reserve it must not queue behind for a few seconds
        # of work per note. On a weeknight, where `week_review_narration` is
        # weekend-only, it sits directly after `observation_tags` anyway.
        #
        # It loads a local MEDIUM model, so `uses_model` is declared honestly
        # and --force may not buy it the daytime clock (TJ-13A item 1). There is
        # no half of "read the trader's words" that runs without a model, so no
        # `model_free_kwargs`. Nothing waiting means no model is loaded at all,
        # and a rejected reply publishes nothing.
        JobSlot(
            name="exit_note_fields",
            run=exit_note_fields.run_exit_note_fields,
            reserve_minutes=15.0,
            description=(
                "Grounded why / felt / watching drafted from the trader's own "
                "exit note - two closed vocabularies, an exact span per value, "
                "and no outcome in the prompt"
            ),
            max_attempts=3,
            uses_model=True,
        ),
        JobSlot(
            name="ticker_briefs",
            run=briefs.run_ticker_briefs,
            reserve_minutes=120.0,
            description="Medium-tier advisory briefs for Focus/watchlist tickers",
            max_attempts=briefs.TICKER_BRIEFS_MAX_ATTEMPTS,
            uses_model=True,
        ),
        # Phase 0.31 / WISHLIST 10D step 3. Appended at the end of the
        # narration stage. It reads only the deterministic story packs above,
        # and a failure preserves the last verified narration.
        JobSlot(
            name="market_story_narration",
            run=market_story_narration.run_market_story_narration,
            reserve_minutes=15.0,
            description=(
                "Grounded weekly/monthly/quarterly Market Journal narration "
                "and one Trade Mentor coaching question"
            ),
            max_attempts=3,
            uses_model=True,
        ),
        # ------------------------------------------------------------------
        # STAGE 3: the model-gated slots. Unchanged, and still last.
        # ------------------------------------------------------------------
        # LOCAL-AI Phase 3, APPENDED. Later phases append; they never reorder.
        #
        # It runs after the digest because its GATE is the digest's - ten
        # CONSECUTIVE clean fact packs plus the trader's recorded spot-audit
        # (Q4) - and below that gate it calls no model and writes nothing, so
        # an ungated night costs a ledger row and a second.
        # Advisory fields only: R7's I7 keeps tags, notes and planned risk with
        # the trader, and this pass writes its own table instead.
        JobSlot(
            name="journal_enrichment",
            run=enrichment.run_journal_enrichment,
            reserve_minutes=20.0,
            description="Advisory summaries and setup tags for the night's journal rows (gated)",
            max_attempts=3,
            uses_model=True,
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
            uses_model=True,
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
            uses_model=True,
        ),
        # TJ-6 (2026-09-20), APPENDED LAST, inside stage 3.
        #
        # Last because it READS what the rest of the night wrote - the packs,
        # the day stories, TJ-15's contrast pack - and feeds nothing. Nothing in
        # the night runs after it, so a night that ran out of window loses the
        # suggestions and never the evidence.
        #
        # `uses_model=True` and no `model_free_kwargs`: there is no half of an
        # IDEA that runs without a model, so a forced daytime run records
        # SKIPPED and loads nothing (TJ-13A item 1). It is NOT in
        # `WEEKEND_ONLY_SLOTS` - the packet says up to three ideas A NIGHT.
        #
        # What it writes is a SUGGESTION and nothing else: no detector, score,
        # alert, watchlist, Focus, review queue or `review_policy.json` reads
        # the ideas store, and a KEEP is the trader's own click on the card.
        JobSlot(
            name="improvement_ideas",
            run=improvement_ideas.run_improvement_ideas,
            reserve_minutes=improvement_ideas.RESERVE_MINUTES,
            description=(
                "Up to three grounded suggestions a night - each citing the "
                "trader's own evidence, and a process idea naming the one "
                "measurable that will check it"
            ),
            max_attempts=2,
            uses_model=True,
        ),
    ]


#: The three night kinds (TJ-13A item 2, plan.md §12.4 TJ-13 item 6).
NIGHT_WEEKNIGHT = "weeknight"
NIGHT_SATURDAY = "saturday"
NIGHT_SUNDAY = "sunday"
NIGHT_KINDS = (NIGHT_WEEKNIGHT, NIGHT_SATURDAY, NIGHT_SUNDAY)

#: The slots that leave the weeknight slate entirely (plan.md TJ-13 item 8).
#: Measured on the live ledger, 2026-09-19: `ai_summary` ran 12,453-18,540 s a
#: night and ended `degraded_no_narrative` on 09-15, 09-16, 09-17 and 09-18. It
#: is the slot the night cannot afford five times a week, so it runs once, on
#: the Saturday slate, with the whole weekend night in front of it.
#:
#: TJ-5's `week_review_narration` joins it (2026-09-20) rather than growing a
#: second constant beside it: this tuple already MEANS "off the weeknight
#: slate, on Saturday, on Sunday only when owed", which is exactly the week
#: story's cadence. It is weekly work on the largest local model the desk owns,
#: and a Tuesday night has a session behind it and another in front.
WEEKEND_ONLY_SLOTS = ("ai_summary", "week_review_narration")

#: The deterministic stage (decision 0018 stage 1), which every night runs. It
#: ENDS at `measured_report`, which closes that stage today; a later packet
#: appending inside stage 1 lands inside this set automatically because the set
#: is derived from the slate, not written out twice.
_STAGE_ONE_LAST_SLOT = "measured_report"


def _night_evening_date(moment: datetime):
    """The calendar date of the EVENING this night started, in ET.

    A night is one night. The scheduled task fires every 30 minutes from 22:00
    to 06:00 Pacific, so most of a night's firings happen on the FOLLOWING
    calendar date - and in ET, where the window is stored, Saturday night's
    firings are already stamped Sunday. Reading the date off the clock is
    exactly the seam that would file Saturday night's heavy slate under Sunday.

    Noon ET is the split. It sits outside every plausible night window (the
    live one is 01:00-09:00 ET and the shipped default 18:30-08:00), so an
    evening firing keeps its own date and a small-hours one belongs to the day
    before, under either shape.
    """
    moment = market_now(moment)
    if moment.hour >= 12:
        return moment.date()
    return moment.date() - timedelta(days=1)


def night_kind(now: datetime | None = None) -> str:
    """Which of the three nights this moment belongs to. Pure.

    Named on the EXCHANGE CALENDAR, never on the weekday number, so the weekend
    slate follows the sessions:

    * ``weeknight`` - the evening it started was a session.
    * ``sunday``    - it was not, and the next day is. This is the last night
      before trading resumes, so it is the backlog night; a Monday holiday
      moves it to Monday night.
    * ``saturday``  - neither. The first night with no session behind it, which
      a Friday holiday starts a night early. In a Friday-holiday week Friday
      AND Saturday night are both ``saturday``; the second is the resume night
      for anything the first did not finish.

    It decides nothing and writes nothing: it names a night. Raises
    :class:`market_calendar.SessionCalendarError` when the calendar cannot
    answer, because guessing a night kind would guess a slate.
    """
    from market_calendar import is_session

    evening = _night_evening_date(now)
    if is_session(evening):
        return NIGHT_WEEKNIGHT
    if is_session(evening + timedelta(days=1)):
        return NIGHT_SUNDAY
    return NIGHT_SATURDAY


def _deterministic_stage(slots: list[JobSlot]) -> list[JobSlot]:
    """Stage 1, up to and including the slot that closes it."""
    out: list[JobSlot] = []
    for slot in slots:
        out.append(slot)
        if slot.name == _STAGE_ONE_LAST_SLOT:
            return out
    return out


def _owed_slot_names(
    slots: list[JobSlot], session_date: str, ledger_path=None
) -> set[str]:
    """Slots this weekend ATTEMPTED and did not finish, still inside their cap.

    Both weekend nights key to the same session date - Friday's - which is what
    makes the ledger the honest record of what the weekend still owes.

    A slot that never ran is NOT owed: the Sunday slate is the backlog, not a
    second weekly slate, so `ai_summary` skipped on a night nobody tried it is
    not resumed here. A slot that answered (`ok`, or a manual run's
    `manual_test`) is done. A slot that burned its attempts, or whose terminal
    marker is already written, is not re-offered to spend the night re-earning
    the same marker.
    """
    session = str(session_date or "").strip()
    if not session:
        return set()
    try:
        finished = ledger.completed_jobs(session, path=ledger_path)
    except (OSError, ValueError):
        return set()
    owed: set[str] = set()
    for slot in slots:
        if slot.name in finished:
            continue
        try:
            attempts = ledger.attempt_rows(slot.name, session, path=ledger_path)
            if not attempts:
                continue
            if ledger.has_terminal_marker(slot.name, session, path=ledger_path):
                continue
            if slot.max_attempts and ledger.attempt_cap_reason(
                slot.name, session, max_attempts=slot.max_attempts, path=ledger_path
            ):
                continue
        except (OSError, ValueError):
            continue
        owed.add(slot.name)
    return owed


def slots_for(
    kind: str,
    *,
    summary_scopes: tuple[str, ...] | None = None,
    session_date: str = "",
    ledger_path=None,
) -> list[JobSlot]:
    """The slate for one night kind (TJ-13A item 2; plan.md TJ-13 item 6).

    `EXPECTED_SLOT_ORDER` stays the order WITHIN a night and decision 0018's
    stage boundaries do not move: this function CHOOSES which slots a night
    holds and never reorders them. Every slate it returns is a subsequence of
    `default_slots()`.

    * **weeknight** - everything except the weekend-only slot. The deterministic
      stage, the short trader-facing narration, `ticker_briefs` and the
      model-gated stage, exactly as they were.
    * **saturday** - the whole slate plus `weekly_synthesis`, which in 476
      ledger rows had never run because it needed a typed
      ``--weekly-synthesis``. It is a model-gated slot, so it joins the end of
      stage 3.
    * **sunday** - the deterministic stage, plus a retry of any slot this
      weekend attempted, did not finish, and is still inside its cap. Nothing
      heavy is offered a second time just for being heavy.

    An unknown kind RAISES rather than running a guess: a typo that silently
    fell back to a full slate would put a four-hour model job on a weeknight.
    """
    name = str(kind or "").strip().lower()
    if name not in NIGHT_KINDS:
        raise ValueError(
            f"unknown night kind {kind!r}; known kinds are {', '.join(NIGHT_KINDS)}"
        )
    slate = default_slots(summary_scopes=summary_scopes)

    if name == NIGHT_WEEKNIGHT:
        return [slot for slot in slate if slot.name not in WEEKEND_ONLY_SLOTS]

    if name == NIGHT_SATURDAY:
        return slate + optional_slots()

    stage_one = _deterministic_stage(slate)
    owed = _owed_slot_names(
        slate + optional_slots(), session_date, ledger_path=ledger_path
    )
    kept = {slot.name for slot in stage_one}
    out = list(stage_one)
    for slot in slate + optional_slots():
        if slot.name in owed and slot.name not in kept:
            out.append(slot)
            kept.add(slot.name)
    # Keep the night's own order: stage 1 first, then whatever it owes, each in
    # the slate's order. `optional_slots()` sits after stage 3, where it does on
    # the Saturday slate.
    order = {slot.name: index for index, slot in enumerate(slate + optional_slots())}
    out.sort(key=lambda slot: order.get(slot.name, len(order)))
    return out


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
            uses_model=True,
        ),
    ]
