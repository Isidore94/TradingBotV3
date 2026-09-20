#!/usr/bin/env python3
"""Standalone overnight AI job runner (plan.md item 13b, Phase 1).

Task Scheduler boots this, it does its work, and it exits. It is deliberately
NOT hosted in the trading GUI:

* the lifecycles are opposed -- the desk is meant to be up during market hours
  and this layer is meant to run when it is not;
* the desk's own launch task relaunches the GUI every 15 minutes through the
  session, which would orphan a long job living inside it;
* a 14GB model load that goes wrong must not be able to take down the window
  the trader watches charts in;
* "no inference during market hours" becomes a scheduler fact rather than only
  a code check.

It imports no Qt and needs only ``requirements-core.txt``.

Exit codes: 0 = nothing to do or everything succeeded; 1 = at least one job
failed; 2 = the AI store was unreachable, so nothing ran.

Usage:
    python scripts/run_ai_jobs.py              # run TONIGHT'S slate, then exit
    python scripts/run_ai_jobs.py --status     # print state, run nothing
    python scripts/run_ai_jobs.py --slot ai_summary
    python scripts/run_ai_jobs.py --slot ticker_briefs
    python scripts/run_ai_jobs.py --force      # re-spend the caps + already-done
    python scripts/run_ai_jobs.py --probe-model large   # MEASURE the big model

Which slate runs is THE NIGHT'S decision, not the operator's (TJ-13A item 2):
`runner.night_kind()` names the night on the exchange calendar and
`runner.slots_for()` builds its slate. A weeknight runs the deterministic stage
plus the short narration; Saturday night carries `ai_summary` and
`weekly_synthesis`; Sunday night is the backlog. Nothing needs typing.

`--force` re-spends the attempt caps and the already-completed check. It does
NOT buy the clock for a slot that calls a local model: inference is night-only,
seven days a week (trader, 2026-09-19).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

#: The ONE slot `--session` may name (TJ-4 change 4). Spelled once.
DAY_REVIEW_SLOT = "day_review_narration"

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if not getattr(sys, "frozen", False) and str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def _print_status() -> int:
    """Print state and run nothing -- including writing nothing.

    --status used to call store_available(), which creates the five-directory
    skeleton and writes a .write_probe file. "Print state, run nothing" that
    mkdirs on a NAS is a contradiction: during market hours it is a write the
    plan sec 2 hard rule never authorised, and on a sleeping share it turns a
    read into a ~20 s spin-up (checkpoint review 2026-08-08 second review).
    """
    from ai_jobs import ledger, runner, store, window

    details = store.get_ai_store_details()
    available, reason = (
        store.store_available(read_only=True)
        if details["enabled"] == "yes"
        else (False, details["error"] or "unset")
    )
    from market_calendar import SessionCalendarError

    # Session identity can refuse to answer, and --status must report that
    # rather than crash or guess.
    try:
        session_date = runner.session_date_for()
        session_note = runner.market_calendar_describe()
    except SessionCalendarError as exc:
        session_date = ""
        session_note = f"session calendar cannot answer: {exc}"

    # TJ-13A item 2: --status reports TONIGHT'S slate, not the full slot list.
    # An operator reading this at 21:00 wants to know what is about to run, and
    # since the slate depends on the night, printing `default_slots()` would
    # promise `ai_summary` on a Tuesday and omit `weekly_synthesis` on a
    # Saturday. Both helpers fail safe, so --status still prints.
    kind = _night_kind_or_weeknight()
    try:
        slate = runner.slots_for(kind, session_date=session_date)
    except Exception as exc:  # --status prints state; it never crashes
        slate = []
        session_note = f"{session_note}; slate unavailable: {exc}".strip("; ")
    payload = {
        "session_date": session_date,
        "session_note": session_note,
        "store": details,
        "store_available": available,
        "store_reason": reason,
        "window": window.describe_window(),
        "night_kind": kind,
        "slots": [
            {"name": slot.name, "reserve_minutes": slot.reserve_minutes,
             "enabled": slot.enabled, "uses_model": slot.uses_model,
             "description": slot.description}
            for slot in slate
        ],
    }
    if available and session_date:
        session = session_date
        try:
            path = ledger.ledger_path(create=False)
            payload["completed_today"] = sorted(ledger.completed_jobs(session, path=path))
            payload["recent"] = [
                {k: row.get(k) for k in ("job", "status", "session_date", "finished_at", "reason", "error")}
                for row in ledger.recent_rows(10, path=path)
            ]
        except (OSError, ValueError) as exc:
            payload["ledger_error"] = str(exc)
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _run_model_probe(tier: str, *, force: bool = False) -> int:
    """TJ-13B: measure one local tier on this desk and print what it cost.

    A COMMAND, not a slot. It builds no slate, runs no other job, and exits
    non-zero when it refused - "it printed something and exited 0" and "it
    measured the model" must not look alike to whoever typed it.

    The real machine lock is handed to the probe here, because this is the one
    caller that actually loads a model: while the nightly runner is working,
    the probe stands down rather than loading a second model beside it.

    ``--force`` is passed through (2026-09-20 review): the probe's own refusal
    says "pass --force to measure it again", and until the flag reached it that
    sentence was false - a desk with one measurement on the session could never
    be re-measured from the command line. Here as everywhere else, --force
    re-spends ONLY the already-measured check. It never buys the clock and
    never beats a held lock.
    """
    from ai_jobs import ledger, model_probe

    try:
        outcome = model_probe.run_model_probe(
            tier=tier, force=force, lock=model_probe.runner_lock
        )
    except ValueError as exc:
        print(f"model probe refused: {exc}")
        return 1

    measurement = outcome.get("measurement")
    if measurement:
        print(json.dumps(measurement, indent=2, default=str))
    print(f"model probe [{outcome.get('status')}]: {outcome.get('reason')}")
    if outcome.get("status") != ledger.STATUS_MANUAL:
        return 1
    reserve = model_probe.reserve_minutes_from_probe(tier=tier)
    print(
        "week story reserve derived from this measurement: "
        + (f"{reserve} min" if reserve is not None else "not derivable (throughput unmeasured)")
    )
    return 0


def _night_kind_or_weeknight() -> str:
    """The night's kind, failing toward the LIGHT slate.

    An unanswerable calendar must not choose a four-hour model job: the
    weeknight slate is the one without `ai_summary` or `weekly_synthesis` on
    it, so "we could not tell" spends the least. The run itself still fails
    closed a moment later - `run_slots` refuses outright rather than keying
    artifacts to a guessed session date.
    """
    from ai_jobs import runner

    try:
        return runner.night_kind()
    except Exception as exc:
        # Deliberately broad: this is slate SELECTION, and any failure here
        # must cost the heavy slots rather than the whole night. The calendar's
        # own SessionCalendarError is the expected one.
        logging.warning(
            "AI jobs: could not name tonight (%s); using the weeknight slate, "
            "which carries no weekly model job.",
            exc,
        )
        return runner.NIGHT_WEEKNIGHT


def _session_date_or_blank() -> str:
    """Friday's session for a weekend night, or "" when the calendar cannot say.

    Only the Sunday slate reads it, and a blank means "no backlog to resume" -
    the deterministic stage still runs.
    """
    from ai_jobs import runner
    from market_calendar import SessionCalendarError

    try:
        return runner.session_date_for()
    except SessionCalendarError:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="print state and exit without running anything")
    parser.add_argument("--slot", default="", help="run only this named slot")
    parser.add_argument("--force", action="store_true",
                        help="manual run: spend the attempt caps and the "
                             "already-completed check again. It does NOT buy the "
                             "clock for a slot that calls a local model - local "
                             "inference is night-only, seven days a week, so a "
                             "forced model slot outside the night window records "
                             "skipped. A deterministic slot (seconds, no model) is "
                             "still forceable by day. Never skips the "
                             "market-session block or its pre-open reserve, and its "
                             "ledger row is manual_test, which never counts as "
                             "session coverage")
    parser.add_argument(
        "--scopes",
        default="",
        help="comma-separated evidence scopes for the ai_summary slot, "
             "overriding the nightly default. Manual runs only - this is how "
             "an opt-in scope such as trader_judgement is exercised on a "
             "weekend without adding it to the unattended slate.",
    )
    parser.add_argument(
        "--weekly-synthesis",
        action="store_true",
        help="run ONLY the weekly trader-judgement synthesis (LOCAL-AI sec 7.3). "
             "Since TJ-13A the SATURDAY-night slate runs it with no typed "
             "command, so this flag is the operator's way to run it alone. Below "
             "its two-week graded-cohort gate it writes deterministic "
             "scaffolding and calls no model. It calls a model above that gate, "
             "so --force will not start it by day.",
    )
    parser.add_argument(
        "--probe-model",
        default="",
        metavar="TIER",
        help="MEASURE one local model tier on this desk and record the numbers "
             "in the job ledger, then exit. It runs no slate and no other job. "
             "A probe is a model LOAD, so it runs only inside the night window "
             "and only when no AI job holds the machine lock; it reads a COPY "
             "of one week's fact packs and never the live store. Exit code 0 "
             "means it measured, 1 means it refused and says why - it is not "
             "night, a job is already running, or this session was measured "
             "already, which is the one --force re-spends. Tiers: large, "
             "medium.",
    )
    parser.add_argument(
        "--session",
        default="",
        metavar="YYYY-MM-DD",
        help="narrate THIS session instead of the one the clock names. Accepted "
             "ONLY together with `--slot day_review_narration` - it is the Day "
             "Review page's Redo button, which asks for one named day. It is "
             "handed to that one slot and to nothing else: the night's own "
             "session date, the night kind and every other slot's "
             "already-done check are untouched, and the night-only rule still "
             "holds, so a forced daytime redo still records skipped.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    # TJ-4 change 4. Refused rather than ignored: a flag that silently did
    # nothing for every other slot would make `--session` look like a general
    # override of the night's own session date, which it is not.
    session_override = str(args.session or "").strip()
    if session_override:
        if args.slot != DAY_REVIEW_SLOT:
            parser.error(
                f"--session is accepted only with --slot {DAY_REVIEW_SLOT}; it "
                "names the one day that slot narrates and reaches no other job"
            )
        # ONE rule for what a redo may name, shared with the Day Review page's
        # own button: exactly `YYYY-MM-DD`, a real exchange session, and one
        # that has CLOSED. The CLI used to check only the first two, so the
        # picker's provisional Today entry parsed here and the slot then
        # answered `skipped` - true, but the operator was told nothing at the
        # door (reviewer round 3, 2026-09-20).
        import day_review_pack

        try:
            session_override = day_review_pack.validated_session(session_override)
        except ValueError as exc:
            parser.error(f"--session: {exc}")

    if args.status:
        return _print_status()

    if args.probe_model:
        return _run_model_probe(str(args.probe_model).strip(), force=args.force)

    from ai_jobs import runner

    scopes = tuple(
        part.strip() for part in str(args.scopes or "").split(",") if part.strip()
    )
    if scopes:
        import ai_summary

        unknown = [scope for scope in scopes if scope not in ai_summary.SCOPE_LABELS]
        if unknown:
            parser.error(
                f"unknown scope(s) {unknown}; known: {sorted(ai_summary.SCOPE_LABELS)}"
            )
    if args.slot:
        # TJ-13A review round. A TYPED slot is the operator's explicit choice
        # and resolves against every registered slot, not against tonight's
        # slate. It used to be filtered by the slate, so `--slot ai_summary` on
        # a weeknight matched nothing, ran nothing and exited 0 - silently,
        # while this module's own docstring advertised that exact command. "I
        # typed it wrong" and "it ran and found nothing" must not look alike.
        #
        # This widens what can be NAMED, never what can RUN: a model slot named
        # by day is still refused by the night-only window inside `run_slots`,
        # and its ledger row still says so.
        registered = runner.default_slots(summary_scopes=scopes or None) + runner.optional_slots()
        known = [slot.name for slot in registered]
        if args.slot not in known:
            parser.error(
                f"unknown slot {args.slot!r}; known slots: {', '.join(sorted(known))}"
            )
        slots = registered
    elif args.weekly_synthesis:
        # Constructed per call and ONLY here when it is asked for by name.
        slots = runner.optional_slots()
    else:
        # TJ-13A item 2: THE NIGHT picks the slate, not the operator. A
        # weeknight runs the deterministic stage plus the short narration;
        # Saturday night carries `ai_summary` and `weekly_synthesis`; Sunday
        # night is the backlog. Nothing needs typing for any of it.
        slots = runner.slots_for(
            _night_kind_or_weeknight(),
            summary_scopes=scopes or None,
            session_date=_session_date_or_blank(),
        )
    report = runner.run_slots(
        slots, force=args.force, only=args.slot, session_override=session_override
    )
    logging.info("%s", report.summary())

    if not report.store_ok:
        return 2
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
