"""Finish the pending M5 outcomes before the nightly reports read them.

This is an owner for scheduling only.  The scanner's canonical finalizer owns
the locks, recovery, CSV transaction and terminal-status semantics.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Callable

from ai_jobs import ledger


def _session_day(value: str) -> date:
    return date.fromisoformat(str(value or "")[:10])


def _default_factory():
    from bounce_bot_lib.legacy import BounceBot

    return BounceBot.for_outcome_sweep()


def _autorun_enabled() -> bool:
    """Read the existing switch without constructing a scanner."""
    try:
        from project_paths import get_local_setting

        raw = str(get_local_setting("outcome_sweep_autorun", "") or "").strip().lower()
    except Exception:
        return False
    return raw in {"on", "1", "true", "yes"}


def _closed_target_gate(day: date, now: datetime | None) -> tuple[bool, str]:
    """Validate the target and the scanner's own close-plus-35 boundary.

    This uses the same market-local clock and early-close calendar that
    ``BounceBot.actual_session_close`` uses, but does so before the lightweight
    factory loads a checkpoint.  A night slot must not construct a writer to
    learn that this session has not closed.
    """
    import market_calendar
    from market_early_close import session_close
    from market_session import normalize_market_local_datetime

    moment = normalize_market_local_datetime(now)
    if not market_calendar.is_session(day):
        return False, f"{day.isoformat()} is not an exchange session"
    completed = market_calendar.last_completed_session(moment)
    if day > completed:
        return False, f"{day.isoformat()} is current, open, or future; last closed is {completed.isoformat()}"
    close = session_close(day).astimezone(moment.tzinfo)
    due = close + timedelta(minutes=35)
    if moment < due:
        return False, f"canonical close+35 gate has not opened (due {due.isoformat(timespec='minutes')})"
    return True, ""


def run_outcome_sweep(
    *,
    session_date: str = "",
    now: datetime | None = None,
    bot_factory: Callable[[], Any] | None = None,
    autorun_enabled: bool | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Run the scanner's finalizer for one verified completed session.

    The factory stays behind both the calendar and autorun gates.  In
    particular, a disabled job must not load a checkpoint, start a client, or
    repair anything merely to decide that it is disabled.
    """
    try:
        day = _session_day(session_date)
        allowed, reason = _closed_target_gate(day, now)
        if not allowed:
            return {"status": ledger.STATUS_SKIPPED, "reason": reason, "session_date": day.isoformat()}
    except Exception as exc:  # Calendar uncertainty must never launch a sweep.
        return {
            "status": ledger.STATUS_FAILED,
            "reason": f"exchange calendar could not verify the target session: {exc}",
            "session_date": str(session_date or "")[:10],
        }

    enabled = _autorun_enabled() if autorun_enabled is None else bool(autorun_enabled)
    if not enabled:
        return {
            "status": ledger.STATUS_SKIPPED,
            "reason": "outcome sweep autorun is disabled",
            "session_date": day.isoformat(),
        }

    try:
        bot = (bot_factory or _default_factory)()
        resolver = getattr(bot, "resolve_unfinished_finalizations", None)
        recovery = dict(resolver()) if callable(resolver) else {}
        counts = dict(bot.sweep_pending_bounce_outcomes(now=now, wait_for_scan_window=False))
    except Exception as exc:  # A failed canonical finalizer is a failed slot.
        return {
            "status": ledger.STATUS_FAILED,
            "reason": f"canonical outcome sweep failed: {exc}",
            "session_date": day.isoformat(),
        }

    outcome = {**counts, "session_date": day.isoformat(), "recovery": recovery}
    if counts.get("deferred"):
        outcome.update({
            "status": ledger.STATUS_SKIPPED,
            "reason": f"canonical outcome sweep deferred: {counts['deferred']}",
        })
        return outcome
    failed = int(counts.get("failed") or 0)
    commit_failed = int(counts.get("commit_failed") or 0)
    if failed or commit_failed:
        outcome.update({
            "status": ledger.STATUS_FAILED,
            "reason": (
                "canonical outcome sweep did not commit every final "
                f"(failed={failed}, commit_failed={commit_failed})"
            ),
        })
        return outcome
    outcome.update({"status": ledger.STATUS_OK, "reason": "canonical outcome sweep completed"})
    return outcome
