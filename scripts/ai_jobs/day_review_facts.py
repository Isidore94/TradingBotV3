"""Refresh the closed session's Day Review facts before narration starts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ai_jobs import ledger


def _closed_target(session: str, now: datetime | None) -> tuple[bool, str]:
    """Refuse a non-session, current or future pack before any service write."""
    from datetime import date

    import market_calendar
    from market_session import normalize_market_local_datetime

    day = date.fromisoformat(session)
    moment = normalize_market_local_datetime(now)
    if not market_calendar.is_session(day):
        return False, f"{session} is not an exchange session"
    completed = market_calendar.last_completed_session(moment)
    if day > completed:
        return False, f"{session} is current, open, or future; last closed is {completed.isoformat()}"
    return True, ""


def run_day_review_facts(
    *,
    session_date: str = "",
    now: datetime | None = None,
    service: Any | None = None,
    root: Path | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Drive the four canonical non-Qt Day Review seams in dependency order.

    The service writes a pack atomically only after it has built the whole one.
    This slot never publishes a replacement itself, which preserves the prior
    verified artifact if any source or the pack build fails.
    """
    session = str(session_date or "")[:10]
    if not session:
        return {"status": ledger.STATUS_FAILED, "reason": "missing target session", "outputs": []}
    try:
        closed, reason = _closed_target(session, now)
        if not closed:
            return {
                "status": ledger.STATUS_SKIPPED,
                "reason": reason,
                "session_date": session,
                "outputs": [],
            }
        if service is None:
            from ui.services.day_review_service import DayReviewService

            service = DayReviewService()
        index = service.build_index_for(session, now=now)
        if not isinstance(index, dict):
            raise RuntimeError("the per-session index was not built")
        bars = service.build_session_bars_for(session, reuse_existing=True)
        reads = service.build_reads_for(session, now=now)
        if reads is None:
            raise RuntimeError("the Day Review reads could not be built")
        pack = service.build_pack_for(session, now=now)
    except Exception as exc:  # The prior atomic pack remains the last good one.
        return {
            "status": ledger.STATUS_FAILED,
            "reason": f"Day Review facts could not be refreshed: {exc}",
            "session_date": session,
            "outputs": [],
        }
    if (
        not isinstance(pack, dict)
        or str(pack.get("session_date") or "")[:10] != session
        or not str(pack.get("inputs_hash") or "")
    ):
        missing = "the day pack"
        if bars is None:
            missing = "the closed-session tape and the day pack"
        return {
            "status": ledger.STATUS_FAILED,
            "reason": f"Day Review facts are incomplete: {missing} could not be built",
            "session_date": session,
            "outputs": [],
        }
    outputs: list[str] = []
    try:
        import day_review_pack

        path = day_review_pack.pack_path(session, root=root)
        if path.is_file():
            outputs.append(str(path))
    except Exception:
        pass
    return {
        "status": ledger.STATUS_OK,
        "reason": "Day Review facts refreshed from the canonical service",
        "session_date": session,
        "pack": pack,
        "reads": len(reads or ()),
        "tape": "available" if bars is not None else "unmeasured",
        "outputs": outputs,
    }
