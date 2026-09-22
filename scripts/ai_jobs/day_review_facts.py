"""Refresh the closed session's Day Review facts before narration starts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ai_jobs import ledger


def refresh_recent_packs(
    *, end_session: str, now: datetime | None = None,
    root: Path | None = None, service: Any, sessions: int = 20,
) -> dict[str, Any]:
    """Refresh only already saved, earlier exchange-session packs.

    The current target is built by the caller. Matured calls and late answers
    change the canonical pack hash. An unchanged pack keeps its exact bytes.
    """
    import day_review_pack
    import market_calendar
    from datetime import date

    end = date.fromisoformat(end_session)
    if not market_calendar.is_session(end):
        raise ValueError(f"{end_session} is not an exchange session")
    counts: dict[str, Any] = {
        "refreshed": [], "unchanged": [], "failed": [], "omitted": [],
    }
    cursor = end
    for _ in range(max(0, min(int(sessions), 20)) - 1):
        cursor = market_calendar.previous_session(cursor)
        day = cursor.isoformat()
        path = day_review_pack.pack_path(day, root=root)
        if not path.is_file():
            counts["omitted"].append(day)
            continue
        try:
            before = path.read_bytes()
            built = service.build_pack_for(day, now=now, strict=True, root=root)
            if not isinstance(built, dict) or not built.get("inputs_hash"):
                raise RuntimeError("a source was unreadable or no verified pack was built")
            after = path.read_bytes()
            counts["unchanged" if after == before else "refreshed"].append(day)
        except Exception as exc:  # prior atomic pack is retained
            counts["failed"].append({"session": day, "reason": str(exc)})
    return counts


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


def _tape_has_rows(session: str, result: Any) -> bool:
    """Whether the canonical bar seam actually yielded evidence rows."""
    if isinstance(result, dict):
        return any(bool(rows) for rows in result.values())
    if result is None:
        return False
    try:
        import day_review_bars

        stored = day_review_bars.read_session_bars(session)
    except Exception:
        return False
    return isinstance(stored, dict) and any(bool(rows) for rows in stored.values())


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
        reads = service.build_reads_for(session, now=now, strict=True)
        if reads is None:
            raise RuntimeError("the Day Review reads could not be built")
        pack = service.build_pack_for(session, now=now, strict=True, root=root)
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
    try:
        recent = refresh_recent_packs(
            end_session=session, now=now, root=root, service=service,
        )
    except Exception as exc:
        recent = {"refreshed": [], "unchanged": [], "omitted": [],
                  "failed": [{"session": "recent window", "reason": str(exc)}]}
    failed = recent["failed"]
    return {
        "status": ledger.STATUS_DEGRADED if failed else ledger.STATUS_OK,
        "reason": (
            f"Current facts refreshed; {len(recent['refreshed'])} older changed, "
            f"{len(recent['unchanged'])} unchanged, {len(failed)} failed, "
            f"{len(recent['omitted'])} absent"
        ),
        "session_date": session,
        "pack": pack,
        "recent": recent,
        # `build_reads_for` returns rows newly appended on this pass, not every
        # read the pack holds; say that narrow count honestly.
        "new_grades": len(reads or ()),
        "tape": "available" if _tape_has_rows(session, bars) else "unmeasured",
        "outputs": outputs,
    }
