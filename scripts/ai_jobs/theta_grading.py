"""The overnight `theta_pick_grading` slot - packet WS-TH item 2 (2026-09-12).

Deterministic. No model is called, it costs seconds, and it reads only files the
desk already holds: `theta_picks.jsonl` (what the scan recorded) and the durable
daily-bar store (what price then did). It writes ONE file,
`master_avwap_theta_outcomes.csv`, through a temp-and-rename.

**Where it sits, and why it may not move.** Decision 0018's stage order is
deterministic slots, then narration, then the model-gated slots; a later phase
APPENDS inside its stage and never reorders across stages. `daily_digest` closes
the deterministic block, so this slot goes directly after it and stays ahead of
`ai_summary`. `EXPECTED_SLOT_ORDER` in `tests/test_ai_jobs_runner.py` gains one
name in that position; nothing above it moves.

**Idempotent, because the arithmetic is.** Every run re-grades the whole store
from the bars in hand and rewrites the CSV in full. A mark the calendar has not
reached is `pending` and is simply measured on a later night; a matured mark
never changes, because the endpoint session's close does not move.

Shadow only: nothing here reaches the theta scan, the theta score, the theta
report, a detector, an alert, a watchlist, Focus, the review queue or
`review_policy.json`.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def run_theta_pick_grading(
    *,
    session_date: str = "",
    now: datetime | None = None,
    picks_path: Path | None = None,
    outcomes_path: Path | None = None,
    daily_bars_dir: Path | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Grade every recorded theta pick against the completed daily bars.

    A store that has never been written is `skipped` with a reason, not a
    failure: the recorder only runs on a scan, so the first night after the
    packet lands has nothing to grade and says so.
    """
    import market_calendar
    from project_paths import MASTER_AVWAP_THETA_OUTCOMES_FILE, THETA_PICKS_FILE
    from theta_pick_tracker import (
        STATUS_MEASURED,
        STATUS_PENDING,
        STATUS_UNMEASURED,
        closes_from_daily_bars,
        grade_theta_picks,
        read_theta_picks,
    )

    picks = Path(picks_path or THETA_PICKS_FILE)
    outcomes = Path(outcomes_path or MASTER_AVWAP_THETA_OUTCOMES_FILE)

    rows = read_theta_picks(picks)
    if not rows:
        return {
            "status": "skipped",
            "reason": f"no theta picks recorded yet at {picks.name}",
            "picks": 0,
        }

    # COMPLETED BARS ONLY. `last_completed_session` raises rather than guess a
    # date, and a guessed `as_of` would mark an unfinished session as a break.
    moment = now or datetime.now(market_calendar.MARKET_TZ)
    as_of = market_calendar.last_completed_session(moment)

    graded = grade_theta_picks(
        rows,
        closes_for=closes_from_daily_bars(daily_bars_dir),
        calendar=market_calendar,
        as_of=as_of,
        path=outcomes,
    )
    counts = {
        status: sum(1 for row in graded if row.get("status") == status)
        for status in (STATUS_MEASURED, STATUS_PENDING, STATUS_UNMEASURED)
    }
    _log.info(
        "Theta pick grading: %d pick(s) at %s -> %s", len(graded), as_of.isoformat(), counts
    )
    return {
        "status": "ok",
        "reason": (
            f"graded {len(graded)} theta pick(s) as of {as_of.isoformat()}: "
            f"{counts[STATUS_MEASURED]} measured, {counts[STATUS_PENDING]} pending, "
            f"{counts[STATUS_UNMEASURED]} unmeasured"
        ),
        "picks": len(rows),
        "as_of": as_of.isoformat(),
        "session_date": str(session_date or ""),
        **counts,
    }
