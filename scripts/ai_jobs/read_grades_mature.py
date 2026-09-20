"""`read_grades_mature` - close the market reads whose horizon has matured (TJ-10).

The trader's five-session call cannot be graded on the day it is made. The Day
Review post-close tick writes it as ``pending <date>`` and this slot is what
comes back for it: every night it re-reads the bars, re-measures every OPEN read
in the ledger, and appends a NEW row naming the one it supersedes.

Deterministic: ``uses_model=False``, no inference, no network, no provider,
seconds of work. It calls exactly one function - `market_read_grades.regrade_matured`
- and that function is pure arithmetic over bars read from files the desk already
has.

Three rules it is built around, all of them from the fix round 2026-09-20:

* **An `unmeasured` result never supersedes a `pending` row.** Missing data is
  uncertainty, never confirmation (plan.md sec 5). A verdict may only move UP
  (`market_read_grades.verdict_rank`), so a night on which the daily store is
  unreachable writes NOTHING rather than retiring a correct pending read with an
  `unmeasured` one. That is the exact defect this slot exists beside: measured
  on the desk, `chart_snapshot.load_d1_bars` answers EMPTY for SPY, QQQ, IWM and
  VXX, so a night that trusted it would have retired every open read.
* **It never costs the night.** A missing ledger, an unreadable store and a
  session with nothing open are each a recorded REASON on an `ok` row. Nothing
  here raises into the runner.
* **Nothing it writes reaches a decision.** The ledger is evidence: no detector,
  score, gate, alert, watchlist, Focus list, review queue or `review_policy.json`
  reads it, and this slot sends no push.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

_log = logging.getLogger(__name__)

#: What the runner records when the ledger holds nothing open.
NOTHING_OPEN = "no open read had matured"


def run_read_grades_mature(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Any = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Re-measure every open market read. Deterministic, model-free, never fatal.

    `session_date` is accepted because every slot is handed one; this job is not
    scoped to a session - a read's horizon matures on its own clock, and the
    ledger is walked whole.
    """
    moment = now or datetime.now()
    try:
        import market_read_grades as grader
    except Exception as exc:  # noqa: BLE001 - an unimportable grader is a reason
        return {
            "status": "ok",
            "model": "",
            "reason": f"no re-grade: the read grader could not be imported ({exc})",
            "outputs": [],
        }

    try:
        written = grader.regrade_matured(moment, root=root)
    except Exception as exc:  # noqa: BLE001 - arithmetic must not cost the night
        _log.exception("read_grades_mature: the ledger could not be re-graded.")
        return {
            "status": "ok",
            "model": "",
            "reason": f"no re-grade: {type(exc).__name__}: {exc}",
            "outputs": [],
        }

    if not written:
        return {"status": "ok", "model": "", "reason": NOTHING_OPEN, "outputs": []}

    sessions = sorted({str(row.get("session") or "") for row in written})
    verdicts: dict[str, int] = {}
    for row in written:
        verdict = str(row.get("verdict") or "").split(" ", 1)[0].split(":", 1)[0]
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
    printed = ", ".join(f"{count} {name}" for name, count in sorted(verdicts.items()))
    try:
        outputs = [str(grader.reads_path(session, root=root)) for session in sessions]
    except Exception:  # noqa: BLE001 - a path is a courtesy, never the work
        outputs = []
    return {
        "status": "ok",
        "model": "",
        "reason": (
            f"closed {len(written)} matured read(s) over {len(sessions)} session(s) "
            f"({printed}); every row appended, none rewritten"
        ),
        "outputs": outputs,
    }


__all__ = ["NOTHING_OPEN", "run_read_grades_mature"]
