"""The nightly auto-tagging slot — V2 item 1. Deterministic, no model.

Decision 0016 answer 10, in the trader's words: *"Journaling's slow part is
tagging. The bot should auto-tag every night and the trader corrects."*

P6a built the whole machine — `journal_bulk_tag` plans, applies at a recorded
0.70 threshold, writes an adjustments trail and marks what it could not decide —
and then left it as a command the trader had to remember to run. This runs it.

**It runs right after `journal_import` and before every other slot.** The import
is what puts the night's trades in the journal, so tagging ahead of it would tag
yesterday's; and the cohort slots that follow read the journal, so tagging after
them would give them a journal one night stale. That is the same reasoning that
put `journal_import` first in the first place.

**It never touches a confirmed row.** The refusal lives in
`JournalStore.apply_provisional_tags`, not here and not in any caller — an
exception that depends on every caller remembering a rule is not a boundary. What
this slot adds is the schedule.

**A journal WRITE fails loudly.** Every other evidence store on the desk swallows
a failed append, because losing the evidence must not cost the event; the journal
is the exception, and a tagging run that could not write says so in its status
rather than reporting a quiet success.
"""

from __future__ import annotations

from typing import Any

#: The threshold P6a recorded and justified from the auto-tagger's own score
#: arithmetic. Named here rather than re-chosen: two places that each pick a
#: threshold eventually pick different ones.
from journal_bulk_tag import DEFAULT_CONFIDENCE_THRESHOLD


def run_journal_auto_tag(
    *,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    db_path=None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Plan and apply provisional tags for the night. Returns a ledger row.

    `refresh=True`, as the hand-run default is: most of this journal was imported
    from broker statements long after the scan files that could explain it, so a
    trade's stored suggestions may predate the evidence that now exists. Deciding
    from stale candidates would tag tonight's trades from whatever the tagger
    believed on some earlier day.
    """
    from journal_bulk_tag import apply_plan, build_plan
    from journal_store import JournalStore

    try:
        store = JournalStore(db_path) if db_path is not None else JournalStore()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "model": "",
            "reason": f"journal unavailable: {exc}",
            "outputs": [],
        }

    try:
        plan = build_plan(store, threshold=threshold, refresh=True)
        summary = apply_plan(store, plan)
    except Exception as exc:  # noqa: BLE001
        # LOUD. A journal write is the one store on this desk that may not fail
        # quietly - a tag that silently did not land is a trade the trader will
        # believe is tagged.
        return {
            "status": "failed",
            "model": "",
            "reason": f"auto-tagging failed: {exc}",
            "outputs": [],
        }

    reason = (
        f"{summary.get('applied', 0)} applied, "
        f"{summary.get('marked', 0)} marked for review, "
        f"{summary.get('refused', 0)} refused "
        f"({plan.considered} closed trade(s) considered, "
        f"{plan.already_confirmed} already confirmed by the trader)"
    )
    return {"status": "ok", "model": "", "reason": reason, "outputs": []}


def trades_awaiting_review(db_path=None) -> int:
    """How many closed trades carry a `needs_review` marker or a provisional tag.

    The Journal tab's badge. Counted from the STORE rather than from the last
    run's summary: a run that wrote nothing new does not mean there is nothing to
    review, and the trader may have confirmed some of them since.

    Returns 0 rather than raising on any failure - a badge is a convenience, and
    a number nobody can compute is not worth a broken page.
    """
    from journal_bulk_tag import ELIGIBLE_STATUS, PARTIAL_STATUS
    from journal_store import (
        TAG_STATUS_NEEDS_REVIEW,
        TAG_STATUS_PROVISIONAL,
        JournalStore,
    )

    try:
        store = JournalStore(db_path) if db_path is not None else JournalStore()
        wanted = {TAG_STATUS_NEEDS_REVIEW, TAG_STATUS_PROVISIONAL}
        total = 0
        for trade in store.list_trades():
            status = str(trade.get("status") or "").upper()
            if status not in {ELIGIBLE_STATUS, PARTIAL_STATUS}:
                continue
            if str(trade.get("tag_status") or "") in wanted:
                total += 1
        return total
    except Exception:  # noqa: BLE001 - a badge is never worth a broken page
        return 0
