"""Sunday bulk confirm: the trader confirms suggested setups in one pass (P8 P2 B).

Lists every trade whose `tag_status` is `needs_review` or `provisional`, with the
Mentor's own setup suggestion and its evidence (`trade_mentor_trade_check`
`setup_guess_for` / `evidence_for_guess`), and writes the rows the trader ticked
through the Mentor's writer (`confirm_setup`). A row already confirmed is never
touched; a journal write failure raises. Pure of Qt: the dialog runs both
halves on a worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import trade_mentor_trade_check as check
from journal_store import TAG_STATUS_NEEDS_REVIEW, TAG_STATUS_PROVISIONAL

#: The two machine states a bulk confirm may move to `confirmed`.
REVIEW_STATUSES = (TAG_STATUS_NEEDS_REVIEW, TAG_STATUS_PROVISIONAL)

#: What the why column says for a claimed like, which has no stored candidate.
CLAIMED_LIKE_WHY = "you claimed this setup before the first fill"


@dataclass(frozen=True)
class BulkConfirmRow:
    """One trade awaiting the trader's setup, with the machine's suggestion."""

    trade_id: str
    symbol: str
    trade_date: str
    direction: str
    opened_at: str
    net_pnl: float | None
    currency: str
    tag_status: str
    suggestion: str = ""
    lane: str = ""
    evidence: str = ""

    @property
    def checked_by_default(self) -> bool:
        """Ticked only when the suggestion comes with evidence."""
        return bool(self.suggestion and self.evidence)


def _status(trade: Any) -> str:
    return str(trade.get("tag_status") or "").strip().lower()


def _pnl(value: Any) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def rows_to_review(store: Any) -> list[BulkConfirmRow]:
    """Every needs_review / provisional trade, with its suggestion and why."""
    claims_by_day: dict[str, list] = {}
    rows: list[BulkConfirmRow] = []
    for trade in store.list_trades():
        status = _status(trade)
        if status not in REVIEW_STATUSES:
            continue
        trade_id = str(trade.get("trade_id") or "")
        if not trade_id:
            continue
        day = str(trade.get("trade_date") or "")[:10]
        if day not in claims_by_day:
            claims_by_day[day] = check.claimed_setup_rows(day) if day else []
        candidates = check._stored_candidates(store, trade_id)
        guess, lane = check.setup_guess_for(trade, claims_by_day[day], candidates)
        evidence = check.evidence_for_guess(guess, candidates)
        if guess and not evidence and lane == check.LANE_CLAIMED_LIKE:
            evidence = CLAIMED_LIKE_WHY
        rows.append(
            BulkConfirmRow(
                trade_id=trade_id,
                symbol=str(trade.get("symbol") or ""),
                trade_date=day,
                direction=str(trade.get("direction") or ""),
                opened_at=str(trade.get("opened_at") or ""),
                net_pnl=_pnl(trade.get("net_pnl")),
                currency=str(trade.get("currency") or ""),
                tag_status=status,
                suggestion=guess,
                lane=lane,
                evidence=evidence,
            )
        )
    return rows


def left_to_review(store: Any) -> int:
    """How many trades still carry a needs_review or provisional tag."""
    return sum(1 for trade in store.list_trades() if _status(trade) in REVIEW_STATUSES)


def confirm(store: Any, choices: Iterable[tuple[BulkConfirmRow, str]]) -> dict[str, Any]:
    """Confirm each (row, setup) the trader ticked; returns counts and refusals.

    The row's CURRENT state is re-read first: anything no longer needs_review or
    provisional (the trader confirmed it meanwhile) is left alone. A store write
    that fails raises - a journal write fails loudly.
    """
    confirmed: list[str] = []
    refused: list[tuple[str, str]] = []
    for row, setup in choices:
        chosen = str(setup or "").strip()
        if not chosen:
            refused.append((row.trade_id, "no setup chosen"))
            continue
        state = store.annotation_state(row.trade_id)
        if _status(state) not in REVIEW_STATUSES:
            refused.append((row.trade_id, "already confirmed"))
            continue
        question = check.TradeQuestion(
            trade_id=row.trade_id,
            symbol=row.symbol,
            direction=row.direction,
            missing=("setup",),
            opened_at=row.opened_at,
            trade_date=row.trade_date,
        )
        result = check.confirm_setup(store, question, setup=chosen)
        if result.get("ok"):
            confirmed.append(row.trade_id)
        else:
            refused.append((row.trade_id, str(result.get("reason") or "refused")))
    return {
        "confirmed": len(confirmed),
        "confirmed_ids": confirmed,
        "refused": refused,
        "left": left_to_review(store),
    }


def summary_text(result: dict[str, Any]) -> str:
    """The line the dialog shows when a confirm finishes."""
    return f"{int(result.get('confirmed') or 0)} confirmed, {int(result.get('left') or 0)} left"
