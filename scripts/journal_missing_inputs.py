"""Which recent trades still lack a stop, a confirmed setup or a thesis (P8 B4).

`missing_inputs` is pure: it takes journal trade rows (`JournalStore.list_trades`,
which already joins `trade_annotations`) and, per trade id, the annotation fields
plus the material fields the Mentor already has an answer for (`RECALLED` rows,
`trade_mentor_trade_check.answered_fields`). A field is missing by the Mentor's
own rule (`trade_mentor_trade_check.missing_fields`): the stop when there is no
`planned_stop` and no answer; the setup when there is no tag or its `tag_status`
is not `confirmed`, and no answer; the thesis when there are no notes and no
answer. Absent data counts as missing; nothing is inferred.

`load` reads the store off the Qt thread for the status-bar chip.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping

import trade_mentor_trade_check as check
from journal_store import TRADE_STATUSES

#: The inputs this nag reads, in the Mentor's asking order.
INPUT_FIELDS = ("stop", "setup", "thesis")
#: The inputs the status-bar chip counts.
CHIP_FIELDS = ("stop", "setup")
SINCE_DAYS_DEFAULT = 30

#: Annotation columns an `annotations` entry may supply over the trade row.
_ANNOTATION_KEYS = ("planned_stop", "setup_tags", "tag_status", "notes")


def _opened_day(trade: Mapping[str, Any]) -> str:
    """The day the trade opened, or its trade date when the open stamp is absent."""
    text = str(trade.get("opened_at") or "").strip() or str(trade.get("trade_date") or "").strip()
    return text[:10] if len(text) >= 10 else ""


def window_start(today: date, since_days: int = SINCE_DAYS_DEFAULT) -> str:
    """The first opened day inside the window, ISO."""
    return (today - timedelta(days=max(0, int(since_days)))).isoformat()


def missing_inputs(
    trades: Iterable[Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    since_days: int = SINCE_DAYS_DEFAULT,
    today: date | None = None,
) -> dict[str, Any]:
    """Trades opened in the last `since_days` days that miss an input, oldest first.

    `annotations[trade_id]` may carry `planned_stop`, `setup_tags`, `tag_status`
    and `notes` (they win over the trade row) and `answered`: the material
    fields the Mentor already holds an answer for. Returns
    ``{"since", "rows", "counts"}``; each row is
    ``{trade_id, symbol, opened_at, missing}``.
    """
    day = today or datetime.now().astimezone().date()
    since = window_start(day, since_days)
    notes = annotations or {}
    rows: list[dict[str, Any]] = []
    for trade in trades or ():
        trade_id = str(trade.get("trade_id") or "")
        if not trade_id:
            continue
        status = str(trade.get("status") or "").upper()
        if status and status not in TRADE_STATUSES:
            continue
        opened = _opened_day(trade)
        if not opened or opened < since:
            continue
        extra = notes.get(trade_id) or {}
        merged = dict(trade)
        for key in _ANNOTATION_KEYS:
            if key in extra:
                merged[key] = extra[key]
        answered = {str(name) for name in (extra.get("answered") or ())}
        gaps = set(check.missing_fields(merged, answered))
        missing = [name for name in INPUT_FIELDS if name in gaps]
        if not missing:
            continue
        rows.append(
            {
                "trade_id": trade_id,
                "symbol": str(trade.get("symbol") or ""),
                "opened_at": str(trade.get("opened_at") or "") or opened,
                "trade_date": str(trade.get("trade_date") or "")[:10],
                "missing": missing,
            }
        )
    rows.sort(key=lambda row: (str(row["opened_at"]), row["trade_id"]))
    counts = {name: sum(1 for row in rows if name in row["missing"]) for name in INPUT_FIELDS}
    counts["trades"] = len(rows)
    counts["stop_or_setup"] = sum(
        1 for row in rows if any(name in row["missing"] for name in CHIP_FIELDS)
    )
    return {"since": since, "since_days": int(since_days), "rows": rows, "counts": counts}


def oldest_chip_row(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The oldest trade missing a stop or setup, or ``None``."""
    for row in result.get("rows") or ():
        if any(name in row.get("missing", ()) for name in CHIP_FIELDS):
            return dict(row)
    return None


def chip_text(result: Mapping[str, Any] | None) -> str:
    """The status-bar chip, or "" when no trade misses a stop or setup."""
    count = int(((result or {}).get("counts") or {}).get("stop_or_setup") or 0)
    if count <= 0:
        return ""
    return f"Inputs: {count} trade{'s' if count != 1 else ''} missing stop/setup"


def load(
    store: Any, *, since_days: int = SINCE_DAYS_DEFAULT, today: date | None = None
) -> dict[str, Any]:
    """Read the journal and answer :func:`missing_inputs` (never on the Qt thread)."""
    day = today or datetime.now().astimezone().date()
    since = window_start(day, since_days)
    trades = [trade for trade in store.list_trades() if _opened_day(trade) >= since]
    annotations = {
        str(trade.get("trade_id") or ""): {
            "answered": sorted(check.answered_fields(store, str(trade.get("trade_id") or "")))
        }
        for trade in trades
    }
    return missing_inputs(trades, annotations, since_days=since_days, today=day)


def question_for(store: Any, row: Mapping[str, Any] | None) -> Any:
    """The Mentor's own `TradeQuestion` for one listed trade, or ``None``.

    Built by `questions_for_session` on the trade's session, so the card gets
    the same gaps, setup guess and exit ask it would draw at 09:00; a trade that
    session does not list is asked from its row with the same field rule.
    """
    if not row:
        return None
    trade_id = str(row.get("trade_id") or "")
    if not trade_id:
        return None
    days: list[str] = []
    for text in (row.get("trade_date"), row.get("opened_at")):
        day = str(text or "")[:10]
        if len(day) == 10 and day not in days:
            days.append(day)
    for day in days:
        for question in check.questions_for_session(store, day) or ():
            if str(question.trade_id) == trade_id:
                return question
    trade = store.get_trade(trade_id) or {}
    if not trade:
        return None
    return check.TradeQuestion(
        trade_id=trade_id,
        symbol=str(trade.get("symbol") or ""),
        direction=str(trade.get("direction") or ""),
        missing=check.missing_fields(trade, check.answered_fields(store, trade_id)),
        opened_at=str(trade.get("opened_at") or ""),
        trade_date=str(trade.get("trade_date") or ""),
    )


def load_chip(
    store: Any, *, since_days: int = SINCE_DAYS_DEFAULT, today: date | None = None
) -> dict[str, Any]:
    """What the status-bar chip needs, read on its worker: the counts and the
    Mentor question for the oldest trade missing a stop or setup."""
    result = load(store, since_days=since_days, today=today)
    result["question"] = question_for(store, oldest_chip_row(result))
    return result
