"""Sunday ritual card (B3): tags waiting, the plan, and exits per setup family.

`read_card` does every read (journal, plan, plan challenges) and runs on a
worker; `family_rows` and `format_card` are pure. Facts only: nothing here
detects, scores, ranks, gates or alerts, and nothing here writes. Confirming
tags goes through `journal_bulk_confirm` behind the trader's click.

Family = the first CONFIRMED setup tag; a provisional or needs_review tag is
not the trader's answer and counts under `NOT_CONFIRMED`.
Exit early = a winner whose trader-confirmed exit reason is `took_profit_early`.
Held loser = a loser held longer than the family's median winner.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from statistics import median
from typing import Any, Callable, Iterable, Mapping

_log = logging.getLogger(__name__)

#: The family of a closed trade whose setup the trader has not confirmed.
NOT_CONFIRMED = "not confirmed yet"
#: The exit-reason code (exit_reasons_v1) that means "exited early".
EARLY_EXIT_CODE = "took_profit_early"
#: The journal event holding the trader's confirmed exit fields.
EXIT_FIELDS_EVENT = "EXIT_NOTE_FIELDS"
#: How many open plan challenges the card lists.
MAX_CHALLENGES = 5
#: How many confirmed family lines the card lists (biggest first); the rest are counted.
MAX_FAMILIES = 12


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _hours(opened: Any, closed: Any) -> float | None:
    try:
        start = datetime.fromisoformat(_text(opened))
        end = datetime.fromisoformat(_text(closed))
    except ValueError:
        return None
    if (start.tzinfo is None) != (end.tzinfo is None):
        return None
    seconds = (end - start).total_seconds()
    return seconds / 3600.0 if seconds >= 0 else None


def family_of(trade: Mapping[str, Any]) -> str:
    """The first confirmed setup tag, lower case; `NOT_CONFIRMED` otherwise."""
    status = _text(trade.get("tag_status")).lower() or "confirmed"
    if status != "confirmed":
        return NOT_CONFIRMED
    for part in re.split(r"[;,]", _text(trade.get("setup_tags"))):
        if part.strip():
            return part.strip().lower()
    return NOT_CONFIRMED


def family_rows(
    trades: Iterable[Mapping[str, Any]], exit_reasons: Mapping[str, str] | None
) -> list[dict[str, Any]]:
    """One row per setup family over closed trades. Pure.

    `exit_reasons` maps trade_id -> the trader's confirmed exit-reason code;
    `None` means they could not be read, so exit early is unknown.
    A count the journal cannot answer is `None` (unknown), never 0.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        if _text(trade.get("status")).upper() != "CLOSED":
            continue
        pnl = _number(trade.get("net_pnl"))
        if pnl is None:
            continue
        groups.setdefault(family_of(trade), []).append(
            {
                "trade_id": _text(trade.get("trade_id")),
                "pnl": pnl,
                "hours": _hours(trade.get("opened_at"), trade.get("closed_at")),
            }
        )
    rows = []
    for family, items in groups.items():
        winners = [item for item in items if item["pnl"] > 0]
        losers = [item for item in items if item["pnl"] < 0]
        reasons = [(exit_reasons or {}).get(item["trade_id"], "") for item in winners]
        win_hours = [item["hours"] for item in winners if item["hours"] is not None]
        loss_hours = [item["hours"] for item in losers if item["hours"] is not None]
        median_win = median(win_hours) if win_hours else None
        median_loss = median(loss_hours) if loss_hours else None
        held = None
        if median_win is not None:
            held = sum(1 for hours in loss_hours if hours > median_win)
        rows.append(
            {
                "family": family,
                "closed": len(items),
                "winners": len(winners),
                "losers": len(losers),
                "exit_early": None if exit_reasons is None else sum(1 for code in reasons if code == EARLY_EXIT_CODE),
                "winners_no_reason": None if exit_reasons is None else sum(1 for code in reasons if not code),
                "held_losers": held,
                "losers_no_hold": len(losers) - len(loss_hours),
                "median_winner_hours": median_win,
                "median_loser_hours": median_loss,
            }
        )
    rows.sort(key=lambda row: (row["family"] == NOT_CONFIRMED, -row["closed"], row["family"]))
    return rows


def _confirmed_exit_reasons(store: Any) -> dict[str, str]:
    """trade_id -> the newest confirmed exit-reason code, in one read."""
    out: dict[str, str] = {}
    for row in store.list_opportunity_events(event_type=EXIT_FIELDS_EVENT, limit=10000):
        payload = row.get("payload") or {}
        why = (payload.get("fields") or {}).get("why") if isinstance(payload, Mapping) else None
        code = _text(why.get("code")) if isinstance(why, Mapping) else ""
        trade_id = _text(row.get("trade_id"))
        if trade_id and code:
            out[trade_id] = code
    return out


def _default_store():
    from ui.services import journal_feed

    return journal_feed._store()


def read_card(
    store_factory: Callable[[], Any] | None = None,
    *,
    plan_reader: Callable[[], Mapping[str, Any]] | None = None,
    challenges_reader: Callable[[], list] | None = None,
) -> dict[str, Any]:
    """Every read the card needs. Worker only. A failed part is stated, never blank."""
    payload: dict[str, Any] = {"waiting": None, "families": [], "plan": {}, "challenges": [], "errors": {}}
    try:
        import journal_bulk_confirm as bulk

        store = (store_factory or _default_store)()
        trades = store.list_trades()
        payload["waiting"] = sum(
            1 for trade in trades if _text(trade.get("tag_status")).lower() in bulk.REVIEW_STATUSES
        )
        try:
            reasons = _confirmed_exit_reasons(store)
        except Exception as exc:  # noqa: BLE001 - stated on the card
            reasons = None
            payload["errors"]["exit_reasons"] = str(exc)
        payload["families"] = family_rows(trades, reasons)
    except Exception as exc:  # noqa: BLE001 - stated on the card
        _log.debug("Sunday card: journal unreadable.", exc_info=True)
        payload["errors"]["journal"] = str(exc)
    try:
        if plan_reader is None:
            import trading_plan

            plan = trading_plan.read_plan(create=False, snapshot=False)
        else:
            plan = plan_reader()
        payload["plan"] = dict(plan or {})
    except Exception as exc:  # noqa: BLE001 - stated on the card
        payload["errors"]["plan"] = str(exc)
    try:
        if challenges_reader is None:
            import plan_challenges

            challenges = plan_challenges.open_challenges()
        else:
            challenges = challenges_reader()
        payload["challenges"] = [dict(row) for row in challenges or () if isinstance(row, Mapping)]
    except Exception as exc:  # noqa: BLE001 - stated on the card
        payload["errors"]["challenges"] = str(exc)
    return payload


def _hours_text(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 24:.1f}d" if value >= 48 else f"{value:.1f}h"


def family_line(row: Mapping[str, Any]) -> str:
    """One family's line, facts only."""
    winners, losers = int(row.get("winners") or 0), int(row.get("losers") or 0)
    if row.get("exit_early") is None:
        early = f"exit early unknown of {winners} winners"
    else:
        early = f"exit early {int(row.get('exit_early') or 0)} of {winners} winners"
    missing = int(row.get("winners_no_reason") or 0)
    if missing:
        early += f" ({missing} with no exit reason)"
    held = row.get("held_losers")
    if losers == 0:
        held_text = "held losers 0 of 0"
    elif held is None:
        held_text = f"held losers unknown of {losers} (no winner to compare)"
    else:
        held_text = f"held losers {held} of {losers} (held past the median winner)"
    return (
        f"{row.get('family')}: {int(row.get('closed') or 0)} closed, {winners} won, {losers} lost · "
        f"{early} · {held_text} · median hold win {_hours_text(row.get('median_winner_hours'))}, "
        f"loss {_hours_text(row.get('median_loser_hours'))}"
    )


def format_card(payload: Mapping[str, Any]) -> dict[str, str]:
    """The card's three texts from one payload. Pure."""
    errors = dict(payload.get("errors") or {})
    waiting = payload.get("waiting")
    if errors.get("journal"):
        tags = f"Setup tags: the journal could not be read ({errors['journal']})."
    elif not waiting:
        tags = "Setup tags: none waiting. Every tag is your own answer."
    else:
        tags = (
            f"Setup tags: {int(waiting)} waiting for you. Press the button to see each "
            "suggestion first; only the rows you tick are confirmed."
        )

    plan = dict(payload.get("plan") or {})
    parsed = plan.get("parsed") if isinstance(plan.get("parsed"), Mapping) else {}
    sections = dict((parsed or {}).get("sections") or {})
    plan_lines: list[str] = []
    if errors.get("plan") or plan.get("error"):
        plan_lines.append(f"Plan: could not be read ({errors.get('plan') or plan.get('error')}).")
    elif not plan.get("exists"):
        plan_lines.append("Plan: no trading plan yet. Open 'My trading plan' on Week Review to start one.")
    else:
        for heading in ("Setups I trade", "What I am testing"):
            lines = [_text(line) for line in sections.get(heading) or () if _text(line)]
            plan_lines.append(f"{heading}: " + ("; ".join(lines) if lines else "(empty)"))
        plan_lines.append("Ask yourself: did this week's trades follow these lines?")
    challenges = list(payload.get("challenges") or ())
    if errors.get("challenges"):
        plan_lines.append(f"Night AI plan challenges: could not be read ({errors['challenges']}).")
    elif challenges:
        plan_lines.append(f"Night AI plan challenges open: {len(challenges)} (answer them in the Mentor).")
        for row in challenges[:MAX_CHALLENGES]:
            plan_lines.append(f"- \"{_text(row.get('plan_line_text'))}\": {_text(row.get('text'))}")
    else:
        plan_lines.append("Night AI plan challenges open: none.")

    families = list(payload.get("families") or ())
    if errors.get("journal"):
        family_text = "Exits by setup family: unknown (journal unreadable)."
    elif not families:
        family_text = "Exits by setup family: no closed trades yet."
    else:
        head = ["Exits by setup family (closed trades, from the journal):"]
        if errors.get("exit_reasons"):
            head.append(f"Exit reasons could not be read ({errors['exit_reasons']}); exit early is unknown.")
        shown = [row for row in families if row.get("family") != NOT_CONFIRMED][:MAX_FAMILIES]
        hidden = sum(1 for row in families if row.get("family") != NOT_CONFIRMED) - len(shown)
        lines = [family_line(row) for row in shown]
        if hidden > 0:
            lines.append(f"... and {hidden} smaller families.")
        lines += [family_line(row) for row in families if row.get("family") == NOT_CONFIRMED]
        family_text = "\n".join(head + lines)
    return {"tags": tags, "plan": "\n".join(plan_lines), "families": family_text}
