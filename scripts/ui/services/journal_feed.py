from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ui.models.journal import JournalTrade


_STORE = None


def _store():
    """Lazily create a shared JournalStore (also initializes the sqlite schema)."""
    global _STORE
    if _STORE is None:
        from journal_store import JournalStore

        _STORE = JournalStore()
    return _STORE


def journal_db_path() -> Path:
    from project_paths import JOURNAL_DB_FILE

    return Path(JOURNAL_DB_FILE)


def distinct_values(column: str) -> list[str]:
    try:
        return list(_store().distinct_values(column))
    except Exception:
        return []


def analytics_summary(trades: list[JournalTrade]) -> dict[str, Any]:
    from journal_analytics import build_analytics_summary

    return build_analytics_summary([trade.raw for trade in trades])


def analytics_text(trades: list[JournalTrade]) -> str:
    from journal_analytics import build_analytics_text

    return build_analytics_text([trade.raw for trade in trades])


def trade_legs(trade_id: str) -> list[dict[str, Any]]:
    try:
        return _store().list_trade_legs(trade_id)
    except Exception:
        return []


def save_annotation(trade_id: str, *, setup_tags: str, notes: str) -> None:
    _store().save_trade_annotation(trade_id, setup_tags=setup_tags, notes=notes)


def record_trade_review(
    trade_id: str,
    *,
    review_outcome: str,
    decision_reason: str = "",
    setup_tags: str = "",
    notes: str = "",
) -> dict[str, Any] | None:
    trade = _store().get_trade(trade_id)
    if trade is None:
        return None
    return _store().record_opportunity_event(
        opportunity_id=f"trade:{trade_id}",
        lifecycle_id=f"trade:{trade_id}",
        event_type="REVIEWED",
        symbol=str(trade.get("symbol") or ""),
        side=str(trade.get("direction") or ""),
        trade_id=trade_id,
        reason=decision_reason,
        payload={
            "review_outcome": str(review_outcome or "").strip(),
            "setup_tags": str(setup_tags or "").strip(),
            "notes": str(notes or "").strip(),
        },
        source="journal_gui",
    )


def latest_trade_review(trade_id: str) -> dict[str, Any] | None:
    try:
        return _store().latest_trade_review(trade_id)
    except Exception:
        return None


def export_trades_csv() -> Path:
    return _store().export_trades_csv()


def rebuild_trades() -> int:
    return _store().rebuild_trades()


def list_import_runs(limit: int = 10) -> list[dict[str, Any]]:
    try:
        return _store().list_import_runs(limit=limit)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# R7 §9 steps 11-13. Every tab reads through here and nothing else, so the UI
# never holds a JournalStore and the whole surface is testable without Qt.
# ---------------------------------------------------------------------------

#: The account-tree groups, in the order they are shown. "Unlabeled" is last and
#: is deliberately a group rather than a hidden default: an account nobody has
#: decided about must be visible as undecided, because a guessed tax status is a
#: wrong number in a tax record (I6).
TAX_GROUPS = ("TAXABLE", "TAX_FREE", "TAX_DEFERRED", "")

TAX_GROUP_LABELS = {
    "TAXABLE": "Taxable",
    "TAX_FREE": "Tax-free",
    "TAX_DEFERRED": "Tax-deferred",
    "": "Unlabeled",
}


def accounts() -> list[dict[str, Any]]:
    try:
        return _store().list_accounts()
    except Exception:
        return []


def account_tree() -> list[dict[str, Any]]:
    """Accounts grouped by tax status, for the shared header's checkable tree."""
    grouped: dict[str, list[dict[str, Any]]] = {group: [] for group in TAX_GROUPS}
    for account in accounts():
        status = str(account.get("tax_status") or "")
        grouped.setdefault(status if status in grouped else "", []).append(account)
    return [
        {"tax_status": group, "label": TAX_GROUP_LABELS[group], "accounts": grouped[group]}
        for group in TAX_GROUPS
        if grouped[group]
    ]


def set_account_tax_status(broker: str, account_number: str, tax_status: str) -> None:
    _store().set_account_tax_status(broker, account_number, tax_status, source="trader")


def selection_spans_tax_groups(selected: Iterable[tuple[str, str]]) -> bool:
    """Does this account selection mix tax treatments? (I6)

    The header badges a blended selection rather than refusing it - the trader
    is allowed to look at everything at once, and is not allowed to do it by
    accident.
    """
    by_key = {
        (str(account.get("broker") or "").upper(), str(account.get("account_number") or "")):
            str(account.get("tax_status") or "")
        for account in accounts()
    }
    statuses = {by_key.get((str(broker).upper(), str(number)), "") for broker, number in selected}
    return len(statuses) > 1


def load_trades(
    *,
    broker: str = "All",
    account: str = "All",
    symbol: str = "",
    date_from: Any = None,
    date_to: Any = None,
    accounts_filter: Iterable[tuple[str, str]] | None = None,
) -> list[JournalTrade]:
    rows = _store().list_trades(
        broker=broker or "All",
        account=account or "All",
        symbol=(symbol or "").strip() or None,
        date_from=date_from,
        date_to=date_to,
    )
    if accounts_filter is not None:
        wanted = {(str(b).upper(), str(a)) for b, a in accounts_filter}
        rows = [
            row for row in rows
            if (str(row.get("broker") or "").upper(), str(row.get("account_number") or "")) in wanted
        ]
    return [JournalTrade.from_mapping(row) for row in rows]


def date_range_bounds(preset: str) -> tuple[Any, Any]:
    """(date_from, date_to) for a header preset. ``All`` is (None, None)."""
    from datetime import date, timedelta

    today = date.today()
    preset = str(preset or "All").strip()
    if preset == "7d":
        return today - timedelta(days=7), today
    if preset == "30d":
        return today - timedelta(days=30), today
    if preset == "QTD":
        quarter_first_month = 3 * ((today.month - 1) // 3) + 1
        return date(today.year, quarter_first_month, 1), today
    if preset == "YTD":
        return date(today.year, 1, 1), today
    return None, None


def convert_amount(trade: JournalTrade, currency_mode: str) -> tuple[float | None, str]:
    """One trade's P&L in the header's chosen currency, plus what to render.

    Returns ``(value, label)``; ``value`` is None when the trade cannot honestly
    be shown in the requested currency, and ``label`` says why. I5: never 0, and
    never the native number quietly relabelled.
    """
    mode = str(currency_mode or "Native").strip().upper()
    raw = trade.raw
    native = raw.get("net_pnl")
    if mode in {"NATIVE", ""}:
        return native, str(raw.get("currency") or "")
    if mode == "CAD":
        value = raw.get("net_pnl_cad")
        return (value, "CAD") if value is not None else (None, "unconverted")
    if mode == "USD":
        if str(raw.get("currency") or "").upper() == "USD":
            return native, "USD"
        return None, "unconverted"
    return native, str(raw.get("currency") or "")


# -- R fields (trader-owned; no import path writes these, I7) ----------------


def save_risk_fields(
    trade_id: str,
    *,
    planned_entry: float | None = None,
    planned_stop: float | None = None,
    planned_risk: float | None = None,
    risk_source: str = "manual",
) -> None:
    _store().save_risk_fields(
        trade_id,
        planned_entry=planned_entry,
        planned_stop=planned_stop,
        planned_risk=planned_risk,
        risk_source=risk_source,
    )


def r_multiple(trade: JournalTrade) -> float | None:
    """``net_pnl_cad / planned_risk``, or None when either half is missing.

    Deliberately CAD: an R computed from a native P&L and a risk the trader
    typed in dollars would silently mix currencies, which is the same defect B8
    was about.
    """
    raw = trade.raw
    risk = raw.get("planned_risk")
    pnl = raw.get("net_pnl_cad")
    try:
        risk_value = float(risk)
        pnl_value = float(pnl)
    except (TypeError, ValueError):
        return None
    if abs(risk_value) < 1e-9:
        return None
    return pnl_value / abs(risk_value)


def _armed_alert_events() -> list[dict[str, Any]]:
    """Alert Center decisions, read straight from the JSONL evidence log.

    Read-only. This stream is analysis evidence and R7 does not write to it -
    the prefill borrows an entry and a stop the trader already set on an alert,
    which is why it is trustworthy enough to offer and still never applied
    without a unique match.
    """
    import json

    from project_paths import ALERT_REVIEW_EVENTS_FILE

    path = Path(ALERT_REVIEW_EVENTS_FILE)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def suggest_planned_risk(trade: JournalTrade) -> dict[str, Any] | None:
    """Prefill entry/stop from an armed alert, on a **unique** match only.

    Joins symbol + direction + opened_at ± 1 session against alert review events
    carrying an entry and a stop. Same-symbol re-entries in one day are
    genuinely ambiguous (spec §11), so anything but a single match returns None
    rather than picking one - a prefilled stop the trader did not set is a
    fabricated R.
    """
    events = _armed_alert_events()
    if not events:
        return None

    from datetime import date, timedelta

    try:
        opened = date.fromisoformat(str(trade.opened_at)[:10])
    except (TypeError, ValueError):
        return None
    window = {opened - timedelta(days=1), opened, opened + timedelta(days=1)}

    matches = []
    for event in events or []:
        if str(event.get("symbol") or "").upper() != trade.symbol:
            continue
        side = str(event.get("side") or event.get("direction") or "").upper()
        if side and side != trade.direction:
            continue
        entry = event.get("entry") or event.get("entry_price")
        stop = event.get("stop") or event.get("stop_price")
        if entry is None or stop is None:
            continue
        try:
            when = date.fromisoformat(str(event.get("occurred_at") or event.get("timestamp") or "")[:10])
        except (TypeError, ValueError):
            continue
        if when not in window:
            continue
        matches.append({"planned_entry": float(entry), "planned_stop": float(stop)})

    if len(matches) != 1:
        return None
    suggestion = dict(matches[0])
    quantity = trade.raw.get("quantity_opened") or 0.0
    try:
        risk = abs(float(suggestion["planned_entry"]) - float(suggestion["planned_stop"])) * float(quantity)
    except (TypeError, ValueError):
        risk = 0.0
    suggestion["planned_risk"] = risk or None
    suggestion["risk_source"] = "alert_prefill"
    return suggestion


# -- corrections -------------------------------------------------------------


def record_adjustment(**kwargs: Any) -> dict[str, Any]:
    return _store().record_adjustment(**kwargs)


def undo_adjustment(adjustment_id: str, *, reason: str) -> dict[str, Any]:
    return _store().undo_adjustment(adjustment_id, reason=reason)


def list_adjustments(**kwargs: Any) -> list[dict[str, Any]]:
    try:
        return _store().list_adjustments(**kwargs)
    except Exception:
        return []


def group_key_for(trade: JournalTrade) -> str:
    from journal_identity import group_key, group_key_text

    return group_key_text(group_key(trade.raw))


def add_manual_execution(fields: dict[str, Any]) -> int:
    """Enter a fill by hand, into a real broker/account (spec §5 fix 3).

    The dialog's broker and account pickers exist because ``broker="MANUAL"``
    was the old default and made every hand-entered fill an orphan that could
    never attach to the position it belonged to (B3).
    """
    from journal_importers import manual_execution_from_fields

    return _store().upsert_executions([manual_execution_from_fields(fields)])


# -- auto tags ---------------------------------------------------------------


def auto_tag_candidates(trade_id: str) -> list[dict[str, Any]]:
    try:
        return _store().list_auto_tag_candidates(trade_id)
    except Exception:
        return []


def accept_auto_tags(trade_id: str, tags: Iterable[str]) -> str:
    """Append accepted suggestions to the trader's own tags and learn from it."""
    accepted = [str(tag).strip() for tag in tags if str(tag).strip()]
    trade = _store().get_trade(trade_id) or {}
    existing = [part.strip() for part in str(trade.get("setup_tags") or "").split(";") if part.strip()]
    merged = existing + [tag for tag in accepted if tag not in existing]
    combined = "; ".join(merged)
    _store().save_trade_annotation(trade_id, setup_tags=combined, notes=str(trade.get("notes") or ""))
    if accepted:
        _store().record_tag_corrections(trade, "; ".join(accepted))
    return combined


# -- calendar / analytics ----------------------------------------------------


def calendar_pnl_by_day(**kwargs: Any) -> dict[str, float]:
    from journal_analytics import calendar_pnl_by_day as _calendar

    trades = [trade.raw for trade in load_trades(**kwargs)]
    return _calendar(trades)


def equity_curve(trades: list[JournalTrade], currency_mode: str = "CAD") -> list[tuple[str, float]]:
    """Cumulative P&L by trade date, in the header's currency.

    Trades that cannot be shown in that currency are **skipped and counted** by
    the caller rather than treated as zero - a curve that silently absorbs an
    unconvertible trade as a flat step is a lie about a real position.
    """
    points: list[tuple[str, float]] = []
    running = 0.0
    for trade in sorted(trades, key=lambda item: (item.trade_date, item.trade_id)):
        if not trade.is_closed:
            continue
        value, _label = convert_amount(trade, currency_mode)
        if value is None:
            continue
        running += float(value)
        points.append((trade.trade_date, running))
    return points


def unconvertible_count(trades: list[JournalTrade], currency_mode: str = "CAD") -> int:
    return sum(
        1 for trade in trades
        if trade.is_closed and convert_amount(trade, currency_mode)[0] is None
    )


def walkaway_summary(since: Any = None, until: Any = None) -> dict[str, Any]:
    """Run walk-away for a window. Heavy - the panel calls it from a worker."""
    from journal_walkaway import run_walkaway_analysis

    return run_walkaway_analysis(source="journal", write_outputs=False, since=since, until=until)


# -- health / fees -----------------------------------------------------------


def coverage_grid(days: int = 30) -> dict[str, Any]:
    """Accounts x recent days, coloured by coverage status (spec §7 Health)."""
    from datetime import date, timedelta

    import journal_coverage

    end = date.today()
    start = end - timedelta(days=max(1, int(days)))
    rows = journal_coverage.coverage_rows(_store(), start=start, end=end)
    by_account: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("broker") or ""), str(row.get("account_number") or ""))
        by_account.setdefault(key, {})[str(row.get("day"))] = str(row.get("status") or "")
    days_list = []
    cursor = start
    while cursor <= end:
        days_list.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return {"days": days_list, "accounts": by_account, "start": start.isoformat(), "end": end.isoformat()}


def find_coverage_gaps(days: int = 365) -> list[dict[str, Any]]:
    from datetime import date, timedelta

    import journal_coverage

    end = date.today()
    start = end - timedelta(days=max(1, int(days)))
    gaps: list[dict[str, Any]] = []
    for broker, account_number in journal_coverage.known_accounts(_store()):
        missing = journal_coverage.find_gaps(
            _store(), broker=broker, account_number=account_number, start=start, end=end
        )
        if missing:
            gaps.append(
                {
                    "broker": broker,
                    "account_number": account_number,
                    "days": [day.isoformat() for day in missing],
                }
            )
    return gaps


def last_reconciliation() -> dict[str, Any] | None:
    import journal_reconcile

    try:
        return journal_reconcile.last_report(_store())
    except Exception:
        return None


def confirm_reconciliation_suggestion(suggestion: dict[str, Any], *, reason: str) -> dict[str, Any]:
    import journal_reconcile

    return journal_reconcile.confirm_suggestion(_store(), suggestion, reason=reason)


def fx_coverage() -> dict[str, Any]:
    import journal_fx

    try:
        return journal_fx.describe_coverage(_store())
    except Exception:
        return {"trades": 0, "converted": 0, "unconverted": [], "booked_rates": 0}


def cash_transactions(**kwargs: Any) -> list[dict[str, Any]]:
    try:
        return _store().list_cash_transactions(**kwargs)
    except Exception:
        return []


def fee_totals(**kwargs: Any) -> list[dict[str, Any]]:
    """Per account x currency: trade costs, plus cash-side fees and dividends.

    Trade commissions and ``cash_transactions`` are reported side by side and
    never added together - the first is already inside each trade's net P&L and
    the second is not, so one column summing both would double-count.
    """
    totals: dict[tuple[str, str, str], dict[str, float]] = {}
    for trade in load_trades(**kwargs):
        raw = trade.raw
        key = (
            str(raw.get("broker") or ""),
            str(raw.get("account_label") or raw.get("account_number") or ""),
            str(raw.get("currency") or ""),
        )
        entry = totals.setdefault(key, {"commission": 0.0, "fees": 0.0, "cash_fees": 0.0, "dividends": 0.0})
        entry["commission"] += float(raw.get("commission") or 0.0)
        entry["fees"] += float(raw.get("fees") or 0.0)
    for row in cash_transactions():
        key = (str(row.get("broker") or ""), str(row.get("account_number") or ""), str(row.get("currency") or ""))
        entry = totals.setdefault(key, {"commission": 0.0, "fees": 0.0, "cash_fees": 0.0, "dividends": 0.0})
        if str(row.get("activity_type") or "") == "FEE":
            entry["cash_fees"] += float(row.get("amount") or 0.0)
        elif str(row.get("activity_type") or "") == "DIVIDEND":
            entry["dividends"] += float(row.get("amount") or 0.0)
    return [
        {"broker": broker, "account": account, "currency": currency, **values}
        for (broker, account, currency), values in sorted(totals.items())
    ]


def export_fees_csv(path: Path | None = None) -> Path:
    import csv

    from project_paths import JOURNAL_EXPORT_DIR

    rows = fee_totals()
    target = Path(path) if path else Path(JOURNAL_EXPORT_DIR) / "journal_fees.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["broker", "account", "currency", "commission", "fees", "cash_fees", "dividends"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def nightly_slot_status() -> dict[str, Any]:
    """Read-only view of the journal_import slot's last ledger entries."""
    try:
        from ai_jobs import ledger

        rows = [
            row for row in ledger.recent_rows(limit=200)
            if str(row.get("job") or "") == "journal_import"
        ]
    except Exception:
        return {"rows": [], "available": False}
    return {"rows": rows[-5:], "available": True}


def self_heal_gaps(max_days: int = 62) -> dict[str, Any]:
    """The Health tab's "Backfill gaps" button. Runs in a worker, never on the GUI thread."""
    import journal_coverage
    from journal_runner import _fetch_one_day

    return journal_coverage.self_heal(
        _store(),
        lambda broker, account, day: _fetch_one_day(_store(), broker, account, day),
        max_days_per_night=max_days,
    )


# ---------------------------------------------------------------------------
# Weekend Prep (R8 §6). The journal side of the weekend routine: which trades
# belong to the reviewed week, and which of them are still open.
# ---------------------------------------------------------------------------


def week_trades(monday: Any, friday: Any) -> dict[str, list[JournalTrade]]:
    """The week's trades, split into closed-in-week and still-open.

    "The week's trades" means **closed** within Mon-Fri of the reviewed week.
    A position opened during the week and still open is not a result yet - it
    has no realized P&L and no exit to learn from - so it is returned separately
    and flagged rather than folded into the numbers. Silently including it would
    put an unfinished trade into a walk-away that is about how exits went.
    """
    from datetime import date as _date

    def _as_date(value):
        if isinstance(value, _date):
            return value
        return _date.fromisoformat(str(value)[:10])

    first, last = _as_date(monday), _as_date(friday)
    closed: list[JournalTrade] = []
    still_open: list[JournalTrade] = []
    for trade in load_trades():
        raw = trade.raw
        status = str(raw.get("status") or "").upper()
        closed_at = str(raw.get("closed_at") or "")[:10]
        opened_at = str(raw.get("opened_at") or "")[:10]
        if status == "CLOSED" and closed_at:
            try:
                when = _as_date(closed_at)
            except ValueError:
                continue
            if first <= when <= last:
                closed.append(trade)
            continue
        if opened_at:
            try:
                when = _as_date(opened_at)
            except ValueError:
                continue
            if first <= when <= last:
                still_open.append(trade)
    return {"closed": closed, "still_open": still_open}


def week_tag_candidates(monday: Any, friday: Any) -> list[dict[str, Any]]:
    """The week's closed trades with their auto-tag proposals, for the sub-pane.

    Only trades that actually have a proposal are returned - an empty suggestion
    list is not a review item, and padding the pane with them would make the
    weekly ritual look like more work than it is.
    """
    rows: list[dict[str, Any]] = []
    for trade in week_trades(monday, friday)["closed"]:
        candidates = auto_tag_candidates(trade.trade_id)
        if not candidates:
            continue
        rows.append(
            {
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "trade_date": trade.trade_date,
                "current_tags": str(trade.raw.get("setup_tags") or ""),
                "candidates": candidates,
            }
        )
    return rows


def correct_auto_tag(trade_id: str, tags: str) -> None:
    """Record a correction: the trader's wording wins and the tagger learns.

    ``accept_auto_tags`` is the confirm path; this is the other one. Both write
    through the trader-owned annotation, so neither is reachable by an import.
    """
    trade = _store().get_trade(trade_id) or {}
    save_annotation(trade_id, setup_tags=str(tags or "").strip(), notes=str(trade.get("notes") or ""))
    if str(tags or "").strip():
        _store().record_tag_corrections(trade, str(tags))
