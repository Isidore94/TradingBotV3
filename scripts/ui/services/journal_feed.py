from __future__ import annotations

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


def load_trades(
    *,
    broker: str = "All",
    account: str = "All",
    symbol: str = "",
) -> list[JournalTrade]:
    rows = _store().list_trades(
        broker=broker or "All",
        account=account or "All",
        symbol=(symbol or "").strip() or None,
    )
    return [JournalTrade.from_mapping(row) for row in rows]


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
