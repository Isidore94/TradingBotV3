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


def export_trades_csv() -> Path:
    return _store().export_trades_csv()


def rebuild_trades() -> int:
    return _store().rebuild_trades()


def list_import_runs(limit: int = 10) -> list[dict[str, Any]]:
    try:
        return _store().list_import_runs(limit=limit)
    except Exception:
        return []
