from __future__ import annotations

from datetime import date, timedelta
from typing import Any


DEFAULT_QUESTRADE_PULL_DAYS = 7


def recent_import_dates(window_days: int, *, today: date | None = None) -> list[date]:
    """Return oldest-to-newest calendar dates for a recent broker import."""
    try:
        days = int(window_days)
    except (TypeError, ValueError):
        days = DEFAULT_QUESTRADE_PULL_DAYS
    days = max(1, min(days, 31))
    end = today or date.today()
    start = end - timedelta(days=days - 1)
    return [start + timedelta(days=offset) for offset in range(days)]


def summarize_import_results(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return "Questrade import returned no day summaries."

    total = sum(int(item.get("total_imported") or 0) for item in summaries if isinstance(item, dict))
    failures = [item for item in summaries if isinstance(item, dict) and item.get("status") == "FAILED"]
    rebuilt = [item.get("trade_count") for item in summaries if isinstance(item, dict) and item.get("trade_count") is not None]
    latest_trade_count = rebuilt[-1] if rebuilt else None
    status = "finished with errors" if failures else "complete"

    parts = [
        f"Questrade import {status}: {total} execution(s) across {len(summaries)} day(s).",
    ]
    if latest_trade_count is not None:
        parts.append(f"Grouped trades now: {latest_trade_count}.")

    notable_messages: list[str] = []
    for item in summaries[-3:]:
        if not isinstance(item, dict):
            continue
        target_date = str(item.get("target_date") or "").strip()
        messages = "; ".join(str(message) for message in item.get("messages") or [] if str(message).strip())
        if target_date and messages:
            notable_messages.append(f"{target_date}: {messages}")
    if notable_messages:
        parts.append("Recent: " + " | ".join(notable_messages))

    return " ".join(parts)
