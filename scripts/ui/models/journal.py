from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class JournalTrade:
    trade_id: str
    trade_date: str = ""
    symbol: str = ""
    direction: str = ""
    status: str = ""
    quantity: float | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    net_pnl: float | None = None
    fees: float | None = None
    account: str = ""
    broker: str = ""
    tags: str = ""
    notes: str = ""
    opened_at: str = ""
    closed_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "JournalTrade":
        quantity = _float_or_none(row.get("quantity_closed")) or _float_or_none(row.get("quantity_opened"))
        return cls(
            trade_id=str(row.get("trade_id") or ""),
            trade_date=str(row.get("trade_date") or "")[:10],
            symbol=str(row.get("symbol") or "").upper(),
            direction=str(row.get("direction") or "").upper(),
            status=str(row.get("status") or "").upper(),
            quantity=quantity,
            entry_price=_float_or_none(row.get("average_entry_price")),
            exit_price=_float_or_none(row.get("average_exit_price")),
            net_pnl=_float_or_none(row.get("net_pnl")),
            fees=_sum_costs(row),
            account=str(row.get("account_label") or row.get("account_number") or ""),
            broker=str(row.get("broker") or ""),
            tags=str(row.get("display_tags") or row.get("setup_tags") or row.get("auto_tag_summary") or ""),
            notes=str(row.get("notes") or ""),
            opened_at=str(row.get("opened_at") or ""),
            closed_at=str(row.get("closed_at") or ""),
            raw=dict(row),
        )

    @property
    def is_closed(self) -> bool:
        return self.status == "CLOSED"


def _sum_costs(row: dict[str, Any]) -> float | None:
    commission = _float_or_none(row.get("commission"))
    fees = _float_or_none(row.get("fees"))
    if commission is None and fees is None:
        return None
    return (commission or 0.0) + (fees or 0.0)


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
