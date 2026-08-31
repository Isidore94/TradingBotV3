from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThetaRow:
    symbol: str
    play_type: str = "sold_put"
    score: int | None = None
    support_count: int | None = None
    close: float | None = None
    atr: str = ""
    next_earnings_days: int | None = None
    next_earnings_label: str = ""
    recommended_expiration: str = ""
    recommended_strike: float | None = None
    recommended_long_strike: float | None = None
    recommended_credit: float | None = None
    recommended_credit_source: str = ""
    primary_strike_band: str = ""
    liquidity_score: str = ""
    # Phase 0.11: the premium facts the report now states per sold-put row.
    # None means "this row has no quoted credit to describe" (a support-only
    # strike zone, or a PCS row measured on credit/width instead) - never zero,
    # which would read as a measurement.
    credit_pct_of_strike: float | None = None
    credit_pct_per_week: float | None = None
    spread_pct: float | None = None
    major_sma_above_strike: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "ThetaRow":
        return cls(
            symbol=str(row.get("symbol") or "").strip().upper(),
            play_type=str(row.get("play_type") or "sold_put").strip() or "sold_put",
            score=_int_or_none(row.get("score")),
            support_count=_int_or_none(row.get("support_count")),
            close=_float_or_none(row.get("close")),
            atr=str(row.get("atr") or ""),
            next_earnings_days=_int_or_none(row.get("next_earnings_days")),
            next_earnings_label=str(row.get("next_earnings_label") or ""),
            recommended_expiration=str(row.get("recommended_expiration") or ""),
            recommended_strike=_float_or_none(row.get("recommended_strike")),
            recommended_long_strike=_float_or_none(row.get("recommended_long_strike")),
            recommended_credit=_float_or_none(row.get("recommended_credit")),
            recommended_credit_source=str(row.get("recommended_credit_source") or ""),
            primary_strike_band=str(row.get("primary_strike_band") or ""),
            liquidity_score=str(row.get("liquidity_score") or ""),
            credit_pct_of_strike=_float_or_none(row.get("credit_pct_of_strike")),
            credit_pct_per_week=_float_or_none(row.get("credit_pct_per_week")),
            spread_pct=_float_or_none(row.get("spread_pct")),
            major_sma_above_strike=_int_or_none(row.get("major_sma_above_strike")),
            raw=dict(row),
        )

    @property
    def play_label(self) -> str:
        return "PCS" if self.play_type == "pcs" else "Sold Put"


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
