"""Rank the RS/RW sweep with the trader's Focus names pinned to the top.

The RS/RW Board tab already ranks every scanned symbol against SPY, its
sector, and its industry. The question it does not answer at a glance is the
one the trader actually asks between alerts: *how are MY names ranking right
now?* This model answers that by splitting one snapshot into two lanes:

- **Focus lane** - every Focus pick that appears anywhere in the sweep, shown
  with the scope it ranks best in and its rank inside that scope.
- **Field lane** - the strongest and weakest of everything else, so the Focus
  names are read against the field rather than in isolation.

A Focus name that ranks on the wrong side of its thesis (a long pick showing
up in relative WEAKNESS) is marked misaligned rather than hidden: that
disagreement is the most decision-relevant thing the board can say.

Pure ranking logic, no widgets - the board widget renders what this returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

from ui.models.rrs import rrs_rows

# Scope order is also the tie-break when a symbol ranks equally well in more
# than one: vs SPY is the headline read, industry the most specialized.
SCOPES = ("SPY", "Sector", "Industry")
# One symbol appears in up to three scopes. Both lanes collapse it to a single
# best row so the board never spends its scarce rows repeating a name.
DEFAULT_FOCUS_LIMIT = 6
DEFAULT_FIELD_LIMIT = 6


@dataclass(frozen=True)
class StrengthRow:
    """One symbol's best relative-strength read in this snapshot."""

    symbol: str
    side: str  # "RS" (relative strength) or "RW" (relative weakness)
    rrs: float
    scope: str  # "SPY" | "Sector" | "Industry"
    rank: int  # 1-based within its scope and side
    in_focus: bool = False
    focus_side: str = ""  # "long" | "short" | "" when not a Focus pick
    focus_category: str = ""  # "swing" | "m5" | "both" | ""

    @property
    def aligned(self) -> bool:
        """Does the ranking agree with the direction the trader focused?

        A name with no Focus side cannot disagree, so it reads aligned.
        """
        if not self.focus_side:
            return True
        return (self.focus_side == "long") == (self.side == "RS")

    def rank_text(self) -> str:
        return f"#{self.rank} vs {self.scope}"


@dataclass
class StrengthBoard:
    """Both lanes plus the honest gaps, ready to render."""

    focus: list[StrengthRow] = dataclass_field(default_factory=list)
    strong: list[StrengthRow] = dataclass_field(default_factory=list)
    weak: list[StrengthRow] = dataclass_field(default_factory=list)
    # Focus picks the sweep did not rank at all. Named rather than silently
    # dropped: "not ranked" and "ranked badly" are different facts.
    unranked_focus: list[str] = dataclass_field(default_factory=list)
    timeframe: str = ""
    threshold: float | None = None
    timestamp: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.focus or self.strong or self.weak)

    @property
    def misaligned(self) -> list[StrengthRow]:
        return [row for row in self.focus if not row.aligned]


def focus_membership(focus_by_category: dict[str, dict[str, list[str]]] | None) -> dict[str, tuple[str, str]]:
    """``{SYMBOL: (side, category)}`` from ``all_focus_by_category()``.

    A symbol focused in both Swing and M5 reports ``"both"``: the categories
    are independent memberships and the board must not silently pick one.
    """
    membership: dict[str, tuple[str, str]] = {}
    for category, sides in (focus_by_category or {}).items():
        if not isinstance(sides, dict):
            continue
        for side, symbols in sides.items():
            if side not in {"long", "short"} or not isinstance(symbols, (list, tuple, set)):
                continue
            for raw in symbols:
                symbol = str(raw or "").strip().upper()
                if not symbol:
                    continue
                previous = membership.get(symbol)
                if previous is None:
                    membership[symbol] = (side, str(category or ""))
                elif previous[1] != str(category or ""):
                    membership[symbol] = (previous[0], "both")
    return membership


def _ranked_rows(payload: dict[str, Any] | None) -> dict[str, StrengthRow]:
    """Best row per symbol across every scope.

    "Best" is the smallest rank inside its own side, so a name that is #1
    against its industry but #14 against SPY is reported where it looks
    strongest - with the scope shown, never implied.
    """
    best: dict[str, StrengthRow] = {}
    for scope in SCOPES:
        rows = rrs_rows(payload, scope)
        strong = sorted((row for row in rows if row.side == "RS"), key=lambda row: -row.rrs)
        weak = sorted((row for row in rows if row.side == "RW"), key=lambda row: row.rrs)
        for ranked in (strong, weak):
            for index, row in enumerate(ranked, start=1):
                symbol = str(row.symbol or "").strip().upper()
                if not symbol:
                    continue
                candidate = StrengthRow(
                    symbol=symbol,
                    side=row.side,
                    rrs=float(row.rrs),
                    scope=scope,
                    rank=index,
                )
                current = best.get(symbol)
                if current is None or candidate.rank < current.rank:
                    best[symbol] = candidate
    return best


def build_strength_board(
    payload: dict[str, Any] | None,
    focus_by_category: dict[str, dict[str, list[str]]] | None = None,
    *,
    focus_limit: int = DEFAULT_FOCUS_LIMIT,
    field_limit: int = DEFAULT_FIELD_LIMIT,
) -> StrengthBoard:
    """Split one RRS snapshot into the Focus lane and the field lane."""
    payload = payload if isinstance(payload, dict) else {}
    membership = focus_membership(focus_by_category)
    best = _ranked_rows(payload)

    focus_rows: list[StrengthRow] = []
    field_rows: list[StrengthRow] = []
    for symbol, row in best.items():
        if symbol in membership:
            side, category = membership[symbol]
            focus_rows.append(
                StrengthRow(
                    symbol=row.symbol,
                    side=row.side,
                    rrs=row.rrs,
                    scope=row.scope,
                    rank=row.rank,
                    in_focus=True,
                    focus_side=side,
                    focus_category=category,
                )
            )
        else:
            field_rows.append(row)

    # Misaligned Focus names lead: a long pick sitting in relative weakness is
    # the row worth reading first. Then strongest-first by absolute conviction.
    focus_rows.sort(key=lambda row: (row.aligned, row.rank, -abs(row.rrs), row.symbol))
    strong = sorted((row for row in field_rows if row.side == "RS"), key=lambda row: (-row.rrs, row.symbol))
    weak = sorted((row for row in field_rows if row.side == "RW"), key=lambda row: (row.rrs, row.symbol))

    return StrengthBoard(
        focus=focus_rows[: max(0, int(focus_limit))],
        strong=strong[: max(0, int(field_limit))],
        weak=weak[: max(0, int(field_limit))],
        unranked_focus=sorted(symbol for symbol in membership if symbol not in best),
        timeframe=str(payload.get("timeframe_key") or ""),
        threshold=_float(payload.get("threshold")),
        timestamp=str(payload.get("timestamp") or ""),
    )


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
