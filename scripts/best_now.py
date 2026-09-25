"""P1-5 5b: one ranked "Best right now" list. Pure; display and ranking only.

Three inputs, one list:
- today's M5 alerts as `live_alert_results` rows (grade, live R, entry, stop);
- the Movers board's dip-strong names (`board["dip"]["long"]`);
- the setups table's D1 names (`swing_context` map). A D1 name enters only
  with an M5 confirmation: an M5 alert on the same symbol and side. That is the
  `held_run_score` rule ("an M5 alert on a name that also carries a D1 setup
  outranks the same alert on a name that does not"), used here for order only.

Order: D1 + M5 first, then M5 alone, then dip-strong alone. Inside a tier:
setup grade, then live R, then the earlier alert. A stopped alert is left out:
it is not "best right now". Nothing here changes an alert, a score or a list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

#: Rows shown by the strip.
BEST_NOW_LIMIT = 5

TIER_D1_M5 = 0
TIER_M5 = 1
TIER_DIP = 2

_STOPPED = "stopped"
_OPEN = "open"


@dataclass(frozen=True)
class BestNowEntry:
    """One row of the strip. Frozen, so two lists compare by value."""

    symbol: str
    side: str
    tier: int
    why: str
    entry: float | None
    stop: float | None

    @property
    def key(self) -> tuple[str, str]:
        return (self.symbol, self.side)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _side(value: Any) -> str:
    text = str(value or "").strip().upper()
    return {"L": "LONG", "BUY": "LONG", "S": "SHORT", "SELL": "SHORT"}.get(text, text)


def _grade_rank(grade: Any) -> int:
    import setup_grades

    return setup_grades.sort_rank(grade)


def _format_r(value: float | None) -> str:
    if value is None:
        return "no data"
    return f"{value:+.1f}R"


def _d1_context(swing_context: Mapping[Any, Any] | None, symbol: str, side: str):
    if not swing_context:
        return None
    try:
        import swing_context as swing_context_module

        return swing_context_module.context_for(swing_context, symbol, side)
    except Exception:  # noqa: BLE001 - a hint never costs the list
        return swing_context.get((symbol, side))


def _d1_text(context: Any) -> str:
    if isinstance(context, Mapping):
        grade = str(context.get("grade") or "").strip()
        return f"D1 {grade}".strip() if grade else "D1 setup"
    return "D1 setup"


def rank_best_now(
    alert_results: Iterable[Mapping[str, Any]] | None,
    dip_strong: Iterable[Mapping[str, Any]] | None = None,
    swing_context: Mapping[Any, Any] | None = None,
    *,
    limit: int | None = BEST_NOW_LIMIT,
) -> list[BestNowEntry]:
    """The ranked list. Pure; the same inputs always give the same list."""
    import setup_grades

    ranked: list[tuple[tuple, BestNowEntry]] = []
    seen: set[tuple[str, str]] = set()
    dip_by_symbol: dict[str, Mapping[str, Any]] = {}
    for row in dip_strong or ():
        symbol = str((row or {}).get("symbol") or "").strip().upper()
        if symbol and symbol not in dip_by_symbol:
            dip_by_symbol[symbol] = row

    for order, row in enumerate(alert_results or ()):
        symbol = str(row.get("symbol") or "").strip().upper()
        side = _side(row.get("side"))
        if not symbol or side not in {"LONG", "SHORT"} or (symbol, side) in seen:
            continue
        if row.get("status") == _STOPPED:
            continue
        seen.add((symbol, side))
        live_r = _number(row.get("r")) if row.get("status") == _OPEN else None
        grade = row.get("grade")
        context = _d1_context(swing_context, symbol, side)
        tier = TIER_D1_M5 if context else TIER_M5
        parts = [f"M5 {setup_grades.badge(grade)} {_format_r(live_r)}"]
        if context:
            parts.insert(0, _d1_text(context))
        dip = dip_by_symbol.get(symbol) if side == "LONG" else None
        if dip is not None:
            parts.append("dip-strong")
        received = row.get("received_at")
        sort_key = (
            tier,
            _grade_rank(grade),
            live_r is None,
            -(live_r if live_r is not None else 0.0),
            str(received or ""),
            order,
        )
        ranked.append(
            (
                sort_key,
                BestNowEntry(
                    symbol=symbol,
                    side=side,
                    tier=tier,
                    why=" + ".join(parts),
                    entry=_number(row.get("entry")),
                    stop=_number(row.get("stop")),
                ),
            )
        )

    for order, (symbol, row) in enumerate(dip_by_symbol.items()):
        if (symbol, "LONG") in seen:
            continue
        seen.add((symbol, "LONG"))
        score = _number(row.get("dip_score"))
        since = _number(row.get("since_start_pct"))
        why = "Dip-strong" + (f" {score:+.1f}" if score is not None else "")
        if since is not None:
            why += f" ({since:+.1f}% since turn)"
        # The board has no trade plan: entry is the last price, stop the day's low.
        ranked.append(
            (
                (TIER_DIP, 0, score is None, -(score or 0.0), "", order),
                BestNowEntry(
                    symbol=symbol,
                    side="LONG",
                    tier=TIER_DIP,
                    why=why,
                    entry=_number(row.get("last")),
                    stop=_number(row.get("lod")),
                ),
            )
        )

    ranked.sort(key=lambda item: item[0])
    entries = [entry for _key, entry in ranked]
    return entries if limit is None else entries[: max(0, int(limit))]


def _price(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "?"


def entry_text(entry: BestNowEntry) -> str:
    """One strip row, two lines: symbol, side and why; then entry (e) and stop (s)."""
    side = "L" if entry.side == "LONG" else "S"
    return f"{entry.symbol} {side}  {entry.why}\n e {_price(entry.entry)} s {_price(entry.stop)}"


def diff_rows(old: Sequence[str], new: Sequence[str]) -> list[int]:
    """The row indexes whose text changed (a length change counts every row past the shorter)."""
    changed = [index for index in range(min(len(old), len(new))) if old[index] != new[index]]
    changed.extend(range(min(len(old), len(new)), max(len(old), len(new))))
    return changed


def dip_strong_rows(board: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """The Movers board's dip-strong list, or [] when the board has none."""
    dip = (board or {}).get("dip") if isinstance(board, Mapping) else None
    rows = dip.get("long") if isinstance(dip, Mapping) else None
    return [row for row in rows or () if isinstance(row, Mapping)]
