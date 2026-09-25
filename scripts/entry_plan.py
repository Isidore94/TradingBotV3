"""P1-6 6b/6c: the D1 trade plan as table cells, position size, and entry grade. Pure.

* `plan_for_row` projects the SAME plan the D1 detail pane shows
  (`setup_docs.build_trade_plan`, fed exactly as `_show_setup_detail` feeds it)
  into entry / stop / TP1 / TP1 R and a stale flag.
* `shares_for` is the trader's fixed-dollar sizing (decided 2026-09-24): shares =
  floor(risk dollars / |entry - stop|), blank when the stop is unknown or the
  setting is off. No account size is stored. NEVER an order.
* `entry_grade` = (actual entry - planned entry) / (planned entry - planned stop):
  positive = paid up (worse than planned), negative = a better fill, in R.

Missing data is blank (None), never zero. Nothing here reads a file.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

#: The one local setting (`local_settings.json`). Blank / absent = sizing off.
RISK_SETTING = "risk_per_trade_dollars"


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        number = float(str(value).strip().replace(",", "").lstrip("$")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------- 6c sizing
def parse_risk_dollars(value: Any) -> float | None:
    """A positive dollar amount, or None (blank, zero, negative or not a number)."""
    number = _number(value)
    return number if number is not None and number > 0 else None


def validate_risk_text(text: str) -> tuple[bool, float | None]:
    """(ok, value) for the Settings box: blank is ok and means off."""
    if not str(text or "").strip():
        return True, None
    value = parse_risk_dollars(text)
    return (value is not None), value


def risk_per_trade_dollars() -> float | None:
    """The saved setting (cached settings read), or None when off or unreadable."""
    try:
        from project_paths import get_local_setting

        return parse_risk_dollars(get_local_setting(RISK_SETTING, None))
    except Exception:  # noqa: BLE001 - sizing is a hint; unreadable means off
        return None


_SIDES = {"LONG": "LONG", "BUY": "LONG", "SHORT": "SHORT", "SELL": "SHORT"}


def shares_for(risk_dollars: Any, entry: Any, stop: Any, side: Any = None) -> int | None:
    """floor(risk / |entry - stop|), or None when any input is missing or the gap is zero.

    With a side, a stop on the wrong side of the entry (a long's stop above it,
    a short's below) sizes nothing - the same rule the D1 plan's stale flag keeps.
    """
    risk = parse_risk_dollars(risk_dollars)
    entry_value, stop_value = _number(entry), _number(stop)
    if risk is None or entry_value is None or stop_value is None:
        return None
    direction = _SIDES.get(str(side or "").strip().upper())
    if direction == "LONG" and stop_value >= entry_value:
        return None
    if direction == "SHORT" and stop_value <= entry_value:
        return None
    gap = abs(entry_value - stop_value)
    if gap <= 0:
        return None
    return int(math.floor(risk / gap + 1e-9))


def notional_text(value: float | None) -> str:
    """`$800`, `$10.0k`, `$2.0M` - the dollars a size puts to work. No cap: the trader decides."""
    if value is None:
        return ""
    if value >= 999_950:
        return f"${value / 1_000_000:,.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:,.1f}k"
    return f"${value:,.0f}"


def size_text(shares: int | None, entry: Any = None) -> str:
    """`1,000 sh · $10.0k` (shares and notional), `10 sh` with no price, '' with no size."""
    if shares is None:
        return ""
    price = _number(entry)
    text = f"{shares:,} sh"
    return f"{text} · {notional_text(shares * price)}" if price else text


# ---------------------------------------------------------------- 6b plan
def plan_for_row(
    *,
    symbol: str,
    side: str,
    setup_family: Any,
    setup_tags: Any = (),
    favorite_signals: Any = None,
    last_close: Any = None,
    levels_by_symbol: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    """The detail pane's plan for one D1 row, or None when the scan has no levels for it."""
    from setup_docs import build_trade_plan, resolve_setup_family_from_candidates

    symbol = str(symbol or "").strip().upper()
    levels = (levels_by_symbol or {}).get(symbol) if symbol else None
    if not isinstance(levels, Mapping):
        return None
    tags = [str(tag) for tag in (setup_tags or ()) if str(tag).strip()]
    family = resolve_setup_family_from_candidates([setup_family, *tags])
    signals = list(favorite_signals or tags or [])
    plan = build_trade_plan(
        side=str(side or "LONG"),
        setup_family=family,
        favorite_signals=signals,
        bands=levels.get("bands") or {},
        vwap=levels.get("vwap"),
        atr20=levels.get("atr20"),
        last_close=_number(last_close) or levels.get("last_close"),
    )
    entry = _number(plan.get("entry_reference"))
    stop = _number(plan.get("stop_price"))
    risk = _number(plan.get("risk_per_share"))
    return {
        "side": str(plan.get("side") or side or "LONG"),
        "entry": entry,
        "stop": stop,
        "stop_label": str(plan.get("stop_label") or ""),
        "tp1": _number(plan.get("partial_price")),
        "tp1_label": str(plan.get("partial_label") or ""),
        "tp1_r": _number(plan.get("partial_r")),
        "risk": risk,
        # The detail pane's stale rule, with the unknown case kept apart: a
        # plan is stale only when both prices are known and price is past the stop.
        "stale": entry is not None and stop is not None and risk is None,
    }


def plan_for_setup_row(row: Any, levels_by_symbol) -> dict[str, Any] | None:
    """`plan_for_row` for a `SetupRow`, fed as `_show_setup_detail` feeds the pane."""
    raw = row.raw if isinstance(getattr(row, "raw", None), dict) else {}
    return plan_for_row(
        symbol=getattr(row, "symbol", ""),
        side=getattr(row, "side", "") or str(raw.get("side") or "LONG"),
        setup_family=raw.get("setup_family"),
        setup_tags=getattr(row, "setup_tags", ()) or (),
        favorite_signals=raw.get("favorite_signals"),
        last_close=raw.get("last_close") or raw.get("previous_close"),
        levels_by_symbol=levels_by_symbol,
    )


def _price(value: float | None) -> str:
    return f"{value:,.2f}" if value is not None else ""


def plan_cells(plan: Mapping[str, Any] | None, risk_dollars: Any = None) -> dict[str, str]:
    """Display text per plan column. Blank = unknown; `stale` rides the R cell."""
    if not plan:
        return {"plan_entry": "", "plan_stop": "", "plan_tp1": "", "plan_r": "", "plan_shares": ""}
    tp1_r = plan.get("tp1_r")
    r_text = "stale" if plan.get("stale") else (f"{tp1_r:+.1f}R" if tp1_r is not None else "")
    shares = plan_shares(plan, risk_dollars)
    return {
        "plan_entry": _price(plan.get("entry")),
        "plan_stop": _price(plan.get("stop")),
        "plan_tp1": _price(plan.get("tp1")),
        "plan_r": r_text,
        "plan_shares": size_text(shares, plan.get("entry")),
    }


def plan_shares(plan: Mapping[str, Any] | None, risk_dollars: Any = None) -> int | None:
    """Shares for a plan: blank when stale, the stop is unknown, or it sits on the wrong side."""
    if not plan or plan.get("stale"):
        return None
    return shares_for(risk_dollars, plan.get("entry"), plan.get("stop"), plan.get("side"))


def plan_line(plan: Mapping[str, Any] | None, risk_dollars: Any = None) -> str:
    """One tooltip line: `Plan: entry 10.00 · stop 9.50 (LOWER_1) · TP1 11.00 (+2.0R) · 40 sh`."""
    if not plan:
        return ""
    cells = plan_cells(plan, risk_dollars)
    parts = [f"entry {cells['plan_entry'] or '?'}"]
    stop = cells["plan_stop"] or "?"
    parts.append(f"stop {stop}" + (f" ({plan['stop_label']})" if plan.get("stop_label") else ""))
    tp1 = cells["plan_tp1"] or "?"
    parts.append(f"TP1 {tp1}" + (f" ({cells['plan_r']})" if cells["plan_r"] else ""))
    if cells["plan_shares"]:
        parts.append(cells["plan_shares"])
    return "Plan: " + " · ".join(parts)


# ---------------------------------------------------------------- 6c journal grade
def entry_grade(actual_entry: Any, planned_entry: Any, planned_stop: Any) -> float | None:
    """(actual - planned) / (planned - stop), in R. None when any part is missing or 1R is zero."""
    actual, planned, stop = _number(actual_entry), _number(planned_entry), _number(planned_stop)
    if actual is None or planned is None or stop is None or not actual or not planned or not stop:
        return None
    one_r = planned - stop
    if abs(one_r) < 1e-12:
        return None
    return (actual - planned) / one_r


def entry_grade_text(grade: float | None) -> str:
    if grade is None:
        return "Entry grade: - (needs planned entry and stop)"
    word = "paid up" if grade > 0.005 else "better than plan" if grade < -0.005 else "on plan"
    return f"Entry grade: {grade:+.2f}R ({word})"
