from __future__ import annotations

"""Context joins + sorting for the RS Window tab.

The bot answers "who led/lagged SPY over the selected M5 window"; this module
answers everything else about those names: is it a current bot pick (favorite
setup / tier / family), how strong is its industry and sector on the boards,
and how strong is the name itself on D1 and weekly timeframes. Pure Python
(no Qt) so every piece is unit-testable.
"""

import csv
from pathlib import Path
from typing import Any

# Session caches: classification/board joins and per-symbol daily strength are
# stable within a session; keyed by file mtimes so a fresh scan invalidates.
_industry_context_cache: dict[str, Any] = {}
_daily_strength_cache: dict[str, tuple[float, dict]] = {}

# yfinance classification sector names -> SPDR sector-board names.
_SECTOR_ALIASES = {
    "financial services": "Financials",
    "financial": "Financials",
    "healthcare": "Health Care",
    "consumer cyclical": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "basic materials": "Materials",
    "communication services": "Communication Services",
}

SORT_CHOICES = (
    ("Window RS/RW (excess vs SPY)", "excess"),
    ("Industry strength (side-aligned)", "industry_rs"),
    ("Sector strength (side-aligned)", "sector_rs"),
    ("D1 strength 20d (side-aligned)", "d1_rs_20d"),
    ("Weekly 8EMA streak (side-aligned)", "weekly_streak"),
)
# Metrics measured as "strength": for SHORT rows the sort flips sign so a
# short in a WEAK industry/sector/tape ranks high (alignment with the trade).
_SIDE_ALIGNED_KEYS = {"industry_rs", "sector_rs", "d1_rs_5d", "d1_rs_20d", "weekly_streak"}


def _read_csv_rows(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _to_float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_bot_pick_map(tier_list_path=None) -> dict[tuple[str, str], dict]:
    """(symbol, side) -> current bot pick row from the tier-list export."""
    if tier_list_path is None:
        from project_paths import MASTER_AVWAP_TIER_LIST_FILE

        tier_list_path = MASTER_AVWAP_TIER_LIST_FILE
    picks: dict[tuple[str, str], dict] = {}
    for row in _read_csv_rows(tier_list_path):
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        if not symbol or not side:
            continue
        picks[(symbol, side)] = {
            "tier": str(row.get("tier") or "").strip().upper(),
            "priority_bucket": str(row.get("priority_bucket") or "").strip(),
            "setup_family": str(row.get("setup_family") or "").strip(),
            "favorite_zone": str(row.get("favorite_zone") or "").strip(),
            "priority_score": _to_float(row.get("priority_score")),
        }
    return picks


def _sector_board_row(sector_name: str, sector_rows: list[dict]) -> dict | None:
    text = str(sector_name or "").strip()
    if not text:
        return None
    wanted = _SECTOR_ALIASES.get(text.lower(), text).lower()
    for row in sector_rows:
        if str(row.get("sector") or "").strip().lower() == wanted:
            return row
    return None


def load_industry_context_map(force_refresh: bool = False) -> dict[str, dict]:
    """symbol -> sector/industry names + board RS scores and ranks.

    Joins the shared symbol-classification cache to the sector board (SPDR
    ETFs vs SPY) and the industry index board (curated composite groups),
    reusing the industry scanner's own membership logic so a symbol lands in
    the same index row the Industry Board tab shows."""
    from industry_scanner import (
        INDUSTRY_BOARD_CSV_FILE,
        SECTOR_BOARD_CSV_FILE,
        collect_industry_members,
        load_custom_industry_groups,
        load_industry_index_definitions,
        load_symbol_classifications,
    )

    mtimes = tuple(
        Path(path).stat().st_mtime if Path(path).exists() else 0.0
        for path in (INDUSTRY_BOARD_CSV_FILE, SECTOR_BOARD_CSV_FILE)
    )
    if not force_refresh and _industry_context_cache.get("mtimes") == mtimes:
        return _industry_context_cache.get("map", {})

    classifications = load_symbol_classifications()
    industry_rows = _read_csv_rows(INDUSTRY_BOARD_CSV_FILE)
    sector_rows = _read_csv_rows(SECTOR_BOARD_CSV_FILE)
    industry_row_by_label = {str(row.get("industry") or "").strip(): row for row in industry_rows}

    members = collect_industry_members(
        classifications,
        load_custom_industry_groups(),
        index_definitions=load_industry_index_definitions(),
    )
    # Invert to symbol -> best board row (a symbol may sit in several indexes;
    # keep the one with the best RS rank so "strength to industry" reads the
    # strongest board line it belongs to).
    best_industry_by_symbol: dict[str, dict] = {}
    for label, group in members.items():
        board_row = industry_row_by_label.get(label)
        if board_row is None:
            continue
        rank = _to_float(board_row.get("rs_rank"))
        for symbol in group:
            symbol = str(symbol or "").strip().upper()
            current = best_industry_by_symbol.get(symbol)
            current_rank = _to_float(current.get("rs_rank")) if current else None
            if current is None or (
                rank is not None and (current_rank is None or rank < current_rank)
            ):
                best_industry_by_symbol[symbol] = {"label": label, **board_row}

    context: dict[str, dict] = {}
    for symbol, classification in classifications.items():
        sector_name = str(classification.get("sector") or "").strip()
        sector_row = _sector_board_row(sector_name, sector_rows)
        industry_row = best_industry_by_symbol.get(symbol)
        context[symbol] = {
            "sector": sector_name,
            "sector_rs": _to_float(sector_row.get("rs_score")) if sector_row else None,
            "sector_rank": _to_float(sector_row.get("rs_rank")) if sector_row else None,
            "industry": (
                str(industry_row.get("label") or "").strip()
                if industry_row
                else str(classification.get("industry") or "").strip()
            ),
            "industry_rs": _to_float(industry_row.get("rs_score")) if industry_row else None,
            "industry_rank": _to_float(industry_row.get("rs_rank")) if industry_row else None,
        }
    _industry_context_cache["mtimes"] = mtimes
    _industry_context_cache["map"] = context
    return context


def _daily_strength_for_symbol(symbol: str, spy_returns: dict[int, float | None]) -> dict:
    """D1 excess returns vs SPY + weekly 8EMA streak from the durable bar store."""
    from setup_playbook_study import _load_daily_frame, compute_weekly_streak_series

    frame = _load_daily_frame(symbol)
    if frame is None or len(frame) < 25:
        return {}
    closes = frame["close"].to_numpy(dtype=float)

    def _return_pct(sessions: int) -> float | None:
        if len(closes) <= sessions or not closes[-1 - sessions]:
            return None
        return (closes[-1] / closes[-1 - sessions] - 1.0) * 100.0

    result: dict[str, Any] = {}
    for sessions, key in ((5, "d1_rs_5d"), (20, "d1_rs_20d")):
        own = _return_pct(sessions)
        spy = spy_returns.get(sessions)
        result[key] = (own - spy) if (own is not None and spy is not None) else own
    streaks = compute_weekly_streak_series(frame)
    result["weekly_streak"] = int(streaks[-1]) if len(streaks) else 0
    return result


def daily_strength_map(symbols) -> dict[str, dict]:
    """symbol -> {d1_rs_5d, d1_rs_20d, weekly_streak}, cached per bar-file mtime."""
    from setup_playbook_study import _load_daily_frame

    spy_frame = _load_daily_frame("SPY")
    spy_returns: dict[int, float | None] = {5: None, 20: None}
    if spy_frame is not None and len(spy_frame) > 21:
        spy_closes = spy_frame["close"].to_numpy(dtype=float)
        for sessions in (5, 20):
            if spy_closes[-1 - sessions]:
                spy_returns[sessions] = (spy_closes[-1] / spy_closes[-1 - sessions] - 1.0) * 100.0

    from master_avwap_lib.legacy import MASTER_AVWAP_DAILY_BARS_DIR

    result: dict[str, dict] = {}
    for symbol in {str(s or "").strip().upper() for s in symbols or []}:
        if not symbol:
            continue
        bar_path = Path(MASTER_AVWAP_DAILY_BARS_DIR) / f"{symbol}.parquet"
        mtime = bar_path.stat().st_mtime if bar_path.exists() else 0.0
        cached = _daily_strength_cache.get(symbol)
        if cached is not None and cached[0] == mtime:
            result[symbol] = cached[1]
            continue
        values = _daily_strength_for_symbol(symbol, spy_returns)
        _daily_strength_cache[symbol] = (mtime, values)
        result[symbol] = values
    return result


def decorate_mover_rows(
    rows: list[dict],
    *,
    pick_map: dict | None = None,
    industry_map: dict | None = None,
    strength_map: dict | None = None,
) -> list[dict]:
    """Join bot pick / industry board / D1-weekly strength onto mover rows."""
    if pick_map is None:
        pick_map = load_bot_pick_map()
    if industry_map is None:
        industry_map = load_industry_context_map()
    if strength_map is None:
        strength_map = daily_strength_map([row.get("symbol") for row in rows])

    decorated = []
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        pick = pick_map.get((symbol, side)) or {}
        merged = {
            **row,
            "tier": pick.get("tier", ""),
            "setup_family": pick.get("setup_family", ""),
            "favorite_zone": pick.get("favorite_zone", ""),
            "favorite_setup": pick.get("priority_bucket", "") == "favorite_setup",
            **{key: value for key, value in (industry_map.get(symbol) or {}).items()},
            **{key: value for key, value in (strength_map.get(symbol) or {}).items()},
        }
        decorated.append(merged)
    return decorated


def filter_mover_rows(rows: list[dict], *, side: str = "", favorites_only: bool = False) -> list[dict]:
    side = str(side or "").strip().upper()
    kept = []
    for row in rows:
        if side in ("LONG", "SHORT") and str(row.get("side") or "").upper() != side:
            continue
        if favorites_only and not row.get("favorite_setup"):
            continue
        kept.append(row)
    return kept


def sort_mover_rows(rows: list[dict], key: str = "excess") -> list[dict]:
    """Best-first sort. Strength metrics are side-aligned: LONG rows rank by
    strength, SHORT rows by weakness, so a mixed list reads "most aligned with
    its trade direction first"."""
    key = key if key in {choice for _label, choice in SORT_CHOICES} else "excess"

    def sort_value(row: dict) -> tuple:
        value = _to_float(row.get(key))
        if value is not None and key in _SIDE_ALIGNED_KEYS:
            if str(row.get("side") or "").upper() == "SHORT":
                value = -value
        return (value is None, -(value if value is not None else 0.0))

    return sorted(rows, key=sort_value)
