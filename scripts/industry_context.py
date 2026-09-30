"""Symbol -> sector/industry context (names, board RS, primary industry members). Pure, Qt-free.

Lifted out of ``ui.services.rs_window_feed`` unchanged so non-UI packages (the Trade
Mentor's packs) can read it; ``rs_window_feed`` re-exports every name.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

# Session cache keyed by the source files' mtimes, so a fresh scan invalidates.
_industry_context_cache: dict[str, Any] = {}

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


def _sector_board_row(sector_name: str, sector_rows: list[dict]) -> dict | None:
    text = str(sector_name or "").strip()
    if not text:
        return None
    wanted = _SECTOR_ALIASES.get(text.lower(), text).lower()
    for row in sector_rows:
        if str(row.get("sector") or "").strip().lower() == wanted:
            return row
    return None


def _norm_label(value: object) -> str:
    return " ".join(str(value or "").split()).lower()


def build_primary_industry_context(
    classifications: Mapping[str, dict],
    industry_rows: list[dict],
    members: Mapping[str, list[str]],
    index_definitions: Mapping[str, dict],
) -> dict[str, dict]:
    """Resolve one deterministic primary board index per symbol.

    Curated industry definitions intentionally overlap. The old GUI chose the
    membership with the strongest current rank, which leaks the outcome into
    classification. Primary now follows the symbol's classification and the
    first declared taxonomy mapping; all other memberships remain visible as
    additional context.
    """

    board_by_norm = {
        _norm_label(row.get("industry")): row
        for row in industry_rows
        if _norm_label(row.get("industry"))
    }
    membership_by_symbol: dict[str, list[str]] = {}
    for label, symbols in members.items():
        for symbol in symbols:
            normalized = str(symbol or "").strip().upper()
            if normalized:
                membership_by_symbol.setdefault(normalized, []).append(str(label))

    raw_members: dict[str, list[str]] = {}
    for raw_symbol, classification in classifications.items():
        raw_label = str(classification.get("industry") or "").strip()
        normalized = str(raw_symbol or "").strip().upper()
        if raw_label and normalized:
            raw_members.setdefault(_norm_label(raw_label), []).append(normalized)

    result: dict[str, dict] = {}
    for raw_symbol, classification in classifications.items():
        symbol = str(raw_symbol or "").strip().upper()
        raw_industry = str(classification.get("industry") or "").strip()
        raw_norm = _norm_label(raw_industry)
        memberships = membership_by_symbol.get(symbol, [])
        primary = ""
        source = "unmapped"
        if raw_norm in board_by_norm:
            primary = str(board_by_norm[raw_norm].get("industry") or raw_industry)
            source = "exact_classification"
        else:
            # Definitions retain JSON declaration order. Only taxonomy matches
            # can become primary; explicit cross-theme ticker memberships stay
            # additional so they cannot displace the true industry.
            for label, spec in index_definitions.items():
                definition_industries = {_norm_label(value) for value in spec.get("industries") or []}
                if raw_norm and raw_norm in definition_industries and label in memberships:
                    primary = str(label)
                    source = "classification_definition"
                    break
        if not primary and raw_industry:
            # A Yahoo/IB classification with no curated board mapping is still
            # safer than borrowing an unrelated ticker/theme membership. It
            # can form an intraday raw-classification composite, while its
            # daily board RS remains honestly unavailable.
            primary = raw_industry
            source = "raw_classification"
        if not primary:
            non_custom = [label for label in memberships if not str(label).endswith("*")]
            if non_custom:
                primary = non_custom[0]
                source = "deterministic_fallback"
            elif memberships:
                primary = memberships[0]
                source = "custom_fallback"

        primary_row = board_by_norm.get(_norm_label(primary))
        additional = [label for label in memberships if _norm_label(label) != _norm_label(primary)]
        member_symbols = list(
            members.get(primary)
            or raw_members.get(_norm_label(primary))
            or []
        )
        result[symbol] = {
            "industry": primary or raw_industry,
            "industry_classification": raw_industry,
            "industry_primary_source": source,
            "additional_industries": additional,
            "industry_member_symbols": member_symbols,
            "industry_expected_members": len(member_symbols),
            "industry_rs": _to_float(primary_row.get("rs_score")) if primary_row else None,
            "industry_rank": _to_float(primary_row.get("rs_rank")) if primary_row else None,
            "industry_return_1d_pct": _to_float(primary_row.get("pct_change_1d")) if primary_row else None,
            "industry_return_5d_pct": _to_float(primary_row.get("return_5d_pct")) if primary_row else None,
        }
    return result


def load_industry_context_map(force_refresh: bool = False) -> dict[str, dict]:
    """symbol -> sector/industry names + board RS scores and ranks.

    Joins the shared symbol-classification cache to the sector board (SPDR
    ETFs vs SPY) and the industry index board (curated composite groups),
    reusing the industry scanner's own membership logic so a symbol lands in
    the same index row the Industry Board tab shows."""
    from industry_scanner import (
        INDUSTRY_BOARD_CSV_FILE,
        CUSTOM_INDUSTRY_GROUPS_FILE,
        INDUSTRY_INDEX_DEFINITIONS_FILE,
        SECTOR_BOARD_CSV_FILE,
        collect_industry_members,
        load_custom_industry_groups,
        load_industry_index_definitions,
        load_symbol_classifications,
    )
    from project_paths import SYMBOL_CLASSIFICATION_CACHE_FILE

    mtimes = tuple(
        Path(path).stat().st_mtime if Path(path).exists() else 0.0
        for path in (
            INDUSTRY_BOARD_CSV_FILE,
            SECTOR_BOARD_CSV_FILE,
            CUSTOM_INDUSTRY_GROUPS_FILE,
            INDUSTRY_INDEX_DEFINITIONS_FILE,
            SYMBOL_CLASSIFICATION_CACHE_FILE,
        )
    )
    if not force_refresh and _industry_context_cache.get("mtimes") == mtimes:
        return _industry_context_cache.get("map", {})

    classifications = load_symbol_classifications()
    industry_rows = _read_csv_rows(INDUSTRY_BOARD_CSV_FILE)
    sector_rows = _read_csv_rows(SECTOR_BOARD_CSV_FILE)
    custom_groups = load_custom_industry_groups()
    index_definitions = load_industry_index_definitions()
    members = collect_industry_members(
        classifications,
        custom_groups,
        index_definitions=index_definitions,
    )
    primary_context = build_primary_industry_context(
        classifications,
        industry_rows,
        members,
        index_definitions,
    )

    context: dict[str, dict] = {}
    for raw_symbol, classification in classifications.items():
        symbol = str(raw_symbol or "").strip().upper()
        sector_name = str(classification.get("sector") or "").strip()
        sector_row = _sector_board_row(sector_name, sector_rows)
        industry = primary_context.get(symbol) or {}
        context[symbol] = {
            "sector": sector_name,
            "sector_rs": _to_float(sector_row.get("rs_score")) if sector_row else None,
            "sector_rank": _to_float(sector_row.get("rs_rank")) if sector_row else None,
            "sector_return_1d_pct": _to_float(sector_row.get("pct_change_1d")) if sector_row else None,
            "sector_return_5d_pct": _to_float(sector_row.get("return_5d_pct")) if sector_row else None,
            **industry,
        }
    _industry_context_cache["mtimes"] = mtimes
    _industry_context_cache["map"] = context
    return context
