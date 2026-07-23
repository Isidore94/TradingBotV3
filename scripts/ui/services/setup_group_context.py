from __future__ import annotations

"""Local sector/industry context for Master AVWAP setup display rows.

This is display-only enrichment. It reads the existing classification/board
snapshots and durable D1 store; it never fetches data or changes setup scores.
"""

import csv
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from project_paths import MASTER_AVWAP_UNMAPPED_CLASSIFICATIONS_FILE
from ui.models.setup import SetupRow
from ui.services.rs_window_feed import load_industry_context_map


def weighted_d1_excess(
    stock_1d: float | None,
    stock_5d: float | None,
    reference_1d: float | None,
    reference_5d: float | None,
) -> float | None:
    """House D1 RS/RW: 35% one-session + 65% five-session excess return."""
    if None in (stock_1d, stock_5d, reference_1d, reference_5d):
        return None
    return round(
        0.35 * (float(stock_1d) - float(reference_1d))
        + 0.65 * (float(stock_5d) - float(reference_5d)),
        3,
    )


def _float_or_none(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _returns_from_bars(bars: list[dict]) -> tuple[float | None, float | None]:
    closes = [float(bar["close"]) for bar in bars if _float_or_none(bar.get("close")) is not None]
    if len(closes) < 6 or closes[-2] == 0 or closes[-6] == 0:
        return None, None
    return (
        (closes[-1] / closes[-2] - 1.0) * 100.0,
        (closes[-1] / closes[-6] - 1.0) * 100.0,
    )


def _supplemental_returns(rows: Iterable[SetupRow]) -> dict[str, tuple[float | None, float | None]]:
    result: dict[str, tuple[float | None, float | None]] = {}
    for row in rows:
        raw = row.raw if isinstance(row.raw, dict) else {}
        one_day = _float_or_none(raw.get("symbol_one_day_return_pct"))
        five_day = _float_or_none(raw.get("symbol_five_day_return_pct"))
        if row.symbol and one_day is not None and five_day is not None:
            result.setdefault(row.symbol, (one_day, five_day))
    return result


def enrich_setup_group_context(
    rows: list[SetupRow],
    *,
    supplemental_rows: Iterable[SetupRow] = (),
    context_map: Mapping[str, dict] | None = None,
    daily_returns: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> list[SetupRow]:
    """Attach group names and D1 stock-vs-group readings to setup rows.

    ``supplemental_rows`` lets a fresh text report reuse the prior rich focus
    payload's already-computed stock returns. Only symbols absent there need a
    durable-parquet read, which keeps GUI refreshes bounded.
    """
    context_map = context_map if context_map is not None else load_industry_context_map()
    return_map = _supplemental_returns(supplemental_rows)
    return_map.update(daily_returns or {})

    missing = {
        row.symbol
        for row in rows
        if row.symbol and row.symbol not in return_map
    }
    if missing:
        from chart_snapshot import load_d1_bars

        for symbol in sorted(missing):
            return_map[symbol] = _returns_from_bars(load_d1_bars(symbol))

    for row in rows:
        context = context_map.get(row.symbol) or {}
        row.sector = str(context.get("sector") or "").strip()
        row.industry = str(context.get("industry") or "").strip()
        row.industry_classification_source = str(
            context.get("industry_primary_source") or "unmapped"
        )
        stock_1d, stock_5d = return_map.get(row.symbol, (None, None))
        row.d1_vs_sector = weighted_d1_excess(
            stock_1d,
            stock_5d,
            _float_or_none(context.get("sector_return_1d_pct")),
            _float_or_none(context.get("sector_return_5d_pct")),
        )
        row.d1_vs_industry = weighted_d1_excess(
            stock_1d,
            stock_5d,
            _float_or_none(context.get("industry_return_1d_pct")),
            _float_or_none(context.get("industry_return_5d_pct")),
        )
    return rows


def unmapped_setup_classifications(rows: Iterable[SetupRow]) -> list[dict[str, str]]:
    """Return one deterministic AI-review row per unresolved symbol."""
    by_symbol: dict[str, dict[str, str]] = {}
    review_sources = {
        "",
        "unmapped",
        "raw_classification",
        "deterministic_fallback",
        "custom_fallback",
    }
    for row in rows:
        source = str(row.industry_classification_source or "").strip()
        reason = ""
        if not row.industry:
            reason = "missing industry classification"
        elif source in review_sources:
            reason = "industry needs curated board mapping"
        if not reason or not row.symbol:
            continue
        by_symbol[row.symbol] = {
            "symbol": row.symbol,
            "sector": row.sector,
            "industry": row.industry,
            "classification_source": source or "unmapped",
            "reason": reason,
        }
    return [by_symbol[symbol] for symbol in sorted(by_symbol)]


def write_unmapped_setup_classification_report(
    rows: Iterable[SetupRow],
    path: Path = MASTER_AVWAP_UNMAPPED_CLASSIFICATIONS_FILE,
) -> list[dict[str, str]]:
    """Atomically rewrite the unresolved-classification CSV for AI review."""
    unresolved = unmapped_setup_classifications(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ("symbol", "sector", "industry", "classification_source", "reason")
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(unresolved)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return unresolved
