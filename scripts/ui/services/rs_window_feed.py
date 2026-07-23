from __future__ import annotations

"""Context joins + sorting for the RS Window tab.

The bot answers "who led/lagged SPY over the selected M5 window"; this module
answers everything else about those names: is it a current bot pick (favorite
setup / tier / family), how strong is its industry and sector on the boards,
and how strong is the name itself on D1 and weekly timeframes. Pure Python
(no Qt) so every piece is unit-testable.
"""

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping

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
    ("Industry M5 vs SPY (advisory)", "industry_m5_vs_spy"),
    ("Stock vs primary industry M5 (advisory)", "stock_vs_industry_m5"),
    ("Industry strength (side-aligned)", "industry_rs"),
    ("Sector strength (side-aligned)", "sector_rs"),
    ("D1 strength 20d (side-aligned)", "d1_rs_20d"),
    ("Weekly 8EMA streak (side-aligned)", "weekly_streak"),
)
# Metrics measured as "strength": for SHORT rows the sort flips sign so a
# short in a WEAK industry/sector/tape ranks high (alignment with the trade).
_SIDE_ALIGNED_KEYS = {
    "industry_m5_vs_spy",
    "industry_rs",
    "sector_rs",
    "d1_rs_5d",
    "d1_rs_20d",
    "weekly_streak",
}

INDUSTRY_M5_MIN_MEMBERS = 3
INDUSTRY_M5_MIN_MEMBER_COVERAGE = 0.20
INDUSTRY_M5_MIN_TIMESTAMP_COVERAGE = 0.80


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


def _bar_field(bar: object, key: str):
    return bar.get(key) if isinstance(bar, Mapping) else getattr(bar, key, None)


def _window_bars(bars: list, start_dt: datetime, end_dt: datetime) -> list:
    selected = [
        bar
        for bar in bars or []
        if isinstance(_bar_field(bar, "dt"), datetime)
        and start_dt <= _bar_field(bar, "dt") <= end_dt
    ]
    return sorted(selected, key=lambda bar: _bar_field(bar, "dt"))


def _window_return(bars: list) -> float | None:
    if len(bars) < 2:
        return None
    first_open = _to_float(_bar_field(bars[0], "open"))
    last_close = _to_float(_bar_field(bars[-1], "close"))
    if not first_open or last_close is None:
        return None
    return (last_close - first_open) / first_open * 100.0


def compute_industry_m5_composites(
    industry_map: Mapping[str, dict],
    bars_by_symbol: Mapping[str, list],
    *,
    start_dt: datetime,
    end_dt: datetime,
    min_members: int = INDUSTRY_M5_MIN_MEMBERS,
    min_member_coverage: float = INDUSTRY_M5_MIN_MEMBER_COVERAGE,
    min_timestamp_coverage: float = INDUSTRY_M5_MIN_TIMESTAMP_COVERAGE,
) -> dict[str, dict]:
    """Primary-industry median returns on exact completed SPY endpoints.

    This is an advisory projection only. Coverage travels with every value so
    a thin watchlist sample can never masquerade as the full industry index.
    """

    normalized_bars = {
        str(symbol or "").strip().upper(): list(bars or [])
        for symbol, bars in bars_by_symbol.items()
    }
    spy_window = _window_bars(normalized_bars.get("SPY", []), start_dt, end_dt)
    if len(spy_window) < 2:
        return {}
    first_ts = _bar_field(spy_window[0], "dt")
    last_ts = _bar_field(spy_window[-1], "dt")
    spy_timestamps = {_bar_field(bar, "dt") for bar in spy_window}
    spy_pct = _window_return(spy_window)
    if spy_pct is None:
        return {}

    members_by_industry: dict[str, list[str]] = {}
    for context in industry_map.values():
        label = str(context.get("industry") or "").strip()
        if label and label not in members_by_industry:
            members_by_industry[label] = [
                str(symbol or "").strip().upper()
                for symbol in context.get("industry_member_symbols") or []
                if str(symbol or "").strip()
            ]

    composites: dict[str, dict] = {}
    for industry, expected_members in members_by_industry.items():
        member_returns: dict[str, float] = {}
        timestamp_coverages: list[float] = []
        for symbol in expected_members:
            member_window = _window_bars(normalized_bars.get(symbol, []), first_ts, last_ts)
            if len(member_window) < 2:
                continue
            member_ts = {_bar_field(bar, "dt") for bar in member_window}
            coverage = len(member_ts & spy_timestamps) / max(1, len(spy_timestamps))
            endpoints_ok = first_ts in member_ts and last_ts in member_ts
            value = _window_return(member_window)
            if not endpoints_ok or value is None or coverage < min_timestamp_coverage:
                continue
            member_returns[symbol] = value
            timestamp_coverages.append(coverage)
        used = len(member_returns)
        expected = len(expected_members)
        member_coverage = used / max(1, expected)
        timestamp_coverage = median(timestamp_coverages) if timestamp_coverages else 0.0
        industry_pct = median(member_returns.values()) if member_returns else None
        if (
            used >= max(1, int(min_members))
            and member_coverage >= float(min_member_coverage)
            and timestamp_coverage >= float(min_timestamp_coverage)
        ):
            status = "QUALIFIED_ADVISORY"
        elif used >= 2:
            status = "THIN_PREVIEW"
        else:
            status = "UNAVAILABLE"
            # One constituent is not an industry. Retain coverage counters but
            # suppress the apparent calculation so missing evidence cannot be
            # mistaken for a confident relative-strength value.
            industry_pct = None
        identity = json.dumps(
            {
                "industry": industry,
                "first": first_ts.isoformat(),
                "last": last_ts.isoformat(),
                "member_returns": sorted(member_returns.items()),
                "members_expected": expected,
                "spy_window_pct": spy_pct,
                "status": status,
            },
            sort_keys=True,
        )
        composites[industry] = {
            "industry": industry,
            "industry_m5_window_pct": industry_pct,
            "industry_m5_vs_spy": (industry_pct - spy_pct) if industry_pct is not None else None,
            "industry_m5_members_used": used,
            "industry_m5_members_expected": expected,
            "industry_m5_member_coverage": member_coverage,
            "industry_m5_timestamp_coverage": timestamp_coverage,
            "industry_m5_status": status,
            "industry_m5_first_ts": first_ts.isoformat(timespec="minutes"),
            "industry_m5_last_ts": last_ts.isoformat(timespec="minutes"),
            "industry_m5_snapshot_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
            "industry_m5_member_returns": member_returns,
            "spy_window_pct": spy_pct,
            "advisory_only": True,
        }
    return composites


def build_intraday_industry_rows(
    industry_map: Mapping[str, dict],
    bars_by_symbol: Mapping[str, list],
    *,
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict]:
    """Sortable GUI rows for every computable primary industry."""
    composites = compute_industry_m5_composites(
        industry_map,
        bars_by_symbol,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    rows = []
    for composite in composites.values():
        row = {
            key: value
            for key, value in composite.items()
            if key != "industry_m5_member_returns"
        }
        row["industry_m5_status_label"] = {
            "QUALIFIED_ADVISORY": "Qualified advisory",
            "THIN_PREVIEW": "Thin preview",
            "UNAVAILABLE": "Unavailable",
        }.get(str(row.get("industry_m5_status") or ""), "Unavailable")
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row.get("industry_m5_vs_spy") is None,
            -float(row.get("industry_m5_vs_spy") or 0.0),
            str(row.get("industry") or "").lower(),
        ),
    )


def add_intraday_industry_context(
    rows: list[dict],
    *,
    industry_map: Mapping[str, dict],
    bars_by_symbol: Mapping[str, list],
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict]:
    composites = compute_industry_m5_composites(
        industry_map,
        bars_by_symbol,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    decorated: list[dict] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        side_sign = -1.0 if str(row.get("side") or "").strip().upper() == "SHORT" else 1.0
        context = industry_map.get(symbol) or {}
        industry = str(context.get("industry") or row.get("industry") or "").strip()
        composite = composites.get(industry) or {}
        member_returns = composite.get("industry_m5_member_returns") or {}
        comparison_includes_symbol = False
        comparison_values = []
        if composite.get("industry_m5_status") != "UNAVAILABLE":
            comparison_values = [
                value
                for member, value in member_returns.items()
                if str(member).upper() != symbol
            ]
            if len(comparison_values) < 2:
                comparison_values = list(member_returns.values())
                comparison_includes_symbol = symbol in member_returns
        comparison_pct = median(comparison_values) if comparison_values else None
        # Use the exactly aligned member calculation, not the row's broader
        # cached-window return, for stock-within-industry comparison.
        stock_pct = _to_float(member_returns.get(symbol))
        stock_vs_industry = (
            side_sign * (stock_pct - comparison_pct)
            if stock_pct is not None and comparison_pct is not None
            else None
        )
        clean_composite = {
            key: value
            for key, value in composite.items()
            if key != "industry_m5_member_returns"
        }
        decorated.append(
            {
                **row,
                **clean_composite,
                "industry_m5_comparison_pct": comparison_pct,
                "industry_m5_stock_window_pct": stock_pct,
                "stock_vs_industry_m5": stock_vs_industry,
                "industry_comparison_includes_symbol": comparison_includes_symbol,
                "industry_m5_status_label": {
                    "QUALIFIED_ADVISORY": "Qualified advisory",
                    "THIN_PREVIEW": "Thin preview",
                    "UNAVAILABLE": "Unavailable",
                }.get(str(composite.get("industry_m5_status") or ""), "Unavailable"),
            }
        )
    return decorated


_INTRADAY_INDUSTRY_FIELDS = (
    "industry",
    "industry_primary_source",
    "industry_m5_window_pct",
    "industry_m5_vs_spy",
    "industry_m5_members_used",
    "industry_m5_members_expected",
    "industry_m5_member_coverage",
    "industry_m5_timestamp_coverage",
    "industry_m5_status",
    "industry_m5_first_ts",
    "industry_m5_last_ts",
    "industry_m5_snapshot_id",
    "spy_window_pct",
    "advisory_only",
)


def build_industry_intraday_snapshot(
    rows: list[dict],
    *,
    start_dt: datetime,
    end_dt: datetime,
    board_snapshot_id: str = "",
    industry_rows: list[dict] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create the immutable advisory artifact consumed by research and AI."""
    industries_by_id: dict[str, dict] = {}
    candidates: list[dict] = []
    for row in [*(industry_rows or []), *rows]:
        industry_snapshot_id = str(row.get("industry_m5_snapshot_id") or "")
        if industry_snapshot_id:
            industries_by_id.setdefault(
                industry_snapshot_id,
                {key: row.get(key) for key in _INTRADAY_INDUSTRY_FIELDS},
            )
    for row in rows:
        industry_snapshot_id = str(row.get("industry_m5_snapshot_id") or "")
        stock_window_pct = _to_float(row.get("window_pct"))
        spy_window_pct = _to_float(row.get("spy_pct"))
        raw_stock_vs_spy = (
            stock_window_pct - spy_window_pct
            if stock_window_pct is not None and spy_window_pct is not None
            else None
        )
        candidates.append(
            {
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "side": str(row.get("side") or "").strip().upper(),
                "industry": str(row.get("industry") or ""),
                "additional_industries": list(row.get("additional_industries") or []),
                "stock_window_pct": stock_window_pct,
                "raw_stock_vs_spy": raw_stock_vs_spy,
                "side_aligned_stock_vs_spy": _to_float(row.get("excess")),
                "aligned_stock_window_pct": _to_float(row.get("industry_m5_stock_window_pct")),
                "side_aligned_stock_vs_primary_industry": _to_float(
                    row.get("stock_vs_industry_m5")
                ),
                "industry_snapshot_id": industry_snapshot_id,
                "industry_status": str(row.get("industry_m5_status") or "UNAVAILABLE"),
            }
        )
    industries = sorted(
        industries_by_id.values(),
        key=lambda value: (str(value.get("industry") or "").lower(), str(value.get("industry_m5_snapshot_id") or "")),
    )
    candidates.sort(key=lambda value: (value["symbol"], value["side"], value["industry"]))
    generated = now or datetime.now().astimezone()
    core = {
        "board_snapshot_id": str(board_snapshot_id or ""),
        "requested_start": start_dt.isoformat(timespec="minutes"),
        "requested_end": end_dt.isoformat(timespec="minutes"),
        "industries": industries,
        "candidates": candidates,
    }
    identity = json.dumps(core, sort_keys=True, separators=(",", ":"), default=str)
    qualified = sum(
        str(value.get("industry_m5_status") or "") == "QUALIFIED_ADVISORY"
        for value in industries
    )
    thin = sum(
        str(value.get("industry_m5_status") or "") == "THIN_PREVIEW"
        for value in industries
    )
    return {
        "schema": "industry_intraday_rs_snapshot_v1",
        "snapshot_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
        "generated_at": generated.isoformat(timespec="seconds"),
        "advisory_only": True,
        "production_score_effect": "none",
        "calculation_contract": {
            "bar_size": "M5",
            "completed_bars_only": True,
            "timestamp_alignment": "exact_spy_endpoints",
            "industry_aggregation": "median_member_return",
            "primary_industry_only": True,
            "stock_comparison": "leave_one_out_when_at_least_two_peers",
        },
        "source_board_snapshot_id": str(board_snapshot_id or ""),
        "requested_window": {
            "start": start_dt.isoformat(timespec="minutes"),
            "end": end_dt.isoformat(timespec="minutes"),
        },
        "industry_count": len(industries),
        "qualified_industry_count": qualified,
        "thin_preview_industry_count": thin,
        "candidate_count": len(candidates),
        "industries": industries,
        "candidates": candidates,
    }


def _read_board_snapshot_id(path: Path) -> str:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("snapshot_id") or "") if isinstance(payload, dict) else ""


def _snapshot_has_signal(payload: Mapping[str, Any] | None) -> bool:
    """True when a snapshot carries any qualified/thin/candidate content."""
    if not isinstance(payload, Mapping):
        return False
    return bool(
        int(payload.get("qualified_industry_count") or 0)
        or int(payload.get("thin_preview_industry_count") or 0)
        or int(payload.get("candidate_count") or 0)
    )


def save_industry_intraday_snapshot(
    rows: list[dict],
    *,
    start_dt: datetime,
    end_dt: datetime,
    output_path: Path | None = None,
    board_state_path: Path | None = None,
    industry_rows: list[dict] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Atomically persist the GUI's last completed-M5 advisory calculation."""
    from project_paths import INDUSTRY_BOARD_STATE_FILE, INDUSTRY_INTRADAY_RS_STATE_FILE

    target = Path(output_path or INDUSTRY_INTRADAY_RS_STATE_FILE)
    board_path = Path(board_state_path or INDUSTRY_BOARD_STATE_FILE)
    payload = build_industry_intraday_snapshot(
        rows,
        start_dt=start_dt,
        end_dt=end_dt,
        board_snapshot_id=_read_board_snapshot_id(board_path),
        industry_rows=industry_rows,
        now=now,
    )
    if not _snapshot_has_signal(payload):
        # 2026-07-17: an after-close regeneration with an empty bar cache
        # overwrote the session's last useful advisory (5/67 qualified) with
        # an all-UNAVAILABLE husk. A no-signal recalculation never replaces
        # a stored snapshot that still has content - the trader's evening
        # review keeps the last real read of the day.
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if _snapshot_has_signal(existing):
            return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return payload


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
    intraday_bars_by_symbol: Mapping[str, list] | None = None,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
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
        merged["additional_industries_display"] = ", ".join(
            str(value) for value in merged.get("additional_industries") or []
        )
        decorated.append(merged)
    if intraday_bars_by_symbol is not None and start_dt is not None and end_dt is not None:
        return add_intraday_industry_context(
            decorated,
            industry_map=industry_map,
            bars_by_symbol=intraday_bars_by_symbol,
            start_dt=start_dt,
            end_dt=end_dt,
        )
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
