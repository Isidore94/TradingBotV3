#!/usr/bin/env python3
"""TC2000-style sector + industry index board.

Two boards, both ranked so the hottest groups float to the top:

- **Sector board**: the SPDR sector ETFs (XLK, XLF, ...) measured against SPY —
  today's % change plus a blended relative-strength score over ~1w/1m/3m, with an
  RS rank exactly like the per-stock RRS ranking.
- **Industry board**: composite "industry indexes" in the TC2000 style. TC2000
  builds its index board from the Morningstar hierarchy (11 sectors -> ~55
  industry groups -> 145 industries); the shared `symbol_classification.csv`
  cache (the one BounceBot maintains) already carries the 145-industry level, so
  the board aggregates those into curated industry-group indexes defined in
  `data/industry_index_definitions.json` (seeded from
  ``DEFAULT_INDUSTRY_INDEX_DEFINITIONS``, user-editable). Definitions may also
  pin explicit tickers, which lets theme indexes (AI Hardware, Uranium &
  Nuclear, Crypto, Space, ...) cut across the official taxonomy — a symbol can
  and should appear in several indexes. Cache industries not claimed by any
  definition still show up as their own raw rows, and user-defined
  micro-industry groups in `data/custom_industry_groups.json` (e.g. Photonics =
  AAOI/LITE/CIEN/AXTI) are appended with a trailing ``*``.
  Each index row reports median member % change, blended RS vs SPY, volume
  buzz (today vs 20-day average volume), member count and the top movers.

Data comes from yfinance in one batched download, so this runs without an IBKR
session and without consuming IB API pacing budget. Outputs land in the home
folder's `output/` directory as a phone-friendly text board plus CSVs.

Run:
    .venv/Scripts/python.exe scripts/industry_scanner.py            # full boards
    .venv/Scripts/python.exe scripts/industry_scanner.py --sectors-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from statistics import median

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_paths import (  # noqa: E402
    DATA_DIR,
    OUTPUT_DIR,
    SYMBOL_CLASSIFICATION_CACHE_FILE,
    LONGS_FILE,
    SHORTS_FILE,
    SWING_LONGS_FILE,
    SWING_SHORTS_FILE,
)
from watchlist_utils import read_watchlist_symbols  # noqa: E402

BENCHMARK_SYMBOL = "SPY"
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Blended relative-strength windows (sessions) and weights: short-term leadership
# is what the TC2000 board surfaces, so ~1 week dominates, ~1 month and ~3 months
# keep persistent leaders from being displaced by one-day noise.
RS_WINDOWS = ((5, 0.5), (20, 0.3), (65, 0.2))
VOLUME_BUZZ_AVG_SESSIONS = 20
INDUSTRY_MIN_MEMBERS = 2
FETCH_PERIOD = "9mo"

CUSTOM_INDUSTRY_GROUPS_FILE = DATA_DIR / "custom_industry_groups.json"
INDUSTRY_INDEX_DEFINITIONS_FILE = DATA_DIR / "industry_index_definitions.json"
INDUSTRY_BOARD_TEXT_FILE = OUTPUT_DIR / "industry_indexes.txt"
INDUSTRY_BOARD_CSV_FILE = OUTPUT_DIR / "industry_indexes.csv"
SECTOR_BOARD_CSV_FILE = OUTPUT_DIR / "sector_indexes.csv"

DEFAULT_CUSTOM_INDUSTRY_GROUPS = {
    "Photonics": ["AAOI", "LITE", "CIEN", "AXTI", "COHR", "FN"],
}

# TC2000-style industry-group indexes. Each entry aggregates Morningstar
# industries (the `industry` column of the classification cache, matched
# case-insensitively) and/or pins explicit tickers. Tickers let theme indexes
# cut across the official taxonomy, so overlap between indexes is expected —
# NVDA belongs in Semiconductors *and* AI Hardware & Data Center.
DEFAULT_INDUSTRY_INDEX_DEFINITIONS: dict[str, dict] = {
    # --- Technology ---
    "Semiconductors": {"industries": ["Semiconductors"]},
    "Semiconductor Equipment": {"industries": ["Semiconductor Equipment & Materials"]},
    "Software - Application": {"industries": ["Software - Application"]},
    "Software - Infrastructure": {"industries": ["Software - Infrastructure"]},
    "Networking & Comm Equipment": {"industries": ["Communication Equipment"]},
    "Computer Hardware & Electronics": {
        "industries": [
            "Computer Hardware",
            "Consumer Electronics",
            "Electronics & Computer Distribution",
            "Electronic Components",
            "Scientific & Technical Instruments",
        ]
    },
    "IT Services & Consulting": {"industries": ["Information Technology Services"]},
    "Solar": {"industries": ["Solar"]},
    # --- Tech / market themes (ticker-pinned, overlap by design) ---
    "AI Hardware & Data Center": {
        "tickers": [
            "NVDA", "AMD", "AVGO", "MRVL", "MU", "SMCI", "DELL", "HPE",
            "ANET", "VRT", "CRDO", "ALAB", "TSM", "COHR", "MOD",
        ]
    },
    "Cybersecurity": {
        "tickers": [
            "CRWD", "PANW", "ZS", "FTNT", "NET", "OKTA", "S", "CYBR",
            "TENB", "QLYS", "RPD", "GEN",
        ]
    },
    "Quantum Computing": {"tickers": ["IONQ", "RGTI", "QBTS", "QUBT", "ARQQ"]},
    "Crypto & Bitcoin Miners": {
        "tickers": [
            "COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "WULF", "CIFR",
            "IREN", "CORZ", "BTBT", "HOOD", "GLXY",
        ]
    },
    "Payments & Fintech": {
        "industries": ["Credit Services"],
        "tickers": ["PYPL", "XYZ", "AFRM", "TOST", "SOFI", "FI", "FIS", "GPN", "MQ", "UPST"],
    },
    # --- Healthcare ---
    "Biotechnology": {"industries": ["Biotechnology"]},
    "Pharmaceuticals": {
        "industries": ["Drug Manufacturers - General", "Drug Manufacturers - Specialty & Generic"]
    },
    "Medical Devices & Supplies": {
        "industries": ["Medical Devices", "Medical Instruments & Supplies"]
    },
    "Diagnostics & Life Science Tools": {"industries": ["Diagnostics & Research"]},
    "Healthcare Providers & Services": {
        "industries": [
            "Healthcare Plans",
            "Medical Care Facilities",
            "Medical Distribution",
            "Health Information Services",
        ]
    },
    "GLP-1 & Obesity": {"tickers": ["LLY", "NVO", "AMGN", "VKTX", "GPCR", "ALT"]},
    # --- Financials ---
    "Banks - Major": {"industries": ["Banks - Diversified"]},
    "Banks - Regional": {"industries": ["Banks - Regional"]},
    "Capital Markets & Exchanges": {
        "industries": ["Capital Markets", "Financial Data & Stock Exchanges"]
    },
    "Asset Management": {"industries": ["Asset Management"]},
    "Insurance": {
        "industries": [
            "Insurance - Diversified",
            "Insurance - Life",
            "Insurance - Property & Casualty",
            "Insurance - Reinsurance",
            "Insurance - Specialty",
            "Insurance Brokers",
        ]
    },
    # --- Energy ---
    "Oil & Gas E&P": {"industries": ["Oil & Gas E&P"]},
    "Oil Services & Drilling": {
        "industries": ["Oil & Gas Equipment & Services", "Oil & Gas Drilling"]
    },
    "Integrated Oil & Refiners": {
        "industries": ["Oil & Gas Integrated", "Oil & Gas Refining & Marketing"]
    },
    "Oil & Gas Midstream": {"industries": ["Oil & Gas Midstream"]},
    "Uranium & Nuclear": {
        "industries": ["Uranium"],
        "tickers": ["CCJ", "UEC", "DNN", "NXE", "UUUU", "LEU", "SMR", "OKLO", "NNE", "BWXT", "CEG"],
    },
    "Coal": {"industries": ["Thermal Coal", "Coking Coal"]},
    # --- Basic Materials ---
    "Gold & Silver Miners": {
        "industries": ["Gold", "Silver", "Other Precious Metals & Mining"]
    },
    "Copper & Base Metals": {
        "industries": ["Copper", "Aluminum", "Other Industrial Metals & Mining"]
    },
    "Steel": {"industries": ["Steel"]},
    "Chemicals": {"industries": ["Chemicals", "Specialty Chemicals"]},
    "Agriculture & Farm Products": {
        "industries": ["Agricultural Inputs", "Farm Products"],
        "tickers": ["DE", "AGCO", "CNH"],
    },
    # --- Industrials ---
    "Aerospace & Defense": {"industries": ["Aerospace & Defense"]},
    "Space": {"tickers": ["RKLB", "ASTS", "LUNR", "RDW", "PL", "BKSY"]},
    "Airlines": {"industries": ["Airlines", "Airports & Air Services"]},
    "Rails & Trucking": {
        "industries": ["Railroads", "Trucking", "Integrated Freight & Logistics"]
    },
    "Marine Shipping": {"industries": ["Marine Shipping"]},
    "Machinery": {
        "industries": [
            "Specialty Industrial Machinery",
            "Farm & Heavy Construction Machinery",
            "Metal Fabrication",
            "Tools & Accessories",
        ]
    },
    "Electrical Equipment & Grid": {
        "industries": ["Electrical Equipment & Parts"],
        "tickers": ["ETN", "VRT", "PWR", "HUBB", "GEV", "NVT", "AZZ"],
    },
    "Engineering & Construction": {"industries": ["Engineering & Construction"]},
    "Building Products & Materials": {
        "industries": ["Building Products & Equipment", "Building Materials"]
    },
    "Waste & Environmental": {
        "industries": ["Waste Management", "Pollution & Treatment Controls"]
    },
    "Business Services": {
        "industries": [
            "Specialty Business Services",
            "Consulting Services",
            "Staffing & Employment Services",
            "Security & Protection Services",
        ]
    },
    "Distribution & Rentals": {
        "industries": ["Industrial Distribution", "Rental & Leasing Services"]
    },
    # --- Consumer Cyclical ---
    "Homebuilders & Home Improvement": {
        "industries": [
            "Residential Construction",
            "Home Improvement Retail",
            "Furnishings, Fixtures & Appliances",
        ]
    },
    "Autos & EV": {"industries": ["Auto Manufacturers"]},
    "Auto Parts & Dealers": {
        "industries": ["Auto Parts", "Auto & Truck Dealerships", "Recreational Vehicles"]
    },
    "Retail - Apparel & Specialty": {
        "industries": [
            "Apparel Retail",
            "Apparel Manufacturing",
            "Footwear & Accessories",
            "Luxury Goods",
            "Department Stores",
            "Specialty Retail",
        ]
    },
    "Internet Retail": {"industries": ["Internet Retail"]},
    "Restaurants": {"industries": ["Restaurants"]},
    "Travel & Leisure": {
        "industries": ["Travel Services", "Lodging", "Resorts & Casinos", "Gambling", "Leisure"]
    },
    "Packaging & Containers": {"industries": ["Packaging & Containers"]},
    # --- Consumer Defensive ---
    "Food & Beverage": {
        "industries": [
            "Packaged Foods",
            "Beverages - Brewers",
            "Beverages - Non - Alcoholic",
            "Confectioners",
            "Food Distribution",
        ]
    },
    "Staples Retail & Discount": {"industries": ["Discount Stores", "Grocery Stores"]},
    "Household & Personal Products": {
        "industries": ["Household & Personal Products", "Tobacco"]
    },
    # --- Communication Services ---
    "Internet Content & Social": {"industries": ["Internet Content & Information"]},
    "Media & Entertainment": {
        "industries": [
            "Entertainment",
            "Broadcasting",
            "Publishing",
            "Advertising Agencies",
            "Electronic Gaming & Multimedia",
        ]
    },
    "Telecom": {"industries": ["Telecom Services"]},
    # --- Real Estate ---
    "REITs - Equity": {
        "industries": [
            "REIT - Diversified",
            "REIT - Healthcare Facilities",
            "REIT - Hotel & Motel",
            "REIT - Industrial",
            "REIT - Office",
            "REIT - Residential",
            "REIT - Retail",
            "REIT - Specialty",
        ]
    },
    "REITs - Mortgage": {"industries": ["REIT - Mortgage", "Mortgage Finance"]},
    "Real Estate Services": {"industries": ["Real Estate Services"]},
    # --- Utilities ---
    "Utilities - Regulated": {
        "industries": [
            "Utilities - Regulated Electric",
            "Utilities - Regulated Gas",
            "Utilities - Regulated Water",
            "Utilities - Diversified",
        ]
    },
    "Utilities - IPP & Renewables": {
        "industries": ["Utilities - Independent Power Producers", "Utilities - Renewable"]
    },
}


# ---------------------------------------------------------------------------
# Inputs: classification cache, custom groups, watchlists
# ---------------------------------------------------------------------------
def load_symbol_classifications(path: Path | None = None) -> dict[str, dict]:
    """Read the shared symbol->sector/industry CSV cache (BounceBot maintains it)."""
    cache_path = Path(path) if path else SYMBOL_CLASSIFICATION_CACHE_FILE
    cache: dict[str, dict] = {}
    if not cache_path.exists():
        return cache
    try:
        with open(cache_path, "r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                symbol = str(row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                cache[symbol] = {
                    "symbol": symbol,
                    "sector": str(row.get("sector") or "").strip(),
                    "industry": str(row.get("industry") or "").strip(),
                }
    except Exception as exc:
        logging.warning("Failed loading symbol classification cache: %s", exc)
    return cache


def load_custom_industry_groups(path: Path | None = None) -> dict[str, list[str]]:
    """User-editable micro-industry groups; seeds a starter file on first run."""
    groups_path = Path(path) if path else CUSTOM_INDUSTRY_GROUPS_FILE
    if not groups_path.exists():
        try:
            groups_path.parent.mkdir(parents=True, exist_ok=True)
            groups_path.write_text(
                json.dumps(DEFAULT_CUSTOM_INDUSTRY_GROUPS, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logging.warning("Could not seed custom industry groups file: %s", exc)
            return {name: list(members) for name, members in DEFAULT_CUSTOM_INDUSTRY_GROUPS.items()}
    try:
        payload = json.loads(groups_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Could not parse %s: %s", groups_path, exc)
        return {}
    groups: dict[str, list[str]] = {}
    if isinstance(payload, dict):
        for name, members in payload.items():
            label = str(name or "").strip()
            if not label or not isinstance(members, list):
                continue
            symbols = sorted({str(item or "").strip().upper() for item in members if str(item or "").strip()})
            if symbols:
                groups[label] = symbols
    return groups


def load_industry_index_definitions(path: Path | None = None) -> dict[str, dict]:
    """TC2000-style index definitions; seeds the default file on first run.

    Each definition is ``{"industries": [...], "tickers": [...]}`` (either key
    optional). A plain list value is shorthand for ``{"tickers": [...]}``.
    """
    defs_path = Path(path) if path else INDUSTRY_INDEX_DEFINITIONS_FILE
    if not defs_path.exists():
        try:
            defs_path.parent.mkdir(parents=True, exist_ok=True)
            defs_path.write_text(
                json.dumps(DEFAULT_INDUSTRY_INDEX_DEFINITIONS, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logging.warning("Could not seed industry index definitions file: %s", exc)
            return {name: dict(spec) for name, spec in DEFAULT_INDUSTRY_INDEX_DEFINITIONS.items()}
    try:
        payload = json.loads(defs_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Could not parse %s: %s", defs_path, exc)
        return {}
    definitions: dict[str, dict] = {}
    if isinstance(payload, dict):
        for name, spec in payload.items():
            label = str(name or "").strip()
            if not label:
                continue
            if isinstance(spec, list):
                spec = {"tickers": spec}
            if not isinstance(spec, dict):
                continue
            industries = [
                str(item or "").strip()
                for item in (spec.get("industries") or [])
                if str(item or "").strip()
            ]
            tickers = sorted(
                {
                    str(item or "").strip().upper()
                    for item in (spec.get("tickers") or [])
                    if str(item or "").strip()
                }
            )
            if industries or tickers:
                definitions[label] = {"industries": industries, "tickers": tickers}
    return definitions


def _norm_industry(name: str) -> str:
    return " ".join(str(name or "").split()).lower()


def collect_industry_members(
    classifications: dict[str, dict],
    custom_groups: dict[str, list[str]] | None = None,
    *,
    index_definitions: dict[str, dict] | None = None,
    restrict_to: set[str] | None = None,
) -> dict[str, list[str]]:
    """Index label -> member symbols.

    Cache industries feed the curated index definitions (matched by industry
    name, case-insensitively); a symbol may land in several indexes because
    definitions overlap on purpose. Industries not claimed by any definition
    fall through as their own raw rows so nothing goes invisible. Custom groups
    are first-class indexes (marked with a trailing ``*`` so the board shows
    which rows are user-defined); like definition tickers, they are never
    restricted: if you bothered to define Photonics, you want the whole group
    measured.
    """
    by_industry: dict[str, set[str]] = {}
    display_name: dict[str, str] = {}
    for symbol, row in (classifications or {}).items():
        industry = str(row.get("industry") or "").strip()
        if not industry:
            continue
        if restrict_to is not None and symbol not in restrict_to:
            continue
        key = _norm_industry(industry)
        by_industry.setdefault(key, set()).add(symbol)
        display_name.setdefault(key, industry)

    members: dict[str, set[str]] = {}
    claimed: set[str] = set()
    for name, spec in (index_definitions or {}).items():
        group: set[str] = set()
        for industry in spec.get("industries") or []:
            key = _norm_industry(industry)
            claimed.add(key)
            group.update(by_industry.get(key, set()))
        group.update(spec.get("tickers") or [])
        if group:
            members.setdefault(name, set()).update(group)
    for key, group in by_industry.items():
        if key not in claimed:
            members.setdefault(display_name[key], set()).update(group)
    for name, symbols in (custom_groups or {}).items():
        members.setdefault(f"{name}*", set()).update(symbols)
    return {label: sorted(group) for label, group in members.items() if group}


def gather_watchlist_symbols() -> set[str]:
    symbols: set[str] = set()
    for path in (LONGS_FILE, SHORTS_FILE, SWING_LONGS_FILE, SWING_SHORTS_FILE):
        try:
            symbols.update(read_watchlist_symbols(path))
        except Exception:
            continue
    return {s.strip().upper() for s in symbols if s.strip()}


# ---------------------------------------------------------------------------
# Metrics (pure; frames in, numbers out)
# ---------------------------------------------------------------------------
def compute_symbol_metrics(frame: pd.DataFrame | None) -> dict | None:
    """Per-symbol board metrics from a daily OHLCV frame (chronological)."""
    if frame is None or getattr(frame, "empty", True):
        return None
    work = frame.dropna(subset=["close"]).reset_index(drop=True)
    if len(work) < 2:
        return None
    closes = pd.to_numeric(work["close"], errors="coerce")
    volumes = pd.to_numeric(work.get("volume"), errors="coerce") if "volume" in work.columns else None
    last_close = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2])
    metrics = {
        "last_close": last_close,
        "pct_change_1d": ((last_close / prev_close) - 1.0) * 100 if prev_close else None,
        "volume_buzz_pct": None,
    }
    for window, _weight in RS_WINDOWS:
        key = f"return_{window}d_pct"
        if len(closes) > window and float(closes.iloc[-1 - window]) > 0:
            metrics[key] = ((last_close / float(closes.iloc[-1 - window])) - 1.0) * 100
        else:
            metrics[key] = None
    if volumes is not None and len(volumes.dropna()) > VOLUME_BUZZ_AVG_SESSIONS:
        recent = volumes.dropna()
        avg_volume = float(recent.iloc[-1 - VOLUME_BUZZ_AVG_SESSIONS : -1].mean())
        today_volume = float(recent.iloc[-1])
        if avg_volume > 0:
            metrics["volume_buzz_pct"] = ((today_volume / avg_volume) - 1.0) * 100
    return metrics


def compute_rs_score(metrics: dict | None, benchmark_metrics: dict | None) -> float | None:
    """Blended excess return vs the benchmark across the RS windows."""
    if not metrics or not benchmark_metrics:
        return None
    score = 0.0
    weight_used = 0.0
    for window, weight in RS_WINDOWS:
        key = f"return_{window}d_pct"
        own = metrics.get(key)
        bench = benchmark_metrics.get(key)
        if own is None or bench is None:
            continue
        score += weight * (float(own) - float(bench))
        weight_used += weight
    if weight_used <= 0:
        return None
    return score / weight_used


def build_sector_board(
    frames_by_symbol: dict[str, pd.DataFrame],
    *,
    sector_etfs: dict[str, str] | None = None,
    benchmark_symbol: str = BENCHMARK_SYMBOL,
) -> list[dict]:
    etfs = sector_etfs if isinstance(sector_etfs, dict) else SECTOR_ETFS
    benchmark_metrics = compute_symbol_metrics(frames_by_symbol.get(benchmark_symbol))
    rows = []
    for etf, sector_name in etfs.items():
        metrics = compute_symbol_metrics(frames_by_symbol.get(etf))
        if not metrics:
            continue
        rows.append(
            {
                "etf": etf,
                "sector": sector_name,
                "pct_change_1d": metrics.get("pct_change_1d"),
                "return_5d_pct": metrics.get("return_5d_pct"),
                "return_20d_pct": metrics.get("return_20d_pct"),
                "return_65d_pct": metrics.get("return_65d_pct"),
                "volume_buzz_pct": metrics.get("volume_buzz_pct"),
                "rs_score": compute_rs_score(metrics, benchmark_metrics),
            }
        )
    rows.sort(key=lambda row: (row["rs_score"] is None, -(row["rs_score"] or 0.0)))
    for rank, row in enumerate(rows, start=1):
        row["rs_rank"] = rank
    return rows


def _median_or_none(values: list) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return median(clean) if clean else None


def build_industry_board(
    frames_by_symbol: dict[str, pd.DataFrame],
    industry_members: dict[str, list[str]],
    *,
    benchmark_symbol: str = BENCHMARK_SYMBOL,
    min_members: int = INDUSTRY_MIN_MEMBERS,
) -> list[dict]:
    """Composite industry rows: median member metrics + RS rank + top movers."""
    benchmark_metrics = compute_symbol_metrics(frames_by_symbol.get(benchmark_symbol))
    metrics_by_symbol: dict[str, dict] = {}
    for symbol in {s for members in industry_members.values() for s in members}:
        metrics = compute_symbol_metrics(frames_by_symbol.get(symbol))
        if metrics:
            metrics_by_symbol[symbol] = metrics

    rows = []
    for industry, members in industry_members.items():
        member_metrics = [(s, metrics_by_symbol[s]) for s in members if s in metrics_by_symbol]
        if len(member_metrics) < max(1, int(min_members)):
            continue
        composite = {
            f"return_{window}d_pct": _median_or_none(
                [m.get(f"return_{window}d_pct") for _, m in member_metrics]
            )
            for window, _w in RS_WINDOWS
        }
        movers = sorted(
            member_metrics,
            key=lambda item: (item[1].get("pct_change_1d") is None, -(item[1].get("pct_change_1d") or 0.0)),
        )
        rows.append(
            {
                "industry": industry,
                "member_count": len(member_metrics),
                "pct_change_1d": _median_or_none([m.get("pct_change_1d") for _, m in member_metrics]),
                "volume_buzz_pct": _median_or_none([m.get("volume_buzz_pct") for _, m in member_metrics]),
                "return_5d_pct": composite.get("return_5d_pct"),
                "return_20d_pct": composite.get("return_20d_pct"),
                "return_65d_pct": composite.get("return_65d_pct"),
                "rs_score": compute_rs_score(composite, benchmark_metrics),
                "top_movers": ", ".join(
                    f"{symbol} {metrics.get('pct_change_1d'):+.1f}%"
                    for symbol, metrics in movers[:3]
                    if metrics.get("pct_change_1d") is not None
                ),
            }
        )
    rows.sort(key=lambda row: (row["rs_score"] is None, -(row["rs_score"] or 0.0)))
    for rank, row in enumerate(rows, start=1):
        row["rs_rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# Fetching (yfinance batch; the only networked part)
# ---------------------------------------------------------------------------
def fetch_daily_frames_yf(symbols: list[str], *, period: str = FETCH_PERIOD) -> dict[str, pd.DataFrame]:
    """One batched yfinance download -> {symbol: OHLCV frame}. Missing symbols skipped."""
    import yfinance as yf

    tickers = sorted({str(s or "").strip().upper() for s in symbols if str(s or "").strip()})
    if not tickers:
        return {}
    raw = yf.download(
        tickers=" ".join(tickers),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    frames: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return frames
    for symbol in tickers:
        try:
            sub = raw[symbol] if isinstance(raw.columns, pd.MultiIndex) else raw
        except (KeyError, TypeError):
            continue
        if sub is None or sub.empty:
            continue
        frame = sub.reset_index().rename(
            columns={
                "Date": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frame = frame.dropna(subset=["close"])
        if not frame.empty:
            frames[symbol] = frame[["datetime", "open", "high", "low", "close", "volume"]]
    return frames


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def _fmt_pct(value) -> str:
    return f"{value:+.1f}%" if value is not None else "  n/a"


def render_board_text(sector_rows: list[dict], industry_rows: list[dict]) -> str:
    lines = [
        f"SECTOR / INDUSTRY INDEX BOARD  (generated {datetime.now().isoformat(timespec='minutes')})",
        "RS score = blended excess return vs SPY over ~1w/1m/3m; rank 1 = strongest.",
        "",
        "== SECTORS (SPDR ETFs vs SPY) ==",
        f"{'rank':<5}{'ETF':<6}{'sector':<26}{'today':>8}{'RS':>8}{'5d':>8}{'20d':>8}{'65d':>8}{'volbuzz':>9}",
    ]
    for row in sector_rows:
        lines.append(
            f"{row.get('rs_rank', ''):<5}{row['etf']:<6}{row['sector']:<26}"
            f"{_fmt_pct(row.get('pct_change_1d')):>8}"
            f"{(f'{row['rs_score']:+.2f}' if row.get('rs_score') is not None else 'n/a'):>8}"
            f"{_fmt_pct(row.get('return_5d_pct')):>8}"
            f"{_fmt_pct(row.get('return_20d_pct')):>8}"
            f"{_fmt_pct(row.get('return_65d_pct')):>8}"
            f"{_fmt_pct(row.get('volume_buzz_pct')):>9}"
        )
    lines += [
        "",
        "== INDUSTRY INDEXES (composite of classified members; * = custom group) ==",
        f"{'rank':<5}{'industry':<38}{'n':>4}{'today':>8}{'RS':>8}{'volbuzz':>9}  top movers",
    ]
    for row in industry_rows:
        lines.append(
            f"{row.get('rs_rank', ''):<5}{row['industry'][:37]:<38}{row['member_count']:>4}"
            f"{_fmt_pct(row.get('pct_change_1d')):>8}"
            f"{(f'{row['rs_score']:+.2f}' if row.get('rs_score') is not None else 'n/a'):>8}"
            f"{_fmt_pct(row.get('volume_buzz_pct')):>9}  {row.get('top_movers', '')}"
        )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to replace {path.name} with an empty board")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def run_industry_scan(
    *,
    extra_symbols: list[str] | None = None,
    sectors_only: bool = False,
    write_outputs: bool = True,
) -> dict:
    """Build both boards and (optionally) write the text/CSV outputs."""
    classifications = load_symbol_classifications()
    custom_groups = load_custom_industry_groups()
    index_definitions = load_industry_index_definitions()
    watchlist_symbols = gather_watchlist_symbols()

    universe: set[str] = set(watchlist_symbols)
    universe.update(classifications.keys())
    universe.update(s for members in custom_groups.values() for s in members)
    universe.update(s for spec in index_definitions.values() for s in spec.get("tickers") or [])
    universe.update(extra_symbols or [])

    fetch_symbols = set(SECTOR_ETFS) | {BENCHMARK_SYMBOL}
    if not sectors_only:
        fetch_symbols |= universe
    frames = fetch_daily_frames_yf(sorted(fetch_symbols))

    sector_rows = build_sector_board(frames)
    industry_rows = []
    if not sectors_only:
        industry_members = collect_industry_members(
            classifications, custom_groups, index_definitions=index_definitions
        )
        industry_rows = build_industry_board(frames, industry_members)

    if write_outputs:
        if not sector_rows or (not sectors_only and not industry_rows):
            raise RuntimeError(
                "Industry provider returned incomplete boards; previous verified files were preserved."
            )
        _write_text(INDUSTRY_BOARD_TEXT_FILE, render_board_text(sector_rows, industry_rows))
        _write_csv(SECTOR_BOARD_CSV_FILE, sector_rows)
        if not sectors_only:
            _write_csv(INDUSTRY_BOARD_CSV_FILE, industry_rows)
        logging.info(
            "Industry scanner wrote %s sector row(s), %s industry row(s) to %s",
            len(sector_rows),
            len(industry_rows),
            INDUSTRY_BOARD_TEXT_FILE,
        )
    return {"sector_rows": sector_rows, "industry_rows": industry_rows, "symbol_count": len(frames)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sector + industry RS index board")
    parser.add_argument("--sectors-only", action="store_true", help="skip the industry composite scan")
    parser.add_argument("--symbols", default="", help="comma-separated extra symbols to include")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    extra = [s for s in (args.symbols or "").replace(",", " ").split() if s]
    result = run_industry_scan(extra_symbols=extra, sectors_only=bool(args.sectors_only))
    print(render_board_text(result["sector_rows"], result["industry_rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
