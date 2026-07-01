#!/usr/bin/env python3
"""TC2000-style sector + industry index board.

Two boards, both ranked so the hottest groups float to the top:

- **Sector board**: the SPDR sector ETFs (XLK, XLF, ...) measured against SPY —
  today's % change plus a blended relative-strength score over ~1w/1m/3m, with an
  RS rank exactly like the per-stock RRS ranking.
- **Industry board**: composite "industry indexes" built from every classified
  symbol in the shared `symbol_classification.csv` cache (the one BounceBot
  maintains) plus user-defined micro-industry groups in
  `data/custom_industry_groups.json` (e.g. Photonics = AAOI/LITE/CIEN/AXTI, which
  trade as their own group long before any official industry list notices).
  Each industry row reports median member % change, blended RS vs SPY, volume
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
import sys
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
INDUSTRY_BOARD_TEXT_FILE = OUTPUT_DIR / "industry_indexes.txt"
INDUSTRY_BOARD_CSV_FILE = OUTPUT_DIR / "industry_indexes.csv"
SECTOR_BOARD_CSV_FILE = OUTPUT_DIR / "sector_indexes.csv"

DEFAULT_CUSTOM_INDUSTRY_GROUPS = {
    "Photonics": ["AAOI", "LITE", "CIEN", "AXTI", "COHR", "FN"],
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


def collect_industry_members(
    classifications: dict[str, dict],
    custom_groups: dict[str, list[str]] | None = None,
    *,
    restrict_to: set[str] | None = None,
) -> dict[str, list[str]]:
    """Industry label -> member symbols, from the cache plus custom groups.

    Custom groups are first-class industries (marked with a trailing ``*`` so the
    board shows which rows are user-defined) and are never restricted: if you
    bothered to define Photonics, you want the whole group measured.
    """
    members: dict[str, set[str]] = {}
    for symbol, row in (classifications or {}).items():
        industry = str(row.get("industry") or "").strip()
        if not industry:
            continue
        if restrict_to is not None and symbol not in restrict_to:
            continue
        members.setdefault(industry, set()).add(symbol)
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
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_industry_scan(
    *,
    extra_symbols: list[str] | None = None,
    sectors_only: bool = False,
    write_outputs: bool = True,
) -> dict:
    """Build both boards and (optionally) write the text/CSV outputs."""
    classifications = load_symbol_classifications()
    custom_groups = load_custom_industry_groups()
    watchlist_symbols = gather_watchlist_symbols()

    universe: set[str] = set(watchlist_symbols)
    universe.update(classifications.keys())
    universe.update(s for members in custom_groups.values() for s in members)
    universe.update(extra_symbols or [])

    fetch_symbols = set(SECTOR_ETFS) | {BENCHMARK_SYMBOL}
    if not sectors_only:
        fetch_symbols |= universe
    frames = fetch_daily_frames_yf(sorted(fetch_symbols))

    sector_rows = build_sector_board(frames)
    industry_rows = []
    if not sectors_only:
        industry_members = collect_industry_members(classifications, custom_groups)
        industry_rows = build_industry_board(frames, industry_members)

    if write_outputs:
        INDUSTRY_BOARD_TEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
        INDUSTRY_BOARD_TEXT_FILE.write_text(render_board_text(sector_rows, industry_rows), encoding="utf-8")
        _write_csv(SECTOR_BOARD_CSV_FILE, sector_rows)
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
