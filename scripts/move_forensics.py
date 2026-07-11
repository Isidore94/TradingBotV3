#!/usr/bin/env python3
"""Move forensics: find every big clean move, then reverse-engineer why.

The setup tracker and playbook study both work *forward*: detect a setup we
already believe in, measure what happened. This module is the bot's critical
thinking in the other direction: it scans the durable daily-bar store for
every stock that made a GOOD MOVE (long and short), snapshots the conditions
that were true just before each move started (using only data through that
day - no lookahead), and mines which conditions - alone and in PAIRS - were
over-represented at move starts relative to ordinary days (lift). Conditions
reuse the exact signals the bot already tracks: AVWAP band breaks/bounces
(incl. 2nd-stdev), EMA 8/15/21 bounces, SMA reclaims, 252d breakouts, volume
thrusts, weekly 8EMA regime, earnings recency, band zone.

Because the mining is outcome-first, it can surface combinations that are NOT
in the current playbook at all ("novel" patterns: no detector member) and
refinements to existing families (detector + context pairs with higher lift
than the detector alone).

Outputs (OUTPUT_DIR/reports/):
- move_forensics_movers.csv     one row per move episode with every flag
- move_forensics_baseline.csv   matched ordinary-day sample, same flags
  (these two are the spreadsheet/database an external AI can deep-dive)
- move_forensics_patterns.csv   single + pair leaderboard with lift
- move_forensics_report.txt     human summary
- move_forensics_ai_digest.json structured digest for AI analysis

Run:
    .venv/Scripts/python.exe scripts/move_forensics.py --days 150
    .venv/Scripts/python.exe scripts/move_forensics.py --days 150 --max-symbols 300
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_paths import OUTPUT_DIR  # noqa: E402
from setup_playbook_study import (  # noqa: E402
    MIN_PRICE,
    PLAYBOOK,
    SymbolContext,
    _durable_symbols,
    _load_daily_frame,
    _tag_and_hold,
    build_symbol_context,
    compute_weekly_streak_series,
)
from master_avwap_lib.legacy import load_earnings_date_cache  # noqa: E402

FORENSICS_MOVERS_CSV = OUTPUT_DIR / "reports" / "move_forensics_movers.csv"
FORENSICS_BASELINE_CSV = OUTPUT_DIR / "reports" / "move_forensics_baseline.csv"
FORENSICS_PATTERNS_CSV = OUTPUT_DIR / "reports" / "move_forensics_patterns.csv"
FORENSICS_REPORT_TXT = OUTPUT_DIR / "reports" / "move_forensics_report.txt"
FORENSICS_AI_DIGEST_JSON = OUTPUT_DIR / "reports" / "move_forensics_ai_digest.json"

# A "good move": >= MOVE_MIN_ATR ATRs of favorable travel within
# MOVE_HORIZON_SESSIONS sessions, with at most MOVE_MAX_ADVERSE_ATR ATRs of
# pain before the peak (clean trend legs, not chop that eventually got there).
MOVE_HORIZON_SESSIONS = 10
MOVE_MIN_ATR = 3.0
MOVE_MAX_ADVERSE_ATR = 1.0
# A condition counts as "present" if it fired on the signal day or the 2
# sessions before it (moves often launch a day or two after the setup bar).
FEATURE_LOOKBACK_SESSIONS = 3
# Ordinary-day baseline: every Nth eligible non-move day per symbol/side.
BASELINE_SAMPLE_EVERY = 5
# Pattern mining gates.
PATTERN_MIN_MOVER_SUPPORT = 8
PAIR_CANDIDATE_FLAG_LIMIT = 25
WEEKLY_STRONG_WEEKS = 5

# Playbook detectors reused as condition flags (baseline control excluded).
_DETECTOR_FLAGS = [name for name in PLAYBOOK if name != "baseline_every5"]


# ---------------------------------------------------------------------------
# Condition snapshot (no lookahead: everything reads data through day i only)
# ---------------------------------------------------------------------------
def _detector_fired_recently(ctx: SymbolContext, i: int, family: str) -> bool:
    spec = PLAYBOOK[family]
    for j in range(max(0, i - FEATURE_LOOKBACK_SESSIONS + 1), i + 1):
        if spec.get("needs_bands") and not np.isfinite(ctx.vwap[j]):
            continue
        try:
            if spec["fn"](ctx, j):
                return True
        except (IndexError, ValueError):
            continue
    return False


def _ema15_bounce_recently(ctx: SymbolContext, i: int) -> bool:
    """The user's named signal: price tags the daily 15EMA and closes back over
    it while the prior close was already on the right side (trend intact)."""
    for j in range(max(1, i - FEATURE_LOOKBACK_SESSIONS + 1), i + 1):
        if not np.isfinite(ctx.ema15[j]) or not np.isfinite(ctx.ema15[j - 1]):
            continue
        if ctx.close[j - 1] > ctx.ema15[j - 1] and _tag_and_hold(ctx, j, ctx.ema15[j]):
            return True
    return False


def collect_condition_flags(
    ctx: SymbolContext,
    i: int,
    *,
    weekly_streak: int,
    side: str,
) -> dict[str, bool]:
    """Boolean condition flags for day ``i`` (mirror-aware via ``ctx``).

    ``weekly_streak`` is in ORIGINAL orientation (signed weeks vs the weekly
    8EMA); alignment flags translate it to the trade direction."""
    flags: dict[str, bool] = {}
    for family in _DETECTOR_FLAGS:
        flags[family] = _detector_fired_recently(ctx, i, family)
    flags["ema15_bounce"] = _ema15_bounce_recently(ctx, i)

    bands_ok = np.isfinite(ctx.vwap[i])
    flags["above_avwape"] = bool(bands_ok and ctx.close[i] > ctx.vwap[i])
    flags["above_first_dev"] = bool(np.isfinite(ctx.upper1[i]) and ctx.close[i] > ctx.upper1[i])
    flags["above_second_dev"] = bool(np.isfinite(ctx.upper2[i]) and ctx.close[i] > ctx.upper2[i])

    aligned_streak = weekly_streak if side == "LONG" else -weekly_streak
    flags["weekly_aligned_strong"] = aligned_streak >= WEEKLY_STRONG_WEEKS
    flags["weekly_against_strong"] = aligned_streak <= -WEEKLY_STRONG_WEEKS

    if i >= 21:
        avg_volume = float(np.mean(ctx.volume[i - 20 : i]))
        flags["volume_2x_recent"] = bool(
            avg_volume > 0
            and any(
                ctx.volume[j] >= 2.0 * avg_volume
                for j in range(max(0, i - FEATURE_LOOKBACK_SESSIONS + 1), i + 1)
            )
        )
    else:
        flags["volume_2x_recent"] = False

    flags["trend20_aligned"] = bool(i >= 20 and ctx.close[i] > ctx.close[i - 20])

    if i >= 60:
        lookback = min(i, 252)
        prior_extreme = float(np.max(ctx.high[i - lookback : i]))
        denom = abs(ctx.close[i]) or 1.0
        flags["near_252_extreme"] = bool((prior_extreme - ctx.close[i]) / denom <= 0.05)
    else:
        flags["near_252_extreme"] = False

    since_earnings = ctx.sessions_since_last_earnings(i)
    flags["earnings_0_5_sessions"] = since_earnings is not None and since_earnings <= 5
    flags["earnings_6_20_sessions"] = since_earnings is not None and 6 <= since_earnings <= 20
    return flags


def condition_flag_names() -> list[str]:
    """Stable column order for the CSV feature matrix."""
    return _DETECTOR_FLAGS + [
        "ema15_bounce",
        "above_avwape",
        "above_first_dev",
        "above_second_dev",
        "weekly_aligned_strong",
        "weekly_against_strong",
        "volume_2x_recent",
        "trend20_aligned",
        "near_252_extreme",
        "earnings_0_5_sessions",
        "earnings_6_20_sessions",
    ]


# ---------------------------------------------------------------------------
# Move-episode detection (arrays are in ctx orientation: shorts are mirrored,
# so "up" always means "favorable")
# ---------------------------------------------------------------------------
def find_move_episodes(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    scan_start: int,
    *,
    horizon: int = MOVE_HORIZON_SESSIONS,
    min_move_atr: float = MOVE_MIN_ATR,
    max_adverse_atr: float = MOVE_MAX_ADVERSE_ATR,
) -> list[dict]:
    """Non-overlapping clean-move episodes; each starts at the FIRST day whose
    forward window clears the bar (the day the pattern was live, pre-liftoff)."""
    episodes: list[dict] = []
    n = len(closes)
    i = max(scan_start, 1)
    last_full_start = n - 1 - horizon  # full forward window required
    while i <= last_full_start:
        atr_value = float(atr[i]) if np.isfinite(atr[i]) else 0.0
        close_value = float(closes[i])
        if atr_value <= 0 or not np.isfinite(close_value) or abs(close_value) < MIN_PRICE:
            i += 1
            continue
        window_high = highs[i + 1 : i + 1 + horizon]
        peak_offset = int(np.argmax(window_high))
        peak_idx = i + 1 + peak_offset
        peak_high = float(window_high[peak_offset])
        move_atr = (peak_high - close_value) / atr_value
        adverse_atr = (close_value - float(np.min(lows[i + 1 : peak_idx + 1]))) / atr_value
        if move_atr >= min_move_atr and adverse_atr <= max_adverse_atr:
            # Qualification uses the in-horizon window; the episode then
            # extends while the run keeps making new highs within a horizon of
            # the current peak, so one continuing move is one episode (the
            # continuation never re-counts as a fresh "move start").
            while peak_idx < n - 1:
                extension = highs[peak_idx + 1 : min(peak_idx + 1 + horizon, n)]
                if not len(extension):
                    break
                ext_offset = int(np.argmax(extension))
                if float(extension[ext_offset]) <= peak_high:
                    break
                peak_idx = peak_idx + 1 + ext_offset
                peak_high = float(extension[ext_offset])
            episodes.append(
                {
                    "signal_idx": i,
                    "peak_idx": peak_idx,
                    "move_atr": (peak_high - close_value) / atr_value,
                    "adverse_atr": max(0.0, adverse_atr),
                    "days_to_peak": peak_idx - i,
                    "move_pct": (peak_high - close_value) / (abs(close_value) or 1.0) * 100.0,
                }
            )
            i = peak_idx + 1
            continue
        i += 1
    return episodes


def _forward_move_atr(highs, closes, atr, i, horizon) -> float | None:
    atr_value = float(atr[i]) if np.isfinite(atr[i]) else 0.0
    if atr_value <= 0 or i + 1 + horizon > len(closes):
        return None
    return (float(np.max(highs[i + 1 : i + 1 + horizon])) - float(closes[i])) / atr_value


def scan_symbol(symbol: str, df, earnings_dates: list[str], *, days: int, **move_kwargs) -> tuple[list[dict], list[dict]]:
    """(mover_rows, baseline_rows) for one symbol, both sides."""
    n = len(df)
    scan_start = max(n - days, 30)
    if scan_start >= n:
        return [], []
    weekly_streaks = compute_weekly_streak_series(df)
    horizon = int(move_kwargs.get("horizon", MOVE_HORIZON_SESSIONS))

    movers: list[dict] = []
    baseline: list[dict] = []
    for side, mirrored in (("LONG", False), ("SHORT", True)):
        ctx = build_symbol_context(symbol, df, earnings_dates, scan_start_idx=scan_start, mirrored=mirrored)
        episodes = find_move_episodes(
            ctx.high, ctx.low, ctx.close, ctx.atr, scan_start, **move_kwargs
        )
        covered: set[int] = set()
        for episode in episodes:
            covered.update(range(episode["signal_idx"], episode["peak_idx"] + 1))
        for episode in episodes:
            i = episode["signal_idx"]
            movers.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "signal_date": ctx.dates[i].isoformat(),
                    "peak_date": ctx.dates[episode["peak_idx"]].isoformat(),
                    "move_atr": round(episode["move_atr"], 2),
                    "move_pct": round(episode["move_pct"], 2),
                    "adverse_atr": round(episode["adverse_atr"], 2),
                    "days_to_peak": episode["days_to_peak"],
                    "weekly_streak": int(weekly_streaks[i]),
                    **collect_condition_flags(ctx, i, weekly_streak=int(weekly_streaks[i]), side=side),
                }
            )
        # Ordinary-day baseline: same flag snapshot on sampled non-move days.
        last_full_start = n - 1 - horizon
        stride_phase = 0
        for i in range(scan_start, last_full_start + 1):
            if i in covered:
                continue
            atr_value = float(ctx.atr[i]) if np.isfinite(ctx.atr[i]) else 0.0
            if atr_value <= 0 or abs(float(ctx.close[i])) < MIN_PRICE:
                continue
            stride_phase += 1
            if stride_phase % BASELINE_SAMPLE_EVERY != 0:
                continue
            forward = _forward_move_atr(ctx.high, ctx.close, ctx.atr, i, horizon)
            baseline.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "signal_date": ctx.dates[i].isoformat(),
                    "forward_move_atr": round(forward, 2) if forward is not None else None,
                    "weekly_streak": int(weekly_streaks[i]),
                    **collect_condition_flags(ctx, i, weekly_streak=int(weekly_streaks[i]), side=side),
                }
            )
    return movers, baseline


# ---------------------------------------------------------------------------
# Pattern mining: which conditions (and pairs) are over-represented at move
# starts relative to ordinary days?
# ---------------------------------------------------------------------------
def _laplace_rate(hits: int, total: int) -> float:
    return (hits + 1.0) / (total + 2.0)


def _pattern_row(
    side: str,
    names: tuple[str, ...],
    mover_rows: list[dict],
    baseline_rows: list[dict],
) -> dict:
    mover_hits = [row for row in mover_rows if all(row.get(name) for name in names)]
    baseline_hits = [row for row in baseline_rows if all(row.get(name) for name in names)]
    mover_rate = _laplace_rate(len(mover_hits), len(mover_rows))
    baseline_rate = _laplace_rate(len(baseline_hits), len(baseline_rows))
    move_sizes = [row["move_atr"] for row in mover_hits if row.get("move_atr") is not None]
    includes_detector = any(name in PLAYBOOK for name in names)
    return {
        "side": side,
        "pattern": " + ".join(names),
        "kind": "pair" if len(names) > 1 else "single",
        "movers_with": len(mover_hits),
        "movers_total": len(mover_rows),
        "mover_rate": mover_rate,
        "baseline_with": len(baseline_hits),
        "baseline_total": len(baseline_rows),
        "baseline_rate": baseline_rate,
        "lift": mover_rate / baseline_rate,
        "avg_move_atr": (sum(move_sizes) / len(move_sizes)) if move_sizes else None,
        "includes_playbook_detector": includes_detector,
        "novel": not includes_detector,
    }


def mine_patterns(mover_rows: list[dict], baseline_rows: list[dict]) -> list[dict]:
    """Single + pair condition patterns per side, ranked by lift.

    Lift = P(pattern | move start) / P(pattern | ordinary day), Laplace
    smoothed. Support below PATTERN_MIN_MOVER_SUPPORT sinks to the bottom
    rather than disappearing (thin evidence stays visible, never decisive)."""
    flag_names = condition_flag_names()
    patterns: list[dict] = []
    for side in ("LONG", "SHORT"):
        side_movers = [row for row in mover_rows if row.get("side") == side]
        side_baseline = [row for row in baseline_rows if row.get("side") == side]
        if not side_movers or not side_baseline:
            continue
        singles = [
            _pattern_row(side, (name,), side_movers, side_baseline) for name in flag_names
        ]
        patterns.extend(singles)
        pair_candidates = sorted(
            (row for row in singles if row["movers_with"] >= PATTERN_MIN_MOVER_SUPPORT),
            key=lambda row: -row["lift"],
        )[:PAIR_CANDIDATE_FLAG_LIMIT]
        candidate_names = [row["pattern"] for row in pair_candidates]
        for name_a, name_b in combinations(candidate_names, 2):
            row = _pattern_row(side, (name_a, name_b), side_movers, side_baseline)
            if row["movers_with"] >= PATTERN_MIN_MOVER_SUPPORT:
                patterns.append(row)
    patterns.sort(
        key=lambda row: (
            row["movers_with"] < PATTERN_MIN_MOVER_SUPPORT,
            -row["lift"],
            -row["movers_with"],
        )
    )
    return patterns


# ---------------------------------------------------------------------------
# Report + AI digest
# ---------------------------------------------------------------------------
def _median(values: list[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def render_report(mover_rows: list[dict], patterns: list[dict], *, days: int, params: dict) -> str:
    lines = [
        f"MOVE FORENSICS  (generated {datetime.now().isoformat(timespec='minutes')})",
        f"Window: last {days} sessions | good move = >= {params['min_move_atr']} ATR favorable within "
        f"{params['horizon']} sessions with <= {params['max_adverse_atr']} ATR adverse before the peak.",
        "Direction: outcome-first. Every mover's conditions were snapshotted with data through the",
        f"signal day only (conditions count if they fired within the prior {FEATURE_LOOKBACK_SESSIONS} sessions).",
        "Lift = how much more often a condition was true at move starts than on ordinary days.",
        "",
    ]
    for side in ("LONG", "SHORT"):
        side_movers = [row for row in mover_rows if row["side"] == side]
        lines.append(
            f"== {side} MOVES: {len(side_movers)} episodes | median size "
            f"{_median([r['move_atr'] for r in side_movers]) or 0:.1f} ATR | median "
            f"{_median([float(r['days_to_peak']) for r in side_movers]) or 0:.0f} days to peak =="
        )
        side_patterns = [
            row
            for row in patterns
            if row["side"] == side and row["movers_with"] >= PATTERN_MIN_MOVER_SUPPORT
        ]
        for title, keep in (
            ("-- top single conditions --", lambda r: r["kind"] == "single"),
            ("-- top pair combos --", lambda r: r["kind"] == "pair"),
            ("-- NOT IN THE PLAYBOOK (no tracked setup family involved) --", lambda r: r["novel"]),
        ):
            lines.append(title)
            shown = 0
            for row in side_patterns:
                if not keep(row):
                    continue
                lines.append(
                    f"  {row['pattern']:<58} lift {row['lift']:>5.2f}x  "
                    f"n={row['movers_with']:>4} ({row['mover_rate']:.0%} of movers vs "
                    f"{row['baseline_rate']:.0%} of ordinary days)  avg {row['avg_move_atr'] or 0:.1f} ATR"
                )
                shown += 1
                if shown >= 12:
                    break
            if not shown:
                lines.append("  (nothing above the support threshold)")
        lines.append("")
    lines += [
        f"Full evidence: {FORENSICS_MOVERS_CSV.name} / {FORENSICS_BASELINE_CSV.name} "
        f"(feature matrix for AI deep-dive), {FORENSICS_PATTERNS_CSV.name} (this leaderboard),",
        f"{FORENSICS_AI_DIGEST_JSON.name} (structured digest - paste or attach it to Claude for analysis).",
    ]
    return "\n".join(lines) + "\n"


def build_ai_digest(mover_rows: list[dict], patterns: list[dict], *, days: int, params: dict) -> dict:
    strong = [row for row in patterns if row["movers_with"] >= PATTERN_MIN_MOVER_SUPPORT]
    return {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "what_this_is": (
            "Outcome-first move forensics: every clean >= "
            f"{params['min_move_atr']} ATR move (long and short) in the last {days} sessions, with the "
            "conditions that were true at each move's start (no lookahead), mined for lift vs "
            "ordinary days. Use it to find setup combinations the live playbook does not scan for."
        ),
        "how_to_analyze": [
            "Trust lift only where movers_with is comfortably above the support threshold.",
            "'novel: true' patterns involve no tracked setup family - candidates for NEW scan families.",
            "Detector+context pairs that out-lift the detector alone are refinement filters for existing families.",
            "Cross-check any candidate against the playbook study / setup tracker before promoting to scoring.",
            "The movers/baseline CSVs carry the full per-episode feature matrix for deeper slicing.",
        ],
        "params": {**params, "feature_lookback_sessions": FEATURE_LOOKBACK_SESSIONS,
                   "baseline_sample_every": BASELINE_SAMPLE_EVERY,
                   "min_support": PATTERN_MIN_MOVER_SUPPORT},
        "window_sessions": days,
        "mover_counts": {
            side: len([row for row in mover_rows if row["side"] == side])
            for side in ("LONG", "SHORT")
        },
        "top_patterns": strong[:40],
        "novel_patterns": [row for row in strong if row["novel"]][:20],
        "caveats": [
            "Survivorship: today's durable-store universe replayed backward.",
            "Lift is association, not tradability: entries/stops/costs are the playbook study's job.",
            "Overlapping conditions inflate each other (e.g. above_second_dev and second_dev_breakout).",
            "One market window; require agreement across windows before believing a pattern.",
        ],
        "files": {
            "movers_csv": str(FORENSICS_MOVERS_CSV),
            "baseline_csv": str(FORENSICS_BASELINE_CSV),
            "patterns_csv": str(FORENSICS_PATTERNS_CSV),
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_move_forensics(
    *,
    days: int = 150,
    symbols: list[str] | None = None,
    max_symbols: int | None = None,
    min_move_atr: float = MOVE_MIN_ATR,
    horizon: int = MOVE_HORIZON_SESSIONS,
    max_adverse_atr: float = MOVE_MAX_ADVERSE_ATR,
    write_outputs: bool = True,
    progress=None,
) -> dict:
    earnings_cache = load_earnings_date_cache()
    universe = symbols or _durable_symbols()
    if max_symbols:
        universe = universe[: int(max_symbols)]

    params = {"min_move_atr": min_move_atr, "horizon": horizon, "max_adverse_atr": max_adverse_atr}
    move_kwargs = {"min_move_atr": min_move_atr, "horizon": horizon, "max_adverse_atr": max_adverse_atr}
    mover_rows: list[dict] = []
    baseline_rows: list[dict] = []
    processed = 0
    for symbol in universe:
        df = _load_daily_frame(symbol)
        if df is None:
            continue
        dates = (earnings_cache["symbols"].get(symbol) or {}).get("dates") or []
        try:
            movers, baseline = scan_symbol(symbol, df, dates, days=days, **move_kwargs)
        except Exception as exc:
            logging.warning("%s: move forensics failed (%s)", symbol, exc)
            continue
        mover_rows.extend(movers)
        baseline_rows.extend(baseline)
        processed += 1
        if progress and processed % 100 == 0:
            progress(f"{processed} symbols scanned, {len(mover_rows)} moves found...")

    patterns = mine_patterns(mover_rows, baseline_rows)
    report = render_report(mover_rows, patterns, days=days, params=params)
    digest = build_ai_digest(mover_rows, patterns, days=days, params=params)
    if write_outputs:
        _write_csv(FORENSICS_MOVERS_CSV, mover_rows)
        _write_csv(FORENSICS_BASELINE_CSV, baseline_rows)
        _write_csv(FORENSICS_PATTERNS_CSV, patterns)
        FORENSICS_REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
        FORENSICS_REPORT_TXT.write_text(report, encoding="utf-8")
        FORENSICS_AI_DIGEST_JSON.write_text(json.dumps(digest, indent=2), encoding="utf-8")
    return {
        "movers": mover_rows,
        "baseline": baseline_rows,
        "patterns": patterns,
        "report": report,
        "digest": digest,
        "symbols_processed": processed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Outcome-first move forensics")
    parser.add_argument("--days", type=int, default=150, help="sessions to scan (default 150)")
    parser.add_argument("--symbols", default="", help="comma-separated symbol subset")
    parser.add_argument("--max-symbols", type=int, default=0, help="cap symbol count (debug)")
    parser.add_argument("--min-move-atr", type=float, default=MOVE_MIN_ATR)
    parser.add_argument("--horizon", type=int, default=MOVE_HORIZON_SESSIONS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    subset = [s.strip().upper() for s in args.symbols.replace(",", " ").split() if s.strip()] or None
    result = run_move_forensics(
        days=args.days,
        symbols=subset,
        max_symbols=args.max_symbols or None,
        min_move_atr=args.min_move_atr,
        horizon=args.horizon,
        progress=lambda msg: logging.info(msg),
    )
    print(result["report"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
