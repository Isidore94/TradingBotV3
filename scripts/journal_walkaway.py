#!/usr/bin/env python3
"""Walkaway analysis for journal trades and human focus picks.

For every closed equity trade in the journal (Questrade + IBKR imports) and
every focus pick, replay the daily bars around the position and answer the
questions that actually train the trader:

- **Could we have held?** Where was price 5/10/20 sessions after the exit, and
  what was the best additional move available?
- **What should the stop have been?** How much heat (adverse excursion) did the
  trade take before its peak — per trade, and the P50/P80 heat across *winning*
  trades, which is the tightest stop that would have kept most winners alive.
- **What should the TP have been?** The max favorable excursion within the hold
  window vs what was actually captured (capture ratio).

Everything is normalized to ATR(20 at entry) so trades across symbols compare,
and per-trade verdicts ("left +1.8 ATR on the table within 10 sessions") plus
aggregate lessons are written to output/journal_walkaway.txt and a per-position
CSV. Focus picks run through the identical engine (entry = pick-day close) so
the flaws in what you *like* get the same treatment as what you *traded*.

Run:
    .venv/Scripts/python.exe scripts/journal_walkaway.py            # both sources
    .venv/Scripts/python.exe scripts/journal_walkaway.py --source journal
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_paths import OUTPUT_DIR  # noqa: E402

WALKAWAY_TEXT_FILE = OUTPUT_DIR / "journal_walkaway.txt"
WALKAWAY_CSV_FILE = OUTPUT_DIR / "journal_walkaway.csv"

HOLD_SESSIONS = 20
WALKAWAY_HORIZONS = (5, 10, 20)
ATR_SESSIONS = 20
STOP_BUFFER_ATR = 0.10
# Verdict thresholds (ATR units).
LEFT_ON_TABLE_MIN_ATR = 0.75
LOW_HEAT_WINNER_MAX_ATR = 0.40
EQUITY_SECURITY_TYPES = {"", "STK", "STOCK", "COMMON", "CS", "EQUITY", "EQ"}


@dataclass
class WalkawayPosition:
    """One position to replay — a journal trade or a focus pick."""

    source: str  # "journal" | "focus"
    symbol: str
    side: str  # LONG | SHORT
    entry_date: str  # ISO date
    entry_price: float | None = None  # focus picks: filled from pick-day close
    exit_date: str = ""
    exit_price: float | None = None
    label: str = ""
    net_pnl: float | None = None
    tags: str = ""
    extra: dict = field(default_factory=dict)


def _direction(side: str) -> float:
    return -1.0 if str(side or "").strip().upper() == "SHORT" else 1.0


def _atr_at_index(bars: pd.DataFrame, idx: int, sessions: int = ATR_SESSIONS) -> float | None:
    if idx < 1:
        return None
    window = bars.iloc[max(0, idx - sessions) : idx + 1]
    highs = pd.to_numeric(window["high"], errors="coerce")
    lows = pd.to_numeric(window["low"], errors="coerce")
    closes = pd.to_numeric(window["close"], errors="coerce")
    prev_close = closes.shift(1)
    tr = pd.concat([highs - lows, (highs - prev_close).abs(), (lows - prev_close).abs()], axis=1).max(axis=1)
    tr = tr.dropna()
    if tr.empty:
        return None
    value = float(tr.mean())
    return value if value > 0 else None


def _index_for_date(dates: list[str], target: str) -> int | None:
    """First bar index with date >= target (the bar that traded that day)."""
    for idx, value in enumerate(dates):
        if value >= target:
            return idx
    return None


def compute_walkaway(
    bars: pd.DataFrame | None,
    position: WalkawayPosition,
    *,
    hold_sessions: int = HOLD_SESSIONS,
    horizons: tuple[int, ...] = WALKAWAY_HORIZONS,
) -> dict | None:
    """Replay one position over daily bars. Returns a flat metrics row or None."""
    if bars is None or getattr(bars, "empty", True):
        return None
    work = bars.dropna(subset=["close"]).reset_index(drop=True)
    if len(work) < 2:
        return None
    dates = [pd.Timestamp(value).date().isoformat() for value in work["datetime"]]
    entry_idx = _index_for_date(dates, position.entry_date)
    if entry_idx is None:
        return None

    entry_bar = work.iloc[entry_idx]
    entry_price = position.entry_price
    if entry_price is None:
        entry_price = float(pd.to_numeric(entry_bar.get("close"), errors="coerce"))
    if not entry_price or entry_price <= 0:
        return None
    direction = _direction(position.side)
    atr = _atr_at_index(work, entry_idx)
    if not atr:
        return None

    row = {
        "source": position.source,
        "label": position.label,
        "symbol": position.symbol,
        "side": "SHORT" if direction < 0 else "LONG",
        "entry_date": dates[entry_idx],
        "entry_price": round(float(entry_price), 4),
        "atr_at_entry": round(float(atr), 4),
        "net_pnl": position.net_pnl,
        "tags": position.tags,
    }

    # Hold-window excursions (post-entry bars only; entry modeled at the close).
    forward = work.iloc[entry_idx + 1 : entry_idx + 1 + hold_sessions]
    if forward.empty:
        return None
    highs = pd.to_numeric(forward["high"], errors="coerce")
    lows = pd.to_numeric(forward["low"], errors="coerce")
    if direction > 0:
        favorable = (highs - entry_price)
        adverse = (entry_price - lows)
        peak_pos = int(favorable.to_numpy().argmax())
        peak_price = float(highs.iloc[peak_pos])
    else:
        favorable = (entry_price - lows)
        adverse = (highs - entry_price)
        peak_pos = int(favorable.to_numpy().argmax())
        peak_price = float(lows.iloc[peak_pos])
    mfe = max(0.0, float(favorable.max()))
    mae = max(0.0, float(adverse.max()))
    row["mfe_atr"] = round(mfe / atr, 2)
    row["mae_atr"] = round(mae / atr, 2)
    row["mfe_pct"] = round(mfe / entry_price * 100, 2)
    row["peak_sessions_after_entry"] = peak_pos + 1

    # Heat before the peak = the stop question. The tightest stop that kept this
    # trade alive sits just past the worst price seen before the peak bar.
    before_peak = forward.iloc[: peak_pos + 1]
    if direction > 0:
        worst_before_peak = float(pd.to_numeric(before_peak["low"], errors="coerce").min())
        heat = max(0.0, entry_price - worst_before_peak)
        suggested_stop = worst_before_peak - STOP_BUFFER_ATR * atr
    else:
        worst_before_peak = float(pd.to_numeric(before_peak["high"], errors="coerce").max())
        heat = max(0.0, worst_before_peak - entry_price)
        suggested_stop = worst_before_peak + STOP_BUFFER_ATR * atr
    row["heat_before_peak_atr"] = round(heat / atr, 2)
    row["suggested_stop_price"] = round(suggested_stop, 4)
    row["suggested_tp_price"] = round(peak_price, 4)

    # Exit + walkaway (journal trades; focus picks have no exit).
    captured = None
    if position.exit_price is not None and position.exit_date:
        captured = (float(position.exit_price) - entry_price) * direction
        row["exit_date"] = position.exit_date
        row["exit_price"] = round(float(position.exit_price), 4)
        row["captured_atr"] = round(captured / atr, 2)
        row["capture_ratio_pct"] = round(captured / mfe * 100, 1) if mfe > 0 else None
        exit_idx = _index_for_date(dates, position.exit_date)
        if exit_idx is not None:
            closes = pd.to_numeric(work["close"], errors="coerce")
            for horizon in horizons:
                key = f"walkaway_{horizon}d_atr"
                target_idx = exit_idx + horizon
                if target_idx < len(work):
                    extra_move = (float(closes.iloc[target_idx]) - float(position.exit_price)) * direction
                    row[key] = round(extra_move / atr, 2)
                else:
                    row[key] = None
            post_exit = work.iloc[exit_idx + 1 : exit_idx + 1 + hold_sessions]
            if not post_exit.empty:
                if direction > 0:
                    best_after = float(pd.to_numeric(post_exit["high"], errors="coerce").max()) - float(position.exit_price)
                else:
                    best_after = float(position.exit_price) - float(pd.to_numeric(post_exit["low"], errors="coerce").min())
                row["best_after_exit_atr"] = round(max(0.0, best_after) / atr, 2)

    # Per-position verdicts, most actionable first.
    verdicts = []
    walk10 = row.get("walkaway_10d_atr")
    if walk10 is not None and walk10 >= LEFT_ON_TABLE_MIN_ATR:
        verdicts.append(f"LEFT +{walk10:.1f} ATR on the table within 10 sessions of exit — could have held")
    if captured is not None and captured < 0 and row["mfe_atr"] >= 1.0:
        verdicts.append(
            f"Loser that offered +{row['mfe_atr']:.1f} ATR first — timing/target problem, not a bad pick"
        )
    if captured is not None and captured > 0 and row.get("capture_ratio_pct") is not None and row["capture_ratio_pct"] < 40:
        verdicts.append(f"Captured only {row['capture_ratio_pct']:.0f}% of the available move")
    if captured is not None and captured < 0 and row["heat_before_peak_atr"] <= LOW_HEAT_WINNER_MAX_ATR and row["mfe_atr"] >= 1.0:
        verdicts.append("Stop was tighter than the trade needed — it barely took heat before working")
    row["verdicts"] = "; ".join(verdicts)
    return row


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------
def _percentile(values: list[float], pct: float) -> float | None:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return None
    pos = (len(clean) - 1) * (pct / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(clean) - 1)
    frac = pos - lower
    return clean[lower] * (1 - frac) + clean[upper] * frac


def summarize_walkaway_rows(rows: list[dict]) -> dict:
    """Aggregate lessons across positions: hold, stop and target guidance."""
    summary: dict = {"count": len(rows)}
    if not rows:
        return summary
    walk10 = [r.get("walkaway_10d_atr") for r in rows if r.get("walkaway_10d_atr") is not None]
    if walk10:
        summary["median_walkaway_10d_atr"] = round(median(walk10), 2)
        summary["exited_early_rate_pct"] = round(
            100.0 * sum(1 for v in walk10 if v >= LEFT_ON_TABLE_MIN_ATR) / len(walk10), 1
        )
    # Winners = positions that offered at least 1 ATR (for focus picks) or made
    # money (for journal trades). Their heat distribution is the stop lesson.
    winner_heat = [
        r.get("heat_before_peak_atr")
        for r in rows
        if r.get("heat_before_peak_atr") is not None
        and ((r.get("captured_atr") or 0) > 0 or (r.get("captured_atr") is None and (r.get("mfe_atr") or 0) >= 1.0))
    ]
    if winner_heat:
        summary["winner_heat_p50_atr"] = round(_percentile(winner_heat, 50), 2)
        summary["winner_heat_p80_atr"] = round(_percentile(winner_heat, 80), 2)
    mfe_values = [r.get("mfe_atr") for r in rows if r.get("mfe_atr") is not None]
    if mfe_values:
        summary["median_mfe_atr"] = round(median(mfe_values), 2)
    capture_values = [r.get("capture_ratio_pct") for r in rows if r.get("capture_ratio_pct") is not None]
    if capture_values:
        summary["median_capture_ratio_pct"] = round(median(capture_values), 1)
    return summary


def render_walkaway_report(journal_rows: list[dict], focus_rows: list[dict]) -> str:
    lines = [
        f"WALKAWAY ANALYSIS  (generated {datetime.now().isoformat(timespec='minutes')})",
        "All distances in ATR(20 at entry). Entry modeled at the entry-day close.",
        "",
    ]

    def _section(title: str, rows: list[dict]) -> None:
        summary = summarize_walkaway_rows(rows)
        lines.append(f"== {title} ({summary.get('count', 0)} positions) ==")
        if not rows:
            lines.append("  (no analyzable positions)")
            lines.append("")
            return
        if summary.get("median_walkaway_10d_atr") is not None:
            lines.append(
                f"  Could you have held?  Median move 10 sessions AFTER your exit: "
                f"{summary['median_walkaway_10d_atr']:+.2f} ATR; "
                f"{summary.get('exited_early_rate_pct', 0):.0f}% of exits left >= {LEFT_ON_TABLE_MIN_ATR} ATR behind."
            )
        if summary.get("winner_heat_p80_atr") is not None:
            lines.append(
                f"  Stop guidance: winners took P50 {summary.get('winner_heat_p50_atr'):+.2f} / "
                f"P80 {summary['winner_heat_p80_atr']:+.2f} ATR of heat before paying — "
                f"stops tighter than ~{summary['winner_heat_p80_atr']:.1f} ATR cut winners."
            )
        if summary.get("median_mfe_atr") is not None:
            capture = summary.get("median_capture_ratio_pct")
            capture_text = f"; median capture ratio {capture:.0f}%" if capture is not None else ""
            lines.append(
                f"  Target guidance: median trade offered {summary['median_mfe_atr']:.1f} ATR "
                f"within {HOLD_SESSIONS} sessions{capture_text}."
            )
        lines.append("")
        flagged = [r for r in rows if r.get("verdicts")]
        flagged.sort(key=lambda r: -(r.get("walkaway_10d_atr") or 0))
        if flagged:
            lines.append("  Actionable flags:")
            for row in flagged[:25]:
                lines.append(
                    f"   {row['entry_date']} {row['symbol']:<6} {row['side']:<5} "
                    f"stop~{row['suggested_stop_price']} tp~{row['suggested_tp_price']} | {row['verdicts']}"
                )
            lines.append("")

    _section("JOURNAL TRADES", journal_rows)
    _section("FOCUS PICKS (what you liked)", focus_rows)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def load_journal_positions() -> tuple[list[WalkawayPosition], int]:
    """Closed equity trades from the journal DB. Returns (positions, skipped)."""
    from journal_store import JournalStore

    store = JournalStore()
    store.initialize_schema()
    trades = store.list_trades()
    positions: list[WalkawayPosition] = []
    skipped = 0
    for trade in trades:
        if str(trade.get("status") or "").upper() not in {"CLOSED", "CLOSED_PARTIAL"}:
            continue
        security_type = str(trade.get("security_type") or "").strip().upper()
        if security_type not in EQUITY_SECURITY_TYPES:
            skipped += 1
            continue
        opened = str(trade.get("opened_at") or "")[:10]
        closed = str(trade.get("closed_at") or "")[:10]
        entry_price = float(trade.get("average_entry_price") or 0.0)
        exit_price = float(trade.get("average_exit_price") or 0.0)
        if not opened or not closed or entry_price <= 0 or exit_price <= 0:
            continue
        positions.append(
            WalkawayPosition(
                source="journal",
                symbol=str(trade.get("symbol") or "").strip().upper(),
                side="SHORT" if str(trade.get("direction") or "").upper() == "SHORT" else "LONG",
                entry_date=opened,
                entry_price=entry_price,
                exit_date=closed,
                exit_price=exit_price,
                label=str(trade.get("trade_id") or ""),
                net_pnl=float(trade.get("net_pnl") or 0.0),
                tags=str(trade.get("display_tags") or ""),
            )
        )
    return positions, skipped


def load_focus_positions() -> list[WalkawayPosition]:
    """Every snapshotted human focus pick (entry = pick-day close, no exit)."""
    from human_focus_tracking import load_human_focus_daily_picks

    positions = []
    for row in load_human_focus_daily_picks():
        symbol = str(row.get("symbol") or "").strip().upper()
        trade_date = str(row.get("trade_date") or "").strip()
        if not symbol or not trade_date:
            continue
        positions.append(
            WalkawayPosition(
                source="focus",
                symbol=symbol,
                side="SHORT" if str(row.get("side") or "").upper() == "SHORT" else "LONG",
                entry_date=trade_date,
                label=f"focus:{trade_date}:{symbol}",
            )
        )
    return positions


def _history_period_for(positions: list[WalkawayPosition]) -> str:
    oldest = min((p.entry_date for p in positions), default="")
    if not oldest:
        return "1y"
    try:
        age_days = (datetime.now().date() - datetime.fromisoformat(oldest).date()).days
    except ValueError:
        return "1y"
    for period, days in (("6mo", 120), ("1y", 300), ("2y", 650), ("5y", 1700)):
        if age_days + 90 <= days:
            return period
    return "max"


def run_walkaway_analysis(*, source: str = "both", write_outputs: bool = True) -> dict:
    journal_positions: list[WalkawayPosition] = []
    focus_positions: list[WalkawayPosition] = []
    skipped_non_equity = 0
    if source in ("journal", "both"):
        try:
            journal_positions, skipped_non_equity = load_journal_positions()
        except Exception as exc:
            logging.warning("Journal positions unavailable (%s).", exc)
    if source in ("focus", "both"):
        try:
            focus_positions = load_focus_positions()
        except Exception as exc:
            logging.warning("Focus picks unavailable (%s).", exc)

    positions = journal_positions + focus_positions
    if not positions:
        logging.info("No positions to analyze.")
        return {"journal_rows": [], "focus_rows": [], "skipped_non_equity": skipped_non_equity}

    from industry_scanner import fetch_daily_frames_yf

    symbols = sorted({p.symbol for p in positions})
    frames = fetch_daily_frames_yf(symbols, period=_history_period_for(positions))

    journal_rows = [r for p in journal_positions if (r := compute_walkaway(frames.get(p.symbol), p))]
    focus_rows = [r for p in focus_positions if (r := compute_walkaway(frames.get(p.symbol), p))]

    if write_outputs:
        WALKAWAY_TEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
        WALKAWAY_TEXT_FILE.write_text(render_walkaway_report(journal_rows, focus_rows), encoding="utf-8")
        all_rows = journal_rows + focus_rows
        if all_rows:
            fieldnames = sorted({key for row in all_rows for key in row})
            with open(WALKAWAY_CSV_FILE, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
        logging.info(
            "Walkaway analysis: %s journal trade(s), %s focus pick(s) -> %s (skipped %s non-equity)",
            len(journal_rows),
            len(focus_rows),
            WALKAWAY_TEXT_FILE,
            skipped_non_equity,
        )
    return {
        "journal_rows": journal_rows,
        "focus_rows": focus_rows,
        "skipped_non_equity": skipped_non_equity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Walkaway analysis for journal trades + focus picks")
    parser.add_argument("--source", choices=("journal", "focus", "both"), default="both")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_walkaway_analysis(source=args.source)
    print(render_walkaway_report(result["journal_rows"], result["focus_rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
