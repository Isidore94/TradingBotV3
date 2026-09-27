"""Regime report over ``market_regime_daily``: how each label behaved (P10). Read-only.

Per axis and label: sessions, segment count and lengths, forward 1/5/10/20-session
close-to-close returns from the label's session close (the label was known at that
close, so nothing here looks ahead of the label), forward 20-session max drawdown
(lowest low vs that close), the segment transition matrix and the full timeline.
Plus agreement of ``structural`` with the trader's typed segments.
"""

from __future__ import annotations

import shutil
import sqlite3
import statistics
import tempfile
from bisect import bisect_right
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # package import
    from . import regime_daily as rd
except ImportError:  # pragma: no cover
    import regime_daily as rd  # type: ignore

HORIZONS = (1, 5, 10, 20)
FWD_DRAWDOWN_SESSIONS = 20


def _pct(value: float | None) -> float | None:
    return None if value is None else round(value * 100.0, 3)


def forward_facts(d1: Any) -> dict[date, dict[str, float | None]]:
    """Per session: forward returns at ``HORIZONS`` and the 20-session forward max drawdown."""
    bars = rd.daily_bars(d1)
    out: dict[date, dict[str, float | None]] = {}
    for i, bar in enumerate(bars):
        close = bar["close"]
        facts: dict[str, float | None] = {}
        for k in HORIZONS:
            facts[f"fwd_{k}"] = bars[i + k]["close"] / close - 1.0 if i + k < len(bars) else None
        window = bars[i + 1: i + 1 + FWD_DRAWDOWN_SESSIONS]
        facts["fwd_mdd"] = (
            min(0.0, min(b["low"] for b in window) / close - 1.0) if len(window) == FWD_DRAWDOWN_SESSIONS else None
        )
        out[bar["session_date"]] = facts
    return out


def segments(pairs: Sequence[tuple[date, str]]) -> list[dict[str, Any]]:
    """Back-to-back sessions with one label, oldest first."""
    out: list[dict[str, Any]] = []
    for day, label in pairs:
        if out and out[-1]["label"] == label:
            out[-1]["end"] = day
            out[-1]["sessions"] += 1
        else:
            out.append({"start": day, "end": day, "label": label, "sessions": 1})
    return out


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean_pct": None, "median_pct": None, "pct_up": None}
    return {
        "n": len(values),
        "mean_pct": _pct(statistics.fmean(values)),
        "median_pct": _pct(statistics.median(values)),
        "pct_up": round(100.0 * sum(1 for v in values if v > 0) / len(values), 1),
    }


def _forward_block(days: Iterable[date], fwd: Mapping[date, Mapping[str, float | None]]) -> dict[str, Any]:
    days = list(days)
    block: dict[str, Any] = {}
    for k in HORIZONS:
        block[f"fwd_{k}"] = _summary([fwd[d][f"fwd_{k}"] for d in days if d in fwd and fwd[d][f"fwd_{k}"] is not None])
    mdd = [fwd[d]["fwd_mdd"] for d in days if d in fwd and fwd[d]["fwd_mdd"] is not None]
    block[f"fwd_max_drawdown_{FWD_DRAWDOWN_SESSIONS}"] = {
        "n": len(mdd),
        "mean_pct": _pct(statistics.fmean(mdd)) if mdd else None,
        "median_pct": _pct(statistics.median(mdd)) if mdd else None,
        "worst_pct": _pct(min(mdd)) if mdd else None,
    }
    return block


def axis_report(pairs: Sequence[tuple[date, str]], fwd: Mapping[date, Mapping[str, float | None]]) -> dict[str, Any]:
    segs = segments(pairs)
    total = len(pairs)
    labels: dict[str, Any] = {}
    for label in sorted({label for _day, label in pairs}):
        days = [day for day, lab in pairs if lab == label]
        lengths = [s["sessions"] for s in segs if s["label"] == label]
        labels[label] = {
            "sessions": len(days),
            "share_pct": round(100.0 * len(days) / total, 1) if total else None,
            "segments": len(lengths),
            "segment_sessions_median": statistics.median(lengths),
            "segment_sessions_mean": round(statistics.fmean(lengths), 1),
            "segment_sessions_max": max(lengths),
            **_forward_block(days, fwd),
        }
    transitions: dict[str, dict[str, int]] = {}
    for prev, cur in zip(segs, segs[1:], strict=False):
        row = transitions.setdefault(prev["label"], {})
        row[cur["label"]] = row.get(cur["label"], 0) + 1
    timeline = [{**s, "start": s["start"].isoformat(), "end": s["end"].isoformat()} for s in segs]
    return {"labels": labels, "transitions": transitions, "timeline": timeline}


def read_trader_segments(db_path: Path | str) -> list[dict[str, Any]]:
    """The trader's typed structural segments from a COPY of the journal (the live file is never opened)."""
    source = Path(db_path)
    if not source.exists():
        return []
    with tempfile.TemporaryDirectory(prefix="regime_report_journal_") as scratch:
        copy = Path(scratch) / source.name
        shutil.copy2(source, copy)
        for suffix in ("-wal", "-shm"):
            side = source.with_name(source.name + suffix)
            if side.exists():
                shutil.copy2(side, copy.with_name(copy.name + suffix))
        conn = sqlite3.connect(str(copy))
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(row) for row in conn.execute("SELECT * FROM structural_regime ORDER BY segment_id")]
        except sqlite3.Error:
            rows = []
        finally:
            conn.close()
    return rows


def trader_agreement(pairs: Sequence[tuple[date, str]], trader_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Session-by-session agreement of the auto label with the trader's typed segment in force."""
    from structural_regime import effective_segments

    timeline = effective_segments(trader_rows)
    typed = [
        {"start": str(seg["start_date"])[:10], "regime": seg["regime"], "note": seg.get("structure_note") or ""}
        for seg in timeline
    ]
    if not timeline:
        return {"status": "no typed segments", "segments_typed": []}
    starts = [date.fromisoformat(seg["start"]) for seg in typed]
    confusion: dict[str, dict[str, int]] = {}
    compared = agree = 0
    for day, auto in pairs:
        index = bisect_right(starts, day) - 1
        if index < 0:
            continue
        trader = typed[index]["regime"]
        row = confusion.setdefault(trader, {})
        row[auto] = row.get(auto, 0) + 1
        if auto == rd.UNKNOWN:
            continue
        compared += 1
        agree += auto == trader
    return {
        "status": "ok",
        "segments_typed": typed,
        "sessions_compared": compared,
        "sessions_agree": agree,
        "agree_pct": round(100.0 * agree / compared, 1) if compared else None,
        "confusion_trader_by_auto": confusion,
    }


def build_report(
    regimes: Any,
    d1: Any,
    *,
    symbol: str = "SPY",
    trader_rows: Iterable[Mapping[str, Any]] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    rows = rd._records(regimes)
    rows = sorted(rows, key=lambda r: rd._as_day(r["session_date"]))
    fwd = forward_facts(d1)
    days = [rd._as_day(r["session_date"]) for r in rows]
    report: dict[str, Any] = {
        "symbol": symbol,
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "dataset": rd.DATASET,
        "rule_version": str(rows[0]["rule_version"]) if rows else None,
        "structural_rule_version": rd.STRUCTURAL_RULE_VERSION,
        "point_in_time": (
            "Each label uses only bars completed by its session's close; forward returns "
            "run from that close, so they are what trading the label would have seen."
        ),
        "span": {"first": days[0].isoformat() if days else None, "last": days[-1].isoformat() if days else None,
                 "sessions": len(days)},
        "thresholds": {
            "rv_cuts": rd.RV_CUTS, "vix_cuts": rd.VIX_CUTS, "drawdown_cuts_pct": rd.DD_CUTS,
            "sma_slow_slope_sessions": rd.SMA_SLOW_SLOPE_SESSIONS, "hysteresis_sessions": rd.HYSTERESIS_SESSIONS,
            "capitulation": {"min_drawdown": rd.CAPITULATION_MIN_DD, "max_ret5": rd.CAPITULATION_MAX_RET5},
            "recovery": {"min_prior_drawdown": rd.RECOVERY_MIN_PRIOR_DD, "min_bounce": rd.RECOVERY_MIN_BOUNCE,
                         "lookback": rd.RECOVERY_LOOKBACK},
            "compression_ratio": rd.COMPRESSION_RATIO, "bull_max_drawdown": rd.BULL_MAX_DD,
            "composite_key": rd.COMPOSITE_KEY,
        },
        "baseline": {"sessions": len(days), **_forward_block(days, fwd)},
        "axes": {},
    }
    for axis in rd.AXES:
        pairs = [(day, str(row.get(axis) or rd.UNKNOWN)) for day, row in zip(days, rows, strict=True)]
        report["axes"][axis] = axis_report(pairs, fwd)
    structural = [(day, str(row.get("structural") or rd.UNKNOWN)) for day, row in zip(days, rows, strict=True)]
    report["trader_agreement"] = trader_agreement(structural, trader_rows)
    # The three regimes the trader described on 2026-09-26 (offered as prefills, not yet
    # confirmed in the journal), reported separately so they never pass for typed rows.
    from structural_regime import PREFILLS

    described = [
        {"segment_id": index + 1, "start_date": p.start_date, "regime": p.regime, "structure_note": p.structure_note}
        for index, p in enumerate(PREFILLS)
    ]
    report["described_agreement"] = trader_agreement(structural, described)
    return report


__all__ = ["build_report", "forward_facts", "read_trader_segments", "segments", "trader_agreement"]
