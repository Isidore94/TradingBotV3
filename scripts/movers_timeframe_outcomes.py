"""What is working on the Movers M30 and Daily boards: each scan's box rows and
their +1/+3/+5 session returns vs SPY (trader 2026-09-29: "we do want to see
whats working and whats not").

Evidence only, append-only JSONL (`MOVERS_TIMEFRAME_PICKS_FILE`):
- `kind: pick` - one row per box row per scan (tf, box, side, symbol, rank,
  score, entry close, SPY entry, session); a symbol/box/side is logged once per
  session.
- `kind: outcome` - one row per pick per horizon once that many sessions have
  closed after the pick session: the side-adjusted return, SPY's return and the
  excess (long: name - SPY; short: SPY - name). Never resolved twice.
A failed write loses those rows, never the board. Pure except `append_records`
/ `load_records` (reused from `movers_outcomes`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import movers_outcomes
import movers_scan
import movers_timeframe

HORIZONS = (1, 3, 5)
#: The box each list feeds, and the sides it holds.
BOXES = (("pop", ("long", "short")), ("dip_strong", ("long",)), ("dip_weak", ("short",)))
#: Summaries read the last this many pick sessions of the timeframe.
LOOKBACK_SESSIONS = 20
#: A pick older than this many of the name's sessions is no longer resolved.
MAX_AGE_SESSIONS = 30
#: The horizon a box title's hover line quotes first.
HEADLINE_HORIZON = 3

append_records = movers_outcomes.append_records
load_records = movers_outcomes.load_records


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (str(row.get("tf") or ""), str(row.get("box") or ""), str(row.get("side") or ""),
            str(row.get("symbol") or "").upper(), str(row.get("session") or ""))


def board_picks(board: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One `kind: pick` row per row on the board's three boxes."""
    tf = str(board.get("tf") or "")
    session = str(board.get("session") or "")
    spy_entry = movers_scan._finite((board.get("state") or {}).get("spy_last"))
    lists = (("pop", "long", "pop", "pop_score"), ("pop", "short", "pop", "pop_score"),
             ("dip_strong", "long", "swing", "dip_score"),
             ("dip_weak", "short", "swing", "dip_score"))
    out: list[dict[str, Any]] = []
    for box, side, key, score_key in lists:
        for rank, row in enumerate(((board.get(key) or {}).get(side)) or [], start=1):
            symbol = str(row.get("symbol") or "").strip().upper()
            entry = movers_scan._finite(row.get("last"))
            if not symbol or entry is None:
                continue
            out.append({"kind": "pick", "tf": tf, "box": box, "side": side, "symbol": symbol,
                        "rank": rank, "score": row.get(score_key), "entry_close": entry,
                        "spy_entry": spy_entry, "session": session,
                        "as_of": board.get("as_of") or "",
                        "scanned_at": board.get("scanned_at") or ""})
    return out


def new_picks(board: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The board's picks not already logged for that session (a re-scan adds nothing)."""
    logged = {_key(r) for r in records if r.get("kind") == "pick"}
    out = []
    for pick in board_picks(board):
        if _key(pick) not in logged:
            logged.add(_key(pick))
            out.append(pick)
    return out


def _closes_by_session(bars: Sequence[Mapping[str, Any]]) -> list[tuple[str, float]]:
    out = []
    for bar in bars or ():
        close = movers_scan._finite(bar.get("close"))
        day = movers_timeframe.session_date(bar.get("dt"))
        if close is None or day is None:
            continue
        out.append((day.isoformat(), close))
    return out


def resolve(
    records: Iterable[Mapping[str, Any]],
    daily_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    resolved_at: str = "",
    horizons: Sequence[int] = HORIZONS,
) -> list[dict[str, Any]]:
    """New `kind: outcome` rows for picks whose horizon session has closed.

    `daily_by_symbol` holds completed daily bars (aware `dt`, `close`) per
    symbol, SPY included. A pick is resolved from its entry close to the close
    N of the name's sessions after the pick session; SPY over the same span.
    A pick/horizon already resolved, or missing either close, is skipped."""
    rows = list(records)
    done = {(_key(r), int(r.get("horizon") or 0)) for r in rows if r.get("kind") == "outcome"}
    spy = dict(_closes_by_session(daily_by_symbol.get("SPY") or ()))
    series = {str(s).upper(): _closes_by_session(b) for s, b in daily_by_symbol.items()}
    out: list[dict[str, Any]] = []
    for pick in rows:
        if pick.get("kind") != "pick":
            continue
        closes = series.get(str(pick.get("symbol") or "").upper()) or []
        sessions = [day for day, _close in closes]
        try:
            start = sessions.index(str(pick.get("session") or ""))
        except ValueError:
            continue
        if len(sessions) - 1 - start > MAX_AGE_SESSIONS:
            continue
        entry = movers_scan._finite(pick.get("entry_close"))
        spy_entry = movers_scan._finite(pick.get("spy_entry"))
        side = "short" if pick.get("side") == "short" else "long"
        for horizon in horizons:
            if (_key(pick), int(horizon)) in done or start + horizon >= len(closes):
                continue
            target, close = closes[start + horizon]
            raw = movers_scan._pct(close, entry)
            spy_raw = movers_scan._pct(spy.get(target), spy_entry)
            excess = movers_outcomes._excess(side, raw, spy_raw)
            if raw is None or excess is None:
                continue
            done.add((_key(pick), int(horizon)))
            out.append({"kind": "outcome", "tf": pick.get("tf"), "box": pick.get("box"),
                        "side": side, "symbol": pick.get("symbol"),
                        "session": pick.get("session"), "rank": pick.get("rank"),
                        "horizon": int(horizon), "target_session": target,
                        "ret_pct": raw if side == "long" else -raw,
                        "spy_ret_pct": spy_raw, "excess_pct": excess, "beat": excess > 0,
                        "resolved_at": resolved_at})
    return out


def summarize(
    records: Iterable[Mapping[str, Any]],
    tf: str,
    box: str,
    side: str,
    lookback_sessions: int = LOOKBACK_SESSIONS,
) -> dict[str, Any]:
    """Per horizon over the last `lookback_sessions` pick sessions of `tf`:
    n, mean excess vs SPY (%), and the share that beat SPY (%)."""
    rows = list(records)
    sessions = sorted({str(r.get("session") or "") for r in rows
                       if r.get("kind") == "pick" and r.get("tf") == tf and r.get("session")})
    window = set(sessions[-max(1, int(lookback_sessions)):])
    picked = [r for r in rows if r.get("kind") == "pick" and r.get("tf") == tf
              and r.get("box") == box and r.get("side") == side and r.get("session") in window]
    out: dict[str, Any] = {"tf": tf, "box": box, "side": side, "sessions": len(window),
                           "lookback_sessions": int(lookback_sessions), "picks": len(picked),
                           "n": {}, "mean_excess_pct": {}, "beat_pct": {}}
    for horizon in HORIZONS:
        values = [float(r["excess_pct"]) for r in rows
                  if r.get("kind") == "outcome" and r.get("tf") == tf and r.get("box") == box
                  and r.get("side") == side and r.get("session") in window
                  and int(r.get("horizon") or 0) == horizon and r.get("excess_pct") is not None]
        key = str(horizon)
        out["n"][key] = len(values)
        out["mean_excess_pct"][key] = (sum(values) / len(values)) if values else None
        out["beat_pct"][key] = (100.0 * sum(1 for v in values if v > 0) / len(values)
                                if values else None)
    return out


def summaries(records: Iterable[Mapping[str, Any]], tf: str,
              lookback_sessions: int = LOOKBACK_SESSIONS) -> dict[str, dict[str, Any]]:
    """{box: {side: summary}} for every box of `tf`."""
    rows = list(records)
    return {box: {side: summarize(rows, tf, box, side, lookback_sessions) for side in sides}
            for box, sides in BOXES}


def summary_line(summary: Mapping[str, Any] | None) -> str:
    """'Last 20 sessions: +0.8% vs SPY at 3d, 57% beat, n=140' (3d, else the longest
    horizon with results), or 'no results yet'."""
    summary = summary or {}
    counts = summary.get("n") or {}
    order = [str(HEADLINE_HORIZON)] + [str(h) for h in sorted(HORIZONS, reverse=True)]
    key = next((k for k in order if int(counts.get(k) or 0) > 0), None)
    if key is None:
        return "no results yet"
    mean = summary["mean_excess_pct"][key]
    beat = summary["beat_pct"][key]
    lookback = int(summary.get("lookback_sessions") or LOOKBACK_SESSIONS)
    return (f"Last {lookback} sessions: {mean:+.1f}% vs SPY at {key}d, "
            f"{beat:.0f}% beat, n={int(counts[key])}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Movers M30 / Daily pick outcomes")
    parser.add_argument("--summary", action="store_true", help="print the summaries as JSON")
    parser.add_argument("--tf", choices=("m30", "d1"), help="one timeframe (default: both)")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_SESSIONS,
                        help="pick sessions to read (default 20)")
    parser.add_argument("--path", help="log (default: project_paths.MOVERS_TIMEFRAME_PICKS_FILE)")
    args = parser.parse_args(argv)
    if not args.summary:
        parser.print_help()
        return 2
    if args.path:
        path = Path(args.path)
    else:
        from project_paths import MOVERS_TIMEFRAME_PICKS_FILE

        path = MOVERS_TIMEFRAME_PICKS_FILE
    records = load_records(path)
    out: dict[str, Any] = {"path": str(path)}
    for tf in ([args.tf] if args.tf else ["m30", "d1"]):
        block = summaries(records, tf, args.lookback)
        out[tf] = {box: {side: dict(s, line=summary_line(s)) for side, s in sides.items()}
                   for box, sides in block.items()}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
