"""Measure what one symbol's D1 level store holds, and what the chart drops.

``chart_levels.horizontal_levels`` draws at most ``MAX_GREEN_HORIZONTALS`` /
``MAX_RED_HORIZONTALS`` / ``MAX_CLOUD_FLATS`` lines per bucket (10 / 6 / 4).
Those numbers were chosen on 2026-08-09 with no level store in view - the
reasoning was "a chart with forty lines on it is a chart nobody reads", which
is true but says nothing about whether a real store holds four levels or four
hundred. A budget that never binds is dead code; one that cuts a store in half
is silently deciding which S/R the trader gets to see.

So this reports it, per symbol, on the desk where the store actually is::

    .venv\\Scripts\\python.exe scripts\\d1_level_store_survey.py --symbol NVDA
    .venv\\Scripts\\python.exe scripts\\d1_level_store_survey.py --symbol AAPL --range 150 260
    .venv\\Scripts\\python.exe scripts\\d1_level_store_survey.py --symbol SPY --as-of 2026-07-01

What each block means:

* **total records in store** - every record in the symbol's JSON, including
  kinds the chart never draws. ``chart_levels._store_levels`` keeps only
  ``hv_horizontal`` and ``cloud_flat``; anything else is counted here and then
  gone, so the difference is visible rather than assumed.
* **the filters, in the chart's own order** - priced, then strength (green/red
  only; a cloud flat carries no strength gate), then effective-on, then the
  price range. Each line shows what survived and what that step cost. A record
  below ``MIN_HORIZONTAL_STRENGTH`` is never drawn under any budget, so it is
  not evidence that the budget is too small.
* **before / after the clutter budget** - the number that answers the actual
  question. "before" is what came through the three filters; "after" is
  ``len(horizontal_levels(...))`` run per bucket, so it is the production
  function's own output and cannot drift from what the chart paints. If
  "truncated" is zero everywhere the budget is not binding on this symbol.

Two honest limits, both about matching the chart exactly:

* The price range comes from the last ``--sessions`` bars of the same durable
  daily store ``chart_snapshot.load_d1_bars`` reads, which is where the chart's
  range comes from too - but the chart also appends today's FORMING candle, and
  that bar can widen the range on a big move. Intraday, treat the range as a
  floor. Pass ``--range`` to pin it.
* The chart's ``as_of`` is the last bar it is showing, not the calendar date.
  During a session those are the same day; on a weekend they are not, and
  ``--as-of`` exists for that.

Read-only. Opens the level store and the daily parquet, writes nothing,
contacts nothing, and touches no detector, score or alert.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The two kinds ``chart_levels._store_levels`` keeps. Everything else in the
#: store is counted and discarded.
DRAWN_KINDS = ("hv_horizontal", "cloud_flat")

BUCKETS = ("green", "red", "cloud")


def _ensure_scripts_on_path() -> None:
    """Make ``scripts/`` importable, exactly as the trendline survey does."""
    root = str(Path(__file__).resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bucket_of(record: Mapping[str, Any]) -> str:
    """green / red / cloud, by the same rule ``horizontal_levels`` applies."""
    if str(record.get("kind") or "") == "cloud_flat":
        return "cloud"
    # horizontal_levels: bucket = str(record.get("bucket") or "red").lower()
    return "green" if str(record.get("bucket") or "red").lower() == "green" else "red"


def survey(
    records: Sequence[Mapping[str, Any]],
    *,
    as_of: date | None = None,
    price_range: tuple[float, float] | None = None,
) -> dict:
    """Stage-by-stage counts for one symbol's raw store records.

    ``records`` is the store's whole ``levels`` list - every kind, unfiltered -
    so the report can show what the chart's kind filter costs. Pure: no I/O,
    no mutation of the inputs.

    The three filter stages replicate ``horizontal_levels`` in its own order,
    using its own constants and ``level_is_effective_on``. The post-budget
    counts do NOT replicate anything: they are ``horizontal_levels`` called
    once per bucket, so the "after" column is the production function's answer.
    """
    _ensure_scripts_on_path()
    from chart_levels import (
        MAX_CLOUD_FLATS,
        MAX_GREEN_HORIZONTALS,
        MAX_RED_HORIZONTALS,
        MIN_HORIZONTAL_STRENGTH,
        horizontal_levels,
    )
    from master_avwap_lib.levels import level_is_effective_on

    rows = [dict(record) for record in records or () if isinstance(record, Mapping)]
    drawn_kinds = set(DRAWN_KINDS)
    kept = [row for row in rows if str(row.get("kind") or "") in drawn_kinds]

    def _empty() -> dict[str, int]:
        return {name: 0 for name in BUCKETS}

    in_store = _empty()
    priced = _empty()
    after_strength = _empty()
    after_effective = _empty()
    before_budget = _empty()
    survivors: dict[str, list[dict]] = {name: [] for name in BUCKETS}

    low, high = price_range if price_range else (None, None)

    for row in kept:
        bucket = _bucket_of(row)
        in_store[bucket] += 1
        # horizontal_levels drops an unpriced or non-positive level first.
        price = _coerce_float(row.get("price"))
        if price is None or price <= 0:
            continue
        priced[bucket] += 1
        # ...then the strength gate, which applies to green/red only.
        if bucket != "cloud":
            strength = _coerce_float(row.get("strength")) or 0.0
            if strength < MIN_HORIZONTAL_STRENGTH:
                continue
        after_strength[bucket] += 1
        # ...then effective-on, which only a cloud flat can fail.
        if as_of is not None and not level_is_effective_on(dict(row), as_of):
            continue
        after_effective[bucket] += 1
        # ...then the chart's visible price range.
        if low is not None and not (low <= price <= high):
            continue
        before_budget[bucket] += 1
        survivors[bucket].append(row)

    budgets = {
        "green": int(MAX_GREEN_HORIZONTALS),
        "red": int(MAX_RED_HORIZONTALS),
        "cloud": int(MAX_CLOUD_FLATS),
    }
    # One call per bucket: each returns exactly that bucket's post-budget
    # lines, so no output has to be re-classified back into a bucket.
    after_budget = {
        name: len(
            horizontal_levels(survivors[name], as_of=as_of, price_range=price_range)
        )
        for name in BUCKETS
    }
    drawn_total = len(horizontal_levels(kept, as_of=as_of, price_range=price_range))

    return {
        "total_records": len(rows),
        "other_kinds": len(rows) - len(kept),
        "in_store": in_store,
        "priced": priced,
        "after_strength": after_strength,
        "after_effective": after_effective,
        "before_budget": before_budget,
        "after_budget": after_budget,
        "truncated": {
            name: before_budget[name] - after_budget[name] for name in BUCKETS
        },
        "budgets": budgets,
        "drawn_total": drawn_total,
        "min_strength": float(MIN_HORIZONTAL_STRENGTH),
        # A canary, not a filter: the per-bucket calls and the single whole-set
        # call must agree. They can only diverge if horizontal_levels grows a
        # cross-bucket rule, at which point this tool is lying and says so.
        "buckets_agree": sum(after_budget.values()) == drawn_total,
    }


def _chart_price_range(symbol: str, sessions: int) -> tuple[tuple[float, float] | None, int]:
    """(min low, max high) over the bars the chart shows, and the bar count.

    Same loader the snapshot path uses (``chart_snapshot.load_d1_bars`` is
    ``build_d1_snapshot``'s default), tailed to the same session count, so this
    is the chart's own range rather than a whole-history one.
    """
    _ensure_scripts_on_path()
    from chart_snapshot import load_d1_bars

    bars = load_d1_bars(symbol) or []
    shown = bars[-sessions:] if sessions and len(bars) > sessions else bars
    lows = [value for value in (_coerce_float(bar.get("low")) for bar in shown) if value]
    highs = [value for value in (_coerce_float(bar.get("high")) for bar in shown) if value]
    if not lows or not highs:
        return None, len(shown)
    return (min(lows), max(highs)), len(shown)


def _row(label: str, counts: Mapping[str, int], note: str = "") -> str:
    total = sum(counts[name] for name in BUCKETS)
    cells = "".join(f"{counts[name]:>7d}" for name in BUCKETS)
    return f"  {label:<26}{cells}{total:>8d}  {note}".rstrip()


def main(argv: list[str] | None = None) -> int:
    _ensure_scripts_on_path()
    from chart_levels import _store_levels
    from chart_snapshot import D1_DEFAULT_SESSIONS
    from master_avwap_lib.levels import level_store_path
    from project_paths import MASTER_AVWAP_LEVELS_DIR

    parser = argparse.ArgumentParser(description="Survey one symbol's D1 level store.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--levels-dir", type=Path, default=Path(MASTER_AVWAP_LEVELS_DIR))
    parser.add_argument(
        "--range",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        default=None,
        help="explicit price range; skips loading the daily bars",
    )
    parser.add_argument(
        "--as-of",
        default="",
        metavar="YYYY-MM-DD",
        help="effective-on date (default today)",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=D1_DEFAULT_SESSIONS,
        help=f"sessions the chart shows, for the price range (default {D1_DEFAULT_SESSIONS})",
    )
    args = parser.parse_args(argv)

    symbol = str(args.symbol or "").strip().upper()
    path = level_store_path(Path(args.levels_dir), symbol)
    if not path.exists():
        print(
            f"no level store for {symbol} at {path} - run a master scan first",
            file=sys.stderr,
        )
        return 2

    if args.as_of:
        try:
            as_of = date.fromisoformat(str(args.as_of)[:10])
        except ValueError:
            print(f"--as-of is not a date: {args.as_of!r}", file=sys.stderr)
            return 2
    else:
        as_of = date.today()

    # The kind filter is the chart's, so read through the chart's own loader
    # and take the raw payload only for the total it discards.
    kept = _store_levels(symbol, Path(args.levels_dir))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("levels") if isinstance(payload, dict) else None
        records = [row for row in (raw or []) if isinstance(row, dict)]
    except (OSError, ValueError) as exc:
        print(f"level store at {path} is unreadable: {exc}", file=sys.stderr)
        return 2
    if len(kept) > len(records):  # cannot happen; a torn read would say so
        records = list(kept)

    range_note = ""
    if args.range:
        price_range = (float(min(args.range)), float(max(args.range)))
        range_note = "from --range"
    else:
        try:
            price_range, bar_count = _chart_price_range(symbol, int(args.sessions))
        except Exception as exc:  # no parquet store, no pandas, no Drive
            price_range, bar_count = None, 0
            range_note = f"daily bars unreachable ({type(exc).__name__})"
        if price_range is not None:
            range_note = f"from {bar_count} D1 sessions"
        elif not range_note:
            range_note = "the daily store holds no bars for this symbol"

    result = survey(records, as_of=as_of, price_range=price_range)

    print(f"level store: {path}")
    print(f"  symbol {symbol}   as-of {as_of.isoformat()}")
    print(f"  total records in store      {result['total_records']:6d}")
    print(
        f"    drawable kinds            {result['total_records'] - result['other_kinds']:6d}"
        f"   hv_horizontal {sum(result['in_store'][name] for name in ('green', 'red')):d}"
        f" (green {result['in_store']['green']:d} / red {result['in_store']['red']:d})"
        f", cloud_flat {result['in_store']['cloud']:d}"
    )
    print(
        f"    other kinds               {result['other_kinds']:6d}   never drawn"
    )
    if price_range is None:
        print(f"\n  price-range filter SKIPPED - {range_note or 'no bars'}")
    else:
        print(
            f"\n  price range {price_range[0]:.2f} - {price_range[1]:.2f}  ({range_note})"
        )

    print(f"\n{'':<28}{'green':>7}{'red':>7}{'cloud':>7}{'total':>8}")
    print(_row("in store", result["in_store"]))
    print(_row("priced", result["priced"]))
    weak = {
        name: result["priced"][name] - result["after_strength"][name] for name in BUCKETS
    }
    print(
        _row(
            f"strength >= {result['min_strength']:.1f}",
            result["after_strength"],
            f"({sum(weak.values())} below, never drawn)",
        )
    )
    dropped = {
        name: result["after_strength"][name] - result["after_effective"][name]
        for name in BUCKETS
    }
    print(
        _row(
            f"effective on {as_of.isoformat()}",
            result["after_effective"],
            f"({sum(dropped.values())} not in force)",
        )
    )
    if price_range is not None:
        outside = {
            name: result["after_effective"][name] - result["before_budget"][name]
            for name in BUCKETS
        }
        print(
            _row("inside price range", result["before_budget"], f"({sum(outside.values())} off-chart)")
        )

    print()
    print(_row("BEFORE clutter budget", result["before_budget"]))
    print(_row("budget", result["budgets"]))
    print(_row("AFTER (drawn)", result["after_budget"]))
    print(_row("truncated", result["truncated"]))

    lost = sum(result["truncated"].values())
    print()
    if lost:
        print(
            f"  The budget CUTS {lost} level(s) this symbol's store holds and the"
            f" chart could otherwise draw."
        )
    else:
        print("  The budget does not bind on this symbol - nothing is truncated.")
    if not result["buckets_agree"]:
        print(
            "  WARNING: per-bucket and whole-set horizontal_levels() disagree"
            f" ({sum(result['after_budget'].values())} vs {result['drawn_total']});"
            " this tool's bucket split no longer matches the chart's."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
