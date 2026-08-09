"""Measure how usable the stored D1 trendline actually is, per symbol.

A4 paints ``priority_trendline_candidate`` from the ai_state file. Whether
that is worth painting is an empirical question about one specific file on
one specific desk, and it cannot be answered from the code: it depends on how
many symbols the last scan promoted to priority candidates, how many of those
produced a trendline at all, and how fresh they are.

So this reports it. Run on the desk::

    .venv\\Scripts\\python.exe scripts\\d1_trendline_survey.py
    .venv\\Scripts\\python.exe scripts\\d1_trendline_survey.py --list 20

What each line means:

* **symbols in ai_state** - the scan's whole per-symbol map.
* **with a trendline record** - how many carry ``priority_trendline_candidate``
  or ``priority_trendline_break_candidate``. This is the coverage ceiling, and
  it is expected to be well under the total: the scan writes the record only
  for rows that reached priority-candidate status.
* **projectable** - has ``slope_log_per_bar`` AND a parseable ``lookback_end``.
  Without both there is no honest way to draw the line, and A4 draws nothing
  (plan.md sec 5: missing data is uncertainty, never confirmation).
* **fresh** - the symbol's ``last_trade_date`` is within
  ``chart_levels.TRENDLINE_MAX_AGE_DAYS`` of today. A trendline projects along
  its slope and goes wrong faster than a moving average.
* **paintable today** - projectable AND fresh. This is the number that decides
  whether the feature earns its place.

Read-only. Touches no store, writes nothing, contacts nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def _parse_date(value) -> date | None:
    text = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def survey(path: Path, *, today: date, max_age_days: int) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    symbols = payload.get("symbols") if isinstance(payload, dict) else {}
    rows = []
    total = 0
    for symbol, entry in (symbols or {}).items():
        total += 1
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("priority_trendline_candidate")
        kind = "candidate"
        if not isinstance(candidate, dict):
            candidate = entry.get("priority_trendline_break_candidate")
            kind = "break"
        if not isinstance(candidate, dict):
            continue
        scan_date = _parse_date(entry.get("last_trade_date"))
        age = (today - scan_date).days if scan_date else None
        rows.append(
            {
                "symbol": str(symbol).strip().upper(),
                "kind": kind,
                "type": str(candidate.get("type") or ""),
                "has_slope": candidate.get("slope_log_per_bar") is not None,
                "lookback_end": str(candidate.get("lookback_end") or ""),
                "has_anchor": _parse_date(candidate.get("lookback_end")) is not None,
                "touches": candidate.get("touch_count"),
                "age_days": age,
                "fresh": age is not None and 0 <= age <= max_age_days,
            }
        )
    for row in rows:
        row["projectable"] = bool(row["has_slope"] and row["has_anchor"])
        row["paintable"] = bool(row["projectable"] and row["fresh"])
    return {"total_symbols": total, "rows": rows}


def _percent(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:5.1f}%" if whole else "    -"


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from chart_levels import TRENDLINE_MAX_AGE_DAYS
    from project_paths import MASTER_AVWAP_AI_STATE_FILE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai-state", type=Path, default=Path(MASTER_AVWAP_AI_STATE_FILE))
    parser.add_argument("--max-age-days", type=int, default=TRENDLINE_MAX_AGE_DAYS)
    parser.add_argument("--list", type=int, default=0, metavar="N",
                        help="also print the first N paintable symbols")
    args = parser.parse_args(argv)

    if not args.ai_state.exists():
        print(f"no ai_state at {args.ai_state} - run a scan first", file=sys.stderr)
        return 2

    result = survey(args.ai_state, today=date.today(), max_age_days=args.max_age_days)
    rows = result["rows"]
    total = result["total_symbols"]
    with_record = len(rows)
    projectable = sum(1 for row in rows if row["projectable"])
    fresh = sum(1 for row in rows if row["fresh"])
    paintable = sum(1 for row in rows if row["paintable"])

    print(f"ai_state: {args.ai_state}")
    print(f"  symbols in ai_state      {total:6d}")
    print(f"  with a trendline record  {with_record:6d}  {_percent(with_record, total)} of symbols")
    print(f"  projectable              {projectable:6d}  {_percent(projectable, with_record)} of records")
    print(f"  fresh (<= {args.max_age_days}d)            {fresh:6d}  {_percent(fresh, with_record)} of records")
    print(f"  PAINTABLE TODAY          {paintable:6d}  {_percent(paintable, total)} of symbols")

    no_slope = sum(1 for row in rows if not row["has_slope"])
    if no_slope:
        print(f"\n  {no_slope} record(s) carry no slope_log_per_bar and are never painted.")
    if args.list:
        print(f"\n  first {args.list} paintable:")
        for row in [row for row in rows if row["paintable"]][: args.list]:
            print(
                f"    {row['symbol']:<8} {row['type']:<8} "
                f"touches={row['touches']} age={row['age_days']}d"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
