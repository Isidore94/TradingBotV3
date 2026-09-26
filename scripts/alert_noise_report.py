"""Alert noise report (B6, goal 9): shown vs hidden vs acted on, per day. Read-only.

`python scripts/alert_noise_report.py --day YYYY-MM-DD [--days N]` prints, per
session, from the review events: shown (chart impressions and distinct names),
hidden_by_show (NAMES the Show filter held back: the panel writes one row per
symbol/side/day on the first hide, so this counts hidden names, never hidden
alerts), skip, remove_today,
acted on (the trader's take actions - `review_learning.TAKE_ACTIONS` plus a
Focus/favourite toggle turned on; claims are `like_advance`), watch_fired and
acted share = acted / shown; plus the Best-right-now hit rate
(`best_now_outcomes`). A day before the first `hidden_by_show` row anywhere in
the store says the hidden count is unmeasured, never zero.

`day_line` is the Day Review "Alerts:" truth line, built on its worker.
Nothing here writes, ranks, filters or alerts.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from typing import Any, Callable, Iterable, Mapping, Sequence

HIDDEN = "hidden_by_show"
COUNTED = ("skip", "remove_today", "watch_fired")
NO_EVENTS = "Alerts: no review events for this day"


def _is_acted(row: Mapping[str, Any]) -> bool:
    from review_learning import TAKE_ACTIONS, TOGGLE_TAKE_ACTIONS

    action = str(row.get("action") or "")
    if action in TAKE_ACTIONS:
        return True
    detail = row.get("detail")
    return action in TOGGLE_TAKE_ACTIONS and isinstance(detail, Mapping) and bool(detail.get("on"))


def hidden_measured_since(rows: Iterable[Mapping[str, Any]]) -> str | None:
    """The first trade date with any `hidden_by_show` row, or None."""
    days = [str(r.get("trade_date") or "")[:10] for r in rows if r.get("action") == HIDDEN]
    days = [day for day in days if day]
    return min(days) if days else None


def day_counts(rows: Iterable[Mapping[str, Any]], day: str, *, hidden_since: str | None) -> dict[str, Any]:
    """One session's counts. `hidden_by_show` is None (unmeasured) before `hidden_since`."""
    todays = [r for r in rows if str(r.get("trade_date") or "")[:10] == day]
    shown = [r for r in todays if r.get("action") == "shown"]
    hidden = [r for r in todays if r.get("action") == HIDDEN]
    acted = [r for r in todays if _is_acted(r)]
    measured_hidden = bool(hidden) or (hidden_since is not None and day >= hidden_since)
    counts: dict[str, Any] = {
        "day": day,
        "events": len(todays),
        "shown": len(shown),
        "shown_names": len({str(r.get("symbol") or "") for r in shown}),
        "hidden_by_show": (
            len({(str(r.get("symbol") or ""), str(r.get("side") or "")) for r in hidden})
            if measured_hidden else None
        ),
        "hidden_names": len({str(r.get("symbol") or "") for r in hidden}) if measured_hidden else None,
        "acted": len(acted),
        "acted_names": len({str(r.get("symbol") or "") for r in acted}),
        "acted_share": (len(acted) / len(shown)) if shown else None,
    }
    for action in COUNTED:
        counts[action] = sum(1 for r in todays if r.get("action") == action)
    return counts


def _pct(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value * 100:.0f}%"


def best_now_text(summary: Mapping[str, Any] | None) -> str:
    """"Best-right-now hit rate R% (n)"; unmeasured / no data said in words."""
    summary = summary or {}
    if not summary.get("logged"):
        return "Best-right-now: no data yet"
    if not summary.get("graded"):
        return f"Best-right-now hit rate unmeasured ({summary['logged']} logged, {summary.get('pending', 0)} pending)"
    return f"Best-right-now hit rate {_pct(summary.get('hit_rate'))} ({summary['graded']})"


def day_line(counts: Mapping[str, Any], best: Mapping[str, Any] | None) -> str:
    """The Day Review line."""
    if not counts.get("events"):
        return NO_EVENTS
    hidden = counts.get("hidden_by_show")
    hidden_text = "hidden names unmeasured" if hidden is None else f"{hidden} names hidden by Show"
    return (
        f"Alerts: {counts['shown']} shown, {hidden_text}, {counts['acted']} acted on "
        f"({_pct(counts.get('acted_share'))}), {best_now_text(best)}"
    )


def report(
    days: Sequence[str],
    events: Sequence[Mapping[str, Any]],
    best_records: Sequence[Mapping[str, Any]],
    *,
    bars_reader: Callable[[str], Any] | None = None,
) -> list[dict[str, Any]]:
    """Counts plus the Best-right-now summary for each day."""
    import best_now_outcomes

    since = hidden_measured_since(events)
    out = []
    for day in days:
        counts = day_counts(events, day, hidden_since=since)
        one = date.fromisoformat(day)
        best = best_now_outcomes.summarize(best_records, start=one, end=one, bars_reader=bars_reader)
        best.pop("rows", None)
        counts["best_now"] = best
        counts["line"] = day_line(counts, best)
        out.append(counts)
    return out


def build_day_line(day: str, *, events=None, best_records=None, bars_reader=None) -> str:
    """The Day Review "Alerts:" line for one session (worker thread; reads the stores)."""
    import best_now_outcomes
    import review_events

    if events is None:
        events = review_events.load_review_events()
    if best_records is None:
        best_records = best_now_outcomes.load_records()
    return report([day], events, best_records, bars_reader=bars_reader)[0]["line"]


def _sessions(day: str, count: int) -> list[str]:
    import walkaway_day

    return [*walkaway_day.earlier_sessions(day, count=max(0, count - 1)), day]


def _lines(row: Mapping[str, Any]) -> list[str]:
    hidden = row.get("hidden_by_show")
    return [
        f"{row['day']}:",
        f"  shown {row['shown']} ({row['shown_names']} names), "
        f"hidden_by_show {'unmeasured' if hidden is None else f'{hidden} names'}, "
        f"skip {row['skip']}, remove_today {row['remove_today']}",
        f"  acted on {row['acted']} ({row['acted_names']} names), acted share "
        f"{_pct(row.get('acted_share'))}, watch_fired {row['watch_fired']}",
        f"  {best_now_text(row.get('best_now'))}",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alert noise: shown / hidden / acted on per day")
    parser.add_argument("--day", required=True, help="last session (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=1, help="sessions ending at --day")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)
    date.fromisoformat(args.day)
    import best_now_outcomes
    import review_events

    rows = report(
        _sessions(args.day, max(1, args.days)),
        review_events.load_review_events(),
        best_now_outcomes.load_records(),
    )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
    else:
        for row in rows:
            print("\n".join(_lines(row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
