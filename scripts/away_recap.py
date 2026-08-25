"""What an AWAY day produced, assembled for the return — R1 amendment 2026-08-24.

The trader came back from a full AWAY day to **317 alerts waiting in the chart
review queue**, plus 128 hidden inside yesterday's range. Their rule: "Auto away
should NOT produce that much noise. In general it should just send nothing until
EOD, where it will show me what produced the best results intraday and also what
focus picks needed managing... Only auto desk should send 317 signals over the
course of the day."

So AWAY's return surface stops being a queue and becomes this: the day's best
output, ranked **exactly as the day already ranked it**, plus the Focus picks
that needed a decision.

**This module ranks nothing.** It has no detector, no score, no model and no
writer — a test asserts it makes no write call at all. Every ordering here came
from something that already produced it: `autopilot_today.txt`'s numbered best
swings are the AWAY push's own ranking, the classified D1 events are the Alert
Center's own classification, the staged picks are what the machine already
staged. The recap's whole job is to put them on one page and say where each
came from, so a reader can tell an ordering the desk produced from one this
page invented — because this page invents none.

Two honesty rules do the load-bearing work:

* **A source that could not be read is NAMED**, not silently empty. A recap
  showing nothing because a file would not open must never look like a quiet
  day.
* **A line that cannot be parsed is kept and marked.** Dropping it would
  quietly narrow the day, and the trader would have no way to know.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

#: Schema NAME for the assembled payload (ground rule 5). It is a VIEW - it is
#: never persisted - but naming it keeps a future reader honest about which
#: shape they are looking at.
RECAP_SCHEMA = "away_day_recap_v1"

#: `1. FTAI (SHORT) | Favorite | AVWAP band bounce @ VWAP to LOWER_1`
_SWING_LINE = re.compile(r"^\s*(?:\d+\.\s*)?([A-Z][A-Z0-9.\-]{0,9})\s*\(([A-Z]+)\)")


def _rows_from_swings(lines: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(lines or (), start=1):
        text = str(raw or "").strip()
        if not text:
            continue
        match = _SWING_LINE.match(text)
        if match is None:
            # Kept, and marked. A line this page cannot read is still a line the
            # day produced, and dropping it would narrow the record silently.
            rows.append(
                {"rank": index, "symbol": "", "side": "", "text": text, "unparsed": True}
            )
            continue
        rows.append(
            {
                "rank": index,
                "symbol": match.group(1),
                "side": match.group(2),
                "text": text,
                "unparsed": False,
            }
        )
    return rows


def _alert_rows(alerts: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The day's alerts, in the order the desk produced them.

    No re-ranking. The tier travels because the desk assigned it; this page
    does not compute one and does not sort on it - the order is the order the
    day happened, which is the only ordering nobody has to defend.
    """
    rows: list[dict[str, Any]] = []
    for alert in alerts or ():
        rows.append(
            {
                "symbol": str(alert.get("symbol") or "").upper(),
                "side": str(alert.get("side") or "").upper(),
                "tier": str(alert.get("tier") or ""),
                "trigger": str(alert.get("trigger") or ""),
                "time_text": str(alert.get("time_text") or ""),
                "is_d1": bool(alert.get("is_d1")),
            }
        )
    return rows


def _side_rows(picks: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for side in ("long", "short"):
        for symbol in (picks or {}).get(side) or []:
            rows.append({"symbol": str(symbol).upper(), "side": side})
    return rows


def build_recap(
    *,
    session_date: str,
    alerts: Iterable[Mapping[str, Any]] = (),
    staged_picks: Mapping[str, Any] | None = None,
    digest_swings: Iterable[str] = (),
    focus_picks: Mapping[str, Any] | None = None,
    unavailable: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble one AWAY day's return view. Reads only; writes nothing."""
    swing_rows = _rows_from_swings(digest_swings)
    alert_rows = _alert_rows(alerts)
    staged_rows = _side_rows(staged_picks or {})
    focus_rows = _side_rows(focus_picks or {})
    missing = {str(name): str(reason) for name, reason in (unavailable or {}).items()}

    counts = {
        "alerts": len(alert_rows),
        "staged": len(staged_rows),
        "swings": len(swing_rows),
        "focus": len(focus_rows),
    }
    return {
        "schema": RECAP_SCHEMA,
        "session_date": str(session_date or ""),
        "best_swings": swing_rows,
        "classified_alerts": alert_rows,
        "staged_picks": staged_rows,
        "focus_to_manage": focus_rows,
        "counts": counts,
        "unavailable": missing,
        # Where each section's ORDER came from. The point of stating it is that
        # a reader can tell an ordering the desk produced from one this page
        # invented - and this page invents none.
        "provenance": {
            "best_swings": (
                "autopilot_today.txt's numbered BEST SWING TRADES, in the order "
                "the AWAY push already ranked them"
            ),
            "classified_alerts": (
                "the Alert Center's own backing list and its own tier/D1 "
                "classification, in the order the day produced them"
            ),
            "staged_picks": (
                "the picks the machine staged and deliberately did not adopt "
                "(AWAY stages, never adopts)"
            ),
            "focus_to_manage": "the current Focus lists, read as they stand",
        },
        "summary": _summary(counts, missing, session_date),
    }


def _summary(counts: Mapping[str, int], missing: Mapping[str, str], session_date: str) -> str:
    parts: list[str] = []
    if any(counts.values()):
        parts.append(
            f"{counts['swings']} ranked swing(s), {counts['alerts']} alert(s), "
            f"{counts['staged']} staged pick(s) and {counts['focus']} Focus name(s) "
            f"for {session_date}."
        )
    else:
        parts.append(f"Nothing was recorded for {session_date}.")
    if missing:
        # Named, never silent: a page that is empty because a file would not
        # open must not read as a quiet day.
        named = ", ".join(f"{name} ({reason})" for name, reason in sorted(missing.items()))
        parts.append(
            f"{len(missing)} source(s) could not be read, so this recap is "
            f"INCOMPLETE rather than empty: {named}."
        )
    return " ".join(parts)


def digest_swing_lines(digest_text: str) -> list[str]:
    """The numbered BEST SWING TRADES block out of `autopilot_today.txt`.

    Parsed rather than re-derived: the ranking is the AWAY push's, and
    recomputing it here would be a second ranking wearing the same name.
    """
    lines: list[str] = []
    in_block = False
    for raw in str(digest_text or "").splitlines():
        text = raw.strip()
        if text.startswith("== BEST SWING TRADES"):
            in_block = True
            continue
        if in_block:
            if text.startswith("=="):
                break
            if not text or text.startswith("TV paste:") or text.startswith("Swing data:"):
                continue
            lines.append(text)
    return lines
