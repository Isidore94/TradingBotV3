"""Today's news & econ for the Trade Mentor: the pasted brief, read back.

Reads the trader's pasted daily forecasts (market journal rows with origin
`external_forecast`) and turns the newest into one view for a session:

* the timed events for that day and the week after it (`econ_events`, a fixed
  parser - times never come from a model);
* a few "what to watch" lines: the night AI's verified summary when it was
  built from this same brief, else plain lines from the brief itself.

File reads only; call it off the Qt thread. Nothing here is fetched or written.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import econ_events
import forecast_brief

SCHEMA = "econ_brief_v1"

ORIGIN_NIGHT = "night"
ORIGIN_TODAY_BRIEF = "today_brief"
ORIGIN_LAST_BRIEF = "last_brief"
ORIGIN_NONE = "none"

ORIGIN_TEXT = {
    ORIGIN_NIGHT: "night AI summary of the last brief",
    ORIGIN_TODAY_BRIEF: "from today's brief",
    ORIGIN_LAST_BRIEF: "from the last brief",
    ORIGIN_NONE: "",
}

NO_BRIEF_TEXT = "No brief pasted."

#: How far past the session the "rest of the week" list reaches, in days.
WEEK_DAYS = 7

_MARKUP = re.compile(r":chatgpt-content-reference\{[^}]*\}|\*\*|__|`")


# ---------------------------------------------------------------------------
# the pasted briefs
# ---------------------------------------------------------------------------
def load_recent_forecasts(
    *, up_to: str = "", limit: int = 2, journal_dir: Path | None = None
) -> list[dict[str, str]]:
    """The newest pasted briefs, one per session, newest session first.

    `up_to` drops briefs about a later session. Read-only; an unreadable
    journal is "no brief", never an error.
    """
    try:
        import market_journal
        from evidence_ledger import EvidenceLedger

        kwargs: dict[str, Any] = {
            "stream": market_journal.STREAM,
            "schema": market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
        }
        if journal_dir is not None:
            kwargs["directory"] = Path(journal_dir)
        rows = market_journal.resolve_entries(EvidenceLedger(**kwargs).read().rows)
    except Exception:  # noqa: BLE001 - an unreadable journal is no brief
        logging.debug("Pasted briefs unreadable.", exc_info=True)
        return []
    return forecasts_from_rows(rows, up_to=up_to, limit=limit)


def forecasts_from_rows(
    rows: Iterable[Mapping[str, Any]], *, up_to: str = "", limit: int = 2
) -> list[dict[str, str]]:
    import market_journal

    by_session: dict[str, dict[str, str]] = {}
    for row in rows or ():
        if str(row.get("origin") or "") != market_journal.ORIGIN_EXTERNAL_FORECAST:
            continue
        text = str(row.get("text") or "")
        session = market_journal.session_of_entry(row)
        if not text.strip() or not session or (up_to and session > up_to):
            continue
        created = str(row.get("created_at") or "")
        seen = by_session.get(session)
        if seen is not None and seen["created_at"] >= created:
            continue
        by_session[session] = {
            "session": session,
            "text": text,
            "entry_id": str(row.get("entry_id") or ""),
            "created_at": created,
        }
    ordered = sorted(by_session.values(), key=lambda item: item["session"], reverse=True)
    return ordered[: max(1, int(limit))]


# ---------------------------------------------------------------------------
# the deterministic pack
# ---------------------------------------------------------------------------
def _event_rows(events, prefix: str) -> list[dict[str, str]]:
    rows = []
    for index, event in enumerate(events, start=1):
        row = econ_events.as_dict(event)
        row["id"] = f"{prefix}{index}"
        rows.append(row)
    return rows


def _plain(text: str, limit: int = 400) -> str:
    cleaned = re.sub(r"\s+", " ", _MARKUP.sub("", str(text or ""))).strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def build_pack(forecasts: list[Mapping[str, str]], *, target_session: str) -> dict[str, Any]:
    """Everything the brief says about `target_session` and the week after it.

    The newest brief about a session at or before the target is used; the one
    before it only when the newest names no event from the target on.
    """
    usable = [item for item in forecasts if str(item.get("session") or "") <= target_session]
    if not usable:
        return {}
    try:
        end = (date.fromisoformat(target_session) + timedelta(days=WEEK_DAYS)).isoformat()
    except ValueError:
        return {}
    chosen = usable[0]
    parsed = econ_events.parse(str(chosen.get("text") or ""))
    events = [event for event in parsed.events if target_session <= event.date <= end]
    if not events and len(usable) > 1:
        older = econ_events.parse(str(usable[1].get("text") or ""))
        older_events = [event for event in older.events if target_session <= event.date <= end]
        if older_events:
            chosen, parsed, events = usable[1], older, older_events
    text = str(chosen.get("text") or "")
    brief = forecast_brief.parse(text)
    turbulence_lines = [
        _plain(line, 200)
        for line in forecast_brief.normalize(text).splitlines()
        if re.search(r"turbulence\s*:", line, re.IGNORECASE)
    ]
    return {
        "schema": SCHEMA,
        "target_session": target_session,
        "brief_session": str(chosen.get("session") or ""),
        "brief_hash": brief.source_hash,
        "events_source": parsed.source,
        "unread_lines": parsed.unread_lines,
        "today": _event_rows([e for e in events if e.date == target_session], "t"),
        "week": _event_rows([e for e in events if e.date > target_session], "w"),
        "bottom_line": _plain(brief.bottom_line, 800),
        "ranked_signals": list(brief.ranked_signals),
        "turbulence_lines": turbulence_lines,
        "playbook_bullish": _plain(brief.playbook_bullish, 400),
        "playbook_bearish": _plain(brief.playbook_bearish, 400),
    }


def deterministic_lines(pack: Mapping[str, Any]) -> list[str]:
    """Plain lines straight from the brief, no model."""
    lines: list[str] = []
    signals = [str(item) for item in pack.get("ranked_signals") or () if str(item).strip()]
    if signals:
        lines.append("Watch, in order: " + " → ".join(signals).rstrip("."))
    bullish = str(pack.get("playbook_bullish") or "")
    if bullish:
        lines.append(_plain(bullish, 240))
    bearish = str(pack.get("playbook_bearish") or "")
    if bearish:
        lines.append(_plain(bearish, 240))
    return lines


# ---------------------------------------------------------------------------
# the night file
# ---------------------------------------------------------------------------
def night_dir(out_dir: Path | None = None) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    from project_paths import ECON_BRIEF_DIR

    return Path(ECON_BRIEF_DIR)


def read_night(target_session: str, *, out_dir: Path | None = None) -> dict[str, Any] | None:
    """The night's verified summary for `target_session`, or None."""
    path = night_dir(out_dir) / f"{target_session}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return None
    lines = payload.get("summary_lines")
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
        return None
    return payload


# ---------------------------------------------------------------------------
# the view the Mentor and the reminders read
# ---------------------------------------------------------------------------
def today_view(
    session: str,
    *,
    forecasts: list[Mapping[str, str]] | None = None,
    out_dir: Path | None = None,
    journal_dir: Path | None = None,
) -> dict[str, Any]:
    """Today's news & econ for `session`. File reads; never on the Qt thread."""
    if forecasts is None:
        forecasts = load_recent_forecasts(up_to=session, journal_dir=journal_dir)
    pack = build_pack(list(forecasts or ()), target_session=session) if forecasts else {}
    if not pack:
        return {
            "session": session,
            "origin": ORIGIN_NONE,
            "origin_text": "",
            "brief_session": "",
            "summary_lines": [],
            "today": [],
            "week": [],
            "unread_lines": 0,
            "note": NO_BRIEF_TEXT,
        }
    if pack["brief_session"] == session:
        origin, lines = ORIGIN_TODAY_BRIEF, deterministic_lines(pack)
    else:
        night = read_night(session, out_dir=out_dir)
        if night is not None and night.get("brief_hash") == pack["brief_hash"]:
            origin, lines = ORIGIN_NIGHT, [str(line) for line in night["summary_lines"]]
        else:
            origin, lines = ORIGIN_LAST_BRIEF, deterministic_lines(pack)
    return {
        "session": session,
        "origin": origin,
        "origin_text": ORIGIN_TEXT[origin],
        "brief_session": pack["brief_session"],
        "brief_hash": pack["brief_hash"],
        "summary_lines": lines,
        "today": pack["today"],
        "week": pack["week"],
        "unread_lines": int(pack.get("unread_lines") or 0),
        "note": "",
    }


__all__ = [
    "NO_BRIEF_TEXT",
    "ORIGIN_LAST_BRIEF",
    "ORIGIN_NIGHT",
    "ORIGIN_NONE",
    "ORIGIN_TODAY_BRIEF",
    "SCHEMA",
    "build_pack",
    "deterministic_lines",
    "forecasts_from_rows",
    "load_recent_forecasts",
    "read_night",
    "today_view",
]
