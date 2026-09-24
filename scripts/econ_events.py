"""Upcoming econ events and their ET clock times, read from the pasted brief.

A fixed parser over the trader's daily ChatGPT brief. Times feed alarms, so
they come from here and never from a model: a misread time is a wrong alarm.

Rules:

* A sentence becomes an event only when it names a known release (JOLTS, PCE,
  a Treasury auction, "<Company> reports ...") or states an explicit clock time,
  AND it can be dated: "today", "this morning", "Tomorrow" (next trading day),
  a weekday, "Tuesday, Sept. 29", "September 30", or the brief's own
  intraday-calendar section. An undated mention ("the most recent CPI") is
  history, not an event.
* Times: "8:30 a.m. ET", "10 a.m.", "1:00 p.m. ET", "noon". A bare "10:00"
  takes the meridiem of the nearest earlier explicit time in its sentence;
  otherwise it is unknown. A time in another zone is unknown.
* Headings are never events ("## Fresh 8:30 data" names a past print). Events
  before the brief date, or on it at or before the brief's "As of" time, are past.
* The same event said twice is one event, keeping the time if either said it.

Since 2026-09-24 every brief ENDS with a machine-readable block, and when it is
present it is the ONLY source of events (prose times are ignored):

    NEXT 7 DAYS — ECONOMIC CALENDAR
    2026-09-25 | 05:30 PT / 08:30 ET | Durable goods orders (Aug)
    2026-09-30 | TIME TBD | ADP employment

The ET time is used; PT + 3 h must equal it or the event keeps no time (never
a guess which one is right). A malformed line is skipped whole and counted.
Calendar rows are never merged with each other.

Pure: no I/O, no clock, no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import forecast_brief


@dataclass(frozen=True)
class EconEvent:
    """One upcoming release. `time_et` is "HH:MM" (24h, ET) or "" if unknown."""

    date: str
    time_et: str
    label: str
    source_line: str
    kind: str = ""


@dataclass(frozen=True)
class BriefEvents:
    brief_date: str = ""
    as_of: str = ""
    events: tuple[EconEvent, ...] = field(default_factory=tuple)
    #: "calendar" (the machine-readable block), "prose" (the fallback) or "".
    source: str = ""
    #: Non-empty lines in the calendar block that could not be read whole.
    unread_lines: int = 0


#: Warnings are only for events at or after 07:00 on the trader's own clock
#: (Pacific; trader, 2026-09-24), converted on each event's date.
ALARM_EARLIEST_LOCAL = time(7, 0)
ALARM_ZONE = ZoneInfo("America/Los_Angeles")
EASTERN = ZoneInfo("America/New_York")

SOURCE_CALENDAR = "calendar"
SOURCE_PROSE = "prose"


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_ALT = "|".join(_WEEKDAYS)

#: "Tuesday, September 29", "Sept 29", "September 30" (abbreviation dots are
#: removed before matching). The day may not run into a year: "July 2007".
_EXPLICIT_DATE = re.compile(
    rf"\b(?:(?:{_WEEKDAY_ALT}),?\s+)?({_MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!\d)",
    re.IGNORECASE,
)
_RELATIVE_DATE = re.compile(
    r"\b(tomorrow|today(?:'s|’s)?|tonight|this\s+(?:morning|afternoon)|yesterday(?:'s|’s)?)\b",
    re.IGNORECASE,
)
_WEEKDAY_DATE = re.compile(rf"\b({_WEEKDAY_ALT})\b", re.IGNORECASE)

#: Explicit clock times, after "a.m." / "p.m." were normalised to am / pm.
_EXPLICIT_TIME = re.compile(
    r"(?<![\d.:$])(\d{1,2})(?::([0-5]\d))?\s*(am|pm)\b(?:\s*(ET|EST|EDT|CT|CST|CDT|PT|PST|PDT|MT)\b)?",
    re.IGNORECASE,
)
_NOON = re.compile(r"\bnoon\b(?:\s*(ET|CT|PT|MT)\b)?", re.IGNORECASE)
_BARE_TIME = re.compile(
    r"(?<![\d.:$])(\d{1,2}):([0-5]\d)(?![\d%])(?!\s*(?:am|pm)\b)(?:\s*(ET|EST|EDT|CT|CST|CDT|PT|PST|PDT|MT)\b)?",
    re.IGNORECASE,
)
_ET_ZONES = {"", "et", "est", "edt"}

_AS_OF = re.compile(r"^\s*as\s+of\b", re.IGNORECASE)

#: Known releases: (kind, pattern, label). A "{n}" in the label takes the
#: pattern's first group (the auction's tenor).
_RELEASES: tuple[tuple[str, re.Pattern[str], str], ...] = tuple(
    (kind, re.compile(pattern, re.IGNORECASE), label)
    for kind, pattern, label in (
        ("jolts", r"\bJOLTS\b", "JOLTS"),
        ("pce", r"\bPCE\b", "PCE"),
        ("gdp", r"\bGDP\b", "GDP"),
        ("adp", r"\bADP\b", "ADP employment"),
        ("payrolls", r"\b(?:employment\s+report|jobs\s+report|nonfarm\s+payrolls|payrolls)\b", "employment report"),
        ("cpi", r"\bCPI\b", "CPI"),
        ("ppi", r"\bPPI\b", "PPI"),
        ("retail_sales", r"\bretail\s+sales\b", "retail sales"),
        ("durable_goods", r"\bdurable[-\s]goods\b", "durable goods orders"),
        ("michigan", r"\bMichigan\b", "Michigan sentiment"),
        ("new_home_sales", r"\bnew[-\s]home\s+sales\b", "new-home sales"),
        ("existing_home_sales", r"\bexisting[-\s]home\s+sales\b", "existing-home sales"),
        ("pending_home_sales", r"\bpending[-\s]home[-\s]sales\b", "pending home sales"),
        ("housing_starts", r"\bhousing\s+starts\b", "housing starts"),
        ("claims", r"\b(?:jobless|unemployment)\s+claims\b", "jobless claims"),
        ("ism", r"\bISM\b", "ISM"),
        ("pmi", r"\bPMI\b", "PMI"),
        ("consumer_confidence", r"\bconsumer\s+confidence\b", "consumer confidence"),
        ("factory_orders", r"\bfactory\s+orders\b", "factory orders"),
        ("fomc", r"\b(?:FOMC|Fed\s+(?:decision|minutes)|rate\s+decision)\b", "Fed decision"),
        ("powell", r"\bPowell\b", "Powell speaks"),
        ("boj", r"\b(?:BOJ|BoJ|Bank\s+of\s+Japan)\b", "Bank of Japan decision"),
        ("ecb", r"\bECB\b", "ECB decision"),
        ("industrial_production", r"\bindustrial\s+production\b", "industrial production"),
        ("auction", r"\b(?:(\d{1,2})-year\s+)?(?:Treasury\s+)?(?:note\s+|bond\s+)?auction\b", "Treasury auction"),
    )
)
#: "<Company> reports fiscal Q4 ..." - earnings, never a news source reporting.
_EARNINGS = re.compile(
    r"\b([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*)?)\s+reports\s+"
    r"(?=fiscal|Q[1-4]|earnings|results|quarterly|first|second|third|fourth|after|before)"
)
_NOT_COMPANIES = {"reuters", "bloomberg", "cnbc", "wsj", "treasury", "the fed", "fed", "it", "he", "she"}
_AFTER_CLOSE = re.compile(r"\bafter\s+(?:today'?s\s+|the\s+|tomorrow'?s\s+)?(?:close|bell)\b", re.IGNORECASE)
_BEFORE_OPEN = re.compile(r"\bbefore\s+(?:today'?s\s+|the\s+|tomorrow'?s\s+)?(?:open|bell)\b", re.IGNORECASE)

_CALENDAR_HEADING = re.compile(
    r"^next\s+7\s+days\s*[—–-]+\s*economic\s+calendar\s*:?$", re.IGNORECASE
)
_CALENDAR_ROW = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)$")
_CALENDAR_TBD = re.compile(r"^time\s+tbd$", re.IGNORECASE)
_CALENDAR_TIMES = re.compile(
    r"^(\d{1,2}):([0-5]\d)\s*(PT|ET)\s*/\s*(\d{1,2}):([0-5]\d)\s*(PT|ET)$", re.IGNORECASE
)
_TABLE_FILLER = re.compile(r"^[\s|:\-]+$|^\|?\s*date\s*\|\s*time", re.IGNORECASE)

_INTRADAY_SECTION = re.compile(r"\b(?:today|intraday)\b", re.IGNORECASE)
_MARKUP = re.compile(r":chatgpt-content-reference\{[^}]*\}|\*\*|__|`")


def parse(text: str) -> BriefEvents:
    """The brief's date, its "As of" time, and its upcoming events."""
    body = forecast_brief.normalize(text)
    brief_iso = forecast_brief.parse(body).title_date
    brief_day = _as_date(brief_iso)
    as_of = ""
    found: list[EconEvent] = []
    section = ""
    for raw in body.splitlines():
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.*)$", raw)
        if heading is not None:
            section = heading.group(1)
            continue
        line = _clean(raw)
        if not line:
            continue
        if _AS_OF.match(line):
            times = _times(_protect(line))
            if times and times[0][2]:
                as_of = times[0][2]
            continue
        if brief_day is None:
            continue
        in_today_section = bool(_INTRADAY_SECTION.search(section))
        for sentence in _sentences(line):
            found.extend(_sentence_events(sentence, brief_day, in_today_section))
    source = SOURCE_PROSE
    calendar = _calendar_block(body)
    unread = 0
    if calendar is not None:
        found, unread = calendar
        source = SOURCE_CALENDAR
    # Prose says one event twice; each calendar row is its own event.
    events = list(found) if source == SOURCE_CALENDAR else _dedup(found)
    if brief_day is not None:
        events = [
            event
            for event in events
            if event.date > brief_iso
            or (event.date == brief_iso and not (event.time_et and as_of and event.time_et <= as_of))
        ]
    events.sort(key=lambda e: (e.date, e.time_et or "99:99", e.label))
    return BriefEvents(
        brief_date=brief_iso,
        as_of=as_of,
        events=tuple(events),
        source=source if brief_iso or calendar is not None else "",
        unread_lines=unread,
    )


def parse_events(text: str) -> tuple[EconEvent, ...]:
    return parse(text).events


def alarm_allowed(day: str, time_et: str) -> bool:
    """Is `time_et` on `day` at or after 07:00 Pacific that day? Unknown -> False."""
    try:
        base = date.fromisoformat(str(day))
        moment = datetime(
            base.year, base.month, base.day, int(time_et[:2]), int(time_et[3:5]), tzinfo=EASTERN
        )
    except (TypeError, ValueError):
        return False
    return moment.astimezone(ALARM_ZONE).time() >= ALARM_EARLIEST_LOCAL


def alarm_events(events, *, day: str) -> tuple[EconEvent, ...]:
    """The events on `day` with a known ET time at or after 07:00 Pacific."""
    return tuple(
        event
        for event in events
        if event.date == day and event.time_et and alarm_allowed(day, event.time_et)
    )


def as_dict(event: EconEvent) -> dict[str, str]:
    return {
        "date": event.date,
        "time_et": event.time_et,
        "label": event.label,
        "source_line": event.source_line,
        "kind": event.kind,
    }


# ---------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------
def _calendar_block(body: str) -> tuple[list[EconEvent], int] | None:
    """Events from the NEXT 7 DAYS block, and how many lines were not read.

    None when the brief has no such block. The block runs to the next
    Markdown heading or the end of the brief.
    """
    lines = body.splitlines()
    start = None
    for index, raw in enumerate(lines):
        text = _clean(re.sub(r"^\s{0,3}#{1,6}\s+", "", raw))
        if _CALENDAR_HEADING.match(text):
            start = index + 1
    if start is None:
        return None
    events: list[EconEvent] = []
    unread = 0
    for raw in lines[start:]:
        if re.match(r"^\s{0,3}#{1,6}\s+", raw):
            break
        text = raw.strip().strip("`").strip()
        if not text or text.startswith("```") or _TABLE_FILLER.match(text):
            continue
        text = re.sub(r"^(?:[-*+]\s+)", "", text).strip().strip("|").strip()
        event = _calendar_row(text)
        if event is None:
            unread += 1
            continue
        events.append(event)
    return events, unread


def _calendar_row(text: str) -> EconEvent | None:
    match = _CALENDAR_ROW.match(text)
    if match is None:
        return None
    day, when, label = match.group(1), match.group(2).strip(), match.group(3).strip()
    if _as_date(day) is None or not label:
        return None
    if _CALENDAR_TBD.match(when):
        time_et = ""
    else:
        times = _CALENDAR_TIMES.match(when)
        if times is None:
            return None
        first = (int(times.group(1)), int(times.group(2)), times.group(3).upper())
        second = (int(times.group(4)), int(times.group(5)), times.group(6).upper())
        zones = {first[2]: first, second[2]: second}
        if set(zones) != {"PT", "ET"} or any(hour > 23 for hour, _m, _z in (first, second)):
            return None
        pt_minutes = zones["PT"][0] * 60 + zones["PT"][1]
        et_minutes = zones["ET"][0] * 60 + zones["ET"][1]
        # PT + 3 h must be ET; a mismatch keeps the event with no time.
        time_et = _hhmm(et_minutes) if pt_minutes + 180 == et_minutes else ""
    hits = _hits(_protect(label))
    kind = hits[0][2] if hits else "other"
    return EconEvent(day, time_et, label, text, kind)


def _as_date(iso: str) -> date | None:
    try:
        return date.fromisoformat(str(iso or ""))
    except ValueError:
        return None


def _clean(line: str) -> str:
    text = _MARKUP.sub("", line)
    text = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _protect(text: str) -> str:
    """Drop dots that are not sentence ends: a.m., p.m., Sept., U.S."""
    text = re.sub(r"\b([ap])\.\s?m\.", lambda m: m.group(1).lower() + "m", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b({_MONTH_ALT})\.", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bU\.S\.", "US", text)
    return text


def _sentences(line: str) -> list[str]:
    protected = _protect(line)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)
    return [part.strip() for part in parts if part.strip()]


def _times(sentence: str) -> list[tuple[int, int, str]]:
    """(start, end, "HH:MM" or "") for every clock time in the sentence."""
    explicit: list[tuple[int, int, int, bool]] = []  # start, end, minutes, known
    for match in _EXPLICIT_TIME.finditer(sentence):
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        if not 1 <= hour <= 12:
            continue
        hour = hour % 12 + (12 if match.group(3).lower() == "pm" else 0)
        zone = (match.group(4) or "").lower()
        explicit.append((match.start(), match.end(), hour * 60 + minute, zone in _ET_ZONES))
    for match in _NOON.finditer(sentence):
        zone = (match.group(1) or "").lower()
        explicit.append((match.start(), match.end(), 12 * 60, zone in _ET_ZONES))
    out: list[tuple[int, int, str]] = [
        (start, end, _hhmm(minutes) if known else "") for start, end, minutes, known in explicit
    ]
    taken = [(start, end) for start, end, _m, _k in explicit]
    for match in _BARE_TIME.finditer(sentence):
        if any(start <= match.start() < end for start, end in taken):
            continue
        hour, minute = int(match.group(1)), int(match.group(2))
        zone = (match.group(3) or "").lower()
        value = ""
        if zone in _ET_ZONES and hour <= 23:
            if hour >= 13:
                value = _hhmm(hour * 60 + minute)
            else:
                earlier = [e for e in explicit if e[1] <= match.start() and e[3]]
                if earlier and 1 <= hour <= 12:
                    anchor = max(earlier, key=lambda e: e[1])
                    pm = anchor[2] >= 12 * 60
                    minutes = (hour % 12 + (12 if pm else 0)) * 60 + minute
                    if minutes >= anchor[2]:
                        value = _hhmm(minutes)
        out.append((match.start(), match.end(), value))
    out.sort()
    return out


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _dates(sentence: str, brief: date) -> list[tuple[int, int, date]]:
    out: list[tuple[int, int, date]] = []
    spans: list[tuple[int, int]] = []
    for match in _EXPLICIT_DATE.finditer(sentence):
        month = _MONTHS[match.group(1).casefold()]
        day = int(match.group(2))
        year = brief.year + (1 if brief.month == 12 and month == 1 else 0)
        try:
            value = date(year, month, day)
        except ValueError:
            continue
        out.append((match.start(), match.end(), value))
        spans.append((match.start(), match.end()))
    for match in _RELATIVE_DATE.finditer(sentence):
        word = match.group(1).casefold()
        if word.startswith("tomorrow"):
            value = _next_session(brief)
        elif word.startswith("yesterday"):
            value = brief - timedelta(days=1)
        else:
            value = brief
        out.append((match.start(), match.end(), value))
    for match in _WEEKDAY_DATE.finditer(sentence):
        if any(start <= match.start() < end for start, end in spans):
            continue
        target = _WEEKDAYS.index(match.group(1).casefold())
        ahead = (target - brief.weekday()) % 7
        out.append((match.start(), match.end(), brief + timedelta(days=ahead or 7)))
    out.sort()
    return out


def _next_session(day: date) -> date:
    try:
        from market_calendar import next_session

        return next_session(day)
    except Exception:  # noqa: BLE001 - outside the calendar: the next weekday
        step = day + timedelta(days=1)
        while step.weekday() >= 5:
            step += timedelta(days=1)
        return step


def _hits(sentence: str) -> list[tuple[int, int, str, str]]:
    """(start, end, kind, label) for every known release in the sentence."""
    hits: list[tuple[int, int, str, str]] = []
    for kind, pattern, label in _RELEASES:
        for match in pattern.finditer(sentence):
            text = label
            if kind == "auction" and match.group(1):
                text = f"{match.group(1)}-year Treasury auction"
            month = _month_before(sentence, match.start())
            if month and kind not in {"auction", "fomc", "powell"}:
                text = f"{month} {text}"
            hits.append((match.start(), match.end(), kind, text))
    for match in _EARNINGS.finditer(sentence):
        name = match.group(1).strip()
        if name.casefold() in _NOT_COMPANIES or name.casefold().startswith("the "):
            continue
        text = f"{name} earnings"
        if _AFTER_CLOSE.search(sentence):
            text += ", after the close"
        elif _BEFORE_OPEN.search(sentence):
            text += ", before the open"
        hits.append((match.start(), match.end(), "earnings:" + name.casefold(), text))
    hits.sort()
    kept: list[tuple[int, int, str, str]] = []
    for hit in hits:
        if kept and hit[0] < kept[-1][1]:
            continue  # overlapping mentions of one thing
        kept.append(hit)
    return kept


_MONTH_QUALIFIER = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES) + r")\s+(?:(?:final|preliminary|flash)\s+)?$", re.IGNORECASE
)


def _month_before(sentence: str, start: int) -> str:
    """"August" in "August JOLTS" - a month name directly before the release."""
    match = _MONTH_QUALIFIER.search(sentence[max(0, start - 30):start])
    return match.group(1).capitalize() if match else ""


def _nearest(items, start: int, end: int, lower: int, upper: int):
    """First item after [start, end) before `upper`, else the last before `start` after `lower`."""
    after = [item for item in items if item[0] >= end and item[0] < upper]
    if after:
        return after[0]
    before = [item for item in items if item[1] <= start and item[0] >= lower]
    return before[-1] if before else None


def _sentence_events(sentence: str, brief: date, in_today_section: bool) -> list[EconEvent]:
    hits = _hits(sentence)
    times = _times(sentence)
    dates = _dates(sentence, brief)
    source = sentence
    out: list[EconEvent] = []
    if not hits:
        # An unknown event still counts when the brief gives it a known ET time.
        known = [t for t in times if t[2]]
        if not known or len(known) > 1:
            return []
        day = dates[0][2] if dates else brief
        label = _short(sentence)
        return [EconEvent(day.isoformat(), known[0][2], label, source, "other")]
    for index, (start, end, kind, label) in enumerate(hits):
        lower = hits[index - 1][1] if index else 0
        upper = hits[index + 1][0] if index + 1 < len(hits) else len(sentence)
        timed = _nearest(times, start, end, lower, upper)
        dated = _nearest(dates, start, end, lower, upper)
        if dated is None:
            # "Tomorrow brings A ... and B": a date earlier in the sentence
            # covers every release after it; a later one belongs to a later release.
            earlier = [item for item in dates if item[1] <= start]
            dated = earlier[-1] if earlier else None
        if dated is not None:
            day = dated[2]
        elif timed is not None or in_today_section:
            day = brief
        else:
            continue  # undated: history, not an event
        out.append(EconEvent(day.isoformat(), timed[2] if timed else "", label, source, kind))
    return out


def _short(sentence: str, limit: int = 70) -> str:
    text = re.sub(r"^(?:At|From|By)\s+[^,]{0,30},\s*", "", sentence).rstrip(". ")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _dedup(events: list[EconEvent]) -> list[EconEvent]:
    out: list[EconEvent] = []
    for event in events:
        for index, seen in enumerate(out):
            if seen.date != event.date or seen.kind != event.kind or event.kind == "other":
                continue
            if seen.time_et and event.time_et and seen.time_et != event.time_et:
                continue
            if event.kind == "auction" and _tenor(seen) and _tenor(event) and _tenor(seen) != _tenor(event):
                continue
            label = seen.label
            if len(event.label) > len(label) and (event.kind == "auction" or not seen.time_et):
                label = event.label
            out[index] = EconEvent(
                seen.date,
                seen.time_et or event.time_et,
                label,
                seen.source_line if seen.time_et or not event.time_et else event.source_line,
                seen.kind,
            )
            break
        else:
            out.append(event)
    return out


def _tenor(event: EconEvent) -> str:
    match = re.match(r"(\d{1,2})-year", event.label)
    return match.group(1) if match else ""


__all__ = ["BriefEvents", "EconEvent", "alarm_events", "as_dict", "parse", "parse_events"]
