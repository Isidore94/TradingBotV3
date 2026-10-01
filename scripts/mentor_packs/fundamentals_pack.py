"""Fundamentals pack (P15b): the morning brief the trader pastes, read-only, as citable rows.

The brief is pasted into the Market Journal (desk "Paste daily forecast", or ``/paste`` in the
app) and stored verbatim as a ``market_journal_entry_v1`` row with origin ``external_forecast``;
a re-paste for the same session supersedes the earlier one. This pack reads the latest
non-superseded brief for the session asked about (else the most recent earlier one, labelled
"last brief: <date>") and turns it into rows a reply can cite:

- ``fund:<session>:asof`` when it was pasted, its source model and claimed write time;
- ``fund:<session>:bottom:<n>`` the bottom line (the ``Bottom line`` heading; a brief without one
  falls back, by heading only, to ``Why it matters`` / ``What matters today``, and says which);
- ``fund:<session>:signal:<n>`` the ranked signals (the bold arrow line);
- ``fund:<session>:bull:<n>`` / ``:bear:<n>`` the playbook's two conditions when the brief bolds them,
  ``fund:<session>:play:<n>`` the bullets under a playbook heading (``Intraday playbook``, ``How it may
  trade``, ``Intraday implications``), ``fund:<session>:watch:<n>`` the ``What to watch`` bullets;
- ``fund:<session>:turb:<n>`` turbulence lines and outlook headings;
- ``fund:<session>:event:<n>`` upcoming releases from ``econ_events.parse``;
- ``fund:<session>:p<n>`` the raw text in order, in paragraphs of at most 400 characters (at most 40),
  so any sentence of the brief can be cited;
- ``fund:<session>:none`` when nothing was pasted for the session.

Headings are matched by text only (a Markdown heading or a line that is all bold); nothing is
inferred, nothing is fetched, nothing is written. Pure and Qt-free.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from mentor_packs.registry import Pack, make_pack

NAME = "fundamentals_pack"
SECTIONS = ("bottom_line", "signals", "playbook", "events", "text")
#: ``compact`` = the as-of row, the bottom line and the playbook, at most COMPACT_ROWS (gate, /tape, memory).
EXTRA_SECTIONS = ("compact",)
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "The morning macro brief the trader pasted (from his scheduled Claude chat): bottom line, ranked "
            "signals, playbook, turbulence, upcoming releases and the full text by paragraph. Use it for the "
            "macro, the Fed, yields, oil, the dollar, CPI/NFP/PCE/FOMC, catalysts and 'what did the brief say'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "day": {"type": "string", "description": "'today' (default) or a session date YYYY-MM-DD."},
                "section": {"type": "string", "enum": ["all", *SECTIONS, *EXTRA_SECTIONS],
                            "description": "One part of the brief, or all of it (default)."},
            },
        },
    },
}

ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")
MAX_ROW_CHARS = 400
MAX_PARAGRAPHS = 40
MAX_BOTTOM = 6
MAX_SIGNALS = 6
MAX_PLAY = 8
MAX_WATCH = 8
MAX_TURB = 4
MAX_EVENTS = 12
COMPACT_ROWS = 8
ORIGIN = "external_forecast"
BOTTOM_HEADINGS = ("bottom line", "why it matters", "what matters today")
PLAY_HEADINGS = ("intraday playbook", "how it may trade", "intraday implications", "playbook")
WATCH_HEADINGS = ("what to watch",)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_BOLD_LINE = re.compile(r"^\s*\*\*([^*]+?)\*\*\s*:?\s*$")
_TURBULENCE = re.compile(r"turbulen", re.IGNORECASE)
_OUTLOOK = re.compile(r"\boutlook\b", re.IGNORECASE)
_BULLET = re.compile(r"^(\s*)(?:[-*+]|\d+\.)\s+(.*)$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class FundPaths:
    """Where the pack reads; tests pass fixture paths, the app uses :func:`live_paths`. None = no store."""

    ledger_dir: Path | None = None  # RUNTIME_DATA_DIR/evidence_ledgers (market_journal-YYYYMM.jsonl)
    theses: Path | None = None  # MARKET_THESES_FILE (the paste's source_model / created_at_claimed sidecar)


def live_paths() -> FundPaths:
    import project_paths as pp
    from evidence_ledger import default_ledger_dir

    return FundPaths(ledger_dir=default_ledger_dir(), theses=Path(pp.MARKET_THESES_FILE))


# ---------------------------------------------------------------- small helpers
def _clean(value: Any, cap: int = MAX_ROW_CHARS) -> str:
    text = " ".join(str(value or "").replace("**", "").split())
    return text if len(text) <= cap else text[: cap - 3].rstrip() + "..."


def _now(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def market_day(now: datetime | None = None) -> date:
    """The session "today" means for the trader: his own (PT) calendar date when it is a session, else the
    next session (a weekend reads ahead to Monday, so Friday's brief is "last brief: Friday")."""
    day = _now(now).astimezone(PT).date()
    try:
        import market_calendar

        return day if market_calendar.is_session(day) else market_calendar.next_session(day)
    except Exception:  # noqa: BLE001 - outside the calendar: the calendar date itself
        return day


def _pt(stamp: Any) -> str:
    try:
        moment = datetime.fromisoformat(str(stamp or "").replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    moment = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(PT).strftime("%Y-%m-%d %H:%M PT")


# ---------------------------------------------------------------- reading the store
def read_briefs(paths: FundPaths) -> list[dict[str, Any]]:
    """Every current (not superseded) pasted brief, oldest first by (session, paste time)."""
    if paths.ledger_dir is None:
        return []
    import market_journal
    from evidence_ledger import EvidenceLedger

    ledger = EvidenceLedger(stream=market_journal.STREAM, schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
                            directory=Path(paths.ledger_dir))
    rows = market_journal.resolve_entries(ledger.read().rows)
    briefs = [dict(row) for row in rows if str(row.get("origin") or "") == ORIGIN and str(row.get("text") or "").strip()]
    for row in briefs:
        row["_session"] = market_journal.session_of_entry(row)
    return sorted(briefs, key=lambda row: (row["_session"], str(row.get("created_at") or "")))


def pick_brief(briefs: Iterable[Mapping[str, Any]], session: str) -> tuple[dict[str, Any] | None, bool]:
    """(the brief for ``session`` or else the latest earlier one, True when it is an earlier one)."""
    on_day = [dict(row) for row in briefs if row.get("_session") == session]
    if on_day:
        return on_day[-1], False
    earlier = [dict(row) for row in briefs if str(row.get("_session") or "") < session]
    return (earlier[-1], True) if earlier else (None, False)


def sidecar_for(paths: FundPaths, entry_id: str) -> dict[str, Any]:
    """The newest forecast sidecar row for one entry; {} when none or unreadable."""
    if paths.theses is None or not entry_id:
        return {}
    try:
        lines = Path(paths.theses).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    found: dict[str, Any] = {}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("entry_id") == entry_id:
            found = row
    return found


# ---------------------------------------------------------------- reading the brief
def _heading(line: str) -> str | None:
    """The heading text of a Markdown heading or an all-bold line; None for body text."""
    if "→" in line:
        return None  # the ranked-signals line is bold, never a heading
    found = _MD_HEADING.match(line) or _BOLD_LINE.match(line)
    return _clean(found.group(1), 200).strip(" :") if found else None


def sections(text: str) -> list[tuple[str, list[str]]]:
    """``[(heading, body lines)]`` in document order; text before the first heading has heading ""."""
    out: list[tuple[str, list[str]]] = [("", [])]
    for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        head = _heading(line)
        if head is not None:
            out.append((head, []))
        else:
            out[-1][1].append(line)
    return [(head, body) for head, body in out if head or any(line.strip() for line in body)]


def _find(parts: list[tuple[str, list[str]]], names: Iterable[str]) -> tuple[str, list[str]] | None:
    """The first section whose heading starts with one of ``names`` (in the order of ``names``)."""
    for name in names:
        for head, body in parts:
            if head.casefold().startswith(name):
                return head, body
    return None


def _items(body: Iterable[str]) -> list[str]:
    """Top-level bullets (sub-bullets folded into their parent) or, without bullets, paragraphs."""
    items: list[str] = []
    paragraph: list[str] = []
    bulleted = False
    for raw in body:
        found = _BULLET.match(raw)
        if found and not found.group(1):
            bulleted = True
            items.append(found.group(2))
        elif found and items and bulleted:
            items[-1] += "; " + found.group(2)
        elif raw.strip():
            if bulleted and items and raw.startswith((" ", "\t")):
                items[-1] += " " + raw.strip()
            else:
                paragraph.append(raw.strip())
        elif paragraph:
            items.append(" ".join(paragraph))
            paragraph = []
    if paragraph:
        items.append(" ".join(paragraph))
    return [item for item in (_clean(i) for i in items) if item]


def _chunks(text: str) -> list[str]:
    """The raw text in order, as paragraphs of at most MAX_ROW_CHARS (long ones split by line, then sentence)."""
    body = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    carry = ""
    for paragraph in re.split(r"\n\s*\n", body):
        lines = [line for line in paragraph.split("\n") if line.strip()]
        if len(lines) == 1 and _heading(lines[0]) is not None:
            carry = f"{carry} {_heading(lines[0])}:".strip()  # a heading alone leads the next paragraph
            continue
        if carry:
            paragraph, carry = f"{carry} {paragraph.lstrip()}", ""
        pieces: list[str] = []
        for line in paragraph.split("\n"):
            line = " ".join(line.split())
            if not line:
                continue
            if len(line) <= MAX_ROW_CHARS:
                pieces.append(line)
                continue
            for sentence in _SENTENCE.split(line):
                while len(sentence) > MAX_ROW_CHARS:
                    pieces.append(sentence[:MAX_ROW_CHARS])
                    sentence = sentence[MAX_ROW_CHARS:]
                if sentence:
                    pieces.append(sentence)
        current = ""
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > MAX_ROW_CHARS:
                out.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
        if current:
            out.append(current)
    if carry:
        out.append(carry)
    return out


def _row(session: str, part: str, text: str, section: str, **extra: Any) -> dict[str, Any]:
    return {"id": f"fund:{session}:{part}", "kind": "fund", "section": section, "date": session,
            "text": _clean(text), **extra}


def brief_rows(entry: Mapping[str, Any], *, sidecar: Mapping[str, Any] | None = None,
               requested: str = "", fallback: bool = False) -> dict[str, list[dict[str, Any]]]:
    """The brief as rows by section (``asof``, ``bottom_line``, ``signals``, ``playbook``, ``events``, ``text``)."""
    import econ_events
    import forecast_brief

    session = str(entry.get("_session") or entry.get("session_date") or "")[:10]
    text = str(entry.get("text") or "")
    parsed = forecast_brief.parse(text)
    parts = sections(text)
    side = dict(sidecar or {})
    model = str(side.get("source_model") or "unknown")
    claimed = str(side.get("created_at_claimed") or "unknown")
    label = f"last brief: {session} (none pasted for {requested})" if fallback else f"brief for {session}"
    asof = [_row(session, "asof", f"Morning brief ({label}): pasted {_pt(entry.get('created_at'))}, source "
                 f"{model}, written {claimed}; the trader's outside commentary, not his own view.", "asof",
                 entry_id=str(entry.get("entry_id") or ""))]

    bottom: list[dict[str, Any]] = []
    if parsed.bottom_line.strip():
        lines = _items(parsed.bottom_line.split("\n"))
        bottom = [_row(session, f"bottom:{n}", f"Bottom line: {line}", "bottom_line")
                  for n, line in enumerate(lines[:MAX_BOTTOM], start=1)]
    else:
        found = _find(parts, BOTTOM_HEADINGS[1:])
        if found is not None:
            head, body = found
            bottom = [_row(session, f"bottom:{n}", f"Bottom line (from '{head}'): {line}", "bottom_line")
                      for n, line in enumerate(_items(body)[:MAX_BOTTOM], start=1)]

    signals = [_row(session, f"signal:{n}", f"Ranked signal {n}: {signal}", "signals")
               for n, signal in enumerate(parsed.ranked_signals[:MAX_SIGNALS], start=1)]
    watch = _find(parts, WATCH_HEADINGS)
    if watch is not None:
        signals += [_row(session, f"watch:{n}", f"What to watch: {line}", "signals")
                    for n, line in enumerate(_items(watch[1])[:MAX_WATCH], start=1)]

    play: list[dict[str, Any]] = []
    if parsed.playbook_bullish.strip():
        play.append(_row(session, "bull:1", f"Playbook, bullish: {parsed.playbook_bullish}", "playbook"))
    if parsed.playbook_bearish.strip():
        play.append(_row(session, "bear:1", f"Playbook, bearish: {parsed.playbook_bearish}", "playbook"))
    if not play:
        found = _find(parts, PLAY_HEADINGS)
        if found is not None:
            head, body = found
            play = [_row(session, f"play:{n}", f"Playbook ('{head}'): {line}", "playbook")
                    for n, line in enumerate(_items(body)[:MAX_PLAY], start=1)]
    turb_lines = [line for line in text.splitlines() if _TURBULENCE.search(line) and not _heading(line)]
    turb_lines += [head for head, _body in parts if head and (_OUTLOOK.search(head) or _TURBULENCE.search(head))]
    play += [_row(session, f"turb:{n}", line if _clean(line).lower().startswith("turbulence") else
                  f"Turbulence: {line}", "playbook") for n, line in enumerate(turb_lines[:MAX_TURB], start=1)]

    events = [event for event in econ_events.parse(text).events if event.date >= session][:MAX_EVENTS]
    event_rows = [_row(session, f"event:{n}", f"Upcoming: {event.date} {event.time_et + ' ET' if event.time_et else 'time TBD'}"
                       f" {event.label}", "events") for n, event in enumerate(events, start=1)]

    chunks = _chunks(text)
    paragraphs = [_row(session, f"p{n}", chunk, "text") for n, chunk in enumerate(chunks[:MAX_PARAGRAPHS], start=1)]
    if len(chunks) > MAX_PARAGRAPHS:
        asof[0]["text"] += f" The text is cut at {MAX_PARAGRAPHS} of {len(chunks)} paragraphs."
    return {"asof": asof, "bottom_line": bottom, "signals": signals, "playbook": play, "events": event_rows,
            "text": paragraphs}


def _none(session: str, text: str) -> dict[str, Any]:
    return {"id": f"fund:{session}:none", "kind": "none", "section": "none", "date": session, "text": text}


def compact(parts: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """The as-of row, the bottom line, then the playbook, at most COMPACT_ROWS rows."""
    return [*parts.get("asof", ()), *parts.get("bottom_line", ()), *parts.get("playbook", ())][:COMPACT_ROWS]


def resolve_session(day: Any, now: datetime | None) -> str | None:
    text = str(day or "today").strip().lower()
    if text in ("", "today"):
        return market_day(now).isoformat()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return None
    if text == "yesterday":
        back = market_day(now) - timedelta(days=1)
        while back.weekday() >= 5:
            back -= timedelta(days=1)
        return back.isoformat()
    return None


def build(day: Any = "today", section: str = "all", *, now: datetime | None = None,
          paths: FundPaths | None = None) -> Pack:
    """Build the fundamentals pack. File reads: call it on a worker."""
    wanted = str(section or "all").strip().lower()
    if wanted != "all" and wanted not in (*SECTIONS, *EXTRA_SECTIONS):
        return make_pack(NAME, (), empty_text=f"fundamentals_pack sections are all, {', '.join(SECTIONS)}, compact; "
                                              f"not {section!r}")
    session = resolve_session(day, now)
    if session is None:
        return make_pack(NAME, (), empty_text=f"fundamentals_pack could not read the day {day!r}; try 'today' or "
                                              "YYYY-MM-DD")
    src = paths or live_paths()
    entry, fallback = pick_brief(read_briefs(src), session)
    if entry is None:
        return make_pack(NAME, [_none(session, f"No morning brief pasted for {session} or before; the fundamentals are "
                                               "unknown, not quiet. /paste stores one.")])
    parts = brief_rows(entry, sidecar=sidecar_for(src, str(entry.get("entry_id") or "")), requested=session,
                       fallback=fallback)
    head = [_none(session, f"No morning brief pasted for {session}; the rows below are the last brief, "
                           f"{entry['_session']}.")] if fallback else []
    if wanted == "compact":
        return make_pack(NAME, [*head, *compact(parts)])
    body = [row for name in (SECTIONS if wanted == "all" else (wanted,)) for row in parts[name]]
    if not body:
        body = [{"id": f"fund:{entry['_session']}:{wanted}:none", "kind": "none", "section": wanted,
                 "date": entry["_session"], "text": f"The brief has no {wanted.replace('_', ' ')} part (by heading)."}]
    return make_pack(NAME, [*head, *parts["asof"], *body])


def as_text(pack: Pack) -> str:
    """Compact: one line per row with its id."""
    return pack.as_text()


def bottom_line_sentence(pack: Pack) -> str:
    """The first sentence of the first bottom-line row (for the paste confirmation), else ""."""
    row = next((row for row in pack.rows if row.get("section") == "bottom_line"), None)
    if row is None:
        return ""
    text = re.sub(r"^Bottom line(?: \(from '[^']*'\))?: ", "", str(row.get("text") or ""))
    return _SENTENCE.split(text, maxsplit=1)[0]


def embed_rows(pack: Pack) -> list[tuple[int, str]]:
    """``(ref_id, text)`` per paragraph row; the ref is stable for the brief's entry id and the row's place."""
    from mentor_packs.night_pack import stable_ref

    entry_id = next((str(row.get("entry_id") or "") for row in pack.rows if row.get("section") == "asof"), "")
    return [(stable_ref(f"{entry_id}:{row['id']}", ""), f"[{row['id']}] {row['text']}")
            for row in pack.rows if row.get("section") == "text" and entry_id]


# ---------------------------------------------------------------- fixture
FIXTURE_NOW = datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc)  # Wed 2026-09-30, 07:00 PT

FIXTURE_BRIEF = """# Morning Brief — September 30, 2026

As of 8:45 ET.

## The setup

- **Rates:** the 10-year is near 5.28%, its highest since 2007.
- **Oil:** WTI about $90; Hormuz headlines keep a bid under crude.
- **Dollar:** the DXY eased to 101 after the data.

**10Y yield → Brent → NVDA/SMH → DXY**

## Intraday playbook

The **bullish continuation** needs the 10-year back below 5.25% and SPY holding its opening range.

The **bearish reversal:** a new 10-year high above 5.30% with oil up 2% sends small caps lower.

Turbulence: ~6/10 today, 7–8/10 into Friday's payrolls.

## Bottom line

Softer core PCE weakens the case for an October hike. Rates lead today; oil is the risk to any rally.

## NEXT 7 DAYS — ECONOMIC CALENDAR

2026-09-30 | 06:45 PT / 09:45 ET | Chicago PMI (Sep)
2026-10-01 | 07:00 PT / 10:00 ET | ISM Manufacturing PMI (Sep)
2026-10-02 | 05:30 PT / 08:30 ET | Employment Situation / Nonfarm Payrolls (Sep)
"""
FIXTURE_FIRST = "# Morning Brief — September 30, 2026\n\n## Bottom line\n\nSUPERSEDED first paste.\n"
FIXTURE_OLD = """**Market Morning Brief: Monday, Sept 28, 2026**

**Why it matters**

Yields are the story. A Fed hike is priced at 66% for October.

**How it may trade today**

- **Rates lead.** Below 5.25% on the 10-year, small caps bounce.
- **Base case:** chop into the afternoon.
"""


def write_fixture_world(root: Path | str) -> FundPaths:
    """Briefs for Mon 2026-09-28 (Claude style, bold headings) and Wed 2026-09-30 (pasted twice; the second wins)."""
    import market_journal
    from evidence_ledger import EvidenceLedger

    base = Path(root)
    ledger_dir = base / "evidence_ledgers"
    ledger = EvidenceLedger(stream=market_journal.STREAM, schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
                            directory=ledger_dir)

    def paste(entry_id: str, session: str, text: str, at: datetime, supersedes: str = "") -> None:
        ledger.append({"entry_id": entry_id, "event_type": "entry", "origin": ORIGIN, "text": text,
                       "created_at": at.isoformat(timespec="seconds"), "supersedes": supersedes, "symbols": [],
                       "timeframe": "D1"}, now=at, subject_session_date=session)

    paste("mj-2026-09-28-old", "2026-09-28", FIXTURE_OLD, datetime(2026, 9, 28, 12, 40, tzinfo=timezone.utc))
    paste("mj-2026-09-30-first", "2026-09-30", FIXTURE_FIRST, datetime(2026, 9, 30, 12, 45, tzinfo=timezone.utc))
    paste("mj-2026-09-30-second", "2026-09-30", FIXTURE_BRIEF, datetime(2026, 9, 30, 12, 50, tzinfo=timezone.utc),
          supersedes="mj-2026-09-30-first")
    theses = base / "market_theses.jsonl"
    theses.write_text(json.dumps({"entry_id": "mj-2026-09-30-second", "kind": "forecast", "source_model": "claude",
                                  "created_at_claimed": "2026-09-30T05:45:00-07:00"}) + "\n", encoding="utf-8")
    return FundPaths(ledger_dir=ledger_dir, theses=theses)


def fixture() -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        return build("today", now=FIXTURE_NOW, paths=write_fixture_world(tmp))
