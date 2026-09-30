"""Auto-attach: which packs a plain-language question needs, decided by the app. Pure, Qt-free.

``plan_attachments(text, known_symbols, now)`` reads tickers, time words and intent words
and returns the packs the brain builds and injects as tool results in the first model
message. The model can still call more tools; nothing is hidden from it. A plain
uppercase token is a ticker only when it is in the trader's universe (Focus, liked, open
book, journal symbols of the last 60 days); ``$SYM`` is a ticker anywhere, any case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
MAX_SYMBOLS = 3
#: A row cited in this many recent turns is not attached again unless the question names its subject.
DEDUPE_TURNS = 6
#: Kept first when the attachment budget is tight (lower = more important).
PRIORITY = {
    "gate_pack": 0,
    "journal_pack": 1,
    "pick_pack": 2,
    "veto_pack": 3,
    "regime_pack": 4,
    "book_pack": 5,
    "news_pack": 6,
    "tilt_pack": 7,
    "mirror_pack": 8,
    "plan_lines": 9,
    "recall": 10,
}
#: Index tickers are the tape, never a pick.
INDEX_SYMBOLS = frozenset({"SPY", "QQQ", "IWM", "DIA", "VIX"})
#: Uppercase words that are desk vocabulary, not tickers, unless typed as ``$SYM``.
NOT_TICKERS = frozenset({"I", "A", "AM", "PM", "ET", "PT", "EOD", "OK", "HOD", "LOD", "VWAP", "AVWAP", "RVOL",
                         "ATH", "ATR", "RRS", "D1", "M5", "R", "PNL", "USD", "CAD", "TFSA", "RRSP", "IRA", "LOL"})

_DOLLAR = re.compile(r"\$([A-Za-z]{1,5}(?:[.\-][A-Za-z]{1,2})?)(?![A-Za-z])")
_PLAIN = re.compile(r"(?<![A-Za-z$.\-])([A-Z]{1,5}(?:[.\-][A-Z]{1,2})?)(?![A-Za-z])")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAY = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")s?\b")
_TODAY = re.compile(r"\b(today|this morning|so far|this session|today's)\b")
_YESTERDAY = re.compile(r"\byesterday\b")
_THIS_WEEK = re.compile(r"\bthis week\b|\bweek so far\b")
_LAST_WEEK = re.compile(r"\blast week\b")
_INTENT = re.compile(
    r"\bthinking (?:of|about)\b|\bshould i\b|\btake\b|\btaking\b|\bgo(?:ing)? (?:long|short)\b|\benter(?:ing)?\b"
    r"|\badd(?:ing)? (?:to )?\b|\bsize\b|\bsizing\b|\bget(?:ting)? (?:in|into)\b|\bworth (?:a|the) (?:trade|shot)\b"
    r"|\bplanning (?:to|on)\b|\bwant to (?:short|buy|long)\b"
)
#: Past-tense trade talk is the journal, never a pre-trade check.
_PAST = re.compile(r"\bdid i\b|\bi took\b|\btook\b|\bwhy did\b|\bhow did\b|\bwhat happened\b|\bwas i\b")
_JOURNAL = re.compile(
    r"\bwhy did i\b|\bwhat happened\b|\bhow did i do\b|\bhow did (?:it|today|the day|my day|that) go\b"
    r"|\bhow(?:'s| is| has) (?:today|the day|my day) (?:going|been)\b|\bmy trades\b|\btrades? did i\b"
    r"|\bdid i (?:take|trade|make|lose|win|do)\b|\bp&l\b|\bpnl\b|\blos[et] money\b|\bmade money\b"
    r"|\bmy (?:day|week|losses|wins|fills)\b|\bhow am i doing today\b|\bgreen or red\b|\bi took\b"
)
_VETO = re.compile(r"\bveto(?:ed|es|s)?\b|\bpassed on\b|\bi passed\b|\bskipped\b")
_TAPE = re.compile(r"\btape\b|\bmarket\b|\bspy\b|\bqqq\b|\biwm\b|\bregime\b|\bsectors?\b|\bbreadth\b|\bmacro\b"
                   r"|\bindex(?:es)?\b|\bfomc\b|\bcpi\b|\bjobs report\b|\bfed\b")
_NEWS = re.compile(r"\bnews\b|\bearnings\b|\breporting\b|\breports?\b|\bheadlines?\b|\bcatalysts?\b")
_BOOK = re.compile(r"\bmy book\b|\bpositions?\b|\bexposure\b|\bholding\b|\bwhat am i in\b|\bopen trades?\b")
_TILT = re.compile(r"\btilt(?:ed|ing)?\b|\brevenge\b|\bovertrad")
_MIRROR = re.compile(r"\bmy record\b|\blately\b|\bmy stats\b|\bmy edge\b|\bhit rate\b|\bwin rate\b|\bpattern in my\b")
_PLAN = re.compile(r"\bmy plan\b|\bmy rules?\b|\btrading plan\b|\bbreak(?:ing)? (?:a|my) rule\b")
_RECALL = re.compile(r"\byou said\b|\bwe (?:said|talked|discussed)\b|\bremember when\b|\blast time we\b")
_GROUP = re.compile(r"\bmy (longs|shorts|focus|names|picks|watchlist|likes|liked)\b|\bfocus (longs|shorts|names)\b")
_SHORT_WORD = re.compile(r"\bshort(?:ing|s|ed)?\b|\bsell(?:ing)? short\b|\bput(?:s)?\b")
_LONG_WORD = re.compile(r"\blong\b|\bbuy(?:ing)?\b|\bgo long\b|\bcalls?\b")


@dataclass(frozen=True)
class AttachRequest:
    """One pack to build and inject: its tool name, arguments, keep-priority and why."""

    name: str
    args: dict[str, Any] = field(default_factory=dict, hash=False)
    priority: int = 50
    reason: str = ""

    def key(self) -> tuple[str, tuple[tuple[str, str], ...]]:
        return self.name, tuple(sorted((str(k), str(v)) for k, v in self.args.items()))


def _market_day(now: datetime | None) -> date:
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.astimezone(ET).date()


def previous_weekday(day: date) -> date:
    back = day - timedelta(days=1)
    while back.weekday() >= 5:
        back -= timedelta(days=1)
    return back


def last_weekday_named(name: str, today: date) -> date:
    """The latest ``name``-day on or before ``today``."""
    target = _WEEKDAYS.index(name)
    return today - timedelta(days=(today.weekday() - target) % 7)


def resolve_day(text: str, now: datetime | None = None) -> str:
    """The journal day a question means: ISO date, ``week``, ``last_week``, or "" (none named)."""
    lowered = str(text or "").lower()
    today = _market_day(now)
    if _LAST_WEEK.search(lowered):
        return "last_week"
    if _THIS_WEEK.search(lowered):
        return "week"
    if _YESTERDAY.search(lowered):
        return previous_weekday(today).isoformat()
    found = _WEEKDAY.search(lowered)
    if found:
        return last_weekday_named(found.group(1), today).isoformat()
    if _TODAY.search(lowered):
        return today.isoformat()
    return ""


def _normalise_known(known_symbols: Mapping[str, Any] | Iterable[str]) -> dict[str, str]:
    """``{SYM: side}`` with side LONG/SHORT or ""; an iterable means no sides."""
    if isinstance(known_symbols, Mapping):
        items = known_symbols.items()
    else:
        items = ((sym, "") for sym in known_symbols or ())
    out: dict[str, str] = {}
    for sym, side in items:
        key = str(sym or "").strip().upper()
        if not key:
            continue
        value = str(side or "").strip().upper()
        if value not in ("LONG", "SHORT"):
            value = ""
        if key not in out or (value and not out[key]):
            out[key] = value
    return out


def find_symbols(text: str, known_symbols: Mapping[str, Any] | Iterable[str]) -> list[str]:
    """Tickers in the question, in order: ``$SYM`` any case; plain uppercase only when known."""
    known = _normalise_known(known_symbols)
    hits: list[tuple[int, str]] = []
    for match in _DOLLAR.finditer(text or ""):
        hits.append((match.start(), match.group(1).upper()))
    for match in _PLAIN.finditer(text or ""):
        token = match.group(1)
        if token in known and token not in NOT_TICKERS:
            hits.append((match.start(), token))
    ordered: list[str] = []
    for _, sym in sorted(hits):
        if sym not in ordered:
            ordered.append(sym)
    return ordered


def _side_words(lowered: str) -> str:
    short, long_ = _SHORT_WORD.search(lowered), _LONG_WORD.search(lowered)
    if short and not long_:
        return "SHORT"
    if long_ and not short:
        return "LONG"
    if short and long_:
        return "SHORT" if short.start() < long_.start() else "LONG"
    return ""


def plan_attachments(
    text: str, known_symbols: Mapping[str, Any] | Iterable[str], now: datetime | None = None
) -> list[AttachRequest]:
    """The packs a plain question needs, most important first. Deterministic for (text, known, now)."""
    raw = str(text or "")
    lowered = raw.lower()
    known = _normalise_known(known_symbols)
    symbols = [sym for sym in find_symbols(raw, known) if sym not in INDEX_SYMBOLS][:MAX_SYMBOLS]
    day = resolve_day(raw, now)
    today = _market_day(now).isoformat()
    past = bool(_PAST.search(lowered))
    wanted: list[AttachRequest] = []

    def add(name: str, reason: str, **args: Any) -> None:
        request = AttachRequest(name, dict(args), PRIORITY.get(name, 50), reason)
        if all(existing.key() != request.key() for existing in wanted):
            wanted.append(request)

    gated: set[str] = set()
    if symbols and _INTENT.search(lowered) and not past:
        side = _side_words(lowered)
        for sym in symbols:
            chosen = side or known.get(sym, "")
            if chosen:
                add("gate_pack", f"trade intent on {sym}", side=chosen, symbol=sym)
                gated.add(sym)
    for sym in symbols:
        if sym not in gated:
            add("pick_pack", f"ticker {sym}", symbol=sym)
        add("news_pack", f"ticker {sym}", symbol=sym)
    if day or _JOURNAL.search(lowered):
        add("journal_pack", "time words" if day else "journal words", day=day or today)
    if day:
        add("regime_pack", "time words")
    if _VETO.search(lowered):
        add("veto_pack", "veto words", date=day if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) else "")
    if _TAPE.search(lowered) or any(sym in INDEX_SYMBOLS for sym in find_symbols(raw, set(INDEX_SYMBOLS))):
        add("regime_pack", "tape words")
    if _NEWS.search(lowered) and not symbols:
        group = _GROUP.search(lowered)
        if group:
            which = next(g for g in group.groups() if g)
            side = "LONG" if which.startswith("long") else "SHORT" if which.startswith("short") else ""
            names = [sym for sym, s in known.items() if sym not in INDEX_SYMBOLS and (not side or s == side)]
            for sym in names[:MAX_SYMBOLS]:
                add("pick_pack", f"earnings/news across {which}", symbol=sym)
    if _BOOK.search(lowered):
        add("book_pack", "book words")
    if _TILT.search(lowered):
        add("tilt_pack", "tilt words")
    if _MIRROR.search(lowered):
        add("mirror_pack", "record words")
    if _PLAN.search(lowered):
        add("plan_lines", "plan words")
    if _RECALL.search(lowered):
        add("recall", "memory words", query=raw[:200])
    return sorted(wanted, key=lambda request: request.priority)


def known_symbols(
    context_rows: Iterable[Mapping[str, Any]] = (),
    liked: Iterable[Any] = (),
    journal_symbols: Iterable[str] = (),
) -> dict[str, str]:
    """The trader's universe as ``{SYM: side}``: Focus names, liked picks, open book, journal names."""
    out: dict[str, str] = {}

    def put(sym: Any, side: Any = "") -> None:
        key = str(sym or "").strip().upper()
        if not key:
            return
        value = str(side or "").strip().upper()
        value = value if value in ("LONG", "SHORT") else ""
        if key not in out or (value and not out[key]):
            out[key] = value

    rows = list(context_rows or ())
    for category in ("swing", "m5"):
        for row in rows:
            if row.get("kind") == "focus" and row.get("category") == category:
                for name in row.get("names") or ():
                    put(name, row.get("side"))
    for row in rows:
        if row.get("kind") == "position":
            put(row.get("symbol"), row.get("direction") or "")
    for item in liked or ():
        if isinstance(item, (tuple, list)) and item:
            put(item[0], item[1] if len(item) > 1 else "")
        else:
            put(item)
    for sym in journal_symbols or ():
        put(sym)
    return out


def cited_ids(text: str) -> list[str]:
    """Every ``[id]`` cited in a reply, in order."""
    return [match.strip() for match in re.findall(r"\[([A-Za-z][^\[\]\s]{2,})\]", str(text or ""))]


def recent_cited_ids(turns: Iterable[Any], last: int = 6) -> set[str]:
    """Ids the mentor cited in the last ``last`` turns of this session."""
    kept = list(turns or ())[-int(last):]
    found: set[str] = set()
    for turn in kept:
        role = getattr(turn, "role", None) or (turn.get("role") if isinstance(turn, Mapping) else "")
        text = getattr(turn, "text", None) or (turn.get("text") if isinstance(turn, Mapping) else "")
        if role == "assistant":
            found.update(cited_ids(text))
    return found


_GENERIC_ID_PARTS = frozenset({"pick", "gate", "tape", "ctx", "jrn", "news", "book", "plan", "veto", "mirror", "tilt",
                               "regime", "hyp", "mem", "none", "asof", "pos", "acct", "hint", "pack"})


def names_subject(question: str, row_id: str) -> bool:
    """True when the question names what a row is about (its ticker or its topic word)."""
    words = set(re.findall(r"[a-z0-9]+", str(question or "").lower()))
    for part in re.split(r"[:_\-]", str(row_id or "").lower()):
        if len(part) >= 2 and part not in _GENERIC_ID_PARTS and not part.isdigit() and part in words:
            return True
    return False
