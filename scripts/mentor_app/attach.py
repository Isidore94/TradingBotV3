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
    "earnings_pack": 2,
    "veto_pack": 3,
    "regime_pack": 4,
    "book_pack": 5,
    "fundamentals_pack": 5,
    "news_pack": 6,
    "tilt_pack": 7,
    "mirror_pack": 8,
    "night_pack": 8,
    "recaps_pack": 8,
    "plan_lines": 9,
    "recall": 10,
}
#: Index tickers are the tape, never a pick.
INDEX_SYMBOLS = frozenset({"SPY", "QQQ", "IWM", "DIA", "VIX"})
#: Uppercase words that are desk vocabulary, not tickers, unless typed as ``$SYM``.
NOT_TICKERS = frozenset({"I", "A", "AM", "PM", "ET", "PT", "EOD", "OK", "HOD", "LOD", "VWAP", "AVWAP", "RVOL",
                         "ATH", "ATR", "RRS", "D1", "M5", "R", "PNL", "USD", "CAD", "TFSA", "RRSP", "IRA", "LOL"})

#: Lowercase ``$word`` forms that are ordinary English: a ticker only when in the trader's universe.
COMMON_WORDS = frozenset({
    "a", "all", "an", "and", "am", "are", "as", "at", "be", "by", "can", "do", "for", "go", "has", "he", "hi", "i",
    "if", "in", "is", "it", "me", "my", "no", "not", "now", "of", "on", "or", "out", "so", "the", "to", "up", "us",
    "was", "we", "you", "big", "low", "high", "run", "see", "new", "key", "real", "fast", "good", "well", "one",
    "open", "next", "life", "love", "fun", "cash", "free", "any", "few", "true", "ever", "safe", "else",
})
_DOLLAR = re.compile(r"\$([A-Za-z]{1,5}(?:[.\-][A-Za-z]{1,2})?)(?![A-Za-z])")
_PLAIN = re.compile(r"(?<![A-Za-z$.\-])([A-Z]{1,5}(?:[.\-][A-Z]{1,2})?)(?![A-Za-z])")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAY = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")s?\b")
_TODAY = re.compile(r"\b(today|this morning|so far|this session|today's)\b")
_YESTERDAY = re.compile(r"\byesterday\b")
_THIS_WEEK = re.compile(r"\bthis week\b|\bweek so far\b|\bmy week\b")
_LAST_WEEK = re.compile(r"\blast week\b")
_THIS_MONTH = re.compile(r"\bthis month\b|\bmonth so far\b|\bmy month\b|\bthe month\b")
_LAST_MONTH = re.compile(r"\blast month\b")
#: The trader talking about himself: a time word alone ("today") is his journal only with one of these.
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|im|i've|ive|me|my|mine)\b|\bgreen or red\b|\bso far\b|\btoday go\b"
                           r"|\bthe day (?:go|going)\b|\bpassed on\b")
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
    r"|\bstop(?:ped)? me out\b|\bmy (?:entry|exit|stop) on\b"
)
_VETO = re.compile(r"\bveto(?:ed|es|s)?\b|\bpassed on\b|\bi passed\b|\bskip(?:ped|ping|s)?\b")
#: P16: the record of his vetoes, not one session's list.
_VETO_AGG = re.compile(r"\bfollowed my vetoes\b|\bwork(?:ed)? out\b|\btrack record\b|\b(?:worst|best) record\b"
                       r"|\bveto reasons?\b|\bwould have worked\b|\bhow would i have done\b")
_WINDOWS = ("week", "month", "last_week", "last_month")
_TAPE = re.compile(r"\btape\b|\bmarkets?\b|\bspy\b|\bqqq\b|\biwm\b|\bregime\b|\bsectors?\b|\bbreadth\b|\bmacro\b"
                   r"|\bindex(?:es)?\b|\bfomc\b|\bcpi\b|\bjobs report\b|\bfed\b|\bfutures\b")
#: Getting ready for a session: the tape is the answer ("what should I look at tomorrow morning").
_PREP = re.compile(r"\bwhat should i (?:look at|watch|focus on)\b|\bgame ?plan\b|\bpre-?market\b"
                   r"|\btomorrow\b.*\b(?:look at|watch|expect|prep)\b|\b(?:look at|watch|expect|prep)\b.*\btomorrow\b")
_NEWS = re.compile(r"\bnews\b|\bearnings\b|\breporting\b|\breports?\b|\bheadlines?\b|\bcatalysts?\b")
_BOOK = re.compile(r"\bmy book\b|\bpositions?\b|\bexposure\b|\b(?:am i|i'm|im|i am) holding\b|\bholdings\b"
                   r"|\bwhat am i in\b|\bopen trades?\b|\b(?:any|my) (?:open )?(?:shorts|longs) on\b"
                   r"|\bdo i have any (?:open )?(?:shorts|longs|positions)\b|\bmy open (?:shorts|longs)\b")
#: P16: a group the trader HOLDS ("my shorts", "my open longs", "my book", "what I'm holding") is the open book only.
_BOOK_SCOPE = re.compile(r"\bmy (?:open )?(?:shorts|longs|positions|book|holdings)\b|\bopen (?:shorts|longs|positions)\b"
                         r"|\b(?:am i|i'm|im|i am) holding\b|\bwhat i'm holding\b")
#: P16: a group he WATCHES ("my focus longs", "watchlist", "my lists", "my picks/likes/names") is Focus and likes.
_FOCUS_SCOPE = re.compile(r"\bmy focus\b|\bfocus (?:longs|shorts|names|list)\b|\bwatch ?lists?\b|\bmy lists?\b"
                          r"|\bmy (?:picks|names|likes|liked)\b")
_TILT = re.compile(r"\btilt(?:ed|ing)?\b|\brevenge\b|\bovertrad")
#: Walking away for the day: the tilt read plus today's journal.
_STOP_DAY = re.compile(r"\bstop trading\b|\bcall it a day\b|\bwalk away\b|\bovertrad|\bquit for (?:the|to)day\b"
                       r"|\bdone for (?:the|to)day\b")
_MIRROR = re.compile(r"\bmy record\b|\blately\b|\bmy stats\b|\bmy edge\b|\bhit rate\b|\bwin rate\b|\bpattern in my\b"
                     r"|\btrade best\b|\bbest time\b|\btime of day\b|\bwhen during the day\b")
#: P16: hold time by outcome and the best / worst trade or setup: the journal's outcome rows plus the mirror.
_OUTCOME = re.compile(r"\bhold(?:ing)? times?\b|\bhow long (?:do|did) i hold\b|\b(?:best|worst) (?:trades?|setups?)\b"
                      r"|\bwinners (?:vs\.?|versus|and) losers\b")
#: P16: "is it the regime or me": his month, his mirror (regime and kind cuts) and the tape.
_REGIME_OR_ME = re.compile(r"\bregime or me\b|\bme or the (?:regime|market|tape)\b|\b(?:market|tape) or me\b"
                           r"|\bam i the problem\b|\bmy fault\b|\bis it (?:just )?me\b")
#: P16: this week against last week reads both weeks, never today.
_WEEK_VS = re.compile(r"\b(?:this|my) week\b.*\blast week\b|\blast week\b.*\bthis week\b"
                      r"|\bweek over week\b|\bweek on week\b")
#: P16: what to watch for the next session: the brief's watch list, the next session's econ, the day review.
_WATCH_NEXT = re.compile(r"\b(?:watch|look for|look at|focus on|important)\b.*\b(?:tomorrow|at the open|next session)\b"
                         r"|\b(?:tomorrow|at the open)\b.*\b(?:watch|look for)\b")
_PLAN = re.compile(r"\bmy plan\b|\bmy rules?\b|\btrading plan\b|\bbreak(?:ing)? (?:a|my) rule\b")
#: P15a: the night's reads (day review verdicts, ideas, contrasts, week review, story, digest).
_NIGHT = re.compile(r"\bwhat did the night say\b|\bovernight\b|\blast night\b|\bnight(?:'s)? read\b|\bideas?\b"
                    r"|\bwhat (?:am|are) (?:i|we) missing\b|\bwhat did i get (?:wrong|right)\b|\bweek(?:ly)? review\b"
                    r"|\bthe night\b")
#: P15b: the morning brief the trader pastes (macro, Fed, yields, oil, the dollar, releases, the playbook).
_FUND = re.compile(r"\bfundamentals?\b|\bbrief\b|\bmacro\b|\bthe paste\b|\bwhat did claude (?:say|flag|write)\b"
                   r"|\bclaude\b|\bcatalysts?\b|\bfed\b|\byields?\b|\boil\b|\bcrude\b|\bdollar\b|\bdxy\b"
                   r"|\bcpi\b|\bnfp\b|\bpce\b|\bfomc\b|\bpayrolls\b|\bbottom line\b|\bplaybook\b|\bscenarios?\b")
#: P15b: his day recaps and the recurring issues computed from them. "lately" / "pattern" alone go to the
#: mirror when the question is about his record or stats ("how has my record been lately").
_RECAP = re.compile(r"\brecaps?\b|\breviews?\b|\b(?:doing|done|did) wrong\b|\bmy issues\b|\bissues\b"
                    r"|\bwhat (?:am|are) (?:i|we) missing\b|\bkeep doing\b|\bsame mistakes?\b|\bmistakes?\b"
                    r"|\bwhat should i (?:stop|keep)\b")
_RECAP_SOFT = re.compile(r"\blately\b|\bpatterns?\b|\brecently\b")
_RECORD = re.compile(r"\bmy record\b|\bmy stats\b|\bmy edge\b|\bhit rate\b|\bwin rate\b|\btrade best\b"
                     r"|\bbest time\b|\btime of day\b")
#: P15b: how a trade felt (a feelings note rides on its journal row).
_FEEL = re.compile(r"\bfeel(?:ing|ings|s)?\b|\bfelt\b")
_RECALL = re.compile(r"\byou said\b|\bwe (?:said|talked|discussed)\b|\bremember when\b|\blast time we\b")
_GROUP = re.compile(r"\bmy (longs|shorts|focus|names|picks|watchlist|likes|liked|book|positions|holdings)\b"
                    r"|\bfocus (longs|shorts|names)\b|\bopen (longs|shorts|positions)\b|\b(?:i'm|im|i am) (holding)\b")
_EARNINGS = re.compile(r"\bearnings\b|\breporting\b|\breports?\b")
#: Earnings as the whole question, with no group named ("earnings this week?"): the book + likes.
_EARNINGS_ALONE = re.compile(
    r"^\W*(?:any\s+)?earnings\b|\b(?:earnings|reports?|reporting)\s+(?:this|next)\s+week\b"
    r"|\b(?:earnings|reports?|reporting)\s+(?:today|tomorrow)\b|\b(?:anything|anyone|who|who's|whos)\s+(?:is\s+)?"
    r"report(?:s|ing)?\b")
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
    """The journal day a question means: ISO date, ``week``, ``last_week``, ``month``, ``last_month`` or ""."""
    lowered = str(text or "").lower()
    today = _market_day(now)
    if _LAST_MONTH.search(lowered):
        return "last_month"
    if _THIS_MONTH.search(lowered):
        return "month"
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
        typed = match.group(1)
        sym = typed.upper()
        # `$IT` typed in capitals is a ticker; `$it` is a common word, a ticker only inside the universe.
        if typed != sym and typed.lower() in COMMON_WORDS and sym not in known:
            continue
        hits.append((match.start(), sym))
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


def market_cue(text: str) -> bool:
    """True when the question is about the market or a trade about to be taken: then tape context is wanted."""
    raw = str(text or "")
    lowered = raw.lower()
    if _TAPE.search(lowered) or _PREP.search(lowered):
        return True
    if any(sym in INDEX_SYMBOLS for sym in find_symbols(raw, set(INDEX_SYMBOLS))):
        return True
    if re.search(r"\bthis morning\b", lowered) and not _FIRST_PERSON.search(lowered):
        return True
    return bool(_INTENT.search(lowered) and not _PAST.search(lowered) and not _STOP_DAY.search(lowered))


def plan_attachments(
    text: str, known_symbols: Mapping[str, Any] | Iterable[str], now: datetime | None = None, *,
    book: Iterable[str] = (), liked: Iterable[Any] = (),
) -> list[AttachRequest]:
    """The packs a plain question needs, most important first. Deterministic for the arguments.

    ``book`` (open positions) and ``liked`` (liked chips) order an earnings read over a group: the book
    first and never capped, then the likes, then Focus; with neither given, every known name with a side.
    """
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
    vetoes = bool(_VETO.search(lowered))
    journal_words = bool(_JOURNAL.search(lowered))
    stop_day = bool(_STOP_DAY.search(lowered))
    group = _GROUP.search(lowered) if not symbols else None
    earnings_group = bool(group and _EARNINGS.search(lowered))
    # "earnings this week?" names no group: it means the open book and the liked chips.
    earnings_alone = bool(not symbols and not group and not _PLAN.search(lowered) and _EARNINGS_ALONE.search(lowered))
    # P14: "today"/"this morning" alone is not the journal ("headlines on MSFT today", "market this morning");
    # yesterday, a weekday, a week or a month is; so is any first-person time question. A veto question
    # (or an earnings question over his book) reads its own pack, not the journal, unless it asks about trades.
    journal_day = bool(day) and (day != today or bool(_FIRST_PERSON.search(lowered)))
    week_vs = bool(_WEEK_VS.search(lowered))
    if not week_vs and (journal_words or stop_day or (journal_day and not vetoes and not earnings_group
                                                      and not earnings_alone)):
        add("journal_pack", "journal words" if journal_words else "time words" if day else "stop words",
            day=day or today)
    if _WEEK_VS.search(lowered):
        add("journal_pack", "this week vs last week", day="week")
        add("journal_pack", "this week vs last week", day="last_week")
    if _REGIME_OR_ME.search(lowered):
        add("journal_pack", "regime or me", day=day if day and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) else "month")
        add("mirror_pack", "regime or me")
        add("regime_pack", "regime or me")
    if _WATCH_NEXT.search(lowered):
        add("fundamentals_pack", "watch next session", section="watch")
        add("regime_pack", "watch next session")
        add("night_pack", "watch next session", section="day_review")
    if _OUTCOME.search(lowered):
        add("journal_pack", "outcome words", day=day if day else "month")
        add("mirror_pack", "outcome words")
    if vetoes:
        if day in _WINDOWS or (_VETO_AGG.search(lowered) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day)):
            # P16: a week's or month's vetoes, or their record, is the aggregate by reason; the mirror rides along.
            add("veto_pack", "veto record words", scope=day if day in _WINDOWS else "month")
            add("mirror_pack", "veto record words")
        else:
            add("veto_pack", "veto words", date=day if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) else "")
    # P14: the tape only on market words, session prep, or trade intent the gate does not already carry.
    tape_words = bool(_TAPE.search(lowered) or _PREP.search(lowered)
                      or any(sym in INDEX_SYMBOLS for sym in find_symbols(raw, set(INDEX_SYMBOLS))))
    if tape_words or (market_cue(raw) and not gated and not symbols) or (
            symbols and not gated and _INTENT.search(lowered) and not past):
        add("regime_pack", "market words" if tape_words else "trade intent")
    held, likes = _names(book), _names(liked)
    scope = group_scope(lowered) if group or earnings_alone else ""
    if scope == "book":
        # P16: a question about what he holds reads the book itself too ("my open shorts into earnings").
        add("book_pack", "book scope")
    if (group and earnings_group) or earnings_alone:
        which = next(g for g in group.groups() if g) if group else "book and likes"
        side = _group_side(lowered, which)
        names = [sym for sym in _scope_pool(scope, held, likes, known, earnings_alone=earnings_alone)
                 if sym not in INDEX_SYMBOLS and (not side or known.get(sym, side) == side)]
        if names:
            # The full list: the pack caps it, keeps every book name, and lists the rest by name. Each name
            # carries where it came from, so a Focus name is never read as a position.
            add("earnings_pack", f"earnings across {which}", symbols=names,
                book=[sym for sym in held if sym in names],
                **_origin_args(names, held, likes, side))
    elif group and _NEWS.search(lowered):
        which = next(g for g in group.groups() if g)
        side = _group_side(lowered, which)
        names = [sym for sym in _scope_pool(scope, held, likes, known)
                 if sym not in INDEX_SYMBOLS and (not side or known.get(sym, "") == side)]
        for sym in names[:MAX_SYMBOLS]:
            add("pick_pack", f"news across {which}", symbol=sym, origin=origin_of(sym, held, likes))
    if _BOOK.search(lowered):
        add("book_pack", "book words")
    if _TILT.search(lowered) or stop_day:
        add("tilt_pack", "tilt words" if _TILT.search(lowered) else "stop words")
    # A windowed win rate ("this month") is the journal's count, not the likes-vs-scan mirror.
    if _MIRROR.search(lowered) and not (day and re.search(r"\b(?:win|hit) rate\b", lowered)):
        add("mirror_pack", "record words")
    if _PLAN.search(lowered):
        add("plan_lines", "plan words")
    if _NIGHT.search(lowered):
        add("night_pack", "night words")
    if _RECAP.search(lowered) or (_RECAP_SOFT.search(lowered) and not _RECORD.search(lowered)):
        add("recaps_pack", "recap words", days=10, section="all")
    if _FEEL.search(lowered) and not any(request.name == "journal_pack" for request in wanted):
        add("journal_pack", "feelings words", day=day or "week")
    if _FUND.search(lowered) and not _WATCH_NEXT.search(lowered):
        add("fundamentals_pack", "fundamentals words", section="all")
    elif any(request.name == "gate_pack" for request in wanted):
        # P15b: a pre-trade check sees today's brief: the bottom line and the playbook (at most 8 rows).
        add("fundamentals_pack", "pre-trade: today's brief", section="compact")
    if _RECALL.search(lowered):
        add("recall", "memory words", query=raw[:200])
    return sorted(wanted, key=lambda request: request.priority)


def _group_side(lowered: str, which: str) -> str:
    """The side a group names: its own word ("shorts"), else the one side word in the question ("my focus longs")."""
    if which.startswith("long") or which.startswith("short"):
        return "LONG" if which.startswith("long") else "SHORT"
    longs, shorts = bool(re.search(r"\blongs\b", lowered)), bool(re.search(r"\bshorts\b", lowered))
    return "LONG" if longs and not shorts else "SHORT" if shorts and not longs else ""


def group_scope(lowered: str) -> str:
    """P16: "book" for a group he holds, "focus" for a group he watches, "both" when he names both, else ""."""
    book, focus = bool(_BOOK_SCOPE.search(lowered)), bool(_FOCUS_SCOPE.search(lowered))
    return "both" if book and focus else "book" if book else "focus" if focus else ""


def _scope_pool(scope: str, held: list[str], likes: list[str], known: Mapping[str, str], *,
                earnings_alone: bool = False) -> list[str]:
    """The names a group question covers, in order: the book only for a book group (never Focus), the likes then
    the sided Focus names (not held) for a Focus group, everything for both or no scope word."""
    sided = [sym for sym, side in known.items() if side]
    if scope == "book":
        pool = list(held)
    elif scope == "focus":
        pool = likes + [sym for sym in sided if sym not in held]
    elif earnings_alone:
        pool = held + likes or sided
    else:
        # Open book, then liked chips, then Focus (the order of ``known``); journal-only names carry no side.
        pool = held + likes + sided
    out: list[str] = []
    for sym in pool:
        if sym not in out:
            out.append(sym)
    return out


def origin_of(sym: str, held: Iterable[str], likes: Iterable[str]) -> str:
    """Where a group name came from: ``book`` (an open position), ``liked`` or ``focus`` (watch names only)."""
    return "book" if sym in set(held) else "liked" if sym in set(likes) else "focus"


def _origin_args(names: list[str], held: list[str], likes: list[str], side: str) -> dict[str, Any]:
    liked = [sym for sym in names if origin_of(sym, held, likes) == "liked"]
    focus = [sym for sym in names if origin_of(sym, held, likes) == "focus"]
    return {"liked": liked, "focus": focus, **({"side": side} if side else {})}


def book_symbols(context_rows: Iterable[Mapping[str, Any]] = ()) -> list[str]:
    """The open book's tickers from the desk context rows, in order."""
    out: list[str] = []
    for row in context_rows or ():
        sym = str(row.get("symbol") or "").strip().upper() if row.get("kind") == "position" else ""
        if sym and sym not in out:
            out.append(sym)
    return out


def _names(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for item in values or ():
        sym = str(item[0] if isinstance(item, (tuple, list)) and item else item or "").strip().upper()
        if sym and sym not in out:
            out.append(sym)
    return out


def known_symbols(
    context_rows: Iterable[Mapping[str, Any]] = (),
    liked: Iterable[Any] = (),
    journal_symbols: Iterable[str] = (),
) -> dict[str, str]:
    """The trader's universe as ``{SYM: side}``, in this order: open book, liked picks, Focus, journal names.

    The order is what a capped group read (``earnings_pack``) keeps first: the book, never the Focus tail.
    """
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
    for row in rows:
        if row.get("kind") == "position":
            put(row.get("symbol"), row.get("direction") or "")
    for item in liked or ():
        if isinstance(item, (tuple, list)) and item:
            put(item[0], item[1] if len(item) > 1 else "")
        else:
            put(item)
    for category in ("swing", "m5"):
        for row in rows:
            if row.get("kind") == "focus" and row.get("category") == category:
                for name in row.get("names") or ():
                    put(name, row.get("side"))
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
                               "regime", "hyp", "mem", "none", "asof", "pos", "acct", "hint", "pack",
                               "night", "brief", "coach"})


def names_subject(question: str, row_id: str) -> bool:
    """True when the question names what a row is about (its ticker or its topic word)."""
    words = set(re.findall(r"[a-z0-9]+", str(question or "").lower()))
    for part in re.split(r"[:_\-]", str(row_id or "").lower()):
        if len(part) >= 2 and part not in _GENERIC_ID_PARTS and not part.isdigit() and part in words:
            return True
    return False
