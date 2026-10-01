"""P18 journal mode: tell the trader's self talk from a question, tag its mood, and say "Noted" in one line.

Pure and Qt-free. ``classify`` is deterministic: a message that opens like a question (wh-/how/should/is/can...,
or a tool cue like "show me") or plan talk ("I stop after two losses", which plan inference reads) is a QUESTION;
otherwise first-person self talk (I/I'm/feeling/annoyed/chased/fomo...) is a STATEMENT, kept as a
``journal_entries`` row; anything else is a QUESTION, as before P18. A first-person sentence with a question inside
("?", a wh-word, "should I", "talk me...") is a QUESTION, unless it carries a mood word: then it is stored AND
answered. ``/journal on`` makes every message a statement. Mood tags are the desk's ``trader_state_tags`` codes,
matched by fixed words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
#: app_state key for ``/journal on|off`` ("on" forces journal mode; anything else = auto).
MODE_KEY = "journal:mode"
#: A closed trade this recent is named in the entry and the "Noted" line.
LAST_TRADE_MINUTES = 30

_QUESTION_START = re.compile(
    r"^\s*(?:who|whom|whose|what|what's|whats|when|where|which|why|how|should|shall|is|isn't|are|aren't|"
    r"can|could|would|will|do|does|did|was|were|am|may|might|any)\b",
    re.IGNORECASE,
)
#: A request for a tool or a pack: a question even without "?".
_TOOL_CUE = re.compile(
    r"\b(?:show me|tell me|give me|pull up|look up|check|compare|explain|list|remind me|what's|whats)\b",
    re.IGNORECASE,
)
_TOOL_START = re.compile(r"^\s*(?:show|tell|give|pull|look|check|compare|explain|list|remind|find|search)\b",
                         re.IGNORECASE)
#: Plan talk ("I stop after two losses", "from now on...") goes to the model, so plan inference sees the turn.
_PLAN_CUE = re.compile(
    r"\b(?:(?<!should )(?<!can )(?<!do )(?<!would )(?<!could )i (?:always|never|only|stop|quit|won't|will no longer|must)|my rule|rules?:|from now on|going forward|"
    r"no new entries|no more trades)\b",
    re.IGNORECASE,
)
_SELF_TALK = re.compile(
    r"\b(?:i|i'm|im|i've|ive|i'd|i'll|me|myself|feeling|feel|felt|annoyed|chased|chasing|tempted|bored|tired|"
    r"revenge|fomo|tilt|tilted|tilting|frustrated|angry|mad|nervous|anxious|impatient|greedy|rushed|rushing|"
    r"calm|focused|confident|exhausted)\b",
    re.IGNORECASE,
)

#: Fixed words per ``state_tags_v1`` code. A code not in the loaded vocabulary is never emitted.
TAG_WORDS: dict[str, tuple[str, ...]] = {
    "calm": ("calm", "settled", "relaxed", "patient"),
    "focused": ("focused", "focus", "dialed in", "locked in"),
    "rushed": ("rushed", "rushing", "hurried", "hurry", "late to", "behind the tape"),
    "fomo": ("fomo", "chased", "chasing", "chase", "tempted", "missing out", "missed it", "fear of missing"),
    "tilted": ("tilt", "tilted", "tilting", "revenge", "annoyed", "angry", "mad", "pissed", "frustrated",
               "rattled", "furious"),
    "bored": ("bored", "boring", "nothing to do"),
    "tired": ("tired", "exhausted", "sleepy", "worn out", "no sleep", "drained"),
    "confident": ("confident", "sure of", "conviction"),
}


@dataclass(frozen=True)
class Kind:
    """``statement`` = store as a journal entry; ``asks`` = also answer it with the model."""

    statement: bool
    asks: bool


#: A question inside a sentence: a wh-word anywhere, or "should I", "is that", "talk me", "walk me through"...
_QUESTION_INSIDE = re.compile(
    r"\b(?:what|what's|whats|when|where|which|why|how|who)\b"
    r"|\b(?:should|can|could|would|will|do|did|does|is|are|was|were|am)\s+(?:i|we|it|that|this|there|you)\b"
    r"|\b(?:talk me|walk me|help me|teach me)\b",
    re.IGNORECASE,
)
_FEELING = re.compile(r"\b(?:feeling|feel|felt|nervous|anxious|impatient|greedy|scared|stressed)\b", re.IGNORECASE)


def classify(text: str, *, forced: bool = False) -> Kind:
    """Deterministic: QUESTION (answer only), STATEMENT (store, one-line reply) or both.

    A sentence with a question inside is a QUESTION unless it also carries a mood word: then it is self talk
    with a question, stored AND answered."""
    words = " ".join(str(text or "").split())
    asks = "?" in words or bool(_TOOL_CUE.search(words) or _QUESTION_INSIDE.search(words))
    if forced:
        return Kind(statement=bool(words), asks=asks)
    if not words or _QUESTION_START.match(words) or _TOOL_START.match(words) or _PLAN_CUE.search(words):
        return Kind(statement=False, asks=True)
    if not _SELF_TALK.search(words):
        return Kind(statement=False, asks=True)
    if asks and not (mood_tags(words) or _FEELING.search(words)):
        return Kind(statement=False, asks=True)  # first person, but a question about trading, not self talk
    return Kind(statement=True, asks=asks)


def vocabulary_codes() -> tuple[tuple[str, ...], int | None]:
    """The desk's state-tag codes and version; the fixed list (version unknown) when the vocabulary is unreadable."""
    try:
        import trader_state_tags

        book = trader_state_tags.load_vocabulary()
        return tuple(str(entry["code"]) for entry in book["entries"]), int(book["vocab_version"])
    except Exception:  # noqa: BLE001 - an unreadable vocabulary still tags with the shipped v1 codes
        return tuple(TAG_WORDS), None


def mood_tags(text: str, codes: Iterable[str] | None = None) -> tuple[str, ...]:
    """The vocabulary codes whose fixed words appear in ``text``, in vocabulary order."""
    lowered = " " + " ".join(str(text or "").lower().replace("’", "'").split()) + " "
    allowed = tuple(codes) if codes is not None else tuple(TAG_WORDS)
    found = []
    for code in allowed:
        for word in TAG_WORDS.get(code, ()):
            if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lowered):
                found.append(code)
                break
    return tuple(found)


def time_bucket(moment: datetime) -> str:
    """The ET 30-minute bucket, ``HH:MM``."""
    local = moment.astimezone(ET)
    return f"{local.hour:02d}:{0 if local.minute < 30 else 30:02d}"


def last_closed_trade(journal: Path | str, now: datetime, minutes: int = LAST_TRADE_MINUTES) -> dict[str, Any] | None:
    """The trader's most recent trade closed in the last ``minutes`` (journal ``mode=ro``); None when none/unreadable."""
    from mentor_packs import journal_read

    moment = now if now.tzinfo else now.astimezone()
    since = (moment.astimezone(ET).date() - timedelta(days=1)).isoformat()
    try:
        trades = journal_read.read_trades(journal, since=since)
    except Exception:  # noqa: BLE001 - an unreadable journal means "no trade known", never a raise
        return None
    best: tuple[datetime, Mapping[str, Any]] | None = None
    for trade in trades:
        if str(trade.get("status") or "").upper() != "CLOSED":
            continue
        closed = journal_read.parse_time(trade.get("closed_at"))
        if closed is None or not (moment - timedelta(minutes=minutes) <= closed <= moment):
            continue
        if best is None or closed > best[0]:
            best = (closed, trade)
    if best is None:
        return None
    trade = best[1]
    symbol = (journal_read.underlying(trade.get("symbol")) if journal_read.is_option(trade)
              else str(trade.get("symbol") or "").upper())
    pnl = journal_read.num(trade.get("net_pnl_usd"))
    if pnl is None:
        pnl = journal_read.num(trade.get("net_pnl"))
    word = "close" if pnl is None else ("stop" if pnl < 0 else ("win" if pnl > 0 else "scratch"))
    return {"trade_id": str(trade.get("trade_id") or ""), "symbol": symbol, "pnl": pnl,
            "text": f"the {symbol} {word}"}


def tape_state(context_rows: Iterable[Mapping[str, Any]]) -> tuple[str, str]:
    """(regime label text, last ``ctx:today`` line) from the context pack rows; "" when not there."""
    regime = today = ""
    for row in context_rows or ():
        if row.get("id") == "ctx:regime":
            regime = str(row.get("text") or "")
        elif row.get("id") == "ctx:today":
            today = str(row.get("text") or "")
    return regime, today


def entry_fields(text: str, now: datetime, *, context_rows: Iterable[Mapping[str, Any]] = (),
                 journal: Path | str | None = None, forced: bool = False, asks: bool = False) -> dict[str, Any]:
    """Every column of one ``journal_entries`` row (worker thread: reads the journal ``mode=ro``)."""
    moment = now if now.tzinfo else now.astimezone()
    codes, version = vocabulary_codes()
    regime, today = tape_state(context_rows)
    trade = last_closed_trade(journal, moment) if journal is not None else None
    local = moment.astimezone(ET)
    return {
        "ts_utc": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "day_et": local.date().isoformat(),
        "text": " ".join(str(text or "").split()),
        "mood_tags": list(mood_tags(text, codes)),
        "vocab_version": version,
        "regime": regime,
        "tape": today,
        "last_trade_id": (trade or {}).get("trade_id") or "",
        "last_trade_text": (trade or {}).get("text") or "",
        "time_bucket_et": time_bucket(moment),
        "weekday": local.strftime("%a"),
        "asks": bool(asks),
        "forced": bool(forced),
    }


def noted_line(fields: Mapping[str, Any], entry_id: int | None) -> str:
    """The one-line reply: "Noted, 10:42, after the TWLO stop (tilted)." No analysis."""
    if entry_id is None:
        return "That journal line was NOT saved (the chat store failed). Say it again."
    stamp = datetime.fromisoformat(str(fields["ts_utc"])).astimezone(ET).strftime("%H:%M")
    after = f", after {fields['last_trade_text']}" if fields.get("last_trade_text") else ""
    tags = f" ({', '.join(fields['mood_tags'])})" if fields.get("mood_tags") else ""
    return f"Noted, {stamp}{after}{tags}. [jrn:{fields['day_et']}:entry:{entry_id}]"


def entries_text(entries: Iterable[Mapping[str, Any]], day: str) -> str:
    """``/journal``: today's entries, one line each, with tags."""
    lines = [f"**Journal {day}**"]
    for row in entries:
        tags = ", ".join(row.get("mood_tags") or ()) or "no tag"
        stamp = datetime.fromisoformat(str(row["ts_utc"])).astimezone(ET).strftime("%H:%M")
        after = f", after {row['last_trade_text']}" if row.get("last_trade_text") else ""
        lines.append(f"- {stamp}{after} ({tags}): {row['text']} [jrn:{day}:entry:{row['id']}]")
    if len(lines) == 1:
        lines.append("Nothing journaled today. Just talk: \"I'm annoyed, I chased that\" is kept.")
    return "\n".join(lines)
