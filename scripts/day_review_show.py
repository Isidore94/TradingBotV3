"""The Day Review Show: one session told as 6-10 slides (TODO R1).

Pure: no store, no thread, no clock, no Qt. The night slot
(`ai_jobs.day_review_show_night`) asks a local model for a deck and keeps it
only if :func:`verify_show` passes; the desk shows that deck, or the
deterministic :func:`fallback_deck` built from the day pack with a
"facts only" badge.

Rules the verifier holds, and it rejects the WHOLE deck on any breach:

* every `source_id` is in `day_review_pack.allowed_source_ids(pack)`;
* every number written in a title, body or stat label appears verbatim among
  the numbers of the slide's cited sources (or the session date);
* an all-caps word is a ticker the pack names or a fixed desk word;
* no mood word shares a sentence with a result word, and no slide cites a
  mood id beside a trade, read or walk-away id (mood is report-only);
* one slide per kind, except `number` and `trade`.

The model never supplies a stat value: it names the source and writes the
caption, and :func:`stat_value` prints the number from the pack.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "day_review_show_v1"

KINDS: tuple[str, ...] = (
    "open", "tape", "number", "scoreboard", "trade",
    "miss", "read", "lesson", "tomorrow", "close",
)
#: Kinds that may appear more than once in one deck.
REPEATABLE_KINDS = frozenset({"number", "trade"})

#: One deterministic glyph per kind (Segoe UI Emoji on the desk).
KIND_GLYPHS: dict[str, str] = {
    "open": "\U0001F305",       # sunrise
    "tape": "\U0001F4C8",       # chart increasing
    "number": "\U0001F522",     # input numbers
    "scoreboard": "\U0001F3C6", # trophy
    "trade": "\U0001F4B5",      # dollar banknote
    "miss": "\U0001F3AF",       # direct hit
    "read": "\U0001F50D",       # magnifying glass
    "lesson": "\U0001F4A1",     # light bulb
    "tomorrow": "\U0001F9ED",   # compass
    "close": "\U0001F3C1",      # chequered flag
}

MIN_SLIDES = 6
MAX_SLIDES = 10
MAX_TITLE = 60
MAX_SLIDE_TITLE = 48
MAX_BODY = 220
MAX_STAT_VALUE = 16
MAX_STAT_LABEL = 40
MIN_SOURCES = 1
MAX_SOURCES = 4
#: Fallback-only bullet lines (the report card's six lines, the truth lines).
MAX_LINES = 8
MAX_LINE = 200

#: What the model returns. The stat is a SOURCE and a caption, never a value.
MODEL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "slides"],
    "properties": {
        "title": {"type": "string", "maxLength": MAX_TITLE},
        "slides": {
            "type": "array",
            "minItems": MIN_SLIDES,
            "maxItems": MAX_SLIDES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "kind", "title", "body", "stat_source_id", "stat_label", "source_ids",
                ],
                "properties": {
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "title": {"type": "string", "maxLength": MAX_SLIDE_TITLE},
                    "body": {"type": "string", "maxLength": MAX_BODY},
                    "stat_source_id": {"type": "string", "maxLength": 160},
                    "stat_label": {"type": "string", "maxLength": MAX_STAT_LABEL},
                    "source_ids": {
                        "type": "array",
                        "minItems": MIN_SOURCES,
                        "maxItems": MAX_SOURCES,
                        "items": {"type": "string", "maxLength": 160},
                    },
                },
            },
        },
    },
}

INSTRUCTIONS = (
    "Turn this ONE trading session into a short slide show of 6 to 10 slides. "
    "Each slide has a kind (open, tape, number, scoreboard, trade, miss, read, "
    "lesson, tomorrow, close); use each kind at most once, except number and "
    "trade. Cite 1 to 4 ids per slide, copied exactly from allowed_source_ids. "
    "Never write a number in a title, body or stat_label unless it is copied "
    "character for character from a source that slide cites; to show a number, "
    "set stat_source_id to the source that holds it and write only a short "
    "stat_label - the desk prints the number. Leave stat_source_id empty for no "
    "stat. Name only tickers the pack names. Do not write in capitals. Never "
    "put a feeling next to a result: mood is reported on its own slide or not "
    "at all. You may not grade, rate or measure anything. "
    "previous_story is last night's verified story: context only, never a source."
)

#: All-caps words that are not tickers.
DESK_WORDS = frozenset({
    "AI", "ATR", "AVWAP", "VWAP", "EOD", "HOD", "LOD", "PT", "ET", "AM", "PM",
    "OK", "PNL", "USD", "CAD", "SMA", "EMA", "RTH", "FOMC", "CPI", "PPI", "GDP",
    "FOMO", "NYSE", "TFSA", "OPEX", "SPY", "DTE", "IV",
})

MOOD_WORDS = frozenset({
    "calm", "focused", "rushed", "fomo", "tilted", "bored", "tired", "confident",
    "anxious", "nervous", "angry", "frustrated", "happy", "sad", "fear",
    "fearful", "greedy", "excited", "stressed", "scared", "upset", "mood",
    "felt", "feeling", "emotional",
})
RESULT_WORDS = frozenset({
    "win", "wins", "won", "winner", "winning", "loss", "losses", "lost", "loser",
    "losing", "profit", "profits", "pnl", "green", "red", "right", "wrong",
    "stopped", "gain", "gains", "paid", "made", "result", "results",
})
#: Pack parts that hold a result; a mood id may not sit on a slide beside one.
_RESULT_PARTS = ("trades", "reads", "walkaway")

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CAPS = re.compile(r"\b[A-Z][A-Z.]{1,5}\b")
_WORD = re.compile(r"[a-z&]+")
_SENTENCE = re.compile(r"[.!?;\n]+")


class ShowRejected(ValueError):
    """The model's deck broke a rule; the whole deck is refused."""


# ---------------------------------------------------------------------------
# the pack, read
# ---------------------------------------------------------------------------
def source_rows(pack: Mapping[str, Any]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """`source_id -> (part, row)` for every citable row of the pack."""
    out: dict[str, tuple[str, Mapping[str, Any]]] = {}
    body = pack or {}

    def _add(part: str, row: Any) -> None:
        if isinstance(row, Mapping):
            sid = str(row.get("source_id") or "").strip()
            if sid and sid not in out:
                out[sid] = (part, row)

    for part in ("trader_said", "environment", "measured", "internals", "reads", "congruence"):
        for row in body.get(part) or ():
            _add(part, row)
    forecast = body.get("forecast")
    if isinstance(forecast, Mapping):
        for cell in (forecast.get("fields") or {}).values():
            _add("forecast", cell)
    walkaway = body.get("walkaway")
    if isinstance(walkaway, Mapping):
        for row in walkaway.get("top") or ():
            _add("walkaway", row)
    _add("skill", body.get("skill"))
    card = body.get("report_card")
    if isinstance(card, Mapping):
        for line in card.get("lines") or ():
            _add("report_card", line)
    trades = body.get("trades")
    if isinstance(trades, Mapping):
        for row in trades.get("rows") or ():
            _add("trades", row)
    mood = body.get("mood")
    if isinstance(mood, Mapping):
        for row in mood.get("recorded") or ():
            _add("mood", row)
    return out


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _fits(text: str) -> str | None:
    text = str(text or "").strip()
    return text if text and len(text) <= MAX_STAT_VALUE else None


def stat_value(pack: Mapping[str, Any], source_id: str) -> str | None:
    """The one number (or short word) the desk prints for a cited source.

    None when that source holds nothing printable; a mood is never a stat.
    """
    found = source_rows(pack).get(str(source_id or "").strip())
    if found is None:
        return None
    part, row = found
    if part == "trades":
        pnl = _float(row.get("net_pnl"))
        if pnl is None:
            pnl = _float(row.get("realized_pnl"))
        return _fits(f"{pnl:+,.2f}") if pnl is not None else None
    if part == "measured":
        change = _float(row.get("change_pct"))
        return _fits(f"{change:+.2f}%") if change is not None else None
    if part == "walkaway":
        ran = _float(row.get("ran_after_pct"))
        return _fits(f"{ran:+.1f}%") if ran is not None else None
    if part == "reads":
        return _fits(row.get("verdict"))
    if part == "report_card":
        n = row.get("n")
        return _fits(str(n)) if isinstance(n, int) and not isinstance(n, bool) else None
    if part == "internals":
        common = ((row.get("context") or {}).get("common") or {}) if isinstance(row.get("context"), Mapping) else {}
        value = common.get("internals") if isinstance(common, Mapping) else None
        if value is None:
            value = row.get("internals")
        return _fits(value) if isinstance(value, (str, int, float)) and not isinstance(value, bool) else None
    if part == "environment":
        return _fits(row.get("label"))
    if part == "trader_said":
        return _fits(row.get("direction"))
    return None


def pack_tickers(pack: Mapping[str, Any]) -> frozenset[str]:
    """Every symbol the pack names under a `symbol`/`ticker`/`underlying` key."""
    found: set[str] = set()

    def _walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in ("symbol", "ticker", "underlying") and isinstance(item, str):
                    if item.strip():
                        found.add(item.strip().upper())
                else:
                    _walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)

    _walk(pack or {})
    return frozenset(found)


# ---------------------------------------------------------------------------
# the verifier
# ---------------------------------------------------------------------------
def _string(value: Any, limit: int, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ShowRejected(f"{where} must be a string")
    if len(value) > limit:
        raise ShowRejected(f"{where} is longer than {limit} characters")
    if not allow_empty and not value.strip():
        raise ShowRejected(f"{where} is empty")
    return value


def _numbers(text: str) -> list[str]:
    return _NUMBER.findall(re.sub(r"(?<=\d),(?=\d{3})", "", str(text or "")))


def _check_shape(reply: Any) -> dict[str, Any]:
    if not isinstance(reply, Mapping):
        raise ShowRejected("the deck was not an object")
    extra = sorted(set(reply) - {"title", "slides"})
    if extra:
        raise ShowRejected(f"the deck carries forbidden field(s): {', '.join(extra)}")
    title = _string(reply.get("title"), MAX_TITLE, "title")
    slides = reply.get("slides")
    if not isinstance(slides, list) or not MIN_SLIDES <= len(slides) <= MAX_SLIDES:
        raise ShowRejected(f"a deck has {MIN_SLIDES} to {MAX_SLIDES} slides")
    allowed_keys = set(MODEL_JSON_SCHEMA["properties"]["slides"]["items"]["properties"])
    out: list[dict[str, Any]] = []
    for index, slide in enumerate(slides):
        where = f"slides[{index}]"
        if not isinstance(slide, Mapping):
            raise ShowRejected(f"{where} was not an object")
        if set(slide) != allowed_keys:
            raise ShowRejected(f"{where} must carry exactly {', '.join(sorted(allowed_keys))}")
        kind = slide.get("kind")
        if kind not in KINDS:
            raise ShowRejected(f"{where}.kind {kind!r} is not a slide kind")
        ids = slide.get("source_ids")
        if not isinstance(ids, list) or not MIN_SOURCES <= len(ids) <= MAX_SOURCES:
            raise ShowRejected(f"{where} cites {MIN_SOURCES} to {MAX_SOURCES} sources")
        out.append({
            "kind": kind,
            "title": _string(slide.get("title"), MAX_SLIDE_TITLE, f"{where}.title"),
            "body": _string(slide.get("body"), MAX_BODY, f"{where}.body"),
            "stat_source_id": _string(
                slide.get("stat_source_id"), 160, f"{where}.stat_source_id", allow_empty=True
            ).strip(),
            "stat_label": _string(
                slide.get("stat_label"), MAX_STAT_LABEL, f"{where}.stat_label", allow_empty=True
            ),
            "source_ids": [
                _string(item, 160, f"{where}.source_ids", allow_empty=False).strip()
                for item in ids
            ],
        })
    return {"title": title, "slides": out}


def _check_mood(text: str, where: str) -> None:
    for sentence in _SENTENCE.split(str(text or "").lower()):
        words = set(_WORD.findall(sentence))
        if "$" in sentence:
            words.add("$")
        mood = words & MOOD_WORDS
        result = (words & RESULT_WORDS) | ({"$"} & words)
        if mood and result:
            raise ShowRejected(
                f"{where} puts a mood word ({sorted(mood)[0]}) next to a result "
                f"({sorted(result)[0]}); mood is report-only"
            )


def verify_show(reply: Any, pack: Mapping[str, Any]) -> dict[str, Any]:
    """The model's deck materialized with pack stats, or `ShowRejected`.

    Whole-reject: any breach refuses the deck; nothing is trimmed or repaired.
    """
    import day_review_pack

    deck = _check_shape(reply)
    allowed = set(day_review_pack.allowed_source_ids(pack))
    rows = source_rows(pack)
    tickers = pack_tickers(pack)
    session = str((pack or {}).get("session_date") or "")
    date_numbers = set(_numbers(session))
    _check_mood(deck["title"], "title")
    for word in _CAPS.findall(deck["title"]):
        if word.strip(".") not in tickers and word.strip(".") not in DESK_WORDS:
            raise ShowRejected(f"title names {word!r}, which the pack does not name")
    for number in _numbers(deck["title"]):
        if number not in date_numbers:
            raise ShowRejected(f"the deck title writes {number}, which no source holds")
    seen_kinds: set[str] = set()
    slides: list[dict[str, Any]] = []
    for index, slide in enumerate(deck["slides"]):
        where = f"slide {index + 1} ({slide['kind']})"
        kind = slide["kind"]
        if kind in seen_kinds and kind not in REPEATABLE_KINDS:
            raise ShowRejected(f"{where} repeats kind {kind!r}; only number and trade repeat")
        seen_kinds.add(kind)
        ids = slide["source_ids"]
        outside = [item for item in ids if item not in allowed]
        if outside:
            raise ShowRejected(f"{where} cites id(s) the pack does not carry: {', '.join(outside)}")
        if len(set(ids)) != len(ids):
            raise ShowRejected(f"{where} cites one id twice")
        parts = {rows[item][0] for item in ids if item in rows}
        if "mood" in parts and parts & set(_RESULT_PARTS):
            raise ShowRejected(f"{where} cites a mood beside a result; mood is report-only")
        cited_numbers = set(date_numbers)
        for item in ids:
            if item in rows:
                cited_numbers.update(
                    _numbers(json.dumps(rows[item][1], sort_keys=True, default=str))
                )
        text_fields = (slide["title"], slide["body"], slide["stat_label"])
        for text in text_fields:
            for number in _numbers(text):
                if number not in cited_numbers:
                    raise ShowRejected(
                        f"{where} writes {number}, which none of its cited sources holds"
                    )
            for word in _CAPS.findall(text):
                bare = word.strip(".")
                if bare not in tickers and bare not in DESK_WORDS:
                    raise ShowRejected(f"{where} names {word!r}, which the pack does not name")
            _check_mood(text, where)
        stat = None
        if slide["stat_source_id"]:
            if slide["stat_source_id"] not in ids:
                raise ShowRejected(f"{where} takes its stat from a source it does not cite")
            if rows.get(slide["stat_source_id"], ("",))[0] == "mood":
                raise ShowRejected(f"{where} makes a mood a number; mood is report-only")
            value = stat_value(pack, slide["stat_source_id"])
            if value is None:
                raise ShowRejected(f"{where} names a stat source with no number in it")
            if not slide["stat_label"].strip():
                raise ShowRejected(f"{where} has a stat with no caption")
            stat = {
                "value": value,
                "label": slide["stat_label"],
                "source_id": slide["stat_source_id"],
            }
        elif slide["stat_label"].strip():
            raise ShowRejected(f"{where} captions a stat it does not name")
        slides.append({
            "kind": kind,
            "title": slide["title"],
            "body": slide["body"],
            "stat": stat,
            "source_ids": list(ids),
        })
    return {"title": deck["title"], "slides": slides}


# ---------------------------------------------------------------------------
# the fallback deck: facts only, every day
# ---------------------------------------------------------------------------
def _clip(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _slide(kind: str, title: str, body: str, *, stat=None, source_ids=(), lines=()) -> dict[str, Any]:
    out = {
        "kind": kind,
        "title": _clip(title, MAX_SLIDE_TITLE),
        "body": _clip(body, MAX_BODY),
        "stat": stat,
        "source_ids": [str(item) for item in source_ids if item][:MAX_SOURCES],
    }
    kept = [_clip(line, MAX_LINE) for line in lines if str(line or "").strip()][:MAX_LINES]
    if kept:
        out["lines"] = kept
    return out


def spy_path(bars: Sequence[Mapping[str, Any]] | None) -> dict[str, float] | None:
    """Open, high, low, close and change of a session tape; None when unread."""
    rows = [bar for bar in bars or () if isinstance(bar, Mapping)]
    if not rows:
        return None
    first, last = _float(rows[0].get("open")), _float(rows[-1].get("close"))
    highs = [value for bar in rows if (value := _float(bar.get("high"))) is not None]
    lows = [value for bar in rows if (value := _float(bar.get("low"))) is not None]
    if first is None or last is None or not highs or not lows or first == 0:
        return None
    return {
        "open": first, "high": max(highs), "low": min(lows), "close": last,
        "change_pct": (last - first) / first * 100.0,
    }


def fallback_deck(
    pack: Mapping[str, Any] | None,
    *,
    session_date: str = "",
    truth_lines: Iterable[str] = (),
    alerts_line: str = "",
    spy_bars: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """A deterministic 7-10 slide deck from the day pack, so Show works every day.

    `truth_lines`, `alerts_line` and `spy_bars` are Day Review payload keys the
    pack does not carry; they are handed in by the worker that already read them.
    """
    body = pack if isinstance(pack, Mapping) else {}
    session = str(body.get("session_date") or session_date or "")[:10]
    slides: list[dict[str, Any]] = []

    label_row = next(
        (row for row in body.get("environment") or ()
         if isinstance(row, Mapping) and row.get("kind") == "d1_label"),
        None,
    )
    slides.append(_slide(
        "open", f"Your day, {session}" if session else "Your day",
        f"Day type: {label_row.get('label')}." if label_row else "Day type: unknown.",
        source_ids=[label_row.get("source_id")] if label_row else (),
    ))

    path = spy_path(spy_bars)
    spy_cell = next(
        (row for row in body.get("measured") or ()
         if isinstance(row, Mapping) and str(row.get("symbol") or "").upper() == "SPY"),
        None,
    )
    internals = [row for row in body.get("internals") or () if isinstance(row, Mapping)]
    tape_ids = [spy_cell.get("source_id")] if spy_cell else []
    if path:
        tape_text = (
            f"SPY opened {path['open']:.2f}, ranged {path['low']:.2f} to "
            f"{path['high']:.2f} and closed {path['close']:.2f}."
        )
        tape_stat = {"value": f"{path['change_pct']:+.2f}%", "label": "SPY open to close", "source_id": ""}
    elif spy_cell and _float(spy_cell.get("change_pct")) is not None:
        tape_text = f"SPY closed {_float(spy_cell.get('close')) or 0:.2f}."
        tape_stat = {
            "value": stat_value(body, spy_cell.get("source_id")) or "",
            "label": "SPY on the day",
            "source_id": str(spy_cell.get("source_id") or ""),
        }
    else:
        tape_text, tape_stat = "The SPY tape was not read.", None
    if internals:
        last = internals[-1]
        reading = stat_value(body, last.get("source_id"))
        if reading:
            tape_text += f" Internals at {str(last.get('kind') or 'the last mark')}: {reading}."
        tape_ids.append(last.get("source_id"))
    slides.append(_slide("tape", "The tape", tape_text, stat=tape_stat, source_ids=tape_ids))

    card = body.get("report_card")
    card_lines = [
        line for line in (card.get("lines") if isinstance(card, Mapping) else ()) or ()
        if isinstance(line, Mapping)
    ]
    slides.append(_slide(
        "scoreboard", "Report card",
        "" if card_lines else "No report card for this session.",
        source_ids=[line.get("source_id") for line in card_lines],
        lines=[str(line.get("text") or "") for line in card_lines],
    ))

    truth = [str(line) for line in truth_lines or () if str(line or "").strip()]
    slides.append(_slide(
        "number", "Your last sessions",
        "" if truth else "The truth lines were not read.",
        lines=truth,
    ))
    slides.append(_slide("number", "Alerts", str(alerts_line or "Alerts: unknown.")))

    walkaway = body.get("walkaway") if isinstance(body.get("walkaway"), Mapping) else {}
    counts = walkaway.get("counts") if isinstance(walkaway.get("counts"), Mapping) else {}
    top = [row for row in walkaway.get("top") or () if isinstance(row, Mapping)]
    count_text = ", ".join(
        f"{str(name).replace('_', ' ')} {int(value)}"
        for name, value in counts.items()
        if isinstance(value, int) and not isinstance(value, bool)
    )
    miss_stat = None
    if top and stat_value(body, top[0].get("source_id")):
        miss_stat = {
            "value": stat_value(body, top[0].get("source_id")),
            "label": f"{top[0].get('symbol') or 'top name'} ran after",
            "source_id": str(top[0].get("source_id") or ""),
        }
    slides.append(_slide(
        "miss", "Walked away from",
        f"Counts: {count_text}." if count_text else "The walk-away tables were not read.",
        stat=miss_stat,
        source_ids=[row.get("source_id") for row in top],
    ))

    reads = [row for row in body.get("reads") or () if isinstance(row, Mapping)]
    trades = body.get("trades") if isinstance(body.get("trades"), Mapping) else {}
    trade_rows = [row for row in trades.get("rows") or () if isinstance(row, Mapping)]
    for row in trade_rows[: MAX_SLIDES - len(slides) - (2 if reads else 1)]:
        sid = str(row.get("source_id") or "")
        value = stat_value(body, sid)
        slides.append(_slide(
            "trade",
            f"{row.get('symbol') or 'Trade'} {str(row.get('direction') or '').lower()}".strip(),
            f"Status: {row.get('status') or 'unknown'}.",
            stat={"value": value, "label": "net", "source_id": sid} if value else None,
            source_ids=[sid],
        ))

    if reads:
        verdicts: dict[str, int] = {}
        for row in reads:
            key = str(row.get("verdict") or "unknown")
            verdicts[key] = verdicts.get(key, 0) + 1
        slides.append(_slide(
            "read", "Your reads",
            "; ".join(f"{name} {count}" for name, count in sorted(verdicts.items())) + ".",
            source_ids=[row.get("source_id") for row in reads],
        ))

    slides.append(_slide(
        "close", "Facts only",
        "The night's story for this deck is not ready, so these are the desk's own facts.",
    ))
    return {"title": _clip(f"Day Review {session}".strip(), MAX_TITLE), "slides": slides}


# ---------------------------------------------------------------------------
# the stored file and the desk's choice
# ---------------------------------------------------------------------------
def show_path(session_date: str, *, root: Path | None = None) -> Path:
    """`<DAY_REVIEW_DIR>/shows/<date>.json`; a path only, nothing is opened."""
    import day_review_pack

    base = Path(root) if root is not None else day_review_pack.default_root()
    return base / "shows" / f"{str(session_date or '').strip()[:10]}.json"


def inputs_hash(pack: Mapping[str, Any], narration: Mapping[str, Any] | None) -> str:
    """Over the pack's own hash and the narration's words: what the deck read."""
    body = {
        "pack": str((pack or {}).get("inputs_hash") or ""),
        "narration": dict(narration) if isinstance(narration, Mapping) else None,
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def desk_deck(
    stored: Mapping[str, Any] | None,
    pack: Mapping[str, Any] | None,
    *,
    session_date: str = "",
    truth_lines: Iterable[str] = (),
    alerts_line: str = "",
    spy_bars: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """What the Show button opens: the verified model deck, or the fallback.

    The stored deck is re-verified against the current pack and must have been
    written for that pack; stats are re-printed from the pack. Anything else
    shows the fallback deck with `facts_only` set and the reason named.
    """
    session = str(session_date or (pack or {}).get("session_date") or "")[:10]
    fallback_reason = ""
    if not isinstance(stored, Mapping):
        fallback_reason = "no show was written for this session"
    elif not isinstance(pack, Mapping) or not pack:
        fallback_reason = "no day facts to check the show against"
    elif str(stored.get("session_date") or "")[:10] != session:
        fallback_reason = "the show was written for another session"
    elif str(stored.get("pack_hash") or "") != str(pack.get("inputs_hash") or ""):
        fallback_reason = "the day facts changed since the show was written"
    else:
        try:
            show = stored.get("show") or {}
            reply = {
                "title": show.get("title"),
                "slides": [
                    {
                        "kind": slide.get("kind"),
                        "title": slide.get("title"),
                        "body": slide.get("body"),
                        "stat_source_id": str((slide.get("stat") or {}).get("source_id") or ""),
                        "stat_label": str((slide.get("stat") or {}).get("label") or ""),
                        "source_ids": list(slide.get("source_ids") or ()),
                    }
                    for slide in show.get("slides") or ()
                    if isinstance(slide, Mapping)
                ],
            }
            deck = verify_show(reply, pack)
        except (ShowRejected, AttributeError, TypeError) as exc:
            fallback_reason = f"the show failed its checks: {exc}"
        else:
            return {
                "deck": deck,
                "facts_only": False,
                "reason": "",
                "model": str(stored.get("model") or ""),
            }
    return {
        "deck": fallback_deck(
            pack, session_date=session, truth_lines=truth_lines,
            alerts_line=alerts_line, spy_bars=spy_bars,
        ),
        "facts_only": True,
        "reason": fallback_reason,
        "model": "",
    }


__all__ = [
    "DESK_WORDS",
    "INSTRUCTIONS",
    "KINDS",
    "KIND_GLYPHS",
    "MAX_SLIDES",
    "MIN_SLIDES",
    "MODEL_JSON_SCHEMA",
    "SCHEMA",
    "ShowRejected",
    "desk_deck",
    "fallback_deck",
    "inputs_hash",
    "pack_tickers",
    "show_path",
    "source_rows",
    "spy_path",
    "stat_value",
    "verify_show",
]
