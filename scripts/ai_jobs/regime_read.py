"""Night slot `regime_read` (S17 half 2): the local model reads the regime table.

Runs after `market_story_narration`. Input: the regime table's last
`WINDOW_SESSIONS` sessions for the indexes, the latest structure facts per
index, and the trader's structural regime journal. Output: one bounded
paragraph whose every timeframe word, date, number and regime word is checked
against those inputs. The checks reuse the day story's verifier
(`day_review_narration._validate`, `_check_sources`, `NarrationRejected`); a
breach rejects the WHOLE read, writes nothing and the slot is `degraded`, so
the last verified read stays.

The AI reads, it does not decide: nothing here feeds a detector, a score, an
alert, Focus, the queue or `review_policy.json`.

Output: `REGIME_READS_DIR/<session>.json`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

_log = logging.getLogger(__name__)

PROMPT_VERSION = "regime_read_v1"
SCHEMA = "regime_read_v1"
#: Sessions of the table the model reads.
WINDOW_SESSIONS = 20
MAX_PARAGRAPH = 900
MAX_SENTENCES = 6
MAX_SOURCES = 24
RESERVE_MINUTES = 15.0

READ_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["paragraph", "sources"],
    "properties": {
        "paragraph": {"type": "string", "maxLength": MAX_PARAGRAPH},
        "sources": {
            "type": "array",
            "maxItems": MAX_SOURCES,
            "items": {"type": "string", "maxLength": 80},
        },
    },
}

INSTRUCTIONS = (
    "Write ONE short paragraph, at most six sentences, that reads the market's regime "
    "across timeframes from these rows: the trader's own regime first, then the weekly and "
    "daily structure, then the intraday timeframes. Name a timeframe only as M5, M30, H1, "
    "H4, D1 or W (or daily / weekly), and call it bullish, bearish or neutral only when a "
    "row you cite says so for that timeframe. Copy every date from the rows (YYYY-MM-DD or "
    "MM-DD) and every number exactly; never write a number or a date that is not in the "
    "rows and never spell a number out. Use the trader's regime words exactly as the "
    "journal has them. You read; you do not predict or advise. Cite the ids you used, "
    "copied exactly from allowed_source_ids."
)

#: Timeframe codes the table carries, and the words that name them.
TIMEFRAME_CODES = ("M5", "M30", "H1", "H4", "D1", "W")
_TIMEFRAME_WORDS = {"weekly": "W", "daily": "D1", "hourly": "H1"}
#: Direction words -> the env_keys that support them.
_DIRECTIONS = {
    "bullish": ("bullish_strong", "bullish_weak"),
    "bearish": ("bearish_strong", "bearish_weak"),
    "neutral": ("neutral_chop",),
    "chop": ("neutral_chop",),
    "choppy": ("neutral_chop",),
}
_CODE = re.compile(r"\b(?:[MHD]\d+|W\d*)\b")
_WORDS = re.compile(r"[A-Za-z]+")
_FULL_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_SHORT_DATE = re.compile(r"\b(\d{1,2})[-/](\d{1,2})\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
#: A clause ends at . ; ! ? or a line break, but never inside a decimal.
_CLAUSE = re.compile(r"(?<!\d)[.;!?]|[.;!?](?!\d)|\n")
_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)
_MONTH_WORD = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b"
)
_SMALL_NUMBER_WORDS = frozenset({
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twice", "thrice",
})
#: Trader regime phrases -> the journal regimes that allow them.
_REGIME_PHRASES = (
    (re.compile(r"\bbull\s+run\b", re.I), ("bull_run",)),
    (re.compile(r"\bbear(?:ish)?\s+channel\b", re.I), ("bear_channel_lower_highs",)),
    (re.compile(r"\bcapitulat\w*", re.I), ("capitulation",)),
    (re.compile(r"\brecover(?:y|ing|ed)\b", re.I), ("recovery",)),
    (re.compile(r"\brange[- ]bound\b|\branging\b|\bin a range\b", re.I), ("range",)),
)
_COMPRESSION = re.compile(r"\bcompress\w*", re.I)
#: Structure phrases -> (daily channel label, pivot kind, pivot direction, weekly count key).
_STRUCTURE_PHRASES = (
    (re.compile(r"\blower[- ]highs?\b", re.I), "lh_ll", "high", -1, "lower_highs"),
    (re.compile(r"\bhigher[- ]highs?\b", re.I), "hh_hl", "high", 1, "higher_highs"),
    (re.compile(r"\blower[- ]lows?\b", re.I), "lh_ll", "low", -1, "lower_lows"),
    (re.compile(r"\bhigher[- ]lows?\b", re.I), "hh_hl", "low", 1, "higher_lows"),
)


def _story():
    from ai_jobs import day_review_narration

    return day_review_narration


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def build_inputs(
    session_date: str,
    table_rows: Iterable[Mapping[str, Any]],
    journal_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """The model's view: last sessions of the table, latest structure, the trader's regime."""
    import market_regimes
    import structural_regime

    day = str(session_date or "")[:10]
    indexes = market_regimes.INDEXES
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in table_rows or ():
        if not isinstance(row, Mapping):
            continue
        session = str(row.get("session_date") or "")[:10]
        symbol = str(row.get("symbol") or "").upper()
        if symbol in indexes and _day(session) is not None and session <= day:
            by_key.setdefault((session, symbol), row)
    window = sorted({session for session, _symbol in by_key})[-WINDOW_SESSIONS:]
    if not window:
        return None
    table = []
    for session in window:
        for symbol in indexes:
            row = by_key.get((session, symbol))
            if row is None:
                continue
            timeframes = row.get("timeframes") if isinstance(row.get("timeframes"), Mapping) else {}
            table.append({
                "source_id": f"table:{session}:{symbol}",
                "session_date": session,
                "symbol": symbol,
                "timeframes": {tf: str(timeframes.get(tf) or "unknown") for tf in TIMEFRAME_CODES},
            })
    structure = []
    for symbol in indexes:
        dated = [session for session in window if (session, symbol) in by_key]
        if not dated:
            continue
        row = by_key[(dated[-1], symbol)]
        structure.append({
            "source_id": f"structure:{dated[-1]}:{symbol}",
            "session_date": dated[-1],
            "symbol": symbol,
            "structure": dict(row.get("structure") or {}),
        })
    rows = [dict(row) for row in journal_rows or () if isinstance(row, Mapping)]
    segments = []
    for segment in structural_regime.effective_segments(rows):
        segments.append({
            "source_id": f"regime:{segment.get('first_segment_id')}",
            "regime": segment.get("regime"),
            "label": structural_regime.label(segment.get("regime")),
            "start_date": str(segment.get("start_date") or "")[:10],
            "structure_note": str(segment.get("structure_note") or ""),
        })
    current = structural_regime.regime_at(rows, day)
    trader_regime = None
    if current:
        trader_regime = {
            "source_id": f"regime:{current.get('first_segment_id')}",
            "regime": current.get("regime"),
            "label": current.get("label"),
            "start_date": str(current.get("start_date") or "")[:10],
            "day_count": current.get("day_count"),
            "session_count": current.get("session_count"),
            "structure_note": str(current.get("structure_note") or ""),
        }
    return {
        "session_date": day,
        "window": [window[0], window[-1]],
        "trader_regime": trader_regime,
        "regime_journal": segments,
        "structure": structure,
        "table": table,
    }


def inputs_hash(inputs: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(inputs), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def allowed_source_ids(inputs: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for part in ("table", "structure", "regime_journal"):
        for row in inputs.get(part) or ():
            source_id = str(row.get("source_id") or "")
            if source_id and source_id not in ids:
                ids.append(source_id)
    return ids


# ---------------------------------------------------------------------------
# the checks: every timeframe word, date, number and regime word
# ---------------------------------------------------------------------------
def _allowed_dates(inputs: Mapping[str, Any]) -> set[date]:
    found: set[date] = set()

    def _add(value: Any) -> None:
        day = _day(value)
        if day is not None:
            found.add(day)

    _add(inputs.get("session_date"))
    for row in inputs.get("table") or ():
        _add(row.get("session_date"))
    for segment in inputs.get("regime_journal") or ():
        _add(segment.get("start_date"))
    for row in inputs.get("structure") or ():
        structure = row.get("structure") or {}
        channel = structure.get("daily_channel") or {}
        for key in ("last_pivot_high", "prev_pivot_high", "last_pivot_low", "prev_pivot_low"):
            _add((channel.get(key) or {}).get("date"))
        _add((structure.get("weekly") or {}).get("last_week"))
    return found


def _source_numbers(inputs: Mapping[str, Any]) -> set[str]:
    """Every number in the inputs, as written and rounded to 0-3 decimals."""
    out: set[str] = set()
    for raw in _NUMBER.findall(json.dumps(dict(inputs), default=str)):
        value = float(raw)
        out.add(raw)
        for places in range(4):
            out.add(f"{round(value, places):.{places}f}")
    return out


def _clause_dates(clause: str, allowed: set[date]) -> set[str]:
    """ISO dates a clause names (full or month-day), resolved against the allowed dates."""
    named: set[str] = set()
    for year, month, day in _FULL_DATE.findall(clause):
        named.add(f"{year}-{month}-{day}")
    stripped = _FULL_DATE.sub(" ", clause)
    for month, day in _SHORT_DATE.findall(stripped):
        named.update(
            item.isoformat() for item in allowed if (item.month, item.day) == (int(month), int(day))
        )
    return named


def _check_dates(text: str, inputs: Mapping[str, Any]) -> None:
    rejected = _story().NarrationRejected
    allowed = _allowed_dates(inputs)
    for year, month, day in _FULL_DATE.findall(text):
        stamp = _day(f"{year}-{month}-{day}")
        if stamp is None or stamp not in allowed:
            raise rejected(f"the read writes {year}-{month}-{day}, a date none of its rows holds")
    for month, day in _SHORT_DATE.findall(_FULL_DATE.sub(" ", text)):
        if not any((item.month, item.day) == (int(month), int(day)) for item in allowed):
            raise rejected(f"the read writes {month}-{day}, a date none of its rows holds")
    notes = " ".join(
        f"{segment.get('structure_note')} {segment.get('label')}"
        for segment in inputs.get("regime_journal") or ()
    )
    months = {_MONTHS[item.month - 1] for item in allowed}
    for word in _MONTH_WORD.findall(text):
        full = next((name for name in _MONTHS if name.startswith(word[:3])), word)
        if full not in months and word not in notes and full not in notes:
            raise rejected(f"the read names {word}, a month none of its rows or the trader's notes holds")


def _check_numbers(text: str, inputs: Mapping[str, Any]) -> None:
    import day_review_show

    rejected = _story().NarrationRejected
    bare = _FULL_DATE.sub(" ", text)
    bare = _SHORT_DATE.sub(" ", bare)
    bare = _CODE.sub(" ", bare)
    known = _source_numbers(inputs)
    for number in day_review_show._numbers(bare):
        if number not in known:
            raise rejected(f"the read writes {number}, which none of its rows holds")
    words = {word.lower() for word in _WORDS.findall(text)}
    spelled = words & (set(day_review_show._NUMBER_WORDS) | _SMALL_NUMBER_WORDS)
    if spelled:
        raise rejected(f"the read spells a number out ({sorted(spelled)[0]}); numbers are copied, never counted")


def _rows_for(inputs: Mapping[str, Any], symbols: set[str], dates: set[str]) -> list[Mapping[str, Any]]:
    """Table rows for the named symbols, narrowed to the named table sessions (a pivot date does not narrow)."""
    rows = [row for row in inputs.get("table") or () if not symbols or row.get("symbol") in symbols]
    sessions = dates & {str(row.get("session_date") or "") for row in rows}
    if sessions:
        rows = [row for row in rows if row.get("session_date") in sessions]
    return rows


def _check_timeframes(text: str, inputs: Mapping[str, Any]) -> None:
    """A timeframe the table does not carry, or a direction no cited-window row says, rejects."""
    rejected = _story().NarrationRejected
    allowed = _allowed_dates(inputs)
    symbols_known = {str(row.get("symbol") or "") for row in inputs.get("table") or ()}
    for clause in _CLAUSE.split(text):
        if not clause.strip():
            continue
        tokens: list[tuple[int, str, str]] = []
        for match in _CODE.finditer(clause):
            code = match.group(0)
            if code not in TIMEFRAME_CODES:
                raise rejected(f"the read names {code}, a timeframe the regime table does not carry")
            tokens.append((match.start(), "tf", code))
        for match in _WORDS.finditer(clause):
            word = match.group(0).lower()
            if word in _TIMEFRAME_WORDS:
                tokens.append((match.start(), "tf", _TIMEFRAME_WORDS[word]))
            elif word in _DIRECTIONS:
                tokens.append((match.start(), "dir", word))
        tokens.sort()
        symbols = {word for word in _WORDS.findall(clause) if word in symbols_known}
        dates = _clause_dates(clause, allowed)
        rows = _rows_for(inputs, symbols, dates)
        for index, (_pos, kind, value) in enumerate(tokens):
            if kind == "tf":
                if not any(row["timeframes"].get(value) not in (None, "", "unknown") for row in rows):
                    raise rejected(f"the read names {value}, which its rows never read")
                continue
            before = [tok for tok in tokens[:index] if tok[1] == "tf"]
            after = [tok for tok in tokens[index + 1:] if tok[1] == "tf"]
            timeframe = before[-1][2] if before else (after[0][2] if after else None)
            wanted = _DIRECTIONS[value]
            timeframes = (timeframe,) if timeframe else TIMEFRAME_CODES
            if not any(row["timeframes"].get(tf) in wanted for row in rows for tf in timeframes):
                where = timeframe or "any timeframe"
                raise rejected(f"the read calls {where} {value}, which no row in its window says")


#: Pivot shorthand, read as the words it stands for.
_ABBREVIATIONS = {"HH": "higher high", "HL": "higher low", "LH": "lower high", "LL": "lower low"}


def _check_regime_words(text: str, inputs: Mapping[str, Any]) -> None:
    rejected = _story().NarrationRejected
    text = re.sub(r"\b(HH|HL|LH|LL)\b", lambda match: _ABBREVIATIONS[match.group(1)], text)
    regimes = {str(segment.get("regime") or "") for segment in inputs.get("regime_journal") or ()}
    structures = [row.get("structure") or {} for row in inputs.get("structure") or ()]
    for pattern, allowed in _REGIME_PHRASES:
        match = pattern.search(text)
        if match and not regimes & set(allowed):
            raise rejected(f"the read says {match.group(0)!r}, a regime the trader's journal does not hold")
    match = _COMPRESSION.search(text)
    if match and "weekly_hh_then_compression" not in regimes and not any(
        (structure.get("atr") or {}).get("compressed") is True for structure in structures
    ):
        raise rejected(f"the read says {match.group(0)!r}, which no row or regime says")
    for pattern, channel, pivot, sign, weekly_key in _STRUCTURE_PHRASES:
        match = pattern.search(text)
        if not match:
            continue
        supported = False
        for structure in structures:
            daily = structure.get("daily_channel") or {}
            last = (daily.get(f"last_pivot_{pivot}") or {}).get("value")
            prev = (daily.get(f"prev_pivot_{pivot}") or {}).get("value")
            moved = (
                isinstance(last, (int, float)) and isinstance(prev, (int, float))
                and (last - prev) * sign > 0
            )
            weekly = (structure.get("weekly") or {}).get(weekly_key)
            if daily.get("label") == channel or moved or (isinstance(weekly, int) and weekly > 0):
                supported = True
                break
        if not supported:
            raise rejected(f"the read says {match.group(0)!r}, which no structure fact says")


def _check_tickers(text: str, inputs: Mapping[str, Any]) -> None:
    """An all-caps word is a symbol the rows carry, a timeframe code or a fixed desk word."""
    import day_review_show

    known = {str(row.get("symbol") or "") for row in inputs.get("table") or ()} | set(_ABBREVIATIONS)
    for word in re.findall(r"\b[A-Z][A-Z]{1,5}\b", text):
        if word not in known and word not in day_review_show.DESK_WORDS:
            raise _story().NarrationRejected(f"the read names {word!r}, which its rows do not carry")


def verify_read(reply: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The checked read, or `NarrationRejected`. Whole-reject: nothing is trimmed."""
    story = _story()
    body = story._validate(reply, READ_JSON_SCHEMA, name="regime_read")
    story._check_sources(body, set(allowed_source_ids(inputs)))
    paragraph = str(body.get("paragraph") or "").strip()
    if not paragraph:
        raise story.NarrationRejected("the read is empty")
    sentences = [part for part in re.split(r"(?<!\d)[.!?](?!\d)", paragraph) if part.strip()]
    if len(sentences) > MAX_SENTENCES:
        raise story.NarrationRejected(f"the read runs {len(sentences)} sentences; at most {MAX_SENTENCES}")
    _check_dates(paragraph, inputs)
    _check_numbers(paragraph, inputs)
    _check_tickers(paragraph, inputs)
    _check_timeframes(paragraph, inputs)
    _check_regime_words(paragraph, inputs)
    return {"paragraph": paragraph, "sources": [str(item) for item in body.get("sources") or ()]}


# ---------------------------------------------------------------------------
# the slot
# ---------------------------------------------------------------------------
def _result(status: str, reason: str, *, model: str = "", outputs=()) -> dict[str, Any]:
    return {"status": status, "model": model, "reason": reason, "outputs": list(outputs)}


def _journal_rows() -> list[dict[str, Any]]:
    from journal_store import JournalStore
    from project_paths import JOURNAL_DB_FILE

    return JournalStore(Path(JOURNAL_DB_FILE)).list_structural_regime()


def run_regime_read(
    *,
    session_date: str = "",
    now: datetime | None = None,
    table_path: Any = None,
    journal_rows: Any = None,
    out_dir: Path | None = None,
    request: Callable[..., Mapping[str, Any]] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Write one verified regime read for the session, or keep the last one. Never raises."""
    import market_regimes
    from ai_jobs.ledger import STATUS_DEGRADED

    story = _story()
    session = str(session_date or "").strip()[:10] or datetime.now().date().isoformat()
    try:
        if table_path is None:
            from project_paths import MARKET_REGIME_TABLE_FILE

            table_path = MARKET_REGIME_TABLE_FILE
        table = market_regimes.read_table(table_path)
        try:
            rows = _journal_rows() if journal_rows is None else list(journal_rows)
        except Exception:  # noqa: BLE001 - an unread journal is an unknown regime
            _log.debug("Regime journal unreadable for the regime read.", exc_info=True)
            rows = []
        inputs = build_inputs(session, table, rows)
    except Exception as exc:  # noqa: BLE001 - the night goes on
        _log.exception("regime_read: inputs failed")
        return _result("failed", f"regime read inputs unreadable ({type(exc).__name__}: {exc})")
    if inputs is None:
        return _result("skipped", "no regime table rows yet; nothing to read")
    digest = inputs_hash(inputs)
    destination = market_regimes.regime_read_path(session, root=out_dir)
    existing = story._read_json(destination) or {}
    if existing.get("inputs_hash") == digest and existing.get("prompt_version") == PROMPT_VERSION:
        return _result(
            "ok", "the verified regime read is unchanged",
            model=str(existing.get("model") or ""), outputs=[str(destination)],
        )
    caller, refusal = story._request_for(request)
    if caller is None:
        return _result(STATUS_DEGRADED, refusal.replace("narration", "regime read"))
    evidence = {
        "package_id": f"regime-read:{digest[:16]}",
        "evidence_hash": digest,
        "instructions": INSTRUCTIONS,
        "allowed_source_ids": allowed_source_ids(inputs),
        **inputs,
    }
    try:
        result = story._call(
            caller,
            evidence=evidence,
            schema=READ_JSON_SCHEMA,
            prompt_version=PROMPT_VERSION,
            schema_name="tradingbot_regime_read",
        )
        reply = result.get("summary") if isinstance(result, Mapping) else None
        read = verify_read(reply, inputs)
        model = str(result.get("model") or "")
        story._atomic_write(destination, {
            "schema": SCHEMA,
            "session_date": session,
            "generated_at": story._moment(now),
            "inputs_hash": digest,
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "source_id": f"regime_read:{session}",
            "window": inputs["window"],
            "trader_regime": (inputs.get("trader_regime") or {}).get("label") or "",
            "read": read,
        })
    except Exception as exc:  # noqa: BLE001 - the last verified read is the fallback
        _log.debug("The regime read was not written.", exc_info=True)
        return _result(STATUS_DEGRADED, f"the regime read was rejected; the last verified read was kept: {exc}")
    return _result("ok", f"verified regime read written for {session}", model=model, outputs=[str(destination)])


__all__ = [
    "PROMPT_VERSION",
    "READ_JSON_SCHEMA",
    "build_inputs",
    "run_regime_read",
    "verify_read",
]
