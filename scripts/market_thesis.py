"""What the trader claimed, extracted deterministically — WISHLIST 10D step 2.

A thesis is a claim with a horizon, sometimes a condition, sometimes a stated
invalidation. The Market Journal already holds all of it, in the trader's own
prose, where nothing can find it again. This module reads that prose with a
small VERSIONED VOCABULARY and writes a draft that quotes itself.

Four rules hold this to evidence rather than interpretation:

1. **Every field carries its source span, and the span must reproduce the
   field.** ``entry["text"][start:end] == draft.invalidation``, exactly. A
   paraphrase cannot satisfy that, which is the point: a thesis the trader is
   shown must be one they can recognise as their own words.
2. **Unstated stays unstated.** :data:`UNSTATED` is a real answer with no span,
   because a span is a quotation and there is nothing to quote. An ambiguous
   note - one bullish word and one bearish word, or an explicit hedge - is
   ``unstated``, never a coin flip on whichever keyword came first.
3. **A contradiction is a stance REVERSAL on the same benchmark, inside the
   horizon.** Nothing subtler. A later note that names the benchmark without a
   stance only :data:`LINK_MENTIONS`; a note about a different benchmark, or
   one written after the horizon, is not this thesis's evidence at all.
4. **The store is append-only and the journal entry is never touched.** A
   trader edit of the interpretation is a NEW row naming the draft it
   supersedes (:func:`record_interpretation`). The draft stays on disk, because
   the interesting question later is what the machine read and what the trader
   corrected - and a store that quietly rewrites the first answer cannot say.

The horizon is counted in **exchange sessions** on the same calendar the rest
of the desk runs on, never in calendar days: a "this week" thesis opened on a
Tuesday is still open on the following Monday.

PURE extraction plus a small JSONL store. No Qt, no model, no detector, no
score: nothing here reaches a detector, score, gate, alert, watchlist, Focus
list, the review queue or `review_policy.json` (plan.md sec 5).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: The vocabulary's version. It travels on every row so a later vocabulary can
#: be told apart from this one without re-reading the prose. **Never assert a
#: literal version in a test** - the same rule the veto vocabulary carries.
EXTRACTOR_VERSION = "market_thesis_vocab_v1"

#: Two different absences, kept apart. `unstated` is "the trader did not say";
#: `unknown` is "nobody supplied it" (an imported forecast's creation time).
UNSTATED = "unstated"
UNKNOWN = "unknown"

STANCE_BULLISH = "bullish"
STANCE_BEARISH = "bearish"
STANCE_NEUTRAL = "neutral"
STANCE_CAUTIOUS = "cautious"
STANCE_UNSTATED = UNSTATED

STATUS_OPEN = "open"
STATUS_CLOSED = "closed"
STATUS_INVALIDATED = "invalidated"

LINK_SUPPORTS = "supports"
LINK_CONTRADICTS = "contradicts"
LINK_MENTIONS = "mentions"

KIND_THESIS = "thesis"
KIND_FORECAST = "forecast"

#: The markets a thesis is allowed to be ABOUT. The story's scope, so the two
#: surfaces speak about the same six names; anything else the trader writes
#: about is carried by the entry's own `symbols`.
from market_story import BENCHMARKS  # noqa: E402  (one scope, named once)

#: How long a thesis with no stated horizon is linked for. It does not CLOSE
#: such a thesis (see `resolve_status`): an unstated horizon is unstated, and
#: closing it on a guessed clock would be the machine inventing the trader's
#: time frame. It only bounds which later notes are considered evidence.
DEFAULT_LINK_WINDOW_SESSIONS = 5

#: Horizon words, longest phrase first so "next week" is not read as "week".
_HORIZON_WORDS: tuple[tuple[str, int], ...] = (
    ("into the close", 1),
    ("by the close", 1),
    ("this afternoon", 1),
    ("tomorrow", 1),
    ("today", 1),
    ("next week", 5),
    ("this week", 5),
    ("the rest of the week", 5),
    ("next month", 21),
    ("this month", 21),
    ("this quarter", 63),
    ("a few days", 3),
    ("a couple of days", 2),
    ("couple of days", 2),
    ("few days", 3),
)

#: Stance words. Whole-word matched (or whole-phrase), so "lower" never fires
#: inside "flowering" and "long" never inside "belong".
_BULLISH_WORDS = (
    "hold above",
    "holds above",
    "held above",
    "reclaim",
    "reclaims",
    "reclaimed",
    "breakout",
    "breaks out",
    "strong",
    "strength",
    "bullish",
    "rally",
    "rallies",
    "higher",
    "upside",
    "grind up",
    "bid",
    "squeeze",
    "hold",
    "holds",
    "held",
    "long",
    "buy",
    "buying",
)
_BEARISH_WORDS = (
    "lose",
    "loses",
    "lost",
    "break down",
    "breaks down",
    "breakdown",
    "weak",
    "weakness",
    "bearish",
    "selloff",
    "sell off",
    "lower",
    "downside",
    "fail",
    "fails",
    "failed",
    "heavy",
    "roll over",
    "rolls over",
    "short",
    "fade",
    "fades",
)
_CAUTIOUS_WORDS = (
    "caution",
    "cautious",
    "careful",
    "sit out",
    "sitting out",
    "stand aside",
    "no size",
    "small size",
    "risk off",
    "risk-off",
)
_NEUTRAL_WORDS = ("chop", "choppy", "range bound", "rangebound", "sideways", "balanced")
#: An explicit hedge. The trader said, in words, that they do not have a view -
#: which is a stated absence of a view, and `unstated` is how that is recorded.
_HEDGE_WORDS = (
    "not sure",
    "unsure",
    "no view",
    "no idea",
    "undecided",
    "either way",
    "could go either way",
    "do not know",
    "don't know",
)

#: A conditional clause. "as long as VIX stays under 20" is a CONDITION the
#: trader attached to their own claim; a question later ("is that still true?")
#: may only be asked about one of these, because it quotes it.
_CONDITION_MARKERS = (
    "as long as",
    "so long as",
    "provided that",
    "provided",
    "unless",
    "only if",
    "on condition that",
    "while",
    "assuming",
)

#: What makes the trader wrong, in their own words. Detected as a SENTENCE, so
#: the quote reads as a sentence rather than as a clipped fragment.
_INVALIDATION_MARKERS = (
    "i am wrong",
    "i'm wrong",
    "i was wrong",
    "invalidate",
    "invalidated",
    "invalidation",
    "thesis is dead",
    "thesis dies",
    "i am out",
    "i'm out",
    "i give up on",
    "no longer works",
    "that is the end of it",
)
_CONDITIONAL_OPENERS = ("if ", "if,", "should ", "once ", "when ")


@dataclass(frozen=True)
class ThesisDraft:
    """One deterministic reading of one entry. Never the last word on it."""

    entry_id: str
    session_date: str
    created_at: str
    extractor_version: str
    kind: str
    claim: str
    horizon: str
    horizon_sessions: int
    stance: str
    condition: str
    invalidation: str
    benchmarks: tuple[str, ...]
    spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    is_prediction: bool = False


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def extract_thesis(entry: Mapping[str, Any]) -> ThesisDraft:
    """Read one journal entry into a draft thesis.

    The invalidation and the condition are found FIRST and removed from the
    text the stance is read out of. They have to be: an invalidation says the
    opposite of the claim by construction ("I expect SPY to hold ... if SPY
    loses 5,400 I am wrong"), so a stance counted over the whole sentence reads
    one bullish word and one bearish word and gives up on a perfectly clear
    view.
    """
    text = str(entry.get("text") or "")
    lowered = text.lower()

    invalidation, invalidation_span = _find_invalidation(text, lowered)
    condition, condition_span = _find_condition(text, lowered)

    claim_text = _without(text, (invalidation_span, condition_span))
    stance, stance_span = _find_stance(text, claim_text, (invalidation_span, condition_span))
    horizon, horizon_span, horizon_sessions = _find_horizon(text, lowered)

    claim, claim_span = _find_claim(text, claim_text)

    spans: dict[str, tuple[int, int]] = {}
    if claim_span:
        spans["claim"] = claim_span
    if stance_span:
        spans["stance"] = stance_span
    if condition_span:
        spans["condition"] = condition_span
    if invalidation_span:
        spans["invalidation"] = invalidation_span
    if horizon_span:
        spans["horizon"] = horizon_span

    return ThesisDraft(
        entry_id=str(entry.get("entry_id") or ""),
        session_date=str(entry.get("session_date") or ""),
        created_at=str(entry.get("created_at") or ""),
        extractor_version=EXTRACTOR_VERSION,
        kind=KIND_THESIS,
        claim=claim,
        horizon=horizon,
        horizon_sessions=horizon_sessions,
        stance=stance,
        condition=condition,
        invalidation=invalidation,
        benchmarks=_benchmarks_in(entry, text),
        spans=spans,
        # A note written after the close DESCRIBES the session; it does not
        # predict it. `written_after_the_session` is computed by `build_entry`
        # from the exchange's own close, never claimed by a caller.
        is_prediction=not bool(entry.get("written_after_the_session")),
    )


def _sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence, the terminator excluded."""
    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(text):
        if char in ".!?\n":
            if index > start:
                spans.append((start, index))
            start = index + 1
    if start < len(text):
        spans.append((start, len(text)))
    return [(a + len(text[a:b]) - len(text[a:b].lstrip()), b) for a, b in spans if text[a:b].strip()]


def _find_invalidation(text: str, lowered: str) -> tuple[str, tuple[int, int] | None]:
    for start, end in _sentences(text):
        chunk = lowered[start:end]
        if not any(marker in chunk for marker in _INVALIDATION_MARKERS):
            continue
        if not any(chunk.lstrip().startswith(opener) for opener in _CONDITIONAL_OPENERS):
            # "I was wrong" on its own is a verdict on a past call, not a
            # statement of what WOULD make this one wrong.
            if " if " not in chunk:
                continue
        span = (start, end)
        return text[start:end].strip(), _tighten(text, span)
    return UNSTATED, None


def _find_condition(text: str, lowered: str) -> tuple[str, tuple[int, int] | None]:
    for start, end in _sentences(text):
        chunk = lowered[start:end]
        for marker in _CONDITION_MARKERS:
            at = chunk.find(marker)
            if at < 0:
                continue
            span = (start + at, end)
            body = text[span[0]:span[1]].strip()
            if not body:
                continue
            return body, _tighten(text, span)
    return UNSTATED, None


def _tighten(text: str, span: tuple[int, int]) -> tuple[int, int]:
    """Trim whitespace INSIDE the span, so `text[start:end]` is the field."""
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end)


def _without(text: str, spans: Sequence[tuple[int, int] | None]) -> str:
    """The text with those spans blanked out, positions preserved."""
    chars = list(text)
    for span in spans:
        if not span:
            continue
        for index in range(span[0], min(span[1], len(chars))):
            chars[index] = " "
    return "".join(chars)


def _word_hits(haystack_lower: str, words: Sequence[str]) -> list[tuple[int, int, str]]:
    hits: list[tuple[int, int, str]] = []
    for word in words:
        for match in re.finditer(r"(?<!\w)" + re.escape(word) + r"(?!\w)", haystack_lower):
            hits.append((match.start(), match.end(), word))
    return hits


def _find_stance(
    text: str, claim_text: str, removed: Sequence[tuple[int, int] | None]
) -> tuple[str, tuple[int, int] | None]:
    lowered = claim_text.lower()
    if any(hedge in lowered for hedge in _HEDGE_WORDS):
        return STANCE_UNSTATED, None
    bullish = _word_hits(lowered, _BULLISH_WORDS)
    bearish = _word_hits(lowered, _BEARISH_WORDS)
    cautious = _word_hits(lowered, _CAUTIOUS_WORDS)
    neutral = _word_hits(lowered, _NEUTRAL_WORDS)
    if bullish and bearish:
        # Both directions and no way to choose. A keyword counter that takes
        # the first match calls this bullish; it is a view the trader did not
        # state, and an invented view is worse than a missing one.
        return STANCE_UNSTATED, None
    if bullish:
        hit = min(bullish)
        return STANCE_BULLISH, (hit[0], hit[1])
    if bearish:
        hit = min(bearish)
        return STANCE_BEARISH, (hit[0], hit[1])
    if cautious:
        hit = min(cautious)
        return STANCE_CAUTIOUS, (hit[0], hit[1])
    if neutral:
        hit = min(neutral)
        return STANCE_NEUTRAL, (hit[0], hit[1])
    return STANCE_UNSTATED, None


def _find_horizon(text: str, lowered: str) -> tuple[str, tuple[int, int] | None, int]:
    for phrase, sessions in _HORIZON_WORDS:
        at = lowered.find(phrase)
        if at < 0:
            continue
        return text[at:at + len(phrase)], (at, at + len(phrase)), sessions
    return UNSTATED, None, 0


def _find_claim(text: str, claim_text: str) -> tuple[str, tuple[int, int] | None]:
    """The first sentence that survives the condition and the invalidation."""
    for start, end in _sentences(claim_text):
        if not claim_text[start:end].strip():
            continue
        # Tightened against the BLANKED text, so the condition and the
        # invalidation are not silently re-attached from the original; quoted
        # out of the original at those offsets, so it is still the trader's
        # own characters and the span still reproduces the field.
        span = _tighten(claim_text, (start, end))
        return text[span[0]:span[1]].strip(" ,;"), span
    body = text.strip()
    return (body, (0, len(text))) if body else (UNSTATED, None)


def _benchmarks_in(entry: Mapping[str, Any], text: str) -> tuple[str, ...]:
    found: list[str] = []
    upper = text.upper()
    for symbol in BENCHMARKS:
        if re.search(r"(?<![A-Z0-9])" + symbol + r"(?![A-Z0-9])", upper):
            found.append(symbol)
    for raw in entry.get("symbols") or ():
        symbol = str(raw).strip().upper()
        if symbol and symbol not in found:
            found.append(symbol)
    return tuple(found)


# ---------------------------------------------------------------------------
# linking later notes to a draft
# ---------------------------------------------------------------------------
def link_entries(
    draft: ThesisDraft, later_entries: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Which later notes are evidence about THIS thesis, and which way.

    Inside the horizon and on the same benchmark, or it is not linked at all.
    A stance reversal CONTRADICTS, the same stance SUPPORTS, and a note that
    names the benchmark without stating a stance only MENTIONS - because a
    price observation is not agreement.
    """
    rows: list[dict[str, Any]] = []
    opened = _as_date(draft.session_date)
    window = draft.horizon_sessions or DEFAULT_LINK_WINDOW_SESSIONS
    mine = {symbol for symbol in draft.benchmarks}
    for entry in later_entries or ():
        entry_id = str(entry.get("entry_id") or "")
        if not entry_id or entry_id == draft.entry_id:
            continue
        session = _as_date(str(entry.get("session_date") or ""))
        if opened and session:
            if session < opened:
                continue
            if _sessions_between(opened, session) > window:
                continue
        other = extract_thesis(entry)
        shared = [symbol for symbol in other.benchmarks if symbol in mine]
        if not shared:
            continue
        if other.stance == STANCE_UNSTATED or draft.stance == STANCE_UNSTATED:
            link = LINK_MENTIONS
        elif other.stance == draft.stance:
            link = LINK_SUPPORTS
        elif _opposes(draft.stance, other.stance):
            link = LINK_CONTRADICTS
        else:
            link = LINK_MENTIONS
        for symbol in shared:
            rows.append(
                {
                    "entry_id": entry_id,
                    "benchmark": symbol,
                    "link": link,
                    "stance": other.stance,
                    "session_date": str(entry.get("session_date") or ""),
                    "created_at": str(entry.get("created_at") or ""),
                    "text": str(entry.get("text") or ""),
                }
            )
    return rows


def _opposes(one: str, other: str) -> bool:
    return {one, other} == {STANCE_BULLISH, STANCE_BEARISH}


def _as_date(value: str) -> date | None:
    try:
        parts = [int(part) for part in str(value)[:10].split("-")]
        return date(parts[0], parts[1], parts[2])
    except Exception:  # noqa: BLE001 - an undated row is simply not linked
        return None


def _sessions_between(start: date, end: date) -> int:
    """Exchange sessions after `start` up to `end`. Calendar-days arithmetic
    closes a "this week" thesis on the Sunday and is wrong by a session."""
    try:
        from market_calendar import trading_days_between

        return int(trading_days_between(start, end))
    except Exception:  # noqa: BLE001 - a coarse answer beats none
        return max(0, (end - start).days)


def resolve_status(
    draft: ThesisDraft, *, as_of: date, links: Iterable[Mapping[str, Any]] = ()
) -> tuple[str, str]:
    """Open, closed or invalidated - and the sentence that says why.

    A thesis is INVALIDATED only when the trader stated what would make them
    wrong AND a later note inside the horizon reversed the stance on the same
    benchmark. Nothing subtler counts, and nothing here reads an outcome: a
    thesis is never graded by the market's move, only by what was written.
    """
    if isinstance(as_of, datetime):
        as_of = as_of.date()
    contradicting = [
        row for row in (links or ()) if str(row.get("link") or "") == LINK_CONTRADICTS
    ]
    if contradicting and draft.invalidation != UNSTATED:
        entry_id = str(contradicting[0].get("entry_id") or "")
        return (
            STATUS_INVALIDATED,
            f"the stated invalidation was met: {entry_id} reversed the stance on "
            f"{contradicting[0].get('benchmark', '')} inside the horizon",
        )
    opened = _as_date(draft.session_date)
    if not opened:
        return STATUS_OPEN, "the entry carries no session date, so no clock can close it"
    if draft.horizon_sessions <= 0:
        return (
            STATUS_OPEN,
            "no horizon was stated, so nothing but a stated invalidation closes it",
        )
    elapsed = _sessions_between(opened, as_of)
    if elapsed >= draft.horizon_sessions:
        return (
            STATUS_CLOSED,
            f"{elapsed} of {draft.horizon_sessions} sessions have elapsed since "
            f"{draft.session_date}",
        )
    return (
        STATUS_OPEN,
        f"{elapsed} of {draft.horizon_sessions} sessions have elapsed since "
        f"{draft.session_date}",
    )


def questions_for(draft: ThesisDraft) -> tuple[str, ...]:
    """One or two grounded questions. Never more, and never about nothing.

    "Your caution was conditioned on X - is X still true?" is asked ONLY when X
    was stated, and it quotes X verbatim. When nothing was conditioned and
    nothing was named as an invalidation, the question asked is the one the
    trader can always answer: what would make you wrong? The word `unstated`
    never reaches the trader as a question - it is a field value, not English.
    """
    questions: list[str] = []
    if draft.condition != UNSTATED:
        questions.append(
            f"Your view was conditioned on “{draft.condition}” - is that still true?"
        )
    if draft.invalidation != UNSTATED:
        questions.append(
            f"You said “{draft.invalidation}” would make you wrong - has that happened?"
        )
    if questions:
        return tuple(questions[:2])
    names = ", ".join(draft.benchmarks) if draft.benchmarks else "this"
    return (f"What would have to happen for you to decide you are wrong about {names}?",)


# ---------------------------------------------------------------------------
# the store: append-only, keyed on entry_id + extractor_version
# ---------------------------------------------------------------------------
def _default_path() -> Path:
    from project_paths import MARKET_THESES_FILE

    return Path(MARKET_THESES_FILE)


def _thesis_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"th-{digest}"


def draft_row(draft: ThesisDraft, *, now: datetime | None = None) -> dict[str, Any]:
    """The JSONL row for a draft. Deterministic: the same draft is the same id."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return {
        "thesis_id": _thesis_id(draft.entry_id, draft.extractor_version, draft.kind),
        "entry_id": draft.entry_id,
        "extractor_version": draft.extractor_version,
        "kind": draft.kind,
        "supersedes": "",
        "recorded_at": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "session_date": draft.session_date,
        "created_at": draft.created_at,
        "claim": draft.claim,
        "horizon": draft.horizon,
        "horizon_sessions": int(draft.horizon_sessions),
        "stance": draft.stance,
        "condition": draft.condition,
        "invalidation": draft.invalidation,
        "benchmarks": list(draft.benchmarks),
        "spans": {name: list(span) for name, span in draft.spans.items()},
        "is_prediction": bool(draft.is_prediction),
        "questions": list(questions_for(draft)),
        "text": "",
    }


def record_draft(
    draft: ThesisDraft, *, path: Path | None = None, now: datetime | None = None
) -> dict[str, Any]:
    """Append the draft, once. Keyed on `entry_id` + `extractor_version`.

    A second call with the same draft returns the row already on disk rather
    than writing a duplicate: the key IS the identity, and an append-only store
    that appends the same fact twice makes every later count wrong.
    """
    row = draft_row(draft, now=now)
    target = Path(path) if path is not None else _default_path()
    for existing in read_rows(target):
        if str(existing.get("thesis_id") or "") == row["thesis_id"]:
            return existing
    _append(target, row)
    return row


def record_interpretation(
    *,
    entry_id: str,
    supersedes: str,
    text: str,
    path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The trader's own reading, as a NEW row naming the draft it replaces.

    The draft stays exactly as the machine wrote it and the journal entry is
    never touched - a correction is a second statement, not an erasure.
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    target = Path(path) if path is not None else _default_path()

    parent: dict[str, Any] = {}
    for existing in read_rows(target):
        if str(existing.get("thesis_id") or "") == str(supersedes or ""):
            parent = existing
    row = {
        "thesis_id": _thesis_id(entry_id, supersedes, stamp, str(text)),
        "entry_id": str(entry_id or ""),
        "extractor_version": str(parent.get("extractor_version") or EXTRACTOR_VERSION),
        "kind": str(parent.get("kind") or KIND_THESIS),
        "supersedes": str(supersedes or ""),
        "recorded_at": stamp,
        # Carried from the draft so the clock this row is judged on is the one
        # the ORIGINAL thought was written with. A trader's later reading of a
        # Tuesday thesis is still about Tuesday.
        "session_date": str(parent.get("session_date") or ""),
        "created_at": str(parent.get("created_at") or ""),
        "claim": str(parent.get("claim") or ""),
        "horizon": str(parent.get("horizon") or UNSTATED),
        "horizon_sessions": int(parent.get("horizon_sessions") or 0),
        "stance": str(parent.get("stance") or UNSTATED),
        "condition": str(parent.get("condition") or UNSTATED),
        "invalidation": str(parent.get("invalidation") or UNSTATED),
        "benchmarks": list(parent.get("benchmarks") or ()),
        "spans": dict(parent.get("spans") or {}),
        "is_prediction": bool(parent.get("is_prediction") or False),
        "questions": list(parent.get("questions") or ()),
        "interpretation_by": "trader",
        "text": str(text or ""),
    }
    _append(target, row)
    return row


def record_forecast(
    *,
    entry_id: str,
    text: str,
    source_model: str = "",
    created_at_claimed: str = "",
    target_week: str = "",
    scenarios: Iterable[str] = (),
    links: Iterable[str] = (),
    session_date: str = "",
    path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """WISHLIST 10K: an imported forecast's sidecar. NEVER a thesis.

    ``created_at_claimed`` stays :data:`UNKNOWN` when nobody supplied it. It is
    NOT filled from the import moment, because a later import is not
    information known earlier - that difference is the entire reason the two
    timestamps are separate fields.
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    imported_at = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    row = {
        "thesis_id": _thesis_id(entry_id, KIND_FORECAST, imported_at),
        "entry_id": str(entry_id or ""),
        "extractor_version": EXTRACTOR_VERSION,
        "kind": KIND_FORECAST,
        "supersedes": "",
        "recorded_at": imported_at,
        "session_date": str(session_date or ""),
        "source_model": str(source_model or "").strip() or UNKNOWN,
        "created_at_claimed": str(created_at_claimed or "").strip() or UNKNOWN,
        "imported_at": imported_at,
        "target_week": str(target_week or "").strip() or UNKNOWN,
        "scenarios": [str(item) for item in (scenarios or ())],
        "links": [str(item) for item in (links or ())],
        "text": str(text or ""),
    }
    _append(Path(path) if path is not None else _default_path(), row)
    return row


def _append(path: Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(row), default=str, sort_keys=True, separators=(",", ":"))
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def read_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Every row on disk, oldest first. A missing store is an empty list."""
    target = Path(path) if path is not None else _default_path()
    rows: list[dict[str, Any]] = []
    try:
        handle = target.open("r", encoding="utf-8")
    except OSError:
        return rows
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                logging.debug("market_theses: unreadable row skipped.")
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def current_theses(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The current view: a superseded row is hidden, never removed."""
    listed = [dict(row) for row in rows or ()]
    replaced = {str(row.get("supersedes") or "") for row in listed if row.get("supersedes")}
    return [row for row in listed if str(row.get("thesis_id") or "") not in replaced]


def active_theses(rows: Iterable[Mapping[str, Any]], *, as_of: date) -> list[dict[str, Any]]:
    """The theses still open on `as_of`. A FORECAST is never one of them."""
    out: list[dict[str, Any]] = []
    for row in current_theses(rows):
        if str(row.get("kind") or "") != KIND_THESIS:
            continue
        status, reason = status_for_row(row, as_of=as_of)
        if status == STATUS_OPEN:
            out.append({**row, "status": status, "status_reason": reason})
    return out


def status_for_row(row: Mapping[str, Any], *, as_of: date) -> tuple[str, str]:
    """`resolve_status` over a stored row rather than a live draft."""
    draft = ThesisDraft(
        entry_id=str(row.get("entry_id") or ""),
        session_date=str(row.get("session_date") or ""),
        created_at=str(row.get("created_at") or ""),
        extractor_version=str(row.get("extractor_version") or EXTRACTOR_VERSION),
        kind=str(row.get("kind") or KIND_THESIS),
        claim=str(row.get("claim") or ""),
        horizon=str(row.get("horizon") or UNSTATED),
        horizon_sessions=int(row.get("horizon_sessions") or 0),
        stance=str(row.get("stance") or UNSTATED),
        condition=str(row.get("condition") or UNSTATED),
        invalidation=str(row.get("invalidation") or UNSTATED),
        benchmarks=tuple(str(item) for item in (row.get("benchmarks") or ())),
        spans={},
        is_prediction=bool(row.get("is_prediction") or False),
    )
    return resolve_status(draft, as_of=as_of)
