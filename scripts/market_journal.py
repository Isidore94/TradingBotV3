"""The trader's written record of the day — R10.H.

R10.G records what the machine saw. This is the other half: what the trader
saw, in their own words. The two together are what lets a later reader — human
or model — understand a session rather than re-derive it, which is the whole
point of the two-tier design.

**One writer, two surfaces.** A "Journal" tab on the Trading Desk after
"Capture" for a note written mid-session, and a left-nav "Market Journal" page
for the sit-down review. Both go through this module, so an entry means the
same thing whichever surface it came from, and there is exactly one store to
reason about.

The existing left-nav "Journal" page stays the **trade/tax** journal. The label
collision is deliberate and recorded: one is a record of what you traded, the
other of what you thought, and merging them would make the tax journal a diary.

**After-the-fact entries are first class, and never backdated** (decision
record §5a). On an AWAY day the trader comes in that evening and writes about
the session. That entry carries `session_date` = the session it is ABOUT and
`created_at` = when it was actually written, both tz-aware. An entry about
Friday written on Saturday says so, because a reader weighing "what did you
think at the time?" needs to know it was not written at the time.

Append-only, schema-NAMED, corrections supersede (ground rule 5). An edit is a
new entry naming the one it replaces; nothing is rewritten, because a journal
that can be quietly rewritten is not evidence about what anyone believed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

#: Schema NAME (ground rule 5).
SCHEMA_MARKET_JOURNAL_ENTRY = "market_journal_entry_v1"
STREAM = "market_journal"

#: What a timeframe label may be. M5 is the in-session default because the desk
#: tab is used while the tape is moving; D1 is the review default.
TIMEFRAME_M5 = "M5"
TIMEFRAME_D1 = "D1"
TIMEFRAMES = (TIMEFRAME_M5, "M15", "H1", TIMEFRAME_D1, "W1")

#: How an entry reached the store, kept because it changes how it should be read.
ORIGIN_DESK_TAB = "desk_tab"
ORIGIN_JOURNAL_PAGE = "journal_page"
ORIGIN_AWAY_RECAP = "away_recap"
#: The desk's own hand. An auto-mode flip writes a row so the journal reads as
#: one timeline - what the trader thought AND what the machine did, in order -
#: rather than requiring two stores to be merged by eye. It is marked as
#: machine-written because a reader weighing "what did you think?" must never
#: count a row nobody thought.
ORIGIN_AUTO_MODE_FLIP = "auto_mode_flip"
MACHINE_ORIGINS = (ORIGIN_AUTO_MODE_FLIP,)
#: WISHLIST 10J. A read the Trade Mentor ASKED for. The trader wrote every word
#: of it, so it is not a machine origin - what the origin records is that the
#: desk chose the moment, which is exactly what a later reader needs to tell a
#: prompted read from a volunteered one.
ORIGIN_TRADE_MENTOR = "trade_mentor"
#: WISHLIST 10K / packet WS-10D. Someone ELSE's words, pasted in whole - the
#: weekly forecast the trader asks a chat model for. It is stored in this
#: journal because it is part of the week's record, and it carries its own
#: origin because it is the one kind of entry the trader did not write: outside
#: commentary, never their adopted view. A story shows it under its own
#: heading, `market_thesis` files it as `kind=forecast`, and nothing turns it
#: into a thesis unless the trader writes an entry of their own adopting it.
ORIGIN_EXTERNAL_FORECAST = "external_forecast"

#: The journal-only RVOL floor. It is an OVERLAY on this page's charts and
#: never touches the canonical D1 level store (trader decision, plan.md L1118).
RVOL_OVERLAY_FLOOR = 1.2

# ---------------------------------------------------------------------------
# A description is not a prediction - TJ-14A item 1, decision 0021 answer 29
# ---------------------------------------------------------------------------
#: The trader's own words, 2026-09-19: *"make sure we differentiate predictions
#: from just 'describe the market and your thoughts'"*. A Mentor row therefore
#: carries TWO keys that never share a field: `mentor.observation` is what they
#: SEE, and `mentor.prediction` is what they EXPECT - a click, with its horizon
#: printed on the row it was made on. Only the click is graded; the words are
#: context. Measured that day, `market_thesis.extract_thesis` read 21 of the
#: trader's 42 real notes as `unstated`, which is why the call is a click.
PREDICTION_SCHEMA = "mentor_prediction_v1"
#: Every card asks the first; only the 08:00 and 12:00 D1 cards ask the second.
HORIZON_REST_OF_DAY = "rest_of_day"
HORIZON_NEXT_5_SESSIONS = "next_5_sessions"
#: "I have no call" is a COMPLETE answer and is never graded.
DIRECTION_NO_VIEW = "no_view"
#: Two vocabularies, because a day has chop and five sessions have a range.
DIRECTIONS = {
    HORIZON_REST_OF_DAY: ("up", "down", "chop", DIRECTION_NO_VIEW),
    HORIZON_NEXT_5_SESSIONS: ("up", "down", "range", DIRECTION_NO_VIEW),
}
#: Forced whenever the direction is not `no_view` (decision 0021 answer 29: a
#: prediction IS direction, horizon and confidence; only `because` is optional).
CONFIDENCE_LEVELS = ("low", "medium", "high")
#: Which horizon one stored entry's timeframe carries. A row's `timeframe` and
#: its prediction's `horizon` ALWAYS agree, and the rule is enforced at the
#: WRITER (:func:`build_entry` raises, :func:`is_publishable` refuses) rather
#: than trusted at each call site. The defect that made it a rule: a
#: `Read unchanged` on the 09:00 card, handed the 08:00 D1 row as "your last
#: read", filed a D1-timeframe entry carrying a `rest_of_day` call - a row that
#: is true of neither timeframe, permanent in an append-only ledger, and
#: poison for TJ-16's calibration.
HORIZON_FOR_TIMEFRAME = {
    TIMEFRAME_M5: HORIZON_REST_OF_DAY,
    TIMEFRAME_D1: HORIZON_NEXT_5_SESSIONS,
}


def note_swallowed(reason, exc=None, **kwargs):
    """Log a swallowed failure via ``swallowed`` (imported lazily: scripts/ may not be on sys.path)."""
    try:
        from swallowed import note_swallowed as _note
    except ImportError:
        return
    _note(reason, exc, **kwargs)


class PredictionTimeframeError(ValueError):
    """A row whose timeframe and prediction horizon disagree. Never stored."""


def _check_prediction_timeframe(timeframe: str, prediction: Any) -> str:
    """`""` when the pair agrees, otherwise the sentence that says why not."""
    if not isinstance(prediction, Mapping):
        return ""
    horizon = str(prediction.get("horizon") or "").strip().lower()
    if not horizon:
        return ""
    expected = HORIZON_FOR_TIMEFRAME.get(str(timeframe or "").strip().upper())
    if expected is None:
        return (
            f"a {timeframe} entry carries no prediction horizon; only "
            f"{TIMEFRAME_M5} and {TIMEFRAME_D1} rows may hold a call"
        )
    if horizon != expected:
        return (
            f"a {timeframe} entry may only carry a {expected} call, not "
            f"{horizon}; a row whose timeframe and horizon disagree is true of "
            "neither"
        )
    return ""


@dataclass(frozen=True)
class Prediction:
    """One clicked call, exactly as it was stored. Never derived from words."""

    direction: str
    horizon: str
    confidence: str
    because: str
    schema: str = PREDICTION_SCHEMA

    @property
    def is_no_view(self) -> bool:
        return self.direction == DIRECTION_NO_VIEW


def build_prediction(
    *, direction: str, horizon: str, confidence: str = "", because: str = ""
) -> dict[str, Any]:
    """The stored shape of one click. `no_view` carries no confidence."""
    call = str(direction or "").strip().lower()
    span = str(horizon or "").strip().lower()
    level = str(confidence or "").strip().lower()
    return {
        "direction": call,
        "horizon": span,
        "confidence": "" if call == DIRECTION_NO_VIEW else level,
        "because": str(because or "").strip(),
        "schema": PREDICTION_SCHEMA,
    }


#: How each horizon prints on a screen. Display only - nothing is stored here.
HORIZON_TEXT = {
    HORIZON_REST_OF_DAY: "Rest of day",
    HORIZON_NEXT_5_SESSIONS: "Next 5 sessions",
}
DIRECTION_TEXT = {
    "up": "Up",
    "down": "Down",
    "chop": "Chop",
    "range": "Range",
    DIRECTION_NO_VIEW: "No view",
}


def prediction_line(entry: Mapping[str, Any]) -> str:
    """One readable line for a clicked call, or ``""``. DISPLAY ONLY.

    A card answered with clicks and no words stores `text == ""` - nobody wrote
    a sentence and the desk does not write one for them (decision 0021 answer
    29). A SCREEN still has to show something other than a blank row, so the
    surfaces that print journal text fall back to this. It is built here, once,
    so two pages cannot word the same call differently, and it is never written
    to the store.
    """
    call = prediction_of(entry)
    if call is None:
        return ""
    horizon = HORIZON_TEXT.get(call.horizon, call.horizon)
    direction = DIRECTION_TEXT.get(call.direction, call.direction)
    line = f"{horizon}: {direction}"
    if call.confidence:
        line += f" ({call.confidence} confidence)"
    if call.because:
        line += f" - {call.because}"
    return line


def prediction_of(entry: Mapping[str, Any]) -> Prediction | None:
    """The clicked call on one entry, or ``None``. The ONE accessor.

    It reads `mentor.prediction` and NOTHING else - never `text`, never
    `observation`. A reader that fell back to the words for a row that has a
    click would quietly re-introduce the extraction this packet replaced, and
    would pool an inferred stance with a stated one (decision 0021 answer 29).

    ``None`` for every row written before TJ-14A. The live September file holds
    four such vintages - no `mentor` key, `mentor == {}`, a mentor payload with
    no context, and a full v1 context - and not one of them is a click.
    """
    if not isinstance(entry, Mapping):
        return None
    mentor = entry.get("mentor")
    if not isinstance(mentor, Mapping):
        return None
    payload = mentor.get("prediction")
    if not isinstance(payload, Mapping):
        return None
    horizon = str(payload.get("horizon") or "").strip().lower()
    direction = str(payload.get("direction") or "").strip().lower()
    if horizon not in DIRECTIONS or direction not in DIRECTIONS[horizon]:
        return None
    confidence = str(payload.get("confidence") or "").strip().lower()
    return Prediction(
        direction=direction,
        horizon=horizon,
        confidence="" if direction == DIRECTION_NO_VIEW else confidence,
        because=str(payload.get("because") or ""),
        schema=str(payload.get("schema") or PREDICTION_SCHEMA),
    )


# ---------------------------------------------------------------------------
# A mood is a field the desk REPORTS - TJ-7 change 1
# ---------------------------------------------------------------------------
#: The stored block's own name. A schema id is a permanent identifier.
MOOD_SCHEMA = "trader_mood_v1"
#: Five faces. Not a scale anyone averages here - TJ-7 ships the bones only and
#: mood-vs-outcome statistics are a later, trader-directed phase.
MOOD_SCALE = (1, 2, 3, 4, 5)
#: `Followed the plan:` - the three answers, plus the fourth state "nothing
#: clicked", which stores ``""`` and is never a default "yes" or "no".
FOLLOWED_PLAN_VALUES = ("yes", "partly", "no")
#: The number `plan.md` TJ-7 change 1 names. 200 stores; 201 is refused.
PROCESS_NOTE_MAX = 200


class MoodFieldError(ValueError):
    """A mood nobody could have clicked. Never stored, never truncated.

    The writer is LOUD for the same reason :class:`PredictionTimeframeError` is:
    this ledger is append-only, so a row written wrong is wrong forever. A UI
    caps its own input; a caller that gets past the cap gets an exception rather
    than a quietly shortened note.
    """


@dataclass(frozen=True)
class Mood:
    """One recorded mood, exactly as it was clicked. The ONE read shape.

    `recorded_after_the_session` is READ off the row's own
    `written_after_the_session`, which :func:`build_entry` COMPUTES and never
    backdates - it is a LABEL on a mood typed in the evening, never a reason to
    drop one.
    """

    score: int | None
    state_tags: tuple[str, ...]
    followed_plan: str
    note: str
    vocab_version: int | None
    recorded_after_the_session: bool
    at: datetime | None
    entry_id: str
    schema: str = MOOD_SCHEMA


def _state_tag_owner():
    """The vocabulary module, read AT CALL TIME. One owner of the cap."""
    import trader_state_tags

    return trader_state_tags


#: The sentence a machine row carrying a mood is refused with. A mood is the
#: TRADER's own click; the desk has no feelings to record and an auto-mode flip
#: that filed one would put a feeling in the ledger nobody had. The structural
#: fence (`test_only_the_traders_own_surfaces_hand_a_mood_to_the_journal`) reads
#: call sites; this one refuses the row itself (reviewer advisory 4).
MACHINE_MOOD_REFUSAL = (
    "a mood is the trader's own click; a machine-written row carries none"
)


def _check_mood_block(block: Any) -> str:
    """``""`` when this stored block is clickable, else the sentence that says why.

    Asked by :func:`build_mood` before anything is stored and again by
    :func:`is_publishable`, because a row assembled as a dict literal never
    calls the writer.
    """
    if block is None or not isinstance(block, Mapping) or not block:
        return ""
    owner = _state_tag_owner()
    score = block.get("score")
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, int) or score not in MOOD_SCALE:
            return (
                f"a mood of {score!r} is not one of {list(MOOD_SCALE)}; nobody "
                "could have clicked it"
            )
    raw_tags = block.get("state_tags") or ()
    if isinstance(raw_tags, (str, bytes)):
        return "a mood's state_tags is a list of codes, never one string"
    tags = [str(code) for code in raw_tags]
    cap = int(owner.MAX_STATE_TAGS)
    if len(tags) > cap:
        return f"a mood carries at most {cap} state_tags, not {len(tags)}"
    known = set(owner.codes())
    unknown = [code for code in tags if code not in known]
    if unknown:
        return f"a mood carries no state_tags outside its vocabulary: {unknown}"
    process = block.get("process")
    if process is not None and not isinstance(process, Mapping):
        return "a mood's process is a block with followed_plan and note"
    answer = str((process or {}).get("followed_plan") or "")
    if answer and answer not in FOLLOWED_PLAN_VALUES:
        return (
            f"a mood's followed_plan is one of {list(FOLLOWED_PLAN_VALUES)} or "
            f"nothing at all, never {answer!r}"
        )
    note = str((process or {}).get("note") or "")
    if len(note) > PROCESS_NOTE_MAX:
        return (
            f"a mood's process note is at most {PROCESS_NOTE_MAX} characters; "
            f"this one is {len(note)} and is refused rather than shortened"
        )
    return ""


def build_mood(
    *,
    score: Any = None,
    state_tags: Iterable[str] = (),
    followed_plan: Any = "",
    note: Any = "",
) -> dict[str, Any]:
    """The stored `mood` block for one row, validated. Raises, never truncates.

    Every part is optional in both directions: a trader who clicks a face and
    nothing else, and one who answers `Followed the plan?` and clicks no face,
    have each given a complete answer. What is missing stores as `None` or
    `""` - never as a neutral 3 and never as a "no".
    """
    tags = tuple(str(code).strip() for code in (state_tags or ()) if str(code).strip())
    plan_answer = str(followed_plan or "").strip().lower()
    body = str(note or "")
    block: dict[str, Any] = {
        "schema": MOOD_SCHEMA,
        "score": score,
        "state_tags": list(tags),
        "process": {"followed_plan": plan_answer, "note": body},
    }
    problem = _check_mood_block(block)
    if problem:
        raise MoodFieldError(problem)
    # Stamped from the vocabulary itself, at write time, so a row stays
    # interpretable against exactly the list that produced it.
    block["vocab_version"] = int(_state_tag_owner().load_vocabulary()["vocab_version"])
    return block


def mood_of(entry: Mapping[str, Any]) -> Mood | None:
    """The mood on one entry, or ``None``. The ONE reader.

    Three absences mean the same thing and all three answer ``None``: the key
    ABSENT (every row written before TJ-7 - 43 of the 84 live rows on
    2026-09-20), the key PRESENT and EMPTY (a row written after TJ-7 with
    nothing clicked, the same shape as `"mentor": {}`), and the key set to
    `None`. A reader that turned any of them into a neutral 3 would invent a
    feeling nobody had.
    """
    if not isinstance(entry, Mapping):
        return None
    block = entry.get("mood")
    if not isinstance(block, Mapping) or not block:
        return None
    process = block.get("process") if isinstance(block.get("process"), Mapping) else {}
    score = block.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or score not in MOOD_SCALE:
        score = None
    version = block.get("vocab_version")
    if isinstance(version, bool) or not isinstance(version, int):
        version = None
    return Mood(
        score=score,
        state_tags=tuple(str(code) for code in (block.get("state_tags") or ())),
        followed_plan=str((process or {}).get("followed_plan") or ""),
        note=str((process or {}).get("note") or ""),
        vocab_version=version,
        recorded_after_the_session=bool(entry.get("written_after_the_session")),
        at=_entry_moment(entry),
        entry_id=str(entry.get("entry_id") or ""),
        schema=str(block.get("schema") or MOOD_SCHEMA),
    )


def _entry_moment(entry: Mapping[str, Any]) -> datetime | None:
    """One row's `created_at` as an AWARE moment, or ``None``."""
    raw = str(entry.get("created_at") or "").strip()
    if not raw:
        return None
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.astimezone()


def mood_at(entries: Iterable[Mapping[str, Any]], stamp: datetime) -> Mood | None:
    """The mood already recorded AT `stamp`, or ``None``. Point-in-time.

    The mood the trader clicks is usually on the session's last card - after
    the trade, and often after the close. A read made at 07:02 was made by
    someone who had not felt it yet, so a later mood is not context for it: it
    would be hindsight dressed as a measurement. Compared with `astimezone`,
    never by stripping a zone off either side.
    """
    if stamp is None:
        return None
    edge = stamp if stamp.tzinfo else stamp.astimezone()
    best: Mood | None = None
    for entry in entries or ():
        mood = mood_of(entry)
        if mood is None or mood.at is None:
            continue
        if mood.at.astimezone(timezone.utc) > edge.astimezone(timezone.utc):
            continue
        if best is None or mood.at.astimezone(timezone.utc) >= best.at.astimezone(timezone.utc):
            best = mood
    return best


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment


def entry_id(session_date: str, created_at: str, text: str, *, salt: str = "") -> str:
    """Stable id for one entry, from what it is about and when it was written.

    `salt` exists for TJ-14A's clicks-only answer. A D1 card filed with two
    calls and no words writes two rows in the same second with the same empty
    text, and without a salt they would share an id - which every join over
    this store uses as an identity. An entry that HAS words never takes a salt,
    so every id written before this packet is unchanged.
    """
    suffix = f"|{salt}" if salt else ""
    digest = hashlib.sha256(
        f"{session_date}|{created_at}|{text}{suffix}".encode("utf-8")
    ).hexdigest()[:12]
    return f"mj-{session_date}-{digest}"


def build_entry(
    *,
    text: str,
    session_date: str,
    timeframe: str = TIMEFRAME_M5,
    symbols: Iterable[str] = (),
    origin: str = ORIGIN_DESK_TAB,
    now: datetime | None = None,
    supersedes: str = "",
    mentor: Mapping[str, Any] | None = None,
    reaffirms: str = "",
    mood: Any = None,
    state_tags: Iterable[str] = (),
    process: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One journal entry.

    `session_date` is what the entry is ABOUT. `created_at` is when it was
    written. They are separate fields precisely so an evening write-up of an
    AWAY day can be honest about both - and `written_after_the_session` is
    computed rather than asserted, so it cannot be set wrongly by a caller.

    `mentor` and `reaffirms` are WISHLIST 10J's two additions, and they are
    fields on THIS row rather than a second store, because a read and the
    prompt it answers are one fact (packet WS-TM). `mentor` carries the slot
    that asked (`slot_id`, `prompt_kind`, `scheduled_at`) and, separately, the
    moment the trader actually replied (`responded_at`) - a reply typed at 09:12
    cannot claim to describe the market at 09:00, and `created_at` alone cannot
    say which hour was being asked about. `reaffirms` names the earlier entry a
    "Read unchanged" restates. It is deliberately NOT `supersedes`: superseding
    would hide the read it reaffirms, and "I still think what I thought at 09:00"
    is a new observation at 11:00, not a correction of the old one.

    `mood`, `state_tags` and `process` are TJ-7's three optional arguments and
    they store as ONE additive key, `mood` - present and empty when nothing was
    clicked, ABSENT on every row written before this packet. They move no other
    field on the row and they do not enter `entry_id`: a mood is a field ON an
    entry, never part of what names it.
    """
    moment = _now(now)
    created_at = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    body = str(text or "").strip()
    session = str(session_date or "").strip()
    written_session = moment.astimezone().date().isoformat()
    timeframe_text = _normalize_timeframe(timeframe)
    # Only a WORDLESS row is salted, and only by the two facts that tell two
    # such rows apart: which timeframe it is about and which call it carries.
    clicked = (mentor or {}).get("prediction") if isinstance(mentor, Mapping) else None
    # Loud, at the writer: a caller that would file a row whose timeframe and
    # horizon disagree gets an exception, not a stored row. An append-only
    # ledger has no second chance at this.
    mismatch = _check_prediction_timeframe(timeframe_text, clicked)
    if mismatch:
        raise PredictionTimeframeError(mismatch)
    salt = (
        ""
        if body
        else f"{timeframe_text}|{(clicked or {}).get('horizon') or ''}"
        if isinstance(clicked, Mapping)
        else ""
    )
    # Nothing clicked is `{}`, not a default mood. `mood is None` rather than
    # `not mood`, because a mood of 0 is a REFUSAL and must reach the writer.
    clicked_a_mood = mood is not None or bool(state_tags) or process is not None
    if process is not None and not isinstance(process, Mapping):
        raise MoodFieldError(
            "a mood's process is a block with followed_plan and note, not "
            f"{type(process).__name__}"
        )
    if clicked_a_mood and is_machine_entry({"origin": str(origin or "")}):
        # `is_machine_entry` is the ONE rule for what a machine row is (TJ-1),
        # asked here rather than re-spelled, so a new machine origin is fenced
        # the day it is added.
        raise MoodFieldError(MACHINE_MOOD_REFUSAL)
    mood_block = (
        build_mood(
            score=mood,
            state_tags=state_tags,
            followed_plan=(process or {}).get("followed_plan"),
            note=(process or {}).get("note"),
        )
        if clicked_a_mood
        else {}
    )
    return {
        "event_type": "entry",
        "entry_id": entry_id(session, created_at, body, salt=salt),
        "session_date": session,
        "created_at": created_at,
        "created_local_date": written_session,
        # Computed, never claimed. An entry about Friday written on Saturday
        # is weaker evidence about what the trader thought at the time, and a
        # reader must not have to work that out from two timestamps.
        #
        # R4 A17: measured against the session's CLOSE, not against its date.
        # A note typed at 21:00 Pacific is written five hours after the market
        # shut, and under the date rule it claimed to have been written during
        # the session - which is the one thing this field exists to deny. The
        # date rule is kept as the fallback for a session the calendar cannot
        # place, because a slightly coarse answer is better than none.
        "written_after_the_session": _written_after_the_session(session, moment, written_session),
        "timeframe": timeframe_text,
        "symbols": [str(item).strip().upper() for item in (symbols or ()) if str(item).strip()],
        "origin": str(origin or ""),
        "text": body,
        "supersedes": str(supersedes or ""),
        # Present and empty on every other entry, never absent: a reader that
        # has to tell "no prompt asked for this" from "this key did not exist
        # yet" is reading two different absences as one.
        "mentor": dict(mentor or {}),
        "reaffirms": str(reaffirms or ""),
        # TJ-7. Present and empty for the same reason `mentor` is: a reader
        # that has to tell "nothing was clicked" from "this key did not exist
        # yet" is reading two different absences as one.
        "mood": mood_block,
    }


def _written_after_the_session(session: str, moment: datetime, written_session: str) -> bool:
    """Had the trader already seen how that session finished?

    The honest reading of the question, which is what the field's name has
    always claimed and what `session_date_for`'s own docstring describes. At or
    after the session's regular close, the answer is yes - the same calendar the
    rest of the desk uses decides when that was.

    Falls back to the calendar-date comparison when the close cannot be read: an
    entry about Friday written on Saturday is still, unambiguously, after it.
    """
    if not session:
        return False
    try:
        from datetime import date as _date

        from market_calendar import session_close

        parts = [int(part) for part in session.split("-")]
        close = session_close(_date(parts[0], parts[1], parts[2]))
        stamp = moment if moment.tzinfo else moment.astimezone()
        return stamp >= close
    except Exception:  # noqa: BLE001 - a coarse answer beats no answer
        return bool(written_session > session)


def session_date_for(now: datetime | None = None) -> str:
    """The session a note typed NOW is about - V2 item 4, decision 0016 answer 11.

    Today's session until the close; then STILL today's session, right up to the
    next session's open. A thought written at 18:00 is about the day that just
    ended, and dating it tomorrow would file it against a session that has not
    happened yet. On a weekend or a holiday it is the last session that traded.

    **The roll is the OPEN, not midnight** (R4 A17). This read the calendar date
    in New York, so a Pacific trader typing at 21:00 PT - which is 00:00 ET the
    next day - filed their note against TOMORROW'S session, on a day that had
    not opened. Worse, `written_after_the_session` then computed False, so the
    row claimed the note was written during a session that had not started. The
    trader's own rule: the session ends at the close, and the note is about that
    session until the next one opens.

    This does NOT touch `written_after_the_session`, which `build_entry` still
    COMPUTES from `created_at`. The two answer different questions - which day
    the note is about, and whether the trader had already seen how that day
    finished - and the second is the one a reader needs to discount the first.

    Falls back to the local date if the calendar cannot be read. A note filed
    against today is a small error; a note that could not be written is a lost
    thought, and this function must never be the reason one is.
    """
    moment = _now(now)
    local = moment.astimezone()
    try:
        from market_calendar import is_session, previous_session
        from market_session import get_market_session_window

        window = get_market_session_window(reference=local)
        market_date = window.market_date
        if is_session(market_date) and local >= window.open_local:
            # Today TRADES and it has OPENED. Whether the close has passed or
            # not, the note is about today - before the close it is the running
            # session, after it the one that just finished.
            return market_date.isoformat()
        # Either today never trades (a weekend or a holiday) or it has not
        # opened yet - the small hours in New York, which is the evening on the
        # trader's own clock. Both mean the same thing: the note is about the
        # last session that actually traded.
        return previous_session(market_date).isoformat()
    except Exception as exc:  # noqa: BLE001 - never the reason a thought is lost
        note_swallowed("market calendar unavailable for the note's session; using the local date", exc, quiet=True)
    return local.date().isoformat()


def session_of_entry(entry: Mapping[str, Any]) -> str:
    """Which session a STORED entry is about - the one selection rule (WS-10D).

    Phase 0.31 preserves an explicit subject in `session_date` and records the
    actual write day separately as `written_session_date`. Older rows predate
    that contract, so this reader keeps the created-at fallback for them.
    """
    subject = str(entry.get("session_date") or "").strip()
    if subject and str(entry.get("written_session_date") or "").strip():
        return subject
    raw = str(entry.get("created_at") or "").strip()
    if raw:
        try:
            return session_date_for(datetime.fromisoformat(raw))
        except Exception as exc:  # noqa: BLE001 - never lose an entry to a calendar
            note_swallowed("entry created_at unparseable; using its stored session date", exc, quiet=True)
    return str(entry.get("session_date") or "")


def _normalize_timeframe(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in TIMEFRAMES else TIMEFRAME_M5


def is_publishable(entry: Mapping[str, Any]) -> tuple[bool, str]:
    """May this entry be written?

    An empty entry is refused rather than stored: a journal full of blanks is
    worse than a shorter one, because it makes the record look denser than the
    thinking behind it.

    TJ-14A item 1 opens ONE door in that rule: a Mentor card answered with a
    CLICK and no words is a complete answer, and its `text` stays empty because
    nobody wrote a sentence. The relaxation is as narrow as it sounds - the row
    must carry a clicked `mentor.prediction` - so every other empty-text entry
    is refused exactly as before.
    """
    mentor = entry.get("mentor")
    mismatch = _check_prediction_timeframe(
        str(entry.get("timeframe") or ""),
        mentor.get("prediction") if isinstance(mentor, Mapping) else None,
    )
    if mismatch:
        # The same rule as `build_entry`'s raise, asked again at the gate every
        # write passes through - a row assembled as a dict literal reaches here
        # without ever calling `build_entry`.
        return False, mismatch
    # TJ-7, asked at the same gate and for the same reason: a row assembled as
    # a dict literal reaches here without ever calling `build_entry`, and an
    # append-only ledger has no second chance at a mood nobody could have
    # clicked.
    bad_mood = _check_mood_block(entry.get("mood"))
    if bad_mood:
        return False, bad_mood
    if entry.get("mood") and is_machine_entry(entry):
        return False, MACHINE_MOOD_REFUSAL
    if not str(entry.get("text") or "").strip() and prediction_of(entry) is None:
        return False, "an empty entry is not a thought; nothing is stored"
    if not str(entry.get("session_date") or "").strip():
        return False, "an entry with no session is unfiled and could never be read back"
    return True, ""


def supersede(entry: Mapping[str, Any], *, text: str, now: datetime | None = None) -> dict[str, Any]:
    """A correction: a NEW entry naming the one it replaces (ground rule 5).

    The original stays exactly as written. A journal that can be quietly
    rewritten is not evidence about what anyone believed - it is evidence about
    what they are willing to admit now.
    """
    return build_entry(
        text=text,
        session_date=str(entry.get("session_date") or ""),
        timeframe=str(entry.get("timeframe") or TIMEFRAME_M5),
        symbols=entry.get("symbols") or (),
        origin=str(entry.get("origin") or ORIGIN_JOURNAL_PAGE),
        now=now,
        supersedes=str(entry.get("entry_id") or ""),
    )


def is_machine_entry(entry: Mapping[str, Any]) -> bool:
    """Did the desk write this row, rather than the trader?

    Asked at READ time off ``origin``, so no row needs a second field that
    could disagree with the first.
    """
    return str(entry.get("origin") or "") in MACHINE_ORIGINS


def resolve_entries(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The current view: superseded entries hidden, originals still on disk.

    Hiding is a READ-side decision. Every row remains in the ledger, so the
    history of a correction is recoverable even though the page shows only the
    latest text.
    """
    entries = [dict(row) for row in rows if str(row.get("event_type") or "") == "entry"]
    replaced = {str(row.get("supersedes") or "") for row in entries if row.get("supersedes")}
    current = [row for row in entries if str(row.get("entry_id") or "") not in replaced]
    current.sort(key=lambda row: (str(row.get("session_date") or ""), str(row.get("created_at") or "")))
    return current


def agreement_rate(shifts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """How often the machine's auto regime matched what the trader forced.

    Measured over regime-shift rows (R10.G). A session where the trader never
    overrode is AGREEMENT by silence and is counted as such; a session with no
    auto read at all is not counted either way, because there was nothing to
    agree with. Returns `None` rather than a rate when nothing is measurable -
    an agreement rate over zero comparable sessions is not 100%.
    """
    by_session: dict[str, dict[str, str]] = {}
    for row in shifts or ():
        session = str(row.get("session_date") or "")
        source = str(row.get("source") or "")
        target = str(row.get("to_regime") or "")
        if not session or not target:
            continue
        by_session.setdefault(session, {})[source] = target

    compared = 0
    agreed = 0
    for sources in by_session.values():
        auto = sources.get("auto")
        if not auto:
            continue
        compared += 1
        user = sources.get("user")
        if user is None or user == auto:
            agreed += 1
    return {
        "sessions_compared": compared,
        "sessions_agreed": agreed,
        "rate": round(agreed / compared, 4) if compared else None,
        "note": (
            "a session the trader never overrode counts as agreement; a session "
            "with no auto read is not counted either way"
            if compared
            else "no session carried an auto regime read, so the rate is UNMEASURED"
        ),
    }
