r"""`exit_note_fields` - the NIGHT reads an exit note and drafts three fields.

Trader, 2026-09-21: *"trade exits should ask for 'why did you exit, what
emotions did you have, what technicals were you observing' ideally we can just
write it out and the AI fills this stuff in overnight."*

So: the trader writes ONE box at 09:00, the words go to the journal at once,
and this slot reads them hours later and drafts `why`, `felt` and `watching`.
Every value is grounded in an exact span of the trader's own sentence, the
draft is `provisional` until the trader presses Confirm or Correct, and the
model is never told how the trade turned out.

The pattern is :mod:`ai_jobs.observation_tags` (TJ-16) and every rule that slot
learned the hard way is repeated here.

1. **The package is the NOTE and the two PICKLISTS, and nothing else.** It is
   BUILT from the note's own five fields - not filtered down from a trade row -
   so there is no key to forget to delete. The P&L of the very trade the note
   is about is one join away in the same store, and a draft derived from the
   result is not a reading of the words, it is a rationalisation of the outcome.
2. **Two closed vocabularies, each read from its OWNER.** `why` comes from
   :mod:`exit_reasons`; `felt` from :mod:`trader_state_tags`, which is TJ-7's
   list - there is ONE vocabulary of feelings on this desk, because an exit's
   feeling and a session's mood are different grains of the same words. Both
   are read at CALL time, so a v2 shipping beside a v1 needs no edit here.
3. **A span is a QUOTATION.** ``text[start:end]`` must equal ``quote`` exactly,
   counted in CHARACTERS. One value that does not reproduce rejects the WHOLE
   reply for that note - not the value - and the last verified file stays
   byte-identical, because a half-accepted answer is a file nobody can trust
   and nobody can tell apart from a whole one.
4. **A JSON schema is a grammar hint, never a guard.** :func:`verify_reply`
   re-checks every bound itself. See :data:`REPLY_JSON_SCHEMA` for why this
   slot's contract deliberately does NOT declare ``additionalProperties`` at
   the top level.
5. **Night-only, and asked once.** The slot declares ``uses_model``, so
   ``--force`` may not buy it the daytime clock (TJ-13A item 1); nothing
   waiting means no model is loaded at all; and the window is re-asked before
   EVERY call, so a run near the close stops cleanly mid-queue and keeps what
   it did not reach for tomorrow.
6. **A field the note does not speak to is ABSENT**, never guessed. A guessed
   `felt` would be the machine putting a feeling in the trader's mouth, and
   TJ-7's whole rule is that only the trader names one.

Nothing it writes reaches a detector, score, alert, watchlist, Focus, the
review queue or `review_policy.json`, and nothing here may become the trader's
record without their own click.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import journal_store
import trader_state_tags
from ai_jobs import ledger, window

_log = logging.getLogger(__name__)

#: The prompt contract's NAME, stamped on every stored file.
PROMPT_VERSION = "exit_note_fields_v1"
#: The stored file's own schema name.
SCHEMA = "exit_note_fields_v1"
#: What the response_format contract is called on the wire.
SCHEMA_NAME = "tradingbot_exit_note_fields"

#: What every drafts file is called, before any superseding sibling.
FILE_PREFIX = "exit_note_fields"

#: The three fields one note may produce. There is no fourth, and `watching`
#: has no vocabulary yet on purpose: nothing is invented before there are notes
#: to read (lead decision 4).
FIELD_WHY = "why"
FIELD_FELT = "felt"
FIELD_WATCHING = "watching"
FIELDS = (FIELD_WHY, FIELD_FELT, FIELD_WATCHING)

#: How many feelings one exit may carry. READ from TJ-7's owner, never typed:
#: two chips is a picklist and five is a form, and that number lives with the
#: vocabulary so the strips, the journal writer and this slot cannot drift.
MAX_FELT = trader_state_tags.MAX_STATE_TAGS

#: How many exact quotes `watching` may carry. A bound, so it is re-checked in
#: :func:`verify_reply` after the answer comes back.
MAX_WATCHING = 3

#: How many notes one night may draft. The cap is PACING, not a filter: what
#: does not fit is COUNTED and SAID in the ledger row and is the oldest work
#: waiting tomorrow. Twenty notes is twenty local calls of a few seconds each.
EXIT_NOTES_PER_NIGHT = 20

#: What one call reserves when the window is re-asked. The slot's own
#: `reserve_minutes` buys the FIRST call; every later one asks again with one
#: call's worth of room, which is `day_review_narration`'s rule.
CALL_MINUTES = 2.0

#: One local call, the same 900 s every other session-scale local call uses.
TIMEOUT_SECONDS = 900

#: The only keys a reply may carry, at each of its levels. They mirror the
#: nested ``additionalProperties: false`` in :data:`REPLY_JSON_SCHEMA`, which
#: for the reason recorded there cannot be relied on.
REPLY_KEYS = frozenset({"fields"})
FIELD_KEYS = frozenset(FIELDS)
CODED_VALUE_KEYS = frozenset({"code", "span", "quote"})
PLAIN_VALUE_KEYS = frozenset({"span", "quote"})

INSTRUCTIONS = (
    "Read what the trader wrote about closing ONE position and fill in at most "
    "three fields from their own words. `why` is ONE code from the exit_reasons "
    "list. `felt` is up to the cap of codes from the state_tags list. "
    "`watching` is up to three exact quotes from the note naming what they were "
    "looking at. For every value give the exact CHARACTER span of the note that "
    "made you choose it and that substring copied out as `quote`: `quote` must "
    "be exactly note.text[start:end], or the whole answer is thrown away. Use "
    "only codes from the lists you were given. You are NOT being told how the "
    "trade turned out and you are not being asked whether the exit was right: "
    "there is no price, no profit or loss, no later data and no grade in this "
    "package, and you must not guess at one. A field the note does not speak to "
    "must be LEFT OUT - do not invent a feeling the trader did not name. An "
    "answer with no fields at all is valid."
)


def _value_schema(coded: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": (["code", "span", "quote"] if coded else ["span", "quote"]),
        "properties": {
            "span": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer"},
            },
            "quote": {"type": "string", "maxLength": 400},
        },
    }
    if coded:
        body["properties"]["code"] = {"type": "string", "maxLength": 64}
    return body


#: THE CONTRACT HANDED TO THE PROVIDER - and a HINT, not a guard.
#:
#: It deliberately declares no ``additionalProperties`` at the TOP level.
#: `ai_summary.validate_structured_output` enforces that one key and RAISES,
#: and `_request_local_summary` answers a raised validation by RE-ASKING the
#: model once. That would spend a second call on an answer this module is about
#: to throw away whole - and the rejection has to be THIS module's, because
#: only it knows the note's text, the two live picklists and the counts. The
#: nested closed objects stay, because a constrained decoder reads them and
#: nothing re-asks on their account.
REPLY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["fields"],
    "properties": {
        "fields": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                FIELD_WHY: _value_schema(True),
                FIELD_FELT: {
                    "type": "array",
                    "maxItems": MAX_FELT,
                    "items": _value_schema(True),
                },
                FIELD_WATCHING: {
                    "type": "array",
                    "maxItems": MAX_WATCHING,
                    "items": _value_schema(False),
                },
            },
        }
    },
}


class ReplyRejected(ValueError):
    """The model's answer was not believed, so nothing was published."""


# ---------------------------------------------------------------------------
# the two vocabularies, each read from its owner at CALL time
# ---------------------------------------------------------------------------
def why_codes() -> tuple[str, ...]:
    """Every exit reason this desk offers, from :mod:`exit_reasons`."""
    import exit_reasons

    return tuple(exit_reasons.codes())


def felt_codes() -> tuple[str, ...]:
    """Every feeling this desk offers, from TJ-7's OWN loader.

    Read through `trader_state_tags.codes()` on every call rather than copied
    into a constant: there is one vocabulary of feelings on this desk, and a
    copy here would keep offering yesterday's list the day a v2 ships.
    """
    return tuple(trader_state_tags.codes())


def _vocabulary_book() -> dict[str, Any]:
    import exit_reasons

    why = exit_reasons.load_vocabulary()
    felt = trader_state_tags.load_vocabulary()
    return {
        FIELD_WHY: {
            "vocabulary_id": str(why.get("vocabulary_id") or ""),
            "vocab_version": int(why.get("vocab_version") or 0),
            "entries": [dict(entry) for entry in why.get("entries") or ()],
        },
        FIELD_FELT: {
            "vocabulary_id": str(felt.get("vocabulary_id") or ""),
            "vocab_version": int(felt.get("vocab_version") or 0),
            "max_codes": int(MAX_FELT),
            "entries": [dict(entry) for entry in felt.get("entries") or ()],
        },
        FIELD_WATCHING: {"max_quotes": int(MAX_WATCHING)},
    }


# ---------------------------------------------------------------------------
# what the model is handed - BUILT, never filtered down from a trade
# ---------------------------------------------------------------------------
def build_evidence(
    note: Mapping[str, Any], *, vocabularies: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The package as the model sees it: ONE note and the two picklists.

    Five keys come off the note - its id, the trader's words, the symbol, the
    side and the session the exit is in - and there is no sixth. This function
    knows about those five strings and two picklists and has no way to reach a
    price, a fill, a later bar or a grade, which is what makes the fence
    structural rather than a discipline somebody has to keep.
    """
    book = dict(vocabularies) if vocabularies is not None else _vocabulary_book()
    offered = {
        "note_id": str(note.get("note_id") or ""),
        "text": str(note.get("text") or note.get("raw_text") or ""),
        "symbol": str(note.get("symbol") or ""),
        "side": str(note.get("side") or ""),
        "exit_session": str(note.get("exit_session") or ""),
    }
    canonical = json.dumps(
        {"note": offered, "vocabularies": book},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "package_id": f"exit-note-fields:{digest[:16]}",
        "evidence_hash": digest,
        "instructions": INSTRUCTIONS,
        "note": offered,
        "vocabularies": book,
    }


# ---------------------------------------------------------------------------
# the grounding check - all of it, or none of it
# ---------------------------------------------------------------------------
def _grounded(value: Any, text: str, *, coded: bool, where: str) -> dict[str, Any]:
    """One value, or :class:`ReplyRejected`. Spans are CHARACTER offsets."""
    if not isinstance(value, Mapping):
        raise ReplyRejected(f"{where} is not an object")
    allowed = CODED_VALUE_KEYS if coded else PLAIN_VALUE_KEYS
    extra = sorted(str(key) for key in value if str(key) not in allowed)
    if extra:
        raise ReplyRejected(f"{where} carries key(s) the contract forbids: {', '.join(extra)}")
    span = value.get("span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise ReplyRejected(f"{where} has no two-number span")
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError) as exc:
        raise ReplyRejected(f"{where} has a span that is not numbers") from exc
    if start < 0 or end > len(text) or start >= end:
        raise ReplyRejected(f"{where} has a span outside the note: {[start, end]}")
    quote = str(value.get("quote") or "")
    if text[start:end] != quote:
        raise ReplyRejected(
            f"{where}: the span does not reproduce its quote - the note reads "
            f"{text[start:end]!r} where the answer said {quote!r}"
        )
    row: dict[str, Any] = {"span": [start, end], "quote": quote}
    if coded:
        row["code"] = str(value.get("code") or "")
    return row


def verify_reply(reply: Any, note: Mapping[str, Any]) -> dict[str, Any]:
    """Every field, or :class:`ReplyRejected`. One bad value rejects it WHOLE.

    Every bound the schema states is re-checked HERE, because the schema is a
    grammar hint the provider may or may not honour. A stray key at any level,
    an unknown code, a count over its cap, a repeated feeling or a span that
    does not reproduce its quote throws the whole answer away - and the last
    verified file is left exactly as it was.

    A repeated value is a rejection rather than a silent de-duplication: a
    model that repeats itself has not been verified, and folding the repeat
    away would store a file that does not say what the model returned.
    """
    if not isinstance(reply, Mapping):
        raise ReplyRejected("the model returned no answer object")
    stray = sorted(str(key) for key in reply if str(key) not in REPLY_KEYS)
    if stray:
        raise ReplyRejected(f"the reply carries key(s) the contract forbids: {', '.join(stray)}")
    fields = reply.get("fields")
    if fields is None:
        raise ReplyRejected("the reply carries no `fields` object")
    if not isinstance(fields, Mapping):
        raise ReplyRejected("`fields` is not an object")
    unknown = sorted(str(key) for key in fields if str(key) not in FIELD_KEYS)
    if unknown:
        raise ReplyRejected(f"`fields` carries key(s) the contract forbids: {', '.join(unknown)}")

    text = str(note.get("text") or note.get("raw_text") or "")
    out: dict[str, Any] = {}

    if FIELD_WHY in fields:
        value = _grounded(fields[FIELD_WHY], text, coded=True, where="`why`")
        if value["code"] not in why_codes():
            raise ReplyRejected(
                f"`why` uses {value['code']!r}, which is outside the closed vocabulary"
            )
        out[FIELD_WHY] = value

    if FIELD_FELT in fields:
        rows = fields[FIELD_FELT]
        if not isinstance(rows, (list, tuple)):
            raise ReplyRejected("`felt` is not a list")
        if len(rows) > MAX_FELT:
            raise ReplyRejected(
                f"`felt` carries {len(rows)} codes, over the cap of {MAX_FELT}"
            )
        allowed = felt_codes()
        seen: set[tuple[str, int, int]] = set()
        kept: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            value = _grounded(row, text, coded=True, where=f"`felt[{index}]`")
            if value["code"] not in allowed:
                raise ReplyRejected(
                    f"`felt` uses {value['code']!r}, which is outside the closed vocabulary"
                )
            identity = (value["code"], value["span"][0], value["span"][1])
            if identity in seen:
                raise ReplyRejected(f"the reply repeats one feeling: {value['code']!r}")
            seen.add(identity)
            kept.append(value)
        out[FIELD_FELT] = kept

    if FIELD_WATCHING in fields:
        rows = fields[FIELD_WATCHING]
        if not isinstance(rows, (list, tuple)):
            raise ReplyRejected("`watching` is not a list")
        if len(rows) > MAX_WATCHING:
            raise ReplyRejected(
                f"`watching` carries {len(rows)} quotes, over the cap of {MAX_WATCHING}"
            )
        spans: set[tuple[int, int]] = set()
        kept = []
        for index, row in enumerate(rows):
            value = _grounded(row, text, coded=False, where=f"`watching[{index}]`")
            identity = (value["span"][0], value["span"][1])
            if identity in spans:
                raise ReplyRejected(f"the reply repeats one quote: {value['quote']!r}")
            spans.add(identity)
            kept.append(value)
        out[FIELD_WATCHING] = kept

    return out


# ---------------------------------------------------------------------------
# reading the notes that are still waiting
# ---------------------------------------------------------------------------
def _default_store():
    """The trade journal, headless. Never a Qt service on a nightly worker."""
    from journal_store import JournalStore
    from project_paths import JOURNAL_DB_FILE

    return JournalStore(Path(JOURNAL_DB_FILE))


def notes_waiting(store: Any, session_date: str, *, root: Any = None) -> list[dict[str, Any]]:
    """Every exit note of one session with no verified draft, OLDEST FIRST.

    Oldest first is what makes the nightly cap honest: a night that can only
    reach twenty notes reaches the twenty that have been waiting longest, and
    says how many it left.

    A note is "drafted" by its own `note_id`, not by its trade: a SECOND note
    on the same exit - the trader changing their mind - is a new note and is
    read again tonight.
    """
    import trade_mentor_trade_check as check

    session = str(session_date or "")[:10]
    if not session:
        return []
    try:
        notes = check.exit_notes_for_session(store, session)
    except Exception:  # noqa: BLE001 - an unreadable store is no work
        _log.debug("exit_note_fields: the exit notes were unreadable.", exc_info=True)
        return []
    done = {
        str(draft.get("note_id") or "")
        for draft in (read_latest(session, root=root) or {}).get("drafts") or ()
    }
    waiting = [
        {
            "note_id": str(note.get("note_id") or ""),
            "trade_id": str(note.get("trade_id") or ""),
            "text": str(note.get("raw_text") or ""),
            "symbol": str(note.get("symbol") or ""),
            "side": str(note.get("side") or ""),
            "exit_session": str(note.get("exit_session") or session),
            "occurred_at": str(note.get("occurred_at") or ""),
        }
        for note in notes.values()
        if str(note.get("raw_text") or "").strip()
        and str(note.get("note_id") or "") not in done
    ]
    waiting.sort(key=lambda note: (note["occurred_at"], note["note_id"]))
    return waiting


# ---------------------------------------------------------------------------
# publishing and reading
# ---------------------------------------------------------------------------
def _pack_dir(root: Any = None, *, create: bool = True) -> Path:
    if root is not None:
        target = Path(root)
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target
    from ai_jobs import store

    return store.digests_dir(create=create)


def drafts_path(session_date: str, root: Any = None) -> Path:
    return _pack_dir(root) / f"{FILE_PREFIX}-{str(session_date)[:10]}.json"


def _publish(payload: Mapping[str, Any], root: Any = None) -> Path:
    from ai_jobs.digest import _publish as publish, superseding_path

    target = superseding_path(drafts_path(str(payload.get("session_date") or ""), root))
    return publish(target, json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")


def read_latest(session_date: str, *, root: Any = None) -> dict[str, Any]:
    """The newest verified drafts file for ``session_date``. Never raises.

    ``{}`` when there is none - a session nobody has drafted yet and a session
    whose file will not open are both "nothing verified", and neither is an
    error a nightly job should stop for.
    """
    session = str(session_date or "")[:10]
    if not session:
        return {}
    try:
        folder = _pack_dir(root, create=False)
        candidates = sorted(folder.glob(f"{FILE_PREFIX}-{session}*.json"))
    except (OSError, ValueError):
        return {}

    def _index(path: Path) -> int:
        tail = path.stem[len(f"{FILE_PREFIX}-{session}") :].lstrip(".")
        try:
            return int(tail)
        except ValueError:
            return 0

    newest: dict[str, Any] = {}
    best = -1
    for path in candidates:
        order = _index(path)
        if order < best:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, Mapping):
            newest, best = dict(payload), order
    return newest


def draft_for(trade_id: str, session_date: str, *, root: Any = None) -> dict[str, Any]:
    """The newest draft for ONE trade's exit in one session, or ``{}``."""
    wanted = str(trade_id or "")
    found: dict[str, Any] = {}
    for draft in (read_latest(session_date, root=root) or {}).get("drafts") or ():
        if isinstance(draft, Mapping) and str(draft.get("trade_id") or "") == wanted:
            found = dict(draft)
    return found


def _draft_row(
    note: Mapping[str, Any], fields: Mapping[str, Any], *, model: str, moment: datetime
) -> dict[str, Any]:
    return {
        "trade_id": str(note.get("trade_id") or ""),
        "note_id": str(note.get("note_id") or ""),
        "exit_session": str(note.get("exit_session") or ""),
        "symbol": str(note.get("symbol") or ""),
        # PROVISIONAL, in the journal's OWN word for "a machine wrote this".
        # Nothing here is the trader's until they press Confirm or Correct.
        "status": journal_store.TAG_STATUS_PROVISIONAL,
        "fields": dict(fields),
        "model": str(model or ""),
        "prompt_version": PROMPT_VERSION,
        "drafted_at": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# the night
# ---------------------------------------------------------------------------
def run_exit_note_fields(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Any = None,
    store: Any = None,
    post: Any = None,
    request: Any = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """One night's exit drafts. Never raises: a crash here is a lost night.

    One local MEDIUM call per waiting note, oldest first, at most
    :data:`EXIT_NOTES_PER_NIGHT` of them, with the window re-asked before every
    call. Nothing is published unless at least one reply verified whole, and a
    failure of any kind leaves the last verified file exactly as it was.
    """
    session = str(session_date or "")[:10]
    try:
        date.fromisoformat(session)
    except ValueError:
        return {
            "status": ledger.STATUS_FAILED,
            "model": "",
            "reason": (
                f"refusing to draft exit fields for session_date {session_date!r}: a "
                "drafts file must be keyed to a readable date"
            ),
            "outputs": [],
        }

    if store is None:
        try:
            store = _default_store()
        except Exception as exc:  # noqa: BLE001 - an unreachable journal is a REASON
            return {
                "status": ledger.STATUS_FAILED,
                "model": "",
                "reason": f"the trade journal could not be opened: {exc}",
                "outputs": [],
            }

    try:
        vocabularies = _vocabulary_book()
    except Exception as exc:  # noqa: BLE001 - a packaging defect, said out loud
        return {
            "status": ledger.STATUS_FAILED,
            "model": "",
            "reason": f"an exit vocabulary could not be loaded: {exc}",
            "outputs": [],
        }

    waiting = notes_waiting(store, session, root=root)
    if not waiting:
        # NOTHING WAITING -> NO MODEL LOAD. This is the state the desk is in on
        # the first night after the merge and on every night the trader writes
        # no exit note: `EXIT_NOTE_RAW` holds zero rows today.
        return {
            "status": ledger.STATUS_SKIPPED,
            "model": "",
            "reason": f"no exit note is waiting for {session}; no model was loaded",
            "outputs": [],
            "extra": {"notes": 0, "drafted": 0},
        }

    import ai_summary

    if request is None:
        if not ai_summary.local_provider_enabled():
            return {
                "status": ledger.STATUS_SKIPPED,
                "model": "",
                "reason": (
                    "local AI is not configured; no model was asked and the last "
                    "verified drafts file was kept"
                ),
                "outputs": [],
            }
        request = ai_summary.request_ai_summary
    try:
        model = ai_summary.local_model("medium")
    except Exception:  # noqa: BLE001 - an injected request needs no configured model
        model = ""

    queue = waiting[:EXIT_NOTES_PER_NIGHT]
    left_for_tomorrow = len(waiting) - len(queue)
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()

    extra: dict[str, Any] = {}
    if post is not None:
        extra["post"] = post

    drafted: list[dict[str, Any]] = []
    rejected = 0
    failed = 0
    answered_by = ""
    stopped = ""
    for note in queue:
        # THE WINDOW IS ASKED BEFORE EVERY CALL. The slot's reserve buys the
        # first one; a sweep that kept going on that single answer could be
        # loading a model past the window's close, which is the night-only rule
        # broken from inside. What it does not reach is KEPT, never dropped.
        allowed, why = window.launch_allowed(moment, reserve_minutes=CALL_MINUTES)
        if not allowed:
            stopped = str(why or "the night window closed")
            break
        evidence = build_evidence(note, vocabularies=vocabularies)
        try:
            result = request(
                provider="local",
                model=model,
                api_key="",
                evidence=evidence,
                timeout_seconds=TIMEOUT_SECONDS,
                schema=REPLY_JSON_SCHEMA,
                schema_name=SCHEMA_NAME,
                prompt_version=PROMPT_VERSION,
                **extra,
            )
        except Exception:  # noqa: BLE001 - the prior verified file is the fallback
            _log.debug("exit_note_fields: the local model did not answer.", exc_info=True)
            failed += 1
            continue
        answered_by = (
            str(result.get("model") or "") if isinstance(result, Mapping) else ""
        ) or answered_by
        try:
            fields = verify_reply(
                result.get("summary") if isinstance(result, Mapping) else None, note
            )
        except ReplyRejected as exc:
            _log.debug("exit_note_fields: a reply was rejected whole: %s", exc)
            rejected += 1
            continue
        drafted.append(_draft_row(note, fields, model=answered_by or model, moment=moment))

    if not drafted:
        # Nothing verified, so nothing is published and the prior file stands
        # byte-identical. This is an ATTEMPT, so the runner's cap bites and the
        # next firing may try again.
        said = (
            f"{rejected} reply/replies rejected whole and {failed} unanswered over "
            f"{len(queue)} note(s) on {session}; nothing was published"
        )
        if stopped:
            said = f"{said}; the night window stopped the run: {stopped}"
        return {
            "status": ledger.STATUS_DEGRADED,
            "model": answered_by,
            "reason": said,
            "outputs": [],
            "extra": {"notes": len(waiting), "drafted": 0, "rejected": rejected},
        }

    # Every draft the session has, in one file: a reader must be able to ask
    # "is this note drafted?" of ONE file, or a second night would re-ask the
    # first night's notes.
    kept = [
        dict(row)
        for row in (read_latest(session, root=root) or {}).get("drafts") or ()
        if isinstance(row, Mapping)
    ]
    known = {str(row.get("note_id") or "") for row in kept}
    kept.extend(row for row in drafted if str(row.get("note_id") or "") not in known)

    payload = {
        "schema": SCHEMA,
        "session_date": session,
        "prompt_version": PROMPT_VERSION,
        "model": answered_by,
        "generated_at": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "notes_offered": len(queue),
        "notes_waiting": int(left_for_tomorrow),
        "rejected": int(rejected),
        "drafts": kept,
    }
    try:
        written = _publish(payload, root)
    except OSError as exc:
        return {
            "status": ledger.STATUS_FAILED,
            "model": answered_by,
            "reason": f"the exit drafts file could not be published: {exc}",
            "outputs": [],
        }
    said = (
        f"{len(drafted)} exit note(s) drafted on {session}, every value grounded in "
        f"the trader's own words; {rejected} rejected whole"
    )
    if left_for_tomorrow:
        said = f"{said}; {left_for_tomorrow} still waiting and kept for tomorrow"
    if stopped:
        said = f"{said}; the night window stopped the run: {stopped}"
    return {
        "status": ledger.STATUS_OK,
        "model": answered_by,
        "reason": said,
        "outputs": [str(written)],
        "extra": {
            "notes": len(waiting),
            "drafted": len(drafted),
            "rejected": rejected,
            "waiting": int(left_for_tomorrow),
        },
    }


__all__ = [
    "CALL_MINUTES",
    "EXIT_NOTES_PER_NIGHT",
    "FIELDS",
    "MAX_FELT",
    "MAX_WATCHING",
    "PROMPT_VERSION",
    "REPLY_JSON_SCHEMA",
    "REPLY_KEYS",
    "SCHEMA",
    "SCHEMA_NAME",
    "ReplyRejected",
    "build_evidence",
    "draft_for",
    "drafts_path",
    "felt_codes",
    "notes_waiting",
    "read_latest",
    "run_exit_note_fields",
    "verify_reply",
    "why_codes",
]
