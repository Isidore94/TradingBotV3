r"""`observation_tags` - the model labels the WORDS, grounded, and never sees a
verdict (TJ-16 item 4).

`plan.md` §12.4 TJ-16 item 4: *"each `observation` and `because` gets codes from
a closed, versioned vocabulary ..., each code with the exact source span that
must reproduce it ... or it is rejected. The codes become context fields in item
3 on the NEXT run. **The model never sees a verdict while tagging**, so a tag
cannot be derived from the outcome."*

Decision 0021 answer 29: tendencies are found by MATH; the model only tags words
with a closed span-grounded vocabulary. It counts nothing, ranks nothing and
judges nothing.

Five rules hold it there.

1. **The payload is the NOTES and the VOCABULARY, and nothing else.** It is
   BUILT from the two texts - not filtered down from the entry - so there is no
   key to forget to delete. A tag derived from the outcome is not a label of the
   words, it is a rationalisation of the result, and the grade ledger of the
   very same session sits on disk beside this job.
2. **A closed vocabulary.** Every code the model may use is in the payload, and
   a code outside it rejects the reply. The vocabulary has its OWN loader here
   rather than widening `ui.annotations.vocabulary`, whose `_parse` is
   veto-shaped (a unique single-character `hotkey`, a boolean `note_required`)
   and which serves the capture rail on every click (lead decision, 2026-09-20).
   The version is read from the file and compared with its own FILENAME, never
   asserted as a literal.
3. **A span is a QUOTATION.** ``text[start:end]`` must equal ``quote`` exactly.
   One row that does not reproduce rejects the WHOLE reply - not the row - and
   the last verified file stays byte-identical, because a half-accepted answer
   is a file nobody can trust and nobody can tell apart from a whole one. This
   is `market_thesis`' rule and every other grounded slot's.
4. **A failure keeps the last verified file.** Nothing is published unless every
   tag in the reply verified; an unconfigured desk dials nothing at all.
5. **Tonight's tags are tonight's.** This is a STAGE 2 slot and
   :mod:`ai_jobs.prediction_contrast` is a stage 1 one, so the codes reach a
   contrast on the NEXT run and never on the same one.

Nothing it writes reaches a detector, score, alert, watchlist, Focus, the review
queue or `review_policy.json`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_log = logging.getLogger(__name__)

#: The prompt contract's NAME, stamped on every stored file.
PROMPT_VERSION = "observation_tags_v1"
#: The stored file's own schema name.
SCHEMA = "observation_tags_v1"
#: What the response_format contract is called on the wire.
SCHEMA_NAME = "tradingbot_observation_tags"

#: What every tags file is called, before any superseding sibling.
FILE_PREFIX = "observation_tags"

#: The vocabulary family. One family owns its own version series.
VOCABULARY_FAMILY = "observation_tags"

#: The two fields that carry the trader's own words (TJ-14A). `mentor.text` is
#: not a third: a clicks-only row's text is ``""`` and the desk never
#: synthesises a sentence for it.
FIELD_OBSERVATION = "observation"
FIELD_BECAUSE = "because"
FIELDS = (FIELD_OBSERVATION, FIELD_BECAUSE)

#: The house's permanent-identifier rule, restated rather than imported private:
#: a code is a column name, a filename fragment and a context-feature suffix.
_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{2,47}$")

#: How many tags one night may return. A bound, not a target.
MAX_TAGS = 60

#: THE ONLY REPLY ACCEPTED. No `maxLength` of exactly 2,000 anywhere in it -
#: that is the grammar-compile defect behind gate #144.
TAGS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tags"],
    "properties": {
        "tags": {
            "type": "array",
            "maxItems": MAX_TAGS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["note_id", "code", "span", "quote"],
                "properties": {
                    "note_id": {"type": "string", "maxLength": 64},
                    "code": {"type": "string", "maxLength": 64},
                    "span": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "integer"},
                    },
                    "quote": {"type": "string", "maxLength": 400},
                },
            },
        }
    },
}

INSTRUCTIONS = (
    "Label what the trader's own words DO. You are given short notes the trader "
    "wrote and a closed list of codes. For each note, return every code that "
    "applies, and for each code the exact character span of the note that made "
    "you choose it, plus that substring copied out as `quote`. `quote` must be "
    "exactly note.text[start:end] - if it is not, the whole answer is thrown "
    "away. Use only codes from the list and only note_id values from the list. "
    "You are NOT being asked whether the trader was right or wrong, and you are "
    "not told: no outcome, no market data and no later information is in this "
    "package. Do not guess at one. A note that carries no code is fine - return "
    "nothing for it rather than stretching a code to fit. An EMPTY answer is "
    "valid."
)


class VocabularyError(RuntimeError):
    """The tag vocabulary is missing, unreadable, or violates its contract."""


# ---------------------------------------------------------------------------
# the vocabulary - its OWN loader, deliberately
# ---------------------------------------------------------------------------
def _vocabulary_dir(directory: Any = None) -> Path:
    if directory is not None:
        return Path(directory)
    # ONE answer to "where do vocabularies live", and it is already resolved
    # file-relative so a frozen run finds it under `sys._MEIPASS`.
    from ui.annotations.vocabulary import VOCABULARY_DIR

    return Path(VOCABULARY_DIR)


def load_vocabulary(*, directory: Any = None) -> dict[str, Any]:
    """The newest `observation_tags_v*.json`, validated. Fail-closed.

    A missing or malformed vocabulary is a packaging defect, not a runtime
    condition to paper over: the alternative is a nightly job that writes codes
    no later reader will recognise. The declared `vocab_version` must match the
    FILENAME, so a v2 ships beside this file and rows stamped with v1 stay
    interpretable against exactly the list that produced them.
    """
    folder = _vocabulary_dir(directory)
    pattern = re.compile(rf"^{re.escape(VOCABULARY_FAMILY)}_v(\d+)\.json$")
    found: list[tuple[int, Path]] = []
    try:
        for path in folder.glob(f"{VOCABULARY_FAMILY}_v*.json"):
            match = pattern.fullmatch(path.name)
            if match:
                found.append((int(match.group(1)), path))
    except OSError as exc:
        raise VocabularyError(f"the tag vocabulary folder is unreadable: {exc}") from exc
    if not found:
        raise VocabularyError(
            f"no {VOCABULARY_FAMILY}_v*.json vocabulary under {folder}"
        )
    version, path = max(found, key=lambda pair: pair[0])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VocabularyError(f"the tag vocabulary at {path} is unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise VocabularyError(f"the tag vocabulary at {path} is not a JSON object")
    declared = payload.get("vocab_version")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared != version:
        raise VocabularyError(
            f"{path.name} declares vocab_version {declared!r}, which its own "
            f"filename says is {version}"
        )
    if str(payload.get("vocabulary_id") or "") != VOCABULARY_FAMILY:
        raise VocabularyError(
            f"{path.name} declares vocabulary_id "
            f"{payload.get('vocabulary_id')!r}, not {VOCABULARY_FAMILY!r}"
        )
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in payload.get("tags") or ():
        if not isinstance(row, Mapping):
            raise VocabularyError(f"{path.name} holds a tag that is not an object")
        code = str(row.get("code") or "")
        if not _CODE_RE.fullmatch(code):
            raise VocabularyError(
                f"{path.name}: code {code!r} must match {_CODE_RE.pattern}"
            )
        if code in seen:
            raise VocabularyError(f"{path.name}: code {code!r} appears twice")
        label = str(row.get("label") or "").strip()
        if not label:
            raise VocabularyError(f"{path.name}: code {code!r} has no label")
        seen.add(code)
        entries.append(
            {"code": code, "label": label, "hint": str(row.get("hint") or "").strip()}
        )
    if not entries:
        raise VocabularyError(f"{path.name} holds no tags")
    return {
        "vocabulary_id": VOCABULARY_FAMILY,
        "vocab_version": int(version),
        "description": str(payload.get("description") or ""),
        "codes": tuple(entry["code"] for entry in entries),
        "entries": tuple(entries),
        "path": str(path),
    }


# ---------------------------------------------------------------------------
# the notes - the trader's own two texts, and nothing else
# ---------------------------------------------------------------------------
def _note_id(entry_id: str, field: str) -> str:
    digest = hashlib.sha1(f"{entry_id}|{field}".encode("utf-8")).hexdigest()[:12]
    return f"nt-{digest}"


def notes_for(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One note per text the trader wrote. TWO fields, never a third.

    `mentor.observation` is What I see and `mentor.prediction.because` is
    Because (TJ-14A). A machine row is never a note - `market_journal.
    is_machine_entry` is the ONE filter (TJ-1) - and an empty text is not a
    note either: a clicks-only row stores ``""`` and the desk does not write a
    sentence for it.
    """
    import market_journal

    out: list[dict[str, Any]] = []
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        if market_journal.is_machine_entry(entry):
            continue
        mentor = entry.get("mentor")
        if not isinstance(mentor, Mapping):
            continue
        entry_id = str(entry.get("entry_id") or "")
        if not entry_id:
            continue
        prediction = mentor.get("prediction")
        because = (
            str(prediction.get("because") or "").strip()
            if isinstance(prediction, Mapping)
            else ""
        )
        texts = {
            FIELD_OBSERVATION: str(mentor.get("observation") or "").strip(),
            FIELD_BECAUSE: because,
        }
        for field in FIELDS:
            text = texts[field]
            if not text:
                continue
            out.append(
                {
                    "note_id": _note_id(entry_id, field),
                    "entry_id": entry_id,
                    "field": field,
                    "text": text,
                }
            )
    return out


def build_evidence(
    notes: Sequence[Mapping[str, Any]], *, vocabulary: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """What the model is handed. BUILT, never filtered down from an entry.

    There is no verdict, no grade, no price and no bar anywhere in it, and that
    is structural: this function knows about two strings and a picklist and has
    no way to reach anything else.
    """
    book = dict(vocabulary or load_vocabulary())
    offered = [
        {
            "note_id": str(note.get("note_id") or ""),
            "entry_id": str(note.get("entry_id") or ""),
            "field": str(note.get("field") or ""),
            "text": str(note.get("text") or ""),
        }
        for note in notes or ()
    ]
    book_payload = {
        "vocabulary_id": str(book.get("vocabulary_id") or VOCABULARY_FAMILY),
        "vocab_version": int(book.get("vocab_version") or 0),
        "entries": [dict(entry) for entry in book.get("entries") or ()],
    }
    canonical = json.dumps(
        {"notes": offered, "vocabulary": book_payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "package_id": f"observation-tags:{digest[:16]}",
        "evidence_hash": digest,
        "instructions": INSTRUCTIONS,
        "allowed_note_ids": [note["note_id"] for note in offered],
        "vocabulary": book_payload,
        "notes": offered,
    }


# ---------------------------------------------------------------------------
# the grounding check - all of it, or none of it
# ---------------------------------------------------------------------------
class ReplyRejected(ValueError):
    """The model's answer was not believed, so nothing was published."""


def verify_reply(
    reply: Any, notes: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Every tag, or :class:`ReplyRejected`. One bad row rejects the whole reply."""
    if not isinstance(reply, Mapping):
        raise ReplyRejected("the model returned no tags object")
    by_id = {str(note.get("note_id") or ""): note for note in notes or ()}
    codes = set(vocabulary.get("codes") or ())
    rows = reply.get("tags")
    if rows is None:
        raise ReplyRejected("the reply carries no `tags` list")
    if not isinstance(rows, (list, tuple)):
        raise ReplyRejected("`tags` is not a list")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReplyRejected("a tag row is not an object")
        note_id = str(row.get("note_id") or "")
        note = by_id.get(note_id)
        if note is None:
            raise ReplyRejected(f"a tag names a note that was not offered: {note_id!r}")
        code = str(row.get("code") or "")
        if code not in codes:
            raise ReplyRejected(
                f"a tag uses {code!r}, which is outside the closed vocabulary"
            )
        span = row.get("span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ReplyRejected(f"a tag on {note_id!r} has no two-number span")
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError) as exc:
            raise ReplyRejected(f"a tag on {note_id!r} has a span that is not numbers") from exc
        text = str(note.get("text") or "")
        if start < 0 or end > len(text) or start >= end:
            raise ReplyRejected(
                f"a tag on {note_id!r} has a span outside the note: {[start, end]}"
            )
        quote = str(row.get("quote") or "")
        if text[start:end] != quote:
            raise ReplyRejected(
                f"a span on {note_id!r} does not reproduce its quote: the note "
                f"reads {text[start:end]!r} where the answer said {quote!r}"
            )
        out.append(
            {
                "note_id": note_id,
                "entry_id": str(note.get("entry_id") or ""),
                "field": str(note.get("field") or ""),
                "code": code,
                "span": [start, end],
                "quote": quote,
            }
        )
    return out


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


def tags_path(session_date: str, root: Any = None) -> Path:
    return _pack_dir(root) / f"{FILE_PREFIX}-{str(session_date)[:10]}.json"


def _publish(payload: Mapping[str, Any], root: Any = None) -> Path:
    from ai_jobs.digest import _publish as publish, superseding_path

    target = superseding_path(tags_path(str(payload.get("session_date") or ""), root))
    return publish(
        target, json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n"
    )


def read_latest(session_date: str, *, root: Any = None) -> dict[str, Any] | None:
    """The newest verified tags file for ``session_date``, or ``None``. Never raises."""
    session = str(session_date or "")[:10]
    if not session:
        return None
    try:
        folder = _pack_dir(root, create=False)
        candidates = sorted(folder.glob(f"{FILE_PREFIX}-{session}*.json"))
    except (OSError, ValueError):
        return None

    def _index(path: Path) -> int:
        tail = path.stem[len(f"{FILE_PREFIX}-{session}") :].lstrip(".")
        try:
            return int(tail)
        except ValueError:
            return 0

    newest: dict[str, Any] | None = None
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


def codes_by_entry(session_date: str, *, root: Any = None) -> dict[str, list[str]]:
    """`{entry_id: [code, ...]}` for one session - what the contrast reads."""
    stored = read_latest(session_date, root=root)
    out: dict[str, list[str]] = {}
    if not isinstance(stored, Mapping):
        return out
    for tag in stored.get("tags") or ():
        if not isinstance(tag, Mapping):
            continue
        entry_id = str(tag.get("entry_id") or "")
        code = str(tag.get("code") or "").strip()
        if not entry_id or not code:
            continue
        codes = out.setdefault(entry_id, [])
        if code not in codes:
            codes.append(code)
    return out


# ---------------------------------------------------------------------------
# the default reader (only used when the caller supplies nothing)
# ---------------------------------------------------------------------------
def _read_entries(session: str) -> tuple[list[dict[str, Any]], str]:
    """The ONE session's Market Journal entries, headless. Never raises.

    Read through `evidence_ledger.EvidenceLedger` rather than
    `shared_journal_service()`, which is a `QObject` and would pull PySide6 into
    the nightly process; `market_story_rollups` set that precedent for a
    deterministic slot reading this same store. The Day Review per-session index
    was checked first (lead decision 2, 2026-09-20) and does not carry the
    journal at all - `day_review_index.INDEXED_SOURCES` is the recap stores - so
    it could not have answered this.

    The session is selected the ONE way this desk selects it
    (`market_journal.session_of_entry`) and a machine row never becomes a note.
    """
    try:
        import market_journal
        from evidence_ledger import EvidenceLedger

        ledger = EvidenceLedger(
            stream=market_journal.STREAM,
            schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
        )
        result = ledger.read()
        rows = [
            row
            for row in market_journal.resolve_entries(result.rows)
            if not market_journal.is_machine_entry(row)
            and market_journal.session_of_entry(row) == str(session or "")[:10]
        ]
        return rows, ""
    except Exception as exc:  # noqa: BLE001 - an unreadable journal is a REASON
        _log.debug("observation_tags: the Market Journal was unreadable.", exc_info=True)
        return [], f"journal unreadable: {type(exc).__name__}: {exc}"


def run_observation_tags(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Any = None,
    entries: Iterable[Mapping[str, Any]] | None = None,
    post: Any = None,
    request: Any = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """The nightly slot. One local MEDIUM call over one session's words.

    Nothing is published unless every tag in the reply reproduced its own span
    and named a code in the closed vocabulary. A failure of any kind leaves the
    last verified file exactly as it was.
    """
    session = str(session_date or "")[:10]
    try:
        date.fromisoformat(session)
    except ValueError:
        return {
            "status": "failed",
            "model": "",
            "reason": (
                f"refusing to tag notes for session_date {session_date!r}: a tags "
                "file must be keyed to a readable date"
            ),
            "outputs": [],
        }

    try:
        vocabulary = load_vocabulary()
    except VocabularyError as exc:
        return {
            "status": "failed",
            "model": "",
            "reason": f"the tag vocabulary could not be loaded: {exc}",
            "outputs": [],
        }

    notes_source = ""
    if entries is None:
        entries, notes_source = _read_entries(session)
    notes = notes_for(entries)
    if not notes:
        reason = f"no observation or because text was written on {session}"
        if notes_source:
            reason = f"{reason}; {notes_source}"
        return {"status": "skipped", "model": "", "reason": reason, "outputs": []}

    import ai_summary

    if request is None:
        if not ai_summary.local_provider_enabled():
            return {
                "status": "skipped",
                "model": "",
                "reason": (
                    "local AI is not configured; no model was asked and the last "
                    "verified tags file was kept"
                ),
                "outputs": [],
            }
        request = ai_summary.request_ai_summary

    evidence = build_evidence(notes, vocabulary=vocabulary)
    extra: dict[str, Any] = {}
    if post is not None:
        extra["post"] = post
    try:
        result = request(
            provider="local",
            model=ai_summary.local_model("medium"),
            api_key="",
            evidence=evidence,
            timeout_seconds=900,
            schema=TAGS_JSON_SCHEMA,
            schema_name=SCHEMA_NAME,
            prompt_version=PROMPT_VERSION,
            **extra,
        )
    except Exception as exc:  # noqa: BLE001 - the prior verified file is the fallback
        return {
            "status": "degraded_no_narrative",
            "model": "",
            "reason": f"the local tagger failed; the prior tags file was kept: {exc}",
            "outputs": [],
        }

    reply = result.get("summary") if isinstance(result, Mapping) else None
    try:
        verified = verify_reply(reply, notes, vocabulary)
    except ReplyRejected as exc:
        return {
            "status": "degraded_no_narrative",
            "model": str(result.get("model") or "") if isinstance(result, Mapping) else "",
            "reason": (
                f"the reply was rejected whole and nothing was published: {exc}"
            ),
            "outputs": [],
        }

    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    payload = {
        "schema": SCHEMA,
        "session_date": session,
        "vocabulary_id": vocabulary["vocabulary_id"],
        "vocab_version": vocabulary["vocab_version"],
        "model": str(result.get("model") or "") if isinstance(result, Mapping) else "",
        "prompt_version": PROMPT_VERSION,
        "generated_at": moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "inputs_hash": evidence["evidence_hash"],
        "notes_offered": len(notes),
        "tags": verified,
    }
    try:
        written = _publish(payload, root)
    except OSError as exc:
        return {
            "status": "failed",
            "model": str(payload["model"]),
            "reason": f"the tags file could not be published: {exc}",
            "outputs": [],
        }
    reason = (
        f"{len(verified)} grounded tag(s) over {len(notes)} note(s) on {session}; "
        "every span reproduced its quote"
    )
    if notes_source:
        reason = f"{reason}; {notes_source}"
    return {
        "status": "ok",
        "model": str(payload["model"]),
        "reason": reason,
        "outputs": [str(written)],
        "extra": {"tags": len(verified), "notes": len(notes)},
    }


__all__ = [
    "FIELDS",
    "MAX_TAGS",
    "PROMPT_VERSION",
    "SCHEMA",
    "SCHEMA_NAME",
    "TAGS_JSON_SCHEMA",
    "VOCABULARY_FAMILY",
    "ReplyRejected",
    "VocabularyError",
    "build_evidence",
    "codes_by_entry",
    "load_vocabulary",
    "notes_for",
    "read_latest",
    "run_observation_tags",
    "tags_path",
    "verify_reply",
]
