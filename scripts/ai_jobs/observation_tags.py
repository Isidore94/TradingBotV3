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
# Stored rows remain v1 for every downstream reader.  v2 changes only the
# provider transport: code owns every offset and exact quote.
PROMPT_VERSION = "observation_tags_v2"
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

#: How many tags one night may return. A bound, not a target - and it is
#: re-checked in :func:`verify_reply`, because the copy of it in
#: :data:`TAGS_JSON_SCHEMA` is a GRAMMAR HINT to the provider and nothing else.
#: `ai_summary._request_local_summary` already ships a documented fallback for a
#: backend that refuses to compile the grammar, and
#: `ai_summary.validate_structured_output` walks the TOP level only, so an array
#: of objects arrives untouched. Reviewer, 2026-09-20: a 10,000-row reply was
#: published `ok`. A bound the verifier does not re-check is not a bound.
MAX_TAGS = 60
MAX_FRAGMENT_LENGTH = 400
MAX_FRAGMENTS = 60

#: The only keys a reply may carry, at each of its two levels. They mirror
#: :data:`TAGS_JSON_SCHEMA`'s `additionalProperties: false`, which for the same
#: reason cannot be relied on.
REPLY_KEYS = frozenset({"tags"})
TAG_KEYS = frozenset({"note_id", "code", "span", "quote"})
FRAGMENT_TAG_KEYS = frozenset({"fragment_id", "code"})

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
                "required": ["fragment_id", "code"],
                "properties": {
                    "fragment_id": {"type": "string", "maxLength": 96},
                    "code": {"type": "string", "maxLength": 64},
                },
            },
        }
    },
}

INSTRUCTIONS = (
    "Label what the trader's own words DO. You are given short notes the trader "
    "wrote as exact fragments and a closed list of codes. For each code, return "
    "only the fragment_id that made you choose it and the code. Do not count "
    "characters, choose an offset, or copy a quote: code owns those links. Use "
    "only codes from the list and only fragment_id values from the list. "
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

    Each note carries the entry's own ``written_after_the_session``, which the
    writer COMPUTES and never backdates. It is a LABEL and it stays out of the
    payload: a `because` written in the evening can say "in hindsight the 50 day
    failed and I lost on this call" (reviewer, 2026-09-20), and a later reader
    must be able to partition those words - but telling the tagger a note was
    written after the close would be telling it something about the outcome,
    which is the one thing this slot may never do. The note is never dropped
    and nothing it produces is re-weighted.
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
                    "written_after_the_session": bool(
                        entry.get("written_after_the_session")
                    ),
                }
            )
    return out


def _fragment_id(note_id: str, start: int, end: int) -> str:
    """A stable id makes equal words at different offsets different evidence."""
    return f"frag-{note_id}-{start}-{end}"


def _fragment_spans(text: str) -> list[tuple[int, int]]:
    """Cover every character with sentence-like chunks under the quote limit."""
    spans: list[tuple[int, int]] = []
    # A terminator ends a sentence only before whitespace or EOF.  That keeps
    # ``!!!hello.`` together instead of silently dropping its punctuation.
    for match in re.finditer(r".+?(?:[.!?]+(?=\s|$)|$)", text, flags=re.DOTALL):
        start, end = match.span()
        while end - start > MAX_FRAGMENT_LENGTH:
            # The range stops BEFORE the over-limit position: including a
            # whitespace at offset 400 then retaining it made a 401-char quote.
            split = text.rfind(" ", start + 1, start + MAX_FRAGMENT_LENGTH)
            if split <= start:
                split = start + MAX_FRAGMENT_LENGTH
            else:
                split += 1  # keep the separator in the preceding exact quote
            spans.append((start, split))
            start = split
        if start < end:
            spans.append((start, end))
    return spans


def fragments_for(
    notes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic exact fragments and name every cap omission.

    The cap limits a request, never a note invisibly.  ``fragments_omitted``
    carries a count plus the first bounded set of ids, so a later request can
    explain what was not offered instead of silently changing the evidence.
    """
    all_fragments: list[dict[str, Any]] = []
    for note in notes or ():
        note_id = str(note.get("note_id") or "")
        text = str(note.get("text") or "")
        if not note_id or not text:
            continue
        for start, end in _fragment_spans(text):
            all_fragments.append(
                {
                    "fragment_id": _fragment_id(note_id, start, end),
                    "note_id": note_id,
                    "entry_id": str(note.get("entry_id") or ""),
                    "field": str(note.get("field") or ""),
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            )
    offered = all_fragments[:MAX_FRAGMENTS]
    omitted = all_fragments[MAX_FRAGMENTS:]
    return offered, {
        "limit": MAX_FRAGMENTS,
        "count": len(omitted),
        "fragment_ids": [row["fragment_id"] for row in omitted[:MAX_FRAGMENTS]],
        "more": max(0, len(omitted) - MAX_FRAGMENTS),
    }


def build_evidence(
    notes: Sequence[Mapping[str, Any]], *, vocabulary: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """What the model is handed. BUILT, never filtered down from an entry.

    There is no verdict, no grade, no price and no bar anywhere in it, and that
    is structural: this function knows about two strings and a picklist and has
    no way to reach anything else.
    """
    book = dict(vocabulary or load_vocabulary())
    offered, omitted = fragments_for(notes)
    book_payload = {
        "vocabulary_id": str(book.get("vocabulary_id") or VOCABULARY_FAMILY),
        "vocab_version": int(book.get("vocab_version") or 0),
        "entries": [dict(entry) for entry in book.get("entries") or ()],
    }
    canonical = json.dumps(
        {"fragments": offered, "fragments_omitted": omitted, "vocabulary": book_payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "package_id": f"observation-tags:{digest[:16]}",
        "evidence_hash": digest,
        "instructions": INSTRUCTIONS,
        "allowed_fragment_ids": [row["fragment_id"] for row in offered],
        "vocabulary": book_payload,
        "fragments": offered,
        "fragments_omitted": omitted,
    }


# ---------------------------------------------------------------------------
# the grounding check - all of it, or none of it
# ---------------------------------------------------------------------------
class ReplyRejected(ValueError):
    """The model's answer was not believed, so nothing was published."""


def verify_fragment_reply(
    reply: Any,
    notes: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, Any],
    fragments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve v2 ids in code, then use the unchanged v1 grounding checker."""
    if not isinstance(reply, Mapping):
        raise ReplyRejected("the model returned no tags object")
    stray = sorted(str(key) for key in reply if str(key) not in REPLY_KEYS)
    if stray:
        raise ReplyRejected(
            f"the reply carries key(s) the contract forbids: {', '.join(stray)}"
        )
    rows = reply.get("tags")
    if not isinstance(rows, (list, tuple)):
        raise ReplyRejected("`tags` is not a list")
    if len(rows) > MAX_TAGS:
        raise ReplyRejected(
            f"the reply carries {len(rows)} tags, over the cap of {MAX_TAGS}"
        )
    by_id = {str(row.get("fragment_id") or ""): row for row in fragments or ()}
    codes = set(vocabulary.get("codes") or ())
    seen: set[tuple[str, str]] = set()
    materialized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReplyRejected("a tag row is not an object")
        extra = sorted(str(key) for key in row if str(key) not in FRAGMENT_TAG_KEYS)
        if extra or set(row) != FRAGMENT_TAG_KEYS:
            raise ReplyRejected("a v2 tag row must contain only fragment_id and code")
        fragment_id = row.get("fragment_id")
        code = row.get("code")
        if not isinstance(fragment_id, str) or not isinstance(code, str):
            raise ReplyRejected("a v2 tag's fragment_id and code must be strings")
        fragment = by_id.get(fragment_id)
        if fragment is None:
            raise ReplyRejected(f"a tag names a fragment that was not offered: {fragment_id!r}")
        if code not in codes:
            raise ReplyRejected(
                f"a tag uses {code!r}, which is outside the closed vocabulary"
            )
        identity = (fragment_id, code)
        if identity in seen:
            raise ReplyRejected(f"the reply repeats {code!r} on {fragment_id!r}")
        seen.add(identity)
        materialized.append(
            {
                "note_id": str(fragment["note_id"]),
                "code": code,
                "span": [int(fragment["start"]), int(fragment["end"])],
                "quote": str(fragment["text"]),
            }
        )
    # Do not fork its old verification rules.  This preserves the v1 stored
    # shape, exact quote check and all future reader expectations.
    return verify_reply({"tags": materialized}, notes, vocabulary)


def verify_reply(
    reply: Any, notes: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Every tag, or :class:`ReplyRejected`. One bad row rejects the whole reply.

    It re-checks the SHAPE of the answer as well as the grounding of each row,
    because the contract in :data:`TAGS_JSON_SCHEMA` is a hint the provider may
    or may not honour (see :data:`MAX_TAGS`). Four shape rules, and each one
    throws the whole reply away:

    * more than :data:`MAX_TAGS` rows;
    * a key outside :data:`REPLY_KEYS` at the top or :data:`TAG_KEYS` on a row;
    * a byte-identical duplicate row - same note, same code, same span. **A
      model that repeats itself has not been verified** (lead decision,
      2026-09-20): de-duping silently would store a file that does not say what
      the model returned, and `codes_by_entry` collapses codes per entry, so no
      number would move and nobody would ever notice.

    Two rows on ONE note with OVERLAPPING spans and DIFFERENT codes are kept:
    one sentence can cite a level and be hedged, and that is the finding.
    """
    if not isinstance(reply, Mapping):
        raise ReplyRejected("the model returned no tags object")
    stray = sorted(str(key) for key in reply if str(key) not in REPLY_KEYS)
    if stray:
        raise ReplyRejected(
            f"the reply carries key(s) the contract forbids: {', '.join(stray)}"
        )
    by_id = {str(note.get("note_id") or ""): note for note in notes or ()}
    codes = set(vocabulary.get("codes") or ())
    rows = reply.get("tags")
    if rows is None:
        raise ReplyRejected("the reply carries no `tags` list")
    if not isinstance(rows, (list, tuple)):
        raise ReplyRejected("`tags` is not a list")
    if len(rows) > MAX_TAGS:
        raise ReplyRejected(
            f"the reply carries {len(rows)} tags, over the cap of {MAX_TAGS}"
        )
    seen: set[tuple[str, str, int, int]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReplyRejected("a tag row is not an object")
        extra = sorted(str(key) for key in row if str(key) not in TAG_KEYS)
        if extra:
            raise ReplyRejected(
                f"a tag row carries key(s) the contract forbids: {', '.join(extra)}"
            )
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
        identity = (note_id, code, start, end)
        if identity in seen:
            raise ReplyRejected(
                f"the reply repeats one tag: {code!r} on {note_id!r} at "
                f"{[start, end]}"
            )
        seen.add(identity)
        out.append(
            {
                "note_id": note_id,
                "entry_id": str(note.get("entry_id") or ""),
                "field": str(note.get("field") or ""),
                "code": code,
                "span": [start, end],
                "quote": quote,
                # Advisory 1: a LABEL, never a filter. The trader's own words
                # are the artifact under study and a hindsight note is tagged
                # like any other; this is what lets a later reader partition.
                "written_after_the_session": bool(
                    note.get("written_after_the_session")
                ),
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


def tagged_entries(session_date: str, *, root: Any = None) -> dict[str, dict[str, Any]]:
    """`{entry_id: {"codes": [...], "written_after_the_session": bool}}`.

    ONE read of the session's tags file answering both questions the contrast
    asks of it: which codes a note carried, and whether the note was written
    after the bell. A code appears once per entry however many spans carried it;
    the contrast is about whether a note cited a thing, not how often.
    """
    stored = read_latest(session_date, root=root)
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(stored, Mapping):
        return out
    for tag in stored.get("tags") or ():
        if not isinstance(tag, Mapping):
            continue
        entry_id = str(tag.get("entry_id") or "")
        code = str(tag.get("code") or "").strip()
        if not entry_id or not code:
            continue
        row = out.setdefault(
            entry_id, {"codes": [], "written_after_the_session": False}
        )
        if code not in row["codes"]:
            row["codes"].append(code)
        if tag.get("written_after_the_session"):
            row["written_after_the_session"] = True
    return out


def codes_by_entry(session_date: str, *, root: Any = None) -> dict[str, list[str]]:
    """`{entry_id: [code, ...]}` for one session - the codes half of the above."""
    return {
        entry_id: list(row["codes"])
        for entry_id, row in tagged_entries(session_date, root=root).items()
    }


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

    **The WINDOW is asked of the ledger, not applied after it.**
    `EvidenceLedger.read` filters by `session_date` WHILE STREAMING, and the
    packet's rule is "never stream the whole journal unbounded". Measured by the
    reviewer on a copy of the live stream (2026-09-20): 84 rows unwindowed
    against 7 for one session, the same answer either way. It is safe because a
    CORRECTION carries the ORIGINAL `session_date` (`market_journal.build_entry`:
    *"session_date is what the entry is ABOUT"*), so both halves of a supersede
    pair stay inside a one-session window and `resolve_entries` can still hide
    the older one.

    The session is then selected the ONE way this desk selects it
    (`market_journal.session_of_entry`) - a row whose `session_date` and
    `session_of_entry` disagree is not silently kept - and a machine row never
    becomes a note.
    """
    try:
        import market_journal
        from evidence_ledger import EvidenceLedger

        ledger = EvidenceLedger(
            stream=market_journal.STREAM,
            schema=market_journal.SCHEMA_MARKET_JOURNAL_ENTRY,
        )
        window = str(session or "")[:10]
        result = ledger.read(start=window, end=window)
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
        if not isinstance(reply, Mapping):
            raise ReplyRejected("the model returned no tags object")
        rows = reply.get("tags")
        has_v2 = isinstance(rows, (list, tuple)) and any(
            isinstance(row, Mapping) and "fragment_id" in row for row in rows
        )
        has_legacy = isinstance(rows, (list, tuple)) and any(
            isinstance(row, Mapping)
            and any(key in row for key in ("note_id", "span", "quote"))
            for row in rows
        )
        if has_v2 and has_legacy:
            raise ReplyRejected("the reply mixed legacy offsets with v2 fragments")
        if has_v2:
            verified = verify_fragment_reply(
                reply, notes, vocabulary, evidence["fragments"]
            )
        else:
            # Compatibility is intentionally strict: an old provider fixture
            # still owns its offsets and one bad span rejects the whole reply.
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
        # Advisory 1, as a COUNT in the header: how many of the entries that got
        # a tag were written after the session closed. Present and zero, never
        # absent - a reader that has to tell "none" from "this build did not
        # measure it" is reading two different absences as one.
        "entries_written_after": len(
            {
                str(tag["entry_id"])
                for tag in verified
                if tag.get("written_after_the_session")
            }
        ),
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
    "REPLY_KEYS",
    "TAG_KEYS",
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
    "tagged_entries",
    "tags_path",
    "verify_reply",
]
