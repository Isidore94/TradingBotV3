"""Append-only store for the trader's Chart Review decisions (schema v1).

This is the decision stream: what the trader passed on and why, what they
claimed as a setup, where they would have put a stop. Outcomes are already
tracked elsewhere; this supplies the middle term - the judgement - that no
other artifact in the program records.

HARD BOUNDARY (plan.md sec 5). Everything written here is analysis-only
evidence. Nothing in the running system reads this file to mute, suppress,
score, gate, rank, or alert, and no such consumer ships in this packet. The
one forward-tracking hook is :mod:`ui.annotations.veto_cohort`, which is
capture-side: it lets forward returns accrue against a veto so the veto can be
graded later. It changes no decision the desk makes today.

Storage rules:

* **Append-only.** Every write opens the file in append mode. Nothing in this
  module truncates, rewrites, reorders, or deletes a row - a mistaken capture
  is corrected by a later row, never by editing an earlier one.
* **Atomic per row.** One row is one line, written inside the machine-local
  writer lock and bounded to :data:`MAX_ROW_BYTES` so a row can never be
  interleaved with another process's row or torn across a flush.
* **One writer.** The desk GUI owns this file. Nothing else appends to it.
* **Extensible, never renamed.** Later schema versions add fields. A field
  that exists at v1 keeps its name and meaning forever, because rows already
  written carry it.

Import-light (no Qt, no pandas): the capture rail calls this on every click.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path
from project_paths import TRADER_ANNOTATIONS_FILE
from ui.annotations.vocabulary import VetoVocabulary, load_veto_vocabulary

SCHEMA_VERSION = 1
ANNOTATION_SOURCE = "chart_review"

EVENT_VETO = "veto"
EVENT_LIKE_CLAIM = "like_claim"
EVENT_HYPO_STOP = "hypo_stop"
EVENT_NOTE = "note"
EVENT_TYPES = (EVENT_VETO, EVENT_LIKE_CLAIM, EVENT_HYPO_STOP, EVENT_NOTE)

#: Notes are a capture surface, not a journal - the journal already exists.
#: The cap is what keeps a row inside :data:`MAX_ROW_BYTES`, which is what
#: makes the append atomic.
MAX_NOTE_CHARS = 2000
#: A single write below the pipe/file buffer size is not split by the OS. Rows
#: are ~300 bytes; the cap only ever trips on a pathological note.
MAX_ROW_BYTES = 4096

_SIDES = ("LONG", "SHORT")


class AnnotationError(ValueError):
    """A caller built an annotation that must not be written.

    Distinct from an I/O failure: this means the row itself is wrong (unknown
    event type, reason code outside the vocabulary, a required note missing),
    which is a programming or validation error the capture rail must surface
    rather than write.
    """


def _session_date_text(session_date: Any = None) -> str:
    if isinstance(session_date, datetime):
        return session_date.date().isoformat()
    if isinstance(session_date, date):
        return session_date.isoformat()
    text = str(session_date or "").strip()
    if text:
        return text[:10]
    try:
        from market_session import get_market_session_window

        return get_market_session_window().market_date.isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def _created_at_text(now: datetime | None = None) -> str:
    """An explicitly zoned timestamp.

    plan.md sec 5: timestamps carry explicit timezones. A naive local stamp is
    unreadable a year later from a different machine, so a naive ``now`` is
    given this machine's offset rather than written bare.
    """
    moment = now or datetime.now().astimezone()
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.isoformat(timespec="microseconds")


def _clean_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def _clean_side(side: Any) -> str:
    text = str(side or "").strip().upper()
    if text.startswith("SHORT"):
        return "SHORT"
    if text.startswith("LONG"):
        return "LONG"
    return ""


def _clean_note(note: Any) -> str:
    text = str(note or "").strip()
    if len(text) > MAX_NOTE_CHARS:
        raise AnnotationError(
            f"note is {len(text)} characters; the cap is {MAX_NOTE_CHARS}"
        )
    return text


def _clean_price(value: Any, *, field: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise AnnotationError(f"{field} is not a number: {value!r}") from exc
    if price != price:  # NaN
        raise AnnotationError(f"{field} is NaN")
    if price <= 0:
        raise AnnotationError(f"{field} must be positive, got {price}")
    return price


def build_annotation(
    event_type: str,
    *,
    symbol: Any,
    session_date: Any = None,
    created_at: datetime | None = None,
    reason_code: str = "",
    vocabulary: VetoVocabulary | None = None,
    claimed_setup_id: str = "",
    stop_price: Any = None,
    side: Any = "",
    last_price: Any = None,
    ref_level_id: str = "",
    ref_level_family: str = "",
    note: Any = "",
    timeframe: str = "",
    event_id: str = "",
) -> dict[str, Any]:
    """Validate and assemble one schema-v1 row. Raises AnnotationError.

    Split from :func:`record_annotation` so the capture rail can validate a
    row - and refuse to arm its button - without touching the filesystem.
    """
    kind = str(event_type or "").strip().lower()
    if kind not in EVENT_TYPES:
        raise AnnotationError(f"unknown event_type {event_type!r}; expected one of {EVENT_TYPES}")
    sym = _clean_symbol(symbol)
    if not sym:
        raise AnnotationError("symbol is required")

    note_text = _clean_note(note)
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": str(event_id or "").strip() or uuid.uuid4().hex,
        "event_type": kind,
        "symbol": sym,
        "session_date": _session_date_text(session_date),
        "created_at": _created_at_text(created_at),
        "source": ANNOTATION_SOURCE,
    }

    if kind == EVENT_VETO:
        vocab = vocabulary if vocabulary is not None else load_veto_vocabulary()
        code = str(reason_code or "").strip().lower()
        reason = vocab.reason(code)
        if reason is None:
            raise AnnotationError(
                f"reason_code {reason_code!r} is not in veto vocabulary "
                f"v{vocab.vocab_version} ({list(vocab.codes)})"
            )
        if not reason.accepts(note_text):
            raise AnnotationError(f"reason {code!r} requires a note")
        row["reason_code"] = reason.code
        row["vocab_version"] = vocab.vocab_version

    if kind == EVENT_LIKE_CLAIM:
        claim = str(claimed_setup_id or "").strip().lower()
        if not claim:
            raise AnnotationError("like_claim requires a claimed_setup_id")
        row["claimed_setup_id"] = claim
    elif claimed_setup_id:
        # A veto or a hypothetical stop may also name the setup it is about.
        row["claimed_setup_id"] = str(claimed_setup_id).strip().lower()

    if kind == EVENT_HYPO_STOP:
        stop = _clean_price(stop_price, field="stop_price")
        if stop is None:
            raise AnnotationError("hypo_stop requires a stop_price")
        resolved_side = _clean_side(side)
        if resolved_side not in _SIDES:
            raise AnnotationError(
                f"hypo_stop requires side LONG or SHORT, got {side!r}"
            )
        row["stop_price"] = stop
        row["side"] = resolved_side
    else:
        resolved_side = _clean_side(side)
        if resolved_side:
            row["side"] = resolved_side

    if kind == EVENT_NOTE and not note_text:
        raise AnnotationError("note events require a note")

    last = _clean_price(last_price, field="last_price")
    if last is not None:
        row["last_price"] = last
    if ref_level_id:
        row["ref_level_id"] = str(ref_level_id).strip()
    if ref_level_family:
        row["ref_level_family"] = str(ref_level_family).strip()
    if note_text:
        row["note"] = note_text
    if timeframe:
        row["timeframe"] = str(timeframe).strip().upper()
    return row


def append_annotation_row(
    row: dict[str, Any],
    *,
    path: Path = TRADER_ANNOTATIONS_FILE,
) -> bool:
    """Append one prepared row. True when it reached disk, False otherwise.

    False is a real outcome the caller must show the trader - a capture that
    silently vanished is worse than one that visibly failed - so the transient
    cases (a cloud-synced folder briefly locking the file, the lock timing
    out) are reported rather than raised or swallowed.
    """
    line = json.dumps(row, sort_keys=True, default=str) + "\n"
    encoded = line.encode("utf-8")
    if len(encoded) > MAX_ROW_BYTES:
        raise AnnotationError(
            f"row is {len(encoded)} bytes; the atomic-append cap is {MAX_ROW_BYTES}"
        )
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with local_writer_lock(lock_key_for_path(target), timeout_seconds=1.0):
            # Append mode only: this module never truncates or rewrites.
            with target.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
    except (OSError, LocalLockUnavailable):
        return False
    return True


def record_annotation(
    event_type: str,
    *,
    path: Path = TRADER_ANNOTATIONS_FILE,
    **fields: Any,
) -> dict[str, Any] | None:
    """Build, validate and append one annotation.

    Returns the written row, or ``None`` when the append failed. Raises
    :class:`AnnotationError` when the row itself is invalid.
    """
    row = build_annotation(event_type, **fields)
    return row if append_annotation_row(row, path=path) else None


def load_annotations(
    path: Path = TRADER_ANNOTATIONS_FILE,
    *,
    session_date: Any = None,
    symbol: Any = None,
    event_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Rows in file order (oldest first). Unreadable lines are skipped.

    A corrupt line is skipped rather than fatal: one torn row must never make
    the rest of the trader's decision history unreadable.
    """
    target = Path(path)
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    wanted_date = _session_date_text(session_date) if session_date is not None else None
    wanted_symbol = _clean_symbol(symbol) if symbol is not None else None
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if wanted_date is not None and str(row.get("session_date") or "") != wanted_date:
            continue
        if wanted_symbol is not None and _clean_symbol(row.get("symbol")) != wanted_symbol:
            continue
        if event_types is not None and str(row.get("event_type") or "") not in event_types:
            continue
        rows.append(row)
    return rows
