"""Append-only store for the trader's Chart Review decisions (schema v1).

This is the decision stream: what the trader passed on and why, what they
claimed as a setup, where they would have put a stop. Outcomes are already
tracked elsewhere; this supplies the middle term - the judgement - that no
other artifact in the program records.

Two kinds of "no" live here and they are not the same answer. A VETO says the
chart in front of the trader is not for today. A PASS (2026-08-31) says the
day trade WAS there and one specific thing stopped them - "I really like this
stock for a daytrade but it has this ONE issue." Separate event type, separate
vocabulary family, and a pass may carry several reasons at once. When the desk
already holds the symbol's M5 bars, a pass also references the chart as it
stood at that moment through :mod:`ui.annotations.pass_bars`.

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
* **One row, one line, one write.** A row is written inside the machine-local
  writer lock as a single bounded (:data:`MAX_ROW_BYTES`) buffered write and
  fsynced before the append reports success, so cooperating writers never
  interleave and "saved" means on-disk, not in a page cache. This is NOT
  all-or-nothing persistence: a crash mid-write can still leave a torn
  half-row at the tail. What the store guarantees instead is CONFINEMENT -
  before appending, a tail that does not end in a newline is healed with one,
  so an earlier torn row can never absorb the next good row; the reader
  skips exactly the torn line and nothing else.
* **One writer.** The desk GUI owns this file. Nothing else appends to it.
* **Extensible, never renamed.** Later schema versions add fields. A field
  that exists at v1 keeps its name and meaning forever, because rows already
  written carry it.

Import-light (no Qt, no pandas): the capture rail calls this on every click.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from local_writer_lock import LocalLockUnavailable, local_writer_lock, lock_key_for_path
from project_paths import TRADER_ANNOTATIONS_FILE
from ui.annotations.vocabulary import (
    VetoVocabulary,
    load_pass_vocabulary,
    load_veto_vocabulary,
)

SCHEMA_VERSION = 1

#: How a LIKE was made (P9, trader 2026-09-02: *"anytime I like and claim a setup
#: or like a day trade setup I just want to let the bot and the future AI know
#: 'something about this was good' and then we can figure out what about it /
#: what's the best entry later"*).
#:
#: `claimed` is the original path - Alt+K, a digit, a why - and is unchanged in
#: every respect. `quick` is one key and nothing else: no claim, no why.
#:
#: SCHEMA_VERSION STAYS 1. This is an ADDITIVE key on a JSONL row, and every
#: reader in the tree takes the row as a mapping and asks for the fields it
#: wants - none enumerates them, none rejects an unknown one. A version bump
#: would force every reader to learn a number that changes nothing about how it
#: reads. A row written before P9 has no `like_mode` at all, and absence means
#: `claimed`, because a claim was REQUIRED until this packet.
LIKE_MODE_CLAIMED = "claimed"
LIKE_MODE_QUICK = "quick"
LIKE_MODES = (LIKE_MODE_CLAIMED, LIKE_MODE_QUICK)


def like_mode_of(row) -> str:
    """The mode a like row was made in. Absence reads as `claimed`.

    One place, because four readers now need the answer and a second copy of
    "if it has no mode it is claimed" would eventually disagree.
    """
    if not hasattr(row, "get"):
        return LIKE_MODE_CLAIMED
    mode = str(row.get("like_mode") or "").strip().lower()
    return mode if mode in LIKE_MODES else LIKE_MODE_CLAIMED
#: WHICH SCREEN the verdict came from (P10, trader 2026-09-02: *"the veto and
#: like+claim tabs are just quicker ways to make a note for a stock"*, and *"a
#: star in Master AVWAP setups and a like in chart review are the SAME thing"*).
#:
#: One bucket, graded together, **the screen it came from is a column**. That is
#: the whole design: `surface` never splits a cohort at write time, because two
#: verdicts about the same chart are the same verdict whichever button reached
#: it. It rides along so a later rollup can ask whether the trader is a better
#: judge on one screen than another - a question that needs the column and would
#: be destroyed by two cohorts.
#:
#: These are NOT `review_events.setup_context_fields`' `surface` values (that one
#: writes `"setups"`). Different file, different vocabulary, and neither is
#: renamed: rows already written carry the spelling they were written with.
SURFACE_MASTER_AVWAP = "master_avwap_setups"
SURFACE_CHART_REVIEW = "chart_review"
SURFACE_FOCUS_PANEL = "focus_panel"
SURFACE_M5_ALERT_BAR = "m5_alert_bar"
SURFACE_RAIL = "rail"
SURFACES = (
    SURFACE_MASTER_AVWAP,
    SURFACE_CHART_REVIEW,
    SURFACE_FOCUS_PANEL,
    SURFACE_M5_ALERT_BAR,
    SURFACE_RAIL,
)

#: The scanner row under the click, when there IS one (P10 B1). Trader: *"anytime
#: I like a D1 it should be treated with respect by the bot in regards to finding
#: out what's good about it, how we can replicate those searches"* - and you
#: cannot replicate a search from a bare symbol and a timestamp.
#:
#: Every one of these is copied from a row the desk was ALREADY showing. **A
#: capture click never fetches** (the pass rule, and it holds here for the same
#: reason): a verdict must cost one write, and a lookup that fails would either
#: block the click or write a field that lies. With no row under the click - a
#: bare symbol lookup - they are simply absent, and absence is the honest answer.
SCAN_CONTEXT_FIELDS = (
    "scan_date",
    "tracker_setup_id",
    "canonical_setup_id",
    "priority_bucket",
    "score",
    "expected_r",
)

ANNOTATION_SOURCE = "chart_review"

EVENT_VETO = "veto"
EVENT_LIKE_CLAIM = "like_claim"
EVENT_HYPO_STOP = "hypo_stop"
EVENT_NOTE = "note"
#: The day-trade pass (2026-08-31). A name the trader LIKED and did not take
#: because of one specific issue, ticked from the ``pass_reasons`` vocabulary.
#: Deliberately not a veto: a veto says the chart is not for today, a pass says
#: the trade was there and one thing was in the way - and, on the surface, a
#: pass never retires the chart (it behaves like a note).
EVENT_PASS = "pass"
EVENT_TYPES = (
    EVENT_VETO,
    EVENT_LIKE_CLAIM,
    EVENT_HYPO_STOP,
    EVENT_NOTE,
    EVENT_PASS,
)

#: A pass is multi-select by trader instruction; the cap only exists so one
#: row can never outgrow MAX_ROW_BYTES. It is larger than any shipped
#: vocabulary, so ticking every box is always writable.
MAX_PASS_REASONS = 16

#: Notes are a capture surface, not a journal - the journal already exists.
#: The cap is what keeps a row inside :data:`MAX_ROW_BYTES`, so every row is
#: one small buffered write.
MAX_NOTE_CHARS = 2000
#: One buffered write per row keeps cooperating writers from interleaving
#: (they also hold the lock); it does not make a crash mid-write impossible,
#: which is why the appender heals torn tails instead of claiming atomicity.
#: Rows are ~300 bytes; the cap only ever trips on a pathological note.
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


def _clean_pass_codes(codes: Any, vocabulary: VetoVocabulary) -> list[str]:
    """Validate a multi-select pass into vocabulary order, without duplicates.

    Ordered by the VOCABULARY, not by the order the trader ticked boxes: two
    passes citing the same two reasons have to compare equal months from now,
    and click order carries no meaning worth preserving over that.
    """
    if isinstance(codes, str):
        raw = [codes]
    else:
        try:
            raw = list(codes or ())
        except TypeError as exc:
            raise AnnotationError(f"reason_codes is not a list: {codes!r}") from exc
    wanted = {str(code or "").strip().lower() for code in raw}
    wanted.discard("")
    if not wanted:
        raise AnnotationError("a pass needs at least one reason")
    unknown = sorted(code for code in wanted if vocabulary.reason(code) is None)
    if unknown:
        raise AnnotationError(
            f"reason_codes {unknown} are not in {vocabulary.vocabulary_id} "
            f"v{vocabulary.vocab_version} ({list(vocabulary.codes)})"
        )
    if len(wanted) > MAX_PASS_REASONS:
        raise AnnotationError(
            f"a pass carries {len(wanted)} reasons; the cap is {MAX_PASS_REASONS}"
        )
    return [code for code in vocabulary.codes if code in wanted]


def build_annotation(
    event_type: str,
    *,
    symbol: Any,
    session_date: Any = None,
    created_at: datetime | None = None,
    reason_code: str = "",
    reason_codes: Any = (),
    vocabulary: VetoVocabulary | None = None,
    m5_bars_ref: str = "",
    m5_bar_count: Any = None,
    m5_first_bar: str = "",
    m5_last_bar: str = "",
    claimed_setup_id: str = "",
    like_mode: str = "",
    stop_price: Any = None,
    side: Any = "",
    last_price: Any = None,
    ref_level_id: str = "",
    ref_level_family: str = "",
    note: Any = "",
    timeframe: str = "",
    event_id: str = "",
    surface: str = "",
    supersedes: str = "",
    scan_context: Any = None,
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
        code = str(reason_code or "").strip().lower()
        if not code:
            # AN UNCODED VETO IS LEGAL AND IT IS NOT A CODED ONE (P10 A1).
            # "Not today" in chart review has never asked for a code - it writes
            # `reason="not today"`, a hardcoded string - and the trader asked for
            # a note box there rather than a picklist. So the row carries no
            # `reason_code` and NO `vocab_version`: a version stamp on a row that
            # cites no vocabulary would put it in a pooled cohort it was never
            # part of, and `_rebuild_pooled_performance` pools on exactly that
            # pair. It grades under its own `veto_uncoded` name instead, which
            # keeps the trader's coded record uncontaminated in both directions.
            pass
        else:
            vocab = vocabulary if vocabulary is not None else load_veto_vocabulary()
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

    if kind == EVENT_PASS:
        vocab = vocabulary if vocabulary is not None else load_pass_vocabulary()
        row["reason_codes"] = _clean_pass_codes(reason_codes, vocab)
        row["vocab_version"] = vocab.vocab_version
        row["vocabulary_id"] = vocab.vocabulary_id

    if kind == EVENT_LIKE_CLAIM:
        mode = str(like_mode or "").strip().lower() or LIKE_MODE_CLAIMED
        if mode not in LIKE_MODES:
            raise AnnotationError(f"unknown like_mode {like_mode!r}; expected one of {LIKE_MODES}")
        claim = str(claimed_setup_id or "").strip().lower()
        if mode == LIKE_MODE_CLAIMED and not claim:
            # Unchanged for the claimed path: naming the setup is the whole
            # point of it.
            raise AnnotationError("like_claim requires a claimed_setup_id")
        if mode == LIKE_MODE_QUICK and claim:
            # A quick like that carried a claim would be a claimed like wearing
            # the wrong label, and every split by mode would be wrong after it.
            raise AnnotationError("a quick like carries no claimed_setup_id")
        row["like_mode"] = mode
        if claim:
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
    if surface:
        screen = str(surface).strip().lower()
        if screen not in SURFACES:
            # A typo here would silently create a sixth screen that no rollup
            # knows about, and rows are never rewritten.
            raise AnnotationError(f"unknown surface {surface!r}; expected one of {SURFACES}")
        row["surface"] = screen
    if supersedes:
        # THE LINEAGE, NOT AN EDIT (P10 A2). The click row is already on disk and
        # is never touched; this row says "the note that belongs with that
        # click". Append-only is the whole contract of this file, so a note
        # typed three seconds later is a second row, and the pair is joined by
        # the id rather than merged.
        row["supersedes"] = str(supersedes).strip()
    if scan_context:
        for key in SCAN_CONTEXT_FIELDS:
            value = scan_context.get(key) if hasattr(scan_context, "get") else None
            if value is None or str(value).strip() == "":
                continue
            row[key] = value
    # The attached chart, when the desk already held one. Written as a
    # reference rather than inline so the row stays one small buffered write -
    # see ui.annotations.pass_bars for why the bars live in a sidecar.
    if m5_bars_ref:
        row["m5_bars_ref"] = str(m5_bars_ref)
        if m5_bar_count is not None:
            row["m5_bar_count"] = int(m5_bar_count)
        if m5_first_bar:
            row["m5_first_bar"] = str(m5_first_bar)
        if m5_last_bar:
            row["m5_last_bar"] = str(m5_last_bar)
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

    True means fsynced: for a decision stream that can never be reconstructed,
    "saved" has to survive a power cut, not just a process exit. And before
    writing, a tail left torn by an earlier crashed write is healed with a
    newline, so that torn fragment can only ever cost its own row - it can
    never fuse with this one and take two decisions down together.
    """
    line = json.dumps(row, sort_keys=True, default=str) + "\n"
    encoded = line.encode("utf-8")
    if len(encoded) > MAX_ROW_BYTES:
        raise AnnotationError(
            f"row is {len(encoded)} bytes; the single-write cap is {MAX_ROW_BYTES}"
        )
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with local_writer_lock(lock_key_for_path(target), timeout_seconds=1.0):
            if _tail_is_torn(target):
                encoded = b"\n" + encoded
            # Append mode only: this module never truncates or rewrites.
            with target.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
    except (OSError, LocalLockUnavailable):
        return False
    return True


def _tail_is_torn(target: Path) -> bool:
    """Whether the file ends mid-line (a write died before its newline).

    Called under the writer lock. A missing or empty file is a clean tail.
    """
    try:
        if target.stat().st_size == 0:
            return False
    except FileNotFoundError:
        return False
    with target.open("rb") as probe:
        probe.seek(-1, os.SEEK_END)
        return probe.read(1) != b"\n"


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


def record_annotation_with_bars(
    event_type: str,
    *,
    m5_bars: Any = (),
    path: Path = TRADER_ANNOTATIONS_FILE,
    **fields: Any,
) -> dict[str, Any] | None:
    """Write one annotation, attaching cached M5 bars when there are any.

    The id is minted HERE rather than inside :func:`build_annotation` because
    the sidecar is named after it and has to be on disk before the row that
    references it - see :mod:`ui.annotations.pass_bars` for why that order is
    the one that cannot lie.

    A sidecar that fails to write costs the bars and never the row: the
    trader's stated judgement is the evidence, and the chart behind it is a
    bonus the desk could only ever offer when it happened to be holding one.

    Generalised from `record_pass_annotation` for P9's quick like, which the
    trader asked to save bars the same way. The FIELD NAME stays `m5_bars_ref`
    and the sidecar directory stays the pass one, so no reader forks: what
    changes is which event types can own a sidecar, not what a sidecar is.
    """
    from ui.annotations import pass_bars

    event_id = str(fields.pop("event_id", "") or "").strip() or uuid.uuid4().hex
    row = build_annotation(event_type, event_id=event_id, **fields)
    reference = pass_bars.write_pass_bars(
        event_id,
        list(m5_bars or ()),
        symbol=row.get("symbol", ""),
        side=row.get("side", ""),
        created_at=row.get("created_at", ""),
        annotations_path=path,
    )
    if reference:
        row.update(reference)
    return row if append_annotation_row(row, path=path) else None


def record_pass_annotation(**kwargs: Any) -> dict[str, Any] | None:
    """One day-trade pass, bars attached. The name every existing caller uses."""
    return record_annotation_with_bars(EVENT_PASS, **kwargs)


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
