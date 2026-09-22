r"""`prediction_contrast` - what leads to a good call (TJ-16 item 3).

`plan.md` §12.4 TJ-16 item 3: *"right against wrong over `LATELY_SESSIONS` and
over the whole ledger, per context field - counts, the Wilson interval,
`observational, top 3 of K`, nothing named under the floor, separately for
`Rest of day` and `Next 5 sessions`. No model."*

TJ-15 asked what the MISSES had in common. This asks the same question of the
trader's own reads: what did the market look like when they were right, and what
did it look like when they were wrong? The method is TJ-15's method, CALLED -
`evidence_contrast.contrast` already holds the rank key, the two floors and the
sentence, and a second contrast would be a second answer to the same question.

Seven rules it is built around.

* **The contrast is TJ-15's, through the module attribute.** Nothing here
  computes an AUC, a rank key or a floor of its own.
* **A CLICK is the population.** A clicked read is what the trader STATED and an
  extracted stance is what the desk INFERRED; they are never pooled (decision
  0021 answer 29). The extracted rows are COUNTED in ``excluded_by_source``
  rather than silently dropped - a population nobody can see was skipped is one
  the reader assumes was included.
* **The two horizons are never pooled.** `Rest of day` and `Next 5 sessions` get
  their own counts, their own two contrasts and their own tables.
* **Two populations, stated apart.** ``lately`` is the last
  `evidence_stats.LATELY_SESSIONS` exchange sessions ending at the pack's
  session; ``all`` is every session the ledger holds a file for. One number over
  both would hide whichever end of the ledger the trader has improved on.
* **A field the desk could not measure is not a zero.** A categorical field
  becomes one feature per VALUE - ``f"{field}:{value}"`` - worth 1.0 on a row
  holding that value and 0.0 on a row holding a DIFFERENT MEASURED value, and
  NOTHING AT ALL on a row whose field reads `unmeasured` (plan.md sec 5). A
  numeric field keeps its own name. Tag codes are ``f"tag:{code}"`` under the
  same rule: an untagged read contributes nothing, a tagged one without the code
  contributes 0.0.
* **`flat` is counted and is in NEITHER contrast group.** The market did
  nothing, so the trader was neither right nor wrong; it stays in the accuracy
  denominator (TJ-10's lead decision 3) and folding it into either side of a
  contrast would be inventing a verdict.
* **Nothing is hidden, and nothing is ranked by result.** Every feature that
  cleared the FEATURE floor is shown, in `|AUC-0.5|` order - the bounded view
  this pack owns is :func:`tendencies`, which names at most three cells and
  orders them by `n`, a SIZE rule, never by a rate (gate #43). A feature under
  the floor keeps its row in ``thin_features`` with both counts and no AUC.

Deterministic: ``uses_model=False``, no inference, seconds of work. A missing
input is a recorded reason on an `ok` row - an evidence job is never allowed to
cost the thing it records. Nothing it writes reaches a detector, score, alert,
watchlist, Focus, the review queue or `review_policy.json`; it is REPORTED
evidence and a reader.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import evidence_contrast
import evidence_stats
import market_read_grades as grades
import prediction_ledger
import trade_mentor_context
import walkaway_day

_log = logging.getLogger(__name__)

#: The pack's own name. A later shape is a new version, never a re-reading.
PACK_SCHEMA = "prediction_contrast_v1"

#: What every pack file is called, before any superseding sibling.
PACK_PREFIX = "prediction_contrast"

#: The two group names. They are the trader's own words for the two outcomes.
LABEL_RIGHT = "right"
LABEL_WRONG = "wrong"

#: Context keys that are not a scalar reading. `internals` is the whole
#: `trade_mentor_context_v2` block - a nested document, not a field - and
#: flattening it here would put the same numbers in twice under invented names.
CONTEXT_SKIP: frozenset[str] = frozenset({"internals", "availability", "reason"})

#: The tables the live gate reads, in the order they are printed.
TABLES = ("by_hour", "by_environment")

#: Which context field each table groups on.
TABLE_FIELDS = {"by_hour": "hour", "by_environment": "d1_environment"}

#: How many tendencies the week story may be handed. A bounded view; the number
#: of cells over the floor is countable from the tables themselves.
TENDENCY_LIMIT = 3


# ---------------------------------------------------------------------------
# the feature encoding
# ---------------------------------------------------------------------------
def _is_unmeasured(value: Any) -> bool:
    """A field the desk could not measure. Never a zero, never a category."""
    if value is None:
        return True
    text = str(value).strip()
    return not text or text == grades.UNMEASURED


def _measured_derived(
    internals: Mapping[str, Any], name: str
) -> Mapping[str, Any] | None:
    """One canonical measured v2 derived line, or no feature at all.

    This is deliberately an allowlist reader.  The context is an archival
    document, so notes, sources, stamps, free-form keys and future lines must
    not silently become outcome features merely because they were stored.
    """
    if str(internals.get("schema") or "") != trade_mentor_context.SCHEMA:
        return None
    if name not in trade_mentor_context.DERIVED_LINES:
        return None
    derived = internals.get("derived")
    line = derived.get(name) if isinstance(derived, Mapping) else None
    if not isinstance(line, Mapping) or str(line.get("status") or "") != "measured":
        return None
    return line


def _recorded_number(value: Any) -> float | None:
    """A finite stored number, preserving zero and never re-measuring it."""
    if isinstance(value, bool):
        return None
    return evidence_contrast.measurement(value)


def _internals_features(context: Mapping[str, Any]) -> dict[str, Any]:
    """The bounded canonical-v2 market-internals projection for one grade row."""
    internals = context.get("internals")
    if not isinstance(internals, Mapping):
        return {}
    out: dict[str, Any] = {}
    for name in ("breadth", "rates", "oil", "offense_vs_defense"):
        line = _measured_derived(internals, name)
        value = _recorded_number(line.get("value")) if line else None
        if value is not None:
            out[f"internals.{name}.value"] = value

    fear = _measured_derived(internals, "fear")
    if fear:
        for field in ("vxx_direction", "spy_direction"):
            value = str(fear.get(field) or "")
            if value in {"up", "down", "flat"}:
                out[f"internals.fear.{field}:{value}"] = 1.0
        divergence = fear.get("divergence")
        if isinstance(divergence, bool):
            out[f"internals.fear.divergence:{divergence}"] = 1.0

    above = _measured_derived(internals, "sectors_above_vwap")
    if above:
        for field in ("count", "denominator"):
            value = _recorded_number(above.get(field))
            if value is not None:
                out[f"internals.sectors_above_vwap.{field}"] = value

    for name in ("sector_leaders", "sector_laggards"):
        line = _measured_derived(internals, name)
        if not line:
            continue
        for window in ("day", "m30"):
            members = line.get(window)
            if not isinstance(members, (list, tuple)):
                continue
            chosen = {str(member) for member in members}
            if not chosen <= set(trade_mentor_context.SECTORS):
                continue
            for sector in trade_mentor_context.SECTORS:
                out[f"internals.{name}.{window}:{sector}"] = (
                    1.0 if sector in chosen else 0.0
                )
    return out


def _scalar_context(row: Mapping[str, Any]) -> dict[str, Any]:
    """One grade row's context, scalars only, in the shape a contrast eats."""
    context = row.get("context")
    if not isinstance(context, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in context.items():
        name = str(key)
        if name in CONTEXT_SKIP:
            continue
        if isinstance(value, (Mapping, list, tuple, set)):
            continue
        out[name] = value
    out.update(_internals_features(context))
    return out


def _tag_codes(tags: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Every code any note carried - the universe a `tag:` feature comes from."""
    found: set[str] = set()
    for codes in (tags or {}).values():
        for code in codes or ():
            text = str(code or "").strip()
            if text:
                found.add(text)
    return tuple(sorted(found))


def encode_rows(
    rows: Sequence[Mapping[str, Any]], *, tags: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """One feature mapping per grade row. PURE, and the encoding rule is pinned.

    A NUMERIC field keeps its own name. A CATEGORICAL one becomes one feature
    per value seen in this population, 1.0 on a row holding that value and 0.0
    on a row holding a different MEASURED value. A row whose field reads
    `unmeasured` contributes nothing at all to that feature - it is unknown, not
    "not above", and reading the blank as a zero would move the statistic with
    rows nobody measured.

    A field that parses as a number ANYWHERE in the population is numeric
    everywhere: `gap_pct` is measured on five of forty reads and reads
    `unmeasured` on the rest, and those thirty-five are not a category.
    """
    scalars = [_scalar_context(row) for row in rows]
    numeric: set[str] = set()
    categories: dict[str, set[str]] = {}
    for block in scalars:
        for name, value in block.items():
            if _is_unmeasured(value):
                continue
            if evidence_contrast.measurement(value) is not None:
                numeric.add(name)
            else:
                categories.setdefault(name, set()).add(str(value))
    for name in numeric:
        categories.pop(name, None)

    codes = _tag_codes(tags)
    tagged = {str(key) for key in (tags or {})}
    out: list[dict[str, Any]] = []
    for row, block in zip(rows, scalars):
        mapping: dict[str, Any] = {}
        for name, value in block.items():
            if _is_unmeasured(value):
                continue
            if name in numeric:
                number = evidence_contrast.measurement(value)
                if number is not None:
                    mapping[name] = number
                continue
            mine = str(value)
            for candidate in sorted(categories.get(name) or ()):
                mapping[f"{name}:{candidate}"] = 1.0 if candidate == mine else 0.0
        entry_id = str(row.get("entry_id") or "")
        if codes and entry_id in tagged:
            mine_codes = {str(code) for code in (tags or {}).get(entry_id) or ()}
            for code in codes:
                mapping[f"tag:{code}"] = 1.0 if code in mine_codes else 0.0
        out.append(mapping)
    return out


# ---------------------------------------------------------------------------
# the contrast and the tables
# ---------------------------------------------------------------------------
def _split(
    rows: Sequence[Mapping[str, Any]], mappings: Sequence[Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """(right, wrong). A `flat`, a `pending` and an `unmeasured` are in NEITHER."""
    right: list[Mapping[str, Any]] = []
    wrong: list[Mapping[str, Any]] = []
    for row, mapping in zip(rows, mappings):
        verdict = str(row.get("verdict") or "")
        if verdict == grades.VERDICT_RIGHT:
            right.append(mapping)
        elif verdict == grades.VERDICT_WRONG:
            wrong.append(mapping)
    return right, wrong


def _contrast(
    rows: Sequence[Mapping[str, Any]],
    mappings: Sequence[Mapping[str, Any]],
    *,
    min_side: int,
    min_total: int,
) -> dict[str, Any]:
    right, wrong = _split(rows, mappings)
    names = sorted({key for mapping in mappings for key in mapping})
    # Every feature that cleared the FEATURE floor is shown. `top` bounds a view
    # for readability, and here the readable unit is the whole (small) context
    # block; the pack's bounded view is `tendencies`, which is ordered by `n`.
    # Nothing is selected by a result either way.
    top = max(evidence_contrast.DEFAULT_TOP, len(names))
    return evidence_contrast.contrast(
        right,
        wrong,
        label_a=LABEL_RIGHT,
        label_b=LABEL_WRONG,
        top=top,
        min_side=min_side,
        min_total=min_total,
    )


def _table(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    """Right against wrong per value of ONE context field, every cell with `n`."""
    buckets: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        context = row.get("context")
        if not isinstance(context, Mapping):
            continue
        value = context.get(field)
        if _is_unmeasured(value):
            continue
        number = evidence_contrast.measurement(value)
        key: Any = int(number) if number is not None and float(number).is_integer() else (
            number if number is not None else str(value)
        )
        buckets.setdefault(key, []).append(row)

    cells: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda item: (isinstance(item, str), item)):
        cell = grades.accuracy(buckets[key])
        measured = evidence_contrast.rate(
            cell["right"], cell["n"], pending=cell["pending"]
        )
        cells.append(
            {
                "key": key,
                "n": cell["n"],
                "right": cell["right"],
                "wrong": cell["wrong"],
                "flat": cell["flat"],
                "pending": cell["pending"],
                "unmeasured": cell["unmeasured"],
                "rate": measured["rate"],
                "low": measured["low"],
                "high": measured["high"],
                "reportable": measured["reportable"],
            }
        )
    return cells


# ---------------------------------------------------------------------------
# the pack
# ---------------------------------------------------------------------------
def _window(session: str, count: int) -> tuple[str, ...]:
    """The ``count`` exchange sessions ENDING at ``session``, oldest first."""
    size = max(1, int(count or 1))
    return tuple(walkaway_day.earlier_sessions(session, count=size - 1)) + (session,)


def build_pack(
    session_date: str,
    *,
    now: datetime | None = None,
    rows: Iterable[Mapping[str, Any]] | None = None,
    tags: Mapping[str, Any] | None = None,
    written_after: Iterable[str] = (),
    window_sessions: int = evidence_stats.LATELY_SESSIONS,
    min_side: int | None = None,
    min_total: int | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """The right-against-wrong pack for one session. Pure: no store, no clock.

    ``rows`` are grade rows in `market_read_grades`' own shape, already reduced
    to the CURRENT row per read. ``tags`` is ``{entry_id: [code, ...]}`` from a
    previous night's :mod:`ai_jobs.observation_tags` file - tonight's tags can
    never be in tonight's pack, because the tagger is a stage 2 slot and this is
    a stage 1 one.

    ``written_after`` names the tagged entries whose note was written AFTER the
    session closed (`market_journal`'s computed `written_after_the_session`). It
    is COUNTED and nothing else: no feature is dropped, no row is re-weighted
    and no ranked number moves. The trader's own words are the artifact under
    study and a hindsight `because` is still their reasoning - but a reader has
    to be able to partition it, and a count nobody can see is not a partition
    (reviewer advisory 1, 2026-09-20).

    ``min_side`` and ``min_total`` are the FEATURE floor, defaulted to the
    desk's constants and passed straight to `evidence_contrast.contrast`. They
    exist so a test can pin the ranking mechanics on a small fixture without
    restating them; no production caller passes either.
    """
    session = str(session_date or "")[:10]
    moment = now or datetime.now()
    side_floor = (
        evidence_contrast.MIN_CONTRAST_SIDE_N if min_side is None else int(min_side)
    )
    total_floor = (
        evidence_stats.MIN_REPORTABLE_N if min_total is None else int(min_total)
    )

    listed = [dict(row) for row in rows or () if isinstance(row, Mapping)]
    excluded: dict[str, int] = {grades.SOURCE_EXTRACTED: 0}
    clicked: list[dict[str, Any]] = []
    for row in listed:
        source = str(row.get("source") or "") or "unknown"
        if source == grades.SOURCE_CLICK:
            clicked.append(row)
        else:
            excluded[source] = excluded.get(source, 0) + 1

    window = _window(session, window_sessions) if session else ()
    inside = set(window)
    tag_map = {str(key): list(value or ()) for key, value in (tags or {}).items()}
    codes = _tag_codes(tag_map)
    hindsight = {str(value) for value in written_after or ()} & set(tag_map)

    horizons: dict[str, Any] = {}
    for name in prediction_ledger.HORIZONS:
        mine = [row for row in clicked if prediction_ledger.horizon_of(row) == name]
        lately = [row for row in mine if str(row.get("session") or "")[:10] in inside]
        cell = grades.accuracy(mine)
        horizons[name] = {
            "horizon": name,
            "label": prediction_ledger.HORIZON_LABELS.get(name, name),
            "n": cell["n"],
            "right": cell["right"],
            "wrong": cell["wrong"],
            "flat": cell["flat"],
            "pending": cell["pending"],
            "unmeasured": cell["unmeasured"],
            "rate": cell["rate"],
            "rate_lb": cell["rate_lb"],
            "meets_floor": cell["meets_floor"],
            "lately": _contrast(
                lately,
                encode_rows(lately, tags=tag_map),
                min_side=side_floor,
                min_total=total_floor,
            ),
            "all": _contrast(
                mine,
                encode_rows(mine, tags=tag_map),
                min_side=side_floor,
                min_total=total_floor,
            ),
            "tables": {table: _table(mine, TABLE_FIELDS[table]) for table in TABLES},
        }

    matched = sum(1 for row in clicked if str(row.get("entry_id") or "") in tag_map)
    empty = not clicked
    if empty:
        statement = (
            "no clicked calls yet, so there is nothing to contrast - this is a "
            "fact about the session, not a failure of the job"
        )
    else:
        statement = (
            f"observational, not causal: {len(clicked)} clicked read(s), right "
            f"against wrong per point-in-time context field, over the last "
            f"{len(window)} session(s) ending {session} (`lately`) and over every "
            "session on disk (`all`), the two horizons kept apart. A `flat` "
            "reading is counted and is in NEITHER group. A feature under the "
            f"floor ({side_floor} rows a side, {total_floor} across both) keeps "
            "its row and its counts in `thin_features` and is never ranked. "
            "Nothing here is acted on."
        )

    return {
        "schema": PACK_SCHEMA,
        "session_date": session,
        "built_at": moment.isoformat(),
        "source": grades.SOURCE_CLICK,
        "window_sessions": int(window_sessions),
        "window_first_session": window[0] if window else "",
        "reads": len(clicked),
        "empty": empty,
        "statement": statement,
        "excluded_by_source": dict(sorted(excluded.items())),
        "feature_floor": {"min_side": side_floor, "min_total": total_floor},
        "tags": {
            "entries_tagged": len(tag_map),
            "reads_matched": matched,
            "codes": list(codes),
            # A LABEL, never a filter: how many of the tagged notes were written
            # after the bell. Present and zero, never absent - a reader that has
            # to tell "none" from "this build did not measure it" is reading two
            # different absences as one.
            "entries_written_after": len(hindsight),
            # NAMED even when no read carried one, so "nobody has tagged a note
            # yet" and "the tags do not join these reads" are different states a
            # reader can tell apart.
            "features": [f"tag:{code}" for code in codes],
        },
        "horizons": horizons,
        "notes": [str(note) for note in notes if str(note or "").strip()],
    }


# ---------------------------------------------------------------------------
# the bounded view TJ-5's week story may narrate
# ---------------------------------------------------------------------------
def tendencies(pack: Mapping[str, Any], *, limit: int = TENDENCY_LIMIT) -> list[dict[str, Any]]:
    """At most ``limit`` cells the week story may say something about.

    Each names its own cell and its `n`, so the model quotes a number it did not
    compute. **Ordered by `n` descending then by name** - a SIZE rule, never a
    rate (gate #43) - and a cell under `evidence_stats.MIN_REPORTABLE_N` is
    never offered at all, however quotable it reads.
    """
    found: list[dict[str, Any]] = []
    horizons = pack.get("horizons") if isinstance(pack, Mapping) else None
    for name, horizon in (horizons or {}).items():
        if not isinstance(horizon, Mapping):
            continue
        tables = horizon.get("tables")
        if not isinstance(tables, Mapping):
            continue
        for table in TABLES:
            for cell in tables.get(table) or ():
                if not isinstance(cell, Mapping):
                    continue
                if not cell.get("reportable") or cell.get("rate") is None:
                    continue
                found.append(
                    {
                        "text": _tendency_text(str(name), table, cell),
                        "n": int(cell.get("n") or 0),
                        "cell": {
                            "table": table,
                            "key": cell.get("key"),
                            "horizon": str(name),
                        },
                    }
                )
    found.sort(key=lambda item: (-item["n"], str(item["cell"]["table"]), str(item["cell"]["key"])))
    return found[: max(0, int(limit))]


def _tendency_text(horizon: str, table: str, cell: Mapping[str, Any]) -> str:
    label = prediction_ledger.HORIZON_LABELS.get(horizon, horizon)
    key = cell.get("key")
    where = (
        f"at {int(key):02d}00" if table == "by_hour" else f"when the desk read {key}"
    )
    try:
        rate = f"{float(cell.get('rate')) * 100:.0f}%"
    except (TypeError, ValueError):
        rate = "unmeasured"
    return (
        f"{label}, {where}: right {rate} of {cell.get('n')} closed call(s) "
        f"({cell.get('right')} right, {cell.get('wrong')} wrong) - observational, "
        "and the interval is on the cell"
    )


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


def pack_path(session_date: str, root: Any = None) -> Path:
    return _pack_dir(root) / f"{PACK_PREFIX}-{str(session_date)[:10]}.json"


def _publish(pack: Mapping[str, Any], root: Any = None) -> Path:
    """Write ONE pack, temp-and-rename, onto a superseding sibling (D6)."""
    from ai_jobs.digest import _publish as publish, superseding_path

    target = superseding_path(pack_path(str(pack.get("session_date") or ""), root))
    return publish(target, json.dumps(pack, indent=1, sort_keys=True, default=str) + "\n")


def read_latest(session_date: str, *, root: Any = None) -> dict[str, Any] | None:
    """The newest pack for ``session_date``, or ``None``. Never raises.

    TJ-5's week story and TJ-12's Day Review line read through here; this packet
    ships the reader and leaves the page hooks to those packets.
    """
    session = str(session_date or "")[:10]
    if not session:
        return None
    try:
        folder = _pack_dir(root, create=False)
        candidates = sorted(folder.glob(f"{PACK_PREFIX}-{session}*.json"))
    except (OSError, ValueError):
        return None

    def _index(path: Path) -> int:
        tail = path.stem[len(f"{PACK_PREFIX}-{session}") :].lstrip(".")
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


# ---------------------------------------------------------------------------
# the default readers (only used when the caller supplies nothing)
# ---------------------------------------------------------------------------
def _read_ledger(reads_root: Any) -> tuple[list[dict[str, Any]], str]:
    """Every CURRENT grade row on disk, or a reason. Never raises.

    The whole ledger rather than the window: ``all`` is a population this pack
    prints beside ``lately``, and a reader that only loaded the window could not
    build it. The store is one small JSONL per session.
    """
    try:
        sessions = prediction_ledger.available_sessions(root=reads_root)
        if not sessions:
            return [], "no read ledger on disk yet - no clicked calls have been graded"
        return prediction_ledger.read_ledger(sessions, root=reads_root), ""
    except Exception as exc:  # noqa: BLE001 - an unreadable store is a REASON
        _log.debug("prediction_contrast: the read ledger was unreadable.", exc_info=True)
        return [], f"read ledger unreadable: {type(exc).__name__}: {exc}"


def _read_tags(
    sessions: Iterable[str], root: Any = None
) -> tuple[dict[str, list[str]], set[str], str]:
    """`({entry_id: [code, ...]}, hindsight entry ids, reason)`. Never raises.

    ONE read of each night's tags file answers both questions: which codes a
    note carried, and whether that note was written after the session closed.
    """
    try:
        from ai_jobs import observation_tags
    except Exception as exc:  # noqa: BLE001
        return {}, set(), f"tags unreadable: {type(exc).__name__}: {exc}"
    out: dict[str, list[str]] = {}
    hindsight: set[str] = set()
    for session in sorted({str(value or "")[:10] for value in sessions if str(value or "").strip()}):
        try:
            found = observation_tags.tagged_entries(session, root=root)
        except Exception:  # noqa: BLE001 - one unreadable night costs one night
            _log.debug("prediction_contrast: a tag file was unreadable.", exc_info=True)
            continue
        for entry_id, row in found.items():
            held = out.setdefault(str(entry_id), [])
            for code in row.get("codes") or ():
                if code not in held:
                    held.append(code)
            if row.get("written_after_the_session"):
                hindsight.add(str(entry_id))
    return out, hindsight, ""


def run_prediction_contrast(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Any = None,
    reads_root: Any = None,
    rows: Iterable[Mapping[str, Any]] | None = None,
    tags: Mapping[str, Any] | None = None,
    written_after: Iterable[str] | None = None,
    window_sessions: int | None = None,
    min_side: int | None = None,
    min_total: int | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """The nightly slot. Deterministic, model-free, and never fails the night.

    Every input is read defensively and a missing one becomes a recorded reason
    on an `ok` row with an empty pack - a session with nothing to contrast is a
    fact about the session, not a failure of the job. On 2026-09-20 that is the
    live state: the click card shipped today and no grade row exists yet.
    """
    notes: list[str] = []
    session = str(session_date or "")[:10]
    moment = now or datetime.now()
    size = int(window_sessions or evidence_stats.LATELY_SESSIONS)

    # A pack keyed to nothing is worse than no pack: every later session would
    # supersede it and no reader could date it (TJ-15's reviewer, 2026-09-19).
    try:
        date.fromisoformat(session)
    except ValueError:
        return {
            "status": "failed",
            "model": "",
            "reason": (
                f"refusing to build a prediction contrast for session_date "
                f"{session_date!r}: a pack must be keyed to a readable date"
            ),
            "outputs": [],
        }

    if rows is None:
        rows, note = _read_ledger(reads_root)
        if note:
            notes.append(note)
    listed = [dict(row) for row in rows or () if isinstance(row, Mapping)]

    if tags is None:
        tags, hindsight, note = _read_tags(
            (str(row.get("session") or "") for row in listed), root=root
        )
        if written_after is None:
            written_after = hindsight
        if note:
            notes.append(note)

    try:
        pack = build_pack(
            session,
            now=moment,
            rows=listed,
            tags=tags,
            written_after=written_after or (),
            window_sessions=size,
            min_side=min_side,
            min_total=min_total,
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001 - arithmetic must not cost the night
        _log.exception("prediction_contrast: the pack could not be built.")
        return {
            "status": "ok",
            "model": "",
            "reason": f"no contrast: {type(exc).__name__}: {exc}",
            "outputs": [],
        }

    try:
        written = _publish(pack, root)
    except OSError as exc:
        return {
            "status": "failed",
            "model": "",
            "reason": f"the prediction contrast pack could not be published: {exc}",
            "outputs": [],
        }

    reason = (
        f"contrasted {pack['reads']} clicked read(s) over {size} session(s) ending "
        f"{session}; {sum(pack['excluded_by_source'].values())} extracted stance(s) "
        "counted and not pooled"
    )
    if notes:
        reason = f"{reason}; {'; '.join(notes)}"
    return {
        "status": "ok",
        "model": "",
        "reason": reason,
        "outputs": [str(written)],
        "extra": {
            "reads": pack["reads"],
            "tagged_reads": pack["tags"]["reads_matched"],
        },
    }


__all__ = [
    "LABEL_RIGHT",
    "LABEL_WRONG",
    "PACK_PREFIX",
    "PACK_SCHEMA",
    "TABLES",
    "build_pack",
    "encode_rows",
    "pack_path",
    "read_latest",
    "run_prediction_contrast",
    "tendencies",
]
