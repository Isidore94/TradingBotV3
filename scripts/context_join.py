"""ONE identity and time contract for thesis, context, opportunity and trade.

Packet WS-10I (WISHLIST 10I, plus 10K's "measure environment, then connect
it"). The trader keeps two accounts of every decision and they are never
merged: what they EXPECTED (the thesis) and what the tape was MEASURED to be
doing (the context). This module attaches the measured one - twice, because
there are two different moments worth asking about - and links the expected one
by scope and validity window.

**Pure.** Rows and a label table in, the SAME rows back with two more fields on
them; frozen tuples out. Nothing here opens a store except `main`, the backfill
CLI at the bottom. Nothing here detects, scores, alerts, ranks, gates or
promotes: it is evidence ABOUT decisions and never an input to one (plan.md
sec 5).

The time rule, which is the whole packet
----------------------------------------

A D1 environment label for session S is a statement about the WHOLE of S, so it
is not available until S has closed. Every join below is that one sentence,
applied to the clock the row actually carries:

* a row whose clock lands BEFORE that close takes the PREVIOUS EXCHANGE
  SESSION's label, `certainty=prior_session`. A 10:35 decision cannot know what
  kind of day today turned out to be;
* a row whose clock is at or after the close takes S's own label,
  `certainty=session`;
* an OBSERVATION dated by session alone (`scan_date = "2026-09-08"`, no time)
  takes that session's own label, `certainty=session`: a swing scan row names
  the session it read completed bars for, and that session's label is exactly
  what it knew;
* a clock that carries no time of day - a bare date on an ENTRY, or a stamp of
  exactly midnight market-local, which is how `journal_statement_import` writes
  a broker fill (`journal_trade_shape.is_date_only`) - takes the PREVIOUS
  completed session's label, `certainty=date_only`, FLAGGED. Handing it the
  session's own label would be the desk claiming it knows the fill happened
  after the close; a date-only fill may never select a midday regime;
* a session nobody labelled reads `unknown`, `certainty=unknown`, and is pooled
  into nothing.

The previous session is walked on the EXCHANGE CALENDAR
(`market_calendar.previous_session`), never in calendar days: one day back from
Tuesday 2026-09-08 is Labor Day, which was never a session and carries no
label, so a calendar-day walk turns a measured cell into `unknown`.

Aware and naive stamps both land in market-local through the journal's own
coercion (`journal_trade_shape._coerce_datetime`: a naive stamp is ATTACHED, an
aware one CONVERTED). 15:35 ET is 19:35 UTC, so a read that strips the offset
decides the close has passed and hands out a label that did not exist yet.

Two refs, side by side
----------------------

`OBSERVATION_CONTEXT_FIELD` and `ENTRY_CONTEXT_FIELD` are separate columns and
one never overwrites the other. An opportunity that was never taken has an
observation context and NO entry context; a trade has both, and they routinely
differ - that difference is the readable part.

Theses
------

`link_theses` links by SCOPE (the benchmark the row's context was measured
against must be one the thesis named) and by VALIDITY WINDOW (`created_at`
until the horizon's last session close, or until `invalidated_at`, whichever
comes first). Never by ticker or date alone. Several live theses all link,
newest first, and **none is chosen** - one opportunity can sit inside three
stances and ambiguity stays ambiguous. A thesis written after the decision is
not evidence about it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import market_calendar
from indicators.d1_environment import LABEL_UNKNOWN, RULE_VERSION

#: The row read its own session's label, which had been published.
CERTAINTY_SESSION = "session"

#: The row landed before its session's close, so it took the one before.
CERTAINTY_PRIOR_SESSION = "prior_session"

#: The clock carried no time of day. Previous completed session, FLAGGED.
CERTAINTY_DATE_ONLY = "date_only"

#: Reconstructed after the fact by the backfill, from what the stores hold
#: today. Never a forward claim - see `backfill_refs`.
CERTAINTY_RECONSTRUCTED = "reconstructed"

#: Nobody labelled that session. `unknown` is its own answer, never a guess.
CERTAINTY_UNKNOWN = "unknown"

#: The label a session with no stored reading carries. One spelling, and it is
#: `indicators.d1_environment`'s own.
UNKNOWN_LABEL = LABEL_UNKNOWN

#: The certainties that mean "read this with care": the answer is the best the
#: stores support and not the thing itself.
FLAGGED_CERTAINTIES = frozenset(
    {CERTAINTY_DATE_ONLY, CERTAINTY_RECONSTRUCTED, CERTAINTY_UNKNOWN}
)

#: The two columns, named once. Side by side, never one instead of the other.
OBSERVATION_CONTEXT_FIELD = "observation_context"
ENTRY_CONTEXT_FIELD = "entry_context"

#: Where the linked theses land. A LIST, newest first, possibly empty.
LINKED_THESES_FIELD = "linked_theses"

#: `when=` takes one of these and nothing else.
WHEN_OBSERVATION = "observation"
WHEN_ENTRY = "entry"

#: The benchmark every scope question is asked against unless a caller says
#: otherwise. `d1_environment_store.BENCHMARKS` holds the three that exist.
DEFAULT_BENCHMARK = "SPY"


@dataclass(frozen=True)
class ContextRef:
    """The measured state a row was decided in, and how sure that join is.

    `context_id` is the CONTEXT's identity - benchmark, rule version and the
    session the label is ABOUT - so two rows that read the same reading carry
    the same id. The certainty belongs to the JOIN, not to the context, and is
    therefore not part of the id: a `session` read and a `reconstructed` read of
    the same reading are the same context, told apart by the certainty and never
    by the label.
    """

    context_id: str
    rule_version: str
    benchmark: str
    label: str
    #: The row's own clock, verbatim as the row spelled it.
    observed_at: str
    #: When the label became knowable: the close of the session it is about,
    #: aware and market-local. Empty when there is no such session.
    available_at: str
    certainty: str

    @property
    def flagged(self) -> bool:
        """Whether this ref needs saying out loud before it is read as a fact."""
        return self.certainty in FLAGGED_CERTAINTIES


def context_id(benchmark: str, rule_version: str, session: str) -> str:
    """The identity of ONE reading. Deterministic, and the only spelling."""
    return f"{benchmark}:{rule_version}:{session}"


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _has_time_of_day(value: Any) -> bool:
    """Whether the raw clock SPELLED a time, not whether it parses to one.

    `2026-09-08` and `2026-09-08 00:00:00` both coerce to midnight, and they are
    different statements: the first names a session, the second names a moment
    the market is shut at. The text is the only place that difference survives.
    """
    if isinstance(value, datetime):
        return True
    if isinstance(value, date):
        return False
    text = str(value or "").strip()
    return len(text) > 10 and (" " in text or "T" in text)


def _previous_session(day: date) -> date | None:
    try:
        return market_calendar.previous_session(day)
    except Exception:  # noqa: BLE001 - a calendar refusal is uncertainty
        return None


def _available_at(session: date | None) -> str:
    if session is None:
        return ""
    try:
        return market_calendar.session_close(session).isoformat()
    except Exception:  # noqa: BLE001
        return ""


def _next_session(day: date) -> date | None:
    cursor = day + timedelta(days=1)
    for _ in range(30):
        try:
            if market_calendar.is_session(cursor):
                return cursor
        except Exception:  # noqa: BLE001
            return None
        cursor += timedelta(days=1)
    return None


def _session_and_certainty(value: Any, when: str) -> tuple[date | None, str]:
    """Which session's label this clock may read, and how sure the answer is."""
    from journal_trade_shape import _coerce_datetime, is_date_only  # noqa: PLC0415

    day = _as_date(value)
    if day is None:
        return None, CERTAINTY_UNKNOWN

    if not _has_time_of_day(value):
        # A date with no clock. For an OBSERVATION that is the session it names;
        # for an ENTRY it is a fill whose time nobody recorded.
        if when == WHEN_ENTRY:
            return _previous_session(day), CERTAINTY_DATE_ONLY
        return day, CERTAINTY_SESSION

    moment = _coerce_datetime(value)
    if moment is None:
        return None, CERTAINTY_UNKNOWN
    if is_date_only(moment):
        return _previous_session(moment.date()), CERTAINTY_DATE_ONLY

    day = moment.date()
    try:
        close = market_calendar.session_close(day)
    except Exception:  # noqa: BLE001
        return None, CERTAINTY_UNKNOWN
    if moment >= close:
        return day, CERTAINTY_SESSION
    return _previous_session(day), CERTAINTY_PRIOR_SESSION


def build_ref(
    value: Any,
    *,
    when: str = WHEN_OBSERVATION,
    labels_by_session: Mapping[str, str] | None = None,
    benchmark: str = DEFAULT_BENCHMARK,
    rule_version: str = RULE_VERSION,
    certainty: str = "",
) -> ContextRef:
    """ONE ref for ONE clock. The whole time rule lives here and nowhere else.

    `certainty=` forces a name - the backfill passes `reconstructed`, because
    what makes a backfilled ref different is not the label (it is the same
    label) but the fact that nobody read it at the time.
    """
    labels = labels_by_session or {}
    session, read_certainty = _session_and_certainty(value, when)
    session_text = session.isoformat() if session is not None else ""
    label = str(labels.get(session_text) or "").strip() if session_text else ""
    if not label:
        label, read_certainty = UNKNOWN_LABEL, CERTAINTY_UNKNOWN
    if certainty:
        read_certainty = certainty
    return ContextRef(
        context_id=context_id(benchmark, rule_version, session_text),
        rule_version=str(rule_version),
        benchmark=str(benchmark),
        label=label,
        observed_at=str(value or ""),
        available_at=_available_at(session),
        certainty=read_certainty,
    )


def attach_context(
    rows: Iterable[MutableMapping[str, Any]],
    *,
    when: str,
    labels_by_session: Mapping[str, str] | None = None,
    clock_field: str = "scan_date",
    benchmark: str = DEFAULT_BENCHMARK,
    rule_version: str = RULE_VERSION,
) -> Any:
    """Write one `ContextRef` onto every row IN PLACE and return the same rows.

    `when="observation"` fills `OBSERVATION_CONTEXT_FIELD`, `when="entry"` fills
    `ENTRY_CONTEXT_FIELD`, and neither ever touches the other's column. In
    place, and deliberately: the Results worker joins tens of thousands of rows
    on a redraw and a second copy of that list is pure cost.

    Idempotent - a second pass writes the same ref over the first.
    """
    if when not in (WHEN_OBSERVATION, WHEN_ENTRY):
        raise ValueError(f"when must be {WHEN_OBSERVATION!r} or {WHEN_ENTRY!r}, not {when!r}")
    field = OBSERVATION_CONTEXT_FIELD if when == WHEN_OBSERVATION else ENTRY_CONTEXT_FIELD
    for row in rows or ():
        try:
            value = row.get(clock_field)
        except AttributeError:
            continue
        row[field] = build_ref(
            value,
            when=when,
            labels_by_session=labels_by_session,
            benchmark=benchmark,
            rule_version=rule_version,
        )
    return rows


# ---------------------------------------------------------------------------
# theses: scope and validity window, never ticker and date
# ---------------------------------------------------------------------------

#: What a thesis row's horizon is worth when it did not say. Five sessions is
#: `evidence_stats.SWING_HORIZON_SESSIONS` - the desk's one declared swing
#: horizon - and it is a FALLBACK, never a correction of a stated one.
DEFAULT_THESIS_SESSIONS = 5

#: The fields a linked entry carries onto the row. Enough to read the link
#: without opening the thesis store, and no word that names a winner.
LINK_FIELDS = (
    "thesis_id",
    "entry_id",
    "created_at",
    "stance",
    "claim",
    "horizon",
    "horizon_sessions",
    "benchmarks",
    "invalidated_at",
)


def _thesis_moment(value: Any) -> datetime | None:
    from journal_trade_shape import _coerce_datetime  # noqa: PLC0415

    return _coerce_datetime(value)


def horizon_end(created: datetime, sessions: int) -> datetime | None:
    """The close of the LAST session a thesis's horizon covers.

    Counted in SESSIONS on the exchange calendar, from the session the thesis
    was written in. "This week" is five sessions, and five sessions across a
    holiday week is six calendar days - the reason nothing here adds days.
    """
    cursor = created.date()
    remaining = max(int(sessions or 0), 0)
    if remaining <= 0:
        try:
            return market_calendar.session_close(cursor)
        except Exception:  # noqa: BLE001
            return None
    for _ in range(remaining):
        nxt = _next_session(cursor)
        if nxt is None:
            return None
        cursor = nxt
    try:
        return market_calendar.session_close(cursor)
    except Exception:  # noqa: BLE001
        return None


def thesis_window(thesis: Mapping[str, Any]) -> tuple[datetime | None, datetime | None]:
    """`(opens, closes)` - `created_at` until the horizon end or invalidation.

    An invalidated thesis stops covering the rows that come after it and keeps
    covering the ones that came before: it WAS the stance at the time, and
    rewriting that would be the desk editing what the trader thought.
    """
    opens = _thesis_moment(thesis.get("created_at") or thesis.get("recorded_at"))
    if opens is None:
        return None, None
    try:
        sessions = int(thesis.get("horizon_sessions") or DEFAULT_THESIS_SESSIONS)
    except (TypeError, ValueError):
        sessions = DEFAULT_THESIS_SESSIONS
    closes = horizon_end(opens, sessions)
    invalidated = _thesis_moment(thesis.get("invalidated_at"))
    if invalidated is not None and (closes is None or invalidated < closes):
        closes = invalidated
    return opens, closes


def _row_moment(row: Mapping[str, Any], when: str) -> datetime | None:
    field = OBSERVATION_CONTEXT_FIELD if when == WHEN_OBSERVATION else ENTRY_CONTEXT_FIELD
    ref = row.get(field)
    if isinstance(ref, ContextRef):
        return _thesis_moment(ref.observed_at)
    return None


def _row_benchmark(row: Mapping[str, Any], when: str, fallback: str) -> str:
    field = OBSERVATION_CONTEXT_FIELD if when == WHEN_OBSERVATION else ENTRY_CONTEXT_FIELD
    ref = row.get(field)
    if isinstance(ref, ContextRef) and ref.benchmark:
        return ref.benchmark
    return fallback


def _link_entry(thesis: Mapping[str, Any], opens: datetime, closes: datetime | None) -> dict:
    entry = {key: thesis.get(key) for key in LINK_FIELDS}
    entry["benchmarks"] = [str(one) for one in (thesis.get("benchmarks") or ())]
    entry["window_start"] = opens.isoformat()
    entry["window_end"] = closes.isoformat() if closes is not None else ""
    return entry


def link_theses(
    rows: Iterable[MutableMapping[str, Any]],
    theses: Sequence[Mapping[str, Any]] | None,
    *,
    when: str = WHEN_OBSERVATION,
    benchmark: str = DEFAULT_BENCHMARK,
) -> Any:
    """Link every thesis whose SCOPE and WINDOW cover the row, IN PLACE.

    Scope is the benchmark the row's own `ContextRef` was measured against -
    which is why `attach_context` runs first: the tier-outcome rows carry no
    benchmark column of their own, and inventing one from the symbol would link
    a QQQ call to an SPY decision.

    Newest `created_at` first. **Nothing is chosen.** A row inside three live
    theses links three, and the thesis review grades each of them separately.
    """
    prepared: list[tuple[datetime, datetime | None, Mapping[str, Any], set[str]]] = []
    for thesis in theses or ():
        if not isinstance(thesis, Mapping):
            continue
        opens, closes = thesis_window(thesis)
        if opens is None:
            continue
        scope = {
            str(one).strip().upper()
            for one in (thesis.get("benchmarks") or ())
            if str(one).strip()
        }
        prepared.append((opens, closes, thesis, scope))
    prepared.sort(key=lambda item: item[0], reverse=True)

    for row in rows or ():
        try:
            moment = _row_moment(row, when)
        except AttributeError:
            continue
        scope_of_row = _row_benchmark(row, when, benchmark).strip().upper()
        linked: list[dict] = []
        if moment is not None:
            for opens, closes, thesis, scope in prepared:
                if scope and scope_of_row not in scope:
                    continue
                if moment < opens:
                    continue
                if closes is not None and moment > closes:
                    continue
                linked.append(_link_entry(thesis, opens, closes))
        row[LINKED_THESES_FIELD] = linked
    return rows


# ---------------------------------------------------------------------------
# the Daily Recap seam (WS-DR renders it)
# ---------------------------------------------------------------------------

#: Where a recap row may spell its two clocks. WS-DR writes one row per
#: opportunity with the matched trade beside it; these are the names that row
#: uses, in the order they are tried.
_RECAP_OBSERVED_FIELDS = ("observed_at", "observation_at", "opportunity_at", "entry_time")
_RECAP_ENTRY_FIELDS = ("entry_at", "filled_at", "opened_at", "trade_opened_at")


def _first_present(row: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return ""


def recap_context_labels(
    row: Mapping[str, Any],
    *,
    labels_by_session: Mapping[str, str] | None = None,
    benchmark: str = DEFAULT_BENCHMARK,
    rule_version: str = RULE_VERSION,
) -> dict[str, Any]:
    """The two labels a Daily Recap row shows, as strings it can print.

    Every row carries the OBSERVATION label - the tape the opportunity was seen
    in. A row with a matched fill carries the ENTRY label BESIDE it, never
    instead of it and never blended with it. An unmatched opportunity gets
    blanks: there is no fill, so there is no entry context to invent.
    """
    observed = _first_present(row, _RECAP_OBSERVED_FIELDS)
    observation = build_ref(
        observed,
        when=WHEN_OBSERVATION,
        labels_by_session=labels_by_session,
        benchmark=benchmark,
        rule_version=rule_version,
    )
    payload: dict[str, Any] = {
        "observation": observation.label,
        "observation_certainty": observation.certainty,
        "observation_flagged": observation.flagged,
        "entry": "",
        "entry_certainty": "",
        "entry_flagged": False,
    }
    entered = _first_present(row, _RECAP_ENTRY_FIELDS)
    if not entered:
        return payload
    entry = build_ref(
        entered,
        when=WHEN_ENTRY,
        labels_by_session=labels_by_session,
        benchmark=benchmark,
        rule_version=rule_version,
    )
    payload["entry"] = entry.label
    payload["entry_certainty"] = entry.certainty
    payload["entry_flagged"] = entry.flagged
    return payload


# ---------------------------------------------------------------------------
# item 4 - the backfill
# ---------------------------------------------------------------------------


def backfill_refs(
    rows: Iterable[Mapping[str, Any]],
    *,
    labels_by_session: Mapping[str, str] | None = None,
    clock_field: str = "scan_date",
    since: str = "",
    when: str = WHEN_OBSERVATION,
    benchmark: str = DEFAULT_BENCHMARK,
    rule_version: str = RULE_VERSION,
) -> list[ContextRef]:
    """Refs for rows that were never labelled at the time, ALL reconstructed.

    Backfill labels only what the contemporaneous stores establish - the D1
    environment store is built from completed daily bars, so a session it holds
    a reading for was readable then - and it says so on every row it writes:
    `certainty=reconstructed`, `flagged=True`. A reconstructed ref is evidence
    about the past and is kept out of every forward claim; a surface that pools
    it with a recorded one would be counting a reading nobody made.

    `since` is an ISO date, inclusive, on the row's own clock. Rows before it
    are not reconstructed at all.
    """
    floor = str(since or "").strip()[:10]
    refs: list[ContextRef] = []
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        value = row.get(clock_field)
        stamp = str(value or "")[:10]
        if floor and (not stamp or stamp < floor):
            continue
        refs.append(
            build_ref(
                value,
                when=when,
                labels_by_session=labels_by_session,
                benchmark=benchmark,
                rule_version=rule_version,
                certainty=CERTAINTY_RECONSTRUCTED,
            )
        )
    return refs


def _backfill_rows(since: str) -> list[dict[str, Any]]:
    """The swing observations the backfill would label. Best effort, read-only."""
    try:
        import project_paths  # noqa: PLC0415
        import swing_evidence  # noqa: PLC0415

        read = swing_evidence.read_eligible_rows(
            project_paths.MASTER_AVWAP_TIER_OUTCOMES_FILE,
            swing_evidence.POLICY_SCANROW_V1,
        )
        clock = swing_evidence.POLICY_SCANROW_V1.clock_field
        return [row for row in read.rows if str(row.get(clock) or "")[:10] >= since]
    except Exception:  # noqa: BLE001 - an absent store is nothing to backfill
        return []


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m context_join backfill --since YYYY-MM-DD` - DRY BY DEFAULT.

    The dry run says where it is pointed and what it would write, and writes
    nothing. `--apply` is the only way to touch a file, and it still refuses to
    label a session the environment store has no reading for.
    """
    parser = argparse.ArgumentParser(prog="context_join", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fill = sub.add_parser("backfill", help="reconstruct context refs for past rows")
    fill.add_argument("--since", default="", help="ISO date, inclusive")
    fill.add_argument("--path", default="", help="where the refs would be written")
    fill.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    fill.add_argument("--apply", action="store_true", help="actually write")
    args = parser.parse_args(list(argv) if argv is not None else None)

    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import project_paths  # noqa: PLC0415

    since = str(args.since or "").strip()[:10]
    target = Path(args.path) if args.path else Path(project_paths.DATA_DIR) / "context_refs.jsonl"

    labels: dict[str, str] = {}
    try:
        import d1_environment_store  # noqa: PLC0415

        labels = dict(
            d1_environment_store.labels_by_session(
                benchmark=args.benchmark, rule_version=RULE_VERSION
            )
        )
    except Exception:  # noqa: BLE001 - no store is no labels, never a guess
        labels = {}

    rows = _backfill_rows(since)
    refs = backfill_refs(
        rows,
        labels_by_session=labels,
        clock_field="scan_date",
        since=since,
        benchmark=args.benchmark,
    )
    measured = len([ref for ref in refs if ref.label != UNKNOWN_LABEL])

    print(f"context_join backfill - data dir {project_paths.DATA_DIR}")
    print(f"  benchmark {args.benchmark} / rule {RULE_VERSION} / since {since or '(everything)'}")
    print(f"  {len(labels)} labelled session(s) in the environment store")
    print(f"  {len(rows)} row(s) read, {len(refs)} ref(s) reconstructed, {measured} labelled")
    print(f"  every reconstructed ref is certainty={CERTAINTY_RECONSTRUCTED} and flagged")
    print(f"  target {target}")
    if not args.apply:
        print("  DRY RUN - nothing was written. Re-run with --apply to write it.")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for ref in refs:
            handle.write(json.dumps(ref.__dict__, sort_keys=True) + "\n")
    temporary.replace(target)
    print(f"  wrote {len(refs)} ref(s) to {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the CLI entry point
    raise SystemExit(main(sys.argv[1:]))
