"""What kind of claim is this family making? — R10.B (D5, D6).

The outcome store treats every registered row as a trade: an entry, a stop, an
R. That is true of a bounce confirmation and false of almost everything else
that reaches it, and the store has no way to say so. `regime_pause_rw` is a
*pause observation*; an H1 `blue_after_red` mark is an *annotation* on a bar
that already closed; an LRSI cross is a real entry claim whose bar was thrown
away before the stop could be computed. Measured together they produce numbers
nobody should act on — `regime_pause_rw` carries an all-time mean of −1.82R,
which is a statement about a family that never claimed to be a trade.

So each family declares a **claim kind**, and the four kinds are answers to
different questions:

``entry_claim``
    "Here is a trade: this entry, this stop." Only these may carry R, an exit
    policy, or a path. This is the only kind an outcome statistic may average.
``annotation``
    "Here is something true about this bar." A mark, not a trade. It may be
    counted and joined; it may never be given an entry price.
``information``
    "Here is the state of the tape." Context for other rows. No symbol-level
    claim at all.
``unconfigured``
    **We do not know what this family claims.** The honest default for a family
    nobody has classified.

`unconfigured` is the load-bearing one. It is *never* silently treated as an
entry claim, because that is exactly how a pause observation acquired a mean R;
it is counted loudly in :func:`coverage`, surfaced on the health tile, and left
out of every statistic that assumes a trade. Missing data is uncertainty, never
confirmation (plan.md sec 5) — a family we have not classified is unmeasured,
not "probably a trade".

This module READS. It never writes, never scores, never gates, and nothing here
may reach a detector, an alert, a watchlist or Focus (plan.md sec 5). A claim
kind is something a rollup prints and a writer consults before deciding whether
a row is even eligible to carry an R.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

#: Registry identity. A changed MEANING is a new name, never a new number
#: (R10 ground rule 5), so a report written in March and one written in
#: September agree about what `entry_claim` meant.
REGISTRY_NAME = "outcome_claim_kinds_v1"

CLAIM_ENTRY = "entry_claim"
CLAIM_ANNOTATION = "annotation"
CLAIM_INFORMATION = "information"
CLAIM_UNCONFIGURED = "unconfigured"

CLAIM_KINDS = (CLAIM_ENTRY, CLAIM_ANNOTATION, CLAIM_INFORMATION, CLAIM_UNCONFIGURED)

#: Only these may be averaged, given an R, or run through an exit policy.
TRADE_BEARING_KINDS = frozenset({CLAIM_ENTRY})


@dataclass(frozen=True)
class FamilySpec:
    """One family's declared semantics."""

    family: str
    claim_kind: str
    why: str
    #: True when the family's registration bar is the bar the signal fired on
    #: and the stop can be taken from it. False for families whose registration
    #: carries no usable bar - they are entry claims that cannot yet be graded,
    #: which is a different thing from not being a trade.
    has_signal_bar: bool = True


def _spec(family: str, kind: str, why: str, *, has_signal_bar: bool = True) -> FamilySpec:
    return FamilySpec(family=family, claim_kind=kind, why=why, has_signal_bar=has_signal_bar)


#: The classified families, enumerated from the LIVE store rather than from
#: memory.
#:
#: The first draft of this table was written from the audit's prose and got two
#: of the three H1 engine names wrong - it invented `h1_red_after_blue` and
#: `h1_reversal`, and missed `h1_ema10_bounce` (92,477 rows, the single largest
#: family in the store) and `h1_green_to_yellow`. Reading the store's 27
#: distinct level names is what corrected it, and is why this list is a
#: measured one rather than a plausible one.
FAMILY_SPECS: tuple[FamilySpec, ...] = (
    # --- entry claims -------------------------------------------------------
    # Each names a LEVEL the bounce plan takes a stop from, so a row carrying
    # it has an entry, a stop and an R that all mean something.
    _spec("10_candle_low", CLAIM_ENTRY, "bounce off the 10-candle low"),
    _spec("10_candle_high", CLAIM_ENTRY, "bounce off the 10-candle high"),
    _spec("vwap", CLAIM_ENTRY, "bounce off session VWAP"),
    _spec("eod_vwap", CLAIM_ENTRY, "bounce off the EOD VWAP"),
    _spec("eod_vwap_upper_band", CLAIM_ENTRY, "bounce off the upper EOD band"),
    _spec("eod_vwap_lower_band", CLAIM_ENTRY, "bounce off the lower EOD band"),
    _spec("vwap_upper_band", CLAIM_ENTRY, "bounce off the upper session band"),
    _spec("vwap_lower_band", CLAIM_ENTRY, "bounce off the lower session band"),
    _spec("dynamic_vwap_upper_band", CLAIM_ENTRY, "bounce off the upper dynamic band"),
    _spec("dynamic_vwap_lower_band", CLAIM_ENTRY, "bounce off the lower dynamic band"),
    _spec("ema_8", CLAIM_ENTRY, "bounce off the 8 EMA"),
    _spec("ema_15", CLAIM_ENTRY, "bounce off the 15 EMA"),
    _spec("ema_21", CLAIM_ENTRY, "bounce off the 21 EMA"),
    _spec("ema8_grind_hod", CLAIM_ENTRY, "8-EMA grind into the high of day"),
    _spec("prev_day_high", CLAIM_ENTRY, "bounce off the previous session high"),
    _spec("prev_day_low", CLAIM_ENTRY, "bounce off the previous session low"),
    _spec("impulse_retest_vwap_eod", CLAIM_ENTRY, "impulse retest of the EOD VWAP"),
    _spec("vwap_eod_confluence", CLAIM_ENTRY, "session and EOD VWAP confluence"),
    _spec("orb_breakout", CLAIM_ENTRY, "opening-range breakout; stop is the opposite edge"),
    _spec("orb_breakdown", CLAIM_ENTRY, "opening-range breakdown; stop is the opposite edge"),
    # H1-derived LEVELS, not the H1 colour engines. An H1 average used as a
    # bounce level on an M5 bar is an ordinary entry claim; the `h1_` prefix
    # they share with the annotations below is a naming collision, and is
    # exactly why this registry matches whole names and never prefixes.
    _spec("h1_ema_15", CLAIM_ENTRY, "bounce off the H1 15-EMA as a level"),
    _spec("h1_sma_20", CLAIM_ENTRY, "bounce off the H1 20-SMA as a level"),
    # LRSI: a real entry claim whose registration bar was synthetic and flat
    # until R10.B. D5a measured ZERO outcome rows for either level.
    _spec("lrsi_cross_20", CLAIM_ENTRY, "LRSI efficiency crossing up through 20"),
    _spec("lrsi_cross_50", CLAIM_ENTRY, "LRSI efficiency crossing up through 50"),
    _spec("m5_confluence", CLAIM_ENTRY, "HA reversal + SMI turn + LRSI cross cluster"),
    # First-candle ORB: classified because that is what it would claim. D5b is
    # UNTESTED rather than proven - the flow has never fired, so there is no
    # row anywhere to check a fix against, and this entry says so.
    _spec("orb_first_candle", CLAIM_ENTRY, "first-candle opening-range claim (never yet fired)"),
    # --- annotations: true about a bar, never a trade ------------------------
    # Audit D6a/D6b. These three ARE the retired H1 colour engines, and they
    # are 82% of every registered row. Their `entry_time` was the bar START on
    # 6,439 of 6,439 rows: a mark on a candle that had already closed.
    _spec("h1_ema10_bounce", CLAIM_ANNOTATION, "H1 10-EMA colour mark on a closed bar"),
    _spec("h1_blue_after_red", CLAIM_ANNOTATION, "H1 blue-reclaim mark on a closed bar"),
    _spec("h1_green_to_yellow", CLAIM_ANNOTATION, "H1 green-to-yellow mark on a closed bar"),
    # --- information: the state of the tape ---------------------------------
    # Audit D7: `regime_pause_rw` all-time n=934, mean -1.82R. It never claimed
    # an entry; that -1.82 is the cost of measuring an observation as a trade.
    _spec("regime_pause_rw", CLAIM_INFORMATION, "relative-weakness pause observation"),
    _spec("regime_pause_rs", CLAIM_INFORMATION, "relative-strength pause observation"),
    _spec("regime_pause", CLAIM_INFORMATION, "regime-pause observation"),
)

#: How a compound family is spelled. `_make_bounce_event_id` builds the family
#: as the sorted level names joined by this, so splitting on it recovers the
#: exact parts - CONSTRUCTION, not similarity, which is why it is allowed here
#: while prefix matching is not.
COMPOUND_SEPARATOR = "-"

_BY_FAMILY: dict[str, FamilySpec] = {spec.family: spec for spec in FAMILY_SPECS}


def _normalize(family: str | None) -> str:
    return str(family or "").strip().lower()


def _unconfigured(key: str, why: str) -> FamilySpec:
    return FamilySpec(
        family=key, claim_kind=CLAIM_UNCONFIGURED, why=why, has_signal_bar=False
    )


def spec_for(family: str | None) -> FamilySpec:
    """The family's spec, or an `unconfigured` one naming itself.

    Whole names only. A family called `lrsi_cross_80` is not read as an LRSI
    entry claim because it looks like one - prefix matching would silently
    enrol every future family into whatever its neighbour claimed, and this
    store already holds the trap for it: `h1_ema_15` is a bounce LEVEL while
    `h1_ema10_bounce` is a colour ANNOTATION, and they share a prefix.

    A COMPOUND family - several levels firing on one bar, joined by
    ``COMPOUND_SEPARATOR`` - is decided by its parts, because that is how the
    id was built. It is an entry claim only when every part is one and they
    agree; a mixture, or a single unknown part, is `unconfigured`. A row whose
    parts disagree about what it claims has not been classified, whatever its
    pieces say individually.
    """
    key = _normalize(family)
    if not key:
        return _unconfigured(key, "the row carries no family at all")
    found = _BY_FAMILY.get(key)
    if found is not None:
        return found
    parts = [part for part in key.split(COMPOUND_SEPARATOR) if part]
    if len(parts) > 1:
        specs = [_BY_FAMILY.get(part) for part in parts]
        if all(spec is not None for spec in specs):
            kinds = {spec.claim_kind for spec in specs}
            if len(kinds) == 1:
                kind = kinds.pop()
                return FamilySpec(
                    family=key,
                    claim_kind=kind,
                    why=(
                        "a compound of "
                        + str(len(parts))
                        + " level(s) that all declare "
                        + kind
                        + ": "
                        + ", ".join(parts)
                    ),
                )
            return _unconfigured(
                key,
                "a compound whose parts disagree about what they claim ("
                + ", ".join(sorted(kinds))
                + "), so what the row as a whole claims is UNMEASURED",
            )
        unknown = [part for part, spec in zip(parts, specs) if spec is None]
        return _unconfigured(
            key,
            "a compound containing level(s) no entry declares: " + ", ".join(unknown),
        )
    return _unconfigured(
        key,
        "no entry in outcome_claim_kinds_v1 declares what this family claims, "
        "so what it means is UNMEASURED - it is not therefore a trade",
    )


def claim_kind(family: str | None) -> str:
    return spec_for(family).claim_kind


def is_trade_bearing(family: str | None) -> bool:
    """May rows of this family carry an R, an exit policy, or a path?

    The one question every writer and every rollup has to ask before it
    averages anything.
    """
    return claim_kind(family) in TRADE_BEARING_KINDS


def coverage(families: Iterable[str | None]) -> dict[str, Any]:
    """Count what each claim kind covers, and name the unconfigured LOUDLY.

    The names are returned, not just the count, because "3 unconfigured
    families" is a number nobody can act on and "3 unconfigured families:
    foo, bar, baz" is a to-do list.
    """
    counts = {kind: 0 for kind in CLAIM_KINDS}
    unconfigured: set[str] = set()
    seen: set[str] = set()
    total = 0
    for family in families:
        total += 1
        key = _normalize(family)
        seen.add(key)
        kind = claim_kind(key)
        counts[kind] += 1
        if kind == CLAIM_UNCONFIGURED:
            unconfigured.add(key)
    return {
        "registry": REGISTRY_NAME,
        "rows": total,
        "distinct_families": len(seen),
        "counts": counts,
        "unconfigured_families": sorted(unconfigured),
        "note": format_coverage(counts, sorted(unconfigured)),
    }


def format_coverage(counts: Mapping[str, int], unconfigured: Iterable[str]) -> str:
    """One line a health tile or a report header can print verbatim."""
    names = [name for name in unconfigured if name]
    parts = [f"{kind}={counts.get(kind, 0)}" for kind in CLAIM_KINDS]
    line = f"{REGISTRY_NAME}: " + ", ".join(parts)
    if names:
        shown = ", ".join(names[:8])
        more = f" (+{len(names) - 8} more)" if len(names) > 8 else ""
        line += (
            f" - UNCONFIGURED families carry no declared meaning and are excluded "
            f"from every trade statistic: {shown}{more}"
        )
    return line


def registered_families() -> tuple[str, ...]:
    return tuple(sorted(_BY_FAMILY))


# ---------------------------------------------------------------------------
# What did this trade's finalization actually MEASURE? - packet M2
# ---------------------------------------------------------------------------
# `claim_kind` above answers "may this family be averaged as a trade at all".
# This half answers the next question down: for a row that IS a trade, did the
# finalizer measure anything, and under what.
#
# The after-close sweep (`sweep_pending_bounce_outcomes`) "needs no bars and no
# IB": it finalizes from what each trade already measured, so it holds no bars
# through the close and every row it writes was labelled `unresolved` by
# construction - INCLUDING a trade whose bars were measured earlier and whose
# stop is a recorded fact. Measured over the twenty sessions to 2026-09-05:
# 4,251 `unresolved` rows, of which **3,607 carry a measured basis** (2,054
# `last_measured_bar`, 1,553 `stop_hit_from_prior_measurement`) and only 644
# measured nothing at all. `setup_scoreboard.exit_policy_r` already reads those
# 3,607 under the policy that measured them; only the LABEL said otherwise.
#
# So: **`unresolved` means UNMEASURED**, and a swept trade that measured its
# bars is `measured_swept`. History is read correctly here rather than
# rewritten - an evidence row is never rewritten (plan.md sec 5).

#: The status column's values, spelled once. `swept_measured` is ADDITIVE: the
#: CSV header is unchanged and nothing in the tree enumerates this domain.
STATUS_OPEN = "open"
STATUS_EOD_COMPLETE = "eod_complete"
STATUS_SWEPT_MEASURED = "swept_measured"
STATUS_UNRESOLVED = "unresolved"

#: `context_json.finalization.basis` - the MECHANISM the numbers came from.
BASIS_MEASURED = "measured"
BASIS_LAST_MEASURED_BAR = "last_measured_bar"
BASIS_STOP_HIT_FROM_PRIOR = "stop_hit_from_prior_measurement"
BASIS_UNRESOLVED = "unresolved"

#: A basis that says bars WERE measured, just not through the close.
SWEPT_MEASURED_BASES = frozenset({BASIS_LAST_MEASURED_BAR, BASIS_STOP_HIT_FROM_PRIOR})

TERMINAL_MEASURED_EOD = "measured_eod"
TERMINAL_MEASURED_SWEPT = "measured_swept"
TERMINAL_UNMEASURED = "unmeasured"
TERMINAL_OPEN = "open"

#: Ordered for a report: the measured pair, then what was not measured, then
#: what is not finished. A reader may iterate this and never invent a fifth.
TERMINAL_KINDS = (
    TERMINAL_MEASURED_EOD,
    TERMINAL_MEASURED_SWEPT,
    TERMINAL_UNMEASURED,
    TERMINAL_OPEN,
)

#: Kinds whose R may be averaged - under the policy that measured them
#: (`setup_scoreboard.exit_policy_r`), never blended into `eod_hold`.
MEASURED_TERMINAL_KINDS = frozenset({TERMINAL_MEASURED_EOD, TERMINAL_MEASURED_SWEPT})

#: What a NEW row's status means on its own. A row the current writer emits
#: needs no basis lookup: `unresolved` is written only when nothing was
#: measured. The historical `unresolved`-with-a-basis case is resolved from the
#: basis in :func:`terminal_kind`.
TERMINAL_KIND_BY_STATUS = {
    STATUS_EOD_COMPLETE: TERMINAL_MEASURED_EOD,
    STATUS_SWEPT_MEASURED: TERMINAL_MEASURED_SWEPT,
    STATUS_UNRESOLVED: TERMINAL_UNMEASURED,
    STATUS_OPEN: TERMINAL_OPEN,
}


def status_for_finalization_basis(basis: str | None) -> str:
    """The status a finalizing row carries, from the basis it was arrived at.

    The writer's one decision, spelled here so the reader and the writer cannot
    drift. `measured` (bars through the close) is `eod_complete`; a basis that
    used a prior measurement is `swept_measured`; anything else is `unresolved`,
    which now means UNMEASURED and nothing else.

    **A blank basis does not reach this from the live writer.**
    `_append_bounce_outcome_row` sets `finalization_basis = "measured"` the
    moment `finalize_eod` is true and only narrows it afterwards, so a final row
    whose risk is non-positive - the one case that skips both narrowing branches
    - still writes `eod_complete` today, exactly as it did before this packet.
    That is pre-existing behaviour and M2 deliberately does not change it; the
    blank-basis mapping here is the safe default for a caller that has one, not
    a description of what the writer produces.
    """
    key = str(basis or "").strip().lower()
    if key == BASIS_MEASURED:
        return STATUS_EOD_COMPLETE
    if key in SWEPT_MEASURED_BASES:
        return STATUS_SWEPT_MEASURED
    return STATUS_UNRESOLVED


def _finalization(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """`context_json.finalization`, from a CSV string or an already-parsed dict."""
    import json

    payload: Any = row.get("context_json")
    if payload is None:
        payload = row.get("context")
    if isinstance(payload, str):
        text = payload.strip()
        # The cheap substring test first: on the live file this function would
        # otherwise parse ~325,000 JSON blobs to find a key most rows lack.
        if not text or "finalization" not in text:
            return {}
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return {}
    if not isinstance(payload, Mapping):
        return {}
    found = payload.get("finalization")
    return found if isinstance(found, Mapping) else {}


#: A `status` cell that carries no status. A pandas frame with no `status`
#: column - `setup_scoreboard`'s, for one - yields NaN per row, and
#: `str(float("nan"))` is `"nan"`, which must never be read as a status the
#: registry has never seen.
_BLANK_STATUS = frozenset({"", "nan", "none", "null", "<na>"})


def _measured_number(value: Any) -> float | None:
    """A finite number, or None. `inf` is a division that should not have run."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _shows_a_measured_close(row: Mapping[str, Any]) -> bool:
    """Did this row record a close at all? `close_r` or `eod_close`, either."""
    return any(
        _measured_number(row.get(name)) is not None
        for name in ("close_r", "eod_close")
    )


def terminal_kind(row: Mapping[str, Any] | None) -> str:
    """What this outcome row's finalization MEASURED - one of `TERMINAL_KINDS`.

    * ``measured_eod`` - bars through the close were in hand; `close_r` is the
      `eod_hold` number.
    * ``measured_swept`` - the after-close sweep settled it from bars measured
      earlier. It counts under the policy that measured it
      (`setup_scoreboard.exit_policy_r`: `stop_exit` or `last_measured`) and
      **never under `eod_hold`**, which it has no number for.
    * ``unmeasured`` - nothing was ever measured after the entry. Nothing about
      this row may be averaged.
    * ``open`` - the row does not claim to be a final: no terminal statement.

    Read the STATUS first and the basis only when the status cannot answer: the
    historical `unresolved` label is ambiguous and the current writer's is not.
    A final row carrying a status this registry has never seen is `unmeasured` -
    missing data is uncertainty, never confirmation (plan.md sec 5). Measured
    whole-file 2026-09-05, those are the **749** pre-R10.A schema-1 finals
    (`stop_seen` 397, `target2_seen` 166, `complete` 129,
    `stop_and_target2_seen` 57); see the M2 checkpoint entry, because `complete`
    at least WAS a measured outcome and an all-history report will understate
    `measured` by up to 749 until those four are classified.

    **A row that claims to be `final` and carries neither a status nor a basis
    is decided by whether it recorded a close** (reviewer advisory, 2026-09-05).
    13,703 of the 14,863 `eod_complete` finals on the live file predate R10.A's
    `finalization` block, and they arrive here status-less through
    `setup_scoreboard`'s frames, which never load the `status` column. Reading
    them as `open` would call a finished, measured trade unfinished - so a
    numeric `close_r` or `eod_close` makes it `measured_eod` (all 13,703 have
    one; **zero** do not), and the absence of both makes it `unmeasured`. It is
    never `open`: the row said it was final. A row that never claims `final`
    still is.

    Note this reports what the FINALIZATION claimed, not whether the number is
    usable: `setup_scoreboard.unsettled_close_mask` separately excludes the old
    `close_r == 0 and eod_close == entry` sentinel from every `eod_hold` mean.
    """
    if not isinstance(row, Mapping):
        return TERMINAL_OPEN
    event_type = str(row.get("event_type") or "").strip().lower()
    if event_type and event_type != "final":
        return TERMINAL_OPEN
    status = str(row.get("status") or "").strip().lower()
    if status in _BLANK_STATUS:
        status = ""
    if status in (STATUS_EOD_COMPLETE, STATUS_SWEPT_MEASURED):
        return TERMINAL_KIND_BY_STATUS[status]
    if status == STATUS_OPEN:
        return TERMINAL_OPEN
    if status and status != STATUS_UNRESOLVED:
        return TERMINAL_UNMEASURED
    basis = str(_finalization(row).get("basis") or "").strip().lower()
    if basis:
        if basis == BASIS_MEASURED:
            return TERMINAL_MEASURED_EOD
        if basis in SWEPT_MEASURED_BASES:
            return TERMINAL_MEASURED_SWEPT
        return TERMINAL_UNMEASURED
    if status == STATUS_UNRESOLVED:
        return TERMINAL_UNMEASURED
    if event_type == "final":
        return TERMINAL_MEASURED_EOD if _shows_a_measured_close(row) else TERMINAL_UNMEASURED
    return TERMINAL_OPEN


def terminal_coverage(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count the four kinds per EVENT, not per row.

    The outcome log carries a `registered` row, a stream of `update`s, the
    milestones and at most one `final` for each event; counting rows would
    weight a long-running trade more heavily than a short one and would count
    every event as `open` at least once. An event with a final row takes that
    row's kind; an event without one is `open`.
    """
    kinds: dict[str, str] = {}
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            continue
        kind = terminal_kind(row)
        if kind != TERMINAL_OPEN or event_id not in kinds:
            kinds[event_id] = kind
    counts = {kind: 0 for kind in TERMINAL_KINDS}
    for kind in kinds.values():
        counts[kind] += 1
    counts["measured"] = counts[TERMINAL_MEASURED_EOD] + counts[TERMINAL_MEASURED_SWEPT]
    counts["events"] = len(kinds)
    return counts


def format_terminal_coverage(
    counts: Mapping[str, int] | None, window_text: str = "over the window"
) -> str:
    """One sentence a status line, a digest or a report prints verbatim.

    ``Outcomes: measured 7,427 (eod 3,820 / swept 3,607), unmeasured 644,
    open 90 over the window.`` Blank when there is nothing to say, so a caller
    never appends an empty clause.
    """
    if not counts or not counts.get("events"):
        return ""
    eod = int(counts.get(TERMINAL_MEASURED_EOD, 0))
    swept = int(counts.get(TERMINAL_MEASURED_SWEPT, 0))
    tail = f" {window_text}" if window_text else ""
    return (
        f"Outcomes: measured {eod + swept:,} (eod {eod:,} / swept {swept:,}), "
        f"unmeasured {int(counts.get(TERMINAL_UNMEASURED, 0)):,}, "
        f"open {int(counts.get(TERMINAL_OPEN, 0)):,}{tail}."
    )
