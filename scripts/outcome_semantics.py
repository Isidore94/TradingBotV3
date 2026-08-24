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
