"""Deterministic tags derived from a trade's own shape, with no bot context.

``journal_analytics.AutoTagger`` answers "which of my setups was this?" by
matching a trade against the scanner's own output files. That is the tag the
trader actually wants, and it is the only one worth having *when it exists*.
It cannot exist for imported history: the scan CSVs hold the current
lookback, not last February, so every trade older than those files scores no
candidates at all and lands in the journal untagged. A year pulled from a
broker statement would arrive as one undifferentiated block.

This module is the floor under that. Every tag here is a **fact about the
trade**, read off its own timestamps, legs and instrument -- no files, no
network, no scanner import (`AutoTagger`'s no-scanner-code boundary is
deliberate and is kept). It answers "what kind of trade was this?", never
"which setup was it?", so the two taggers stack rather than compete.

Three rules make these tags safe to average later:

1. **Never derive a tag from the outcome.** No win/loss, no R, no
   "good_trade". A tag that encodes the result makes every per-tag statistic
   circular -- the ``winners`` bucket would post a 100% win rate and mean
   nothing. The outcome is the thing being explained; it may never also be
   the explanation.
2. **Unmeasurable yields no tag.** A missing or unparseable timestamp emits
   nothing rather than a default bucket, because an invented "midday" is
   indistinguishable from a measured one once it is in the store.
3. **Timestamps normalise by ATTACHING market-local to a naive value**, never
   by stripping the zone off an aware one -- the same seam rule the adoption
   gate uses. A Pacific desk writing naive local times is exactly how a
   session bucket silently shifts three hours.

The session-bucket names on their shared domain are identical to
``bounce_bot_lib.learning.time_bucket_for`` (``opening_drive``,
``late_morning``, ``midday``, ``afternoon``, ``closing_window``) so the
journal and the review-learning loop speak one vocabulary. They are restated
here rather than imported: the journal does not import scanner code, and that
boundary is worth more than the four saved lines. ``premarket`` and
``after_hours`` are added because a broker fills extended-hours orders and
that module never had to name them -- its own ``minutes < 60`` branch would
call an 08:00 fill an opening drive.

One known imprecision, stated rather than hidden: buckets are elapsed minutes
from the 09:30 ET open, which is fixed, but an early-close session ends at
13:00. On those few days a 13:30 fill cannot exist, and if one does it is
labelled ``afternoon``. The alternative -- threading the early-close calendar
through every tag -- buys accuracy on ~9 sessions a year at the cost of a
dependency this module deliberately does not have.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, time, timedelta
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")

#: Regular-hours open. Buckets are elapsed minutes from here.
SESSION_OPEN = time(9, 30)

#: Minutes from the open, exclusive upper bounds. The middle five names and
#: their cutoffs match ``bounce_bot_lib.learning.time_bucket_for`` exactly.
SESSION_BUCKETS: tuple[tuple[float, str], ...] = (
    (0.0, "premarket"),
    (60.0, "opening_drive"),
    (150.0, "late_morning"),
    (270.0, "midday"),
    (360.0, "afternoon"),
    (390.0, "closing_window"),
)
#: Anything at or past the last bound above.
AFTER_HOURS_BUCKET = "after_hours"

#: A same-session round trip shorter than this is a scalp.
SCALP_MAX_MINUTES = 5.0

#: Sessions held, exclusive upper bounds, for a trade held at least one night.
#: A trade that opened and closed inside one session never reaches this table
#: -- it is a ``scalp`` or a ``day_trade``, decided by elapsed minutes.
HOLD_BUCKETS: tuple[tuple[int, str], ...] = (
    (2, "overnight"),
    (10, "swing"),
)

#: The two same-session names, kept beside the table above so
#: ``describe_vocabulary`` cannot list one set and the code emit another.
SAME_SESSION_BUCKETS: tuple[str, ...] = ("scalp", "day_trade")
#: Anything at or past the last bound above.
POSITION_BUCKET = "position"

#: Sessions counted before the walk gives up. A hold longer than this is a
#: ``position`` whatever the exact count, so there is nothing to gain by
#: walking a decade of calendar days one at a time.
MAX_SESSION_WALK_DAYS = 400

#: Security types that are just "a share of something" and say nothing worth
#: tagging -- the symbol column already carries more.
_UNREMARKABLE_SECURITY_TYPES = frozenset({"", "STOCK", "STK", "EQUITY", "COMMON"})

#: Ordering of kinds inside a trade's tag list. Stable so a stored summary
#: does not reshuffle between rebuilds.
KIND_ORDER: tuple[str, ...] = ("hold", "entry_time", "execution", "instrument")


class ShapeTag(NamedTuple):
    """One derived tag plus why it was derived."""

    tag: str
    kind: str
    rationale: str


def _coerce_datetime(value: Any) -> datetime | None:
    """Parse a stored timestamp into an aware market-local datetime.

    Naive values get market-local ATTACHED; aware values are converted. Both
    directions end in market-local, so two timestamps are always comparable.
    """
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=MARKET_TZ)
    return moment.astimezone(MARKET_TZ)


def session_bucket(when: Any) -> str | None:
    """Name the part of the session ``when`` falls in, or ``None``.

    ``None`` for anything unparseable -- rule 2. Weekends and holidays are
    bucketed by clock time like any other day; a fill cannot happen on one, so
    the case does not arise from broker data, and refusing to name it would
    only lose the tag on a manually entered row.
    """
    moment = _coerce_datetime(when)
    if moment is None:
        return None
    open_moment = datetime.combine(moment.date(), SESSION_OPEN, tzinfo=MARKET_TZ)
    minutes = (moment - open_moment).total_seconds() / 60.0
    for bound, bucket in SESSION_BUCKETS:
        if minutes < bound:
            return bucket
    return AFTER_HOURS_BUCKET


def _sessions_held(opened: datetime, closed: datetime) -> int | None:
    """Count NYSE sessions strictly after the open date, up to the close date.

    ``0`` means both fills landed in one session. ``None`` means the calendar
    could not answer -- it is authoritative only over a fixed span and raises
    outside it rather than extrapolating, and a guessed hold is worse than an
    absent one.
    """
    start = opened.date()
    end = closed.date()
    if end < start:
        return None
    if end == start:
        return 0
    try:
        from market_calendar import is_session
    except Exception:
        return None
    held = 0
    cursor = start
    for _ in range(MAX_SESSION_WALK_DAYS):
        cursor += timedelta(days=1)
        if cursor > end:
            return held
        try:
            if is_session(cursor):
                held += 1
        except Exception:
            return None
        if held >= HOLD_BUCKETS[-1][0]:
            # Already past the last named bucket; the exact count changes
            # nothing and the walk can stop.
            return held
    return held


def hold_bucket(opened_at: Any, closed_at: Any) -> tuple[str, str] | None:
    """``(tag, rationale)`` for how long the trade was held, or ``None``.

    ``None`` for an open trade or an unreadable pair of timestamps: a hold is
    not measurable until the position is flat, and a trade that is still on
    has no answer yet rather than a short one.
    """
    opened = _coerce_datetime(opened_at)
    closed = _coerce_datetime(closed_at)
    if opened is None or closed is None or closed < opened:
        return None
    held = _sessions_held(opened, closed)
    if held is None:
        return None
    if held == 0:
        minutes = (closed - opened).total_seconds() / 60.0
        if minutes < SCALP_MAX_MINUTES:
            return "scalp", f"closed {minutes:.1f} min after entry, same session"
        return "day_trade", f"opened and closed in one session ({minutes:.0f} min)"
    for bound, bucket in HOLD_BUCKETS:
        if held < bound:
            return bucket, f"held {held} session(s)"
    return POSITION_BUCKET, f"held {held}+ session(s)"


def execution_shape(legs: Sequence[Mapping[str, Any]] | None) -> tuple[str, str] | None:
    """``(tag, rationale)`` for how the position was built and unwound.

    Reads leg ROLES, not quantities: ``rebuild_trades`` already decided which
    fills opened, added to and closed the position, and re-deriving that from
    numbers here would be a second opinion that can disagree with the store.

    A ``SYNTHETIC_OPEN`` leg means the opening fill was never imported, so the
    entry shape is unknown and this returns ``None`` rather than calling a
    reconstructed position a clean single entry.
    """
    if not legs:
        return None
    roles = [str(leg.get("role") or "").upper() for leg in legs]
    if "SYNTHETIC_OPEN" in roles:
        return None
    entries = sum(1 for role in roles if role in {"OPEN", "SCALE"})
    exits = sum(1 for role in roles if role in {"CLOSE", "SYNTHETIC_CLOSE"})
    if entries <= 0:
        return None
    if entries > 1 and exits > 1:
        return "scaled_both", f"{entries} entries, {exits} exits"
    if entries > 1:
        return "scaled_in", f"{entries} entries"
    if exits > 1:
        return "scaled_out", f"{exits} exits"
    if exits == 1:
        return "one_and_done", "single entry, single exit"
    return None


def instrument_tag(security_type: Any) -> str | None:
    """Lower-cased security type, or ``None`` when it says nothing useful."""
    text = str(security_type or "").strip().upper()
    if text in _UNREMARKABLE_SECURITY_TYPES:
        return None
    return text.lower()


def shape_tags(
    trade: Mapping[str, Any],
    *,
    legs: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[ShapeTag, ...]:
    """Every fact-derived tag for one trade, in ``KIND_ORDER``.

    ``legs`` is optional: without it the execution-shape tag is simply absent,
    which keeps the function usable from a caller that only has the trade row.
    """
    found: dict[str, ShapeTag] = {}

    hold = hold_bucket(trade.get("opened_at"), trade.get("closed_at"))
    if hold is not None:
        found["hold"] = ShapeTag(hold[0], "hold", hold[1])

    bucket = session_bucket(trade.get("opened_at"))
    if bucket is not None:
        found["entry_time"] = ShapeTag(bucket, "entry_time", "entry timestamp, market-local")

    shape = execution_shape(legs)
    if shape is not None:
        found["execution"] = ShapeTag(shape[0], "execution", shape[1])

    instrument = instrument_tag(trade.get("security_type"))
    if instrument is not None:
        found["instrument"] = ShapeTag(instrument, "instrument", "broker security type")

    return tuple(found[kind] for kind in KIND_ORDER if kind in found)


def describe_vocabulary() -> dict[str, tuple[str, ...]]:
    """Every tag this module can ever emit, by kind.

    The tag filter and the rename tool both need to tell a derived tag from
    one the trader typed, and a hardcoded second copy of these names would
    drift the first time a bucket is renamed.
    """
    session_names = tuple(name for _, name in SESSION_BUCKETS) + (AFTER_HOURS_BUCKET,)
    hold_names = SAME_SESSION_BUCKETS + tuple(name for _, name in HOLD_BUCKETS) + (POSITION_BUCKET,)
    return {
        "hold": hold_names,
        "entry_time": session_names,
        "execution": ("one_and_done", "scaled_in", "scaled_out", "scaled_both"),
    }


def is_shape_tag(tag: Any) -> bool:
    """True when ``tag`` is one this module emits.

    Instrument tags are excluded on purpose: they come from the broker's own
    security-type vocabulary, which is open-ended, so membership cannot be
    decided from a fixed list.
    """
    text = str(tag or "").strip().lower()
    if not text:
        return False
    return any(text in names for names in describe_vocabulary().values())


__all__ = [
    "AFTER_HOURS_BUCKET",
    "HOLD_BUCKETS",
    "KIND_ORDER",
    "MARKET_TZ",
    "POSITION_BUCKET",
    "SAME_SESSION_BUCKETS",
    "SCALP_MAX_MINUTES",
    "SESSION_BUCKETS",
    "SESSION_OPEN",
    "ShapeTag",
    "describe_vocabulary",
    "execution_shape",
    "hold_bucket",
    "instrument_tag",
    "is_shape_tag",
    "session_bucket",
    "shape_tags",
]
