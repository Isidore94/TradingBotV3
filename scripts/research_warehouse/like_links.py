"""Which warehouse occurrence was the trader looking at when they liked it — P10 B2.

Trader, 2026-09-02: *"anytime I like a D1 it should be treated with respect by
the bot in regards to finding out what's good about it, how we can replicate
those searches, and then how we can improve the entries. if I like a stock one
day it may not be for 3-5 days later that the best entry is."*

Nothing in the tree joined a like to a warehouse occurrence — the round-1 audit's
item 6, still unbuilt on 2026-09-02. Without the join, a like is a symbol and a
timestamp: there is no way to ask what the setup looked like, and no way to ask
what a different entry would have done.

**What this is not.** It is not a detector, a score, an alert, or an opinion. It
writes one row per (like, occurrence) pair saying *these two are about the same
thing, and here is how confident that is*. Everything downstream reads the basis
and decides for itself.

**Absence is a first-class fact.** A like with no occurrence in its window is
written with basis `none`. It has to be: a study that silently dropped the
unmatched likes would report on the subset the scanner happened to find, which is
exactly the population whose behaviour differs.

The window is deliberately asymmetric — **one session back, five forward**. Back
one because a like is usually made on a setup that has already triggered, and the
trader may be looking at the previous session's close. Forward five because they
said so: *"if I like a stock one day it may not be for 3-5 days later that the
best entry is."* A wider window buys more matches and buys them at the cost of
matching the wrong thesis, which is the failure that cannot be undone later.

Reads go through `ResearchStore.read_rows` narrowed **Arrow-side by symbol and
by interval**, month/year-keyed per the partitioning, never a materialised list
(BD-74). The whole module is READ-ONLY over the gold datasets and writes only to
its own bronze artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable

#: Sessions before and after the like that an occurrence may trigger in.
WINDOW_SESSIONS_BEFORE = 1
WINDOW_SESSIONS_AFTER = 5

#: How the pair was matched, best first. These are the ONLY three answers, and
#: the caller never has to infer one from a null.
BASIS_EXACT_FAMILY = "exact_family"
BASIS_ANY_FAMILY = "any_family"
BASIS_NONE = "none"
MATCH_BASES = (BASIS_EXACT_FAMILY, BASIS_ANY_FAMILY, BASIS_NONE)

#: The bronze artifact these rows land in. Bronze, and NOT a new gold schema:
#: the slice datasets are FROZEN (plan sec 7.1) and the bronze namespace exists
#: precisely so an additive artifact needs no schema change. The link fields ride
#: in the shared record's JSON `payload`.
ARTIFACT = "like_occurrence_link"

_SIDES = ("LONG", "SHORT")


@dataclass(frozen=True)
class LikeLink:
    """One (like, occurrence) pair, or one like with nothing to pair it to."""

    event_id: str
    symbol: str
    side: str
    like_date: str
    occurrence_id: str
    canonical_setup_id: str
    trigger_at: str
    match_basis: str
    #: How many occurrences were in the window at all, whatever the family. A
    #: row with basis `any_family` and 11 candidates is a much weaker claim than
    #: one with 1, and the number is the only thing that says so.
    candidates_in_window: int

    def as_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "symbol": self.symbol,
            "side": self.side,
            "like_date": self.like_date,
            "occurrence_id": self.occurrence_id,
            "canonical_setup_id": self.canonical_setup_id,
            "trigger_at": self.trigger_at,
            "match_basis": self.match_basis,
            "candidates_in_window": self.candidates_in_window,
        }


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def window_for(like_date: Any) -> tuple[datetime, datetime] | None:
    """The half-open `[start, end)` an occurrence may trigger in.

    CALENDAR days, not trading days, and that is a deliberate simplification
    with a stated cost: a Friday like reaches through the weekend to the
    following Friday rather than the one after. It is stated here rather than
    hidden because the alternative - a trading-day walk - would make the window
    depend on `market_calendar` inside a warehouse read, and the extra reach is
    two idle days rather than two extra sessions of candidates.
    """
    day = _as_date(like_date)
    if day is None:
        return None
    start = datetime.combine(day - timedelta(days=WINDOW_SESSIONS_BEFORE), time.min)
    end = datetime.combine(
        day + timedelta(days=WINDOW_SESSIONS_AFTER + 1), time.min
    )
    return start.replace(tzinfo=timezone.utc), end.replace(tzinfo=timezone.utc)


def _occurrence_rows(store, symbol: str, window: tuple[datetime, datetime]) -> list[dict]:
    """Occurrences for one symbol inside the window, narrowed Arrow-side.

    `setup_occurrence` partitions by YEAR on `event_at`, so a window that spans a
    New Year reads two partitions; `resolve_files(dataset, None)` covers every
    partition and the interval predicate does the narrowing, which is why no
    partition string is computed here.
    """
    start, end = window
    try:
        return store.read_rows(
            "setup_occurrence",
            symbols=[symbol],
            interval_start_range=(start, end),
            # `setup_occurrence` has no `interval_start`; its trigger is what the
            # window is about, and naming it keeps the narrowing Arrow-side.
            time_column="trigger_at",
            columns=[
                "occurrence_id",
                "symbol",
                "canonical_setup_id",
                "side",
                "trigger_at",
                "dependency_cluster_id",
                "status",
            ],
        )
    except TypeError:
        # An older store without the interval predicate would silently read the
        # whole partition, which is the thing BD-74 exists to prevent.
        raise


def link_one_like(
    store,
    like: dict[str, Any],
    *,
    now: datetime | None = None,
) -> LikeLink | None:
    """The best occurrence for one like, or a `none` row. Never a guess.

    Preference order, and it stops at the first that produces anything:

    1. the SAME family the click recorded (`canonical_setup_id` from B1), whose
       trigger is nearest the like;
    2. any family, nearest trigger — the trader liked the CHART, and the family
       the scanner filed it under is the scanner's opinion, not theirs;
    3. nothing, written as `none`.

    "Nearest" is by absolute distance from the like's own session, so a setup
    that triggered the morning of the like beats one that triggered four days
    later. Ties break toward the EARLIER trigger: the trader was looking at
    something that already existed.
    """
    symbol = str(like.get("symbol") or "").strip().upper()
    side = str(like.get("side") or "").strip().upper()
    like_date = str(like.get("session_date") or "").strip()
    event_id = str(like.get("event_id") or "").strip()
    if not symbol or not event_id or side not in _SIDES:
        return None
    window = window_for(like_date)
    if window is None:
        return None

    rows = [
        row
        for row in _occurrence_rows(store, symbol, window)
        if str(row.get("side") or "").strip().upper() == side
    ]
    anchor = _as_date(like_date)

    def _distance(row: dict) -> tuple[int, str]:
        triggered = _as_date(row.get("trigger_at"))
        gap = abs((triggered - anchor).days) if triggered and anchor else 99
        return (gap, str(row.get("trigger_at") or ""))

    wanted = str(like.get("canonical_setup_id") or "").strip()
    exact = [
        row
        for row in rows
        if wanted and str(row.get("canonical_setup_id") or "").strip() == wanted
    ]
    chosen, basis = (
        (min(exact, key=_distance), BASIS_EXACT_FAMILY)
        if exact
        else (min(rows, key=_distance), BASIS_ANY_FAMILY)
        if rows
        else (None, BASIS_NONE)
    )
    return LikeLink(
        event_id=event_id,
        symbol=symbol,
        side=side,
        like_date=like_date,
        occurrence_id=str((chosen or {}).get("occurrence_id") or ""),
        canonical_setup_id=str((chosen or {}).get("canonical_setup_id") or ""),
        trigger_at=str((chosen or {}).get("trigger_at") or ""),
        match_basis=basis,
        candidates_in_window=len(rows),
    )


def link_likes(
    store,
    likes: Iterable[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[LikeLink]:
    """One row per like, in input order. A like that cannot be keyed is skipped.

    Skipped means it had no symbol, no side or no id — it is not a like with no
    match, it is a row this join cannot address, and inventing a `none` for it
    would put a phantom in the denominator.
    """
    links = []
    for like in likes:
        link = link_one_like(store, like, now=now)
        if link is not None:
            links.append(link)
    return links


def like_event_at(like_date: Any, observed_at: datetime) -> datetime:
    """The market fact's own moment: the day the trader liked it.

    R4 fix round 1. This used to be the RUN STAMP, and `partition_ts` with it -
    which put a September like into the October partition the moment a nightly
    pass ran on 1 October. The dataset is month-partitioned, so the caller's
    dedup (which reads the row's own partition, as BD-74 requires) could not see
    the September copy, and every like inside the lookback was republished at
    each month boundary. Reproduced: one like dated 2026-09-25, three passes on
    09-26 / 10-01 / 10-02, and the same `record_hash` landed twice.

    `observed_at` still means what the ERD says it means - when this installation
    received the row - and stays the run stamp. `event_at` is the market fact and
    is now the like's own date, which is both correct and what makes the row's
    partition stable for the life of the like.

    Midnight UTC on the like date. A like carries a session date, not a time; the
    hour is a placement, not a measurement, and the payload keeps the exact
    `like_date` string either way. A date that cannot be read falls back to the
    run stamp - a row filed under the wrong month is better than a row that
    cannot be written, and the fallback is bounded to likes with no readable
    date.
    """
    text = str(like_date or "").strip()[:10]
    if text:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed.replace(tzinfo=observed_at.tzinfo or timezone.utc)
    return observed_at


def link_rows_for_bronze(
    links: Iterable[LikeLink],
    *,
    observed_at: datetime,
    run_id: str = "",
) -> list[dict[str, Any]]:
    """Bronze records, ready for `ResearchStore.write_rows`.

    The record hash is over the payload, so re-running the join on an unchanged
    lake produces the same rows, and the row is partitioned by the LIKE'S OWN
    DATE (:func:`like_event_at`) so a re-run lands in the same partition as the
    original however many months later it happens - which is what makes the
    caller's partition-scoped dedup an exact identity rather than a
    within-the-month one.
    """
    import hashlib

    try:  # package import, as every other module in this package does it
        from . import schemas
    except ImportError:  # pragma: no cover - flat sys.path layout
        import schemas  # type: ignore

    rows = []
    for link in links:
        payload = json.dumps(link.as_payload(), sort_keys=True)
        rows.append(
            {
                "source_artifact": ARTIFACT,
                "source_path": "trader_annotations.jsonl",
                "source_sha256": "",
                "source_offset": 0,
                "record_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "legacy_id": link.event_id,
                "payload": payload,
                "payload_format": schemas.BRONZE_FORMAT_JSON,
                "quality": "OK",
                # THE MARKET FACT, not the run (R4 fix round 1). `observed_at`
                # is still when this installation received the row.
                "event_at": like_event_at(link.like_date, observed_at),
                "observed_at": observed_at,
                "partition_ts": like_event_at(link.like_date, observed_at),
                "capture_mode": "derived",
                "run_id": run_id,
            }
        )
    return rows


def basis_counts(links: Iterable[LikeLink]) -> dict[str, int]:
    """`{basis: n}` over every basis, including the ones with zero.

    Every key always present, because "no exact-family matches tonight" and
    "the exact-family count was never computed" are different facts and a
    missing key cannot tell them apart.
    """
    counts = {basis: 0 for basis in MATCH_BASES}
    for link in links:
        counts[link.match_basis] = counts.get(link.match_basis, 0) + 1
    return counts
