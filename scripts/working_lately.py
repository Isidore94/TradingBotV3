"""ONE declared leader for "what is working lately" - packet ST2.3, 2026-09-06.

The Setup Tracker had two answers to the same question on one screen. The recent
types TABLE ranked by the Wilson lower bound on the win rate (R4 B3); the "BEST
PERFORMING RIGHT NOW" BANNER above it picked ``max(avg_closed_r)`` over any row
with three closed setups, across the live AND study namespaces. Reproduced on
`main` at ``84ee24d6`` with two live families:

    fat_but_wide   54-36  (+2.50R)  Wilson lower bound 0.497
    tight_and_hot  24-6   (+0.40R)  Wilson lower bound 0.627

the table listed ``tight_and_hot`` first and the banner crowned ``fat_but_wide``.
Worse, a three-episode STUDY with a big R outranked a ninety-episode live family,
so an unpromoted idea could be presented as the desk's best performer purely for
having high R on almost no evidence.

**What this module is.** The one place that decides which family is leading, and
it either names one or says why it cannot. It is pure: no Qt, no file I/O, no
store reads, no clock of its own - the caller passes the rows and the last
completed session in. Nothing here reaches a detector, a score, a rank that
gates, an alert, a watchlist, Focus, the review queue or ``review_policy.json``;
it decides what a banner SAYS.

**Its inputs are integer counts, never a rate.** ``n_wins`` and ``n_losses`` come
off the evidence rows the same table reads
(``legacy.build_recent_tracker_setup_family_rows``, ST2.1). A rate that was
recency-weighted cannot be turned back into a count, which is the defect ST2.1
fixes; a leader chosen from a reconstructed count would inherit it.

**Four states, because "no answer" is an answer.**

``leader``
    One eligible family, or a top bound clear of the runner-up's by
    ``LEADER_MARGIN_LB``.
``no_clear_leader``
    Two eligible families inside the margin. The reason names BOTH and the gap,
    because printing the winner of a coin flip is how a banner starts lying.
``last_reliable_reading``
    The evidence is stale and the caller passed a ``previous`` verdict. The
    leader and the ``as_of`` are the PREVIOUS one's, unchanged - this state says
    "this is the last thing we could honestly read", not "this is current".
``no_evidence``
    Nothing eligible. The reason names which of the three gates closed: the n
    floor, study-only candidates, or freshness.

**Two declared numbers, both fixed 2026-09-06 BEFORE any forward evaluation.**
See ``LEADER_MARGIN_LB`` and ``LEADER_FRESHNESS_SESSIONS`` below. Neither was
tuned to make a winner appear, and neither may be moved to produce one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from evidence_stats import MIN_REPORTABLE_N
from swing_headline import wilson_lower_bound

#: How far the top Wilson lower bound must clear the runner-up's before the desk
#: calls one family the leader.
#:
#: **Declared 2026-09-06, before any forward evaluation, and NOT tuned to make a
#: winner appear.** Five points of lower bound is roughly the gap between "these
#: two families are different" and "these two families are the same family read
#: twice" at the n this table carries (30-90 episodes). It is deliberately a
#: margin on the BOUND rather than on the rate: two rates can differ by fifteen
#: points and still be one sample apart when one of them is thin, and the bound
#: is the number that already knows that.
LEADER_MARGIN_LB = 0.05

#: How many exchange sessions of staleness the leader's own evidence may carry
#: before it stops being a reading of NOW.
#:
#: **Declared 2026-09-06, before any forward evaluation, and NOT tuned.** Two
#: sessions is one weekend plus a holiday: the tracker writes at the close slot,
#: so a reading older than that means the tracker did not write, which is news
#: about the desk and not about the family. Counted on the exchange calendar
#: (``market_calendar``), never in calendar days.
#:
#: **What is measured is the ENTRY session**, not the exit. The tracker's family
#: rows carry no exit date, so ``latest_measured_session`` is the newest
#: scan_date among the episodes that produced a readable R. That is the
#: conservative reading - a family whose newest ENTRY is old cannot have a newer
#: measured close than its own scan - and every surface that shows a verdict
#: says so in words, because "fresh" without its clock is not a fact.
LEADER_FRESHNESS_SESSIONS = 2

#: How the freshness rule reads in a sentence, for every surface that shows one.
FRESHNESS_SENTENCE = (
    f"fresh = an entry inside {LEADER_FRESHNESS_SESSIONS} exchange sessions of "
    f"the last completed one"
)

#: The states ``select_leader`` can return.
LEADER_STATES = ("leader", "no_clear_leader", "last_reliable_reading", "no_evidence")


@dataclass(frozen=True)
class LeaderVerdict:
    """What the desk is willing to say about which family is working.

    ``leader`` and ``runner_up`` are the caller's own ROW MAPPINGS, handed back
    untouched, so a renderer reads ``verdict.leader["setup_family"]`` and never
    has to re-derive anything this module already decided.
    """

    state: str
    leader: Mapping[str, Any] | None
    runner_up: Mapping[str, Any] | None
    reason: str
    as_of: str
    policy_line: str
    coverage: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# small coercions - CSV rows arrive as strings
# ---------------------------------------------------------------------------


def _int(value: Any) -> int | None:
    """An integer, or None when the cell is absent/blank/unreadable.

    None is load-bearing: "this export has no count column" and "this family
    counted zero" are different facts and only the second one is a statistic.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def counted_pair(row: Mapping[str, Any]) -> tuple[int, int] | None:
    """``(wins, losses)`` from a row's INTEGER columns, or None.

    Never reconstructed from a rate. A row whose counts were not exported has
    no pair here and is not eligible for anything - the honest reading, and the
    one the panel prints as "counts not exported yet".
    """
    wins = _int(row.get("n_wins"))
    losses = _int(row.get("n_losses"))
    if wins is None or losses is None:
        return None
    return max(0, wins), max(0, losses)


def _bound(row: Mapping[str, Any]) -> float | None:
    pair = counted_pair(row)
    if pair is None:
        return None
    wins, losses = pair
    return wilson_lower_bound(wins, wins + losses)


def _sessions_stale(row: Mapping[str, Any], last_completed_session: date) -> int | None:
    """How many exchange sessions the row's evidence is behind, or None.

    None means the row does not say - a missing or unreadable
    ``latest_measured_session``. Uncertainty never confirms freshness, so the
    caller treats None as NOT fresh; the reason says the row carried no session
    rather than pretending it was old.
    """
    measured = _as_date(row.get("latest_measured_session"))
    if measured is None:
        return None
    import market_calendar

    try:
        return market_calendar.trading_days_between(measured, last_completed_session)
    except Exception:
        # Outside the validated calendar range. Fail CLOSED: not fresh.
        return None


def _describe(row: Mapping[str, Any]) -> str:
    side = _text(row.get("side")) or "?"
    family = _text(row.get("setup_family")) or "?"
    return f"{side} {family}"


def _policy_line(row: Mapping[str, Any], *, kind: str, last_completed_session: date) -> str:
    """Side, outcome meaning, measurement horizon, window and coverage.

    The trader's requirement 4, in one sentence: a leader that does not say what
    it measured, over what, and on how much, is a headline and not evidence.
    """
    pair = counted_pair(row) or (0, 0)
    wins, losses = pair
    bits = [_describe(row)]
    outcome_kind = _text(row.get("outcome_kind"))
    if outcome_kind:
        bits.append(f"outcome {outcome_kind}")
    horizon = _text(row.get("horizon_basis"))
    if horizon:
        bits.append(f"horizon {horizon}")
    measured = _text(row.get("latest_measured_session"))
    bits.append(
        f"window through {measured or 'an unstated session'}"
        f" (read against {last_completed_session.isoformat()}; {FRESHNESS_SENTENCE})"
    )
    coverage_bits = [f"{wins + losses} graded ({wins}W/{losses}L)"]
    for label, key in (
        ("flat", "n_flats"),
        ("unmeasured", "n_unmeasured"),
        ("pending", "n_pending"),
    ):
        value = _int(row.get(key))
        if value is not None:
            coverage_bits.append(f"{value} {label}")
    for label, key in (("symbol", "n_symbols"), ("entry session", "n_entry_sessions")):
        value = _int(row.get(key))
        if value is not None:
            coverage_bits.append(f"{value} {label}{'' if value == 1 else 's'}")
    bits.append(", ".join(coverage_bits))
    bits.append(f"read as {kind}")
    return "; ".join(bits)


# ---------------------------------------------------------------------------
# the one decision
# ---------------------------------------------------------------------------


def select_leader(
    rows: Iterable[Mapping[str, Any]],
    *,
    kind: str,
    last_completed_session: date,
    previous: LeaderVerdict | None = None,
    min_n: int = MIN_REPORTABLE_N,
) -> LeaderVerdict:
    """The one eligible leader, or the reason there is not one.

    ``rows`` are evidence-table rows carrying ``namespace``, ``side``,
    ``setup_family``, ``outcome_kind``, ``latest_measured_session`` and the
    INTEGER ``n_wins`` / ``n_losses``.

    ``min_n`` is an argument rather than a constant so the two-session
    discovery block can pass its own, smaller floor without either block
    inventing a second statistics contract; the default is the desk's one floor,
    ``evidence_stats.MIN_REPORTABLE_N``. Whatever it is, it is NAMED in the
    reason, so a reader is never guessing which floor a verdict was measured
    against.

    A ``namespace == "study"`` row is never eligible for any floor. A study is
    an unpromoted idea; letting one lead because its R is big is precisely the
    confusion between "interesting" and "working" that the promotion ladder
    (plan.md sec 7) exists to prevent. Studies are COUNTED in
    ``coverage["studies_excluded"]`` so the exclusion is visible rather than
    silent.
    """
    all_rows: Sequence[Mapping[str, Any]] = [row for row in rows if isinstance(row, Mapping)]
    floor = max(1, int(min_n))

    studies: list[Mapping[str, Any]] = []
    live: list[Mapping[str, Any]] = []
    other: list[Mapping[str, Any]] = []
    for row in all_rows:
        namespace = _text(row.get("namespace")).lower()
        if namespace == "study":
            studies.append(row)
        elif namespace == "live":
            live.append(row)
        else:
            other.append(row)

    counted = [row for row in live if counted_pair(row) is not None]
    uncounted = len(live) - len(counted)

    fresh: list[Mapping[str, Any]] = []
    stale: list[Mapping[str, Any]] = []
    undated: list[Mapping[str, Any]] = []
    for row in counted:
        behind = _sessions_stale(row, last_completed_session)
        if behind is None:
            undated.append(row)
        elif behind <= LEADER_FRESHNESS_SESSIONS:
            fresh.append(row)
        else:
            stale.append(row)

    def _order(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        def key(row: Mapping[str, Any]):
            wins, losses = counted_pair(row) or (0, 0)
            bound = _bound(row)
            return (
                bound is None,
                -(bound if bound is not None else 0.0),
                -(wins + losses),
                _text(row.get("setup_family")),
            )

        return sorted(candidates, key=key)

    eligible = _order([row for row in fresh if sum(counted_pair(row) or (0, 0)) >= floor])
    thin = _order([row for row in fresh if sum(counted_pair(row) or (0, 0)) < floor])

    coverage: dict[str, Any] = {
        "kind": kind,
        "min_n": floor,
        "freshness_sessions": LEADER_FRESHNESS_SESSIONS,
        "margin_lb": LEADER_MARGIN_LB,
        "last_completed_session": last_completed_session.isoformat(),
        "rows_considered": len(all_rows),
        "live_rows": len(live),
        "studies_excluded": len(studies),
        "other_namespace_excluded": len(other),
        "counts_not_exported": uncounted,
        "eligible": len(eligible),
        "under_floor": len(thin),
        "not_fresh": len(stale),
        "no_session_on_row": len(undated),
        # Lead decision 2026-09-06: when nothing is eligible the banner may
        # still show the best LIVE row that was kept out - clearly labelled
        # discovery, never called a leader, and never the word "leader" beside
        # its name. It is None whenever a leader exists, so a renderer cannot
        # print both. `discovery_reason` names WHICH gate kept it out, so the
        # banner's sentence is the one the trader asked for ("the count floor
        # alone is not proof") rather than a generic apology.
        "discovery_leader": None,
        "discovery_reason": "",
    }

    if eligible:
        top = eligible[0]
        runner_up = eligible[1] if len(eligible) > 1 else None
        top_bound = _bound(top) or 0.0
        policy_line = _policy_line(top, kind=kind, last_completed_session=last_completed_session)
        as_of = _text(top.get("latest_measured_session"))
        if runner_up is None:
            return LeaderVerdict(
                state="leader",
                leader=top,
                runner_up=None,
                reason=(
                    f"{_describe(top)} is the only eligible family at the n={floor} "
                    f"floor (bound {top_bound:.3f})."
                ),
                as_of=as_of,
                policy_line=policy_line,
                coverage=coverage,
            )
        runner_bound = _bound(runner_up) or 0.0
        gap = top_bound - runner_bound
        if gap > LEADER_MARGIN_LB:
            return LeaderVerdict(
                state="leader",
                leader=top,
                runner_up=runner_up,
                reason=(
                    f"{_describe(top)} leads {_describe(runner_up)} by "
                    f"{gap:.3f} of Wilson lower bound ({top_bound:.3f} vs "
                    f"{runner_bound:.3f}), clear of the declared "
                    f"{LEADER_MARGIN_LB:.2f} margin."
                ),
                as_of=as_of,
                policy_line=policy_line,
                coverage=coverage,
            )
        return LeaderVerdict(
            state="no_clear_leader",
            leader=None,
            runner_up=runner_up,
            reason=(
                f"{_text(top.get('setup_family'))} ({top_bound:.3f}) and "
                f"{_text(runner_up.get('setup_family'))} ({runner_bound:.3f}) are "
                f"{gap:.3f} apart, inside the declared {LEADER_MARGIN_LB:.2f} "
                f"margin - no clear leader."
            ),
            as_of=as_of,
            policy_line=policy_line,
            coverage=coverage,
        )

    # Nothing eligible. Name the best live row that was kept out - as DISCOVERY,
    # with the gate that kept it out - then say which gate closed, in the order
    # a reader would ask.
    if thin:
        coverage["discovery_reason"] = "floor"
    elif stale:
        coverage["discovery_reason"] = "not_fresh"
    elif undated:
        coverage["discovery_reason"] = "no_session"
    # Fresh-but-thin beats stale beats undated, so the row shown as discovery is
    # always the one whose single missing gate is the one named beside it.
    #
    # **`min_n` binds the stale and undated pools too** (fix round): without it
    # a 2-session family with three samples could be shown as discovery under a
    # floor of six, which is a floor that does not hold. `thin` is exempt by
    # definition - being under the floor IS what put it there, and the sentence
    # beside it says so.
    def _at_floor(candidates):
        return [row for row in candidates if sum(counted_pair(row) or (0, 0)) >= floor]

    discovery_pool = (
        _order(thin) or _order(_at_floor(stale)) or _order(_at_floor(undated))
    )
    coverage["discovery_leader"] = discovery_pool[0] if discovery_pool else None

    if thin:
        wins, losses = counted_pair(thin[0]) or (0, 0)
        return LeaderVerdict(
            state="no_evidence",
            leader=None,
            runner_up=None,
            reason=(
                f"no live family reached the n={floor} floor; the best live row "
                f"is {_describe(thin[0])} on n={wins + losses}, which is "
                f"discovery, not a leader."
            ),
            as_of="",
            policy_line=_policy_line(
                thin[0], kind=kind, last_completed_session=last_completed_session
            ),
            coverage=coverage,
        )

    if stale or undated:
        source = (stale or undated)[0]
        if stale:
            behind = _sessions_stale(source, last_completed_session)
            detail = (
                f"its newest measured session is {behind} exchange session(s) "
                f"behind {last_completed_session.isoformat()}"
            )
        else:
            detail = "the export carries no measured session at all"
        if previous is not None and previous.leader is not None:
            carried = dict(coverage)
            carried["carried_from"] = previous.as_of
            # A carried leader IS the answer, so no discovery row rides beside
            # it - a banner may never print both.
            carried["discovery_leader"] = None
            carried["discovery_reason"] = ""
            return LeaderVerdict(
                state="last_reliable_reading",
                leader=previous.leader,
                runner_up=previous.runner_up,
                reason=(
                    f"the evidence is not fresh ({detail}), so this is the last "
                    f"reliable reading, taken {previous.as_of or 'earlier'}."
                ),
                as_of=previous.as_of,
                policy_line=previous.policy_line,
                coverage=carried,
            )
        return LeaderVerdict(
            state="no_evidence",
            leader=None,
            runner_up=None,
            reason=(
                f"the evidence is not fresh ({detail}) and there is no previous "
                f"verdict to fall back on."
            ),
            as_of="",
            policy_line="",
            coverage=coverage,
        )

    if uncounted:
        return LeaderVerdict(
            state="no_evidence",
            leader=None,
            runner_up=None,
            reason=(
                f"{uncounted} live row(s) carry no exported win/loss counts, and "
                f"a count is never reconstructed from a stored rate."
            ),
            as_of="",
            policy_line="",
            coverage=coverage,
        )

    if studies:
        return LeaderVerdict(
            state="no_evidence",
            leader=None,
            runner_up=None,
            reason=(
                f"every candidate is a study ({len(studies)} excluded); a study "
                f"is an unpromoted idea and never leads, whatever its R."
            ),
            as_of="",
            policy_line="",
            coverage=coverage,
        )

    return LeaderVerdict(
        state="no_evidence",
        leader=None,
        runner_up=None,
        reason="no rows to read.",
        as_of="",
        policy_line="",
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# the two-session discovery block
# ---------------------------------------------------------------------------


def short_term_evidence_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Short-horizon rows in the shape ``select_leader`` reads.

    ``legacy.build_tracker_short_horizon_rows`` writes its OWN ``n_wins`` /
    ``n_losses`` / ``n_flats`` / ``n_unmeasured`` and its own
    ``latest_measured_session`` since the ST2 fix round (2026-09-06, the ask
    answered yes), so a row from a current export is read exactly like a recent
    row: counts, freshness, floor.

    **A row from an OLDER export has neither.** For those, and only those, the
    pair is derived from ``win_rate_2d`` x ``samples_2d`` - the one case where
    ``round(rate * n)`` is EXACT, because that rate is a plain unweighted mean
    of ``1.0 if value > 0 else 0.0`` over exactly ``samples_2d`` values, which
    is the legitimate use ``swing_headline.headline_from_rate`` documents and
    not the recency-weighted one ST2.1 removed. Such a row still carries no
    session, so it stays UNDATED and can only ever be shown as discovery.
    """
    from swing_headline import headline_from_rate

    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        adapted = dict(row)
        adapted.setdefault("namespace", "live")
        adapted["latest_measured_session"] = _text(row.get("latest_measured_session"))
        adapted["outcome_kind"] = _text(row.get("outcome_kind")) or "trade_r_close_2d"
        if not _text(adapted.get("horizon_basis")):
            adapted["horizon_basis"] = "2 sessions after entry, close to close"
        if counted_pair(row) is None:
            record = headline_from_rate(
                _text(row.get("setup_family")),
                win_rate=row.get("win_rate_2d"),
                n=row.get("samples_2d"),
            )
            adapted["n_wins"] = record.wins
            adapted["n_losses"] = record.losses
        out.append(adapted)
    return out
