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
#: **``latest_measured_session`` means what its name says, wherever the export
#: can answer.** The 2-session rows date themselves by the BAR the R was read
#: from (``legacy._short_horizon_measured_session``), because entry dating made
#: a family entered eight weeks ago and measured two sessions later read as 58
#: sessions stale on a file written that morning. The recent FAMILY rows still
#: date by the entry session, and only because they have nothing better: those
#: rows carry no exit date (ST4 adds ``representative_exit_date``; until it
#: lands this is the conservative reading, since a family whose newest ENTRY is
#: old cannot have a newer measured close than its own scan). Every surface that
#: shows a verdict states the rule in words - "fresh" without its clock is not a
#: fact - and an export that cannot date a row leaves it EMPTY, which reads as
#: not fresh and never as a guess.
LEADER_FRESHNESS_SESSIONS = 2

#: Which DATE each kind of evidence row is stamped with, per kind.
#:
#: Re-check advisory 1: the two surfaces are dated differently and one sentence
#: for both made whichever one it did not describe say something false. The
#: 2-session rows carry the MEASURED bar's date; the recent family rows carry
#: the ENTRY session, because they have nothing better until ST4's
#: ``representative_exit_date`` lands. Default is `"entry"` - the conservative
#: reading, and the one a new caller is most likely to have.
DATING_BASIS_BY_KIND = {
    "swing": "entry",
    "swing_short_term": "measured",
}


def freshness_sentence(kind: Any = "") -> str:
    """How the freshness rule reads for THIS kind of row, in one clause.

    "fresh" without its clock is not a fact, and a clock the reader thinks is
    the exit when it is the entry is worse than none.
    """
    basis = DATING_BASIS_BY_KIND.get(_text(kind), "entry")
    stamped = "measured" if basis == "measured" else "entry-dated"
    return (
        f"fresh = {stamped} inside {LEADER_FRESHNESS_SESSIONS} exchange sessions "
        f"of the last completed one"
    )

#: The states ``select_leader`` can return.
LEADER_STATES = ("leader", "no_clear_leader", "last_reliable_reading", "no_evidence")

#: The one spelling of each state, for a label that must follow the verdict.
#:
#: Re-review blocker 2: a caller hardcoded "2-session discovery" into its label
#: and then printed it over a real `leader` verdict. A label that does not come
#: from the verdict will eventually contradict it.
_STATE_WORDS = {
    "leader": "leader",
    "no_clear_leader": "no clear leader",
    "last_reliable_reading": "last reliable reading",
    "no_evidence": "no evidence",
}


def verdict_label_suffix(verdict: "LeaderVerdict") -> str:
    """What this verdict IS, in two or three words, for a surface's label."""
    if verdict.state == "no_evidence" and verdict.coverage.get("discovery_leader") is not None:
        return "discovery"
    return _STATE_WORDS.get(verdict.state, "no verdict")


def discovery_basis_phrase(discovery_reason: Any) -> str:
    """How to describe the evidence behind a discovery row, by the gate it failed.

    Re-review advisory 1: a row kept out for being OLD is not thin. It has the
    evidence - it is simply not current - and calling it thin names the wrong
    gate, which is the same class of error as calling a weighted rate a count.
    """
    reason = _text(discovery_reason)
    if reason == "not_fresh":
        return "leading on older evidence"
    if reason == "no_session":
        return "leading on undated evidence"
    return "leading on thin evidence"


def discovery_note(discovery_reason: Any) -> str:
    """The extra sentence a discovery row earns, or "" - beside its phrase.

    Re-check advisory 2. This lived in the panel as a `discovery_note=` argument
    passed on every call and dropped inside for every reason but one, so the
    caller and the condition were in different files and could drift. There is
    exactly one reason that earns a sentence, and it lives next to the phrase
    that names the same gate.
    """
    if _text(discovery_reason) == "no_session":
        return "The export carries no measured session, so its freshness is unstated."
    return ""


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
        f" (read against {last_completed_session.isoformat()};"
        f" {freshness_sentence(kind)})"
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

    # **The floor is judged BEFORE freshness**, because a family that has not
    # been measured enough times is under the floor whatever the clock says, and
    # telling a reader "not fresh" about three samples answers a question they
    # did not ask. Freshness only ever decides between families that already
    # have enough evidence to be compared.
    at_floor = [row for row in counted if sum(counted_pair(row) or (0, 0)) >= floor]
    under_floor = [row for row in counted if sum(counted_pair(row) or (0, 0)) < floor]

    fresh: list[Mapping[str, Any]] = []
    stale: list[Mapping[str, Any]] = []
    undated: list[Mapping[str, Any]] = []
    for row in at_floor:
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

    eligible = _order(fresh)
    thin = _order(under_floor)

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
    # A row that CLEARS the floor and is merely old is a better discovery note
    # than a current row with three samples, so the stale and undated pools are
    # preferred and the thin one is the last resort. The reason names whichever
    # pool the shown row came from.
    if stale:
        coverage["discovery_reason"] = "not_fresh"
    elif undated:
        coverage["discovery_reason"] = "no_session"
    elif thin:
        coverage["discovery_reason"] = "floor"
    # **`min_n` binds the stale and undated pools too** (fix round): `stale` and
    # `undated` are drawn from `at_floor` above, so a family with three samples
    # can never be shown as a stale discovery under a floor of six - a floor
    # that only holds when the clock is right is not a floor. `thin` is the one
    # pool below the floor, by definition, and the sentence beside it says so.
    discovery_pool = _order(stale) or _order(undated) or _order(thin)
    coverage["discovery_leader"] = discovery_pool[0] if discovery_pool else None

    # The branches below are in the SAME order as the discovery pools above, so
    # the reason a verdict gives always names the gate that kept out the row it
    # is showing. Rows that cleared the floor come first: "measured enough, just
    # old" is a different piece of news from "not measured enough yet".
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


# ===========================================================================
# ST6 - ONE deterministic evidence snapshot, 2026-09-06
# ===========================================================================
#
# ST2 gave the Setup Tracker banner one leader. ST6 gives the whole DESK one
# READING: the strip above the M5 list, the tracker banner, Weekend Prep's
# verdict card and the AWAY Recap all print the same `snapshot_id`, so two
# screens read a minute apart can be reconciled instead of argued about.
#
# **The snapshot is PURE.** `build_snapshot` opens no file, starts no thread,
# reads no clock that decides anything and asks no store a question. Its caller
# (`ui.services.working_lately_service`) does the reading on a worker and hands
# the rows in. Identical inputs give an identical `snapshot_id` and identical
# verdicts - which is what makes the leader-change events a record of the
# EVIDENCE rather than a log of the half-hourly timer.
#
# **Three kinds, never pooled.** A swing trade-R rate, a swing favorable-
# direction rate and a day-trade held x ran are three different questions with
# three outcome definitions, three horizons and three units. They get three
# verdicts. `pool_cells` is the one helper that would combine cells and it
# REFUSES across kind, side or outcome kind - the refusal is the mechanism,
# because no formula makes those comparable.
#
# **Dependence is answered by refusing, not by a new number.** The trader:
# *"Any new confidence calculation must account for shared sessions and
# overlapping holdings and must be validated; ordinary Wilson bounds alone do
# not solve dependence or multiple testing."* So no new confidence calculation
# was written. A cell whose top symbol or top session supplies MORE than
# `CONCENTRATION_LIMIT` of its own sample is not eligible to lead, and the
# reason says `concentrated`. The multiple-testing exposure is PRINTED rather
# than corrected: `observational leader among K cells`. Nothing is called
# proven.


import hashlib
from dataclasses import asdict, fields as dataclass_fields

#: How many consecutive snapshots with DISTINCT `as_of` a NEW leader must lead
#: before the desk announces it.
#:
#: **Declared 2026-09-06, before any forward evaluation, and NOT tuned.** The
#: trader: *"Choose any margin/persistence rule before inspecting its forward
#: evaluation."* Two is the smallest number that is not one: it costs a real
#: leader a single session of delay and it stops a one-session wobble - a
#: correction landing, a stale export, one heavy name reporting - from being
#: announced as a change of regime. Until it is met the verdict is
#: `no_clear_leader` with `awaiting persistence (1 of 2)`, which is a true
#: sentence about the evidence rather than a hedge.
LEADER_PERSISTENCE_SNAPSHOTS = 2

#: How far the top day-trade cell's bound must clear the runner-up's, in SCORE
#: units (`held_run_score` = P(held 30m) x trimmed-mean MFE_R, so R).
#:
#: **Declared 2026-09-06, before any forward evaluation, and NOT tuned.** It
#: exists because `LEADER_MARGIN_LB` (0.05) is a margin on a WIN RATE - a
#: quantity bounded in [0, 1] - and applying it to an R-scale bound would make
#: two day-trade cells 0.06R apart read as a clear leader. A tenth of an R of
#: held-run separation is the smallest gap this desk would act on: below it the
#: two names are the same name as far as the next trade is concerned.
LEADER_MARGIN_HELD_RUN_R = 0.10

#: The largest share of a cell's own sample ONE symbol or ONE session may
#: supply and still be eligible to lead.
#:
#: **Declared 2026-09-06, before any forward evaluation.** Half is the point
#: past which a cell is mostly one thing wearing a large n. Deliberately
#: EXCEEDS, not "reaches": a cell split evenly over two sessions sits at exactly
#: 0.5 and is the smallest honest spread, so refusing it would refuse the
#: two-session case the desk sees every week.
CONCENTRATION_LIMIT = 0.5

#: The three questions, and the fact that they are three. A fourth kind is a
#: fourth verdict, never a fourth row inside one of these.
SNAPSHOT_KINDS = ("swing_trade_r", "swing_favorable", "daytrade_held_run")

#: The causes a leader change may carry. Checked REFUSAL-FIRST (lead decision,
#: 2026-09-06): `lost_coverage`, then `corrected_data`, then `window_rollover`,
#: then `new_outcomes`. The packet listed the rollover first, but `as_of` moves
#: on nearly every build, which would make the two interesting causes
#: unreachable.
EVENT_CAUSES = ("lost_coverage", "corrected_data", "window_rollover", "new_outcomes")

#: What the day-trade statistic IS, spelled once. NAME-SELECTION evidence -
#: which name to look at - and never captured P&L; the snapshot carries no P&L
#: field for it and a test asserts the dataclass has none.
HELD_RUN_STATISTIC_NAME = "held_run_score (P(held 30m) x trimmed MFE_R)"

#: The one sentence that says the leader is observational. Printed, never
#: corrected away: K cells were looked at and the best of K was named.
#:
#: **K is that KIND's own cell count** (re-review advisory 1). A caveat summed
#: across kinds would overstate the search for the kind it is standing beside -
#: the swing leader was not chosen against the day-trade cells; they were never
#: comparable and `pool_cells` refuses to make them so.
OBSERVATIONAL_CAVEAT = "observational leader among {k} cells"

#: Which column carries the session a favorable-direction row was MEASURED on,
#: in the order they are tried.
#:
#: **Re-review blocker 4.** `swing_favorable` was dated by `scan_date`, which is
#: the ENTRY, while the outcome is measured `horizon_sessions` later - so a file
#: whose newest entry is exactly 5 sessions back (the live one on 2026-09-04:
#: newest `scan_date` 2026-08-28, horizon 5) could never be fresh and the kind
#: could never produce a leader at all. The freshness rule asks "when was this
#: last MEASURED", and for a v1 row that is `future_scan_date`, for a v2 row
#: `target_session`.
FAVORABLE_MEASURED_FIELDS = ("future_scan_date", "target_session")


@dataclass(frozen=True)
class EvidenceCell:
    """One (kind, side, family) reading, carrying everything it took to make it.

    The trader's requirement, verbatim: *"The snapshot should identify:
    side/family, population, outcome version, entry/knowledge basis, actual
    horizon, evidence window, latest measured session, maturity and coverage
    counts, independent-day/name concentration, statistic and uncertainty, and
    the reason the leader is eligible or withheld."* Every one of those is a
    field here except the last, which is the verdict's job.

    For a swing cell `n_eligible` is the group's ELIGIBLE ROW COUNT and the
    uncertainty's denominator is `n_graded`: a FLAT is a measured outcome with
    no answer to the win/loss question, so it belongs in the population and
    never in the rate.

    **Side keeps the case its source spells it in** - the swing exports say
    `LONG`, the outcome log says `long` - and every COMPARISON upper-cases
    (`name`, `pool_cells`, the priority order). Rewriting a source's own
    spelling into the cell would make the cell disagree with the file a reader
    opens next.
    """

    kind: str
    side: str
    family: str
    outcome_kind: str
    outcome_version: str
    knowledge_basis: str
    horizon: str
    window_sessions: int
    latest_measured_session: str
    n_eligible: int
    n_pending: int
    n_excluded: int
    n_symbols: int
    n_sessions: int
    top_symbol_share: float | None
    top_session_share: float | None
    statistic: float | None
    statistic_name: str
    uncertainty_low: float | None
    uncertainty_kind: str
    namespace: str
    #: The rate's own denominator. Zero for a cell whose statistic is not a rate
    #: (the day-trade score is not one).
    n_graded: int = 0
    #: The floor this cell was measured against, and whether it cleared it.
    #: Carried rather than re-derived: the day-trade cells are floored by the
    #: AGGREGATOR (`Segment.summary`'s own `min_n`) and the swing cells by the
    #: desk's `MIN_REPORTABLE_N`, and a reader that guessed would get one wrong.
    n_floor: int = 0
    meets_floor: bool = False

    @property
    def name(self) -> str:
        """`"{SIDE} {family}"` - the ONE spelling of a cell's identity."""
        return f"{str(self.side or '').upper()} {self.family}".strip()

    @property
    def concentrated(self) -> bool:
        """More than half of the sample from one name or one session."""
        for share in (self.top_symbol_share, self.top_session_share):
            if share is not None and float(share) > CONCENTRATION_LIMIT:
                return True
        return False

    def identity_tuple(self) -> tuple:
        """The cell as the `snapshot_id` hashes it - every field, in order."""
        return tuple(_hashable(getattr(self, spec.name)) for spec in dataclass_fields(self))

    def as_row(self) -> dict[str, Any]:
        """The cell as a ROW MAPPING, which is what `LeaderVerdict` carries.

        `setup_family` and `side` are spelled the way every renderer on the desk
        already reads them (`verdict.leader["setup_family"]`), so ST2's banner
        blocks render an ST6 verdict without knowing anything changed.
        """
        row = asdict(self)
        row["setup_family"] = self.family
        row["leader_name"] = self.name
        return row

    def line(self) -> str:
        """One line for the tooltip: what it measured, over what, on how much."""
        statistic = "unmeasured" if self.statistic is None else f"{self.statistic:.4g}"
        low = "unmeasured" if self.uncertainty_low is None else f"{self.uncertainty_low:.4g}"
        bits = [
            f"{self.name} [{self.kind}]",
            f"{self.statistic_name} {statistic}",
            f"{self.uncertainty_kind} >= {low}",
            f"n={self.n_eligible} ({self.n_graded} graded, {self.n_pending} pending, "
            f"{self.n_excluded} excluded)",
            f"{self.n_symbols} symbol(s) / {self.n_sessions} session(s)",
            (
                # Re-review advisory 3: a cell whose export states coverage as
                # COUNTS carries no shares, and "top symbol unmeasured" reads
                # like a measurement that came back empty. It was never taken.
                "concentration unmeasured (this export states coverage as counts, "
                "not shares, so no cell of this kind can be refused as concentrated)"
                if self.top_symbol_share is None and self.top_session_share is None
                else f"top symbol {_share_text(self.top_symbol_share)}, "
                f"top session {_share_text(self.top_session_share)}"
            ),
            f"outcome {self.outcome_kind} ({self.outcome_version})",
            f"basis {self.knowledge_basis}",
            f"horizon {self.horizon}",
            f"window {self.window_sessions} sessions through "
            f"{self.latest_measured_session or 'an unstated session'}",
            f"namespace {self.namespace}",
        ]
        if self.concentrated:
            bits.append("CONCENTRATED - not eligible to lead")
        if not self.meets_floor:
            bits.append(f"below the n>={self.n_floor} floor")
        return "; ".join(bits)


def _share_text(share: float | None) -> str:
    return "unmeasured" if share is None else f"{float(share):.2f}"


def _hashable(value: Any) -> Any:
    """A float rounded far below anything printed and far above float noise."""
    if isinstance(value, float):
        return round(value, 10)
    return value


def pool_cells(cells: Iterable["EvidenceCell"]) -> list["EvidenceCell"]:
    """The ONE helper that would combine cells - and it refuses across the axes.

    Nothing on the desk pools a swing cell with a day-trade cell, a long with a
    short, or one outcome kind with another. This is what keeps that true: the
    refusal IS the mechanism, and there is deliberately no formula behind it.
    Cells agreeing on all three axes are handed back untouched, because "the
    same question asked twice" is the only case where combining is even defined.
    """
    kept = [cell for cell in cells if isinstance(cell, EvidenceCell)]
    for axis, getter in (
        ("kind", lambda cell: cell.kind),
        ("side", lambda cell: str(cell.side or "").upper()),
        ("outcome kind", lambda cell: cell.outcome_kind),
    ):
        distinct = {getter(cell) for cell in kept}
        if len(distinct) > 1:
            raise ValueError(
                f"cells differ in {axis} "
                f"({', '.join(sorted(str(value) for value in distinct))}); a snapshot "
                f"never pools across kind, side or outcome kind"
            )
    return kept


def leader_name(verdict: "LeaderVerdict | None") -> str:
    """`"{SIDE} {family}"` for a verdict that names a leader, else `""`.

    The events file uses it in BOTH directions - the prior leader and the new
    one - so a name written on the way out is byte-identical to the one written
    on the way in, and a dedupe key over the two can be trusted.
    """
    if verdict is None or verdict.leader is None:
        return ""
    row = verdict.leader
    existing = str(row.get("leader_name") or "").strip()
    if existing:
        return existing
    side = str(row.get("side") or "").upper()
    family = str(row.get("setup_family") or "").strip()
    return f"{side} {family}".strip()


# ---------------------------------------------------------------------------
# the cells - one builder per source, none of them reading a file
# ---------------------------------------------------------------------------


def _float_or_none(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _shares(labels: Sequence[str], total: int) -> tuple[int, float | None]:
    """`(distinct, top_share)` over a population of `total` rows.

    The denominator is the CELL's own row count, not the number of labelled
    rows: a cell where half the rows carry no symbol is not concentrated in the
    half that do, it is half unlabelled, and dividing by the labelled ones would
    read as the opposite.
    """
    counts: dict[str, int] = {}
    for raw in labels:
        label = str(raw or "").strip()
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts or total <= 0:
        return len(counts), None
    return len(counts), max(counts.values()) / float(total)


def swing_trade_r_cells(recent_rows: Iterable[Mapping[str, Any]]) -> list[EvidenceCell]:
    """ST2.1's recent setup-type rows, one cell each.

    The counts are the export's OWN integers (`n_wins` / `n_losses` /
    `n_flats`); nothing here reconstructs one from a rate, which is the defect
    ST2.1 removed. A row whose counts were never exported becomes a cell with a
    `None` statistic - present in the population, ineligible to lead, and
    visible as "counts not exported yet" rather than missing.
    """
    from evidence_stats import LATELY_SESSIONS

    cells: list[EvidenceCell] = []
    for row in recent_rows or ():
        if not isinstance(row, Mapping):
            continue
        pair = counted_pair(row)
        wins, losses = pair or (0, 0)
        graded = wins + losses
        flats = _int(row.get("n_flats")) or 0
        statistic = (wins / graded) if (pair is not None and graded) else None
        cells.append(
            EvidenceCell(
                kind="swing_trade_r",
                side=_text(row.get("side")),
                family=_text(row.get("setup_family")),
                outcome_kind=_text(row.get("outcome_kind")),
                outcome_version=_text(row.get("outcome_version")),
                knowledge_basis=_text(row.get("knowledge_basis")),
                horizon=_text(row.get("horizon_basis")),
                window_sessions=int(LATELY_SESSIONS),
                latest_measured_session=_text(row.get("latest_measured_session")),
                # The MEASURED population: a flat is a measured outcome with no
                # answer to the win/loss question, so it counts here and never
                # in the rate's denominator.
                n_eligible=graded + flats if pair is not None else (_int(row.get("closed_setups")) or 0),
                n_pending=_int(row.get("n_pending")) or 0,
                n_excluded=_int(row.get("n_unmeasured")) or 0,
                n_symbols=_int(row.get("n_symbols")) or 0,
                n_sessions=_int(row.get("n_entry_sessions")) or 0,
                # This export states its coverage as counts, not as shares. An
                # unstated share is None - never a zero, which would read as
                # "measured, and perfectly spread".
                top_symbol_share=None,
                top_session_share=None,
                statistic=statistic,
                statistic_name="win rate (closed basis)",
                uncertainty_low=wilson_lower_bound(wins, graded) if graded else None,
                uncertainty_kind="wilson_lower_bound_95",
                namespace=_text(row.get("namespace")).lower() or "live",
                n_graded=graded,
                n_floor=int(MIN_REPORTABLE_N),
                meets_floor=bool(pair is not None and graded >= int(MIN_REPORTABLE_N)),
            )
        )
    return cells


#: The policy object behind a favorable-direction read, named so a cell can say
#: which one measured it without importing `swing_evidence` to find out.
_FAVORABLE_POLICY_VERSIONS = {
    "favorable_direction_scanrow_v1": "POLICY_SCANROW_V1",
    "favorable_direction_session_v2": "POLICY_SESSION_V2",
}


def _measured_stamp(row: Mapping[str, Any], policy: Any) -> str:
    """The session this favorable-direction row was MEASURED on.

    `future_scan_date` (v1) or `target_session` (v2). Falls back to the entry
    clock only when the row carries neither, and a fallback is NOT a measurement
    date - it dates the cell earlier than the truth, which can only ever make
    the freshness test stricter, never looser.
    """
    for field in FAVORABLE_MEASURED_FIELDS:
        stamp = str(row.get(field) or "")[:10]
        if stamp:
            return stamp
    return str(row.get(policy.clock_field) or "")[:10]


def swing_favorable_cells(read: Any) -> list[EvidenceCell]:
    """ST1's eligible rows, grouped by (side, family) - a FAVORABLE rate.

    Not a win rate, and the cell says so in three places: `outcome_kind` is the
    policy's own (`favorable_direction_scanrow_v1`), `statistic_name` says
    favorable, and `horizon` carries its UNIT ("5 scan rows", never a bare 5).
    ST1's whole point was that this file's `win` is the sign of a close-to-close
    percent move at a scan-row offset; calling it a win on a trader surface is
    the thing that packet stopped.

    A row whose move is exactly zero is a FLAT: it is eligible evidence, it
    counts in the population, and it is not in the rate's denominator - the same
    rule the trade-R cells follow, so the two read the same way.
    """
    if read is None:
        return []
    policy = read.policy
    horizon = f"{int(policy.horizon_sessions)} {policy.horizon_unit}"
    version = _FAVORABLE_POLICY_VERSIONS.get(policy.outcome_kind, policy.outcome_kind)

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in read.rows or ():
        grouped.setdefault(
            (_text(row.get("side")), _text(row.get("setup_family"))), []
        ).append(row)
    pending: dict[tuple[str, str], int] = {}
    for row in getattr(read, "pending", ()) or ():
        key = (_text(row.get("side")), _text(row.get("setup_family")))
        pending[key] = pending.get(key, 0) + 1

    cells: list[EvidenceCell] = []
    for (side, family), rows in grouped.items():
        favorable = against = flat = unreadable = 0
        for row in rows:
            move = _float_or_none(row.get("side_return_pct"))
            if move is None:
                unreadable += 1
            elif move > 0:
                favorable += 1
            elif move < 0:
                against += 1
            else:
                flat += 1
        graded = favorable + against
        distinct_symbols, symbol_share = _shares(
            [str(row.get("symbol") or "") for row in rows], len(rows)
        )
        sessions = [str(row.get(policy.clock_field) or "")[:10] for row in rows]
        distinct_sessions, session_share = _shares(sessions, len(rows))
        # The WINDOW is measured on the policy's own clock (`scan_date` for v1 -
        # "lately" asks which entries are recent). FRESHNESS is a different
        # question - "when was this last measured" - and for a horizon-5 row the
        # answer is five sessions after the entry. Dating the cell by the entry
        # made this kind permanently stale (re-review blocker 4).
        measured = [_measured_stamp(row, policy) for row in rows]
        cells.append(
            EvidenceCell(
                kind="swing_favorable",
                side=side,
                family=family,
                outcome_kind=policy.outcome_kind,
                outcome_version=version,
                knowledge_basis=policy.knowledge_basis,
                horizon=horizon,
                window_sessions=int(policy.window_sessions),
                latest_measured_session=max((stamp for stamp in measured if stamp), default=""),
                n_eligible=len(rows),
                n_pending=pending.get((side, family), 0),
                # The GROUP's own exclusion. The policy's file-level exclusions
                # (wrong horizon, outside window, stale) were never grouped, so
                # they ride in the snapshot's `sources` entry rather than being
                # divided across families they cannot be attributed to.
                n_excluded=unreadable,
                n_symbols=distinct_symbols,
                n_sessions=distinct_sessions,
                top_symbol_share=symbol_share,
                top_session_share=session_share,
                statistic=(favorable / graded) if graded else None,
                statistic_name="favorable-direction rate (percent move, not R)",
                uncertainty_low=wilson_lower_bound(favorable, graded) if graded else None,
                uncertainty_kind="wilson_lower_bound_95",
                namespace="live",
                n_graded=graded,
                n_floor=int(MIN_REPORTABLE_N),
                meets_floor=bool(graded >= int(MIN_REPORTABLE_N)),
            )
        )
    return cells


def daytrade_held_run_cells(summaries: Mapping[Any, Mapping[str, Any]] | None) -> list[EvidenceCell]:
    """`held_run_score.dimension_summaries`, bounce_type x SIDE only.

    **The pooled `("bounce_type", "all", ...)` row is deliberately skipped**
    (lead decision, 2026-09-06): it is the same episodes counted a second time
    under a second name, and a snapshot holding both would compare a cell with
    its own superset.

    The uncertainty is the SESSION-BLOCK bootstrap's lower percentile, not a
    Wilson bound - the statistic is a product of a rate and a trimmed mean, and
    a binomial interval on it would be a number about a different quantity.
    Where the bootstrap refuses (one session in the sample) the cell says
    unmeasured and cannot lead.
    """
    if summaries is None:
        return []
    try:
        from held_run_score import ALL_DIRECTIONS, ROLLING_SESSIONS
    except Exception:  # pragma: no cover - the module ships beside this one
        ALL_DIRECTIONS, ROLLING_SESSIONS = "all", 20

    cells: list[EvidenceCell] = []
    for key, summary in (summaries or {}).items():
        if not isinstance(key, tuple) or len(key) != 3:
            continue
        dimension, direction, value = key
        if str(dimension) != "bounce_type" or str(direction) == ALL_DIRECTIONS:
            continue
        concentration = summary.get("concentration") or {}
        by_symbol = concentration.get("by_symbol") or {}
        by_session = concentration.get("by_session") or {}
        # **`score_bootstrap`, never `bootstrap`** (re-review blocker 3). The
        # second one is an interval on the MFEs of the held episodes - a real
        # answer to "how far did the held ones run" and the WRONG number to put
        # beside `held_run_score`, which is a product measured over two
        # denominators. Live, it printed `held x ran 1.21 (>= 2.070)`: a lower
        # bound above the statistic, and the cell it crowned was not the cell
        # with the best held x ran.
        bootstrap = summary.get("score_bootstrap") or {}
        measured_interval = bool(bootstrap.get("measured"))
        cells.append(
            EvidenceCell(
                kind="daytrade_held_run",
                side=str(direction),
                family=str(value),
                outcome_kind="intraday_held_30m_mfe_r",
                outcome_version="held_run_score_v1",
                knowledge_basis=(
                    "M5 alert entry; the stop intact through the first 30 minutes; "
                    "MFE in R from the intraday outcome log"
                ),
                horizon="first 30 minutes held, then MFE to the episode's last measured bar",
                window_sessions=int(ROLLING_SESSIONS),
                latest_measured_session=str(summary.get("latest_session") or ""),
                n_eligible=int(summary.get("n_measured") or 0),
                n_pending=int(summary.get("n_pending") or 0),
                n_excluded=int(summary.get("n_unmeasured") or 0),
                n_symbols=int(summary.get("n_symbols") or 0),
                n_sessions=int(summary.get("n_sessions") or 0),
                top_symbol_share=by_symbol.get("top_share"),
                top_session_share=by_session.get("top_share"),
                statistic=summary.get("held_run_score"),
                statistic_name=HELD_RUN_STATISTIC_NAME,
                uncertainty_low=bootstrap.get("low") if measured_interval else None,
                uncertainty_kind=(
                    "held_run_score_session_block_low"
                    if measured_interval
                    else f"held_run_score_session_block_unmeasured ({bootstrap.get('reason') or 'no reason given'})"
                ),
                namespace="live",
                n_graded=int(summary.get("n_held") or 0),
                n_floor=int(summary.get("n_floor") or 0),
                meets_floor=bool(summary.get("meets_floor")),
            )
        )
    return cells


# ---------------------------------------------------------------------------
# the one decision, over cells
# ---------------------------------------------------------------------------


def _sessions_behind(stamp: Any, last_completed_session: date) -> int | None:
    """Exchange sessions between a cell's newest reading and the last close.

    None means the cell does not say. Uncertainty never confirms freshness, so
    a `None` is treated as NOT fresh everywhere below - the same rule
    `_sessions_stale` applies to a row.
    """
    measured = _as_date(stamp)
    if measured is None:
        return None
    import market_calendar

    try:
        return market_calendar.trading_days_between(measured, last_completed_session)
    except Exception:
        return None


#: What each kind RANKS on, and the margin the top must clear, in that kind's
#: own units. Two entries rather than one rule because the desk has two
#: headlines and decision 0016 named them separately.
#:
#: The swing kinds rank on the WILSON LOWER BOUND, as everywhere else: two
#: rates can differ by a mile and still be one sample apart when one is thin,
#: and the bound is the number that already knows that.
#:
#: **The day-trade kind ranks on `held_run_score` itself** (re-review blocker 3,
#: lead decision 2026-09-06). It is THE day-trade headline - decision 0016
#: answer 4 - and ranking on its interval's lower edge instead named a different
#: cell from the one the whole desk calls best: live, the bound crowned
#: `LONG regime_pause_rs` while the headline's own leader was
#: `LONG lrsi_cross_50`. The interval is still REQUIRED (a cell whose score
#: cannot be bounded is not eligible) and is still printed - it just is not the
#: sort key.
RANK_BASIS = {
    "swing_trade_r": ("uncertainty_low", LEADER_MARGIN_LB),
    "swing_favorable": ("uncertainty_low", LEADER_MARGIN_LB),
    "daytrade_held_run": ("statistic", LEADER_MARGIN_HELD_RUN_R),
}


def rank_basis(kind: str) -> tuple[str, float]:
    """`(field, margin)` for one kind - the ONE place either is decided."""
    return RANK_BASIS.get(kind, ("uncertainty_low", LEADER_MARGIN_LB))


def _rank_value(cell: EvidenceCell) -> float | None:
    field, _margin = rank_basis(cell.kind)
    value = getattr(cell, field, None)
    return None if value is None else float(value)


def _cell_order(cells: Sequence[EvidenceCell]) -> list[EvidenceCell]:
    """Best first, on that kind's OWN basis, then n, then name.

    A cell with no rank value sorts last: it is not ranked at all, and a
    fallback to another quantity would be a second basis wearing the first's
    name.
    """

    def key(cell: EvidenceCell):
        value = _rank_value(cell)
        return (
            value is None,
            -(value if value is not None else 0.0),
            -int(cell.n_eligible or 0),
            cell.name,
        )

    return sorted(cells, key=key)


def _persistence(
    candidate: EvidenceCell,
    previous: LeaderVerdict | None,
    *,
    as_of: str,
) -> tuple[bool, int]:
    """`(announce, snapshots_led)` for a candidate that is not already leading.

    A NEW leader must lead in `LEADER_PERSISTENCE_SNAPSHOTS` snapshots with
    DISTINCT `as_of` before it is announced. Distinct on purpose: the desk
    rebuilds on a half-hourly timer, so counting builds would let a leader
    "persist" through thirty minutes of the same evidence.
    """
    coverage = dict(previous.coverage) if previous is not None else {}
    pending_name = str(coverage.get("pending_leader") or "")
    pending_as_of = str(coverage.get("pending_leader_as_of") or "")
    led = int(coverage.get("pending_leader_snapshots") or 0)
    if pending_name == candidate.name and pending_as_of and pending_as_of != as_of:
        led += 1
    elif pending_name == candidate.name and pending_as_of == as_of:
        led = max(1, led)
    else:
        led = 1
    return led >= int(LEADER_PERSISTENCE_SNAPSHOTS), led


def select_cell_leader(
    cells: Iterable[EvidenceCell],
    *,
    kind: str,
    last_completed_session: date,
    previous: LeaderVerdict | None = None,
    source_rows: int | None = 0,
) -> LeaderVerdict:
    """One verdict for one KIND. Never across kinds - that is `pool_cells`' refusal.

    The four states are ST2's, unchanged, plus one gate ST2 did not have:
    **concentration**. A cell whose top symbol or top session supplies more than
    `CONCENTRATION_LIMIT` of its own sample is withheld with the reason
    `concentrated`, because dependence between overlapping holdings and shared
    sessions is not something a wider interval fixes.

    `source_rows` is the SOURCE's row count. `None` means the source could not
    be read at all - an absent source is a question that was not asked, and it
    can never produce a leader; a zero is an answer.
    """
    mine = [cell for cell in cells if isinstance(cell, EvidenceCell) and cell.kind == kind]
    coverage: dict[str, Any] = {
        "kind": kind,
        "cells_considered": len(mine),
        "margin_lb": LEADER_MARGIN_LB,
        "freshness_sessions": LEADER_FRESHNESS_SESSIONS,
        "persistence_snapshots": LEADER_PERSISTENCE_SNAPSHOTS,
        "concentration_limit": CONCENTRATION_LIMIT,
        "last_completed_session": last_completed_session.isoformat(),
        "prior_leader": leader_name(previous),
        "prior_state": str(previous.state) if previous is not None else "",
        "pending_leader": "",
        "pending_leader_as_of": "",
        "pending_leader_snapshots": 0,
        "studies_excluded": 0,
        "unreadable": 0,
        "not_fresh": 0,
        "no_session_on_cell": 0,
        "under_floor": 0,
        "concentrated": [],
        "eligible": 0,
        "source_rows": source_rows,
    }

    def _verdict(state, leader, runner_up, reason, as_of="", policy_line=""):
        return LeaderVerdict(
            state=state,
            leader=leader,
            runner_up=runner_up,
            reason=reason,
            as_of=as_of,
            policy_line=policy_line,
            coverage=coverage,
        )

    if source_rows is None:
        # The one sentence that must survive a rewrite: an unreadable source is
        # never a zero, and it never becomes news about the market.
        detail = (
            f"the {kind} source unavailable - a source that could not be read is a "
            f"question that was not asked, never a zero"
        )
        if previous is not None and previous.leader is not None:
            carried = dict(coverage)
            carried["carried_from"] = previous.as_of
            return LeaderVerdict(
                state="last_reliable_reading",
                leader=previous.leader,
                runner_up=previous.runner_up,
                reason=f"{detail}, so this is the last reliable reading, taken {previous.as_of or 'earlier'}.",
                as_of=previous.as_of,
                policy_line=previous.policy_line,
                coverage=carried,
            )
        return _verdict("no_evidence", None, None, f"{detail}.")

    studies = [cell for cell in mine if cell.namespace == "study"]
    live = [cell for cell in mine if cell.namespace != "study"]
    coverage["studies_excluded"] = len(studies)
    rank_field, margin = rank_basis(kind)
    coverage["rank_basis"] = rank_field
    coverage["margin"] = margin
    # A cell needs BOTH: a statistic to be read and an interval to be bounded by.
    # The day-trade kind ranks on the statistic, but a score nobody could put an
    # interval around is a number without a claim, and the trader's own words
    # were that ordinary bounds alone do not solve dependence - the answer to
    # which is to require the session-block one, not to drop it.
    readable = [
        cell
        for cell in live
        if cell.statistic is not None and cell.uncertainty_low is not None
    ]
    coverage["unreadable"] = len(live) - len(readable)

    fresh: list[EvidenceCell] = []
    stale: list[EvidenceCell] = []
    undated: list[EvidenceCell] = []
    for cell in readable:
        behind = _sessions_behind(cell.latest_measured_session, last_completed_session)
        if behind is None:
            undated.append(cell)
        elif behind <= LEADER_FRESHNESS_SESSIONS:
            fresh.append(cell)
        else:
            stale.append(cell)
    coverage["not_fresh"] = len(stale)
    coverage["no_session_on_cell"] = len(undated)

    on_floor = [cell for cell in fresh if cell.meets_floor]
    coverage["under_floor"] = len(fresh) - len(on_floor)
    concentrated = [cell for cell in on_floor if cell.concentrated]
    coverage["concentrated"] = [cell.name for cell in _cell_order(concentrated)]
    eligible = _cell_order([cell for cell in on_floor if not cell.concentrated])
    coverage["eligible"] = len(eligible)

    withheld = ""
    if concentrated:
        withheld = (
            f" {len(concentrated)} cell(s) withheld as concentrated "
            f"({', '.join(coverage['concentrated'])}): more than "
            f"{CONCENTRATION_LIMIT:.0%} of the sample from one name or one session."
        )

    if eligible:
        top = eligible[0]
        runner_up = eligible[1] if len(eligible) > 1 else None
        top_rank = _rank_value(top) or 0.0
        basis_words = (
            "Wilson lower bound" if rank_field == "uncertainty_low" else top.statistic_name
        )
        if runner_up is not None:
            gap = top_rank - (_rank_value(runner_up) or 0.0)
            if gap <= margin:
                return _verdict(
                    "no_clear_leader",
                    None,
                    runner_up.as_row(),
                    f"{top.name} ({top_rank:.3f}) and {runner_up.name} "
                    f"({_rank_value(runner_up) or 0.0:.3f}) are {gap:.3f} apart on "
                    f"{basis_words}, inside the declared {margin:.2f} margin - no clear "
                    f"leader.{withheld}",
                    as_of=top.latest_measured_session,
                    policy_line=top.line(),
                )
            lead_sentence = (
                f"{top.name} leads {runner_up.name} by {gap:.3f} of {basis_words} "
                f"({top_rank:.3f} vs {_rank_value(runner_up) or 0.0:.3f}), clear of the "
                f"declared {margin:.2f} margin."
            )
        else:
            lead_sentence = (
                f"{top.name} is the only eligible cell at the n>={top.n_floor} floor "
                f"({basis_words} {top_rank:.3f})."
            )
        prior_name = str(coverage["prior_leader"])
        if top.name == prior_name:
            return _verdict(
                "leader",
                top.as_row(),
                runner_up.as_row() if runner_up is not None else None,
                f"{lead_sentence}{withheld}",
                as_of=top.latest_measured_session,
                policy_line=top.line(),
            )
        announce, led = _persistence(
            top, previous, as_of=last_completed_session.isoformat()
        )
        coverage["pending_leader"] = top.name
        coverage["pending_leader_as_of"] = last_completed_session.isoformat()
        coverage["pending_leader_snapshots"] = led
        if announce:
            coverage["pending_leader"] = ""
            coverage["pending_leader_as_of"] = ""
            coverage["pending_leader_snapshots"] = 0
            return _verdict(
                "leader",
                top.as_row(),
                runner_up.as_row() if runner_up is not None else None,
                f"{lead_sentence}{withheld}",
                as_of=top.latest_measured_session,
                policy_line=top.line(),
            )
        return _verdict(
            "no_clear_leader",
            None,
            top.as_row(),
            f"{top.name} is ahead but is a NEW leader: awaiting persistence "
            f"({led} of {LEADER_PERSISTENCE_SNAPSHOTS}) - a new leader is "
            f"announced only once it has led in that many snapshots with a "
            f"distinct as_of. {lead_sentence}{withheld}",
            as_of=top.latest_measured_session,
            policy_line=top.line(),
        )

    if stale or undated:
        source = _cell_order(stale or undated)[0]
        if stale:
            behind = _sessions_behind(source.latest_measured_session, last_completed_session)
            detail = (
                f"its newest measured session is {behind} exchange session(s) behind "
                f"{last_completed_session.isoformat()}"
            )
        else:
            detail = "the export carries no measured session at all"
        if previous is not None and previous.leader is not None:
            carried = dict(coverage)
            carried["carried_from"] = previous.as_of
            return LeaderVerdict(
                state="last_reliable_reading",
                leader=previous.leader,
                runner_up=previous.runner_up,
                reason=(
                    f"the evidence is not fresh ({detail}), so this is the last "
                    f"reliable reading, taken {previous.as_of or 'earlier'}.{withheld}"
                ),
                as_of=previous.as_of,
                policy_line=previous.policy_line,
                coverage=carried,
            )
        return _verdict(
            "no_evidence",
            None,
            None,
            f"the evidence is not fresh ({detail}) and there is no previous verdict "
            f"to fall back on.{withheld}",
        )

    if concentrated:
        return _verdict(
            "no_evidence",
            None,
            None,
            f"every fresh cell over the floor is concentrated, so none of them may "
            f"lead.{withheld}",
        )
    if coverage["under_floor"]:
        return _verdict(
            "no_evidence",
            None,
            None,
            f"no cell reached its own n floor; {coverage['under_floor']} fresh cell(s) "
            f"are below it, which is discovery and not a leader.",
        )
    if coverage["unreadable"]:
        return _verdict(
            "no_evidence",
            None,
            None,
            f"{coverage['unreadable']} cell(s) carry no measurable statistic, and a "
            f"statistic is never reconstructed from a stored rate.",
        )
    if studies:
        return _verdict(
            "no_evidence",
            None,
            None,
            f"every candidate is a study ({len(studies)} excluded); a study is an "
            f"unpromoted idea and never leads, whatever its R.",
        )
    return _verdict("no_evidence", None, None, "no rows to read.")


# ---------------------------------------------------------------------------
# the snapshot
# ---------------------------------------------------------------------------


#: The declared policy, hashed into every `snapshot_id`. Changing one of these
#: numbers changes the identity of every snapshot after it, which is the point:
#: two snapshots with the same id were measured under the same rules.
def market_local_now() -> str:
    """An ISO stamp WITH ITS OFFSET, market-local (advisory 6).

    Every other clock in this chain is an exchange session, so the wall clock
    beside it has to be in the same calendar or the two cannot be compared. The
    desk runs on PT and market-local is a different DAY for three hours every
    evening - which is when the overnight run reads these files.
    """
    try:
        import market_calendar

        return datetime.now(market_calendar.MARKET_TZ).isoformat(timespec="seconds")
    except Exception:  # noqa: BLE001 - never worth the build
        return datetime.now().astimezone().isoformat(timespec="seconds")


def _policy_lines() -> tuple[str, ...]:
    return (
        f"leader_margin_lb={LEADER_MARGIN_LB}",
        f"leader_freshness_sessions={LEADER_FRESHNESS_SESSIONS}",
        f"leader_persistence_snapshots={LEADER_PERSISTENCE_SNAPSHOTS}",
        f"concentration_limit={CONCENTRATION_LIMIT}",
        f"min_reportable_n={int(MIN_REPORTABLE_N)}",
        f"kinds={','.join(SNAPSHOT_KINDS)}",
    )


#: The persisted shape's name. A later shape gets a later name; nothing reads a
#: payload whose schema it does not recognise as this one.
SNAPSHOT_SCHEMA = "working_lately_snapshot_v2"

#: The fields that are a property of the KIND rather than of the cell. Written
#: once per kind in the payload and put back by `cells_from_payload`.
KIND_POLICY_FIELDS = (
    "outcome_kind",
    "outcome_version",
    "knowledge_basis",
    "horizon",
    "window_sessions",
    "statistic_name",
    "uncertainty_kind",
    "n_floor",
)


def _kind_policy(cells: Sequence[EvidenceCell]) -> dict[str, dict[str, Any]]:
    """`{kind: {field: value}}` for every field a kind's cells all agree on.

    The seven in `KIND_POLICY_FIELDS` are shared BY DEFINITION - they are
    properties of the kind, and a kind whose cells disagreed on one would be a
    kind that is really two, which `pool_cells` already refuses to combine. The
    rest are found by MEASUREMENT: on the live recent-types export every cell
    also carries the same `namespace`, the same two unmeasured concentration
    shares and the same zeros, and writing 150 copies of each is most of what
    made the first version 133 KB.

    Lossless in both directions: the per-cell value wins on read, so a field
    lifted here is a field every cell agreed on and every cell gets back.
    """
    by_kind: dict[str, list[EvidenceCell]] = {}
    for cell in cells:
        by_kind.setdefault(cell.kind, []).append(cell)
    names = [spec.name for spec in dataclass_fields(EvidenceCell) if spec.name != "kind"]
    out: dict[str, dict[str, Any]] = {}
    for kind, group in by_kind.items():
        first = group[0]
        shared: dict[str, Any] = {}
        for field in names:
            value = getattr(first, field)
            if field in KIND_POLICY_FIELDS or all(
                getattr(cell, field) == value for cell in group
            ):
                shared[field] = value
        out[kind] = shared
    return out


def _compact_cell(cell: EvidenceCell, shared: Mapping[str, Any]) -> dict[str, Any]:
    """One cell's own facts - everything except what the kind already said."""
    return {
        key: value
        for key, value in asdict(cell).items()
        if key == "kind" or key not in shared or value != shared[key]
    }


def cells_from_payload(payload: Mapping[str, Any] | None) -> list[EvidenceCell]:
    """Rebuild the cells, kind policy folded back in. `[]` when absent."""
    policy = dict((payload or {}).get("kind_policy") or {})
    names = {spec.name for spec in dataclass_fields(EvidenceCell)}
    out: list[EvidenceCell] = []
    for raw in (payload or {}).get("cells") or []:
        if not isinstance(raw, Mapping):
            continue
        merged = dict(policy.get(str(raw.get("kind")), {}))
        merged.update(raw)
        try:
            out.append(EvidenceCell(**{k: v for k, v in merged.items() if k in names}))
        except TypeError:
            continue
    return out


def payload_cells(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The cells as MAPPINGS with the kind policy folded in - what sorts read."""
    policy = dict((payload or {}).get("kind_policy") or {})
    out: list[dict[str, Any]] = []
    for raw in (payload or {}).get("cells") or []:
        if not isinstance(raw, Mapping):
            continue
        merged = dict(policy.get(str(raw.get("kind")), {}))
        merged.update(raw)
        out.append(merged)
    return out


@dataclass(frozen=True)
class EvidenceSnapshot:
    """One reading of "what is working lately", and its identity.

    `snapshot_id` is a sha1 over the SORTED cell tuples, the policy lines and
    `as_of` - and over nothing else. Not `built_at`, not a source's mtime, not
    the verdicts. A snapshot that changed identity every half-hourly tick would
    make the leader-change events a log of the timer, and a snapshot whose id
    moved when a file was merely re-saved would announce news that never
    happened.
    """

    snapshot_id: str
    as_of: str
    built_at: str
    cells: tuple[EvidenceCell, ...]
    verdicts: dict[str, LeaderVerdict]
    sources: dict[str, dict[str, Any]]

    def to_payload(self) -> dict[str, Any]:
        """The JSON the service persists and every surface renders. Small.

        **Small is a requirement, not a hope** (re-review advisory 2): on live
        data the first version was 133 KB, because seven of a cell's fields are
        the SAME STRING on every cell of its kind - the outcome kind, the outcome
        version, the knowledge basis, the horizon, the window, the statistic's
        name and the uncertainty's kind are properties of the KIND, not of the
        family. They are written ONCE per kind in `kind_policy` and rehydrated by
        `cells_from_payload`, so nothing the trader's requirement asks the
        snapshot to identify is lost and the file is roughly a third of the size.
        """
        kind_policy = _kind_policy(self.cells)
        return {
            "schema": SNAPSHOT_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "as_of": self.as_of,
            "built_at": self.built_at,
            "policy": list(_policy_lines()),
            "kind_policy": kind_policy,
            "cells": [
                _compact_cell(cell, kind_policy.get(cell.kind, {})) for cell in self.cells
            ],
            "verdicts": {
                kind: {
                    "state": verdict.state,
                    "leader": dict(verdict.leader) if verdict.leader is not None else None,
                    "runner_up": dict(verdict.runner_up) if verdict.runner_up is not None else None,
                    "reason": verdict.reason,
                    "as_of": verdict.as_of,
                    "policy_line": verdict.policy_line,
                    "coverage": dict(verdict.coverage),
                    "leader_name": leader_name(verdict),
                }
                for kind, verdict in self.verdicts.items()
            },
            "sources": {name: dict(entry) for name, entry in self.sources.items()},
        }


def verdicts_from_payload(payload: Mapping[str, Any] | None) -> dict[str, LeaderVerdict]:
    """Rebuild `{kind: LeaderVerdict}` from a persisted snapshot. `{}` when absent.

    The round trip has to be exact in one respect and one only: the coverage
    keys the persistence rule reads (`pending_leader`,
    `pending_leader_snapshots`, `pending_leader_as_of`). A restart that lost
    them would restart the two-snapshot clock and delay a real leader by a day.
    """
    verdicts: dict[str, LeaderVerdict] = {}
    for kind, entry in ((payload or {}).get("verdicts") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        verdicts[str(kind)] = LeaderVerdict(
            state=str(entry.get("state") or "no_evidence"),
            leader=dict(entry["leader"]) if isinstance(entry.get("leader"), Mapping) else None,
            runner_up=dict(entry["runner_up"]) if isinstance(entry.get("runner_up"), Mapping) else None,
            reason=str(entry.get("reason") or ""),
            as_of=str(entry.get("as_of") or ""),
            policy_line=str(entry.get("policy_line") or ""),
            coverage=dict(entry.get("coverage") or {}),
        )
    return verdicts


def _rows_by_session(stamps: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw in stamps:
        stamp = str(raw or "")[:10]
        if stamp:
            counts[stamp] = counts.get(stamp, 0) + 1
    return counts


def build_snapshot(
    *,
    recent_rows: Iterable[Mapping[str, Any]] = (),
    favorable_read: Any = None,
    held_run_summaries: Mapping[Any, Mapping[str, Any]] | None = None,
    last_completed_session: date,
    previous_verdicts: Mapping[str, LeaderVerdict] | None = None,
    sources: Mapping[str, Mapping[str, Any]] | None = None,
) -> EvidenceSnapshot:
    """The ONE reading. Pure: identical inputs give an identical id and verdicts.

    `recent_rows` are ST2.1's recent setup-type rows, `favorable_read` is an
    ST1 `swing_evidence.EligibleRead` (or None when the file could not be
    read), `held_run_summaries` is `held_run_score.dimension_summaries`'s dict
    (or None). `sources` lets the caller state where each came from - a path
    and an mtime - WITHOUT either entering the id.
    """
    recent = [row for row in (recent_rows or ()) if isinstance(row, Mapping)]
    cells: list[EvidenceCell] = []
    cells.extend(swing_trade_r_cells(recent))
    cells.extend(swing_favorable_cells(favorable_read))
    cells.extend(daytrade_held_run_cells(held_run_summaries))

    day_cells = [cell for cell in cells if cell.kind == "daytrade_held_run"]
    computed_sources: dict[str, dict[str, Any]] = {
        "swing_trade_r": {
            "rows": len(recent),
            "rows_by_session": _rows_by_session(
                row.get("latest_measured_session") for row in recent
            ),
        },
        "swing_favorable": {
            "rows": None if favorable_read is None else len(favorable_read.rows),
            "rows_by_session": (
                {}
                if favorable_read is None
                else _rows_by_session(
                    row.get(favorable_read.policy.clock_field)
                    for row in favorable_read.rows
                )
            ),
            "excluded": (
                {}
                if favorable_read is None
                else {str(k): int(v) for k, v in dict(favorable_read.excluded).items()}
            ),
        },
        "daytrade_held_run": {
            "rows": None if held_run_summaries is None else len(day_cells),
            "rows_by_session": _rows_by_session(
                cell.latest_measured_session for cell in day_cells
            ),
        },
    }
    merged: dict[str, dict[str, Any]] = {}
    for name, computed in computed_sources.items():
        entry = dict((sources or {}).get(name) or {})
        entry.setdefault("path", "")
        entry.setdefault("mtime", None)
        entry.update(computed)
        merged[name] = entry
    for name, value in (sources or {}).items():
        if name in merged:
            continue
        entry = dict(value or {})
        entry.setdefault("path", "")
        entry.setdefault("mtime", None)
        entry.setdefault("rows", None)
        entry.setdefault("rows_by_session", {})
        merged[str(name)] = entry

    as_of = last_completed_session.isoformat()
    digest = hashlib.sha1()
    for line in _policy_lines():
        digest.update(line.encode("utf-8"))
        digest.update(b"\x1e")
    for identity in sorted(repr(cell.identity_tuple()) for cell in cells):
        digest.update(identity.encode("utf-8"))
        digest.update(b"\x1e")
    digest.update(as_of.encode("utf-8"))

    previous = dict(previous_verdicts or {})
    # Advisory 6: MARKET-local and AWARE. A naive stamp on this desk (PT) is a
    # different day from market-local for three hours every evening, which is
    # exactly when the overnight run reads these files; and `as_of` beside it is
    # an exchange session, so the two would have been in different calendars.
    verdicts = {
        kind: select_cell_leader(
            cells,
            kind=kind,
            last_completed_session=last_completed_session,
            previous=previous.get(kind),
            source_rows=merged[kind]["rows"],
        )
        for kind in SNAPSHOT_KINDS
    }
    return EvidenceSnapshot(
        snapshot_id=digest.hexdigest(),
        as_of=as_of,
        built_at=market_local_now(),
        cells=tuple(cells),
        verdicts=verdicts,
        sources=merged,
    )


def snapshot_stamp(snapshot_or_payload: Any) -> str:
    """The ONE short identity every surface prints.

    Four surfaces answered "what is working" and nothing tied their answers to
    one reading. This is the tie: a screenshot of the strip and a screenshot of
    the tracker banner can be reconciled by eye.
    """
    if isinstance(snapshot_or_payload, EvidenceSnapshot):
        snapshot_id = snapshot_or_payload.snapshot_id
        as_of = snapshot_or_payload.as_of
    else:
        payload = snapshot_or_payload or {}
        snapshot_id = str(payload.get("snapshot_id") or "")
        as_of = str(payload.get("as_of") or "")
    return f"snapshot {snapshot_id[:8]} as of {as_of or 'an unstated session'}"


# ---------------------------------------------------------------------------
# what the surfaces read off one snapshot
# ---------------------------------------------------------------------------


def alert_priority_key(alert: Any) -> tuple[str, str]:
    """`(bounce_type, SIDE)` for one M5 alert - derived exactly once.

    The SAME derivation `alert_center_panel._attach_held_run_suffix` already
    uses for the held x ran suffix: the first of the alert's own
    `feedback.bounce_types`, falling back to its trigger text. Deriving it
    twice is how the bar, the waiting list and the suffix would come to
    disagree about which cell a row belongs to.
    """
    payload = getattr(alert, "payload", None)
    feedback = payload.get("feedback") if isinstance(payload, Mapping) else None
    feedback = feedback if isinstance(feedback, Mapping) else {}
    bounce_type = str((feedback.get("bounce_types") or "").split(";")[0]).strip()
    if not bounce_type:
        bounce_type = str(getattr(alert, "trigger", "") or "").strip()
    return bounce_type, str(getattr(alert, "side", "") or "").strip().upper()


def _verdict_entries(payload: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    entries = (payload or {}).get("verdicts") or {}
    return {str(kind): entry for kind, entry in entries.items() if isinstance(entry, Mapping)}


def _ordered_cells(payload: Mapping[str, Any] | None, kind: str) -> list[Mapping[str, Any]]:
    """The kind's cells, best bound first - the order the switch reads.

    Off the PAYLOAD rather than off live objects, so the desk sorts by exactly
    the reading it is displaying and cannot drift from the strip above it.
    """
    cells = [cell for cell in payload_cells(payload) if str(cell.get("kind")) == kind]

    field, _margin = rank_basis(kind)

    def key(cell: Mapping[str, Any]):
        value = cell.get(field)
        return (
            value is None,
            -(float(value) if value is not None else 0.0),
            -int(cell.get("n_eligible") or 0),
            str(cell.get("family") or ""),
        )

    return sorted(cells, key=key)


def daytrade_order(payload: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """`[(bounce_type, SIDE)]`, best first - what the priority switch sorts by.

    Presentation only. It REORDERS and never withholds: every list that reads
    it shows the same rows in a different order, and a cell that is not in this
    list simply sorts after the ones that are.
    """
    return [
        (str(cell.get("family") or ""), str(cell.get("side") or "").upper())
        for cell in _ordered_cells(payload, "daytrade_held_run")
    ]


def swing_order(payload: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """`[(SIDE, family)]`, best first - the setups table's optional first key."""
    return [
        (str(cell.get("side") or "").upper(), str(cell.get("family") or ""))
        for cell in _ordered_cells(payload, "swing_trade_r")
    ]


def priority_rank(order: Sequence[tuple[str, str]] | None, key: tuple[str, str]) -> int:
    """Where `key` sits in `order`, or a rank after every named cell.

    Case-insensitive on both halves: the swing exports spell a side `LONG` and
    the intraday outcome log spells it `long`, and a sort that disagreed with
    itself about that would silently rank every short row last.
    """
    if not order:
        return 0
    wanted = (str(key[0] or "").strip().lower(), str(key[1] or "").strip().lower())
    for index, entry in enumerate(order):
        if (str(entry[0] or "").strip().lower(), str(entry[1] or "").strip().lower()) == wanted:
            return index
    return len(order)


def prioritise_enabled() -> bool:
    """The persisted switch, read AT SORT TIME and never at write time.

    Default OFF. Nothing about this flag reaches a write path: the backing
    lists, the evidence files and the repetition/movers filters are built
    before any sort and are byte-identical either way.
    """
    try:
        import project_paths

        return bool(project_paths.get_local_setting("prioritise_working_lately", False))
    except Exception:  # noqa: BLE001 - a display preference never costs a list
        return False


def _cell_for(payload: Mapping[str, Any] | None, verdict: Mapping[str, Any]) -> Mapping[str, Any]:
    leader = verdict.get("leader")
    return leader if isinstance(leader, Mapping) else {}


def observational_caveat(payload: Mapping[str, Any] | None, kind: str) -> str:
    """`observational leader among K cells` for ONE kind (advisory 1).

    K is that kind's own `cells_considered`, never a total across kinds: the
    swing leader was not chosen against the day-trade cells, they were never
    comparable, and `pool_cells` refuses to make them so. A summed K would
    overstate the search behind whichever leader it stood beside.
    """
    entry = _verdict_entries(payload).get(kind) or {}
    coverage = entry.get("coverage") or {}
    considered = coverage.get("cells_considered")
    if considered is None:
        considered = sum(
            1 for cell in payload_cells(payload) if str(cell.get("kind")) == kind
        )
    return f"{kind}: " + OBSERVATIONAL_CAVEAT.format(k=int(considered or 0))


def kind_phrase(payload: Mapping[str, Any] | None, kind: str, label: str) -> str:
    """One kind's sentence. PUBLIC - the banner prints the two the strip elides."""
    return _kind_phrase(payload, kind, label)


def _kind_phrase(payload: Mapping[str, Any] | None, kind: str, label: str) -> str:
    entry = _verdict_entries(payload).get(kind)
    if entry is None:
        return f"{label} - no reading"
    state = str(entry.get("state") or "")
    leader = _cell_for(payload, entry)
    if state in {"leader", "last_reliable_reading"} and leader:
        name = str(leader.get("leader_name") or "").strip()
        low = leader.get("uncertainty_low")
        low_text = "unmeasured" if low is None else f"{float(low):.3f}"
        n = int(leader.get("n_eligible") or 0)
        if kind == "daytrade_held_run":
            statistic = leader.get("statistic")
            head = f"held x ran {'unmeasured' if statistic is None else f'{float(statistic):.2f}'}"
            body = f"{head} (>= {low_text}, n={n})"
        else:
            statistic = leader.get("statistic")
            head = "unmeasured" if statistic is None else f"{float(statistic) * 100:.0f}%"
            sessions = int(leader.get("n_sessions") or 0)
            body = f"{head} (>= {low_text}, n={n}, {sessions} entry days)"
        if state == "last_reliable_reading":
            return (
                f"{label} - last reliable reading "
                f"{str(entry.get('as_of') or 'an unstated session')}: {name}, {body}"
            )
        return f"{label} - {name}, {body}"
    if state == "last_reliable_reading":
        return f"{label} - Last reliable reading: {entry.get('as_of') or 'an unstated session'}"
    # The STATE's own word, not one word for every state that is not a leader.
    # "no clear leader" and "no evidence" are different facts: the first says two
    # cells were too close, the second says nothing could be read at all.
    headline = "no evidence" if state == "no_evidence" else "no clear leader"
    return f"{label} - {headline} - {str(entry.get('reason') or 'no reason given')}"


def snapshot_line(payload: Mapping[str, Any] | None) -> str:
    """The one Working-lately line. The SAME string on every surface.

    Ends with the multiple-testing caveat, which is printed rather than
    corrected: K cells were read and the best of K was named, so the leader is
    OBSERVATIONAL. Nothing on this line is ever called proven.
    """
    from evidence_stats import LATELY_SESSIONS

    if not payload:
        return f"Working lately ({int(LATELY_SESSIONS)} sessions): no snapshot yet."
    cells = payload_cells(payload)
    window = int(
        next(
            (cell.get("window_sessions") for cell in cells if cell.get("kind") == "swing_trade_r"),
            LATELY_SESSIONS,
        )
        or LATELY_SESSIONS
    )
    # **EVERY kind, always** (re-review blocker 4b). The favorable line used to
    # be printed only when it named a leader, so a kind that was withheld left
    # no trace at all and the strip read as though the desk had two questions
    # instead of three. A withheld kind says WHY it is withheld; silence is the
    # one thing it may not say.
    parts = [
        _kind_phrase(payload, "swing_trade_r", "Swing"),
        _kind_phrase(payload, "swing_favorable", "Swing (favorable)"),
        _kind_phrase(payload, "daytrade_held_run", "Day"),
    ]
    parts.append(f"as of {payload.get('as_of') or 'an unstated session'}")
    parts.append(
        " / ".join(observational_caveat(payload, kind) for kind in SNAPSHOT_KINDS)
    )
    return f"Working lately ({window} sessions): " + " · ".join(parts)


def snapshot_sentence(payload: Mapping[str, Any] | None) -> str:
    """`snapshot_line` plus the identity - what a shared surface prints."""
    if not payload:
        return snapshot_line(payload)
    return f"{snapshot_line(payload)} [{snapshot_stamp(payload)}]"


def snapshot_cell_lines(payload: Mapping[str, Any] | None) -> list[str]:
    """Every cell, one line each - the strip's tooltip and the recap's detail."""
    names = {spec.name for spec in dataclass_fields(EvidenceCell)}
    lines: list[str] = []
    for kind in SNAPSHOT_KINDS:
        for cell in _ordered_cells(payload, kind):
            try:
                lines.append(
                    EvidenceCell(
                        **{key: value for key, value in cell.items() if key in names}
                    ).line()
                )
            except TypeError:
                # A payload written by an older build is still worth showing;
                # it is simply shown as the mapping it is.
                lines.append(", ".join(f"{key}={value}" for key, value in sorted(cell.items())))
    return lines
