"""Win rate leads every trader-facing swing surface — V3 item 1.

Decision 0016 answer 3, in the trader's words: *"the D1 support (long) or
resistance (short) level holds, then the move follows. The trader gives swings
room; losses run about 1.5x the best wins, so **win rate is the number that
matters**, not average R."*

That is a statement about the trader's own exits, and it has a consequence the
desk had not applied: a surface that leads with mean R is ranking their swings by
a statistic their loss profile makes misleading. So win rate goes FIRST on every
trader-facing swing surface, and **mean R stays beside it, never replaced** - the
two answer different questions and dropping either is how a number starts lying.

**Every win rate here carries three things.**

* `n` - the number of graded episodes behind it. A rate without one is not a
  statistic.
* The **Wilson lower bound** at 95%. A raw 100% on three trades and a 62% on
  ninety are the same number to a reader skimming a column; their lower bounds
  are 44% and 52%, which is the honest ordering. Wilson rather than the normal
  approximation because the normal one is meaningless near 0 and 1, which is
  exactly where a thin cell sits.
* A **floor flag**, from `evidence_stats.MIN_REPORTABLE_N` - the desk's one
  statistics contract, not a second threshold invented here.

**Sorting is by the LOWER BOUND, never by the raw rate.** That is the whole point
of computing it: ranking by the raw rate puts the three-trade cell on top of the
ninety-trade one every time.

Pure arithmetic. No Qt, no network, no store reads - the callers pass rows in.
Nothing here reaches a detector, a score that gates, an alert, a watchlist, Focus,
the review queue or `review_policy.json`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from evidence_stats import (  # noqa: F401 - SWING_HORIZON_SESSIONS is RE-EXPORTED (ST6), never redefined
    LATELY_SESSIONS,
    MIN_REPORTABLE_N,
    SWING_HORIZON_SESSIONS,
)

#: `evidence_stats` owns the statistics contract; this module owns the
#: headline. A caller that needs the declared horizon beside the Wilson bound
#: should not have to import two modules to get one policy - and a second
#: literal is exactly how the desk grew two answers to "5 what?" (ST1).

#: The confidence level for the lower bound. 95% two-sided, so z = 1.96.
#:
#: **THIS IS THE ONE z FOR EVERY TRADER-FACING WIN RATE** (R4 B3). The desk has
#: a second Wilson - `master_avwap_lib/expected_r.wilson_lower_bound`, default
#: z = 1.28, ~90% one-sided - and the two must never appear on one screen: a
#: reader comparing "at least 52%" on one table with "at least 55%" on another
#: is comparing two different questions and has no way to know it.
#:
#: The other one is deliberately left where it is. It is not a column anybody
#: reads; it is a PARAMETER of the Expected-R model's proven-quality score
#: (`DEFAULT_PQS_CONFIG["wilson_z"]`), which is calibrated end to end and lives
#: in a fenced scoring file. Changing it would move every Expected R on the desk
#: and needs a golden fixture and a sec-7 promotion, which is a different packet
#: from putting a win rate on a table. What B3 requires - and what
#: `test_r4b_swing_headline_wired.py` asserts - is that no trader-facing surface
#: reaches for it.
WILSON_Z = 1.959963984540054

#: The column order every trader-facing swing surface uses. WIN RATE FIRST, and
#: the robust columns beside it rather than instead of it.
HEADLINE_COLUMNS = ("win_rate", "win_rate_lb", "n", "avg_r", "avg_unit", "meets_floor")
HEADLINE_LABELS = ("Win %", "Win % (low)", "n", "Avg", "unit", "")

#: ST1 item 1. **A PERCENT MOVE IS NOT A WIN RATE.** The tracker's `win` column
#: is the sign of a close-to-close move measured at a scan-row offset; the
#: stop-at-a-level, two-closes grading lives on the tracker JSON scenarios, a
#: different export entirely. A headline built from the first says "favorable";
#: only a headline built from graded R says "win rate".
OUTCOME_KIND_FAVORABLE_DIRECTION = "favorable_direction"
OUTCOME_KIND_TRADE_R = "trade_r"

#: The same six columns, headed the way the outcome kind earns.
FAVORABLE_HEADLINE_LABELS = ("Favorable %", "Favorable % (low)", "n", "Avg", "unit", "")


def headline_labels(outcome_kind: str = OUTCOME_KIND_TRADE_R) -> tuple[str, ...]:
    """The column labels for one outcome kind. Never a local string on a surface."""
    if str(outcome_kind or "").strip() == OUTCOME_KIND_FAVORABLE_DIRECTION:
        return FAVORABLE_HEADLINE_LABELS
    return HEADLINE_LABELS


@dataclass(frozen=True)
class Headline:
    """One family's swing record, as every surface shows it."""

    name: str
    wins: int
    losses: int
    #: Graded rows that came back EXACTLY FLAT - R4 B6. A scratch is not a loss.
    #: `headline_from_outcomes` counted `close_r == 0.0` as a loss (`value > 0`
    #: else loss), which understates every rate it touches, and the more
    #: disciplined the exit the more of them there are. It is a MEASURED outcome,
    #: so it stays in `avg_r`; it simply has no answer to the win/loss question
    #: and is kept out of `n`, which is that question's denominator.
    flats: int = 0
    avg_r: float | None = None
    sessions: int = LATELY_SESSIONS
    #: What `avg_r` IS. The tracker grades in percent move and the recipe grids
    #: grade in R, and a column headed "Avg R" showing a percent is exactly the
    #: kind of number this packet exists to stop. Set by whichever constructor
    #: knows, never guessed from the magnitude.
    avg_unit: str = "R"
    #: What the RATE is - ST1 item 1. `trade_r` is a win rate over graded trades;
    #: `favorable_direction` is the share of favorable close-to-close moves, and
    #: is never printed as a win rate. Set by the constructor that knows which
    #: file it read, never guessed.
    outcome_kind: str = OUTCOME_KIND_TRADE_R

    @property
    def n(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float | None:
        return (self.wins / self.n) if self.n else None

    @property
    def win_rate_lb(self) -> float | None:
        return wilson_lower_bound(self.wins, self.n)

    @property
    def meets_floor(self) -> bool:
        return self.n >= MIN_REPORTABLE_N

    def as_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "win_rate": self.win_rate,
            "win_rate_lb": self.win_rate_lb,
            "n": self.n,
            "flats": self.flats,
            "avg_r": self.avg_r,
            "avg_unit": self.avg_unit,
            "meets_floor": self.meets_floor,
            "sessions": self.sessions,
            "outcome_kind": self.outcome_kind,
        }

    @property
    def is_favorable_direction(self) -> bool:
        return self.outcome_kind == OUTCOME_KIND_FAVORABLE_DIRECTION

    def sentence(self) -> str:
        """One line for a setup doc or a card. Says when it cannot say.

        `setup_docs` renders this at READ TIME from the tracker file, so a doc
        never carries a hardcoded number that quietly ages.

        **ST1 item 1: the words follow the outcome kind.** A favorable-direction
        headline says "favorable moves" and names its own basis - a
        price-direction rate measured close to close - because calling it a win
        rate claims a stop rule that was never applied.
        """
        if not self.n:
            # Unchanged for both kinds: "nothing graded" makes no claim about a
            # rate, so it needs no relabelling and every caller keeps its words.
            return f"{self.name}: no graded swings in the last {self.sessions} sessions."
        rate = self.win_rate or 0.0
        bound = self.win_rate_lb or 0.0
        tail = "" if self.meets_floor else f" - under the n={MIN_REPORTABLE_N} floor, so read it as discovery"
        mean = (
            f", avg {self.avg_r:+.2f}{self.avg_unit}" if self.avg_r is not None else ""
        )
        if self.is_favorable_direction:
            return (
                f"{self.name}: {rate * 100:.0f}% favorable moves over the last "
                f"{self.sessions} sessions (n={self.n}, at least {bound * 100:.0f}%{mean})"
                f"{tail} - a price-direction rate measured close to close, not a "
                f"stop-rule verdict."
            )
        return (
            f"{self.name}: {rate * 100:.0f}% win rate over the last {self.sessions} "
            f"sessions (n={self.n}, at least {bound * 100:.0f}%{mean}){tail}."
        )


def wilson_lower_bound(wins: int, n: int, *, z: float = WILSON_Z) -> float | None:
    """The 95% Wilson lower bound on a win rate, or None when n is 0.

    Wilson rather than `p - z*sqrt(p(1-p)/n)`: the normal approximation gives a
    lower bound of exactly p when p is 0 or 1, which is the one place a reader
    most needs the interval to say something. Wilson stays inside [0, 1] and
    keeps widening as n shrinks.
    """
    total = int(n or 0)
    if total <= 0:
        return None
    hits = max(0, min(int(wins or 0), total))
    phat = hits / total
    denominator = 1.0 + (z * z) / total
    centre = phat + (z * z) / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)
    return max(0.0, (centre - margin) / denominator)


def headline_from_counts(
    name: str,
    *,
    wins: Any,
    losses: Any,
    flats: Any = 0,
    avg_r: Any = None,
    sessions: int = LATELY_SESSIONS,
) -> Headline:
    """A headline from the integer counts a store actually observed.

    ``flats`` is optional and additive (ST2 fix round): a store that counts an
    exactly-flat outcome separately - the tracker's exports do since ST2.1 -
    passes it, and `Headline` keeps it out of `n` while leaving it a MEASURED
    outcome. A caller that has no flat count keeps today's behaviour exactly.
    """

    def _int(value: Any) -> int:
        try:
            return max(0, int(float(value)))
        except (TypeError, ValueError):
            return 0

    def _float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return None if number != number else number

    return Headline(
        name=str(name or ""),
        wins=_int(wins),
        losses=_int(losses),
        flats=_int(flats),
        avg_r=_float(avg_r),
        sessions=int(sessions),
    )


def headline_from_rate(
    name: str,
    *,
    win_rate: Any,
    n: Any,
    avg_r: Any = None,
    avg_unit: str = "R",
    sessions: int = LATELY_SESSIONS,
) -> Headline:
    """A headline from a STORED rate and count, for a surface that has no rows.

    **NEVER hand this a weighted rate.** Rebuilding wins as `round(rate * n)` is
    exact if and only if the stored rate was computed as `wins / n` over exactly
    those `n` rows. Hand it a recency-weighted, regime-weighted or otherwise
    reweighted mean and it returns an integer pair NOBODY OBSERVED, with a
    Wilson bound computed from the invented pair - which is what the Setup
    Tracker's recent-types table printed until ST2 (2026-09-06): two 28-day-old
    wins at weight .25 and two same-day losses at weight 1.0 gave 0.2, and the
    cell read `25% (>=5%, n=4)` where the truth was 2-2 and 50%.

    **The legitimate callers, each of which writes `wins / n`:**

    * the veto cohort CSV and the like cohort CSV (`review_learning`'s pooled
      performance rows), read by the Weekend Prep cohort tables;
    * `legacy.build_tracker_short_horizon_rows`' `win_rate_2d`, a plain
      unweighted mean of `1.0 if r_close_2d > 0 else 0.0` over exactly
      `samples_2d` values, adapted in `working_lately.short_term_evidence_rows`.

    The tracker's recent-types export is NO LONGER one of them: it exports
    `n_wins` / `n_losses` and its readers use `headline_from_counts`.

    An n of 0 gives a headline with no rate at all rather than a zero: "nothing
    graded" and "graded and lost everything" are different facts.
    """
    try:
        total = max(0, int(float(n)))
    except (TypeError, ValueError):
        total = 0
    try:
        rate = float(win_rate)
    except (TypeError, ValueError):
        rate = float("nan")
    if total <= 0 or rate != rate:
        return Headline(name=str(name or ""), wins=0, losses=0, sessions=sessions)
    wins = max(0, min(total, int(round(min(max(rate, 0.0), 1.0) * total))))
    return Headline(
        name=str(name or ""),
        wins=wins,
        losses=total - wins,
        avg_r=_optional_float(avg_r),
        avg_unit=avg_unit,
        sessions=sessions,
    )


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def headline_from_outcomes(
    name: str,
    outcomes: Iterable[Mapping[str, Any]],
    *,
    r_field: str = "close_r",
    sessions: int = LATELY_SESSIONS,
) -> Headline:
    """Count wins and losses from graded rows.

    A row whose R cannot be read is counted in NEITHER - it is unmeasured, not a
    loss, and folding it into the denominator would drift every rate downward by
    however much the data is missing.
    """
    wins = losses = flats = 0
    values: list[float] = []
    for row in outcomes or ():
        try:
            value = float(row.get(r_field))
        except (TypeError, ValueError):
            continue
        if value != value:
            continue
        values.append(value)
        if value > 0:
            wins += 1
        elif value < 0:
            losses += 1
        else:
            # R4 B6. A FLAT IS NEITHER. This branch used to be the loss branch,
            # so every scratched trade was scored as a loser - and unmeasured and
            # flat are different facts, which is why an unreadable row is
            # `continue`d above and a 0.0 is counted here.
            flats += 1
    average = sum(values) / len(values) if values else None
    return Headline(
        name=str(name or ""),
        wins=wins,
        losses=losses,
        flats=flats,
        avg_r=average,
        sessions=sessions,
    )


def headline_from_tracker_rows(
    name: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    sessions: int = LATELY_SESSIONS,
) -> Headline:
    """Count from the tracker's own `win` column and `side_return_pct`.

    **ST1 item 1 - what that column actually is.** This docstring used to say
    the tracker "decides what a win is - its stop-at-a-level, two-closes rule".
    That was false, and the wording was how a percent move became a win rate on
    four surfaces. `master_avwap_tier_outcomes.csv`'s `win` is
    `side_return_pct > 0`: the SIGN OF A CLOSE-TO-CLOSE PERCENT MOVE between two
    of that symbol's own scan rows (`legacy.build_scan_factor_observation_rows`).
    A long that broke its D1 support intraday and closed higher counts as a
    `win` there; a stop-at-a-level rule would have been out. The stop/two-closes
    grading is real but lives on the tracker JSON scenarios
    (`_summarize_tracker_setup_outcome`), a different export with a different
    grain.

    So the headline this builds declares `favorable_direction`, and every
    surface fed by it says "favorable" rather than "win rate". The NUMBER is
    unchanged; only the claim it makes is.

    A row whose `win` cannot be read is counted in NEITHER: unmeasured is not a
    loss, and folding it into the denominator drifts every rate downward by
    however much the data is missing.
    """
    wins = losses = 0
    returns: list[float] = []
    for row in rows or ():
        verdict = str(row.get("win") if row.get("win") is not None else "").strip().lower()
        if verdict in {"1", "true", "yes", "win"}:
            wins += 1
        elif verdict in {"0", "false", "no", "loss"}:
            losses += 1
        else:
            continue
        try:
            value = float(row.get("side_return_pct"))
        except (TypeError, ValueError):
            continue
        if value == value:
            returns.append(value)
    average = sum(returns) / len(returns) if returns else None
    return Headline(
        name=str(name or ""),
        wins=wins,
        losses=losses,
        avg_r=average,
        sessions=sessions,
        # PERCENT MOVE, not R: `side_return_pct` is what the tracker grades in.
        avg_unit="%",
        outcome_kind=OUTCOME_KIND_FAVORABLE_DIRECTION,
    )


def rank(headlines: Iterable[Headline]) -> list[Headline]:
    """Best first, BY THE LOWER BOUND. Unmeasured families sort last.

    Ranking by the raw rate puts a 100%-on-three-trades cell above a
    62%-on-ninety every time, which is the failure this column exists to stop.
    """
    def _key(item: Headline):
        bound = item.win_rate_lb
        return (bound is None, -(bound or 0.0), -item.n)

    return sorted(headlines, key=_key)


def as_rows(headlines: Iterable[Headline]) -> list[dict[str, Any]]:
    """Ranked rows, ready for a table that uses `HEADLINE_COLUMNS`."""
    return [item.as_row() for item in rank(headlines)]


def format_win_rate(row: Mapping[str, Any]) -> str:
    """`62% (>=52%, n=90)`, or a dash. The one spelling every surface uses.

    ST1 item 1: a favorable-direction row reads `62% favorable (>=52%, n=90)`.
    The word comes from the row's own `outcome_kind`, so a surface can never
    relabel a rate with a local string - and an old row without the key reads as
    a trade-R win rate, which is what every caller of this function meant before
    the column existed.
    """
    rate = row.get("win_rate")
    if rate is None:
        return "-"
    bound = row.get("win_rate_lb")
    count = row.get("n") or 0
    tail = f">={bound * 100:.0f}%, " if bound is not None else ""
    kind = str(row.get("outcome_kind") or "").strip()
    noun = " favorable" if kind == OUTCOME_KIND_FAVORABLE_DIRECTION else ""
    return f"{float(rate) * 100:.0f}%{noun} ({tail}n={count})"
