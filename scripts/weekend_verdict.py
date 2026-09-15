"""The five-to-eight lines at the top of Weekend Prep — V2 item 2b.

Decision 0016 answer 10, in the trader's words: Weekend Prep's *"first screen is a
wall of text whose three CALLOUT lines are the only part that matters ... Wanted:
one Refresh for the whole tab, a short verdict card on top, tables not prose, ten
visible rows."*

This builds the card. Deterministic — no model, no network — and every line
carries the n it rests on, because a verdict with no sample size is an opinion.

**Every number here is REPORTED, never acted on.** Nothing in this module reaches
a detector, a score, an alert, a watchlist, Focus, the review queue or
`review_policy.json`. It reads what the week already recorded and writes one card.

The lines, in the order a Saturday reader wants them:

1. the take rate — how much of what the desk showed the trader acted on;
2. the blind spots, BY NAME — segments the trader passes on that go on to work;
3. the leaks, BY NAME — segments they take that do not;
4. the liked cohort that worked best over three sessions, with n;
5. the rejected cohort worth another look - the one that ran hardest after
   it was thrown away - over the same three sessions, with n;
6. the week's journal net P&L and win rate, **confirmed tags only**;
7. how many trades are waiting for a tag review.

A line whose inputs are missing SAYS SO rather than printing a zero. "No graded
likes yet" and "your likes averaged 0.00R" are different facts and the second one
is a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

#: The horizon the card reads, IN SESSIONS. Three is the desk's own
#: three-session forward mark - the shortest one that has had time to say
#: anything by Saturday - and it is the integer the rollup's `horizon_sessions`
#: column carries (1/3/5/10). It was the string `"h3"` until WS-5A, which is
#: why the card composed a column name out of it instead of comparing it.
CARD_HORIZON = 3

#: What the rollup writes on the row that pools both sides. A pooled row is
#: both sides added together, so it can be READ but it is never a reason to
#: act on and never leads a ranking of reasons. `BOTH` is carried too because
#: the pass and rejection rollups have used it.
POOLED_SIDES = frozenset({"ALL", "BOTH"})

#: A cohort under this is named and NOT ranked. `evidence_stats.MIN_REPORTABLE_N`
#: (30) is the floor for a REPORTABLE claim and stays the floor everywhere it
#: already rules; this card keeps its own 5, deliberately, because the card is a
#: pointer at a row the trader then opens the table to read, and because raising
#: it to 30 would silence the like line entirely on the live rollup (2 of 21
#: three-session side rows clear 30, against 8 that clear 5). Moving this floor
#: is a trader decision about what the card may say, not a repair.
MIN_COHORT_N = 5


@dataclass
class VerdictLine:
    """One line of the card, with what it rests on."""

    key: str
    text: str
    n: int | None = None
    measured: bool = True

    def rendered(self) -> str:
        if not self.measured:
            return self.text
        return f"{self.text} (n={self.n})" if self.n is not None else self.text


@dataclass
class Verdict:
    lines: list[VerdictLine] = field(default_factory=list)

    def add(self, key: str, text: str, *, n: int | None = None, measured: bool = True) -> None:
        self.lines.append(VerdictLine(key=key, text=text, n=n, measured=measured))

    def rendered(self) -> list[str]:
        return [line.rendered() for line in self.lines]


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def working_lately_line(payload: Mapping[str, Any] | None) -> VerdictLine | None:
    """The desk's shared Working-lately reading, with its identity - ST6.4.

    `None` when there is no snapshot, and the caller then prints NO line: the
    desk builds the snapshot on a desk session, so a weekend before the first
    build has nothing to report and an empty sentence in a capped card would
    cost a line that does.
    """
    import working_lately as wl

    if not payload:
        return None
    return VerdictLine(
        key="working_lately",
        text=f"{wl.snapshot_line(payload)} [{wl.snapshot_stamp(payload)}]",
        measured=True,
    )


def take_rate_line(state: Mapping[str, Any] | None) -> VerdictLine:
    """How much of what the desk showed, the trader acted on.

    READ FROM THE STATE, never recomputed (R4 A13). This added `takes + skips +
    rejects` and `build_review_learning_state` publishes NO `skips` and NO
    `rejects` key - `aggregate_dimensions` returns `episodes`, `shown`, `takes`,
    `overall_take_rate` and `dimensions`, and nothing else at the top level. So
    the denominator was `takes + 0 + 0`, the card printed "100% of 94 shown" on
    a week whose real answer was 30% of 318, and the one number the trader reads
    first said the opposite of the truth.

    `shown` is the scoreboard's own count - a chart the trader was shown, which
    is the denominator every other take-rate number on the desk uses - and
    `overall_take_rate` is its own rate rather than a second division here. Zero
    shown is not a zero take rate: it is a week with nothing to act on.
    """
    state = state or {}
    takes = _as_int(state.get("takes"))
    shown = _as_int(state.get("shown"))
    if shown <= 0:
        return VerdictLine(
            key="take_rate",
            text="Take rate: nothing was shown for review this week.",
            measured=False,
        )
    rate = _as_float(state.get("overall_take_rate"))
    if rate is None:
        rate = takes / shown
    return VerdictLine(
        key="take_rate",
        text=f"Take rate: {rate * 100:.0f}% of {shown} shown ({takes} taken)",
        n=shown,
    )


def _callout_line(state: Mapping[str, Any] | None, key: str, label: str, verb: str) -> VerdictLine:
    """One callout family, BY NAME rather than as an integer.

    "Blind Spots: 3" is a number a reader cannot act on. The scoreboard has
    always known WHICH segments and by how much.
    """
    items = list((state or {}).get(key) or ())
    if not items:
        return VerdictLine(
            key=key, text=f"{label}: none this week.", measured=False
        )
    named = []
    for item in items[:3]:
        if isinstance(item, Mapping):
            name = str(
                item.get("segment")
                or item.get("name")
                or item.get("dimension")
                or ""
            ).strip()
            named.append(name or "unnamed")
        else:
            named.append(str(item))
    more = f", +{len(items) - len(named)} more" if len(items) > len(named) else ""
    return VerdictLine(
        key=key,
        text=f"{label}: {', '.join(named)}{more} - {verb}",
        n=len(items),
    )


def best_cohort_line(
    rows: Iterable[Mapping[str, Any]],
    *,
    key: str,
    label: str,
    family: str,
    value_suffix: str = "",
    horizon: int = CARD_HORIZON,
    min_n: int = MIN_COHORT_N,
) -> VerdictLine:
    """The cohort with the HIGHEST side-adjusted return at one horizon, with n.

    WS-5A. This read `avg_r_{horizon}` and `n_{horizon}` - columns nothing on
    the desk has ever written - off rows whose return was already a formatted
    percent string. Every cell was skipped, so both lines printed "nothing with
    enough behind it yet" against 115 graded rejection rows and 129 graded like
    rows; and the first row that had matched would have printed a PERCENT with
    an `R` after it. The contract is now numeric and explicit:

    - ``horizon_sessions`` is compared to `CARD_HORIZON` as an INT, so a better
      number at another horizon cannot leak into a three-session claim;
    - ``side`` must be a real side. `ALL` is what the rollup writes for the
      pooled row (both sides added together) and it is not a reason anyone can
      act on, so it is excluded from a ranking of reasons and named in neither
      line;
    - ``avg_side_return_pct`` is a PERCENT number and is printed as a percent.

    Polarity: HIGHEST, on both lines. For likes that is "which claim worked".
    For rejections it is the one worth questioning - the name that went on to
    run after it was thrown away. `min()` named the rejection that was RIGHT,
    which is the one reading a trader never has to do anything about.

    Rows under `min_n` are EXCLUDED and the line says the best n it saw against
    the floor, so a thin week reads as thin rather than as a confident answer
    resting on two observations.
    """
    usable: list[tuple[str, float, int]] = []
    best_thin_n = 0
    thin = 0
    off_horizon = 0
    for row in rows or ():
        if _as_int(row.get("horizon_sessions")) != int(horizon):
            off_horizon += 1
            continue
        side = str(row.get("side") or "").strip()
        if not side or side.upper() in POOLED_SIDES:
            continue
        value = _as_float(row.get("avg_side_return_pct"))
        count = _as_int(row.get("n"))
        name = str(
            row.get("cohort") or row.get("source") or row.get("reason_code") or ""
        ).strip()
        if value is None or not name:
            continue
        if count < min_n:
            thin += 1
            best_thin_n = max(best_thin_n, count)
            continue
        usable.append((f"{name} {side}", value, count))
    if not usable:
        if not thin:
            # Three absences, three sentences. "Never graded", "graded but only
            # at another horizon" and "graded and too thin to rank" are
            # different facts, and printing one of them for another is the
            # class of false statement this whole packet repaired.
            if off_horizon:
                return VerdictLine(
                    key=key,
                    text=(
                        f"{label}: nothing has matured to {int(horizon)} sessions "
                        f"yet ({off_horizon} row(s) at other horizons)."
                    ),
                    measured=False,
                )
            return VerdictLine(
                key=key,
                text=f"{label}: no {family} cohorts measured yet.",
                measured=False,
            )
        return VerdictLine(
            key=key,
            text=(
                f"{label}: nothing with enough behind it yet "
                f"(best n was {best_thin_n} against a floor of {min_n})."
            ),
            measured=False,
        )
    name, value, count = max(usable, key=lambda item: (item[1], item[2], item[0]))
    return VerdictLine(
        key=key, text=f"{label}: {name} {value:+.2f}%{value_suffix}", n=count
    )


def journal_week_line(trades: Iterable[Mapping[str, Any]]) -> VerdictLine:
    """Net P&L and win rate for the week — CONFIRMED TAGS ONLY.

    Confirmed only, because "my setups" counts confirmed tags and a card that
    blended the tagger's provisional guesses into the trader's own record would
    be reporting the machine's opinion as the trader's week.
    """
    rows = [
        row
        for row in (trades or ())
        if str(row.get("tag_status") or "") == "confirmed"
        and str(row.get("setup_tags") or "").strip()
    ]
    values = [_as_float(row.get("net_pnl")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return VerdictLine(
            key="journal_week",
            text="This week's trades: none with a confirmed tag yet.",
            measured=False,
        )
    wins = sum(1 for value in values if value > 0)
    return VerdictLine(
        key="journal_week",
        text=(
            f"This week (confirmed tags): {sum(values):+,.2f} net, "
            f"{wins / len(values) * 100:.0f}% win rate"
        ),
        n=len(values),
    )


def awaiting_review_line(count: int) -> VerdictLine:
    """How many trades the nightly tagger left for the trader."""
    total = max(0, _as_int(count))
    if total <= 0:
        return VerdictLine(
            key="awaiting_review",
            text="Nothing waiting for a tag review.",
            measured=False,
        )
    return VerdictLine(
        key="awaiting_review",
        text=f"{total} trade(s) waiting for a tag review - Trades tab, Provisional filter",
        n=total,
    )


def research_line(pack: Mapping[str, Any] | None) -> VerdictLine:
    """The nightly fact pack's headline, on a surface the trader opens - V3 item 5.

    Decision 0016: *"The Research tab is the builder's surface, not the trader's.
    Nothing the trader must see may live only there."* The pack's eligible-cell
    count and its best cell were reachable only from Research, which the trader
    never opens - so the answer to "did the overnight research find anything?"
    was a screen away from every screen they use.

    ONE LINE, and the full panel stays in Research. This is a pointer with a
    number on it, not a second readout.

    Eligible cells only, and it says DISCOVERY: a cell that has cleared the
    evidence floor has still not closed its registered window, and the ledger -
    not this card - is what says when it may be read for a verdict.
    """
    block = (pack or {}).get("gate") or {}
    cells = list((pack or {}).get("eligible_policies") or ())
    if not cells:
        # R4 B1: the OLDER pack shape. `eligible_policies` was added on
        # 2026-09-01; every pack before it carries the same cells under
        # `policies` with eligibility at `cell["stats"]["eligible"]`, and the two
        # lists are the same thing - on the live 2026-09-01 pack,
        # `eligible_policies` is exactly the 33 of 73 `policies` whose stats say
        # eligible. Falling through to "no cell has cleared the floor" for a pack
        # that measured nine of them states a different fact, and the wrong one.
        cells = [
            cell
            for cell in ((pack or {}).get("policies") or ())
            if isinstance(cell, Mapping) and (cell.get("stats") or {}).get("eligible")
        ]
    count = _as_int(block.get("eligible_policy_cells") or len(cells))
    if not count or not cells:
        return VerdictLine(
            key="research",
            text="Research: no cell has cleared the evidence floor yet.",
            measured=False,
        )
    best = max(
        cells,
        key=lambda cell: (
            _as_float(((cell.get("stats") or {}).get("clipped") or {}).get("trimmed_mean"))
            or _as_float(cell.get("trimmed_mean_r"))
            or float("-inf")
        ),
    )
    stats = best.get("stats") or {}
    mean = _as_float((stats.get("clipped") or {}).get("trimmed_mean"))
    if mean is None:
        mean = _as_float(best.get("trimmed_mean_r"))
    n = _as_int(stats.get("n") or best.get("n") or 0)
    name = " ".join(
        str(best.get(key) or "").strip()
        for key in ("family", "side", "recipe_id")
        if str(best.get(key) or "").strip()
    )
    mean_text = f"{mean:+.2f}R" if mean is not None else "unmeasured"
    return VerdictLine(
        key="research",
        text=f"Research: {count} eligible cell(s), best {name} {mean_text} - discovery",
        n=n or count,
    )


def build_verdict(
    *,
    learning_state: Mapping[str, Any] | None = None,
    like_rows: Iterable[Mapping[str, Any]] = (),
    veto_rows: Iterable[Mapping[str, Any]] = (),
    week_trades: Iterable[Mapping[str, Any]] = (),
    awaiting_review: int = 0,
    research_pack: Mapping[str, Any] | None = None,
    horizon: int = CARD_HORIZON,
    working_lately: Mapping[str, Any] | None = None,
) -> Verdict:
    """The whole card. Pure: every input is passed in, nothing is read here.

    Pure on purpose - the panel reads the stores on its worker and hands the
    rows over, so this is testable without a journal, a lake or a Qt event loop.

    `working_lately` is the desk's SHARED evidence snapshot (ST6.4). The card
    prints the same line and the same `snapshot_id` as the strip above the M5
    list and as the Setup Tracker banner, so the weekend read and the weekday
    read can be reconciled instead of compared from memory.
    """
    verdict = Verdict()
    # The head line, and ONLY when there is a snapshot to print. The card is
    # capped at eight lines by the trader ("a card, not a second wall of text")
    # and a weekend before the desk has ever built a snapshot has nothing to say
    # here - an absent reading is not a line, it is silence.
    shared = working_lately_line(working_lately)
    if shared is not None:
        verdict.lines.append(shared)
    verdict.lines.append(take_rate_line(learning_state))
    verdict.lines.append(
        _callout_line(learning_state, "blind_spots", "Blind spots", "shown, passed on, then worked")
    )
    verdict.lines.append(
        _callout_line(learning_state, "leaks", "Leaks", "taken, and did not work")
    )
    # WS-5A. Eight lines is the cap the trader set and these two are lines five
    # and six of eight (take rate, blind spots, leaks, THESE TWO, the week's
    # record, the tag backlog, research) - so there is no room for a third
    # cohort line and "Rejections that were right" is not printed. It is the
    # reading a trader never has to act on, and it would cost a line that does.
    verdict.lines.append(
        best_cohort_line(
            like_rows,
            key="best_like",
            label="Likes that work",
            family="like",
            value_suffix=f" over {int(horizon)} sessions",
            horizon=horizon,
        )
    )
    verdict.lines.append(
        best_cohort_line(
            veto_rows,
            key="veto_second_look",
            label="Rejections worth another look",
            family="veto",
            value_suffix=" side-adjusted",
            horizon=horizon,
        )
    )
    verdict.lines.append(journal_week_line(week_trades))
    verdict.lines.append(awaiting_review_line(awaiting_review))
    verdict.lines.append(research_line(research_pack))
    return verdict
