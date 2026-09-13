"""Packet WS-10I - thesis, context, opportunity and trade share ONE identity/time contract.

Written by the TESTER on `claude/ws-10i-context-join`, off
`origin/claude/wishlist-sweep-2026-09-12`, and proven RED there before any of it
existed. The builder makes these pass; it may ADD tests and may not weaken, skip
or delete one.

WISHLIST 10I (plus 10K's "measure environment, then connect it"), as the packet
rules it: the state known at OPPORTUNITY OBSERVATION and the state known at
ACTUAL ENTRY are attached SEPARATELY and carried side by side, each with its
context id, rule version, benchmark, `observed_at`, `available_at` and a named
match CERTAINTY; a thesis links by SCOPE and VALIDITY WINDOW, never by
ticker/date alone; one thesis -> many opportunities, one opportunity -> several
decisions/fills, and MONEY IS COUNTED ONCE PER TRADE; ambiguous stays ambiguous;
a date-only fill cannot select a midday regime; theta, day trades and swings are
never pooled; every cell carries n, distinct sessions, distinct symbols,
coverage, window, outcome definition and uncertainty through `evidence_stats`,
against the setup's own baseline. **Shadow evidence only** - zero detector,
score, alert, watchlist, Focus or promotion influence.

===========================================================================
THE API THESE TESTS PIN
===========================================================================

`scripts/context_join.py` - PURE (rows and label tables in, rows and frozen
tuples out; no store opened except by the backfill CLI)::

    CERTAINTY_SESSION        = "session"
    CERTAINTY_PRIOR_SESSION  = "prior_session"
    CERTAINTY_DATE_ONLY      = "date_only"
    CERTAINTY_RECONSTRUCTED  = "reconstructed"
    CERTAINTY_UNKNOWN        = "unknown"
    UNKNOWN_LABEL            = "unknown"

    OBSERVATION_CONTEXT_FIELD = "observation_context"
    ENTRY_CONTEXT_FIELD       = "entry_context"
    LINKED_THESES_FIELD       = "linked_theses"

    ContextRef(context_id, rule_version, benchmark, label, observed_at,
               available_at, certainty)                       # frozen
        .flagged -> bool      # True for date_only / reconstructed / unknown,
                              # False for session / prior_session

    attach_context(rows, *, when, labels_by_session, clock_field,
                   benchmark="SPY", rule_version=...) -> rows   # IN PLACE
        # when="observation" -> writes OBSERVATION_CONTEXT_FIELD
        # when="entry"       -> writes ENTRY_CONTEXT_FIELD
    link_theses(rows, theses, *, when="observation") -> rows    # IN PLACE
    recap_context_labels(row) -> dict                           # WS-DR rows
    backfill_refs(rows, *, labels_by_session, clock_field, since="")
        -> list[ContextRef]                                     # all reconstructed
    main(argv) -> int                                           # `backfill`, DRY by default

**The time rule, which is the whole packet.** A D1 label for session S is
available at S's CLOSE. So:

* a row whose clock lands BEFORE that close takes the PREVIOUS EXCHANGE
  SESSION's label, `certainty=prior_session`;
* a row whose clock is a bare date, or lands at/after the close, takes S's own
  label, `certainty=session`;
* a journal fill stamped MIDNIGHT market-local carries no time of day
  (`journal_trade_shape.is_date_only`), so it takes the previous completed
  session's label, `certainty=date_only`, FLAGGED - it may never be given the
  label of the session it sits on, which is a midday regime it could not know;
* a session nobody labelled reads `unknown`, `certainty=unknown`, pooled into
  nothing.

The previous session is walked on the EXCHANGE CALENDAR, never in calendar
days: the fixtures below are built around **Monday 2026-09-07, Labor Day**, so
an implementation that subtracts one day from 2026-09-08 lands on a holiday
with no label and fails.

Aware and naive both land in market-local through `journal_trade_shape`'s own
coercion - a naive stamp is ATTACHED, an aware one CONVERTED, and the fixture
carries one of each for the same instant so an offset-stripping read fails.

`scripts/setup_environment_evidence.py`::

    opportunity_cells(rows, *, environment_key="d1_environment",
                      population_kind="swing", labels_by_session=None,
                      date_field="scan_date") -> list[dict]
    personal_cells(trades, *, labels_by_session, horizon="swing",
                   benchmark="SPY") -> dict
    thesis_review(theses, *, opportunity_rows, trades, benchmark_paths,
                  as_of) -> list[dict]

`scripts/research_results.py`::

    ENVIRONMENT_ALL = "all"
    build_results_view(..., environment_filter=ENVIRONMENT_ALL)
    ResultsView.environment_filter / .environment_basis / .environment_line

`scripts/ui/panels/research_results_panel.py`::

    _read((population, horizon, window_key, environment), window, payload)
        # the 3-tuple selection still works; the filter reaches the view

===========================================================================
WHAT DEPENDS ON WS-DR
===========================================================================

`test_a_daily_recap_row_carries_both_context_labels_side_by_side` pins the pure
seam the Daily Recap renders (`context_join.recap_context_labels`), against the
packet's stated row contract - an observation-context label on every recap row
and an entry-context label BESIDE it on a matched trade. The function is pure
(a row dict in, a mapping out) and is buildable today; only the RENDERING of it
in the Daily Recap needs WS-DR on the branch.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# the calendar the fixtures are built on
# ---------------------------------------------------------------------------

#: Monday 2026-09-07 is LABOR DAY. The sessions around it, in order:
#:     Fri 2026-09-04 | (Mon 2026-09-07 CLOSED) | Tue 2026-09-08 | Wed 2026-09-09
#: so `previous_session(2026-09-08)` is 2026-09-04 and not 2026-09-07.
LABELS = {
    "2026-09-02": "mixed",
    "2026-09-03": "mixed",
    "2026-09-04": "trending_up",
    # NOTE: 2026-09-07 is deliberately ABSENT - it was never a session.
    "2026-09-08": "compressed",
    "2026-09-09": "mixed",
    "2026-09-10": "trending_down",
    "2026-09-11": "compressed",
}


def test_the_fixture_calendar_is_the_exchange_calendar_and_not_a_weekday_count():
    """A guard on the fixtures themselves, so a failure below is never the
    calendar's. Labor Day 2026 falls on the Monday between the two sessions the
    prior-session tests hinge on."""
    import market_calendar

    assert market_calendar.is_session(date(2026, 9, 7)) is False
    assert market_calendar.is_session(date(2026, 9, 8)) is True
    assert market_calendar.previous_session(date(2026, 9, 8)) == date(2026, 9, 4)


# ---------------------------------------------------------------------------
# item 1 - two contexts per row, each with its certainty named
# ---------------------------------------------------------------------------


def test_a_mid_session_m5_row_gets_the_prior_sessions_label_with_the_certainty_named():
    """A 10:35 decision on 2026-09-08 cannot know what kind of day 2026-09-08
    turned out to be - that label is published at the close. It gets the label
    of the previous EXCHANGE session (2026-09-04, because 09-07 was Labor Day),
    and the ref says `prior_session` out loud."""
    import context_join

    rows = [
        # The M5 outcome log's own spelling: a naive market-local stamp.
        {"event_id": "AAPL_long_20260908_10_35_00_h1_blue_after_red",
         "trade_date": "2026-09-08",
         "entry_time": "2026-09-08 10:35:00",
         "symbol": "AAPL", "direction": "long"},
    ]
    context_join.attach_context(
        rows, when="observation", labels_by_session=LABELS, clock_field="entry_time"
    )
    ref = rows[0][context_join.OBSERVATION_CONTEXT_FIELD]

    assert ref.certainty == context_join.CERTAINTY_PRIOR_SESSION
    assert ref.label == "trending_up"
    # The two wrong answers, named so a regression says which one it made:
    assert ref.label != "compressed", "took 2026-09-08's own label, which was not published yet"
    assert ref.label != context_join.UNKNOWN_LABEL, "walked back one CALENDAR day onto Labor Day"
    assert ref.flagged is False
    assert ref.benchmark == "SPY"
    assert ref.observed_at == "2026-09-08 10:35:00"
    # Available at the close of the session the label is ABOUT, aware and ET.
    assert ref.available_at.startswith("2026-09-04T16:00:00")
    assert ref.available_at.endswith(("-04:00", "-05:00"))
    # The entry ref is not invented from an observation row.
    assert context_join.ENTRY_CONTEXT_FIELD not in rows[0]


def test_an_eod_swing_row_gets_its_own_sessions_label():
    """A swing scan row is decided on the session's own closed bars, so the
    label of that session IS available to it. Bare date in, `session` out."""
    import context_join

    rows = [{"observation_id": "obs-1", "scan_date": "2026-09-08", "symbol": "MSFT"}]
    context_join.attach_context(
        rows, when="observation", labels_by_session=LABELS, clock_field="scan_date"
    )
    ref = rows[0][context_join.OBSERVATION_CONTEXT_FIELD]

    assert ref.certainty == context_join.CERTAINTY_SESSION
    assert ref.label == "compressed"
    assert ref.flagged is False
    assert ref.available_at.startswith("2026-09-08T16:00:00")

    # A session the store never labelled is `unknown` and carries its own
    # certainty - it is not pooled into a measured cell and not guessed at.
    blank = [{"observation_id": "obs-2", "scan_date": "2026-09-07", "symbol": "MSFT"}]
    context_join.attach_context(
        blank, when="observation", labels_by_session=LABELS, clock_field="scan_date"
    )
    unlabelled = blank[0][context_join.OBSERVATION_CONTEXT_FIELD]
    assert unlabelled.label == context_join.UNKNOWN_LABEL
    assert unlabelled.certainty == context_join.CERTAINTY_UNKNOWN
    assert unlabelled.flagged is True


def test_an_aware_and_a_naive_stamp_for_the_same_instant_get_the_same_context():
    """15:35 ET is 19:35 UTC. An implementation that strips the offset reads
    19:35, decides the close has passed and hands out 2026-09-08's own label."""
    import context_join

    rows = [
        {"observation_id": "naive", "at": "2026-09-08 15:35:00"},
        {"observation_id": "aware", "at": "2026-09-08T19:35:00+00:00"},
    ]
    context_join.attach_context(
        rows, when="observation", labels_by_session=LABELS, clock_field="at"
    )
    naive = rows[0][context_join.OBSERVATION_CONTEXT_FIELD]
    aware = rows[1][context_join.OBSERVATION_CONTEXT_FIELD]

    assert naive.label == aware.label == "trending_up"
    assert naive.certainty == aware.certainty == context_join.CERTAINTY_PRIOR_SESSION

    # And 16:00 ET exactly - the close - IS the moment the label is available.
    at_close = [{"observation_id": "close", "at": "2026-09-08 16:00:00"}]
    context_join.attach_context(
        at_close, when="observation", labels_by_session=LABELS, clock_field="at"
    )
    ref = at_close[0][context_join.OBSERVATION_CONTEXT_FIELD]
    assert ref.certainty == context_join.CERTAINTY_SESSION
    assert ref.label == "compressed"


def test_a_date_only_fill_is_flagged_and_never_picks_a_midday_label():
    """A Questrade/IBKR statement stamps every fill MIDNIGHT market-local. The
    time of day is not known, so the entry context is the previous completed
    session's label, `date_only`, and FLAGGED. Giving it 2026-09-08's own label
    would be the machine claiming it knows the fill happened after the close."""
    import context_join
    import journal_trade_shape

    trades = [
        # As `journal_statement_import` writes one.
        {"trade_id": "t-statement", "opened_at": "2026-09-08 00:00:00",
         "closed_at": "2026-09-10 00:00:00", "status": "CLOSED", "symbol": "NVDA"},
        # As the live sync writes one - a real fill time on the same session.
        {"trade_id": "t-sync", "opened_at": "2026-09-08 11:02:00",
         "closed_at": "2026-09-10 15:50:00", "status": "CLOSED", "symbol": "NVDA"},
    ]
    assert journal_trade_shape.is_date_only(
        journal_trade_shape._coerce_datetime(trades[0]["opened_at"])  # noqa: SLF001
    ) is True

    context_join.attach_context(
        trades, when="entry", labels_by_session=LABELS, clock_field="opened_at"
    )
    statement = trades[0][context_join.ENTRY_CONTEXT_FIELD]
    sync = trades[1][context_join.ENTRY_CONTEXT_FIELD]

    assert statement.certainty == context_join.CERTAINTY_DATE_ONLY
    assert statement.flagged is True
    assert statement.label == "trending_up"
    assert statement.label != "compressed", "a date-only fill was given a midday regime"

    # The richer source is unaffected: an 11:02 fill is before the close, so it
    # takes the prior session too - but it is NOT flagged, because its time is
    # known and the answer is certain.
    assert sync.certainty == context_join.CERTAINTY_PRIOR_SESSION
    assert sync.label == "trending_up"
    assert sync.flagged is False

    # Both refs sit side by side with the observation ref, never replacing it.
    context_join.attach_context(
        trades, when="observation", labels_by_session=LABELS, clock_field="closed_at"
    )
    assert context_join.OBSERVATION_CONTEXT_FIELD in trades[1]
    assert trades[1][context_join.ENTRY_CONTEXT_FIELD] == sync
    assert (
        trades[1][context_join.OBSERVATION_CONTEXT_FIELD]
        != trades[1][context_join.ENTRY_CONTEXT_FIELD]
    )


# ---------------------------------------------------------------------------
# item 1 - a thesis links by SCOPE and VALIDITY WINDOW, never by ticker/date
# ---------------------------------------------------------------------------


def _thesis(thesis_id, created_at, benchmarks, *, stance="bullish", horizon_sessions=5,
            invalidated_at=""):
    """A `market_theses.jsonl` row shaped the way `market_thesis.draft_row`
    writes one (WS-10D), with only the fields a link needs filled in."""
    return {
        "thesis_id": thesis_id,
        "entry_id": f"entry-{thesis_id}",
        "extractor_version": "market_thesis_vocab_v1",
        "kind": "thesis",
        "supersedes": "",
        "recorded_at": created_at,
        "session_date": str(created_at)[:10],
        "created_at": created_at,
        "claim": "SPY holds 640 into the end of the week",
        "horizon": "this week",
        "horizon_sessions": horizon_sessions,
        "stance": stance,
        "condition": "unstated",
        "invalidation": "unstated",
        "invalidated_at": invalidated_at,
        "benchmarks": list(benchmarks),
        "spans": {},
        "is_prediction": True,
        "questions": [],
        "text": "",
    }


def _observed(rows):
    import context_join

    context_join.attach_context(
        rows, when="observation", labels_by_session=LABELS, clock_field="at"
    )
    return rows


def test_a_late_note_does_not_link_to_an_earlier_observation():
    """A thesis written on Thursday is not evidence about a Tuesday decision.
    The validity window opens at `created_at` and nothing before it is inside."""
    import context_join

    rows = _observed([{"observation_id": "obs-1", "at": "2026-09-08 10:35:00"}])
    theses = [_thesis("th-late", "2026-09-10T09:20:00-04:00", ["SPY"])]

    context_join.link_theses(rows, theses)
    assert rows[0][context_join.LINKED_THESES_FIELD] == []


def test_overlapping_theses_all_link_newest_first_and_none_is_chosen():
    """One opportunity may sit inside three live theses. All three link, newest
    first; the machine names no winner - `ambiguous stays ambiguous`."""
    import context_join

    rows = _observed([{"observation_id": "obs-1", "at": "2026-09-09 10:35:00"}])
    theses = [
        _thesis("th-old", "2026-09-04T16:30:00-04:00", ["SPY"]),
        _thesis("th-new", "2026-09-09T09:15:00-04:00", ["SPY"], stance="bearish"),
        _thesis("th-mid", "2026-09-08T16:10:00-04:00", ["SPY"]),
        # Expired: created 2026-09-02 with a 5-session horizon, so its window
        # closed on 2026-09-09's predecessor. Counted in SESSIONS, not days.
        _thesis("th-expired", "2026-08-31T16:10:00-04:00", ["SPY"]),
    ]

    context_join.link_theses(rows, theses)
    linked = rows[0][context_join.LINKED_THESES_FIELD]

    assert [entry["thesis_id"] for entry in linked] == ["th-new", "th-mid", "th-old"]
    # An opposite STANCE is still evidence about the same decision - it is the
    # thesis review that grades it, not the link.
    assert {entry["stance"] for entry in linked} == {"bullish", "bearish"}
    forbidden = {"chosen", "primary", "selected", "best", "winner"}
    for entry in linked:
        assert forbidden.isdisjoint(set(entry)), f"the link named a winner: {entry}"
    assert isinstance(linked, list)


def test_a_thesis_scoped_to_another_benchmark_does_not_link_even_on_the_same_date():
    """Scope, not date. A live QQQ thesis covering the same minute is not this
    SPY-scoped row's evidence, and an invalidated thesis stops covering the
    rows that come after it."""
    import context_join

    rows = _observed([{"observation_id": "obs-1", "at": "2026-09-09 10:35:00"}])
    theses = [
        _thesis("th-qqq", "2026-09-09T09:15:00-04:00", ["QQQ"]),
        _thesis("th-iwm", "2026-09-08T16:10:00-04:00", ["IWM", "QQQ"]),
        _thesis(
            "th-spy-dead",
            "2026-09-08T16:10:00-04:00",
            ["SPY"],
            invalidated_at="2026-09-09T09:40:00-04:00",
        ),
    ]

    context_join.link_theses(rows, theses)
    assert rows[0][context_join.LINKED_THESES_FIELD] == []

    # The same three theses DO cover a row inside the invalidated one's live
    # window, so the refusal above is about scope and validity, not about a
    # linker that never links.
    earlier = _observed([{"observation_id": "obs-2", "at": "2026-09-09 09:35:00"}])
    context_join.link_theses(earlier, theses)
    assert [
        entry["thesis_id"] for entry in earlier[0][context_join.LINKED_THESES_FIELD]
    ] == ["th-spy-dead"]


def test_attaching_and_linking_twice_is_the_same_answer_as_once():
    """Idempotent. The Results worker re-runs this on every redraw and a second
    pass that appended a second copy of every link would double every count."""
    import context_join

    rows = _observed([{"observation_id": "obs-1", "at": "2026-09-09 10:35:00"}])
    theses = [
        _thesis("th-a", "2026-09-08T16:10:00-04:00", ["SPY"]),
        _thesis("th-b", "2026-09-09T09:15:00-04:00", ["SPY"]),
    ]
    context_join.link_theses(rows, theses)
    once = copy.deepcopy(rows)

    context_join.attach_context(
        rows, when="observation", labels_by_session=LABELS, clock_field="at"
    )
    context_join.link_theses(rows, theses)

    assert rows == once


# ---------------------------------------------------------------------------
# item 2 - opportunity cells
# ---------------------------------------------------------------------------


def _opportunity_rows():
    """Tier-outcome rows shaped like the live CSV, with a KNOWN true answer.

    (family avwap_reclaim, side LONG), cut by environment:

      compressed  : 26-14 -> 65.0%, n=40, 4 sessions x 10, 20 symbols x 2
      trending_up : 34-6  -> 85.0%, n=40, 4 sessions x 10, but AAPL x 30
                              -> top symbol share 0.75, CONCENTRATED
      mixed       : 24-16 -> 60.0%, n=40, 2 sessions x 20
                              -> top session share EXACTLY 0.50, which is not
                                 over the limit and is therefore eligible
      unknown     : 3-1   -> n=4, under the floor, plus ONE row whose `win` is
                              present and EMPTY (unmeasured is not a loss)

    So the family/LONG baseline is 87 wins over 124 graded rows = 0.7016129...,
    and (family avwap_reclaim, side SHORT) is its own population at 12-18.
    """
    rows = []
    counter = 0

    def add(environment, side, session, symbol, win):
        nonlocal counter
        counter += 1
        rows.append(
            {
                "observation_id": f"obs-{counter}",
                "scan_row_id": f"row-{counter}",
                "scan_date": session,
                "future_scan_date": "2026-09-11",
                "horizon_sessions": "5",
                "setup_family": "avwap_reclaim",
                "side": side,
                "symbol": symbol,
                "tier": "A",
                "win": "" if win is None else ("True" if win else "False"),
                "side_return_pct": "1.5" if win else "-1.0",
                "stale_horizon": "False",
                "d1_environment": environment,
            }
        )

    sessions = ("2026-09-02", "2026-09-03", "2026-09-04", "2026-09-08")

    # compressed / LONG: 26-14 over four sessions and twenty symbols.
    for index in range(40):
        add("compressed", "LONG", sessions[index % 4], f"C{index % 20:02d}", index < 26)
    # trending_up / LONG: 34-6, but thirty of the forty are AAPL.
    for index in range(40):
        symbol = "AAPL" if index < 30 else f"T{index:02d}"
        add("trending_up", "LONG", sessions[index % 4], symbol, index < 34)
    # mixed / LONG: 24-16 over exactly two sessions.
    for index in range(40):
        add("mixed", "LONG", sessions[index % 2], f"M{index % 20:02d}", index < 24)
    # unknown / LONG: four graded rows and one unmeasured.
    for index in range(4):
        add("unknown", "LONG", sessions[index % 2], f"U{index:02d}", index < 3)
    add("unknown", "LONG", sessions[0], "UBLANK", None)
    # compressed / SHORT: its own cell, 12-18.
    for index in range(30):
        add("compressed", "SHORT", sessions[index % 3], f"S{index % 15:02d}", index < 12)

    return rows


def _cells_by_key(cells):
    return {(cell["environment"], cell["side"]): cell for cell in cells}


def test_a_sparse_opportunity_cell_says_it_is_below_the_floor_and_is_never_dropped():
    import evidence_stats
    import setup_environment_evidence

    cells = setup_environment_evidence.opportunity_cells(_opportunity_rows())
    by_key = _cells_by_key(cells)

    thin = by_key[("unknown", "LONG")]
    # Four GRADED rows out of five read. The blank `win` is not a loss.
    assert thin["n"] == 4
    assert thin["n_rows"] == 5
    assert thin["meets_floor"] is False
    assert thin["n_floor"] == evidence_stats.MIN_REPORTABLE_N
    assert thin["can_lead"] is False
    assert "floor" in thin["lead_refusal"].lower()
    # Labelled, not hidden, and `unknown` is its own cell pooled into nothing.
    assert thin["environment"] == "unknown"
    assert by_key[("compressed", "LONG")]["n"] == 40
    assert sum(
        cell["n"] for cell in cells if cell["side"] == "LONG"
    ) == 40 + 40 + 40 + 4


def test_a_concentrated_opportunity_cell_refuses_to_lead():
    """The best-looking rate on the page is thirty rows of one name. It is
    REPORTED, with its share, and it is not allowed to lead - and a cell that
    sits at EXACTLY the limit still can, because the limit is exceeded and not
    reached."""
    import setup_environment_evidence
    import working_lately

    cells = setup_environment_evidence.opportunity_cells(_opportunity_rows())
    by_key = _cells_by_key(cells)

    limit = working_lately.CONCENTRATION_LIMIT
    hot = by_key[("trending_up", "LONG")]
    assert hot["statistic"] == pytest.approx(0.85)
    assert hot["top_symbol_share"] == pytest.approx(30 / 40)
    assert hot["top_symbol_share"] > limit
    assert hot["concentrated"] is True
    assert hot["can_lead"] is False
    assert "concentrat" in hot["lead_refusal"].lower()

    # Exactly at the limit: two sessions, twenty rows each.
    even = by_key[("mixed", "LONG")]
    assert even["top_session_share"] == pytest.approx(limit)
    assert even["concentrated"] is False
    assert even["can_lead"] is True

    spread = by_key[("compressed", "LONG")]
    assert spread["top_symbol_share"] == pytest.approx(2 / 40)
    assert spread["top_session_share"] == pytest.approx(10 / 40)
    assert spread["can_lead"] is True

    # The refusal is the whole point: the highest statistic is not the leader.
    leaders = [cell for cell in cells if cell["can_lead"]]
    assert leaders, "every cell was refused - the refusal is not a rule, it is a bug"
    assert max(cell["statistic"] for cell in leaders) < hot["statistic"]


def test_every_opportunity_cell_carries_its_coverage_its_bound_and_its_baseline():
    import evidence_stats
    import swing_headline
    import setup_environment_evidence

    cells = setup_environment_evidence.opportunity_cells(_opportunity_rows())
    by_key = _cells_by_key(cells)
    spread = by_key[("compressed", "LONG")]

    assert spread["population_kind"] == "swing"
    assert spread["family"] == "avwap_reclaim"
    assert spread["n"] == 40
    assert spread["n_sessions"] == 4
    assert spread["n_symbols"] == 20
    assert spread["statistic"] == pytest.approx(26 / 40)
    # ONE Wilson, `swing_headline`'s own - never a second bound on one screen.
    assert spread["lower_bound"] == pytest.approx(
        swing_headline.wilson_lower_bound(26, 40)
    )
    assert spread["outcome_kind"] == swing_headline.OUTCOME_KIND_FAVORABLE_DIRECTION
    assert spread["window_sessions"] == evidence_stats.LATELY_SESSIONS

    # The setup's own baseline sits beside each cell, and it is the FAMILY x
    # SIDE record across every environment - 87 wins over 124 graded rows.
    for key in (("compressed", "LONG"), ("trending_up", "LONG"), ("unknown", "LONG")):
        assert by_key[key]["baseline_statistic"] == pytest.approx(87 / 124)
        assert by_key[key]["baseline_n"] == 124
    # SHORT is a different population and keeps its own baseline.
    short = by_key[("compressed", "SHORT")]
    assert short["baseline_n"] == 30
    assert short["baseline_statistic"] == pytest.approx(12 / 30)
    assert short["statistic"] == pytest.approx(12 / 30)


def _m5_outcome_rows():
    """M5 outcome-log rows for two alerts on 2026-09-09, both HELD past the
    thirty-minute window, carrying the ALERT's own `market_environment` in
    `context_json` - which is a different vocabulary from the D1 label and must
    not be mistaken for it."""
    rows = []
    for symbol, mfe in (("AAPL", 1.4), ("MSFT", 0.6)):
        event_id = f"{symbol}_long_20260909_10_35_00_h1_blue_after_red"
        rows.append(
            {
                "event_id": event_id, "event_type": "registered",
                "trade_date": "2026-09-09", "symbol": symbol, "direction": "long",
                "entry_time": "2026-09-09 10:35:00", "bars_elapsed": "0",
                "minutes_elapsed": "", "stop_hit": "", "mfe_r": "",
                "context_json": json.dumps({"market_environment": "quiet"}),
            }
        )
        rows.append(
            {
                "event_id": event_id, "event_type": "update",
                "trade_date": "2026-09-09", "symbol": symbol, "direction": "long",
                "entry_time": "2026-09-09 10:35:00", "bars_elapsed": "9",
                "minutes_elapsed": "45", "stop_hit": "", "mfe_r": str(mfe),
                "context_json": json.dumps({"market_environment": "quiet"}),
            }
        )
    return rows


def test_a_day_trade_cell_is_cut_by_the_d1_label_not_by_the_alerts_own_environment():
    """The day-trade population keeps its own headline (`held_run_score`) and is
    cut by the D1 environment of its session - the alert's registration-time
    `market_environment` is a different vocabulary and is not this label."""
    import held_run_score
    import setup_environment_evidence

    cells = setup_environment_evidence.opportunity_cells(
        _m5_outcome_rows(),
        population_kind="day_trade",
        labels_by_session=LABELS,
        date_field="trade_date",
    )
    assert len(cells) == 1
    cell = cells[0]

    assert cell["population_kind"] == "day_trade"
    assert cell["environment"] == LABELS["2026-09-09"] == "mixed"
    assert cell["environment"] != "quiet", "read the alert's env key as the D1 label"
    assert cell["statistic_name"] == held_run_score.HELD_RUN_STATISTIC_NAME
    assert cell["n"] == 2
    assert cell["n_held"] == 2
    assert cell["hold_rate"] == pytest.approx(1.0)
    # A day-trade cell may never be pooled with a swing one.
    assert cell["outcome_kind"] != "favorable_direction"


# ---------------------------------------------------------------------------
# item 2 - personal cells: money once, and nothing pooled
# ---------------------------------------------------------------------------


def _journal_trades():
    """Journal rows as `JournalStore` stores them, with the traps that matter.

    * `t-multi` carries TWO confirmed setup tags - it must appear in both tag
      cells and contribute its money EXACTLY ONCE to the total.
    * `t-theta` is a SOLD PUT: premium collected, `bullish_or_neutral`, an
      instrument the swing population does not pool.
    * `t-day` opened and closed inside one session - a day trade.
    * `t-short` is a short STOCK swing, which IS a swing and stays in.
    """
    return [
        {
            "trade_id": "t-multi", "symbol": "NVDA", "status": "CLOSED",
            "security_type": "STK", "direction": "LONG",
            "opened_at": "2026-09-08 11:02:00", "closed_at": "2026-09-10 15:50:00",
            "net_pnl": -120.0, "fees": 4.95, "commission": -4.95,
            "planned_risk": "150", "tag_status": "confirmed",
            "setup_tags": "avwap-reclaim; earnings-gap",
        },
        {
            "trade_id": "t-single", "symbol": "AMD", "status": "CLOSED",
            "security_type": "STK", "direction": "LONG",
            "opened_at": "2026-09-08 10:15:00", "closed_at": "2026-09-11 15:30:00",
            "net_pnl": 300.0, "fees": 2.50, "commission": -2.50,
            "planned_risk": "", "tag_status": "confirmed",
            "setup_tags": "avwap-reclaim",
        },
        {
            "trade_id": "t-short", "symbol": "TSLA", "status": "CLOSED",
            "security_type": "STK", "direction": "SHORT",
            "opened_at": "2026-09-08 09:45:00", "closed_at": "2026-09-10 11:20:00",
            "net_pnl": 80.0, "fees": 1.10, "commission": -1.10,
            "planned_risk": "60", "tag_status": "confirmed",
            "setup_tags": "avwap-reclaim",
        },
        {
            "trade_id": "t-theta", "symbol": "DRAM   261218P00055000",
            "status": "CLOSED", "security_type": "OPT", "direction": "SHORT",
            "opened_at": "2026-09-08 10:05:00", "closed_at": "2026-09-11 14:00:00",
            "net_pnl": 210.0, "fees": 1.25, "commission": -1.25,
            "planned_risk": "", "tag_status": "confirmed",
            "setup_tags": "wheel-put",
        },
        {
            "trade_id": "t-day", "symbol": "SPY", "status": "CLOSED",
            "security_type": "STK", "direction": "LONG",
            "opened_at": "2026-09-09 10:35:00", "closed_at": "2026-09-09 12:10:00",
            "net_pnl": 45.0, "fees": 0.90, "commission": -0.90,
            "planned_risk": "40", "tag_status": "confirmed",
            "setup_tags": "avwap-reclaim",
        },
        {
            "trade_id": "t-provisional", "symbol": "META", "status": "CLOSED",
            "security_type": "STK", "direction": "LONG",
            "opened_at": "2026-09-08 10:40:00", "closed_at": "2026-09-10 15:00:00",
            "net_pnl": 500.0, "fees": 3.00, "commission": -3.00,
            "planned_risk": "", "tag_status": "provisional",
            "setup_tags": "avwap-reclaim",
        },
    ]


def test_one_trade_with_many_tags_counts_money_once():
    """Two confirmed tags are two statements about ONE trade. It shows in both
    tag cells - that is what the cells are for - and the total counts its P&L
    once, the way `trade_level_summary` counts a statement file."""
    import setup_environment_evidence

    result = setup_environment_evidence.personal_cells(
        _journal_trades(), labels_by_session=LABELS, horizon="swing"
    )
    cells = {(cell["setup"], cell["environment"]): cell for cell in result["cells"]}

    # `t-multi` entered on 2026-09-08 at 11:02 - before the close - so its ENTRY
    # context is the prior session's label.
    reclaim = cells[("avwap-reclaim", "trending_up")]
    gap = cells[("earnings-gap", "trending_up")]
    assert "t-multi" in reclaim["trade_ids"]
    assert "t-multi" in gap["trade_ids"]

    totals = result["totals"]
    # t-multi -120 + t-single 300 + t-short 80 = 260. Counting t-multi twice
    # (once per tag) gives 140, which is the defect this test exists for.
    assert totals["net_pnl"] == pytest.approx(260.0)
    assert totals["n_trades"] == 3
    assert totals["duplicate_tag_rows"] == 1
    # The sum over cells is deliberately LARGER than the total - both grains are
    # real and the report says which is which.
    assert sum(cell["n_trades"] for cell in result["cells"]) == 4
    # Fees and planned-risk validity are shown, never inferred.
    assert totals["fees"] == pytest.approx(4.95 + 2.50 + 1.10)
    assert totals["n_with_planned_risk"] == 2
    # Commission keeps the sign the importer gave it; nothing here abs()es it.
    assert totals["commission"] == pytest.approx(-(4.95 + 2.50 + 1.10))

    # No "best" word below the floor - three trades is not an answer.
    import evidence_stats

    assert result["best_setup"] is None
    assert result["n_floor"] == evidence_stats.MIN_REPORTABLE_N
    assert result["meets_floor"] is False


def test_short_and_theta_exposure_stay_out_of_the_swing_cells():
    """A sold put is premium, not a swing; a same-session trade is a day trade.
    Neither is pooled into the swing cells, both are ACCOUNTED for by name, and
    a short STOCK swing is still a swing."""
    import setup_environment_evidence

    result = setup_environment_evidence.personal_cells(
        _journal_trades(), labels_by_session=LABELS, horizon="swing"
    )
    inside = {
        trade_id for cell in result["cells"] for trade_id in cell["trade_ids"]
    }

    assert "t-theta" not in inside
    assert "t-day" not in inside
    # A provisional tag is a machine's guess and never "my setup".
    assert "t-provisional" not in inside
    # A short stock swing is a swing.
    assert "t-short" in inside

    excluded = result["excluded"]
    assert set(excluded) == {"t-theta", "t-day", "t-provisional"}
    # Three different reasons, never one bucket: theta, day trade, unconfirmed.
    assert len({excluded["t-theta"], excluded["t-day"], excluded["t-provisional"]}) == 3
    for reason in excluded.values():
        assert reason.strip()
    # Every trade is accounted for: nothing is silently dropped.
    assert result["totals"]["n_trades"] + len(excluded) == len(_journal_trades())

    # The theta population is reported, separately, with its own money.
    theta = result["populations"]["theta"]
    assert theta["n_trades"] == 1
    assert theta["net_pnl"] == pytest.approx(210.0)
    assert theta["net_pnl"] != result["totals"]["net_pnl"]


def test_personal_cells_partition_by_status_and_never_pool_an_open_mark():
    """ST5's partition survives the environment cut: an open position has NO
    result, and its money is never summed into a closed one."""
    import setup_environment_evidence

    trades = _journal_trades() + [
        {
            "trade_id": "t-open", "symbol": "AAPL", "status": "OPEN",
            "security_type": "STK", "direction": "LONG",
            "opened_at": "2026-09-10 10:00:00", "closed_at": "",
            "net_pnl": 9999.0, "fees": 1.00, "commission": -1.00,
            "planned_risk": "", "tag_status": "confirmed",
            "setup_tags": "avwap-reclaim",
        }
    ]
    result = setup_environment_evidence.personal_cells(
        trades, labels_by_session=LABELS, horizon="swing"
    )

    assert result["populations"]["open_exposure"]["net_pnl"] is None
    assert result["totals"]["net_pnl"] == pytest.approx(260.0)
    assert "t-open" not in {
        trade_id for cell in result["cells"] for trade_id in cell["trade_ids"]
    }


# ---------------------------------------------------------------------------
# item 2 - three thesis verdicts, never merged
# ---------------------------------------------------------------------------


def test_the_three_thesis_verdicts_are_separate_fields():
    """`market call`, `setup held` and `trade profitable` answer three different
    questions and are never blended into one grade. The fixture is the case that
    proves it: the call was RIGHT and the trade LOST."""
    import setup_environment_evidence

    theses = [
        _thesis("th-right", "2026-09-08T16:10:00-04:00", ["SPY"], stance="bullish"),
        _thesis("th-open", "2026-09-11T16:10:00-04:00", ["SPY"], stance="bullish"),
    ]
    # SPY closes, session by session. 2026-09-08 -> 2026-09-15 is five sessions
    # (09-09, 09-10, 09-11, 09-14, 09-15) and the tape went up.
    paths = {
        "SPY": {
            "2026-09-08": 640.00,
            "2026-09-09": 641.20,
            "2026-09-10": 639.80,
            "2026-09-11": 644.10,
            "2026-09-14": 650.00,
            "2026-09-15": 655.00,
        }
    }
    rows = _observed([{"observation_id": "obs-1", "at": "2026-09-09 10:35:00",
                       "setup_family": "avwap_reclaim", "side": "LONG",
                       "symbol": "NVDA", "scan_date": "2026-09-09",
                       "horizon_sessions": "5", "win": "True",
                       "side_return_pct": "1.5", "stale_horizon": "False",
                       "d1_environment": "mixed"}])
    trades = [
        {"trade_id": "t-multi", "symbol": "NVDA", "status": "CLOSED",
         "security_type": "STK", "direction": "LONG",
         "opened_at": "2026-09-09 11:02:00", "closed_at": "2026-09-11 15:50:00",
         "net_pnl": -120.0, "fees": 4.95, "commission": -4.95,
         "planned_risk": "150", "tag_status": "confirmed",
         "setup_tags": "avwap-reclaim"},
    ]

    reviews = setup_environment_evidence.thesis_review(
        theses,
        opportunity_rows=rows,
        trades=trades,
        benchmark_paths=paths,
        as_of=date(2026, 9, 13),
    )
    by_id = {review["thesis_id"]: review for review in reviews}

    right = by_id["th-right"]
    assert right["market_call"] == "right"
    assert right["setup_held"]["n"] == 1
    assert right["trade_profitable"]["profitable"] is False
    assert right["trade_profitable"]["net_pnl"] == pytest.approx(-120.0)
    # The three are separate KEYS and there is no combined grade anywhere.
    assert {"market_call", "setup_held", "trade_profitable"} <= set(right)
    assert {"grade", "verdict", "score", "overall", "result"}.isdisjoint(set(right))
    # Money once per trade here too.
    assert right["linked_trade_ids"] == ["t-multi"]

    # A thesis whose horizon has not been reached is OPEN, never "wrong".
    assert by_id["th-open"]["market_call"] == "open"


# ---------------------------------------------------------------------------
# item 3 - the Research > Results By environment control
# ---------------------------------------------------------------------------

_GOLDEN_SECTION_KEYS = ["swing_trade_r", "swing_favorable"]

_GOLDEN_TRADE_R_VERDICT = (
    "swing_trade_r: no_clear_leader - LONG alpha is ahead but is a NEW leader: "
    "awaiting persistence (1 of 2) - a new leader is announced only once it has "
    "led in that many snapshots with a distinct as_of. LONG alpha leads SHORT "
    "beta by 0.080 of Wilson lower bound (0.610 vs 0.530), clear of the declared "
    "0.05 margin. - LONG alpha [swing_trade_r]; win rate (closed, unweighted) "
    "0.72; wilson lower bound >= 0.61; n=613 (613 graded, 0 pending, 0 excluded); "
    "17 symbol(s) / 12 session(s); top symbol 0.21, top session 0.18; outcome "
    "trade_r_representative_exit (recent_types_v2); basis "
    "entry_scan_row_close_to_representative_exit; horizon 30d lookback, "
    "representative exit; window 20 sessions through 2026-09-11; namespace live"
)

_GOLDEN_FRESHNESS = (
    "The snapshot owns this window: 20 sessions ending 2026-09-11. snapshot "
    "ws10ifix; as of 2026-09-11; built 2026-09-11T17:31:00-04:00; the snapshot "
    "did not record its sources"
)


def _bot_snapshot():
    """A small real `EvidenceSnapshot.to_payload()` with both swing kinds."""
    from working_lately import (
        SNAPSHOT_KINDS,
        EvidenceCell,
        EvidenceSnapshot,
        select_cell_leader,
    )

    def cell(**kwargs):
        base = dict(
            outcome_kind="trade_r_representative_exit",
            outcome_version="recent_types_v2",
            knowledge_basis="entry_scan_row_close_to_representative_exit",
            horizon="30d lookback, representative exit",
            window_sessions=20,
            latest_measured_session="2026-09-11",
            n_pending=0,
            n_excluded=0,
            n_symbols=17,
            n_sessions=12,
            top_symbol_share=0.21,
            top_session_share=0.18,
            statistic_name="win rate (closed, unweighted)",
            uncertainty_kind="wilson lower bound",
            namespace="live",
            n_floor=30,
            meets_floor=True,
        )
        base.update(kwargs)
        return EvidenceCell(**base)

    cells = [
        cell(kind="swing_trade_r", side="LONG", family="alpha", n_eligible=613,
             n_graded=613, statistic=0.72, uncertainty_low=0.61),
        cell(kind="swing_trade_r", side="SHORT", family="beta", n_eligible=614,
             n_graded=614, statistic=0.64, uncertainty_low=0.53),
        cell(kind="swing_favorable", side="LONG", family="fav_a", n_eligible=615,
             n_graded=615, statistic=58.0, uncertainty_low=49.0,
             outcome_kind="favorable_direction",
             outcome_version="favorable_direction_session_v2",
             knowledge_basis="scan row close to target session close",
             horizon="5 sessions",
             statistic_name="favorable direction (percent)",
             uncertainty_kind="wilson lower bound (percent)"),
    ]
    verdicts = {
        kind: select_cell_leader(
            cells,
            kind=kind,
            last_completed_session=date(2026, 9, 11),
            previous=None,
            source_rows=len([one for one in cells if one.kind == kind]),
        )
        for kind in SNAPSHOT_KINDS
    }
    return EvidenceSnapshot(
        snapshot_id="ws10ifixture00000000000000000000000000",
        as_of="2026-09-11",
        built_at="2026-09-11T17:31:00-04:00",
        cells=tuple(cells),
        verdicts=verdicts,
        sources={},
    ).to_payload()


def test_the_existing_results_goldens_are_unchanged_when_no_environment_is_chosen():
    """The champion sections are what they were. These strings were captured by
    running the PRE-CHANGE code on this branch (2026-09-13) and are not
    regenerated by the code under test."""
    import research_results

    view = research_results.build_results_view(
        population="bot",
        horizon="swing",
        window="recent",
        snapshot=_bot_snapshot(),
        as_of=date(2026, 9, 11),
    )

    # The goldens are asserted FIRST, deliberately: if one of these strings were
    # wrong the tester would see a string mismatch here rather than the
    # AttributeError below, which is the only thing this test may fail on today.
    assert [section.key for section in view.sections] == _GOLDEN_SECTION_KEYS
    assert view.sections[0].title == "Swing - closed R on the representative exit"
    assert view.sections[0].verdict_line == _GOLDEN_TRADE_R_VERDICT
    assert view.sections[1].title == "Swing - favorable direction at the declared horizon"
    assert view.freshness_line == _GOLDEN_FRESHNESS
    assert view.window_sentence == "The snapshot owns this window: 20 sessions ending 2026-09-11"
    assert view.window_applies is False
    # And the new control, defaulted, changes none of it.
    assert view.environment_filter == research_results.ENVIRONMENT_ALL


def test_the_by_environment_control_recuts_every_population_and_names_its_basis():
    """One page-level control. Bot is cut by the environment known at
    OBSERVATION, My trades by the one known at ENTRY, and the control says which
    benchmark and rule version it is showing."""
    import research_results
    from indicators.d1_environment import RULE_VERSION

    snapshot = _bot_snapshot()
    rows = copy.deepcopy(_opportunity_rows())

    swing = research_results.build_results_view(
        population="bot", horizon="swing", window="recent", snapshot=snapshot,
        as_of=date(2026, 9, 11), environment_rows=rows,
        environment_filter="compressed",
    )
    assert swing.environment_filter == "compressed"
    assert swing.environment_basis == "observation"
    assert "SPY" in swing.environment_line and RULE_VERSION in swing.environment_line
    assert research_results.ENVIRONMENT_ALL in swing.environment_choices
    assert "compressed" in swing.environment_choices
    assert "unknown" in swing.environment_choices

    section = [
        one for one in swing.sections
        if one.key == research_results.ENVIRONMENT_SECTION_KEY
    ][0]
    assert section.rows, "the chosen environment has rows and the page showed none"
    assert {one.values["environment"] for one in section.rows} == {"compressed"}

    day = research_results.build_results_view(
        population="bot", horizon="day", window="recent", snapshot=snapshot,
        as_of=date(2026, 9, 11), environment_filter="compressed",
    )
    assert day.environment_filter == "compressed"
    assert day.environment_basis == "observation"

    mine = research_results.build_results_view(
        population="mine", horizon="swing", window="recent", snapshot=snapshot,
        journal_trades=[], as_of=date(2026, 9, 11), environment_filter="compressed",
    )
    assert mine.environment_filter == "compressed"
    assert mine.environment_basis == "entry"


def test_the_results_worker_passes_the_chosen_environment_through():
    """Computed on the WORKER, never the Qt thread - and the legacy three-part
    selection still works."""
    import research_results
    from ui.panels import research_results_panel as panel

    seen = {}
    real = research_results.build_results_view

    def capture(**kwargs):
        seen.update(kwargs)
        return real(**{
            key: value for key, value in kwargs.items()
            if key != "environment_filter"
        })

    original = research_results.build_results_view
    research_results.build_results_view = capture
    try:
        panel._read(("bot", "swing", "recent", "compressed"), "recent", _bot_snapshot())
        assert seen["environment_filter"] == "compressed"
        seen.clear()
        panel._read(("bot", "swing", "recent"), "recent", _bot_snapshot())
        assert seen["environment_filter"] == research_results.ENVIRONMENT_ALL
    finally:
        research_results.build_results_view = original


# ---------------------------------------------------------------------------
# item 3 - the Daily Recap row (DEPENDS ON WS-DR for the rendering only)
# ---------------------------------------------------------------------------


def test_a_daily_recap_row_carries_both_context_labels_side_by_side():
    """WS-DR's session reader writes one row per opportunity, with the matched
    trade beside it when there is one. The recap shows the OBSERVATION context
    on every row and the ENTRY context beside it only where a fill exists - two
    labels, never one blended into the other, and never an entry label invented
    for an unmatched opportunity."""
    import context_join

    matched = {
        "opportunity_id": "obs-1",
        "symbol": "NVDA",
        "side": "LONG",
        "observed_at": "2026-09-08 10:35:00",
        "matched_trade_id": "t-multi",
        "entry_at": "2026-09-08 00:00:00",
    }
    unmatched = {
        "opportunity_id": "obs-2",
        "symbol": "AMD",
        "side": "LONG",
        "observed_at": "2026-09-08 15:59:00",
        "matched_trade_id": "",
        "entry_at": "",
    }

    both = context_join.recap_context_labels(matched, labels_by_session=LABELS)
    assert both["observation"] == "trending_up"
    assert both["observation_certainty"] == context_join.CERTAINTY_PRIOR_SESSION
    assert both["entry"] == "trending_up"
    assert both["entry_certainty"] == context_join.CERTAINTY_DATE_ONLY
    assert both["entry_flagged"] is True

    alone = context_join.recap_context_labels(unmatched, labels_by_session=LABELS)
    assert alone["observation"] == "trending_up"
    assert alone["entry"] == ""
    assert alone["entry_certainty"] == ""


# ---------------------------------------------------------------------------
# item 4 - the backfill CLI
# ---------------------------------------------------------------------------


def test_the_backfill_dry_run_writes_nothing_and_says_where_it_is_pointed(tmp_path, capsys):
    import context_join
    import project_paths

    assert not str(project_paths.DATA_DIR).lower().startswith("c:\\tradingbotdata")

    target = tmp_path / "context_refs.jsonl"
    code = context_join.main(["backfill", "--since", "2026-09-01", "--path", str(target)])
    printed = capsys.readouterr().out

    assert code == 0
    assert target.exists() is False, "the dry run wrote the store"
    assert "DRY RUN" in printed
    assert str(project_paths.DATA_DIR) in printed
    assert "--apply" in printed


def test_a_backfilled_ref_is_labelled_reconstructed_and_kept_out_of_forward_claims():
    """Backfill only what contemporaneous evidence establishes, and say so. A
    reconstructed ref is never mistaken for one recorded at the time."""
    import context_join

    rows = [
        {"observation_id": "obs-1", "scan_date": "2026-09-04"},
        {"observation_id": "obs-2", "scan_date": "2026-09-08"},
        # Older than `since`: out of range, so nothing is reconstructed for it.
        {"observation_id": "obs-3", "scan_date": "2026-08-28"},
    ]
    refs = context_join.backfill_refs(
        rows, labels_by_session=LABELS, clock_field="scan_date", since="2026-09-01"
    )

    assert len(refs) == 2
    assert [ref.label for ref in refs] == ["trending_up", "compressed"]
    for ref in refs:
        assert ref.certainty == context_join.CERTAINTY_RECONSTRUCTED
        assert ref.flagged is True
    # A live read of the same rows is NOT reconstructed - the two are told apart
    # by the certainty and never by the label.
    context_join.attach_context(
        rows, when="observation", labels_by_session=LABELS, clock_field="scan_date"
    )
    live = rows[0][context_join.OBSERVATION_CONTEXT_FIELD]
    assert live.label == refs[0].label
    assert live.certainty != refs[0].certainty


# ---------------------------------------------------------------------------
# shadow only
# ---------------------------------------------------------------------------


def test_the_context_join_reaches_no_detector_score_alert_or_policy_store():
    """plan.md sec 5. This is evidence about decisions, never an input to one."""
    import inspect

    import context_join
    import setup_environment_evidence

    for module in (context_join, setup_environment_evidence):
        source = inspect.getsource(module)
        for forbidden in (
            "review_policy",
            "bounce_bot",
            "m5_signal_engines",
            "focus_pick_store",
            "CandidateRegistry",
        ):
            assert forbidden not in source, f"{module.__name__} reaches {forbidden}"
