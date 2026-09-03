"""P10 Part C - what happened after the like.

Trader, 2026-09-02: *"anytime I like a D1 it should be treated with respect by
the bot in regards to finding out what's good about it, how we can replicate
those searches, and then how we can improve the entries. **if I like a stock one
day it may not be for 3-5 days later that the best entry is.**"*

Twenty cells: five day offsets x four entry rules, ONE structural stop and ONE
target, so a winning cell cannot have won on its stop or its target. Shadow only,
every recipe `is_diagnostic`, and the trial ledger row is written before any
outcome is inspected.

The fixture is P8's, deliberately: the parity test only means something if both
grids are reading the same bars.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

FIXTURE = ROOT / "tests" / "fixtures" / "setup_entry_timing_parity_v1.json"

UTC = timezone.utc


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _inputs():
    import build_setup_entry_timing_fixture as builder

    return builder.materialise(_fixture()["setup_entry_timing_parity_input_v1"])


def _as_of() -> datetime:
    return datetime.fromisoformat(_fixture()["as_of"])


def _recipe(recipe_id: str):
    from research_warehouse import outcomes

    return next(r for r in outcomes.AFTER_LIKE_RECIPES if r.recipe_id == recipe_id)


def _like(session_date="2026-08-04", *, symbol="PARITY", side="LONG"):
    return {
        "event_id": "like-1",
        "symbol": symbol,
        "side": side,
        "session_date": session_date,
    }


# ---------------------------------------------------------------------------
# C1 - the grid is what it declared, and it was declared first
# ---------------------------------------------------------------------------


def test_the_grid_is_twenty_bounded_cells_and_never_a_cartesian_search():
    from research_warehouse import outcomes

    recipes = outcomes.AFTER_LIKE_RECIPES
    assert len(recipes) == 20
    assert len(recipes) == len(outcomes.AFTER_LIKE_OFFSETS) * len(
        outcomes.AFTER_LIKE_ENTRIES
    )
    assert len({r.recipe_id for r in recipes}) == 20


def test_only_the_offset_and_the_entry_vary():
    """One stop, one target, one time stop - or a cell might have won on those."""
    from research_warehouse import outcomes

    recipes = outcomes.AFTER_LIKE_RECIPES
    assert {r.target_r for r in recipes} == {outcomes.AFTER_LIKE_TARGET_R}
    assert {r.stop_selector for r in recipes} == {outcomes.AFTER_LIKE_STOP_SELECTOR}
    assert {r.stop_atr_multiple for r in recipes} == {None}
    assert {r.time_stop_sessions for r in recipes} == {outcomes.SWING_TIME_STOP_SESSIONS}
    assert all(r.is_diagnostic for r in recipes), "shadow only"
    assert {r.timeframe for r in recipes} == {"AFTER_LIKE"}


def test_the_recipe_id_says_which_day_and_which_entry():
    """A row has to state what produced it without a lookup table."""
    from research_warehouse import outcomes

    for recipe in outcomes.AFTER_LIKE_RECIPES:
        assert recipe.entry_variant in recipe.recipe_id
        offset = outcomes.after_like_offset_for(recipe)
        assert offset in outcomes.AFTER_LIKE_OFFSETS
        assert f"_d{offset}_" in recipe.recipe_id


def test_the_trial_is_registered_with_the_window_fixed_and_status_collecting():
    from research_warehouse import outcomes, trial_ledger

    row = next(
        item
        for item in trial_ledger.BACKFILL_TRIALS
        if item["trial_id"] == outcomes.AFTER_LIKE_TRIAL_ID
    )
    assert row["declared_cell_count"] == len(outcomes.AFTER_LIKE_RECIPES)
    assert row["status"] == trial_ledger.STATUS_COLLECTING
    assert row["declared_window"]["sessions"] == 20
    assert row["analysis_unit"] == "like_episode"
    assert row["authorization"], "an experiment nobody authorized is not a trial"
    # The named failure mode is the survivorship one, and it must be there before
    # any number is: it is what the reader checks the row counts against.
    assert "survivorship" in row["failure_mode"].lower() or "straight down" in row["failure_mode"]


def test_the_ledger_row_is_written_before_any_outcome_is_read(tmp_path):
    """The whole point of a pre-declaration is that it predates the numbers."""
    from research_warehouse import outcomes, trial_ledger

    written = trial_ledger.backfill(tmp_path)
    assert outcomes.AFTER_LIKE_TRIAL_ID in written
    # And it refuses to be rewritten afterwards.
    assert trial_ledger.backfill(tmp_path) == []
    rows = trial_ledger.load(tmp_path)
    row = next(r for r in rows if r["trial_id"] == outcomes.AFTER_LIKE_TRIAL_ID)
    assert row["registered_at"], "stamped by the ledger, not by the caller"


# ---------------------------------------------------------------------------
# The offset is a TRADING-day walk
# ---------------------------------------------------------------------------


def test_the_offset_counts_trading_days_never_calendar_days():
    """Three days after a Thursday like is the following Tuesday, not the Sunday."""
    from research_warehouse import outcomes

    # 2026-09-03 is a Thursday; 2026-09-07 is Labor Day.
    assert outcomes.after_like_entry_session("2026-09-03", 0).session_date == date(2026, 9, 3)
    assert outcomes.after_like_entry_session("2026-09-03", 1).session_date == date(2026, 9, 4)
    assert outcomes.after_like_entry_session("2026-09-03", 2).session_date == date(2026, 9, 8)
    assert outcomes.after_like_entry_session("2026-09-03", 3).session_date == date(2026, 9, 9)
    assert outcomes.after_like_entry_session("2026-09-03", 5).session_date == date(2026, 9, 11)


def test_a_like_made_when_the_market_was_shut_starts_at_the_next_open():
    """A like typed at the weekend is about the Monday."""
    from research_warehouse import outcomes

    saturday = outcomes.after_like_entry_session("2026-09-05", 0)
    assert saturday.session_date == date(2026, 9, 8), "Labor Day is the Monday"


def test_an_unreadable_like_date_yields_no_session_rather_than_today():
    from research_warehouse import outcomes

    assert outcomes.after_like_entry_session("not-a-date", 0) is None


# ---------------------------------------------------------------------------
# C2 - the parity, and one case per entry per offset
# ---------------------------------------------------------------------------


def test_offset_zero_first_close_reproduces_the_p8_control_field_for_field():
    """The two grids must agree where they ask the same question.

    P8's control enters at the first completed M5 close after the D1 trigger.
    An after-like offset-0 entry on the session that follows that trigger is the
    same bar, so every field of the row - entry price, stop, target, checkpoints,
    outcome - has to match. Anything else means one of the two grids has its own
    copy of the exit machine, which is exactly what neither may have.
    """
    from research_warehouse import outcomes

    occurrence, bars = _inputs()
    as_of = _as_of()

    p8 = outcomes.simulate_setup_entry_timing(
        occurrence,
        bars,
        next(
            r
            for r in outcomes.SETUP_ENTRY_TIMING_RECIPES
            if r.recipe_id == "setupentry_m5_first_close_2r_v1"
        ),
        as_of=as_of,
        computed_at=as_of,
        run_id="test",
    )
    p10 = outcomes.simulate_after_like_entry(
        _like("2026-08-04"),
        occurrence,
        bars,
        _recipe("afterlike_d0_first_m5_close_2r_v1"),
        as_of=as_of,
        computed_at=as_of,
        run_id="test",
    )

    assert p8 is not None and p10 is not None
    ignored = {"recipe_id", "entry_rule", "entry_variant", "recipe_note", "note"}
    shared = {
        key: value for key, value in p8.items() if key not in ignored
    }
    for key, value in shared.items():
        assert p10.get(key) == value, f"{key}: {p10.get(key)!r} != {value!r}"


def test_every_entry_rule_produces_a_row_or_no_row_and_never_an_invented_one():
    """Each of the four, at offset 0, on the same fixture."""
    from research_warehouse import outcomes

    occurrence, bars = _inputs()
    as_of = _as_of()

    for entry in outcomes.AFTER_LIKE_ENTRIES:
        row = outcomes.simulate_after_like_entry(
            _like("2026-08-04"),
            occurrence,
            bars,
            _recipe(f"afterlike_d0_{entry}_2r_v1"),
            as_of=as_of,
            computed_at=as_of,
            run_id="test",
        )
        if row is None:
            continue  # unmeasurable is a real answer
        assert row["recipe_id"] == f"afterlike_d0_{entry}_2r_v1"
        entry_at = row.get("entry_at")
        assert isinstance(entry_at, datetime)
        assert entry_at.date() >= date(2026, 8, 4), entry


def test_a_later_offset_never_enters_before_its_own_session():
    """The offset is the whole variable; an entry before it would erase it."""
    from research_warehouse import outcomes

    occurrence, bars = _inputs()
    as_of = _as_of()

    row = outcomes.simulate_after_like_entry(
        _like("2026-08-04"),
        occurrence,
        bars,
        _recipe("afterlike_d1_first_m5_close_2r_v1"),
        as_of=as_of,
        computed_at=as_of,
        run_id="test",
    )

    assert row is not None
    assert row["entry_at"].date() >= date(2026, 8, 5)


def test_a_day_with_no_bars_is_no_row_and_never_an_invented_entry():
    """The question is which day was best; a day with no entry is an answer."""
    from research_warehouse import outcomes

    occurrence, bars = _inputs()

    # The fixture stops on 2026-08-06, so offset 5 from a 2026-08-04 like lands
    # past the end of the bars entirely.
    row = outcomes.simulate_after_like_entry(
        _like("2026-08-04"),
        occurrence,
        bars,
        _recipe("afterlike_d5_first_m5_close_2r_v1"),
        as_of=_as_of(),
        computed_at=_as_of(),
        run_id="test",
    )

    assert row is None


def test_the_cluster_key_makes_one_like_one_episode():
    """A name liked on consecutive days is one opinion held twice."""
    from research_warehouse import outcomes

    monday = outcomes.after_like_cluster_id(_like("2026-08-03"))
    tuesday = outcomes.after_like_cluster_id(_like("2026-08-04"))
    same_day_short = outcomes.after_like_cluster_id(_like("2026-08-03", side="SHORT"))

    assert monday != tuesday, "two days is two episodes"
    assert monday != same_day_short, "two sides are two theses"
    assert monday == outcomes.after_like_cluster_id(_like("2026-08-03"))


def test_the_grid_registers_no_outcome_semantics(monkeypatch):
    """BD-80: a diagnostic grid does not enter the trade-claim vocabulary."""
    import outcome_semantics

    for recipe_id in ("afterlike_d0_first_m5_close_2r_v1", "afterlike_d5_m15_acceptance_2r_v1"):
        assert recipe_id not in getattr(outcome_semantics, "CLAIM_KINDS", {})


# ---------------------------------------------------------------------------
# The nightly pass, and what it refuses
# ---------------------------------------------------------------------------


def test_every_cell_of_one_like_shares_a_cluster_and_names_its_like():
    from research_warehouse import after_like

    occurrence, bars = _inputs()
    rows = after_like.simulate_after_like_rows(
        _like("2026-08-04"),
        occurrence,
        bars,
        as_of=_as_of(),
        computed_at=_as_of(),
        run_id="test",
    )

    assert rows, "the fixture must produce at least the control"
    assert len({row["dependency_cluster_id"] for row in rows}) == 1
    assert {row["like_event_id"] for row in rows} == {"like-1"}
    assert len({row["recipe_id"] for row in rows}) == len(rows), "one row per cell"


def test_an_unlinked_like_is_counted_and_named_never_silently_dropped():
    """A measured limit, not an omission.

    The registered grid declares ONE structural stop, and that level comes from
    the occurrence's own tracker geometry. A like the scanner never found has no
    anchor to place it at. Giving the unlinked bucket a substitute stop would
    mean the grid no longer has one stop model, so an unlinked-vs-linked
    difference could be a difference in STOPS - and dropping them quietly would
    hide how many of the trader's likes the scanner never found.
    """
    from research_warehouse import after_like

    result = after_like.run_after_like(
        [_like("2026-08-04")],
        {},  # no link at all
        {},
        {},
        as_of=_as_of(),
        run_id="test",
    )

    assert result.likes_seen == 1
    assert result.episodes_graded == 0
    assert result.rows == []
    assert result.excluded_by_reason == {after_like.EXCLUDED_NO_OCCURRENCE: 1}


def test_a_linked_like_with_no_bars_is_a_different_named_exclusion():
    """"The scanner never found it" and "we have no bars" are different facts."""
    from research_warehouse import after_like
    from research_warehouse.like_links import BASIS_EXACT_FAMILY, LikeLink

    occurrence, _bars = _inputs()
    link = LikeLink(
        event_id="like-1",
        symbol="PARITY",
        side="LONG",
        like_date="2026-08-04",
        occurrence_id="occ-1",
        canonical_setup_id="AVWAPE_TO_FIRST_DEV",
        trigger_at="2026-08-03T20:00:00+00:00",
        match_basis=BASIS_EXACT_FAMILY,
        candidates_in_window=1,
    )

    result = after_like.run_after_like(
        [_like("2026-08-04")],
        {"like-1": link},
        {"occ-1": occurrence},
        {},  # no bars materialised for this symbol
        as_of=_as_of(),
        run_id="test",
    )

    assert result.excluded_by_reason == {after_like.EXCLUDED_NO_BARS: 1}


def test_a_linked_like_with_bars_grades_one_episode():
    from research_warehouse import after_like
    from research_warehouse.like_links import BASIS_EXACT_FAMILY, LikeLink

    occurrence, bars = _inputs()
    link = LikeLink(
        event_id="like-1",
        symbol="PARITY",
        side="LONG",
        like_date="2026-08-04",
        occurrence_id="occ-1",
        canonical_setup_id="AVWAPE_TO_FIRST_DEV",
        trigger_at="2026-08-03T20:00:00+00:00",
        match_basis=BASIS_EXACT_FAMILY,
        candidates_in_window=1,
    )

    result = after_like.run_after_like(
        [_like("2026-08-04")],
        {"like-1": link},
        {"occ-1": occurrence},
        {"PARITY": bars},
        as_of=_as_of(),
        computed_at=_as_of(),
        run_id="test",
    )

    assert result.likes_seen == 1
    assert result.episodes_graded == 1, "ONE episode, however many cells measured"
    assert result.rows
    assert result.excluded_by_reason == {}


def test_two_likes_on_one_occurrence_do_not_collide_on_the_outcome_grain():
    """`outcome_path`'s grain is (occurrence_id, recipe_id, definition).

    Two likes on two days that link to the same occurrence are different rows -
    the offsets are measured from each like's own session - and under the
    occurrence's own id the second would silently replace the first.
    """
    from research_warehouse import after_like

    occurrence, bars = _inputs()
    monday = after_like.simulate_after_like_rows(
        {**_like("2026-08-04"), "event_id": "like-1"},
        occurrence,
        bars,
        as_of=_as_of(),
        computed_at=_as_of(),
        run_id="test",
    )
    tuesday = after_like.simulate_after_like_rows(
        {**_like("2026-08-05"), "event_id": "like-2"},
        occurrence,
        bars,
        as_of=_as_of(),
        computed_at=_as_of(),
        run_id="test",
    )

    monday_keys = {(row["occurrence_id"], row["recipe_id"]) for row in monday}
    tuesday_keys = {(row["occurrence_id"], row["recipe_id"]) for row in tuesday}
    assert monday_keys and tuesday_keys
    assert not (monday_keys & tuesday_keys), "the two likes shared a grain key"
    # What was linked is not lost.
    assert {row["linked_occurrence_id"] for row in monday} == {
        occurrence["occurrence_id"]
    }


# ---------------------------------------------------------------------------
# C3 - the readout
# ---------------------------------------------------------------------------


def _cell_rows(count, *, offset=0, entry="first_m5_close", net_r=0.5):
    return [
        {
            "recipe_id": f"afterlike_d{offset}_{entry}_2r_v1",
            "occurrence_id": f"afterlike|SYM{index}|LONG|2026-09-02",
            "net_r": net_r,
            "first_hit": "TARGET",
            "entry_at": "2026-09-09T13:35:00",
        }
        for index in range(count)
    ]


def test_the_pack_carries_an_after_like_block_keyed_by_offset_and_entry():
    from ai_jobs import setup_research

    block = setup_research.after_like_block(
        _cell_rows(3, offset=0) + _cell_rows(2, offset=3, entry="m15_acceptance")
    )

    cells = {(cell["day_offset"], cell["entry"]): cell for cell in block["cells"]}
    assert set(cells) == {(0, "first_m5_close"), (3, "m15_acceptance")}
    assert cells[(0, "first_m5_close")]["n"] == 3
    assert cells[(0, "first_m5_close")]["n_episodes"] == 3
    # THREE, not five. The two cells were built from the same three episode ids,
    # and `episodes` counts distinct likes rather than summing the cells - which
    # is the whole reason the count is taken over the id and not over the rows.
    assert block["episodes"] == 3


def test_the_episode_is_the_unit_never_the_row():
    """Twenty cells over one like are twenty views of one decision."""
    from ai_jobs import setup_research

    one_like = [
        {
            "recipe_id": f"afterlike_d{offset}_first_m5_close_2r_v1",
            "occurrence_id": "afterlike|NVDA|LONG|2026-09-02",
            "net_r": 0.4,
            "entry_at": "2026-09-09T13:35:00",
        }
        for offset in (0, 1, 2, 3, 5)
    ]

    block = setup_research.after_like_block(one_like)

    assert block["episodes"] == 1, "one like, however many cells it filled"
    assert all(cell["n_episodes"] == 1 for cell in block["cells"])


def test_the_block_says_the_family_split_is_not_available_rather_than_omitting_it():
    """`outcome_path` has no column for the linked occurrence."""
    from ai_jobs import setup_research

    block = setup_research.after_like_block(_cell_rows(1))
    assert "NOT AVAILABLE" in block["family_split"]
    assert "like_occurrence_link" in block["family_split"]
    assert "window closes" in block["read_before_the_window_closes"]


def test_a_thin_cell_is_reported_and_never_eligible():
    from ai_jobs import setup_research

    block = setup_research.after_like_block(_cell_rows(3))
    cell = block["cells"][0]

    assert cell["n"] == 3
    assert cell["eligible"] is False
    assert cell["meets_n_floor"] is False
    assert cell["evidence_label"] == "discovery"


def test_the_weekend_table_shows_eligible_cells_only_and_says_which_blank_it_is():
    from ui.panels.weekend_prep_panel import after_like_view

    rows, note = after_like_view({})
    assert rows == [] and "graded yet" in note

    thin = {
        "after_like": {
            "episodes": 4,
            "cells": [{"day_offset": 0, "entry": "first_m5_close", "eligible": False}],
        }
    }
    rows, note = after_like_view(thin)
    assert rows == [], "a cell under the floor is not a weak answer, it is none"
    assert "evidence floor" in note and "4 like episode" in note

    ready = {
        "after_like": {
            "episodes": 40,
            "cells": [
                {"day_offset": 0, "entry": "first_m5_close", "eligible": True, "trimmed_mean_r": 0.2},
                {"day_offset": 3, "entry": "m15_acceptance", "eligible": True, "trimmed_mean_r": 0.9},
                {"day_offset": 5, "entry": "m5_retest_trigger", "eligible": False, "trimmed_mean_r": 5.0},
            ],
        }
    }
    rows, note = after_like_view(ready)
    assert [row["day_offset"] for row in rows] == [3, 0], "best mean R first"
    assert "DISCOVERY" in note and "window has not closed" in note


def test_the_narration_view_carries_the_eligible_after_like_cells_only():
    """R3's budget: handing the model twenty thin cells is how it outgrew it."""
    from ai_jobs import setup_research

    pack = setup_research.build_fact_pack([], {}, {}, coverage={"outcomes": 0})
    pack["after_like"] = {
        "cells": [
            {"day_offset": 0, "entry": "first_m5_close", "eligible": True},
            {"day_offset": 5, "entry": "m15_acceptance", "eligible": False},
        ]
    }

    view = setup_research.narration_view(pack)

    assert view["after_like_eligible"] == [
        {"day_offset": 0, "entry": "first_m5_close", "eligible": True}
    ]


def test_a_large_after_like_cell_is_eligible_rather_than_reading_an_absent_key():
    """R4 A1: `eligible` was read off `evidence_stats.summarize`, which never sets it.

    The key is absent from every `summarize` result, so `bool(summary.get(...))`
    was `False` for every cell of this grid no matter how large - a 60-episode,
    60-symbol, 28-session cell reported ineligible and the readout showed
    nothing. Tested THROUGH `after_like_block` rather than against a hand-written
    summary dict, because a hand-written dict is exactly the thing that could not
    have caught this.
    """
    from ai_jobs import setup_research

    rows = []
    for index in range(60):
        day = 2 + index % 28  # 28 distinct entry sessions
        rows.append(
            {
                "recipe_id": "afterlike_d0_first_m5_close_2r_v1",
                "occurrence_id": f"afterlike|SYM{index}|LONG|2026-09-02",
                "net_r": 0.4 if index % 3 else -0.6,
                "first_hit": "TARGET" if index % 3 else "STOP",
                "entry_at": f"2026-09-{day:02d}T13:35:00",
            }
        )

    block = setup_research.after_like_block(rows)
    cell = block["cells"][0]

    assert cell["n"] == 60
    assert cell["n_episodes"] == 60
    assert cell["meets_n_floor"] is True
    assert cell["eligible"] is True, "the floors are met; the key was simply absent"


def test_the_after_like_floors_are_the_packs_own_and_not_just_the_n_floor():
    """Sixty rows on ONE session still fails the session floor."""
    from ai_jobs import setup_research

    rows = [
        {
            "recipe_id": "afterlike_d1_m15_acceptance_2r_v1",
            "occurrence_id": f"afterlike|SYM{index}|LONG|2026-09-02",
            "net_r": 0.4,
            "first_hit": "TARGET",
            "entry_at": "2026-09-09T13:35:00",
        }
        for index in range(60)
    ]

    cell = setup_research.after_like_block(rows)["cells"][0]
    assert cell["meets_n_floor"] is True
    assert cell["eligible"] is False, "one entry session is not five"


def test_a_cell_measures_the_same_alone_as_it_does_after_its_siblings():
    """R4 A3: one series cache served twenty cells that look at different windows.

    `simulate_after_like_rows` hands ONE `series_cache` to all twenty cells of a
    like, and `_entry_from_derived` keyed it `(symbol, timeframe, as_of)` - no
    offset anywhere in the key. Each cell passes a different `ordered`, the bars
    from its own day offset onward, so the offset-2 cell was handed the offset-0
    cell's longer derived series. What a cell measured therefore depended on
    which sibling had run first, which is the one thing a grid comparing cells
    may never do.

    Reproduced on this fixture: the d2 M30 cell simulated alone saw 13 derived
    bars and refused (the EMA floor is 21); simulated after d0 it saw 39 and
    produced a row.
    """
    from research_warehouse import after_like, outcomes

    occurrence, bars = _inputs()
    as_of = _as_of()
    like = _like("2026-08-04")

    after_siblings = {
        row["recipe_id"]: row
        for row in after_like.simulate_after_like_rows(
            dict(like), dict(occurrence), bars, as_of=as_of, computed_at=as_of, run_id="test"
        )
    }

    for recipe in outcomes.AFTER_LIKE_RECIPES:
        alone = outcomes.simulate_after_like_entry(
            dict(like),
            dict(occurrence),
            bars,
            recipe,
            as_of=as_of,
            computed_at=as_of,
            run_id="test",
            series_cache={},
        )
        shared = after_siblings.get(recipe.recipe_id)
        if alone is None:
            assert shared is None, (
                f"{recipe.recipe_id} measured nothing alone but produced a row "
                "after its siblings ran"
            )
            continue
        assert shared is not None, recipe.recipe_id
        assert shared["entry_at"] == alone["entry_at"], recipe.recipe_id
        assert shared["net_r"] == alone["net_r"], recipe.recipe_id


def test_a_short_window_is_never_measured_against_a_longer_siblings_series():
    """The measurable/unmeasurable verdict itself was order-dependent.

    `_entry_from_derived` refuses below `min_bars` completed derived bars - the
    M30 EMA rule needs 21. An RTH session is 13 M30 bars, so a one-session
    window is under the floor and a three-session window is over it. Sharing a
    cache keyed without the window meant the one-session cell was handed the
    three-session series, cleared a floor it does not clear, and produced a row
    that the same cell run alone refuses. Which of the two answers a night got
    depended on which cell the loop reached first.
    """
    from research_warehouse import outcomes

    occurrence, bars = _inputs()
    as_of = _as_of()
    last_day = max(row["interval_start"].date() for row in bars)
    one_session = [row for row in bars if row["interval_start"].date() == last_day]
    assert len(one_session) < len(bars), "the fixture must span more than one session"

    always = lambda *_args: True  # noqa: E731 - the floor is what is under test

    shared: dict = {}
    outcomes._entry_from_derived(
        dict(occurrence), bars, timeframe="M30", as_of=as_of,
        qualifies=always, min_bars=outcomes.SETUP_ENTRY_TIMING_MIN_EMA_BARS,
        series_cache=shared,
    )
    after_sibling = outcomes._entry_from_derived(
        dict(occurrence), one_session, timeframe="M30", as_of=as_of,
        qualifies=always, min_bars=outcomes.SETUP_ENTRY_TIMING_MIN_EMA_BARS,
        series_cache=shared,
    )
    alone = outcomes._entry_from_derived(
        dict(occurrence), one_session, timeframe="M30", as_of=as_of,
        qualifies=always, min_bars=outcomes.SETUP_ENTRY_TIMING_MIN_EMA_BARS,
        series_cache={},
    )

    assert alone == (None, None), "one session is under the 21-bar EMA floor"
    assert after_sibling == alone, "a sibling's longer series cleared this cell's floor"


def test_the_derived_series_cache_is_keyed_by_the_window_it_was_built_from():
    """Two different windows over one symbol are two different series."""
    from research_warehouse import outcomes

    occurrence, bars = _inputs()
    as_of = _as_of()
    cache: dict = {}

    outcomes._entry_from_derived(
        dict(occurrence),
        bars,
        timeframe="M30",
        as_of=as_of,
        qualifies=lambda *_args: False,
        series_cache=cache,
    )
    first_keys = set(cache)
    outcomes._entry_from_derived(
        dict(occurrence),
        bars[len(bars) // 2 :],
        timeframe="M30",
        as_of=as_of,
        qualifies=lambda *_args: False,
        series_cache=cache,
    )

    assert set(cache) - first_keys, "the shorter window reused the longer one's key"
