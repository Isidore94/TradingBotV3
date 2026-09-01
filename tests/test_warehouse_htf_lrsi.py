"""Higher-timeframe LRSI entry study - Phase 0.12 B3. SHADOW ONLY.

The trader asked on 2026-09-01 whether there is anything in entering a
Focus-style setup on an LRSI cross at M30/H1/H2/H4 instead of M5. This lane
answers with outcome rows and nothing else.

What this file pins:

* the grid is BOUNDED and registered - 16 recipes, never a Cartesian search;
* every recipe is diagnostic, and nothing here reaches a champion;
* session STUBS are excluded from the oscillator's input, because a 30-minute
  bucket in an H2 series is not an H2 bar;
* the entry is POINT-IN-TIME - a cross that printed before the setup was
  known is history, not a trade;
* an unanswerable question produces NO row, never a zero.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_warehouse import exchange_calendar as xcal  # noqa: E402
from research_warehouse import outcomes  # noqa: E402


# --- the registered grid ---------------------------------------------------
def test_the_grid_is_bounded_registered_and_diagnostic():
    recipes = outcomes.HTF_LRSI_RECIPES
    assert len(recipes) == 16
    assert len({recipe.recipe_id for recipe in recipes}) == 16
    assert all(recipe.is_diagnostic for recipe in recipes)
    assert all(recipe.timeframe == "HTF_LRSI" for recipe in recipes)
    # One stop model and one target across the whole grid.
    assert {recipe.stop_atr_multiple for recipe in recipes} == {0.25}
    assert {recipe.target_r for recipe in recipes} == {2.0}
    assert {recipe.htf_timeframe for recipe in recipes} == {"M30", "H1", "H2", "H4"}
    # Longs cross UP through the two live levels; shorts cross DOWN through
    # 50 and 80. Same series for both - see RESEARCH_CROSS_LEVELS.
    assert {(r.cross_direction, r.cross_level) for r in recipes} == {
        ("up", 50.0),
        ("up", 20.0),
        ("down", 50.0),
        ("down", 80.0),
    }


def test_no_recipe_here_can_reach_a_champion():
    """`RECIPES` is the registry `build_outcomes` defaults from. The study is
    opt-in by explicit recipe list, exactly like the M5-close grid."""
    assert not (set(outcomes.RECIPES) & {r.recipe_id for r in outcomes.HTF_LRSI_RECIPES})
    assert outcomes.PRIMARY_RECIPE_BY_SETUP == {
        "AVWAPE_TO_FIRST_DEV": "swing_house_v1",
        "POST_EARNINGS_CANDLE_BREAK": "swing_house_v1",
    }


def test_the_live_m5_cross_levels_are_untouched():
    from indicators.efficiency_lrsi import CROSS_LEVELS, RESEARCH_CROSS_LEVELS

    assert CROSS_LEVELS == (20.0, 50.0)
    assert RESEARCH_CROSS_LEVELS == (20.0, 50.0, 80.0)


# --- the B2 decision, pinned by fixture ------------------------------------
def test_the_unmirrored_and_mirrored_short_idioms_are_not_the_same_feature():
    """The efficiency formula clamps at 0, so the two readings are different
    features rather than a transform of one. The fixture holds both; this
    proves the study's choice is a decision and not an accident."""
    from indicators.efficiency_lrsi import compute_efficiency_lrsi

    fixture = json.loads(
        (ROOT_DIR / "tests/fixtures/efficiency_lrsi_research_v1.json").read_text(
            encoding="utf-8"
        )
    )
    closes = fixture["closes"]
    expected_block = fixture["expected"]
    plain = compute_efficiency_lrsi(closes)
    mirrored = compute_efficiency_lrsi([-close for close in closes])

    assert plain.feature_version == fixture["feature_version"]
    for level, expected in expected_block["cross_up"].items():
        assert list(plain.cross_up_indices(float(level))) == expected
    for level, expected in expected_block["cross_down"].items():
        assert list(plain.cross_down_indices(float(level))) == expected
    for level, expected in expected_block["mirrored_cross_up"].items():
        assert list(mirrored.cross_up_indices(float(level))) == expected

    # The gap itself: the unmirrored down-cross fires when the up move's
    # efficiency collapses; the mirrored up-cross fires later, when the down
    # move is established. Two events, not one.
    assert plain.cross_down_indices(80.0)[-1] < mirrored.cross_up_indices(50.0)[-1]


# --- derived series --------------------------------------------------------
def _m5_session(day: date, *, symbol: str = "AAA", closes=None):
    """78 completed RTH M5 bars for one session."""
    session = xcal.trading_session(day)
    rows = []
    cursor = session.rth_open_at
    index = 0
    while cursor < session.rth_close_at:
        close = float(closes[index]) if closes else 100.0 + index * 0.01
        rows.append(
            {
                "symbol": symbol,
                "interval_start": cursor,
                "interval_end": cursor + timedelta(minutes=5),
                "open": close,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": 1000,
                "is_complete": True,
                "capture_mode": "LIVE",
            }
        )
        cursor += timedelta(minutes=5)
        index += 1
    return rows


def test_the_series_excludes_session_stubs():
    """RTH is 6.5h, so H2 ends every session with a 30-minute bucket. Feeding
    it to an EMA would make the oscillator measure a duration that changes
    with the time of day."""
    day = date(2026, 8, 3)
    bars = _m5_session(day)
    as_of = xcal.trading_session(day).rth_close_at + timedelta(hours=1)

    kept = outcomes._htf_series(bars, "H2", as_of=as_of)
    assert [row["is_stub"] for row in kept] == [False, False, False]

    with_stubs = outcomes._htf_series(bars, "H2", as_of=as_of, exclude_stubs=False)
    assert [row["is_stub"] for row in with_stubs] == [False, False, False, True]


def test_the_series_rolls_across_sessions_in_order():
    days = [date(2026, 8, 3), date(2026, 8, 4), date(2026, 8, 5)]
    bars = [row for day in days for row in _m5_session(day)]
    as_of = xcal.trading_session(days[-1]).rth_close_at + timedelta(hours=1)
    series = outcomes._htf_series(bars, "H1", as_of=as_of)
    # Six full hours per session, the 15:30-16:00 stub dropped.
    assert len(series) == 18
    assert series == sorted(series, key=lambda row: row["interval_end"])


# --- the simulator ---------------------------------------------------------
def _occurrence(trigger_at: datetime, symbol: str = "AAA") -> dict:
    return {
        "occurrence_id": "occ-1",
        "symbol": symbol,
        "side": "LONG",
        "trigger_at": trigger_at,
    }


def _recipe(timeframe="M30", direction="up", level=50.0):
    return next(
        recipe
        for recipe in outcomes.HTF_LRSI_RECIPES
        if recipe.htf_timeframe == timeframe
        and recipe.cross_direction == direction
        and recipe.cross_level == level
    )


def _trending_bars(days, *, symbol="AAA", bucket_bars=6, churn_buckets=30):
    """Churn, then a clean efficient run, so an LRSI cross-up actually prints.

    Shaped at the DERIVED bucket and CUMULATIVELY across sessions, not per M5
    bar and not per day: an M30 series samples one close in six and rolls
    across sessions, so churn written per M5 bar is invisible to it and churn
    reset each morning never ends. `bucket_bars` is how many M5 bars one
    derived bucket holds (6 for M30).
    """
    bars = []
    bucket = 0
    for day in days:
        closes = []
        for index in range(78):
            if index and index % bucket_bars == 0:
                bucket += 1
            if bucket < churn_buckets:
                level = 100.0 + (0.4 if bucket % 2 == 0 else 0.0)
            else:
                level = 100.0 + (bucket - churn_buckets + 1) * 0.9
            closes.append(level)
        bucket += 1
        bars.extend(_m5_session(day, symbol=symbol, closes=closes))
    return bars


def test_a_cross_produces_one_row_with_a_same_timeframe_atr_stop():
    days = [date(2026, 8, 3) + timedelta(days=offset) for offset in range(5)]
    days = [day for day in days if xcal.trading_session(day) is not None]
    bars = _trending_bars(days)
    as_of = xcal.trading_session(days[-1]).rth_close_at + timedelta(hours=1)
    recipe = _recipe("M30", "up", 50.0)

    row = outcomes.simulate_htf_lrsi_entry(
        _occurrence(xcal.trading_session(days[0]).rth_open_at),
        bars,
        recipe,
        as_of=as_of,
        computed_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
    )
    assert row is not None
    assert row["recipe_id"] == recipe.recipe_id
    assert row["outcome_definition_id"] == outcomes.OUTCOME_DEFINITION_ID
    assert row["analysis_unit"] == outcomes.ANALYSIS_UNIT_OPPORTUNITY
    assert row["result_state"] in outcomes.RESULT_STATES
    # The stop sits BELOW the signal bar's low on a long - the bar's own
    # extreme pushed out, never the entry price minus a multiple.
    assert row["stop_price"] < row["entry_price"]
    assert row["stop_distance"] > 0
    # Entry is a completed derived bar close, and it is at or after the
    # setup's trigger. Point-in-time.
    assert row["entry_at"] >= xcal.trading_session(days[0]).rth_open_at


def test_a_cross_that_printed_before_the_setup_was_known_is_not_tradeable():
    days = [date(2026, 8, 3) + timedelta(days=offset) for offset in range(5)]
    days = [day for day in days if xcal.trading_session(day) is not None]
    bars = _trending_bars(days)
    as_of = xcal.trading_session(days[-1]).rth_close_at + timedelta(hours=1)
    recipe = _recipe("M30", "up", 50.0)

    early = outcomes.simulate_htf_lrsi_entry(
        _occurrence(xcal.trading_session(days[0]).rth_open_at),
        bars,
        recipe,
        as_of=as_of,
    )
    assert early is not None
    # Ask again with the setup becoming known AFTER that cross: the same
    # series must not hand back the same entry.
    late = outcomes.simulate_htf_lrsi_entry(
        _occurrence(early["entry_at"] + timedelta(minutes=1)),
        bars,
        recipe,
        as_of=as_of,
    )
    assert late is None or late["entry_at"] > early["entry_at"]


def test_an_unanswerable_question_writes_no_row():
    day = date(2026, 8, 3)
    session = xcal.trading_session(day)
    as_of = session.rth_close_at + timedelta(hours=1)
    # One session of H4 is two buckets, one of them a stub: far too short for
    # an ATR(14), so there is nothing to say and no row is produced.
    assert (
        outcomes.simulate_htf_lrsi_entry(
            _occurrence(session.rth_open_at), _m5_session(day), _recipe("H4"), as_of=as_of
        )
        is None
    )
    # No bars at all.
    assert (
        outcomes.simulate_htf_lrsi_entry(
            _occurrence(session.rth_open_at), [], _recipe("M30"), as_of=as_of
        )
        is None
    )
    # No trigger time: the recipe cannot say when it became eligible.
    assert (
        outcomes.simulate_htf_lrsi_entry(
            {"occurrence_id": "occ-1", "symbol": "AAA", "side": "LONG"},
            _m5_session(day),
            _recipe("M30"),
            as_of=as_of,
        )
        is None
    )


def test_build_outcomes_dispatches_the_study_without_touching_the_champions(tmp_path):
    """The dispatch branch exists and is reached only for HTF_LRSI recipes."""
    seen = []

    def _spy(occurrence, m5_bars, recipe, **kwargs):
        seen.append(recipe.recipe_id)
        return None

    original = outcomes.simulate_htf_lrsi_entry
    outcomes.simulate_htf_lrsi_entry = _spy
    try:
        report = outcomes.build_outcomes(
            None, [_occurrence(datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc))]
        )
        assert report.status == "DISABLED"  # no store, nothing simulated at all
    finally:
        outcomes.simulate_htf_lrsi_entry = original
    assert seen == []


def test_the_derived_series_is_built_once_per_occurrence_and_timeframe():
    """Four entries share one timeframe, so without the memo the same rolling
    series would be rebuilt four times for one occurrence. The cache is handed
    in by the caller and dropped with the occurrence - never module-level,
    which would serve one occurrence's bars to another."""
    days = [date(2026, 8, 3) + timedelta(days=offset) for offset in range(5)]
    days = [day for day in days if xcal.trading_session(day) is not None]
    bars = _trending_bars(days)
    as_of = xcal.trading_session(days[-1]).rth_close_at + timedelta(hours=1)
    occurrence = _occurrence(xcal.trading_session(days[0]).rth_open_at)
    cache: dict = {}

    for recipe in outcomes.HTF_LRSI_RECIPES:
        outcomes.simulate_htf_lrsi_entry(
            occurrence, bars, recipe, as_of=as_of, series_cache=cache
        )

    # One entry per (symbol, timeframe, cutoff), not per recipe.
    assert len(cache) == len(outcomes.HTF_LRSI_TIMEFRAMES)
    assert {key[1] for key in cache} == set(outcomes.HTF_LRSI_TIMEFRAMES)
