"""Recipe simulation and ``house_default_v1`` (plan Phase 6, sec 12, 14.2).

Pinned here: the cost arithmetic, STOP_FIRST as the primary reading of an
ambiguous bar with TARGET_FIRST kept as ``r_upper_bound``, MATURED as a derived
predicate rather than a stored state, only the slice's result-state subset, the
normative recipe-to-setup mapping, and that alternative recipes on one
occurrence stay one episode.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.research_warehouse import exchange_calendar as xcal, outcomes, schemas  # noqa: E402
from scripts.research_warehouse.store import ResearchStore  # noqa: E402

UTC = timezone.utc
TRIGGER_DAY = date(2026, 8, 3)
TRIGGER_AT = xcal.trading_session(TRIGGER_DAY).rth_close_at
NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def _occurrence(**overrides):
    row = {
        "occurrence_id": "occ-1",
        "symbol": "AAPL",
        "canonical_setup_id": "AVWAPE_TO_FIRST_DEV",
        "side": "LONG",
        "structural_timeframe": "D1",
        "status": "TRIGGERED",
        "trigger_at": TRIGGER_AT,
        "entry_price_ref": 100.0,
        "stop_price_ref": 95.0,  # a 5.00 risk unit
        "event_at": TRIGGER_AT,
    }
    row.update(overrides)
    return row


def _d1(closes, *, start=TRIGGER_DAY, highs=None, lows=None):
    """Sessions starting at the trigger day; closes[0] is the trigger bar."""
    rows = []
    day = start
    for index, close in enumerate(closes):
        while not xcal.is_trading_day(day):
            day += timedelta(days=1)
        rows.append(
            {
                "symbol": "AAPL",
                "session_date": day,
                "session_id": xcal.session_id_for(day),
                "open": close,
                "high": highs[index] if highs else close + 0.5,
                "low": lows[index] if lows else close - 0.5,
                "close": close,
                "volume": 1_000_000,
                "capture_mode": "BACKFILL",
                "is_complete": True,
            }
        )
        day += timedelta(days=1)
    return rows


# --- house_default_v1 cost arithmetic --------------------------------------
def test_net_r_arithmetic_is_the_declared_formula():
    # gross 2R on a $5.00 stop, fallback half-spread = max(0.01, 2bp x 100) = 0.02
    expected_cost = 2 * (0.0035 + 0.02) / 5.0
    assert outcomes.half_spread(100.0) == pytest.approx(0.02)
    assert outcomes.net_r(2.0, 5.0, 100.0) == pytest.approx(2.0 - expected_cost)

    # A penny stock's floor is the $0.01 minimum, not 2bp.
    assert outcomes.half_spread(2.0) == pytest.approx(0.01)
    # An observed NBBO half-spread wins over the fallback.
    assert outcomes.half_spread(100.0, 0.005) == pytest.approx(0.005)
    assert outcomes.net_r(1.0, 5.0, 100.0, observed_half_spread=0.005) == pytest.approx(
        1.0 - 2 * (0.0035 + 0.005) / 5.0
    )
    # A tight stop pays proportionally more: costs are in R, not dollars.
    assert outcomes.net_r(1.0, 0.5, 100.0) < outcomes.net_r(1.0, 5.0, 100.0)
    assert outcomes.COMMISSION_PER_SHARE == 0.0035


def test_net_r_includes_the_market_entry_slippage_bullet():
    # Sec 14.2's separate bullet: +1 half_spread slippage on stop/market
    # entries. Every slice recipe enters via a declared market-type order
    # (precommitted MOC or completed-bar close), so every simulated row pays it.
    spread = 0.02  # fallback half-spread at $100
    assert outcomes.ENTRY_SLIPPAGE_HALF_SPREADS == 1.0
    assert outcomes.net_r(2.0, 5.0, 100.0, entry_slippage_half_spreads=1.0) == pytest.approx(
        2.0 - (2 * (0.0035 + spread) + spread) / 5.0
    )
    bars = _d1([100.0, 97.0, 94.0, 93.0, 99.0])
    row = outcomes.simulate_swing(_occurrence(), bars, outcomes.SWING_HOUSE_V1, as_of=NOW)
    assert row["net_r"] == pytest.approx(row["gross_r"] - (2 * (0.0035 + spread) + spread) / 5.0)


def test_matured_is_derived_never_stored():
    assert "MATURED" not in schemas.RESULT_STATES
    row = {"maturity_at": datetime(2026, 8, 20, 20, 0, tzinfo=UTC)}
    assert outcomes.is_matured(row, datetime(2026, 8, 21, tzinfo=UTC)) is True
    assert outcomes.is_matured(row, datetime(2026, 8, 19, tzinfo=UTC)) is False
    assert outcomes.is_matured({"maturity_at": None}, NOW) is False


# --- swing simulation ------------------------------------------------------
def test_a_clean_stop_is_stopped_with_the_declared_close_failures():
    # Long: two consecutive closes below 95 invalidate (swing_house_v1).
    bars = _d1([100.0, 97.0, 94.0, 93.0, 99.0])
    row = outcomes.simulate_swing(_occurrence(), bars, outcomes.SWING_HOUSE_V1, as_of=NOW)

    assert row["result_state"] == "STOPPED" and row["first_hit"] == "STOP"
    assert row["path_resolution"] == "EXACT"
    assert row["stop_distance"] == pytest.approx(5.0)
    assert row["gross_r"] == pytest.approx((93.0 - 100.0) / 5.0)
    assert row["net_r"] < row["gross_r"]  # costs always subtract
    assert row["analysis_unit"] == "OPPORTUNITY"
    assert row["outcome_definition_id"] == "house_default_v1"


def test_a_single_close_below_does_not_stop_a_two_close_recipe():
    bars = _d1([100.0, 94.0, 99.0, 101.0, 102.0])
    row = outcomes.simulate_swing(_occurrence(), bars, outcomes.SWING_HOUSE_V1, as_of=NOW)
    assert row["result_state"] != "STOPPED"


def test_post_earnings_families_invalidate_on_one_close():
    occurrence = _occurrence(
        canonical_setup_id="POST_EARNINGS_CANDLE_BREAK", side="SHORT", entry_price_ref=100.0, stop_price_ref=105.0
    )
    bars = _d1([100.0, 106.0, 101.0, 99.0])
    row = outcomes.simulate_swing(occurrence, bars, outcomes.SWING_HOUSE_V1, as_of=NOW)
    assert outcomes.POST_EARNINGS_CLOSE_FAILURES["POST_EARNINGS_CANDLE_BREAK"] == 1
    assert row["result_state"] == "STOPPED"
    # Short side: a close ABOVE the stop is the failure, and R is mirrored.
    assert row["gross_r"] == pytest.approx((100.0 - 106.0) / 5.0)


def test_same_bar_ambiguity_is_stop_first_with_the_target_bound_kept():
    # Session 2 trades through both the 2R target (110) and the stop.
    occurrence = _occurrence()
    bars = _d1([100.0, 99.0, 94.0], highs=[100.5, 100.0, 112.0], lows=[99.5, 98.0, 90.0])
    row = outcomes.simulate_swing(occurrence, bars, outcomes.CONTROL_FIXED_1R2R_V1, as_of=NOW)

    assert row["result_state"] == "AMBIGUOUS_BAR"
    assert row["path_resolution"] == "AMBIGUOUS"
    assert row["first_hit"] == "STOP"  # the preregistered conservative primary
    assert row["r_lower_bound"] == pytest.approx(-1.0)
    assert row["r_upper_bound"] == pytest.approx(2.0)
    # The primary estimate IS the lower bound; the optimistic read is retained,
    # never averaged in.
    assert row["gross_r"] == pytest.approx(row["r_lower_bound"])


def test_the_fixed_control_targets_2r():
    bars = _d1([100.0, 103.0, 107.0], highs=[100.5, 104.0, 111.0], lows=[99.5, 102.0, 106.0])
    row = outcomes.simulate_swing(_occurrence(), bars, outcomes.CONTROL_FIXED_1R2R_V1, as_of=NOW)
    assert row["result_state"] == "TARGETED" and row["first_hit"] == "TARGET"
    assert row["gross_r"] == pytest.approx(2.0)


def test_the_time_only_control_never_stops():
    closes = [100.0] + [80.0] * 20  # a brutal drawdown
    row = outcomes.simulate_swing(_occurrence(), _d1(closes), outcomes.CONTROL_TIME_ONLY_V1, as_of=NOW)
    assert row["result_state"] == "EXPIRED" and row["first_hit"] == "NEITHER"
    assert row["mae_r"] < 0
    assert row["gross_r"] == pytest.approx((80.0 - 100.0) / 5.0)


def test_the_18_session_time_stop_and_checkpoint_grid():
    closes = [100.0] + [100.0 + index for index in range(1, 25)]
    row = outcomes.simulate_swing(_occurrence(), _d1(closes), outcomes.CONTROL_TIME_ONLY_V1, as_of=NOW)

    assert outcomes.SWING_TIME_STOP_SESSIONS == 18
    assert row["r_at_s1"] == pytest.approx(1 / 5.0)
    assert row["r_at_s5"] == pytest.approx(5 / 5.0)
    assert row["r_at_s18"] == pytest.approx(18 / 5.0)
    # The swing grid is filled; intraday checkpoints stay null without M5 bars.
    assert row["r_at_60m"] is None and row["r_at_eod"] is None
    assert row["maturity_at"] is not None


def test_maturity_is_a_calendar_fact_not_a_data_artifact():
    row = outcomes.simulate_swing(_occurrence(), _d1([100.0, 101.0]), outcomes.SWING_HOUSE_V1, as_of=NOW)
    # Only two sessions of path exist, but maturity is still 18 sessions out.
    assert row["maturity_at"] is not None
    assert row["maturity_at"] > TRIGGER_AT + timedelta(days=20)
    # And an unresolved path carries no realized R - checkpoints and MFE/MAE
    # are path facts, but an interim reading must never enter a mean.
    assert row["result_state"] == "TRUNCATED"
    assert row["gross_r"] is None and row["net_r"] is None


def test_maturity_is_min_of_resolution_and_the_time_stop():
    # sec 14.2: min(+18 sessions, stop/target/expiry). A trade that stopped on
    # session 3 matured on session 3, not weeks later.
    bars = _d1([100.0, 97.0, 94.0, 93.0, 99.0])
    row = outcomes.simulate_swing(_occurrence(), bars, outcomes.SWING_HOUSE_V1, as_of=NOW)
    assert row["result_state"] == "STOPPED"
    assert row["maturity_at"] == row["first_hit_at"]
    assert row["maturity_at"] < TRIGGER_AT + timedelta(days=10)
    assert outcomes.is_matured(row, TRIGGER_AT + timedelta(days=10)) is True


def test_house_management_takes_a_partial_at_band2_and_runs_to_band3():
    bands = {"UPPER_1": 105.0, "UPPER_2": 110.0, "UPPER_3": 120.0}
    bars = _d1([100.0, 108.0, 112.0, 121.0], highs=[101.0, 110.5, 113.0, 122.0], lows=[99.0, 107.0, 111.0, 119.0])
    row = outcomes.simulate_swing(_occurrence(), bars, outcomes.SWING_HOUSE_V1, bands=bands, as_of=NOW)

    partial_r = (110.0 - 100.0) / 5.0  # 2R on half
    runner_r = (120.0 - 100.0) / 5.0  # 4R on the rest
    assert row["gross_r"] == pytest.approx(0.5 * partial_r + 0.5 * runner_r)


def test_house_management_trails_to_band1_after_the_partial():
    bands = {"UPPER_1": 105.0, "UPPER_2": 110.0, "UPPER_3": 130.0}
    bars = _d1([100.0, 108.0, 111.0, 104.0], highs=[101.0, 110.5, 112.0, 106.0], lows=[99.0, 107.0, 110.0, 103.0])
    row = outcomes.simulate_swing(_occurrence(), bars, outcomes.SWING_HOUSE_V1, bands=bands, as_of=NOW)

    partial_r = (110.0 - 100.0) / 5.0
    trail_r = (104.0 - 100.0) / 5.0  # closed back under band 1
    assert row["gross_r"] == pytest.approx(0.5 * partial_r + 0.5 * trail_r)


def test_management_credits_the_partial_when_the_runner_stops_out():
    # 50% off at band 2 on session 1, then the runner trails out at the close
    # back through band 1. The partial is policy arithmetic and stays credited;
    # reporting a full-size stop here was the D4 defect.
    bands = {"UPPER_1": 105.0, "UPPER_2": 110.0, "UPPER_3": 130.0}
    bars = _d1(
        [100.0, 108.0, 94.5, 93.0],
        highs=[101.0, 111.0, 109.0, 95.0],
        lows=[99.0, 107.0, 94.0, 92.0],
    )
    row = outcomes.simulate_swing(_occurrence(), bars, outcomes.SWING_HOUSE_V1, bands=bands, as_of=NOW)

    partial_r = (110.0 - 100.0) / 5.0
    runner_r = (94.5 - 100.0) / 5.0  # the trail exit at that bar's close
    assert row["result_state"] == "STOPPED"
    assert row["gross_r"] == pytest.approx(0.5 * partial_r + 0.5 * runner_r)
    assert row["first_hit"] == "TARGET"  # the band-2 partial filled first
    # min(): the position matured when its last piece exited, on the trail bar.
    assert row["maturity_at"] > row["first_hit_at"]


def test_management_never_looks_past_the_time_stop():
    # A band touch on forward session 24 is outside the 18-session window and
    # must contribute nothing - crediting it was the D3 defect.
    bands = {"UPPER_1": 105.0, "UPPER_2": 110.0, "UPPER_3": 120.0}
    highs = [101.0] * 25
    highs[24] = 121.0
    bars = _d1([100.0] * 25, highs=highs, lows=[99.0] * 25)
    row = outcomes.simulate_swing(_occurrence(), bars, outcomes.SWING_HOUSE_V1, bands=bands, as_of=NOW)
    assert row["result_state"] == "EXPIRED"
    assert row["gross_r"] == pytest.approx(0.0)


def test_a_setup_that_never_triggered_is_recorded_as_evidence():
    row = outcomes.simulate_swing(
        _occurrence(trigger_at=None, status="ELIGIBLE"), _d1([100.0]), outcomes.SWING_HOUSE_V1, as_of=NOW
    )
    assert row["result_state"] == "NO_TRIGGER"
    assert row["entry_at"] is None and row["gross_r"] is None
    assert row["censor_reason"] == "never triggered"


def test_mfe_and_mae_track_the_whole_path():
    bars = _d1([100.0, 104.0, 97.0, 102.0], highs=[100.5, 110.0, 105.0, 103.0], lows=[99.5, 103.0, 92.0, 101.0])
    row = outcomes.simulate_swing(_occurrence(), bars, outcomes.CONTROL_TIME_ONLY_V1, as_of=NOW)
    assert row["mfe_r"] == pytest.approx((110.0 - 100.0) / 5.0)
    assert row["mae_r"] == pytest.approx((92.0 - 100.0) / 5.0)
    assert row["time_to_mfe_min"] == 24 * 60


# --- intraday bounce recipe ------------------------------------------------
def _m5(session, count, base=100.0, step=0.5):
    return [
        {
            "symbol": "AAPL",
            "interval_start": session.rth_open_at + timedelta(minutes=5 * index),
            "interval_end": session.rth_open_at + timedelta(minutes=5 * (index + 1)),
            "open": base + index * step,
            "high": base + index * step + 0.25,
            "low": base + index * step - 0.25,
            "close": base + index * step,
            "capture_mode": "LIVE",
        }
        for index in range(count)
    ]


def test_intraday_bounce_uses_the_linked_event_and_the_production_stop():
    session = xcal.trading_session(TRIGGER_DAY)
    bars = _m5(session, 20)
    bounce_bar = bars[2]
    event = {
        "symbol": "AAPL",
        "bounce_at": bounce_bar["interval_start"],
        "bounce_type": "vwap_band",
        "stop_price": bounce_bar["close"] - 1.0,
    }
    row = outcomes.simulate_intraday_bounce(
        _occurrence(), event, bars, outcomes.INTRADAY_BOUNCE_V1, as_of=NOW, session=session
    )
    assert row["recipe_id"] == "intraday_bounce_v1" and row["analysis_unit"] == "ATTEMPT"
    assert row["entry_price"] == pytest.approx(bounce_bar["close"])
    assert row["stop_distance"] == pytest.approx(1.0)
    assert row["maturity_at"] == session.rth_close_at  # EOD maturity
    assert row["r_at_60m"] is not None
    assert row["r_at_s1"] is None  # swing checkpoints stay null
    assert row["result_state"] in schemas.RESULT_STATES


def test_intraday_bounce_never_walks_past_the_entry_session():
    session = xcal.trading_session(TRIGGER_DAY)
    bars = _m5(session, 78)  # the full RTH session, rising
    next_session = xcal.trading_session(TRIGGER_DAY + timedelta(days=1))
    # A next-day collapse through the stop must be invisible to an EOD recipe.
    bars += _m5(next_session, 10, base=50.0)
    bounce_bar = bars[2]
    event = {"symbol": "AAPL", "bounce_at": bounce_bar["interval_start"], "stop_price": bounce_bar["close"] - 1.0}
    row = outcomes.simulate_intraday_bounce(
        _occurrence(), event, bars, outcomes.INTRADAY_BOUNCE_V1, as_of=NOW, session=session
    )
    assert row["result_state"] == "EXPIRED"
    assert row["maturity_at"] == session.rth_close_at
    # r_at_eod means the ENTRY session's close, never the last bar provided.
    assert row["r_at_eod"] == pytest.approx((bars[77]["close"] - bounce_bar["close"]) / 1.0)


def test_intraday_bounce_mid_session_is_open_and_a_short_archive_is_truncated():
    session = xcal.trading_session(TRIGGER_DAY)
    bars = _m5(session, 20)
    bounce_bar = bars[2]
    event = {"symbol": "AAPL", "bounce_at": bounce_bar["interval_start"], "stop_price": bounce_bar["close"] - 1.0}

    # Simulated while the session is still trading: the trade is simply OPEN.
    mid_session = session.rth_open_at + timedelta(minutes=110)
    open_row = outcomes.simulate_intraday_bounce(
        _occurrence(), event, bars, outcomes.INTRADAY_BOUNCE_V1, as_of=mid_session, session=session
    )
    assert open_row["result_state"] == "OPEN"
    assert open_row["gross_r"] is None and open_row["net_r"] is None

    # Same bars after the close: the archive stops early, so the evidence is
    # truncated - never a finished trade at whatever bar happened to be last.
    late_row = outcomes.simulate_intraday_bounce(
        _occurrence(), event, bars, outcomes.INTRADAY_BOUNCE_V1, as_of=NOW, session=session
    )
    assert late_row["result_state"] == "TRUNCATED"
    assert late_row["gross_r"] is None and late_row["net_r"] is None


def test_intraday_gap_through_the_stop_fills_at_the_open():
    session = xcal.trading_session(TRIGGER_DAY)
    bars = _m5(session, 6)
    bounce_bar = bars[2]
    bars[3] = dict(bars[3], open=95.0, high=95.5, low=94.0, close=94.5)
    event = {"symbol": "AAPL", "bounce_at": bounce_bar["interval_start"], "stop_price": bounce_bar["close"] - 1.0}
    row = outcomes.simulate_intraday_bounce(
        _occurrence(), event, bars, outcomes.INTRADAY_BOUNCE_V1, as_of=NOW, session=session
    )
    assert row["result_state"] == "STOPPED"
    # The market never traded at the stop: the fill is the gap open (sec 14.3).
    assert row["gross_r"] == pytest.approx((95.0 - bounce_bar["close"]) / 1.0)
    assert row["gross_r"] < -1.0
    assert row["maturity_at"] == row["first_hit_at"]  # min(EOD, stop)


def test_no_linked_bounce_event_means_no_intraday_row(store):
    report = outcomes.build_outcomes(
        store,
        [_occurrence()],
        d1_by_symbol={"AAPL": _d1([100.0, 101.0])},
        recipes=[outcomes.INTRADAY_BOUNCE_V1],
        as_of=NOW,
    )
    # The warehouse never re-detects a bounce to manufacture a row (sec 19.3).
    assert report.status == "NOTHING_TO_SIMULATE"
    assert report.skipped == {"NO_LINKED_BOUNCE_EVENT": 1}


# --- the build job ---------------------------------------------------------
def test_alternative_recipes_share_one_occurrence_and_one_episode(store):
    occurrence = _occurrence()
    report = outcomes.build_outcomes(
        store, [occurrence], d1_by_symbol={"AAPL": _d1([100.0] + [101.0] * 20)}, as_of=NOW
    )
    # house + the band challenger's twin (M4.2) + two controls
    assert report.rows == 4
    rows = store.read_table("outcome_path").to_pylist()
    assert {row["recipe_id"] for row in rows} == {
        "swing_house_v1",
        "swing_house_variant_v1",
        "control_fixed_1r2r_v1",
        "control_time_only_v1",
    }
    # Four recipes, ONE occurrence: correlated diagnostics, never four trades.
    assert len({row["occurrence_id"] for row in rows}) == 1
    by_recipe = {row["recipe_id"]: row for row in rows}
    assert {
        row["outcome_definition_id"]
        for recipe_id, row in by_recipe.items()
        if recipe_id != "swing_house_variant_v1"
    } == {"house_default_v1"}
    # The twin's own definition id is a FENCE: every reader that filters on the
    # house default (queries.slice_readout) can never see a challenger row.
    assert by_recipe["swing_house_variant_v1"]["outcome_definition_id"] == "band_variant_v1"

    again = outcomes.build_outcomes(store, [occurrence], d1_by_symbol={"AAPL": _d1([100.0])}, as_of=NOW)
    assert again.status == "NOTHING_TO_SIMULATE" and again.skipped["ALREADY_SIMULATED"] == 4
    assert store.read_table("outcome_path").num_rows == 4


def test_an_open_outcome_is_resimulated_and_superseded(store):
    occurrence = _occurrence()
    early_as_of = TRIGGER_AT + timedelta(days=3)
    first = outcomes.build_outcomes(
        store,
        [occurrence],
        d1_by_symbol={"AAPL": _d1([100.0, 105.0, 105.0])},
        recipes=[outcomes.CONTROL_TIME_ONLY_V1],
        as_of=early_as_of,
        now=early_as_of,
    )
    assert first.rows == 1
    interim = next(iter(outcomes.latest_outcomes(store).values()))
    assert interim["result_state"] == "OPEN"
    assert interim["gross_r"] is None  # +1R interim must never be quotable

    # Weeks later the full path exists: the trade actually finished at -0.8R.
    full_path = _d1([100.0] + [96.0] * 20)
    second = outcomes.build_outcomes(
        store,
        [occurrence],
        d1_by_symbol={"AAPL": full_path},
        recipes=[outcomes.CONTROL_TIME_ONLY_V1],
        as_of=NOW,
        now=NOW,
    )
    assert second.rows == 1
    # Append-only: the interim row stays as history, the reader takes latest.
    assert store.read_table("outcome_path").num_rows == 2
    final = outcomes.latest_outcomes(store)[outcomes.outcome_key(interim)]
    assert final["result_state"] == "EXPIRED"
    assert final["gross_r"] == pytest.approx((96.0 - 100.0) / 5.0)

    # Terminal evidence is immutable: a third build writes nothing.
    third = outcomes.build_outcomes(
        store,
        [occurrence],
        d1_by_symbol={"AAPL": full_path},
        recipes=[outcomes.CONTROL_TIME_ONLY_V1],
        as_of=NOW,
        now=NOW + timedelta(days=1),
    )
    assert third.status == "NOTHING_TO_SIMULATE" and third.skipped == {"ALREADY_SIMULATED": 1}
    assert store.read_table("outcome_path").num_rows == 2


def test_a_rebuild_that_learned_nothing_writes_nothing(store):
    occurrence = _occurrence()
    short_path = _d1([100.0, 105.0, 105.0])
    early_as_of = TRIGGER_AT + timedelta(days=3)
    outcomes.build_outcomes(
        store,
        [occurrence],
        d1_by_symbol={"AAPL": short_path},
        recipes=[outcomes.CONTROL_TIME_ONLY_V1],
        as_of=early_as_of,
        now=early_as_of,
    )
    again = outcomes.build_outcomes(
        store,
        [occurrence],
        d1_by_symbol={"AAPL": short_path},
        recipes=[outcomes.CONTROL_TIME_ONLY_V1],
        as_of=early_as_of,
        now=early_as_of + timedelta(hours=1),
    )
    # The OPEN row is recompute-eligible, but identical knowledge is not a
    # new revision - re-running with the same inputs is a no-op.
    assert again.status == "NOTHING_TO_SIMULATE" and again.skipped == {"UNCHANGED": 1}
    assert store.read_table("outcome_path").num_rows == 1


def test_only_the_slice_result_states_are_emitted(store):
    outcomes.build_outcomes(
        store,
        [_occurrence(), _occurrence(occurrence_id="occ-2", trigger_at=None)],
        d1_by_symbol={"AAPL": _d1([100.0, 94.0, 93.0])},
        as_of=NOW,
    )
    states = {row["result_state"] for row in store.read_table("outcome_path").to_pylist()}
    assert states <= set(schemas.RESULT_STATES)
    assert "MATURED" not in states


def test_the_recipe_mapping_is_the_normative_one():
    assert outcomes.PRIMARY_RECIPE_BY_SETUP == {
        "AVWAPE_TO_FIRST_DEV": "swing_house_v1",
        "POST_EARNINGS_CANDLE_BREAK": "swing_house_v1",
    }
    assert outcomes.CONTROL_FIXED_1R2R_V1.is_control and outcomes.CONTROL_TIME_ONLY_V1.is_control
    assert outcomes.DIAGNOSTIC_ATR_STOP_V1.is_diagnostic
    assert outcomes.DIAGNOSTIC_ATR_STOP_V1.stop == "signal_bar_extreme_plus_0_25_atr_m5_14"
    assert set(outcomes.RECIPES) == {
        "swing_house_v1",
        # M4.2: the band challenger's twin. Additive - it is in the registry so
        # a stored row resolves back to its policy, and deliberately NOT in
        # `PRIMARY_RECIPE_BY_SETUP`, which is asserted unchanged above.
        "swing_house_variant_v1",
        "intraday_bounce_v1",
        "control_fixed_1r2r_v1",
        "control_time_only_v1",
        "diag_signal_bar_atr_stop_v1",
    }


def test_outcomes_are_disabled_without_a_store():
    assert outcomes.build_outcomes(None, [_occurrence()]).status == "DISABLED"
