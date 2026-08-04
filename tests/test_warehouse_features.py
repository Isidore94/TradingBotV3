"""Tier-1 feature snapshots (plan Phase 5; frozen columns in sec 7.1).

Pinned here: the snapshots contain exactly the frozen columns, they are
deterministic (same sealed inputs -> identical values and the same
``input_manifest_hash``), they are point-in-time (a snapshot never sees a bar
that had not completed at its own event time), and production context values
are stored as production computed them rather than recomputed here.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.research_warehouse import (  # noqa: E402
    exchange_calendar as xcal,
    features,
    schemas,
)
from scripts.research_warehouse.store import ResearchStore  # noqa: E402

UTC = timezone.utc
NOW = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def _d1_row(day: date, index: int, symbol="AAPL", capture_mode="BACKFILL"):
    base = 100.0 + index
    return {
        "symbol": symbol,
        "session_id": xcal.session_id_for(day),
        "session_date": day,
        "open": base,
        "high": base + 2.0,
        "low": base - 1.5,
        "close": base + 1.0,
        "volume": 1_000_000 + index * 1000,
        "adjustment_version": None,
        "corporate_action_id": None,
        "provider": "IBKR",
        "quality": "COMPLETE",
        "is_complete": True,
        "event_at": datetime(day.year, day.month, day.day, tzinfo=UTC),
        "observed_at": datetime(day.year, day.month, day.day, 21, tzinfo=UTC),
        "capture_mode": capture_mode,
        "revision_id": "",
        "supersedes_revision_id": "",
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "d1",
    }


def _history(days=30, symbol="AAPL", capture_mode="BACKFILL"):
    rows = []
    day = date(2026, 6, 1)
    index = 0
    while len(rows) < days:
        if xcal.is_trading_day(day):
            rows.append(_d1_row(day, index, symbol=symbol, capture_mode=capture_mode))
            index += 1
        day += timedelta(days=1)
    return rows


def _m5_row(session, index, symbol="AAPL"):
    start = session.rth_open_at + timedelta(minutes=5 * index)
    base = 200.0 + index
    return {
        "symbol": symbol,
        "interval_start": start,
        "interval_end": start + timedelta(minutes=5),
        "session_id": session.session_id,
        "session_phase": "RTH",
        "open": base,
        "high": base + 0.5,
        "low": base - 0.5,
        "close": base + 0.25,
        "volume": 10_000 + index,
        "vwap": None,
        "trade_count": None,
        "provider": "IBKR",
        "is_complete": True,
        "quality": "COMPLETE",
        "source_hash": "",
        "event_at": start + timedelta(minutes=5),
        "observed_at": start + timedelta(minutes=6),
        "capture_mode": "LIVE",
        "revision_id": "",
        "supersedes_revision_id": "",
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "tee",
    }


# --- daily snapshots -------------------------------------------------------
def test_daily_snapshot_has_exactly_the_frozen_columns():
    history = _history()
    row = features.compute_daily_features(
        "AAPL", history, session_date=history[-1]["session_date"], anchor_index=5, computed_at=NOW
    )
    assert set(row) == set(schemas.FEATURE_SNAPSHOT_DAILY.names)
    assert row["feature_set_version"] == features.FEATURE_SET_VERSION
    assert row["event_at"] == xcal.trading_session(history[-1]["session_date"]).rth_close_at
    assert row["input_capture_mode_worst"] == "BACKFILL"


def test_daily_grid_uses_the_champion_indicator_conventions():
    history = _history(days=60)
    row = features.compute_daily_features(
        "AAPL", history, session_date=history[-1]["session_date"], computed_at=NOW
    )
    grid = features.indicator_grid(history)
    assert row["ema8"] == pytest.approx(float(grid["ema_8"].iloc[-1]), abs=1e-12)
    assert row["ema21"] == pytest.approx(float(grid["ema_21"].iloc[-1]), abs=1e-12)
    assert row["sma50"] == pytest.approx(float(grid["sma_50"].iloc[-1]), abs=1e-12)
    # 60 sessions cannot support a 100- or 200-day mean: null, never a partial
    # average dressed up as one.
    assert row["sma100"] is None and row["sma200"] is None
    assert row["dist_sma100_atr"] is None
    assert row["dist_sma50_atr"] == pytest.approx((row["close"] - row["sma50"]) / row["atr14"], abs=1e-12)


def test_atr14_matches_the_house_true_range_method():
    history = _history(days=20)
    expected_window = history[-14:]
    ranges = []
    previous = None
    for bar in expected_window:
        if previous is None:
            ranges.append(bar["high"] - bar["low"])
        else:
            ranges.append(max(bar["high"] - bar["low"], abs(bar["high"] - previous), abs(bar["low"] - previous)))
        previous = bar["close"]
    assert features.atr(history) == pytest.approx(sum(ranges) / len(ranges), abs=1e-12)


def test_daily_snapshot_is_point_in_time():
    history = _history(days=40)
    cutoff = history[20]["session_date"]
    row = features.compute_daily_features("AAPL", history, session_date=cutoff, computed_at=NOW)
    truncated = features.compute_daily_features(
        "AAPL", history[:21], session_date=cutoff, computed_at=NOW
    )
    # Bars after the snapshot date are invisible to it, so the value computed
    # from the full history equals the value computed from history-so-far.
    assert row == truncated
    assert row["close"] == history[20]["close"]


def test_avwap_block_and_favorite_zone_are_filled_from_the_anchor():
    history = _history(days=30)
    anchor_index = 10
    row = features.compute_daily_features(
        "AAPL", history, session_date=history[-1]["session_date"], anchor_index=anchor_index, computed_at=NOW
    )
    vwap, _stdev, bands = features.anchored_vwap_bands(history, anchor_index)
    assert row["avwape_value"] == pytest.approx(vwap, abs=1e-9)
    assert row["avwape_upper_1"] == pytest.approx(bands["UPPER_1"], abs=1e-9)
    assert row["avwape_lower_3"] == pytest.approx(bands["LOWER_3"], abs=1e-9)
    # coord = (close - AVWAPE) / (UPPER_1 - AVWAPE), exactly as sec 6.2 states.
    expected = (row["close"] - vwap) / (bands["UPPER_1"] - vwap)
    assert row["favorite_zone_coord"] == pytest.approx(expected, abs=1e-12)


def test_no_anchor_means_null_avwap_columns_not_a_guess():
    history = _history()
    row = features.compute_daily_features(
        "AAPL", history, session_date=history[-1]["session_date"], computed_at=NOW
    )
    assert row["avwape_value"] is None and row["favorite_zone_coord"] is None
    assert row["first_dev_touch_order"] is None and row["second_band_streak"] is None


def test_favorite_zone_definitions():
    def bar(close, high=None, low=None):
        return {"open": close, "high": high if high is not None else close + 0.1, "low": low if low is not None else close - 0.1, "close": close, "volume": 1}

    bands = {"UPPER_1": 110.0, "UPPER_2": 120.0, "UPPER_3": 130.0}
    # Three consecutive closes inside [100, 110] ending at the snapshot.
    inside = [bar(95.0), bar(102.0), bar(105.0), bar(108.0)]
    zone = features.favorite_zone_block(inside, 100.0, bands)
    assert zone.residence_bars == 3
    assert zone.coord == pytest.approx((108.0 - 100.0) / 10.0)
    assert zone.first_dev_touch_order is None  # nothing reached UPPER_1

    # Two separate touch episodes of UPPER_1, then a rejection bar.
    touches = [bar(105.0, high=111.0), bar(104.0), bar(106.0, high=112.0, low=104.0)]
    zone = features.favorite_zone_block(touches, 100.0, bands)
    assert zone.first_dev_touch_order == 2
    assert zone.band1_rejection_strength == pytest.approx((112.0 - 106.0) / (112.0 - 104.0))

    power_hold = [bar(118.0), bar(121.0), bar(122.0)]
    assert features.favorite_zone_block(power_hold, 100.0, bands).second_band_streak == 2


def test_daily_snapshot_job_is_deterministic_and_idempotent(store):
    history = _history(days=30)
    store.publish("bar_d1", history, job_id="d1")
    session_date = history[-1]["session_date"]

    first = features.build_daily_snapshots(store, session_date, symbols=["AAPL"], now=NOW, run_id="run-1")
    assert first.rows == 1
    row = store.read_table("feature_snapshot_daily").to_pylist()[0]
    assert len(row["input_manifest_hash"]) == 64

    # Re-running against the same sealed inputs is a no-op...
    again = features.build_daily_snapshots(store, session_date, symbols=["AAPL"], now=NOW)
    assert again.status == "NOTHING_TO_COMPUTE"
    assert again.skipped == {"ALREADY_COMPUTED": 1}
    assert store.read_table("feature_snapshot_daily").num_rows == 1

    # ...and recomputing in a fresh lake from the same inputs is identical.
    second_store = ResearchStore.open(store.root.parent / "lake2")
    second_store.publish("bar_d1", history, job_id="d1")
    features.build_daily_snapshots(second_store, session_date, symbols=["AAPL"], now=NOW, run_id="run-1")
    twin = second_store.read_table("feature_snapshot_daily").to_pylist()[0]
    assert {key: value for key, value in twin.items() if key != "input_manifest_hash"} == {
        key: value for key, value in row.items() if key != "input_manifest_hash"
    }


def test_input_manifest_hash_tracks_the_input_file_set(store):
    history = _history(days=30)
    store.publish("bar_d1", history[:20], job_id="d1")
    before = features.input_manifest_hash(store, {("bar_d1", "year=2026")})
    store.publish("bar_d1", history[20:], job_id="d1")
    after = features.input_manifest_hash(store, {("bar_d1", "year=2026")})
    assert before != after  # a new sealed input changes the reproducibility key


def test_a_missing_anchor_bar_is_reported_not_guessed(store):
    history = _history(days=30)
    store.publish("bar_d1", history, job_id="d1")
    report = features.build_daily_snapshots(
        store,
        history[-1]["session_date"],
        symbols=["AAPL"],
        anchors_by_symbol={"AAPL": date(2020, 1, 2)},  # long before the history
        now=NOW,
    )
    assert report.skipped.get("ANCHOR_BAR_NOT_IN_HISTORY") == 1
    row = store.read_table("feature_snapshot_daily").to_pylist()[0]
    assert row["avwape_value"] is None  # computed without an anchor, not faked


# --- anchors ---------------------------------------------------------------
def test_anchor_instances_are_deterministic_and_bitemporal(store):
    anchors = [
        {"symbol": "aapl", "anchor_type": "EARNINGS_CURRENT", "anchor_bar_date": "2026-07-30"},
        {"symbol": "AAPL", "anchor_type": "EARNINGS_PREVIOUS", "anchor_bar_date": date(2026, 4, 30)},
        {"symbol": "", "anchor_type": "EARNINGS_CURRENT", "anchor_bar_date": None},
    ]
    report = features.build_anchor_instances(store, anchors, now=NOW, run_id="anchors")
    assert report.rows == 2 and report.skipped == {"INCOMPLETE_ANCHOR": 1}

    rows = {row["anchor_type"]: row for row in store.read_table("anchor_instance").to_pylist()}
    current = rows["EARNINGS_CURRENT"]
    assert current["anchor_instance_id"] == schemas.anchor_instance_id(
        "AAPL", "EARNINGS_CURRENT", date(2026, 7, 30), features.AVWAP_FORMULA_VERSION
    )
    assert current["formula_version"] == features.AVWAP_FORMULA_VERSION
    assert current["system_from"] == NOW and current["system_to"] is None
    assert current["valid_from"] == datetime(2026, 7, 30, tzinfo=UTC)

    again = features.build_anchor_instances(store, anchors, now=NOW)
    assert again.status == "NOTHING_TO_COMPUTE"
    assert store.read_table("anchor_instance").num_rows == 2


# --- intraday snapshots ----------------------------------------------------
def test_intraday_snapshot_has_exactly_the_frozen_columns_and_wraps_the_champion_vwap():
    session = xcal.trading_session(date(2026, 8, 3))
    bars = [_m5_row(session, index) for index in range(12)]
    boundary = bars[-1]["interval_start"]

    row = features.compute_intraday_features(
        "AAPL", bars, interval_start=boundary, session=session, computed_at=NOW
    )
    assert set(row) == set(schemas.FEATURE_SNAPSHOT_INTRADAY.names)
    assert row["vwap_algorithm"] == "STANDARD"
    vwap, upper, lower = features.session_vwap_bands(bars)
    assert row["session_vwap"] == pytest.approx(vwap, abs=1e-12)
    assert row["session_vwap_upper_1"] == pytest.approx(upper, abs=1e-12)
    assert row["session_vwap_lower_1"] == pytest.approx(lower, abs=1e-12)
    closes = [bar["close"] for bar in bars]
    assert row["ema8_m5"] == pytest.approx(features.ema_series(closes, 8), abs=1e-12)
    assert row["session_phase"] == "RTH"


def test_intraday_snapshot_never_sees_a_later_bar():
    session = xcal.trading_session(date(2026, 8, 3))
    bars = [_m5_row(session, index) for index in range(12)]
    boundary = bars[3]["interval_start"]

    early = features.compute_intraday_features(
        "AAPL", bars, interval_start=boundary, session=session, computed_at=NOW
    )
    truncated = features.compute_intraday_features(
        "AAPL", bars[:4], interval_start=boundary, session=session, computed_at=NOW
    )
    assert early == truncated


def test_production_context_is_stored_verbatim_never_recomputed():
    session = xcal.trading_session(date(2026, 8, 3))
    bars = [_m5_row(session, index) for index in range(6)]
    context = {
        "rvol_tc2000": 0.83,
        "rvol_gate_pass": False,  # sub-1.0 rows are kept as the denominator
        "rs_rw_vs_spy": -1.25,
        "group_rs_debiased": 0.4,
        "market_internals_negative": True,
        "session_structure_gate": "CHOP_VETO",
        "pullback_count_in_current_leg": 2,
    }
    row = features.compute_intraday_features(
        "AAPL", bars, interval_start=bars[-1]["interval_start"], session=session, context=context, computed_at=NOW
    )
    for key, value in context.items():
        assert row[key] == value
    # Without production evidence the columns are null, not a second opinion.
    bare = features.compute_intraday_features(
        "AAPL", bars, interval_start=bars[-1]["interval_start"], session=session, computed_at=NOW
    )
    assert bare["rvol_tc2000"] is None and bare["session_structure_gate"] is None


def test_m15_and_m30_emas_come_from_completed_derived_bars():
    session = xcal.trading_session(date(2026, 8, 3))
    bars = [_m5_row(session, index) for index in range(12)]
    derived = {
        "M15": [
            {
                "symbol": "AAPL",
                "interval_start": session.rth_open_at + timedelta(minutes=15 * index),
                "interval_end": session.rth_open_at + timedelta(minutes=15 * (index + 1)),
                "close": 300.0 + index,
            }
            for index in range(4)
        ]
    }
    boundary = session.rth_open_at + timedelta(minutes=45)
    row = features.compute_intraday_features(
        "AAPL", bars, interval_start=boundary, session=session, derived_by_timeframe=derived, computed_at=NOW
    )
    # Only the three M15 bars that had closed by 10:15 contribute - and three
    # bars cannot seed an 8-period EMA under the stated lookback rule (D5).
    assert row["ema8_m15"] is None
    assert row["ema8_m30"] is None  # no M30 bars supplied

    # With eight completed M15 bars the EMA publishes, on those bars alone.
    derived["M15"] = [
        {
            "symbol": "AAPL",
            "interval_start": session.rth_open_at + timedelta(minutes=15 * index),
            "interval_end": session.rth_open_at + timedelta(minutes=15 * (index + 1)),
            "close": 300.0 + index,
        }
        for index in range(8)
    ]
    later = features.compute_intraday_features(
        "AAPL",
        bars,
        interval_start=session.rth_open_at + timedelta(minutes=120),
        session=session,
        derived_by_timeframe=derived,
        computed_at=NOW,
    )
    assert later["ema8_m15"] == pytest.approx(
        features.ema_series([300.0 + index for index in range(8)], 8), abs=1e-12
    )


def test_intraday_job_is_idempotent(store):
    session = xcal.trading_session(date(2026, 8, 3))
    bars = [_m5_row(session, index) for index in range(6)]
    store.publish("bar_m5", bars, job_id="tee")

    first = features.build_intraday_snapshots(store, session.session_date, symbols=["AAPL"], now=NOW)
    assert first.rows == 6
    again = features.build_intraday_snapshots(store, session.session_date, symbols=["AAPL"], now=NOW)
    assert again.status == "NOTHING_TO_COMPUTE" and again.skipped["ALREADY_COMPUTED"] == 6
    assert store.read_table("feature_snapshot_intraday").num_rows == 6


# --- D5: the windowing rule ------------------------------------------------
def _two_years_to(session_date: date, symbol="AAPL"):
    """Completed D1 bars for the two calendar years ending at ``session_date``."""
    rows = []
    day = date(session_date.year - 1, 1, 1)
    index = 0
    while day <= session_date:
        if xcal.is_trading_day(day):
            rows.append(_d1_row(day, index, symbol=symbol))
            index += 1
        day += timedelta(days=1)
    return rows


def test_a_midyear_daily_snapshot_matches_the_champions_full_history_frame(store):
    """D5: sma100/sma200 and the D1 EMAs must be the champion's numbers.

    August is the worst case for the old year-partition window: the session's
    own year holds ~150 sessions, so sma200 was null and the EMAs were seeded
    on 1 January instead of on the real history.
    """
    session_date = date(2026, 8, 3)
    history = _two_years_to(session_date)
    assert len(history) > features.DAILY_HISTORY_MIN_SESSIONS
    store.publish("bar_d1", history, job_id="d1")

    report = features.build_daily_snapshots(store, session_date, symbols=["AAPL"], now=NOW)
    assert report.rows == 1
    row = store.read_table("feature_snapshot_daily").to_pylist()[0]

    # The champion's own frame over the same full history is the reference.
    grid = features.indicator_grid(history)
    for column, source in (
        ("sma50", "sma_50"),
        ("sma100", "sma_100"),
        ("sma200", "sma_200"),
        ("ema8", "ema_8"),
        ("ema15", "ema_15"),
        ("ema21", "ema_21"),
    ):
        expected = float(grid[source].iloc[-1])
        assert row[column] == pytest.approx(expected, abs=1e-9), column

    # The long SMAs are genuinely populated, not null-and-equal.
    assert row["sma200"] is not None and row["sma100"] is not None


def test_the_daily_window_always_reads_year_and_the_prior_year(store):
    """D5: the floor is year + year-1, whatever the month."""
    session_date = date(2026, 8, 3)
    store.publish("bar_d1", _two_years_to(session_date), job_id="d1")

    partitions, rows_by_symbol = features.daily_history_window(store, session_date)
    assert ("bar_d1", "year=2026") in partitions
    assert ("bar_d1", "year=2025") in partitions
    # Deep enough to stop walking, and no bar after the session date leaks in.
    assert len(rows_by_symbol["AAPL"]) >= features.DAILY_HISTORY_MIN_SESSIONS
    assert max(_as_day(row["session_date"]) for row in rows_by_symbol["AAPL"]) <= session_date
    assert len(partitions) <= features.DAILY_HISTORY_MAX_YEARS


def _as_day(value):
    return value.date() if isinstance(value, datetime) else value


def test_the_intraday_ema_lookback_is_the_session_and_needs_span_bars():
    """D5: the M5 EMA frame is the champion's ``today_df``, guarded by span.

    BounceBot fetches "5 D" for prev-day extremes and the dynamic/EOD VWAPs but
    computes ema_8/15/21 on today's bars only, and only once the session has at
    least ``span`` of them. The warehouse column must be that same number.
    """
    session = xcal.trading_session(date(2026, 8, 3))
    prior = xcal.trading_session(date(2026, 7, 31))
    bars = [_m5_row(session, index) for index in range(12)]
    # Prior-session bars are offered to the snapshot and must not be seen.
    intruders = [_m5_row(prior, index) for index in range(12)]

    boundary = bars[-1]["interval_start"]
    row = features.compute_intraday_features(
        "AAPL", intruders + bars, interval_start=boundary, session=session, computed_at=NOW
    )
    session_only = features.compute_intraday_features(
        "AAPL", bars, interval_start=boundary, session=session, computed_at=NOW
    )
    assert row == session_only

    closes = [bar["close"] for bar in bars]
    assert row["ema8_m5"] == pytest.approx(features.ema_series(closes, 8), abs=1e-12)
    # 12 completed bars: 8 seeds, 15 and 21 do not - exactly as the champion.
    assert row["ema15_m5"] is None and row["ema21_m5"] is None

    # Before the eighth bar even ema8 is null rather than a mostly-seed number.
    early = features.compute_intraday_features(
        "AAPL", bars, interval_start=bars[4]["interval_start"], session=session, computed_at=NOW
    )
    assert early["ema8_m5"] is None


def test_the_intraday_ema_matches_the_champions_own_computation():
    """The stored ema8_m5 equals BounceBot's ``today_df`` EMA, bar for bar."""
    import pandas as pd

    session = xcal.trading_session(date(2026, 8, 3))
    bars = [_m5_row(session, index) for index in range(30)]
    boundary = bars[-1]["interval_start"]
    row = features.compute_intraday_features(
        "AAPL", bars, interval_start=boundary, session=session, computed_at=NOW
    )
    today_df = pd.DataFrame([{"close": bar["close"]} for bar in bars])
    for column, span in (("ema8_m5", 8), ("ema15_m5", 15), ("ema21_m5", 21)):
        champion = today_df["close"].ewm(span=span, adjust=False).mean().iloc[-1]
        assert row[column] == pytest.approx(float(champion), abs=1e-12), column


def test_features_are_disabled_without_a_store():
    assert features.build_daily_snapshots(None, date(2026, 8, 3)).status == "DISABLED"
    assert features.build_intraday_snapshots(None, date(2026, 8, 3)).status == "DISABLED"
    assert features.build_anchor_instances(None, []).status == "DISABLED"
