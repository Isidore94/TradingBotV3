"""Session table and deterministic aggregation (plan sec 5.4, Phase 4).

Exit criteria pinned here: the session-anchored M5->M15/M30/H1 rules, DST,
half days, stub-bar duration flags, and boundary parity with IB's native
``useRTH=1`` H1. Plus the two rules that make a derived bar trustworthy -
completed buckets only, and short buckets visible as PARTIAL rather than
averaged into a full bar.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from scripts.research_warehouse import aggregate, exchange_calendar as xcal
from scripts.research_warehouse.schemas import SCHEMA_VERSION
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def _m5(session, count=78, *, symbol="AAPL", skip=(), capture_mode="LIVE"):
    rows = []
    for index in range(count):
        start = session.rth_open_at + timedelta(minutes=5 * index)
        if index in skip:
            continue
        rows.append(
            {
                "symbol": symbol,
                "interval_start": start,
                "interval_end": start + timedelta(minutes=5),
                "session_id": session.session_id,
                "session_phase": "RTH",
                "open": 100.0 + index,
                "high": 100.5 + index,
                "low": 99.5 + index,
                "close": 100.25 + index,
                "volume": 1000 + index,
                "vwap": None,
                "trade_count": None,
                "provider": "IBKR",
                "is_complete": True,
                "quality": "COMPLETE",
                "source_hash": "",
                "event_at": start + timedelta(minutes=5),
                "observed_at": start + timedelta(minutes=6),
                "capture_mode": capture_mode,
                "revision_id": "",
                "supersedes_revision_id": "",
                "schema_version": SCHEMA_VERSION,
                "run_id": "tee",
            }
        )
    return rows


# --- the calendar ----------------------------------------------------------
def test_holidays_and_observance_rules():
    assert xcal.is_trading_day(date(2026, 8, 3)) is True
    assert xcal.is_trading_day(date(2026, 8, 1)) is False  # Saturday
    for day, name in sorted(xcal.holidays(2026).items()):
        assert day.weekday() < 5, name
    names = set(xcal.holidays(2026).values())
    assert "Good Friday" in names and "Juneteenth National Independence Day" in names
    assert xcal.easter_sunday(2026) == date(2026, 4, 5)
    assert date(2026, 4, 3) in xcal.holidays(2026)  # Good Friday
    assert date(2026, 11, 26) in xcal.holidays(2026)  # Thanksgiving

    # A Saturday holiday moves to the preceding Friday...
    assert date(2027, 12, 24) in xcal.holidays(2027)  # Christmas 2027 is a Saturday
    # ...except New Year's Day, which is never pulled back to 31 December.
    assert date(2021, 12, 31) not in xcal.holidays(2021)
    # A Sunday holiday moves to the following Monday.
    assert date(2027, 7, 5) in xcal.holidays(2027)  # 4 July 2027 is a Sunday
    # Juneteenth only from 2022.
    assert not any("Juneteenth" in name for name in xcal.holidays(2021).values())


def test_half_days_close_early_and_shrink_the_expected_bar_counts():
    black_friday = date(2026, 11, 27)
    assert xcal.is_half_day(black_friday) is True
    session = xcal.trading_session(black_friday)
    assert session.is_half_day is True
    assert session.rth_minutes == 210  # 09:30-13:00
    assert session.expected_m5_bars_rth == 42 and session.expected_m1_bars_rth == 210
    # No post-market on an early-close day.
    assert session.eth_close_at == session.rth_close_at

    full = xcal.trading_session(date(2026, 8, 3))
    assert full.expected_m5_bars_rth == 78 and full.expected_m1_bars_rth == 390
    assert xcal.is_half_day(date(2026, 12, 24)) is True  # Thursday
    # 4 July 2025 is a Friday, so 3 July closes at 13:00...
    assert xcal.is_half_day(date(2025, 7, 3)) is True
    # ...while in 2026 the 4th is a Saturday, so 3 July is the observed holiday
    # and closed outright - an early close and a closure are different facts.
    assert date(2026, 7, 3) in xcal.holidays(2026)
    assert xcal.is_half_day(date(2026, 7, 3)) is False
    assert xcal.trading_session(date(2026, 7, 3)) is None


def test_dst_is_handled_by_the_zone_not_an_offset():
    summer = xcal.trading_session(date(2026, 8, 3))
    winter = xcal.trading_session(date(2026, 12, 1))
    assert summer.rth_open_at == datetime(2026, 8, 3, 13, 30, tzinfo=UTC)  # EDT
    assert winter.rth_open_at == datetime(2026, 12, 1, 14, 30, tzinfo=UTC)  # EST
    # The session is 6.5 hours either way; only its UTC offset moves.
    assert summer.rth_minutes == winter.rth_minutes == 390

    # The US DST switch is a Sunday, so no session straddles the change.
    switch_friday = xcal.trading_session(date(2026, 3, 6))
    switch_monday = xcal.trading_session(date(2026, 3, 9))
    assert switch_friday.rth_open_at.hour == 14 and switch_monday.rth_open_at.hour == 13


def test_trading_session_rows_carry_the_calendar_version(store):
    report = aggregate.build_trading_sessions(store, date(2026, 11, 23), date(2026, 11, 29))
    assert report.rows == 4  # Mon-Wed + the half-day Friday; Thanksgiving closed
    assert report.half_days == 1 and report.holidays_skipped == 3  # Thu + Sat + Sun

    rows = {row["session_id"]: row for row in store.read_table("trading_session").to_pylist()}
    assert "XNYS-2026-11-26" not in rows  # Thanksgiving
    friday = rows["XNYS-2026-11-27"]
    assert friday["is_half_day"] is True and friday["expected_m5_bars_rth"] == 42
    assert friday["calendar_version"] == xcal.CALENDAR_VERSION
    assert friday["exchange_calendar"] == "XNYS"

    again = aggregate.build_trading_sessions(store, date(2026, 11, 23), date(2026, 11, 29))
    assert again.status == "ALREADY_PUBLISHED" and store.read_table("trading_session").num_rows == 4


# --- session-anchored aggregation -----------------------------------------
def test_full_session_bucket_counts_match_the_contract():
    session = xcal.trading_session(date(2026, 8, 3))
    assert len(list(aggregate.session_buckets(session, "M15"))) == 26
    assert len(list(aggregate.session_buckets(session, "M30"))) == 13
    hourly = list(aggregate.session_buckets(session, "H1"))
    assert len(hourly) == 7  # six full hours plus the 15:30-16:00 stub
    assert [bucket[3] for bucket in hourly] == [False] * 6 + [True]
    assert hourly[-1][2] == 6  # the stub expects six M5 constituents


def test_derived_bars_are_session_anchored_with_correct_ohlcv():
    session = xcal.trading_session(date(2026, 8, 3))
    rows = aggregate.derive_session_bars(
        _m5(session), session, "M15", as_of=session.rth_close_at + timedelta(hours=1)
    )
    assert len(rows) == 26
    first = rows[0]
    assert first["interval_start"] == session.rth_open_at
    assert first["interval_end"] == session.rth_open_at + timedelta(minutes=15)
    assert first["open"] == 100.0 and first["close"] == 100.25 + 2
    assert first["high"] == 100.5 + 2 and first["low"] == 99.5
    assert first["volume"] == 1000 + 1001 + 1002
    assert first["constituent_count"] == 3 and first["constituent_expected"] == 3
    assert first["is_complete"] is True and first["quality"] == "COMPLETE"
    assert first["is_stub"] is False and first["stub_duration_min"] is None
    assert first["aggregation_contract_id"] == "xnys_rth_m15_v1"
    assert first["input_capture_mode_worst"] == "LIVE"


def test_the_h1_stub_keeps_its_true_duration():
    session = xcal.trading_session(date(2026, 8, 3))
    rows = aggregate.derive_session_bars(
        _m5(session), session, "H1", as_of=session.rth_close_at + timedelta(hours=1)
    )
    assert len(rows) == 7
    stub = rows[-1]
    assert stub["is_stub"] is True and stub["stub_duration_min"] == 30
    assert stub["interval_start"] == session.rth_close_at - timedelta(minutes=30)
    assert stub["constituent_expected"] == 6 and stub["constituent_count"] == 6
    # A complete stub is COMPLETE for its own contract; the flag is what stops
    # it being compared with a full hour as equivalent.
    assert stub["is_complete"] is True
    assert [row["is_stub"] for row in rows[:-1]] == [False] * 6


def test_derived_h1_boundaries_match_ib_native_use_rth_bars():
    """The sentinel parity check: derived boundaries vs native useRTH=1 H1."""
    for day in (date(2026, 8, 3), date(2026, 12, 1), date(2026, 11, 27)):
        session = xcal.trading_session(day)
        derived = aggregate.derive_session_bars(
            _m5(session, count=session.expected_m5_bars_rth),
            session,
            "H1",
            as_of=session.rth_close_at + timedelta(hours=1),
        )
        native = aggregate.native_h1_boundaries(session)
        assert [(row["interval_start"], row["interval_end"]) for row in derived] == native


def test_half_day_aggregation_uses_the_half_day_contract_variant():
    session = xcal.trading_session(date(2026, 11, 27))
    rows = aggregate.derive_session_bars(
        _m5(session, count=42), session, "H1", as_of=session.rth_close_at + timedelta(hours=1)
    )
    assert len(rows) == 4  # 3 full hours + a 30-minute stub
    assert rows[-1]["is_stub"] is True and rows[-1]["stub_duration_min"] == 30
    assert {row["aggregation_contract_id"] for row in rows} == {"xnys_rth_half_h1_v1"}

    m30 = aggregate.derive_session_bars(
        _m5(session, count=42), session, "M30", as_of=session.rth_close_at + timedelta(hours=1)
    )
    assert len(m30) == 7 and not any(row["is_stub"] for row in m30)


def test_missing_constituents_are_partial_never_averaged_away():
    session = xcal.trading_session(date(2026, 8, 3))
    rows = aggregate.derive_session_bars(
        _m5(session, skip={1}), session, "M15", as_of=session.rth_close_at + timedelta(hours=1)
    )
    first = rows[0]
    assert first["constituent_count"] == 2 and first["constituent_expected"] == 3
    assert first["is_complete"] is False and first["quality"] == "PARTIAL"
    # A bucket with no constituents at all is absent, not a zero-volume bar.
    empty = aggregate.derive_session_bars(
        _m5(session, skip=set(range(3))), session, "M15", as_of=session.rth_close_at + timedelta(hours=1)
    )
    assert empty[0]["interval_start"] == session.rth_open_at + timedelta(minutes=15)


def test_forming_buckets_are_never_derived():
    session = xcal.trading_session(date(2026, 8, 3))
    midday = session.rth_open_at + timedelta(minutes=47)
    rows = aggregate.derive_session_bars(_m5(session), session, "M15", as_of=midday)
    # 09:30, 09:45, 10:00 have closed; the 10:15 bucket has not.
    assert len(rows) == 3
    assert rows[-1]["interval_end"] == session.rth_open_at + timedelta(minutes=45)


def test_forming_and_eth_m5_bars_are_excluded_from_derivation():
    session = xcal.trading_session(date(2026, 8, 3))
    rows = _m5(session, count=3)
    rows[1]["is_complete"] = False  # a forming M5 bar is preview
    premarket = dict(rows[0])
    premarket["interval_start"] = session.rth_open_at - timedelta(minutes=30)
    derived = aggregate.derive_session_bars(
        rows + [premarket], session, "M15", as_of=session.rth_close_at + timedelta(hours=1)
    )
    assert derived[0]["constituent_count"] == 2  # v1 aggregates are RTH-only


def test_worst_capture_mode_wins():
    session = xcal.trading_session(date(2026, 8, 3))
    rows = _m5(session, count=3)
    rows[2]["capture_mode"] = "BACKFILL"
    derived = aggregate.derive_session_bars(
        rows, session, "M15", as_of=session.rth_close_at + timedelta(hours=1)
    )
    assert derived[0]["input_capture_mode_worst"] == "BACKFILL"


def test_unsupported_timeframes_are_refused():
    session = xcal.trading_session(date(2026, 8, 3))
    with pytest.raises(ValueError, match="unsupported derived timeframe"):
        list(aggregate.session_buckets(session, "H3"))


def test_h2_is_supported_now_that_a_consumer_exists():
    """H2 was CUT by the locked plan (sec 5.2) for having no consumer. The
    Phase 0.12 B3 higher-timeframe LRSI recipe grid is that consumer, which is
    the cut's own reopen condition (BD-78).

    RTH is 6.5h, so two-hour buckets do not divide it: three full buckets and
    a 30-minute STUB. The stub keeps its true duration and is flagged, exactly
    as the H1 stub is - a research lane that averaged it into a full H2 would
    be measuring a bar that never existed."""
    session = xcal.trading_session(date(2026, 8, 3))
    buckets = list(aggregate.session_buckets(session, "H2"))
    # UTC, as every bucket boundary in this module is: 13:30Z is 09:30 ET.
    assert [bucket[0].strftime("%H:%M") for bucket in buckets] == [
        "13:30",
        "15:30",
        "17:30",
        "19:30",
    ]
    assert [bucket[3] for bucket in buckets] == [False, False, False, True]
    assert [bucket[2] for bucket in buckets] == [24, 24, 24, 6]


# --- the build job ---------------------------------------------------------
def test_build_derived_bars_partitions_by_timeframe_and_is_idempotent(store):
    session = xcal.trading_session(date(2026, 8, 3))
    store.publish("bar_m5", _m5(session) + _m5(session, symbol="MSFT"), job_id="tee")

    report = aggregate.build_derived_bars(
        store, [session.session_date], as_of=session.rth_close_at + timedelta(hours=1)
    )
    assert report.by_timeframe == {"M15": 52, "M30": 26, "H1": 14}
    assert report.stubs == 2  # one H1 stub per symbol
    assert store.read_table("bar_derived", "timeframe=M15/month=2026-08").num_rows == 52
    assert store.read_table("bar_derived", "timeframe=H1/month=2026-08").num_rows == 14

    again = aggregate.build_derived_bars(
        store, [session.session_date], as_of=session.rth_close_at + timedelta(hours=1)
    )
    assert again.status == "NOTHING_TO_DERIVE" and store.read_table("bar_derived").num_rows == 92


def test_build_derived_bars_skips_closed_days(store):
    report = aggregate.build_derived_bars(store, [date(2026, 11, 26), date(2026, 8, 1)])
    assert report.sessions == 0 and report.status == "NOTHING_TO_DERIVE"


# --- W1 from canonical D1 --------------------------------------------------
def _d1(day: date, symbol="AAPL", index=0):
    return {
        "symbol": symbol,
        "session_id": xcal.session_id_for(day),
        "session_date": day,
        "open": 100.0 + index,
        "high": 105.0 + index,
        "low": 95.0 + index,
        "close": 101.0 + index,
        "volume": 1_000_000 + index,
        "adjustment_version": None,
        "corporate_action_id": None,
        "provider": "IBKR",
        "quality": "COMPLETE",
        "is_complete": True,
        "event_at": datetime(day.year, day.month, day.day, tzinfo=UTC),
        "observed_at": datetime(day.year, day.month, day.day, 21, tzinfo=UTC),
        "capture_mode": "BACKFILL",
        "revision_id": "",
        "supersedes_revision_id": "",
        "schema_version": SCHEMA_VERSION,
        "run_id": "d1",
    }


def test_weekly_bars_derive_from_canonical_d1(store):
    week = [date(2026, 8, 3) + timedelta(days=offset) for offset in range(5)]
    store.publish("bar_d1", [_d1(day, index=index) for index, day in enumerate(week)], job_id="d1")

    friday_close = xcal.trading_session(week[-1]).rth_close_at
    report = aggregate.build_weekly_bars(store, [week[0]], as_of=friday_close + timedelta(hours=1))
    assert report.by_timeframe == {"W1": 1}

    bar = store.read_table("bar_derived", "timeframe=W1/month=2026-08").to_pylist()[0]
    assert bar["aggregation_contract_id"] == aggregate.W1_CONTRACT_ID
    assert bar["interval_start"] == xcal.trading_session(week[0]).rth_open_at
    assert bar["interval_end"] == friday_close
    assert bar["session_id"] == "XNYS-2026-08-07"  # completes at the final close
    assert bar["open"] == 100.0 and bar["close"] == 101.0 + 4
    assert bar["high"] == 105.0 + 4 and bar["low"] == 95.0
    assert bar["constituent_count"] == 5 and bar["constituent_expected"] == 5
    assert bar["is_stub"] is False and bar["is_complete"] is True


def test_a_forming_week_is_never_published(store):
    week = [date(2026, 8, 3) + timedelta(days=offset) for offset in range(3)]
    store.publish("bar_d1", [_d1(day) for day in week], job_id="d1")
    midweek = xcal.trading_session(week[-1]).rth_close_at

    report = aggregate.build_weekly_bars(store, [week[0]], as_of=midweek)
    assert report.skipped_forming == 1 and report.status == "NOTHING_TO_DERIVE"
    assert store.read_table("bar_derived").num_rows == 0


def test_a_short_week_is_flagged(store):
    # Thanksgiving week 2026: Mon-Wed, closed Thursday, half-day Friday.
    week = [date(2026, 11, 23), date(2026, 11, 24), date(2026, 11, 25), date(2026, 11, 27)]
    store.publish("bar_d1", [_d1(day, index=index) for index, day in enumerate(week)], job_id="d1")
    close = xcal.trading_session(date(2026, 11, 27)).rth_close_at

    aggregate.build_weekly_bars(store, [week[0]], as_of=close + timedelta(hours=1))
    bar = store.read_table("bar_derived", "timeframe=W1/month=2026-11").to_pylist()[0]
    assert bar["is_stub"] is True  # four sessions, not five
    assert bar["constituent_count"] == 4 and bar["constituent_expected"] == 4
    assert bar["is_complete"] is True  # complete for the week that existed


def test_a_missing_session_makes_the_week_partial(store):
    week = [date(2026, 8, 3), date(2026, 8, 4), date(2026, 8, 6), date(2026, 8, 7)]
    store.publish("bar_d1", [_d1(day) for day in week], job_id="d1")
    close = xcal.trading_session(date(2026, 8, 7)).rth_close_at

    aggregate.build_weekly_bars(store, [week[0]], as_of=close + timedelta(hours=1))
    bar = store.read_table("bar_derived", "timeframe=W1/month=2026-08").to_pylist()[0]
    assert bar["constituent_count"] == 4 and bar["constituent_expected"] == 5
    assert bar["is_complete"] is False and bar["quality"] == "PARTIAL"


def test_aggregation_is_disabled_without_a_store():
    assert aggregate.build_trading_sessions(None, date(2026, 8, 3), date(2026, 8, 3)).status == "DISABLED"
    assert aggregate.build_derived_bars(None, [date(2026, 8, 3)]).status == "DISABLED"
    assert aggregate.build_weekly_bars(None, [date(2026, 8, 3)]).status == "DISABLED"
