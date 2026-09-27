"""The P10 history reader: one consistent series per symbol, completed bars only."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from scripts.research_warehouse import history_reader as hr
from scripts.research_warehouse import schemas
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc
OBSERVED = datetime(2026, 9, 27, 3, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def _d1(symbol, day, close, *, provider="YAHOO", revision="r1", supersedes="", observed=OBSERVED, dataset_row=True):
    return {
        "symbol": symbol,
        "session_id": f"XNYS-{day.isoformat()}",
        "session_date": day,
        "open": close - 1,
        "high": close + 1,
        "low": close - 2,
        "close": close,
        "volume": 1000,
        "adjustment_version": "yahoo_split_v1" if provider == "YAHOO" else None,
        "corporate_action_id": None,
        "provider": provider,
        "quality": "COMPLETE",
        "is_complete": True,
        "event_at": datetime.combine(day, datetime.min.time(), UTC) + timedelta(hours=20),
        "observed_at": observed,
        "capture_mode": "BACKFILL",
        "revision_id": revision,
        "supersedes_revision_id": supersedes,
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "t",
    }


def _h1(symbol, start, close=10.0, *, revision="r1", minutes=60):
    return {
        "symbol": symbol,
        "interval_start": start,
        "interval_end": start + timedelta(minutes=minutes),
        "session_id": f"XNYS-{start.date().isoformat()}",
        "session_phase": "RTH",
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 10,
        "vwap": None,
        "trade_count": None,
        "provider": "YAHOO",
        "is_complete": True,
        "quality": "COMPLETE",
        "source_hash": "x",
        "adjustment_version": "yahoo_split_v1",
        "event_at": start + timedelta(minutes=minutes),
        "observed_at": OBSERVED,
        "capture_mode": "BACKFILL",
        "revision_id": revision,
        "supersedes_revision_id": "",
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "t",
    }


def test_read_d1_prefers_yahoo_latest_revision_and_never_mixes(store):
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2025, 1, 2)]
    store.publish(
        "bar_d1_history",
        [_d1("AAPL", day, 100.0, revision="r1") for day in days]
        + [_d1("AAPL", day, 25.0, revision="r2", supersedes="r1") for day in days]
        + [_d1("AAPL", days[0], 99.0, provider="IBKR", revision="i1")],
    )
    # Legacy lake rows for the same symbol must be ignored once history exists.
    store.publish("bar_d1", [_d1("AAPL", days[0], 55.0, provider="UNKNOWN", revision="")])

    out = hr.read_d1(["aapl"], store=store)

    frame = out["AAPL"]
    assert list(frame.columns) == hr.D1_COLUMNS
    assert list(frame["session_date"]) == days
    assert set(frame["close"]) == {25.0}
    assert set(frame["provider"]) == {"YAHOO"}
    assert set(frame["revision_id"]) == {"r2"}
    assert set(frame["source_dataset"]) == {"bar_d1_history"}


def test_read_d1_falls_back_to_legacy_rows_and_says_so(store):
    store.publish("bar_d1", [_d1("MSFT", date(2026, 9, 1), 10.0, provider="UNKNOWN", revision="")])
    store.publish("bar_d1_history", [_d1("AAPL", date(2026, 9, 1), 20.0)])

    out = hr.read_d1(store=store)

    assert sorted(out) == ["AAPL", "MSFT"]
    assert set(out["MSFT"]["provider"]) == {"UNKNOWN"}
    assert set(out["MSFT"]["source_dataset"]) == {"bar_d1"}
    assert hr.available_symbols(store=store) == ["AAPL", "MSFT"]


def test_read_d1_date_window_is_inclusive(store):
    days = [date(2023, 12, 29), date(2024, 1, 2), date(2024, 1, 3)]
    store.publish("bar_d1_history", [_d1("SPY", day, 400.0) for day in days])
    frame = hr.read_d1("SPY", start=date(2024, 1, 2), end="2024-01-03", store=store)["SPY"]
    assert list(frame["session_date"]) == days[1:]


def test_read_intraday_is_completed_only_and_in_eastern_time(store):
    first = datetime(2026, 9, 24, 13, 30, tzinfo=UTC)  # 09:30 ET
    rows = [_h1("SPY", first + timedelta(hours=offset)) for offset in range(3)]
    store.publish("bar_h1", rows)

    now = first + timedelta(hours=2)  # the third bar closes at +3h: still forming
    frame = hr.read_intraday("H1", ["SPY"], store=store, now=now)["SPY"]

    assert len(frame) == 2
    assert str(frame["interval_start"].dt.tz) == "America/New_York"
    assert frame["interval_start"].iloc[0].hour == 9 and frame["interval_start"].iloc[0].minute == 30


def test_read_intraday_h4_uses_the_current_source_revision(store):
    start = datetime(2026, 9, 24, 13, 30, tzinfo=UTC)

    def _h4(close, revision, computed):
        return {
            "symbol": "SPY",
            "timeframe": "H4",
            "aggregation_contract_id": hr.H4_CONTRACT_ID,
            "interval_start": start,
            "interval_end": start + timedelta(hours=4),
            "session_id": "XNYS-2026-09-24",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1,
            "is_stub": False,
            "stub_duration_min": None,
            "constituent_count": 4,
            "constituent_expected": 4,
            "is_complete": True,
            "quality": "COMPLETE",
            "event_at": start + timedelta(hours=4),
            "computed_at": computed,
            "input_capture_mode_worst": "BACKFILL",
            "provider": "YAHOO",
            "source_revision_id": revision,
            "schema_version": schemas.SCHEMA_VERSION,
            "run_id": "t",
        }

    store.publish("bar_derived_history", [_h4(100.0, "r1", OBSERVED), _h4(25.0, "r2", OBSERVED + timedelta(days=1))])
    frame = hr.read_intraday("H4", store=store, now=OBSERVED + timedelta(days=2))["SPY"]
    assert list(frame["close"]) == [25.0]
    assert list(frame["revision_id"]) == ["r2"]


def test_read_intraday_rejects_unknown_timeframes(store):
    with pytest.raises(ValueError):
        hr.read_intraday("M5", store=store)


def test_read_earnings_dates_merges_sources_sorted(store):
    def _e(day, source):
        return {
            "symbol": "AAPL",
            "earnings_date": day,
            "time_of_day": "AMC",
            "earnings_at": None,
            "eps_estimate": None,
            "eps_reported": None,
            "surprise_pct": None,
            "source": source,
            "observed_at": OBSERVED,
            "capture_mode": "BACKFILL",
            "schema_version": schemas.SCHEMA_VERSION,
            "run_id": "t",
        }

    store.publish(
        "earnings_date",
        [_e(date(2025, 1, 30), "yahoo"), _e(date(2024, 2, 1), "yahoo"), _e(date(2025, 1, 30), "other")],
    )
    assert hr.read_earnings_dates(store=store) == {"AAPL": [date(2024, 2, 1), date(2025, 1, 30)]}
    assert hr.read_earnings_dates(["MSFT"], store=store) == {}


def test_read_earnings_events_one_row_per_date_prefers_a_row_with_eps(store):
    def _e(day, source, *, tod="AMC", est=None, rep=None, surprise=None, observed=OBSERVED):
        return {
            "symbol": "AAPL", "earnings_date": day, "time_of_day": tod, "earnings_at": None,
            "eps_estimate": est, "eps_reported": rep, "surprise_pct": surprise, "source": source,
            "observed_at": observed, "capture_mode": "BACKFILL",
            "schema_version": schemas.SCHEMA_VERSION, "run_id": "t",
        }

    store.publish("earnings_date", [
        _e(date(2025, 1, 30), "calendar"),
        _e(date(2025, 1, 30), "yahoo", est=2.0, rep=2.4, surprise=20.0),
        _e(date(2024, 2, 1), "yahoo", tod="BMO"),
    ])
    frame = hr.read_earnings_events(store=store)
    assert list(frame.columns) == ["symbol", "earnings_date", "time_of_day", "eps_estimate",
                                   "eps_reported", "surprise_pct", "source"]
    assert frame["earnings_date"].tolist() == [date(2024, 2, 1), date(2025, 1, 30)]
    row = frame.iloc[1]
    assert (row["source"], row["surprise_pct"], row["eps_reported"]) == ("yahoo", 20.0, 2.4)
    assert frame.iloc[0]["time_of_day"] == "BMO"
    assert hr.read_earnings_events(["MSFT"], store=store).empty


def test_empty_lake_reads_empty(store):
    assert hr.read_d1(store=store) == {}
    assert hr.read_intraday("M30", store=store) == {}
    assert hr.available_symbols(store=store) == []


def test_unconfigured_lake_raises(monkeypatch):
    monkeypatch.setattr(ResearchStore, "open", classmethod(lambda cls, root=None: None))
    with pytest.raises(hr.LakeNotConfigured):
        hr.read_d1()
