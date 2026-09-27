"""P10 IB intraday history: M30 from IB, H1/H4 derived, offline with a fake fetcher."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from scripts.research_warehouse import cli
from scripts.research_warehouse import exchange_calendar as xcal
from scripts.research_warehouse import history as hist
from scripts.research_warehouse import history_ib as hib
from scripts.research_warehouse import history_reader as hr
from scripts.research_warehouse import pacer as pacer_mod
from scripts.research_warehouse import schemas
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc
SATURDAY = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
WEEK = date(2026, 9, 21)  # Monday; the week ends Friday 2026-09-25


@pytest.fixture
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def _quiet(*_args, **_kwargs):
    return None


def session_bars(days, *, close=100.0, skip=()):
    """Parsed IB M30 bars (volume in lots) for every RTH slot of each session."""
    out = []
    for day in days:
        session = xcal.trading_session(day)
        if session is None:
            continue
        moment = session.rth_open_at
        while moment < session.rth_close_at:
            if moment not in skip:
                out.append({"interval_start": moment, "open": close, "high": close + 1, "low": close - 1,
                            "close": close, "volume": 10, "vwap": close, "trade_count": 5})
            moment += timedelta(minutes=30)
    return out


def _days(first, last):
    return [s.session_date for s in xcal.sessions_between(first, last)]


class FakeFetcher:
    def __init__(self, bars=None, script=None):
        self.bars = bars or {}
        self.script = script or {}
        self.calls = []

    def fetch(self, symbol, *, end, duration):
        self.calls.append((symbol, end, duration))
        queue = self.script.get(symbol)
        if queue:
            status = queue.pop(0)
            if status != hib.OK:
                return hib.IbFetch(status=status, message=status)
        low = end - timedelta(days=365)  # IB "1 Y" = 365 days
        found = [b for b in self.bars.get(symbol, []) if low < b["interval_start"].astimezone(xcal.EXCHANGE_TZ).date() <= end]
        if not found:
            return hib.IbFetch(status=hib.NO_DATA, message="HMDS query returned no data")
        return hib.IbFetch(bars=found)


class Clock:
    def __init__(self, start):
        self.now = start
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)


def _run(store, symbols, fetcher, *, start=WEEK, now=SATURDAY, clock=None, **kwargs):
    clock = clock or Clock(now)
    return hib.run_ib_backfill(
        store, symbols, fetcher=fetcher, now=now, start=start, clock=clock, sleep=clock.sleep, log=_quiet, **kwargs
    )


def _yahoo_h1(symbol, start, close):
    end = start + timedelta(hours=1)
    return {
        "symbol": symbol, "interval_start": start, "interval_end": end, "session_id": xcal.session_for(start).session_id,
        "session_phase": "RTH", "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 10,
        "vwap": None, "trade_count": None, "provider": "YAHOO", "is_complete": True, "quality": "COMPLETE",
        "source_hash": "x", "adjustment_version": "yahoo_split_v1", "event_at": end, "observed_at": SATURDAY,
        "capture_mode": "BACKFILL", "revision_id": "y1", "supersedes_revision_id": "", "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "t",
    }


# --- pure pieces -------------------------------------------------------------
def test_windows_are_fixed_years_from_the_start_and_end_at_the_last_session():
    windows = hib.ib_windows(date(2021, 9, 27), date(2026, 9, 25))
    assert windows[0] == (date(2021, 9, 27), date(2022, 9, 26))
    # IB's "1 Y" is 365 days: the window over 29 Feb 2024 ends a day earlier
    # (2023-09-27 was lost when windows were calendar years - live run 2026-09-27).
    assert windows[2] == (date(2023, 9, 27), date(2024, 9, 25))
    assert windows[-1] == (date(2025, 9, 26), date(2026, 9, 25))
    assert len(windows) == 5
    assert all((last - first).days + 1 <= 365 for first, last in windows)
    assert hib.duration_for(*windows[0]) == "1 Y"
    assert hib.duration_for(date(2026, 9, 26), date(2026, 10, 2)) == "1 M"
    assert hib.ib_windows(date(2021, 9, 27), date(2026, 10, 2))[-1] == (date(2026, 9, 26), date(2026, 10, 2))


def test_request_is_m30_trades_rth_epoch_and_maps_share_classes():
    request = hib.ib_request("BRK.B", date(2022, 9, 26), "1 Y")
    assert request["symbol"] == "BRK B"
    assert request["barSizeSetting"] == "30 mins" and request["whatToShow"] == "TRADES"
    assert request["useRTH"] == 1 and request["formatDate"] == 2
    assert request["endDateTime"] == "20220926 16:00:00 US/Eastern"


def test_market_guard_is_0915_to_1615_et_on_trading_days_only():
    monday_1000_et = datetime(2026, 9, 28, 14, 0, tzinfo=UTC)
    lift = hib.market_guard_until(monday_1000_et)
    assert lift == datetime(2026, 9, 28, 20, 15, tzinfo=UTC)  # 16:15 ET = 13:15 PT
    assert hib.market_guard_until(datetime(2026, 9, 28, 13, 14, tzinfo=UTC)) is None  # 09:14 ET
    assert hib.market_guard_until(datetime(2026, 9, 28, 13, 15, tzinfo=UTC)) is not None  # 09:15 ET
    assert hib.market_guard_until(datetime(2026, 9, 28, 20, 15, tzinfo=UTC)) is None  # 16:15 ET
    assert hib.market_guard_until(SATURDAY) is None


def test_classify_separates_no_data_no_contract_pacing_and_timeouts():
    pacer = pacer_mod.IbPacer()
    assert hib.classify([], (162, "Historical Market Data Service error message:HMDS query returned no data"), pacer).status == hib.NO_DATA
    assert pacer.snapshot().backoff_until == ""  # no data is an answer, not pacing
    assert hib.classify([], (200, "No security definition has been found"), pacer).status == hib.NO_CONTRACT
    assert hib.classify([], (-1, "no historicalDataEnd within 180s"), pacer).status == hib.RETRY
    assert hib.classify([], (162, "Historical data request pacing violation"), pacer).status == hib.RETRY
    assert pacer.snapshot().backoff_until != ""
    assert hib.classify([], (321, "Error validating request"), pacer).status == hib.ERROR


def test_ib_420_backs_capture_off():
    pacer = pacer_mod.IbPacer()
    assert pacer.note_error(420, "Invalid Real-time Query", capture=True) is True


def test_fetcher_uses_client_1011_and_rebuilds_a_dead_connection():
    built = []

    class Transport:
        def __init__(self, spec):
            self.spec = spec
            self.up = True
            built.append(self)

        def is_connected(self):
            return self.up

        def disconnect(self):
            self.up = False

        def request_historical(self, *, timeout, **request):
            assert request["useRTH"] == 1
            return [{"date": "1695821400", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 3}], None

    fetcher = hib.IbHistoryFetcher(Transport, pacer=pacer_mod.IbPacer(capture_allowance=100), sleep=_quiet)
    assert fetcher.spec.client_id == pacer_mod.CLIENT_ID_NIGHTLY_BACKFILL == 1011
    assert fetcher.fetch("SPY", end=date(2023, 9, 27), duration="1 M").status == hib.OK
    built[0].up = False  # the nightly TWS restart
    assert fetcher.fetch("SPY", end=date(2023, 9, 28), duration="1 M").status == hib.OK
    assert len(built) == 2 and built[1].spec.client_id == 1011
    assert hib.IbHistoryFetcher(Transport).pacer.capture_allowance == hib.CAPTURE_PER_WINDOW == 45


# --- the job -----------------------------------------------------------------
def test_backfill_stores_rth_m30_quarantines_the_rest_and_is_idempotent(store):
    days = _days(WEEK, date(2026, 9, 25))
    monday = xcal.trading_session(WEEK)
    bars = session_bars(days)
    bars[3] = {**bars[3], "high": 1.0}  # OHLC out of order
    bars.append({**bars[0], "interval_start": monday.rth_open_at - timedelta(hours=1)})  # premarket
    bars.append({**bars[0], "interval_start": monday.rth_open_at + timedelta(minutes=45)})  # off the :00/:30 grid
    bars.append({**bars[0], "interval_start": datetime(2026, 9, 20, 14, 0, tzinfo=UTC)})  # a Sunday
    bars.append({**bars[0], "interval_start": datetime(2026, 9, 26, 14, 0, tzinfo=UTC)})  # after the window
    fetcher = FakeFetcher({"SPY": bars})

    report = _run(store, ["SPY"], fetcher, start=date(2026, 9, 19))

    assert len(fetcher.calls) == 1 and fetcher.calls[0][2] == "1 M"
    assert report.rows_published["bar_m30"] == 5 * 13 - 1
    assert report.rows_quarantined["bar_m30"] == 4
    ledger = hist.HistoryLedger(store.root, hib.LEDGER_NAME).latest()
    assert ledger["SPY|2026-09-19"]["status"] == hib.DONE
    m30 = hr.read_intraday("M30", ["SPY"], store=store, now=SATURDAY)["SPY"]
    assert len(m30) == 5 * 13 - 1
    assert set(m30["provider"]) == {"IBKR"}
    assert m30["volume"].iloc[0] == 1000  # IB lots of 100 -> shares

    again = _run(store, ["SPY"], fetcher, start=date(2026, 9, 19))
    assert len(fetcher.calls) == 1  # the sealed window is never requested again
    assert again.rows_published.get("bar_m30", 0) == 0
    assert again.rows_published.get("bar_derived_history", 0) == 0


def test_a_window_sealed_under_the_old_window_size_is_asked_again_and_fills_its_gap(store):
    import json

    days = _days(WEEK, date(2026, 9, 25))
    fetcher = FakeFetcher({"SPY": [b for b in session_bars(days) if b["interval_start"].astimezone(xcal.EXCHANGE_TZ).date() != WEEK]})
    _run(store, ["SPY"], fetcher)
    path = hist.HistoryLedger(store.root, hib.LEDGER_NAME).path
    old_style = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in old_style:
        record.pop("window_days", None)  # written before 365-day windows (live 2026-09-27)
    path.write_text("".join(json.dumps(r) + "\n" for r in old_style), encoding="utf-8")

    fetcher.bars["SPY"] = session_bars(days)
    report = _run(store, ["SPY"], fetcher)
    assert len(fetcher.calls) == 2
    assert report.rows_published["bar_m30"] == 13  # only the missing Monday
    assert len(hr.read_intraday("M30", ["SPY"], store=store, now=SATURDAY)["SPY"]) == 65


def test_h1_and_h4_are_derived_from_m30_and_the_reader_serves_one_basis(store):
    days = _days(WEEK, date(2026, 9, 25))
    fetcher = FakeFetcher({"SPY": session_bars(days)})
    _run(store, ["SPY"], fetcher)

    h1 = hr.read_intraday("H1", ["SPY"], store=store, now=SATURDAY)["SPY"]
    assert len(h1) == 5 * 7
    first_day = h1[h1["interval_start"].dt.date == WEEK]
    assert [t.strftime("%H:%M") for t in first_day["interval_start"]] == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    assert list(first_day["constituent_count"]) == [2, 2, 2, 2, 2, 2, 1]
    assert bool(first_day["is_stub"].iloc[-1]) is True
    assert first_day["volume"].iloc[0] == 2000
    assert set(h1["provider"]) == {"IBKR"}

    h4 = hr.read_intraday("H4", ["SPY"], store=store, now=SATURDAY)["SPY"]
    assert len(h4) == 10 and set(h4["provider"]) == {"IBKR"}
    assert list(h4["constituent_count"])[:2] == [4, 3]


def test_half_day_h1_is_clipped_at_the_1300_close(store):
    half = date(2025, 11, 28)  # the day after Thanksgiving
    assert xcal.trading_session(half).is_half_day
    fetcher = FakeFetcher({"SPY": session_bars([half])})
    now = datetime(2025, 11, 29, 12, tzinfo=UTC)
    _run(store, ["SPY"], fetcher, start=half, now=now)
    h1 = hr.read_intraday("H1", ["SPY"], store=store, now=now)["SPY"]
    assert [t.strftime("%H:%M") for t in h1["interval_start"]] == ["09:30", "10:30", "11:30", "12:30"]
    assert h1["interval_end"].iloc[-1].strftime("%H:%M") == "13:00"
    h4 = hr.read_intraday("H4", ["SPY"], store=store, now=now)["SPY"]
    assert len(h4) == 1 and h4["constituent_count"].iloc[0] == 4


def test_reader_prefers_the_longer_intraday_basis_and_never_mixes(store):
    days = _days(WEEK, date(2026, 9, 25))
    # Yahoo holds a longer H1 history (a week earlier too); IB only this week.
    earlier = xcal.trading_session(date(2026, 9, 14)).rth_open_at
    friday_last = xcal.trading_session(date(2026, 9, 25)).rth_close_at - timedelta(minutes=30)
    store.publish("bar_h1", [_yahoo_h1("SPY", earlier, 50.0), _yahoo_h1("SPY", friday_last, 50.0)])
    _run(store, ["SPY"], FakeFetcher({"SPY": session_bars(days)}))
    assert hr.intraday_basis(["SPY"], store=store) == {"SPY": "YAHOO"}
    h1 = hr.read_intraday("H1", ["SPY"], store=store, now=SATURDAY)["SPY"]
    assert set(h1["provider"]) == {"YAHOO"} and len(h1) == 2
    m30 = hr.read_intraday("M30", ["SPY"], store=store, now=SATURDAY)
    assert "SPY" not in m30 or set(m30["SPY"]["provider"]) == {"YAHOO"}

    # Once IB reaches further back, every timeframe switches to IBKR.
    older = _days(date(2026, 9, 7), date(2026, 9, 18))
    _run(store, ["SPY"], FakeFetcher({"SPY": session_bars(older)}), start=date(2026, 9, 7), run_id="second")
    assert hr.intraday_basis(["SPY"], store=store) == {"SPY": "IBKR"}
    for frame in ("M30", "H1", "H4"):
        series = hr.read_intraday(frame, ["SPY"], store=store, now=SATURDAY)["SPY"]
        assert set(series["provider"]) == {"IBKR"}, frame


def test_ib_and_yahoo_h1_disagreeing_over_half_a_percent_is_flagged_not_dropped(store):
    days = _days(WEEK, date(2026, 9, 25))
    yahoo = []
    for day in days:
        session = xcal.trading_session(day)
        for begin, _end in hib.h1_buckets(session):
            off = 1.01 if (day == date(2026, 9, 23) and begin.astimezone(xcal.EXCHANGE_TZ).hour == 11) else 1.004
            yahoo.append(_yahoo_h1("SPY", begin, 100.0 * off))
    store.publish("bar_h1", yahoo)
    _run(store, ["SPY"], FakeFetcher({"SPY": session_bars(days)}))

    flags = hr.read_quality_flags(store=store)
    mismatch = flags[flags["check"] == hib.FLAG_IB_YAHOO_MISMATCH]
    assert list(mismatch["flag_date"]) == [date(2026, 9, 23)]
    assert "-0.99%" in mismatch["detail"].iloc[0]
    derived = hr._scan(store, "bar_derived_history", symbols=["SPY"]).to_pandas()
    assert (derived["aggregation_contract_id"] == hr.H1_FROM_M30_CONTRACT_ID).sum() == 5 * 7  # nothing dropped


def test_missing_and_partial_sessions_are_flagged_once_the_pull_is_whole(store):
    days = _days(WEEK, date(2026, 9, 25))
    wednesday = xcal.trading_session(date(2026, 9, 23))
    thursday = xcal.trading_session(date(2026, 9, 24))
    bars = [b for b in session_bars(days, skip={thursday.rth_open_at}) if b["interval_start"].astimezone(xcal.EXCHANGE_TZ).date() != wednesday.session_date]
    _run(store, ["SPY"], FakeFetcher({"SPY": bars}))
    flags = hr.read_quality_flags("bar_m30", store=store)
    assert {(row.check, row.flag_date) for row in flags.itertuples()} == {
        (hist.FLAG_MISSING_SESSION, date(2026, 9, 23)),
        (hib.FLAG_PARTIAL_SESSION, date(2026, 9, 24)),
    }


def test_a_day_spy_also_lacks_is_flagged_on_spy_only(store):
    days = [d for d in _days(date(2025, 1, 6), date(2025, 1, 10)) if d != date(2025, 1, 8)]
    now = datetime(2025, 1, 11, 12, tzinfo=UTC)
    fetcher = FakeFetcher({"SPY": session_bars(days), "AAPL": session_bars(days)})
    _run(store, ["SPY", "AAPL"], fetcher, start=date(2025, 1, 6), now=now)
    flags = hr.read_quality_flags("bar_m30", store=store)
    assert list(zip(flags["symbol"], flags["flag_date"], strict=True)) == [("SPY", date(2025, 1, 8))]


def test_2025_01_09_is_a_closure_never_flagged_and_an_old_flag_is_skipped_on_read(store):
    assert xcal.trading_session(date(2025, 1, 9)) is None  # national day of mourning (Carter)
    assert xcal.trading_session(date(2025, 1, 8)) is not None
    days = _days(date(2025, 1, 6), date(2025, 1, 10))
    assert date(2025, 1, 9) not in days
    now = datetime(2025, 1, 11, 12, tzinfo=UTC)
    _run(store, ["SPY"], FakeFetcher({"SPY": session_bars(days)}), start=date(2025, 1, 6), now=now)
    assert hr.read_quality_flags("bar_m30", store=store).empty
    # A flag written before the calendar knew stays stored but is not served.
    old = hist.flag_row("bar_m30", "SPY", hist.FLAG_MISSING_SESSION, date(2025, 1, 9), "no IBKR M30 bar for an exchange session", detected_at=now, run_id="old")
    real = hist.flag_row("bar_m30", "SPY", hist.FLAG_MISSING_SESSION, date(2025, 1, 8), "no IBKR M30 bar for an exchange session", detected_at=now, run_id="old")
    store.publish("history_quality_flag", [old, real])
    flags = hr.read_quality_flags(store=store)
    assert list(flags["flag_date"]) == [date(2025, 1, 8)]
    assert hr._scan(store, "history_quality_flag").num_rows == 2


def test_windows_before_the_listing_are_inferred_empty_without_a_request(store):
    start = date(2024, 9, 23)
    now = SATURDAY
    listed = _days(date(2025, 12, 1), date(2026, 9, 25))
    fetcher = FakeFetcher({"NEWCO": session_bars(listed)})
    _run(store, ["NEWCO"], fetcher, start=start, now=now)
    assert len(fetcher.calls) == 2  # the newest window and the listing window; the oldest is inferred
    ledger = hist.HistoryLedger(store.root, hib.LEDGER_NAME).latest()
    assert ledger["NEWCO|2024-09-23"]["status"] == hib.EMPTY_INFERRED
    assert ledger["NEWCO|2025-09-23"]["status"] == hib.DONE


def test_never_requests_in_market_hours_and_resumes_after_the_close(store):
    monday_1000_et = datetime(2026, 9, 28, 14, 0, tzinfo=UTC)
    clock = Clock(monday_1000_et)
    seen = []

    class Watch(FakeFetcher):
        def fetch(self, symbol, *, end, duration):
            seen.append(clock())
            return super().fetch(symbol, end=end, duration=duration)

    days = _days(WEEK, date(2026, 9, 25))
    _run(store, ["SPY"], Watch({"SPY": session_bars(days)}), now=monday_1000_et, clock=clock)
    assert seen and all(hib.market_guard_until(moment) is None for moment in seen)
    assert seen[0] >= datetime(2026, 9, 28, 20, 15, tzinfo=UTC)


def test_max_hours_stops_sealed_and_the_next_run_resumes(store):
    days = _days(WEEK, date(2026, 9, 25))
    monday_1000_et = datetime(2026, 9, 28, 14, 0, tzinfo=UTC)
    bars = {"SPY": session_bars(days), "QQQ": session_bars(days)}
    first = FakeFetcher(bars)
    report = _run(store, ["SPY", "QQQ"], first, now=monday_1000_et, clock=Clock(monday_1000_et), max_hours=1)
    assert report.status == "PARTIAL" and first.calls == []  # market hours outlast the budget

    second = FakeFetcher(bars)
    report = _run(store, ["SPY", "QQQ"], second, batch_symbols=1)
    assert report.status == "OK" and [c[0] for c in second.calls] == ["SPY", "QQQ"]


def test_retryable_errors_are_retried_and_a_lost_contract_is_recorded(store):
    days = _days(WEEK, date(2026, 9, 25))
    fetcher = FakeFetcher({"SPY": session_bars(days)}, script={"SPY": [hib.RETRY, hib.RETRY], "ZZZZ": [hib.NO_CONTRACT]})
    clock = Clock(SATURDAY)
    report = _run(store, ["SPY", "ZZZZ"], fetcher, clock=clock)
    assert report.rows_published["bar_m30"] == 65
    assert clock.slept[:2] == [hib.RETRY_SLEEP_SECONDS, 2 * hib.RETRY_SLEEP_SECONDS]
    flags = hr.read_quality_flags("bar_m30", store=store)
    assert list(flags[flags["check"] == hist.FLAG_NO_DATA]["symbol"]) == ["ZZZZ"]
    calls = len(fetcher.calls)
    _run(store, ["ZZZZ"], fetcher)
    assert len(fetcher.calls) == calls  # a lost contract is not re-asked for a week


def test_a_split_after_the_first_pull_starts_a_new_revision(store):
    days = _days(WEEK, date(2026, 9, 25))
    _run(store, ["AAPL"], FakeFetcher({"AAPL": session_bars(days, close=200.0)}), run_id="first")
    store.publish("corporate_action", [{
        "symbol": "AAPL", "action_type": "SPLIT", "ex_date": date(2026, 9, 28), "value": 2.0,
        "corporate_action_id": "YAHOO:AAPL:SPLIT:2026-09-28", "provider": "YAHOO",
        "event_at": datetime(2026, 9, 28, 13, 30, tzinfo=UTC), "observed_at": datetime(2026, 9, 29, tzinfo=UTC),
        "capture_mode": "BACKFILL", "revision_id": "", "supersedes_revision_id": "",
        "schema_version": schemas.SCHEMA_VERSION, "run_id": "t",
    }])
    later = datetime(2026, 10, 3, 12, tzinfo=UTC)
    new_days = _days(WEEK, date(2026, 10, 2))
    fetcher = FakeFetcher({"AAPL": session_bars(new_days, close=100.0)})
    report = _run(store, ["AAPL"], fetcher, now=later, run_id="second")
    assert report.repulled == ["AAPL"]
    m30 = hr.read_intraday("M30", ["AAPL"], store=store, now=later)["AAPL"]
    assert set(m30["close"]) == {100.0} and len(m30) == 10 * 13
    assert set(m30["revision_id"]) == {"IBKR:AAPL:second"}
    h1 = hr.read_intraday("H1", ["AAPL"], store=store, now=later)["AAPL"]
    assert set(h1["close"]) == {100.0}


def test_priority_names_first_then_dollar_volume(store, monkeypatch):
    monkeypatch.setattr(hist, "sector_etfs", lambda: ("XLK",))
    rows = []
    for symbol, price, volume in (("AAA", 10.0, 1000), ("BBB", 100.0, 1000), ("CCC", 1.0, 10)):
        for day in _days(date(2026, 9, 1), date(2026, 9, 25)):
            rows.append({
                "symbol": symbol, "session_id": f"XNYS-{day}", "session_date": day, "open": price, "high": price,
                "low": price, "close": price, "volume": volume, "adjustment_version": "yahoo_split_v1",
                "corporate_action_id": None, "provider": "YAHOO", "quality": "COMPLETE", "is_complete": True,
                "event_at": datetime.combine(day, datetime.min.time(), UTC), "observed_at": SATURDAY,
                "capture_mode": "BACKFILL", "revision_id": "r", "supersedes_revision_id": "",
                "schema_version": schemas.SCHEMA_VERSION, "run_id": "t",
            })
    store.publish("bar_d1_history", rows)
    order = hib.order_symbols(store, ["CCC", "ZZZ", "AAA", "XLK", "^VIX", "BBB", "SPY"], through=date(2026, 9, 25))
    assert order == ["SPY", "XLK", "BBB", "AAA", "CCC", "ZZZ"]


def test_cli_dry_run_counts_owed_windows_and_coverage_reports_ib(store):
    report = cli.run_history_ib_backfill(store, symbols="SPY,QQQ", dry_run=True, now=SATURDAY)
    assert report["dry_run"] is True and report["requests_owed"] == 10 and report["windows_per_symbol"] == 5
    days = _days(WEEK, date(2026, 9, 25))
    _run(store, ["SPY"], FakeFetcher({"SPY": session_bars(days)}))
    coverage = hist.coverage_report(store)
    assert coverage["ib_intraday"]["symbols"] == 1 and coverage["ib_intraday"]["bars"] == 65
    assert "IB intraday" in cli.format_history_coverage(coverage)


def test_2018_12_05_bush_mourning_day_is_a_closure():
    assert xcal.trading_session(date(2018, 12, 5)) is None  # national day of mourning (G. H. W. Bush)
    assert xcal.trading_session(date(2018, 12, 4)) is not None
    assert xcal.trading_session(date(2018, 12, 6)) is not None


# --- market-hours guard at send time (review 2026-09-27) -----------------------
MONDAY_0914_ET = datetime(2026, 9, 28, 13, 14, tzinfo=UTC)


class GuardTransport:
    """A fake TWS that records the (fake) time each request is sent."""

    def __init__(self, clock, *, slow_connect_checks=0):
        self.clock = clock
        self.sent = []
        self.checks = slow_connect_checks

    def is_connected(self):
        if self.checks > 0:
            self.checks -= 1
            return False
        return True

    def disconnect(self):
        pass

    def request_historical(self, *, timeout, **request):
        self.sent.append(self.clock())
        raw = []
        for bar in session_bars(_days(WEEK, date(2026, 9, 25))):
            item = dict(bar, date=str(int(bar["interval_start"].timestamp())))
            item.pop("interval_start")
            raw.append(item)
        return raw, None


def _guarded_fetcher(clock, transport, pacer):
    return hib.IbHistoryFetcher(lambda spec: transport, pacer=pacer, sleep=clock.sleep, clock=clock)


def test_a_pacer_backoff_that_runs_into_0915_et_never_sends():
    clock = Clock(MONDAY_0914_ET)
    pacer = pacer_mod.IbPacer(clock=clock)
    pacer.note_error(420, "pacing", capture=True)
    pacer.note_error(420, "pacing", capture=True)  # backoff until 09:16 ET
    transport = GuardTransport(clock)
    result = _guarded_fetcher(clock, transport, pacer).fetch("SPY", end=date(2026, 9, 25), duration="1 M")
    assert transport.sent == []
    assert result.status in (hib.MARKET_HOURS, hib.RETRY)
    assert clock.now <= datetime(2026, 9, 28, 13, 15, tzinfo=UTC)  # never waited into the guard


def test_a_reconnect_that_runs_into_0915_et_never_sends():
    clock = Clock(MONDAY_0914_ET + timedelta(seconds=59))
    transport = GuardTransport(clock, slow_connect_checks=10)  # ~2 s to connect
    fetcher = _guarded_fetcher(clock, transport, pacer_mod.IbPacer(clock=clock))
    result = fetcher.fetch("SPY", end=date(2026, 9, 25), duration="1 M")
    assert result.status == hib.MARKET_HOURS and transport.sent == []


def test_the_job_with_the_real_pacer_waits_out_market_hours(store):
    clock = Clock(MONDAY_0914_ET)
    pacer = pacer_mod.IbPacer(clock=clock)
    pacer.note_error(420, "pacing", capture=True)
    pacer.note_error(420, "pacing", capture=True)
    transport = GuardTransport(clock)
    fetcher = _guarded_fetcher(clock, transport, pacer)
    report = hib.run_ib_backfill(
        store, ["SPY"], fetcher=fetcher, now=MONDAY_0914_ET, start=WEEK, clock=clock, sleep=clock.sleep, log=_quiet
    )
    assert transport.sent and all(hib.market_guard_until(moment) is None for moment in transport.sent)
    assert transport.sent[0] >= datetime(2026, 9, 28, 20, 15, tzinfo=UTC)
    assert report.rows_published["bar_m30"] == 65


# --- the nightly IB top-up and the fresh-basis rule -------------------------
def _older_ib(store, symbols=("SPY",)):
    older = _days(date(2026, 9, 1), date(2026, 9, 18))
    _run(store, list(symbols), FakeFetcher({s: session_bars(older) for s in symbols}), start=date(2026, 9, 1),
         now=datetime(2026, 9, 19, 12, tzinfo=UTC))
    return older


def test_ib_is_the_basis_only_while_as_fresh_as_yahoo(store):
    older = _older_ib(store)
    fresh = xcal.trading_session(date(2026, 9, 25))
    store.publish("bar_h1", [_yahoo_h1("SPY", begin, 100.0) for begin, _end in hib.h1_buckets(fresh)])
    assert hr.intraday_basis(["SPY"], store=store) == {"SPY": "YAHOO"}  # IB is longer but a week stale
    assert set(hr.read_intraday("H1", ["SPY"], store=store, now=SATURDAY)["SPY"]["provider"]) == {"YAHOO"}

    fetcher = FakeFetcher({"SPY": session_bars(older + _days(WEEK, date(2026, 9, 25)))})
    report = hib.run_ib_topup(store, fetcher=fetcher, now=SATURDAY, clock=Clock(SATURDAY), log=_quiet)
    assert report.rows_published["bar_m30"] == 65 and report.by_outcome["TOPPED_UP"] == 1
    assert fetcher.calls == [("SPY", date(2026, 9, 25), "1 M")]
    assert hr.intraday_basis(["SPY"], store=store) == {"SPY": "IBKR"}
    h1 = hr.read_intraday("H1", ["SPY"], store=store, now=SATURDAY)["SPY"]
    assert set(h1["provider"]) == {"IBKR"} and h1["interval_start"].iloc[-1].date() == date(2026, 9, 25)
    again = hib.run_ib_topup(store, fetcher=FakeFetcher({}), now=SATURDAY, clock=Clock(SATURDAY), log=_quiet)
    assert again.by_outcome == {"FRESH": 1}


def test_ib_topup_skips_quietly_when_tws_is_down_or_in_market_hours(store):
    _older_ib(store, ("SPY", "QQQ"))
    down = FakeFetcher(script={"SPY": [hib.RETRY], "QQQ": [hib.RETRY]})
    report = hib.run_ib_topup(store, fetcher=down, now=SATURDAY, clock=Clock(SATURDAY), log=_quiet)
    assert report.status == "SKIPPED" and len(down.calls) == 1  # stops at the first unreachable answer
    busy = FakeFetcher({})
    moment = MONDAY_0914_ET + timedelta(minutes=5)
    report = hib.run_ib_topup(store, fetcher=busy, now=moment, clock=Clock(moment), log=_quiet)
    assert report.status == "SKIPPED" and busy.calls == []


def test_night_slot_runs_ib_after_yahoo_and_never_fails_for_it(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from ai_jobs import lake_history_topup
    from research_warehouse import cli as top_cli

    order = []
    monkeypatch.setattr(top_cli, "run_history_topup", lambda lake, **k: order.append("yahoo") or {"status": "OK"})

    def ib(lake, *, fetcher=None, log=None):
        order.append("ib")
        return {"status": "SKIPPED", "notes": ["RETRY: TWS not reachable"], "rows_published": {}}

    monkeypatch.setattr(top_cli, "run_history_ib_topup", ib)
    result = lake_history_topup.run_lake_history_topup(store=object(), ib_fetcher=object())
    assert order == ["yahoo", "ib"] and result["status"] == "ok"
    assert "IB M30 top-up skipped (RETRY: TWS not reachable)" in result["reason"]

    def broken(lake, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(top_cli, "run_history_ib_topup", broken)
    assert lake_history_topup.run_lake_history_topup(store=object(), ib_fetcher=object())["status"] == "ok"
    order.clear()
    lake_history_topup.run_lake_history_topup(store=object(), client=object())
    assert order == ["yahoo"]  # an injected Yahoo client (tests) never reaches real IB
