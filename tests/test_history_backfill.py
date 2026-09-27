"""P10 history backfill/top-up/quality: offline, with a fake provider."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.research_warehouse import history as hist
from scripts.research_warehouse import history_reader as hr
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc
ET = "America/New_York"
NAN = math.nan


@pytest.fixture
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


class FakeClient:
    def __init__(self):
        self.bars: dict = {}  # interval -> {symbol: frame}
        self.earnings: dict = {}
        self.calls: list = []

    def fetch_bars(self, symbols, *, interval, start=None, period=None):
        self.calls.append((interval, tuple(symbols), start, period))
        return {s: f for s, f in self.bars.get(interval, {}).items() if s in symbols}

    def fetch_earnings(self, symbol, *, limit=hist.EARNINGS_LIMIT):
        self.calls.append(("earnings", symbol))
        return self.earnings.get(symbol)


def _daily(rows):
    """rows: (iso date, o, h, l, c, v[, dividend, split])."""
    index, data = [], []
    for row in rows:
        day, o, h, l_, c, v, *rest = row
        dividend, split = (rest + [0.0, 0.0])[:2]
        index.append(pd.Timestamp(day))
        data.append({"Open": o, "High": h, "Low": l_, "Close": c, "Adj Close": c, "Volume": v,
                     "Dividends": dividend, "Stock Splits": split})
    return pd.DataFrame(data, index=pd.DatetimeIndex(index, name="Date"))


def _intraday(stamps, close=10.0):
    index = pd.DatetimeIndex([pd.Timestamp(s, tz=ET) for s in stamps], name="Datetime")
    data = [{"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Adj Close": close,
             "Volume": 100, "Dividends": 0.0, "Stock Splits": 0.0} for _ in stamps]
    return pd.DataFrame(data, index=index)


def _week(close=100.0):
    return [
        ("2026-09-18", NAN, NAN, NAN, NAN, NAN),  # alignment padding: never stored
        ("2026-09-19", 1, 2, 0.5, 1, 10),  # a Saturday: quarantined
        ("2026-09-21", close, close + 1, close - 1, close, 1000),
        ("2026-09-22", close, close - 5, close - 1, close, 1000),  # high < low: quarantined
        ("2026-09-23", close, close + 1, close - 1, close, 1000, 0.25),
        ("2026-09-24", close, close + 1, close - 1, close, 1000),
        ("2026-09-25", close, close + 1, close - 1, close, 1000),
    ]


SATURDAY = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
DEC1 = datetime(2026, 12, 1, 12, tzinfo=UTC)


def test_d1_backfill_stores_clean_rows_quarantines_dirty_and_is_idempotent(store):
    client = FakeClient()
    client.bars["1d"] = {"AAPL": _daily(_week())}

    report = hist.run_d1(store, ["AAPL"], client=client, now=SATURDAY, log=lambda *_: None)

    assert report.rows_published["bar_d1_history"] == 4
    assert report.rows_quarantined["bar_d1_history"] == 2
    assert report.rows_published["corporate_action"] == 1
    reasons = {tuple(e.extra.get("reasons", [])) for e in store.manifest.quarantine_entries()}
    assert reasons == {("VALIDATOR_REJECTED",)}
    frame = hr.read_d1(["AAPL"], store=store)["AAPL"]
    assert [d.isoformat() for d in frame["session_date"]] == ["2026-09-21", "2026-09-23", "2026-09-24", "2026-09-25"]
    assert set(frame["provider"]) == {"YAHOO"} and set(frame["adjustment_version"]) == {"yahoo_split_v1"}

    again = hist.run_d1(store, ["AAPL"], client=client, now=SATURDAY, mode="topup", log=lambda *_: None)
    assert again.rows_published.get("bar_d1_history", 0) == 0
    assert again.rows_quarantined.get("bar_d1_history", 0) == 0
    assert again.rows_published.get("corporate_action", 0) == 0


def test_d1_never_stores_a_forming_session(store):
    client = FakeClient()
    client.bars["1d"] = {"AAPL": _daily(_week())}
    # 16:30 ET Friday: the Friday bar is not an hour past the close yet.
    hist.run_d1(store, ["AAPL"], client=client, now=datetime(2026, 9, 25, 20, 30, tzinfo=UTC), log=lambda *_: None)
    frame = hr.read_d1(["AAPL"], store=store)["AAPL"]
    assert frame["session_date"].iloc[-1] == date(2026, 9, 24)


def test_topup_appends_to_the_same_revision(store):
    client = FakeClient()
    client.bars["1d"] = {"AAPL": _daily(_week())}
    hist.run_d1(store, ["AAPL"], client=client, now=SATURDAY, log=lambda *_: None)
    client.bars["1d"] = {"AAPL": _daily(_week()[2:] + [("2026-09-28", 100, 101, 99, 100.5, 900)])}

    report = hist.run_d1(store, ["AAPL"], client=client, now=datetime(2026, 9, 29, 12, tzinfo=UTC), mode="topup", log=lambda *_: None)

    assert report.rows_published["bar_d1_history"] == 1
    frame = hr.read_d1(["AAPL"], store=store)["AAPL"]
    assert len(frame) == 5 and frame["revision_id"].nunique() == 1


def test_a_new_split_repulls_the_whole_history_as_a_superseding_revision(store):
    client = FakeClient()
    client.bars["1d"] = {"AAPL": _daily(_week())}
    hist.run_d1(store, ["AAPL"], client=client, now=SATURDAY, log=lambda *_: None)
    old_revision = hr.read_d1(["AAPL"], store=store)["AAPL"]["revision_id"].iloc[0]

    halved = [row if math.isnan(row[4]) or row[0] == "2026-09-19" else (row[0], row[1] / 2, row[2] / 2, row[3] / 2, row[4] / 2, row[5] * 2, *row[6:]) for row in _week()]
    client.bars["1d"] = {"AAPL": _daily(halved + [("2026-09-28", 50, 51, 49, 50, 2000, 0.0, 2.0)])}
    report = hist.run_d1(store, ["AAPL"], client=client, now=datetime(2026, 9, 29, 12, tzinfo=UTC), mode="topup", log=lambda *_: None)

    assert report.repulled == ["AAPL"]
    frame = hr.read_d1(["AAPL"], store=store)["AAPL"]
    assert frame["revision_id"].nunique() == 1 and frame["revision_id"].iloc[0] != old_revision
    assert frame["close"].iloc[0] == 50.0
    raw = store.read_table("bar_d1_history").to_pandas()
    assert set(raw.loc[raw["revision_id"] == frame["revision_id"].iloc[0], "supersedes_revision_id"]) == {old_revision}
    assert (raw["revision_id"] == old_revision).sum() == 4  # never edited in place
    splits = hr.read_corporate_actions(["AAPL"], store=store)
    assert list(splits.loc[splits["action_type"] == "SPLIT", "value"]) == [2.0]


def test_no_data_symbol_is_flagged_not_guessed(store):
    client = FakeClient()
    hist.run_d1(store, ["GONE"], client=client, now=SATURDAY, log=lambda *_: None)
    flags = hr.read_quality_flags(store=store)
    assert list(flags["check"]) == ["PROVIDER_NO_DATA"]
    # Within the retry window it is not asked again.
    client.calls.clear()
    hist.run_d1(store, ["GONE"], client=client, now=SATURDAY + timedelta(days=1), log=lambda *_: None)
    assert client.calls == []


def test_h1_backfill_derives_h4_and_handles_the_half_day(store):
    half = [f"2026-11-27 {h}:30" for h in (9, 10, 11, 12)]
    full = [f"2026-11-30 {h}:30" for h in range(9, 16)]
    client = FakeClient()
    client.bars["1h"] = {"SPY": _intraday(["2026-11-30 08:30"] + half + full)}

    report = hist.run_intraday(store, ["SPY"], "H1", client=client, now=datetime(2026, 12, 1, 12, tzinfo=UTC), log=lambda *_: None)

    assert report.rows_published["bar_h1"] == 11
    assert report.rows_quarantined["bar_h1"] == 1  # the 08:30 pre-market bar
    h1 = hr.read_intraday("H1", ["SPY"], store=store, now=DEC1)["SPY"]
    assert h1["interval_end"].iloc[3].strftime("%H:%M") == "13:00"  # clipped at the early close
    h4 = hr.read_intraday("H4", ["SPY"], store=store, now=DEC1)["SPY"]
    assert [t.strftime("%m-%d %H:%M") for t in h4["interval_start"]] == ["11-27 09:30", "11-30 09:30", "11-30 13:30"]
    assert list(h4["constituent_count"]) == [4, 4, 3]
    assert list(h4["is_stub"]) == [True, False, True]

    again = hist.run_intraday(store, ["SPY"], "H1", client=client, now=datetime(2026, 12, 1, 12, tzinfo=UTC), mode="topup", log=lambda *_: None)
    assert again.rows_published.get("bar_h1", 0) == 0
    assert again.rows_published.get("bar_derived_history", 0) == 0
    assert again.rows_quarantined.get("bar_h1", 0) == 0


def test_an_intraday_split_carries_older_bars_into_the_new_revision(store):
    old_day = [f"2026-10-01 {h}:30" for h in range(9, 16)]
    recent = [f"2026-11-30 {h}:30" for h in range(9, 16)]
    client = FakeClient()
    client.bars["1h"] = {"SPY": pd.concat([_intraday(old_day, 100.0), _intraday(recent, 100.0)])}
    hist.run_intraday(store, ["SPY"], "H1", client=client, now=DEC1, log=lambda *_: None)

    # The provider's window has moved past October and the basis halved (2:1 split).
    client.bars["1h"] = {"SPY": _intraday(recent, 50.0)}
    report = hist.run_intraday(store, ["SPY"], "H1", client=client, now=DEC1, mode="topup", log=lambda *_: None)

    assert report.repulled == ["SPY"]
    frame = hr.read_intraday("H1", ["SPY"], store=store, now=DEC1)["SPY"]
    assert len(frame) == 14 and frame["revision_id"].nunique() == 1
    assert set(frame["close"]) == {50.0}  # October re-based, not dropped
    raw = store.read_table("bar_h1").to_pandas()
    carried = raw[raw["capture_mode"] == "RECONSTRUCTED"]
    assert len(carried) == 7 and set(carried["adjustment_version"]) == {"yahoo_split_v1+carried"}
    assert set(carried["volume"]) == {200}
    assert (raw["close"] == 100.0).sum() == 14  # the old revision is untouched
    h4 = hr.read_intraday("H4", ["SPY"], store=store, now=DEC1)["SPY"]
    assert len(h4) == 4 and set(h4["close"]) == {50.0}


def test_a_lagging_series_is_caught_up_into_its_revision_not_duplicated(store):
    july = [f"2026-07-17 {h}:30" for h in range(9, 16)]
    client = FakeClient()
    client.bars["1h"] = {"EQR": _intraday(july)}  # the provider cut the series short
    hist.run_intraday(store, ["EQR"], "H1", client=client, now=DEC1, log=lambda *_: None)

    again = hist.run_intraday(store, ["EQR"], "H1", client=client, now=DEC1, mode="topup", log=lambda *_: None)
    assert again.rows_published.get("bar_h1", 0) == 0
    assert store.read_table("bar_h1").to_pandas()["revision_id"].nunique() == 1

    client.bars["1h"] = {"EQR": _intraday(july + [f"2026-11-30 {h}:30" for h in range(9, 16)])}
    caught = hist.run_intraday(store, ["EQR"], "H1", client=client, now=DEC1, mode="topup", log=lambda *_: None)
    assert caught.rows_published["bar_h1"] == 7
    raw = store.read_table("bar_h1").to_pandas()
    assert raw["revision_id"].nunique() == 1 and len(raw) == 14


def test_m30_goes_to_its_own_dataset(store):
    client = FakeClient()
    client.bars["30m"] = {"QQQ": _intraday(["2026-11-30 09:30", "2026-11-30 10:00"])}
    hist.run_intraday(store, ["QQQ"], "M30", client=client, now=datetime(2026, 12, 1, 12, tzinfo=UTC), log=lambda *_: None)
    frame = hr.read_intraday("M30", store=store, now=DEC1)["QQQ"]
    assert len(frame) == 2 and (frame["interval_end"] - frame["interval_start"]).iloc[0] == pd.Timedelta(minutes=30)
    assert store.read_table("bar_h1").num_rows == 0


def test_earnings_dates_timing_gaps_etf_skip_and_resume(store):
    client = FakeClient()
    client.earnings["AAPL"] = pd.DataFrame(
        {"EPS Estimate": [1.9, 1.5, 1.0, 0.5], "Reported EPS": [2.0, NAN, 1.1, 0.4], "Surprise(%)": [5.0, NAN, 10.0, -20.0]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-07-30 16:00", tz=ET), pd.Timestamp("2026-04-30 07:00", tz=ET),
             pd.Timestamp("2026-01-29 00:00", tz=ET), pd.Timestamp("2017-01-31 16:00", tz=ET)],
            name="Earnings Date",
        ),
    )
    report = hist.run_earnings(store, ["AAPL", "MSFT", "SPY", "^VIX"], client=client, now=SATURDAY, log=lambda *_: None)

    assert ("earnings", "SPY") not in client.calls and ("earnings", "^VIX") not in client.calls
    assert report.rows_published["earnings_date"] == 3  # 2017 is before the window
    table = store.read_table("earnings_date").to_pandas().sort_values("earnings_date")
    assert list(table["time_of_day"]) == ["UNKNOWN", "BMO", "AMC"]
    flags = hr.read_quality_flags("earnings_date", store=store)
    assert list(flags["symbol"]) == ["MSFT"] and list(flags["check"]) == ["EARNINGS_NOT_AVAILABLE"]

    client.calls.clear()
    hist.run_earnings(store, ["AAPL", "MSFT"], client=client, now=SATURDAY + timedelta(days=2), log=lambda *_: None)
    assert client.calls == []  # refreshed within a week: resumed, not re-asked


def test_a_date_listed_twice_by_the_provider_is_stored_once(store):
    client = FakeClient()
    client.earnings["AAPL"] = pd.DataFrame(
        {"EPS Estimate": [1.0, 1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-30 16:00", tz=ET), pd.Timestamp("2026-07-30 16:05", tz=ET)]),
    )
    hist.run_earnings(store, ["AAPL"], client=client, now=SATURDAY, log=lambda *_: None)
    assert store.read_table("earnings_date").num_rows == 1


def test_a_crash_before_the_seal_leaves_earnings_owed_not_done(store):
    class Crashing(FakeClient):
        def fetch_earnings(self, symbol, *, limit=hist.EARNINGS_LIMIT):
            if symbol == "CCC":
                raise RuntimeError("power loss")
            return super().fetch_earnings(symbol, limit=limit)

    client = Crashing()
    for name in ("AAA", "BBB"):
        client.earnings[name] = pd.DataFrame(
            {"EPS Estimate": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-07-30 16:00", tz=ET)])
        )
    with pytest.raises(RuntimeError):
        hist.run_earnings(store, ["AAA", "BBB", "CCC"], client=client, now=SATURDAY, log=lambda *_: None)
    assert store.read_table("earnings_date").num_rows == 0

    retry = FakeClient()
    retry.earnings = client.earnings
    hist.run_earnings(store, ["AAA", "BBB"], client=retry, now=SATURDAY, log=lambda *_: None)
    assert ("earnings", "AAA") in retry.calls  # still owed after the crash
    assert store.read_table("earnings_date").num_rows == 2


def test_quality_flags_missing_stale_and_unexplained_jump_once(store):
    client = FakeClient()
    spy = [(d, 400, 410, 399, 400 + i, 10) for i, d in enumerate(["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25"])]
    bad = [
        ("2026-09-21", 10, 11, 9, 10, 100),
        ("2026-09-22", 10, 11, 9, 10, 100),  # identical to the day before
        ("2026-09-24", 16, 17, 15, 16, 100),  # 09-23 missing, +60% with no split
        ("2026-09-25", 16, 17, 15, 16.5, 100),
    ]
    client.bars["1d"] = {"SPY": _daily(spy), "ZZZ": _daily(bad)}
    hist.run_d1(store, ["SPY", "ZZZ"], client=client, now=SATURDAY, log=lambda *_: None)

    report = hist.run_quality(store, now=SATURDAY)

    assert report.by_outcome == {"MISSING_SESSION": 1, "STALE_REPEAT_BAR": 1, "UNEXPLAINED_JUMP": 1}
    assert hist.run_quality(store, now=SATURDAY + timedelta(days=1)).by_outcome == {}
    coverage = hist.coverage_report(store)
    assert coverage["per_symbol"]["ZZZ"] == {
        "first": "2026-09-21", "last": "2026-09-25", "sessions": 4, "source": "bar_d1_history",
        "provider": "YAHOO", "missing_sessions": 1, "stale": 1, "jumps": 1,
    }
    assert coverage["datasets"]["bar_d1_history"]["rows"] == 9
    assert "survivorship" in coverage["survivorship"]


def test_universe_puts_the_regime_inputs_first(tmp_path, store):
    universe = tmp_path / "universe_all.txt"
    universe.write_text("AAPL\nMSFT # note\n", encoding="utf-8")
    industry = tmp_path / "industry.json"
    industry.write_text('{"yahoo_industryKey_to_ref": {"a": {"etf": "SMH"}}}', encoding="utf-8")
    names = hist.history_universe(store, universe_file=universe, industry_path=industry)
    assert names[:5] == ["SPY", "QQQ", "IWM", "DIA", "^VIX"]
    assert {"XLK", "XLE", "SMH", "TLT", "HYG", "USO", "GLD", "AAPL", "MSFT"} <= set(names)
    assert len(names) == len(set(names))


def test_row_problem_reasons():
    good = {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0, "session_id": "XNYS-2026-09-25"}
    assert hist.row_problem(good) is None
    assert hist.row_problem({**good, "close": NAN}) == hist.BAD_PRICE
    assert hist.row_problem({**good, "low": -1}) == hist.BAD_PRICE
    assert hist.row_problem({**good, "high": 1.2}) == hist.BAD_OHLC
    assert hist.row_problem({**good, "volume": -1}) == hist.BAD_VOLUME
    assert hist.row_problem({**good, "volume": None}) == hist.BAD_VOLUME
    assert hist.row_problem({**good, "session_id": ""}) == hist.NOT_A_SESSION
    assert hist.row_problem({**good, "session_phase": "PRE"}) == hist.OUTSIDE_RTH


def test_topup_cli_path_runs_every_step_offline(store, tmp_path):
    from scripts.research_warehouse import cli

    client = FakeClient()
    client.bars["1d"] = {"SPY": _daily(_week(400.0))}
    client.bars["1h"] = {"SPY": _intraday(["2026-09-25 09:30", "2026-09-25 10:30"])}
    report = cli.run_history_topup(
        store, symbols="SPY", client=client, lock_path=tmp_path / "build.lock", log=lambda *_: None
    )
    assert report["status"] == "OK"
    assert report["d1"]["rows_published"]["bar_d1_history"] == 4
    assert report["h1"]["rows_published"]["bar_h1"] == 2
    assert report["coverage"]["history_symbols"] == 1
    assert not (tmp_path / "build.lock").exists()  # held only around each seal


def test_night_slot_tops_up_and_sits_after_the_regime_table(store, monkeypatch, tmp_path):
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    monkeypatch.syspath_prepend(scripts_dir)
    from ai_jobs import lake_history_topup, runner

    names = [slot.name for slot in runner.default_slots()]
    assert names[names.index("lake_history_topup") - 1] == "market_regime_table"
    slot = next(slot for slot in runner.default_slots() if slot.name == "lake_history_topup")
    assert slot.uses_model is False

    client = FakeClient()
    client.bars["1d"] = {"SPY": _daily(_week(400.0))}
    monkeypatch.setattr(hist, "history_universe", lambda *_a, **_k: ["SPY"])
    from research_warehouse import history as pkg_history

    monkeypatch.setattr(pkg_history, "history_universe", lambda *_a, **_k: ["SPY"])
    monkeypatch.setenv("TRADINGBOTV3_RESEARCH_DIR", str(tmp_path / "unused"))
    lake = sys.modules["research_warehouse.store"].ResearchStore(store.root)
    result = lake_history_topup.run_lake_history_topup(store=lake, client=client)
    assert result["status"] == "ok"
    assert "bar_d1_history +4" in result["reason"]
    again = lake_history_topup.run_lake_history_topup(store=lake, client=client)
    assert "bar_d1_history" not in again["reason"]


def test_night_slot_is_a_no_op_without_a_lake(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from ai_jobs import lake_history_topup
    from research_warehouse.store import ResearchStore as PkgStore

    monkeypatch.setattr(PkgStore, "open", classmethod(lambda cls, root=None: None))
    result = lake_history_topup.run_lake_history_topup(client=object())
    assert result["status"] == "ok" and "not configured" in result["reason"]


def test_yahoo_client_splits_batches_and_retries_missing_once(monkeypatch):
    import yfinance

    frame = _daily([("2026-09-25", 1, 2, 0.5, 1.5, 10)])
    both = pd.concat({"SPY": frame}, axis=1)
    calls, sleeps = [], []

    def fake_download(tickers, **kwargs):
        calls.append(list(tickers))
        return both if "SPY" in tickers else pd.DataFrame()

    monkeypatch.setattr(yfinance, "download", fake_download)
    client = hist.YahooClient(pause=1.0, backoff=10.0, sleep=sleeps.append)
    out = client.fetch_bars(["SPY", "BRK.B"], interval="1d", start=date(2018, 1, 1))

    assert list(out) == ["SPY"]
    assert calls == [["SPY", "BRK-B"], ["BRK-B"]]  # Yahoo spelling; one second pass only
    assert sleeps == [1.0]  # the pause between requests; no backoff for the single retry

    with pytest.raises(hist.ProviderError):
        hist.YahooClient(pause=1.0, retries=3, backoff=10.0, sleep=sleeps.append).fetch_bars(["XXX"], interval="1d", period="5d")
    assert sleeps[1:] == [10.0, 1.0, 20.0, 1.0]  # backoff between tries, none after the last


def test_history_lock_waits_for_a_running_build(tmp_path):
    from scripts.research_warehouse import cli

    lock = tmp_path / "build.lock"
    waits = []
    with cli.single_flight(lock):
        held = cli.history_lock(lock, wait_seconds=60, sleep=waits.append)
        with pytest.raises(cli.SingleFlightError):
            with held():
                pass
    assert waits == [30, 30]
    with cli.history_lock(lock)():
        assert lock.exists()
    assert not lock.exists()
