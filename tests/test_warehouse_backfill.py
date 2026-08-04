"""Nightly/weekly backfill and the yfinance seed (plan Phase 3b, sec 5.2).

Pinned here:

* every net-new request is gated by the shared pacer, so capture can never
  crowd a champion;
* backfill is ETH-inclusive (``useRTH=0``) and lands as ``capture_mode=BACKFILL``
  so the AS_OBSERVED filter excludes it from coverage/latency/promotion claims;
* a job interrupted by the ~23:45 ET TWS restart resumes with no duplicate and
  no hole - already-archived sessions are not re-requested;
* nothing skipped is silent: paced-out and no-response work leaves gap rows;
* the yfinance seed trickles in batches with a per-symbol ledger and is
  resumable, never a bulk scrape (risk R11).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from scripts.research_warehouse import backfill, pacer as pacer_mod
from scripts.research_warehouse.backfill import FetchResult
from scripts.research_warehouse.store import ResearchStore

UTC = timezone.utc
NOW = datetime(2026, 8, 4, 2, 0, tzinfo=UTC)
SESSION = date(2026, 8, 3)


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


@pytest.fixture()
def pacer():
    return pacer_mod.IbPacer(clock=lambda: NOW, capture_allowance=100)


def _bars(day: date, count: int = 3, *, premarket: bool = False):
    first = datetime(day.year, day.month, day.day, 8 if premarket else 13, 30, tzinfo=UTC)
    return [
        {
            "interval_start": first + timedelta(minutes=5 * index),
            "interval_end": first + timedelta(minutes=5 * (index + 1)),
            "open": 10.0 + index,
            "high": 11.0 + index,
            "low": 9.0 + index,
            "close": 10.5 + index,
            "volume": 500 + index,
        }
        for index in range(count)
    ]


def _fetcher(calls, *, bars=None, error=None):
    def fetch(symbol, day, *, timeframe, use_rth):
        calls.append({"symbol": symbol, "day": day, "timeframe": timeframe, "use_rth": use_rth})
        if error is not None:
            return FetchResult(error_code=error[0], error_message=error[1])
        return FetchResult(bars=list(bars if bars is not None else _bars(day)))

    return fetch


def test_nightly_backfill_is_eth_inclusive_and_marked_backfill(store, pacer):
    calls = []
    report = backfill.run_nightly_backfill(
        store,
        ["AAPL", "MSFT"],
        fetcher=_fetcher(calls, bars=_bars(SESSION, premarket=True)),
        session_date=SESSION,
        pacer=pacer,
        now=NOW,
    )

    assert report.by_outcome == {"OK": 2} and report.rows_published == 6
    # useRTH=0 on every request: premarket history is not recoverable later.
    assert {call["use_rth"] for call in calls} == {False}
    rows = store.read_table("bar_m5").to_pylist()
    assert {row["capture_mode"] for row in rows} == {"BACKFILL"}
    assert {row["provider"] for row in rows} == {"IBKR"}
    assert {row["session_phase"] for row in rows} == {"PRE"}
    assert {row["is_complete"] for row in rows} == {True}


def test_backfill_resumes_across_a_tws_restart_without_duplicates(store, pacer):
    calls = []
    first = backfill.run_nightly_backfill(
        store, ["AAPL"], fetcher=_fetcher(calls), session_date=SESSION, pacer=pacer, now=NOW
    )
    assert first.rows_published == 3

    # The 23:45 restart lands here; the job runs again from scratch.
    second = backfill.run_nightly_backfill(
        store, ["AAPL"], fetcher=_fetcher(calls), session_date=SESSION, pacer=pacer, now=NOW
    )
    assert second.by_outcome == {"ALREADY_HAVE": 1} and second.rows_published == 0
    assert store.read_table("bar_m5").num_rows == 3
    assert len(calls) == 1  # the second run made no provider request at all


def test_a_disconnect_stops_cleanly_and_records_what_was_missed(store, pacer):
    calls = []
    connected = {"value": True}

    def is_connected():
        # Connected for the first symbol, gone for the rest (TWS restart).
        was = connected["value"]
        connected["value"] = False
        return was

    report = backfill.run_backfill(
        store,
        ["AAPL", "MSFT", "NVDA"],
        fetcher=_fetcher(calls),
        job="nightly_backfill",
        days=[SESSION],
        pacer=pacer,
        is_connected=is_connected,
        now=NOW,
    )

    assert report.status == "STOPPED" and report.stopped_reason == "provider disconnected"
    assert report.by_outcome["OK"] == 1 and report.by_outcome["DISCONNECTED"] == 2
    gaps = store.read_table("collection_gap").to_pylist()
    assert {gap["symbol"] for gap in gaps} == {"MSFT", "NVDA"}
    assert {gap["reason"] for gap in gaps} == {"NO_RESPONSE"}


def test_capture_requests_are_gated_by_the_pacer(store):
    calls = []
    tight = pacer_mod.IbPacer(clock=lambda: NOW, capture_allowance=2)
    report = backfill.run_backfill(
        store,
        ["AAPL", "MSFT", "NVDA", "TSLA"],
        fetcher=_fetcher(calls),
        job="nightly_backfill",
        days=[SESSION],
        pacer=tight,
        now=NOW,
    )

    assert len(calls) == 2  # the budget, not the cohort, decided
    assert report.by_outcome["OK"] == 2 and report.by_outcome["PACED_OUT"] == 2
    paced = store.read_table("collection_gap").to_pylist()
    # A pacing shortfall is intended-but-not-collected. Sec 5.4 keeps
    # NOT_COLLECTED_BY_POLICY for genuine policy absence, so it must never
    # absorb a real gap (review defect D7).
    assert {gap["reason"] for gap in paced} == {backfill.REASON_TIMED_OUT}
    assert backfill.NOT_COLLECTED_BY_POLICY not in {gap["reason"] for gap in paced}


def test_a_pacing_error_backs_capture_off_and_is_tagged_capture(store, pacer):
    calls = []
    report = backfill.run_backfill(
        store,
        ["AAPL", "MSFT"],
        fetcher=_fetcher(calls, error=(162, "pacing violation")),
        job="nightly_backfill",
        days=[SESSION],
        pacer=pacer,
        now=NOW,
    )

    assert report.by_outcome["PACED_OUT"] >= 1
    snapshot = pacer.snapshot(now=NOW)
    assert snapshot.capture_errors >= 1 and snapshot.champion_errors == 0
    assert snapshot.backoff_until  # capture is now in cool-off
    assert store.read_table("bar_m5").num_rows == 0


def test_champion_traffic_is_never_delayed_by_a_backfill(store, pacer):
    """The pacer meters capture only; a champion request is pass-through."""
    calls = []
    backfill.run_backfill(
        store,
        [f"SYM{index}" for index in range(20)],
        fetcher=_fetcher(calls),
        job="nightly_backfill",
        days=[SESSION],
        pacer=pacer,
        now=NOW,
    )
    before = pacer.snapshot(now=NOW).champion_requests_in_window
    pacer.note_champion_request("daily_bars", "AAPL", now=NOW)
    assert pacer.snapshot(now=NOW).champion_requests_in_window == before + 1


def test_no_response_is_recorded_as_a_gap_not_a_hole(store, pacer):
    report = backfill.run_backfill(
        store,
        ["AAPL"],
        fetcher=_fetcher([], bars=[]),
        job="nightly_backfill",
        days=[SESSION],
        pacer=pacer,
        now=NOW,
    )
    assert report.by_outcome == {"NO_RESPONSE": 1}
    gap = store.read_table("collection_gap").to_pylist()[0]
    assert gap["symbol"] == "AAPL" and gap["reason"] == "NO_RESPONSE"


def test_naive_bar_timestamps_are_dropped_not_localized(store, pacer):
    naive = [{"interval_start": datetime(2026, 8, 3, 13, 30), "open": 1, "high": 2, "low": 0.5, "close": 1.5}]
    report = backfill.run_backfill(
        store,
        ["AAPL"],
        fetcher=_fetcher([], bars=naive),
        job="nightly_backfill",
        days=[SESSION],
        pacer=pacer,
        now=NOW,
    )
    assert report.rows_published == 0 and report.by_outcome == {"NO_RESPONSE": 1}


def test_weekly_sweep_covers_the_weeks_sessions(store, pacer):
    calls = []
    report = backfill.run_weekly_universe_sweep(
        store,
        ["AAPL"],
        fetcher=_fetcher(calls),
        week_ending=date(2026, 8, 8),  # Saturday
        pacer=pacer,
        now=NOW,
    )
    days = sorted({call["day"] for call in calls})
    assert days == [date(2026, 8, 3), date(2026, 8, 4), date(2026, 8, 5), date(2026, 8, 6), date(2026, 8, 7)]
    assert report.by_outcome["OK"] == 5  # weekends are not sessions


# --- D6: the tee must not block the ETH job -------------------------------
def _tee_rows(day: date, count: int = 3):
    """RTH bars as the tee archives them: capture_mode=LIVE, not BACKFILL."""
    rows = []
    first = datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC)
    for index in range(count):
        start = first + timedelta(minutes=5 * index)
        rows.append(
            {
                "symbol": "AAPL",
                "interval_start": start,
                "interval_end": start + timedelta(minutes=5),
                "session_id": f"XNYS:{day.isoformat()}",
                "session_phase": "RTH",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "vwap": None,
                "trade_count": None,
                "provider": "IBKR",
                "is_complete": True,
                "quality": "COMPLETE",
                "source_hash": "",
                "event_at": start + timedelta(minutes=5),
                "observed_at": NOW,
                "capture_mode": "LIVE",
                "revision_id": "",
                "supersedes_revision_id": "",
                "schema_version": backfill.SCHEMA_VERSION,
                "run_id": "tee",
            }
        )
    return rows


def test_the_eth_job_still_runs_for_symbols_the_tee_already_captured(store, pacer):
    """D6: day-level "already have" defeated LD-03 entirely.

    The tee archives RTH bars for the whole cohort every session, so a
    day-level check made the ETH-inclusive nightly job skip exactly the
    symbols it exists to extend. Premarket bars must publish, and the tee's
    RTH bars must not be duplicated.
    """
    store.publish("bar_m5", _tee_rows(SESSION), job_id="tee")
    assert store.read_table("bar_m5").num_rows == 3

    calls = []
    # The ETH answer repeats the RTH bars the tee already has and adds premarket.
    eth = _bars(SESSION, count=2, premarket=True) + [
        {
            "interval_start": datetime(2026, 8, 3, 13, 30, tzinfo=UTC),
            "interval_end": datetime(2026, 8, 3, 13, 35, tzinfo=UTC),
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100,
        }
    ]
    report = backfill.run_nightly_backfill(
        store, ["AAPL"], fetcher=_fetcher(calls, bars=eth), session_date=SESSION, pacer=pacer, now=NOW
    )

    assert len(calls) == 1, "the tee's RTH rows must not suppress the ETH request"
    assert report.by_outcome == {"OK": 1}
    assert report.rows_published == 2  # the two premarket bars
    assert report.rows_duplicate == 1  # the repeated RTH bar, not written twice

    rows = store.read_table("bar_m5").to_pylist()
    assert len(rows) == 5  # 3 tee + 2 premarket, no loss and no duplicate
    starts = [row["interval_start"] for row in rows]
    assert len(starts) == len(set(starts))
    assert {row["session_phase"] for row in rows} == {"RTH", "PRE"}
    # The tee's own rows keep their LIVE capture_mode; only new rows are BACKFILL.
    by_phase = {row["session_phase"]: row["capture_mode"] for row in rows}
    assert by_phase["RTH"] == "LIVE" and by_phase["PRE"] == "BACKFILL"

    # And the ETH job is still idempotent: a re-run requests nothing.
    again = backfill.run_nightly_backfill(
        store, ["AAPL"], fetcher=_fetcher(calls, bars=eth), session_date=SESSION, pacer=pacer, now=NOW
    )
    assert again.by_outcome == {"ALREADY_HAVE": 1} and len(calls) == 1
    assert store.read_table("bar_m5").num_rows == 5


# --- D7: the pacer clock must advance -------------------------------------
class _AdvancingClock:
    """A fake wall clock the run advances as it works."""

    def __init__(self, start: datetime, step_seconds: float = 30.0):
        self.at = start
        self.step = timedelta(seconds=step_seconds)

    def __call__(self) -> datetime:
        return self.at

    def sleep(self, seconds: float) -> None:
        self.at += timedelta(seconds=max(float(seconds), 0.0))

    def work(self) -> None:
        self.at += self.step


def test_one_run_gets_far_past_a_single_pacer_window(store, pacer):
    """D7: a frozen stamp capped any invocation at one window's allowance.

    With the real clock the 10-minute window advances as the run works, so
    the token bucket refills and a nightly pass can clear its cohort.
    """
    # 5s per request means 15 grants fit well inside one 10-minute window, so
    # the 16th genuinely has to wait for the bucket to refill.
    clock = _AdvancingClock(NOW, step_seconds=5.0)
    metered = pacer_mod.IbPacer(clock=clock, capture_allowance=15)
    cohort = [f"SYM{index:03d}" for index in range(40)]

    calls = []

    def fetch(symbol, day, *, timeframe, use_rth):
        calls.append(symbol)
        clock.work()  # the request itself takes time
        return FetchResult(bars=list(_bars(day, count=1)))

    report = backfill.run_backfill(
        store,
        cohort,
        fetcher=fetch,
        job="nightly_backfill",
        days=[SESSION],
        pacer=metered,
        clock=clock,
        sleep=clock.sleep,
        time_budget_seconds=3600.0,
        now=NOW,
    )

    assert report.requested == 40, "the whole cohort, not one window's worth"
    assert report.by_outcome == {"OK": 40}
    assert len(calls) == 40
    assert report.gaps_recorded == 0
    # The pacer was genuinely consulted and did make the run wait.
    assert report.seconds_waited > 0


def test_a_frozen_run_stamp_would_have_capped_the_run(store):
    """The same cohort against a frozen clock: the old behaviour, pinned."""
    frozen = pacer_mod.IbPacer(clock=lambda: NOW, capture_allowance=15)
    calls = []
    report = backfill.run_backfill(
        store,
        [f"SYM{index:03d}" for index in range(40)],
        fetcher=_fetcher(calls),
        job="nightly_backfill",
        days=[SESSION],
        pacer=frozen,
        clock=lambda: NOW,
        now=NOW,
    )
    assert report.requested == 15 and len(calls) == 15
    assert report.by_outcome["PACED_OUT"] == 25
    assert {gap["reason"] for gap in store.read_table("collection_gap").to_pylist()} == {
        backfill.REASON_TIMED_OUT
    }


# --- D7: gap dedupe and resolution ----------------------------------------
def test_a_repeated_miss_does_not_inflate_the_gap_table(store, pacer):
    """D7: _record_missed appended blindly, so re-runs multiplied gap rows."""
    empty = _fetcher([], bars=[])
    for _ in range(3):
        backfill.run_backfill(
            store, ["AAPL"], fetcher=empty, job="nightly_backfill", days=[SESSION], pacer=pacer, now=NOW
        )
    gaps = store.read_table("collection_gap").to_pylist()
    assert len(gaps) == 1, "one open gap, however many runs observed it"
    assert gaps[0]["reason"] == backfill.REASON_NO_RESPONSE
    assert gaps[0]["resolved_at"] is None


def test_a_later_run_that_fills_a_gap_resolves_it(store, pacer):
    """D7: nothing ever set resolved_at/BACKFILLED on a filled gap."""
    backfill.run_backfill(
        store,
        ["AAPL"],
        fetcher=_fetcher([], bars=[]),
        job="nightly_backfill",
        days=[SESSION],
        pacer=pacer,
        now=NOW,
    )
    partition = f"month={SESSION:%Y-%m}"
    assert len(backfill.open_gap_keys(store, [partition])) == 1

    later = NOW + timedelta(days=1)
    # The next night is a new run: a fresh pacer, past the identical-request
    # cooldown that correctly suppressed an immediate retry.
    report = backfill.run_backfill(
        store,
        ["AAPL"],
        fetcher=_fetcher([], bars=_bars(SESSION)),
        job="nightly_backfill",
        days=[SESSION],
        pacer=pacer_mod.IbPacer(clock=lambda: later, capture_allowance=100),
        clock=lambda: later,
        now=later,
    )
    assert report.rows_published == 3 and report.gaps_resolved == 1

    # The original row is preserved; the superseding row closes it.
    rows = store.read_table("collection_gap").to_pylist()
    assert len(rows) == 2
    assert backfill.open_gap_keys(store, [partition]) == {}
    closed = max(rows, key=lambda row: row["detected_at"])
    assert closed["resolution"] == backfill.RESOLUTION_BACKFILLED
    assert closed["resolved_at"] == later


def test_backfill_is_a_no_op_when_the_warehouse_is_disabled():
    report = backfill.run_nightly_backfill(None, ["AAPL"], fetcher=lambda *a, **k: FetchResult())
    assert report.status == "DISABLED" and report.rows_published == 0


# --- the yfinance 60-day seed ----------------------------------------------
def _seed_fetcher(calls, *, fail_for=()):
    def fetch(symbol, start, end):
        calls.append({"symbol": symbol, "start": start, "end": end})
        if symbol in fail_for:
            return FetchResult(error_message="Too Many Requests")
        return FetchResult(bars=_bars(SESSION, 2))

    return fetch


def test_seed_trickles_in_batches_and_resumes_from_its_ledger(store, tmp_path):
    calls = []
    universe = ["AAPL", "MSFT", "NVDA", "TSLA"]
    spool = tmp_path / "spool"

    first = backfill.run_yahoo_seed(
        store, universe, fetcher=_seed_fetcher(calls), spool_dir=spool, batch_size=2, now=NOW
    )
    assert first.symbols_completed == 2 and first.remaining == 2
    assert [call["symbol"] for call in calls] == ["AAPL", "MSFT"]
    # 60 days of history is the whole point of the seed.
    assert (calls[0]["end"] - calls[0]["start"]).days == backfill.YAHOO_M5_WINDOW_DAYS

    second = backfill.run_yahoo_seed(
        store, universe, fetcher=_seed_fetcher(calls), spool_dir=spool, batch_size=2, now=NOW
    )
    assert second.symbols_completed == 2 and second.remaining == 0
    assert [call["symbol"] for call in calls[2:]] == ["NVDA", "TSLA"]

    done = backfill.run_yahoo_seed(store, universe, fetcher=_seed_fetcher(calls), spool_dir=spool, now=NOW)
    assert done.status == "COMPLETE" and len(calls) == 4

    rows = store.read_table("bar_m5").to_pylist()
    assert {row["provider"] for row in rows} == {"YAHOO"}
    assert {row["capture_mode"] for row in rows} == {"BACKFILL"}


def test_a_throttled_seed_symbol_is_retried_on_a_later_night(store, tmp_path):
    calls = []
    backoff_calls = []
    spool = tmp_path / "spool"

    first = backfill.run_yahoo_seed(
        store,
        ["AAPL", "MSFT"],
        fetcher=_seed_fetcher(calls, fail_for={"AAPL"}),
        spool_dir=spool,
        now=NOW,
        backoff=lambda symbol, error: backoff_calls.append((symbol, error)),
    )
    assert first.symbols_failed == 1 and first.symbols_completed == 1
    assert backoff_calls == [("AAPL", "Too Many Requests")]

    ledger = backfill.load_seed_ledger(spool)
    assert ledger["AAPL"]["status"] == "FAILED" and ledger["MSFT"]["status"] == "COMPLETE"

    second = backfill.run_yahoo_seed(store, ["AAPL", "MSFT"], fetcher=_seed_fetcher(calls), spool_dir=spool, now=NOW)
    assert second.symbols_completed == 1  # only the failed symbol is retried
    assert backfill.load_seed_ledger(spool)["AAPL"]["status"] == "COMPLETE"


def test_seed_never_touches_the_ib_budget(store, tmp_path):
    arbiter = pacer_mod.reset_pacer(pacer_mod.IbPacer(clock=lambda: NOW))
    backfill.run_yahoo_seed(store, ["AAPL"], fetcher=_seed_fetcher([]), spool_dir=tmp_path / "spool", now=NOW)
    snapshot = arbiter.snapshot(now=NOW)
    assert snapshot.capture_requests_in_window == 0 and snapshot.grants == 0
    pacer_mod.reset_pacer()


# --- the capture-side IB adapter (offline parts) ---------------------------
def test_capture_requests_are_eth_scoped_and_unambiguous():
    from scripts.research_warehouse import ib_capture

    request = ib_capture.historical_request("aapl", SESSION, timeframe="M5", use_rth=False)
    assert request["symbol"] == "AAPL"
    assert request["barSizeSetting"] == "5 mins" and request["durationStr"] == "1 D"
    assert request["useRTH"] == 0
    # ETH close, so premarket and postmarket bars come back too (LD-03).
    assert request["endDateTime"] == "20260803 20:00:00 US/Eastern"
    # Epoch seconds: no timezone inference anywhere downstream.
    assert request["formatDate"] == 2
    assert request["keepUpToDate"] is False

    rth = ib_capture.historical_request("AAPL", SESSION, use_rth=True)
    assert rth["useRTH"] == 1 and rth["endDateTime"].endswith("16:00:00 US/Eastern")
    with pytest.raises(ValueError, match="unsupported capture timeframe"):
        ib_capture.historical_request("AAPL", SESSION, timeframe="H2")  # H2 is CUT


def test_capture_bars_parse_from_epoch_seconds():
    from scripts.research_warehouse import ib_capture

    parsed = ib_capture.parse_bar(
        {"date": "1786109400", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 9, "barCount": 3}
    )
    assert parsed["interval_start"].tzinfo is not None
    assert parsed["interval_end"] - parsed["interval_start"] == timedelta(minutes=5)
    assert parsed["trade_count"] == 3
    assert ib_capture.parse_bar({"date": "", "open": 1}) is None
    assert ib_capture.parse_bar({"date": "1786109400", "open": "n/a"}) is None


def test_a_naive_ib_timestamp_is_dropped_not_rezoned():
    """D17: the fallback read IB's naive strings as UTC.

    IB's naive strings are exchange-local, so a TWS build answering
    formatDate=1-style to this connection would have shifted every bar 4-5
    hours, silently. BD-06's rule: naive is uncertainty, never localized.
    """
    from scripts.research_warehouse import ib_capture

    for naive in ("20260803  13:30:00", "20260803 13:30:00", "20260803"):
        assert ib_capture.parse_bar(
            {"date": naive, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 9}
        ) is None
    assert ib_capture._epoch_to_utc(datetime(2026, 8, 3, 13, 30)) is None
    # An already-aware datetime is still accepted unchanged.
    aware = datetime(2026, 8, 3, 13, 30, tzinfo=UTC)
    assert ib_capture._epoch_to_utc(aware) == aware


def test_capture_fetcher_tags_errors_as_capture_and_reconnects(store):
    from scripts.research_warehouse import ib_capture

    arbiter = pacer_mod.IbPacer(clock=lambda: NOW)

    class FakeTransport:
        def __init__(self):
            self.connected = False
            self.connects = 0
            self.requests = []
            self.next_error = None

        def is_connected(self):
            return self.connected

        def connect(self, spec):
            self.connects += 1
            self.connected = True

        def request_historical(self, **kwargs):
            self.requests.append(kwargs)
            if self.next_error:
                error, self.next_error = self.next_error, None
                return [], error
            return [{"date": "1786109400", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 5}], None

    transport = FakeTransport()
    fetcher = ib_capture.IbCaptureFetcher(
        transport, spec=ib_capture.backfill_connection_spec(), pacer=arbiter
    )
    # Disconnected at first (the TWS restart): the fetcher reconnects itself.
    result = fetcher("AAPL", SESSION, timeframe="M5", use_rth=False)
    assert transport.connects == 1 and result.ok and len(result.bars) == 1

    transport.next_error = (162, "pacing violation")
    paced = fetcher("MSFT", SESSION, timeframe="M5", use_rth=False)
    assert paced.error_code == 162 and not paced.ok
    snapshot = arbiter.snapshot(now=NOW)
    assert snapshot.capture_errors == 1 and snapshot.champion_errors == 0
    assert snapshot.backoff_until


def test_capture_connection_specs_use_the_allocated_client_ids():
    from scripts.research_warehouse import ib_capture

    assert ib_capture.backfill_connection_spec().client_id == 1011
    assert ib_capture.streaming_connection_spec().client_id == 1010
    bad = ib_capture.CaptureConnectionSpec(role=pacer_mod.ROLE_NIGHTLY_BACKFILL, client_id=1003)
    with pytest.raises(pacer_mod.ClientIdError, match="retired"):
        bad.validate()
