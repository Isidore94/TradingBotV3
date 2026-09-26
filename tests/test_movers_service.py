"""MoversService: bot-cache reads, yfinance fallback, baseline caching, last-good board."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ui.services import movers_service as svc  # noqa: E402

NY = ZoneInfo("America/New_York")
LA = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 9, 22, 10, 40, 30, tzinfo=NY)  # 14 bars completed


@pytest.fixture(autouse=True)
def _la_market_tz(monkeypatch):
    monkeypatch.setattr(svc, "_market_local_tz", lambda: LA)


def _naive_la_bars(day_closes, *, volume=1000.0, day=22):
    """Naive LA bars from 06:30 local (09:30 NY), the bot's shape (volume in lots)."""
    start = datetime(2026, 9, day, 6, 30)
    out = []
    prev = day_closes[0]
    for i, close in enumerate(day_closes):
        out.append({"dt": start + timedelta(minutes=5 * i), "open": prev,
                    "high": max(prev, close) + 0.5, "low": min(prev, close) - 0.5,
                    "close": close, "volume": volume})
        prev = close
    return out


class FakeBot:
    is_process_proxy = False

    def __init__(self, universe, bars):
        self.universe = set(universe)
        self.bars = bars
        self.calls: list[str] = []

    def get_scan_symbol_set(self):
        self.calls.append("get_scan_symbol_set")
        return set(self.universe)

    def m5_chart_bars(self, symbol, max_sessions=2):
        self.calls.append("m5_chart_bars")
        return list(self.bars.get(symbol, []))

    def fetch_m5_chart_bars(self, *args, **kwargs):  # the IB path: must never run
        raise AssertionError("Movers must never fetch from IB")


def _frame(day_closes_by_day, *, volume=100_000.0):
    rows, index = [], []
    for day, closes in day_closes_by_day:
        start = datetime(2026, 9, day, 9, 30, tzinfo=NY)
        prev = closes[0]
        for i, close in enumerate(closes):
            index.append(pd.Timestamp(start + timedelta(minutes=5 * i)))
            rows.append({"Open": prev, "High": max(prev, close) + 0.5,
                         "Low": min(prev, close) - 0.5, "Close": close, "Volume": volume})
            prev = close
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


class FakeDownloader:
    def __init__(self, frames):
        self.frames = frames
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def __call__(self, symbols, *, period, interval):
        self.calls.append((tuple(symbols), period))
        if len(symbols) == 1:
            return self.frames.get(symbols[0], pd.DataFrame())
        return {s: self.frames.get(s, pd.DataFrame()) for s in symbols}


def _history_frame():
    days = [(d, [100.0] * 78) for d in (8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 21)]
    return _frame(days + [(22, [100.0] * 11 + [100.5, 101.0, 101.5])])


def _service(bot, downloader, *, universe=("QQQ",), focus=None):
    return svc.MoversService(
        bot_provider=lambda: bot,
        focus_provider=(lambda: focus) if focus is not None else None,
        downloader=downloader,
        universe_provider=lambda: list(universe),
        clock=lambda: NOW,
        autostart=False,
    )


def test_read_bot_bars_converts_round_lots_and_skips_empty():
    bot = FakeBot(["AAA"], {"AAA": _naive_la_bars([100.0, 101.0], volume=12.0)})
    out = svc.read_bot_bars(bot, ["AAA", "NONE"], rpc_gap=0)
    assert list(out) == ["AAA"]
    assert out["AAA"][0]["volume"] == 1200.0


def test_choose_freshest_prefers_the_newer_series_and_yahoo_on_a_tie():
    bot_bars = {"AAA": _naive_la_bars([100.0] * 14), "BBB": _naive_la_bars([100.0] * 12)}
    yahoo = {"AAA": svc_rows(_frame([(22, [100.0] * 13)])),
             "BBB": svc_rows(_frame([(22, [100.0] * 12)], volume=7.0))}
    chosen = svc.choose_freshest(bot_bars, yahoo, now=NOW, local_tz=LA)
    assert len(chosen["AAA"]) == 14  # bot is newer
    assert chosen["BBB"][-1]["volume"] == 7.0  # tie -> Yahoo


def svc_rows(frame):
    import autopilot_core as core

    return core._frame_rows(frame)


def test_small_bot_universe_adds_the_yahoo_sweep_and_caches_baselines():
    bot = FakeBot(["AAA"], {
        "AAA": _naive_la_bars([100.0] * 11 + [100.5, 101.0, 101.5]),
        "SPY": _naive_la_bars([400.0] * 14),
    })
    frames = {"QQQ": _history_frame(), "AAA": _history_frame(), "SPY": _history_frame()}
    downloader = FakeDownloader(frames)
    service = _service(bot, downloader, focus={"m5": {"long": ["AAA"]}})
    emitted = []
    service.moversChanged.connect(emitted.append)
    service._run_once(service._focus_snapshot())

    periods = [period for _symbols, period in downloader.calls]
    assert svc.YAHOO_TODAY_PERIOD in periods and svc.YAHOO_BASELINE_PERIOD in periods
    board = emitted[-1]
    assert board["bot_universe"] == 1
    assert board["yahoo_universe"] >= 2
    assert board["mine"]["long"][0]["symbol"] == "AAA"
    assert board["mine"]["long"][0]["rvol"] is not None  # baseline arrived
    assert "fetch_m5_chart_bars" not in bot.calls

    downloader.calls.clear()
    service._run_once(service._focus_snapshot())
    assert all(period != svc.YAHOO_BASELINE_PERIOD for _s, period in downloader.calls)
    # Five-minute cadence: the second tick a minute later does not re-sweep either.
    assert all(period != svc.YAHOO_TODAY_PERIOD for _s, period in downloader.calls)


def test_large_bot_universe_skips_the_yahoo_sweep():
    names = [f"S{i:03d}" for i in range(svc.BOT_UNIVERSE_MIN)]
    bot = FakeBot(names, {"SPY": _naive_la_bars([400.0] * 14)})
    downloader = FakeDownloader({})
    service = _service(bot, downloader)
    service._run_once({"long": [], "short": []})
    assert all(period != svc.YAHOO_TODAY_PERIOD for _s, period in downloader.calls)
    assert service.bot_universe_size == svc.BOT_UNIVERSE_MIN


def test_failed_refresh_keeps_the_last_good_board():
    bot = FakeBot([], {"SPY": _naive_la_bars([400.0] * 14)})
    service = _service(bot, FakeDownloader({}))
    service._worker({"long": [], "short": []})
    good = service.board()
    assert good and "state" in good

    def boom():
        raise RuntimeError("child gone")

    service.set_bot_provider(boom)
    service._worker({"long": [], "short": []})
    assert service.board() == good
    assert "FAILED" in service.status_text()


def test_failed_yahoo_sweep_does_not_move_the_sweep_clock():
    class Broken:
        calls = 0

        def __call__(self, symbols, *, period, interval):
            Broken.calls += 1
            raise RuntimeError("yahoo down")

    bot = FakeBot(["AAA"], {"SPY": _naive_la_bars([400.0] * 14)})
    service = _service(bot, Broken())
    service._run_once({"long": [], "short": []})
    assert service._yahoo_at is None
    assert service._yahoo_due(NOW)  # the next tick retries the sweep


def test_next_tick_is_the_bar_boundary_plus_grace():
    at = datetime(2026, 9, 22, 10, 41, 0, tzinfo=NY)
    assert svc.next_tick_delay_ms(at) == (4 * 60 + svc.TICK_GRACE_SECONDS) * 1000
    just_after = datetime(2026, 9, 22, 10, 45, 5, tzinfo=NY)
    assert svc.next_tick_delay_ms(just_after) == (svc.TICK_GRACE_SECONDS - 5) * 1000
    past_grace = datetime(2026, 9, 22, 10, 45, 30, tzinfo=NY)
    assert svc.next_tick_delay_ms(past_grace) == (5 * 60 - 10) * 1000


def test_timer_is_single_shot_and_rearms_after_a_tick():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    service = _service(None, FakeDownloader({}))
    assert service._timer.isSingleShot()
    service._clock = lambda: datetime(2026, 9, 22, 8, 0, tzinfo=NY)  # idle hours
    service._tick()
    assert service._timer.isActive()
    service.shutdown()
    assert not service._timer.isActive()
    service._tick()
    assert not service._timer.isActive()  # a stopped service never re-arms


def _big_universe_bot():
    names = [f"S{i:03d}" for i in range(svc.BOT_UNIVERSE_MIN)]
    bars = {"SPY": _naive_la_bars([400.0] * 14),
            "S000": _naive_la_bars([100.0] * 14),  # fresh
            "S001": _naive_la_bars([50.0] * 10, volume=5000.0)}  # stale (ends 10:15 NY)
    return FakeBot(names, bars)


def test_gap_fill_fetches_only_stale_or_missing_names(monkeypatch):
    monkeypatch.setattr(svc, "BOT_UNIVERSE_MIN", 300)
    bot = _big_universe_bot()
    downloader = FakeDownloader({"S001": _history_frame()})
    service = _service(bot, downloader)
    service._run_once({"long": [], "short": []})
    gap_calls = [symbols for symbols, period in downloader.calls if period == svc.GAP_FILL_PERIOD]
    fetched = {s for chunk in gap_calls for s in chunk}
    assert "S001" in fetched and "S002" in fetched  # stale and missing
    assert "S000" not in fetched and "SPY" not in fetched  # fresh from the bot
    assert service._gap_at == NOW
    assert "S001" in service._yahoo_bars


def test_gap_fill_cap_takes_the_most_liquid_first(monkeypatch):
    monkeypatch.setattr(svc, "GAP_FILL_MAX", 1)
    bot = _big_universe_bot()
    downloader = FakeDownloader({})
    service = _service(bot, downloader)
    service._run_once({"long": [], "short": []})
    gap_calls = [symbols for symbols, period in downloader.calls if period == svc.GAP_FILL_PERIOD]
    assert gap_calls == [("S001",)]  # the only stale name with known volume


def test_failed_gap_fill_keeps_last_bars_and_clock():
    class Broken:
        def __call__(self, symbols, *, period, interval):
            raise RuntimeError("down")

    bot = _big_universe_bot()
    service = _service(bot, Broken())
    service._yahoo_bars = {"S001": ["kept"]}
    service._run_once({"long": [], "short": []})
    assert service._gap_at is None
    assert service._yahoo_bars["S001"] == ["kept"]


def test_board_carries_persistence_group_and_earnings_tags():
    names = ["AAA", "BBB", "CCC"]
    rising = _naive_la_bars([100.0] * 78, day=21) + _naive_la_bars(
        [100.0] * 11 + [101.0, 102.0, 103.0]
    )  # a prior session so ATR14 is measurable
    bot = FakeBot(names, {"SPY": _naive_la_bars([400.0] * 14), **{n: rising for n in names}})
    service = svc.MoversService(
        bot_provider=lambda: bot, downloader=FakeDownloader({}),
        universe_provider=lambda: [], clock=lambda: NOW, autostart=False,
        industry_provider=lambda: {n: "Semiconductors" for n in names},
        earnings_provider=lambda today: {"BBB"},
    )
    service._run_once({"long": [], "short": []})
    service._run_once({"long": [], "short": []})
    rows = {r["symbol"]: r for r in service.board()["pop"]["long"]}
    assert set(rows) == set(names)
    assert rows["BBB"]["er"] is True and rows["AAA"]["er"] is False
    assert rows["AAA"]["group"] == "Semis"
    assert service.board()["groups"]["pop"]["long"] == [["Semis", 3]]
    assert rows["AAA"]["streak"] == 2 and rows["AAA"]["rank_change"] == 0


def test_outcome_rows_are_appended_and_a_failed_write_keeps_the_board(tmp_path, monkeypatch):
    bot = FakeBot([], {"SPY": _naive_la_bars([400.0] * 14)})
    service = svc.MoversService(
        bot_provider=lambda: bot, downloader=FakeDownloader({}),
        universe_provider=lambda: [], clock=lambda: NOW, autostart=False,
        outcomes_path=tmp_path / "out.jsonl",
    )
    monkeypatch.setattr(service._tracker, "observe",
                        lambda *a, **k: [{"kind": "flag", "symbol": "X"}])
    service._run_once({"long": [], "short": []})
    assert (tmp_path / "out.jsonl").read_text(encoding="utf-8").count('"flag"') == 1
    service._outcomes_path = tmp_path  # a directory: the write fails
    service._run_once({"long": [], "short": []})
    assert service.board() and "state" in service.board()


def test_first_tick_restores_todays_flags_from_the_log(tmp_path):
    import movers_outcomes

    path = tmp_path / "out.jsonl"
    movers_outcomes.append_records(path, [
        {"kind": "flag", "session": "2026-09-22", "episode": "2026-09-22T10:15:00-04:00",
         "side": "long", "symbol": "HOLD", "flagged_bar": "2026-09-22T10:30:00-04:00"},
        {"kind": "flag", "session": "2026-09-19", "episode": "old", "side": "long",
         "symbol": "OLD", "flagged_bar": "2026-09-19T10:30:00-04:00"},
    ])
    bot = FakeBot([], {"SPY": _naive_la_bars([400.0] * 14)})
    service = svc.MoversService(
        bot_provider=lambda: bot, downloader=FakeDownloader({}),
        universe_provider=lambda: [], clock=lambda: NOW, autostart=False,
        outcomes_path=path,
    )
    service._run_once({"long": [], "short": []})
    flagged = {s for ep in service._tracker.episodes.values() for s in ep["flagged"]}
    assert flagged == {"HOLD"}


def test_a_name_yahoo_returns_empty_is_skipped_for_three_ticks():
    bot = _big_universe_bot()
    downloader = FakeDownloader({})
    service = _service(bot, downloader)

    def gap_names():
        return {s for symbols, period in downloader.calls if period == svc.GAP_FILL_PERIOD
                for s in symbols}

    service._run_once({"long": [], "short": []})
    assert "S001" in gap_names()
    for _ in range(svc.GAP_EMPTY_BACKOFF_TICKS):
        downloader.calls.clear()
        service._run_once({"long": [], "short": []})
        assert "S001" not in gap_names()
    downloader.calls.clear()
    service._run_once({"long": [], "short": []})
    assert "S001" in gap_names()


def test_tick_is_idle_outside_regular_hours():
    service = _service(None, FakeDownloader({}))
    service._clock = lambda: datetime(2026, 9, 22, 8, 0, tzinfo=NY)
    started = []
    service._start = lambda: started.append(1) or True
    service._tick()
    assert started == []
    service._clock = lambda: NOW
    service._tick()
    assert started == [1]
    assert not svc.in_regular_hours(datetime(2026, 9, 26, 11, 0, tzinfo=NY))  # Saturday


# ------------------------------------------------------------ IB scanner source
class FakeScanner:
    """Stands in for the IB market scanner: a list of answers, or an exception."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _scanner_service(bot, downloader, scanner, *, clock=lambda: NOW):
    swept = []

    def universe():
        swept.append(1)
        return ["QQQ"]

    service = svc.MoversService(
        bot_provider=lambda: bot, downloader=downloader, universe_provider=universe,
        clock=clock, autostart=False, scanner=scanner,
    )
    return service, swept


def _small_bot():
    return FakeBot(["AAA"], {"SPY": _naive_la_bars([400.0] * 14),
                             "AAA": _naive_la_bars([100.0] * 14)})


def _gap_names(downloader):
    return {s for symbols, period in downloader.calls if period == svc.GAP_FILL_PERIOD
            for s in symbols}


def test_scanner_names_enter_the_pool_and_replace_the_universe_sweep():
    downloader = FakeDownloader({"GAIN": _history_frame(), "LOSE": _history_frame()})
    scanner = FakeScanner({"TOP_PERC_GAIN": ["GAIN"], "TOP_PERC_LOSE": ["LOSE", "AAA"],
                           "HOT_BY_VOLUME": []})
    service, swept = _scanner_service(_small_bot(), downloader, scanner)
    emitted = []
    service.moversChanged.connect(emitted.append)
    service._run_once({"long": [], "short": []})

    assert scanner.calls == 1
    assert swept == []  # no liquid-universe read
    assert all(period != svc.YAHOO_TODAY_PERIOD for _s, period in downloader.calls)
    gap = _gap_names(downloader)
    assert {"GAIN", "LOSE"} <= gap
    assert "AAA" not in gap  # fresh in the bot cache: no Yahoo for it
    board = emitted[-1]
    assert board["candidate_source"] == "ib_scanner"
    assert board["scanner_names"] == 3
    assert board["scanner_error"] == ""


def test_scanner_failure_reuses_todays_last_list_then_reports_stale():
    downloader = FakeDownloader({})
    scanner = FakeScanner({"TOP_PERC_GAIN": ["GAIN"]}, RuntimeError("TWS gone"))
    service, swept = _scanner_service(_small_bot(), downloader, scanner)
    service._run_once({"long": [], "short": []})
    downloader.calls.clear()
    service._gap_skip_until.clear()
    service._run_once({"long": [], "short": []})

    assert swept == []
    assert "GAIN" in _gap_names(downloader)
    board = service.board()
    assert board["candidate_source"] == "ib_scanner_stale"
    assert "TWS gone" in board["scanner_error"]


def test_scanner_failure_with_no_list_today_falls_back_to_the_universe_sweep():
    downloader = FakeDownloader({})
    scanner = FakeScanner(RuntimeError("no scanner permission"))
    service, swept = _scanner_service(_small_bot(), downloader, scanner)
    service._run_once({"long": [], "short": []})

    assert swept == [1]
    assert any(period == svc.YAHOO_TODAY_PERIOD for _s, period in downloader.calls)
    board = service.board()
    assert board["candidate_source"] == "yahoo_universe"
    assert "no scanner permission" in board["scanner_error"]


def test_yesterdays_scanner_list_is_not_reused():
    downloader = FakeDownloader({})
    moments = [NOW - timedelta(days=1), NOW]
    scanner = FakeScanner({"TOP_PERC_GAIN": ["OLD"]}, RuntimeError("down"))
    service, swept = _scanner_service(_small_bot(), downloader, scanner,
                                      clock=lambda: moments[0])
    service._run_once({"long": [], "short": []})
    moments.pop(0)
    downloader.calls.clear()
    service._gap_skip_until.clear()
    service._run_once({"long": [], "short": []})

    assert "OLD" not in _gap_names(downloader)
    assert service.board()["candidate_source"] == "yahoo_universe"


def test_an_empty_scanner_answer_counts_as_a_failure():
    downloader = FakeDownloader({})
    scanner = FakeScanner({"TOP_PERC_GAIN": [], "TOP_PERC_LOSE": []})
    service, swept = _scanner_service(_small_bot(), downloader, scanner)
    service._run_once({"long": [], "short": []})
    assert swept == [1]
    assert service.board()["candidate_source"] == "yahoo_universe"


def test_a_hung_worker_is_named_in_the_status_after_four_minutes(caplog):
    service = _service(None, FakeDownloader({}))
    service._running = True
    service._run_started = NOW - timedelta(minutes=svc.HUNG_WORKER_MINUTES, seconds=1)
    statuses = []
    service.statusChanged.connect(statuses.append)
    with caplog.at_level("WARNING"):
        service._tick()
    assert "stuck" in service.status_text()
    assert statuses and "stuck" in statuses[-1]
    assert any("stuck" in record.getMessage() for record in caplog.records)
    service.shutdown()


def test_a_short_running_worker_is_not_called_stuck():
    service = _service(None, FakeDownloader({}))
    service._running = True
    service._run_started = NOW - timedelta(minutes=1)
    assert "stuck" not in service.status_text()
    service.shutdown()


class _PerfClock:
    """A perf_counter the test moves by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class _SlowChase:
    def __init__(self, clock, seconds):
        self.clock, self.seconds = clock, seconds

    def run(self, board, prices, *, now):
        self.clock.now += self.seconds

    def annotate(self, board):
        return dict(board)

    def close(self):
        pass


def _timed_service(chase_seconds=None, board_seconds=0.0):
    """A service whose bar read costs `board_seconds` and chase `chase_seconds` of fake time."""
    clock = _PerfClock()

    class SlowBot(FakeBot):
        def get_scan_symbol_set(self):
            clock.now += board_seconds
            return super().get_scan_symbol_set()

    bot = SlowBot([], {"SPY": _naive_la_bars([400.0] * 14)})
    service = svc.MoversService(
        bot_provider=lambda: bot,
        downloader=FakeDownloader({}),
        universe_provider=lambda: [],
        clock=lambda: NOW,
        autostart=False,
        options_chase=_SlowChase(clock, chase_seconds) if chase_seconds is not None else None,
        push_sender=lambda *_a: {},
        mode_provider=lambda: "OFF",
        scanner=lambda: {},
    )
    service._perf_clock = clock
    return service, clock


def test_a_tick_records_its_wall_time_and_the_chase_share():
    service, _clock = _timed_service(chase_seconds=1.1, board_seconds=3.1)
    assert service.last_tick_s is None
    service._worker({"long": [], "short": []})
    assert service.last_chase_s == pytest.approx(1.1)
    assert service.last_tick_s == pytest.approx(4.2)
    assert "tick 4.2 s (chase 1.1 s)" in service.status_text()


def test_a_tick_without_the_chase_shows_the_tick_only():
    service, _clock = _timed_service(chase_seconds=None, board_seconds=0.5)
    service._worker({"long": [], "short": []})
    assert service.last_chase_s is None
    assert "tick 0.5 s" in service.status_text()
    assert "(chase" not in service.status_text()


def test_a_slow_tick_warns_once_per_ten_minutes(caplog):
    service, clock = _timed_service(chase_seconds=11.0)
    with caplog.at_level("WARNING"):
        service._run_once({"long": [], "short": []})
        service._run_once({"long": [], "short": []})  # 11 s later: rate-limited
        clock.now += svc.SLOW_TICK_WARN_SECONDS
        service._run_once({"long": [], "short": []})
    slow = [r for r in caplog.records if "Movers tick took" in r.getMessage()]
    assert len(slow) == 2
    assert "11.0 s" in slow[0].getMessage()


def test_a_fast_tick_does_not_warn(caplog):
    service, _clock = _timed_service(chase_seconds=2.0)
    with caplog.at_level("WARNING"):
        service._run_once({"long": [], "short": []})
    assert not [r for r in caplog.records if "Movers tick took" in r.getMessage()]
