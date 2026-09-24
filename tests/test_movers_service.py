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
