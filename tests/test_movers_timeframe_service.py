"""MoversTimeframeService: 12:00 ET M30 once a day, 16:15 ET Daily, restart catch-up,
the night window, last good board on a failed scan (injected clock and downloader)."""

from __future__ import annotations

import json
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

from ui.services import movers_timeframe_service as svc  # noqa: E402

NY = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")


def ny(day, hour, minute=0):
    return datetime(2026, 9, day, hour, minute, tzinfo=NY)


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ------------------------------------------------------------------ schedule (pure)
def test_night_window_is_22_to_06_pacific():
    assert svc.in_night_window(datetime(2026, 9, 22, 22, 0, tzinfo=PT))
    assert svc.in_night_window(datetime(2026, 9, 23, 5, 59, tzinfo=PT))
    assert not svc.in_night_window(datetime(2026, 9, 23, 6, 0, tzinfo=PT))
    assert not svc.in_night_window(ny(22, 12))


def test_m30_is_due_once_per_trading_day_from_12_new_york():
    assert not svc.m30_due(ny(22, 11, 59), None)
    assert svc.m30_due(ny(22, 12, 0), None)
    assert svc.m30_due(ny(22, 12, 0), "2026-09-21")
    assert not svc.m30_due(ny(22, 12, 0), "2026-09-22")  # done today
    assert svc.m30_due(ny(22, 15, 0), None)  # the desk started late: run once
    assert not svc.m30_due(ny(26, 12, 0), None)  # Saturday
    assert svc.m30_expected_session(ny(22, 11)).isoformat() == "2026-09-21"
    assert svc.m30_expected_session(ny(22, 12)).isoformat() == "2026-09-22"


def test_d1_is_due_after_16_15_and_catches_up_outside_market_hours():
    assert not svc.d1_due(ny(22, 16, 14), "2026-09-21")
    assert svc.d1_due(ny(22, 16, 15), "2026-09-21")
    assert not svc.d1_due(ny(22, 16, 30), "2026-09-22")
    # A two-session-old board: not in market hours, not in the night window.
    assert not svc.d1_due(ny(22, 10, 0), "2026-09-18")
    assert not svc.d1_due(ny(22, 8, 59), "2026-09-18")  # 05:59 PT: night window
    assert svc.d1_due(ny(22, 9, 0), "2026-09-18")  # 06:00 PT: catch up
    assert svc.d1_due(ny(26, 10, 0), None)  # Saturday: Friday's board is owed
    assert svc.d1_target_session(ny(26, 10)).isoformat() == "2026-09-25"
    # 21:00 PT is before the night window; 22:30 PT is inside it.
    assert svc.d1_due(datetime(2026, 9, 22, 21, 0, tzinfo=PT), "2026-09-21")
    assert not svc.d1_due(datetime(2026, 9, 22, 22, 30, tzinfo=PT), "2026-09-21")


def test_next_check_lands_on_the_scan_times_and_never_sleeps_past_30_min():
    assert svc.next_check_ms(ny(22, 11, 50)) == pytest.approx(10 * 60_000 + 1000)
    assert svc.next_check_ms(ny(22, 16, 10)) == pytest.approx(5 * 60_000 + 1000)
    assert svc.next_check_ms(ny(22, 13, 0)) == svc.MAX_SLEEP_MS
    assert svc.next_check_ms(datetime(2026, 9, 23, 5, 55, tzinfo=PT)) == 5 * 60_000 + 1000


# ------------------------------------------------------------------ service
def _sessions(count, end_day=22):
    out, cursor = [], datetime(2026, 9, end_day)
    while len(out) < count:
        if cursor.weekday() < 5:
            out.append(cursor.date())
        cursor -= timedelta(days=1)
    return sorted(out)


def _m30_frame(today_closes, *, price):
    rows, index = [], []
    for day in _sessions(21)[:-1]:
        for i in range(13):
            index.append(pd.Timestamp(datetime(day.year, day.month, day.day, 9, 30, tzinfo=NY)
                                      + timedelta(minutes=30 * i)))
            rows.append({"Open": price, "High": price + 0.5, "Low": price - 0.5,
                         "Close": price, "Volume": 20_000.0})
    prev = price
    for i, close in enumerate(today_closes):
        index.append(pd.Timestamp(ny(22, 9, 30) + timedelta(minutes=30 * i)))
        rows.append({"Open": prev, "High": max(prev, close) + 0.5, "Low": min(prev, close) - 0.5,
                     "Close": close, "Volume": 40_000.0})
        prev = close
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


def _daily_frame(closes, end_day=22):
    days = _sessions(len(closes), end_day)
    rows = [{"Open": c, "High": c + 0.5, "Low": c - 0.5, "Close": c, "Volume": 2_000_000.0}
            for c in closes]
    return pd.DataFrame(rows, index=pd.DatetimeIndex([pd.Timestamp(d) for d in days]))


class FakeDownloader:
    def __init__(self, m30=None, daily=None, fail=False):
        self.m30, self.daily, self.fail = m30 or {}, daily or {}, fail
        self.calls: list[tuple[tuple[str, ...], str, str]] = []

    def __call__(self, symbols, *, period, interval):
        self.calls.append((tuple(symbols), period, interval))
        if self.fail:
            raise RuntimeError("yahoo down")
        frames = self.daily if interval == "1d" else self.m30
        if len(symbols) == 1:
            return frames.get(symbols[0], pd.DataFrame())
        return {s: frames.get(s, pd.DataFrame()) for s in symbols}


class Clock:
    def __init__(self, moment):
        self.moment = moment

    def __call__(self):
        return self.moment


def _service(tmp_path, downloader, clock, **kwargs):
    return svc.MoversTimeframeService(
        downloader=downloader, clock=clock, autostart=False,
        universe_provider=lambda: ["AAA"],
        picks_path=tmp_path / "picks.jsonl",
        board_paths={"m30": tmp_path / "m30.json", "d1": tmp_path / "d1.json"},
        **kwargs,
    )


def _m30_downloader():
    return FakeDownloader(
        m30={"SPY": _m30_frame([400.0] * 5, price=400.0),
             "AAA": _m30_frame([100.0, 100.0, 100.5, 101.0, 101.5], price=100.0)},
        daily={"SPY": _daily_frame([400.0] * 210, end_day=21),
               "AAA": _daily_frame([50.0] * 210, end_day=21)},
    )


def test_m30_scan_emits_saves_logs_and_runs_once_a_day(app, tmp_path):
    clock = Clock(ny(22, 12, 5))
    downloader = _m30_downloader()
    service = _service(tmp_path, downloader, clock, focus_provider=lambda: {"m5": {"long": ["FOC"]}},
                       m5_board_provider=lambda: {"pop": {"long": [{"symbol": "HOT"}]}})
    emitted = []
    service.timeframeBoardChanged.connect(lambda tf, board: emitted.append((tf, board)))
    assert service.due() == ["m30"]  # the owed Daily waits for the close (market hours)
    service._worker(["m30"], service._snapshot())
    tf, board = emitted[-1]
    assert tf == "m30" and board["session"] == "2026-09-22" and board["stale"] is False
    assert board["as_of"].startswith("2026-09-22T11:30")  # measured as at 12:00
    assert [r["symbol"] for r in board["pop"]["long"]] == ["AAA"]
    assert board["summaries"]["pop"]["long"]["picks"] == 1
    # Universe: liquid + Focus + today's M5 names; 30m over 1mo; daily for the SMAs.
    m30_call = next(c for c in downloader.calls if c[2] == "30m")
    assert {"SPY", "AAA", "FOC", "HOT"} <= set(m30_call[0]) and m30_call[1] == "1mo"
    assert any(c[2] == "1d" and c[1] == "1y" for c in downloader.calls)
    saved = json.loads((tmp_path / "m30.json").read_text(encoding="utf-8"))
    assert saved["session"] == "2026-09-22"
    picks = [json.loads(line) for line in (tmp_path / "picks.jsonl").read_text().splitlines()]
    assert [(p["tf"], p["box"], p["symbol"]) for p in picks] == [
        ("m30", "pop", "AAA"), ("m30", "dip_strong", "AAA")]
    assert "m30" not in service.due()
    # A later run the same day (a manual refresh) logs nothing twice.
    service._worker(["m30"], service._snapshot())
    assert len((tmp_path / "picks.jsonl").read_text().splitlines()) == 2


def test_restart_shows_saved_boards_and_catches_up_only_what_is_owed(app, tmp_path):
    (tmp_path / "m30.json").write_text(json.dumps({"tf": "m30", "session": "2026-09-22",
                                                   "pop": {}, "swing": {}}), encoding="utf-8")
    (tmp_path / "d1.json").write_text(json.dumps({"tf": "d1", "session": "2026-09-18",
                                                  "pop": {}, "swing": {}}), encoding="utf-8")
    clock = Clock(ny(22, 14, 0))
    service = _service(tmp_path, FakeDownloader(), clock)
    boards = service.boards()
    assert boards["m30"]["stale"] is False and boards["d1"]["stale"] is True
    assert service.due() == []  # market hours: the Daily catch-up waits
    clock.moment = ny(22, 16, 20)
    assert service.due() == ["d1"]
    # A wrong-timeframe or broken file loads as nothing.
    (tmp_path / "d1.json").write_text("{not json", encoding="utf-8")
    assert svc.load_board(tmp_path / "d1.json", "d1") == {}
    assert svc.load_board(tmp_path / "m30.json", "d1") == {}


def test_failed_scan_keeps_and_reshows_the_last_good_board(app, tmp_path):
    clock = Clock(ny(22, 12, 5))
    service = _service(tmp_path, _m30_downloader(), clock)
    service._worker(["m30"], service._snapshot())
    good = service.boards()["m30"]
    service._downloader = FakeDownloader(fail=True)
    emitted, status = [], []
    service.timeframeBoardChanged.connect(lambda tf, board: emitted.append((tf, board)))
    service.statusChanged.connect(status.append)
    service._worker(["m30"], service._snapshot())
    tf, board = emitted[-1]
    assert tf == "m30" and board["pop"] == good["pop"]
    assert "no SPY bars" in board["last_error"]
    assert "M30 scan FAILED" in status[-1]
    assert json.loads((tmp_path / "m30.json").read_text())["session"] == "2026-09-22"


def test_d1_scan_resolves_earlier_picks_and_one_worker_at_a_time(app, tmp_path):
    closes = [100.0] * 205 + [101.0, 102.0, 103.0, 104.0, 105.0]
    pick = {"kind": "pick", "tf": "m30", "box": "pop", "side": "long", "symbol": "AAA",
            "rank": 1, "score": 2.0, "entry_close": 103.0, "spy_entry": 400.0,
            "session": "2026-09-18"}
    (tmp_path / "picks.jsonl").write_text(json.dumps(pick) + "\n", encoding="utf-8")
    downloader = FakeDownloader(daily={"SPY": _daily_frame([400.0] * 210),
                                       "AAA": _daily_frame(closes)})
    service = _service(tmp_path, downloader, Clock(ny(22, 16, 20)))
    service._worker(["d1"], service._snapshot())
    rows = [json.loads(line) for line in (tmp_path / "picks.jsonl").read_text().splitlines()]
    outcomes = [r for r in rows if r["kind"] == "outcome"]
    # Picked Friday 9/18: +1 = Monday 9/21 has closed; +3 has not.
    assert [(r["horizon"], r["target_session"]) for r in outcomes] == [(1, "2026-09-21")]
    assert outcomes[0]["excess_pct"] == pytest.approx((104.0 / 103.0 - 1) * 100)
    board = service.boards()["d1"]
    assert board["session"] == "2026-09-22"
    assert board["summaries"]["pop"]["long"]["tf"] == "d1"
    service._running = True
    assert service.refresh_now() is False


def _d1_downloader():
    closes = [100.0] * 200 + [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                              110.0]
    return FakeDownloader(daily={"SPY": _daily_frame([400.0] * 210),
                                 "AAA": _daily_frame(closes)})


def test_d1_since_rebuilds_the_daily_board_from_cache_without_logging_picks(app, tmp_path):
    # Trader 2026-09-30: "Daily can just be raw strength and weakness maybe let me pick a date?"
    from datetime import date

    downloader = _d1_downloader()
    service = _service(tmp_path, downloader, Clock(ny(22, 16, 20)))
    service._worker(["d1"], service._snapshot())
    board = service.boards()["d1"]
    assert board["swing_anchor"]["long"]["date"] == "2026-08-25"  # 20 sessions back
    assert [r["symbol"] for r in board["swing"]["long"]] == ["AAA"]
    picks_before = (tmp_path / "picks.jsonl").read_text()
    calls_before = len(downloader.calls)
    emitted = []
    service.timeframeBoardChanged.connect(lambda tf, b: emitted.append((tf, b)))
    started = []
    service._spawn = lambda target: (started.append(target), target())
    service.set_d1_since(date(2026, 9, 19))  # a Saturday: Monday 9/21's close
    assert len(started) == 1 and not service.running
    tf, rebuilt = emitted[-1]
    assert tf == "d1" and rebuilt["swing_anchor"]["long"]["date"] == "2026-09-21"
    assert rebuilt["swing_anchor"]["short"] == rebuilt["swing_anchor"]["long"]
    assert rebuilt["swing"]["long"][0]["since_start_pct"] == pytest.approx((110 / 109 - 1) * 100)
    assert len(downloader.calls) == calls_before  # from the cached bars, no download
    assert (tmp_path / "picks.jsonl").read_text() == picks_before  # no pick rows
    saved = json.loads((tmp_path / "d1.json").read_text(encoding="utf-8"))
    assert saved["swing_anchor"]["long"]["date"] == "2026-09-21"
    # The next normal Daily scan uses the stored date too.
    service._worker(["d1"], service._snapshot())
    assert service.boards()["d1"]["swing_anchor"]["long"]["date"] == "2026-09-21"


def test_d1_since_waits_for_a_running_worker_and_scans_without_a_cache(app, tmp_path):
    from datetime import date

    service = _service(tmp_path, _d1_downloader(), Clock(ny(22, 16, 20)))
    started = []
    service._spawn = lambda target: started.append(target)
    service._running = True
    service.set_d1_since(date(2026, 9, 21))
    assert started == []  # one worker at a time: the date waits
    service._running = False
    service._after_worker()
    # No cached daily bars for 9/22: a normal Daily scan runs.
    assert len(started) == 1 and service.running
    service._running = False
    service._after_worker()
    assert len(started) == 1  # the pending date was applied once
    # Restoring the saved date on start stores it without a scan.
    other = _service(tmp_path, _d1_downloader(), Clock(ny(22, 16, 20)))
    other._spawn = lambda target: started.append(target)
    other.set_d1_since(date(2026, 9, 2), rebuild=False)
    assert other.d1_since == date(2026, 9, 2) and len(started) == 1


def test_a_missing_daily_bar_retries_at_most_3_times_per_session(app, tmp_path):
    # Yahoo still lacks 9/22's daily bar at 16:15: the board stays on 9/21.
    downloader = FakeDownloader(daily={"SPY": _daily_frame([400.0] * 210, end_day=21),
                                       "AAA": _daily_frame([100.0] * 210, end_day=21)})
    clock = Clock(ny(22, 16, 20))
    service = _service(tmp_path, downloader, clock)
    status = []
    service.statusChanged.connect(status.append)
    for attempt in range(svc.MAX_TRIES_PER_SESSION):
        assert "d1" in service.due(), attempt
        service._worker(["d1"], service._snapshot())
        clock.moment += timedelta(minutes=30)
    assert service.boards()["d1"]["session"] == "2026-09-21"  # last good board kept
    assert "d1" not in service.due()
    assert "Daily: no 2026-09-22 bars after 3 tries" in status[-1]
    # The next session's target starts a fresh count.
    clock.moment = ny(23, 16, 20)
    assert "d1" in service.due()
