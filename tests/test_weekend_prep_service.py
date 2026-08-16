"""R8 §9 step 5 - the weekend routine's one owner, offline.

Injected downloader, frozen clock, temp state file. Nothing here touches a
network, a broker, or the trader's real state file.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

import weekend_strength as ws  # noqa: E402
from ui.services import weekend_prep_service as wps  # noqa: E402

_app = QApplication.instance() or QApplication([])

SERVICE_SOURCE = (SCRIPTS_DIR / "ui" / "services" / "weekend_prep_service.py").read_text(encoding="utf-8")


@pytest.fixture
def service(tmp_path):
    svc = wps.WeekendPrepService(state_path=tmp_path / "weekend_prep_state.json",
                                 now=datetime(2026, 8, 15, 10, 0))
    yield svc
    svc.shutdown()


# ---------------------------------------------------------------------------
# No timer. Structurally, not by promise.
# ---------------------------------------------------------------------------


def test_the_service_owns_no_timer_at_all():
    """Not "a timer that is usually off" - none exists.

    The weekend quiet-hours gate already refuses automatic work on a Saturday,
    and manual buttons are the documented carve-out. A timer here would start
    fetching on the trader's weekend by itself, which is the exact behaviour
    that gate exists to prevent. Parsed rather than grepped so the docstring
    explaining the absence cannot satisfy the check.
    """
    tree = ast.parse(SERVICE_SOURCE)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
    for forbidden in ("QTimer", "singleShot", "startTimer", "timerEvent"):
        assert forbidden not in names, f"{forbidden} has no business in this service"
        assert forbidden not in imported


def test_constructing_the_service_starts_nothing(service, tmp_path):
    assert service.board("h1") is None and service.board("d1") is None
    assert service.week_ahead_markdown == ""
    assert not any(service.is_running(a) for a in ("board:h1:long", "week_ahead"))


# ---------------------------------------------------------------------------
# Weekend identity
# ---------------------------------------------------------------------------


def test_saturday_and_sunday_land_on_the_same_weekend():
    """Or the routine would silently start over halfway through."""
    saturday = wps.weekend_id(datetime(2026, 8, 15, 10, 0))
    sunday = wps.weekend_id(datetime(2026, 8, 16, 20, 0))
    assert saturday == sunday == "2026-08-14"


def test_the_week_runs_monday_to_friday():
    assert wps.week_bounds("2026-08-14") == (date(2026, 8, 10), date(2026, 8, 14))


def test_the_id_is_anchored_to_the_session_calendar_not_the_wall_clock():
    """Tuesday after Labor Day still reviews the week ending that Friday."""
    assert wps.weekend_id(datetime(2026, 9, 8, 9, 0)) == "2026-09-04"


def test_an_always_on_service_rolls_forward_to_the_new_weekend(service):
    assert service.weekend == "2026-08-14"
    service.set_step_status("week_review", "done")

    service._now_provider = lambda: datetime(2026, 8, 22, 10, 0)

    assert service.weekend == "2026-08-21"
    assert service.week_bounds == (date(2026, 8, 17), date(2026, 8, 21))
    assert service.step_status("week_review") == "pending"


# ---------------------------------------------------------------------------
# State: atomic, pruned, forgiving
# ---------------------------------------------------------------------------


def test_progress_survives_closing_the_app_mid_routine(tmp_path):
    path = tmp_path / "state.json"
    first = wps.WeekendPrepService(state_path=path, now=datetime(2026, 8, 15, 10, 0))
    first.set_step_status("week_review", "done")
    first.set_step_status("focus_review", "skipped")
    first.shutdown()

    second = wps.WeekendPrepService(state_path=path, now=datetime(2026, 8, 16, 9, 0))
    try:
        assert second.weekend == first.weekend
        assert second.step_status("week_review") == "done"
        assert second.step_status("focus_review") == "skipped"
        assert second.step_status("discovery") == "pending"
    finally:
        second.shutdown()


def test_the_state_file_is_written_atomically_and_leaves_no_tmp(service, tmp_path):
    service.set_step_status("week_review", "done")
    path = tmp_path / "weekend_prep_state.json"
    assert path.is_file()
    assert list(tmp_path.glob("*.tmp")) == []
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["weekends"]["2026-08-14"]["steps"]["week_review"]["status"] == "done"


def test_the_file_is_pruned_to_eight_weekends(service, tmp_path):
    weekends = service._state.setdefault("weekends", {})
    for index in range(14):
        friday = date(2026, 1, 2) + timedelta(days=7 * index)
        weekends[friday.isoformat()] = wps._empty_weekend()
    service.set_step_status("week_review", "done")
    payload = json.loads((tmp_path / "weekend_prep_state.json").read_text(encoding="utf-8"))
    assert len(payload["weekends"]) == wps.KEEP_WEEKENDS
    # The most recent survive, which includes the one being worked on.
    assert service.weekend in payload["weekends"]


def test_a_corrupt_state_file_loses_progress_not_the_routine(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    svc = wps.WeekendPrepService(state_path=path, now=datetime(2026, 8, 15, 10, 0))
    try:
        assert svc.step_status("week_review") == "pending"
    finally:
        svc.shutdown()


def test_an_older_file_missing_a_step_id_is_filled_not_crashed(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({"version": 1, "weekends": {"2026-08-14": {"steps": {"week_review": {"status": "done", "at": ""}}}}}),
        encoding="utf-8",
    )
    svc = wps.WeekendPrepService(state_path=path, now=datetime(2026, 8, 15, 10, 0))
    try:
        assert svc.step_status("week_review") == "done"
        assert svc.step_status("week_ahead") == "pending"
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def test_a_routine_is_complete_when_every_step_is_done_or_skipped(service):
    assert service.routine_complete is False
    for step in wps.STEP_IDS[:-1]:
        service.set_step_status(step, "done")
    assert service.routine_complete is False
    # Skipping is a decision, not a gap.
    service.set_step_status(wps.STEP_IDS[-1], "skipped")
    assert service.routine_complete is True


def test_an_unknown_step_or_status_is_refused(service):
    with pytest.raises(ValueError, match="unknown weekend prep step"):
        service.set_step_status("make_coffee", "done")
    with pytest.raises(ValueError, match="unknown step status"):
        service.set_step_status("discovery", "probably")


# ---------------------------------------------------------------------------
# Refreshes: manual, single-flight, last-good
# ---------------------------------------------------------------------------


def _bars(count=60, step=timedelta(days=1), drift=0.5):
    out, price = [], 100.0
    stamp = datetime(2026, 1, 1, 10, 0)
    for index in range(count):
        close = price + drift
        out.append({"timestamp": stamp + step * index, "open": price, "close": close,
                    "high": max(price, close) + 0.5, "low": min(price, close) - 0.5, "volume": 1000})
        price = close
    return out


def _fake_downloader(calls=None, fail=False):
    def download(chunk, period=None, interval=None):
        if calls is not None:
            calls.append((tuple(chunk), period, interval))
        if fail:
            raise RuntimeError("provider unavailable")
        import pandas as pd

        frames = {}
        for symbol in chunk:
            rows = _bars()
            frames[symbol] = pd.DataFrame(
                {"Open": [r["open"] for r in rows], "High": [r["high"] for r in rows],
                 "Low": [r["low"] for r in rows], "Close": [r["close"] for r in rows],
                 "Volume": [r["volume"] for r in rows]},
                index=pd.to_datetime([r["timestamp"] for r in rows]),
            )
        # yfinance returns a flat frame for a single symbol and a (symbol, field)
        # MultiIndex for several. The fake mirrors that, because the difference
        # is exactly what the caller branches on.
        if len(chunk) == 1:
            return frames[chunk[0]]
        return pd.concat(frames, axis=1)

    return download


def test_a_board_refresh_uses_the_injected_downloader_and_its_timeframes_periods(service):
    calls: list = []
    service.refresh_board("d1", downloader=_fake_downloader(calls),
                          symbols=["AAA", "BBB"], now=datetime(2026, 4, 1, 12, 0), blocking=True)
    assert calls and calls[0][1] == ws.D1.yf_period and calls[0][2] == ws.D1.yf_interval
    board = service.board("d1")
    assert board is not None and board.offered == 2


def test_zero_ib_traffic_the_service_only_knows_a_downloader():
    """The fetch path is batched yfinance, mirroring the R2 board deliberately."""
    tree = ast.parse(SERVICE_SOURCE)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
    for forbidden in ("ibapi", "ib_insync", "bounce_bot"):
        assert forbidden not in imported


def test_a_second_refresh_while_one_is_running_is_refused(service, monkeypatch):
    service._inflight.add("board:d1")
    assert service.refresh_board("d1", symbols=[], blocking=True) is False


def test_refreshing_one_timeframe_does_not_block_another(service):
    service._inflight.add("board:h1")
    assert service.refresh_board("d1", downloader=_fake_downloader(), symbols=["AAA"],
                                 now=datetime(2026, 4, 1, 12, 0), blocking=True) is True


def test_one_timeframe_fetch_derives_both_sides_and_has_one_race_slot(service):
    calls = []
    service.refresh_board(
        "d1", side="short", downloader=_fake_downloader(calls), symbols=["AAA", "BBB"],
        now=datetime(2026, 4, 1, 12, 0), blocking=True,
    )

    assert len(calls) == 1
    assert service.board("d1", "long") is not None
    assert service.board("d1", "short") is not None
    assert service.board("d1").side == "short"

    service._inflight.add("board:d1")
    assert service.refresh_board("d1", side="long", blocking=True) is False


def test_last_good_boards_survive_a_service_restart(tmp_path):
    path = tmp_path / "state.json"
    first = wps.WeekendPrepService(state_path=path, now=datetime(2026, 8, 15, 10, 0))
    first.refresh_board(
        "d1", side="short", downloader=_fake_downloader(), symbols=["AAA", "BBB"],
        now=datetime(2026, 4, 1, 12, 0), blocking=True,
    )
    first.shutdown()

    second = wps.WeekendPrepService(state_path=path, now=datetime(2026, 8, 16, 10, 0))
    try:
        assert second.board("d1").side == "short"
        assert second.board("d1", "long").offered == 2
        assert second.board("d1", "short").offered == 2
    finally:
        second.shutdown()


def test_a_failed_fetch_keeps_the_last_good_board(service):
    """An empty board after a blip reads as "nothing is strong this week",
    which is a different and much worse claim than "the fetch failed"."""
    service.refresh_board("d1", downloader=_fake_downloader(), symbols=["AAA", "BBB"],
                          now=datetime(2026, 4, 1, 12, 0), blocking=True)
    good = service.board("d1")
    assert good is not None

    messages: list[str] = []
    service.statusChanged.connect(messages.append)
    service.refresh_board("d1", downloader=_fake_downloader(fail=True), symbols=["AAA", "BBB"],
                          now=datetime(2026, 4, 1, 12, 0), blocking=True)
    assert service.board("d1") is good, "the last good board is still there"
    assert any("last good" in m for m in messages)


def test_an_all_empty_provider_response_keeps_the_last_good_board(service):
    import pandas as pd

    service.refresh_board("d1", downloader=_fake_downloader(), symbols=["AAA"],
                          now=datetime(2026, 4, 1, 12, 0), blocking=True)
    good = service.board("d1")
    messages: list[str] = []
    service.statusChanged.connect(messages.append)

    service.refresh_board(
        "d1", downloader=lambda *args, **kwargs: pd.DataFrame(), symbols=["AAA"],
        now=datetime(2026, 4, 1, 12, 0), blocking=True,
    )

    assert service.board("d1") is good
    assert any("last good" in message for message in messages)


def test_a_bad_chunk_costs_one_chunk_not_the_board(service, monkeypatch):
    import autopilot_core as core

    monkeypatch.setattr(core, "AUTOPILOT_OPEN_SCAN_CHUNK_SIZE", 1)
    working = _fake_downloader()
    seen: list = []

    def flaky(chunk, period=None, interval=None):
        seen.append(chunk[0])
        if chunk[0] == "BAD":
            raise RuntimeError("chunk failed")
        return working(chunk, period=period, interval=interval)

    service.refresh_board("d1", downloader=flaky, symbols=["AAA", "BAD", "BBB"],
                          now=datetime(2026, 4, 1, 12, 0), blocking=True)
    board = service.board("d1")
    assert board.offered == 3 and board.measured == 2
    assert "had too little history" in board.accounting or board.measured == 2


def test_an_unknown_timeframe_is_refused(service):
    with pytest.raises(ValueError, match="unknown timeframe"):
        service.refresh_board("w1", blocking=True)


def test_the_week_ahead_runs_only_when_asked_and_keeps_its_last_report(service):
    service.refresh_week_ahead(runner=lambda: "# Week ahead\n\nsomething", blocking=True)
    assert "Week ahead" in service.week_ahead_markdown
    assert service.weekend_state()["week_ahead"]["ran_at"]

    def _boom():
        raise RuntimeError("market_prep unavailable")

    service.refresh_week_ahead(runner=_boom, blocking=True)
    assert "Week ahead" in service.week_ahead_markdown, "the last report survives"


def test_week_ahead_accepts_the_orchestrators_real_return_shape(monkeypatch):
    from market_prep.orchestrator import MarketPrepOrchestrator

    real_shape = {
        "weekly_report": {"markdown": "# Nested copy"},
        "report": "# Week ahead\n\nProduction-shaped report",
    }
    monkeypatch.setattr(MarketPrepOrchestrator, "run_weekly_prep", lambda self: real_shape)

    assert wps._run_weekly_prep() == real_shape["report"]


def test_adoptions_and_tag_reviews_are_recorded_in_state(service):
    service.record_adopted("AAPL", "long", "d1")
    service.record_tag_review("trade-1")
    service.record_tag_review("trade-2", corrected_to="avwap-reclaim")
    entry = service.weekend_state()
    assert entry["adopted"][0]["symbol"] == "AAPL"
    assert entry["tag_review"]["confirmed"] == ["trade-1"]
    assert entry["tag_review"]["corrected"] == {"trade-2": "avwap-reclaim"}
