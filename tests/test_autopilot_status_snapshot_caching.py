"""Desk snappiness packet 1 item 3: the Auto Pilot status row stops re-reading.

`status_snapshot()` ran on the GUI thread from a 5 s panel timer with no
visibility check, plus twice back-to-back in the 30 s tick, doing 2 watchlist
reads + 2 auto-watchlist reads + 2 JSON parses per call - most of the 10
minutes the 2026-08-31 stall log charged to `watchlist_utils.py:33` and 3.9
minutes to `project_paths.py:165`. The fixes: memoize each file-backed piece
on `(st_mtime_ns, st_size)`, compute the tick's snapshot once, skip the hidden
panel's 5 s poll, and only restyle a status label when its text/tone moved.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _fresh_memo():
    from ui.services import autopilot_service as mod

    # getattr so a pre-fix checkout runs the behavioral assertions instead of
    # erroring in the fixture.
    memo = getattr(mod, "_status_file_memo", None)
    if memo is not None:
        memo.clear()
    yield
    if memo is not None:
        memo.clear()


class _ReadCounter:
    """Count `Path.read_text` calls per file name, pass everything through."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.counts: dict[str, int] = {}
        original = Path.read_text
        counter = self

        def counting_read_text(path_self, *args, **kwargs):
            counter.counts[path_self.name] = counter.counts.get(path_self.name, 0) + 1
            return original(path_self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", counting_read_text)


def _bump_mtime(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


# ---------------------------------------------------------------------------
# the file memo itself
# ---------------------------------------------------------------------------


def test_an_unchanged_watchlist_is_read_once(tmp_path, monkeypatch):
    from ui.services.autopilot_service import AutopilotService

    watchlist = tmp_path / "auto_longs.txt"
    watchlist.write_text("AAPL\nMSFT\n", encoding="utf-8")
    counter = _ReadCounter(monkeypatch)

    first = AutopilotService._read_auto_watchlist(watchlist)
    second = AutopilotService._read_auto_watchlist(watchlist)

    assert first == ["AAPL", "MSFT"]
    assert second == first
    assert counter.counts.get(watchlist.name, 0) == 1


def test_a_changed_watchlist_invalidates_the_memo(tmp_path, monkeypatch):
    from ui.services.autopilot_service import AutopilotService

    watchlist = tmp_path / "auto_longs.txt"
    watchlist.write_text("AAPL\n", encoding="utf-8")
    counter = _ReadCounter(monkeypatch)

    assert AutopilotService._read_auto_watchlist(watchlist) == ["AAPL"]
    watchlist.write_text("AAPL\nNVDA\n", encoding="utf-8")
    _bump_mtime(watchlist)

    assert AutopilotService._read_auto_watchlist(watchlist) == ["AAPL", "NVDA"]
    assert counter.counts.get(watchlist.name, 0) == 2


# ---------------------------------------------------------------------------
# status_snapshot: unchanged files, zero reads
# ---------------------------------------------------------------------------


def _snapshot_service(monkeypatch, tmp_path):
    from ui.services import autopilot_service as mod

    for name, filename, text in (
        ("LONGS_FILE", "longs.txt", "AAPL\n"),
        ("SHORTS_FILE", "shorts.txt", "TSLA\n"),
        ("AUTO_LONGS_FILE", "auto_longs.txt", "MSFT\n"),
        ("AUTO_SHORTS_FILE", "auto_shorts.txt", "AMD\n"),
        ("INDUSTRY_BOARD_STATE_FILE", "industry_state.json", "{}"),
        ("INDUSTRY_INTRADAY_RS_STATE_FILE", "industry_rs_state.json", "{}"),
    ):
        target = tmp_path / filename
        target.write_text(text, encoding="utf-8")
        monkeypatch.setattr(mod, name, target)

    service = mod.AutopilotService.__new__(mod.AutopilotService)
    service._enabled = False
    service._profile = "desk"
    service._state = {}
    service._active_scan_slot = None
    service._waiting_scan_slot = None
    service._scan_service = type("S", (), {"running": False})()
    service._universe_rebuild_running = False
    service._wrapup_running = False
    service._swing_slots = lambda now: []  # type: ignore[method-assign]
    service._ib_status_text = lambda: "connected"  # type: ignore[method-assign]
    service._regime_text = lambda: "unknown"  # type: ignore[method-assign]
    service._universe_line = lambda now=None: "Universe: fresh"  # type: ignore[method-assign]
    return service


def test_a_second_snapshot_over_unchanged_files_does_zero_reads(tmp_path, monkeypatch):
    service = _snapshot_service(monkeypatch, tmp_path)
    tracked = {
        "longs.txt",
        "shorts.txt",
        "auto_longs.txt",
        "auto_shorts.txt",
        "industry_state.json",
        "industry_rs_state.json",
    }

    first = service.status_snapshot()
    counter = _ReadCounter(monkeypatch)
    second = service.status_snapshot()

    assert not tracked & set(counter.counts), (
        f"the second snapshot re-read {tracked & set(counter.counts)}"
    )
    for key in ("longs_count", "shorts_count", "auto_longs_count", "auto_shorts_count", "industry_line"):
        assert second[key] == first[key], "caching must not change the snapshot"
    assert second["longs_count"] == 1
    assert second["auto_shorts_count"] == 1


def test_a_snapshot_after_a_watchlist_change_sees_the_change(tmp_path, monkeypatch):
    service = _snapshot_service(monkeypatch, tmp_path)
    service.status_snapshot()

    longs = tmp_path / "longs.txt"
    longs.write_text("AAPL\nNVDA\nMETA\n", encoding="utf-8")
    _bump_mtime(longs)

    assert service.status_snapshot()["longs_count"] == 3


# ---------------------------------------------------------------------------
# the tick computes its snapshot once
# ---------------------------------------------------------------------------


def test_the_tick_calls_status_snapshot_exactly_once(monkeypatch, tmp_path):
    _qapp()
    from ui.services import autopilot_service as mod

    service = _snapshot_service(monkeypatch, tmp_path)
    service._enabled = True
    service._last_report_write = None

    for name in (
        "_roll_day_state", "_apply_scan_window", "_apply_quiet_hours",
        "_maybe_auto_arm", "_maybe_clear_stale_auto_lists",
        "_maybe_add_near_extreme_names", "_maybe_score_picks_daily",
        "_ensure_bot_running", "_ensure_universe_fresh",
        "_maybe_build_watchlists", "_maybe_run_swing_slot",
        "_maybe_run_wrapup", "_maybe_run_evening_prep",
        "_maybe_hourly_away_report", "_maybe_push_d1_events",
        "_maybe_push_spy_alarm",
    ):
        setattr(service, name, lambda *a, **k: None)
    monkeypatch.setattr(mod.core, "write_heartbeat", lambda **_k: None)
    # A weekday: the tick short-circuits on weekends before the heartbeat.
    thursday = datetime(2026, 8, 27, 10, 0)
    monkeypatch.setattr(
        "ui.services.autopilot_service.datetime",
        type("D", (datetime,), {"now": staticmethod(lambda tz=None: thursday)}),
    )

    calls: list[int] = []
    original = service.status_snapshot

    def counting_snapshot():
        calls.append(1)
        return original()

    service.status_snapshot = counting_snapshot  # type: ignore[method-assign]
    # statusChanged.emit needs a real QObject half; this instance was built
    # with __new__, so stub the signal at CLASS level (monkeypatch restores
    # the real Signal afterwards) rather than touch the C++ side.
    monkeypatch.setattr(
        mod.AutopilotService,
        "statusChanged",
        type("Sig", (), {"emit": staticmethod(lambda *_a: None)})(),
    )

    service._tick()

    assert len(calls) == 1, f"the tick took {len(calls)} snapshots; one is enough"


# ---------------------------------------------------------------------------
# the hidden panel and the restyle guard
# ---------------------------------------------------------------------------


def _snapshot_payload(**overrides) -> dict:
    payload = {
        "ib_status": "connected",
        "regime": "trending",
        "next_slot": "10:30",
        "slots_done": [],
        "scan_running": False,
        "watchlist_built_at": "07:31",
        "longs_count": 1,
        "shorts_count": 1,
        "auto_longs_count": 0,
        "auto_shorts_count": 0,
        "universe_line": "Universe: fresh (built 2026-08-31 06:45)",
        "industry_line": "Industry Board: ok",
        "wrapup_running": False,
        "wrapup_done_at": "",
        "report_path": "C:/somewhere/autopilot_today.txt",
        "report_last_attempt": "",
        "report_last_verified": "",
        "report_error": "",
    }
    payload.update(overrides)
    return payload


def _panel(monkeypatch):
    from ui.panels.autopilot_panel import AutopilotPanel
    from ui.services.autopilot_service import AutopilotService

    monkeypatch.setattr(AutopilotService, "_load_state", lambda self: {"enabled": False})
    monkeypatch.setattr(AutopilotService, "_save_state", lambda self: None)
    monkeypatch.setattr("job_ledger.get_default_ledger", lambda: None)
    return AutopilotPanel(bounce_service=None)


def test_the_hidden_panel_refresh_does_no_work(monkeypatch):
    _qapp()
    panel = _panel(monkeypatch)
    try:
        calls: list[int] = []
        panel.service.status_snapshot = lambda: (calls.append(1), _snapshot_payload())[1]  # type: ignore[method-assign]

        assert not panel.isVisible()
        panel._refresh_status()
        assert calls == [], "a hidden panel must not poll status_snapshot"

        panel.show()
        panel._refresh_status()
        assert len(calls) == 1, "a visible panel polls exactly once per tick"
        assert panel._refresh_timer.isActive(), "the timer keeps running while hidden"
    finally:
        panel.shutdown()
        panel.deleteLater()
        _qapp().processEvents()


def test_status_labels_restyle_only_on_change(monkeypatch):
    _qapp()
    panel = _panel(monkeypatch)
    try:
        restyles: list[str] = []
        for name in ("ib_value", "universe_value", "industry_value", "report_value"):
            widget = getattr(panel, name)
            original = widget.setStyleSheet

            def counting(style, _original=original, _name=name):
                restyles.append(_name)
                return _original(style)

            widget.setStyleSheet = counting  # type: ignore[method-assign]

        panel._apply_status(_snapshot_payload())
        after_first = len(restyles)
        panel._apply_status(_snapshot_payload())
        assert len(restyles) == after_first, "an unchanged status must not restyle"

        panel._apply_status(_snapshot_payload(ib_status="DISCONNECTED - waiting to reconnect"))
        assert restyles[after_first:] == ["ib_value"], "only the changed label restyles"
    finally:
        panel.shutdown()
        panel.deleteLater()
        _qapp().processEvents()
