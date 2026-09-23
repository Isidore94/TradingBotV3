"""An arm click shows QUEUED at once; the slow half runs on one ordered worker.

Trader, 2026-09-23: arming may take 3-10 s to really arm, but the GUI must
show at once that it is queued, the arm must finish even if they move to the
next chart, and the button/row then says ARMED, or FAILED with a reason.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

CLICK_BUDGET_S = 0.05
RPC_DELAY_S = 2.0
NOON = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)


def _bar(minute, high, low):
    mid = (high + low) / 2
    return {
        "dt": NOON.replace(hour=11, minute=minute),
        "open": mid,
        "high": high,
        "low": low,
        "close": mid,
        "volume": 1000.0,
    }


BARS = [_bar(20, 110.0, 99.0), _bar(25, 108.0, 100.0)]


class _SlowProxy:
    is_process_proxy = True

    def __init__(self, delay=RPC_DELAY_S):
        self.delay = delay
        self.calls: list[threading.Thread] = []
        self.d1_zone_arms = {}

    def m5_chart_bars(self, symbol, max_sessions=2):
        self.calls.append(threading.current_thread())
        time.sleep(self.delay)
        return list(BARS)

    def get_market_environment(self):
        time.sleep(self.delay)
        return ""


class _Service:
    def __init__(self, bot):
        self._bot = bot

    def current_bot(self):
        return self._bot


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    from ui.services import m5_bar_cache
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_a, **_k: None)
    m5_bar_cache.reset_shared_m5_cache()
    yield
    m5_bar_cache.reset_shared_m5_cache()


@pytest.fixture()
def events(monkeypatch):
    """Every review row the panel writes, with the thread that wrote it."""
    from ui.panels import alert_center_panel as panel_module

    rows: list[tuple[str, dict, threading.Thread]] = []

    def record(action, **kwargs):
        rows.append((action, kwargs, threading.current_thread()))

    monkeypatch.setattr(panel_module, "record_review_event", record)
    return rows


def _panel(tmp_path, bot):
    from ui.models.bounce import BounceAlert
    from ui.panels.alert_center_panel import AlertCenterPanel

    panel = AlertCenterPanel(
        parked_symbols_path=tmp_path / "parked.json",
        focus_d1_flags_path=tmp_path / "focus_flags.json",
        chart_watches_path=tmp_path / "chart_watches.json",
        d1_level_watches_path=tmp_path / "d1_level_watches.json",
        d1_event_watches_path=tmp_path / "d1_event_watches.json",
        review_events_path=tmp_path / "review_events.jsonl",
    )
    panel._review_movers_only = False
    panel._bounce_service = _Service(bot)
    assert panel.chart_symbol("NVDA", side="LONG")
    assert isinstance(panel._current_review_alert, BounceAlert)
    return panel


def _wait(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    QApplication.processEvents()
    return condition()


def _close(panel):
    panel.shutdown()
    panel.close()
    panel.deleteLater()


def _armed_rows(panel):
    table = panel.armed_list.table
    return [
        " | ".join(table.item(row, col).text() for col in range(6) if table.item(row, col))
        for row in range(table.rowCount())
    ]


def test_an_arm_click_returns_at_once_and_shows_queued(tmp_path, events):
    bot = _SlowProxy()
    panel = _panel(tmp_path, bot)
    statuses: list[str] = []
    panel.statusChanged.connect(statuses.append)
    try:
        symbol = panel._current_review_alert.symbol
        button = panel.chart_review.watch_buttons["new_hod"]
        main = threading.current_thread()

        started = time.perf_counter()
        button.click()
        elapsed = time.perf_counter() - started

        assert elapsed < CLICK_BUDGET_S, f"the click blocked the GUI for {elapsed * 1000:.0f} ms"
        assert main not in bot.calls, "no child RPC on the Qt thread"
        assert button.text().startswith("⏳") and button.isChecked()
        assert any(f"{symbol}: arming" in line for line in statuses)
        assert any("⏳" in row and symbol in row for row in _armed_rows(panel))
        assert panel._chart_watches == [], "nothing is armed until the save lands"
    finally:
        _close(panel)


def test_a_queued_arm_flips_to_armed_with_the_same_records(tmp_path, events):
    from chart_watch import load_chart_watches

    bot = _SlowProxy(delay=0.3)
    panel = _panel(tmp_path, bot)
    try:
        alert = panel._current_review_alert
        panel.chart_review.watch_buttons["new_hod"].click()
        assert _wait(lambda: bool(panel._chart_watches))
        watch = panel._chart_watches[0]
        assert (watch.symbol, watch.kind, watch.baseline) == (alert.symbol, "new_hod", 110.0)
        assert load_chart_watches(tmp_path / "chart_watches.json") == [watch]
        button = panel.chart_review.watch_buttons["new_hod"]
        assert "armed" in button.text() and not button.text().startswith("⏳")
        assert not any("⏳" in row for row in _armed_rows(panel))

        assert panel._arm_queue().wait_idle(5.0)
        arm_rows = [row for row in events if row[0] == "arm_watch"]
        assert len(arm_rows) == 1
        _action, kwargs, thread = arm_rows[0]
        assert thread is not threading.current_thread(), "the review row is written off the Qt thread"
        assert kwargs["symbol"] == alert.symbol and kwargs["alert"] is alert
        assert kwargs["detail"] == {"kind": "new_hod", "baseline": 110.0}
    finally:
        _close(panel)


def test_the_arm_completes_after_the_chart_moves_on(tmp_path, events):
    bot = _SlowProxy(delay=0.3)
    panel = _panel(tmp_path, bot)
    try:
        clicked = panel._current_review_alert
        panel.chart_review.watch_buttons["new_hod"].click()
        panel.chart_symbol("MSFT")
        assert panel._current_review_alert.symbol == "MSFT"
        msft_button = panel.chart_review.watch_buttons["new_hod"]
        assert not msft_button.text().startswith("⏳"), "NVDA's queued arm is not shown on MSFT"

        assert _wait(lambda: bool(panel._chart_watches))
        assert [w.symbol for w in panel._chart_watches] == [clicked.symbol]
        assert panel._arm_queue().wait_idle(5.0)
        _action, kwargs, _thread = next(row for row in events if row[0] == "arm_watch")
        assert kwargs["alert"] is clicked, "the row names the chart that was clicked"
        # The MSFT chart's buttons are not lit by another symbol's arm.
        assert not panel.chart_review.watch_buttons["new_hod"].isChecked()
    finally:
        _close(panel)


def test_a_second_click_while_queued_cancels_the_arm(tmp_path, events):
    from chart_watch import load_chart_watches

    bot = _SlowProxy(delay=0.5)
    panel = _panel(tmp_path, bot)
    statuses: list[str] = []
    panel.statusChanged.connect(statuses.append)
    try:
        button = panel.chart_review.watch_buttons["new_hod"]
        button.click()
        assert button.text().startswith("⏳")
        button.click()
        assert any("cancelled" in line for line in statuses)
        assert not button.isChecked()

        assert panel._arm_queue().wait_idle(5.0)
        _wait(lambda: False, timeout=0.3)
        assert panel._chart_watches == []
        assert load_chart_watches(tmp_path / "chart_watches.json") == []
        assert not [row for row in events if row[0] == "arm_watch"]
        assert not _armed_rows(panel)
    finally:
        _close(panel)


def test_a_failed_save_shows_failed_with_the_reason(tmp_path, events, monkeypatch):
    from ui.panels import alert_center_panel as panel_module

    monkeypatch.setattr(
        panel_module,
        "save_chart_watches",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    bot = _SlowProxy(delay=0.1)
    panel = _panel(tmp_path, bot)
    statuses: list[str] = []
    panel.statusChanged.connect(statuses.append)
    try:
        button = panel.chart_review.watch_buttons["new_hod"]
        button.click()
        assert _wait(lambda: any("FAILED" in line for line in statuses))
        assert panel._chart_watches == []
        assert not button.isChecked()
        failed = [row for row in _armed_rows(panel) if "failed" in row]
        assert failed and "could not be saved" in failed[0]
    finally:
        _close(panel)


def test_d1_event_and_level_clicks_queue_too(tmp_path, events):
    bot = _SlowProxy(delay=0.2)
    panel = _panel(tmp_path, bot)
    try:
        symbol = panel._current_review_alert.symbol
        d1_button = panel.chart_review.arm_bar.d1_event_buttons["new_5d_high"]
        started = time.perf_counter()
        d1_button.click()
        panel._arm_level_from_dock(symbol, "above", 123.45)
        elapsed = time.perf_counter() - started
        assert elapsed < CLICK_BUDGET_S * 2
        assert d1_button.text().startswith("⏳")
        assert sum("⏳" in row for row in _armed_rows(panel)) == 2

        assert _wait(lambda: bool(panel._d1_event_watches) and bool(panel._d1_level_watches))
        assert [w.kind for w in panel._d1_event_watches] == ["new_5d_high"]
        assert [(w.direction, w.level) for w in panel._d1_level_watches] == [("above", 123.45)]
        assert d1_button.text().endswith("✓")
    finally:
        _close(panel)


def test_shutdown_drains_the_queue_and_logs_what_it_could_not_arm(tmp_path, events, caplog):
    from ui.services.arm_queue import ArmQueue

    bot = _SlowProxy(delay=0.3)
    panel = _panel(tmp_path, bot)
    try:
        panel.chart_review.watch_buttons["new_hod"].click()
        panel.shutdown()  # before the event loop ever delivers the result
        assert [w.kind for w in panel._chart_watches] == ["new_hod"], "a finished arm is not lost"
    finally:
        panel.close()
        panel.deleteLater()

    queue = ArmQueue()
    gate = threading.Event()
    queue.submit(("watch", "AAA", "new_hod"), "AAA", "New HOD", gate.wait, lambda _r: (True, ""))
    queue.submit(("watch", "BBB", "new_hod"), "BBB", "New HOD", None, lambda _r: (True, ""))
    with caplog.at_level(logging.ERROR):
        dropped = queue.shutdown(timeout=0.2)
    gate.set()
    assert {job.symbol for job in dropped} == {"AAA", "BBB"}
    assert "ARM DROPPED" in caplog.text and "BBB" in caplog.text


def test_without_a_process_proxy_the_click_arms_inline_as_before(tmp_path, events):
    class _InProcess:
        def m5_chart_bars(self, symbol, max_sessions=2):
            return list(BARS)

    panel = _panel(tmp_path, _InProcess())
    try:
        panel.chart_review.watch_buttons["new_hod"].click()
        assert [w.kind for w in panel._chart_watches] == ["new_hod"]
        assert panel._arm_queue_obj is None
    finally:
        _close(panel)


def test_the_snapshot_popup_queues_its_arms_through_the_panel(tmp_path, events):
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotDialog

    bot = _SlowProxy()
    panel = _panel(tmp_path, bot)
    dialog = SymbolSnapshotDialog()
    try:
        dialog.show_symbol("AMD", bot=bot, side="LONG", watch_host=panel)
        started = time.perf_counter()
        dialog._toggle_watch("new_hod")
        dialog._on_d1_level_alert("AMD", "below", 90.5, "2026-09-22")
        elapsed = time.perf_counter() - started
        assert elapsed < CLICK_BUDGET_S * 2, f"the popup blocked the GUI for {elapsed * 1000:.0f} ms"
        assert threading.current_thread() not in bot.calls
        assert dialog.watch_buttons["new_hod"].text().startswith("⏳")
        assert sum("⏳" in row and "AMD" in row for row in _armed_rows(panel)) == 2

        assert _wait(lambda: bool(panel._chart_watches) and bool(panel._d1_level_watches))
        assert "armed" in dialog.watch_buttons["new_hod"].text()
        assert panel._d1_level_watches[0].candle_date == "2026-09-22"
    finally:
        dialog.close()
        dialog.deleteLater()
        _close(panel)
