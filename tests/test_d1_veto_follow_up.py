"""A saved veto reason arms its follow-up alert (trader's word 2026-09-24).

`too_extended_from_base` -> Pullback (unchanged, pinned by
tests/test_extended_veto_pullback.py); `incoming_trendline` ->
trendline_break_retest on the side's incoming scan line; `sma_incoming` ->
sma_break_retest; `compressed` -> range_breakout; anything else arms nothing.
The veto retires normally either way. Also: the snapshot popup's D1 alerts
use the same grouped menu as the arm bar.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
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

KNOWN_AT = datetime(2026, 9, 24, 6, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _scan(symbol: str, side: str = "LONG"):
    from ui.models.bounce import BounceAlert

    return BounceAlert(
        time_text="10:05:00",
        symbol=symbol,
        side=side,
        trigger=f"({side.lower()}) zone1 bounce off AVWAPE",
        timeframe="D1",
        tag=f"d1_flag_{side.lower()}",
        raw_text=f"MASTER_AVWAP_D1_ZONE: {symbol} ({side.lower()}) zone1 bounce",
        is_d1=True,
    )


def _panel(tmp_path, monkeypatch):
    import pick_feedback
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *_a, **_k: None)
    pick_feedback.clear_reviewed_today_cache()
    panel = AlertCenterPanel(
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        chart_watches_path=tmp_path / "chart_watches.json",
        d1_event_watches_path=tmp_path / "d1_event_watches.json",
        review_events_path=tmp_path / "review_events.jsonl",
    )
    monkeypatch.setattr(panel, "_auto_mode_now", lambda: "DESK")
    monkeypatch.setattr(panel, "_alerts_may_sound", lambda: False)
    monkeypatch.setattr(panel, "_review_movers_only", False, raising=False)
    monkeypatch.setattr(panel.chart_review, "_reviewed_symbols", lambda: set())
    rail = panel.chart_review.capture_rail
    monkeypatch.setattr(rail, "_annotations_path", tmp_path / "annotations.jsonl")
    rail._merge_veto_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_like_cohort = lambda **_kwargs: {"written": True, "added": 0}
    rail._merge_pass_cohort = lambda **_kwargs: {"written": True, "added": 0}
    return panel


def _queue(panel, symbol="AAPL", side="LONG"):
    from PySide6.QtWidgets import QPushButton

    panel.add_alert(_scan(symbol, side))
    panel.add_alert(_scan("NVDA", "SHORT"))
    buttons = [b for b in panel.findChildren(QPushButton) if "show all" in b.text().casefold()]
    if len(buttons) == 1:
        buttons[0].click()
        QApplication.processEvents()
    assert panel._current_review_alert.symbol == symbol


def _veto(panel, code):
    from ui.widgets.capture_rail import _REASON_ROLE

    rail = panel.chart_review.capture_rail
    for row in range(rail.reason_list.count()):
        item = rail.reason_list.item(row)
        if item.data(_REASON_ROLE) == code:
            rail.reason_list.setCurrentItem(item)
            rail.reason_list.itemActivated.emit(item)
            QApplication.processEvents()
            return
    raise AssertionError(f"no {code!r} veto reason loaded")


def _status(panel) -> str:
    return panel.chart_review.capture_rail.status_label.text().casefold()


def _d1_kinds(panel):
    return [(w.symbol, w.kind, w.side) for w in panel._d1_event_watches]


def _run(tmp_path, monkeypatch, code, *, side="LONG", setup=None):
    panel = _panel(tmp_path, monkeypatch)
    if setup is not None:
        setup(panel)
    _queue(panel, "AAPL", side)
    _veto(panel, code)
    return panel


def test_the_mapping_table():
    from ui.panels.alert_center_panel import AlertCenterPanel

    assert AlertCenterPanel.VETO_FOLLOW_UPS == {
        "too_extended_from_base": "pullback",
        "incoming_trendline": "trendline_break_retest",
        "sma_incoming": "sma_break_retest",
        "compressed": "range_breakout",
    }


def test_sma_incoming_arms_an_sma_break_retest_on_the_side_and_retires(tmp_path, monkeypatch):
    panel = _run(tmp_path, monkeypatch, "sma_incoming", side="SHORT")
    try:
        assert _d1_kinds(panel) == [("AAPL", "sma_break_retest", "SHORT")]
        assert panel._current_review_alert.symbol == "NVDA", "the veto still retires"
        stored = json.loads((tmp_path / "d1_event_watches.json").read_text(encoding="utf-8"))
        assert stored["watches"][0]["side"] == "SHORT"
        # A second veto of the same chart never duplicates it.
        panel.chart_alert(_scan("AAPL", "SHORT"))
        _veto(panel, "sma_incoming")
        assert _d1_kinds(panel) == [("AAPL", "sma_break_retest", "SHORT")]
    finally:
        panel.close()
        panel.deleteLater()


def test_sma_incoming_does_not_touch_an_other_side_arm(tmp_path, monkeypatch):
    def setup(panel):
        assert panel.arm_d1_event_watch("AAPL", "sma_break_retest", side="SHORT")

    panel = _run(tmp_path, monkeypatch, "sma_incoming", side="LONG", setup=setup)
    try:
        assert _d1_kinds(panel) == [("AAPL", "sma_break_retest", "SHORT")]
        assert "not armed" in _status(panel)
        assert panel._current_review_alert.symbol == "NVDA"
    finally:
        panel.close()
        panel.deleteLater()


def test_compressed_arms_a_range_breakout_once(tmp_path, monkeypatch):
    panel = _run(tmp_path, monkeypatch, "compressed")
    try:
        assert [(s, k) for s, k, _side in _d1_kinds(panel)] == [("AAPL", "range_breakout")]
        assert panel._current_review_alert.symbol == "NVDA"
        panel.chart_alert(_scan("AAPL"))
        _veto(panel, "compressed")
        assert len(panel._d1_event_watches) == 1
    finally:
        panel.close()
        panel.deleteLater()


def _daily(count=40, close=98.0):
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=count)
    return [
        {
            "dt": start + timedelta(days=i),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        }
        for i in range(count)
    ]


def _line(kind, price, lookback_end, **extra):
    record = {
        "line_id": f"d1_trendline:{kind}:2026-08-03_2026-08-20",
        "type": kind,
        "start_date": "2026-08-03",
        "end_date": "2026-08-20",
        "start_price": price + 4.0,
        "end_price": price + 2.0,
        "current_line_price": price,
        "slope_log_per_bar": 0.0,
        "lookback_end": lookback_end,
    }
    record.update(extra)
    return record


def test_incoming_trendline_arms_a_frozen_break_retest_on_the_nearest_line(tmp_path, monkeypatch):
    daily = _daily()
    last = daily[-1]["dt"].date().isoformat()
    far = _line("H-", 106.0, last)
    near = dict(_line("H-", 101.0, last), line_id="d1_trendline:H-:2026-08-04_2026-08-21",
                start_date="2026-08-04", end_date="2026-08-21")
    support = _line("L+", 95.0, last, line_id="d1_trendline:L+:2026-08-03_2026-08-20")
    broken = _line("H-break", 99.0, last, break_date=last,
                   line_id="d1_trendline:H-break:2026-08-03_2026-08-20")

    def setup(panel):
        monkeypatch.setattr(panel, "_d1_bars_for", lambda _s: list(daily))
        monkeypatch.setattr(panel, "_wall_trendlines_for", lambda _s: [far, broken, support, near])
        monkeypatch.setattr(panel, "_wall_trendline_knowledge_at", lambda: KNOWN_AT)

    panel = _run(tmp_path, monkeypatch, "incoming_trendline", setup=setup)
    try:
        assert len(panel._d1_event_watches) == 1
        watch = panel._d1_event_watches[0]
        assert watch.kind == "trendline_break_retest" and watch.side == "LONG"
        assert watch.trendline_candidate["line_id"] == near["line_id"]
        assert watch.trendline_knowledge_at == KNOWN_AT
        assert panel._current_review_alert.symbol == "NVDA"
        panel.chart_alert(_scan("AAPL"))
        _veto(panel, "incoming_trendline")
        assert len(panel._d1_event_watches) == 1
    finally:
        panel.close()
        panel.deleteLater()


def test_incoming_trendline_with_no_line_is_not_armed_and_says_why(tmp_path, monkeypatch):
    daily = _daily()

    def setup(panel):
        monkeypatch.setattr(panel, "_d1_bars_for", lambda _s: list(daily))
        monkeypatch.setattr(panel, "_wall_trendlines_for", lambda _s: [])
        monkeypatch.setattr(panel, "_wall_trendline_knowledge_at", lambda: KNOWN_AT)

    panel = _run(tmp_path, monkeypatch, "incoming_trendline", setup=setup)
    try:
        assert panel._d1_event_watches == []
        assert "not armed" in _status(panel) and "no trendline" in _status(panel)
        assert panel._current_review_alert.symbol == "NVDA", "the veto still retires"
    finally:
        panel.close()
        panel.deleteLater()


def test_incoming_trendline_with_no_daily_bars_is_not_armed(tmp_path, monkeypatch):
    last = datetime.now().date().isoformat()

    def setup(panel):
        monkeypatch.setattr(panel, "_d1_bars_for", lambda _s: [])
        monkeypatch.setattr(panel, "_wall_trendlines_for", lambda _s: [_line("H-", 101.0, last)])
        monkeypatch.setattr(panel, "_wall_trendline_knowledge_at", lambda: KNOWN_AT)

    panel = _run(tmp_path, monkeypatch, "incoming_trendline", setup=setup)
    try:
        assert panel._d1_event_watches == []
        assert "not armed" in _status(panel)
        assert panel._current_review_alert.symbol == "NVDA"
    finally:
        panel.close()
        panel.deleteLater()


@pytest.mark.parametrize("code", ["volume_dry", "overhead_horizontal", "earnings_too_close"])
def test_other_reasons_arm_nothing(tmp_path, monkeypatch, code):
    panel = _run(tmp_path, monkeypatch, code)
    try:
        assert panel._d1_event_watches == []
        assert panel._chart_watches == []
        assert panel._current_review_alert.symbol == "NVDA"
    finally:
        panel.close()
        panel.deleteLater()


# ------------------------------------------------------ snapshot popup menu
def test_snapshot_popup_uses_the_grouped_d1_menu(tmp_path, monkeypatch):
    from ui.widgets.arm_bar import _LEGACY_HEADER
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotDialog

    panel = _panel(tmp_path, monkeypatch)
    dialog = SymbolSnapshotDialog()
    try:
        dialog.show_symbol("NVDA", side="LONG", watch_host=panel)
        assert dialog.d1_menu_button.isVisibleTo(dialog)
        menu = dialog.d1_menu_button.menu()
        dialog.sync_d1_menu()
        rows = []
        for action in menu.actions():
            if not action.isVisible():
                continue
            rows.append("--" if action.isSeparator() else action.text())
        assert rows == [
            "PULLBACK — it ran, let it calm down",
            "Pullback (fast)",
            "Pullback to D1 line",
            "--",
            "BREAKOUT — it was tight, let it go",
            "Range breakout",
            "--",
            "LINE BREAK — it crossed a big line",
            "Line break",
            "SMA break + 15EMA retest",
            "Trendline break",
            "Trendline break + retest",
        ]
        # Pullback left the flat M5 row; no D1 button sits in the row itself.
        assert not dialog.watch_buttons["pullback"].isVisibleTo(dialog)
        assert not any(b.isVisibleTo(dialog) for b in dialog.d1_event_buttons.values())

        dialog.d1_actions["range_breakout"].trigger()
        QApplication.processEvents()
        assert "range_breakout" in panel.armed_d1_event_kinds("NVDA")

        dialog.d1_actions["sma_break_retest"].trigger()
        QApplication.processEvents()
        assert any(
            w.kind == "sma_break_retest" and w.side == "LONG" for w in panel._d1_event_watches
        ), "the popup passes its side"

        # A legacy kind armed elsewhere shows so it can be disarmed.
        assert not dialog.d1_actions["ema15_reject"].isVisible()
        assert panel.arm_d1_event_watch("NVDA", "ema15_reject")
        dialog._refresh_watch_actions()
        dialog.sync_d1_menu()
        assert dialog.d1_actions["ema15_reject"].isVisible()
        assert any(a.text() == _LEGACY_HEADER and a.isVisible() for a in menu.actions())
        dialog.d1_actions["ema15_reject"].trigger()
        QApplication.processEvents()
        assert "ema15_reject" not in panel.armed_d1_event_kinds("NVDA")
    finally:
        dialog.close()
        dialog.deleteLater()
        panel.close()
        panel.deleteLater()
