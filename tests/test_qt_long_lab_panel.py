"""Research -> Long lab reads nothing until pressed, then formats a worker's report."""

import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from ui.panels import long_lab_panel as mod  # noqa: E402


def _settle(panel, timeout_ms: int = 15000) -> None:
    worker = getattr(panel, "_worker", None)
    if worker is not None:
        worker.wait(timeout_ms)
    for _ in range(20):
        _app.processEvents()


def _cell(axis, regime, horizon, n=40, thin=False):
    return {"rule": "leader_pullback", "axis": axis, "regime": regime, "horizon": horizon, "n": n,
            "thin": thin, "win_raw": 0.55, "wilson_lb": 0.4, "win_vs_spy": 0.5, "mean_raw": 0.012,
            "median_raw": 0.004, "mean_vs_spy": 0.005, "median_vs_spy": -0.002, "mfe_atr": 1.45,
            "mae_atr": 1.3, "limit_fill_rate": 0.78, "best_exit": "time_10"}


REPORT = {
    "schema": "long_lab_v1", "sessions": 194, "first_session": "2025-12-17",
    "last_session": "2026-09-25", "universe": {"symbols_in_window": 1996},
    "generated_at": "2026-09-26T12:00:00+00:00",
    "rules": [{"key": "leader_pullback", "label": "(a) Leader pullback"}],
    "cells": [_cell("spy_trend", "above_rising_20d", 10), _cell("spy_trend", "below_20d", 10, n=12, thin=True),
              _cell("month", "2026-04", 10)],
    "sweep": [{"knob": "pullback_depth", "bucket": "8%-12%", "axis": "spy_trend", "regime": "below_20d",
               "horizon": 10, "n": 50, "win_vs_spy": 0.6, "mean_vs_spy": 0.01, "mean_raw": 0.02, "thin": False}],
    "knob_spread": [{"knob": "rs_decile", "axis": "spy_trend", "regime": "below_20d", "buckets": 3,
                     "spread_vs_spy": 0.021}],
}


def test_construction_reads_nothing_and_load_uses_a_worker(monkeypatch):
    from research_warehouse import long_lab

    calls = []
    monkeypatch.setattr(long_lab, "read_report", lambda path=None: calls.append("read") or REPORT)
    monkeypatch.setattr(long_lab, "lab_from_live_inputs", lambda **_k: calls.append("run"))
    panel = mod.LongLabPanel()
    assert calls == [] and panel.row_count() == 0
    panel.load()
    _settle(panel)
    assert calls == ["read"]
    assert panel.row_count() == 2  # the default axis only: SPY trend
    cells = [panel.table.item(0, c).text() for c in range(panel.table.columnCount())]
    assert cells == ["(a) Leader pullback", "above_rising_20d", "10", "40", "55%", "40%", "50%",
                     "+1.2%", "+0.4%", "+0.5%", "-0.2%", "+1.45", "+1.30", "78%", "time_10"]
    assert panel.table.item(1, 3).text() == "12 (thin)"
    assert panel.sweep_table.rowCount() == 1
    assert "rs_decile 2.1 pts" in panel.sweep_label.text()
    panel.axis_combo.setCurrentIndex(2)  # Month
    assert panel.row_count() == 1 and panel.table.item(0, 1).text() == "2026-04"
    panel.shutdown()


def test_run_computes_and_saves_on_the_worker(monkeypatch):
    from research_warehouse import long_lab

    saved = []
    monkeypatch.setattr(long_lab, "lab_from_live_inputs", lambda **_k: REPORT)
    monkeypatch.setattr(long_lab, "write_report", lambda report, path=None: saved.append(report))
    panel = mod.LongLabPanel()
    panel.run()
    _settle(panel)
    assert saved == [REPORT] and panel.row_count() == 2
    panel.shutdown()


def test_no_report_is_a_message(monkeypatch):
    from research_warehouse import long_lab

    monkeypatch.setattr(long_lab, "read_report", lambda path=None: None)
    panel = mod.LongLabPanel()
    panel.load()
    _settle(panel)
    assert panel.status_label.text() == mod.NO_REPORT_TEXT and panel.row_count() == 0
    panel.shutdown()


def test_the_research_page_carries_the_tab_before_retest_entry():
    from ui.panels.research_panel import ResearchPanel

    panel = ResearchPanel(None)
    try:
        titles = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
        assert titles[-2:] == ["Long lab", "Retest entry"]
        assert panel.tabs.widget(len(titles) - 2) is panel.long_lab_panel
        panel.shutdown()
    finally:
        panel.close()
