"""P1-4 / 4c - Research -> Setup keys: read off the Qt thread, populations never pooled."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the tab is a Qt panel")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])
MAIN_THREAD = threading.get_ident()

from ui.panels import setup_keys_panel as module  # noqa: E402


def _report():
    key = {"rank": 1, "depth": 1, "label": "ma_support=sma100_support", "facets": {"ma_support": "sma100_support"},
           "lift_pp": 31.5, "selection": {"n": 240, "sessions": 40, "win_rate": 0.85, "wilson_lb": 0.8,
                                         "mean_r": 0.7},
           "holdout": {"n": 120, "win_rate": 0.83, "passed": True}}
    family = {"verdict": "key_found", "keys": [key], "baseline": {"n": 800, "sessions": 40, "win_rate": 0.53},
              "holdout_baseline": {"win_rate": 0.51}}
    none = {"verdict": "no_key_found", "keys": [], "baseline": {"n": 300, "sessions": 40, "win_rate": 0.5,
                                                               "wilson_lb": 0.44, "mean_r": 0.0},
            "holdout_baseline": {"win_rate": 0.49}}
    return {
        "schema": "setup_permutation_report_v1", "generated_at": "2026-09-26T12:00:00+00:00",
        "populations": {
            "swing": {"horizons": {"1": {"families": {"avwap_band_bounce SHORT": family,
                                                      "post_earnings_candle_break SHORT": none}},
                                   "10": {"families": {}}}},
            "m5": {"horizons": {"0": {"families": {"ema_15 LONG": none}}}},
        },
    }


def _settle(panel, timeout_ms: int = 15000) -> None:
    if panel._worker is not None:
        panel._worker.wait(timeout_ms)
    for _ in range(20):
        _app.processEvents()


def test_report_rows_show_keys_and_no_key_found_per_population():
    rows = module.report_rows(_report(), "swing", "1")
    assert [row["family"] for row in rows] == ["avwap_band_bounce SHORT", "post_earnings_candle_break SHORT"]
    assert rows[0]["key"] == "ma_support=sma100_support"
    assert rows[0]["lift_pp"] == "+31.5"
    assert rows[0]["win_rate"] == "85%"
    assert rows[0]["holdout"].startswith("passed: 83% on n=120")
    assert rows[1]["key"] == "no key found"
    m5 = module.report_rows(_report(), "m5", "0")
    assert [row["family"] for row in m5] == ["ema_15 LONG"]
    assert module.horizons_in(_report(), "swing") == ["1", "10"]
    assert module.report_rows(None, "swing", "1") == []


def test_the_report_is_read_off_the_qt_thread(tmp_path, monkeypatch):
    path = tmp_path / "permutation_report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    threads = []
    real = module.read_report
    monkeypatch.setattr(module, "read_report", lambda p: (threads.append(threading.get_ident()), real(p))[1])
    panel = module.SetupKeysPanel(report_path=path)
    assert panel.row_count() == 0  # constructing reads nothing
    panel.refresh()
    _settle(panel)
    assert threads and MAIN_THREAD not in threads
    assert panel.row_count() == 2
    panel.population_input.setCurrentIndex(panel.population_input.findData("m5"))
    assert panel.row_count() == 1
    assert panel.table.item(0, 2).text() == "no key found"
    panel.shutdown()


def test_no_report_and_a_failed_read_keep_what_is_shown(tmp_path, monkeypatch):
    path = tmp_path / "permutation_report.json"
    panel = module.SetupKeysPanel(report_path=path)
    panel.refresh()
    _settle(panel)
    assert "No setup-keys report yet" in panel.status_label.text()
    path.write_text(json.dumps(_report()), encoding="utf-8")
    panel.refresh()
    _settle(panel)
    assert panel.row_count() == 2
    monkeypatch.setattr(module, "read_report", lambda p: (_ for _ in ()).throw(OSError("disk gone")))
    panel.refresh()
    _settle(panel)
    assert panel.row_count() == 2
    assert "disk gone" in panel.status_label.text()
    panel.shutdown()


def test_the_research_page_has_the_setup_keys_tab():
    source = (SCRIPTS_DIR / "ui" / "panels" / "research_panel.py").read_text(encoding="utf-8")
    assert 'tabs.addTab(self.setup_keys_panel, "Setup keys")' in source
    assert "self.setup_keys_panel.shutdown()" in source
