"""S8: the Research -> Retest entry tab reads nothing until Run, then formats a worker's report."""

import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from ui.panels import retest_study_panel as mod  # noqa: E402


def _settle(panel, timeout_ms: int = 15000) -> None:
    worker = getattr(panel, "_worker", None)
    if worker is not None:
        worker.wait(timeout_ms)
    for _ in range(20):
        _app.processEvents()


REPORT = {
    "families": [{
        "source": "m5_alert", "family": "vwap", "side": "long", "n": 4,
        "flag_ev_r": -0.25, "flag_win_share": 0.25, "retest_n_filled": 2,
        "retest_no_fill_share": 0.5, "retest_ev_r_per_fill": 0.5, "retest_ev_r_per_alert": 0.25,
    }],
    "skipped": {"no_bars": 3}, "sessions": 2,
    "first_session": "2026-09-24", "last_session": "2026-09-25",
}


def test_construction_reads_nothing_and_run_uses_a_worker(monkeypatch):
    from research_warehouse import retest_entry
    from research_warehouse.store import ResearchStore

    calls = []
    monkeypatch.setattr(ResearchStore, "open", classmethod(lambda cls, root=None: "store"))
    monkeypatch.setattr(retest_entry, "study_from_live_inputs",
                        lambda store, **_k: calls.append(store) or REPORT)
    panel = mod.RetestStudyPanel()
    assert calls == [] and panel.row_count() == 0
    panel.refresh()
    _settle(panel)
    assert calls == ["store"]
    assert panel.row_count() == 1
    cells = [panel.table.item(0, c).text() for c in range(panel.table.columnCount())]
    assert cells == ["m5_alert", "vwap", "long", "4", "-0.25", "25%", "2", "50%", "+0.50", "+0.25"]
    assert "no_bars 3" in panel.status_label.text()
    panel.shutdown()


def test_disabled_warehouse_is_a_message(monkeypatch):
    from research_warehouse.store import ResearchStore

    monkeypatch.setattr(ResearchStore, "open", classmethod(lambda cls, root=None: None))
    panel = mod.RetestStudyPanel()
    panel.refresh()
    _settle(panel)
    assert panel.status_label.text() == mod.DISABLED_TEXT and panel.row_count() == 0
    panel.shutdown()


def test_the_research_page_carries_and_joins_the_tab():
    from ui.panels.research_panel import ResearchPanel

    panel = ResearchPanel(None)
    try:
        titles = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
        assert titles[-1] == "Retest entry"
        assert panel.tabs.widget(len(titles) - 1) is panel.retest_study_panel
        panel.shutdown()
    finally:
        panel.close()
