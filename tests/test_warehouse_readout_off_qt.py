"""The warehouse readout read a NETWORK SHARE on the Qt thread, in a click.

G-P1.5. `WarehouseReadoutPanel.refresh` called `ResearchStore.open()` and then
`queries.slice_readout(store)` inline. The research lake lives on the DAS at
`\\\\MINI-PC\\Trading Bot Data\\research_lake`, and that share is known to drop
and re-establish (CLAUDE.md records exactly that happening mid-measurement on
2026-08-21). An SMB read against a dropped share does not fail fast - it blocks
until it times out, and every one of those seconds was a frozen desk with
nothing on screen to explain it.

This is the one panel in the audit where the read leaves the machine, which is
why it went first.

The second defect is in the same method and is about honesty rather than speed:
every failure path called `self._set_rows([])`. An unreadable lake is not an
empty lake. Blanking the table on a failed read replaces "here is what we last
saw" with a silent claim that there is nothing there.
"""

from __future__ import annotations

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
pytest.importorskip("PySide6", reason="the readout is a Qt panel")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

MAIN_THREAD = threading.get_ident()


class _Snapshot:
    def __init__(self, rows, files=2, manifest_seq=7):
        self.rows = rows
        self.files = files
        self.manifest_seq = manifest_seq


def _rows(count=3):
    return [
        {"setup": f"s{i}", "n": i, "win_rate": 0.5, "avg_r": 0.1} for i in range(count)
    ]


def _panel():
    from ui.panels.warehouse_readout_panel import WarehouseReadoutPanel

    return WarehouseReadoutPanel()


def _settle(panel, timeout_ms: int = 15000) -> None:
    worker = getattr(panel, "_worker", None)
    if worker is not None:
        worker.wait(timeout_ms)
    for _ in range(20):
        _app.processEvents()


@pytest.fixture
def lake(monkeypatch):
    """Stand in for the DAS: records the thread it was read on."""
    from research_warehouse import queries
    from research_warehouse.store import ResearchStore

    state = {"idents": [], "snapshot": _Snapshot(_rows()), "raise_on": None}

    def fake_open(*args, **kwargs):
        state["idents"].append(threading.get_ident())
        if state["raise_on"] == "open":
            raise OSError("the DAS share is unreachable")
        return object()

    def fake_readout(store):
        state["idents"].append(threading.get_ident())
        if state["raise_on"] == "read":
            raise OSError("the DAS share is unreachable")
        return state["snapshot"]

    monkeypatch.setattr(ResearchStore, "open", staticmethod(fake_open))
    monkeypatch.setattr(queries, "slice_readout", fake_readout)
    return state


def test_the_lake_is_never_read_on_the_qt_thread(lake):
    panel = _panel()
    panel.refresh()
    _settle(panel)

    assert lake["idents"], "the lake was never read"
    assert MAIN_THREAD not in lake["idents"], (
        "the research lake - a network share that is known to drop - was read "
        "on the Qt thread"
    )
    assert panel.row_count() == 3


def test_a_failed_read_keeps_the_last_good_rows(lake):
    """An unreadable lake is not an empty lake."""
    panel = _panel()
    panel.refresh()
    _settle(panel)
    assert panel.row_count() == 3

    lake["raise_on"] = "read"
    panel.refresh()
    _settle(panel)

    assert panel.row_count() == 3, "a failed read blanked the table"
    assert "unreachable" in panel.status_label.text().lower(), panel.status_label.text()


def test_a_failed_open_also_keeps_the_last_good_rows(lake):
    panel = _panel()
    panel.refresh()
    _settle(panel)
    assert panel.row_count() == 3

    lake["raise_on"] = "open"
    panel.refresh()
    _settle(panel)

    assert panel.row_count() == 3
    assert "unavailable" in panel.status_label.text().lower(), panel.status_label.text()


def test_a_genuinely_empty_lake_still_empties_the_table(lake):
    """Last-good is for FAILURES. A successful read of nothing is a real answer."""
    panel = _panel()
    panel.refresh()
    _settle(panel)
    assert panel.row_count() == 3

    lake["snapshot"] = _Snapshot([])
    panel.refresh()
    _settle(panel)

    assert panel.row_count() == 0, (
        "a successful read that found nothing must clear the table - otherwise "
        "the panel shows rows the lake no longer has"
    )


def test_refresh_is_single_flight(lake):
    """The share is slow when it is unwell; three clicks must not be three reads."""
    release = threading.Event()
    started = threading.Event()
    from research_warehouse import queries

    def slow_readout(store):
        started.set()
        release.wait(10)
        return _Snapshot(_rows())

    queries.slice_readout = slow_readout

    panel = _panel()
    panel.refresh()
    assert started.wait(10)
    panel.refresh()
    panel.refresh()
    release.set()
    _settle(panel)

    assert lake["idents"].count(MAIN_THREAD) == 0
    # `open` is called once per accepted refresh; the two extra clicks were
    # refused while one was already in flight.
    assert len(lake["idents"]) == 1, lake["idents"]


def test_the_research_page_joins_the_readouts_worker(lake):
    """A worker must not outlive the widget it was going to update.

    `ResearchPanel.shutdown` names its children one by one, and had not been
    updated when this panel grew a thread.
    """
    from ui.panels.research_panel import ResearchPanel

    panel = ResearchPanel()
    try:
        readout = panel.warehouse_readout_panel
        readout.refresh()
        panel.shutdown()
        worker = getattr(readout, "_worker", None)
        assert worker is None or not worker.isRunning(), (
            "shutdown returned while the lake read was still running"
        )
    finally:
        try:
            panel.close()
        except Exception:
            pass
