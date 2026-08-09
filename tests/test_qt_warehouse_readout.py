"""The Phase-7 Research-tab readout panel (plan sec 17, 20).

Two properties matter here and nothing else: the panel performs **no lake read
on the render path** - constructing it touches no files, a read happens only
when Refresh is pressed - and it is inert with a message rather than an error
when the warehouse is not configured.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from ui.panels.warehouse_readout_panel import WarehouseReadoutPanel  # noqa: E402


def test_constructing_the_panel_reads_nothing(monkeypatch):
    import research_warehouse.store as store_module

    opened = []
    monkeypatch.setattr(
        store_module.ResearchStore, "open", classmethod(lambda cls, root=None: opened.append(1))
    )

    panel = WarehouseReadoutPanel()
    # No provider call, no lake read, no file touched by construction.
    assert opened == [] and panel.row_count() == 0
    assert "Refresh" in panel.refresh_button.text()

    panel.refresh()  # the explicit action, and only then
    assert opened == [1]


def test_a_disabled_warehouse_is_a_message_not_a_crash(monkeypatch):
    import research_warehouse.store as store_module

    monkeypatch.setattr(store_module.ResearchStore, "open", classmethod(lambda cls, root=None: None))
    panel = WarehouseReadoutPanel()
    panel.refresh()

    assert panel.row_count() == 0
    assert "not configured" in panel.status_label.text()


def test_a_failing_read_is_reported_not_raised(monkeypatch):
    import research_warehouse.store as store_module

    def explode(cls, root=None):
        raise RuntimeError("DAS unmounted")

    monkeypatch.setattr(store_module.ResearchStore, "open", classmethod(explode))
    panel = WarehouseReadoutPanel()
    panel.refresh()

    assert panel.row_count() == 0
    assert "DAS unmounted" in panel.status_label.text()


def test_the_panel_states_that_it_is_exploratory_and_shadow_only():
    panel = WarehouseReadoutPanel()
    caveat = panel.caveat_label.text()
    assert "EXPLORATORY" in caveat
    assert "no shrinkage" in caveat and "no ranking" in caveat
    assert "nothing here affects a score or an alert" in caveat
    # Episodes are the sample size, and the panel says so.
    assert "Episodes" in [label for _key, label in __import__(
        "ui.panels.warehouse_readout_panel", fromlist=["COLUMNS"]
    ).COLUMNS]
