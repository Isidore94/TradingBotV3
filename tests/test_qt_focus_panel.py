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


def _service(tmp_path):
    from focus_picks import FocusPickStore
    from ui.services.focus_service import FocusService

    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "focus_pick_membership.json",
    )
    return FocusService(store)


def test_focus_panel_add_renders_chips(tmp_path):
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    editor = panel.long_editor
    editor.add_input.setText("nvda, aapl")
    editor.add_from_input()

    assert panel.service.focus_symbols("long") == ["NVDA", "AAPL"]
    assert editor.chip_flow.count() == 2  # focusChanged rebuilt the chips
    assert editor.add_input.text() == ""  # input cleared


def test_focus_panel_chip_remove_updates_store(tmp_path):
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.long_editor.add_input.setText("NVDA AAPL")
    panel.long_editor.add_from_input()

    panel.long_editor._remove("NVDA")  # simulates a chip's × button

    assert panel.service.focus_symbols("long") == ["AAPL"]
    assert panel.long_editor.chip_flow.count() == 1


def test_focus_panel_sides_are_independent(tmp_path):
    from ui.panels.focus_picks_panel import FocusPicksPanel

    panel = FocusPicksPanel(_service(tmp_path))
    panel.long_editor.add_input.setText("NVDA")
    panel.long_editor.add_from_input()
    panel.short_editor.add_input.setText("TSLA")
    panel.short_editor.add_from_input()

    assert panel.service.focus_symbols("long") == ["NVDA"]
    assert panel.service.focus_symbols("short") == ["TSLA"]
    assert panel.long_editor.chip_flow.count() == 1
    assert panel.short_editor.chip_flow.count() == 1
