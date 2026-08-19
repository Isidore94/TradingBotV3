"""R4's held-back reviewed-today marker for Focus Picks (built 2026-08-18).

R4 deliberately did NOT put a marker inside these editors: they hold editable
watchlist text that is written back to the shared watchlists, so a glyph in a
row is one careless save away from becoming a symbol name. The hold is honored
by rendering the answer BESIDE the editors instead, and these tests are what
keep that true:

- the marker text never reaches the editors or the watchlist bytes;
- unreadable evidence says nothing rather than claiming nothing was reviewed;
- a reviewed name that is not in Focus is counted, not listed as if it were.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6")


def _panel(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    import focus_picks
    import project_paths

    monkeypatch.setattr(project_paths, "PERSISTENT_DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr(focus_picks, "FOCUS_FILES", getattr(focus_picks, "FOCUS_FILES", {}), raising=False)

    from ui.panels.focus_picks_panel import FocusPicksPanel
    from ui.services.focus_service import FocusService

    service = FocusService()
    panel = FocusPicksPanel(service)
    return panel


def test_a_reviewed_focus_name_is_named_beside_the_editors(monkeypatch, tmp_path):
    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(
        panel.service, "focus_symbols", lambda side, category: ["AAPL"] if side == "long" else []
    )
    import pick_feedback

    monkeypatch.setattr(pick_feedback, "reviewed_symbols_today", lambda *a, **k: {"AAPL", "MSFT"})

    panel.refresh_reviewed_today()
    assert "AAPL" in panel.reviewed_today_label.text()


def test_reviewed_names_outside_focus_are_counted_not_listed(monkeypatch, tmp_path):
    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(panel.service, "focus_symbols", lambda side, category: [])
    import pick_feedback

    monkeypatch.setattr(pick_feedback, "reviewed_symbols_today", lambda *a, **k: {"MSFT"})

    panel.refresh_reviewed_today()
    text = panel.reviewed_today_label.text()
    assert "none of them in Focus" in text
    assert "MSFT" not in text


def test_nothing_reviewed_says_so_plainly(monkeypatch, tmp_path):
    panel = _panel(monkeypatch, tmp_path)
    import pick_feedback

    monkeypatch.setattr(pick_feedback, "reviewed_symbols_today", lambda *a, **k: set())
    panel.refresh_reviewed_today()
    assert panel.reviewed_today_label.text() == "Reviewed today: none yet."


def test_unreadable_evidence_is_silence_not_a_claim(monkeypatch, tmp_path):
    """"Nothing reviewed" and "I could not read the record" are different."""
    panel = _panel(monkeypatch, tmp_path)
    import pick_feedback

    def _boom(*_args, **_kwargs):
        raise OSError("locked")

    monkeypatch.setattr(pick_feedback, "reviewed_symbols_today", _boom)
    panel.refresh_reviewed_today()
    assert panel.reviewed_today_label.text() == ""


def test_the_marker_never_touches_the_editors(monkeypatch, tmp_path):
    """The whole reason R4 held this back."""
    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(
        panel.service, "focus_symbols", lambda side, category: ["AAPL"] if side == "long" else []
    )
    import pick_feedback

    monkeypatch.setattr(pick_feedback, "reviewed_symbols_today", lambda *a, **k: {"AAPL"})

    panel.refresh_reviewed_today()
    for editor in panel.editors:
        # The add box is the only editable text on the editor, and it must be
        # exactly as the trader left it.
        assert editor.add_input.text() == ""
