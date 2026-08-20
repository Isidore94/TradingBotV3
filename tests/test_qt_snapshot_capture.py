"""R4 sections 2 and 5: capture and the reviewed-today badge on the snapshot.

The trader's ask was *"anytime I bring up a chart from master avwap setups or
the RS/RW board or anywhere it would be nice if it had all the functions of
chart review"*. The mechanism that delivers that is a `CaptureRail` living in
`SymbolSnapshotDialog` itself, so every host that opens the popup inherits
capture without having to know anything about it.

These tests go through the real dialog, not a fake: the R8 review found six
blockers at seams the tests had bypassed, and a stub host proves nothing about
whether the RS/RW board actually gets a rail.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

_QT = pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = _QT.QApplication.instance() or _QT.QApplication([])
    yield app


@pytest.fixture
def dialog(tmp_path, monkeypatch):
    """A real SymbolSnapshotDialog whose captures land in tmp_path."""
    import pick_feedback
    from ui.widgets import symbol_snapshot_dialog as mod

    pick_feedback.clear_reviewed_today_cache()
    dlg = mod.SymbolSnapshotDialog(annotations_path=tmp_path / "trader_annotations.jsonl")
    # The popup owns a 30s refresh timer; a test that leaves it running leaks a
    # live QTimer into every later Qt test in the process.
    dlg._refresh_timer.stop()
    yield dlg
    dlg.deleteLater()


# --------------------------------------------------------------------------
# section 2 - the rail is present, wired, and analysis-only
# --------------------------------------------------------------------------
def test_the_snapshot_popup_carries_a_capture_rail(dialog):
    from ui.widgets.capture_rail import CaptureRail

    assert isinstance(dialog.capture_rail, CaptureRail)


def test_capture_is_available_without_any_host(dialog, monkeypatch):
    """A quick look opened from a board with no watch host still captures.

    This is the whole point of section 2: the RS/RW board and the Industry
    panel pass no review_host, and before R4 that left them without even a
    Dislike button."""
    monkeypatch.setattr(dialog.snapshot, "set_symbol", lambda *a, **k: None)
    dialog.show_symbol("AAPL")
    assert dialog.capture_rail.isVisible() or not dialog.isVisible()
    assert dialog.capture_rail.veto_button.isEnabled()


def test_showing_a_symbol_points_the_rail_at_it(dialog, monkeypatch):
    monkeypatch.setattr(dialog.snapshot, "set_symbol", lambda *a, **k: None)
    dialog.show_symbol("MSFT", side="SHORT")
    assert dialog.capture_rail._symbol == "MSFT"
    assert dialog.capture_rail._side == "SHORT"


def test_selecting_a_painted_level_re_points_the_rail(dialog, monkeypatch):
    """Same wiring chart_review_panel uses: a clicked level becomes the
    capture's reference, so a veto records WHICH level it was about."""
    monkeypatch.setattr(dialog.snapshot, "set_symbol", lambda *a, **k: None)
    dialog.show_symbol("NVDA")
    dialog.snapshot._symbol = "NVDA"
    dialog.snapshot.d1LevelSelected.emit("NVDA", "d1_horizontal:2026-06-01:100.00", "d1_horizontal", 100.0)
    assert dialog.capture_rail._ref_level_id == "d1_horizontal:2026-06-01:100.00"
    assert dialog.capture_rail._ref_level_family == "d1_horizontal"


def test_a_level_selected_for_another_symbol_is_ignored(dialog, monkeypatch):
    monkeypatch.setattr(dialog.snapshot, "set_symbol", lambda *a, **k: None)
    dialog.show_symbol("NVDA")
    dialog.snapshot.d1LevelSelected.emit("AMD", "x:y:1.00", "d1_horizontal", 1.0)
    assert dialog.capture_rail._ref_level_id == ""


def test_switching_symbols_clears_the_previous_level_reference(dialog, monkeypatch):
    """A stale ref_level_id would attribute a veto on one name to a level the
    trader clicked on a different chart."""
    monkeypatch.setattr(dialog.snapshot, "set_symbol", lambda *a, **k: None)
    dialog.show_symbol("NVDA")
    dialog.snapshot._symbol = "NVDA"
    dialog.snapshot.d1LevelSelected.emit("NVDA", "d1_horizontal:2026-06-01:100.00", "d1_horizontal", 100.0)
    dialog.show_symbol("AMD")
    assert dialog.capture_rail._ref_level_id == ""


def test_a_capture_from_the_popup_writes_only_annotations(dialog, tmp_path, monkeypatch):
    """The invariant: capture never writes Focus or a watchlist."""
    monkeypatch.setattr(dialog.snapshot, "set_symbol", lambda *a, **k: None)
    dialog.show_symbol("AAPL")
    dialog.capture_rail.note_input.setText("looks heavy into the 50")
    row = dialog.capture_rail.commit_note()
    assert row is not None
    written = (tmp_path / "trader_annotations.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(written)["symbol"] == "AAPL"
    # Nothing else was created beside it.
    assert {path.name for path in tmp_path.iterdir()} == {"trader_annotations.jsonl"}


# --------------------------------------------------------------------------
# section 5 - the reviewed-today badge
# --------------------------------------------------------------------------
def _decide(path: Path, symbol: str, market_date: str) -> None:
    path.write_text(
        json.dumps(
            {
                "ts": f"{market_date}T09:31:00",
                "trade_date": market_date,
                "symbol": symbol,
                "side": "LONG",
                "verdict": "dislike",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_a_symbol_decided_today_shows_the_badge(dialog, tmp_path, monkeypatch):
    import pick_feedback

    feedback = tmp_path / "pick_feedback.jsonl"
    today = datetime.now().date().isoformat()
    _decide(feedback, "AAPL", today)
    pick_feedback.clear_reviewed_today_cache()
    monkeypatch.setattr(
        dialog,
        "_reviewed_symbols",
        lambda: pick_feedback.reviewed_symbols_today(
            market_date=today,
            pick_feedback_path=feedback,
            review_events_path=tmp_path / "none.jsonl",
            annotations_path=tmp_path / "none2.jsonl",
        ),
    )
    monkeypatch.setattr(dialog.snapshot, "set_symbol", lambda *a, **k: None)
    dialog.show_symbol("AAPL")
    assert "Reviewed today" in dialog.reviewed_badge.text()
    assert dialog.reviewed_badge.isVisible() or not dialog.isVisible()


def test_an_undecided_symbol_shows_no_badge(dialog, tmp_path, monkeypatch):
    import pick_feedback

    feedback = tmp_path / "pick_feedback.jsonl"
    today = datetime.now().date().isoformat()
    _decide(feedback, "AAPL", today)
    pick_feedback.clear_reviewed_today_cache()
    monkeypatch.setattr(
        dialog,
        "_reviewed_symbols",
        lambda: pick_feedback.reviewed_symbols_today(
            market_date=today,
            pick_feedback_path=feedback,
            review_events_path=tmp_path / "none.jsonl",
            annotations_path=tmp_path / "none2.jsonl",
        ),
    )
    monkeypatch.setattr(dialog.snapshot, "set_symbol", lambda *a, **k: None)
    dialog.show_symbol("TSLA")
    assert dialog.reviewed_badge.text() == ""


def test_capturing_makes_the_badge_appear_without_reopening(dialog, tmp_path, monkeypatch):
    """The trader's stated want is 'very obvious I have already checked that
    chart today' - which is worth nothing if it only lands on the next open."""
    monkeypatch.setattr(dialog.snapshot, "set_symbol", lambda *a, **k: None)
    seen: set[str] = set()
    monkeypatch.setattr(dialog, "_reviewed_symbols", lambda: set(seen))
    dialog.show_symbol("AAPL")
    assert dialog.reviewed_badge.text() == ""
    seen.add("AAPL")
    dialog.capture_rail.note_input.setText("checked")
    dialog.capture_rail.commit_note()
    assert "Reviewed today" in dialog.reviewed_badge.text()


def test_a_badge_lookup_failure_never_takes_down_the_popup(dialog, monkeypatch):
    def boom():
        raise OSError("the home folder went away")

    monkeypatch.setattr(dialog, "_reviewed_symbols", boom)
    monkeypatch.setattr(dialog.snapshot, "set_symbol", lambda *a, **k: None)
    dialog.show_symbol("AAPL")
    assert dialog.reviewed_badge.text() == ""


# --------------------------------------------------------------------------
# section 2.2 - the boards really do inherit it
# --------------------------------------------------------------------------
def test_the_rs_board_opens_a_chart_that_can_capture(tmp_path, monkeypatch):
    """Through the real panel call site, not a reimplementation of it."""
    from ui.widgets import symbol_snapshot_dialog as mod

    owner = _QT.QWidget()
    dialog = mod.show_symbol_snapshot(owner, "AAPL", side="LONG")
    try:
        dialog._refresh_timer.stop()
        assert dialog.capture_rail is not None
        assert dialog.capture_rail._symbol == "AAPL"
    finally:
        dialog.deleteLater()
        owner.deleteLater()


# --------------------------------------------------------------------------
# 2026-08-20: the popup has to be typeable.
#
# Trader: "i cant type in the master avwap charts that I double click on in
# the notes section." Cause: Qt.WindowDoesNotAcceptFocus was set on the
# dialog. That flag tells the window system the window may NEVER take keyboard
# focus, so nothing inside it could receive a keystroke - clicking into the
# note field worked and typing did nothing.
#
# The offscreen platform does not enforce OS focus rules, so a hasFocus()
# assertion passes either way and would not have caught this. The flag itself
# is the contract, so the flag is what these assert.
# --------------------------------------------------------------------------
def test_the_popup_can_take_keyboard_focus(dialog):
    from PySide6.QtCore import Qt

    assert not bool(
        dialog.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    ), "this flag makes every field in the popup untypeable"


def test_the_popup_still_does_not_steal_focus_when_it_appears(dialog):
    """The original intent, kept: appearing must not pull the caret out of a
    watchlist editor or the live feed."""
    from PySide6.QtCore import Qt

    assert dialog.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert bool(dialog.windowFlags() & Qt.WindowType.Tool)


def test_every_capture_field_accepts_input(dialog):
    """Belt and braces: the fields themselves are editable and focusable."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    rail = dialog.capture_rail
    for name in ("note_input", "veto_note_input", "like_note_input"):
        field = getattr(rail, name)
        assert field.isEnabled() and not field.isReadOnly()
        assert field.focusPolicy() != Qt.FocusPolicy.NoFocus
        field.clear()
        field.setFocus()
        QTest.keyClicks(field, "abc")
        assert field.text() == "abc", f"{name} swallowed the keystrokes"
