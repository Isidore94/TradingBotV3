"""V2 item 4 - the Market Journal capture is one box and one Enter.

Decision 0016 answer 11, in the trader's words: *"one box, one Enter, one or two
thesis entries a day."*

The desk-tab capture had a timeframe picker, a box, a Save button and a status
line - four decisions for a thought you have at 10:40 and would otherwise lose.
Everything except the box leaves the SURFACE. Nothing leaves the SCHEMA.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PT = timezone(timedelta(hours=-7))


# ---------------------------------------------------------------------------
# Which session a note is about
# ---------------------------------------------------------------------------


def test_a_note_typed_during_the_session_is_about_today():
    import market_journal

    assert market_journal.session_date_for(
        datetime(2026, 9, 2, 10, 0, tzinfo=PT)
    ) == "2026-09-02"


def test_a_note_typed_after_the_close_is_still_about_the_day_that_just_ended():
    """Dating it tomorrow would file it against a session that has not happened."""
    import market_journal

    assert market_journal.session_date_for(
        datetime(2026, 9, 2, 18, 0, tzinfo=PT)
    ) == "2026-09-02"


def test_a_weekend_note_is_about_the_last_session_that_traded():
    import market_journal

    for moment in (
        datetime(2026, 9, 5, 11, 0, tzinfo=PT),  # Saturday
        datetime(2026, 9, 6, 11, 0, tzinfo=PT),  # Sunday
        datetime(2026, 9, 7, 11, 0, tzinfo=PT),  # Labor Day
    ):
        assert market_journal.session_date_for(moment) == "2026-09-04", moment


def test_the_after_the_fact_flag_is_still_computed_and_not_replaced():
    """Two different questions: which day it is about, and whether the trader
    had already seen how that day finished."""
    import market_journal

    saturday = datetime(2026, 9, 5, 11, 0, tzinfo=PT)
    entry = market_journal.build_entry(
        text="Friday ran out of buyers into the close",
        session_date=market_journal.session_date_for(saturday),
        now=saturday,
    )

    assert entry["session_date"] == "2026-09-04"
    assert entry["written_after_the_session"] is True


def test_it_never_costs_the_thought(monkeypatch):
    """A note filed against today is a small error; a lost thought is not."""
    import market_journal

    def _explode(*_args, **_kwargs):
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr("market_session.get_market_session_window", _explode)

    stamp = market_journal.session_date_for(datetime(2026, 9, 2, 10, 0, tzinfo=PT))
    assert stamp == "2026-09-02"


# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------


def test_the_desk_capture_has_one_box_and_no_picker_or_button():
    source = (ROOT / "scripts" / "ui" / "panels" / "alert_center_panel.py").read_text(
        encoding="utf-8"
    )
    tab = source.split("def _build_journal_tab", 1)[1].split("def eventFilter", 1)[0]

    assert "QPlainTextEdit()" in tab
    assert "QComboBox()" not in tab, "the timeframe picker is gone from the surface"
    assert 'QPushButton("Save entry' not in tab, "Enter saves; no button"
    # And Ctrl+Enter still works, because the trader has been typing it for weeks.
    assert 'QKeySequence("Ctrl+Return")' in tab


def test_the_schema_keeps_the_timeframe_the_surface_stopped_asking_for():
    """A field that exists at v1 keeps its name and meaning forever."""
    source = (ROOT / "scripts" / "ui" / "panels" / "alert_center_panel.py").read_text(
        encoding="utf-8"
    )
    assert "_journal_timeframe_value = market_journal.TIMEFRAME_M5" in source
    assert "timeframe=self._journal_timeframe_value" in source

    import market_journal

    entry = market_journal.build_entry(text="x", session_date="2026-09-02")
    for field in ("timeframe", "symbols", "origin", "supersedes", "session_date"):
        assert field in entry, field


def test_plain_enter_saves_and_shift_enter_does_not():
    """An event filter, not a shortcut: Return must mean "save" only in the box."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication, QPlainTextEdit

    QApplication.instance() or QApplication([])

    from ui.panels.alert_center_panel import AlertCenterPanel

    saved = []

    class _Fake:
        _journal_text = QPlainTextEdit()

        def _commit_journal_entry(self):
            saved.append(True)

    fake = _Fake()

    plain = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
    )
    shifted = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier
    )

    # Unbound so the fake's own `_commit_journal_entry` is what runs; `super()`
    # is never reached on the accepted path, which is what `True` means.
    handled = AlertCenterPanel.eventFilter.__wrapped__(fake, fake._journal_text, plain) \
        if hasattr(AlertCenterPanel.eventFilter, "__wrapped__") \
        else _call_filter(fake, fake._journal_text, plain)
    assert handled is True
    assert saved == [True]

    saved.clear()
    _call_filter(fake, fake._journal_text, shifted)
    assert saved == [], "Shift+Enter is a newline"


def _call_filter(fake, watched, event):
    """Run the filter's own body against a stand-in.

    `AlertCenterPanel.eventFilter` calls `super().eventFilter` on the path it
    does not handle, which needs a real QObject; the handled path returns before
    that. Copying the guard here would test a copy, so this calls the real method
    and lets the unhandled path raise into a caught AttributeError.
    """
    from ui.panels.alert_center_panel import AlertCenterPanel

    try:
        return AlertCenterPanel.eventFilter(fake, watched, event)
    except (AttributeError, TypeError):
        return False
