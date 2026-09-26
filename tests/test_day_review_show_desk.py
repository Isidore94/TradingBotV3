"""R1 Day Review Show on the desk, and B12 (the page's early eventFilter)."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _extra in (ROOT_DIR / "scripts", ROOT_DIR / "tests"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6", reason="the Day Review page uses PySide6")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QTableWidgetItem, QWidget  # noqa: E402

import day_review_pack  # noqa: E402
import day_review_show  # noqa: E402
from test_day_review_show import SESSION, _good_reply, _pack  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Service:
    def read_day(self, session_date, **_kwargs):
        from ui.services.day_review_service import empty_payload

        return empty_payload(session_date)


@pytest.fixture
def panel(app, monkeypatch):
    from ui.panels import day_review_panel

    widget = day_review_panel.DayReviewPanel(
        service=_Service(), clock=lambda: datetime(2026, 9, 26, 7, 0)
    )
    monkeypatch.setattr(widget, "reload", lambda: None)
    yield widget
    widget.close_show()
    widget.shutdown()
    widget.deleteLater()
    app.processEvents()


def _chosen(facts_only=False):
    pack = _pack()
    if facts_only:
        return {"deck": day_review_show.fallback_deck(pack), "facts_only": True,
                "reason": "no show was written for this session", "model": ""}
    return {"deck": day_review_show.verify_show(_good_reply(), pack), "facts_only": False,
            "reason": "", "model": "gemma3:12b"}


def _render(panel, **extra):
    from ui.services.day_review_service import empty_payload

    payload = empty_payload(SESSION)
    payload.update(extra)
    panel.session_date = lambda: SESSION
    panel.render(payload)


def _key(widget, key):
    widget.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))


# ---------------------------------------------------------------------------
# B12
# ---------------------------------------------------------------------------
def test_b12_the_event_filter_is_quiet_before_entry_text_exists(panel, monkeypatch):
    """Tables install the filter before `entry_text` is built; J must still work."""
    from ui.panels import day_review_panel

    swallowed: list[str] = []
    monkeypatch.setattr(
        day_review_panel, "note_swallowed", lambda reason, *a, **k: swallowed.append(reason)
    )
    table = panel.miss_table
    table.setRowCount(2)
    table.setColumnCount(2)
    for row in range(2):
        table.setItem(row, 1, QTableWidgetItem(str(row)))
    table.setCurrentCell(0, 1)
    del panel.entry_text  # the state while __init__ is still building the page
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_J, Qt.KeyboardModifier.NoModifier)
    assert panel.eventFilter(table, event) is True
    assert swallowed == []
    assert table.currentRow() == 1


# ---------------------------------------------------------------------------
# the worker payload
# ---------------------------------------------------------------------------
def test_the_payload_carries_the_show_present_and_empty():
    from ui.services.day_review_service import PAYLOAD_KEYS, empty_payload

    assert "show" in PAYLOAD_KEYS
    assert empty_payload(SESSION)["show"] == {}


def test_the_worker_picks_the_verified_show_or_the_fallback(tmp_path, monkeypatch):
    import project_paths
    from ui.services.day_review_service import DayReviewService

    monkeypatch.setattr(project_paths, "DAY_REVIEW_DIR", tmp_path, raising=False)
    pack = _pack()
    payload = {"truth": {"lines": ["Last 20 sessions:", "Stocks 3-2", "Alerts: 5 shown"]},
               "spy_m5_bars": []}
    chosen = DayReviewService._show(SESSION, payload, pack)
    assert chosen["facts_only"] is True
    bodies = [slide["body"] for slide in chosen["deck"]["slides"]]
    assert "Alerts: 5 shown" in bodies
    truth = next(s for s in chosen["deck"]["slides"] if s["title"] == "Your last sessions")
    assert truth["lines"] == ["Last 20 sessions:", "Stocks 3-2"]

    path = day_review_show.show_path(SESSION)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "session_date": SESSION, "pack_hash": pack["inputs_hash"], "model": "gemma3:12b",
        "show": day_review_show.verify_show(_good_reply(), pack),
    }), encoding="utf-8")
    chosen = DayReviewService._show(SESSION, payload, pack)
    assert chosen["facts_only"] is False and chosen["model"] == "gemma3:12b"
    # With no current facts the saved pack is the check.
    day_review_pack.write_pack(pack, root=tmp_path)
    assert DayReviewService._show(SESSION, payload, None)["facts_only"] is False


# ---------------------------------------------------------------------------
# the page and the overlay
# ---------------------------------------------------------------------------
def test_show_opens_a_full_window_deck_and_the_keys_drive_it(panel, app):
    host = QWidget()
    host.resize(900, 600)
    host.show()
    panel.setParent(host)
    _render(panel, show=_chosen())
    assert panel.show_button.isEnabled()
    panel.show_button.click()
    overlay = panel.show_overlay()
    assert overlay is not None and overlay.parent() is host
    assert overlay.geometry() == host.rect()
    assert overlay.footer.text() == "1/6 - told by gemma3:12b - sources on hover"
    assert not overlay.badge.isVisibleTo(overlay)
    _key(overlay, Qt.Key.Key_Right)
    assert overlay.index == 1 and overlay.stat_value.text() == "+0.81%"
    assert "measured:SPY" in overlay.title.toolTip()
    _key(overlay, Qt.Key.Key_Space)
    _key(overlay, Qt.Key.Key_Left)
    assert overlay.index == 1
    _key(overlay, Qt.Key.Key_A)
    assert overlay._auto.isActive() and overlay._auto.interval() == 8000
    _key(overlay, Qt.Key.Key_A)
    assert not overlay._auto.isActive()
    host.resize(1000, 700)
    app.processEvents()
    assert overlay.geometry() == host.rect()
    _key(overlay, Qt.Key.Key_Escape)
    assert panel.show_overlay() is None
    panel.setParent(None)
    host.deleteLater()


def test_a_failed_deck_shows_facts_only(panel):
    _render(panel, show=_chosen(facts_only=True))
    panel.open_show()
    overlay = panel.show_overlay()
    assert overlay.badge.isVisibleTo(overlay)
    assert "facts only" in overlay.footer.text()
    assert overlay.badge.toolTip() == "no show was written for this session"


def test_show_with_no_worker_deck_still_opens_the_fallback(panel):
    _render(panel)
    panel.open_show()
    overlay = panel.show_overlay()
    assert overlay.facts_only and len(overlay.slides) >= day_review_show.MIN_SLIDES


def test_fonts_are_set_in_code_and_sized_through_theme_px(app):
    from ui import theme
    from ui.widgets.day_review_show_overlay import DayReviewShow

    overlay = DayReviewShow(_chosen())
    try:
        assert overlay.title.font().families()[0] == "Bahnschrift SemiBold"
        assert overlay.title.font().pixelSize() == theme.px(44)
        assert overlay.stat_value.font().families()[0] == "Cascadia Mono SemiBold"
        assert overlay.body.font().families()[0] == "Georgia"
        assert overlay.footer.font().families()[0] == "Segoe UI"
        assert overlay.glyph.font().families()[0] == "Segoe UI Emoji"
        assert overlay.glyph.text() == day_review_show.KIND_GLYPHS["open"]
    finally:
        overlay.deleteLater()


def test_one_qss_block_styles_the_show_and_no_widget_sets_a_stylesheet():
    source = (ROOT_DIR / "scripts" / "ui" / "widgets" / "day_review_show_overlay.py").read_text(
        encoding="utf-8"
    )
    assert "setStyleSheet" not in source
    qss = (ROOT_DIR / "scripts" / "ui" / "theme.qss").read_text(encoding="utf-8")
    assert qss.count("QFrame#DayReviewShow {") == 1
    from ui import theme

    built = theme.build_stylesheet()
    start = built.index("QFrame#DayReviewShow {")
    assert "@" not in built[start:built.index("Market Journal reader", start)]
