"""TJ-14A item 5 - the card SHOWS what the desk already knows.

RED BEFORE THE FIX. On `claude/tj14a-mentor-card`'s base (`8077a758`) the card
stores `_current_context` SILENTLY (`_on_context_ready`) and draws nothing: the
trader types "RSP is lagging, VXX bid" into a box while the same numbers sit
unread on the row that answer is about. There is no `internals_strip`.

WHAT THIS PINS
--------------
* the strip is built from the context the service ALREADY delivers - no second
  `request_context`, no fetch, no loader, no network;
* it sits ABOVE **What I see**, proven by geometry inside a shown host rather
  than by layout index, so a builder may group the text box how they like;
* an `unmeasured` derived line is SHOWN as unmeasured, never hidden and never
  rendered as a zero;
* it is styled by OBJECT NAME out of `theme.qss` (CLAUDE.md: no stylesheet work
  on the Qt thread).
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Trade Mentor card is Qt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QObject, QPoint, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget  # noqa: E402

import tj14a_support as fixture  # noqa: E402

_app = QApplication.instance() or QApplication([])

PACIFIC = ZoneInfo("America/Los_Angeles")
SESSION = date(2026, 9, 14)
STRIP_OBJECT_NAME = "MentorInternalsStrip"


class _FakeContextService(QObject):
    """The service boundary only. It never loads bars and never fetches."""

    contextReady = Signal(str, object)
    contextUnavailable = Signal(str, object)

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[str, datetime | None]] = []

    def request_context(self, request_id: str, *, now=None) -> bool:
        self.requests.append((str(request_id), now))
        return True


def _slot(hour: int):
    from trade_mentor_schedule import slots_for_session

    for slot in slots_for_session(SESSION):
        if slot.scheduled_at.hour == hour:
            return slot
    raise AssertionError(f"no {hour:02d}:00 slot")


def _card(tmp_path: Path, service):
    from ui.widgets.trade_mentor_card import TradeMentorCard

    return TradeMentorCard(
        journal=None,
        clock=lambda: datetime(2026, 9, 14, 11, 3, tzinfo=PACIFIC),
        drafts_path=tmp_path / "drafts.json",
        context_service=service,
    )


def _strip_text(card) -> str:
    strip = card.internals_strip
    parts = [strip.property("plainText") or ""]
    for label in strip.findChildren(QLabel):
        parts.append(label.text())
    if hasattr(strip, "text"):
        parts.append(strip.text())
    if hasattr(strip, "toPlainText"):
        parts.append(strip.toPlainText())
    return "\n".join(str(part) for part in parts)


def _shown(card):
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(card)
    host.resize(640, 900)
    host.show()
    _app.processEvents()
    return host


def test_the_card_shows_the_internals_it_already_holds_without_fetching_again(tmp_path):
    """One request when the card comes up; the strip is drawn from its answer."""
    service = _FakeContextService()
    card = _card(tmp_path, service)
    slot = _slot(11)

    card.show_slot(slot)
    assert [request[0] for request in service.requests] == [slot.slot_id]

    service.contextReady.emit(slot.slot_id, fixture.context())
    _app.processEvents()

    assert [request[0] for request in service.requests] == [slot.slot_id], (
        "drawing the strip must not start a second read"
    )
    strip = card.internals_strip
    assert strip.isVisibleTo(card)
    text = _strip_text(card)
    for word in ("breadth", "fear", "XLE"):
        assert word.lower() in text.lower(), f"the strip never says {word}"


def test_an_unmeasured_derived_line_is_shown_as_unmeasured(tmp_path):
    """Missing data is uncertainty. A blank row on the strip would read as calm."""
    service = _FakeContextService()
    card = _card(tmp_path, service)
    slot = _slot(11)

    card.show_slot(slot)
    service.contextReady.emit(slot.slot_id, fixture.context(drop=("TLT",)))
    _app.processEvents()

    text = _strip_text(card).lower()
    assert "unmeasured" in text
    assert "rates" in text


def test_the_strip_sits_above_what_i_see(tmp_path):
    """The trader reads what the desk has, THEN writes what it cannot see."""
    service = _FakeContextService()
    card = _card(tmp_path, service)
    slot = _slot(11)

    card.show_slot(slot)
    service.contextReady.emit(slot.slot_id, fixture.context())
    _app.processEvents()
    host = _shown(card)
    try:
        strip_y = card.internals_strip.mapTo(card, QPoint(0, 0)).y()
        box_y = card.text_box.mapTo(card, QPoint(0, 0)).y()
        assert card.internals_strip.height() > 0, "the strip has to occupy real space"
        assert strip_y < box_y
    finally:
        host.close()


def test_the_strip_is_styled_by_object_name_out_of_the_theme(tmp_path):
    """No per-widget stylesheet on the Qt thread (CLAUDE.md, performance rules)."""
    service = _FakeContextService()
    card = _card(tmp_path, service)
    card.show_slot(_slot(11))

    strip = card.internals_strip
    assert strip.objectName() == STRIP_OBJECT_NAME
    assert not strip.styleSheet().strip(), "the variant belongs in theme.qss"
    theme = (SCRIPTS_DIR / "ui" / "theme.qss").read_text(encoding="utf-8")
    assert STRIP_OBJECT_NAME in theme
