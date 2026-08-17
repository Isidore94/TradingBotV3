"""R4 section 6.2: the feed's Focus action reads as one.

Trader wording, 2026-08-14: *"If I like a stock I can add it to m5 focus picks.
Then I get flagged on pullbacks."* The action already had exactly those
semantics -- it just wore a bare star glyph, so the one verb on the feed that
places membership was the least legible thing on the row.

This is a LABEL change on a placement verb. It must not blur into R4's
CaptureRail LIKE, which is analysis-only and never writes Focus membership;
tests in test_qt_alert_capture.py hold that other half.
"""

from __future__ import annotations

import sys
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


def _item(*, is_focus: bool = False, hint: str = "M5 Focus", symbol: str = "AAPL"):
    from ui.models.bounce import BounceAlert
    from ui.widgets.alert_feed_item import AlertFeedItem

    alert = BounceAlert(
        time_text="09:31:00",
        symbol=symbol,
        side="LONG",
        trigger="Bounce confirmed",
        timeframe="M5",
        tag="green",
        raw_text=f"[B-TIER] {symbol}: Bounce confirmed",
    )
    return AlertFeedItem(
        alert,
        focus_category="swing" if (is_focus and "Swing" in hint) else "m5" if is_focus else "",
        show_favorite_button=True,
        favorite_hint=hint,
    )


def test_the_unlit_action_names_the_list_it_places_into():
    item = _item(hint="M5 Focus")
    assert "Like" in item.favorite_button.text()
    assert "M5 Focus" in item.favorite_button.text()


def test_a_swing_alert_names_swing_focus():
    item = _item(hint="Swing Focus")
    assert "Swing Focus" in item.favorite_button.text()


def test_the_lit_state_still_offers_removal():
    """The trader must be able to undo placement from the same control -- that
    affordance existed on the lit star and the label must not eat it."""
    item = _item(is_focus=True, hint="M5 Focus")
    assert "M5 Focus" in item.favorite_button.text()
    assert "remove" in item.favorite_button.toolTip().lower()


def test_the_action_still_emits_the_same_signal():
    """A label change must not move the wiring: the hosting panel's
    _toggle_favorite is what actually writes membership."""
    item = _item()
    fired: list[int] = []
    item.favoriteToggled.connect(lambda: fired.append(1))
    item.favorite_button.click()
    assert fired == [1]


def test_no_action_without_a_focus_service():
    """show_favorite_button is False when the panel has no FocusService: a
    placement verb with nothing to place through would be a dead button."""
    from ui.models.bounce import BounceAlert
    from ui.widgets.alert_feed_item import AlertFeedItem

    alert = BounceAlert(time_text="09:31:00", symbol="AAPL", side="LONG", tag="green")
    item = AlertFeedItem(alert, show_favorite_button=False)
    assert item.favorite_button is None


def test_an_alert_with_no_symbol_gets_no_action():
    from ui.models.bounce import BounceAlert
    from ui.widgets.alert_feed_item import AlertFeedItem

    alert = BounceAlert(time_text="09:31:00", symbol="", side="WATCH", tag="green")
    item = AlertFeedItem(alert, show_favorite_button=True)
    assert item.favorite_button is None
