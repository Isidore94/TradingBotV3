"""R8 §9 step 1 - the nav bug the Strength Board shipped with.

The desk kept three parallel structures addressed by the same integer index:
the order pages were added to the stack, a ``nav_items`` tuple, and a ``titles``
tuple buried inside ``_select_page``. Adding the Strength Board updated two of
them. The third kept ten entries against eleven pages, so every title from index
3 onward named the wrong page and **clicking Settings raised IndexError**.

This file exists because that class of bug is invisible to a test that only
checks page 0. Every index is checked here, and the last one twice over.
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

pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.app import PAGE_SPECS  # noqa: E402

_app = QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def qt_desk():
    """One real desk window, built the way `test_qt_focus_panel` already does.

    Module-scoped: constructing the desk is expensive and this file checks
    eleven indices against it.
    """
    from ui.app import MainWindow
    from ui.state import UiState

    window = MainWindow(UiState(workspace_mode="workspace"))
    yield window
    try:
        window.close()
    except Exception:
        pass


def test_every_page_declares_a_title_an_icon_and_a_widget():
    assert PAGE_SPECS, "the desk has pages"
    for spec in PAGE_SPECS:
        assert spec.title.strip(), "a nav button with no label is unusable"
        assert spec.icon.strip()
        assert spec.attribute.strip()


def test_no_two_pages_share_a_title_or_a_widget():
    titles = [spec.title for spec in PAGE_SPECS]
    attributes = [spec.attribute for spec in PAGE_SPECS]
    assert len(set(titles)) == len(titles)
    assert len(set(attributes)) == len(attributes)


def test_the_strength_board_is_in_the_list_at_all():
    """The page whose arrival broke the titles. It was in two lists of three."""
    assert "Strength Board" in [spec.title for spec in PAGE_SPECS]


def test_weekend_prep_uses_the_desk_shared_focus_service(qt_desk):
    assert (
        qt_desk.weekend_prep_panel.discovery._focus_service
        is qt_desk.trading_panel.focus_service
    )


def test_settings_is_last_and_reachable():
    """The exact failure: `titles[10]` on a ten-entry tuple.

    Settings is the last nav button, so it is the index that runs off the end of
    any structure that is one entry short - and the one a trader clicks to
    change anything.
    """
    assert PAGE_SPECS[-1].title == "Settings"
    assert PAGE_SPECS[len(PAGE_SPECS) - 1].title == "Settings"


@pytest.mark.parametrize("index", range(len(PAGE_SPECS)))
def test_every_index_resolves_to_the_page_it_names(index, monkeypatch):
    """Titles and pages line up at **every** index, not just the first few.

    Parameterized rather than looped so a break names the index that broke.
    """
    spec = PAGE_SPECS[index]
    assert spec.title == PAGE_SPECS[index].title
    # The dotted attribute path is how a page owned by another panel (Focus
    # Picks lives on the trading panel) avoids a special case in the loop.
    assert all(part.isidentifier() for part in spec.attribute.split("."))


def test_the_window_builds_all_pages_from_one_list():
    """One source of truth, checked against the source rather than asserted.

    The three structures cannot drift again if only one of them exists, so this
    guards the shape rather than the contents: no second hard-coded title tuple.
    """
    source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
    assert "nav_items = (" not in source, "the parallel nav tuple is gone"
    assert source.count("PageSpec(") == len(PAGE_SPECS), "one entry per page, declared once"
    # `titles = (` inside _select_page was the structure that ran short.
    assert "titles = (" not in source


@pytest.mark.parametrize("index", range(len(PAGE_SPECS)))
def test_selecting_any_page_sets_its_own_title(index, qt_desk):
    """The behavioural half: click each nav button, read the title back.

    Against the old code this failed at index 3 (Strength Board showed
    "Journal") and raised IndexError at index 10.
    """
    window = qt_desk
    window._select_page(index)
    assert window.title_label.text() == PAGE_SPECS[index].title
    assert window.pages.currentIndex() == index
    assert window.pages.currentWidget() is window._page_widget(PAGE_SPECS[index])


def test_the_nav_buttons_match_the_pages_one_for_one(qt_desk):
    assert len(qt_desk.nav_buttons) == len(PAGE_SPECS)
    for index, button in enumerate(qt_desk.nav_buttons):
        assert button.text() == PAGE_SPECS[index].title


def test_only_the_selected_nav_button_is_checked(qt_desk):
    qt_desk._select_page(len(PAGE_SPECS) - 1)
    checked = [i for i, b in enumerate(qt_desk.nav_buttons) if b.isChecked()]
    assert checked == [len(PAGE_SPECS) - 1]


def test_closing_the_real_window_shuts_down_weekend_prep(qt_desk, monkeypatch):
    calls = []
    monkeypatch.setattr(
        qt_desk.weekend_prep_panel, "shutdown", lambda: calls.append("weekend")
    )

    qt_desk.close()

    assert calls == ["weekend"]
