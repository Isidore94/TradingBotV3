"""TJ-1 item 3 - one Day Review page takes the place of two.

Decision 0021 answer 1: *"One page, Day Review, replaces Market Journal and Daily
Recap."* Left nav before (`ui/app.py:82-110`, 11 specs): Trading Desk · Journal ·
Market Journal · Daily Recap · Weekend Prep · Universe · Research · Auto Pilot ·
A.I. Summary · System Health · Settings. After: the same list with ONE `Day
Review` where the two used to be - ten specs.

`tests/test_qt_page_specs.py` already guards that the list is the only structure
and that every index resolves. This file guards the CONTENTS the packet names, so
a half-done swap (a Day Review spec added beside a Daily Recap spec still in the
nav) cannot pass.

No Qt window is built here: `PAGE_SPECS` is a module-level tuple.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("PySide6", reason="ui.app imports PySide6 at module scope")

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

#: The nav, top to bottom, after TJ-1. Ten entries.
EXPECTED_TITLES = (
    "Trading Desk",
    "Journal",
    "Day Review",
    "Weekend Prep",
    "Universe",
    "Research",
    "Auto Pilot",
    "A.I. Summary",
    "System Health",
    "Settings",
)


def test_the_nav_is_exactly_these_ten_pages_in_this_order():
    from ui.app import PAGE_SPECS

    assert tuple(spec.title for spec in PAGE_SPECS) == EXPECTED_TITLES


def test_day_review_takes_the_market_journal_slot_rather_than_joining_the_end():
    """Third from the top, where "what the trader thought" already lived. A page
    appended after Settings is a different page to reach for."""
    from ui.app import PAGE_SPECS

    assert PAGE_SPECS[2].title == "Day Review"
    assert PAGE_SPECS[2].icon == "mdi.calendar-text"
    assert PAGE_SPECS[2].attribute == "day_review_panel"


def test_neither_old_page_is_in_the_nav_by_title_or_by_widget():
    from ui.app import PAGE_SPECS

    titles = [spec.title for spec in PAGE_SPECS]
    attributes = [spec.attribute for spec in PAGE_SPECS]
    assert "Market Journal" not in titles
    assert "Daily Recap" not in titles
    assert "market_journal_panel" not in attributes
    assert "daily_recap_panel" not in attributes


def test_the_trade_and_tax_journal_page_is_untouched():
    """The label collision was always deliberate: `Journal` is what you traded.
    TJ-1 merges the two JOURNAL-ADJACENT pages, never this one."""
    from ui.app import PAGE_SPECS

    assert PAGE_SPECS[1].title == "Journal"
    assert PAGE_SPECS[1].attribute == "journal_panel"


def test_the_page_is_matched_by_title_through_one_renamed_constant():
    """`DAILY_RECAP_PAGE_TITLE` BECOMES `DAY_REVIEW_PAGE_TITLE` (TJ-1 item 3).
    Two constants, one live page, is the drift `test_qt_page_specs` exists for."""
    import ui.app as app

    assert app.DAY_REVIEW_PAGE_TITLE == "Day Review"
    assert app.DAY_REVIEW_PAGE_TITLE in [spec.title for spec in app.PAGE_SPECS]
    assert not hasattr(app, "DAILY_RECAP_PAGE_TITLE")


def test_the_two_retired_panel_modules_are_still_on_disk():
    """TJ-1 item 7 / TJ-8: unregistered now, deleted only after the gates pass.
    A builder who deletes them here breaks the reviewer's reproduction."""
    assert (SCRIPTS_DIR / "ui" / "panels" / "market_journal_panel.py").is_file()
    assert (SCRIPTS_DIR / "ui" / "panels" / "daily_recap_panel.py").is_file()


def test_the_new_page_modules_exist_where_the_packet_puts_them():
    assert (SCRIPTS_DIR / "ui" / "panels" / "day_review_panel.py").is_file()
    assert (SCRIPTS_DIR / "ui" / "services" / "day_review_service.py").is_file()
