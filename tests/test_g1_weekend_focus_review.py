"""Packet G1 item G1.4 - Weekend Prep > Focus Review: one table, a selector, a pane.

Nine `QTableWidget`s, each carrying a 260 px ten-row floor, stacked in ONE
`QVBoxLayout` with about twenty labels between them: 9 x 260 = 2,340 px of floor
in roughly 1,900 px of page at 2160. The overlap the trader reported is
arithmetic, and no font size fixes it. G1 moves the nine into a
`QStackedWidget` behind a nine-button view selector, drops the floors on THIS
page so the one visible table takes the height, and puts a read-only detail pane
beside the stack for the selected row.

Every test here drives the real widgets offscreen at the trader's two real
screen sizes and the real render pass (`_on_focus_ready`, the slot the page's
own `_ReadWorker` emits into) with a fixture payload. Nothing asserts on source
text, and no test calls a helper with a hand-written dict where the widget path
exists.

Written RED by the tester before any fix (packet G1, 2026-09-06). A builder may
add to this file; it may not weaken, skip or delete anything in it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# The nine views, in the packet's order: (button label, table attr, note attr)
# ---------------------------------------------------------------------------

FOCUS_VIEWS: tuple[tuple[str, str, str], ...] = (
    ("Week's picks", "table", "note"),
    ("Picks graded", "performance_table", "performance_note"),
    ("Vetoes", "cohort_table", "cohort_note"),
    ("Likes", "like_table", "like_note"),
    ("After-like", "after_like_table", "after_like_note"),
    ("Passes", "pass_table", "pass_note"),
    ("Not-today", "rejection_table", "rejection_note"),
    ("Said vs did", "preference_table", "preference_note"),
    ("Said at the time", "feedback_table", "feedback_note"),
)

#: The two views whose rows the horizon combo actually drives.
HORIZON_VIEWS = frozenset({"Vetoes", "Likes"})

TABLE_ATTRS: tuple[str, ...] = tuple(attr for _, attr, _ in FOCUS_VIEWS)

#: The row count each fixture section puts in its own table. Numbers, not
#: shapes: a render that dropped the horizon filter, or re-read on a view
#: change, moves one of these.
EXPECTED_ROWS: dict[str, int] = {
    "table": 4,
    "performance_table": 3,
    "cohort_table": 5,
    "like_table": 3,
    "after_like_table": 2,
    "pass_table": 4,
    "rejection_table": 3,
    "preference_table": 5,
    "feedback_table": 6,
}

#: A real reason is a paragraph, not a word. 400 characters is the packet's
#: figure and it is what proves the pane shows the long text IN FULL rather
#: than the elided cell.
LONG_REASON = (
    "Vetoed the D1 setup because the anchored VWAP band was already eight days "
    "old and the volume that built it came from a single earnings gap, so the "
    "band is describing one print rather than a value area; the name also sits "
    "under a falling 20-day and the sector tape was red all morning, which is "
    "the combination that has cost me the most this quarter when I took it. "
    "Revisit it if it bases above the anchor for a week and the sector turns."
)
assert len(LONG_REASON) >= 400, len(LONG_REASON)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _process(qapp, rounds: int = 4) -> None:
    for _ in range(rounds):
        qapp.processEvents()


@pytest.fixture
def make_panel(qapp):
    """Build a real `WeekendPrepPanel` offscreen at a real screen size."""
    built = []

    def _build(width: int, height: int):
        from ui.services.weekend_prep_service import STEP_IDS
        from ui.panels import weekend_prep_panel as panel_module

        panel = panel_module.WeekendPrepPanel()
        built.append(panel)
        panel.resize(width, height)
        panel.show()
        _process(qapp)
        # The real path onto the page: the stepper rail, not `setCurrentIndex`.
        panel.rail.setCurrentRow(STEP_IDS.index("focus_review"))
        _process(qapp)
        return panel

    yield _build

    for panel in built:
        try:
            panel.shutdown()
        finally:
            panel.hide()
            panel.deleteLater()
    _process(qapp)


# ---------------------------------------------------------------------------
# Finding the new furniture without pinning an attribute name the packet
# does not give. The BEHAVIOUR is pinned; the private names are the builder's.
# ---------------------------------------------------------------------------


def _selector_buttons(page) -> dict:
    from PySide6.QtWidgets import QToolButton

    wanted = {label for label, _, _ in FOCUS_VIEWS}
    found: dict[str, object] = {}
    for button in page.findChildren(QToolButton):
        text = button.text().strip()
        if text in wanted:
            found[text] = button
    return found


def _require_selector(page) -> dict:
    buttons = _selector_buttons(page)
    missing = [label for label, _, _ in FOCUS_VIEWS if label not in buttons]
    assert not missing, (
        "the Focus Review page has no view selector: no exclusive QToolButton "
        f"is labelled {missing}"
    )
    return buttons


def _select_view(qapp, page, label: str) -> None:
    buttons = _require_selector(page)
    buttons[label].click()
    _process(qapp)


def _detail_pane(page):
    from PySide6.QtWidgets import QTextBrowser

    panes = page.findChildren(QTextBrowser)
    assert len(panes) == 1, (
        "the Focus Review page must hold exactly ONE read-only detail pane "
        f"(a QTextBrowser beside the stack); found {len(panes)}"
    )
    return panes[0]


def _visible_tables(page) -> list[str]:
    return [attr for attr in TABLE_ATTRS if getattr(page, attr).isVisible()]


def _rows_in_viewport(table) -> int:
    step = table.verticalHeader().defaultSectionSize()
    assert step > 0, "a table with no row height cannot be measured"
    return table.viewport().height() // step


def _row_counts(page) -> dict[str, int]:
    return {attr: getattr(page, attr).rowCount() for attr in TABLE_ATTRS}


# ---------------------------------------------------------------------------
# The fixture payload: exactly the dict `_read_everything` returns, so
# `_on_focus_ready` - the slot the page's own worker emits into - is the path
# under test.
# ---------------------------------------------------------------------------


def _cohort_row(cohort: str, side: str, *, n: int, meets: bool, lb: float) -> dict:
    """One cohort row shaped like `human_focus_tracking` writes it.

    Old rows have the key PRESENT and EMPTY, never absent, so `median` and
    `ci` are empty strings on the row that has not matured rather than missing.
    """
    return {
        "cohort": cohort,
        "side": side,
        "horizon": "3",
        "n": str(n),
        "win_rate": "0.55",
        "avg_return": "0.012",
        "median": "0.009" if meets else "",
        "trimmed": "0.010",
        "profit_factor": "1.4",
        "symbols": "7",
        "sessions": "12",
        "top_share": "0.21",
        "ci": "0.01 to 0.02" if meets else "",
        "ci_basis": "block bootstrap" if meets else "",
        "evidence": "graded" if meets else "thin",
        "win_rate_lb": lb,
        "_meets_floor": meets,
        "_sort_value": 0.010,
    }


def _p5_row(cohort: str, side: str, *, meets: str) -> dict:
    return {
        "cohort": cohort,
        "side": side,
        "horizon": "3",
        "n": "18",
        "win_rate": "0.44",
        "avg_return": "-0.004",
        "profit_factor": "0.9",
        "meets_n_floor": meets,
        "evidence": "graded" if meets == "1" else "thin",
    }


def fixture_payload() -> dict:
    """One payload, pinned by hand - never generated by the page it drives."""
    return {
        "week": [
            {
                "date": "2026-08-31", "symbol": "AMD", "side": "long",
                "source": "focus_auto", "h1": "0.4", "h3": "1.1",
                "h5": "", "h10": "", "matured": "2",
            },
            {
                "date": "2026-09-01", "symbol": "NVDA", "side": "long",
                "source": "focus_manual", "h1": "-0.2", "h3": "",
                "h5": "", "h10": "", "matured": "1",
            },
            {
                "date": "2026-09-02", "symbol": "TSLA", "side": "short",
                "source": "strength_board", "h1": "", "h3": "",
                "h5": "", "h10": "", "matured": "0",
            },
            {
                "date": "2026-09-03", "symbol": "SMCI", "side": "long",
                "source": "no pick snapshot", "h1": "0.8", "h3": "0.3",
                "h5": "", "h10": "", "matured": "2",
            },
        ],
        "performance": [
            {
                "cohort": "human_focus_intraday", "side": "long", "horizon": "3",
                "n": "48", "win_rate": "0.52", "avg_return": "0.006",
                "median": "0.004", "profit_factor": "1.2", "symbols": "22",
                "sessions": "31", "ci": "0.00 to 0.01",
                "updated_at": "2026-09-05T04:12:00-04:00",
            },
            {
                "cohort": "human_focus_intraday", "side": "short", "horizon": "3",
                "n": "19", "win_rate": "0.47", "avg_return": "-0.002",
                "median": "", "profit_factor": "0.9", "symbols": "11",
                "sessions": "9", "ci": "",
                "updated_at": "2026-09-05T04:12:00-04:00",
            },
            {
                "cohort": "human_focus_swing_vetted", "side": "long", "horizon": "3",
                "n": "27", "win_rate": "0.59", "avg_return": "0.021",
                "median": "0.018", "profit_factor": "1.7", "symbols": "17",
                "sessions": "24", "ci": "0.01 to 0.03",
                "updated_at": "2026-09-05T04:12:00-04:00",
            },
        ],
        "cohort": [
            _cohort_row("veto_extended_from_anchor", "long", n=31, meets=True, lb=0.41),
            _cohort_row("veto_wrong_sector_tape", "long", n=24, meets=True, lb=0.38),
            _cohort_row("veto_thin_volume", "short", n=17, meets=True, lb=0.33),
            _cohort_row("veto_earnings_in_window", "long", n=6, meets=False, lb=0.11),
            _cohort_row("human_focus_veto", "long", n=78, meets=True, lb=0.45),
        ],
        "like": [
            _cohort_row("avwap_reclaim", "long", n=22, meets=True, lb=0.40),
            _cohort_row("like_unclaimed", "long", n=14, meets=True, lb=0.31),
            _cohort_row("gap_and_go", "short", n=4, meets=False, lb=0.05),
        ],
        "claim_caveat": "Claimed setups come from the pick list, not a free text box.",
        # The WHOLE nightly fact pack, exactly as `_read_after_like_block`
        # returns it - the `after_like` block is one key inside it.
        "after_like": {
            "after_like": {
                "episodes": 61,
                "cells": [
                    {
                        "day_offset": 1, "entry": "open", "n_episodes": 34,
                        "trimmed_mean_r": 0.42, "win_rate": 0.56,
                        "eligible": True,
                    },
                    {
                        "day_offset": 2, "entry": "vwap", "n_episodes": 27,
                        "trimmed_mean_r": 0.18, "win_rate": 0.51,
                        "eligible": True,
                    },
                    {
                        "day_offset": 3, "entry": "open", "n_episodes": 5,
                        "trimmed_mean_r": 1.90, "win_rate": 0.80,
                        "eligible": False,
                    },
                ],
            },
        },
        "pass": [
            _p5_row("pass_all", "long", meets="1"),
            _p5_row("pass_spread_too_wide", "long", meets="1"),
            _p5_row("pass_already_extended", "short", meets="1"),
            _p5_row("pass_no_volume", "long", meets="0"),
        ],
        "rejection": [
            _p5_row("focus__m5_not_today", "long", meets="1"),
            _p5_row("focus__swing_dislike", "long", meets="1"),
            _p5_row("human_focus_rejection", "long", meets="1"),
        ],
        "preference": [
            {
                "session_date": "2026-08-31", "symbol": "AMD", "side": "long",
                "statement": "liked the reclaim off the anchor", "traded": "yes",
                "match_confidence": "high", "journal_r": "1.2",
                "paper_forward_return_h5": "0.031", "trade_id": "T-101",
            },
            {
                "session_date": "2026-09-01", "symbol": "NVDA", "side": "long",
                "statement": "said I would take the pullback", "traded": "yes",
                "match_confidence": "medium", "journal_r": "-0.4",
                "paper_forward_return_h5": "-0.008", "trade_id": "T-102",
            },
            {
                "session_date": "2026-09-02", "symbol": "TSLA", "side": "short",
                "statement": "wanted the fade under VWAP", "traded": "no",
                "match_confidence": "no match", "journal_r": "",
                "paper_forward_return_h5": "0.014", "trade_id": "",
            },
            {
                "session_date": "2026-09-03", "symbol": "SMCI", "side": "long",
                "statement": "named it and skipped it", "traded": "no",
                "match_confidence": "no match", "journal_r": "",
                "paper_forward_return_h5": "0.022", "trade_id": "",
            },
            {
                "session_date": "2026-09-04", "symbol": "MU", "side": "long",
                "statement": "second entry on the same idea", "traded": "yes",
                "match_confidence": "high", "journal_r": "0.6",
                "paper_forward_return_h5": "0.011", "trade_id": "T-101",
            },
        ],
        "feedback": [
            {
                "date": "2026-08-31", "symbol": "AMD", "side": "long",
                "verdict": "like", "category": "intraday", "origin": "vetted",
                "reason": LONG_REASON,
            },
            {
                "date": "2026-09-01", "symbol": "NVDA", "side": "long",
                "verdict": "veto", "category": "intraday", "origin": "auto",
                "reason": "too extended from the anchor",
            },
            {
                "date": "2026-09-01", "symbol": "TSLA", "side": "short",
                "verdict": "not_today", "category": "intraday", "origin": "auto",
                "reason": "",
            },
            {
                "date": "2026-09-02", "symbol": "SMCI", "side": "long",
                "verdict": "like", "category": "swing", "origin": "vetted",
                "reason": "clean higher low on the daily",
            },
            {
                "date": "2026-09-03", "symbol": "MU", "side": "long",
                "verdict": "veto", "category": "swing", "origin": "manual",
                "reason": "earnings inside the window",
            },
            {
                "date": "2026-09-04", "symbol": "INTC", "side": "short",
                "verdict": "dislike", "category": "swing", "origin": "manual",
                "reason": "",
            },
        ],
    }


def _render_fixture(qapp, page) -> dict:
    """Drive the page's real render pass with the fixture payload."""
    payload = fixture_payload()
    page._on_focus_ready(payload)
    _process(qapp)
    return payload


# ---------------------------------------------------------------------------
# 1 - No overflow at 3456 x 2160
# ---------------------------------------------------------------------------


def test_focus_review_page_fits_its_own_height_at_2160(make_panel, qapp):
    """9 x 260 px of floor does not fit in the height 2160 gives the page.

    The packet's stated measure - `page.minimumSizeHint().height() <=
    page.height()` - cannot fail: a Qt child that does not fit is GROWN past
    its container rather than clipped, so the page's own height is always at
    least its minimum. Measured on the current code the page is 2,824 px tall
    inside a panel asked for 2,160, and `WeekendPrepPanel.resize(3456, 2160)`
    comes back 2,934 px tall because the panel cannot go below the minimum
    this one page forces on it. Focus Review is the sole driver: the next
    worst page (Tag week) asks for 700 px.

    So the overflow is measured against the height the trader's screen
    actually has, three ways: the panel fits, the page's minimum fits the
    space above the panel's bottom edge, and every table the page shows is
    inside that space instead of below it.
    """
    from PySide6.QtCore import QPoint, QRect

    screen_height = 2160
    panel = make_panel(3456, screen_height)
    page = panel.focus_review
    _render_fixture(qapp, page)

    assert page.height() > 0, "the page must be laid out to be measured"
    assert panel.height() <= screen_height, (
        "Weekend Prep cannot be shown at 2160: it insists on "
        f"{panel.minimumSizeHint().height()} px of height, and Focus Review is "
        f"the page demanding {page.minimumSizeHint().height()} px of it"
    )

    stack_top = panel.pages.mapTo(panel, QPoint(0, 0)).y()
    available = screen_height - stack_top - panel.layout().contentsMargins().bottom()
    assert available > 0, "the header alone must not eat the screen"
    assert page.minimumSizeHint().height() <= available, (
        f"Focus Review demands {page.minimumSizeHint().height()} px of minimum "
        f"height in the {available} px the panel has for it at 2160"
    )

    rects: dict[str, QRect] = {}
    for attr in TABLE_ATTRS:
        table = getattr(page, attr)
        if not table.isVisible():
            continue
        rects[attr] = QRect(table.mapTo(page, QPoint(0, 0)), table.size())

    below = {
        attr: rect.bottom()
        for attr, rect in rects.items()
        if rect.bottom() > available
    }
    assert not below, (
        f"table(s) drawn past the bottom of a {screen_height} px screen "
        f"(usable {available} px): {below}"
    )

    overlaps = [
        (left, right)
        for index, left in enumerate(sorted(rects))
        for right in sorted(rects)[index + 1 :]
        if rects[left].intersects(rects[right])
    ]
    assert not overlaps, f"visible Focus Review tables overlap: {overlaps}"


# ---------------------------------------------------------------------------
# 2 - Still readable at 2560 x 1440
#
# This REPLACES the Focus-Review half of
# `tests/test_r4_market_journal_page_and_tables.py::
#  test_every_weekend_prep_table_shows_ten_rows`. The old test asserted a 260 px
# minimumHeight on all nine tables of this page; the ten-row promise is now kept
# by the ONE VISIBLE table taking the height, which is what is measured here.
# The other Weekend pages keep the floor and the old assertion.
# ---------------------------------------------------------------------------


def test_the_visible_focus_review_table_shows_ten_rows_at_1440(make_panel, qapp):
    panel = make_panel(2560, 1440)
    page = panel.focus_review
    _render_fixture(qapp, page)

    # The default view first - it needs no selector, so this measures
    # readability rather than the presence of the new furniture.
    default_rows = _rows_in_viewport(page.table)
    assert default_rows >= 10, (
        "the default Focus Review view shows only "
        f"{default_rows} row(s) of the week's picks at 1440"
    )

    thin: list[tuple[str, int]] = []
    for label, table_attr, _note_attr in FOCUS_VIEWS:
        _select_view(qapp, page, label)
        rows = _rows_in_viewport(getattr(page, table_attr))
        if rows < 10:
            thin.append((label, rows))
    assert not thin, f"views showing fewer than ten rows at 1440: {thin}"


# ---------------------------------------------------------------------------
# 3 - One visible table per view
# ---------------------------------------------------------------------------


def test_each_selector_button_shows_exactly_its_own_table_and_note(make_panel, qapp):
    panel = make_panel(3456, 2160)
    page = panel.focus_review
    _render_fixture(qapp, page)

    buttons = _require_selector(page)

    # The default view is the week's picks, before anything is clicked.
    assert _visible_tables(page) == ["table"], (
        "Focus Review must open on Week's picks with one table visible; visible "
        f"today: {_visible_tables(page)}"
    )

    for label, table_attr, note_attr in FOCUS_VIEWS:
        buttons[label].click()
        _process(qapp)
        assert _visible_tables(page) == [table_attr], (
            f"view {label!r} should show only {table_attr}; visible: "
            f"{_visible_tables(page)}"
        )
        assert getattr(page, note_attr).isVisible(), (
            f"view {label!r} hides its own note ({note_attr})"
        )
        assert page.cohort_horizon_input.isVisible() is (label in HORIZON_VIEWS), (
            f"the horizon combo is visible on {label!r} but drives only "
            f"{sorted(HORIZON_VIEWS)}"
        )

    # Exclusive: the buttons are a group, so the last click left exactly one
    # button checked.
    checked = [label for label in buttons if buttons[label].isChecked()]
    assert checked == ["Said at the time"], f"selector is not exclusive: {checked}"


# ---------------------------------------------------------------------------
# 4 - The read is unchanged; switching a view reads nothing
# ---------------------------------------------------------------------------


def test_switching_view_rebuilds_nothing_and_reads_nothing(make_panel, qapp, monkeypatch):
    from ui.panels import weekend_prep_panel as panel_module

    panel = make_panel(3456, 2160)
    page = panel.focus_review
    _render_fixture(qapp, page)

    counts = _row_counts(page)
    assert counts == EXPECTED_ROWS, (
        "one render pass must fill all nine tables with the fixture's own row "
        f"counts; got {counts}"
    )

    reads: list[str] = []
    for name in (
        "_read_veto_cohort",
        "_read_like_cohort",
        "_read_after_like_block",
        "_read_pass_cohort",
        "_read_rejection_cohort",
        "_read_preference_trade_rows",
        "_read_focus_performance",
        "_read_pick_feedback_week",
        "_join_focus_week",
        "_claim_picklist_caveat",
    ):
        monkeypatch.setattr(
            panel_module,
            name,
            lambda *a, _name=name, **k: reads.append(_name) or [],
        )

    for label, _table_attr, _note_attr in FOCUS_VIEWS:
        _select_view(qapp, page, label)
        assert _row_counts(page) == EXPECTED_ROWS, (
            f"selecting {label!r} changed a table's row count: {_row_counts(page)}"
        )

    assert reads == [], f"selecting a view touched a store: {reads}"
    worker = getattr(page, "_worker", None)
    assert worker is None or not worker.isRunning(), (
        "selecting a view started the page's read worker"
    )


# ---------------------------------------------------------------------------
# 5 - The chosen view survives a re-render
# ---------------------------------------------------------------------------


def test_the_chosen_view_survives_a_reload(make_panel, qapp):
    panel = make_panel(3456, 2160)
    page = panel.focus_review
    _render_fixture(qapp, page)

    _select_view(qapp, page, "Passes")
    assert _visible_tables(page) == ["pass_table"]

    _render_fixture(qapp, page)

    assert _visible_tables(page) == ["pass_table"], (
        "a refresh threw the trader back to the default view; visible: "
        f"{_visible_tables(page)}"
    )
    assert page.pass_table.rowCount() == EXPECTED_ROWS["pass_table"]


# ---------------------------------------------------------------------------
# 6 - The detail pane
# ---------------------------------------------------------------------------


def test_the_detail_pane_shows_every_column_of_the_selected_row(make_panel, qapp):
    panel = make_panel(3456, 2160)
    page = panel.focus_review
    _render_fixture(qapp, page)

    pane = _detail_pane(page)
    assert pane.isReadOnly(), "the detail pane is read-only"

    _select_view(qapp, page, "Said at the time")
    table = page.feedback_table
    assert table.item(0, 6).text() == LONG_REASON, (
        "the render must keep the reason whole in the cell - the elision is a "
        "paint-time affair"
    )
    table.selectRow(0)
    _process(qapp)

    text = pane.toPlainText()
    headers = [
        table.horizontalHeaderItem(column).text()
        for column in range(table.columnCount())
    ]
    missing = [header for header in headers if header not in text]
    assert not missing, f"the detail pane omits column(s) {missing}: {text!r}"
    assert LONG_REASON in text, (
        "the detail pane must carry the row's reason IN FULL, not the elided "
        f"cell: {text!r}"
    )

    _select_view(qapp, page, "Passes")
    assert pane.toPlainText().strip() == "", (
        "switching view must clear the detail pane; it still reads "
        f"{pane.toPlainText()!r}"
    )


# ---------------------------------------------------------------------------
# 7 - The floor left THIS page and stayed on the other five
#
# The other half of the replacement in item 2's comment, written where the
# packet can be read against it: `test_every_weekend_prep_table_shows_ten_rows`
# now excludes Focus Review, and this is the test that says the exclusion is
# narrow. A builder that deleted `_ten_row_table` or `TABLE_TEN_ROWS_PX`
# outright, or that left the floors on, fails here.
# ---------------------------------------------------------------------------


def test_the_ten_row_floor_left_focus_review_and_stayed_everywhere_else(
    make_panel, qapp
):
    from PySide6.QtWidgets import QTableWidget
    from ui.panels import weekend_prep_panel as panel_module

    panel = make_panel(3456, 2160)
    page = panel.focus_review

    assert panel_module.TABLE_TEN_ROWS_PX == 260, (
        "the floor is still one constant, for the pages that keep it"
    )

    floored = {
        attr: getattr(page, attr).minimumHeight()
        for attr in TABLE_ATTRS
        if getattr(page, attr).minimumHeight() >= panel_module.TABLE_TEN_ROWS_PX
    }
    assert not floored, (
        "Focus Review tables must carry no ten-row floor - the visible one "
        f"takes the height instead: {floored}"
    )

    focus_tables = set(page.findChildren(QTableWidget))
    others = [
        table
        for table in panel.findChildren(QTableWidget)
        if table not in focus_tables
    ]
    assert others, "the other Weekend pages must still have tables to be about"
    thin = [
        table.columnCount()
        for table in others
        if table.minimumHeight() < panel_module.TABLE_TEN_ROWS_PX
    ]
    assert not thin, (
        f"the floor was removed from a page G1 does not touch: {thin}"
    )


# ---------------------------------------------------------------------------
# 8 and 9 - ADDED BY THE BUILDER (packet G1). Nothing above was weakened.
#
# The packet binds two things the tester's seven do not measure: the new
# furniture is themed by OBJECT NAME and never by a per-widget stylesheet, and
# the horizon combo - which moved onto the selector row - still re-filters both
# cohort tables from what is already in memory, touching no store.
# ---------------------------------------------------------------------------


def test_the_new_furniture_is_themed_by_object_name_not_a_stylesheet(
    make_panel, qapp
):
    from PySide6.QtWidgets import QTextBrowser, QToolButton, QWidget

    panel = make_panel(3456, 2160)
    page = panel.focus_review

    qss = (ROOT / "scripts" / "ui" / "theme.qss").read_text(encoding="utf-8")

    pane = _detail_pane(page)
    assert pane.objectName(), "the detail pane must carry a theme object name"
    assert f"QTextBrowser#{pane.objectName()}" in qss, (
        f"theme.qss has no rule for QTextBrowser#{pane.objectName()}"
    )

    buttons = _require_selector(page)
    names = {button.objectName() for button in buttons.values()}
    assert names and "" not in names, (
        "every selector button must carry a theme object name"
    )
    for name in names:
        assert f"QToolButton#{name}" in qss, (
            f"theme.qss has no rule for QToolButton#{name}"
        )

    # A stylesheet is expensive on the Qt thread; the page must not set one.
    styled = [
        widget.objectName() or widget.__class__.__name__
        for widget in page.findChildren(QWidget)
        if widget.styleSheet()
    ]
    assert not styled, f"per-widget stylesheets on the Focus Review page: {styled}"
    assert not page.styleSheet(), "the page itself must not set a stylesheet"
    assert isinstance(pane, QTextBrowser)
    assert all(isinstance(button, QToolButton) for button in buttons.values())


def test_the_horizon_combo_still_refilters_both_cohort_tables_from_memory(
    make_panel, qapp, monkeypatch
):
    from ui.panels import weekend_prep_panel as panel_module

    panel = make_panel(3456, 2160)
    page = panel.focus_review
    _render_fixture(qapp, page)

    _select_view(qapp, page, "Vetoes")
    assert page.cohort_horizon_input.isVisible()
    assert page.cohort_table.rowCount() == EXPECTED_ROWS["cohort_table"]
    assert page.like_table.rowCount() == EXPECTED_ROWS["like_table"]

    reads: list[str] = []
    for name in ("_read_veto_cohort", "_read_like_cohort"):
        monkeypatch.setattr(
            panel_module,
            name,
            lambda *a, _name=name, **k: reads.append(_name) or [],
        )

    # Every fixture cohort row is at the default horizon, so another horizon
    # empties both tables - which is the filter working, not a lost read.
    other = next(
        horizon
        for horizon in panel_module.COHORT_HORIZONS
        if horizon != panel_module.DEFAULT_COHORT_HORIZON
    )
    page.cohort_horizon_input.setCurrentIndex(
        page.cohort_horizon_input.findData(other)
    )
    _process(qapp)
    assert page.cohort_table.rowCount() == 0
    assert page.like_table.rowCount() == 0

    page.cohort_horizon_input.setCurrentIndex(
        page.cohort_horizon_input.findData(panel_module.DEFAULT_COHORT_HORIZON)
    )
    _process(qapp)
    assert page.cohort_table.rowCount() == EXPECTED_ROWS["cohort_table"]
    assert page.like_table.rowCount() == EXPECTED_ROWS["like_table"]
    assert reads == [], f"changing the horizon touched a store: {reads}"


# ---------------------------------------------------------------------------
# G1 fix round (reviewer NO-GO, 2026-09-06). Added by the builder; nothing
# above this line is weakened, skipped or rewritten.
# ---------------------------------------------------------------------------


def _feedback_payload(rows) -> dict:
    """The fixture payload with a hand-written `feedback` section."""
    payload = fixture_payload()
    payload["feedback"] = rows
    return payload


#: The refreshed week: the SAME number of rows, every value on row 0
#: different. A render that keeps the row count leaves the row SELECTED, so
#: this is the shape that makes a stale pane visible.
REFRESHED_REASON = (
    "Re-read after the overnight grade landed: the band rebuilt on three "
    "sessions of real two-sided volume rather than the earnings print, the "
    "20-day turned up under it, and the sector tape closed green four days "
    "running - the same name, the opposite read, which is exactly the change "
    "the trader must not be shown last week's words for."
)
assert len(REFRESHED_REASON) >= 200, len(REFRESHED_REASON)


def _refreshed_feedback_rows() -> list[dict]:
    original = fixture_payload()["feedback"]
    rows = [dict(row) for row in original]
    rows[0] = {
        "date": "2026-09-05",
        "symbol": "ORCL",
        "side": "short",
        "verdict": "veto",
        "category": "swing",
        "origin": "manual",
        "reason": REFRESHED_REASON,
    }
    assert len(rows) == len(original), "the refresh must keep the row count"
    return rows


def test_the_detail_pane_re_reads_the_row_after_a_refresh(make_panel, qapp):
    """The pane describes the cells on screen NOW, never the previous read.

    Reviewer's reproduction (packet G1, round 1): the pane is wired to
    `itemSelectionChanged` only, and a re-render that keeps the row count
    leaves the row SELECTED without re-emitting it - so the table shows the
    new week and the pane goes on reading the old one, under the same row
    number, with nothing on screen saying which is which.
    """
    panel = make_panel(3456, 2160)
    page = panel.focus_review
    _render_fixture(qapp, page)

    _select_view(qapp, page, "Said at the time")
    page.feedback_table.selectRow(0)
    _process(qapp)

    pane = _detail_pane(page)
    before = pane.toPlainText()
    assert "AMD" in before, before[:400]
    assert LONG_REASON[:60] in before, before[:400]

    # The same page, re-rendered with the same row COUNT and new values.
    page._on_focus_ready(_feedback_payload(_refreshed_feedback_rows()))
    _process(qapp)

    assert page.feedback_table.rowCount() == EXPECTED_ROWS["feedback_table"], (
        "the refresh must keep the row count - otherwise this test is not "
        "reproducing the defect"
    )
    assert page.feedback_table.item(0, 1).text() == "ORCL"

    after = pane.toPlainText()
    assert "ORCL" in after, (
        "the detail pane still describes the PREVIOUS read after a refresh: "
        f"{after[:400]!r}"
    )
    assert REFRESHED_REASON[:60] in after, after[:400]
    assert "AMD" not in after, (
        f"the pane kept a value the refresh replaced: {after[:400]!r}"
    )
    assert LONG_REASON[:60] not in after, after[:400]


def test_a_refresh_that_drops_the_selected_row_empties_the_detail_pane(
    make_panel, qapp
):
    """A selection the new read cannot carry leaves the pane EMPTY, not frozen."""
    panel = make_panel(3456, 2160)
    page = panel.focus_review
    _render_fixture(qapp, page)

    _select_view(qapp, page, "Said at the time")
    last = EXPECTED_ROWS["feedback_table"] - 1
    page.feedback_table.selectRow(last)
    _process(qapp)

    pane = _detail_pane(page)
    assert pane.toPlainText().strip(), "the pane must be filled before the refresh"

    # A shorter week: the selected row no longer exists.
    page._on_focus_ready(_feedback_payload(fixture_payload()["feedback"][:2]))
    _process(qapp)

    assert page.feedback_table.rowCount() == 2
    assert not page.feedback_table.selectedItems(), (
        "the shorter render must have dropped the selection"
    )
    assert not pane.toPlainText().strip(), (
        "the pane must be EMPTY when the refresh dropped the selected row, "
        f"not the row it used to describe: {pane.toPlainText()[:400]!r}"
    )


def test_the_horizon_re_render_also_re_reads_the_detail_pane(make_panel, qapp):
    """The cohort re-render is a render too, and the pane follows it.

    The horizon combo swaps the rows of both cohort tables from memory. When
    the other horizon holds the SAME NUMBER of rows the selection survives, so
    this is the second door onto the same staleness as a refresh - a different
    horizon's numbers under the previous horizon's words.
    """
    import ui.panels.weekend_prep_panel as panel_module

    panel = make_panel(3456, 2160)
    page = panel.focus_review

    other = next(
        horizon
        for horizon in panel_module.COHORT_HORIZONS
        if horizon != panel_module.DEFAULT_COHORT_HORIZON
    )
    payload = fixture_payload()
    at_other = []
    for index, row in enumerate(payload["cohort"]):
        twin = dict(row)
        twin["horizon"] = other
        twin["cohort"] = f"veto_only_at_h{other}_{index}"
        at_other.append(twin)
    # The same COUNT at both horizons: the switch keeps the row selected.
    payload["cohort"] = list(payload["cohort"]) + at_other
    page._on_focus_ready(payload)
    _process(qapp)

    _select_view(qapp, page, "Vetoes")
    assert page.cohort_table.rowCount() == EXPECTED_ROWS["cohort_table"]
    page.cohort_table.selectRow(0)
    _process(qapp)

    pane = _detail_pane(page)
    before = pane.toPlainText()
    assert before.strip()
    first_cohort = page.cohort_table.item(0, 0).text()
    assert first_cohort in before, before[:400]

    page.cohort_horizon_input.setCurrentIndex(
        page.cohort_horizon_input.findData(other)
    )
    _process(qapp)

    assert page.cohort_table.rowCount() == EXPECTED_ROWS["cohort_table"], (
        "both horizons must hold the same row count, or this test is not "
        "reproducing the defect"
    )
    swapped = page.cohort_table.item(0, 0).text()
    assert swapped != first_cohort, "the horizon switch must have swapped the rows"
    after = pane.toPlainText()
    assert swapped in after, (
        "the pane still describes the row of the PREVIOUS horizon: "
        f"{after[:400]!r}"
    )
    assert first_cohort not in after, after[:400]


def test_clicking_the_view_already_shown_is_a_no_op(make_panel, qapp):
    """Re-clicking the current view must not throw the trader's row away.

    An exclusive checkable button still emits `clicked` when it is already
    checked, so the selector re-ran the view change and cleared the selection
    and the pane - a click that changed nothing on screen except the one thing
    the trader was reading.
    """
    panel = make_panel(3456, 2160)
    page = panel.focus_review
    _render_fixture(qapp, page)

    _select_view(qapp, page, "Vetoes")
    page.cohort_table.selectRow(1)
    _process(qapp)

    pane = _detail_pane(page)
    before = pane.toPlainText()
    assert before.strip(), "the pane must be filled before the second click"
    assert {item.row() for item in page.cohort_table.selectedItems()} == {1}

    _select_view(qapp, page, "Vetoes")

    assert _visible_tables(page) == ["cohort_table"]
    assert {item.row() for item in page.cohort_table.selectedItems()} == {1}, (
        "clicking the view already shown cleared the selected row"
    )
    assert pane.toPlainText() == before, (
        "clicking the view already shown emptied the detail pane"
    )

    # Switching to a DIFFERENT view still clears - that rule is unchanged.
    _select_view(qapp, page, "Likes")
    assert not pane.toPlainText().strip()


def test_the_two_cross_reference_notes_name_a_view_not_a_position(
    make_panel, qapp
):
    """Nothing is "above" another table once the nine are a stack.

    The like note pointed at "the veto table above" and the feedback note at
    "the rollup above"; both are now views the trader reaches with a button,
    and a note naming a position that no longer exists is a wrong direction.
    """
    panel = make_panel(3456, 2160)
    page = panel.focus_review
    _render_fixture(qapp, page)

    like_note = page.like_note.text()
    assert "Vetoes view" in like_note, like_note
    assert "table above" not in like_note, like_note

    feedback_note = page.feedback_note.text()
    assert "Picks graded view" in feedback_note, feedback_note
    assert "above" not in feedback_note, feedback_note
