"""Packet G2a item 3 - the Journal's Trades table and Weekend Prep's Tag Week
tables name their text column.

The trader's report (GUI review, 2026-09-06): on a 3,456 px window the Trades
table uses a fraction of the width, option symbols and tag lists clip, and the
right-hand third is empty. `docs/archive/GUI_REDESIGN_PLAN_2026-08-25.md` §12
already says what should happen and `ui/widgets/data_table.py` already
implements it once - the two pages in this packet simply never call it.

**Every test here drives the panel's own render seam.** Trades goes through
`TradesTab.reload()` -> `_populate_table()`, the one fill in that file; Tag Week
goes through `TagWeekPage._on_rows_ready()` -> `_render()` ->
`_render_missing_risk()`, which is the slot the page's `_ReadWorker` calls. No
test here asserts on source text, and none opens the live journal or any live
store: `journal_feed`'s readers and the page's payload are patched, so the only
thing under test is what the render leaves on the widgets.

Widths are read from the widgets after a real resize + `show()` + one event
loop turn, because a `QHeaderView` that has never been laid out reports the
sizes of an 638 px default viewport rather than of the desk's window.

The column indices below are taken from the panels' OWN header lists
(`TAG_WEEK_COLUMNS`, `MISSING_RISK_COLUMNS`, the Trades tab's header labels),
never written as literals - a column inserted tomorrow must move the test, not
break it silently.
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

pytestmark = pytest.mark.qt
pytest.importorskip("PySide6", reason="the Qt desk needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QHeaderView  # noqa: E402

#: The trader's monitor, and the width the GUI review was written against.
DESK_WIDTH_PX = 3456
DESK_HEIGHT_PX = 2160

#: A real OCC option symbol: a six-character padded root, then the expiry, the
#: right and the eight-digit strike. That is 21 characters, not the 22 the
#: packet says - the identity ("260918C00230000") is in the TAIL, which is the
#: whole reason this column elides in the middle.
OPTION_SYMBOL = "AAPL  260918C00230000"

#: A realistic multi-tag cell. `setup_tags` is comma-joined free text.
LONG_TAGS = "avwap_reclaim, gap_and_go, held_the_open_range, earnings_drift"


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _lay_out(widget) -> None:
    """Give the widget the desk's width for real, then let Qt lay it out.

    `resize()` alone is not enough: `QHeaderView.sectionSize` answers from the
    viewport, and an unshown widget's viewport is still its default 638 px, so
    a Stretch column would measure ~660 px on a 3,456 px window and the
    assertions below would be about nothing.
    """
    widget.resize(DESK_WIDTH_PX, DESK_HEIGHT_PX)
    widget.show()
    QApplication.processEvents()


def _sizes(table) -> list[int]:
    header = table.horizontalHeader()
    return [int(header.sectionSize(c)) for c in range(table.columnCount())]


def _modes(table) -> list:
    header = table.horizontalHeader()
    return [header.sectionResizeMode(c) for c in range(table.columnCount())]


def _column(headers, name: str) -> int:
    """The index of a column BY NAME, from the panel's own header list."""
    labels = list(headers)
    assert name in labels, f"{name!r} is not one of {labels!r}"
    return labels.index(name)


def _trades_headers(table) -> list[str]:
    return [
        table.horizontalHeaderItem(c).text() for c in range(table.columnCount())
    ]


def _split_three_to_two(tab) -> None:
    """Put the tab's splitter where the tab's own code asks for it.

    `TradesTab.__init__` declares `setStretchFactor(0, 3)` / `(1, 2)` - the
    table is meant to be three fifths of the tab. Qt does not honour that at
    show time: measured offscreen at 3,456 px the splitter opens
    `[1347, 2105]`, so the detail pane - the one the GUI review called "blank
    vertical space with editors exposed and no selection" - takes 61% of the
    desk and the table gets 39%. **That is a real defect and it is NOT G2a's**
    (the packet says nothing else in the tab moves), so the tests below enact
    the 3:2 the code already declares rather than measuring around it. Take the
    splitter out of this helper and every width below shrinks by 35%.
    """
    from PySide6.QtWidgets import QSplitter

    splitter = tab.findChild(QSplitter)
    assert splitter is not None, "the Trades tab is meant to be a splitter"
    width = splitter.width()
    splitter.setSizes([width * 3 // 5, width - width * 3 // 5])
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# The Journal's Trades tab
# ---------------------------------------------------------------------------


def _trade(**overrides):
    """One `JournalTrade` as `journal_feed.load_trades` hands it over.

    Modelled on a real row: `raw` carries every key the render reads, PRESENT
    and empty where the journal has nothing, never absent. `reconcile_status`
    is present and empty - a row that HAS the key set to `NEEDS_REVIEW` already
    carries a tooltip, and the width rule must not overwrite one, which is why
    the option row below deliberately does not have it.
    """
    from ui.models.journal import JournalTrade

    raw = {
        "trade_id": "T1",
        "symbol": "AAPL",
        "currency": "USD",
        "net_pnl": 1204.0,
        "net_pnl_cad": 1650.0,
        "net_pnl_usd": 1204.0,
        "planned_risk": 1000.0,
        "tag_status": "",
        "setup_tags": "",
        "reconcile_status": "",
    }
    raw.update(overrides.pop("raw", {}))
    fields = {
        "trade_id": raw["trade_id"],
        "trade_date": "2026-09-01",
        "symbol": str(raw["symbol"]),
        "direction": "LONG",
        "status": "CLOSED",
        "quantity": 100.0,
        "tags": str(raw.get("setup_tags") or ""),
        "raw": raw,
    }
    fields.update(overrides)
    return JournalTrade(**fields)


def _three_trades():
    """Three rows: a long option symbol, a long tag list, and a plain one."""
    option = _trade(
        trade_id="T-OPT",
        symbol=OPTION_SYMBOL,
        tags="earnings_hold",
        raw={
            "trade_id": "T-OPT",
            "symbol": OPTION_SYMBOL,
            "setup_tags": "earnings_hold",
            "reconcile_status": "",
        },
    )
    tagged = _trade(
        trade_id="T-TAGS",
        symbol="NVDA",
        tags=LONG_TAGS,
        raw={
            "trade_id": "T-TAGS",
            "symbol": "NVDA",
            "setup_tags": LONG_TAGS,
            "reconcile_status": "",
        },
    )
    plain = _trade(
        trade_id="T-PLAIN",
        symbol="AMD",
        direction="SHORT",
        tags="",
        raw={
            "trade_id": "T-PLAIN",
            "symbol": "AMD",
            "setup_tags": "",
            "reconcile_status": "",
            "net_pnl": -240.5,
            "net_pnl_cad": -330.0,
            "net_pnl_usd": -240.5,
            "planned_risk": 1000.0,
            "currency": "USD",
            "tag_status": "",
        },
    )
    return [option, tagged, plain]


@pytest.fixture
def trades_tab(app, monkeypatch):
    """A real `TradesTab` over a real `JournalHeader`, with the feed patched.

    `autoload=False` plus the three patched readers means no store of any kind
    is opened - not the live journal, not a temp one. `load_trades` is the seam
    `reload()` calls, so `_populate_table` runs for real over fixture rows.
    """
    import journal_fx
    from ui.panels.journal.header import JournalHeader
    from ui.panels.journal.trades_tab import TradesTab
    from ui.services import journal_feed

    monkeypatch.setattr(journal_fx, "manual_usd_rate", lambda *a, **k: None)
    monkeypatch.setattr(journal_feed, "account_tree", lambda *a, **k: [])
    monkeypatch.setattr(journal_feed, "tag_names", lambda *a, **k: [])

    rows: list = []
    monkeypatch.setattr(journal_feed, "load_trades", lambda **_kwargs: list(rows))

    header = JournalHeader(autoload=False)
    tab = TradesTab(header)

    def render(trades):
        rows[:] = list(trades)
        _lay_out(tab)
        _split_three_to_two(tab)
        tab.reload()
        QApplication.processEvents()
        return tab

    tab._g2a_render = render  # noqa: SLF001 - the fixture's own handle
    yield tab
    tab.hide()
    tab.deleteLater()
    header.deleteLater()
    QApplication.processEvents()


def test_the_trades_table_gives_the_tags_column_the_slack_at_desk_width(trades_tab):
    """Item 3.1 - Tags takes the slack; nothing else is over the 260 px cap.

    On a 3,456 px window the eight columns share ~800 px today and the rest of
    the row is blank. `MAX_COLUMN_WIDTH` is 260, so with the rule applied the
    seven budgeted columns cannot exceed 260 and `Tags` gets what is left -
    well over 1,000 px even after the detail pane takes its share of the
    splitter.
    """
    from ui.widgets.data_table import MAX_COLUMN_WIDTH

    tab = trades_tab._g2a_render(_three_trades())
    table = tab.table
    headers = _trades_headers(table)
    tags = _column(headers, "Tags")

    assert table.rowCount() == 3
    sizes = _sizes(table)
    assert sizes[tags] >= 1000, (
        f"Tags is {sizes[tags]} px on a {DESK_WIDTH_PX} px desk; "
        f"widths were {dict(zip(headers, sizes))}"
    )
    for index, name in enumerate(headers):
        if index == tags:
            continue
        assert sizes[index] <= MAX_COLUMN_WIDTH, (
            f"{name} is {sizes[index]} px, over the {MAX_COLUMN_WIDTH} px cap; "
            f"widths were {dict(zip(headers, sizes))}"
        )
    assert _modes(table)[tags] == QHeaderView.ResizeMode.Stretch
    assert table.horizontalHeader().stretchLastSection() is False
    # ALL of the slack, not merely some of it: the table fills its own viewport
    # and Tags is exactly what the seven budgeted columns did not take. A rule
    # that leaves a gap, or that lets `stretchLastSection` share the space,
    # fails here rather than passing on a floor.
    budgeted = sum(size for index, size in enumerate(sizes) if index != tags)
    assert sizes[tags] == table.viewport().width() - budgeted


def test_a_long_option_symbol_middle_elides_and_keeps_its_tail_in_the_tooltip(
    trades_tab,
):
    """Item 3.2 - the Symbol column carries the identity in its TAIL.

    `AAPL  260918C00230000` and `AAPL  260918C00240000` differ in the last four
    characters, so Qt's default end elision renders them identically. The
    column gets `MiddleElideDelegate` and every item on it carries the full
    value as its tooltip.
    """
    from ui.widgets.data_table import MiddleElideDelegate

    tab = trades_tab._g2a_render(_three_trades())
    table = tab.table
    headers = _trades_headers(table)
    symbol = _column(headers, "Symbol")

    assert table.item(0, symbol).text() == OPTION_SYMBOL
    assert len(OPTION_SYMBOL) == 21, "the fixture must stay a real OCC symbol"
    assert isinstance(table.itemDelegateForColumn(symbol), MiddleElideDelegate), (
        "Symbol has "
        f"{type(table.itemDelegateForColumn(symbol)).__name__}, not the middle-elide delegate"
    )
    assert table.item(0, symbol).toolTip() == OPTION_SYMBOL


def test_an_empty_trades_render_still_names_tags_as_the_text_column(trades_tab):
    """Item 3.3 - the assignment is by NAME, not by what is in the table.

    With zero rows every column measures its header only, `classify_columns`
    calls all eight of them text, and `_widest_text_column` breaks the tie on
    the LOWEST index - so a rule applied without `text_columns=` would hand the
    slack to `Date`. Naming the column is what makes the empty table look like
    the full one.
    """
    tab = trades_tab._g2a_render([])
    table = tab.table
    headers = _trades_headers(table)
    tags = _column(headers, "Tags")

    assert table.rowCount() == 0
    modes = _modes(table)
    assert modes[tags] == QHeaderView.ResizeMode.Stretch
    stretching = [headers[i] for i, mode in enumerate(modes)
                  if mode == QHeaderView.ResizeMode.Stretch]
    assert stretching == ["Tags"], f"these columns stretch on an empty table: {stretching}"
    assert _sizes(table)[tags] >= 1000


# ---------------------------------------------------------------------------
# Weekend Prep - Tag Week
# ---------------------------------------------------------------------------


def _tag_week_rows():
    """`_read_week_tag_rows`' own row shape, hand-written.

    Old rows have the key PRESENT and empty (`setup_tags` on a `needs_review`
    row is the real example - 132 of the week's rows carried one), so the
    fixture spells every key on every row.
    """
    return [
        {
            "trade_id": "t-opt",
            "trade_date": "2026-09-01T00:00:00",
            "symbol": OPTION_SYMBOL,
            "tag_status": "provisional",
            "setup_tags": "earnings_hold",
            "net_pnl": 1204.0,
            "in_review_week": True,
            "current_tags": "earnings_hold",
        },
        {
            "trade_id": "t-tags",
            "trade_date": "2026-09-02T00:00:00",
            "symbol": "NVDA",
            "tag_status": "provisional",
            "setup_tags": LONG_TAGS,
            "net_pnl": -240.5,
            "in_review_week": True,
            "current_tags": LONG_TAGS,
        },
        {
            "trade_id": "t-blank",
            "trade_date": "2026-08-28T00:00:00",
            "symbol": "AMD",
            "tag_status": "needs_review",
            "setup_tags": "",
            "net_pnl": None,
            "in_review_week": False,
            "current_tags": "",
        },
    ]


def _missing_risk_rows():
    return [
        {
            "trade_id": "r-opt",
            "trade_date": "2026-09-01T00:00:00",
            "symbol": OPTION_SYMBOL,
            "direction": "LONG",
            "net_pnl": 1204.0,
            "setup_tags": LONG_TAGS,
            "missing_risk_total": 3,
        },
        {
            "trade_id": "r-two",
            "trade_date": "2026-08-31T00:00:00",
            "symbol": "NVDA",
            "direction": "SHORT",
            "net_pnl": -240.5,
            "setup_tags": "gap_and_go",
            "missing_risk_total": 3,
        },
        {
            "trade_id": "r-three",
            "trade_date": "2026-08-30T00:00:00",
            "symbol": "AMD",
            "direction": "LONG",
            "net_pnl": None,
            "setup_tags": "",
            "missing_risk_total": 3,
        },
    ]


@pytest.fixture
def tag_week_page(app, tmp_path, monkeypatch):
    """A real `TagWeekPage` whose state file is a temp file and whose payload
    is handed straight to the worker's own slot - no store is opened."""
    import project_paths
    from ui.panels import weekend_prep_panel
    from ui.services.weekend_prep_service import WeekendPrepService

    monkeypatch.setattr(
        project_paths, "WEEKEND_PREP_STATE_FILE", tmp_path / "state.json", raising=False
    )
    page = weekend_prep_panel.TagWeekPage(WeekendPrepService())

    def render(tags, missing_risk):
        _lay_out(page)
        page._on_rows_ready(
            {"tags": list(tags), "missing_risk": list(missing_risk), "coverage": ""}
        )
        QApplication.processEvents()
        return page

    page._g2a_render = render  # noqa: SLF001 - the fixture's own handle
    yield page
    page.shutdown()
    page.hide()
    page.deleteLater()
    QApplication.processEvents()


def test_tag_week_and_missing_risk_tables_name_their_text_column(tag_week_page):
    """Item 3.4 - the same three assertions on both of this page's tables.

    The text column is `Tag` on BOTH. `MISSING_RISK_COLUMNS` is
    `("Date", "Symbol", "Direction", "Net", "Tag")` - it carries no description
    or reason column at all, so `Tag` is the only free-text one and it is the
    column named here. (The packet's "whichever column carries the trade's
    description / reason" describes a column that does not exist; see the
    handoff.)

    `TAG_WEEK_COLUMNS` likewise has SIX entries, not the packet's five - `Week`
    was added when the list widened past the current week - which is why every
    index below is looked up by name.
    """
    from ui.panels import weekend_prep_panel
    from ui.widgets.data_table import MAX_COLUMN_WIDTH, MiddleElideDelegate

    page = tag_week_page._g2a_render(_tag_week_rows(), _missing_risk_rows())

    for table, headers in (
        (page.table, weekend_prep_panel.TAG_WEEK_COLUMNS),
        (page.risk_table, weekend_prep_panel.MISSING_RISK_COLUMNS),
    ):
        names = list(headers)
        tag = _column(names, "Tag")
        symbol = _column(names, "Symbol")
        assert table.columnCount() == len(names)
        assert table.rowCount() == 3

        # (a) the slack goes to Tag and every other column is inside the cap.
        sizes = _sizes(table)
        assert sizes[tag] >= 1000, (
            f"{names} Tag is {sizes[tag]} px on a {DESK_WIDTH_PX} px desk; "
            f"widths were {dict(zip(names, sizes))}"
        )
        for index, name in enumerate(names):
            if index == tag:
                continue
            assert sizes[index] <= MAX_COLUMN_WIDTH, (
                f"{names} {name} is {sizes[index]} px, over the "
                f"{MAX_COLUMN_WIDTH} px cap"
            )
        assert _modes(table)[tag] == QHeaderView.ResizeMode.Stretch
        assert table.horizontalHeader().stretchLastSection() is False

        # (b) the option symbol middle-elides and keeps its whole value.
        assert table.item(0, symbol).text() == OPTION_SYMBOL
        assert isinstance(
            table.itemDelegateForColumn(symbol), MiddleElideDelegate
        ), f"{names} Symbol has no middle-elide delegate"
        assert table.item(0, symbol).toolTip() == OPTION_SYMBOL

    # (c) an empty render still names Tag - by NAME, not by content.
    page = tag_week_page._g2a_render([], [])
    for table, headers in (
        (page.table, weekend_prep_panel.TAG_WEEK_COLUMNS),
        (page.risk_table, weekend_prep_panel.MISSING_RISK_COLUMNS),
    ):
        names = list(headers)
        tag = _column(names, "Tag")
        assert table.rowCount() == 0
        modes = _modes(table)
        stretching = [names[i] for i, mode in enumerate(modes)
                      if mode == QHeaderView.ResizeMode.Stretch]
        assert stretching == ["Tag"], (
            f"{names} - these columns stretch on an empty table: {stretching}"
        )
        assert _sizes(table)[tag] >= 1000


# ---------------------------------------------------------------------------
# The golden: widths only
# ---------------------------------------------------------------------------


#: Every cell of the Trades table over `_three_trades()`, captured from the
#: render on `main` at `a1dab8fa` - BEFORE any width rule exists. The header
#: opens on `CURRENCY_MODES[0]`, which is `CAD`, so the P&L column is the
#: BOOKED `net_pnl_cad` and the R column is `net_pnl_cad / planned_risk` - both
#: read straight off the old code's output rather than re-derived here.
TRADES_GOLDEN = [
    [
        "2026-09-01",
        OPTION_SYMBOL,
        "LONG",
        "CLOSED",
        "100",
        "1,650.00 CAD",
        "1.65R",
        "earnings_hold",
    ],
    [
        "2026-09-01",
        "NVDA",
        "LONG",
        "CLOSED",
        "100",
        "1,650.00 CAD",
        "1.65R",
        LONG_TAGS,
    ],
    [
        "2026-09-01",
        "AMD",
        "SHORT",
        "CLOSED",
        "100",
        "-330.00 CAD",
        "-0.33R",
        "",
    ],
]


def test_the_trades_rows_texts_and_order_are_unchanged_by_the_width_rule(trades_tab):
    """Item 3.5 - the golden. **This test is GREEN by design, before and after.**

    Unlike tests 1-4 it does NOT fail on `main`, and it is not supposed to: it
    is the guard that says the layout lane changed layout and nothing else. The
    literal above was captured from the render on `main` at `a1dab8fa`, so a
    builder who reaches past widths - re-sorting the rows, reformatting a
    number, moving a column - turns it red.
    """
    tab = trades_tab._g2a_render(_three_trades())
    table = tab.table

    assert _trades_headers(table) == [
        "Date", "Symbol", "Dir", "Status", "Qty", "P&L", "R", "Tags",
    ]
    rendered = [
        [table.item(row, column).text() for column in range(table.columnCount())]
        for row in range(table.rowCount())
    ]
    assert rendered == TRADES_GOLDEN
    assert tab.tag_filter_note.text() == "3 of 3 shown; 0 provisional"
