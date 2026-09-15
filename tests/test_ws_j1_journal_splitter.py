"""WISHLIST sweep packet WS-J1 - the Journal's Trades splitter opens at its
declared 3:2, and nothing sits blank above the table.

Verified offscreen (recon 2026-09-12, this packet): `TradesTab.__init__`
(`scripts/ui/panels/journal/trades_tab.py`) builds a `QSplitter` with
`setStretchFactor(0, 3)` / `(1, 2)` and never calls `setSizes`, so the first
show splits on the children's size hints - measured `[1347, 2105]` on a
3,456 px tab (39%/61%), matching the trader's report and
`tests/test_g2a_named_text_columns.py`'s own comment about the same numbers.

The blank band above the table is a SEPARATE bug in the same file: the
tag-review row's note label (`self.tag_filter_note`, an empty `QLabel`) keeps
Qt's default `(Preferred, Preferred)` size policy. A `QLabel` with a growable
vertical policy is the one item in that row that CAN take extra height, so the
row - not the splitter below it - absorbed almost half the tab (measured
`958 px` of a `2160 px` tab, offscreen). `self.tag_filter` (a `QComboBox`,
always `Fixed` vertically) sat centred inside that inflated row rather than
at the top, which is what a human reading the screenshot calls a "blank band"
between the filter line and the table.

Both are layout-only: no number, sort, read or write in this file changes,
which the golden test at the bottom pins.
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
# No `QT_QPA_FONTDIR` here (unlike the packet's standalone offscreen render
# scripts): this module shares one `QApplication`/font database with the rest
# of the pytest session, and setting it here - before anything else in the
# session has created a `QApplication` - measurably changed font metrics for
# unrelated tests collected later (`test_table_width_rule.py`,
# `test_qt_desk_layout.py`), which is state pollution this packet must not
# cause. Nothing this file asserts depends on which fonts are loaded.

from PySide6.QtWidgets import QApplication, QSplitter  # noqa: E402

#: A width partway between the desk's 3,456 px and a typical windowed size -
#: the packet says "after show at 2400 px wide". The height is the desk
#: monitor's own (3,456x2,160): the blank-band bug is HEIGHT-dependent (a
#: VBoxLayout only over-grows the filter row once there is real spare height
#: to hand it - measured offscreen as no bug at 2,400x1,000 but a 965 px
#: splitter offset at 2,400x2,160), so a short test window would silently not
#: exercise it.
WIDTH_PX = 2400
HEIGHT_PX = 2160

#: The declared ratio (`setStretchFactor(0, 3)` / `(1, 2)`): the table pane
#: is 3/5 of the splitter's width, the detail pane 2/5.
TABLE_FRACTION = 0.6
DETAIL_FRACTION = 0.4
#: "within 2%" per the packet.
TOLERANCE = 0.02


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _lay_out(widget, width: int = WIDTH_PX, height: int = HEIGHT_PX) -> None:
    """Reparent to top-level, size it for real, then let Qt lay it out.

    Reparenting with `setParent(None)` before `show()` (memory: offscreen Qt
    render checks) is what makes a repeated resize in the same test actually
    take effect - a widget already shown as a child keeps its old geometry
    hints otherwise.
    """
    widget.setParent(None)
    widget.resize(width, height)
    widget.show()
    QApplication.processEvents()
    QApplication.processEvents()


def _splitter(tab) -> QSplitter:
    """The tab's splitter, found by TYPE rather than by the `.splitter`
    attribute the fix adds - `__init__` on the un-fixed file only ever binds
    it to a local variable, so a lookup by name would raise `AttributeError`
    before the assertion below ever ran, which proves nothing about the
    layout itself."""
    found = tab.findChild(QSplitter)
    assert found is not None, "the Trades tab is meant to be a splitter"
    return found


@pytest.fixture
def trades_tab(app, monkeypatch):
    """A real `TradesTab` over a real `JournalHeader`, with the feed patched.

    `autoload=False` plus the three patched readers means no store of any kind
    is opened - not the live journal, not a temp one.
    """
    import journal_fx
    from ui.panels.journal.header import JournalHeader
    from ui.panels.journal.trades_tab import TradesTab
    from ui.services import journal_feed

    monkeypatch.setattr(journal_fx, "manual_usd_rate", lambda *a, **k: None)
    monkeypatch.setattr(journal_feed, "account_tree", lambda *a, **k: [])
    monkeypatch.setattr(journal_feed, "tag_names", lambda *a, **k: [])
    monkeypatch.setattr(journal_feed, "load_trades", lambda **_kwargs: [])

    header = JournalHeader(autoload=False)
    tab = TradesTab(header)
    yield tab
    tab.hide()
    tab.deleteLater()
    header.deleteLater()
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# Item 1 - the splitter opens at 3:2
# ---------------------------------------------------------------------------


def test_the_splitter_opens_at_the_declared_three_to_two_ratio_on_first_show(
    trades_tab,
):
    """`setStretchFactor` alone never sets the OPENING sizes - Qt derives those
    from the children's size hints, which is how the trader saw 39/61. This
    fails on the un-fixed file (measured `[1347, 2105]`, i.e. ~39%/61%, on a
    3,456 px tab - well outside the 2% band around 60/40 at any width)."""
    _lay_out(trades_tab)

    sizes = _splitter(trades_tab).sizes()
    total = sum(sizes)
    assert total > 0
    table_frac = sizes[0] / total
    detail_frac = sizes[1] / total

    assert table_frac == pytest.approx(TABLE_FRACTION, abs=TOLERANCE), (
        f"table pane is {table_frac:.3f} of the splitter width ({sizes}); "
        f"wanted {TABLE_FRACTION:.2f} +/- {TOLERANCE}"
    )
    assert detail_frac == pytest.approx(DETAIL_FRACTION, abs=TOLERANCE), (
        f"detail pane is {detail_frac:.3f} of the splitter width ({sizes}); "
        f"wanted {DETAIL_FRACTION:.2f} +/- {TOLERANCE}"
    )


def test_the_three_to_two_ratio_holds_at_more_than_one_width(trades_tab):
    """Not a one-width fluke: the desk's 3,456 px monitor and a smaller
    windowed size both open at 3:2."""
    for width, height in [(3456, 2160), (1600, 900)]:
        _lay_out(trades_tab, width=width, height=height)
        sizes = _splitter(trades_tab).sizes()
        total = sum(sizes)
        table_frac = sizes[0] / total
        assert table_frac == pytest.approx(TABLE_FRACTION, abs=TOLERANCE), (
            f"at {width}x{height}: table pane is {table_frac:.3f} ({sizes})"
        )


def test_a_users_drag_survives_a_refresh_and_a_later_resize(trades_tab):
    """Once the trader drags the handle, `_apply_splitter_ratio` must stand
    down for the rest of the desk session - a `reload()` (the tab's own
    refresh) and a later window resize must not silently put the ratio back
    to 3:2 underneath the trader's own choice.

    The drag is simulated the way Qt itself distinguishes it from a
    programmatic resize: `splitterMoved` fires only for an interactive
    move, never for `setSizes()` - `_on_splitter_moved` is the tab's own
    handler for that signal, called directly here rather than driving a real
    `QTest` mouse drag through an offscreen splitter handle.
    """
    _lay_out(trades_tab)
    assert trades_tab._splitter_user_sized is False

    # A programmatic resize alone (e.g. from `_apply_splitter_ratio` itself)
    # must never look like a drag.
    trades_tab.splitter.setSizes([500, 500])
    assert trades_tab._splitter_user_sized is False

    # The trader's own drag.
    trades_tab.splitter.splitterMoved.emit(500, 1)
    assert trades_tab._splitter_user_sized is True
    dragged_sizes = list(trades_tab.splitter.sizes())

    trades_tab.reload()
    QApplication.processEvents()
    assert list(trades_tab.splitter.sizes()) == dragged_sizes, (
        "a refresh must not move the handle back to 3:2 after a drag"
    )

    trades_tab.resize(1800, 1000)
    QApplication.processEvents()
    total = sum(trades_tab.splitter.sizes())
    table_frac = trades_tab.splitter.sizes()[0] / total
    assert table_frac != pytest.approx(TABLE_FRACTION, abs=TOLERANCE), (
        "a later window resize reset the trader's dragged ratio back to 3:2"
    )


# ---------------------------------------------------------------------------
# Item 2 - no blank band above the table
# ---------------------------------------------------------------------------


def test_no_blank_band_between_the_filter_row_and_the_table(trades_tab):
    """The vertical gap between the tag-review row and the table is under
    8 px. Measured from `tag_filter` (the combobox): it is `Fixed` vertically
    and never grows, so its own bottom edge marks where the row's real
    content ends regardless of whether the row around it is bugged.

    This fails on the un-fixed file: the empty `tag_filter_note` label's
    growable size policy stretches the ROW (not the combobox) to fill nearly
    the whole tab, so the combobox is measured centred far from the row's top
    and the gap to the splitter below balloons into the hundreds of pixels
    (measured 475 px on a 2,400x2,160 offscreen render) rather than shrinking
    to a couple of layout-spacing pixels.
    """
    _lay_out(trades_tab)

    combo_bottom = trades_tab.tag_filter.geometry().bottom()
    splitter_top = _splitter(trades_tab).geometry().top()
    gap = splitter_top - combo_bottom

    assert gap < 8, (
        f"gap between the tag-review row and the table is {gap} px "
        f"(combo bottom={combo_bottom}, splitter top={splitter_top})"
    )


def test_the_filter_row_stays_one_slim_line(trades_tab):
    """The row's own height stays close to a single control's height - the
    row must not silently balloon again if a future edit adds another label
    to it. `tag_filter`'s own height (22 px, `Fixed`) is the floor; a slim
    row is at most a small multiple of that, never hundreds of pixels."""
    _lay_out(trades_tab)

    row_height = _splitter(trades_tab).geometry().top()
    control_height = trades_tab.tag_filter.sizeHint().height()
    assert row_height < control_height * 3, (
        f"the tag-review row occupies {row_height} px above the splitter; "
        f"the combobox itself only needs {control_height} px"
    )


# ---------------------------------------------------------------------------
# Golden - nothing else about the table moved
# ---------------------------------------------------------------------------


def _trade(**overrides):
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


def test_the_trades_table_headers_and_row_count_are_unchanged(trades_tab, monkeypatch):
    """This test is GREEN by design, before and after: it is the guard that
    says the layout lane changed layout and nothing else - no number, sort,
    read or write in `trades_tab.py` moved.
    """
    from ui.panels.journal.trades_tab import TRADES_COLUMNS
    from ui.services import journal_feed

    trades = [_trade(trade_id="T1"), _trade(trade_id="T2", symbol="NVDA")]
    monkeypatch.setattr(journal_feed, "load_trades", lambda **_kwargs: list(trades))

    _lay_out(trades_tab)
    trades_tab.reload()
    QApplication.processEvents()

    headers = [
        trades_tab.table.horizontalHeaderItem(c).text()
        for c in range(trades_tab.table.columnCount())
    ]
    assert headers == list(TRADES_COLUMNS)
    assert trades_tab.table.rowCount() == 2
    assert trades_tab.tag_filter_note.text() == "2 of 2 shown; 0 provisional"
