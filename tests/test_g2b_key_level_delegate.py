"""Packet G2b.3 - the Desk Setups `Key Level / Entry` column keeps BOTH
behaviours: the setups delegate's painting and the middle elision.

ADDED BY THE BUILDER (2026-09-07) on top of the tester's
`tests/test_g2b_named_columns.py`, which asserts only that the column carries a
`MiddleElideDelegate`. That assertion is satisfied by a plain one - and a plain
one would REPLACE `SetupTableDelegate` for this column, so `key_level` alone
would lose the alternating background, the favorite tint, the selection fill
and the hairline separator every other column on its own row still draws.

It is also satisfied by a delegate that never actually elides in the middle:
`SetupTableDelegate._text` hard-codes `ElideRight`, so deferring `paint` to it
without more would keep painting the end elision §12 forbids for an identifier.

So these four tests pin what the isinstance check cannot:

1. the column's delegate is a `SetupTableDelegate` as well as a
   `MiddleElideDelegate`;
2. its `sizeHint` is the setups row height, which is the setups delegate
   answering - Qt's default is shorter;
3. painting a long key level goes through `elide_middle`, with the WHOLE value,
   so the anchor date in the tail survives;
4. the compact profile gets the column back, unelided.
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

from PySide6.QtCore import QRect, Qt  # noqa: E402
from PySide6.QtGui import QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QStyleOptionViewItem,
)

from ui.widgets.data_table import MiddleElideDelegate  # noqa: E402
from ui.widgets.setup_delegate import SetupTableDelegate  # noqa: E402


#: The setups delegate's own row height (`setup_delegate._ROW_HEIGHT`). Read as
#: a floor, not copied as a contract: the point is that the setups delegate is
#: the one answering, and Qt's default row is well under this.
SETUPS_ROW_HEIGHT = 40

LONG_KEY_LEVEL = (
    "$412.50 2nd dev band, anchored 2026-05-12 post-earnings, retested 2026-08-27"
)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _rows():
    from ui.models.setup import SetupRow

    return [
        SetupRow(
            symbol="AMD",
            side="SHORT",
            score=71.2,
            bucket="favorite_setup",
            setup_tags=["breakdown"],
            key_level=LONG_KEY_LEVEL,
            supports=1,
            expected_r=0.61,
            sector="Technology",
            industry="Semiconductors",
            d1_vs_sector=-1.1,
            d1_vs_industry=-0.4,
            last_trade_date="2026-09-04",
            raw={"setup_family": "avwap_breakdown"},
        ),
    ]


def _panel(app, *, profile: str = "full"):
    from ui.panels.master_avwap_panel import MasterAvwapPanel

    panel = MasterAvwapPanel()
    panel.resize(3456, 2160)
    panel.show()
    app.processEvents()
    panel.set_rows(_rows())
    panel.set_column_profile(profile)
    app.processEvents()
    return panel


def _key_level_column() -> int:
    from ui.models.setup_table_model import SetupTableModel

    for index, (key, _label) in enumerate(SetupTableModel.COLUMNS):
        if key == "key_level":
            return index
    raise AssertionError("key_level is not a SetupTableModel column")


def _option(table):
    option = QStyleOptionViewItem()
    option.initFrom(table)
    option.font = table.font()
    option.rect = QRect(0, 0, 160, SETUPS_ROW_HEIGHT)
    return option


def test_the_key_level_delegate_is_both_delegates(app):
    panel = _panel(app)
    try:
        delegate = panel.table.itemDelegateForColumn(_key_level_column())
        assert isinstance(delegate, MiddleElideDelegate), (
            "the elision and the full-value tooltip come from here"
        )
        assert isinstance(delegate, SetupTableDelegate), (
            "a per-column delegate REPLACES the view's delegate, so this one has "
            "to paint the setups row itself - backgrounds, favorite tint, "
            "selection and separator included"
        )
    finally:
        panel.deleteLater()


def test_the_key_level_delegate_still_sizes_a_setups_row(app):
    """The cheap, unambiguous proof that `SetupTableDelegate` is in charge of
    painting: its `sizeHint` floors the row at the setups height, and Qt's
    default `QStyledItemDelegate` does not."""
    panel = _panel(app)
    try:
        table = panel.table
        delegate = table.itemDelegateForColumn(_key_level_column())
        index = table.model().index(0, _key_level_column())
        assert delegate.sizeHint(_option(table), index).height() >= SETUPS_ROW_HEIGHT

        plain = MiddleElideDelegate(table)
        assert plain.sizeHint(_option(table), index).height() < SETUPS_ROW_HEIGHT, (
            "if Qt's default row were already this tall the assertion above "
            "would prove nothing"
        )
    finally:
        panel.deleteLater()


def test_painting_a_long_key_level_elides_it_in_the_middle(app, monkeypatch):
    """`SetupTableDelegate._text` elides RIGHT. The override has to shorten the
    text in the MIDDLE first, with the WHOLE value, or the anchor and retest
    dates in the tail are what gets cut."""
    from ui.panels import master_avwap_panel

    seen: list[tuple[str, int]] = []
    real = master_avwap_panel.elide_middle

    def spy(text, metrics, width):
        result = real(text, metrics, width)
        seen.append((str(text), int(width)))
        return result

    monkeypatch.setattr(master_avwap_panel, "elide_middle", spy)

    panel = _panel(app)
    try:
        table = panel.table
        column = _key_level_column()
        index = table.model().index(0, column)
        assert str(index.data(Qt.ItemDataRole.DisplayRole)) == LONG_KEY_LEVEL

        pixmap = QPixmap(200, SETUPS_ROW_HEIGHT)
        painter = QPainter(pixmap)
        try:
            table.itemDelegateForColumn(column).paint(painter, _option(table), index)
        finally:
            painter.end()

        assert seen, "the key-level paint path never reached `elide_middle`"
        text, width = seen[-1]
        assert text == LONG_KEY_LEVEL, (
            "the elision must see the whole value, not something already cut"
        )
        assert 0 < width < 160, "the elision width must be the padded cell, not the raw rect"

        metrics = table.fontMetrics()
        shortened = real(LONG_KEY_LEVEL, metrics, width)
        assert shortened != LONG_KEY_LEVEL, "the fixture no longer overflows its cell"
        # The tail is the whole reason §12 asks for a MIDDLE elision here: what
        # survives is the END of the value, not an ellipsis where the end was.
        assert "…" in shortened
        assert not shortened.endswith("…"), (
            "this is Qt's end elision, which is what loses the retest date"
        )
        tail = shortened.rsplit("…", 1)[-1]
        assert len(tail) >= 3 and LONG_KEY_LEVEL.endswith(tail), (
            f"the surviving tail {tail!r} is not the end of the real value"
        )
    finally:
        panel.deleteLater()


def test_the_compact_profile_gets_the_column_back(app):
    """G2b.3 is the full profile only: switching back to compact takes the
    per-column delegate off again, so `key_level` at its pinned 116px reads
    exactly as it did before this packet."""
    panel = _panel(app, profile="full")
    try:
        column = _key_level_column()
        assert panel.table.itemDelegateForColumn(column) is not None
        panel.set_column_profile("compact")
        assert panel.table.itemDelegateForColumn(column) is None
    finally:
        panel.deleteLater()
