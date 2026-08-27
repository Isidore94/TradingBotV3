"""§12: tables stretch, and identifiers elide in the MIDDLE.

The 2026-08-26 live session found this on two pages and the trader's words were
"the data seems useful i just cant read it": Weekend Prep ▸ Focus pick review
rendered `human_foc…` in the FIRST column of three tables, so every row read the
same, and AWAY Recap's ranked swings truncated to `1. FROG …` while two thirds
of a 4K window sat empty.

Both are one rule, and it is applied through one helper
(`ui.widgets.data_table.apply_width_rule`) rather than per panel:

* the widest TEXT column takes the slack, numeric and badge columns keep their
  measured width, and the last section is not the only one that stretches;
* long identifiers elide in the MIDDLE so the distinguishing TAIL survives, the
  full value is the tooltip, and the split is deterministic;
* an elision that leaves every row reading the same is a rendering defect - so
  two different keys must never render as the same string.
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

from PySide6.QtGui import QFontMetrics, QStandardItem, QStandardItemModel  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

from ui.widgets.data_table import (  # noqa: E402
    MAX_COLUMN_WIDTH,
    MIN_COLUMN_WIDTH,
    DataTable,
    MiddleElideDelegate,
    apply_width_rule,
    apply_width_rule_to_table_widget,
    classify_columns,
    elide_middle,
    looks_numeric,
)


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _model(headers, rows):
    model = QStandardItemModel(len(rows), len(headers))
    model.setHorizontalHeaderLabels(list(headers))
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            model.setItem(row_index, column, QStandardItem(str(value)))
    return model


COHORT_ROWS = [
    ("human_focus_tracking", "LONG", "41", "0.51", "0.18", "1.24"),
    ("veto_v3_sma_incoming", "SHORT", "12", "0.33", "-0.09", "0.71"),
    ("human_focus_tracking_second_dev_breakout", "LONG", "7", "0.42", "0.02", "1.01"),
]
COHORT_HEADERS = ("Cohort", "Side", "n", "Win rate", "Avg return", "PF")


# -- the width half ----------------------------------------------------------


def test_a_wider_window_gives_the_text_column_the_extra_slack(app):
    """The complaint, measured: the same table in a laptop and a 4K viewport.

    1680 x 954 is the existing laptop gate and 2304 x 1392 the primary target
    (§12). The text column must absorb the difference; if it does not, the extra
    width goes nowhere and the page hugs the left edge.
    """
    table = DataTable()
    table.setModel(_model(COHORT_HEADERS, COHORT_ROWS))
    apply_width_rule(table, text_columns=(0,), elide_columns=(0,))

    table.resize(1680, 400)
    table.show()
    narrow = table.horizontalHeader().sectionSize(0)
    table.resize(2304, 400)
    wide = table.horizontalHeader().sectionSize(0)
    table.hide()

    assert narrow > 0
    assert wide > narrow, (
        "the text column did not take the extra width - this is exactly the "
        f"defect §3.4 A reports ({narrow}px at 1680, {wide}px at 2304)"
    )


def test_numeric_and_badge_columns_keep_their_measured_width(app):
    table = DataTable()
    table.setModel(_model(COHORT_HEADERS, COHORT_ROWS))
    apply_width_rule(table, text_columns=(0,))
    header = table.horizontalHeader()

    table.resize(1680, 400)
    table.show()
    before = [header.sectionSize(column) for column in range(1, 6)]
    table.resize(2304, 400)
    after = [header.sectionSize(column) for column in range(1, 6)]
    table.hide()

    assert before == after, "a numeric column absorbed slack that belongs to the text column"
    for width in after:
        assert MIN_COLUMN_WIDTH <= width <= MAX_COLUMN_WIDTH


def test_the_last_section_is_not_the_only_one_that_stretches(app):
    table = DataTable()
    table.setModel(_model(COHORT_HEADERS, COHORT_ROWS))
    apply_width_rule(table, text_columns=(0,))
    header = table.horizontalHeader()

    assert header.stretchLastSection() is False
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(5) != QHeaderView.ResizeMode.Stretch


def test_the_widest_text_column_is_found_without_being_named(app):
    """A caller that names nothing still gets the rule, deterministically."""
    table = DataTable()
    table.setModel(_model(COHORT_HEADERS, COHORT_ROWS))
    apply_width_rule(table)
    header = table.horizontalHeader()

    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch


def test_a_numbers_only_table_stretches_its_last_section_rather_than_nothing(app):
    """No text column is not a licence to hug the left edge."""
    table = DataTable()
    table.setModel(_model(("a", "b"), [("1", "2"), ("3", "4")]))
    apply_width_rule(table)

    assert table.horizontalHeader().stretchLastSection() is True


def test_numeric_classification_covers_the_shapes_these_tables_hold():
    for value in ("12", "-3.5", "$1,204", "(0.42)", "88%", "2.3x", "+7"):
        assert looks_numeric(value), value
    for value in ("human_focus_tracking", "LONG", "1. FROG above prev high", "", "n/a"):
        assert not looks_numeric(value), value


def test_an_empty_column_is_treated_as_text_not_frozen_narrow(app):
    table = DataTable()
    table.setModel(_model(("Symbol", "Note"), [("FROG", ""), ("OKTA", "")]))

    assert classify_columns(table) == [True, True]


def test_an_empty_table_is_safe(app):
    table = DataTable()
    table.setModel(_model(COHORT_HEADERS, []))
    apply_width_rule(table, text_columns=(0,))  # must not raise

    table_without_model = DataTable()
    apply_width_rule(table_without_model)  # must not raise


# -- the elision half --------------------------------------------------------


def test_the_elision_keeps_the_tail(app):
    """`human_f…tracking`, never `human_foc…` - the tail is the identity."""
    metrics = QFontMetrics(DataTable().font())
    for width in (70, 110, 160):
        rendered = elide_middle("human_focus_tracking", metrics, width)
        assert rendered != "human_focus_tracking", width
        assert "…" in rendered, rendered
        head, _, tail = rendered.partition("…")
        assert tail, f"nothing survived after the ellipsis at {width}px: {rendered!r}"
        assert "human_focus_tracking".endswith(tail), rendered
        assert "human_focus_tracking".startswith(head), rendered


def test_the_elision_is_deterministic(app):
    metrics = QFontMetrics(DataTable().font())
    first = elide_middle("human_focus_tracking", metrics, 70)
    second = elide_middle("human_focus_tracking", metrics, 70)

    assert first == second


def test_two_different_long_keys_never_elide_to_the_same_string(app):
    """The defect §12 names: an elision that leaves every row reading the same."""
    metrics = QFontMetrics(DataTable().font())
    keys = (
        "human_focus_tracking",
        "human_focus_review",
        "human_focus_tracking_second_dev_breakout",
    )
    for width in (60, 80, 110, 160):
        rendered = [elide_middle(key, metrics, width) for key in keys]
        assert len(set(rendered)) == len(keys), (width, rendered)


def test_the_identifier_column_gets_the_middle_elide_delegate(app):
    table = DataTable()
    table.setModel(_model(COHORT_HEADERS, COHORT_ROWS))
    apply_width_rule(table, text_columns=(0,), elide_columns=(0,))

    assert isinstance(table.itemDelegateForColumn(0), MiddleElideDelegate)
    assert not isinstance(table.itemDelegateForColumn(1), MiddleElideDelegate)


def test_the_delegate_is_reused_across_refreshes(app):
    table = DataTable()
    table.setModel(_model(COHORT_HEADERS, COHORT_ROWS))
    apply_width_rule(table, text_columns=(0,), elide_columns=(0,))
    first = table.itemDelegateForColumn(0)
    apply_width_rule(table, text_columns=(0,), elide_columns=(0,))

    assert table.itemDelegateForColumn(0) is first


def test_a_table_widget_carries_the_full_value_as_a_tooltip(app):
    table = QTableWidget(len(COHORT_ROWS), len(COHORT_HEADERS))
    table.setHorizontalHeaderLabels(list(COHORT_HEADERS))
    for row_index, row in enumerate(COHORT_ROWS):
        for column, value in enumerate(row):
            table.setItem(row_index, column, QTableWidgetItem(str(value)))
    apply_width_rule_to_table_widget(table, text_columns=(0,), elide_columns=(0,))

    for row_index, row in enumerate(COHORT_ROWS):
        assert table.item(row_index, 0).toolTip() == row[0]
    assert table.horizontalHeader().sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch

