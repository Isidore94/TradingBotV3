"""System Health rebuilt every cell of three tables every 15 seconds.

G-P1.4. `_fill` called `setRowCount` and then constructed a fresh
`QTableWidgetItem` for every cell, and the checks table did the same inline.
On a 15-second timer that is a steady churn of Qt objects the trader never
asked for - and it is also where the scroll position went: a rebuilt table
jumps back to the top, so a trader reading the bottom of the jobs list was
pulled away from it mid-read, every fifteen seconds, with no way to tell why.

The repair is to write into the cells that are already there and only create
what is genuinely new. Same rows, same text, same colours - fewer objects, and
the view stays where the trader put it.

What these tests deliberately do NOT assert: that the tables became faster.
They assert the churn is gone and the view survives, which is what can be
proved off a real desk.
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
pytest.importorskip("PySide6", reason="the health panel is a Qt panel")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _check(index: int, status: str = "healthy") -> dict:
    return {
        "id": f"check_{index}",
        "label": f"Component {index}",
        "status": status,
        "summary": f"summary {index}",
        "updated_at": "12:29",
        "source": "somewhere.json",
        "details": {"n": index},
    }


def _payload(count: int = 25, *, status: str = "healthy") -> dict:
    return {
        "status": status,
        "generated_at": "2026-08-26T12:30:00-07:00",
        "market_phase": "regular",
        "market_session": "06:30-13:00",
        "summary": {"healthy": count, "degraded": 0, "unhealthy": 0, "unknown": 0, "total": count},
        "checks": [_check(i) for i in range(count)],
        "jobs": [
            {
                "job": f"job_{i}",
                "status": "OK",
                "last_success": "13:00",
                "attempts": 1,
                "last_run": "13:00",
                "duration_s": 1.0,
                "detail": f"detail {i}",
            }
            for i in range(20)
        ],
    }


def _panel():
    from ui.panels.health_panel import HealthPanel

    return HealthPanel(refresh_interval_ms=60_000)


def _identities(table) -> list[int]:
    return [
        id(table.item(row, column))
        for row in range(table.rowCount())
        for column in range(table.columnCount())
    ]


def test_an_unchanged_refresh_reuses_the_cells_it_already_has():
    """The churn itself. Same data in, same cell objects still there."""
    panel = _panel()
    try:
        panel.set_payload(_payload())
        before = _identities(panel.table)
        assert before, "the table rendered nothing"

        panel.set_payload(_payload())
        after = _identities(panel.table)

        assert after == before, (
            "the checks table replaced every cell on an unchanged refresh"
        )
    finally:
        panel.shutdown()


def test_the_jobs_table_reuses_its_cells_too():
    panel = _panel()
    try:
        panel.set_payload(_payload())
        before = _identities(panel.jobs_table)
        assert before

        panel.set_payload(_payload())
        assert _identities(panel.jobs_table) == before
    finally:
        panel.shutdown()


def test_a_refresh_does_not_pull_the_trader_away_from_where_they_were_reading():
    """Scroll position survives. This is the visible half of the defect."""
    panel = _panel()
    try:
        panel.set_payload(_payload(60))
        # The table has to be shorter than its contents, with a sized viewport,
        # before it can scroll at all. Deliberately WITHOUT `show()`: showing a
        # child of an unshown panel leaves a live top-level widget behind, and
        # a later Qt test in the same process then dies on it - this test
        # segfaulted `test_qt_alert_capture` two files later before it was
        # written this way.
        panel.table.setFixedHeight(120)
        panel.table.viewport().resize(400, 120)
        _app.processEvents()
        bar = panel.table.verticalScrollBar()
        bar.setValue(bar.maximum())
        parked = bar.value()
        assert parked > 0, "the table did not scroll; the test proves nothing"

        panel.set_payload(_payload(60))

        assert panel.table.verticalScrollBar().value() == parked, (
            "the refresh scrolled the trader back to the top"
        )
    finally:
        panel.shutdown()


def test_the_selected_check_still_survives_a_refresh():
    """Already true by id, and it must stay true through the rewrite."""
    panel = _panel()
    try:
        panel.set_payload(_payload(25))
        panel.table.selectRow(7)
        selected = panel.table.item(7, 1).text()

        panel.set_payload(_payload(25))

        assert panel.table.item(panel.table.currentRow(), 1).text() == selected
    finally:
        panel.shutdown()


def test_new_and_removed_rows_are_still_rendered_correctly():
    """A diff that only ever updates would show stale rows after a shrink."""
    panel = _panel()
    try:
        panel.set_payload(_payload(25))
        assert panel.table.rowCount() == 25

        panel.set_payload(_payload(4))
        assert panel.table.rowCount() == 4
        assert panel.table.item(3, 1).text() == "Component 3"

        panel.set_payload(_payload(30))
        assert panel.table.rowCount() == 30
        assert panel.table.item(29, 1).text() == "Component 29"
        # Every cell of a grown table is populated, not left as a None hole.
        assert all(
            panel.table.item(row, column) is not None
            for row in range(panel.table.rowCount())
            for column in range(panel.table.columnCount())
        )
    finally:
        panel.shutdown()


def test_changed_text_and_tone_still_reach_the_cell():
    """Reusing a cell must not mean keeping its old contents."""
    panel = _panel()
    try:
        panel.set_payload(_payload(3))
        assert panel.table.item(0, 0).text() == "HEALTHY"
        first = panel.table.item(0, 0)

        changed = _payload(3)
        changed["checks"][0]["status"] = "unhealthy"
        changed["checks"][0]["summary"] = "it broke"
        panel.set_payload(changed)

        assert panel.table.item(0, 0) is first, "the cell was replaced, not updated"
        assert panel.table.item(0, 0).text() == "UNHEALTHY"
        assert panel.table.item(0, 2).text() == "it broke"
    finally:
        panel.shutdown()


# ==========================================================================
# the audit thread must not outlive the panel it emits into
# ==========================================================================
def test_shutdown_joins_the_audit_thread():
    """Found by a segfault two test files away, not by this panel's own tests.

    Constructing a `HealthPanel` schedules a refresh, which starts a daemon
    thread that emits a Qt signal back into the panel. `shutdown` stopped the
    timer and left that thread running, so it could emit into a panel whose C++
    half had already been freed - an access violation, which the
    `except RuntimeError` around the emit cannot catch because it is not a
    Python exception. Four runs in six crashed `test_qt_alert_capture` merely
    because a HealthPanel had been constructed earlier in the same process.
    """
    import threading

    panel = _panel()
    panel.refresh()
    panel.shutdown()

    thread = getattr(panel, "_audit_thread", None)
    assert thread is None or not thread.is_alive(), (
        "shutdown returned while the audit thread was still running"
    )
    assert not any(
        t.name == "qt-health-audit" and t.is_alive() for t in threading.enumerate()
    ), "an audit thread outlived its panel"
