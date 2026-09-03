"""R4 A15 - "Confirm all shown" cannot confirm a blank, and there is now an edit.

`JournalStore.confirm_tags` only flips the LANE. Confirming a `needs_review` row
that carries no tag therefore wrote `confirmed` over a blank, and the nightly
`journal_auto_tag` then found a closed trade with no confirmed tag and marked it
`needs_review` again - forever, every night, for as long as the trade exists.
132 of the live rows are exactly that shape, and no button on this page could
give one of them a tag: the trader could only confirm what the machine guessed,
and for those rows it guessed nothing.

The SQLite writes also ran on the Qt thread - one UPDATE per row on "Confirm all
shown".
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


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture()
def page(qapp):
    from ui.panels import weekend_prep_panel as panel_module

    panel = panel_module.WeekendPrepPanel()
    yield panel._pages["tag_week"]
    panel.shutdown()


def _row(trade_id, *, tag, status="needs_review", symbol="NVDA"):
    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "trade_date": "2026-08-28",
        "tag_status": status,
        "setup_tags": tag,
        "net_pnl": 10.0,
    }


def test_confirm_all_skips_a_row_with_no_tag_and_says_how_many(page, monkeypatch):
    """A confirmed blank is a row the nightly tagger re-flags every night."""
    from ui.panels import weekend_prep_panel as panel_module

    asked: list[tuple] = []
    monkeypatch.setattr(
        panel_module,
        "_confirm_tag_ids",
        lambda ids: asked.append(tuple(ids)) or len(ids),
    )
    page._rows = [
        _row("t1", tag="avwape_retest", status="provisional"),
        _row("t2", tag=""),
        _row("t3", tag="   "),
    ]

    notes: list[str] = []
    monkeypatch.setattr(page.note, "setText", notes.append)

    page._confirm_all_shown()
    _drain(page)

    assert asked == [("t1",)], "an untagged row must never reach the store"
    # Every note the click wrote, because `_on_write_done` reloads and the
    # reload's own "reading..." line lands last.
    assert any("2 row(s) skipped" in note for note in notes), notes
    assert any("1 tag(s) confirmed" in note for note in notes), notes


def test_confirming_only_untagged_rows_writes_nothing_and_points_at_the_edit(page):
    page._rows = [_row("t1", tag=""), _row("t2", tag="")]

    page._confirm_all_shown()

    assert "Nothing to confirm" in page.note.text()
    assert "Edit tag" in page.note.text()
    assert page._write_worker is None, "no write may start for a page of blanks"


def test_the_confirm_writes_run_off_the_qt_thread(page):
    """One UPDATE per row was running on the thread that draws."""
    source = (ROOT / "scripts" / "ui" / "panels" / "weekend_prep_panel.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def _confirm(self", 1)[1].split("\n    def ", 1)[0]

    assert "_ReadWorker(" in body
    assert "JournalStore()" not in body, "the store must be opened on the worker"
    worker_side = source.split("def _confirm_tag_ids(", 1)[1].split("\ndef ", 1)[0]
    assert "JournalStore" in worker_side
    assert "setText" not in worker_side, "the worker must touch no widget"


def test_the_edit_writes_the_traders_own_tag_and_confirms_it(page, monkeypatch):
    """The path 132 untagged rows had no button for."""
    from ui.panels import weekend_prep_panel as panel_module

    written: list[tuple] = []
    monkeypatch.setattr(
        panel_module,
        "_write_trader_tag",
        lambda trade_id, tag: written.append((trade_id, tag)) or 1,
    )
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("my own read", True))
    )
    page._rows = [_row("t9", tag="")]
    page._render()
    page.table.selectRow(0)

    page._edit_selected()
    _drain(page)

    assert written == [("t9", "my own read")]


def test_a_cancelled_edit_writes_nothing(page, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    from ui.panels import weekend_prep_panel as panel_module

    written: list[tuple] = []
    monkeypatch.setattr(
        panel_module, "_write_trader_tag", lambda trade_id, tag: written.append((trade_id, tag))
    )
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("", False)))
    page._rows = [_row("t9", tag="")]
    page._render()
    page.table.selectRow(0)

    page._edit_selected()

    assert written == []
    assert page._write_worker is None


def test_an_edit_with_no_row_selected_says_so(page):
    page._rows = [_row("t1", tag=""), _row("t2", tag="")]
    page._render()
    page.table.clearSelection()

    page._edit_selected()

    assert "Select exactly one trade" in page.note.text()


def test_a_failed_write_is_loud(page):
    """The journal is the one store on this desk that may not fail quietly."""
    page._on_write_failed("database is locked")

    assert "NOT SAVED" in page.note.text()
    assert "database is locked" in page.note.text()


def _drain(page) -> None:
    """Let the page's write worker finish before the test reads the result."""
    from PySide6.QtWidgets import QApplication

    worker = page._write_worker
    if worker is not None:
        worker.wait(2000)
    QApplication.processEvents()
