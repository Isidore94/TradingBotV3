"""P8 P2 B: Sunday bulk confirm of suggested setups (trader's go 2026-09-25).

Live: 1 confirmed setup, 139 `needs_review`, 58 `provisional`. The Trades tab's
"Confirm suggested setups" lists every needs_review / provisional trade with the
Mentor's suggestion and why, and one Confirm writes the ticked rows through the
journal writer. Golden: a trade the trader already confirmed is never touched.

Hand-built journal: AAA provisional (tracker evidence), BBB needs_review with a
below-threshold scanner candidate, CCC needs_review with nothing, DDD the
trader's own confirmed tag.
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
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from tj14b_support import REVIEWED, add_round_trip, new_store  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _seed_candidate(store, trade_id, tag, confidence, rationale):
    with store.connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO auto_tag_candidates(
                trade_id, tag, confidence, source, rationale, created_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (trade_id, tag, float(confidence), "setup_tracker", rationale, "2026-09-10T00:00:00"),
        )


def _annotation_row(store, trade_id):
    with store.connection() as conn:
        row = conn.execute(
            "SELECT * FROM trade_annotations WHERE trade_id = ?", (trade_id,)
        ).fetchone()
    return dict(row) if row is not None else None


@pytest.fixture
def journal(tmp_path):
    store = new_store(tmp_path)
    ids = {
        symbol: add_round_trip(store, symbol, day=REVIEWED, entry_hour=hour)
        for symbol, hour in (("AAA", 7), ("BBB", 8), ("CCC", 9), ("DDD", 10))
    }
    _seed_candidate(store, ids["AAA"], "avwap_breakout", 0.80, "tracker named AAA, context 2026-09-10")
    assert store.apply_provisional_tags(ids["AAA"], "avwap_breakout")
    _seed_candidate(store, ids["BBB"], "compression_break", 0.60, "avwap signal BBB, context 2026-09-10")
    assert store.mark_tags_needing_review(ids["BBB"])
    assert store.mark_tags_needing_review(ids["CCC"])
    store.save_trade_annotation(ids["DDD"], setup_tags="my_own_setup", notes="mine")
    _seed_candidate(store, ids["DDD"], "avwap_breakout", 0.95, "tracker named DDD, context 2026-09-10")
    return store, ids


def test_rows_list_needs_review_and_provisional_with_their_why(journal):
    import journal_bulk_confirm as bulk

    store, ids = journal
    rows = {row.symbol: row for row in bulk.rows_to_review(store)}

    assert set(rows) == {"AAA", "BBB", "CCC"}, "the confirmed DDD is never listed"
    assert rows["AAA"].suggestion == "avwap_breakout"
    assert "tracker named AAA" in rows["AAA"].evidence
    assert rows["BBB"].suggestion == "compression_break"
    assert "avwap signal BBB" in rows["BBB"].evidence
    assert rows["CCC"].suggestion == ""
    assert [rows[s].checked_by_default for s in ("AAA", "BBB", "CCC")] == [True, True, False]
    assert rows["AAA"].trade_date == REVIEWED and rows["AAA"].direction
    assert rows["AAA"].net_pnl == pytest.approx(100.0, abs=5.0)


def test_confirm_writes_ticked_rows_and_never_touches_a_confirmed_one(journal):
    import journal_bulk_confirm as bulk

    store, ids = journal
    before_ddd = _annotation_row(store, ids["DDD"])
    rows = {row.symbol: row for row in bulk.rows_to_review(store)}

    result = bulk.confirm(
        store, [(rows["AAA"], "avwap_breakout"), (rows["BBB"], "pullback_sma_reclaim")]
    )

    assert result["confirmed"] == 2 and result["left"] == 1
    assert bulk.summary_text(result) == "2 confirmed, 1 left"
    aaa = store.annotation_state(ids["AAA"])
    assert (aaa["tag_status"], aaa["setup_tags"]) == ("confirmed", "avwap_breakout")
    bbb = store.annotation_state(ids["BBB"])
    assert (bbb["tag_status"], bbb["setup_tags"]) == ("confirmed", "pullback_sma_reclaim")
    assert store.annotation_state(ids["CCC"])["tag_status"] == "needs_review", "unticked is left alone"
    assert _annotation_row(store, ids["DDD"]) == before_ddd, "golden: the trader's row is byte-identical"


def test_a_row_confirmed_meanwhile_is_left_alone(journal):
    import journal_bulk_confirm as bulk

    store, ids = journal
    rows = {row.symbol: row for row in bulk.rows_to_review(store)}
    store.save_trade_annotation(ids["AAA"], setup_tags="trader_said_this", notes="")
    before = _annotation_row(store, ids["AAA"])

    result = bulk.confirm(store, [(rows["AAA"], "avwap_breakout")])

    assert result["confirmed"] == 0
    assert _annotation_row(store, ids["AAA"]) == before
    assert bulk.summary_text(result) == "0 confirmed, 1 refused (already confirmed), 2 left"


def test_a_failed_journal_write_raises(journal, monkeypatch):
    import journal_bulk_confirm as bulk

    store, _ids = journal
    rows = {row.symbol: row for row in bulk.rows_to_review(store)}

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_trade_annotation", boom)
    with pytest.raises(OSError):
        bulk.confirm(store, [(rows["AAA"], "avwap_breakout")])


@pytest.mark.qt
def test_the_dialog_confirms_ticked_rows_and_says_how_many_are_left(journal, monkeypatch):
    pytest.importorskip("PySide6", reason="the dialog is Qt")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    import trade_mentor_trade_check as check
    from ui.panels.journal import bulk_confirm_dialog as module

    QApplication.instance() or QApplication([])
    store, ids = journal
    dialog = module.BulkConfirmDialog(store_factory=lambda: store, threaded=False)
    dialog.load()

    assert dialog.isModal() is False
    assert dialog.table.rowCount() == 3
    by_symbol = {dialog.table.item(r, module.COL_SYMBOL).text(): r for r in range(3)}
    checks = {s: dialog.table.item(r, module.COL_CHECK).checkState() for s, r in by_symbol.items()}
    assert checks == {"AAA": Qt.Checked, "BBB": Qt.Checked, "CCC": Qt.Unchecked}
    combo = dialog.setup_box(by_symbol["BBB"])
    assert set(check.setup_vocabulary()) <= {combo.itemData(i) for i in range(combo.count())}
    assert "avwap signal BBB" in dialog.table.item(by_symbol["BBB"], module.COL_WHY).text()
    combo.setCurrentIndex(combo.findData("pullback_sma_reclaim"))

    dialog.confirm_button.click()

    assert dialog.status_label.text() == "2 confirmed, 1 left"
    assert store.annotation_state(ids["BBB"])["setup_tags"] == "pullback_sma_reclaim"
    assert store.annotation_state(ids["CCC"])["tag_status"] == "needs_review"
    assert dialog.table.rowCount() == 1, "confirmed rows leave the list"

    # A ticked row with no setup is refused: named in the line, kept, reason shown.
    dialog.table.item(0, module.COL_CHECK).setCheckState(Qt.Checked)
    dialog.confirm_button.click()
    assert dialog.status_label.text() == "0 confirmed, 1 refused (no setup chosen), 1 left"
    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, module.COL_WHY).text() == "refused: no setup chosen"

    # A failed write is loud: a warning box, and the status says FAILED.
    warned = []
    monkeypatch.setattr(module.QMessageBox, "warning", lambda *a, **_k: warned.append(a))

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_trade_annotation", boom)
    dialog.setup_box(0).setCurrentIndex(dialog.setup_box(0).findData("avwap_breakout"))
    dialog.table.item(0, module.COL_CHECK).setCheckState(Qt.Checked)
    dialog.confirm_button.click()
    assert warned and "FAILED" in dialog.status_label.text()


@pytest.mark.qt
def test_the_trades_tab_opens_the_bulk_confirm_non_modal(monkeypatch):
    pytest.importorskip("PySide6", reason="the tab is Qt")
    from PySide6.QtWidgets import QApplication

    from ui.panels.journal import bulk_confirm_dialog as module
    from ui.panels.journal.trades_tab import TradesTab

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(module.BulkConfirmDialog, "load", lambda self: None)
    tab = TradesTab(header=None, threaded=False)

    assert tab.bulk_confirm_button.text() == "Confirm suggested setups"
    tab.bulk_confirm_button.click()
    dialog = tab._bulk_confirm_dialog
    assert isinstance(dialog, module.BulkConfirmDialog)
    assert dialog.isModal() is False
    dialog.close()
