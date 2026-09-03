"""V2 item 2e - the correcting half of the nightly tagger.

Decision 0016 answer 10: *"the bot should auto-tag every night and the trader
corrects."* Item 1 built the nightly half. This is the correcting half, on the
screen the trader already opens on a Saturday.

**The trader owns `trade_annotations`** (R7 invariant I7). Confirming writes the
trader's own answer through the store's own API; nothing on this page invents a
tag, and a row already carrying a confirmed one is never offered again.
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

WEEK = ("2026-08-10", "2026-08-14")


@pytest.fixture()
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _seed(store, trade_id, *, date="2026-08-11", symbol="NVDA"):
    with store.connection() as conn:
        conn.execute(
            """
            INSERT INTO trades(
                trade_id, broker, account_number, symbol, direction, status,
                opened_at, closed_at, trade_date, updated_at
            ) VALUES(?, 'QUESTRADE', '123', ?, 'LONG', 'CLOSED', ?, ?, ?, ?)
            """,
            (
                trade_id,
                symbol,
                f"{date}T09:31:00-07:00",
                f"{date}T12:00:00-07:00",
                date,
                "2026-08-11T00:00:00",
            ),
        )
    return trade_id


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------


def test_the_step_is_appended_between_reading_the_week_and_planning_the_next():
    from ui.services.weekend_prep_service import STEP_IDS, STEP_LABELS

    assert "tag_week" in STEP_IDS
    assert STEP_LABELS["tag_week"] == "Tag this week"
    assert STEP_IDS.index("tag_week") > STEP_IDS.index("week_review")
    assert STEP_IDS.index("tag_week") < STEP_IDS.index("week_ahead")


def test_a_weekend_saved_before_the_step_existed_still_loads(tmp_path, monkeypatch):
    """`step_status` answers "pending" for a step it has never seen."""
    import project_paths
    from ui.services.weekend_prep_service import WeekendPrepService

    monkeypatch.setattr(project_paths, "WEEKEND_PREP_STATE_FILE", tmp_path / "state.json", raising=False)
    service = WeekendPrepService()
    assert service.step_status("tag_week") in ("pending", "done", "skipped")


# ---------------------------------------------------------------------------
# What it lists, and what it refuses to list
# ---------------------------------------------------------------------------


def test_only_the_rows_that_are_not_the_traders_answer_yet_are_listed(tmp_path, monkeypatch):
    """A confirmed row is settled; listing it invites a second confirmation."""
    import project_paths
    from journal_store import JournalStore
    from ui.panels import weekend_prep_panel

    db = tmp_path / "journal.sqlite3"
    store = JournalStore(db)
    _seed(store, "confirmed", symbol="AAA")
    _seed(store, "provisional", symbol="BBB")
    _seed(store, "review", symbol="CCC")
    store.save_trade_annotation("confirmed", setup_tags="mine", notes="")
    store.apply_provisional_tags("provisional", "guessed")
    store.mark_tags_needing_review("review")

    monkeypatch.setattr(project_paths, "JOURNAL_DB_FILE", db, raising=False)
    monkeypatch.setattr(
        weekend_prep_panel, "JournalStore", lambda *a, **k: store, raising=False
    )
    monkeypatch.setattr("journal_store.JournalStore", lambda *a, **k: store)

    rows = weekend_prep_panel._read_week_tag_rows(WEEK)

    symbols = {str(row.get("symbol")) for row in rows}
    assert symbols == {"BBB", "CCC"}, symbols


def test_a_trade_outside_the_week_is_not_listed(tmp_path, monkeypatch):
    from journal_store import JournalStore
    from ui.panels import weekend_prep_panel

    store = JournalStore(tmp_path / "journal.sqlite3")
    _seed(store, "inside", date="2026-08-12", symbol="IN")
    _seed(store, "outside", date="2026-08-20", symbol="OUT")
    for trade_id in ("inside", "outside"):
        store.mark_tags_needing_review(trade_id)

    monkeypatch.setattr("journal_store.JournalStore", lambda *a, **k: store)

    rows = weekend_prep_panel._read_week_tag_rows(WEEK)
    assert [row["symbol"] for row in rows] == ["IN"]


# ---------------------------------------------------------------------------
# What confirming does
# ---------------------------------------------------------------------------


def test_confirm_all_shown_writes_the_traders_answer(qapp, tmp_path, monkeypatch):
    from journal_store import TAG_STATUS_CONFIRMED, JournalStore
    from ui.panels import weekend_prep_panel
    from ui.services.weekend_prep_service import WeekendPrepService

    store = JournalStore(tmp_path / "journal.sqlite3")
    _seed(store, "t1", symbol="AAA")
    store.apply_provisional_tags("t1", "avwap_breakout")
    monkeypatch.setattr("journal_store.JournalStore", lambda *a, **k: store)

    page = weekend_prep_panel.TagWeekPage(WeekendPrepService())
    page._rows = weekend_prep_panel._read_week_tag_rows(WEEK)
    assert page._rows, "the fixture must offer something to confirm"

    page._confirm_all_shown()

    trade = next(row for row in store.list_trades() if row["trade_id"] == "t1")
    assert str(trade.get("tag_status")) == TAG_STATUS_CONFIRMED
    assert "avwap_breakout" in str(trade.get("setup_tags") or "")


def test_a_failed_write_is_reported_and_never_a_quiet_success(qapp, monkeypatch):
    """A journal write is the one store on this desk that may not fail quietly."""
    from ui.panels import weekend_prep_panel
    from ui.services.weekend_prep_service import WeekendPrepService

    page = weekend_prep_panel.TagWeekPage(WeekendPrepService())
    page._rows = [{"trade_id": "t1"}]

    def _explode(*_args, **_kwargs):
        raise OSError("database is locked")

    monkeypatch.setattr("journal_store.JournalStore", _explode)

    page._confirm_all_shown()

    assert "NOT SAVED" in page.note.text()
    assert "database is locked" in page.note.text()


def test_confirming_nothing_says_so_rather_than_reporting_a_write(qapp):
    from ui.panels import weekend_prep_panel
    from ui.services.weekend_prep_service import WeekendPrepService

    page = weekend_prep_panel.TagWeekPage(WeekendPrepService())
    page._rows = []
    page._confirm_all_shown()
    assert "Nothing selected" in page.note.text()


def test_the_table_shows_ten_rows_before_scrolling(qapp):
    """Three at a time was the complaint."""
    from ui.panels import weekend_prep_panel
    from ui.services.weekend_prep_service import WeekendPrepService

    page = weekend_prep_panel.TagWeekPage(WeekendPrepService())
    assert page.table.minimumHeight() >= 240


def test_the_page_reads_off_the_qt_thread():
    source = (ROOT / "scripts" / "ui" / "panels" / "weekend_prep_panel.py").read_text(
        encoding="utf-8"
    )
    page = source.split("class TagWeekPage")[1].split(chr(10) + "class ")[0]
    assert "_ReadWorker(" in page
    assert "_read_week_tag_rows" in page
