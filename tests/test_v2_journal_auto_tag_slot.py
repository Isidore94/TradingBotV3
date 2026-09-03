"""V2 item 1 - the nightly auto-tagger, and the badge that asks for review.

Decision 0016 answer 10: *"Journaling's slow part is tagging. The bot should
auto-tag every night and the trader corrects."*

P6a built the whole machine and left it as a command the trader had to remember
to run. This packet runs it, right after `journal_import` and before everything
else, and puts the resulting review queue where the trader can see it.
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


DB_NAME = "journal.sqlite3"


@pytest.fixture()
def journal(tmp_path):
    """An empty journal. Trades are seeded the way P6a's own tests seed them.

    `JournalStore` builds trades from EXECUTIONS through `rebuild_trades`; there
    is no `upsert_trade`. Writing rows straight into `trades` is what the P6a
    suite does, and reusing that shape keeps the two suites describing one store.
    """
    from journal_store import JournalStore

    return JournalStore(tmp_path / DB_NAME)


def _closed_trade(store, trade_id, *, symbol="NVDA", date="2026-09-01"):
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
                "2026-09-01T00:00:00",
            ),
        )
    return trade_id


def _candidate(store, trade_id, tag, confidence):
    with store.connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO auto_tag_candidates(
                trade_id, tag, confidence, source, rationale, created_at
            ) VALUES(?, ?, ?, 'setup_tracker', 'seeded', '2026-09-01T00:00:00')
            """,
            (trade_id, tag, float(confidence)),
        )


# ---------------------------------------------------------------------------
# Where the slot sits, and why
# ---------------------------------------------------------------------------


def test_the_slot_runs_after_the_import_and_before_everything_else():
    """The import puts the night's trades in; every cohort slot reads them.

    Ahead of the import it would tag yesterday's trades; after the cohorts it
    would hand them a journal one night stale.
    """
    from ai_jobs.runner import default_slots

    names = [slot.name for slot in default_slots()]

    assert names[0] == "journal_import"
    assert names[1] == "journal_auto_tag"
    # PAIRWISE for the rest - an index assertion has been edited by four packets.
    for later in ("ai_summary", "veto_cohort_grading", "evidence_report"):
        assert names.index("journal_auto_tag") < names.index(later), later


def test_the_slot_calls_no_model_and_reserves_five_minutes():
    from ai_jobs.runner import default_slots

    slot = next(item for item in default_slots() if item.name == "journal_auto_tag")
    assert slot.reserve_minutes == 5.0
    assert "no model" in slot.description
    assert "confirmed" in slot.description


# ---------------------------------------------------------------------------
# What it does to the journal
# ---------------------------------------------------------------------------


def test_a_second_run_the_same_night_applies_nothing_new(journal, tmp_path):
    """Idempotent by construction: `build_plan` skips a row it already tagged."""
    from ai_jobs.journal_auto_tag import run_journal_auto_tag

    _closed_trade(journal, "t1")

    _candidate(journal, "t1", "avwap_breakout", 0.91)

    first = run_journal_auto_tag(db_path=tmp_path / DB_NAME)
    second = run_journal_auto_tag(db_path=tmp_path / DB_NAME)

    assert first["status"] == "ok" and second["status"] == "ok"
    assert "0 applied" in second["reason"], second["reason"]


def test_a_confirmed_row_survives_the_nightly_run(journal, tmp_path):
    """The trader's own answer. Nothing here improves on it."""
    from journal_store import TAG_STATUS_CONFIRMED

    from ai_jobs.journal_auto_tag import run_journal_auto_tag

    _closed_trade(journal, "t1")
    _candidate(journal, "t1", "the_taggers_guess", 0.95)
    # The trader's own tag, written through the surface the trader uses.
    journal.save_trade_annotation("t1", setup_tags="my_own_setup", notes="")

    run_journal_auto_tag(db_path=tmp_path / DB_NAME)

    trade = next(row for row in journal.list_trades() if row["trade_id"] == "t1")
    assert "my_own_setup" in str(trade.get("setup_tags") or "")
    assert str(trade.get("tag_status")) == TAG_STATUS_CONFIRMED


def test_a_failed_write_is_reported_and_never_a_quiet_success(tmp_path, monkeypatch):
    """A journal WRITE is the one store on this desk that may not fail quietly.

    Every other evidence store swallows a failed append, because losing the
    evidence must not cost the event. Here a tag that silently did not land is a
    trade the trader will believe is tagged.

    MEASURED, not assumed: `JournalStore` creates its parent directory, so a
    path that does not exist yet is not a failure and would have tested nothing.
    """
    import journal_bulk_tag

    from ai_jobs.journal_auto_tag import run_journal_auto_tag

    def _explode(*_args, **_kwargs):
        raise OSError("database is locked")

    monkeypatch.setattr(journal_bulk_tag, "build_plan", _explode)

    result = run_journal_auto_tag(db_path=tmp_path / DB_NAME)

    assert result["status"] == "failed"
    assert "database is locked" in result["reason"]


# ---------------------------------------------------------------------------
# The badge
# ---------------------------------------------------------------------------


def test_the_count_matches_the_store(journal, tmp_path):
    from ai_jobs.journal_auto_tag import trades_awaiting_review

    _closed_trade(journal, "t1")
    _closed_trade(journal, "t2", symbol="AMD")
    _closed_trade(journal, "t3", symbol="TSLA")
    journal.save_trade_annotation("t1", setup_tags="x", notes="")  # the trader's
    journal.mark_tags_needing_review("t2")  # the tagger could not decide

    # t2 needs review; t3 was never tagged at all and so is not in the queue
    # until the tagger has looked at it; t1 is the trader's own.
    assert trades_awaiting_review(tmp_path / DB_NAME) == 1


def test_the_count_never_raises_and_never_costs_the_page(tmp_path):
    """A badge is a convenience; a number nobody can compute is not worth a
    broken nav bar."""
    from ai_jobs.journal_auto_tag import trades_awaiting_review

    assert trades_awaiting_review(tmp_path / "nothing" / "here.db") == 0


def test_the_badge_is_computed_off_the_qt_thread_and_only_once_shown():
    """Two rules, and the second was learned the hard way.

    It opens SQLite and walks every trade, so it runs on a worker - not work a
    page build may do. And it starts from `showEvent`, not `__init__`: a thread
    that starts during construction runs while a test is still monkeypatching
    the journal's module globals, and it made an unrelated journal test fail
    from a hundred tests away - green alone, red in the suite.
    """
    source = (ROOT / "scripts" / "ui" / "app.py").read_text(encoding="utf-8")

    assert "ReadWorker(_count, self)" in source
    assert "def showEvent" in source
    show = source.split("def showEvent", 1)[1].split("def closeEvent", 1)[0]
    assert "_start_tag_review_badge()" in show

    # Never on the construction path, and never inline anywhere.
    build = source.split("def _start_tag_review_badge", 1)[0]
    assert "self._start_tag_review_badge()" not in build
    assert "trades_awaiting_review()" not in build

    # And the reader is joined when the window closes.
    assert "_join_tag_review_badge()" in source


def test_zero_leaves_the_label_alone(qtbot=None):
    """"Journal (0 to review)" would be a badge for nothing to do."""
    from PySide6.QtWidgets import QApplication, QPushButton

    QApplication.instance() or QApplication([])

    from ui.app import PAGE_SPECS, MainWindow

    index = next(i for i, spec in enumerate(PAGE_SPECS) if spec.title == "Journal")
    button = QPushButton("Journal")

    class _Fake:
        nav_buttons = [QPushButton("") for _ in PAGE_SPECS]

    fake = _Fake()
    fake.nav_buttons[index] = button
    MainWindow._apply_tag_review_badge(fake, 0)
    assert button.text() == "Journal"

    MainWindow._apply_tag_review_badge(fake, 12)
    assert button.text() == "Journal (12 to review)"
