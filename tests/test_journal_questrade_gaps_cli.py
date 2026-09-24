"""The Questrade gaps CLI: lists FAILED days and mismatches, retries only on --apply."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_coverage  # noqa: E402
import journal_questrade_gaps  # noqa: E402
import journal_reconcile  # noqa: E402
from journal_store import JournalStore  # noqa: E402

FAILED_DAYS = ("2026-06-10", "2026-06-11", "2026-08-13")


def _seeded(tmp_path: Path) -> Path:
    db = tmp_path / "trade_journal.sqlite3"
    store = JournalStore(db)
    store.initialize_schema()
    for day in FAILED_DAYS:
        journal_coverage.mark_coverage(
            store, broker="QUESTRADE", account_number="51830546", day=day,
            status=journal_coverage.FAILED, source="QT_API",
            message=journal_questrade_gaps.EXECUTIONS_MISSING_REASON,
        )
    journal_coverage.mark_coverage(
        store, broker="QUESTRADE", account_number="51830546", day="2026-06-12",
        status=journal_coverage.COVERED, source="QT_API",
    )
    journal_reconcile.store_report(store, {
        "checked_at": "2026-09-23T02:00:37",
        "mismatched": [{
            "account_number": "29347316", "symbol": "AAL", "journal_quantity": -50.0,
            "broker_quantity": 0.0, "kind": "JOURNAL_OPEN_BROKER_FLAT",
        }],
    })
    return db


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _never(*_args):
    raise AssertionError("the broker must not be contacted")


def test_the_default_run_lists_failed_days_and_mismatches_and_writes_nothing(tmp_path, capsys):
    db = _seeded(tmp_path)
    before = _sha(db)

    assert journal_questrade_gaps.main(["--db", str(db)], fetch=_never) == 0

    assert _sha(db) == before
    out = capsys.readouterr().out
    assert "Questrade FAILED import days: 3" in out
    assert "2026-06-10, 2026-06-11, 2026-08-13" in out
    assert "statement import fixes" in out
    assert "Reconciliation mismatches: 1" in out
    assert "AAL" in out and "JOURNAL_OPEN_BROKER_FLAT" in out


def test_reimport_without_apply_fetches_nothing(tmp_path, capsys):
    db = _seeded(tmp_path)
    before = _sha(db)

    assert journal_questrade_gaps.main(["--db", str(db), "--reimport"], fetch=_never) == 0

    assert _sha(db) == before
    assert "would retry 3 account-day(s)" in capsys.readouterr().out


def test_reimport_apply_retries_each_failed_day_once_and_backs_up(tmp_path, capsys):
    db = _seeded(tmp_path)
    fetched = []

    def fake_fetch(broker, account, day):
        fetched.append((broker, account, day.isoformat()))
        return 0

    assert journal_questrade_gaps.main(["--db", str(db), "--reimport", "--apply"], fetch=fake_fetch) == 0

    assert sorted(fetched) == [("QUESTRADE", "51830546", day) for day in FAILED_DAYS]
    assert list(tmp_path.glob("*.pre-qt-reimport-*.bak"))
    assert journal_questrade_gaps.gap_report(db)["failed_days"] == []
    assert "3 imported, 0 still failed" in capsys.readouterr().out


def test_reimport_apply_refuses_the_live_folder_without_the_traders_flag(tmp_path, monkeypatch):
    db = _seeded(tmp_path)
    import journal_reclassify

    monkeypatch.setattr(journal_reclassify, "_is_live_store", lambda _path: True)
    before = _sha(db)

    code = journal_questrade_gaps.main(["--db", str(db), "--reimport", "--apply"], fetch=_never)

    assert code == journal_questrade_gaps.EXIT_REFUSED_TO_START
    assert _sha(db) == before
