"""S14: Health lists the journal's traded names the scan never sees, once a week."""

from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

import project_paths  # noqa: E402
import ui.panels.health_panel as hp  # noqa: E402


def _journal(path: Path, trades: list[tuple[str, str, str]]) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE trades (symbol TEXT, security_type TEXT, trade_date TEXT)")
        conn.executemany("INSERT INTO trades VALUES (?, ?, ?)", trades)
        conn.commit()


def _stores(tmp_path, monkeypatch, trades, bars=("AAPL", "MRNA")):
    db = tmp_path / "trade_journal.sqlite3"
    _journal(db, trades)
    bars_dir = tmp_path / "daily_bars"
    bars_dir.mkdir()
    for symbol in bars:
        (bars_dir / f"{symbol}.parquet").write_bytes(b"")
    monkeypatch.setattr(project_paths, "JOURNAL_DB_FILE", db)
    monkeypatch.setattr(project_paths, "MASTER_AVWAP_DAILY_BARS_DIR", bars_dir)
    hp._TRADED_GAP_CACHE.clear()
    return db, bars_dir


TRADES = [
    ("SPCX", "STK", "2026-09-21"),
    ("DRAM", "STK", "2026-09-22"),
    ("DRAM261218P00050000", "OPT", "2026-09-23"),
    ("MRNA", "STK", "2026-09-21"),
    ("ZZZ", "STK", "2026-01-02"),   # older than the lookback
    ("USD.CAD", "CASH", "2026-09-21"),
]


def test_traded_names_without_daily_bars_are_listed(tmp_path, monkeypatch):
    _stores(tmp_path, monkeypatch, TRADES)
    row = hp.traded_names_missing_check(today=date(2026, 9, 26))
    assert row["status"] == "degraded"
    assert row["details"]["missing"] == ["DRAM", "SPCX"]
    assert row["details"]["week_of"] == "2026-09-21"
    assert "DRAM, SPCX" in row["summary"]


def test_all_traded_names_in_the_scan_is_healthy(tmp_path, monkeypatch):
    _stores(tmp_path, monkeypatch, [("MRNA", "STK", "2026-09-21")])
    row = hp.traded_names_missing_check(today=date(2026, 9, 26))
    assert row["status"] == "healthy" and row["details"]["missing"] == []


def test_the_list_is_fixed_for_the_week_and_moves_the_next(tmp_path, monkeypatch):
    db, _bars = _stores(tmp_path, monkeypatch, TRADES)
    first = hp.traded_names_missing_check(today=date(2026, 9, 22))
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("INSERT INTO trades VALUES ('NEWX', 'STK', '2026-09-24')")
        conn.commit()
    assert hp.traded_names_missing_check(today=date(2026, 9, 26)) == first
    later = hp.traded_names_missing_check(today=date(2026, 9, 28))
    assert "NEWX" in later["details"]["missing"]


def test_no_journal_is_unknown_not_green(tmp_path, monkeypatch):
    monkeypatch.setattr(project_paths, "JOURNAL_DB_FILE", tmp_path / "missing.sqlite3")
    monkeypatch.setattr(project_paths, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    hp._TRADED_GAP_CACHE.clear()
    assert hp.traded_names_missing_check(today=date(2026, 9, 26))["status"] == "unknown"


def test_the_audit_worker_appends_the_row(tmp_path, monkeypatch):
    _stores(tmp_path, monkeypatch, TRADES)
    monkeypatch.setattr(hp, "universe_floor_check", lambda: hp._health_row("u", "U", "healthy", "", ""))
    monkeypatch.setattr(hp, "ib_status_check", lambda _bot: hp._health_row("i", "I", "healthy", "", ""))
    payload = hp._with_universe_and_ib_checks({"status": "healthy", "checks": [], "summary": {}}, None)
    ids = [row["id"] for row in payload["checks"]]
    assert "traded_names_universe" in ids
    hp._TRADED_GAP_CACHE.clear()
