"""Desk snappiness packet 2, item 3: the journal's worst per-click costs.

Three measured problems, all behind one OK button or one checkbox:

* accepting a correction called `rebuild_trades()` -> `refresh_auto_tags()` ->
  `AutoTagger.load_context_rows`, which `json.loads` the **1.08 GB**
  `master_avwap_setup_tracker.json` plus a 73 MB CSV, synchronously on the GUI
  thread;
* `list_trades` opened a fresh sqlite connection PER TRADE to look up that
  trade's regime;
* the filter header re-queried the whole store on every checkbox and combo
  signal, with no debounce.

Threading, batching and caching only. No schema change, no P&L, tax or
statistics change, and every regime block is the same one as before.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import journal_analytics  # noqa: E402
from journal_store import JournalStore  # noqa: E402


# --------------------------------------------------------------- 3b: parsing
class TestTheTrackerIsParsedOncePerFileVersion:
    @staticmethod
    def _tracker(path: Path, count: int = 5) -> None:
        payload = {
            "setups": {
                f"S{index}": {
                    "symbol": f"SYM{index}",
                    "side": "LONG",
                    "scan_date": "2026-08-20",
                    "setup_family": "avwap",
                    "priority_score": 10.0,
                }
                for index in range(count)
            }
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_unchanged_files_parse_once_across_taggers(self, tmp_path, monkeypatch):
        """Two retags in a row - accept a correction, then add an execution -
        parsed a gigabyte twice for byte-identical input."""
        journal_analytics.clear_context_row_cache()
        tracker = tmp_path / "tracker.json"
        self._tracker(tracker)

        parses = []
        real = journal_analytics._load_json

        def counted(path):
            parses.append(Path(path).name)
            return real(path)

        monkeypatch.setattr(journal_analytics, "_load_json", counted)

        def _tagger():
            return journal_analytics.AutoTagger(
                setup_tracker_path=tracker,
                focus_path=tmp_path / "missing_focus.json",
                avwap_signals_path=tmp_path / "missing.csv",
                intraday_bounces_path=tmp_path / "missing2.csv",
            )

        first = _tagger().load_context_rows()
        second = _tagger().load_context_rows()

        assert parses.count("tracker.json") == 1
        assert first == second
        assert len(first) == 5

    def test_a_changed_file_is_parsed_again(self, tmp_path, monkeypatch):
        journal_analytics.clear_context_row_cache()
        tracker = tmp_path / "tracker.json"
        self._tracker(tracker, count=2)

        parses = []
        real = journal_analytics._load_json
        monkeypatch.setattr(
            journal_analytics,
            "_load_json",
            lambda path: (parses.append(Path(path).name), real(path))[1],
        )

        def _tagger():
            return journal_analytics.AutoTagger(
                setup_tracker_path=tracker,
                focus_path=tmp_path / "missing_focus.json",
                avwap_signals_path=tmp_path / "missing.csv",
                intraday_bounces_path=tmp_path / "missing2.csv",
            )

        before = _tagger().load_context_rows()
        # A new scan rewrites the file; a stamp cache that missed this would
        # tag today's trades against yesterday's setups.
        self._tracker(tracker, count=4)
        os.utime(tracker, (0, 0))
        after = _tagger().load_context_rows()

        assert parses.count("tracker.json") == 2
        assert len(after) == 4
        assert len(before) == 2

    def test_a_missing_file_is_not_cached_as_an_answer(self, tmp_path):
        """A source that appears later must be picked up, so "absent" is never
        remembered."""
        journal_analytics.clear_context_row_cache()
        tracker = tmp_path / "tracker.json"

        def _tagger():
            return journal_analytics.AutoTagger(
                setup_tracker_path=tracker,
                focus_path=tmp_path / "missing_focus.json",
                avwap_signals_path=tmp_path / "missing.csv",
                intraday_bounces_path=tmp_path / "missing2.csv",
            )

        assert _tagger().load_context_rows() == []
        self._tracker(tracker, count=3)
        assert len(_tagger().load_context_rows()) == 3


# --------------------------------------------------------------- 3c: regimes
class _CountingStore(JournalStore):
    """A store that counts how many sqlite connections it opens."""

    def connect(self):
        self.connection_count = getattr(self, "connection_count", 0) + 1
        return super().connect()


class TestTheRegimeLookupIsOneQuery:
    @staticmethod
    def _seed(store: JournalStore) -> None:
        with store.connection() as conn:
            for row in (
                ("2026-08-24", "bull", "bull", "trend", "seeded"),
                ("2026-08-26", "", "", "chop", "no carry labels"),
            ):
                conn.execute(
                    "INSERT INTO regimes"
                    " (trade_date, mid_term_regime, short_term_regime, intraday_regime, notes, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    row + ("2026-08-31T00:00:00",),
                )

    def test_the_batch_matches_the_single_date_lookup(self, tmp_path):
        store = JournalStore(tmp_path / "journal.db")
        self._seed(store)
        dates = ["2026-08-23", "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27"]

        batch = store.get_regimes_for_dates(dates)

        for value in dates:
            assert batch[value] == store.get_regime_for_date(value), value

    def test_the_carry_row_is_the_latest_labelled_one_at_or_before_the_date(self, tmp_path):
        store = JournalStore(tmp_path / "journal.db")
        self._seed(store)

        batch = store.get_regimes_for_dates(["2026-08-26", "2026-08-27"])

        # 08-26's own row has no mid/short labels, so both carry 08-24's...
        assert batch["2026-08-26"]["mid_term_regime"] == "bull"
        # ...but intraday and notes come from the EXACT row only.
        assert batch["2026-08-26"]["intraday_regime"] == "chop"
        assert batch["2026-08-27"]["intraday_regime"] == ""

    def test_a_date_before_every_regime_row_carries_nothing(self, tmp_path):
        store = JournalStore(tmp_path / "journal.db")
        self._seed(store)

        batch = store.get_regimes_for_dates(["2020-01-01"])

        assert batch["2020-01-01"] == {
            "mid_term_regime": "",
            "short_term_regime": "",
            "intraday_regime": "",
            "regime_notes": "",
        }

    def test_listing_many_trades_opens_one_connection_for_the_regimes(self, tmp_path):
        """It used to be one connection - with its PRAGMA - per trade."""
        store = _CountingStore(tmp_path / "journal.db")
        self._seed(store)
        with store.connection() as conn:
            for index in range(12):
                conn.execute(
                    "INSERT INTO trades"
                    " (trade_id, broker, account_number, symbol, direction, status, opened_at,"
                    "  trade_date, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"T{index}",
                        "IBKR",
                        "U1234567",
                        "NVDA",
                        "LONG",
                        "CLOSED",
                        f"2026-08-25T09:{index:02d}:00",
                        "2026-08-25",
                        "2026-08-31T00:00:00",
                    ),
                )
        store.connection_count = 0

        trades = store.list_trades()

        assert len(trades) == 12
        assert store.connection_count == 2, "one for the trades, one for the regimes"
        assert {trade["mid_term_regime"] for trade in trades} == {"bull"}


# ------------------------------------------------------- 3a / 3d: the Qt side
@pytest.mark.qt
class TestTheRebuildIsOffTheGuiThread:
    @staticmethod
    def _service(monkeypatch, worker):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.services import journal_feed
        from ui.services.journal_rebuild_service import JournalRebuildService

        monkeypatch.setattr(journal_feed, "rebuild_trades", worker)
        return JournalRebuildService()

    def test_the_rebuild_runs_on_a_worker_thread(self, monkeypatch):
        seen = {}

        def _rebuild():
            seen["thread"] = threading.get_ident()
            return 7

        service = self._service(monkeypatch, _rebuild)
        results = []
        service.finished.connect(results.append)

        token = service.request("correction recorded")
        assert token
        worker = service._worker
        assert worker is not None
        worker.wait(5000)
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

        assert seen["thread"] != threading.get_ident(), "not the GUI thread"
        assert results and results[0]["ok"] is True
        assert results[0]["trades"] == 7
        assert results[0]["token"] == token

    def test_a_second_request_while_one_runs_is_refused(self, monkeypatch):
        release = threading.Event()

        def _rebuild():
            release.wait(5)
            return 1

        service = self._service(monkeypatch, _rebuild)
        first = service.request("one")
        try:
            assert first
            assert service.request("two") == "", "single-flight, not a queue"
        finally:
            release.set()
            service.shutdown(5000)

    def test_a_failure_is_reported_and_never_swallowed(self, monkeypatch):
        """The journal's loud-write rule: a rebuild that did not finish must
        not look like one that did."""

        def _rebuild():
            raise RuntimeError("database is locked")

        service = self._service(monkeypatch, _rebuild)
        results = []
        service.finished.connect(results.append)

        service.request("correction recorded", blocking=True)

        assert results and results[0]["ok"] is False
        assert "database is locked" in results[0]["reason"]


@pytest.mark.qt
class TestTheFilterHeaderIsDebounced:
    def test_a_burst_of_toggles_is_one_reload(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.journal.header import JournalHeader

        header = JournalHeader(autoload=False)
        header._change_coalescer.cancel()
        reloads = []
        header.selectionChanged.connect(lambda: reloads.append(datetime.now()))

        for _ in range(6):
            header._emit_changed()

        assert reloads == [], "nothing fires inside the window"
        header.flush_pending_change()
        assert len(reloads) == 1, "a burst of filter toggles is ONE query"

    def test_loading_still_emits_nothing(self):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.journal.header import JournalHeader

        header = JournalHeader(autoload=False)
        # Construction wires the widgets, which can leave one request owed.
        header._change_coalescer.cancel()
        reloads = []
        header.selectionChanged.connect(lambda: reloads.append(1))
        header._loading = True

        header._emit_changed()
        header.flush_pending_change()

        assert reloads == []
