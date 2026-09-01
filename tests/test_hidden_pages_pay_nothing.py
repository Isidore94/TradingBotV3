"""Packet 3 item 3: the three remaining ungated timers stop paying while hidden.

All three are pure presentation - none feeds an alert, a file write or a push -
and all three ran their full tick whether or not anyone was looking at the page.
The pattern is the one `chart_review_panel` already uses: the TIMER keeps
running, the work early-returns while hidden, and `showEvent` catches the page up
once on the way back in.

The watchlist viewer also gained a signature guard. `setPlainText` resets the
scroll position and the caret, and a 30-second timer called it unconditionally -
so reading the Auto Longs list meant being yanked back to the top every thirty
seconds, whether or not Auto Pilot had written anything.
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

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class TestTheWatchlistViewer:
    @staticmethod
    def _viewer(path: Path):
        from ui.panels.watchlists_panel import AutoWatchlistViewerPanel

        return AutoWatchlistViewerPanel("Auto Longs", path)

    def test_an_unchanged_file_never_re_sets_the_text(self, tmp_path, monkeypatch):
        """`setPlainText` yanks scroll and caret. Every 30 s, for nothing."""
        path = tmp_path / "autolongs.txt"
        path.write_text("NVDA\nAMD\n", encoding="utf-8")
        viewer = self._viewer(path)
        viewer.refresh_from_disk()

        sets = []
        monkeypatch.setattr(viewer.text, "setPlainText", lambda text: sets.append(text))
        for _ in range(5):
            viewer.refresh_from_disk()

        assert sets == []

    def test_a_changed_file_does_re_set_the_text(self, tmp_path):
        path = tmp_path / "autolongs.txt"
        path.write_text("NVDA\n", encoding="utf-8")
        viewer = self._viewer(path)
        viewer.refresh_from_disk()
        assert viewer.current_symbols() == ["NVDA"]

        path.write_text("NVDA\nTSLA\n", encoding="utf-8")
        os.utime(path, (0, 0))
        viewer.refresh_from_disk()

        assert viewer.current_symbols() == ["NVDA", "TSLA"]

    def test_force_re_reads_even_when_unchanged(self, tmp_path, monkeypatch):
        path = tmp_path / "autolongs.txt"
        path.write_text("NVDA\n", encoding="utf-8")
        viewer = self._viewer(path)
        viewer.refresh_from_disk()

        sets = []
        monkeypatch.setattr(viewer.text, "setPlainText", lambda text: sets.append(text))
        viewer.refresh_from_disk(force=True)

        assert sets == ["NVDA"]

    def test_a_hidden_page_reads_nothing(self, tmp_path, monkeypatch):
        from ui.panels.watchlists_panel import AutoWatchlistViewerArea

        panel = AutoWatchlistViewerArea()
        reads = []
        for viewer in panel.symbol_panels():
            monkeypatch.setattr(
                viewer, "refresh_from_disk", lambda **kwargs: reads.append(1)
            )

        assert panel.isVisible() is False
        panel.refresh()

        assert reads == []

    def test_showing_the_page_catches_it_up(self, tmp_path, monkeypatch):
        from ui.panels.watchlists_panel import AutoWatchlistViewerArea

        panel = AutoWatchlistViewerArea()
        reads = []
        for viewer in panel.symbol_panels():
            monkeypatch.setattr(
                viewer, "refresh_from_disk", lambda **kwargs: reads.append(1)
            )
        try:
            panel.show()
            assert len(reads) == 2, "both viewers, once"
        finally:
            panel.close()


class TestTheMasterSchedulerTick:
    @staticmethod
    def _panel():
        from ui.panels.master_avwap_panel import MasterAvwapPanel
        from ui.services.focus_service import FocusService

        return MasterAvwapPanel(FocusService())

    def test_a_hidden_page_does_no_scheduler_work(self, monkeypatch):
        panel = self._panel()
        try:
            calls = []
            monkeypatch.setattr(panel, "_reset_scheduler_state_for_day", lambda now: calls.append(1))

            assert panel.isVisible() is False
            panel._scheduler_tick()

            assert calls == []
        finally:
            panel.close()

    def test_an_externally_owned_scheduler_does_no_work_either(self, monkeypatch):
        """It can do nothing when another process owns scheduled scans - the
        tick's only outcome there was to disable itself and rewrite a label."""
        panel = self._panel()
        try:
            panel.show()
            panel.external_scheduler_owner = "the scheduled task"
            calls = []
            monkeypatch.setattr(panel, "_reset_scheduler_state_for_day", lambda now: calls.append(1))

            panel._scheduler_tick()

            assert calls == []
        finally:
            panel.close()

    def test_showing_the_page_ticks_once(self, monkeypatch):
        panel = self._panel()
        try:
            calls = []
            monkeypatch.setattr(panel, "_scheduler_tick", lambda: calls.append(1))
            panel.show()
            assert calls == [1]
        finally:
            panel.close()


class TestTheRsWindowAutoTick:
    @staticmethod
    def _panel():
        from ui.panels.rs_window_panel import RsWindowPanel

        class _Service:
            def current_bot(self):
                raise AssertionError("a hidden page must not even ask for the bot")

        return RsWindowPanel(_Service())

    def test_a_hidden_page_does_not_even_ask_for_the_bot(self):
        panel = self._panel()
        try:
            assert panel.isVisible() is False
            panel._auto_tick()  # the stub raises if the gate is missing
        finally:
            panel.close()

    def test_showing_the_page_ticks_once(self, monkeypatch):
        from ui.panels.rs_window_panel import RsWindowPanel

        class _Service:
            def current_bot(self):
                return None

        panel = RsWindowPanel(_Service())
        try:
            calls = []
            monkeypatch.setattr(panel, "_auto_tick", lambda: calls.append(1))
            panel.show()
            assert calls == [1]
        finally:
            panel.close()
