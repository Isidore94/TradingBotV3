"""Packet 3 item 4: the measured drips.

Eight small costs, each evidenced in the 2026-08-31 stall log or the thread
counts beside it. Threading, caching, batching and debouncing only - nothing
computes anything different, and the one observable change (4g) was authorized
explicitly.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import technical_integrity as ti  # noqa: E402


# ---------------------------------------------------- 4c snapshot parse memo
class TestTheIntegritySnapshotIsParsedOncePerVersion:
    @staticmethod
    def _snapshot(path: Path, value: int = 1) -> None:
        path.write_text(
            json.dumps({"schema": ti.SNAPSHOT_SCHEMA, "value": value}), encoding="utf-8"
        )

    def test_repeat_reads_parse_once(self, tmp_path, monkeypatch):
        """The GUI polls this every 30 s; the ~453 KB file changes hourly."""
        ti.clear_technical_integrity_snapshot_cache()
        path = tmp_path / "snapshot.json"
        self._snapshot(path)

        reads = []
        real = Path.read_text
        monkeypatch.setattr(
            Path, "read_text", lambda self, **kw: (reads.append(self.name), real(self, **kw))[1]
        )
        for _ in range(30):
            assert ti.load_technical_integrity_snapshot(path)["value"] == 1

        assert reads.count("snapshot.json") == 1

    def test_a_rewritten_snapshot_is_parsed_again(self, tmp_path):
        ti.clear_technical_integrity_snapshot_cache()
        path = tmp_path / "snapshot.json"
        self._snapshot(path, value=1)
        assert ti.load_technical_integrity_snapshot(path)["value"] == 1

        self._snapshot(path, value=2)
        os.utime(path, (0, 0))

        assert ti.load_technical_integrity_snapshot(path)["value"] == 2

    def test_a_missing_snapshot_is_not_remembered(self, tmp_path):
        ti.clear_technical_integrity_snapshot_cache()
        path = tmp_path / "snapshot.json"
        assert ti.load_technical_integrity_snapshot(path) == {}
        self._snapshot(path, value=7)
        assert ti.load_technical_integrity_snapshot(path)["value"] == 7


# ------------------------------------------------------------ 4h paused loop
def test_the_paused_loop_sleeps_five_seconds_not_half_of_one():
    """120 wake-ups a minute for two calls that are date/time guarded and can
    change at most once per 5-minute bar. Shutdown latency 0.5 s -> 5 s, which
    the trader accepted."""
    source = (SCRIPTS_DIR / "bounce_bot_lib" / "legacy.py").read_text(encoding="utf-8")
    assert "self._stop_event.wait(5.0)" in source
    assert "self._stop_event.wait(0.5)" not in source


# --------------------------------------------------------------- Qt surfaces
@pytest.mark.qt
class TestTheBounceServiceDrips:
    @staticmethod
    def _service():
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.services.bounce_service import BounceService

        return BounceService()

    def test_the_entry_board_snapshot_runs_off_the_gui_thread(self):
        """It walks the bot's cached bars for every board name, once a minute,
        and used to do it on the Qt thread."""
        service = self._service()
        seen = {}
        done = threading.Event()

        class _Bot:
            def entry_assist_board_snapshot(self):
                seen["thread"] = threading.get_ident()
                done.set()
                return {"rows": []}

        service._current_bot = lambda: _Bot()
        service._is_live = lambda: True
        service.refresh_entry_board()

        assert done.wait(5), "the worker never ran"
        assert seen["thread"] != threading.get_ident()

    def test_the_entry_board_refresh_is_single_flight(self):
        service = self._service()
        release = threading.Event()
        started = threading.Event()
        calls = []

        class _Bot:
            def entry_assist_board_snapshot(self):
                calls.append(1)
                started.set()
                release.wait(5)
                return {}

        service._current_bot = lambda: _Bot()
        service._is_live = lambda: True
        try:
            service.refresh_entry_board()
            assert started.wait(5)
            service.refresh_entry_board()
            service.refresh_entry_board()
            assert calls == [1], "one in flight, not a queue of three"
        finally:
            release.set()

    def test_the_health_tick_makes_no_thread_when_the_file_has_not_moved(
        self, tmp_path, monkeypatch
    ):
        """The stat is microseconds; the PARSE is what needed a thread."""
        import project_paths

        signals = tmp_path / "avwap_signals.csv"
        signals.write_text("a,b\n", encoding="utf-8")
        monkeypatch.setattr(project_paths, "AVWAP_SIGNALS_FILE", signals)

        service = self._service()
        stat = signals.stat()
        service._active_bounces_signature = (int(stat.st_size), int(stat.st_mtime_ns))

        assert service._active_bounces_signature_moved() is False

        signals.write_text("a,b\nc,d\n", encoding="utf-8")
        os.utime(signals, (0, 0))
        assert service._active_bounces_signature_moved() is True

    def test_an_unreadable_signals_file_still_gets_a_worker(self, tmp_path, monkeypatch):
        """The worker is what knows how to turn that into an honest zero."""
        import project_paths

        monkeypatch.setattr(project_paths, "AVWAP_SIGNALS_FILE", tmp_path / "gone.csv")
        service = self._service()
        service._active_bounces_signature = (0, 0)

        assert service._active_bounces_signature_moved() is True


@pytest.mark.qt
class TestTheAlertCenterPrefsWriteOnce:
    def test_one_batched_settings_write_not_two(self, tmp_path, monkeypatch):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels import alert_center_panel as panel_mod

        panel = panel_mod.AlertCenterPanel(
            parked_symbols_path=tmp_path / "parked.json",
            focus_d1_flags_path=tmp_path / "flags.json",
        )
        writes = []
        monkeypatch.setattr(panel_mod, "save_local_settings", lambda values: writes.append(values))
        monkeypatch.setattr(panel, "_rebuild_feed", lambda: None)

        panel._on_prefs_changed()

        assert len(writes) == 1
        assert set(writes[0]) == {"qt_alert_min_tier", "qt_alert_sound"}


@pytest.mark.qt
class TestTheHoldExpiryEvaluatesEachAlertOnce:
    def test_the_current_alert_is_evaluated_once_and_fires_one_event(self, tmp_path, monkeypatch):
        """`survives()` rewrites the caption AND writes a `hold_expired` review
        event, and the current alert was run through it twice - once in the
        queue filter, once on its own. Two events, two caption mutations, on the
        tick it expired. The trader authorized the fix and its consequence."""
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.models.bounce import BounceAlert
        from ui.panels import alert_center_panel as panel_mod

        alert = BounceAlert(
            time_text="07:09:19",
            symbol="NVDA",
            side="LONG",
            trigger=f"{panel_mod.REGIME_PAUSE_TRIGGER_PREFIX} holding highs",
            timeframe="5m",
            tag="green",
            raw_text="regime pause NVDA",
        )

        class _Verdict:
            keep = False
            reason = "stale hold"

            class hold:
                distance_atr = 2.0
                bars_since_extreme = 9

        panel = panel_mod.AlertCenterPanel(
            parked_symbols_path=tmp_path / "parked.json",
            focus_d1_flags_path=tmp_path / "flags.json",
        )
        evaluated = []
        events = []
        monkeypatch.setattr(panel_mod, "is_regime_pause_alert", lambda _alert: True)
        monkeypatch.setattr(
            panel, "_hold_verdict_for", lambda a: evaluated.append(a.symbol) or _Verdict()
        )
        monkeypatch.setattr(panel, "_apply_hold_caption", lambda a, v: None)
        monkeypatch.setattr(
            panel, "_record_review_event", lambda action, **kw: events.append(action)
        )
        monkeypatch.setattr(panel, "_advance_review_queue", lambda: None)
        panel._review_queue = [alert]
        panel._current_review_alert = alert

        panel._expire_stale_hold_alerts()

        assert evaluated == ["NVDA"], "exactly once per alert per tick"
        assert events == ["hold_expired"], "one event, not two"


@pytest.mark.qt
class TestTheSetupTrackerPage:
    def test_the_spinbox_is_debounced(self, monkeypatch):
        """The arrows step one at a time and every step re-ran the WHOLE page:
        ten CSV parses, ten model resets, ten column fits."""
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.setup_tracker_panel import SetupTrackerPanel

        panel = SetupTrackerPanel()
        try:
            refreshes = []
            panel._refresh_coalescer._target = lambda: refreshes.append(1)
            panel._refresh_coalescer.cancel()

            for value in range(6, 12):
                panel.min_closed_input.setValue(value)

            assert refreshes == [], "nothing fires inside the window"
            panel._refresh_coalescer.flush()
            assert len(refreshes) == 1, "six steps, one page rebuild"
        finally:
            panel.close()

    def test_an_unchanged_export_is_parsed_once(self, tmp_path, monkeypatch):
        from ui.panels import setup_tracker_panel as tracker_mod

        tracker_mod.clear_setup_tracker_csv_cache()
        path = tmp_path / "export.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")

        parses = []
        real = tracker_mod._load_csv_rows
        monkeypatch.setattr(
            tracker_mod, "_load_csv_rows", lambda p: (parses.append(str(p)), real(p))[1]
        )

        first = tracker_mod._load_csv_rows_cached(path)
        for _ in range(5):
            tracker_mod._load_csv_rows_cached(path)

        assert len(parses) == 1
        assert first == [{"a": "1", "b": "2"}]

    def test_a_rewritten_export_is_parsed_again(self, tmp_path):
        from ui.panels import setup_tracker_panel as tracker_mod

        tracker_mod.clear_setup_tracker_csv_cache()
        path = tmp_path / "export.csv"
        path.write_text("a\n1\n", encoding="utf-8")
        assert len(tracker_mod._load_csv_rows_cached(path)) == 1

        path.write_text("a\n1\n2\n", encoding="utf-8")
        os.utime(path, (0, 0))
        assert len(tracker_mod._load_csv_rows_cached(path)) == 2


@pytest.mark.qt
class TestTheFocusChip:
    @staticmethod
    def _chip(monkeypatch):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from ui.panels.focus_picks_panel import FocusStatusChip

        return FocusStatusChip("NVDA", tone="long", state={})

    def test_the_badge_stylesheet_is_inside_the_look_guard(self, monkeypatch):
        """It ran a stylesheet parse per chip per update - on a 45-name board,
        for every bounce alert and every mover pass."""
        chip = self._chip(monkeypatch)
        state = {"bounce": {"text": "BOUNCE 5m", "tone": "long"}}
        chip.update_state(state)

        sets = []
        monkeypatch.setattr(chip.live_flag, "setStyleSheet", lambda css: sets.append(css))
        for _ in range(10):
            chip.update_state(dict(state))

        assert sets == [], "the look did not change, so nothing is re-parsed"

    def test_a_changed_accent_still_re_applies(self, monkeypatch):
        chip = self._chip(monkeypatch)
        chip.update_state({"bounce": {"text": "BOUNCE", "tone": "long"}})

        sets = []
        monkeypatch.setattr(chip.live_flag, "setStyleSheet", lambda css: sets.append(css))
        chip.update_state({"rrs": {"text": "RRS +1.2", "tone": "short"}})

        assert len(sets) == 1
        assert chip.live_flag.text() == "RRS"

    def test_an_rrs_snapshot_is_coalesced_like_every_other_rebuild(self, tmp_path, monkeypatch):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from focus_picks import FocusPickStore
        from ui.panels.focus_picks_panel import FocusPicksPanel
        from ui.services.focus_service import FocusService
        from ui.services.price_alert_service import PriceAlertService

        service = FocusService(
            FocusPickStore(
                focus_longs_path=tmp_path / "focus_longs.txt",
                focus_shorts_path=tmp_path / "focus_shorts.txt",
                longs_path=tmp_path / "longs.txt",
                shorts_path=tmp_path / "shorts.txt",
                membership_path=tmp_path / "membership.json",
            )
        )
        panel = FocusPicksPanel(service, PriceAlertService(engine_enabled=False))
        try:
            rebuilds = []
            panel._refresh_coalescer._target = lambda: rebuilds.append(1)
            panel._refresh_coalescer.cancel()

            for _ in range(4):
                panel.record_rrs_snapshot({})

            assert rebuilds == [], "held inside the window"
            panel._refresh_coalescer.flush()
            assert len(rebuilds) == 1
        finally:
            panel.close()
