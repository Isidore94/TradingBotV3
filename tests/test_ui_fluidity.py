"""The GUI-thread costs behind the trader's "so many minor hitch-ups".

Measured on the live desk, 2026-08-21 07:52-11:11: **1843 stalls over 50 ms,
1008 seconds blocked**, plus two garbage-collection freezes of 298 s and 200 s.
The server was the trader's first suspicion and is ruled out - every hot path
resolves to the local SSD, and a miss on the (unmounted) DAS costs 0.0 ms.

What was left, and what each of these pins:

* per-call re-reads of the machine-local settings file - 100 call sites;
* a full re-parse of the 5.8 MB review-event store on every call;
* ~490 bar dicts materialized per symbol per D1 poll, on Qt, against that
  function's own "worker thread, never the GUI thread" contract;
* whole-list widget rebuilds with a stylesheet parse per widget;
* a font warning storm written to the console from inside ``paint()``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _count_reads(monkeypatch, target: Path) -> dict:
    """Count read_text calls against one file, leaving every other read alone."""
    seen = {"count": 0}
    real_read = Path.read_text

    def counting_read(self, *args, **kwargs):
        if self == target:
            seen["count"] += 1
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read)
    return seen


# -- the settings file, read on every get_local_setting ------------------


def test_settings_are_parsed_once_per_change(tmp_path, monkeypatch):
    import project_paths as pp

    settings = tmp_path / "local_settings.json"
    settings.write_text('{"probe": "one"}', encoding="utf-8")
    monkeypatch.setattr(pp, "LOCAL_SETTINGS_FILE", settings)
    pp.invalidate_local_settings_cache()
    reads = _count_reads(monkeypatch, settings)

    for _ in range(50):
        assert pp.get_local_setting("probe") == "one"
    assert reads["count"] == 1, f"parsed {reads['count']} times, not once"


def test_a_changed_settings_file_is_picked_up_immediately(tmp_path, monkeypatch):
    """A cache that can go stale is worse than no cache."""
    import project_paths as pp

    settings = tmp_path / "local_settings.json"
    settings.write_text('{"probe": "one"}', encoding="utf-8")
    monkeypatch.setattr(pp, "LOCAL_SETTINGS_FILE", settings)
    pp.invalidate_local_settings_cache()
    assert pp.get_local_setting("probe") == "one"

    settings.write_text('{"probe": "two"}', encoding="utf-8")
    pp.invalidate_local_settings_cache()
    assert pp.get_local_setting("probe") == "two"


def test_the_cached_settings_cannot_be_mutated_by_a_caller(tmp_path, monkeypatch):
    import project_paths as pp

    settings = tmp_path / "local_settings.json"
    settings.write_text('{"probe": "one"}', encoding="utf-8")
    monkeypatch.setattr(pp, "LOCAL_SETTINGS_FILE", settings)
    pp.invalidate_local_settings_cache()

    payload = pp._load_local_settings()
    payload["probe"] = "mutated"
    assert pp.get_local_setting("probe") == "one"


def test_a_missing_settings_file_reads_as_empty(tmp_path, monkeypatch):
    import project_paths as pp

    monkeypatch.setattr(pp, "LOCAL_SETTINGS_FILE", tmp_path / "absent.json")
    pp.invalidate_local_settings_cache()
    assert pp.get_local_setting("probe") is None


# -- the review-event store ----------------------------------------------


def _write_events(path: Path, count: int) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"ts": f"2026-08-21T09:{index:02d}:00", "action": "shown"})
            for index in range(count)
        )
        + "\n",
        encoding="utf-8",
    )


def test_review_events_are_parsed_once_per_change(tmp_path, monkeypatch):
    import review_events

    store = tmp_path / "alert_review_events.jsonl"
    _write_events(store, 40)
    reads = _count_reads(monkeypatch, store)

    first = review_events.load_review_events(store, include_shards=False)
    for _ in range(10):
        review_events.load_review_events(store, include_shards=False)
    assert reads["count"] == 1
    assert len(first) == 40


def test_an_appended_event_invalidates_the_cache(tmp_path):
    """The store is append-only; a new row must be visible on the next call."""
    import review_events

    store = tmp_path / "alert_review_events.jsonl"
    _write_events(store, 5)
    assert len(review_events.load_review_events(store, include_shards=False)) == 5
    _write_events(store, 6)
    assert len(review_events.load_review_events(store, include_shards=False)) == 6


def test_cached_review_rows_cannot_be_mutated_by_a_caller(tmp_path):
    import review_events

    store = tmp_path / "alert_review_events.jsonl"
    _write_events(store, 3)
    rows = review_events.load_review_events(store, include_shards=False)
    rows[0]["action"] = "mutated"
    again = review_events.load_review_events(store, include_shards=False)
    assert again[0]["action"] == "shown"


# -- fonts on the paint path ---------------------------------------------


@pytest.mark.qt
def test_a_pixel_sized_font_never_produces_a_zero_point_size():
    """The console flood: ``pointSizeF()`` is -1 for a px font, so ``+1.0``
    asked Qt for a zero-point font once per visible row per repaint."""
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui.widgets.setup_delegate import _resized

    base = QFont()
    base.setPixelSize(13)
    for delta in (2.0, 1.0, -1.0):
        out = _resized(base, delta)
        assert out.pixelSize() > 0, "a px font must stay a usable px font"
    assert _resized(base, 2.0).pixelSize() > _resized(base, -1.0).pixelSize()


@pytest.mark.qt
def test_a_point_sized_font_still_scales_in_points():
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui.widgets.setup_delegate import _resized

    base = QFont()
    base.setPointSizeF(10.0)
    assert _resized(base, 2.0).pointSizeF() == pytest.approx(12.0)
    assert _resized(base, -1.0, minimum=7.5).pointSizeF() == pytest.approx(9.0)
    assert _resized(base, -99.0, minimum=7.5).pointSizeF() == pytest.approx(7.5)


@pytest.mark.qt
def test_every_delegate_font_is_one_qt_would_accept():
    """Qt rejects a font whose size is <= 0 in whichever unit it carries."""
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui.widgets.setup_delegate import _resized

    for base_setup in (lambda f: f.setPixelSize(13), lambda f: f.setPointSizeF(10.0)):
        base = QFont()
        base_setup(base)
        for delta in (2.0, 1.0, -1.0):
            resized = _resized(base, delta)
            assert resized.pointSizeF() > 0 or resized.pixelSize() > 0


# -- the Qt message rate limiter -----------------------------------------


@pytest.mark.qt
def test_a_repeated_qt_message_is_printed_once_and_counted(capsys):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import qWarning
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui import app as ui_app

    ui_app._qt_message_counts.clear()
    ui_app.install_qt_message_rate_limit()
    for _ in range(500):
        qWarning("QFont::setPointSizeF: Point size <= 0 (0.000000)")
    printed = capsys.readouterr().err
    assert printed.count("Point size") == 1, printed[:200]
    assert any(count >= 500 for count, _text in ui_app.report_qt_messages())


@pytest.mark.qt
def test_a_new_qt_message_is_always_printed(capsys):
    """Rate limiting must never become silencing."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import qWarning
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui import app as ui_app

    ui_app._qt_message_counts.clear()
    ui_app.install_qt_message_rate_limit()
    qWarning("first complaint")
    qWarning("second complaint")
    printed = capsys.readouterr().err
    assert "first complaint" in printed and "second complaint" in printed


@pytest.mark.qt
def test_messages_differing_only_by_number_collapse_together(capsys):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import qWarning
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui import app as ui_app

    ui_app._qt_message_counts.clear()
    ui_app.install_qt_message_rate_limit()
    for index in range(20):
        qWarning(f"QLayout: attempting to add QLayout to row {index}")
    assert capsys.readouterr().err.count("attempting to add") == 1
