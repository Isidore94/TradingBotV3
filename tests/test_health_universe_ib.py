"""P2-11c: Health shows the universe count against its floor, and IB status.

Both are read on the audit worker, never on the Qt thread; missing evidence
is UNKNOWN, never green and never "disconnected".
"""

from __future__ import annotations

import os
import sys
import threading
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
from universe_builder import UNIVERSE_FLOOR_MIN_SYMBOLS  # noqa: E402


def _universe(tmp_path, monkeypatch, count: int | None) -> None:
    path = tmp_path / "universe_all.txt"
    if count is not None:
        path.write_text("\n".join(f"S{i}" for i in range(count)), encoding="utf-8")
    monkeypatch.setattr(project_paths, "UNIVERSE_ALL_FILE", path)


def test_universe_below_the_floor_is_unhealthy(tmp_path, monkeypatch):
    _universe(tmp_path, monkeypatch, UNIVERSE_FLOOR_MIN_SYMBOLS - 1)
    row = hp.universe_floor_check()
    assert row["status"] == "unhealthy"
    assert f"floor of {UNIVERSE_FLOOR_MIN_SYMBOLS}" in row["summary"]


def test_universe_at_or_above_the_floor_is_healthy_and_says_both_numbers(tmp_path, monkeypatch):
    _universe(tmp_path, monkeypatch, UNIVERSE_FLOOR_MIN_SYMBOLS + 12)
    row = hp.universe_floor_check()
    assert row["status"] == "healthy"
    assert row["details"] == {"count": UNIVERSE_FLOOR_MIN_SYMBOLS + 12, "floor": UNIVERSE_FLOOR_MIN_SYMBOLS}
    assert f"floor {UNIVERSE_FLOOR_MIN_SYMBOLS}" in row["summary"]


def test_no_universe_file_is_unknown_not_empty(tmp_path, monkeypatch):
    _universe(tmp_path, monkeypatch, None)
    assert hp.universe_floor_check()["status"] == "unknown"


class _Bot:
    def __init__(self, connected=True, pacing=0.0, broken=False):
        self._connected = connected
        self._pacing = pacing
        self._broken = broken

    @property
    def connection_status(self):
        if self._broken:
            raise RuntimeError("pipe closed")
        return self._connected

    def pacing_delay_remaining(self):
        return self._pacing


def test_ib_status_rows():
    assert hp.ib_status_check(None)["status"] == "unknown"
    assert hp.ib_status_check(lambda: None)["status"] == "unknown"
    assert hp.ib_status_check(lambda: _Bot(broken=True))["status"] == "unknown"
    assert hp.ib_status_check(lambda: _Bot(connected=False))["status"] == "unhealthy"
    assert hp.ib_status_check(lambda: _Bot(pacing=12))["status"] == "degraded"
    assert hp.ib_status_check(lambda: _Bot())["status"] == "healthy"


def test_the_panel_shows_both_rows_and_asks_the_bot_off_the_qt_thread(tmp_path, monkeypatch):
    _universe(tmp_path, monkeypatch, UNIVERSE_FLOOR_MIN_SYMBOLS + 1)
    monkeypatch.setattr(
        hp,
        "build_operations_audit",
        lambda: {"status": "healthy", "summary": {"healthy": 0, "total": 0}, "checks": []},
    )
    asked_on: list[threading.Thread] = []

    def provider():
        asked_on.append(threading.current_thread())
        return _Bot(connected=False)

    panel = hp.HealthPanel(refresh_interval_ms=3_600_000, bot_provider=provider)
    try:
        panel.refresh()
        panel.wait_for_audit()
        _app.processEvents()
        ids = {row["id"]: row["status"] for row in panel._payload.get("checks", [])}
        assert ids.get("universe_floor") == "healthy"
        assert ids.get("ib_status") == "unhealthy"
        assert panel._payload["status"] == "unhealthy"
        assert asked_on and all(t is not threading.main_thread() for t in asked_on)
    finally:
        panel.shutdown()
        panel.deleteLater()
        _app.processEvents()
