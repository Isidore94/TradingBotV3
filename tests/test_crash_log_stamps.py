"""gui_crash.log stamps its own time and keeps Qt's last words (B0, 2026-09-24 abort).

The 09-24 desk died with "Fatal Python error: Aborted" and no "Current thread"
in the dump: the abort came from a thread with no Python state (a Qt/native
thread; a CRT abort on a native thread reproduces that exact shape). The dump
carries no time. A Qt fatal message is a different death: Qt6 fast-fails on
Windows, so faulthandler writes nothing and the reason only reached the console.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ZONED = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}"

OLD_CRASH = (
    "\n=== GUI start 2026-09-24 06:23:25 pid=10552 ===\n"
    "Fatal Python error: Aborted\n\n"
    'Thread 0x00002ca8 (most recent call first):\n  File "app.py", line 2386 in main\n'
)


@pytest.fixture
def armed_log(tmp_path, monkeypatch):
    """Arm the desk's crash log into tmp_path without touching pytest's own faulthandler."""
    import faulthandler
    import importlib

    import launch_gui  # puts scripts/ on sys.path

    crash_log = importlib.import_module("crash_log")
    project_paths = importlib.import_module("project_paths")

    monkeypatch.setattr(project_paths, "LOCAL_LOG_DIR", tmp_path)
    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: None)
    monkeypatch.setattr(crash_log, "_HANDLE", None, raising=False)
    handles = []

    def arm():
        launch_gui._enable_crash_log()
        handle = getattr(crash_log, "_HANDLE", None) or launch_gui._CRASH_LOG_HANDLE
        handles.append(handle)
        return tmp_path / "gui_crash.log"

    yield arm
    for handle in handles:
        if handle is not None:
            handle.close()
    monkeypatch.setattr(launch_gui, "_CRASH_LOG_HANDLE", None)


def _write_old_crash(path: Path, mtime: float) -> None:
    path.write_text(OLD_CRASH, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_a_previous_unstamped_crash_is_stamped_with_its_write_time(armed_log, tmp_path):
    log = tmp_path / "gui_crash.log"
    crash_at = datetime(2026, 9, 24, 9, 41, 7).timestamp()
    _write_old_crash(log, crash_at)

    armed_log()

    text = log.read_text(encoding="utf-8")
    stamp = datetime.fromtimestamp(crash_at).astimezone().isoformat(timespec="seconds")
    assert f"=== crash above written {stamp}" in text
    # The stamp sits between the dump and the new run's header.
    assert text.index("Fatal Python error") < text.index(stamp) < text.rindex("=== GUI start")
    # No "Current thread" in the dump: the stamp says a native thread crashed.
    assert "native (non-Python) thread" in text


def test_the_start_header_carries_a_zoned_time(armed_log):
    log = armed_log()
    header = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert re.fullmatch(rf"=== GUI start {ZONED} pid=\d+ ===", header), header


def test_a_clean_run_or_an_already_stamped_crash_is_not_stamped_again(armed_log, tmp_path):
    log = tmp_path / "gui_crash.log"
    _write_old_crash(log, time.time() - 3600)
    armed_log()
    armed_log()
    armed_log()
    assert log.read_text(encoding="utf-8").count("=== crash above written") == 1


@pytest.mark.qt
def test_a_qt_critical_message_lands_in_the_crash_log_with_a_time(armed_log):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import qCritical
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui import app as ui_app

    log = armed_log()
    ui_app._qt_message_counts.clear()
    ui_app.install_qt_message_rate_limit()
    qCritical("b0 probe: something Qt could not recover from")
    text = log.read_text(encoding="utf-8")
    assert re.search(rf"\[{ZONED}\] Qt critical: b0 probe: something Qt", text), text[-400:]


_FATAL_CHILD = """
import sys
sys.path.insert(0, {scripts!r})
sys.path.insert(0, {root!r})
import launch_gui
launch_gui._enable_crash_log()
import shiboken6
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication
app = QApplication([])
from ui import app as ui_app
ui_app.install_qt_message_rate_limit()
worker = QThread()
worker.setObjectName("b0-probe")
worker.start()
shiboken6.delete(worker)  # Qt6: qFatal "QThread: Destroyed while thread ... is still running"
"""


@pytest.mark.qt
def test_a_qt_fatal_abort_leaves_its_reason_and_time_in_the_crash_log(tmp_path):
    """Qt kills the process without a faulthandler dump; the log must still say why and when."""
    pytest.importorskip("PySide6")
    local = tmp_path / "local"
    data = tmp_path / "data"
    local.mkdir()
    data.mkdir()
    env = dict(os.environ, LOCALAPPDATA=str(local), TRADINGBOTV3_DATA_DIR=str(data))
    env["QT_QPA_PLATFORM"] = "offscreen"
    code = _FATAL_CHILD.format(scripts=str(ROOT / "scripts"), root=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    log = local / "TradingBotV3" / "logs" / "gui_crash.log"
    text = log.read_text(encoding="utf-8")
    assert re.search(
        rf"\[{ZONED}\] Qt fatal: QThread: Destroyed while thread '?b0-probe'? is still running",
        text,
    ), text[-600:]
