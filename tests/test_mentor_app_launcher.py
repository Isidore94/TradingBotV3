"""The desk's Trade Mentor button: starts or focuses the app without blocking the Qt thread."""

from __future__ import annotations

import functools
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ui.services import mentor_launcher  # noqa: E402


def _slow_popen(calls):
    def popen(cmd, **kwargs):
        time.sleep(1.0)
        calls.append((cmd, kwargs))
        return SimpleNamespace(pid=1)

    return popen


def test_the_click_handler_returns_in_under_50_ms(monkeypatch):
    from ui.app import MainWindow

    calls: list = []
    fast = functools.partial(
        mentor_launcher.launch_or_focus, popen=_slow_popen(calls), probe=lambda: True, ping=lambda: False
    )
    started: list = []
    monkeypatch.setattr(mentor_launcher, "launch_or_focus", lambda: started.append(fast()) or started[-1])
    t0 = time.perf_counter()
    MainWindow._open_trade_mentor_app(SimpleNamespace())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50, f"the desk's click took {elapsed_ms:.0f} ms"
    started[0].join(5)
    assert calls and calls[0][0][-1].endswith("launch_mentor.py")


def test_a_free_slot_launches_the_app_from_the_repo_root():
    calls: list = []
    thread = mentor_launcher.launch_or_focus(popen=lambda cmd, **kw: calls.append((cmd, kw)), probe=lambda: True)
    thread.join(5)
    cmd, kwargs = calls[0]
    assert Path(cmd[1]) == ROOT_DIR / "launch_mentor.py"
    assert Path(kwargs["cwd"]) == ROOT_DIR
    assert Path(cmd[0]).name.lower() in ("python.exe", "pythonw.exe", "python", "python3")


def test_a_running_app_is_focused_not_launched_twice():
    calls: list = []
    pings: list = []
    thread = mentor_launcher.launch_or_focus(
        popen=lambda cmd, **kw: calls.append(cmd), probe=lambda: False, ping=lambda: pings.append(1) or True
    )
    thread.join(5)
    assert pings == [1] and calls == []


def test_a_frozen_desk_does_nothing(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    calls: list = []
    assert mentor_launcher.launch_or_focus(popen=lambda *a, **k: calls.append(a)) is None
    assert calls == []


def test_a_failed_launch_never_raises_into_the_desk():
    def boom(*args, **kwargs):
        raise OSError("no python")

    assert mentor_launcher._work(boom, lambda: True, lambda: False) == "failed"


def test_the_desk_has_the_button_wired():
    source = (SCRIPTS_DIR / "ui" / "app.py").read_text(encoding="utf-8")
    assert 'QPushButton("Trade Mentor")' in source
    assert "self.trade_mentor_app_button.clicked.connect(self._open_trade_mentor_app)" in source
