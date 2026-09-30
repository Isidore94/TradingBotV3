"""Follow the desk: a desk-launched Trade Mentor app (`--follow-desk`) exits once the desk's
single-instance slot has been free for two checks in a row; a hand-launched app never does."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from PySide6.QtWidgets import QApplication  # noqa: E402

from mentor_app import settings  # noqa: E402
from mentor_app.prefetch import PrefetchQueue  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402


class _Tunnel:
    endpoint = "http://127.0.0.1:11436"

    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1


def _window(tmp_path, monkeypatch, *, follow, probe, quits):
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    posted: list = []
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(), tunnel=_Tunnel(),
        mentor_enabled=False, follow_desk=follow, desk_probe=probe, quit_app=lambda: quits.append(1),
        post=lambda url, payload, timeout: posted.append((url, payload)) or {},
    )
    win.posted = posted
    win._brain_ok, win._endpoint, win._model = True, "http://127.0.0.1:11436", "gpt-oss:20b"
    return win


@pytest.fixture
def made():
    windows: list = []
    yield windows
    for win in windows:
        win.shutdown()
        win.deleteLater()


def test_two_free_checks_shut_down_and_exit(tmp_path, monkeypatch, made, caplog):
    quits: list = []
    win = _window(tmp_path, monkeypatch, follow=True, probe=lambda: True, quits=quits)
    made.append(win)
    with caplog.at_level("INFO"):
        win._on_desk_state(True)
        assert quits == [] and not win._shut
        win._on_desk_state(True)

    assert quits == [1] and win._shut
    assert win._tunnel.stops == 1
    unloaded = sorted(p["model"] for _, p in win.posted if p.get("keep_alive") == 0)
    assert unloaded == sorted(["gpt-oss:20b", settings.EMBED_MODEL])
    assert any("desk closed; Trade Mentor following" in r.getMessage() for r in caplog.records)


def test_one_free_check_then_busy_does_not_exit(tmp_path, monkeypatch, made):
    quits: list = []
    win = _window(tmp_path, monkeypatch, follow=True, probe=lambda: True, quits=quits)
    made.append(win)
    for state in (True, False, True, None, True):
        win._on_desk_state(state)
    assert quits == [] and not win._shut


def test_a_hand_launched_app_never_exits(tmp_path, monkeypatch, made):
    quits: list = []
    probes: list = []
    win = _window(tmp_path, monkeypatch, follow=False, probe=lambda: probes.append(1) or True, quits=quits)
    made.append(win)
    for _ in range(3):
        win.check_desk()
        win._on_desk_state(True)
    win.bring_to_front()  # a focus ping from a desk launch changes nothing
    assert quits == [] and not win._shut and probes == []
    assert not win._desk_timer.isActive()


def test_the_probe_runs_off_the_qt_thread_and_reports_back(tmp_path, monkeypatch, made):
    import threading
    import time

    quits: list = []
    threads: list = []
    win = _window(tmp_path, monkeypatch, follow=True,
                  probe=lambda: threads.append(threading.current_thread()) or True, quits=quits)
    made.append(win)
    for _ in range(2):
        win.check_desk()
        deadline = time.monotonic() + 5
        while len(threads) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        deadline = time.monotonic() + 5
        while not quits and time.monotonic() < deadline:
            QApplication.processEvents()
            if win._desk_free_checks >= 1 and len(threads) == 1:
                break
    deadline = time.monotonic() + 5
    while not quits and time.monotonic() < deadline:
        QApplication.processEvents()
    assert quits == [1]
    assert all(thread is not threading.main_thread() for thread in threads)


def test_main_follows_the_desk_only_with_the_flag(monkeypatch):
    from ui.services import mentor_launcher

    assert mentor_launcher.FOLLOW_DESK_FLAG == "--follow-desk"
    source = (SCRIPTS_DIR / "mentor_app" / "__init__.py").read_text(encoding="utf-8")
    assert "MentorWindow(follow_desk=FOLLOW_DESK_FLAG in list(argv or ()))" in source


def test_the_desk_launch_passes_the_flag_and_a_focus_ping_does_not():
    from ui.services import mentor_launcher

    calls: list = []
    mentor_launcher._work(lambda cmd, **kw: calls.append(cmd), lambda: True, lambda: True)
    assert calls[-1][-1] == "--follow-desk"
    calls.clear()
    assert mentor_launcher._work(lambda cmd, **kw: calls.append(cmd), lambda: False, lambda: True) == "focused"
    assert calls == []


def test_the_default_probe_reads_the_desk_slot(monkeypatch):
    import single_instance
    from mentor_app import window

    keys: list = []
    monkeypatch.setattr(single_instance, "slot_is_free", lambda key=None, **kw: keys.append(key) or True)
    assert window._desk_slot_is_free() is True
    assert keys == [single_instance.DESK_LOCK_KEY]
