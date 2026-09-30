"""Trade Mentor window: commands, brain-off banner, a streamed turn stored, GPU hand-back, focus ping."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mentor_app import settings  # noqa: E402
from mentor_app.prefetch import PrefetchQueue  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _answer(text):
    return [
        json.dumps({"message": {"content": text}, "done": False}).encode(),
        json.dumps({"message": {"content": ""}, "done": True, "prompt_eval_count": 50, "eval_count": 5}).encode(),
    ]


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    from mentor_app.window import MentorWindow

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    posted: list = []
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        queue=PrefetchQueue(),
        stream_post=lambda url, payload, cancelled: _answer("Auto is DESK [ctx:auto_mode]."),
        post=lambda url, payload, timeout: posted.append((url, payload)) or {},
    )
    win.posted = posted
    yield win
    win.shutdown()
    win.deleteLater()


def _flush(win):
    win._io.submit(lambda: None).result(5)


def _text(win):
    return win.transcript.toPlainText()


def test_help_is_answered_locally(window):
    window.send("/help")
    assert "Commands" in _text(window) and "/remember" in _text(window)


def test_with_the_brain_off_the_turn_is_kept_and_the_banner_says_why(window):
    window._brain_reason = "the night AI owns the GPU"
    window.send("Is NVDA worth it?")
    assert "brain is off" in _text(window).lower()
    _flush(window)
    turns = window.store.turns()
    assert [row["text"] for row in turns] == ["Is NVDA worth it?"]
    assert window.banner.isVisibleTo(window)


def test_a_streamed_turn_renders_and_both_turns_are_stored(window, app):
    window._brain_ok, window._endpoint, window._model = True, "http://127.0.0.1:11436", "qwen3:14b"
    window.send("mode?")
    worker = window._worker
    assert worker is not None and worker.wait(5000)
    deadline = time.monotonic() + 5
    while window._worker is not None and time.monotonic() < deadline:
        app.processEvents()
    assert "Auto is DESK [ctx:auto_mode]." in _text(window)
    _flush(window)
    rows = window.store.turns()
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert rows[1]["model"] == "qwen3:14b" and rows[1]["completion_tokens"] == 5
    _flush(window)
    assert "embed_turns" in window.queue.pending(), "new turns are queued for embedding"
    assert window.send_button.isEnabled() and not window.stop_button.isEnabled()


def test_enter_sends_and_shift_enter_is_a_new_line(window):
    window.input.setPlainText("/help")
    QTest.keyClick(window.input, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert "\n" in window.input.toPlainText()
    window.input.setPlainText("/help")
    QTest.keyClick(window.input, Qt.Key.Key_Return)
    assert window.input.toPlainText() == "" and "Commands" in _text(window)


def test_the_night_takes_the_model_back(window, monkeypatch):
    window._brain_ok, window._endpoint, window._model = True, "http://127.0.0.1:11436", "gpt-oss:20b"
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "the night AI starts within 15 minutes")
    window.check_gpu_share()
    for thread in window._threads:
        thread.join(5)
    assert not window._brain_ok
    url, payload = window.posted[-1]
    assert url.endswith("/api/chat") and payload["keep_alive"] == 0 and payload["model"] == "gpt-oss:20b"
    window.send("still there?")
    assert "brain is off" in _text(window).lower()


def test_connect_refuses_in_the_night_without_touching_the_host(window, monkeypatch):
    touched: list = []
    window._tunnel = SimpleNamespace(preflight=lambda: touched.append(1), stop=lambda: None)
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "night")
    monkeypatch.setattr(settings, "mentor_model", lambda: "gpt-oss:20b")
    states: list = []
    window._bridge.brain_state.connect(states.append)
    window._connect_worker()
    assert touched == [] and states and states[-1]["ok"] is False and states[-1]["reason"] == "night"


def test_connect_warms_the_model_on_the_tunnel(window, monkeypatch, app):
    window._tunnel = SimpleNamespace(
        preflight=lambda: SimpleNamespace(ok=True, reason="ready", host="192.168.0.220"),
        endpoint="http://127.0.0.1:11436",
        stop=lambda: None,
    )
    monkeypatch.setattr(settings, "mentor_model", lambda: "gpt-oss:20b")
    monkeypatch.setattr(settings, "keep_alive", lambda: -1)
    window._connecting = True
    window._connect_worker()
    app.processEvents()
    assert window._brain_ok and window._host == "192.168.0.220"
    url, payload = window.posted[-1]
    assert url == "http://127.0.0.1:11436/api/chat" and payload["keep_alive"] == -1
    from mentor_packs import recall

    recall.set_searcher(None)


def test_tape_and_chips_show_the_context_pack(window):
    from mentor_packs import context_pack

    window._on_context(context_pack.fixture())
    assert window.chips["auto_mode"].text().startswith("Auto mode: DESK")
    assert window.chips["positions"].text() == "Open: 1"
    window.send("/tape")
    assert "[ctx:d1_env]" in _text(window)


def test_the_inbox_shows_a_badge_and_never_touches_the_transcript(window):
    before = _text(window)
    assert window.post_to_inbox("question", "How did the NVDA short go?")
    assert window.inbox_header.text() == "Inbox (1)"
    assert _text(window) == before


def test_a_focus_ping_reaches_the_running_app(app):
    from mentor_app import focus_link

    name = f"tradingbotv3-mentor-focus-test-{time.monotonic_ns()}"
    hits: list = []
    server = focus_link.make_focus_server(None, lambda: hits.append(1), name=name)
    assert server is not None
    result: dict = {}
    thread = threading.Thread(target=lambda: result.setdefault("sent", focus_link.send_focus_ping(2000, name=name)))
    thread.start()
    deadline = time.monotonic() + 5
    while (thread.is_alive() or not hits) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    thread.join(1)
    server.close()
    assert result.get("sent") is True and hits
    assert focus_link.send_focus_ping(200, name=name + "-nobody") is False
