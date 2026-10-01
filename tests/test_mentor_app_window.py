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
    url, payload = next((u, p) for u, p in window.posted if u.endswith("/api/chat"))
    assert url.endswith("/api/chat") and payload["keep_alive"] == 0 and payload["model"] == "gpt-oss:20b"
    window.send("still there?")
    assert "brain is off" in _text(window).lower()


def _unloaded(win):
    return sorted(
        p["model"] for url, p in win.posted if url.endswith(("/api/chat", "/api/embed")) and p.get("keep_alive") == 0
    )


def test_at_2145_both_the_chat_model_and_the_embedder_are_unloaded(window, monkeypatch):
    window._brain_ok, window._endpoint, window._model = True, "http://127.0.0.1:11436", "gpt-oss:20b"
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "the night AI starts within 15 minutes")
    window.check_gpu_share()
    for thread in window._threads:
        thread.join(5)
    assert _unloaded(window) == sorted(["gpt-oss:20b", settings.EMBED_MODEL])


def test_closing_the_app_unloads_both_models_and_stops_the_tunnel(app, tmp_path, monkeypatch):
    from mentor_app.window import MentorWindow

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    posted: list = []
    stopped: list = []
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        queue=PrefetchQueue(),
        tunnel=SimpleNamespace(stop=lambda: stopped.append(1)),
        post=lambda url, payload, timeout: posted.append((url, payload)) or {},
    )
    win._brain_ok, win._endpoint, win._model = True, "http://127.0.0.1:11436", "gemma3:12b"
    win.posted = posted
    win.shutdown()
    win.deleteLater()
    assert _unloaded(win) == sorted(["gemma3:12b", settings.EMBED_MODEL])
    assert stopped == [1]


def test_a_night_started_ollama_makes_the_queue_yield_fully(window, monkeypatch, app, caplog):
    window._tunnel = SimpleNamespace(
        preflight=lambda: SimpleNamespace(ok=True, reason="ollama: already up", host="192.168.0.220", slots=1),
        endpoint="http://127.0.0.1:11436",
        stop=lambda: None,
    )
    monkeypatch.setattr(settings, "mentor_model", lambda: "gpt-oss:20b")
    window._connecting = True
    with caplog.at_level("INFO"):
        window._connect_worker()
    app.processEvents()
    assert "1 slot, night-started" in caplog.text
    assert window.queue.single_slot
    window.queue.begin_interactive()
    assert window.queue.should_yield()
    from mentor_packs import recall

    recall.set_searcher(None)


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


def test_connect_picks_gemma4_when_the_host_has_it_and_the_pill_says_native_tools(window, monkeypatch, app):
    window._tunnel = SimpleNamespace(
        preflight=lambda: SimpleNamespace(ok=True, reason="ready", host="192.168.0.220"),
        endpoint="http://127.0.0.1:11436",
        stop=lambda: None,
    )
    shown: list = []

    def post(url, payload, timeout):
        if url.endswith("/api/show"):
            shown.append(payload["model"])
            if payload["model"] == "gemma4:12b":
                return {"details": {}, "capabilities": ["completion", "tools"]}
            raise RuntimeError("HTTP 404")
        return {}

    window._post = post
    monkeypatch.setattr(settings, "mentor_model", lambda: "gemma3:12b-tbv3ctx-64k")
    monkeypatch.setattr(settings, "explicit_model", lambda: "")
    window._connecting = True
    window._connect_worker()
    app.processEvents()
    assert window._model == "gemma4:12b" and window._native_tools is True
    assert "gemma4:12b" in window.status_pill.text() and "tools: native" in window.status_pill.text()
    # A second connect reads the capability cache (only the presence check asks the host again).
    shown.clear()
    window._connecting = True
    window._connect_worker()
    app.processEvents()
    assert shown == ["gemma4:12b"]
    from mentor_packs import recall

    recall.set_searcher(None)


def test_connect_without_gemma4_keeps_the_medium_model_on_the_fallback(window, monkeypatch, app):
    window._tunnel = SimpleNamespace(
        preflight=lambda: SimpleNamespace(ok=True, reason="ready", host="192.168.0.220"),
        endpoint="http://127.0.0.1:11436",
        stop=lambda: None,
    )

    def post(url, payload, timeout):
        if url.endswith("/api/show"):
            if payload["model"] == "gemma3:12b-tbv3ctx-64k":
                return {"details": {}, "capabilities": ["completion", "vision"]}
            raise RuntimeError("HTTP 404")
        return {}

    window._post = post
    monkeypatch.setattr(settings, "mentor_model", lambda: "gemma3:12b-tbv3ctx-64k")
    monkeypatch.setattr(settings, "explicit_model", lambda: "")
    window._connecting = True
    window._connect_worker()
    app.processEvents()
    assert window._model == "gemma3:12b-tbv3ctx-64k" and window._native_tools is False
    assert "tools: fallback" in window.status_pill.text()
    from mentor_packs import recall

    recall.set_searcher(None)


def _run_turn(window, app, text):
    window.send(text)
    worker = window._worker
    assert worker is not None and worker.wait(5000)
    deadline = time.monotonic() + 5
    while window._worker is not None and time.monotonic() < deadline:
        app.processEvents()
    _flush(window)


def test_a_plain_pre_trade_question_auto_attaches_the_gate_and_the_log_says_so(app, tmp_path, monkeypatch):
    from mentor_app.window import MentorWindow
    from mentor_packs import context_pack
    from mentor_packs.registry import make_pack

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    sent: list = []
    built: list = []

    def build(name, args):
        built.append((name, dict(args)))
        if name == "gate_pack":
            return make_pack(name, [{"id": "gate:TSLA:pick:TSLA:earn", "text": "earnings far"},
                                    {"id": "gate:TSLA:tape:d1env", "text": "D1 bearish"},
                                    {"id": "gate:TSLA:book:industry", "text": "no Autos open"}])
        return make_pack(name, [{"id": f"news:{args.get('symbol')}:1", "text": "a headline"}])

    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(),
        stream_post=lambda url, payload, cancelled: sent.append(payload) or _answer(
            "Earnings are far [gate:TSLA:pick:TSLA:earn]."),
        post=lambda url, payload, timeout: {}, pack_builder=build,
    )
    try:
        win._brain_ok, win._endpoint, win._model, win._native_tools = True, "http://x", "gemma4:12b", True
        win._on_context(context_pack.fixture())  # TSLA is a swing short on Focus
        _run_turn(win, app, "im thinking of shorting TSLA thoughts?")
        assert built[0] == ("gate_pack", {"side": "SHORT", "symbol": "TSLA"})
        # P15b: a pre-trade question also carries today's brief (compact) between the gate and the news.
        assert [m["role"] for m in sent[0]["messages"]][-4:] == ["assistant", "tool", "tool", "tool"]
        shown = _text(win)
        assert "Not covered:" in shown and "[gate:TSLA:book:industry]" in shown
        row = win.store.turns()[-1]
        calls = json.loads(row["tool_calls_json"])
        assert [(c["name"], c["source"]) for c in calls] == [("gate_pack", "auto"), ("fundamentals_pack", "auto"),
                                                            ("news_pack", "auto")]
        timings = json.loads(row["timings_json"])
        assert timings["auto_packs"] == 3 and timings["prompt_tokens"] == 50 and "attach_ms" in timings
        assert "Not covered:" in row["text"], "the stored turn is what the trader saw"
        win._io.submit(lambda: None).result(5)
        win.send("/latency")
        _flush(win)
        app.processEvents()
        assert "Last answers" in _text(win) and "gemma4:12b" in _text(win)
    finally:
        win.shutdown()
        win.deleteLater()


def test_today_answers_from_the_journal(app, tmp_path, monkeypatch):
    from mentor_app.window import MentorWindow
    from mentor_packs import journal_pack

    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: "")
    journal = journal_pack.write_fixture_journal(tmp_path / "trade_journal.sqlite3")
    win = MentorWindow(store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"), queue=PrefetchQueue(),
                       post=lambda url, payload, timeout: {}, journal_path=journal,
                       now=lambda: journal_pack.FIXTURE_NOW)
    try:
        win.send("/today")
        assert win.queue.run_one()
        app.processEvents()
        assert "[jrn:2026-09-30:totals]" in _text(win) and "LONG NVDA" in _text(win)
        assert win._read_journal_symbols() == ["TSLA", "NVDA", "AMD", "ALL", "MSFT"]
    finally:
        win.shutdown()
        win.deleteLater()


def test_tape_and_chips_show_the_context_pack(window):
    from mentor_packs import context_pack

    window._on_context(context_pack.fixture())
    assert window.chips["auto_mode"].text().startswith("Auto mode: DESK")
    assert window.chips["positions"].text() == "Open: 1"
    window._show_chip("d1_env")
    assert "[ctx:d1_env]" in _text(window)
    # P5: /tape is the regime pack now (tests/test_mentor_app_tape.py); it is built off-thread.
    window.send("/tape")
    assert "reading the desk" in _text(window) and window.queue.pending_keys() == ["tape-build"]


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
