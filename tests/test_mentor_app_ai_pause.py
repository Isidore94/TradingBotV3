"""Pause AI in the Trade Mentor app: `/ai off|on`, the header button, a distinct "paused"
state, both models unloaded and the tunnel closed, no model call while paused, and the
normal start again on resume or expiry."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from PySide6.QtWidgets import QApplication  # noqa: E402

import ai_pause  # noqa: E402
import project_paths  # noqa: E402
from mentor_app import commands, settings  # noqa: E402
from mentor_app.prefetch import PrefetchQueue  # noqa: E402
from mentor_app.store import MentorChatStore  # noqa: E402

PT = ai_pause.PT
#: A Wednesday, 10:00 PT: the app may use the model.
NOW = datetime(2026, 9, 30, 10, 0, tzinfo=PT)


class _Tunnel:
    endpoint = "http://127.0.0.1:11436"

    def __init__(self):
        self.stops = 0
        self.preflights = 0

    def stop(self):
        self.stops += 1

    def preflight(self):
        self.preflights += 1
        raise AssertionError("no preflight while AI is paused")


@pytest.fixture(autouse=True)
def scratch_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_FILE", tmp_path / "local_settings.json")
    project_paths.invalidate_local_settings_cache()
    yield
    project_paths.invalidate_local_settings_cache()


@pytest.fixture
def clock():
    return [NOW]


@pytest.fixture
def window(tmp_path, monkeypatch, clock):
    from mentor_app.window import MentorWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    streamed: list = []
    posted: list = []
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        queue=PrefetchQueue(),
        tunnel=_Tunnel(),
        now=lambda: clock[0],
        mentor_enabled=False,
        stream_post=lambda url, payload, cancelled: streamed.append(payload) or [],
        post=lambda url, payload, timeout: posted.append((url, payload)) or {},
    )
    win.streamed, win.posted = streamed, posted
    yield win
    win.shutdown()
    win.deleteLater()


def _join(win):
    for thread in win._threads:
        thread.join(5)


def _unloaded(win):
    return sorted(p["model"] for url, p in win.posted if p.get("keep_alive") == 0)


def _up(win):
    win._brain_ok, win._endpoint, win._model = True, "http://127.0.0.1:11436", "gpt-oss:20b"


# ---------------------------------------------------------------- commands
@pytest.mark.parametrize(
    "text, action, arg",
    [
        ("/ai", "ai_status", None),
        ("/ai on", "ai_on", None),
        ("/ai off", "ai_off", "tonight"),
        ("/ai off tonight", "ai_off", "tonight"),
        ("/ai off 2h", "ai_off", timedelta(hours=2)),
        ("/ai off 4h", "ai_off", timedelta(hours=4)),
        ("/ai off forever", "ai_off", "until_resumed"),
        ("/ai off soon", "error", None),
        ("/ai maybe", "error", None),
        ("/pause", "pause", None),
    ],
)
def test_ai_commands_parse_and_pause_stays_the_mentor_pause(text, action, arg):
    result = commands.handle(text)
    assert result.action == action
    if arg is not None:
        assert result.arg == arg


def test_help_lists_ai_off_and_on():
    assert "/ai off" in commands.HELP_TEXT and "/ai on" in commands.HELP_TEXT


# ---------------------------------------------------------------- pausing
def test_ai_off_unloads_both_models_closes_the_tunnel_and_shows_paused(window):
    _up(window)
    window.send("/ai off 2h")
    _join(window)

    assert ai_pause.is_paused(NOW)
    assert not window._brain_ok
    assert {"gpt-oss:20b", settings.EMBED_MODEL} <= set(_unloaded(window))
    assert window._tunnel.stops == 1
    assert window.status_pill.text() == "AI paused until 12:00"
    assert window.banner.isVisibleTo(window) and "`/ai on` resumes" in window.banner.text()
    assert window.ai_pause_button.text() == "AI paused"


def test_a_pause_also_unloads_the_nights_model_tags(window):
    project_paths.save_local_settings({"ai_local_model_medium": "gemma3:12b", "ai_local_model_large": "gpt-oss:120b"})
    _up(window)
    window.send("/ai off 2h")
    _join(window)

    assert _unloaded(window) == sorted(["gpt-oss:20b", "gemma3:12b", "gpt-oss:120b", settings.EMBED_MODEL])


def test_a_pause_before_the_model_is_known_unloads_the_configured_chat_model(window):
    project_paths.save_local_settings({settings.MODEL_KEY: "qwen3:14b", "ai_local_model_medium": "qwen3:14b",
                                       "ai_local_model_large": "qwen3:14b"})
    window._brain_ok, window._endpoint, window._model = True, "http://127.0.0.1:11436", ""
    window.send("/ai off 2h")
    _join(window)

    assert _unloaded(window) == sorted(["qwen3:14b", settings.EMBED_MODEL])


def test_a_chat_turn_while_paused_makes_no_model_call(window):
    _up(window)
    window.send("/ai off 2h")
    window.send("Is NVDA worth it?")

    assert window._worker is None and window.streamed == []
    assert "AI is paused until 12:00 — `/ai on` to resume." in window._blocks[-1]


def test_a_pause_from_the_desk_is_picked_up_by_the_5s_check(window):
    _up(window)
    ai_pause.pause_for("4h", NOW)
    window.check_ai_pause()
    _join(window)

    assert not window._brain_ok and window._tunnel.stops == 1
    assert window.status_pill.text() == "AI paused until 14:00"


def test_paused_minutes_are_not_outage_minutes_and_nothing_reconnects(window, monkeypatch):
    connects = []
    monkeypatch.setattr(window, "connect_brain", lambda: connects.append(1))
    ai_pause.pause_for("2h", NOW)
    window.check_gpu_share()
    window.check_gpu_share()

    assert connects == []
    window._io.submit(lambda: None).result(5)
    assert not window.store.day_stats(NOW.date().isoformat()).get("brain_offline_min")


def test_the_connect_worker_never_touches_the_host_while_paused(window):
    ai_pause.pause_for("2h", NOW)
    states = []
    window._bridge.brain_state.connect(states.append)
    window._connect_worker()
    QApplication.processEvents()

    assert window._tunnel.preflights == 0
    assert states and not states[-1]["ok"] and states[-1]["reason"] == "AI paused until 12:00"


def test_a_pause_during_the_warm_up_hands_the_model_back(window):
    state = {"endpoint": "http://127.0.0.1:11436", "model": "gpt-oss:20b"}
    assert window._paused_while_connecting(state, warmed=True) is False
    ai_pause.pause_for("2h", NOW)
    assert window._paused_while_connecting(state, warmed=True) is True
    assert _unloaded(window) == sorted(["gpt-oss:20b", settings.EMBED_MODEL])
    assert window._tunnel.stops == 1


def test_a_pick_card_while_paused_shows_the_evidence_with_ai_paused(window):
    from mentor_packs.registry import Pack

    ai_pause.pause_for("2h", NOW)
    window.check_ai_pause()
    pack = Pack(name="pick_pack", rows=({"id": "pick:NVDA:price", "text": "NVDA 180.00"},))
    window._pick_blocks["NVDA"] = len(window._blocks)
    window._add_block("**Pick NVDA**: building...")
    window._on_pick_built({"symbol": "NVDA", "side": "LONG", "pack": pack, "hash": "h1", "assessment": None})

    text = window.transcript.toPlainText()
    assert "AI paused until 12:00" in text
    assert not any(key.startswith("pick-assess") for key in window.queue.pending_keys())


# ---------------------------------------------------------------- resuming
def test_ai_on_runs_the_normal_start_again(window, monkeypatch):
    connects = []
    monkeypatch.setattr(window, "connect_brain", lambda: connects.append(1))
    window.send("/ai off 2h")
    window.send("/ai on")

    assert not ai_pause.is_paused(NOW)
    assert connects == [1]
    assert "brain" in window.status_pill.text()


def test_an_expired_pause_reconnects_on_the_minute_check(window, monkeypatch, clock):
    connects = []
    monkeypatch.setattr(window, "connect_brain", lambda: connects.append(1))
    window.send("/ai off 2h")
    clock[0] = NOW + timedelta(hours=2, minutes=1)
    window.check_gpu_share()

    assert connects == [1]
    assert window._paused_until is None


def test_resume_inside_the_night_window_stays_off(window, monkeypatch, clock):
    connects = []
    monkeypatch.setattr(window, "connect_brain", lambda: connects.append(1))
    clock[0] = datetime(2026, 9, 30, 23, 0, tzinfo=PT)
    window.send("/ai off 2h")
    window.send("/ai on")

    assert connects == []
    assert "night" in window._brain_reason


def test_the_gpu_block_reason_says_paused_so_the_tunnel_watchdog_stops_too():
    ai_pause.pause_for("2h", NOW)
    assert settings.gpu_block_reason(NOW) == "AI paused until 12:00"
    ai_pause.resume()
    assert settings.gpu_block_reason(NOW) == ""


def test_a_tunnel_stopped_by_a_pause_opens_again_on_resume():
    import threading
    import time

    from mentor_app.tunnel import Tunnel

    procs: list = []

    class _Proc:
        def __init__(self):
            self.ended = threading.Event()

        def wait(self):
            self.ended.wait(5)
            time.sleep(0.3)  # ssh takes a moment to exit after terminate

        def terminate(self):
            self.ended.set()

    ticks = iter(range(0, 10_000, 100))
    tunnel = Tunnel(
        "claude-host", 11436,
        popen=lambda *a, **k: procs.append(_Proc()) or procs[-1],
        port_open=lambda host, port: len(procs) >= 2,
        sleep=lambda seconds: None,
        clock=lambda: next(ticks),
    )
    try:
        assert tunnel.open() is False and len(procs) == 1
        tunnel.stop()  # Pause AI: the watchdog is still winding down here
        assert tunnel.open() is True, "resume starts a fresh watchdog"
        assert len(procs) == 2
    finally:
        tunnel.stop()
        for proc in procs:
            proc.terminate()


def test_the_header_button_writes_the_same_setting(window):
    window.ai_pause_button.pause("4h")
    _join(window)
    saved = json.loads(project_paths.LOCAL_SETTINGS_FILE.read_text(encoding="utf-8"))
    assert saved[ai_pause.PAUSE_KEY] == "2026-09-30T14:00:00-07:00"
    assert window._paused_until is not None
    window.ai_pause_button.resume()
    assert not ai_pause.is_paused(NOW)
