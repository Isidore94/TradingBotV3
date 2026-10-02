"""The plan-rule gate (trader 2026-10-02): one yes/no question to a local CPU decision model
before plan inference. Its reply parsing, its settings, the server owner, and the window's
wiring (Pause AI and the night window stop it; a shadow record per inference).

Never starts the real llama-server and never touches the network: every launcher, GET and POST is fake.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import project_paths  # noqa: E402
from mentor_app import rule_gate, settings  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 10, 2, 8, 0, tzinfo=PT)


@pytest.fixture(autouse=True)
def scratch_settings(tmp_path, monkeypatch):
    path = tmp_path / "local_settings.json"
    monkeypatch.setattr(project_paths, "LOCAL_SETTINGS_FILE", path)
    project_paths.invalidate_local_settings_cache()

    def write(**values):
        path.write_text(json.dumps(values), encoding="utf-8")
        project_paths.invalidate_local_settings_cache()

    yield write
    project_paths.invalidate_local_settings_cache()


# ---------------------------------------------------------------- score
def _reply(value):
    return {"model": "kev", "answers": {"rule": {"type": "noul", "noul": value}}, "usage": {}}


def test_score_sends_the_tested_request_and_reads_noul():
    sent = []

    def post(url, payload, timeout):
        sent.append((url, payload, timeout))
        return _reply(0.35)

    assert rule_gate.score("I trade the vwap setup", endpoint="http://127.0.0.1:11438/", post=post) == 0.35
    url, payload, timeout = sent[0]
    assert url == "http://127.0.0.1:11438/v1/systemone" and timeout == 10.0
    assert payload == {
        "state": "Trader message: I trade the vwap setup",
        "questions": {"rule": {"type": "noul", "instructions": rule_gate.RULE_INSTRUCTIONS}},
    }
    assert rule_gate.RULE_INSTRUCTIONS == (
        "Does the trader clearly state a standing rule he follows or has decided to follow from now on "
        "(a limit, a time, a goal, a setup he trades, a risk rule, or something he is testing)? A question, "
        "a hypothetical, a maybe, a feeling, a market observation, or a plan for one single trade is NOT a rule."
    )
    assert rule_gate.CUT == 0.5


def _raises(url, payload, timeout):
    raise ConnectionError("server down")


@pytest.mark.parametrize("post", [
    _raises,
    lambda url, payload, timeout: {},
    lambda url, payload, timeout: {"answers": {"rule": {}}},
    lambda url, payload, timeout: {"answers": []},
    lambda url, payload, timeout: None,
    lambda url, payload, timeout: _reply("0.9"),
    lambda url, payload, timeout: _reply(None),
    lambda url, payload, timeout: _reply(True),
    lambda url, payload, timeout: _reply(float("nan")),
    lambda url, payload, timeout: _reply(1.5),
])
def test_score_is_none_on_any_failure(post):
    assert rule_gate.score("x", endpoint="http://127.0.0.1:11438", post=post) is None


# ---------------------------------------------------------------- settings
def test_mode_defaults_to_shadow_and_anything_else_is_off(scratch_settings):
    assert rule_gate.mode() == "shadow"
    for raw, expected in (("on", "on"), (" Shadow ", "shadow"), ("OFF", "off"), ("yes", "off"), (True, "off"),
                          (None, "off"), (1, "off")):
        scratch_settings(mentor_rule_gate=raw)
        assert rule_gate.mode() == expected, raw


def test_default_paths_come_from_the_home_dir(monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("C:/Users/someone")))
    assert rule_gate.server_path() == Path("C:/Users/someone/llama-decision/bin2/llama-server.exe")
    assert rule_gate.model_path() == Path("C:/Users/someone/llama-decision/models/Kev-4B-Q4_K_M.gguf")


def test_path_settings_override(scratch_settings, tmp_path):
    scratch_settings(mentor_rule_gate_server=str(tmp_path / "s.exe"), mentor_rule_gate_model=str(tmp_path / "m.gguf"))
    assert rule_gate.server_path() == tmp_path / "s.exe" and rule_gate.model_path() == tmp_path / "m.gguf"


def test_port_never_collides_with_the_tunnel_ports(scratch_settings):
    assert rule_gate.port() == 11438
    scratch_settings(mentor_rule_gate_port=11500)
    assert rule_gate.port() == 11500
    for taken in (11435, 11436, 11437):
        scratch_settings(mentor_rule_gate_port=taken)
        assert rule_gate.port() == 11438, taken
    scratch_settings(mentor_rule_gate_port=11438, mentor_tunnel_port=11438)
    assert settings.tunnel_port() == 11438 and rule_gate.port() == 11439
    scratch_settings(mentor_rule_gate_port="junk")
    assert rule_gate.port() == 11438


# ---------------------------------------------------------------- the server owner
class _Proc:
    def __init__(self):
        self.alive = True
        self.terminated = 0

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated += 1
        self.alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.alive = False


class _Popen:
    def __init__(self):
        self.calls = []
        self.procs = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        proc = _Proc()
        self.procs.append(proc)
        return proc


def _server(popen, *, exists=lambda path: True, get=None):
    return rule_gate.GateServer(Path("C:/x/llama-server.exe"), Path("C:/x/kev.gguf"), 11438, popen=popen,
                                exists=exists, get=get or (lambda url, timeout: {"status": "ok"}))


def test_start_launches_the_tested_command_once_without_a_console_and_stop_terminates():
    popen = _Popen()
    server = _server(popen)
    assert server.start() is True and server.start() is True
    assert len(popen.calls) == 1, "a running server is not launched twice"
    cmd, kwargs = popen.calls[0]
    assert cmd == [str(Path("C:/x/llama-server.exe")), "-m", str(Path("C:/x/kev.gguf")), "--host", "127.0.0.1",
                   "--port", "11438", "-c", "4096", "-np", "1", "-t", "4"]
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["stdout"] is subprocess.DEVNULL
    assert server.running()
    server.stop()
    assert popen.procs[0].terminated == 1 and not server.running()
    server.stop()  # nothing to stop: no error


def test_start_with_a_missing_file_does_nothing_and_logs_once(caplog):
    popen = _Popen()
    server = _server(popen, exists=lambda path: path.suffix != ".gguf")
    with caplog.at_level(logging.INFO):
        assert server.start() is False
        assert server.start() is False
    assert popen.calls == []
    assert len([r for r in caplog.records if "rule gate off" in r.getMessage()]) == 1


def test_a_failed_launch_never_raises():
    def broken(cmd, **kwargs):
        raise OSError("bad exe")

    server = _server(broken)
    assert server.start() is False and not server.running()


def test_ready_is_health_ok():
    assert _server(_Popen()).ready() is True
    assert _server(_Popen(), get=lambda url, timeout: {"status": "loading"}).ready() is False

    def loading(url, timeout):
        raise RuntimeError("HTTP 503")

    assert _server(_Popen(), get=loading).ready() is False


# ---------------------------------------------------------------- the window
PLAN = "## Rules\n\n- Respect the stop.\n\n## Risk\n\n## What I am testing\n\n## Decisions\n"


class _FakeServer:
    endpoint = "http://127.0.0.1:11438"

    def __init__(self, score=0.2):
        self.alive = False
        self.starts = 0
        self.stops = 0
        self.scored: list[str] = []
        self.score = score

    def running(self):
        return self.alive

    def start(self):
        self.starts += 1
        self.alive = True
        return True

    def stop(self):
        self.stops += 1
        self.alive = False

    def scorer(self, post):
        def gate(text):
            self.scored.append(text)
            return self.score
        return gate


def _answer(text):
    return [json.dumps({"message": {"content": text}, "done": False}).encode(),
            json.dumps({"message": {"content": ""}, "done": True}).encode()]


@pytest.fixture()
def window(tmp_path, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from mentor_app import plan_infer
    from mentor_app.inbox import Inbox
    from mentor_app.prefetch import PrefetchQueue
    from mentor_app.store import MentorChatStore
    from mentor_app.window import MentorWindow

    app = QApplication.instance() or QApplication([])
    block = [""]
    monkeypatch.setattr(settings, "gpu_block_reason", lambda now=None: block[0])
    monkeypatch.setattr(settings, "context_tokens", lambda: 8192)
    plan = tmp_path / "trading_plan.md"
    plan.write_text(PLAN, encoding="utf-8")
    calls: list = []

    def post(url, payload, timeout):
        if payload.get("format") == plan_infer.SCHEMA:
            calls.append(url)
            return {"message": {"content": json.dumps({"ops": []})}}
        return {}

    server = _FakeServer()
    win = MentorWindow(
        store=MentorChatStore(tmp_path / "mentor_chat.sqlite3"),
        queue=PrefetchQueue(blocked=lambda: "", model_ready=lambda: True),
        inbox=Inbox(per_day_cap=6, now=lambda: NOW),
        stream_post=lambda url, payload, cancelled: _answer("Noted."),
        post=post,
        now=lambda: NOW,
        mentor_enabled=False,
        liked_source=lambda: [],
        memory_root=tmp_path / "ai",
        plan_path=plan,
        rule_gate_server=server,
    )
    win.app, win.calls, win.server, win.block = app, calls, server, block
    win._submit_io(win._open_session)
    win._io.submit(lambda: None).result(5)
    yield win
    win.shutdown()
    win.deleteLater()


def _wait(win, done, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        win._io.submit(lambda: None).result(5)
        win.app.processEvents()
        if done():
            return True
        time.sleep(0.02)
    return False


def _settle():
    for thread in threading.enumerate():
        if thread.name.startswith("mentor-rule-gate"):
            thread.join(5)


def _chat(win, text):
    win._brain_ok, win._endpoint, win._model = True, "http://127.0.0.1:11436", "qwen3:14b"
    win.send(text)
    assert win._worker is not None and win._worker.wait(5000)
    assert _wait(win, lambda: win._worker is None)


def test_the_server_runs_only_while_the_app_may_use_a_model(window, scratch_settings):
    window._sync_rule_gate()
    _settle()
    assert window.server.starts == 1 and window.server.alive
    window.block[0] = "the night AI starts within 15 minutes; the model is handed back"
    window.check_gpu_share()
    _settle()
    assert window.server.stops == 1 and not window.server.alive, "the refusal window stops the server"
    window.block[0] = ""
    window.check_gpu_share()
    _settle()
    assert window.server.starts == 2 and window.server.alive, "it starts again when the window ends"
    scratch_settings(mentor_rule_gate="off")
    window._sync_rule_gate()
    _settle()
    assert not window.server.alive, "mode off stops it"


def test_pause_ai_stops_the_server_and_the_gate_is_not_asked(window, monkeypatch):
    import ai_pause

    window._sync_rule_gate()
    _settle()
    assert window.server.alive
    ai_pause.pause_for("until_resumed", NOW)
    window.check_ai_pause()
    _settle()
    assert window.server.stops >= 1 and not window.server.alive
    window.server.alive = True  # even with the server somehow up, a paused app never asks the gate
    window._brain_ok = True
    window._store_turn("user", "I stop after two losses.")
    window.queue.start()
    window._queue_plan_inference()
    assert _wait(window, lambda: window.calls)
    assert window.server.scored == []


def test_shadow_record_is_kept_in_the_store_and_gemma_still_asked(window):
    window._sync_rule_gate()
    _settle()
    window.queue.start()
    _chat(window, "I stop after two losses.")
    turn_id = [row for row in window.store.turns(window._session_id) if row["role"] == "user"][-1]["id"]
    key = rule_gate.RECORD_KEY.format(turn_id=turn_id)
    assert _wait(window, lambda: window.store.get_state(key))
    record = json.loads(window.store.get_state(key))
    assert record == {"turn_ids": [f"turn:{turn_id}"], "score": 0.2, "mode": "shadow", "cut": 0.5,
                      "would_skip": True, "skipped": False, "ops": 0}
    assert window.server.scored == ["I stop after two losses."] and len(window.calls) == 1
