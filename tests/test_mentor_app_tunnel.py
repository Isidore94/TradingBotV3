"""Trade Mentor tunnel + settings: own port, GPU time-share edges, preflight steps."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ai_summary  # noqa: E402
import project_paths  # noqa: E402
from mentor_app import settings, tunnel  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
LIVE_LIKE = {
    "ai_offhours_start": "05:00",
    "ai_offhours_end": "09:00",
    "ai_local_model_medium": "gemma3:12b-tbv3ctx-64k",
    "ai_remote_gpu_ssh_alias": "claude-host",
}


@pytest.fixture
def local_settings(monkeypatch):
    values = dict(LIVE_LIKE)

    def get(key, default=None):
        return values.get(key, default)

    monkeypatch.setattr(project_paths, "get_local_setting", get)
    monkeypatch.setattr(ai_summary, "get_local_setting", get)
    # ai_jobs.window may resolve `scripts.project_paths`, a second module object.
    import ai_jobs.window

    monkeypatch.setattr(ai_jobs.window, "_paths", lambda: project_paths)
    return values


def test_the_app_port_defaults_to_11436_and_is_never_the_nights(local_settings):
    assert settings.tunnel_port() == 11436
    assert settings.tunnel_port() != 11435
    assert settings.tunnel_port() != settings.night_tunnel_port()
    local_settings["ai_remote_gpu_tunnel_port"] = 11436
    assert settings.tunnel_port() not in (11436, settings.night_tunnel_port())
    local_settings["mentor_tunnel_port"] = 11435
    local_settings.pop("ai_remote_gpu_tunnel_port")
    assert settings.tunnel_port() == 11436


def test_model_and_keep_alive_defaults(local_settings):
    assert settings.mentor_model() == "gemma3:12b-tbv3ctx-64k"
    local_settings["mentor_model"] = "gpt-oss:20b"
    assert settings.mentor_model() == "gpt-oss:20b"
    assert settings.keep_alive() == -1
    assert settings.ssh_alias() == "claude-host"


@pytest.mark.parametrize(
    ("hh", "mm", "blocked"),
    [
        (1, 44, False),   # the last free minute before the hand-back
        (1, 45, True),    # 15 minutes before the 02:00 PT night window
        (3, 30, True),
        (5, 59, True),    # the night still owns it
        (6, 0, False),    # window closed: the app may load the model again
        (12, 0, False),
    ],
)
def test_gpu_time_share_edges(local_settings, hh, mm, blocked):
    moment = datetime(2026, 9, 30, hh, mm, tzinfo=PT)
    assert bool(settings.gpu_block_reason(moment)) is blocked


def test_the_start_script_gets_two_parallel_slots_without_changing_the_file():
    real = tunnel.UP_SCRIPT.read_text(encoding="utf-8")
    assert tunnel.SERVE_ANCHOR in real, "ollama_up.sh changed its serve line; update SERVE_ANCHOR"
    text = tunnel.up_script_text()
    assert f'{tunnel.SERVE_ANCHOR}OLLAMA_NUM_PARALLEL=2 OLLAMA_MODELS=' in text
    assert "OLLAMA_NUM_PARALLEL" not in real
    assert "\r" not in text


class _Proc:
    def __init__(self, on_wait=None):
        self.on_wait = on_wait
        self.terminated = False

    def wait(self):
        if self.on_wait:
            self.on_wait()

    def terminate(self):
        self.terminated = True


def _fake_run(calls, *, host_line="hostname 192.168.0.220", start_rc=0):
    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["ssh", "-G"]:
            return SimpleNamespace(stdout=f"user aaron\n{host_line}\nport 22\n", returncode=0)
        if cmd[0] == "ssh":
            assert isinstance(kwargs["input"], bytes) and b"OLLAMA_NUM_PARALLEL=2" in kwargs["input"]
            return SimpleNamespace(stdout=b"ollama: started\n", returncode=start_rc)
        return SimpleNamespace(stdout="", returncode=0)

    return run


def test_preflight_happy_path_opens_the_app_port(tmp_path):
    calls: list = []
    opened: dict = {"tunnel": False}
    stop = {}

    def popen(cmd, **kwargs):
        calls.append(cmd)
        opened["tunnel"] = True
        return _Proc(on_wait=lambda: stop["t"]._stop.wait(5))

    def port_open(host, port):
        return host != "127.0.0.1" or opened["tunnel"]

    t = tunnel.Tunnel(
        "claude-host", 11436, run=_fake_run(calls), popen=popen, port_open=port_open,
        get=lambda url, timeout: SimpleNamespace(status_code=200), sleep=lambda s: None,
        wake_script=tmp_path / "missing.ps1",
    )
    stop["t"] = t
    status = t.preflight()
    t.stop()
    assert status.ok, status.reason
    assert status.host == "192.168.0.220"
    forward = next(cmd for cmd in calls if "-N" in cmd)
    assert "127.0.0.1:11436:127.0.0.1:11434" in forward and forward[-1] == "claude-host"


def test_a_sleeping_host_is_woken_through_host_on(tmp_path):
    wake = tmp_path / "host-on.ps1"
    wake.write_text("# wol", encoding="utf-8")
    calls: list = []
    state = {"awake": False}

    def run(cmd, **kwargs):
        if cmd[0] == "powershell.exe":
            calls.append(cmd)
            state["awake"] = True
            return SimpleNamespace(stdout="", returncode=0)
        return _fake_run(calls)(cmd, **kwargs)

    t = tunnel.Tunnel(
        "claude-host", 11436, run=run, port_open=lambda host, port: state["awake"], wake_script=wake,
        sleep=lambda s: None,
    )
    assert t.ensure_host_up("192.168.0.220") == (True, True)
    assert any(str(wake) in cmd for cmd in calls)


def test_no_alias_and_an_unresolvable_alias_fail_softly(tmp_path):
    assert not tunnel.Tunnel("", 11436).preflight().ok
    calls: list = []
    t = tunnel.Tunnel("nowhere", 11436, run=_fake_run(calls, host_line=""), wake_script=tmp_path / "x.ps1")
    status = t.preflight()
    assert not status.ok and "resolve" in status.reason


def test_the_watchdog_reopens_a_dropped_tunnel():
    starts: list = []
    t = tunnel.Tunnel("claude-host", 11436, sleep=lambda s: None)

    def popen(cmd, **kwargs):
        starts.append(cmd)
        if len(starts) >= 3:
            t._stop.set()
        return _Proc()

    t._popen = popen
    t._watch()
    assert len(starts) == 3 and t.restarts == 2


def test_a_passing_health_check_is_trusted_for_sixty_seconds():
    now = {"t": 1000.0}
    hits: list = []
    t = tunnel.Tunnel(
        "claude-host", 11436, clock=lambda: now["t"],
        get=lambda url, timeout: hits.append(url) or SimpleNamespace(status_code=200),
    )
    assert t.alive() and t.alive()
    assert hits == ["http://127.0.0.1:11436/api/version"]
    now["t"] += 61
    assert t.alive() and len(hits) == 2


def test_an_unreachable_host_is_probed_every_five_minutes_without_respawning_ssh():
    starts: list = []
    pauses: list = []
    probes = {"n": 0}
    t = tunnel.Tunnel("claude-host", 11436)
    t.host = "192.168.0.220"

    def port_open(host, port):
        probes["n"] += 1
        if probes["n"] >= 4:
            t._stop.set()
        return False

    def popen(cmd, **kwargs):
        starts.append(cmd)
        if len(starts) >= 3:  # a respawning watchdog would loop forever; end the test
            t._stop.set()
        return _Proc()

    t._popen = popen
    t._port_open = port_open
    t._sleep = pauses.append
    t._watch()
    assert len(starts) == 1, "ssh -N must not be respawned while the host is unreachable"
    assert pauses and all(p == tunnel.HOST_PROBE_SECONDS == 300 for p in pauses)


def test_the_watchdog_respawns_once_the_host_answers_again():
    starts: list = []
    answers = iter([False, True])
    t = tunnel.Tunnel("claude-host", 11436, sleep=lambda s: None)
    t.host = "192.168.0.220"

    def popen(cmd, **kwargs):
        starts.append(cmd)
        if len(starts) >= 2:
            t._stop.set()
        return _Proc()

    t._popen = popen
    t._port_open = lambda host, port: next(answers)
    t._watch()
    assert len(starts) == 2


def test_the_watchdog_stops_entirely_inside_the_night_window():
    starts: list = []
    state = {"night": ""}
    t = tunnel.Tunnel("claude-host", 11436, sleep=lambda s: None, blocked=lambda: state["night"])
    t.host = "192.168.0.220"

    def popen(cmd, **kwargs):
        starts.append(cmd)
        state["night"] = "the night AI owns the GPU"
        if len(starts) >= 3:  # a watchdog that ignores the night would loop forever
            t._stop.set()
        return _Proc()

    t._popen = popen
    t._port_open = lambda host, port: True
    t._watch()  # returns by itself: no respawn once the night opens
    assert len(starts) == 1
    assert tunnel.from_settings()._blocked is settings.gpu_block_reason


def test_a_night_started_ollama_reports_one_slot(tmp_path):
    def run(cmd, **kwargs):
        if cmd[:2] == ["ssh", "-G"]:
            return SimpleNamespace(stdout="hostname 192.168.0.220\n", returncode=0)
        return SimpleNamespace(stdout=b"ollama: already up\n", returncode=0)

    common = dict(
        port_open=lambda host, port: True, get=lambda url, timeout: SimpleNamespace(status_code=200),
        sleep=lambda s: None, wake_script=tmp_path / "missing.ps1",
    )
    status = tunnel.Tunnel("claude-host", 11436, run=run, **common).preflight()
    assert status.ok and status.slots == 1
    assert tunnel.Tunnel("claude-host", 11436, run=_fake_run([]), **common).preflight().slots == 2
