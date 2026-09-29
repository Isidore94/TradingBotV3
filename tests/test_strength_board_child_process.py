"""The strength board build runs in a spawned child, never on a desk thread.

2026-09-29: the in-desk `strength-board` thread used 30-36 CPU-s per minute
and starved the Qt thread of the GIL (GUI stalls of 23.7 s, 16 s, 13.6 s).
"""

from __future__ import annotations

import importlib.util
import os
import pickle
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FIXTURE = Path(__file__).with_name("fixtures") / "strength_board_child_targets.py"
GOOD = {"long": [{"symbol": "AAPL"}], "short": [], "offered": 1, "measured": 1}


def _fixture_module():
    spec = importlib.util.spec_from_file_location("sb_child_targets_inproc", FIXTURE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _service(monkeypatch, target: str):
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - PySide6 is on the desk
        pytest.skip("PySide6 is not installed")
    QApplication.instance() or QApplication([])
    from ui.services import strength_board_service as module

    # The child resolves the target by spec; the in-process patch only exists so
    # a build that (wrongly) ran on the desk thread would use the same stand-in.
    monkeypatch.setattr(module, "BOARD_TARGET_SPEC", f"{FIXTURE}:{target}", raising=False)
    monkeypatch.setattr(module, "build_board", getattr(_fixture_module(), target))
    service = module.StrengthBoardService()
    service._timer.stop()
    return service, module


def _wait_for(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_build_board_runs_in_a_different_process(monkeypatch):
    service, _module = _service(monkeypatch, "board_with_pid")
    published: list[dict] = []
    service.boardChanged.connect(published.append)

    service._worker()

    board = service.board()
    assert board.get("pid") not in (None, os.getpid()), "built in the desk process"
    assert board["long"] == [{"symbol": "NVDA"}]
    assert board["fraction"] == service._fraction(), "fraction reaches the child"
    assert published and published[-1]["pid"] == board["pid"]
    assert service._last_error == ""


def test_a_child_timeout_keeps_the_last_good_board_and_reports(monkeypatch):
    service, module = _service(monkeypatch, "sleep_long")
    monkeypatch.setattr(module, "BOARD_CHILD_TIMEOUT_SECONDS", 1.5, raising=False)
    service._board = dict(GOOD)
    service._last_success = service._last_attempt = __import__("datetime").datetime.now()

    started = time.monotonic()
    service._worker()

    assert time.monotonic() - started < 7.0, "the timeout did not stop the wait"
    assert service.board()["long"] == [{"symbol": "AAPL"}], "last good survives"
    assert "timed out" in service._last_error
    assert "FAILED" in service.status_text()
    assert service._child is None


def test_a_child_exception_is_an_error_not_a_crash(monkeypatch):
    service, _module = _service(monkeypatch, "raise_error")
    service._board = dict(GOOD)

    service._worker()

    assert service.board()["long"] == [{"symbol": "AAPL"}]
    assert "boom in child" in service._last_error
    assert service.running is False


def test_a_child_that_dies_without_answering_is_an_error(monkeypatch):
    service, module = _service(monkeypatch, "hard_exit")
    # In-process, os._exit would kill the test runner: never run it on this thread.
    monkeypatch.setattr(module, "build_board", _fixture_module().raise_error)
    service._board = dict(GOOD)

    service._worker()

    assert service.board()["long"] == [{"symbol": "AAPL"}]
    assert "exit" in service._last_error.lower() and "3" in service._last_error


def test_shutdown_terminates_a_running_child(monkeypatch):
    service, _module = _service(monkeypatch, "sleep_long")
    assert service.refresh_now() is True

    assert _wait_for(lambda: getattr(service, "_child", None) is not None
                     and service._child.is_alive()), "child never started"
    child = service._child
    started = time.monotonic()
    service.shutdown()

    assert _wait_for(lambda: not child.is_alive(), timeout=5.0), "orphan child"
    assert time.monotonic() - started < 5.0
    assert _wait_for(lambda: service.running is False, timeout=5.0)
    assert service.board()["long"] == [], "a killed build publishes nothing"


def test_shutdown_with_no_child_is_quiet(monkeypatch):
    service, _module = _service(monkeypatch, "board_with_pid")
    service.shutdown()
    assert service.running is False
    assert threading.active_count() >= 1


def test_a_real_board_survives_the_pipe_unchanged():
    """What crosses the pipe is pickled: the board must round-trip exactly."""
    import test_strength_board_service as base
    from ui.services.strength_board_service import build_board

    mapping = {
        f"N{index}": base._bars(
            105.0, prev_high=100.0, prev_low=98.0, opening=101.0,
            today_volume=1000.0 * (index + 1),
        )
        for index in range(4)
    }
    daily = {symbol: base._daily(50.0) for symbol in mapping}
    board = build_board(
        symbols=list(mapping), downloader=base._downloader(mapping, daily),
        fraction=1.0, now=base.NOW,
    )
    assert board["long"], "fixture produced an empty board"
    assert pickle.loads(pickle.dumps(board)) == board


def test_the_frozen_app_answers_spawn_children_first():
    """A frozen spawn child re-runs TradingBotV3.exe; without freeze_support it
    would start a second desk instead of the board build."""
    import ast

    tree = ast.parse((ROOT_DIR / "launch_gui.py").read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    first_calls = [
        ast.unparse(node) for node in main.body[:3] if isinstance(node, ast.Expr)
    ]
    assert any("freeze_support()" in text for text in first_calls), first_calls
