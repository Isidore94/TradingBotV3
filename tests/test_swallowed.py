"""P2-11a: a swallowed failure leaves a rate-limited, reasoned log line."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import swallowed  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_state():
    swallowed.reset_swallowed()
    yield
    swallowed.reset_swallowed()


def test_note_logs_reason_and_exception(caplog):
    caplog.set_level(logging.DEBUG, logger="tradingbot.swallowed")
    swallowed.note_swallowed("store write failed", OSError("disk locked"))
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "store write failed" in record.getMessage()
    assert "OSError: disk locked" in record.getMessage()


def test_note_is_rate_limited_per_reason_but_counts_every_hit(caplog):
    caplog.set_level(logging.DEBUG, logger="tradingbot.swallowed")
    for _ in range(50):
        swallowed.note_swallowed("hot loop", ValueError("x"))
    swallowed.note_swallowed("other reason")
    assert [r.getMessage().split(":")[0] for r in caplog.records] == [
        "swallowed hot loop",
        "swallowed other reason",
    ]
    assert swallowed.swallowed_counts() == {"hot loop": 50, "other reason": 1}


def test_zero_interval_logs_every_time_with_seen_count(caplog):
    caplog.set_level(logging.DEBUG, logger="tradingbot.swallowed")
    swallowed.note_swallowed("r", interval_s=0)
    swallowed.note_swallowed("r", interval_s=0)
    assert "(seen 2x)" in caplog.records[-1].getMessage()


def test_quiet_logs_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="tradingbot.swallowed")
    swallowed.note_swallowed("temp file not removed", OSError("gone"), quiet=True)
    assert caplog.records[0].levelno == logging.DEBUG


def test_a_broken_logger_never_raises():
    class Broken:
        def log(self, *_args, **_kwargs):
            raise RuntimeError("handler exploded")

    swallowed.note_swallowed("anything", logger=Broken())


def test_focus_store_write_failure_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    """A store site: the write is still best-effort, but the failure now leaves a line."""
    import focus_picks

    def _locked(*_args, **_kwargs):
        raise PermissionError("locked by AV scan")

    monkeypatch.setattr(focus_picks.os, "replace", _locked)
    caplog.set_level(logging.DEBUG)
    focus_picks._atomic_json_write(tmp_path / "sidecar.json", {"a": 1})
    messages = [r.getMessage() for r in caplog.records]
    assert any("focus store JSON write failed" in m and "locked by AV scan" in m for m in messages)


def test_modules_imported_as_scripts_dot_x_do_not_need_scripts_on_sys_path(tmp_path):
    """`from scripts import earnings_history` (and friends) must keep working
    without scripts/ on sys.path: those modules import the helper lazily."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        f"sys.path[:] = [{str(root)!r}] + [p for p in sys.path if p and 'scripts' not in p]\n"
        "from scripts import earnings_history, project_paths, trading_plan\n"
        "from scripts.diagnostics import artifact_io, run_manifest\n"
        "from scripts.research_warehouse import schemas, outcomes\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env.update(
        {
            "TRADINGBOTV3_DATA_DIR": str(tmp_path / "data"),
            "LOCALAPPDATA": str(tmp_path / "local"),
            "PYTHONPATH": "",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120, cwd=str(tmp_path)
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout
