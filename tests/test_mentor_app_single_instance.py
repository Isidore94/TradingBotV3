"""Trade Mentor app: its own single-instance slot, never the desk's."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import single_instance  # noqa: E402


def test_the_mentor_key_is_not_the_desk_key():
    assert single_instance.MENTOR_LOCK_KEY == "tradingbotv3-mentor"
    assert single_instance.MENTOR_LOCK_KEY != single_instance.DESK_LOCK_KEY


def test_desk_and_mentor_slots_are_held_together():
    with single_instance.desk_slot(key="tradingbotv3-test-desk-pair") as desk:
        with single_instance.mentor_slot(key="tradingbotv3-test-mentor-pair") as mentor:
            assert "desk slot held" in desk
            assert "mentor slot held" in mentor


def test_a_taken_mentor_slot_raises_the_mentor_error(monkeypatch):
    from local_writer_lock import LocalLockUnavailable

    def busy(key, **kwargs):
        raise LocalLockUnavailable("another thread in this process has held the writer lock")

    monkeypatch.setattr("local_writer_lock.local_writer_lock", busy)
    with pytest.raises(single_instance.AnotherMentorIsRunning) as excinfo:
        with single_instance.mentor_slot(key="tradingbotv3-test-mentor-busy"):
            pass
    assert "Trade Mentor app" in str(excinfo.value)


def test_the_probe_reads_free_and_does_not_keep_the_slot():
    key = "tradingbotv3-test-mentor-probe"
    assert single_instance.slot_is_free(key) is True
    with single_instance.mentor_slot(key=key):
        pass
    assert single_instance.slot_is_free(key) is True


def test_the_probe_reads_held_when_another_holder_has_it(monkeypatch):
    from local_writer_lock import LocalLockUnavailable

    def busy(key, **kwargs):
        raise LocalLockUnavailable("held elsewhere")

    monkeypatch.setattr("local_writer_lock.local_writer_lock", busy)
    assert single_instance.slot_is_free("tradingbotv3-test-mentor-held") is False


def test_launch_mentor_uses_the_mentor_slot_and_pings_when_taken(monkeypatch):
    import importlib

    launch_mentor = importlib.import_module("launch_mentor") if str(ROOT_DIR) in sys.path else None
    if launch_mentor is None:
        sys.path.insert(0, str(ROOT_DIR))
        launch_mentor = importlib.import_module("launch_mentor")
    from contextlib import contextmanager

    @contextmanager
    def taken(**kwargs):
        raise single_instance.AnotherMentorIsRunning("taken")
        yield  # pragma: no cover

    pings: list[int] = []
    monkeypatch.setattr(single_instance, "mentor_slot", taken)
    monkeypatch.setattr("mentor_app.focus_link.send_focus_ping", lambda timeout_ms=500, **k: pings.append(timeout_ms) or True)
    assert launch_mentor.main([]) == 0
    assert pings, "a second launch must bring the running app to the front"
