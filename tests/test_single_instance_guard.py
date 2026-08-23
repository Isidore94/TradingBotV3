"""R10.A / Sol blocker 3, second layer - one desk per machine.

`launch_gui_auto.ps1` already refuses a second desk, but only on that path. A
double-click, a shortcut, a terminal, a second scheduled task and the frozen exe
all reach `launch_gui.py` directly, and R10.0 measured the consequence: pid
31848 overlapped three other desks on 2026-08-20, the worst by **3.8 hours**.

**This is defence in depth, not the transaction.** The outcome finalizer fences
itself with the same machine-local primitive and re-reads the authoritative
checkpoint inside that fence, and it stays correct with two desks running -
because this guard can be overridden and a second process can always be started
another way. `test_outcome_finalization_durability.py` is where that is proven.

The guard fails **open** when the machine has no exclusion primitive at all:
refusing to start the trader's desk over that is a worse failure than the
duplicate it prevents. It fails **closed** only when it can see somebody else
holding the slot.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import single_instance  # noqa: E402


def test_the_slot_is_held_with_a_real_primitive():
    with single_instance.desk_slot(key="tradingbotv3-test-held") as protection:
        assert "desk slot held" in protection
        assert "mutex: held" in protection or "file lock: held" in protection


def test_the_override_skips_the_guard_entirely():
    with single_instance.desk_slot(allow_second=True, key="tradingbotv3-test-override") as note:
        assert "override" in note


def test_a_missing_primitive_fails_open(monkeypatch):
    """A box that cannot exclude anything is a warning, not a locked-out desk."""
    from local_writer_lock import LocalLockUnavailable

    def no_primitive(key, **kwargs):
        raise LocalLockUnavailable(
            "no machine-local exclusion primitive is available (named mutex: unavailable, "
            "file lock: unavailable); refusing to publish"
        )

    monkeypatch.setattr("local_writer_lock.local_writer_lock", no_primitive)
    with single_instance.desk_slot(key="tradingbotv3-test-noprim") as note:
        assert "no OS primitive" in note


def test_another_holder_fails_closed(monkeypatch):
    from local_writer_lock import LocalLockUnavailable

    def busy(key, **kwargs):
        raise LocalLockUnavailable("another thread in this process has held the writer lock")

    monkeypatch.setattr("local_writer_lock.local_writer_lock", busy)
    with pytest.raises(single_instance.AnotherDeskIsRunning) as excinfo:
        with single_instance.desk_slot(key="tradingbotv3-test-busy"):
            pass
    assert single_instance.OVERRIDE_FLAG in str(excinfo.value)


def test_an_unexpected_failure_fails_open(monkeypatch):
    monkeypatch.setattr(
        "local_writer_lock.local_writer_lock",
        lambda key, **kwargs: (_ for _ in ()).throw(RuntimeError("something else")),
    )
    with single_instance.desk_slot(key="tradingbotv3-test-odd") as note:
        assert "guard unavailable" in note


HOLDER = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, r"{scripts}")
    from single_instance import desk_slot

    with desk_slot(key="{key}") as note:
        print("HELD", flush=True)
        time.sleep(float("{seconds}"))
    """
)

CHALLENGER = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, r"{scripts}")
    from single_instance import AnotherDeskIsRunning, desk_slot

    try:
        with desk_slot(key="{key}"):
            print("ACQUIRED")
    except AnotherDeskIsRunning:
        print("REFUSED")
    """
)


def test_a_second_real_process_is_refused(tmp_path):
    """The case the in-process RLock could never see."""
    key = "tradingbotv3-test-twoproc"
    holder_script = tmp_path / "holder.py"
    holder_script.write_text(HOLDER.format(scripts=SCRIPTS_DIR, key=key, seconds=8), encoding="utf-8")
    challenger_script = tmp_path / "challenger.py"
    challenger_script.write_text(CHALLENGER.format(scripts=SCRIPTS_DIR, key=key), encoding="utf-8")

    holder = subprocess.Popen(
        [sys.executable, str(holder_script)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "HELD", "the holder never acquired"
        result = subprocess.run(
            [sys.executable, str(challenger_script)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "REFUSED", result.stdout
    finally:
        holder.kill()
        holder.communicate(timeout=30)


def test_the_slot_is_released_when_the_holder_dies(tmp_path):
    """A killed desk must not wedge the machine out of its own slot."""
    key = "tradingbotv3-test-killed"
    holder_script = tmp_path / "holder.py"
    holder_script.write_text(HOLDER.format(scripts=SCRIPTS_DIR, key=key, seconds=120), encoding="utf-8")
    challenger_script = tmp_path / "challenger.py"
    challenger_script.write_text(CHALLENGER.format(scripts=SCRIPTS_DIR, key=key), encoding="utf-8")

    holder = subprocess.Popen(
        [sys.executable, str(holder_script)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert holder.stdout.readline().strip() == "HELD"
    holder.kill()
    holder.communicate(timeout=30)

    result = subprocess.run(
        [sys.executable, str(challenger_script)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.stdout.strip() == "ACQUIRED", result.stdout + result.stderr


# ---------------------------------------------------------------------------
# the launcher wiring
# ---------------------------------------------------------------------------
def test_the_launcher_guards_the_desk_and_only_the_desk():
    """`--selftest` and `--run-scan` are legitimately concurrent with a desk."""
    source = (ROOT_DIR / "launch_gui.py").read_text(encoding="utf-8")
    guard_at = source.index("desk_slot(")
    selftest_at = source.index('"--selftest" in argv')
    scan_at = source.index('"--run-scan" in argv')
    assert selftest_at < guard_at and scan_at < guard_at, (
        "the guard must sit after the two entry points that are not the desk"
    )


def test_the_launcher_exits_zero_when_a_desk_is_already_running():
    """Double-clicking twice is a normal thing to do, not a failure to report."""
    source = (ROOT_DIR / "launch_gui.py").read_text(encoding="utf-8")
    block = source[source.index("except AnotherDeskIsRunning"):]
    assert "return 0" in block.split("\n\n")[0]


def test_the_override_flag_is_not_passed_through_to_the_app():
    source = (ROOT_DIR / "launch_gui.py").read_text(encoding="utf-8")
    assert "arg != OVERRIDE_FLAG" in source
