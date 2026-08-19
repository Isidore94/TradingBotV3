"""Marker-based scan completion in the Qt scan service.

The scan subprocess prints SCAN_SUBPROCESS_OK after every report is written,
then stays alive for the deferred theta option enrichment thread. The GUI must
treat the marker as completion instead of waiting for process exit, or the
enrichment tail delays the table refresh and the next scheduler slot.
"""

import os
import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_wait_for_scan_marker_returns_before_process_exit():
    from ui.services.scan_service import _wait_for_scan_marker

    code = "print('SCAN_SUBPROCESS_OK', flush=True); import time; time.sleep(5)"
    start = time.monotonic()
    stdout_text = _wait_for_scan_marker(
        [sys.executable, "-c", code], cwd=os.getcwd(), env=os.environ.copy()
    )
    elapsed = time.monotonic() - start
    assert "SCAN_SUBPROCESS_OK" in stdout_text
    assert elapsed < 4  # returned on the marker, not the 5s exit


def test_wait_for_scan_marker_raises_on_failure_without_marker():
    from ui.services.scan_service import _wait_for_scan_marker

    code = "import sys; print('boom-detail', file=sys.stderr); sys.exit(3)"
    with pytest.raises(RuntimeError) as excinfo:
        _wait_for_scan_marker(
            [sys.executable, "-c", code], cwd=os.getcwd(), env=os.environ.copy()
        )
    message = str(excinfo.value)
    assert "exited with code 3" in message
    assert "boom-detail" in message


def test_wait_for_scan_marker_accepts_marker_on_clean_fast_exit():
    from ui.services.scan_service import _wait_for_scan_marker

    code = "print('SCAN_SUBPROCESS_OK', flush=True)"
    stdout_text = _wait_for_scan_marker(
        [sys.executable, "-c", code], cwd=os.getcwd(), env=os.environ.copy()
    )
    assert "SCAN_SUBPROCESS_OK" in stdout_text


def test_wait_for_scan_marker_reports_child_pid():
    from ui.services.scan_service import _wait_for_scan_marker

    pids = []
    _wait_for_scan_marker(
        [sys.executable, "-c", "print('SCAN_SUBPROCESS_OK', flush=True)"],
        cwd=os.getcwd(),
        env=os.environ.copy(),
        on_process_started=pids.append,
    )
    assert len(pids) == 1 and isinstance(pids[0], int) and pids[0] > 0


def _child_program(lines: list[str]) -> str:
    """A child program built without escape sequences of our own."""
    return chr(10).join(["import sys"] + lines)


def test_failure_first_line_names_the_child_s_own_error():
    """Auto Pilot's activity feed keeps only the first line.

    ``AutopilotService._on_scan_failed`` logs ``detail.splitlines()[0]`` to
    ``autopilot.log``, so the three real desk failures (2026-08-17 07:30 and
    10:00, 2026-08-18 12:00) read only "Master AVWAP scan process exited with
    code 1." and named nothing. The cause sat in the child's stderr the whole
    time; it belongs on the line the trader actually sees.
    """
    from ui.services.scan_service import _wait_for_scan_marker

    code = _child_program([
        "err = sys.stderr",
        "print('Traceback (most recent call last):', file=err)",
        "print('  File legacy.py, line 2122, in _write_text_atomic', file=err)",
        "print('    os.replace(temp_path, path)', file=err)",
        "print('PermissionError: [WinError 5] Access is denied: master_avwap_market_prep.txt', file=err)",
        "sys.exit(1)",
    ])
    with pytest.raises(RuntimeError) as excinfo:
        _wait_for_scan_marker(
            [sys.executable, "-c", code], cwd=os.getcwd(), env=os.environ.copy()
        )
    first_line = str(excinfo.value).splitlines()[0]
    assert "exited with code 1" in first_line
    assert "PermissionError" in first_line
    assert "master_avwap_market_prep.txt" in first_line


def test_failure_first_line_ignores_indented_traceback_frames():
    """Only the unindented exception line is a cause; a frame is not."""
    from ui.services.scan_service import _wait_for_scan_marker

    code = _child_program([
        "err = sys.stderr",
        "print('Traceback (most recent call last):', file=err)",
        "print('  File runner.py, line 2265, in _run_master_impl', file=err)",
        "sys.exit(1)",
    ])
    with pytest.raises(RuntimeError) as excinfo:
        _wait_for_scan_marker(
            [sys.executable, "-c", code], cwd=os.getcwd(), env=os.environ.copy()
        )
    first_line = str(excinfo.value).splitlines()[0]
    assert "Traceback (most recent call last):" in first_line
    assert "runner.py" not in first_line


def test_failure_first_line_survives_a_child_with_no_stderr():
    from ui.services.scan_service import _wait_for_scan_marker

    with pytest.raises(RuntimeError) as excinfo:
        _wait_for_scan_marker(
            [sys.executable, "-c", "import sys; sys.exit(4)"],
            cwd=os.getcwd(),
            env=os.environ.copy(),
        )
    first_line = str(excinfo.value).splitlines()[0]
    assert first_line.strip() == "Master AVWAP scan process exited with code 4."


def test_failure_first_line_is_bounded():
    """A 5,000-character exception must not swamp the feed or the phone."""
    from ui.services.scan_service import _wait_for_scan_marker

    code = _child_program([
        "print('ValueError: ' + 'x' * 5000, file=sys.stderr)",
        "sys.exit(1)",
    ])
    with pytest.raises(RuntimeError) as excinfo:
        _wait_for_scan_marker(
            [sys.executable, "-c", code], cwd=os.getcwd(), env=os.environ.copy()
        )
    first_line = str(excinfo.value).splitlines()[0]
    assert len(first_line) < 400
    assert "ValueError" in first_line
