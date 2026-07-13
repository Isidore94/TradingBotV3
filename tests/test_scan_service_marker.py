"""Marker-based scan completion in the Qt scan service.

The scan subprocess prints SCAN_SUBPROCESS_OK after every report is written,
then stays alive for the deferred theta option enrichment thread. The GUI must
treat the marker as completion instead of waiting for process exit, or the
enrichment tail delays the table refresh and the next scheduler slot.
"""

import os
import sys
import time

import pytest


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
