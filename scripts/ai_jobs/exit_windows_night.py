"""The `exit_windows` night slot (S11): deterministic, no model.

Reads the M5 outcome log in chunks, builds `exit_windows.build_payload` over the
last `exit_windows.WINDOW_SESSIONS` sessions and replaces `EXIT_WINDOWS_FILE`.
A failed read or build returns `failed` and leaves the last good file alone; an
empty window returns `ok` and writes nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def run_exit_windows(
    *,
    session_date: str = "",
    outcomes_path: Any = None,
    out_path: Any = None,
    window: tuple[str, str] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    import exit_windows

    session = str(session_date or "")[:10]
    try:
        if window is None:
            import held_run_score

            window = held_run_score.window_bounds(
                sessions=exit_windows.WINDOW_SESSIONS, as_of=session or None
            )
        if outcomes_path is None:
            from project_paths import INTRADAY_BOUNCE_OUTCOMES_FILE

            outcomes_path = INTRADAY_BOUNCE_OUTCOMES_FILE
        if out_path is None:
            from project_paths import EXIT_WINDOWS_FILE

            out_path = EXIT_WINDOWS_FILE
        rows = exit_windows.read_window_rows(Path(outcomes_path), tuple(window))
        payload = exit_windows.build_payload(rows, as_of=session, window=tuple(window))
    except Exception as exc:  # noqa: BLE001 - the night goes on; the last file stays
        _log.exception("exit_windows: build failed")
        return {
            "status": "failed", "model": "",
            "reason": f"exit windows not built ({type(exc).__name__}: {exc}); last file kept",
            "outputs": [],
        }
    if not payload["alerts"]:
        return {
            "status": "ok", "model": "",
            "reason": f"no finished M5 alerts in {window[0]}..{window[1]}; last file kept",
            "outputs": [],
        }
    try:
        written = exit_windows.write_payload(payload, Path(out_path))
    except OSError as exc:
        return {
            "status": "failed", "model": "",
            "reason": f"exit windows not written ({exc}); last file kept",
            "outputs": [],
        }
    return {
        "status": "ok", "model": "",
        "reason": (
            f"{payload['alerts']} finished M5 alerts in {window[0]}..{window[1]}, "
            f"{len(payload['cells'])} cells"
        ),
        "outputs": [str(written)],
    }
