"""Out-of-process Master AVWAP scan entry point.

The GUI runs scanner work in a child process so a native fault in the scanner
cannot close the desk. That child used to be spawned as ``sys.executable -c
"<code>"``, which is correct from a source checkout and **impossible when
frozen**: under PyInstaller ``sys.executable`` is ``TradingBotV3.exe``, so the
``-c`` and the code string are handed to the application's own argument parser,
which rejects them and exits 2 before a single bar is fetched.

That is exactly what happened on the desk. Every scheduled swing scan from
2026-08-12 07:30 onward failed one second after launching:

    Swing scan for slot 09:00 FAILED: Master AVWAP scan process exited with
    code 2.
    TradingBotV3.exe: error: unrecognized arguments: -c import faulthandler; ...

Eleven D1 evidence sources went stale while the desk looked healthy, because
everything that runs in-process -- BounceBot, the open scan, the away report --
was unaffected.

So the transport differs by build and the *work* does not. Both spawn forms call
:func:`run` with the same JSON payload; only the argv shape changes, and
:func:`ui.services.scan_service.scan_worker_command` owns that choice.

This module must stay import-light: it is imported inside the child process and
inside ``launch_gui.main`` before the Qt application exists. It imports no Qt.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

#: Printed on stdout once every report file is written. The parent waits for
#: this rather than for process exit, because the scanner stays alive after it
#: for the deferred option-enrichment thread.
SCAN_OK_MARKER = "SCAN_SUBPROCESS_OK"


def parse_payload(payload: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalise the spawn payload. A malformed one is a bug, not a default.

    Guessing here would run *some* scan rather than the requested one -- with
    or without a setup-tracker write, against the wrong watchlists -- and the
    tracker write is not a thing to perform speculatively.
    """
    if payload is None:
        raise ValueError("scan worker payload is missing")
    spec = json.loads(payload) if isinstance(payload, str) else dict(payload)
    if not isinstance(spec, Mapping):
        raise ValueError(f"scan worker payload must be an object, got {type(spec).__name__}")
    return {
        "use_shared_watchlists": bool(spec.get("use_shared_watchlists")),
        # None is meaningful: it selects the caller-chooses-the-entry-point
        # branch, which is not the same as "do not update the tracker".
        "update_setup_tracker": (
            None if spec.get("update_setup_tracker") is None
            else bool(spec.get("update_setup_tracker"))
        ),
    }


def run(payload: str | Mapping[str, Any] | None) -> int:
    """Run one Master AVWAP scan in this process and print the marker."""
    import faulthandler

    faulthandler.enable()
    spec = parse_payload(payload)

    from master_avwap_lib.runner import run_master, run_master_with_shared_watchlists

    if spec["update_setup_tracker"] is None:
        if spec["use_shared_watchlists"]:
            run_master_with_shared_watchlists()
        else:
            run_master()
    else:
        run_master(
            use_shared_watchlists=True,
            update_setup_tracker=spec["update_setup_tracker"],
            require_ib_for_setup_tracker=True,
        )
    print(SCAN_OK_MARKER, flush=True)
    return 0
