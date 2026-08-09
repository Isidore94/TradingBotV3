#!/usr/bin/env python3
"""Standalone overnight AI job runner (plan.md item 13b, Phase 1).

Task Scheduler boots this, it does its work, and it exits. It is deliberately
NOT hosted in the trading GUI:

* the lifecycles are opposed -- the desk is meant to be up during market hours
  and this layer is meant to run when it is not;
* the desk's own launch task relaunches the GUI every 15 minutes through the
  session, which would orphan a long job living inside it;
* a 14GB model load that goes wrong must not be able to take down the window
  the trader watches charts in;
* "no inference during market hours" becomes a scheduler fact rather than only
  a code check.

It imports no Qt and needs only ``requirements-core.txt``.

Exit codes: 0 = nothing to do or everything succeeded; 1 = at least one job
failed; 2 = the AI store was unreachable, so nothing ran.

Usage:
    python scripts/run_ai_jobs.py              # run every due slot, then exit
    python scripts/run_ai_jobs.py --status     # print state, run nothing
    python scripts/run_ai_jobs.py --slot ai_summary
    python scripts/run_ai_jobs.py --force      # ignore window + already-done
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if not getattr(sys, "frozen", False) and str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def _print_status() -> int:
    """Print state and run nothing -- including writing nothing.

    --status used to call store_available(), which creates the five-directory
    skeleton and writes a .write_probe file. "Print state, run nothing" that
    mkdirs on a NAS is a contradiction: during market hours it is a write the
    plan sec 2 hard rule never authorised, and on a sleeping share it turns a
    read into a ~20 s spin-up (checkpoint review 2026-08-08 second review).
    """
    from ai_jobs import ledger, runner, store, window

    details = store.get_ai_store_details()
    available, reason = (
        store.store_available(read_only=True)
        if details["enabled"] == "yes"
        else (False, details["error"] or "unset")
    )
    payload = {
        "session_date": runner.session_date_for(),
        "store": details,
        "store_available": available,
        "store_reason": reason,
        "window": window.describe_window(),
        "slots": [
            {"name": slot.name, "reserve_minutes": slot.reserve_minutes,
             "enabled": slot.enabled, "description": slot.description}
            for slot in runner.default_slots()
        ],
    }
    if available:
        session = payload["session_date"]
        try:
            path = ledger.ledger_path(create=False)
            payload["completed_today"] = sorted(ledger.completed_jobs(session, path=path))
            payload["recent"] = [
                {k: row.get(k) for k in ("job", "status", "session_date", "finished_at", "reason", "error")}
                for row in ledger.recent_rows(10, path=path)
            ]
        except (OSError, ValueError) as exc:
            payload["ledger_error"] = str(exc)
    print(json.dumps(payload, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="print state and exit without running anything")
    parser.add_argument("--slot", default="", help="run only this named slot")
    parser.add_argument("--force", action="store_true",
                        help="ignore the off-hours window and the already-completed check")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    if args.status:
        return _print_status()

    from ai_jobs import runner

    report = runner.run_slots(runner.default_slots(), force=args.force, only=args.slot)
    logging.info("%s", report.summary())

    if not report.store_ok:
        return 2
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
