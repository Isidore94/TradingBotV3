#!/usr/bin/env python3
"""Compatibility entrypoint for the consolidated TradingBot GUI."""

from __future__ import annotations

import argparse
import sys


def _launch_from_cli() -> int:
    parser = argparse.ArgumentParser(description="Launch the TradingBotV3 desktop GUI.", add_help=False)
    parser.add_argument("--ui", choices=("tk", "qt"), default="qt")
    args, remaining = parser.parse_known_args()

    if args.ui == "qt":
        try:
            from ui import app as qt_app
        except ModuleNotFoundError as exc:
            if exc.name == "PySide6":
                print(
                    "PySide6 is not installed. Install the GUI dependencies with "
                    "`pip install -r requirements.txt`, then rerun the GUI.",
                    file=sys.stderr,
                )
                return 2
            raise
        return int(qt_app.main(remaining) or 0)

    # Import the legacy Tk stack only when it is actually requested
    # (plan.md 23.7): the Qt launch path must not pay its import cost.
    from gui_app import app as _app

    sys.argv = [sys.argv[0], *remaining]
    _app.main()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_launch_from_cli())
    except KeyboardInterrupt:
        pass
else:
    from gui_app import app as _app

    sys.modules[__name__] = _app
