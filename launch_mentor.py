#!/usr/bin/env python3
"""Start the Trade Mentor app (its own process), or bring the running one to the front."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from single_instance import OVERRIDE_FLAG, AnotherMentorIsRunning, mentor_slot

    try:
        with mentor_slot(allow_second=OVERRIDE_FLAG in argv) as protection:
            print(f"TradingBotV3 Trade Mentor: {protection}")
            import mentor_app

            return int(mentor_app.main([arg for arg in argv if arg != OVERRIDE_FLAG]) or 0)
    except AnotherMentorIsRunning as exc:
        # Already running is a normal outcome: bring that window forward instead.
        from mentor_app.focus_link import send_focus_ping

        sent = send_focus_ping(1500)
        print(str(exc) if not sent else "TradingBotV3 Trade Mentor: already running; brought it to the front.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
