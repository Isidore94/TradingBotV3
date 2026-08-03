#!/bin/bash
# Compatibility shortcut only. launch_gui.py is the one supported entrypoint;
# choose Main/Satellite inside Settings -> Desk Link.
set -u
cd "$(dirname "$0")" || exit 1

PYTHON=""
if [ -x ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "python3 was not found. Run setup_macos.command first." >&2
    read -r -p "Press Return to close..." _
    exit 1
fi

"$PYTHON" launch_gui.py --desk-role satellite "$@"
status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "TradingBotV3 Satellite Desk exited with an error (status $status)."
    read -r -p "Press Return to close..." _
fi
exit "$status"
