#!/bin/bash
# macOS launcher for the Trading Desk (Finder runs .command files in Terminal).
# Mirrors TradingBotV3_GUI.cmd: prefer the repo-local venv, fall back to python3.
# Goes through launch_gui.py so native-crash logging (faulthandler) stays on.
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

"$PYTHON" launch_gui.py "$@"
status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "TradingBotV3 GUI exited with an error (status $status)."
    read -r -p "Press Return to close..." _
fi
exit "$status"
