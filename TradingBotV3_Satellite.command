#!/bin/bash
# macOS launcher for the Desk Link satellite (view-only mirror of the main
# desk — no TWS needed on this machine). First launch opens a connect
# dialog: enter the main PC's IP and the token from its Settings page.
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

"$PYTHON" scripts/gui.py --ui qt --satellite "$@"
status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "TradingBotV3 Satellite exited with an error (status $status)."
    read -r -p "Press Return to close..." _
fi
exit "$status"
