#!/bin/bash
# macOS launcher for the SATELLITE DESK: the full Trading Desk UI fed by the
# main PC's Desk Link relay instead of TWS. Alerts land in the real Alert
# Center as if this machine were connected to the API. Uses the pairing
# saved by the satellite window / connect dialog (prompts if missing).
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

"$PYTHON" scripts/gui.py --ui qt --satellite-desk "$@"
status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "TradingBotV3 Satellite Desk exited with an error (status $status)."
    read -r -p "Press Return to close..." _
fi
exit "$status"
